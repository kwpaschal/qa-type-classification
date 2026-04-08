'''
generate_question_answering_roberta_base_squad
Author: Keith Paschal, Mahmood Ahmed
Class: CSC 790, NLP

Fine-tunes RoBERTa-base for extractive reading comprehension on SQuAD 1.1.

Key difference from BERT variants:
  RoBERTa has no Next Sentence Prediction (NSP) task, so token_type_ids
  are NOT passed to the model — they are all-zero and unused.

Evaluation uses the official SQuAD 1.1 metrics:
  - Exact Match (EM): prediction matches any gold answer exactly after normalisation
  - F1 Score:         token-level F1 averaged over the full validation set

Expected performance for RoBERTa-base on SQuAD 1.1 (4 epochs, lr=2e-5):
  EM ≈ 83-84%   F1 ≈ 90-91%

Output: ./squad_qa_roberta_base/
'''

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import RobertaProcessing
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from collections import Counter
import collections
import string
import re
import os
import sys
import glob
import random
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# ===========================================================================
# DDP / Device setup
# ===========================================================================

def _running_under_wsl():
    try:
        with open('/proc/version', encoding='utf-8') as f:
            return 'microsoft' in f.read().lower()
    except OSError:
        return False


def _distributed_backend():
    """
    NCCL is fastest on Linux servers but often fails on WSL2 / some Windows
    dual-GPU setups (CUDA 999 during DDP's first collective). Default to gloo
    there unless DDP_BACKEND is set explicitly.
    """
    if os.environ.get('DDP_BACKEND'):
        return os.environ['DDP_BACKEND']
    if sys.platform == 'win32':
        return 'gloo'
    if _running_under_wsl():
        return 'gloo'
    return 'nccl'


# Optional: helps NCCL on consumer GPUs / odd topologies (use if you still want NCCL).
if os.environ.get('NCCL_SAFE', '').lower() in ('1', 'true', 'yes'):
    os.environ.setdefault('NCCL_P2P_DISABLE', '1')
    os.environ.setdefault('NCCL_IB_DISABLE', '1')

local_rank = int(os.environ.get('LOCAL_RANK', 0))
if torch.cuda.is_available() and 'LOCAL_RANK' in os.environ:
    torch.cuda.set_device(local_rank)
    requested_backend = _distributed_backend()
    try:
        dist.init_process_group(backend=requested_backend)
    except Exception as e:
        if requested_backend == 'nccl':
            print(f'NCCL init failed ({e}). Falling back to gloo backend.')
            dist.init_process_group(backend='gloo')
        else:
            raise
    if dist.get_rank() == 0:
        print(f'Distributed backend: {dist.get_backend()}')
device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
is_main = (not dist.is_initialized()) or dist.get_rank() == 0

# Reproducibility across runs and workers.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# Prefer speed over full determinism for faster multi-GPU training.
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cudnn.allow_tf32 = True

# ===========================================================================
# Hyperparameters
# ===========================================================================

MAX_LENGTH     = int(os.environ.get('MAX_LENGTH', 384))
DOC_STRIDE     = int(os.environ.get('DOC_STRIDE', 128))
BATCH_SIZE     = int(os.environ.get('BATCH_SIZE', 16))
EPOCHS         = int(os.environ.get('EPOCHS', 4))
LR             = float(os.environ.get('LR', 2.0e-5))
WARMUP_RATIO   = float(os.environ.get('WARMUP_RATIO', 0.1))
GRAD_ACCUM_STEPS = int(os.environ.get('GRAD_ACCUM_STEPS', 1))
NUM_WORKERS    = int(os.environ.get('NUM_WORKERS', min(16, os.cpu_count() or 8)))
PREFETCH_FACTOR = int(os.environ.get('PREFETCH_FACTOR', 4))
N_BEST         = 20
MAX_ANS_LENGTH = 30

if is_main:
    print('Training config:')
    print(f'  MAX_LENGTH={MAX_LENGTH} DOC_STRIDE={DOC_STRIDE}')
    print(f'  BATCH_SIZE={BATCH_SIZE} GRAD_ACCUM_STEPS={GRAD_ACCUM_STEPS}')
    print(f'  NUM_WORKERS={NUM_WORKERS} PREFETCH_FACTOR={PREFETCH_FACTOR}')
    print(f'  EPOCHS={EPOCHS} LR={LR} WARMUP_RATIO={WARMUP_RATIO}')

# ===========================================================================
# Official SQuAD normalisation and scoring
# ===========================================================================

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def compute_exact(gold, pred):
    return int(normalize_answer(gold) == normalize_answer(pred))

def compute_f1(gold, pred):
    gold_toks = normalize_answer(gold).split()
    pred_toks = normalize_answer(pred).split()
    common    = Counter(gold_toks) & Counter(pred_toks)
    num_same  = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall    = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)

def squad_scores(predictions, references):
    exact_scores, f1_scores = {}, {}
    for ref in references:
        qid   = ref['id']
        golds = ref['answers']['text']
        pred  = predictions.get(qid, '')
        if not golds:
            exact_scores[qid] = int(pred == '')
            f1_scores[qid]    = int(pred == '')
        else:
            exact_scores[qid] = max(compute_exact(g, pred) for g in golds)
            f1_scores[qid]    = max(compute_f1(g, pred)    for g in golds)
    total = len(exact_scores)
    return {
        'exact_match': 100.0 * sum(exact_scores.values()) / total,
        'f1':          100.0 * sum(f1_scores.values())    / total,
    }

# ===========================================================================
# Tokenizer
# ===========================================================================

def _build_roberta_tokenizer_from_files(model_name: str) -> RobertaTokenizerFast:
    """
    Build RobertaTokenizerFast directly from vocab.json + merges.txt.

    tokenizers>=0.19 introduced a breaking format change: it no longer
    recognises the old tokenizer.json layout where 'model.type' is absent
    (None). When loaded that way the BPE merges are silently skipped and
    every character becomes its own token, which destroys model performance.

    Building from the raw BPE files bypasses that load path entirely and
    restores correct subword tokenization.
    """
    hf_cache  = os.path.expanduser('~/.cache/huggingface/hub')
    model_dir = 'models--' + model_name.replace('/', '--')
    snap_dirs = sorted(glob.glob(os.path.join(hf_cache, model_dir, 'snapshots', '*')))
    if not snap_dirs:
        raise FileNotFoundError(
            f'No HF cache snapshot found for {model_name}. '
            'Download the model files first.'
        )
    snap        = snap_dirs[-1]
    vocab_path  = os.path.join(snap, 'vocab.json')
    merges_path = os.path.join(snap, 'merges.txt')

    bpe = BPE.from_file(vocab=vocab_path, merges=merges_path, unk_token='<unk>')
    backend = Tokenizer(bpe)
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False, trim_offsets=True)
    backend.decoder       = ByteLevelDecoder(trim_offsets=True)
    backend.post_processor = RobertaProcessing(
        sep=('</s>', 2),
        cls=('<s>', 0),
        trim_offsets=True,
        add_prefix_space=False,
    )
    backend.add_special_tokens([
        AddedToken('<s>',    special=True),
        AddedToken('<pad>',  special=True),
        AddedToken('</s>',   special=True),
        AddedToken('<unk>',  special=True),
        AddedToken('<mask>', special=True, lstrip=True),
    ])
    return RobertaTokenizerFast(
        tokenizer_object=backend,
        bos_token='<s>', eos_token='</s>', sep_token='</s>',
        cls_token='<s>', unk_token='<unk>', pad_token='<pad>',
        mask_token='<mask>',
    )


tokenizer = _build_roberta_tokenizer_from_files('roberta-base')

# ===========================================================================
# Dataset loading
# ===========================================================================

print('Loading SQuAD 1.1...')
raw       = load_dataset('squad')
train_raw = raw['train']
val_raw   = raw['validation']
print(f'  Train: {len(train_raw)}  |  Val: {len(val_raw)}')

# ===========================================================================
# Preprocessing
# RoBERTa uses <s> Q </s> </s> Context </s> — two separator tokens between
# question and context. sequence_ids() handles this transparently.
# ===========================================================================

def preprocess_train(examples):
    questions = [q.strip() for q in examples['question']]
    inputs = tokenizer(
        questions, examples['context'],
        max_length=MAX_LENGTH, truncation='only_second', stride=DOC_STRIDE,
        return_overflowing_tokens=True, return_offsets_mapping=True, padding='max_length',
    )
    sample_map      = inputs.pop('overflow_to_sample_mapping')
    offset_mapping  = inputs.pop('offset_mapping')
    answers         = examples['answers']
    start_positions, end_positions = [], []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer     = answers[sample_idx]
        start_char = answer['answer_start'][0]
        end_char   = start_char + len(answer['text'][0])
        seq_ids    = inputs.sequence_ids(i)
        idx = 0
        while seq_ids[idx] != 1:
            idx += 1
        ctx_start = idx
        while idx < len(seq_ids) and seq_ids[idx] == 1:
            idx += 1
        ctx_end = idx - 1
        if offsets[ctx_start][0] > start_char or offsets[ctx_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = ctx_start
            while idx <= ctx_end and offsets[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)
            idx = ctx_end
            while idx >= ctx_start and offsets[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs['start_positions'] = start_positions
    inputs['end_positions']   = end_positions
    return inputs

def preprocess_val(examples):
    questions = [q.strip() for q in examples['question']]
    inputs = tokenizer(
        questions, examples['context'],
        max_length=MAX_LENGTH, truncation='only_second', stride=DOC_STRIDE,
        return_overflowing_tokens=True, return_offsets_mapping=True, padding='max_length',
    )
    sample_map  = inputs.pop('overflow_to_sample_mapping')
    example_ids = []
    for i in range(len(inputs['input_ids'])):
        sample_idx  = sample_map[i]
        example_ids.append(examples['id'][sample_idx])
        seq_ids     = inputs.sequence_ids(i)
        inputs['offset_mapping'][i] = [
            o if seq_ids[k] == 1 else None
            for k, o in enumerate(inputs['offset_mapping'][i])
        ]
    inputs['example_id'] = example_ids
    return inputs

print('Tokenizing training set...')
tokenized_train = train_raw.map(preprocess_train, batched=True, remove_columns=train_raw.column_names)
print(f'  Training chunks: {len(tokenized_train)}')

print('Tokenizing validation set...')
tokenized_val = val_raw.map(preprocess_val, batched=True, remove_columns=val_raw.column_names)
print(f'  Validation chunks: {len(tokenized_val)}')

# ===========================================================================
# Dataset wrappers  — NO token_type_ids for RoBERTa
# ===========================================================================

class SQuADTrainDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids':       torch.tensor(item['input_ids'],       dtype=torch.long),
            'attention_mask':  torch.tensor(item['attention_mask'],  dtype=torch.long),
            'start_positions': torch.tensor(item['start_positions'], dtype=torch.long),
            'end_positions':   torch.tensor(item['end_positions'],   dtype=torch.long),
        }

class SQuADValDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids':      torch.tensor(item['input_ids'],      dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
        }

train_dataset = SQuADTrainDataset(tokenized_train)
val_dataset   = SQuADValDataset(tokenized_val)

train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
train_loader  = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=(train_sampler is None), sampler=train_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
)

# ===========================================================================
# Model
# ===========================================================================

model     = RobertaForQuestionAnswering.from_pretrained('roberta-base').to(device)
if dist.is_initialized():
    try:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    except Exception:
        if dist.get_rank() == 0:
            print(
                'DDP failed (often NCCL on WSL/Windows/dual-GPU). Retry with e.g.:\n'
                '  DDP_BACKEND=gloo torchrun --standalone --nproc_per_node=2 ...\n'
                '  or NCCL_SAFE=1 DDP_BACKEND=nccl ...'
            )
        raise
scaler    = GradScaler('cuda')

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps_per_epoch = (len(train_loader) + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS
total_steps = steps_per_epoch * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * WARMUP_RATIO),
    num_training_steps=total_steps,
)

# ===========================================================================
# Postprocessing
# ===========================================================================

def postprocess_predictions(val_examples, val_features, start_logits_all, end_logits_all):
    example_id_to_index  = {ex['id']: i for i, ex in enumerate(val_examples)}
    features_per_example = collections.defaultdict(list)
    for feat_idx, feat in enumerate(val_features):
        features_per_example[example_id_to_index[feat['example_id']]].append(feat_idx)

    predictions = {}
    for ex_idx, example in enumerate(val_examples):
        context       = example['context']
        feat_indices  = features_per_example[ex_idx]
        valid_answers = []
        for feat_idx in feat_indices:
            start_logits   = start_logits_all[feat_idx]
            end_logits     = end_logits_all[feat_idx]
            offset_mapping = val_features[feat_idx]['offset_mapping']
            start_indexes  = np.argsort(start_logits)[-1:-N_BEST-1:-1].tolist()
            end_indexes    = np.argsort(end_logits)[-1:-N_BEST-1:-1].tolist()
            for s in start_indexes:
                for e in end_indexes:
                    if offset_mapping[s] is None or offset_mapping[e] is None:
                        continue
                    if e < s or (e - s + 1) > MAX_ANS_LENGTH:
                        continue
                    valid_answers.append({
                        'score': start_logits[s] + end_logits[e],
                        'text':  context[offset_mapping[s][0]:offset_mapping[e][1]],
                    })
        predictions[example['id']] = (
            max(valid_answers, key=lambda x: x['score'])['text'] if valid_answers else ''
        )
    return predictions

# ===========================================================================
# Training loop
# ===========================================================================

print('\nFine-tuning RoBERTa-base on SQuAD 1.1...')
best_f1 = 0.0

for epoch in range(EPOCHS):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    model.train()
    total_loss = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(train_loader):
        input_ids       = batch['input_ids'].to(device)
        attention_mask  = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions   = batch['end_positions'].to(device)

        with autocast('cuda'):
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                start_positions=start_positions, end_positions=end_positions,
            )
            loss = outputs.loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()
        should_step = ((batch_idx + 1) % GRAD_ACCUM_STEPS == 0) or ((batch_idx + 1) == len(train_loader))
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() == old_scale:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * GRAD_ACCUM_STEPS
        if (batch_idx + 1) % 200 == 0:
            print(f'  Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}')

    print(f'  Epoch {epoch+1} done — avg loss: {total_loss/len(train_loader):.4f}')

    if is_main:
        print('  Running SQuAD evaluation...')
        model_to_eval = model.module if dist.is_initialized() else model
        model_to_eval.eval()
        all_start_logits, all_end_logits = [], []
        with torch.no_grad():
            for batch in val_loader:
                with autocast('cuda'):
                    outputs = model_to_eval(
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                    )
                all_start_logits.append(outputs.start_logits.float().cpu().numpy())
                all_end_logits.append(outputs.end_logits.float().cpu().numpy())

        all_start_logits = np.concatenate(all_start_logits, axis=0)
        all_end_logits   = np.concatenate(all_end_logits,   axis=0)
        predictions      = postprocess_predictions(val_raw, tokenized_val, all_start_logits, all_end_logits)
        references       = [{'id': ex['id'], 'answers': ex['answers']} for ex in val_raw]
        metrics          = squad_scores(predictions, references)

        print(f'\n  ── Epoch {epoch+1} Validation ──')
        print(f'     Exact Match : {metrics["exact_match"]:.2f}%')
        print(f'     F1 Score    : {metrics["f1"]:.2f}%')

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            model_to_eval.save_pretrained('squad_qa_roberta_base')
            tokenizer.save_pretrained('squad_qa_roberta_base')
            print(f'     ✓ New best F1 — model saved to ./squad_qa_roberta_base/')

print(f'\nTraining complete. Best F1: {best_f1:.2f}%')
print('Model saved to ./squad_qa_roberta_base/')

if dist.is_initialized():
    dist.destroy_process_group()
