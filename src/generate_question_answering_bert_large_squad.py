'''
generate_question_answering_bert_large_squad
Author: Keith Paschal, Mahmood Ahmed
Class: CSC 790, NLP

Fine-tunes BERT-large-uncased for extractive reading comprehension on SQuAD 1.1.
Given a question and a passage, the model predicts the start and end token
positions of the answer span within the passage.

Evaluation uses the official SQuAD 1.1 metrics:
  - Exact Match (EM): prediction matches any gold answer exactly after normalisation
  - F1 Score:         token-level F1 averaged over the full validation set

Expected performance for BERT-large on SQuAD 1.1 (3 epochs, lr=2e-5):
  EM ≈ 83-84%   F1 ≈ 90-91%

Output: ./squad_qa_bert_large/
'''

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForQuestionAnswering
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from collections import Counter
import collections
import string
import re
import os
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# ===========================================================================
# DDP / Device setup
# ===========================================================================

local_rank = int(os.environ.get('LOCAL_RANK', 0))
if torch.cuda.is_available() and 'LOCAL_RANK' in os.environ:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='gloo')
device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
is_main = (not dist.is_initialized()) or dist.get_rank() == 0

# ===========================================================================
# Hyperparameters
# BERT-large needs a lower lr and smaller batch than BERT-base to remain stable.
# ===========================================================================

MAX_LENGTH     = 384
DOC_STRIDE     = 128
BATCH_SIZE     = 16    # dual RTX 4500 (24GB VRAM) + AMP handles full batch
EPOCHS         = 3
LR             = 2e-5  # lower than bert-base to prevent instability
N_BEST         = 20
MAX_ANS_LENGTH = 30

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

tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')

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
# Dataset wrappers
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
            'token_type_ids':  torch.tensor(item['token_type_ids'],  dtype=torch.long),
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
            'token_type_ids': torch.tensor(item['token_type_ids'], dtype=torch.long),
        }

train_dataset = SQuADTrainDataset(tokenized_train)
val_dataset   = SQuADValDataset(tokenized_val)

train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
train_loader  = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=(train_sampler is None), sampler=train_sampler,
    num_workers=4, pin_memory=True,
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

# ===========================================================================
# Model
# ===========================================================================

model     = BertForQuestionAnswering.from_pretrained('bert-large-uncased').to(device)
if dist.is_initialized():
    model = DDP(model, device_ids=[local_rank])
scaler    = GradScaler('cuda')

optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
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

print('\nFine-tuning BERT-large on SQuAD 1.1...')
best_f1 = 0.0

for epoch in range(EPOCHS):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids       = batch['input_ids'].to(device)
        attention_mask  = batch['attention_mask'].to(device)
        token_type_ids  = batch['token_type_ids'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions   = batch['end_positions'].to(device)

        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=start_positions, end_positions=end_positions,
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() == old_scale:  # only step if no inf/NaN skip
            scheduler.step()

        total_loss += loss.item()
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
                        token_type_ids=batch['token_type_ids'].to(device),
                    )
                all_start_logits.append(outputs.start_logits.cpu().numpy())
                all_end_logits.append(outputs.end_logits.cpu().numpy())

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
            model_to_eval.save_pretrained('squad_qa_bert_large')
            tokenizer.save_pretrained('squad_qa_bert_large')
            print(f'     ✓ New best F1 — model saved to ./squad_qa_bert_large/')

print(f'\nTraining complete. Best F1: {best_f1:.2f}%')
print('Model saved to ./squad_qa_bert_large/')

if dist.is_initialized():
    dist.destroy_process_group()
