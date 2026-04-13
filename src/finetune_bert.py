'''
finetune_bert_qa.py
Author: Keith Paschal, Mahmood Ahmed
Class: CSC 790, NLP

This script fine-tunes BertForQuestionAnswering on the SQuAD 1.1 dataset.
The fine-tuned model is saved to ./bert_qa_finetuned/ and loaded by
qa_with_type_matching.py for evaluation.

Following Devlin et al. (2019) [7], all BERT layers are unfrozen
during fine-tuning. The small learning rate of 3e-5 with linear
warmup prevents catastrophic forgetting of pretrained knowledge.
'''

import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForQuestionAnswering,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from datasets import load_dataset

# set DEBUG=False for full training run
DEBUG            = False
DEBUG_TRAIN_SIZE = 5000
DEBUG_VAL_SIZE   = 500

# agreed hyperparameters - must match Mahmood's baseline scripts
MAX_LENGTH        = 384   # standard for QA - covers most SQuAD contexts
DOC_STRIDE        = 128   # overlap between chunks for long contexts
BATCH_SIZE        = 16    
EPOCHS            = 3
LEARNING_RATE     = 3e-5  # standard BERT fine-tuning rate
WARMUP_RATIO      = 0.1   # first 10% of steps = warmup
MAX_ANSWER_LENGTH = 30    # max tokens in a valid answer span

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class SQuADDataset(Dataset):
    '''
    Dataset class for SQuAD 1.1 fine-tuning.
    Handles tokenization and conversion of character-level answer
    positions to token-level positions for BERT.
    '''
    def __init__(self, examples, tokenizer, max_length=MAX_LENGTH,
                 doc_stride=DOC_STRIDE):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.features   = []
        self._process_examples(examples)

    def _process_examples(self, examples):
        '''
        Tokenize all examples and convert character-level answer
        positions to token-level start and end positions.

        Some examples are skipped if the answer is truncated out
        of the context due to MAX_LENGTH limit.
        '''
        print("Processing examples...")
        skipped = 0

        for i, example in enumerate(examples):
            question     = example['question']
            context      = example['context']
            answer_text  = example['answers']['text'][0]
            answer_start = example['answers']['answer_start'][0]
            answer_end   = answer_start + len(answer_text)

            # tokenize question and context together
            encoding = self.tokenizer(
                question,
                context,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_offsets_mapping=True,
                return_tensors='pt'
            )

            offset_mapping = encoding['offset_mapping'][0].tolist()
            sequence_ids   = encoding.sequence_ids(0)

            # find token positions for answer
            # sequence_ids: None=special tokens, 0=question, 1=context
            token_start = None
            token_end   = None

            for idx, (seq_id, (char_start, char_end)) in enumerate(
                zip(sequence_ids, offset_mapping)
            ):
                # only look in context tokens (sequence_id == 1)
                if seq_id != 1:
                    continue

                if char_start <= answer_start < char_end:
                    token_start = idx

                if char_start < answer_end <= char_end:
                    token_end = idx

            # skip if answer not found in tokenized context
            # this happens when the answer is truncated out
            if token_start is None or token_end is None:
                skipped += 1
                continue

            self.features.append({
                'input_ids':       encoding['input_ids'][0],
                'attention_mask':  encoding['attention_mask'][0],
                'token_type_ids':  encoding['token_type_ids'][0],
                'start_position':  torch.tensor(token_start),
                'end_position':    torch.tensor(token_end),
            })

            if (i + 1) % 10000 == 0:
                print(f"  Processed {i+1}/{len(examples)} examples "
                      f"({skipped} skipped due to truncation)...")

        print(f"Total features: {len(self.features)} "
              f"({skipped} skipped due to truncation)")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

def evaluate(model, loader, desc="Validating"):
    '''
    Compute average loss on validation set.
    Full EM/F1 evaluation is handled by qa_with_type_matching.py
    since that requires the full pipeline including type matching.
    '''
    model.eval()
    total_loss = 0
    total      = 0

    with torch.no_grad():
        for batch in loader:
            input_ids       = batch['input_ids'].to(device)
            attention_mask  = batch['attention_mask'].to(device)
            token_type_ids  = batch['token_type_ids'].to(device)
            start_positions = batch['start_position'].to(device)
            end_positions   = batch['end_position'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=start_positions,
                end_positions=end_positions
            )

            total_loss += outputs.loss.item()
            total      += 1

    avg_loss = total_loss / total
    print(f"  {desc} Loss: {avg_loss:.4f}")
    return avg_loss


# Main loop
print("Loading SQuAD dataset...")
dataset = load_dataset("squad")

train_examples = list(dataset['train'])
val_examples   = list(dataset['validation'])

# shuffle training data - SQuAD is sorted by article
# without shuffling the model sees all Notre Dame questions
# then all Warsaw Pact questions etc causing loss spikes
random.seed(42)
random.shuffle(train_examples)
random.shuffle(val_examples)

# debug mode uses smaller subset for faster testing
if DEBUG:
    print(f"\nDEBUG MODE - using {DEBUG_TRAIN_SIZE} train, "
          f"{DEBUG_VAL_SIZE} val examples")
    train_examples = train_examples[:DEBUG_TRAIN_SIZE]
    val_examples   = val_examples[:DEBUG_VAL_SIZE]
else:
    print("\nFULL TRAINING MODE")

print(f"Train examples: {len(train_examples)}")
print(f"Val examples:   {len(val_examples)}")

print("\nLoading tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print("\nBuilding training dataset...")
train_dataset = SQuADDataset(train_examples, tokenizer)

print("\nBuilding validation dataset...")
val_dataset = SQuADDataset(val_examples, tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)

print(f"\nTrain batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")

# load pretrained BERT for question answering
print("\nLoading BertForQuestionAnswering...")
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)

# unfreeze all layers - following Devlin et al. 2019
for param in model.parameters():
    param.requires_grad = True

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

'''
Warmup phase: learning rate increases gradually for the first
WARMUP_RATIO of steps to avoid destroying pretrained BERT weights.
After warmup, learning rate decays linearly to zero (cooldown).
'''
total_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * WARMUP_RATIO),
    num_training_steps=total_steps
)

# Training
print("\nStarting fine-tuning...")
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total      = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids       = batch['input_ids'].to(device)
        attention_mask  = batch['attention_mask'].to(device)
        token_type_ids  = batch['token_type_ids'].to(device)
        start_positions = batch['start_position'].to(device)
        end_positions   = batch['end_position'].to(device)

        # clear gradients from previous batch
        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions
        )

        '''
        BertForQuestionAnswering automatically computes:
        loss = (start_loss + end_loss) / 2
        '''
        loss = outputs.loss
        loss.backward()

        # clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total      += 1

        if (batch_idx + 1) % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1} | "
                  f"Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"LR: {current_lr:.2e}")

    avg_train_loss = total_loss / total
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {avg_train_loss:.4f}")

    val_loss = evaluate(model, val_loader)

    # accumulator pattern - save only the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save_pretrained("bert_qa_finetuned")
        tokenizer.save_pretrained("bert_qa_finetuned")
        print(f"  ✅ New best model saved! Val loss: {val_loss:.4f}")
    else:
        print(f"  No improvement. Best so far: {best_val_loss:.4f}")
    print()

print(f"Training complete. Best val loss: {best_val_loss:.4f}")
print("Model saved to ./bert_qa_finetuned/")