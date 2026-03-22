'''
generate_question_classifier
Author: Keith Paschal, Mahmood Ahmed
Class: CSC 790, NLP

This script uses spaCy and BERT to learn labels for expected data types of questions from the
SQuAD 1.1 dataset.  The Labels are initially mapped to categories of: PERSON, DATE, LOC, QUANTITY,
ORG and DESC.  The training of this resulted in a 76-79% accuracy ot mapping the labels to the
questions in the SQuAD dataset.
'''

import spacy
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random

def get_label_from_answer(answer_text, context, question):
    """
    This function will return the votes for a label based on
    the answer text, context and question.
    ===========
    Parameters:
    ===========
    answer_text: str
        This is the text of the answer in the SQuAD dataset
    context: str
        This is the paragraph or text to read from in the 
        SQuAD dataset
    question: str
        This is the question asked in the SQuAD dataset
    
    ===========
    Returns
    ===========
    Vote output for the Label, or DESC if unsure
    """
    votes = Counter()
    
    # The first vote is based on hard-coded rules which helps BERT
    # figure out future versions of questions.  This is weighted double
    # of the other votes as it helped out a lot during development.
    
    q = question.lower().strip()
    if q.startswith("who") or "to whom" in q or "by whom" in q:
        votes["PERSON"] += 2
    elif (q.startswith("when") or "what year" in q or 
          "in what year" in q or "in which year" in q):
        votes["DATE"] += 2
    elif (q.startswith("where") or "in what city" in q or 
          "in which city" in q or "in what country" in q):
        votes["LOC"] += 2
    elif q.startswith("how much") or q.startswith("how many"):
        votes["QUANTITY"] += 2

    # I noticed in the dataset, if the answer starts with a, an, the, some
    # or is longer than 4 tokens, then it tends to be a descriptive answer

    answer_lower = answer_text.lower().strip()
    if answer_lower.startswith(("a ", "an ", "the ", "some ")):
        votes["DESC"] += 2
    if len(answer_text.split()) > 4:
        votes["DESC"] += 2
    elif len(answer_text.split()) > 2:
        votes["DESC"] += 1

    # Started with just spaCy NER, but this proved to be the least effective on its
    # own because the answer fragments lack enough context, but adds some value when
    # used in conjumction with the other votes
    
    doc_answer = nlp(answer_text)
    if doc_answer.ents:
        ner_label = SPACY_TO_LABEL.get(doc_answer.ents[0].label_, None)
        if ner_label:
            votes[ner_label] += 1

    # Calculate the votes and return the most common Label or DESC
    
    if votes:
        return votes.most_common(1)[0][0]
    return "DESC"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(f"Using device: {device}")

# Label mapping, IDs to Labels and the other way around.  Also defines labels for
# spaCy mappings.

LABEL2ID = {
    "PERSON":   0,
    "DATE":     1,
    "LOC":      2,
    "QUANTITY":  3,
    "ORG":      4,
    "DESC":     5
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

SPACY_TO_LABEL = {
    "PERSON":   "PERSON",
    "DATE":     "DATE",
    "TIME":     "DATE",
    "GPE":      "LOC",
    "LOC":      "LOC",
    "FAC":      "LOC",
    "MONEY":    "QUANTITY",
    "QUANTITY": "QUANTITY",
    "CARDINAL": "QUANTITY",
    "ORDINAL":  "QUANTITY",
    "PERCENT":  "QUANTITY",
    "ORG":      "ORG",
    "NORP":     "ORG",
    "EVENT":    "DESC",
    "WORK_OF_ART": "DESC",
    "LANGUAGE":  "DESC",
}

#print("Loading spaCy...")
# Load spaCy (which is trained with a neural network)

nlp = spacy.load("en_core_web_sm")

# Load the SQuAD dataset, split the training questions, labels, context,
# and questions

#print("Loading SQuAD and generating labels...")
dataset = load_dataset("squad")

train_questions = []
train_labels = []

for i, ex in enumerate(dataset['train']):
    label = get_label_from_answer(
        ex['answers']['text'][0],
        ex['context'],
        ex['question']
    )
    train_questions.append(ex['question'])
    train_labels.append(LABEL2ID[label])
    
    # Use index to print out status so you know it is running.
    
    if (i + 1) % 10000 == 0:
        print(f"  Processed {i+1}/{len(dataset['train'])}...")

# Printed distributions.  DESC has many more values than the rest which are hard
# to classify, so we will need to also add weights.  DESC is the cathc-all category
# for answers that don't fit another type.  Due to the imbalance of the categories
# we had to add weights, otherwise DESC would be the only thing that gets predicted.

label_counts = Counter(train_labels)
print("\nQuestion type distribution:")
for label_id in range(len(LABEL2ID)):
    count = label_counts[label_id]
    print(f"  {ID2LABEL[label_id]:10} ({label_id}): "
          f"{count:6} ({count/len(train_labels)*100:.1f}%)")

total = len(train_labels)
class_weights = []
print("\nClass weights:")
for label_id in range(len(LABEL2ID)):
    count = label_counts[label_id]
    weight = total / (len(LABEL2ID) * count)
    class_weights.append(weight)
    print(f"  {ID2LABEL[label_id]:10}: {weight:.3f}")
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

class QuestionTypeDataset(Dataset):
    '''
    This is the class for the QuestionTypeDataset.  It needs to be iterable
    so we need len and getitem methods also.
    '''
    def __init__(self, questions, labels, tokenizer, max_length=64):
        self.encodings = tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

# Start the real work.  Tokenize the questions using the small BERT model
# due to GPU limitations.

#print("Tokenizing questions...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

combined = list(zip(train_questions, train_labels))
random.shuffle(combined)
questions_shuffled, labels_shuffled = zip(*combined)

# Split 80 percent training and 20 percent validation

split = int(0.8 * len(questions_shuffled))
train_q = list(questions_shuffled[:split])
train_l = list(labels_shuffled[:split])
val_q = list(questions_shuffled[split:])
val_l = list(labels_shuffled[split:])

#print(f"Train: {len(train_q)} | Val: {len(val_q)}")

# Stick training data in the dataset class.  Makes it easier to work with while testing.

train_dataset = QuestionTypeDataset(train_q, train_l, tokenizer)
val_dataset = QuestionTypeDataset(val_q, val_l, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# This is where we load up BERT with the labels to learn.  Dropout is configured here
# as it worked better than the default of .1.  Without dropout the model memorized the
# labels.

#print("Loading BERT question classifier...")
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(LABEL2ID),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3
).to(device)

for param in model.bert.parameters():
    param.requires_grad = True

optimizer = Adam(model.parameters(), lr=2e-5, weight_decay=0.01)

EPOCHS = 3
total_steps = len(train_loader) * EPOCHS
# Warmup: The learning rate gradually increases for the first 10% of the
# steps to avoid 'catastrophic forgetting', which I did in development.  After
# the warmup the learning rate decays back down towards zero.

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss(weight=class_weights)

def evaluate(model, loader):
    '''
    This is the main model eval loop
    '''
    model.eval()
    correct = 0
    total = 0
    class_correct = Counter()
    class_total = Counter()
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += len(labels)
            
            for pred, label in zip(predictions, labels):
                class_total[label.item()] += 1
                if pred == label:
                    class_correct[label.item()] += 1
    
    overall_acc = correct / total
    # We can print the accuracy to determine which model to save.  I found it was overfitting the data
    
    #print(f"  Overall accuracy: {overall_acc:.4f}")
    #print(f"  Per-class accuracy:")
    #for label_id in range(len(LABEL2ID)):
    #    if class_total[label_id] > 0:
    #        acc = class_correct[label_id] / class_total[label_id]
    #        print(f"{ID2LABEL[label_id]:10}: {acc:.4f} ")
    #        print(f"({class_correct[label_id]}/{class_total[label_id]})")
    return overall_acc

# Start training
# Use accumulator pattern to save only the best model

#print("\nTraining question type classifier...")

best_val_acc = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        predictions = torch.argmax(outputs.logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += len(labels)
        total_loss += loss.item()
        
        # Print out the batches every 50 steps so we know it is working
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} "
                  f"| Loss: {loss.item():.4f}")
    
    train_acc = correct / total
    #print(f"\nEpoch {epoch+1}/{EPOCHS}")
    #print(f"  Train Loss: {total_loss/len(train_loader):.4f} ")
    #print(f"| Train Acc: {train_acc:.4f}")
    #print("  Validation:")
    val_acc = evaluate(model, val_loader)
    
    # save only if best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained("question_classifier")
        tokenizer.save_pretrained("question_classifier")
        #print(f"New best model saved! Val acc: {val_acc:.4f}")
    #else:
        #print(f"  No improvement. Best so far: {best_val_acc:.4f}")
    #print()

print(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
print("Best model saved to ./question_classifier/")

# Found some examples which were initially wrong and so use them as a 'test' to determine
# how much better the model got

#print("\n=== Testing classifier ===")
model.eval()
test_questions = [
    "Who was the president of Notre Dame in 2012?",
    "When did the Scholastic Magazine begin publishing?",
    "Where is the headquarters of the Holy Cross?",
    "How many students attend Notre Dame?",
    "What is the Grotto at Notre Dame?",
    "To whom did the Virgin Mary appear in 1858?",
    "In what year was the College of Engineering formed?",
    "What organization oversees Notre Dame?",
    "What company did Ray Kroc own?",
    "What is a common name for the Touchdown Jesus mural?",
]

with torch.no_grad():
    inputs = tokenizer(
        test_questions,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors='pt'
    ).to(device)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    probs = torch.softmax(outputs.logits, dim=1)
    
    for q, pred, prob in zip(test_questions, predictions, probs):
        label = ID2LABEL[pred.item()]
        confidence = prob[pred].item()
        #print(f"Q: {q}")
        #print(f"   Predicted type: {label} ({confidence*100:.1f}% confident)\n")