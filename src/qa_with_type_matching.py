'''
qa_with_type_matching.py
Author: Keith Paschal, Mahmood Ahmed
Class: CSC 790, NLP

Evaluates BertForQuestionAnswering on SQuAD 1.1 validation set with
a novel type-matching layer on top of the neural span extraction.

Pipeline per question:
    1. Classify question type via fine-tuned BERT classifier
       → expected type (PERSON, DATE, LOC, QUANTITY, ORG, DESC)
    2. Extract answer span via fine-tuned BertForQuestionAnswering
       → predicted span + logit scores
    3. Run spaCy NER on predicted span
       → actual entity type of the extracted span
    4. Collect top-5 candidate spans from BERT (by logit score)
    5. Re-rank candidates: final_score = bert_score * (1 + ALPHA * type_match)
       where type_match = 1 if spaCy NER on that span matches expected type
    6. Compute EM and F1 under three conditions for ablation table:
       - Baseline BERT (top-1 span, no type matching)
       - Hard filter  (top-1 span among type-matching candidates only)
       - Soft boost   (top-5 re-ranked by type-match bonus — our method)

Hyperparameters must match finetune_bert_qa.py:
    MAX_LENGTH = 384
    DOC_STRIDE = 128

DESC is the catch-all label — a DESC prediction applies no bonus
and no penalty, since it covers heterogeneous answer types.
'''

import re
import string
import torch
import spacy
from collections import Counter
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForQuestionAnswering,
    BertForSequenceClassification,
)

# ── configuration ──────────────────────────────────────────────────────────────
DEBUG          = False        # set True for a quick 200-example sanity check
DEBUG_VAL_SIZE = 200

MAX_LENGTH        = 384       # must match finetune_bert_qa.py
DOC_STRIDE        = 128       # must match finetune_bert_qa.py
MAX_ANSWER_LENGTH = 30        # max tokens in a valid answer span
TOP_N             = 5         # number of candidate spans to collect for re-ranking
ALPHA             = 0.15      # soft-boost strength; tune on dev set

# paths to saved models
QA_MODEL_PATH   = "./bert_qa_finetuned"
CLS_MODEL_PATH  = "./question_classifier"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate labels to ID and the reverse.
LABEL2ID = {
    "PERSON":   0,
    "DATE":     1,
    "LOC":      2,
    "QUANTITY": 3,
    "ORG":      4,
    "DESC":     5,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# maps spaCy NER labels to 6 Class defined above
SPACY_TO_LABEL = {
    "PERSON":      "PERSON",
    "DATE":        "DATE",
    "TIME":        "DATE",
    "GPE":         "LOC",
    "LOC":         "LOC",
    "FAC":         "LOC",
    "MONEY":       "QUANTITY",
    "QUANTITY":    "QUANTITY",
    "CARDINAL":    "QUANTITY",
    "ORDINAL":     "QUANTITY",
    "PERCENT":     "QUANTITY",
    "ORG":         "ORG",
    "NORP":        "ORG",
    "EVENT":       "DESC",
    "WORK_OF_ART": "DESC",
    "LANGUAGE":    "DESC",
}

#print("Loading spaCy...")
nlp = spacy.load("en_core_web_sm")

#print("Loading QA tokenizer and model...")
qa_tokenizer = BertTokenizer.from_pretrained(QA_MODEL_PATH)
qa_model     = BertForQuestionAnswering.from_pretrained(QA_MODEL_PATH).to(device)
qa_model.eval()

#print("Loading question classifier...")
cls_tokenizer = BertTokenizer.from_pretrained(CLS_MODEL_PATH)
cls_model     = BertForSequenceClassification.from_pretrained(CLS_MODEL_PATH).to(device)
cls_model.eval()

#print("Loading SQuAD validation set...")
dataset     = load_dataset("squad")
val_examples = list(dataset["validation"])

if DEBUG:
    print(f"DEBUG MODE — using first {DEBUG_VAL_SIZE} examples")
    val_examples = val_examples[:DEBUG_VAL_SIZE]

print(f"Evaluating on {len(val_examples)} examples\n")


def normalize_answer(s):
    """Lower, strip punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(prediction, ground_truths):
    pred_norm = normalize_answer(prediction)
    return int(any(pred_norm == normalize_answer(gt) for gt in ground_truths))


def f1_score(prediction, ground_truths):
    pred_tokens = normalize_answer(prediction).split()
    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        common    = Counter(pred_tokens) & Counter(gt_tokens)
        num_same  = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall    = num_same / len(gt_tokens)
        f1        = 2 * precision * recall / (precision + recall)
        best_f1   = max(best_f1, f1)
    return best_f1


def classify_question(question: str) -> str:
    """
    Returns one of: PERSON, DATE, LOC, QUANTITY, ORG, DESC
    DESC means 'unknown / catch-all' — no bonus is applied.
    """
    with torch.no_grad():
        inputs = cls_tokenizer(
            question,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        outputs     = cls_model(**inputs)
        predicted_id = torch.argmax(outputs.logits, dim=1).item()
    return ID2LABEL[predicted_id]


def get_span_type(span_text: str) -> str:
    """
    Returns the mapped label for the first named entity found in span_text,
    or DESC if nothing is found (no match → no bonus, no penalty).
    """
    doc = nlp(span_text)
    if doc.ents:
        return SPACY_TO_LABEL.get(doc.ents[0].label_, "DESC")
    return "DESC"


def extract_candidates(question: str, context: str, top_n: int = TOP_N):
    """
    Tokenizes question+context with a sliding window (DOC_STRIDE) to handle
    contexts longer than MAX_LENGTH.

    Returns a list of (answer_text, bert_score) tuples, sorted by bert_score
    descending, deduplicated by answer text, up to top_n entries.

    bert_score = start_logit + end_logit for a given (start, end) pair.
    The caller re-ranks these with the type-match bonus.
    """
    inputs = qa_tokenizer(
        question,
        context,
        truncation=True,
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )

    offset_mappings = inputs.pop("offset_mapping")
    inputs.pop("overflow_to_sample_mapping", None)

    input_ids_all      = inputs["input_ids"].to(device)
    attention_mask_all = inputs["attention_mask"].to(device)
    token_type_ids_all = inputs.get("token_type_ids")
    if token_type_ids_all is not None:
        token_type_ids_all = token_type_ids_all.to(device)

    # collect all (score, answer_text) across every chunk
    all_candidates = []

    with torch.no_grad():
        for chunk_idx in range(input_ids_all.size(0)):
            input_ids      = input_ids_all[chunk_idx].unsqueeze(0)
            attention_mask = attention_mask_all[chunk_idx].unsqueeze(0)
            token_type_ids = (
                token_type_ids_all[chunk_idx].unsqueeze(0)
                if token_type_ids_all is not None else None
            )

            outputs = qa_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            start_logits = outputs.start_logits[0]
            end_logits   = outputs.end_logits[0]

            sequence_ids = inputs.sequence_ids(chunk_idx) if hasattr(inputs, "sequence_ids") else None
            offset_map   = offset_mappings[chunk_idx].tolist()

            # find valid context token range
            context_start = None
            context_end   = None
            if sequence_ids is not None:
                for i, sid in enumerate(sequence_ids):
                    if sid == 1:
                        if context_start is None:
                            context_start = i
                        context_end = i
            else:
                ttype = token_type_ids[0].tolist() if token_type_ids is not None else [0] * input_ids.size(1)
                for i, t in enumerate(ttype):
                    if t == 1:
                        if context_start is None:
                            context_start = i
                        context_end = i

            if context_start is None:
                continue

            # score every valid (start, end) pair and keep top_n from this chunk
            chunk_candidates = []
            for s in range(context_start, context_end + 1):
                for e in range(s, min(s + MAX_ANSWER_LENGTH, context_end + 1)):
                    score      = start_logits[s].item() + end_logits[e].item()
                    char_start = offset_map[s][0]
                    char_end   = offset_map[e][1]
                    answer     = context[char_start:char_end].strip()
                    if answer:
                        chunk_candidates.append((score, answer))

            # keep only the best top_n from this chunk to avoid O(n²) blowup
            chunk_candidates.sort(key=lambda x: x[0], reverse=True)
            all_candidates.extend(chunk_candidates[:top_n])

    # global sort, deduplicate by answer text (keep highest score per text)
    seen   = {}
    for score, answer in sorted(all_candidates, key=lambda x: x[0], reverse=True):
        if answer not in seen:
            seen[answer] = score
        if len(seen) >= top_n:
            break

    # return as list of (answer_text, score), best first
    ranked = sorted(seen.items(), key=lambda x: x[1], reverse=True)
    return ranked  # [(answer_text, score), ...]


# Three conditions evaluated per question:
#
#   baseline — top-1 BERT span, no type info used
#   hard     — top-1 span among candidates whose NER type matches expected type;
#              falls back to top-1 BERT span if no candidate matches
#              (avoids throwing away the answer entirely for DESC questions)
#   soft     — re-rank all top-N candidates by:
#                final_score = bert_score * (1 + ALPHA * type_match)
#              pick the highest re-ranked candidate

metrics = {
    "baseline": {"em": 0.0, "f1": 0.0},
    "hard":     {"em": 0.0, "f1": 0.0},
    "soft":     {"em": 0.0, "f1": 0.0},
}

# per-type accumulators for soft boost (to show where it helps most)
type_metrics = {t: {"em": 0.0, "f1": 0.0, "n": 0}
                for t in LABEL2ID}

total = len(val_examples)

for i, example in enumerate(val_examples):
    question      = example["question"]
    context       = example["context"]
    ground_truths = example["answers"]["text"]

    # Figure out the type we expect
    expected_type = classify_question(question)

    # get top 5 (default) candidate spans
    candidates = extract_candidates(question, context, top_n=TOP_N)
    # candidates = [(answer_text, bert_score), ...] sorted best-first
    # guaranteed non-empty (extract_candidates returns at least 1 if context
    # has any tokens, but guard anyway)
    if not candidates:
        candidates = [("", 0.0)]

    # This gives the first span of answer used when it is DESC (no bonus)
    baseline_answer = candidates[0][0]

    # See if the span matches DESC...
    if expected_type == "DESC":
        # DESC is the catch-all; don't filter, take BERT's top answer
        hard_answer = baseline_answer
    else:
        hard_answer = None
        for answer_text, _ in candidates:
            if get_span_type(answer_text) == expected_type:
                hard_answer = answer_text
                break
        if hard_answer is None:
            # no candidate matched — fall back to BERT top-1
            hard_answer = baseline_answer

    # If not DESC, add the matching bonus to maybe give a new value
    def reranked_score(answer_text, bert_score):
        if expected_type == "DESC":
            return bert_score   # no bonus for catch-all type
        # Don't reward a candidate that is a strict substring of the baseline —
        # spaCy tends to tag short numeric/entity fragments (e.g. "84" from
        # "84 hours") as the right type, promoting degraded answers.
        if answer_text != baseline_answer and answer_text in baseline_answer:
            return bert_score   # no bonus for substring degradations
        span_type  = get_span_type(answer_text)
        type_match = (span_type != "DESC" and span_type == expected_type)
        return bert_score * (1.0 + ALPHA * int(type_match))

    reranked    = sorted(candidates,
                         key=lambda c: reranked_score(c[0], c[1]),
                         reverse=True)
    soft_answer = reranked[0][0]

    # Compute em & f1
    metrics["baseline"]["em"] += exact_match(baseline_answer, ground_truths)
    metrics["baseline"]["f1"] += f1_score(baseline_answer, ground_truths)

    metrics["hard"]["em"] += exact_match(hard_answer, ground_truths)
    metrics["hard"]["f1"] += f1_score(hard_answer, ground_truths)

    metrics["soft"]["em"] += exact_match(soft_answer, ground_truths)
    metrics["soft"]["f1"] += f1_score(soft_answer, ground_truths)

    # per-type tracking (soft condition)
    type_metrics[expected_type]["em"] += exact_match(soft_answer, ground_truths)
    type_metrics[expected_type]["f1"] += f1_score(soft_answer, ground_truths)
    type_metrics[expected_type]["n"]  += 1

    if (i + 1) % 500 == 0:
        pct     = (i + 1) / total * 100
        cur_em  = metrics["baseline"]["em"] / (i + 1) * 100
        cur_f1  = metrics["baseline"]["f1"] / (i + 1) * 100
        soft_em = metrics["soft"]["em"] / (i + 1) * 100
        soft_f1 = metrics["soft"]["f1"] / (i + 1) * 100
        print(f"  [{pct:5.1f}%] {i+1}/{total} | "
              f"Baseline EM: {cur_em:.2f} F1: {cur_f1:.2f} | "
              f"Soft EM: {soft_em:.2f} F1: {soft_f1:.2f}")

# Print results!!
print("\n" + "="*60)
print("ABLATION RESULTS — SQuAD 1.1 Validation")
print("="*60)
print(f"{'System':<35} {'EM':>8} {'F1':>8}")
print("-"*60)

for condition, label in [
    ("baseline", "BERT baseline (no type matching)"),
    ("hard",     "+ NER filter (hard, discard mismatch)"),
    ("soft",     "+ NER boost  (soft, alpha={:.2f})".format(ALPHA)),
]:
    em = metrics[condition]["em"] / total * 100
    f1 = metrics[condition]["f1"] / total * 100
    print(f"  {label:<33} {em:>7.2f}  {f1:>7.2f}")

print("\n" + "="*60)
print("PER-TYPE BREAKDOWN (soft boost condition)")
print("="*60)
print(f"  {'Type':<12} {'N':>6} {'EM':>8} {'F1':>8}")
print("-"*60)

for type_name in ["PERSON", "DATE", "LOC", "QUANTITY", "ORG", "DESC"]:
    t = type_metrics[type_name]
    if t["n"] == 0:
        continue
    em = t["em"] / t["n"] * 100
    f1 = t["f1"] / t["n"] * 100
    print(f"  {type_name:<12} {t['n']:>6} {em:>7.2f}  {f1:>7.2f}")

print("="*60)
print(f"\nALPHA (soft boost strength): {ALPHA}")
print(f"TOP_N (candidates re-ranked): {TOP_N}")
print("Note: soft boost score = bert_score * (1 + ALPHA * type_match)")
print("      DESC questions receive no bonus (catch-all type).")
print("      Hard filter falls back to BERT top-1 if no candidate matches.")