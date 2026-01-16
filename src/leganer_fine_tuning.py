"""
Script used to fine-tune Transformer based models for NER on the COMtext SR legal dataset.

Supports two execution modes:
- Standard: Train on fixed train/test split (single run per model)
- CV: 10-Fold Cross-Validation with averaged results

Implements two evaluation settings:
- Default: Type-level matching (ignores BIO prefixes)
- Strict: Exact tag matching with separate B/I reporting

Usage:
    uv run python src/reproduce_results.py --mode standard
    uv run python src/reproduce_results.py --mode cv
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import cyrtranslit
import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import KFold, train_test_split
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Load seqeval metric once at module level
seqeval_metric = evaluate.load("seqeval")

# ===== CONFIGURATION =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

FILE_PATH = DATA_DIR / "comtext.sr.legal.ekavica.conllu"

# Model configurations
MODELS = {
    "BERTić": {
        "name": "classla/bcms-bertic",
        "display_name": "BERTić-COMtext-SR-legal-NER-ijekavica",
        "use_cyrillic": False,
    },
    "SrBERTa": {
        "name": "nemanjaPetrovic/SrBERTa",
        "display_name": "SrBERTa",
        "use_cyrillic": True,
    },
    "mmBert-base": {
        "name": "jhu-clsp/mmBERT-base",
        "display_name": "mmBERT-base",
        "use_cyrillic": False,
    },
    "XLM-R-Bertic": {
        "name": "classla/xlm-r-bertic",
        "display_name": "XLM-R-Bertic",
        "use_cyrillic": False,
    },
}


# Training hyperparameters (matching reference configuration)
SEED = 64
NUM_EPOCHS = 10
LEARNING_RATE = 3e-5  # 4e-5 for bertic and srberta
TRAIN_BATCH_SIZE = 16  # 8 for bertic and srberta by reference
EVAL_BATCH_SIZE = 64
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01  # 0.0 for bertic and srberta by reference
NUM_FOLDS = 10
TEST_SIZE = 0.1
USE_EARLY_STOPPING = True

# Entity types to report (mapped from dataset labels)
ENTITY_TYPES_DISPLAY = [
    "PER",
    "LOC",
    "ADR",
    "COURT",
    "INST",
    "COM",
    "OTHORG",
    "LAW",
    "REF",
    "IDPER",
    "IDCOM",
    "IDTAX",
    "NUMACC",
    "NUMDOC",
    "NUMCAR",
    "NUMPLOT",
    "IDOTH",
    "CONTACT",
    "DATE",
    "MONEY",
    "MISC",
]

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(exist_ok=True)
(OUTPUT_DIR / "models").mkdir(exist_ok=True)


# ===== DATA LOADING FUNCTION =====
def load_corpus_sentences(
    filepath_list: list[Path], use_cyrillic: bool = False, conllup: bool = False
) -> tuple[list[dict], set]:
    """
    Parses CoNLL-U files into a list of sentence dictionaries.
    Ensures tokens are grouped by sentence to preserve context.

    Args:
        filepath_list: List of paths to CoNLL-U files
        use_cyrillic: If True, transliterate to Cyrillic (for SrBERTa)
        conllup: If True, use column 4 for NER tag, else column 3

    Returns:
        Tuple of (dataset, wordlist)
    """
    dataset = []
    wordlist = set()
    global_sent_index = 0

    for filepath in filepath_list:
        with open(filepath, encoding="utf-8") as f:
            content = f.read().strip()

        raw_sentences = content.split("\n\n")

        for raw_sent in raw_sentences:
            if not raw_sent.strip():
                continue

            sentence_id_str = ""
            document_id = None
            tokens = []
            labels = []

            lines = raw_sent.split("\n")

            for line in lines:
                line = line.strip()
                if line.startswith("#"):
                    if "sent_id" in line:
                        sentence_id_str = line.split("=")[1].strip()
                    elif "newdoc id" in line:
                        document_id = line.split("=")[1].strip()
                    continue

                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                token = parts[1]
                lemma = parts[2] if len(parts) > 2 else token

                if not conllup:
                    tag = parts[3] if len(parts) > 3 else "O"
                else:
                    try:
                        tag = parts[4]
                    except IndexError:
                        tag = "O"

                # Transliteration for SrBERTa (Cyrillic)
                if use_cyrillic:
                    token = cyrtranslit.to_cyrillic(token, "sr")
                    lemma = cyrtranslit.to_cyrillic(lemma, "sr")

                tokens.append(token)
                labels.append(tag)
                wordlist.add(token)

            if len(tokens) > 0:
                dataset.append(
                    {
                        "sentence_id": global_sent_index,
                        "conllu_id": sentence_id_str,
                        "document_id": document_id,
                        "words": tokens,
                        "labels": labels,
                    }
                )
                global_sent_index += 1

    return dataset, wordlist


# ===== TOKENIZATION & ALIGNMENT FUNCTION =====
def tokenize_and_align_labels(examples: dict, tokenizer, label2id: dict, max_length: int = 512) -> dict:
    """
    Tokenizes sentences and aligns labels with sub-word tokens.
    Assigns -100 to special tokens and subsequent sub-word pieces.
    """
    tokenized_inputs = tokenizer(examples["words"], truncation=True, is_split_into_words=True, max_length=max_length)

    labels = []
    for i, label_seq in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label_seq[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# ===== HELPER FUNCTIONS =====
def convert_ids_to_labels(
    predictions: np.ndarray, labels: np.ndarray, id2label: dict
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Convert prediction and label IDs to string labels, filtering out -100 tokens.

    Args:
        predictions: Model predictions (batch_size, seq_len, num_labels) or (batch_size, seq_len)
        labels: Ground truth labels (batch_size, seq_len)
        id2label: Mapping from label ID to label string

    Returns:
        Tuple of (true_labels_list, pred_labels_list) as nested lists of strings
    """
    # Handle logits vs already-argmaxed predictions
    if predictions.ndim == 3:
        preds = np.argmax(predictions, axis=2)
    else:
        preds = predictions

    true_labels_list = []
    pred_labels_list = []

    for i, label_seq in enumerate(labels):
        true_seq = []
        pred_seq = []
        for j, label_id in enumerate(label_seq):
            if label_id != -100:
                true_seq.append(id2label[label_id])
                pred_seq.append(id2label[preds[i][j]])
        true_labels_list.append(true_seq)
        pred_labels_list.append(pred_seq)

    return true_labels_list, pred_labels_list


def compute_token_accuracy(true_labels: list[list[str]], pred_labels: list[list[str]]) -> float:
    """
    Compute token-level accuracy.

    Args:
        true_labels: Nested list of true label strings
        pred_labels: Nested list of predicted label strings

    Returns:
        Token-level accuracy as a float
    """
    true_flat = [t for seq in true_labels for t in seq]
    pred_flat = [p for seq in pred_labels for p in seq]

    if not true_flat:
        return 0.0

    correct = sum(1 for t, p in zip(true_flat, pred_flat) if t == p)
    return correct / len(true_flat)


def compute_o_class_metrics(true_labels: list[list[str]], pred_labels: list[list[str]]) -> dict:
    """
    Compute precision, recall, and F1 for the O class at token level.
    Seqeval doesn't report O class metrics, so we compute them separately.

    Args:
        true_labels: Nested list of true label strings
        pred_labels: Nested list of predicted label strings

    Returns:
        Dict with precision, recall, f1, and support for O class
    """
    true_flat = [t for seq in true_labels for t in seq]
    pred_flat = [p for seq in pred_labels for p in seq]

    o_tp = sum(1 for t, p in zip(true_flat, pred_flat) if t == "O" and p == "O")
    o_fp = sum(1 for t, p in zip(true_flat, pred_flat) if t != "O" and p == "O")
    o_fn = sum(1 for t, p in zip(true_flat, pred_flat) if t == "O" and p != "O")

    precision = o_tp / (o_tp + o_fp) if (o_tp + o_fp) > 0 else 0.0
    recall = o_tp / (o_tp + o_fn) if (o_tp + o_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": sum(1 for t in true_flat if t == "O"),
    }


# ===== EVALUATION FUNCTIONS (using HuggingFace evaluate + seqeval) =====
def compute_metrics_default(true_labels: list[list[str]], pred_labels: list[list[str]]) -> dict:
    """
    Compute metrics with default (entity-level) matching using seqeval.
    Uses 'default' mode which evaluates based on entity spans.

    Returns:
        Dict with accuracy, macro_f1_with_o, macro_f1_without_o, and per-class metrics
    """
    # Compute seqeval metrics with default mode (entity-level evaluation)
    results = seqeval_metric.compute(
        predictions=pred_labels,
        references=true_labels,
        mode="default",
        zero_division=0,
    )

    # Token-level accuracy using evaluate library
    accuracy = compute_token_accuracy(true_labels, pred_labels)

    # Extract per-class metrics from seqeval results
    per_class = {}
    entity_f1_scores = []

    for key, value in results.items():
        if isinstance(value, dict) and "f1" in value:
            per_class[key] = {
                "precision": value["precision"],
                "recall": value["recall"],
                "f1": value["f1"],
                "support": value["number"],
            }
            entity_f1_scores.append(value["f1"])

    # Add O class metrics (seqeval doesn't report O)
    o_metrics = compute_o_class_metrics(true_labels, pred_labels)
    per_class["O"] = o_metrics

    # Macro F1 without O (from seqeval overall_f1 which excludes O)
    macro_f1_without_o = results.get("overall_f1", 0.0)

    # Macro F1 with O (include O class in the average)
    all_f1_scores = entity_f1_scores + [o_metrics["f1"]]
    macro_f1_with_o = np.mean(all_f1_scores) if all_f1_scores else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1_with_o": macro_f1_with_o,
        "macro_f1_without_o": macro_f1_without_o,
        "overall_precision": results.get("overall_precision", 0.0),
        "overall_recall": results.get("overall_recall", 0.0),
        "per_class": per_class,
    }


def compute_metrics_strict(true_labels: list[list[str]], pred_labels: list[list[str]]) -> dict:
    """
    Compute metrics with strict (exact) matching using seqeval.
    Uses 'strict' mode where entity boundaries must match exactly.
    Reports B- and I- tags separately at the token level.

    Returns:
        Dict with accuracy, macro_f1_with_o, macro_f1_without_o, and per-class F1 (B/I separate)
    """
    # Compute seqeval metrics with strict mode for entity-level metrics
    results = seqeval_metric.compute(
        predictions=pred_labels,
        references=true_labels,
        mode="strict",
        zero_division=0,
    )

    # Token-level accuracy using evaluate library
    accuracy = compute_token_accuracy(true_labels, pred_labels)

    # For strict mode with B/I reporting, we need token-level per-tag metrics
    # Use seqeval with scheme=IOBES or compute token-level metrics
    true_flat = [t for seq in true_labels for t in seq]
    pred_flat = [p for seq in pred_labels for p in seq]

    # Compute per-tag metrics at token level (B-X, I-X separately)
    all_tags = sorted(set(true_flat) | set(pred_flat))
    per_class = {}
    f1_scores_with_o = []
    f1_scores_without_o = []

    for tag in all_tags:
        tp = sum(1 for t, p in zip(true_flat, pred_flat) if t == tag and p == tag)
        fp = sum(1 for t, p in zip(true_flat, pred_flat) if t != tag and p == tag)
        fn = sum(1 for t, p in zip(true_flat, pred_flat) if t == tag and p != tag)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[tag] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for t in true_flat if t == tag),
        }

        f1_scores_with_o.append(f1)
        if tag != "O":
            f1_scores_without_o.append(f1)

    # Macro F1 scores
    macro_f1_with_o = np.mean(f1_scores_with_o) if f1_scores_with_o else 0.0
    macro_f1_without_o = np.mean(f1_scores_without_o) if f1_scores_without_o else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1_with_o": macro_f1_with_o,
        "macro_f1_without_o": macro_f1_without_o,
        "overall_precision": results.get("overall_precision", 0.0),
        "overall_recall": results.get("overall_recall", 0.0),
        "per_class": per_class,
    }


def compute_trainer_metrics(eval_preds, id2label: dict) -> dict:
    """Compute metrics for Trainer callback during training using seqeval."""
    predictions, labels = eval_preds

    # Use helper function to convert IDs to labels
    true_labels_list, pred_labels_list = convert_ids_to_labels(predictions, labels, id2label)

    # Use seqeval for evaluation during training
    results = seqeval_metric.compute(
        predictions=pred_labels_list,
        references=true_labels_list,
        mode="default",
        zero_division=0,
    )

    # Token-level accuracy using evaluate library
    accuracy = compute_token_accuracy(true_labels_list, pred_labels_list)

    return {
        "accuracy": accuracy,
        "f1": results.get("overall_f1", 0.0),
        "precision": results.get("overall_precision", 0.0),
        "recall": results.get("overall_recall", 0.0),
    }


# ===== TRAINING FUNCTION =====
def train_and_evaluate(
    train_data: list[dict],
    test_data: list[dict],
    model_config: dict,
    label2id: dict,
    id2label: dict,
    output_base: Path,
    fold_num: int | None = None,
) -> tuple[dict, dict]:
    """
    Train a model and evaluate using both default and strict settings.

    Returns:
        Tuple of (default_metrics, strict_metrics)
    """
    model_name = model_config["name"]

    fold_suffix = f"_fold_{fold_num}" if fold_num is not None else ""
    output_path = output_base / f"{model_name.replace('/', '_')}{fold_suffix}"

    print(f"  Loading tokenizer: {model_name}")
    # RoBERTa-based models (like SrBERTa) need add_prefix_space=True for pre-tokenized inputs
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, add_prefix_space=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Create HuggingFace datasets
    hf_train = Dataset.from_pandas(pd.DataFrame(train_data))
    hf_test = Dataset.from_pandas(pd.DataFrame(test_data))

    print("  Tokenizing datasets...")
    tokenized_train = hf_train.map(
        tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer, "label2id": label2id}
    )
    tokenized_test = hf_test.map(
        tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer": tokenizer, "label2id": label2id}
    )

    # Initialize model
    print(f"  Loading model: {model_name}")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    )

    # Training arguments (matching reference configuration)
    args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        disable_tqdm=True,
        save_total_limit=2,
        seed=SEED,
    )

    # Create compute_metrics with id2label in closure
    def get_compute_metrics(id2label_map):
        def _compute_metrics(eval_preds):
            return compute_trainer_metrics(eval_preds, id2label_map)

        return _compute_metrics

    # Setup callbacks based on configuration
    callbacks = []
    if USE_EARLY_STOPPING:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=get_compute_metrics(id2label),
        callbacks=callbacks,  # Empty list when early stopping disabled (matches reference)
    )

    print("  Training...")
    trainer.train()

    # Save the best model (currently loaded in memory after training)
    print("  Saving best model...")
    trainer.save_model(str(output_path / "best_model"))
    tokenizer.save_pretrained(str(output_path / "best_model"))

    # Generate predictions
    print("  Generating predictions...")
    predictions, labels, _ = trainer.predict(tokenized_test)

    # Convert IDs back to tags using helper function
    true_labels_list, pred_labels_list = convert_ids_to_labels(predictions, labels, id2label)

    # Compute both evaluation settings
    default_metrics = compute_metrics_default(true_labels_list, pred_labels_list)
    strict_metrics = compute_metrics_strict(true_labels_list, pred_labels_list)

    # Cleanup intermediate checkpoints (keep only best_model)
    print("  Cleaning up intermediate checkpoints...")
    if output_path.exists():
        for checkpoint_dir in output_path.glob("checkpoint-*"):
            shutil.rmtree(checkpoint_dir, ignore_errors=True)

    return default_metrics, strict_metrics


# ===== RESULTS AGGREGATION =====
def aggregate_cv_results(fold_results: list[tuple[dict, dict]]) -> tuple[dict, dict]:
    """
    Aggregate results from multiple CV folds.

    Args:
        fold_results: List of (default_metrics, strict_metrics) tuples

    Returns:
        Tuple of (aggregated_default, aggregated_strict)
    """

    def aggregate_metrics(metrics_list: list[dict]) -> dict:
        # Aggregate scalar metrics
        result = {
            "accuracy": np.mean([m["accuracy"] for m in metrics_list]),
            "macro_f1_with_o": np.mean([m["macro_f1_with_o"] for m in metrics_list]),
            "macro_f1_without_o": np.mean([m["macro_f1_without_o"] for m in metrics_list]),
            "per_class": {},
        }

        # Aggregate per-class metrics
        all_classes = set()
        for m in metrics_list:
            all_classes.update(m["per_class"].keys())

        for cls in all_classes:
            f1_scores = [m["per_class"][cls]["f1"] for m in metrics_list if cls in m["per_class"]]
            if f1_scores:
                precision_scores = [m["per_class"][cls]["precision"] for m in metrics_list if cls in m["per_class"]]
                recall_scores = [m["per_class"][cls]["recall"] for m in metrics_list if cls in m["per_class"]]
                result["per_class"][cls] = {
                    "f1": np.mean(f1_scores),
                    "precision": np.mean(precision_scores),
                    "recall": np.mean(recall_scores),
                }

        return result

    default_list = [r[0] for r in fold_results]
    strict_list = [r[1] for r in fold_results]

    return aggregate_metrics(default_list), aggregate_metrics(strict_list)


# ===== CSV RESULTS SAVING =====
def save_results_to_csv(results: dict[str, dict[str, dict]], output_path: Path, mode: str = "cv") -> None:
    """
    Save results to CSV file.

    Args:
        results: Dict with structure {model_key: {"default": metrics, "strict": metrics}}
        output_path: Path to save CSV file
        mode: Training mode (cv or standard)
    """
    rows = []

    for model_key, model_results in results.items():
        model_display = MODELS[model_key]["display_name"]

        for eval_type in ["default", "strict"]:
            metrics = model_results.get(eval_type, {})
            if not metrics:
                continue

            # Base row with overall metrics
            base_row = {
                "model": model_key,
                "model_display": model_display,
                "eval_type": eval_type,
                "mode": mode,
                "accuracy": metrics.get("accuracy"),
                "macro_f1_with_o": metrics.get("macro_f1_with_o"),
                "macro_f1_without_o": metrics.get("macro_f1_without_o"),
            }

            # Add per-class F1 scores
            per_class = metrics.get("per_class", {})

            if eval_type == "default":
                # For default, add entity type F1 (without B-/I- prefix)
                for entity_type in ENTITY_TYPES_DISPLAY + ["O"]:
                    f1 = per_class.get(entity_type, {}).get("f1")
                    base_row[f"f1_{entity_type}"] = f1
            else:
                # For strict, add both B- and I- F1 scores
                for entity_type in ENTITY_TYPES_DISPLAY + ["O"]:
                    if entity_type == "O":
                        f1 = per_class.get("O", {}).get("f1")
                        base_row["f1_O"] = f1
                    else:
                        b_f1 = per_class.get(f"B-{entity_type}", {}).get("f1")
                        i_f1 = per_class.get(f"I-{entity_type}", {}).get("f1")
                        base_row[f"f1_B-{entity_type}"] = b_f1
                        base_row[f"f1_I-{entity_type}"] = i_f1

            rows.append(base_row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Results saved to {output_path}")


# ===== MAIN EXECUTION =====
def run_standard_mode(sentences_by_script: dict[bool, list[dict]], label2id: dict, id2label: dict) -> dict:
    """Run standard train/test split mode with sequential (non-shuffled) split.

    Args:
        sentences_by_script: Dict with keys False (Latin) and True (Cyrillic) containing pre-loaded sentences
        label2id: Label to ID mapping
        id2label: ID to label mapping
    """
    print("\n" + "=" * 60)
    print("STANDARD MODE: Single train/test split (sequential)")
    print("=" * 60)

    # Use Latin script sentences for splitting (both models have same sentence order)
    base_sentences = sentences_by_script[False]

    # Sequential split (no shuffle) - matches reference methodology
    train_data, test_data = train_test_split(base_sentences, test_size=TEST_SIZE, random_state=SEED, shuffle=False)
    print(f"Train: {len(train_data)} sentences, Test: {len(test_data)} sentences")
    print("  (Using sequential split - no shuffle)")

    # Get train/test indices
    train_indices = [s["sentence_id"] for s in train_data]
    test_indices = [s["sentence_id"] for s in test_data]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = OUTPUT_DIR / "models" / f"standard_{timestamp}"

    results = {}

    for model_key, model_config in MODELS.items():
        print(f"\n{'=' * 40}")
        print(f"Training {model_key}")
        print("=" * 40)

        # Use pre-loaded sentences with appropriate script
        model_sentences = sentences_by_script[model_config["use_cyrillic"]]

        # Use same indices for consistency
        model_train = [model_sentences[i] for i in train_indices]
        model_test = [model_sentences[i] for i in test_indices]

        default_metrics, strict_metrics = train_and_evaluate(
            model_train,
            model_test,
            model_config,
            label2id,
            id2label,
            output_base,
        )

        results[model_key] = {
            "default": default_metrics,
            "strict": strict_metrics,
        }

        csv_path = OUTPUT_DIR / f"ner_results_standard_{timestamp}.csv"
        save_results_to_csv(results, csv_path, mode="standard")

    return results


def run_cv_mode(sentences_by_script: dict[bool, list[dict]], label2id: dict, id2label: dict) -> dict:
    """Run 10-fold cross-validation mode with sequential splits (no shuffle).

    Args:
        sentences_by_script: Dict with keys False (Latin) and True (Cyrillic) containing pre-loaded sentences
        label2id: Label to ID mapping
        id2label: ID to label mapping
    """
    print("\n" + "=" * 60)
    print(f"CROSS-VALIDATION MODE: {NUM_FOLDS}-Fold CV (sequential)")
    print("=" * 60)

    # Use Latin script sentences for fold splitting (both models have same sentence order)
    base_sentences = sentences_by_script[False]
    print(f"Loaded {len(base_sentences)} sentences")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = OUTPUT_DIR / "models" / f"cv_{timestamp}"

    # Sequential splits (no shuffle) - matches reference methodology
    kf = KFold(n_splits=NUM_FOLDS, shuffle=False)
    print("Using sequential splits (no shuffle) - matches reference methodology")
    print("  (Sequential splits create natural document-level folds)")

    results = {}

    for model_key, model_config in MODELS.items():
        print(f"\n{'=' * 50}")
        print(f"Training {model_key} ({NUM_FOLDS}-Fold CV)")
        print("=" * 50)

        # Use pre-loaded sentences with appropriate script
        model_sentences = sentences_by_script[model_config["use_cyrillic"]]

        fold_results = []

        for fold_num, (train_index, test_index) in enumerate(kf.split(model_sentences), 1):
            print(f"\n===== FOLD {fold_num}/{NUM_FOLDS} =====")

            # Sequential sentence-level split
            train_data = [model_sentences[i] for i in train_index]
            test_data = [model_sentences[i] for i in test_index]

            print(f"  Train: {len(train_data)} sentences, Test: {len(test_data)} sentences")

            default_metrics, strict_metrics = train_and_evaluate(
                train_data,
                test_data,
                model_config,
                label2id,
                id2label,
                output_base,
                fold_num=fold_num,
            )

            fold_results.append((default_metrics, strict_metrics))

        # Aggregate fold results
        agg_default, agg_strict = aggregate_cv_results(fold_results)

        results[model_key] = {
            "default": agg_default,
            "strict": agg_strict,
        }

        # Save results to CSV after each model (incremental save)
        csv_path = OUTPUT_DIR / f"ner_results_cv_{timestamp}.csv"
        save_results_to_csv(results, csv_path, mode="cv")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce NER finetuning results using sequential KFold splits (matches reference methodology)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "cv"],
        default="standard",
        help="Execution mode: 'standard' for single split, 'cv' for 10-fold cross-validation. "
        "Both modes use sequential (non-shuffled) splits matching the reference implementation.",
    )
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data once for each script variant (Latin and Cyrillic)
    print("Loading data...")
    print("  Loading Latin script version...")
    sentences_latin, wordlist = load_corpus_sentences([FILE_PATH], use_cyrillic=False, conllup=True)
    print(f"  Loaded {len(sentences_latin)} sentences with {len(wordlist)} unique tokens")

    print("  Loading Cyrillic script version...")
    sentences_cyrillic, _ = load_corpus_sentences([FILE_PATH], use_cyrillic=True, conllup=True)
    print(f"  Loaded {len(sentences_cyrillic)} sentences (Cyrillic)")

    # Store both versions in a dictionary keyed by use_cyrillic flag
    sentences_by_script = {
        False: sentences_latin,  # For BERTić
        True: sentences_cyrillic,  # For SrBERTa
    }

    # Create label maps (using Latin version)
    unique_labels = set()
    for s in sentences_latin:
        unique_labels.update(s["labels"])
    sorted_labels = sorted(list(unique_labels))
    label2id = {label: idx for idx, label in enumerate(sorted_labels)}
    id2label = {idx: label for idx, label in enumerate(sorted_labels)}

    print(f"Found {len(unique_labels)} unique labels: {sorted_labels}")

    # Run appropriate mode (both use sequential splits matching reference)
    if args.mode == "standard":
        results = run_standard_mode(sentences_by_script, label2id, id2label)
    else:
        results = run_cv_mode(sentences_by_script, label2id, id2label)

    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print()

    for model_key, model_results in results.items():
        default_m = model_results.get("default", {})
        print(f"{model_key}:")
        print(f"  Accuracy: {default_m.get('accuracy', 'N/A'):.4f}")
        print(f"  Macro F1 (with O): {default_m.get('macro_f1_with_o', 'N/A'):.4f}")
        print(f"  Macro F1 (without O): {default_m.get('macro_f1_without_o', 'N/A'):.4f}")
        print()

    # Save final JSON with all results
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = OUTPUT_DIR / f"ner_results_{args.mode}_{timestamp_str}.json"

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    print(f"\nFinal JSON results saved to {output_json}")


if __name__ == "__main__":
    main()
