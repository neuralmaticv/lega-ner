import os as _os
import warnings

warnings.filterwarnings('ignore')

# Disable progress bars and warnings for cleaner logs
_os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
_os.environ['HF_DATASETS_DISABLE_PROGRESS_BARS'] = '1'

import gc
from datetime import datetime
from pathlib import Path

import cyrtranslit
import numpy as np
import polars as pl
import torch
import transformers
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer,
                          TrainingArguments)

print(f"Transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: CUDA not available, will use CPU!")


def parse_conllu(file_path):
    """
    Parse CoNLL-U format file.
    Returns: (sentences, labels) where each is a list of lists.
    """
    sentences = []
    labels = []
    current_tokens = []
    current_labels = []
    
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and blank lines (end of sentence)
            if line.startswith("#") or not line:
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
                continue
            
            # Parse token line: ID FORM LEMMA POS NER
            parts = line.split("\t")
            if len(parts) >= 5 and parts[0].isdigit():
                token = parts[1]       # Column 2: word form
                ner_tag = parts[4]     # Column 5: NER tag
                current_tokens.append(token)
                current_labels.append(ner_tag)
        
        # Don't forget last sentence
        if current_tokens:
            sentences.append(current_tokens)
            labels.append(current_labels)
    
    return sentences, labels

def tokenize_and_align_labels(examples, tokenizer, label2id, max_length=512):
    """
    Tokenize text and align labels with subword tokens.
    
    Args:
        examples: Dict with 'tokens' and 'ner_tags' keys
        tokenizer: HuggingFace tokenizer
        label2id: Label to ID mapping
    
    Returns:
        Tokenized inputs with aligned labels
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
    )
    
    labels = []
    for i, label_seq in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            # Special tokens (CLS, SEP, PAD) get -100
            if word_idx is None:
                label_ids.append(-100)
            # First subword of a word gets the label
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label_seq[word_idx]])
            # Subsequent subwords get -100 (ignored)
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

def strip_bio_prefix(labels):
    """Convert B-PER, I-PER → PER (entity type only)"""
    stripped = []
    for label in labels:
        if label == 'O':
            stripped.append('O')
        else:
            # Remove B- or I- prefix
            entity_type = label.split('-', 1)[1] if '-' in label else label
            stripped.append(entity_type)
    return stripped

def compute_metrics(pred):
    """
    Compute metrics for model predictions.
    This gets called automatically during evaluation.
    """
    predictions, labels = pred
    
    # Get predicted label IDs (argmax over logits)
    predictions = np.argmax(predictions, axis=2)
    
    # Flatten and remove ignored indices (-100)
    true_labels = []
    pred_labels = []
    
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] != -100:
                true_labels.append(id2label[labels[i][j]])
                pred_labels.append(id2label[predictions[i][j]])
    
    # Convert to arrays
    y_true = np.array(true_labels)
    y_pred = np.array(pred_labels)
    
    # DEFAULT EVALUATION (entity type only)
    y_true_default = strip_bio_prefix(y_true)
    y_pred_default = strip_bio_prefix(y_pred)
    
    default_acc = accuracy_score(y_true_default, y_pred_default)
    
    unique_labels_default = sorted(set(y_true_default) | set(y_pred_default))
    entity_labels_default = [l for l in unique_labels_default if l != 'O']
    
    default_f1_with_o = f1_score(y_true_default, y_pred_default, labels=unique_labels_default, average='macro', zero_division=0)
    default_f1_without_o = f1_score(y_true_default, y_pred_default, labels=entity_labels_default, average='macro', zero_division=0)
    
    # STRICT EVALUATION (full BIO tags)
    strict_acc = accuracy_score(y_true, y_pred)
    
    unique_labels = sorted(set(y_true) | set(y_pred))
    entity_labels = [l for l in unique_labels if l != 'O']
    
    strict_f1_with_o = f1_score(y_true, y_pred, labels=unique_labels, average='macro', zero_division=0)
    strict_f1_without_o = f1_score(y_true, y_pred, labels=entity_labels, average='macro', zero_division=0)
    
    return {
        # Default mode
        'default_accuracy': default_acc,
        'default_f1_with_o': default_f1_with_o,
        'default_f1_without_o': default_f1_without_o,

        # Strict mode
        'strict_accuracy': strict_acc,
        'strict_f1_with_o': strict_f1_with_o,
        'strict_f1_without_o': strict_f1_without_o,
    }


if __name__ == "__main__":
    # Setup paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    OUTPUT_DIR = PROJECT_ROOT / 'outputs'
    RESULTS_DIR = PROJECT_ROOT / 'results' / 'custom'

    # Create directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    data_path = DATA_DIR / "comtext.sr.legal.ijekavica.conllu"
    sentences, labels = parse_conllu(data_path)

    print(f"Loaded {len(sentences)} sentences")
    print(f"Total tokens: {sum(len(s) for s in sentences)}")
    print("\nExample sentence 1:")
    print(f"Tokens: {sentences[0][:10]}...")
    print(f"Labels: {labels[0][:10]}...")

    all_labels = set()
    for label_seq in labels:
        all_labels.update(label_seq)

    unique_labels = sorted(list(all_labels))
    print(f"Total unique labels: {len(unique_labels)}")
    print(f"\nAll labels:\n{unique_labels}")

    # Count occurrences
    label_counts = {}
    for label_seq in labels:
        for label in label_seq:
            label_counts[label] = label_counts.get(label, 0) + 1

    # Show top 10 most frequent
    print("\nTop 10 most frequent labels:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {label:8}: {count:6,d}")

    # Label mappings
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    print(f"Created mappings for {len(label2id)} labels")
    print("\nFirst 10 label mappings:")
    for label, idx in list(label2id.items())[:10]:
        print(f"  {label:10s} -> {idx}")

    print(f"\nTest mapping: 'B-PER' -> {label2id['B-PER']} -> '{id2label[label2id['B-PER']]}'")

    N_FOLDS = 10
    SEED = 64

    # Create folds (using KFold instead of StratifiedKFold to match original implementation)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_splits = list(enumerate(kf.split(sentences), 1))

    print(f"Created {N_FOLDS} folds:")
    for fold_num, (train_idx, eval_idx) in fold_splits:
        print(f"  Fold {fold_num}: Train={len(train_idx)}, Eval={len(eval_idx)}")

    model_name = "classla/bcms-bertic"
    #model_name = "nemanjaPetrovic/SrBERTa"
    model_short_name = model_name.split('/')[-1]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = OUTPUT_DIR / "models" / f"{model_short_name}_10fold_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)

    all_fold_results = []

    print(f"Starting training and evaluation across {N_FOLDS} folds...")
    print("=" * 70)

    # Check if we need Cyrillic conversion for SrBERTa
    use_cyrillic = 'SrBERTa' in model_name or 'srberta' in model_name.lower()
    if use_cyrillic:
        print("-> SrBERTa detected: Converting text to Cyrillic\n")

    for fold_num, (train_idx, eval_idx) in fold_splits:
        # Convert to Cyrillic if using SrBERTa (matches original implementation)
        if use_cyrillic:
            train_dataset = Dataset.from_dict({
                "tokens": [[cyrtranslit.to_cyrillic(token, "sr") for token in sentences[i]]
                           for i in train_idx],
                "ner_tags": [labels[i] for i in train_idx],
            })
            eval_dataset = Dataset.from_dict({
                "tokens": [[cyrtranslit.to_cyrillic(token, "sr") for token in sentences[i]]
                           for i in eval_idx],
                "ner_tags": [labels[i] for i in eval_idx],
            })
        else:
            # BERTić - use Latin as-is
            train_dataset = Dataset.from_dict({
                "tokens": [sentences[i] for i in train_idx],
                "ner_tags": [labels[i] for i in train_idx],
            })
            eval_dataset = Dataset.from_dict({
                "tokens": [sentences[i] for i in eval_idx],
                "ner_tags": [labels[i] for i in eval_idx],
            })

        print(f"Loading tokenizer from {model_name}...")
        # RoBERTa (SrBERTa) needs add_prefix_space=True for pre-tokenized inputs
        if use_cyrillic:
            tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Loading model from {model_name}...")
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )

        tokenized_train = train_dataset.map(
            lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        tokenized_eval = eval_dataset.map(
            lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
            batched=True,
            remove_columns=eval_dataset.column_names,
        )

        training_args = TrainingArguments(
            output_dir= f"{output_base}/fold_{fold_num}",

            # Training schedule
            num_train_epochs=20,
            per_device_train_batch_size=8, # used in original implementation
            per_device_eval_batch_size=256, # original 100

            # Optimization
            learning_rate=4e-05,
            weight_decay=0,
            warmup_ratio=0.06,

            # Evaluation
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="strict_f1_without_o",
            greater_is_better=True,

            # Performance
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,

            # Logging
            logging_dir=str(OUTPUT_DIR / "logs" / timestamp),
            logging_strategy="steps",
            logging_steps=50,
            report_to="none",
            disable_tqdm=True,

            # Checkpointing
            save_total_limit=2,

            # Reproducibility
            seed=SEED
        )

        data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
        )

        print(f"Training fold {fold_num}/{N_FOLDS}...")
        trainer.train()
        print("\n" + "=" * 70)
        print("Training completed!")
        print("=" * 70)

        eval_metrics = trainer.evaluate()

        fold_results = {
            "fold_num": fold_num,
            "train_samples": len(train_idx),
            "eval_samples": len(eval_idx),
            **eval_metrics,
        }
        all_fold_results.append(fold_results)

        # print(f"\nFold {fold_num} results:")
        # print("DEFAULT EVALUATION (entity type only):")
        # print(f"  Accuracy:           {eval_metrics['eval_default_accuracy']:.4f}")
        # print(f"  F1-Macro (with O):  {eval_metrics['eval_default_f1_with_o']:.4f}")
        # print(f"  F1-Macro (no O):    {eval_metrics['eval_default_f1_without_o']:.4f}")
        # print()
        # print("STRICT EVALUATION (full BIO tags):")
        # print(f"  Accuracy:           {eval_metrics['eval_strict_accuracy']:.4f}")
        # print(f"  F1-Macro (with O):  {eval_metrics['eval_strict_f1_with_o']:.4f}")
        # print(f"  F1-Macro (no O):    {eval_metrics['eval_strict_f1_without_o']:.4f}")
        # print("=" * 70)

        # cleanup
        del trainer, model, tokenizer, tokenized_train, tokenized_eval
        gc.collect()
        torch.cuda.empty_cache()



    # Aggregate results
    results_df = pl.DataFrame(all_fold_results)

    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS: {model_name}")
    print(f"{'='*70}\n")

    # Calculate statistics for all metrics
    metrics = {
        'default_accuracy': results_df['eval_default_accuracy'],
        'default_f1_with_o': results_df['eval_default_f1_with_o'],
        'default_f1_without_o': results_df['eval_default_f1_without_o'],
        'strict_accuracy': results_df['eval_strict_accuracy'],
        'strict_f1_with_o': results_df['eval_strict_f1_with_o'],
        'strict_f1_without_o': results_df['eval_strict_f1_without_o'],
    }

    print("DEFAULT EVALUATION (Entity Type Only):")
    print("-" * 70)
    print(f"  Accuracy:          {metrics['default_accuracy'].mean():.4f} ± {metrics['default_accuracy'].std():.4f}")
    print(f"  F1-Macro (with O): {metrics['default_f1_with_o'].mean():.4f} ± {metrics['default_f1_with_o'].std():.4f}")
    print(f"  F1-Macro (no O):   {metrics['default_f1_without_o'].mean():.4f} ± {metrics['default_f1_without_o'].std():.4f}")

    print("\nSTRICT EVALUATION (Full BIO Tags):")
    print("-" * 70)
    print(f"  Accuracy:          {metrics['strict_accuracy'].mean():.4f} ± {metrics['strict_accuracy'].std():.4f}")
    print(f"  F1-Macro (with O): {metrics['strict_f1_with_o'].mean():.4f} ± {metrics['strict_f1_with_o'].std():.4f}")
    print(f"  F1-Macro (no O):   {metrics['strict_f1_without_o'].mean():.4f} ± {metrics['strict_f1_without_o'].std():.4f}")


    csv_path = f"{output_base}/results.csv"
    results_df.write_csv(csv_path)
    print(f"\n{'='*70}")
    print(f"  Detailed results saved to: {csv_path}")
    print(f"{'='*70}\n")

    # Save summary statistics
    summary_data = {
        'model': [model_name],
        'n_folds': [N_FOLDS],
        'default_accuracy_mean': [metrics['default_accuracy'].mean()],
        'default_accuracy_std': [metrics['default_accuracy'].std()],
        'default_f1_with_o_mean': [metrics['default_f1_with_o'].mean()],
        'default_f1_with_o_std': [metrics['default_f1_with_o'].std()],
        'default_f1_without_o_mean': [metrics['default_f1_without_o'].mean()],
        'default_f1_without_o_std': [metrics['default_f1_without_o'].std()],
        'strict_accuracy_mean': [metrics['strict_accuracy'].mean()],
        'strict_accuracy_std': [metrics['strict_accuracy'].std()],
        'strict_f1_with_o_mean': [metrics['strict_f1_with_o'].mean()],
        'strict_f1_with_o_std': [metrics['strict_f1_with_o'].std()],
        'strict_f1_without_o_mean': [metrics['strict_f1_without_o'].mean()],
        'strict_f1_without_o_std': [metrics['strict_f1_without_o'].std()],
    }

    summary_df = pl.DataFrame(summary_data)
    summary_path = f"{output_base}/summary.csv"
    summary_df.write_csv(summary_path)
    print(f"  Summary statistics saved to: {summary_path}\n")

    # Also save to results directory (version controlled)
    results_file = RESULTS_DIR / f"{model_short_name}_10fold_{timestamp}_results.csv"
    results_df.write_csv(str(results_file))
    print(f"  Results also saved to: {results_file}")

    summary_results_file = RESULTS_DIR / f"{model_short_name}_10fold_{timestamp}_summary.csv"
    summary_df.write_csv(str(summary_results_file))
    print(f"  Summary also saved to: {summary_results_file}\n")
