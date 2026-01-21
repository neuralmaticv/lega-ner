"""
Simple inference script for trained NER models.
"""

import argparse
import re
from pathlib import Path

import cyrtranslit
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

# Model configurations matching training script
MODEL_CONFIGS = {
    "bertic": {
        "pattern": "classla_bcms-bertic",
        "use_cyrillic": False,
        "description": "BERTić (BCMS ELECTRA) - Latin script",
    },
    "srberta": {
        "pattern": "nemanjaPetrovic_SrBERTa",
        "use_cyrillic": True,
        "description": "SrBERTa (Serbian RoBERTa) - Cyrillic script",
    },
    "mmbert": {
        "pattern": "jhu-clsp_mmBERT-base",
        "use_cyrillic": False,
        "description": "mmBERT (Multilingual BERT) - Latin script",
    },
    "xlmrbert": {
        "pattern": "classla_xlm-r-bertic",
        "use_cyrillic": False,
        "description": "XLM-R-BERTić Latin script",
    },
    "jerteh": {
        "pattern": "jerteh_Jerteh-355",
        "use_cyrillic": False,
        "description": "Jerteh 355 - Latin script",
    },
}


def find_available_models() -> dict[str, list[Path]]:
    """Find all available trained models in the outputs directory."""
    available = {}

    if not MODELS_DIR.exists():
        return available

    for cv_dir in MODELS_DIR.iterdir():
        if not cv_dir.is_dir():
            continue

        for model_key, config in MODEL_CONFIGS.items():
            pattern = config["pattern"]
            model_dirs = sorted(cv_dir.glob(f"{pattern}_fold_*/best_model"))
            if model_dirs:
                if model_key not in available:
                    available[model_key] = []
                available[model_key].extend(model_dirs)

    return available


def extract_fold_number(path: Path) -> int:
    """Extract fold number from path for proper numeric sorting."""
    match = re.search(r"_fold_(\d+)", str(path))
    return int(match.group(1)) if match else 0


def get_model_path(model_key: str, fold: int | None = None) -> Path:
    """Get path to a specific model, defaulting to last fold."""
    available = find_available_models()

    if model_key not in available:
        raise ValueError(f"Model '{model_key}' not found. Available: {list(available.keys())}")

    # Sort by fold number numerically
    model_paths = sorted(available[model_key], key=extract_fold_number)

    if fold is not None:
        # Find specific fold
        for path in model_paths:
            if f"_fold_{fold}" in str(path):
                return path
        raise ValueError(f"Fold {fold} not found for model '{model_key}'")

    # Default to last fold
    return model_paths[-1]


def load_model(model_path: Path) -> tuple:
    """Load model and tokenizer from path."""
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("Using CUDA")

    return model, tokenizer


def predict(
    text: str,
    model,
    tokenizer,
    use_cyrillic: bool = False,
) -> list[tuple[str, str]]:
    """
    Run NER prediction on input text.

    Returns list of (token, label) tuples.
    """
    # Transliterate to Cyrillic for SrBERTa
    if use_cyrillic:
        text = cyrtranslit.to_cyrillic(text, "sr")

    # Simple whitespace tokenization to get words
    words = text.split()

    # Tokenize
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = encoding
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)

    # Map predictions back to words
    word_ids = encoding.word_ids()

    predictions = predictions[0].cpu().numpy()
    id2label = model.config.id2label

    results = []
    prev_word_idx = None

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != prev_word_idx:
            label = id2label[predictions[idx]]
            results.append((words[word_idx], label))
        prev_word_idx = word_idx

    return results


def format_results(results: list[tuple[str, str]], show_all: bool = False) -> str:
    """Format prediction results for display."""
    lines = []

    for token, label in results:
        if show_all or label != "O":
            lines.append(f"  {token:20} -> {label}")

    if not lines:
        return "  No named entities detected."

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run NER inference on text using trained models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python src/inference.py --text "Petar Petrović živi u Beogradu."
    uv run python src/inference.py --model srberta --text "Firma ABC d.o.o."
    uv run python src/inference.py --model bertic --fold 5 --text "Test"
    uv run python src/inference.py --list-models
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        default="bertic",
        help="Model to use for inference (default: bertic)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Specific fold to use (default: last available fold)",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to analyze for named entities",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all tokens including O labels",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available trained models and exit",
    )

    args = parser.parse_args()

    # List available models
    if args.list_models:
        available = find_available_models()
        if not available:
            print("No trained models found in outputs/models/")
            return

        print("Available trained models:")
        for model_key, paths in available.items():
            config = MODEL_CONFIGS[model_key]
            print(f"\n  {model_key}: {config['description']}")
            folds = [p.parent.name.split("_fold_")[-1] for p in paths]
            print(f"    Folds available: {', '.join(folds)}")
        return

    # Require text for inference
    if not args.text:
        parser.error("--text is required for inference (or use --list-models)")

    # Get model configuration
    config = MODEL_CONFIGS[args.model]

    # Load model
    model_path = get_model_path(args.model, args.fold)
    model, tokenizer = load_model(model_path)

    # Run prediction
    print(f"\nInput: {args.text}")
    if config["use_cyrillic"]:
        print(f"(Transliterated to Cyrillic for {args.model})")

    results = predict(args.text, model, tokenizer, use_cyrillic=config["use_cyrillic"])

    # Display results
    print("\nNamed Entities:")
    print(format_results(results, show_all=args.show_all))


if __name__ == "__main__":
    main()
