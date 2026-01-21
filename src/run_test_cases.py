"""
Run NER inference on test cases from JSON file.

Usage:
    uv run python src/run_test_cases.py
    uv run python src/run_test_cases.py --model srberta
    uv run python src/run_test_cases.py --id 5
    uv run python src/run_test_cases.py --region Croatia
    uv run python src/run_test_cases.py --ids 1,5,10
"""

import argparse
import json
from pathlib import Path

from inference import MODEL_CONFIGS, get_model_path, load_model, predict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_CASES_PATH = PROJECT_ROOT / "tests" / "ner_test_cases.json"


def load_test_cases() -> list[dict]:
    """Load test cases from JSON file."""
    with open(TEST_CASES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["test_cases"]


def format_predictions(results: list[tuple[str, str]]) -> list[str]:
    """Format predictions as entity strings."""
    entities = []
    current_entity = []
    current_label = None

    for token, label in results:
        if label == "O":
            if current_entity:
                entities.append(f"{' '.join(current_entity)} ({current_label})")
                current_entity = []
                current_label = None
        elif label.startswith("B-"):
            if current_entity:
                entities.append(f"{' '.join(current_entity)} ({current_label})")
            current_entity = [token]
            current_label = label[2:]
        elif label.startswith("I-"):
            if current_entity:
                current_entity.append(token)
            else:
                current_entity = [token]
                current_label = label[2:]

    if current_entity:
        entities.append(f"{' '.join(current_entity)} ({current_label})")

    return entities


def run_test_case(test_case: dict, model, tokenizer, use_cyrillic: bool) -> dict:
    """Run a single test case and return results."""
    text = test_case["text"]
    results = predict(text, model, tokenizer, use_cyrillic=use_cyrillic)
    predicted_entities = format_predictions(results)

    return {
        "id": test_case["id"],
        "region": test_case.get("region", "Unknown"),
        "text": text,
        "expected": test_case["expected_entities"],
        "predicted": predicted_entities,
    }


def print_result(result: dict, verbose: bool = True) -> None:
    """Print test case result."""
    print(f"\n{'=' * 60}")
    print(f"Test #{result['id']} [{result['region']}]")
    print(f"{'=' * 60}")
    print(f"Text: {result['text']}")

    if verbose:
        print(f"\nExpected:  {result['expected']}")
        print(f"Predicted: {result['predicted']}")

        # Simple match check
        expected_set = set(result["expected"])
        predicted_set = set(result["predicted"])

        missing = expected_set - predicted_set
        extra = predicted_set - expected_set

        if expected_set == predicted_set:
            print("Status: MATCH")
        elif not missing and not extra:
            print("Status: MATCH")
        else:
            print("Status: MISMATCH")
            if missing:
                print(f"  Missing: {list(missing)}")
            if extra:
                print(f"  Extra:   {list(extra)}")
    else:
        print(f"Entities: {result['predicted']}")


def main():
    parser = argparse.ArgumentParser(description="Run NER test cases from JSON file.")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        default="bertic",
        help="Model to use (default: bertic)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Specific fold to use (default: last fold)",
    )
    parser.add_argument(
        "--id",
        type=int,
        default=None,
        help="Run specific test case by ID",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Run specific test cases by IDs (comma-separated, e.g., 1,5,10)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Run test cases from specific region (e.g., Serbia, Croatia, Bosnia, Montenegro)",
    )
    parser.add_argument(
        "--brief",
        action="store_true",
        help="Brief output (no expected/comparison)",
    )

    args = parser.parse_args()

    # Load test cases
    test_cases = load_test_cases()

    # Filter test cases
    if args.id is not None:
        test_cases = [tc for tc in test_cases if tc["id"] == args.id]
    elif args.ids is not None:
        ids = [int(i.strip()) for i in args.ids.split(",")]
        test_cases = [tc for tc in test_cases if tc["id"] in ids]
    elif args.region is not None:
        test_cases = [tc for tc in test_cases if tc.get("region", "").lower() == args.region.lower()]

    if not test_cases:
        print("No matching test cases found.")
        return

    # Load model
    config = MODEL_CONFIGS[args.model]
    model_path = get_model_path(args.model, args.fold)
    model, tokenizer = load_model(model_path)

    print(f"\nRunning {len(test_cases)} test case(s) with {args.model}...")

    # Run test cases
    for tc in test_cases:
        result = run_test_case(tc, model, tokenizer, use_cyrillic=config["use_cyrillic"])
        print_result(result, verbose=not args.brief)

    print(f"\n{'=' * 60}")
    print(f"Completed {len(test_cases)} test case(s)")


if __name__ == "__main__":
    main()
