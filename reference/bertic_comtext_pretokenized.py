import argparse
import gc
import json
import random
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from load_data import load_corpus_tokens
from seqeval.metrics import accuracy_score
from simpletransformers.ner import ner_model
from sklearn.model_selection import KFold

warnings.filterwarnings(action="ignore", category=UserWarning, module="seqeval")

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reference"
RESULTS_DIR = PROJECT_ROOT / "results" / "reference"

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 64

# Model options: 0=BERTic, 1=SrBERTa
MODEL_TYPE = ["electra", "roberta"]
MODEL_NAME = ["classla/bcms-bertic", "nemanjaPetrovic/SrBERTa"]

# Dialect options: 0=Ekavica, 1=Ijekavica
DIALECT_NAME = ["Ekavica", "Ijekavica"]
DATA_PATHS = (
    [str(PROJECT_ROOT / "data" / "comtext.sr.legal.ekavica.conllu")],
    [str(PROJECT_ROOT / "data" / "comtext.sr.legal.ijekavica.conllu")],
)


def get_labels(all_data):
    labels = []
    for item in all_data:
        tag = item[2]
        if tag not in labels:
            labels.append(tag)

    return labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train NER model on COMtext.SR dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bertic_comtext_pretokenized.py --model 0 --dialect 0 --epochs 20
  python bertic_comtext_pretokenized.py -m 1 -d 1 -e 10
        """,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=int,
        choices=[0, 1],
        default=0,
        help="Model: 0=BERTic (classla/bcms-bertic), 1=SrBERTa (default: 0)",
    )
    parser.add_argument(
        "-d",
        "--dialect",
        type=int,
        choices=[0, 1],
        default=0,
        help="Dialect: 0=Ekavica, 1=Ijekavica (default: 0)",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    model_index = cli_args.model
    dialect = cli_args.dialect
    num_epochs = cli_args.epochs

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print(f"Model: {MODEL_NAME[model_index]}")
    print(f"Dialect: {DIALECT_NAME[dialect]}")
    print(f"Epochs: {num_epochs}")

    X_tokens, wordlist = load_corpus_tokens(DATA_PATHS[dialect], model_name=MODEL_NAME[model_index], conllup=True)

    labels_list = get_labels(X_tokens)
    results = {}

    # added shuffle=True, random_state=64
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = MODEL_NAME[model_index].split("/")[1]
    dialect_name = DIALECT_NAME[dialect]

    train_args = {}
    train_args["num_train_epochs"] = num_epochs
    train_args["manual_seed"] = RANDOM_SEED
    train_args["max_seq_length"] = 512
    train_args["silent"] = True
    train_args["output_dir"] = str(OUTPUT_DIR / f"{model_short}_{dialect_name}_{timestamp}")
    train_args["overwrite_output_dir"] = False
    train_args["reprocess_input_data"] = True
    train_args["no_cache"] = True
    train_args["save_eval_checkpoints"] = False
    train_args["save_model_every_epoch"] = False
    train_args["use_cached_eval_features"] = False
    train_args["do_lower_case"] = False

    # workaround for multiprocessing issue
    train_args["use_multiprocessing"] = False
    train_args["process_count"] = 1

    results[num_epochs] = []
    fold_index = 0

    for train_index, test_index in kf.split(X_tokens, X_tokens):
        fold_index += 1
        conllu_id_train = []
        id_train = []
        X_train = []
        y_train = []
        conllu_id_test = []
        id_test = []
        X_test = []
        y_test = []

        for elem in train_index:
            id_train.append(X_tokens[elem][0])
            X_train.append(X_tokens[elem][1])
            y_train.append(X_tokens[elem][2])
            conllu_id_train.append(X_tokens[elem][4])
        gold_lemmas = []
        for elem in test_index:
            id_test.append(X_tokens[elem][0])
            X_test.append(X_tokens[elem][1])
            y_test.append(X_tokens[elem][2])
            gold_lemmas.append(X_tokens[elem][3])
            conllu_id_test.append(X_tokens[elem][4])

        train_df = pd.DataFrame(
            list(zip(id_train, X_train, y_train)),
            columns=["sentence_id", "words", "labels"],
        )
        test_df = pd.DataFrame(
            list(zip(id_test, X_test, y_test)),
            columns=["sentence_id", "words", "labels"],
        )

        model = ner_model.NERModel(
            MODEL_TYPE[model_index],
            MODEL_NAME[model_index],
            use_cuda=True,
            labels=labels_list,
            args=train_args,
        )

        print(f"Model training for {num_epochs} epochs, fold {fold_index}")
        model.train_model(train_df, acc=accuracy_score)

        print(f"Model evaluation, fold {fold_index}")
        msd_result, model_outputs, preds_list = model.eval_model(test_df, acc=accuracy_score)
        print(f"NER accuracy after fine-tuning for {num_epochs} epochs, fold {fold_index}:")
        print(msd_result)

        results[num_epochs].append(msd_result)

        # clean up cache to prevent memory leak
        del model
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Failed to clear CUDA cache: {e}")

    # Save results
    results_filename = f"results_pretokenized_CV_{model_short}_{dialect_name}_{timestamp}.json"
    results_path = RESULTS_DIR / results_filename

    with open(results_path, "w") as outfile:
        json.dump(results, outfile)

    print(f"\nResults saved to: {results_path}")
