import gc
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from load_data import load_corpus_tokens
from seqeval.metrics import accuracy_score
from simpletransformers.ner import ner_model
from sklearn.model_selection import KFold

warnings.filterwarnings(action='ignore', category=UserWarning, module='seqeval')

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reference"
RESULTS_DIR = PROJECT_ROOT / "results" / "reference"

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 64

MODEL_INDEX = 0 # 0: BERTic, 1: SrBERTa
MODEL_TYPE = ["electra", "roberta"]
MODEL_NAME = ["classla/bcms-bertic", "nemanjaPetrovic/SrBERTa"]

DIALECT = 0 # 0: EKAVICA, 1: IJEKAVICA
DIALECT_NAME = ["Ekavica", "Ijekavica"]
DATA_PATHS = (
    [str(PROJECT_ROOT / "data" / "comtext.sr.legal.ekavica.conllu")],
    [str(PROJECT_ROOT / "data" / "comtext.sr.legal.ijekavica.conllu")]
)


def get_labels(all_data):
    labels = []
    for item in all_data:
        tag = item[2]
        if tag not in labels:
            labels.append(tag)

    return labels


if __name__ == '__main__':
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    X_tokens, wordlist = load_corpus_tokens(DATA_PATHS[DIALECT], model_name=MODEL_NAME[MODEL_INDEX])

    labels_list = get_labels(X_tokens)
    results = {}

    # added shuffle=True, random_state=64
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

    # changed to 20 epochs
    for i in [20]:
        args = {}
        args["num_train_epochs"] =  i
        args["manual_seed"] = RANDOM_SEED
        args["max_seq_length"] = 512
        args["silent"] = True
        args['output_dir'] = str(OUTPUT_DIR / 'reference_models')  # Save to outputs/
        args['overwrite_output_dir'] = True
        args['reprocess_input_data'] = True
        args['no_cache'] = True
        args['save_eval_checkpoints'] = False
        args['save_model_every_epoch'] = False
        args['use_cached_eval_features'] = False
        args['do_lower_case'] = False

        # workaround for multiprocessing issue
        args["use_multiprocessing"] = False
        args["process_count"] = 1
        results[i] = []

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

            train_df = pd.DataFrame(list(zip(id_train, X_train, y_train)), columns=["sentence_id", "words", "labels"])
            test_df = pd.DataFrame(list(zip(id_test, X_test, y_test)), columns=["sentence_id", "words", "labels"])

            model = ner_model.NERModel(MODEL_TYPE[MODEL_INDEX],
                                MODEL_NAME[MODEL_INDEX],
                                use_cuda=True,
                                labels=labels_list,
                                args=args
                                )

            print(f"Model training for {i} epochs, fold {fold_index}")
            model.train_model(train_df, acc=accuracy_score)

            print(f"Model evaluation, fold {fold_index}")
            msd_result, model_outputs, preds_list = model.eval_model(test_df, acc=accuracy_score)
            print(f"NER accuracy after fine-tuning for {i} epochs, fold {fold_index}:")
            print(msd_result)

            results[i].append(msd_result)

            # clean up cache to prevent memory leak
            del model
            gc.collect()
            torch.cuda.empty_cache()

    # Save results
    results_filename = f"results_pretokenized_CV_{MODEL_NAME[MODEL_INDEX].split('/')[1]}_{DIALECT_NAME[DIALECT]}.json"
    results_path = RESULTS_DIR / results_filename

    with open(results_path, 'w') as outfile:
        json.dump(results, outfile)

    print(f"\nResults saved to: {results_path}")