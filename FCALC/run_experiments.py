import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


FCALC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FCALC_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FCALC_ROOT))
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))

import fcalc  # noqa: E402
from src.dataset import Dataset, known_datasets  # noqa: E402
from src.dataset_preprocessor import DatasetPreprocessor  # noqa: E402


STANDARD_METHODS = ["standard", "standard-support", "ratio-support"]
PROXIMITY_METHODS = ["proximity", "proximity-non-falsified", "proximity-support"]
N_ITERS_GRID = [10, 20, 30, 40, 50]
SUBSAMPLE_SIZE_GRID = list(range(1, 11))


def encode_labels(y_train, y_test):
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    return y_train_encoded, y_test_encoded


def load_dataset(name):
    dataset = Dataset(dataset_name=name)
    DatasetPreprocessor(dataset).preprocess()

    scaler = MinMaxScaler().fit(dataset.X_train)
    x_train = scaler.transform(dataset.X_train)
    x_test = scaler.transform(dataset.X_test)
    y_train, y_test = encode_labels(dataset.y_train, dataset.y_test)
    return x_train, x_test, y_train, y_test


def score_predictions(y_true, y_pred):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "F1 score": round(f1_score(y_true, y_pred, average="macro"), 4),
        "Unclassified": round(float((y_pred == -1).sum()) / len(y_pred), 4),
    }


def can_sample(y, subsample_size):
    _, counts = np.unique(y, return_counts=True)
    return counts.min() >= subsample_size


def can_sample_cv(y, subsample_size, n_splits=5, seed=1998):
    kf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    for train_index, _ in kf.split(np.zeros_like(y), y):
        if not can_sample(y[train_index], subsample_size):
            return False
    return True


def predict_standard(x_train, y_train, x_test, method, randomize=False, num_iters=10, subsample_size=1):
    classifier = fcalc.classifier.PatternClassifier(
        x_train,
        y_train,
        method=method,
        randomize=randomize,
        num_iters=num_iters,
        subsample_size=subsample_size,
    )
    classifier.predict(x_test)
    return classifier.predictions


def supp_prox(context, labels, test, method="proximity", num_iters=None, subsample_size=None, seed=42):
    classes = np.unique(labels)
    class_lengths = np.array([len(context[labels == c]) for c in classes])
    support = []
    distances = []
    rng = np.random.default_rng(seed=seed)

    for c in classes:
        train_pos = context[labels == c]
        train_neg = context[labels != c]

        if num_iters is None:
            sampled_groups = train_pos[:, np.newaxis, :]
        else:
            sampled_groups = np.zeros((num_iters, subsample_size, context.shape[1]))
            for j in range(num_iters):
                sampled_groups[j] = rng.choice(
                    train_pos, size=subsample_size, replace=False, shuffle=True
                )

        positive_support = np.zeros((len(test), len(sampled_groups)))
        positive_counter = np.zeros((len(test), len(sampled_groups)))
        pos_dists = np.zeros((len(test), len(sampled_groups)))

        for i in range(len(test)):
            for j, group in enumerate(sampled_groups):
                low = np.minimum(test[i], np.min(group, axis=0))
                high = np.maximum(test[i], np.max(group, axis=0))
                pos_mask = (~((low <= train_pos) & (train_pos <= high))).sum(axis=1) == 0
                cnt_mask = (~((low <= train_neg) & (train_neg <= high))).sum(axis=1) == 0
                pos_dists[i][j] = (
                    1
                    - np.linalg.norm(train_pos[pos_mask] - test[i], axis=1).mean()
                    / np.sqrt(context.shape[1])
                )
                positive_support[i][j] = pos_mask.sum()
                positive_counter[i][j] = cnt_mask.sum()

        support.append(np.array((positive_support, positive_counter)))
        distances.append(pos_dists)

    return support, distances, classes, class_lengths


def proximity_based(proximity, support, classes, class_lengths):
    preds = np.full(proximity[0].shape[0], -1.0)
    criter = np.zeros((len(classes), proximity[0].shape[0]))
    for j in range(len(classes)):
        criter[j] = proximity[j].mean(axis=1)
    criter = criter.T
    pred_mask = (np.max(criter, axis=1)[:, None] == criter).sum(axis=-1) < 2
    preds[pred_mask] = classes[np.argmax(criter[pred_mask], axis=-1)]
    return preds


def proximity_non_falsified(proximity, support, classes, class_lengths):
    preds = np.full(proximity[0].shape[0], -1.0)
    criter = np.zeros((len(classes), proximity[0].shape[0]))
    for j in range(len(classes)):
        criter[j] = (proximity[j] * (support[j][1] == 0)).mean(axis=1)
    criter = criter.T
    pred_mask = (np.max(criter, axis=1)[:, None] == criter).sum(axis=-1) < 2
    preds[pred_mask] = classes[np.argmax(criter[pred_mask], axis=-1)]
    return preds


def proximity_support(proximity, support, classes, class_lengths):
    preds = np.full(proximity[0].shape[0], -1.0)
    criter = np.zeros((len(classes), proximity[0].shape[0]))
    for j in range(len(classes)):
        criter[j] = (support[j][0] * proximity[j] * (support[j][1] == 0)).sum(axis=1)
    criter = criter.T / class_lengths
    pred_mask = (np.max(criter, axis=1)[:, None] == criter).sum(axis=-1) < 2
    preds[pred_mask] = classes[np.argmax(criter[pred_mask], axis=-1)]
    return preds


def predict_proximity(x_train, y_train, x_test, method, num_iters=None, subsample_size=None):
    support, proximity, classes, class_lengths = supp_prox(
        x_train,
        y_train,
        x_test,
        method=method,
        num_iters=num_iters,
        subsample_size=subsample_size,
    )
    if method == "proximity":
        return proximity_based(proximity, support, classes, class_lengths)
    if method == "proximity-non-falsified":
        return proximity_non_falsified(proximity, support, classes, class_lengths)
    if method == "proximity-support":
        return proximity_support(proximity, support, classes, class_lengths)
    raise ValueError(f"Unknown proximity method: {method}")


def cross_val_score_config(x_train, y_train, method, family, num_iters, subsample_size):
    kf = StratifiedKFold(n_splits=5, random_state=1998, shuffle=True)
    scores = []
    for train_index, valid_index in kf.split(x_train, y_train):
        if family == "standard":
            predictions = predict_standard(
                x_train[train_index],
                y_train[train_index],
                x_train[valid_index],
                method,
                randomize=True,
                num_iters=num_iters,
                subsample_size=subsample_size,
            )
        else:
            predictions = predict_proximity(
                x_train[train_index],
                y_train[train_index],
                x_train[valid_index],
                method,
                num_iters=num_iters,
                subsample_size=subsample_size,
            )
        scores.append(f1_score(y_train[valid_index], predictions, average="macro"))
    return float(np.mean(scores))


def tune_randomized(x_train, y_train, methods, family):
    best = None
    for method in methods:
        for num_iters in N_ITERS_GRID:
            for subsample_size in SUBSAMPLE_SIZE_GRID:
                if not can_sample_cv(y_train, subsample_size):
                    continue
                score = cross_val_score_config(
                    x_train, y_train, method, family, num_iters, subsample_size
                )
                if best is None or score > best["cv_f1"]:
                    best = {
                        "method": method,
                        "num_iters": num_iters,
                        "subsample_size": subsample_size,
                        "cv_f1": score,
                    }
    if best is None:
        raise ValueError("No feasible randomized FCALC configuration was found")
    return best


def evaluate_deterministic(x_train, x_test, y_train, y_test, methods, family):
    rows = []
    for method in methods:
        start = time.time()
        if family == "standard":
            predictions = predict_standard(x_train, y_train, x_test, method)
        else:
            predictions = predict_proximity(x_train, y_train, x_test, method)
        elapsed = round(time.time() - start, 2)
        row = {"method": method, **score_predictions(y_test, predictions), "time (sec.)": elapsed}
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def evaluate_randomized(x_train, x_test, y_train, y_test, methods, family):
    best = tune_randomized(x_train, y_train, methods, family)
    start = time.time()
    if family == "standard":
        predictions = predict_standard(
            x_train,
            y_train,
            x_test,
            best["method"],
            randomize=True,
            num_iters=best["num_iters"],
            subsample_size=best["subsample_size"],
        )
    else:
        predictions = predict_proximity(
            x_train,
            y_train,
            x_test,
            best["method"],
            num_iters=best["num_iters"],
            subsample_size=best["subsample_size"],
        )
    elapsed = round(time.time() - start, 2)
    return pd.DataFrame(
        [
            {
                **score_predictions(y_test, predictions),
                "method": best["method"],
                "cv_f1": round(best["cv_f1"], 4),
                "num_iters": best["num_iters"],
                "subsample_size": best["subsample_size"],
                "time (sec.)": elapsed,
            }
        ]
    )


def ensure_output_dirs():
    (FCALC_ROOT / "all-results" / "results").mkdir(parents=True, exist_ok=True)
    (FCALC_ROOT / "all-results" / "results-randomized").mkdir(parents=True, exist_ok=True)


def run_dataset(name):
    print(f"Running FCALC on {name}")
    ensure_output_dirs()
    x_train, x_test, y_train, y_test = load_dataset(name)

    standard_res = evaluate_deterministic(
        x_train, x_test, y_train, y_test, STANDARD_METHODS, "standard"
    )
    standard_res.to_csv(FCALC_ROOT / "all-results" / "results" / f"{name}-res.csv")

    proximity_res = evaluate_deterministic(
        x_train, x_test, y_train, y_test, PROXIMITY_METHODS, "proximity"
    )
    proximity_res.to_csv(FCALC_ROOT / "all-results" / "results" / f"{name}-prox-res.csv")

    randomized_res = evaluate_randomized(
        x_train, x_test, y_train, y_test, STANDARD_METHODS, "standard"
    )
    randomized_res.to_csv(
        FCALC_ROOT / "all-results" / "results-randomized" / f"{name}-fcalc-rand-res.csv",
        index=False,
    )

    randomized_prox_res = evaluate_randomized(
        x_train, x_test, y_train, y_test, PROXIMITY_METHODS, "proximity"
    )
    randomized_prox_res.to_csv(
        FCALC_ROOT
        / "all-results"
        / "results-randomized"
        / f"{name}-fcalc-rand-prox-res.csv",
        index=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run FCALC on the repository datasets.")
    parser.add_argument(
        "datasets",
        nargs="*",
        default=known_datasets,
        help="Dataset names to run. Defaults to all known datasets.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    unknown = sorted(set(args.datasets) - set(known_datasets))
    if unknown:
        raise ValueError(f"Unknown dataset names: {unknown}")
    for name in args.datasets:
        run_dataset(name)


if __name__ == "__main__":
    main()
