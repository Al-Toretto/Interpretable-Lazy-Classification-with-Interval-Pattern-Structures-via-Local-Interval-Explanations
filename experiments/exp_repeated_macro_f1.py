from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from scipy.stats import t
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "FCALC"))
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))

from FCALC.run_experiments import (  # noqa: E402
    PROXIMITY_METHODS,
    STANDARD_METHODS,
    can_sample_cv,
    cross_val_score_config,
    predict_proximity,
    predict_standard,
)
from src.dataset import Dataset, known_datasets  # noqa: E402
from src.dataset_preprocessor import DatasetPreprocessor  # noqa: E402
from src.ips_knn_classifier import IPSKNNClassifier  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "output" / "repeated_macro_f1"
RAW_RESULTS_PATH = OUTPUT_DIR / "raw_repeat_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
LATEX_PATH = OUTPUT_DIR / "latex_table_body.tex"
TEXT_SUMMARY_PATH = OUTPUT_DIR / "summary.txt"

DEFAULT_REPEATS = 10
REPEAT_SEEDS = [1998 + i for i in range(DEFAULT_REPEATS)]
CV_SEED = 1998

SMALL_FIRST_DATASETS = [
    "wine",
    "breast_cancer",
    "rice",
    "sonar",
    "parkinsons",
    "spam",
    "glass",
    "ionosphere",
    "page_blocks",
    "waveform",
    "vehicle",
    "image_segmentation",
]


PARAM_GRIDS = {
    "ips_knn": {"k": list(range(1, 101))},
    "knn": {
        "classifier__n_neighbors": list(range(1, 101)),
        "classifier__weights": ["uniform", "distance"],
    },
    "naive_bayes": {"var_smoothing": [1e-9, 1e-8, 1e-7]},
    "logistic_regression": [
        {
            "classifier__penalty": ["l1"],
            "classifier__C": [10**x for x in range(-5, 5, 1)],
            "classifier__solver": ["liblinear"],
        },
        {
            "classifier__penalty": ["l2"],
            "classifier__C": [10**x for x in range(-5, 5, 1)],
            "classifier__solver": ["liblinear", "lbfgs"],
        },
    ],
    "svm": {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__kernel": ["linear", "rbf"],
        "classifier__gamma": ["scale", "auto"],
    },
    "decision_tree": {
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": [None, "sqrt"],
        "min_impurity_decrease": [0.0],
    },
    "random_forest": {
        "n_estimators": [100, 300],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", None],
        "min_impurity_decrease": [0.0],
        "bootstrap": [True],
    },
    "xgboost": {
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [50, 100, 200],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "scale_pos_weight": [1],
        "reg_alpha": [0],
        "reg_lambda": [1e-5],
    },
}


CLASSIFIER_ORDER = [
    "fcalc",
    "fcalc_rand",
    "ips_knn",
    "knn",
    "naive_bayes",
    "logistic_regression",
    "svm",
    "decision_tree",
    "random_forest",
    "xgboost",
]


CLASSIFIER_LABELS = {
    "naive_bayes": "Naive Bayes",
    "xgboost": "XGBoost",
    "ips_knn": "IPS-KNN",
    "knn": "k-NN",
    "logistic_regression": "Logistic Regression",
    "svm": "SVM",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "fcalc": "FCALC",
    "fcalc_rand": "FCALC(rand.)",
}

DATASET_LABELS = {
    "wine": "Wine",
    "breast_cancer": "Breast Cancer",
    "rice": "Rice",
    "sonar": "Sonar",
    "parkinsons": "Parkinson's",
    "spam": "SpamBase",
    "glass": "Glass",
    "ionosphere": "Ionosphere",
    "page_blocks": "Page Blocks",
    "waveform": "Waveform",
    "vehicle": "Vehicle",
    "image_segmentation": "Segmentation",
}


def normalize_labels_for_scoring(values):
    normalized = []
    for value in pd.Series(values).tolist():
        if pd.isna(value):
            normalized.append("__missing__")
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            normalized.append(str(value))
            continue
        if math.isfinite(numeric_value) and numeric_value.is_integer():
            normalized.append(str(int(numeric_value)))
        else:
            normalized.append(str(value))
    return normalized


def macro_f1_safe(y_true, y_pred):
    y_true_safe = normalize_labels_for_scoring(y_true)
    y_pred_safe = normalize_labels_for_scoring(y_pred)
    return f1_score(y_true_safe, y_pred_safe, average="macro", zero_division=0)


def clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key.replace("classifier__", ""): value for key, value in params.items()}


def load_full_dataset(name: str):
    dataset = Dataset(dataset_name=name)
    preprocessor = DatasetPreprocessor(dataset)
    preprocessor._load_data()
    preprocessor._known_datasets_preprocessor[name]()
    dataset.y = dataset.df[dataset.dataset_class_column_name]
    dataset.X = dataset.df.drop(dataset.dataset_class_column_name, axis="columns")
    return dataset.X, dataset.y


def ordered_datasets(selected: list[str] | None):
    available = [name for name in SMALL_FIRST_DATASETS if name in known_datasets]
    available.extend([name for name in known_datasets if name not in available])
    if selected:
        unknown = sorted(set(selected) - set(known_datasets))
        if unknown:
            raise ValueError(f"Unknown dataset names: {unknown}")
        return [name for name in available if name in selected]
    return available


def classifier_factory(name: str, seed: int, n_jobs: int):
    if name == "naive_bayes":
        return GaussianNB(), PARAM_GRIDS[name]
    if name == "xgboost":
        return (
            xgb.XGBClassifier(
                random_state=seed,
                n_jobs=1,
                eval_metric="logloss",
            ),
            PARAM_GRIDS[name],
        )
    if name == "ips_knn":
        return IPSKNNClassifier(), PARAM_GRIDS[name]
    if name == "knn":
        return (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", KNeighborsClassifier(n_jobs=1)),
                ]
            ),
            PARAM_GRIDS[name],
        )
    if name == "logistic_regression":
        return (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(random_state=seed, n_jobs=1, max_iter=5000),
                    ),
                ]
            ),
            PARAM_GRIDS[name],
        )
    if name == "svm":
        return (
            Pipeline([("scaler", StandardScaler()), ("classifier", SVC(random_state=seed))]),
            PARAM_GRIDS[name],
        )
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed), PARAM_GRIDS[name]
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=1), PARAM_GRIDS[name]
    raise ValueError(f"Unknown sklearn classifier: {name}")


def tune_sklearn_classifier(name, x_train, y_train, seed, n_jobs):
    estimator, param_grid = classifier_factory(name, seed, n_jobs)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=make_scorer(macro_f1_safe),
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
        error_score=np.nan,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        search.fit(x_train, y_train)
    return search.best_estimator_, clean_params(search.best_params_), search.best_score_


def encode_for_xgboost(y_train, y_test):
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    return y_train_encoded, y_test_encoded


def evaluate_sklearn_classifier(name, x_train, x_test, y_train, y_test, seed, n_jobs):
    if name == "xgboost":
        y_train_fit, y_test_eval = encode_for_xgboost(y_train, y_test)
        estimator, best_params, cv_f1 = tune_sklearn_classifier(
            name, x_train, y_train_fit, seed, n_jobs
        )
        predictions = estimator.predict(x_test)
        score = macro_f1_safe(y_test_eval, predictions)
        return score, best_params, cv_f1

    x_train_fit = x_train
    x_test_eval = x_test
    if name == "ips_knn":
        scaler = StandardScaler().fit(x_train)
        x_train_fit = pd.DataFrame(
            scaler.transform(x_train), columns=x_train.columns, index=x_train.index
        )
        x_test_eval = pd.DataFrame(
            scaler.transform(x_test), columns=x_test.columns, index=x_test.index
        )

    estimator, best_params, cv_f1 = tune_sklearn_classifier(
        name, x_train_fit, y_train, seed, n_jobs
    )
    predictions = estimator.predict(x_test_eval)
    score = macro_f1_safe(y_test, predictions)
    return score, best_params, cv_f1


def encode_labels(y_train, y_test):
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    return y_train_encoded, y_test_encoded


def prepare_fcalc_data(x_train, x_test, y_train, y_test):
    scaler = MinMaxScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    y_train_encoded, y_test_encoded = encode_labels(y_train, y_test)
    return x_train_scaled, x_test_scaled, y_train_encoded, y_test_encoded


def tune_fcalc_randomized(x_train, y_train, rule_configs, n_jobs):
    configs = [
        (family, method, num_iters, subsample_size)
        for family, method in rule_configs
        for num_iters in [10, 20, 30, 40, 50]
        for subsample_size in range(1, 11)
        if can_sample_cv(y_train, subsample_size)
    ]
    if not configs:
        raise ValueError("No feasible randomized FCALC configuration was found")

    def score_config(family, method, num_iters, subsample_size):
        return {
            "family": family,
            "method": method,
            "num_iters": num_iters,
            "subsample_size": subsample_size,
            "cv_f1": cross_val_score_config(
                x_train, y_train, method, family, num_iters, subsample_size
            ),
        }

    results = Parallel(n_jobs=n_jobs)(
        delayed(score_config)(family, method, num_iters, subsample_size)
        for family, method, num_iters, subsample_size in configs
    )
    return max(results, key=lambda item: item["cv_f1"])


def evaluate_fcalc_classifier(
    name, x_train, x_test, y_train, y_test, n_jobs, progress_label=None
):
    x_train_f, x_test_f, y_train_f, y_test_f = prepare_fcalc_data(
        x_train, x_test, y_train, y_test
    )
    if name == "fcalc":
        best = tune_fcalc_deterministic(
            x_train_f,
            y_train_f,
            STANDARD_METHODS + PROXIMITY_METHODS,
            progress_label=progress_label,
        )
        if best["family"] == "standard":
            predictions = predict_standard(x_train_f, y_train_f, x_test_f, best["method"])
        else:
            predictions = predict_proximity(x_train_f, y_train_f, x_test_f, best["method"])
    elif name == "fcalc_rand":
        best = tune_fcalc_randomized(
            x_train_f,
            y_train_f,
            [
                *[("standard", method) for method in STANDARD_METHODS],
                *[("proximity", method) for method in PROXIMITY_METHODS],
            ],
            n_jobs,
        )
        if best["family"] == "standard":
            predictions = predict_standard(
                x_train_f,
                y_train_f,
                x_test_f,
                best["method"],
                randomize=True,
                num_iters=best["num_iters"],
                subsample_size=best["subsample_size"],
            )
        else:
            predictions = predict_proximity(
                x_train_f,
                y_train_f,
                x_test_f,
                best["method"],
                num_iters=best["num_iters"],
                subsample_size=best["subsample_size"],
            )
    else:
        raise ValueError(f"Unknown FCALC classifier: {name}")

    score = macro_f1_safe(y_test_f, predictions)
    return score, best, best["cv_f1"]


def tune_fcalc_deterministic(x_train, y_train, methods, progress_label=None):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    best = None
    total_steps = len(methods) * cv.get_n_splits()
    completed_steps = 0
    start = time.time()
    last_percent = -1
    if progress_label:
        print(
            f"[fcalc progress] {progress_label} | deterministic CV started "
            f"| steps={total_steps}",
            flush=True,
        )
    for method_idx, method in enumerate(methods, start=1):
        family = "standard" if method in STANDARD_METHODS else "proximity"
        scores = []
        for fold_idx, (train_index, valid_index) in enumerate(cv.split(x_train, y_train), start=1):
            if family == "standard":
                predictions = predict_standard(
                    x_train[train_index], y_train[train_index], x_train[valid_index], method
                )
            else:
                predictions = predict_proximity(
                    x_train[train_index], y_train[train_index], x_train[valid_index], method
                )
            scores.append(macro_f1_safe(y_train[valid_index], predictions))
            completed_steps += 1
            percent = int(100 * completed_steps / total_steps)
            if progress_label and percent > last_percent:
                last_percent = percent
                print(
                    f"[fcalc progress] {progress_label} | "
                    f"method {method_idx}/{len(methods)} {family}:{method} "
                    f"| fold {fold_idx}/5 | step {completed_steps}/{total_steps} "
                    f"| {percent}% | elapsed={format_duration(time.time() - start)}",
                    flush=True,
                )
        cv_f1 = float(np.mean(scores))
        if best is None or cv_f1 > best["cv_f1"]:
            best = {"family": family, "method": method, "cv_f1": cv_f1}
            if progress_label:
                print(
                    f"[fcalc progress] {progress_label} | new best "
                    f"{family}:{method} cv_macro_f1={cv_f1 * 100:.2f}",
                    flush=True,
                )
    if progress_label:
        print(
            f"[fcalc progress] {progress_label} | deterministic CV finished "
            f"| best={best['family']}:{best['method']} "
            f"| cv_macro_f1={best['cv_f1'] * 100:.2f} "
            f"| elapsed={format_duration(time.time() - start)}",
            flush=True,
        )
    return best


def load_completed_keys(path):
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    return set(zip(df["dataset"], df["classifier"], df["repeat"]))


def append_raw_result(row, raw_results_path):
    raw_results_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        raw_results_path, mode="a", header=not raw_results_path.exists(), index=False
    )


def format_duration(seconds):
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def evaluate_one(
    classifier,
    x_train,
    x_test,
    y_train,
    y_test,
    split_seed,
    n_jobs,
    progress_label=None,
):
    if classifier in {
        "naive_bayes",
        "xgboost",
        "ips_knn",
        "knn",
        "logistic_regression",
        "svm",
        "decision_tree",
        "random_forest",
    }:
        return evaluate_sklearn_classifier(
            classifier, x_train, x_test, y_train, y_test, split_seed, n_jobs
        )
    return evaluate_fcalc_classifier(
        classifier,
        x_train,
        x_test,
        y_train,
        y_test,
        n_jobs,
        progress_label=progress_label,
    )


def run_raw(args):
    datasets = ordered_datasets(args.datasets)
    classifiers = args.classifiers or CLASSIFIER_ORDER
    unknown_classifiers = sorted(set(classifiers) - set(CLASSIFIER_ORDER))
    if unknown_classifiers:
        raise ValueError(f"Unknown classifier names: {unknown_classifiers}")

    repeat_seed_pairs = [
        (repeat_idx, split_seed)
        for repeat_idx, split_seed in enumerate(REPEAT_SEEDS[: args.repeats], start=1)
        if args.repeat_indices is None or repeat_idx in set(args.repeat_indices)
    ]
    completed = load_completed_keys(args.raw_results_path)
    selected_keys = {
        (dataset, classifier, repeat)
        for dataset in datasets
        for classifier in classifiers
        for repeat, _ in repeat_seed_pairs
    }
    total = len(datasets) * len(classifiers) * len(repeat_seed_pairs)
    done = len(completed & selected_keys)
    start_all = time.time()

    for dataset_idx, dataset_name in enumerate(datasets, start=1):
        x, y = load_full_dataset(dataset_name)
        for repeat_position, (repeat_idx, split_seed) in enumerate(repeat_seed_pairs, start=1):
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=0.2,
                random_state=split_seed,
                stratify=y,
            )
            for classifier_idx, classifier in enumerate(classifiers, start=1):
                key = (dataset_name, classifier, repeat_idx)
                if key in completed:
                    continue
                current = done + 1
                print(
                    f"[{current}/{total}] dataset {dataset_idx}/{len(datasets)} "
                    f"{dataset_name} | repeat {repeat_idx} "
                    f"({repeat_position}/{len(repeat_seed_pairs)}) "
                    f"seed={split_seed} | classifier {classifier_idx}/{len(classifiers)} "
                    f"{classifier}",
                    flush=True,
                )
                start = time.time()
                status = "ok"
                error = ""
                try:
                    score, best_params, cv_f1 = evaluate_one(
                        classifier,
                        x_train,
                        x_test,
                        y_train,
                        y_test,
                        split_seed,
                        args.n_jobs,
                        progress_label=(
                            f"dataset={dataset_name} repeat={repeat_idx} classifier={classifier}"
                        ),
                    )
                except Exception as exc:
                    score = np.nan
                    best_params = {}
                    cv_f1 = np.nan
                    status = "failed"
                    error = repr(exc)
                elapsed = time.time() - start
                append_raw_result(
                    {
                        "dataset": dataset_name,
                        "classifier": classifier,
                        "repeat": repeat_idx,
                        "split_seed": split_seed,
                        "macro_f1": score,
                        "macro_f1_percent": score * 100 if not np.isnan(score) else np.nan,
                        "cv_macro_f1": cv_f1,
                        "best_params": json.dumps(best_params, sort_keys=True),
                        "elapsed_seconds": round(elapsed, 3),
                        "status": status,
                        "error": error,
                    },
                    args.raw_results_path,
                )
                done += 1
                completed.add(key)
                avg = (time.time() - start_all) / max(done, 1)
                remaining = max(total - done, 0) * avg
                print(
                    f"    {status}; macro-F1={score * 100 if not np.isnan(score) else np.nan:.2f}; "
                    f"elapsed={format_duration(elapsed)}; rough remaining={format_duration(remaining)}",
                    flush=True,
                )


def summarize_best_params(params_series):
    params = [json.loads(value) for value in params_series if isinstance(value, str)]
    return json.dumps(params, sort_keys=True)


def build_summary(repeats, raw_results_path):
    df = pd.read_csv(raw_results_path)
    ok = df[df["status"] == "ok"].copy()
    rows = []
    for (dataset, classifier), group in ok.groupby(["dataset", "classifier"], sort=False):
        scores = group.sort_values("repeat")["macro_f1_percent"].to_numpy()
        mean = float(np.mean(scores))
        std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        ci = float(t.ppf(0.975, len(scores) - 1) * std / math.sqrt(len(scores))) if len(scores) > 1 else 0.0
        rows.append(
            {
                "dataset": dataset,
                "classifier": classifier,
                "mean_macro_f1_percent": round(mean, 4),
                "ci95_half_width_percent": round(ci, 4),
                "std_macro_f1_percent": round(std, 4),
                "repeat_scores": json.dumps([round(float(score), 4) for score in scores]),
                "best_params_summary": summarize_best_params(group["best_params"]),
                "successful_repeats": len(scores),
            }
        )
    summary = pd.DataFrame(rows)
    dataset_rank = {name: index for index, name in enumerate(SMALL_FIRST_DATASETS)}
    classifier_rank = {name: index for index, name in enumerate(CLASSIFIER_ORDER)}
    summary["_dataset_rank"] = summary["dataset"].map(dataset_rank).fillna(999).astype(int)
    summary["_classifier_rank"] = summary["classifier"].map(classifier_rank).fillna(999).astype(int)
    summary = summary.sort_values(["_dataset_rank", "_classifier_rank"]).drop(
        columns=["_dataset_rank", "_classifier_rank"]
    )
    summary.to_csv(SUMMARY_PATH, index=False)
    build_latex(summary)
    build_text_summary(summary, df, repeats)


def latex_escape(value):
    return value.replace("_", "\\_")


def build_latex(summary):
    dataset_order = ordered_datasets(None)
    classifier_order = [c for c in CLASSIFIER_ORDER if c in set(summary["classifier"])]
    lines = []
    for dataset in dataset_order:
        group = summary[summary["dataset"] == dataset]
        if group.empty:
            continue
        max_mean = group["mean_macro_f1_percent"].max()
        cells = [latex_escape(DATASET_LABELS.get(dataset, dataset))]
        for classifier in classifier_order:
            row = group[group["classifier"] == classifier]
            if row.empty:
                cells.append("--")
                continue
            mean = row.iloc[0]["mean_macro_f1_percent"]
            ci = row.iloc[0]["ci95_half_width_percent"]
            value = f"{mean:.1f} \\pm {ci:.1f}"
            if mean == max_mean:
                value = f"\\mathbf{{{value}}}"
            cells.append(f"\\({value}\\)")
        lines.append(" & ".join(cells) + r" \\")
    LATEX_PATH.write_text("\n".join(lines) + "\n")


def build_text_summary(summary, raw, repeats):
    failed = raw[raw["status"] != "ok"]
    lines = [
        f"Repeated splits used: R={repeats}.",
        "95% CI formula: t_{0.975, R-1} * sample_std(scores, ddof=1) / sqrt(R), reported in percentage points.",
        f"Successful dataset-classifier summaries: {len(summary)}.",
        f"Failed raw runs: {len(failed)}.",
        "Old single-split comparison: not computed automatically; inspect the new summary against the previous table.",
    ]
    if not failed.empty:
        lines.append("Failures:")
        for _, row in failed.iterrows():
            lines.append(
                f"- {row['dataset']} / {row['classifier']} / repeat {row['repeat']}: {row['error']}"
            )
    TEXT_SUMMARY_PATH.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Repeated-split macro-F1 experiment runner.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--classifiers", nargs="*", default=None)
    parser.add_argument(
        "--repeat-indices",
        nargs="*",
        type=int,
        default=None,
        help="1-based repeat indices to run, e.g. --repeat-indices 1 3 5.",
    )
    parser.add_argument(
        "--raw-results-path",
        type=Path,
        default=RAW_RESULTS_PATH,
        help="CSV path for raw per-repeat rows. Use separate paths for parallel workers.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip summary/LaTeX/text generation after raw runs.",
    )
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for summary.csv, latex_table_body.tex, and summary.txt.",
    )
    return parser.parse_args()


def main():
    global OUTPUT_DIR, SUMMARY_PATH, LATEX_PATH, TEXT_SUMMARY_PATH
    args = parse_args()
    OUTPUT_DIR = args.output_dir
    SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
    LATEX_PATH = OUTPUT_DIR / "latex_table_body.tex"
    TEXT_SUMMARY_PATH = OUTPUT_DIR / "summary.txt"
    if args.repeats > len(REPEAT_SEEDS):
        raise ValueError(f"Only {len(REPEAT_SEEDS)} repeat seeds are defined")
    if args.repeat_indices is not None:
        bad = [idx for idx in args.repeat_indices if idx < 1 or idx > args.repeats]
        if bad:
            raise ValueError(f"Repeat indices out of range 1..{args.repeats}: {bad}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        run_raw(args)
    if not args.no_summary:
        build_summary(args.repeats, args.raw_results_path)
        print(f"Wrote summary to {SUMMARY_PATH}")
        print(f"Wrote LaTeX table body to {LATEX_PATH}")
        print(f"Wrote text summary to {TEXT_SUMMARY_PATH}")
    print(f"Wrote raw results to {args.raw_results_path}")


if __name__ == "__main__":
    main()
