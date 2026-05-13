from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))

from src.dataset import Dataset, known_datasets  # noqa: E402
from src.dataset_preprocessor import DatasetPreprocessor  # noqa: E402
from src.ips_knn_classifier import IPSKNNClassifier  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "output" / "repeated_sizes"
DEFAULT_SOURCE_RESULTS_PATH = (
    PROJECT_ROOT / "output" / "repeated_macro_f1" / "raw_repeat_results.csv"
)
RAW_SIZES_PATH = OUTPUT_DIR / "raw_repeat_sizes.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
COMPACTNESS_PATH = OUTPUT_DIR / "compactness.csv"
LATEX_PATH = OUTPUT_DIR / "latex_table_body.tex"

DEFAULT_REPEATS = 10
REPEAT_SEEDS = [1998 + i for i in range(DEFAULT_REPEATS)]

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

RAW_COLUMNS = [
    "dataset",
    "classifier",
    "repeat",
    "split_seed",
    "n_train",
    "n_test",
    "n_features",
    "n_classes",
    "best_params",
    "primary_metric",
    "primary_value",
    "secondary_metric",
    "secondary_value",
    "tertiary_metric",
    "tertiary_value",
    "avg_rrc_size",
    "max_rrc_size",
    "rrc_rc_ratio",
    "elapsed_seconds",
    "status",
    "error",
]


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


def format_duration(seconds):
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def load_source_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Source results file does not exist: {path}")
    df = pd.read_csv(path)
    required = {"dataset", "classifier", "repeat", "split_seed", "best_params", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Source results file is missing columns: {sorted(missing)}")
    df = df[df["status"] == "ok"].copy()
    df["repeat"] = df["repeat"].astype(int)
    df["split_seed"] = df["split_seed"].astype(int)
    return df


def load_completed_keys(path: Path):
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    ok = df[df["status"] == "ok"]
    return set(zip(ok["dataset"], ok["classifier"], ok["repeat"]))


def append_raw_size(row: dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    pd.DataFrame([row], columns=RAW_COLUMNS).to_csv(
        path, mode="a", header=write_header, index=False
    )


def metric_row(
    dataset,
    classifier,
    repeat,
    split_seed,
    n_train,
    n_test,
    n_features,
    n_classes,
    best_params,
    metrics,
    elapsed,
    status="ok",
    error="",
    avg_rrc_size=np.nan,
    max_rrc_size=np.nan,
    rrc_rc_ratio=np.nan,
):
    padded = list(metrics)[:3] + [(None, np.nan)] * (3 - len(metrics))
    return {
        "dataset": dataset,
        "classifier": classifier,
        "repeat": repeat,
        "split_seed": split_seed,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": n_features,
        "n_classes": n_classes,
        "best_params": json.dumps(best_params, sort_keys=True),
        "primary_metric": padded[0][0],
        "primary_value": padded[0][1],
        "secondary_metric": padded[1][0],
        "secondary_value": padded[1][1],
        "tertiary_metric": padded[2][0],
        "tertiary_value": padded[2][1],
        "avg_rrc_size": avg_rrc_size,
        "max_rrc_size": max_rrc_size,
        "rrc_rc_ratio": rrc_rc_ratio,
        "elapsed_seconds": round(elapsed, 3),
        "status": status,
        "error": error,
    }


def count_path_features(classifier, x_test):
    decision_paths = classifier.decision_path(x_test)
    n_features_per_sample = []
    for row in range(decision_paths.shape[0]):
        start = decision_paths.indptr[row]
        end = decision_paths.indptr[row + 1]
        sample_node_indices = decision_paths.indices[start:end]
        features_used = classifier.tree_.feature[sample_node_indices]
        unique_features = np.unique(features_used[features_used >= 0])
        n_features_per_sample.append(len(unique_features))
    return n_features_per_sample


def evaluate_size(classifier, x_train, x_test, y_train, best_params, split_seed):
    n_features = x_train.shape[1]
    n_train = len(x_train)
    n_classes = pd.Series(y_train).nunique()

    if classifier == "naive_bayes":
        return [("size", 4 * n_features + 2)], {}

    if classifier == "xgboost":
        return [
            ("n_estimators", best_params["n_estimators"]),
            ("max_depth", best_params["max_depth"]),
        ], {}

    if classifier == "ips_knn":
        scaler = StandardScaler().fit(x_train)
        x_train_scaled = pd.DataFrame(
            scaler.transform(x_train), columns=x_train.columns, index=x_train.index
        )
        x_test_scaled = pd.DataFrame(
            scaler.transform(x_test), columns=x_test.columns, index=x_test.index
        )
        model = IPSKNNClassifier(**best_params)
        model.fit(x_train_scaled, y_train)
        _, _, size_dict, _, _ = model._predict_with_explanation(
            x_test_scaled,
            include_reduced_reason_for_classification=True,
            include_feature_importance=False,
        )
        sizes = np.array(list(size_dict.values()), dtype=float)
        avg_rrc_size = float(np.mean(sizes))
        max_rrc_size = float(np.max(sizes))
        rrc_rc_ratio = avg_rrc_size / n_features
        return [
            ("avg_rrc_size", avg_rrc_size),
            ("max_rrc_size", max_rrc_size),
        ], {
            "avg_rrc_size": avg_rrc_size,
            "max_rrc_size": max_rrc_size,
            "rrc_rc_ratio": rrc_rc_ratio,
        }

    if classifier == "knn":
        return [("size", n_features * best_params["n_neighbors"])], {}

    if classifier == "logistic_regression":
        return [("size", n_features + 1)], {}

    if classifier == "svm":
        if best_params["kernel"] == "linear":
            return [("size", n_features + 1)], {}
        scaler = StandardScaler().fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        model = SVC(random_state=split_seed, **best_params)
        model.fit(x_train_scaled, y_train)
        n_support_vectors = int(np.sum(model.n_support_))
        size = n_features * n_support_vectors + n_support_vectors + 2
        return [
            ("n_support_vectors", n_support_vectors),
            ("size", size),
        ], {}

    if classifier == "decision_tree":
        model = DecisionTreeClassifier(random_state=split_seed, **best_params)
        model.fit(x_train, y_train)
        path_sizes = count_path_features(model, x_test)
        return [
            ("configured_max_depth", best_params.get("max_depth")),
            ("max_path_features", float(np.max(path_sizes))),
            ("avg_path_features", float(np.mean(path_sizes))),
        ], {}

    if classifier == "random_forest":
        max_depth = best_params.get("max_depth")
        if max_depth is None:
            model = RandomForestClassifier(random_state=split_seed, n_jobs=1, **best_params)
            model.fit(x_train, y_train)
            max_depth = max(estimator.tree_.max_depth for estimator in model.estimators_)
        return [
            ("n_estimators", best_params["n_estimators"]),
            ("max_depth", max_depth),
        ], {}

    if classifier == "fcalc":
        return [("size", n_train * n_features)], {}

    if classifier == "fcalc_rand":
        return [
            ("size", n_classes * best_params["num_iters"] * n_features),
            ("num_iters", best_params["num_iters"]),
        ], {}

    raise ValueError(f"Unknown classifier: {classifier}")


def source_row_for(source_df, dataset, classifier, repeat):
    rows = source_df[
        (source_df["dataset"] == dataset)
        & (source_df["classifier"] == classifier)
        & (source_df["repeat"] == repeat)
    ]
    if rows.empty:
        raise KeyError(f"No source row for {dataset}/{classifier}/repeat={repeat}")
    if len(rows) > 1:
        raise ValueError(f"Duplicate source rows for {dataset}/{classifier}/repeat={repeat}")
    return rows.iloc[0]


def run_raw(args):
    source = load_source_results(args.source_results_path)
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
    completed = load_completed_keys(args.raw_sizes_path)
    selected = [
        (dataset, classifier, repeat_idx, split_seed)
        for dataset in datasets
        for classifier in classifiers
        for repeat_idx, split_seed in repeat_seed_pairs
    ]
    total = len(selected)
    done = len(completed)
    start_all = time.time()

    for dataset_idx, dataset_name in enumerate(datasets, start=1):
        x, y = load_full_dataset(dataset_name)
        for repeat_position, (repeat_idx, split_seed) in enumerate(repeat_seed_pairs, start=1):
            x_train, x_test, y_train, _ = train_test_split(
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
                best_params = {}
                metrics = []
                extra = {}
                try:
                    try:
                        source_row = source_row_for(source, dataset_name, classifier, repeat_idx)
                    except KeyError:
                        if classifier != "fcalc":
                            raise
                        source_row = None
                    if source_row is not None:
                        if int(source_row["split_seed"]) != split_seed:
                            raise ValueError(
                                f"Source split seed {source_row['split_seed']} != expected {split_seed}"
                            )
                        best_params = json.loads(source_row["best_params"])
                    metrics, extra = evaluate_size(
                        classifier, x_train, x_test, y_train, best_params, split_seed
                    )
                except Exception as exc:
                    status = "failed"
                    error = repr(exc)
                elapsed = time.time() - start
                row = metric_row(
                    dataset_name,
                    classifier,
                    repeat_idx,
                    split_seed,
                    len(x_train),
                    len(x_test),
                    x_train.shape[1],
                    pd.Series(y_train).nunique(),
                    best_params,
                    metrics,
                    elapsed,
                    status=status,
                    error=error,
                    **extra,
                )
                append_raw_size(row, args.raw_sizes_path)
                done += 1
                completed.add(key)
                avg = (time.time() - start_all) / max(done, 1)
                remaining = max(total - done, 0) * avg
                metric_text = ", ".join(f"{name}={value}" for name, value in metrics)
                print(
                    f"    {status}; {metric_text}; elapsed={format_duration(elapsed)}; "
                    f"rough remaining={format_duration(remaining)}",
                    flush=True,
                )


def mean_or_nan(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.mean())


def metric_values(group: pd.DataFrame, metric_name: str) -> list[float]:
    values = []
    for metric_col, value_col in [
        ("primary_metric", "primary_value"),
        ("secondary_metric", "secondary_value"),
        ("tertiary_metric", "tertiary_value"),
    ]:
        matched = group.loc[group[metric_col] == metric_name, value_col]
        values.extend(pd.to_numeric(matched, errors="coerce").dropna().astype(float).tolist())
    return values


def metric_summary(group: pd.DataFrame, metric_name: str):
    values = metric_values(group, metric_name)
    if not values:
        return np.nan, []
    return float(np.mean(values)), [round(float(value), 6) for value in values]


SUMMARY_METRICS = {
    "fcalc": ["size"],
    "fcalc_rand": ["size", "num_iters"],
    "ips_knn": ["avg_rrc_size", "max_rrc_size"],
    "knn": ["size"],
    "naive_bayes": ["size"],
    "logistic_regression": ["size"],
    "svm": ["size", "n_support_vectors"],
    "decision_tree": ["avg_path_features", "max_path_features", "configured_max_depth"],
    "random_forest": ["n_estimators", "max_depth"],
    "xgboost": ["n_estimators", "max_depth"],
}


def build_summary(raw_sizes_path: Path):
    df = pd.read_csv(raw_sizes_path)
    ok = df[df["status"] == "ok"].copy()
    rows = []
    for (dataset, classifier), group in ok.groupby(["dataset", "classifier"], sort=False):
        metric_names = list(SUMMARY_METRICS[classifier])
        summaries = [metric_summary(group, metric_name) for metric_name in metric_names]
        while len(metric_names) < 3:
            metric_names.append(None)
            summaries.append((np.nan, []))
        primary_mean, primary_values = summaries[0]
        secondary_mean, secondary_values = summaries[1]
        tertiary_mean, tertiary_values = summaries[2]
        rows.append(
            {
                "dataset": dataset,
                "classifier": classifier,
                "primary_metric": metric_names[0],
                "mean_primary_value": round(primary_mean, 6)
                if not pd.isna(primary_mean)
                else np.nan,
                "secondary_metric": metric_names[1],
                "mean_secondary_value": round(secondary_mean, 6)
                if not pd.isna(secondary_mean)
                else np.nan,
                "tertiary_metric": metric_names[2],
                "mean_tertiary_value": round(tertiary_mean, 6)
                if not pd.isna(tertiary_mean)
                else np.nan,
                "primary_repeat_values": json.dumps(primary_values),
                "secondary_repeat_values": json.dumps(secondary_values),
                "tertiary_repeat_values": json.dumps(tertiary_values),
                "successful_repeats": len(group),
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


def latex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def display_dataset_name(dataset: str) -> str:
    return {
        "breast_cancer": "Breast Cancer",
        "page_blocks": "Page Blocks",
        "image_segmentation": "Segmentation",
        "parkinsons": "Parkinson's",
        "spam": "SpamBase",
    }.get(dataset, dataset.replace("_", " ").title())


def format_table_value(value):
    if value is None or pd.isna(value):
        return "--"
    value = float(value)
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.1f}"


def maybe_bold(text: str, is_best: bool) -> str:
    if not is_best or text == "--":
        return text
    return f"\\textbf{{{text}}}"


def mean_metric(group: pd.DataFrame, metric_name: str):
    values = metric_values(group, metric_name)
    if not values:
        return np.nan
    return float(np.mean(values))


def table_values_for_group(group: pd.DataFrame) -> dict[str, float]:
    classifier = group["classifier"].iloc[0]
    if classifier in {
        "fcalc",
        "fcalc_rand",
        "knn",
        "naive_bayes",
        "logistic_regression",
        "svm",
    }:
        return {"size": mean_metric(group, "size")}
    if classifier == "ips_knn":
        return {
            "avg": mean_metric(group, "avg_rrc_size"),
            "max": mean_metric(group, "max_rrc_size"),
        }
    if classifier == "decision_tree":
        return {
            "avg": mean_metric(group, "avg_path_features"),
            "max": mean_metric(group, "max_path_features"),
        }
    if classifier in {"random_forest", "xgboost"}:
        return {
            "n": mean_metric(group, "n_estimators"),
            "h": mean_metric(group, "max_depth"),
        }
    raise ValueError(f"Unknown classifier for table: {classifier}")


def build_latex(raw_sizes_path: Path):
    df = pd.read_csv(raw_sizes_path)
    ok = df[df["status"] == "ok"].copy()
    grouped = {
        (dataset, classifier): table_values_for_group(group)
        for (dataset, classifier), group in ok.groupby(["dataset", "classifier"], sort=False)
    }

    lines = []
    for dataset in ordered_datasets(None):
        dataset_values = {
            classifier: grouped.get((dataset, classifier), {})
            for classifier in CLASSIFIER_ORDER
        }
        comparable = []
        comparable.extend(
            dataset_values[classifier].get("size", np.nan)
            for classifier in [
                "fcalc",
                "fcalc_rand",
                "knn",
                "naive_bayes",
                "logistic_regression",
                "svm",
            ]
        )
        comparable.extend(
            [
                dataset_values["ips_knn"].get("avg", np.nan),
                dataset_values["ips_knn"].get("max", np.nan),
                dataset_values["decision_tree"].get("avg", np.nan),
                dataset_values["decision_tree"].get("max", np.nan),
            ]
        )
        comparable = [value for value in comparable if not pd.isna(value)]
        best = min(comparable) if comparable else np.nan

        def cell(value):
            text = format_table_value(value)
            is_best = not pd.isna(best) and not pd.isna(value) and math.isclose(
                float(value), float(best), rel_tol=1e-12, abs_tol=1e-12
            )
            return maybe_bold(text, is_best)

        cells = [
            latex_escape(display_dataset_name(dataset)),
            cell(dataset_values["fcalc"].get("size", np.nan)),
            cell(dataset_values["fcalc_rand"].get("size", np.nan)),
            cell(dataset_values["ips_knn"].get("avg", np.nan)),
            cell(dataset_values["ips_knn"].get("max", np.nan)),
            cell(dataset_values["knn"].get("size", np.nan)),
            cell(dataset_values["naive_bayes"].get("size", np.nan)),
            cell(dataset_values["logistic_regression"].get("size", np.nan)),
            cell(dataset_values["svm"].get("size", np.nan)),
            cell(dataset_values["decision_tree"].get("avg", np.nan)),
            cell(dataset_values["decision_tree"].get("max", np.nan)),
            format_table_value(dataset_values["random_forest"].get("n", np.nan)),
            format_table_value(dataset_values["random_forest"].get("h", np.nan)),
            format_table_value(dataset_values["xgboost"].get("n", np.nan)),
            format_table_value(dataset_values["xgboost"].get("h", np.nan)),
        ]
        lines.append(" & ".join(cells) + r" \\")
    LATEX_PATH.write_text("\n".join(lines) + "\n")


def build_compactness(raw_sizes_path: Path):
    df = pd.read_csv(raw_sizes_path)
    ips = df[(df["classifier"] == "ips_knn") & (df["status"] == "ok")].copy()
    rows = []
    for dataset, group in ips.groupby("dataset", sort=False):
        rows.append(
            {
                "dataset": dataset,
                "mean_avg_rrc_size": round(float(group["avg_rrc_size"].mean()), 6),
                "max_max_rrc_size": round(float(group["max_rrc_size"].max()), 6),
                "mean_rrc_rc_ratio": round(float(group["rrc_rc_ratio"].mean()), 6),
                "successful_repeats": len(group),
            }
        )
    compactness = pd.DataFrame(rows)
    dataset_rank = {name: index for index, name in enumerate(SMALL_FIRST_DATASETS)}
    compactness["_dataset_rank"] = (
        compactness["dataset"].map(dataset_rank).fillna(999).astype(int)
    )
    compactness = compactness.sort_values("_dataset_rank").drop(columns=["_dataset_rank"])
    compactness.to_csv(COMPACTNESS_PATH, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Repeated-split classifier size experiment.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
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
        "--source-results-path",
        type=Path,
        default=DEFAULT_SOURCE_RESULTS_PATH,
        help="Raw repeated macro-F1 CSV containing the selected best_params.",
    )
    parser.add_argument(
        "--raw-sizes-path",
        type=Path,
        default=RAW_SIZES_PATH,
        help="CSV path for raw per-repeat size rows.",
    )
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
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
        build_summary(args.raw_sizes_path)
        build_compactness(args.raw_sizes_path)
        build_latex(args.raw_sizes_path)


if __name__ == "__main__":
    main()
