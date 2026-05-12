import argparse
import os
import sys

import pandas as pd


project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = project_root
sys.path.append(project_root)


from config.optimal_hyperparameters import optimal_params  # noqa: E402
from src.dataset import Dataset, known_datasets  # noqa: E402
from src.dataset_preprocessor import DatasetPreprocessor  # noqa: E402
from src.ips_knn_classifier import IPSKNNClassifier  # noqa: E402


def filter_reduced_reason(
    reason: pd.Series, reduced_reason: pd.Series
) -> pd.Series:
    kept_features = {
        feature: reduced_interval
        for feature, reduced_interval in reduced_reason.items()
        if reduced_interval == reason[feature]
    }
    return pd.Series(kept_features, dtype=object)


def has_non_zero_length_interval(reason: pd.Series) -> bool:
    return any(interval[0] != interval[1] for interval in reason.tolist())


def main(dataset_name: str) -> None:
    dataset = Dataset(dataset_name=dataset_name)
    preprocessor = DatasetPreprocessor(dataset).preprocess()
    preprocessor.standardize()

    classifier = IPSKNNClassifier(**optimal_params["ips_knn"][dataset_name])
    classifier.fit(dataset.X_train, dataset.y_train)

    (
        y_pred,
        reason_for_classification_df,
        _,
        reduced_reason_for_classification_df,
        _,
    ) = classifier.predict_with_explanation(
        dataset.X_test,
        dataset_preprocessor=preprocessor,
        include_reduced_reason_for_classification=True,
        include_feature_importance=False,
    )

    candidates = []
    for sample_id in dataset.X_test.index:
        reason = reason_for_classification_df.loc[sample_id]
        reduced_reason = filter_reduced_reason(
            reason, reduced_reason_for_classification_df.loc[sample_id]
        )

        if len(reduced_reason) >= len(reason):
            continue
        if len(reduced_reason) == 0:
            continue
        if not has_non_zero_length_interval(reduced_reason):
            continue

        candidates.append(
            (
                sample_id,
                len(reduced_reason),
                y_pred.loc[sample_id],
                dataset.y_test.loc[sample_id],
            )
        )

    candidates.sort(key=lambda item: (item[1], item[0]))

    for sample_id, reduced_reason_size, predicted_class, actual_class in candidates:
        print(
            f"{sample_id}\t{reduced_reason_size}\t{predicted_class}\t{actual_class}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Find test-set sample ids whose reduced IPS-KNN reason is smaller than "
            "the full reason and contains at least one non-zero-length interval."
        )
    )
    parser.add_argument(
        "dataset_name",
        choices=known_datasets,
        help="Name of the dataset to scan.",
    )
    args = parser.parse_args()
    main(args.dataset_name)
