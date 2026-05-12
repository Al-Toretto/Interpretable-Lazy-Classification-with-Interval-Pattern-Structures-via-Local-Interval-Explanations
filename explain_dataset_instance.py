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


def format_series(series: pd.Series) -> str:
    return series.to_string()


def format_reason(reason: pd.Series) -> str:
    return "\n".join(
        f"{feature}: [{interval[0]}, {interval[1]}]"
        for feature, interval in reason.items()
    )


def format_feature_importance(scores: pd.Series) -> str:
    return "\n".join(
        f"{feature}: {score}"
        for feature, score in scores.sort_values(ascending=False).items()
    )


def format_feature_ranges(df: pd.DataFrame) -> str:
    return "\n".join(
        f"{feature}: [{df[feature].min()}, {df[feature].max()}]"
        for feature in df.columns
    )


def format_explanation_metadata(metadata: dict) -> str:
    return "\n".join(
        [
            f"k: {metadata['k']}",
            f"supporter_label: {metadata['predicted_label']}",
            f"opposer_label: {metadata['opposer_label']}",
            f"supporter_count: {metadata['supporter_count']}",
            f"opposer_count: {metadata['opposer_count']}",
            f"original_supporter_score: {metadata['original_supporter_score']}",
            f"original_opposer_score: {metadata['original_opposer_score']}",
            f"taken_supporter_count: {metadata['taken_supporter_count']}",
            f"taken_supporter_score: {metadata['taken_supporter_score']}",
            f"taken_supporter_indices: {metadata['taken_supporter_indices']}",
        ]
    )


def filter_reduced_reason(
    reason: pd.Series, reduced_reason: pd.Series
) -> pd.Series:
    kept_features = {
        feature: reduced_interval
        for feature, reduced_interval in reduced_reason.items()
        if reduced_interval == reason[feature]
    }
    return pd.Series(kept_features, dtype=object)


def main(dataset_name: str, sample_id: int) -> None:
    dataset = Dataset(dataset_name=dataset_name)
    preprocessor = DatasetPreprocessor(dataset).preprocess()

    if sample_id not in dataset.X_test.index:
        available_ids = sorted(dataset.X_test.index.tolist())
        raise ValueError(
            f"Sample id {sample_id} is not in the test split. "
            f"Choose one of these test ids: {available_ids}"
        )

    preprocessor.standardize()

    classifier = IPSKNNClassifier(**optimal_params["ips_knn"][dataset_name])
    classifier.fit(dataset.X_train, dataset.y_train)

    sample_standardized = dataset.X_test.loc[[sample_id]]
    sample_original = preprocessor.destandardize_df(sample_standardized)
    explanation_metadata = classifier.find_explanation_metadata_for_one_sample(
        sample_standardized.loc[sample_id]
    )

    (
        y_pred,
        reason_for_classification_df,
        _,
        reduced_reason_for_classification_df,
        feature_importance_scores_df,
    ) = classifier.predict_with_explanation(
        sample_standardized,
        dataset_preprocessor=preprocessor,
        include_reduced_reason_for_classification=True,
        include_feature_importance=True,
    )

    predicted_class = y_pred.loc[sample_id]
    correct_class = dataset.y_test.loc[sample_id]
    reason = reason_for_classification_df.loc[sample_id]
    reduced_reason = filter_reduced_reason(
        reason, reduced_reason_for_classification_df.loc[sample_id]
    )
    feature_importance_scores = feature_importance_scores_df.loc[sample_id]
    feature_importance_scores = feature_importance_scores[
        feature_importance_scores.index.isin(reduced_reason.index)
    ]

    print(f"dataset: {dataset_name}")
    print(f"sample_id: {sample_id}")
    print(f"predicted_class: {predicted_class}")
    print(f"correct_class: {correct_class}")
    print(f"reason_size: {len(reason)}")
    print(f"reduced_reason_size: {len(reduced_reason)}")
    print("\nclassification_vote_context:")
    print(format_explanation_metadata(explanation_metadata))
    print("\noriginal_feature_values:")
    print(format_series(sample_original.loc[sample_id]))
    print("\nfull_feature_ranges:")
    print(format_feature_ranges(dataset.X))
    print("\nreason_for_classification:")
    print(format_reason(reason))
    print("\nreduced_reason_for_classification:")
    print(format_reason(reduced_reason))
    print("\nfeature_importance_scores:")
    print(format_feature_importance(feature_importance_scores))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Explain one IPS-KNN prediction for a predefined dataset. "
            "The sample id must be an index from the test split produced by the repo."
        )
    )
    parser.add_argument(
        "dataset_name",
        choices=known_datasets,
        help="Name of the dataset to load.",
    )
    parser.add_argument(
        "sample_id",
        type=int,
        help="Row index of the sample in the test split.",
    )
    args = parser.parse_args()
    main(args.dataset_name, args.sample_id)
