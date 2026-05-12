from __future__ import annotations

import numpy as np
import pandas as pd
import functools
from typing import Tuple, Dict, Any

from src.information_gain_analyzer import InformationGainAnalyzer
from src.hyperrectangle import Hyperrectangle
from src.dataset_preprocessor import DatasetPreprocessor


def expand_hyperrectangle_by_information_gain(
    rect: Dict[Any, Tuple[float, float]], X: pd.DataFrame, y: pd.Series
) -> Tuple[Dict[Any, Tuple[float, float]], int]:
    hyperrectangle = Hyperrectangle([(rect[col][0], rect[col][1]) for col in X.columns])
    condition_list = hyperrectangle.find_condition_list_inside_hyperrectangle(X)

    mask = functools.reduce(lambda x, y: x & y, condition_list)
    count_inside_rect = mask.sum()

    cols = list(X.columns)
    used_cols = []

    X_new = X.copy()
    y_new = y.copy()
    while len(X_new) > count_inside_rect:
        max_gain = -1
        max_gain_col = None
        for col in cols:
            if col in used_cols:
                continue
            gain = (
                InformationGainAnalyzer.find_information_gain_for_splitting_by_interval(
                    X_new, y_new, col, rect[col]
                )
            )
            if gain > max_gain:
                max_gain = gain
                max_gain_col = col
        used_cols.append(max_gain_col)
        X_new = X_new[
            (X_new.loc[:, max_gain_col] >= rect[max_gain_col][0])
            & (X_new.loc[:, max_gain_col] <= rect[max_gain_col][1])
        ]
        y_new = y_new[X_new.index]
    new_rect = rect.copy()
    used_cols = set(used_cols)

    for col in new_rect:
        if col not in used_cols:
            new_rect[col] = (X[col].min(), X[col].max())
    return new_rect, len(used_cols)


def find_feature_importance_scores_for_hyperrectangle(
    rect: Dict[Any, Tuple[float, float]], X: pd.DataFrame, y: pd.Series
) -> Dict[Any, float]:
    scores = {
        col: InformationGainAnalyzer.find_information_gain_for_splitting_by_interval(
            X, y, col, interval
        )
        for col, interval in rect.items()
    }
    sum_scores = sum(scores.values())
    if sum_scores != 0:
        scores = {col: score / sum_scores for col, score in scores.items()}
    return scores


class IPSKNNClassifier:
    def __init__(self, k=3, p=2, weights="distance"):
        self.k = k
        self.p = p
        self.weights = weights
        self.X_train = None
        self.y_train = None
        self.org_X_train = None
        self.dataset_preprocessor: DatasetPreprocessor = None
        self.classifier_sizes = []

    def _find_sorted_distances_and_samples_mask_supporting_one_sample(
        self, sample: pd.Series, df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        distances = np.power(
            np.sum(np.power(np.subtract(df, sample), self.p), axis="columns"),
            1 / self.p,
        ).sort_values(kind="mergesort")
        knn_indeces = np.array(distances[: self.k].index)
        knn = df.loc[knn_indeces]

        knn_hyperrectangle = Hyperrectangle(
            [(knn[col].min(), knn[col].max()) for col in df.columns]
        )
        conditions_list = knn_hyperrectangle.find_condition_list_inside_hyperrectangle(
            df
        )
        inside_hyperrectangle_mask = functools.reduce(
            lambda x, y: x & y, conditions_list
        )
        return distances, inside_hyperrectangle_mask

    def get_params(self, deep=True):
        return {"k": self.k, "p": self.p, "weights": self.weights}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def destandardize_features(self, dataset_preprocessor: DatasetPreprocessor):
        self.dataset_preprocessor = dataset_preprocessor
        self.org_X_train = dataset_preprocessor.destandardize_df(self.X_train)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.X_train = X_train
        self.y_train = y_train

    def _find_vote_score_for_distance(self, distance: float) -> float:
        if self.weights == "uniform":
            return 1
        if self.weights == "distance":
            return 1 / distance if distance != 0 else 10000
        raise ValueError(f"Unknown weights value: {self.weights}")

    def _find_votes_for_one_sample(
        self, sample: pd.Series
    ) -> Tuple[Dict[Any, float], pd.Series, pd.Series]:
        distances, mask = (
            self._find_sorted_distances_and_samples_mask_supporting_one_sample(
                sample, self.X_train
            )
        )
        supporting_samples_labels = self.y_train[mask]
        votes = {}
        if self.weights == "uniform":
            votes = supporting_samples_labels.value_counts().to_dict()
        elif self.weights == "distance":
            for index, label in supporting_samples_labels.items():
                if label in votes:
                    votes[label] += self._find_vote_score_for_distance(
                        distances[index]
                    )
                else:
                    votes[label] = self._find_vote_score_for_distance(
                        distances[index]
                    )
        return votes, distances, mask

    def _predict_one_sample(self, sample: pd.Series) -> Any:
        votes, _, _ = self._find_votes_for_one_sample(sample)
        max_support = -1
        max_label = ""
        for label, support in votes.items():
            if support > max_support:
                max_support = support
                max_label = label
        return max_label

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        y_pred = pd.Series(data=np.nan, index=X_test.index, dtype=object)
        for index, sample in X_test.iterrows():
            y_pred[index] = self._predict_one_sample(sample)
        return y_pred

    def _find_reduced_reason_for_classification(
        self, reason_for_classification: Dict[Any, Tuple[float, float]]
    ) -> Tuple[Dict[Any, Tuple[float, float]], int]:
        return expand_hyperrectangle_by_information_gain(
            reason_for_classification, self.X_train, self.y_train
        )

    def _find_feature_importance_scores(
        self, rect: Dict[Any, Tuple[float, float]]
    ) -> Dict[Any, float]:
        return find_feature_importance_scores_for_hyperrectangle(
            rect, self.X_train, self.y_train
        )

    def _predict_with_explanation_one_sample(
        self,
        sample: pd.Series,
    ) -> Tuple[
        Any,
        Dict[Any, Tuple[float, float]],
        Dict[Any, Tuple[float, float]] | None,
    ]:
        reason_for_classification = None

        (
            largest_label,
            reason_for_classification,
            _,
        ) = self._predict_with_explanation_one_sample_with_metadata(sample)

        return (
            largest_label,
            reason_for_classification,
        )

    def _predict_with_explanation_one_sample_with_metadata(
        self,
        sample: pd.Series,
    ) -> Tuple[
        Any,
        Dict[Any, Tuple[float, float]],
        Dict[str, Any],
    ]:
        votes, distances, mask = self._find_votes_for_one_sample(sample)

        sorted_votes_list = sorted(
            votes.items(), key=lambda item: item[1], reverse=True
        )
        largest_label = sorted_votes_list[0][0]
        largest_score = sorted_votes_list[0][1]
        second_largest_label = None
        second_largest_score = 0
        if len(sorted_votes_list) > 1:
            second_largest_label = sorted_votes_list[1][0]
            second_largest_score = sorted_votes_list[1][1]

        mask_voters_with_largest_label = mask & (self.y_train == largest_label)
        mask_voters_with_other_labels = mask & (self.y_train != largest_label)
        distances_supporting_samples_with_largest_label = distances[
            mask_voters_with_largest_label
        ]

        explanation_score = 0
        explanation_indices = []
        for (
            dist_index,
            dist_value,
        ) in distances_supporting_samples_with_largest_label.items():
            explanation_score += self._find_vote_score_for_distance(dist_value)
            explanation_indices.append(dist_index)
            if explanation_score >= second_largest_score:
                break

        explanation_samples = self.X_train.loc[explanation_indices]

        reason_for_classification = {
            col: (
                min(explanation_samples[col].min(), sample.loc[col]),
                max(explanation_samples[col].max(), sample.loc[col]),
            )
            for col in self.X_train.columns
        }

        return (
            largest_label,
            reason_for_classification,
            {
                "k": self.k,
                "predicted_label": largest_label,
                "opposer_label": second_largest_label,
                "supporter_count": int(mask_voters_with_largest_label.sum()),
                "opposer_count": int(mask_voters_with_other_labels.sum()),
                "original_supporter_score": largest_score,
                "original_opposer_score": second_largest_score,
                "taken_supporter_count": len(explanation_indices),
                "taken_supporter_score": explanation_score,
                "taken_supporter_indices": explanation_indices,
            },
        )

    def find_explanation_metadata_for_one_sample(
        self, sample: pd.Series
    ) -> Dict[str, Any]:
        _, _, metadata = self._predict_with_explanation_one_sample_with_metadata(
            sample
        )
        return metadata

    def _predict_with_explanation(
        self,
        X_test: pd.DataFrame,
        include_reduced_reason_for_classification: bool = False,
        include_feature_importance: bool = False,
    ) -> Tuple[
        pd.Series,
        pd.DataFrame,
        Dict[Any, int],
        pd.DataFrame,
        pd.DataFrame,
    ]:
        y_pred = pd.Series(data=np.nan, index=X_test.index, dtype=object)

        reason_for_classification_dict = {}
        reduced_reason_for_classification_dict = {}
        feature_importance_scores_dict = {}

        classifier_size_dict = {}

        reduced_reason_for_classification_df = None
        feature_importance_scores_df = None

        for index, sample in X_test.iterrows():
            (
                predicted_label,
                reason_for_classification,
            ) = self._predict_with_explanation_one_sample(
                sample,
            )
            y_pred[index] = predicted_label

            reason_for_classification_dict[index] = reason_for_classification
            classifier_size = len(reason_for_classification)
            if include_reduced_reason_for_classification:
                reduced_reason_for_classification_dict[index], classifier_size = (
                    self._find_reduced_reason_for_classification(
                        reason_for_classification
                    )
                )

            classifier_size_dict[index] = classifier_size

            if include_feature_importance:
                if include_reduced_reason_for_classification:
                    feature_importance_scores = self._find_feature_importance_scores(
                        reduced_reason_for_classification_dict[index]
                    )
                else:
                    feature_importance_scores = self._find_feature_importance_scores(
                        reason_for_classification
                    )

                feature_importance_scores_dict[index] = feature_importance_scores

        reason_for_classification_df = pd.DataFrame.from_dict(
            reason_for_classification_dict, orient="index"
        )
        if include_reduced_reason_for_classification:
            reduced_reason_for_classification_df = pd.DataFrame.from_dict(
                reduced_reason_for_classification_dict, orient="index"
            )
        if include_feature_importance:
            feature_importance_scores_df = pd.DataFrame.from_dict(
                feature_importance_scores_dict, orient="index"
            )

        return (
            y_pred,
            reason_for_classification_df,
            classifier_size_dict,
            reduced_reason_for_classification_df,
            feature_importance_scores_df,
        )

    def predict_with_explanation(
        self,
        X_test,
        dataset_preprocessor : DatasetPreprocessor,
        include_reduced_reason_for_classification: bool = False,
        include_feature_importance: bool = False,    
    ) -> Tuple[
        pd.Series,
        pd.DataFrame,
        Dict[Any, int],
        pd.DataFrame,
        pd.DataFrame,
    ]:
        self.dataset_preprocessor = dataset_preprocessor
        (
            y_pred,
            reason_for_classification_df,
            classifier_size_dict,
            reduced_reason_for_classification_df,
            feature_importance_scores_df,
        ) = self._predict_with_explanation(
            X_test,
            include_reduced_reason_for_classification,
            include_feature_importance,
        )

        reason_for_classification_df = (
            self.dataset_preprocessor.destandardize_df_of_ranges(
                reason_for_classification_df
            )
        )
        if include_reduced_reason_for_classification:
            reduced_reason_for_classification_df = (
                self.dataset_preprocessor.destandardize_df_of_ranges(
                    reduced_reason_for_classification_df
                )
            )

        return (
            y_pred,
            reason_for_classification_df,
            classifier_size_dict,
            reduced_reason_for_classification_df,
            feature_importance_scores_df,
        )
