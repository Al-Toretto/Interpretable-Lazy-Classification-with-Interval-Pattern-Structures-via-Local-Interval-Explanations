import pandas as pd
import numpy as np
from typing import Tuple, Any, Dict, List


class InformationGainAnalyzer:
    def __init___(self):
        pass

    def find_entropy(y: pd.Series) -> float:
        _, counts = np.unique(y, return_counts=True)
        probalbilities = counts / len(y)
        return -np.sum(probalbilities * np.log2(probalbilities))

    def find_informatio_gain(
        parent_entropy: float, left_branch: pd.Series, right_branch: pd.Series
    ) -> float:
        total_samples = len(left_branch) + len(right_branch)
        left_weight = len(left_branch) / total_samples
        right_weight = len(right_branch) / total_samples
        return parent_entropy - (
            left_weight * InformationGainAnalyzer.find_entropy(left_branch)
            + right_weight * InformationGainAnalyzer.find_entropy(right_branch)
        )

    def find_information_gain_for_splitting_by_interval(
        X: pd.DataFrame,
        y: pd.Series,
        numerical_feature: Any,
        interval: Tuple[float, float],
    ) -> float:
        y_inside = y[
            (interval[0] <= X[numerical_feature])
            & (X[numerical_feature] <= interval[1])
        ]
        y_outside = y[y.index.difference(y_inside.index)]
        original_entropy = InformationGainAnalyzer.find_entropy(y)
        information_gain = InformationGainAnalyzer.find_informatio_gain(
            original_entropy, y_inside, y_outside
        )
        return information_gain

    def order_hyperrectangle_intervals_by_information_gain(
        hyperrectangle: Dict[Any, Tuple[float, float]], X: pd.DataFrame, y: pd.Series
    ) -> List[Tuple[Any, float]]:
        ordered_features = []
        for feature, interval in hyperrectangle.items():
            ordered_features.append(
                (
                    feature,
                    InformationGainAnalyzer.find_information_gain_for_splitting_by_interval(
                        X, y, feature, interval
                    ),
                )
            )
        return sorted(ordered_features, key=lambda x: x[1], reverse=True)
