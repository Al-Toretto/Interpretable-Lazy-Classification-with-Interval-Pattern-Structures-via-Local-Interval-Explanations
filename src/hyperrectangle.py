from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple
import copy


class Hyperrectangle:
    def __init__(
        self, intervals: List[Tuple[float, float]], is_postive: bool | None = None
    ):
        # a list of two elements for the shape [(min1, max1), (min2, max2), ...]
        self.intervals = intervals
        self.is_positive = is_postive
        self.volume = self._calculate_volume()

    def __copy__(self):
        # Create a shallow copy of the object
        intervals_copy = copy.copy(self.intervals)
        return Hyperrectangle(intervals_copy, self.is_positive)

    def __deepcopy__(self, memo):
        # Create a deep copy of the object
        intervals_copy = copy.deepcopy(self.intervals, memo)
        return Hyperrectangle(intervals_copy, self.is_positive)

    def flatten(self) -> List[float]:
        return [val for tup in self.intervals for val in tup]

    def _calculate_volume(self) -> float:
        return np.prod([abs(tup[1] - tup[0]) for tup in self.intervals])

    def __str__(self) -> str:
        if self.is_positive is not None:
            hyperrectangle_class = (
                f"is {'positive' if self.is_positive else 'negative'}"
            )
        else:
            hyperrectangle_class = ""
        intervals_str = ", ".join(
            [f"({min_val}, {max_val})" for min_val, max_val in self.intervals]
        )
        return intervals_str + " " + hyperrectangle_class

    def contains_point(self, point: Tuple[float, ...] | List[float]) -> bool:
        for i, interval in enumerate(self.intervals):
            if not interval[0] <= point[i] <= interval[1]:
                return False
        return True

    def l1_distance_vector_from_point(
        self, point: Tuple[float, ...] | List[float]
    ) -> List[float]:
        l1_distances = []
        for i in range(len(point)):
            if point[i] < min(self.intervals[i]):
                l1_distances.append(min(self.intervals[i]) - point[i])
            elif point[i] > max(self.intervals[i]):
                l1_distances.append(max(self.intervals[i]) - point[i])
            else:
                l1_distances.append(0.0)
        return l1_distances

    def distance_to_point(
        self, point: Tuple[float, ...] | List[float], p: int = 2
    ) -> float:
        if p == 2:
            squared_distances = [
                max(min(upper, lower) - coord, 0, coord - max(upper, lower)) ** 2
                for coord, (lower, upper) in zip(point, self.intervals)
            ]
            return np.sqrt(sum(squared_distances))
        elif p == 1:
            return np.sum(np.abs(self.l1_distance_vector_from_point(point)))
        else:
            raise ValueError("Unsupported value for p. Choose 1 or 2.")

    def furthest_distance_to_point(
        self, point: Tuple[float, ...] | List[float], p: int = 2
    ) -> float:
        l1_distances = []
        for i in range(len(point)):
            lower = min(self.intervals[i])
            upper = max(self.intervals[i])
            if lower <= point[i] and point[i] <= upper:
                l1_distances.append(0.0)
            else:
                l1_distances.append(max(upper - point[i], point[i] - lower))
        if p == 2:
            return np.sqrt(np.sum([x**2 for x in l1_distances]))
        elif p == 1:
            return np.sum(l1_distances)
        else:
            raise ValueError("Unsupported value for p. Choose 1 or 2.")

    def find_condition_list_inside_hyperrectangle(self, df: pd.DataFrame):
        return [
            (val[0] <= df.iloc[:, i]) & (df.iloc[:, i] <= val[1])
            for i, val in enumerate(self.intervals)
        ]
