"""Best decision-tree split under weighted Gini impurity.

Interview prompt: enumerate valid thresholds, score the two partitions, and
return the feature and threshold with the lowest weighted impurity.
"""

from __future__ import annotations

import numpy as np


def gini_impurity(labels: np.ndarray) -> float:
    """Compute Gini impurity for a one-dimensional label array."""
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if len(values) == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / len(values)
    return float(1.0 - np.sum(probabilities**2))


def best_gini_split(
    features: np.ndarray, labels: np.ndarray
) -> tuple[int, float, float] | None:
    """Find the feature and threshold with minimum weighted Gini impurity."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    if x.ndim != 2 or not x.size or not np.all(np.isfinite(x)):
        raise ValueError("features must be a non-empty finite [N, D] matrix")
    if y.shape != (x.shape[0],):
        raise ValueError("labels must have one value per row")

    best: tuple[int, float, float] | None = None
    for feature_index in range(x.shape[1]):
        unique_values = np.unique(x[:, feature_index])
        lower_values = unique_values[:-1]
        upper_values = unique_values[1:]
        with np.errstate(over="ignore"):
            gaps = upper_values - lower_values
            thresholds = lower_values + gaps / 2.0
        overflowed = ~np.isfinite(thresholds)
        thresholds[overflowed] = (
            lower_values[overflowed] / 2.0 + upper_values[overflowed] / 2.0
        )
        for threshold in thresholds:
            left = x[:, feature_index] <= threshold
            if left.all() or (~left).all():
                continue
            weighted_impurity = (
                left.mean() * gini_impurity(y[left])
                + (~left).mean() * gini_impurity(y[~left])
            )
            candidate = (feature_index, float(threshold), float(weighted_impurity))
            if best is None or candidate[2] < best[2]:
                best = candidate
    return best

