"""Segmentation quality metrics for the benchmark (spec 14.3)."""

from __future__ import annotations

import numpy as np


def intersection_over_union(predicted: np.ndarray, truth: np.ndarray) -> float:
    """IoU (Jaccard index) of two binary masks.

    Both masks empty counts as a perfect match (1.0).
    """
    predicted_bool = predicted.astype(bool)
    truth_bool = truth.astype(bool)
    union = int(np.logical_or(predicted_bool, truth_bool).sum())
    if union == 0:
        return 1.0
    intersection = int(np.logical_and(predicted_bool, truth_bool).sum())
    return intersection / union


def dice_score(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Dice coefficient of two binary masks.

    Both masks empty counts as a perfect match (1.0).
    """
    predicted_bool = predicted.astype(bool)
    truth_bool = truth.astype(bool)
    total = int(predicted_bool.sum()) + int(truth_bool.sum())
    if total == 0:
        return 1.0
    intersection = int(np.logical_and(predicted_bool, truth_bool).sum())
    return 2.0 * intersection / total
