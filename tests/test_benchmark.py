"""Tests for the benchmark dataset, metrics, and pipeline runners."""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.generate_dataset import generate_cases
from benchmark.metrics import dice_score, intersection_over_union
from benchmark.run_benchmark import PIPELINES, run_pipeline


def test_metric_known_values() -> None:
    first = np.zeros((4, 4), dtype=bool)
    second = np.zeros((4, 4), dtype=bool)
    first[0:2] = True  # 8 px
    second[1:3] = True  # 8 px, 4 px overlap

    assert intersection_over_union(first, second) == pytest.approx(4 / 12)
    assert dice_score(first, second) == pytest.approx(0.5)


def test_metrics_are_symmetric() -> None:
    rng = np.random.default_rng(0)
    first = rng.random((16, 16)) > 0.5
    second = rng.random((16, 16)) > 0.5

    assert intersection_over_union(first, second) == intersection_over_union(second, first)
    assert dice_score(first, second) == dice_score(second, first)


def test_metrics_handle_empty_masks() -> None:
    empty = np.zeros((8, 8), dtype=bool)
    full = np.ones((8, 8), dtype=bool)

    assert intersection_over_union(empty, empty) == 1.0
    assert dice_score(empty, empty) == 1.0
    assert intersection_over_union(empty, full) == 0.0
    assert dice_score(empty, full) == 0.0


def test_dataset_is_deterministic_and_well_formed() -> None:
    first = generate_cases()
    second = generate_cases()

    assert len(first) >= 5
    assert len({case.name for case in first}) == len(first)
    for case_a, case_b in zip(first, second, strict=True):
        assert np.array_equal(case_a.image, case_b.image)
        assert np.array_equal(case_a.ground_truth, case_b.ground_truth)
        assert case_a.image.dtype == np.uint8
        assert case_a.ground_truth.dtype == bool
        assert case_a.true_object_count > 0
        assert case_a.ground_truth.any()


def test_all_pipelines_perform_well_on_easy_case() -> None:
    easy = next(case for case in generate_cases() if case.name == "large_low_noise")

    for pipeline in PIPELINES:
        mask, runtime, predicted = run_pipeline(pipeline, easy)
        assert intersection_over_union(mask, easy.ground_truth) > 0.7, pipeline
        assert runtime >= 0
        assert predicted > 0

    # With cleanup, the count should match the ground truth exactly.
    _, _, cleaned_count = run_pipeline("otsu_cleaned", easy)
    assert cleaned_count == easy.true_object_count


def test_cleanup_reduces_count_error_on_debris_case() -> None:
    debris = next(case for case in generate_cases() if case.name == "debris_specks")

    _, _, raw_count = run_pipeline("otsu_only", debris)
    _, _, cleaned_count = run_pipeline("otsu_cleaned", debris)

    raw_error = abs(raw_count - debris.true_object_count)
    cleaned_error = abs(cleaned_count - debris.true_object_count)
    assert cleaned_error <= raw_error
    assert cleaned_count == debris.true_object_count


def test_unknown_pipeline_is_rejected() -> None:
    case = generate_cases()[0]
    with pytest.raises(ValueError, match="Unknown pipeline"):
        run_pipeline("magic", case)
