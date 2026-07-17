"""Tests for visualization helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.errors import ToolInputError
from src.tools.measurement import label_objects
from src.visualization import create_labelled_image, create_overlay, to_display_mask


def _mask() -> np.ndarray:
    mask = np.zeros((12, 12), dtype=bool)
    mask[3:7, 3:7] = True
    return mask


def test_display_mask_is_binary_uint8() -> None:
    display = to_display_mask(_mask())

    assert display.dtype == np.uint8
    assert set(np.unique(display)) == {0, 255}


def test_overlay_changes_only_masked_pixels() -> None:
    image = np.full((12, 12), 100, dtype=np.uint8)
    mask = _mask()

    overlay = create_overlay(image, mask, overlay_color=(255, 0, 0), alpha=0.5)

    assert overlay.shape == (12, 12, 3)
    assert overlay.dtype == np.uint8
    # Outside the mask: unchanged grayscale replicated to RGB.
    assert tuple(overlay[0, 0]) == (100, 100, 100)
    # Inside the mask: blended toward red.
    assert overlay[4, 4, 0] > 100
    assert overlay[4, 4, 1] < 100


def test_overlay_alpha_zero_keeps_image() -> None:
    image = np.full((12, 12), 60, dtype=np.uint8)

    overlay = create_overlay(image, _mask(), alpha=0.0)

    assert np.all(overlay == 60)


def test_overlay_rejects_shape_mismatch() -> None:
    with pytest.raises(ToolInputError, match="does not match"):
        create_overlay(np.zeros((10, 10), dtype=np.uint8), np.zeros((5, 5), dtype=bool))


def test_overlay_rejects_invalid_alpha() -> None:
    with pytest.raises(ToolInputError, match="alpha"):
        create_overlay(np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 10), dtype=bool), alpha=2)


def test_labelled_image_colors_objects() -> None:
    labels = label_objects(_mask())

    rendered = create_labelled_image(labels)

    assert rendered.shape == (12, 12, 3)
    assert rendered.dtype == np.uint8
    assert tuple(rendered[0, 0]) == (0, 0, 0)  # background stays black
    assert rendered[4, 4].max() > 0  # object got a color


def test_labelled_image_with_backdrop_matches_shape() -> None:
    labels = label_objects(_mask())
    backdrop = np.full((12, 12), 90, dtype=np.uint8)

    rendered = create_labelled_image(labels, image=backdrop)

    assert rendered.shape == (12, 12, 3)


def test_labelled_image_rejects_mismatched_backdrop() -> None:
    labels = label_objects(_mask())
    with pytest.raises(ToolInputError, match="does not match"):
        create_labelled_image(labels, image=np.zeros((5, 5), dtype=np.uint8))
