"""Unit tests for the approved image-processing tools."""

from __future__ import annotations

import numpy as np
import pytest

from src.errors import ToolInputError
from src.tools.measurement import (
    MEASUREMENT_COLUMNS,
    label_objects,
    measure_objects,
)
from src.tools.preprocessing import (
    convert_to_grayscale,
    denoise_median,
    enhance_contrast,
)
from src.tools.segmentation import clean_mask, segment_otsu

# --- convert_to_grayscale -------------------------------------------------


def test_grayscale_converts_rgb() -> None:
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb[:5] = (255, 255, 255)

    gray = convert_to_grayscale(rgb)

    assert gray.ndim == 2
    assert gray.dtype == np.uint8
    assert gray[0, 0] == 255
    assert gray[9, 9] == 0


def test_grayscale_passthrough_returns_copy() -> None:
    image = np.full((6, 6), 42, dtype=np.uint8)

    result = convert_to_grayscale(image)

    assert np.array_equal(result, image)
    assert result is not image
    result[0, 0] = 0
    assert image[0, 0] == 42  # input untouched


@pytest.mark.parametrize("bad", [None, "image", np.zeros((4, 4, 4), dtype=np.uint8)])
def test_grayscale_rejects_invalid_input(bad) -> None:
    with pytest.raises(ToolInputError):
        convert_to_grayscale(bad)


# --- denoise_median -------------------------------------------------------


def test_median_removes_impulse_noise() -> None:
    image = np.full((21, 21), 50, dtype=np.uint8)
    image[10, 10] = 255

    result = denoise_median(image, radius=1)

    assert result[10, 10] == 50
    assert result.dtype == np.uint8
    assert result.shape == image.shape


@pytest.mark.parametrize("radius", [0, 6, -3, 2.5, True, "two"])
def test_median_rejects_invalid_radius(radius) -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ToolInputError):
        denoise_median(image, radius=radius)


def test_median_requires_grayscale() -> None:
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ToolInputError, match="grayscale"):
        denoise_median(rgb, radius=2)


# --- enhance_contrast -----------------------------------------------------


def test_contrast_enhancement_increases_spread() -> None:
    gradient = np.tile(np.linspace(100, 130, 64, dtype=np.uint8), (64, 1))

    result = enhance_contrast(gradient, clip_limit=0.02)

    assert result.dtype == np.uint8
    assert result.shape == gradient.shape
    assert result.std() > gradient.std()


@pytest.mark.parametrize("clip_limit", [0, 0.5, -0.01, "strong", True])
def test_contrast_rejects_invalid_clip_limit(clip_limit) -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ToolInputError):
        enhance_contrast(image, clip_limit=clip_limit)


def test_contrast_requires_grayscale() -> None:
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ToolInputError, match="grayscale"):
        enhance_contrast(rgb)


# --- segment_otsu ---------------------------------------------------------


def test_otsu_segments_bright_square() -> None:
    image = np.zeros((50, 50), dtype=np.uint8)
    image[10:20, 15:30] = 200

    mask = segment_otsu(image, polarity="bright")

    assert mask.dtype == bool
    expected = np.zeros((50, 50), dtype=bool)
    expected[10:20, 15:30] = True
    assert np.array_equal(mask, expected)


def test_otsu_dark_polarity_selects_dark_objects() -> None:
    image = np.full((30, 30), 220, dtype=np.uint8)
    image[5:10, 5:10] = 20

    mask = segment_otsu(image, polarity="dark")

    assert mask[7, 7]
    assert not mask[20, 20]


def test_otsu_constant_image_returns_empty_mask() -> None:
    image = np.full((16, 16), 77, dtype=np.uint8)

    mask = segment_otsu(image)

    assert mask.dtype == bool
    assert not mask.any()


def test_otsu_rejects_invalid_polarity() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ToolInputError, match="polarity"):
        segment_otsu(image, polarity="both")


def test_otsu_requires_grayscale() -> None:
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ToolInputError):
        segment_otsu(rgb)


# --- clean_mask -----------------------------------------------------------


def _mask_with_blob_and_speck() -> np.ndarray:
    mask = np.zeros((30, 30), dtype=bool)
    mask[2:12, 2:12] = True  # 100 px blob
    mask[20:22, 20:22] = True  # 4 px speck
    return mask


def test_clean_mask_removes_small_objects() -> None:
    result = clean_mask(_mask_with_blob_and_speck(), minimum_object_size=30)

    assert result[5, 5]
    assert not result[20, 20]


def test_clean_mask_zero_size_keeps_everything() -> None:
    mask = _mask_with_blob_and_speck()

    result = clean_mask(mask, minimum_object_size=0)

    assert np.array_equal(result, mask)


def test_clean_mask_fills_holes_when_enabled() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[4:16, 4:16] = True
    mask[9:11, 9:11] = False  # 4 px hole

    without_fill = clean_mask(mask, minimum_object_size=30, fill_holes=False)
    with_fill = clean_mask(mask, minimum_object_size=30, fill_holes=True)

    assert not without_fill[9, 9]
    assert with_fill[9, 9]


def test_clean_mask_accepts_numeric_masks() -> None:
    numeric = _mask_with_blob_and_speck().astype(np.uint8)

    result = clean_mask(numeric, minimum_object_size=30)

    assert result.dtype == bool
    assert result[5, 5]


@pytest.mark.parametrize("size", [-1, 200_000, "big", 2.5, True])
def test_clean_mask_rejects_invalid_size(size) -> None:
    with pytest.raises(ToolInputError):
        clean_mask(np.zeros((5, 5), dtype=bool), minimum_object_size=size)


def test_clean_mask_rejects_non_boolean_fill_holes() -> None:
    with pytest.raises(ToolInputError, match="fill_holes"):
        clean_mask(np.zeros((5, 5), dtype=bool), minimum_object_size=10, fill_holes="yes")


# --- label_objects and measure_objects ------------------------------------


def _three_object_mask() -> np.ndarray:
    mask = np.zeros((32, 32), dtype=bool)
    mask[2:5, 3:7] = True  # area 12
    mask[10:12, 10:15] = True  # area 10
    mask[20, 20] = True  # area 1
    return mask


def test_labelling_uses_eight_connectivity() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True  # touches only diagonally

    labels = label_objects(mask)

    assert labels.max() == 1


def test_measurements_table_and_summary() -> None:
    measurements, summary = measure_objects(_three_object_mask())

    assert list(measurements.columns) == MEASUREMENT_COLUMNS
    assert len(measurements) == 3
    assert sorted(measurements["area"]) == [1, 10, 12]

    assert summary.object_count == 3
    assert summary.total_segmented_area == 23
    assert summary.minimum_area == 1
    assert summary.maximum_area == 12
    assert summary.median_area == 10
    assert summary.segmented_area_percent == pytest.approx(23 / (32 * 32) * 100)


def test_measurement_geometry_is_correct() -> None:
    measurements, _ = measure_objects(_three_object_mask())
    first = measurements.sort_values("label").iloc[0]

    assert first["area"] == 12
    assert first["centroid_row"] == pytest.approx(3.0)
    assert first["centroid_col"] == pytest.approx(4.5)
    assert (
        first["bbox_min_row"],
        first["bbox_min_col"],
        first["bbox_max_row"],
        first["bbox_max_col"],
    ) == (2, 3, 5, 7)


def test_mean_intensity_measured_when_intensity_given() -> None:
    mask = _three_object_mask()
    intensity = np.zeros((32, 32), dtype=np.uint8)
    intensity[2:5, 3:7] = 100
    intensity[10:12, 10:15] = 200
    intensity[20, 20] = 50

    measurements, _ = measure_objects(mask, intensity_image=intensity)

    assert "mean_intensity" in measurements.columns
    values = sorted(measurements["mean_intensity"])
    assert values == [50, 100, 200]


def test_empty_mask_yields_empty_table_and_zero_summary() -> None:
    measurements, summary = measure_objects(np.zeros((10, 10), dtype=bool))

    assert measurements.empty
    assert list(measurements.columns) == MEASUREMENT_COLUMNS
    assert summary.object_count == 0
    assert summary.total_segmented_area == 0
    assert summary.segmented_area_percent == 0


def test_measure_rejects_mismatched_intensity_shape() -> None:
    with pytest.raises(ToolInputError, match="does not"):
        measure_objects(
            np.zeros((10, 10), dtype=bool),
            intensity_image=np.zeros((5, 5), dtype=np.uint8),
        )


def test_measure_rejects_non_2d_mask() -> None:
    with pytest.raises(ToolInputError):
        measure_objects(np.zeros((5, 5, 3), dtype=bool))
