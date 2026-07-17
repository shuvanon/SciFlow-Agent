"""Integration tests: the complete Phase 2 pipeline on the bundled examples.

These tests exercise the full workflow from image loading to measurements
without any planner, mirroring the reference use cases in the specification.
"""

from __future__ import annotations

import numpy as np

from src.config import EXAMPLES_DIR
from src.image_io import load_image_file
from src.tools.measurement import label_objects, measure_objects
from src.tools.preprocessing import convert_to_grayscale, denoise_median, enhance_contrast
from src.tools.segmentation import clean_mask, segment_otsu
from src.visualization import create_labelled_image, create_overlay


def test_full_pipeline_on_cells_example() -> None:
    loaded = load_image_file(EXAMPLES_DIR / "example_cells.png")

    grayscale = convert_to_grayscale(loaded.original)
    denoised = denoise_median(grayscale, radius=2)
    enhanced = enhance_contrast(denoised, clip_limit=0.01)
    raw_mask = segment_otsu(enhanced, polarity="bright")
    mask = clean_mask(raw_mask, minimum_object_size=40, fill_holes=True)

    _, raw_summary = measure_objects(raw_mask)
    measurements, summary = measure_objects(mask, intensity_image=denoised)

    # The image contains dozens of large cells plus small specks and noise.
    assert raw_summary.object_count > summary.object_count  # cleanup removed specks
    assert summary.object_count >= 10
    assert len(measurements) == summary.object_count
    assert 0 < summary.segmented_area_percent < 60
    assert summary.minimum_area >= 40  # cleanup enforced the size threshold
    assert measurements["mean_intensity"].min() > 60  # cells are bright

    overlay = create_overlay(loaded.original, mask)
    labelled = create_labelled_image(label_objects(mask), grayscale)
    assert overlay.shape == (*mask.shape, 3)
    assert labelled.shape == (*mask.shape, 3)


def test_rgb_objects_example_segments_after_grayscale() -> None:
    loaded = load_image_file(EXAMPLES_DIR / "example_objects.png")
    assert loaded.original.ndim == 3  # RGB input

    grayscale = convert_to_grayscale(loaded.original)
    mask = clean_mask(segment_otsu(grayscale, polarity="bright"), minimum_object_size=50)
    measurements, summary = measure_objects(mask, intensity_image=grayscale)

    assert grayscale.ndim == 2
    assert summary.object_count >= 5
    assert len(measurements) == summary.object_count
    assert summary.mean_area > 100  # geometric shapes are large


def test_pipeline_handles_blank_image_without_crashing() -> None:
    blank = np.full((64, 64), 128, dtype=np.uint8)

    mask = clean_mask(segment_otsu(blank), minimum_object_size=10)
    measurements, summary = measure_objects(mask)

    assert not mask.any()
    assert measurements.empty
    assert summary.object_count == 0
