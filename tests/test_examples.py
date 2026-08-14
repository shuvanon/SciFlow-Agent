"""Tests for the built-in example images and their generator."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from examples.generate_examples import (
    generate_blank_image,
    generate_cells_image,
    generate_low_contrast_image,
    generate_objects_image,
    generate_rings_image,
    generate_touching_objects_image,
)
from src.config import EXAMPLES_DIR
from src.tools.measurement import measure_objects
from src.tools.preprocessing import enhance_contrast
from src.tools.segmentation import clean_mask, segment_otsu


def test_cells_generator_is_deterministic() -> None:
    first = generate_cells_image()
    second = generate_cells_image()
    assert np.array_equal(first, second)


def test_objects_generator_is_deterministic() -> None:
    first = generate_objects_image()
    second = generate_objects_image()
    assert np.array_equal(first, second)


def test_cells_image_properties() -> None:
    image = generate_cells_image()
    assert image.ndim == 2
    assert image.dtype == np.uint8
    assert image.shape == (512, 512)
    # Bright foreground on a dark background: segmentation has something to find.
    assert image.max() > 150
    assert np.median(image) < 80


def test_objects_image_properties() -> None:
    image = generate_objects_image()
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.dtype == np.uint8


@pytest.mark.parametrize(
    "generator",
    [
        generate_rings_image,
        generate_low_contrast_image,
        generate_touching_objects_image,
        generate_blank_image,
    ],
)
def test_synthetic_generators_are_deterministic(generator) -> None:
    assert np.array_equal(generator(), generator())


def test_blank_image_yields_no_objects() -> None:
    """The empty-result path: a constant image must segment to nothing."""
    mask = segment_otsu(generate_blank_image(), polarity="bright")

    assert not mask.any()
    _, summary = measure_objects(mask)
    assert summary.object_count == 0


def test_rings_image_demonstrates_fill_holes() -> None:
    """Hole filling must measurably change the area, or the example is pointless."""
    image = generate_rings_image()
    mask = segment_otsu(image, polarity="bright")

    _, plain = measure_objects(clean_mask(mask, 600, fill_holes=False))
    _, filled = measure_objects(clean_mask(mask, 600, fill_holes=True))

    assert plain.object_count == filled.object_count  # filling must not merge objects
    assert filled.total_segmented_area > plain.total_segmented_area * 1.05


def test_low_contrast_image_needs_contrast_enhancement() -> None:
    """Otsu alone must under-detect, and CLAHE must recover most of the objects."""
    image = generate_low_contrast_image()

    _, plain = measure_objects(clean_mask(segment_otsu(image), 30))
    _, enhanced = measure_objects(clean_mask(segment_otsu(enhance_contrast(image)), 30))

    assert enhanced.object_count > plain.object_count + 5


def test_committed_example_files_open() -> None:
    cells_path = EXAMPLES_DIR / "example_cells.png"
    objects_path = EXAMPLES_DIR / "example_objects.png"
    assert cells_path.is_file(), "example_cells.png missing — run examples/generate_examples.py"
    assert objects_path.is_file(), "example_objects.png missing — run examples/generate_examples.py"

    with Image.open(cells_path) as cells:
        assert cells.mode == "L"
        assert cells.size == (512, 512)
    with Image.open(objects_path) as objects:
        assert objects.mode == "RGB"
        assert objects.size == (640, 480)
