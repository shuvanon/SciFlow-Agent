"""Tests for the built-in example images and their generator."""

from __future__ import annotations

import numpy as np
from PIL import Image

from examples.generate_examples import generate_cells_image, generate_objects_image
from src.config import EXAMPLES_DIR


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
