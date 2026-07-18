"""Tests for image loading, validation, and normalization."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src.errors import ImageValidationError
from src.image_io import load_image_bytes, load_image_file


def _encode(array: np.ndarray, image_format: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=image_format)
    return buffer.getvalue()


def test_metadata_includes_sha256_of_original_bytes() -> None:
    import hashlib

    array = np.zeros((10, 10), dtype=np.uint8)
    data = _encode(array)

    loaded = load_image_bytes(data, "hashme.png")

    assert loaded.metadata.sha256 == hashlib.sha256(data).hexdigest()


def test_loads_grayscale_png_with_metadata() -> None:
    array = np.zeros((10, 20), dtype=np.uint8)
    array[2:5, 3:9] = 200

    loaded = load_image_bytes(_encode(array), "sample.png")

    assert loaded.original.dtype == np.uint8
    assert loaded.original.shape == (10, 20)
    assert np.array_equal(loaded.original, array)
    assert loaded.metadata.filename == "sample.png"
    assert loaded.metadata.width == 20
    assert loaded.metadata.height == 10
    assert loaded.metadata.channels == 1
    assert loaded.metadata.mode == "L"
    assert loaded.metadata.dtype == "uint8"
    assert loaded.metadata.minimum_intensity == 0
    assert loaded.metadata.maximum_intensity == 200


def test_loads_rgb_png() -> None:
    array = np.zeros((8, 8, 3), dtype=np.uint8)
    array[..., 0] = 120

    loaded = load_image_bytes(_encode(array), "color.png")

    assert loaded.original.shape == (8, 8, 3)
    assert loaded.metadata.channels == 3
    assert loaded.metadata.mode == "RGB"


def test_rgba_is_converted_to_rgb() -> None:
    array = np.full((6, 6, 4), 255, dtype=np.uint8)

    loaded = load_image_bytes(_encode(array), "transparent.png")

    assert loaded.original.shape == (6, 6, 3)
    assert loaded.metadata.mode == "RGBA"  # original mode is preserved in metadata
    assert loaded.metadata.channels == 3


def test_sixteen_bit_tiff_is_rescaled_to_uint8() -> None:
    array = np.linspace(1000, 3000, num=64, dtype=np.uint16).reshape(8, 8)

    loaded = load_image_bytes(_encode(array, "TIFF"), "depth.tiff")

    assert loaded.original.dtype == np.uint8
    assert loaded.original.min() == 0
    assert loaded.original.max() == 255
    assert loaded.metadata.maximum_intensity == 3000
    assert loaded.metadata.minimum_intensity == 1000


def test_unsupported_extension_is_rejected() -> None:
    array = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(ImageValidationError, match="Unsupported file type"):
        load_image_bytes(_encode(array), "image.bmp")


def test_corrupt_bytes_are_rejected() -> None:
    with pytest.raises(ImageValidationError, match="corrupt or not a valid image"):
        load_image_bytes(b"definitely not image data", "broken.png")


def test_oversized_image_is_rejected() -> None:
    array = np.zeros((10, 30), dtype=np.uint8)
    with pytest.raises(ImageValidationError, match="exceeds the maximum"):
        load_image_bytes(_encode(array), "large.png", max_width=20, max_height=20)


def test_missing_file_is_rejected(tmp_path) -> None:
    with pytest.raises(ImageValidationError, match="not found"):
        load_image_file(tmp_path / "absent.png")


def test_load_image_file_roundtrip(tmp_path) -> None:
    array = np.full((12, 12), 99, dtype=np.uint8)
    path = tmp_path / "disk.png"
    Image.fromarray(array).save(path)

    loaded = load_image_file(path)

    assert loaded.metadata.filename == "disk.png"
    assert np.array_equal(loaded.original, array)
