"""Tests for DICOM image loading (FR-01 medical format support)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.errors import ImageValidationError
from src.image_io import load_image_bytes

pydicom = pytest.importorskip("pydicom")
from pydicom.data import get_testdata_file  # noqa: E402


def _sample_dicom_bytes() -> bytes:
    """Bytes of a small DICOM image bundled with pydicom (no download)."""
    path = get_testdata_file("CT_small.dcm")
    assert path, "pydicom bundled test file CT_small.dcm not found"
    return Path(path).read_bytes()


def test_reads_bundled_dicom_to_uint8_grayscale() -> None:
    loaded = load_image_bytes(_sample_dicom_bytes(), "scan.dcm")

    assert loaded.original.dtype == np.uint8
    assert loaded.original.ndim == 2
    assert loaded.metadata.channels == 1
    assert loaded.metadata.mode.startswith("DICOM")
    assert loaded.metadata.width > 0 and loaded.metadata.height > 0
    assert loaded.metadata.sha256  # hash of the original file bytes


def test_dicom_alias_extension_accepted() -> None:
    loaded = load_image_bytes(_sample_dicom_bytes(), "scan.dicom")
    assert loaded.original.ndim == 2


def test_corrupt_dicom_is_rejected() -> None:
    with pytest.raises(ImageValidationError, match="Could not read DICOM"):
        load_image_bytes(b"this is not a DICOM file", "bad.dcm")


def test_oversized_dicom_is_rejected() -> None:
    with pytest.raises(ImageValidationError, match="exceeds the maximum"):
        load_image_bytes(_sample_dicom_bytes(), "scan.dcm", max_width=10, max_height=10)
