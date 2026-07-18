"""Image loading, validation, and normalization (FR-01, FR-02).

Accepted formats: PNG, JPG/JPEG, TIFF. Any accepted image is normalized to
the internal representation used by every tool: a uint8 array with shape
(H, W) for grayscale or (H, W, 3) for RGB.

Normalization rules:

- 8-bit grayscale (``L``) and ``RGB`` images pass through unchanged.
- Palette, RGBA, CMYK, and YCbCr images are converted to RGB (alpha is
  dropped, not composited).
- Bilevel (``1``) and grayscale-with-alpha (``LA``) images are converted
  to ``L``.
- Higher-depth images (16/32-bit integer or float, common in scientific
  TIFFs) are rescaled to uint8 using their own minimum-maximum intensity
  range, which preserves relative intensities within the image.
- Multi-page TIFFs: only the first page is used.

The reported metadata (mode, dtype, intensity range) always describes the
image *before* uint8 rescaling.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.errors import ImageValidationError
from src.models import ImageMetadata, LoadedImage

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

_CONVERT_TO_L_MODES = ("1", "LA")
_CONVERT_TO_RGB_MODES = ("P", "RGBA", "CMYK", "YCbCr")
_HIGH_DEPTH_MODES = ("I", "I;16", "I;16B", "I;16L", "I;16N", "F")


def _validate_extension(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise ImageValidationError(
            f"Unsupported file type {suffix or '(none)'!r} for {filename!r}. "
            f"Supported types: {supported}."
        )


def _normalize_mode(pil_image: Image.Image) -> Image.Image:
    mode = pil_image.mode
    if mode in _CONVERT_TO_L_MODES:
        return pil_image.convert("L")
    if mode in _CONVERT_TO_RGB_MODES:
        return pil_image.convert("RGB")
    if mode in ("L", "RGB") or mode in _HIGH_DEPTH_MODES:
        return pil_image
    return pil_image.convert("RGB")  # defensive fallback for exotic modes


def _rescale_to_uint8(array: np.ndarray) -> np.ndarray:
    """Rescale a non-uint8 array to uint8 using its own intensity range."""
    values = array.astype(np.float64)
    lowest = float(values.min())
    highest = float(values.max())
    if highest <= lowest:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (values - lowest) / (highest - lowest) * 255.0
    return np.round(scaled).astype(np.uint8)


def load_image_bytes(
    data: bytes,
    filename: str,
    *,
    max_width: int = 4096,
    max_height: int = 4096,
) -> LoadedImage:
    """Validate and normalize raw image bytes into a :class:`LoadedImage`.

    Raises:
        ImageValidationError: If the file type is unsupported, the data is
            not a readable image, or the dimensions exceed the configured
            limits.
    """
    _validate_extension(filename)

    try:
        with Image.open(io.BytesIO(data)) as pil_image:
            pil_image.load()
            original_mode = pil_image.mode
            converted = _normalize_mode(pil_image)
            array = np.asarray(converted)
    except UnidentifiedImageError as exc:
        raise ImageValidationError(
            f"Could not read {filename!r}: the file is corrupt or not a valid image."
        ) from exc
    except OSError as exc:
        raise ImageValidationError(f"Could not read {filename!r}: {exc}") from exc

    height, width = array.shape[0], array.shape[1]
    if width > max_width or height > max_height:
        raise ImageValidationError(
            f"Image is {width}×{height} px, which exceeds the maximum supported "
            f"size of {max_width}×{max_height} px."
        )

    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] != 3):
        raise ImageValidationError(
            f"Unsupported image layout with shape {array.shape}; expected a 2D "
            "grayscale or RGB image."
        )

    original_dtype = str(array.dtype)
    minimum_intensity = float(array.min())
    maximum_intensity = float(array.max())

    normalized = array.copy() if array.dtype == np.uint8 else _rescale_to_uint8(array)

    metadata = ImageMetadata(
        filename=filename,
        width=width,
        height=height,
        channels=1 if normalized.ndim == 2 else normalized.shape[2],
        mode=original_mode,
        dtype=original_dtype,
        minimum_intensity=minimum_intensity,
        maximum_intensity=maximum_intensity,
        sha256=hashlib.sha256(data).hexdigest(),
    )
    return LoadedImage(original=normalized, metadata=metadata)


def load_image_file(
    path: str | Path,
    *,
    max_width: int = 4096,
    max_height: int = 4096,
) -> LoadedImage:
    """Load an image from disk with the same validation as uploads."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ImageValidationError(f"Image file not found: {file_path}")
    return load_image_bytes(
        file_path.read_bytes(),
        file_path.name,
        max_width=max_width,
        max_height=max_height,
    )
