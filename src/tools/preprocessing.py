"""Preprocessing tools: grayscale conversion, median denoising, contrast enhancement.

All tools follow the same contract: they accept a uint8 image, never modify
the input in place, validate their parameters defensively, and return a new
uint8 image. Grayscale images are 2D arrays; RGB images are (H, W, 3).
"""

from __future__ import annotations

import numpy as np
from skimage import color, exposure, filters, morphology, util

from src.errors import ToolInputError

MIN_MEDIAN_RADIUS = 1
MAX_MEDIAN_RADIUS = 5
DEFAULT_MEDIAN_RADIUS = 2

MIN_CLIP_LIMIT = 0.001
MAX_CLIP_LIMIT = 0.1
DEFAULT_CLIP_LIMIT = 0.01


def _require_grayscale(image: np.ndarray, tool_name: str) -> None:
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ToolInputError(
            f"{tool_name} requires a 2D grayscale image; "
            f"got shape {getattr(image, 'shape', None)}. Run convert_to_grayscale first."
        )


def _require_integer(value: object, name: str, tool_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolInputError(f"{tool_name}: parameter {name!r} must be an integer, got {value!r}.")
    return value


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale; pass grayscale images through.

    Output is always a new 2D uint8 array in the range 0-255.
    """
    if not isinstance(image, np.ndarray):
        raise ToolInputError("convert_to_grayscale requires a numpy array image.")
    if image.ndim == 2:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 3:
        grayscale = color.rgb2gray(image)  # float64 in [0, 1]
        return util.img_as_ubyte(grayscale)
    raise ToolInputError(
        f"convert_to_grayscale expects a (H, W) or (H, W, 3) image; got shape {image.shape}."
    )


def denoise_median(image: np.ndarray, radius: int = DEFAULT_MEDIAN_RADIUS) -> np.ndarray:
    """Apply median filtering with a disk footprint to reduce impulse noise.

    Args:
        image: 2D uint8 grayscale image.
        radius: Disk radius in pixels, allowed range
            [``MIN_MEDIAN_RADIUS``, ``MAX_MEDIAN_RADIUS``].
    """
    _require_grayscale(image, "denoise_median")
    radius = _require_integer(radius, "radius", "denoise_median")
    if not MIN_MEDIAN_RADIUS <= radius <= MAX_MEDIAN_RADIUS:
        raise ToolInputError(
            f"denoise_median: radius must be between {MIN_MEDIAN_RADIUS} and "
            f"{MAX_MEDIAN_RADIUS}, got {radius}."
        )
    return filters.median(image, footprint=morphology.disk(radius))


def enhance_contrast(image: np.ndarray, clip_limit: float = DEFAULT_CLIP_LIMIT) -> np.ndarray:
    """Enhance contrast using CLAHE (adaptive histogram equalization).

    CLAHE was chosen over global contrast stretching because it also improves
    unevenly illuminated microscopy images. ``clip_limit`` bounds how strongly
    local contrast is amplified.

    Args:
        image: 2D uint8 grayscale image.
        clip_limit: CLAHE clipping limit, allowed range
            [``MIN_CLIP_LIMIT``, ``MAX_CLIP_LIMIT``].
    """
    _require_grayscale(image, "enhance_contrast")
    if isinstance(clip_limit, bool) or not isinstance(clip_limit, int | float):
        raise ToolInputError(f"enhance_contrast: clip_limit must be a number, got {clip_limit!r}.")
    if not MIN_CLIP_LIMIT <= clip_limit <= MAX_CLIP_LIMIT:
        raise ToolInputError(
            f"enhance_contrast: clip_limit must be between {MIN_CLIP_LIMIT} and "
            f"{MAX_CLIP_LIMIT}, got {clip_limit}."
        )
    enhanced = exposure.equalize_adapthist(image, clip_limit=float(clip_limit))  # float in [0, 1]
    return util.img_as_ubyte(enhanced)
