"""Segmentation tools: Otsu thresholding and binary-mask cleanup.

Masks are 2D boolean arrays: ``True`` marks foreground (detected objects).
"""

from __future__ import annotations

import numpy as np
from skimage import filters, morphology

from src.errors import ToolInputError

POLARITY_BRIGHT = "bright"
POLARITY_DARK = "dark"
VALID_POLARITIES = (POLARITY_BRIGHT, POLARITY_DARK)

MIN_OBJECT_SIZE_LIMIT = 0
MAX_OBJECT_SIZE_LIMIT = 100_000
DEFAULT_MINIMUM_OBJECT_SIZE = 30


def _require_2d(array: np.ndarray, tool_name: str, kind: str) -> None:
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ToolInputError(
            f"{tool_name} requires a 2D {kind}; got shape {getattr(array, 'shape', None)}."
        )


def segment_otsu(image: np.ndarray, polarity: str = POLARITY_BRIGHT) -> np.ndarray:
    """Create a binary mask by Otsu thresholding.

    Args:
        image: 2D uint8 grayscale image.
        polarity: ``"bright"`` selects pixels above the threshold (bright
            objects on a dark background); ``"dark"`` selects the exact
            complement (pixels at or below the threshold).

    Returns:
        Boolean mask. If the image has a single constant intensity there is
        nothing to separate and an all-background mask is returned.
    """
    _require_2d(image, "segment_otsu", "grayscale image")
    if polarity not in VALID_POLARITIES:
        raise ToolInputError(
            f"segment_otsu: polarity must be one of {VALID_POLARITIES}, got {polarity!r}."
        )
    if image.min() == image.max():
        return np.zeros(image.shape, dtype=bool)
    threshold = filters.threshold_otsu(image)
    if polarity == POLARITY_BRIGHT:
        return image > threshold
    return image <= threshold


def clean_mask(
    mask: np.ndarray,
    minimum_object_size: int = DEFAULT_MINIMUM_OBJECT_SIZE,
    fill_holes: bool = False,
) -> np.ndarray:
    """Remove small objects from a binary mask and optionally fill small holes.

    Args:
        mask: 2D binary mask (boolean, or numeric where nonzero means
            foreground).
        minimum_object_size: Objects with fewer pixels are removed. Allowed
            range [``MIN_OBJECT_SIZE_LIMIT``, ``MAX_OBJECT_SIZE_LIMIT``].
            When ``fill_holes`` is enabled, holes smaller than this size are
            filled as well.
        fill_holes: Whether to fill small holes inside objects.
    """
    _require_2d(mask, "clean_mask", "binary mask")
    if isinstance(minimum_object_size, bool) or not isinstance(minimum_object_size, int):
        raise ToolInputError(
            f"clean_mask: minimum_object_size must be an integer, got {minimum_object_size!r}."
        )
    if not MIN_OBJECT_SIZE_LIMIT <= minimum_object_size <= MAX_OBJECT_SIZE_LIMIT:
        raise ToolInputError(
            f"clean_mask: minimum_object_size must be between {MIN_OBJECT_SIZE_LIMIT} and "
            f"{MAX_OBJECT_SIZE_LIMIT}, got {minimum_object_size}."
        )
    if not isinstance(fill_holes, bool):
        raise ToolInputError(f"clean_mask: fill_holes must be a boolean, got {fill_holes!r}.")

    cleaned = mask.astype(bool)
    # skimage's max_size removes/fills regions with area <= max_size, so
    # minimum_object_size - 1 keeps objects of exactly minimum_object_size.
    if minimum_object_size > 0:
        cleaned = morphology.remove_small_objects(cleaned, max_size=minimum_object_size - 1)
        if fill_holes:
            cleaned = morphology.remove_small_holes(cleaned, max_size=minimum_object_size - 1)
    elif fill_holes:
        cleaned = morphology.remove_small_holes(cleaned)
    return cleaned
