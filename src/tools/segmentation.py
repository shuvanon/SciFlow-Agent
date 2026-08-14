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

#: Threshold methods approved for ``segment_threshold``.
#:
#: ``otsu``, ``li``, ``yen``, ``triangle`` and ``isodata`` each split the
#: histogram in two. ``multiotsu`` splits it into ``classes`` levels and keeps
#: only the extreme one, which is the difference that matters when an image
#: holds three or more distinct intensity populations — a CT slice, for
#: instance, contains air, soft tissue and bone, and a single cut cannot
#: separate bone from soft tissue no matter where it is placed.
THRESHOLD_METHODS = ("otsu", "multiotsu", "li", "yen", "triangle", "isodata")
DEFAULT_THRESHOLD_METHOD = "otsu"

MIN_THRESHOLD_CLASSES = 2
MAX_THRESHOLD_CLASSES = 5
DEFAULT_THRESHOLD_CLASSES = 3

_TWO_CLASS_FUNCTIONS = {
    "otsu": filters.threshold_otsu,
    "li": filters.threshold_li,
    "yen": filters.threshold_yen,
    "triangle": filters.threshold_triangle,
    "isodata": filters.threshold_isodata,
}


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


def segment_threshold(
    image: np.ndarray,
    method: str = DEFAULT_THRESHOLD_METHOD,
    classes: int = DEFAULT_THRESHOLD_CLASSES,
    polarity: str = POLARITY_BRIGHT,
) -> np.ndarray:
    """Create a binary mask using a chosen thresholding method.

    Generalizes :func:`segment_otsu`. The reason it exists is ``multiotsu``:
    a two-class threshold assumes the image holds a foreground and a
    background, and returns a poor mask whenever it holds more than that.
    On a CT slice — air, soft tissue, bone — Otsu separates air from
    everything else, so asking for "the bone" yields the whole body.
    ``multiotsu`` splits the histogram into ``classes`` levels and keeps only
    the brightest (or darkest) one.

    Args:
        image: 2D uint8 grayscale image.
        method: One of :data:`THRESHOLD_METHODS`.
        classes: Number of intensity classes for ``multiotsu``, in
            [``MIN_THRESHOLD_CLASSES``, ``MAX_THRESHOLD_CLASSES``]. Ignored by
            the two-class methods.
        polarity: ``"bright"`` keeps the brightest class, ``"dark"`` the
            darkest.

    Returns:
        Boolean mask. A single-intensity image has nothing to separate and
        yields an all-background mask.

    Raises:
        ToolInputError: If the method, class count, or polarity is invalid, or
            if the image has too few distinct intensities for ``classes``
            levels.
    """
    _require_2d(image, "segment_threshold", "grayscale image")
    if method not in THRESHOLD_METHODS:
        raise ToolInputError(
            f"segment_threshold: method must be one of {THRESHOLD_METHODS}, got {method!r}."
        )
    if polarity not in VALID_POLARITIES:
        raise ToolInputError(
            f"segment_threshold: polarity must be one of {VALID_POLARITIES}, got {polarity!r}."
        )
    if isinstance(classes, bool) or not isinstance(classes, int):
        raise ToolInputError(f"segment_threshold: classes must be an integer, got {classes!r}.")
    if not MIN_THRESHOLD_CLASSES <= classes <= MAX_THRESHOLD_CLASSES:
        raise ToolInputError(
            f"segment_threshold: classes must be between {MIN_THRESHOLD_CLASSES} and "
            f"{MAX_THRESHOLD_CLASSES}, got {classes}."
        )
    if image.min() == image.max():
        return np.zeros(image.shape, dtype=bool)

    if method == "multiotsu":
        try:
            thresholds = filters.threshold_multiotsu(image, classes=classes)
        except ValueError as exc:
            raise ToolInputError(
                f"segment_threshold: the image does not have enough distinct intensities for "
                f"{classes} classes ({exc}). Try fewer classes."
            ) from exc
        if polarity == POLARITY_BRIGHT:
            return image > thresholds[-1]
        return image <= thresholds[0]

    threshold = _TWO_CLASS_FUNCTIONS[method](image)
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
