"""Shared data models for images and measurement results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImageMetadata:
    """Descriptive metadata of an accepted input image (spec section 10.1).

    ``mode``, ``dtype``, and the intensity range describe the image as it was
    loaded, before normalization to the internal uint8 representation.
    """

    filename: str
    width: int
    height: int
    channels: int
    mode: str
    dtype: str
    minimum_intensity: float
    maximum_intensity: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class LoadedImage:
    """An accepted image in the normalized internal representation.

    ``original`` is always uint8: shape (H, W) for grayscale or (H, W, 3)
    for RGB. The array is preserved unchanged for comparison and reporting;
    tools always work on copies.
    """

    original: np.ndarray
    metadata: ImageMetadata


@dataclass(frozen=True)
class SummaryStatistics:
    """Aggregate statistics over all detected objects (FR-12).

    Areas are in pixels. For an empty segmentation every field is zero
    (instead of NaN) so the values stay JSON-friendly.
    """

    object_count: int
    mean_area: float
    median_area: float
    minimum_area: float
    maximum_area: float
    total_segmented_area: float
    segmented_area_percent: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)
