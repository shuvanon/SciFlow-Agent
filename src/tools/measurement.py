"""Measurement tools: connected-component labelling, region properties, summaries.

``measure_objects`` implements FR-11 and FR-12: it labels a binary mask,
measures every object, and aggregates summary statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage import measure

from src.errors import ToolInputError
from src.models import SummaryStatistics

#: 8-connectivity: diagonal neighbors belong to the same object.
LABEL_CONNECTIVITY = 2

_BASE_PROPERTIES = [
    "label",
    "area",
    "bbox",
    "centroid",
    "perimeter",
    "axis_major_length",
    "axis_minor_length",
]

_COLUMN_RENAMES = {
    "bbox-0": "bbox_min_row",
    "bbox-1": "bbox_min_col",
    "bbox-2": "bbox_max_row",
    "bbox-3": "bbox_max_col",
    "centroid-0": "centroid_row",
    "centroid-1": "centroid_col",
    "axis_major_length": "major_axis_length",
    "axis_minor_length": "minor_axis_length",
    "intensity_mean": "mean_intensity",
}

MEASUREMENT_COLUMNS = [
    "label",
    "area",
    "perimeter",
    "major_axis_length",
    "minor_axis_length",
    "centroid_row",
    "centroid_col",
    "bbox_min_row",
    "bbox_min_col",
    "bbox_max_row",
    "bbox_max_col",
]


def _require_2d(array: np.ndarray, tool_name: str, kind: str) -> None:
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ToolInputError(
            f"{tool_name} requires a 2D {kind}; got shape {getattr(array, 'shape', None)}."
        )


def label_objects(mask: np.ndarray) -> np.ndarray:
    """Label connected components of a binary mask (0 is background)."""
    _require_2d(mask, "label_objects", "binary mask")
    return measure.label(mask.astype(bool), connectivity=LABEL_CONNECTIVITY)


def _empty_measurements(with_intensity: bool) -> pd.DataFrame:
    columns = [*MEASUREMENT_COLUMNS, "mean_intensity"] if with_intensity else MEASUREMENT_COLUMNS
    return pd.DataFrame({column: [] for column in columns})


def _summarize(areas: np.ndarray, mask: np.ndarray) -> SummaryStatistics:
    total_pixels = int(mask.size)
    segmented_pixels = int(np.count_nonzero(mask))
    if areas.size == 0:
        return SummaryStatistics(
            object_count=0,
            mean_area=0.0,
            median_area=0.0,
            minimum_area=0.0,
            maximum_area=0.0,
            total_segmented_area=float(segmented_pixels),
            segmented_area_percent=(
                100.0 * segmented_pixels / total_pixels if total_pixels else 0.0
            ),
        )
    return SummaryStatistics(
        object_count=int(areas.size),
        mean_area=float(np.mean(areas)),
        median_area=float(np.median(areas)),
        minimum_area=float(np.min(areas)),
        maximum_area=float(np.max(areas)),
        total_segmented_area=float(segmented_pixels),
        segmented_area_percent=100.0 * segmented_pixels / total_pixels if total_pixels else 0.0,
    )


def measure_objects(
    mask: np.ndarray,
    intensity_image: np.ndarray | None = None,
) -> tuple[pd.DataFrame, SummaryStatistics]:
    """Measure every object in a binary mask.

    Args:
        mask: 2D binary mask.
        intensity_image: Optional 2D grayscale image of the same shape; when
            given, the per-object mean intensity is measured on it.

    Returns:
        A tuple of (per-object measurements table, summary statistics). The
        table contains one row per object with the columns in
        ``MEASUREMENT_COLUMNS`` (plus ``mean_intensity`` when an intensity
        image is provided). An empty mask yields an empty table and an
        all-zero summary rather than an error.
    """
    _require_2d(mask, "measure_objects", "binary mask")
    if intensity_image is not None:
        _require_2d(intensity_image, "measure_objects", "intensity image")
        if intensity_image.shape != mask.shape:
            raise ToolInputError(
                f"measure_objects: intensity image shape {intensity_image.shape} does not "
                f"match mask shape {mask.shape}."
            )

    boolean_mask = mask.astype(bool)
    labels = label_objects(boolean_mask)

    if labels.max() == 0:
        return _empty_measurements(intensity_image is not None), _summarize(
            np.array([]), boolean_mask
        )

    properties = list(_BASE_PROPERTIES)
    if intensity_image is not None:
        properties.append("intensity_mean")

    table = measure.regionprops_table(
        labels, intensity_image=intensity_image, properties=properties
    )
    measurements = pd.DataFrame(table).rename(columns=_COLUMN_RENAMES)

    ordered_columns = [*MEASUREMENT_COLUMNS]
    if intensity_image is not None:
        ordered_columns.append("mean_intensity")
    measurements = measurements[ordered_columns]

    areas = measurements["area"].to_numpy(dtype=float)
    return measurements, _summarize(areas, boolean_mask)
