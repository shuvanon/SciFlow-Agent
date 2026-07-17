"""Fixed registry of approved tools — the only functions a plan can execute.

The registry maps each approved tool name to its callable, its Pydantic
parameter model, its data-flow types (used for workflow-order validation),
and a human-readable description (also used to build the LLM system prompt).

Nothing outside this registry is ever dispatched by the executor, and the
registry itself is never modified at runtime.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from src.agent import schemas
from src.errors import UnknownToolError
from src.tools import preprocessing, segmentation
from src.tools.measurement import measure_objects
from src.tools.preprocessing import convert_to_grayscale, denoise_median, enhance_contrast
from src.tools.segmentation import clean_mask, segment_otsu

INPUT_IMAGE = "image"  # accepts grayscale or RGB
INPUT_GRAYSCALE = "grayscale_image"
INPUT_MASK = "mask"

OUTPUT_GRAYSCALE = "grayscale_image"
OUTPUT_MASK = "mask"
OUTPUT_TABLE = "table"


@dataclass(frozen=True)
class ToolDefinition:
    """A single approved tool: callable, parameter schema, and data-flow types."""

    name: str
    function: Callable[..., Any]
    parameter_model: type[BaseModel]
    input_type: str
    output_type: str
    description: str


TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "convert_to_grayscale": ToolDefinition(
        name="convert_to_grayscale",
        function=convert_to_grayscale,
        parameter_model=schemas.NoParameters,
        input_type=INPUT_IMAGE,
        output_type=OUTPUT_GRAYSCALE,
        description="Convert an RGB image to grayscale; grayscale input passes through. "
        "No parameters.",
    ),
    "denoise_median": ToolDefinition(
        name="denoise_median",
        function=denoise_median,
        parameter_model=schemas.MedianDenoiseParameters,
        input_type=INPUT_GRAYSCALE,
        output_type=OUTPUT_GRAYSCALE,
        description="Reduce noise with a median filter. Parameter: radius (integer, "
        f"{preprocessing.MIN_MEDIAN_RADIUS}-{preprocessing.MAX_MEDIAN_RADIUS}, "
        f"default {preprocessing.DEFAULT_MEDIAN_RADIUS}).",
    ),
    "enhance_contrast": ToolDefinition(
        name="enhance_contrast",
        function=enhance_contrast,
        parameter_model=schemas.ContrastParameters,
        input_type=INPUT_GRAYSCALE,
        output_type=OUTPUT_GRAYSCALE,
        description="Improve contrast using CLAHE adaptive histogram equalization. "
        f"Parameter: clip_limit (number, {preprocessing.MIN_CLIP_LIMIT}-"
        f"{preprocessing.MAX_CLIP_LIMIT}, default {preprocessing.DEFAULT_CLIP_LIMIT}).",
    ),
    "segment_otsu": ToolDefinition(
        name="segment_otsu",
        function=segment_otsu,
        parameter_model=schemas.OtsuParameters,
        input_type=INPUT_GRAYSCALE,
        output_type=OUTPUT_MASK,
        description="Create a binary mask by Otsu thresholding. Parameter: polarity "
        "('bright' for bright objects on dark background, 'dark' for the opposite; "
        "default 'bright').",
    ),
    "clean_mask": ToolDefinition(
        name="clean_mask",
        function=clean_mask,
        parameter_model=schemas.CleanMaskParameters,
        input_type=INPUT_MASK,
        output_type=OUTPUT_MASK,
        description="Remove small objects from the mask and optionally fill small holes. "
        "Parameters: minimum_object_size (integer, "
        f"{segmentation.MIN_OBJECT_SIZE_LIMIT}-{segmentation.MAX_OBJECT_SIZE_LIMIT}, "
        f"default {segmentation.DEFAULT_MINIMUM_OBJECT_SIZE}), fill_holes (boolean, "
        "default false).",
    ),
    "measure_objects": ToolDefinition(
        name="measure_objects",
        function=measure_objects,
        parameter_model=schemas.MeasureObjectsParameters,
        input_type=INPUT_MASK,
        output_type=OUTPUT_TABLE,
        description="Label connected objects in the mask and measure area, perimeter, "
        "axes, centroid, bounding box, and mean intensity. No parameters.",
    ),
}


def get_tool(name: str) -> ToolDefinition:
    """Resolve an approved tool by name.

    Raises:
        UnknownToolError: If the name is not in the registry. Callers must
            treat this as a rejected plan, never fall back to any other
            lookup mechanism.
    """
    try:
        return TOOL_REGISTRY[name]
    except KeyError:
        allowed = ", ".join(sorted(TOOL_REGISTRY))
        raise UnknownToolError(
            f"Tool {name!r} is not an approved tool. Approved tools: {allowed}."
        ) from None
