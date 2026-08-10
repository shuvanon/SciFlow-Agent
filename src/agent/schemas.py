"""Pydantic schemas for execution plans and tool parameters.

These models are the first validation boundary for any planner output, no
matter whether it came from the LLM planner or the deterministic demo
planner. Every model forbids unknown fields, tool names are restricted to
the approved set, and numeric parameters carry hard bounds imported from
the tool modules so each limit is defined in exactly one place.

Parsing untrusted planner output through these models can fail with
``pydantic.ValidationError``; callers must treat that as "plan rejected",
never as something to work around.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.tools import ml_segmentation, preprocessing, segmentation

AllowedToolName = Literal[
    "convert_to_grayscale",
    "denoise_median",
    "enhance_contrast",
    "segment_otsu",
    "segment_ml",
    "clean_mask",
    "measure_objects",
]

#: Runtime tuple of the same names (kept in sync with AllowedToolName by a test).
ALLOWED_TOOL_NAMES: tuple[str, ...] = (
    "convert_to_grayscale",
    "denoise_median",
    "enhance_contrast",
    "segment_otsu",
    "segment_ml",
    "clean_mask",
    "measure_objects",
)

#: Structural upper bound on plan length; the configured limit
#: (MAX_WORKFLOW_STEPS, default 8) is enforced by the plan validator.
MAX_PLAN_STEPS_HARD_LIMIT = 20


class _StrictModel(BaseModel):
    """Base model that rejects any field not declared in the schema."""

    model_config = ConfigDict(extra="forbid")


class NoParameters(_StrictModel):
    """Parameter model for tools that accept no parameters."""


class MedianDenoiseParameters(_StrictModel):
    """Parameters for ``denoise_median``."""

    radius: int = Field(
        default=preprocessing.DEFAULT_MEDIAN_RADIUS,
        ge=preprocessing.MIN_MEDIAN_RADIUS,
        le=preprocessing.MAX_MEDIAN_RADIUS,
        description="Median filter disk radius in pixels.",
    )


class ContrastParameters(_StrictModel):
    """Parameters for ``enhance_contrast`` (CLAHE)."""

    clip_limit: float = Field(
        default=preprocessing.DEFAULT_CLIP_LIMIT,
        ge=preprocessing.MIN_CLIP_LIMIT,
        le=preprocessing.MAX_CLIP_LIMIT,
        description="CLAHE clipping limit controlling contrast amplification.",
    )


class OtsuParameters(_StrictModel):
    """Parameters for ``segment_otsu``."""

    polarity: Literal["bright", "dark"] = Field(
        default=segmentation.POLARITY_BRIGHT,
        description="Whether objects are brighter or darker than the background.",
    )

    @field_validator("polarity", mode="before")
    @classmethod
    def _normalize_polarity(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class CleanMaskParameters(_StrictModel):
    """Parameters for ``clean_mask``."""

    minimum_object_size: int = Field(
        default=segmentation.DEFAULT_MINIMUM_OBJECT_SIZE,
        ge=segmentation.MIN_OBJECT_SIZE_LIMIT,
        le=segmentation.MAX_OBJECT_SIZE_LIMIT,
        description="Objects with fewer pixels than this are removed.",
    )
    fill_holes: bool = Field(
        default=False,
        description="Fill holes smaller than minimum_object_size inside objects.",
    )


class MlSegmentParameters(_StrictModel):
    """Parameters for ``segment_ml`` (pretrained deep-learning segmentation)."""

    model_name: Literal["cxr_lung"] = Field(
        default=ml_segmentation.DEFAULT_MODEL,
        description="Approved pretrained model to run.",
    )
    threshold: float = Field(
        default=ml_segmentation.DEFAULT_THRESHOLD,
        ge=ml_segmentation.MIN_THRESHOLD,
        le=ml_segmentation.MAX_THRESHOLD,
        description="Foreground probability cutoff for the mask.",
    )


class MeasureObjectsParameters(_StrictModel):
    """Parameters for ``measure_objects`` (always the standard measurement set)."""


class ToolStep(_StrictModel):
    """One step of an execution plan: an approved tool plus raw parameters.

    ``parameters`` stays a plain dictionary at parse time; the plan validator
    checks it against the tool-specific parameter model.
    """

    tool: AllowedToolName
    parameters: dict[str, Any] = Field(default_factory=dict)


class ExecutionPlan(_StrictModel):
    """A structured, reviewable workflow produced by a planner (FR-05)."""

    goal: str = Field(min_length=1, max_length=200)
    supported: bool
    explanation: str = Field(min_length=1, max_length=2000)
    steps: list[ToolStep] = Field(default_factory=list, max_length=MAX_PLAN_STEPS_HARD_LIMIT)
    warnings: list[str] = Field(default_factory=list, max_length=10)
