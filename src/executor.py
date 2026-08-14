"""Controlled workflow executor (FR-09).

The executor runs a validated plan step by step. Safety properties:

- Tools are resolved exclusively through the fixed registry; there is no
  other dispatch path. Nothing from the plan is evaluated, imported, or
  executed as code.
- Parameters are re-validated against the tool's Pydantic model immediately
  before the call, even though the plan validator already checked them.
- Execution stops at the first failing step; prior results are preserved.
- Every step records its runtime and warnings; failures become readable
  error messages without stack traces (full tracebacks go to the log only).

Intensity measurement uses the **photometric baseline**: the first 2D
grayscale image in the pipeline — the input itself, or the output of
``convert_to_grayscale`` — never the enhanced image. Enhancement steps change
pixel values (CLAHE is non-linear and spatially adaptive, so a post-CLAHE mean
carries no photometric meaning), and ``mean_intensity`` is meant to describe
the sample, not the preprocessing. Segmentation still runs on the fully
processed image; only the reported intensity is taken from the baseline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.agent.schemas import ExecutionPlan
from src.errors import ToolInputError, UnknownToolError
from src.models import SummaryStatistics
from src.tool_registry import INPUT_MASK, OUTPUT_MASK, OUTPUT_TABLE, get_tool
from src.tools.measurement import label_objects

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepResult:
    """Record of one executed step."""

    tool: str
    parameters: dict[str, Any]
    runtime_seconds: float
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Complete outcome of a workflow execution.

    ``images`` maps ``"NN_toolname"`` keys (in execution order) to the
    intermediate array produced by that step: uint8 images for image tools
    and boolean masks for segmentation tools.
    """

    success: bool
    steps: list[StepResult] = field(default_factory=list)
    images: dict[str, np.ndarray] = field(default_factory=dict)
    mask: np.ndarray | None = None
    labels: np.ndarray | None = None
    measurements: pd.DataFrame | None = None
    summary: SummaryStatistics | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_runtime_seconds: float = 0.0


def execute_plan(plan: ExecutionPlan, image: np.ndarray) -> ExecutionResult:
    """Execute a validated plan on an image.

    Args:
        plan: A plan that passed :func:`src.plan_validator.validate_plan`.
        image: The normalized input image (uint8). It is never modified.

    Returns:
        An :class:`ExecutionResult`; ``success`` is False if any step failed,
        with prior step results preserved.
    """
    total_start = time.perf_counter()
    result = ExecutionResult(success=True)

    current_image = image.copy()
    current_mask: np.ndarray | None = None
    # Photometric baseline for intensity measurement: the first 2D grayscale
    # image in the pipeline (the input itself, or the output of
    # convert_to_grayscale). Deliberately *not* updated by later enhancement
    # steps — see the note in the module docstring.
    intensity_reference: np.ndarray | None = current_image if current_image.ndim == 2 else None

    for index, step in enumerate(plan.steps, start=1):
        label = f"Step {index} ({step.tool})"

        try:
            definition = get_tool(step.tool)
        except UnknownToolError as exc:
            result.errors.append(str(exc))
            break

        try:
            parameters = definition.parameter_model.model_validate(step.parameters)
        except ValidationError:
            result.errors.append(f"{label}: parameters failed validation; execution stopped.")
            break
        kwargs = parameters.model_dump()

        step_warnings: list[str] = []
        step_start = time.perf_counter()
        try:
            if definition.output_type == OUTPUT_TABLE:
                if current_mask is None:
                    raise ToolInputError("no segmentation mask exists; run segment_otsu first.")
                result.measurements, result.summary = definition.function(
                    current_mask, intensity_image=intensity_reference, **kwargs
                )
                if result.summary.object_count == 0:
                    step_warnings.append("No objects detected in the segmentation mask.")
            elif definition.input_type == INPUT_MASK:
                if current_mask is None:
                    raise ToolInputError("no segmentation mask exists; run segment_otsu first.")
                new_mask = definition.function(current_mask, **kwargs)
                if current_mask.any() and not new_mask.any():
                    step_warnings.append("Mask cleanup removed every object.")
                current_mask = new_mask
                result.images[f"{index:02d}_{step.tool}"] = new_mask
            elif definition.output_type == OUTPUT_MASK:
                current_mask = definition.function(current_image, **kwargs)
                if not current_mask.any():
                    step_warnings.append("Segmentation produced an empty mask.")
                result.images[f"{index:02d}_{step.tool}"] = current_mask
            else:
                current_image = definition.function(current_image, **kwargs)
                result.images[f"{index:02d}_{step.tool}"] = current_image
                if intensity_reference is None and current_image.ndim == 2:
                    # First grayscale image only: convert_to_grayscale sets the
                    # baseline, denoise/enhance_contrast must not overwrite it.
                    intensity_reference = current_image
        except ToolInputError as exc:
            result.errors.append(f"{label}: {exc}")
            break
        except Exception as exc:  # unexpected tool failure: log details, keep UI clean
            logger.exception("%s raised an unexpected error", label)
            result.errors.append(f"{label} failed: {type(exc).__name__}: {exc}")
            break

        runtime = time.perf_counter() - step_start

        # Optional per-tool provenance (e.g. ML model name/version/weights hash)
        # for the reproducibility report. Never fatal to the run.
        step_metadata: dict[str, Any] = {}
        if definition.metadata_fn is not None:
            try:
                step_metadata = definition.metadata_fn(kwargs)
            except Exception:  # metadata is best-effort, never breaks execution
                logger.warning("Could not collect metadata for %s", step.tool)

        result.steps.append(
            StepResult(
                tool=step.tool,
                parameters=kwargs,
                runtime_seconds=runtime,
                warnings=step_warnings,
                metadata=step_metadata,
            )
        )
        result.warnings.extend(f"{label}: {message}" for message in step_warnings)

    result.success = not result.errors
    result.mask = current_mask
    if current_mask is not None and result.success:
        result.labels = label_objects(current_mask)

    result.total_runtime_seconds = time.perf_counter() - total_start
    logger.info(
        "Executed %d/%d steps in %.3fs (success=%s)",
        len(result.steps),
        len(plan.steps),
        result.total_runtime_seconds,
        result.success,
    )
    return result
