"""Semantic validation of execution plans against the tool registry (FR-06).

Pydantic parsing (``src.agent.schemas``) guarantees structure; this module
checks everything structure alone cannot:

- every tool exists in the fixed registry,
- tool-specific parameters are valid (bounds, types, no extras),
- tools appear in a workable order (grayscale before grayscale-consuming
  tools, a segmentation mask before mask-consuming tools),
- the configured maximum number of steps is respected,
- plans the planner itself marked unsupported never execute.

Validation returns a normalized copy of the plan in which every parameter
has been parsed through its model and defaults are filled in, so the
executor runs exactly what was reviewed and reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from src.agent.schemas import ExecutionPlan, ToolStep
from src.tool_registry import (
    INPUT_GRAYSCALE,
    INPUT_MASK,
    OUTPUT_GRAYSCALE,
    OUTPUT_MASK,
    TOOL_REGISTRY,
)

DEFAULT_MAX_STEPS = 8


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of plan validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    normalized_plan: ExecutionPlan | None = None


def _format_validation_error(error: ValidationError) -> str:
    parts = []
    for issue in error.errors():
        location = ".".join(str(item) for item in issue["loc"]) or "value"
        parts.append(f"{location}: {issue['msg']}")
    return "; ".join(parts)


def validate_plan(
    plan: ExecutionPlan,
    *,
    channels: int | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> ValidationResult:
    """Validate a parsed plan for safe execution.

    Args:
        plan: A structurally valid :class:`ExecutionPlan`.
        channels: Channel count of the input image when known (1 means the
            image is already grayscale). When unknown, the image is treated
            as possibly-RGB, so grayscale-consuming tools require an explicit
            ``convert_to_grayscale`` step first.
        max_steps: Configured workflow-length limit (MAX_WORKFLOW_STEPS).

    Returns:
        A :class:`ValidationResult`; when valid, ``normalized_plan`` carries
        parameters parsed through the tool models with defaults filled in.
    """
    errors: list[str] = []

    if not plan.supported:
        return ValidationResult(
            valid=False,
            errors=[f"The planner marked this request as unsupported: {plan.explanation}"],
        )
    if not plan.steps:
        return ValidationResult(valid=False, errors=["The plan contains no steps."])
    if len(plan.steps) > max_steps:
        errors.append(
            f"The plan has {len(plan.steps)} steps, exceeding the maximum of {max_steps}."
        )

    grayscale_available = channels == 1
    mask_available = False
    normalized_steps: list[ToolStep] = []

    for index, step in enumerate(plan.steps, start=1):
        label = f"Step {index} ({step.tool})"

        definition = TOOL_REGISTRY.get(step.tool)
        if definition is None:
            errors.append(f"{label}: not an approved tool.")
            continue

        try:
            parameters = definition.parameter_model.model_validate(step.parameters)
        except ValidationError as exc:
            errors.append(f"{label}: invalid parameters — {_format_validation_error(exc)}")
            parameters = None

        if definition.input_type == INPUT_GRAYSCALE and not grayscale_available:
            errors.append(
                f"{label}: requires a grayscale image. Add convert_to_grayscale before it."
            )
        if definition.input_type == INPUT_MASK and not mask_available:
            errors.append(f"{label}: requires a segmentation mask. Add segment_otsu before it.")

        # Apply data-flow transitions even after an error so one root cause
        # does not cascade into misleading follow-up messages.
        if definition.output_type == OUTPUT_GRAYSCALE:
            grayscale_available = True
        if definition.output_type == OUTPUT_MASK:
            mask_available = True

        if parameters is not None:
            normalized_steps.append(ToolStep(tool=step.tool, parameters=parameters.model_dump()))

    if errors:
        return ValidationResult(valid=False, errors=errors)

    normalized_plan = plan.model_copy(update={"steps": normalized_steps})
    return ValidationResult(valid=True, normalized_plan=normalized_plan)
