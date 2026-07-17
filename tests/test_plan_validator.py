"""Tests for semantic plan validation (FR-06)."""

from __future__ import annotations

from src.agent.schemas import ExecutionPlan, ToolStep
from src.plan_validator import validate_plan


def _plan(steps: list[ToolStep], supported: bool = True) -> ExecutionPlan:
    return ExecutionPlan(
        goal="test_goal",
        supported=supported,
        explanation="test plan",
        steps=steps,
    )


def _reference_steps() -> list[ToolStep]:
    return [
        ToolStep(tool="convert_to_grayscale"),
        ToolStep(tool="denoise_median", parameters={"radius": 2}),
        ToolStep(tool="segment_otsu"),
        ToolStep(tool="clean_mask", parameters={"minimum_object_size": 40}),
        ToolStep(tool="measure_objects"),
    ]


def test_reference_plan_is_valid_and_normalized() -> None:
    result = validate_plan(_plan(_reference_steps()), channels=3)

    assert result.valid
    assert not result.errors
    normalized = result.normalized_plan
    assert normalized is not None
    # Defaults are filled in for review and reproducibility.
    otsu_step = normalized.steps[2]
    assert otsu_step.parameters == {"polarity": "bright"}
    clean_step = normalized.steps[3]
    assert clean_step.parameters == {"minimum_object_size": 40, "fill_holes": False}


def test_unsupported_plan_is_rejected() -> None:
    plan = _plan([], supported=False)

    result = validate_plan(plan)

    assert not result.valid
    assert "unsupported" in result.errors[0]


def test_empty_plan_is_rejected() -> None:
    result = validate_plan(_plan([]))

    assert not result.valid
    assert "no steps" in result.errors[0]


def test_step_limit_is_enforced() -> None:
    steps = [ToolStep(tool="convert_to_grayscale")] + [
        ToolStep(tool="enhance_contrast") for _ in range(9)
    ]

    result = validate_plan(_plan(steps), max_steps=8)

    assert not result.valid
    assert any("maximum of 8" in error for error in result.errors)


def test_unregistered_tool_is_rejected_even_if_schema_bypassed() -> None:
    # model_construct skips validation, simulating a corrupted or tampered plan.
    rogue_step = ToolStep.model_construct(tool="delete_all_files", parameters={})
    plan = ExecutionPlan.model_construct(
        goal="g",
        supported=True,
        explanation="e",
        steps=[rogue_step],
        warnings=[],
    )

    result = validate_plan(plan)

    assert not result.valid
    assert any("not an approved tool" in error for error in result.errors)


def test_grayscale_required_before_grayscale_tools_on_rgb() -> None:
    steps = [ToolStep(tool="denoise_median")]

    result = validate_plan(_plan(steps), channels=3)

    assert not result.valid
    assert any("requires a grayscale image" in error for error in result.errors)


def test_grayscale_input_skips_conversion_requirement() -> None:
    steps = [
        ToolStep(tool="denoise_median"),
        ToolStep(tool="segment_otsu"),
        ToolStep(tool="measure_objects"),
    ]

    result = validate_plan(_plan(steps), channels=1)

    assert result.valid


def test_unknown_channels_are_treated_as_rgb() -> None:
    result = validate_plan(_plan([ToolStep(tool="denoise_median")]), channels=None)

    assert not result.valid


def test_mask_required_before_mask_tools() -> None:
    steps = [
        ToolStep(tool="convert_to_grayscale"),
        ToolStep(tool="clean_mask"),
    ]

    result = validate_plan(_plan(steps))

    assert not result.valid
    assert any("requires a segmentation mask" in error for error in result.errors)


def test_measure_requires_segmentation_first() -> None:
    steps = [
        ToolStep(tool="convert_to_grayscale"),
        ToolStep(tool="measure_objects"),
    ]

    result = validate_plan(_plan(steps))

    assert not result.valid


def test_invalid_parameters_reported_with_step_number() -> None:
    steps = [
        ToolStep(tool="convert_to_grayscale"),
        ToolStep(tool="denoise_median", parameters={"radius": 99}),
    ]

    result = validate_plan(_plan(steps))

    assert not result.valid
    assert any(error.startswith("Step 2 (denoise_median)") for error in result.errors)


def test_extra_parameters_reported() -> None:
    steps = [
        ToolStep(tool="convert_to_grayscale", parameters={"path": "/etc/passwd"}),
    ]

    result = validate_plan(_plan(steps))

    assert not result.valid
    assert any("path" in error for error in result.errors)


def test_multiple_errors_are_collected() -> None:
    steps = [
        ToolStep(tool="denoise_median", parameters={"radius": 99}),
        ToolStep(tool="clean_mask"),
    ]

    result = validate_plan(_plan(steps), channels=3)

    # radius bound + grayscale missing + mask missing
    assert not result.valid
    assert len(result.errors) >= 3
