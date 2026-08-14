"""Tests for the controlled workflow executor (FR-09)."""

from __future__ import annotations

import numpy as np
import pytest

from src import executor as executor_module
from src.agent.schemas import ExecutionPlan, ToolStep
from src.executor import execute_plan
from src.plan_validator import validate_plan
from src.tool_registry import TOOL_REGISTRY


def _synthetic_image() -> np.ndarray:
    """Dark background with two bright squares and one tiny speck."""
    image = np.full((64, 64), 20, dtype=np.uint8)
    image[10:20, 10:20] = 220  # 100 px object
    image[40:52, 30:42] = 200  # 144 px object
    image[5, 55] = 210  # 1 px speck, removed by cleanup
    return image


def _reference_plan() -> ExecutionPlan:
    return ExecutionPlan(
        goal="segment_and_measure",
        supported=True,
        explanation="Denoise, segment, clean, and measure the image.",
        steps=[
            ToolStep(tool="convert_to_grayscale"),
            ToolStep(tool="denoise_median", parameters={"radius": 1}),
            ToolStep(tool="segment_otsu"),
            ToolStep(tool="clean_mask", parameters={"minimum_object_size": 20}),
            ToolStep(tool="measure_objects"),
        ],
    )


def _validated(plan: ExecutionPlan, channels: int = 1) -> ExecutionPlan:
    result = validate_plan(plan, channels=channels)
    assert result.valid, result.errors
    assert result.normalized_plan is not None
    return result.normalized_plan


def test_valid_plan_executes_successfully() -> None:
    image = _synthetic_image()

    result = execute_plan(_validated(_reference_plan()), image)

    assert result.success
    assert not result.errors
    assert result.summary is not None
    assert result.summary.object_count == 2  # speck removed by clean_mask
    assert result.measurements is not None
    assert len(result.measurements) == 2
    assert result.mask is not None
    assert result.labels is not None
    assert result.labels.max() == 2


def test_execution_preserves_step_order_and_runtimes() -> None:
    plan = _validated(_reference_plan())

    result = execute_plan(plan, _synthetic_image())

    executed_tools = [step.tool for step in result.steps]
    assert executed_tools == [step.tool for step in plan.steps]
    assert all(step.runtime_seconds >= 0 for step in result.steps)
    assert result.total_runtime_seconds >= sum(step.runtime_seconds for step in result.steps)


def test_executed_parameters_are_recorded_with_defaults() -> None:
    result = execute_plan(_validated(_reference_plan()), _synthetic_image())

    by_tool = {step.tool: step.parameters for step in result.steps}
    assert by_tool["denoise_median"] == {"radius": 1}
    assert by_tool["segment_otsu"] == {"polarity": "bright"}
    assert by_tool["clean_mask"] == {"minimum_object_size": 20, "fill_holes": False}


def test_intermediate_images_are_stored_in_order() -> None:
    result = execute_plan(_validated(_reference_plan()), _synthetic_image())

    keys = list(result.images)
    assert keys == [
        "01_convert_to_grayscale",
        "02_denoise_median",
        "03_segment_otsu",
        "04_clean_mask",
    ]
    assert result.images["03_segment_otsu"].dtype == bool


def test_input_image_is_not_modified() -> None:
    image = _synthetic_image()
    reference = image.copy()

    execute_plan(_validated(_reference_plan()), image)

    assert np.array_equal(image, reference)


def test_unknown_tool_cannot_execute() -> None:
    rogue = ExecutionPlan.model_construct(
        goal="g",
        supported=True,
        explanation="e",
        steps=[ToolStep.model_construct(tool="os.system", parameters={})],
        warnings=[],
    )

    result = execute_plan(rogue, _synthetic_image())

    assert not result.success
    assert any("not an approved tool" in error for error in result.errors)
    assert not result.steps


def test_invalid_parameters_cannot_execute() -> None:
    rogue = ExecutionPlan.model_construct(
        goal="g",
        supported=True,
        explanation="e",
        steps=[ToolStep.model_construct(tool="denoise_median", parameters={"radius": 500})],
        warnings=[],
    )

    result = execute_plan(rogue, _synthetic_image())

    assert not result.success
    assert any("parameters failed validation" in error for error in result.errors)


def test_executor_stops_on_failure_and_preserves_prior_steps(monkeypatch) -> None:
    definition = TOOL_REGISTRY["segment_otsu"]

    def _explode(*args, **kwargs):
        raise RuntimeError("simulated tool crash")

    monkeypatch.setattr(
        executor_module,
        "get_tool",
        lambda name: (
            definition.__class__(**{**definition.__dict__, "function": _explode})
            if name == "segment_otsu"
            else TOOL_REGISTRY[name]
        ),
    )

    result = execute_plan(_validated(_reference_plan()), _synthetic_image())

    assert not result.success
    assert [step.tool for step in result.steps] == ["convert_to_grayscale", "denoise_median"]
    assert any("segment_otsu" in error and "RuntimeError" in error for error in result.errors)
    assert result.summary is None


def test_empty_segmentation_yields_warning_not_error() -> None:
    constant = np.full((32, 32), 128, dtype=np.uint8)
    plan = _validated(
        ExecutionPlan(
            goal="segment_blank",
            supported=True,
            explanation="Segment a constant image.",
            steps=[
                ToolStep(tool="segment_otsu"),
                ToolStep(tool="measure_objects"),
            ],
        )
    )

    result = execute_plan(plan, constant)

    assert result.success
    assert result.summary is not None
    assert result.summary.object_count == 0
    assert any("empty mask" in warning for warning in result.warnings)
    assert any("No objects detected" in warning for warning in result.warnings)


def test_cleanup_removing_everything_warns() -> None:
    image = np.full((32, 32), 20, dtype=np.uint8)
    image[10:13, 10:13] = 200  # 9 px object only
    plan = _validated(
        ExecutionPlan(
            goal="clean_all",
            supported=True,
            explanation="Cleanup removes the only object.",
            steps=[
                ToolStep(tool="segment_otsu"),
                ToolStep(tool="clean_mask", parameters={"minimum_object_size": 50}),
            ],
        )
    )

    result = execute_plan(plan, image)

    assert result.success
    assert any("removed every object" in warning for warning in result.warnings)


def test_measure_uses_grayscale_intensity() -> None:
    image = _synthetic_image()

    result = execute_plan(_validated(_reference_plan()), image)

    assert result.measurements is not None
    assert "mean_intensity" in result.measurements.columns
    assert result.measurements["mean_intensity"].min() > 150  # bright objects


def test_intensity_uses_unenhanced_baseline_not_the_processed_image() -> None:
    """Enhancement must not move mean_intensity: it describes the sample.

    CLAHE is non-linear and spatially adaptive, so measuring on its output
    would make the reported intensity an artifact of the plan rather than a
    property of the image.
    """
    image = _synthetic_image()

    plain = execute_plan(_validated(_reference_plan()), image)
    with_clahe = execute_plan(
        _validated(
            ExecutionPlan(
                goal="enhance_and_measure",
                supported=True,
                explanation="Enhance contrast before segmenting.",
                steps=[
                    ToolStep(tool="convert_to_grayscale"),
                    ToolStep(tool="denoise_median", parameters={"radius": 1}),
                    ToolStep(tool="enhance_contrast", parameters={"clip_limit": 0.05}),
                    ToolStep(tool="segment_otsu"),
                    ToolStep(tool="clean_mask", parameters={"minimum_object_size": 20}),
                    ToolStep(tool="measure_objects"),
                ],
            )
        ),
        image,
    )

    assert plain.measurements is not None
    assert with_clahe.measurements is not None
    # Guard: the comparison below is only meaningful if both runs found the
    # same objects, so a mask difference must fail loudly rather than silently.
    assert with_clahe.summary.object_count == plain.summary.object_count
    assert with_clahe.measurements["area"].tolist() == plain.measurements["area"].tolist()

    assert with_clahe.measurements["mean_intensity"].tolist() == pytest.approx(
        plain.measurements["mean_intensity"].tolist()
    )


@pytest.mark.parametrize("channels,expected_first_tool", [(3, "convert_to_grayscale")])
def test_validator_and_executor_work_together_on_rgb(channels, expected_first_tool) -> None:
    rgb = np.zeros((48, 48, 3), dtype=np.uint8)
    rgb[10:30, 10:30] = (200, 210, 190)

    plan = _validated(_reference_plan(), channels=channels)
    result = execute_plan(plan, rgb)

    assert plan.steps[0].tool == expected_first_tool
    assert result.success
    assert result.summary is not None
    assert result.summary.object_count == 1
