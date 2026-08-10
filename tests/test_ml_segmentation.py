"""Tests for the deep-learning segmentation tool (segment_ml).

These run WITHOUT the optional ML backend (torch/torchxrayvision): the
graceful-degradation and postprocessing paths are exercised with monkeypatched
seams, and the rest is pure schema/registry/validator/planner wiring. Real GPU
inference is verified manually in the browser, not in CI.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from src.agent.demo_planner import generate_demo_plan
from src.agent.schemas import ExecutionPlan, MlSegmentParameters, ToolStep
from src.errors import ToolInputError
from src.plan_validator import validate_plan
from src.tool_registry import TOOL_REGISTRY
from src.tools import ml_segmentation as ml

# --- input / parameter validation (no backend needed) ---------------------


@pytest.mark.parametrize("bad", [None, np.zeros((5,), dtype=np.uint8), "image"])
def test_rejects_invalid_image(bad) -> None:
    with pytest.raises(ToolInputError):
        ml.segment_ml(bad)


def test_rejects_unknown_model() -> None:
    with pytest.raises(ToolInputError, match="unknown model"):
        ml.segment_ml(np.zeros((8, 8), dtype=np.uint8), model_name="bogus")


@pytest.mark.parametrize("threshold", [0.0, 1.5, -0.2, True, "high"])
def test_rejects_invalid_threshold(threshold) -> None:
    with pytest.raises(ToolInputError):
        ml.segment_ml(np.zeros((8, 8), dtype=np.uint8), threshold=threshold)


# --- graceful degradation when the ML backend is absent -------------------


def test_missing_backend_raises_actionable_error(monkeypatch) -> None:
    def _raise() -> tuple:
        raise ToolInputError("segment_ml needs the optional ML dependencies; pip install .[ml]")

    monkeypatch.setattr(ml, "_import_backend", _raise)
    with pytest.raises(ToolInputError, match="optional ML dependencies"):
        ml.segment_ml(np.zeros((16, 16), dtype=np.uint8))


# --- postprocessing (thresholding) with a stubbed predictor ---------------


def test_thresholding_produces_bool_mask_at_input_shape(monkeypatch) -> None:
    prob = np.zeros((20, 30), dtype=float)
    prob[5:15, 8:20] = 0.9

    monkeypatch.setattr(ml, "_import_backend", lambda: (None, None))
    monkeypatch.setattr(ml, "_predict_lung_probability", lambda image, model_name, torch, xrv: prob)

    mask = ml.segment_ml(np.zeros((20, 30), dtype=np.uint8), threshold=0.5)
    assert mask.dtype == bool
    assert mask.shape == (20, 30)
    assert mask[10, 12] and not mask[0, 0]

    # A cutoff above the peak probability removes everything.
    empty = ml.segment_ml(np.zeros((20, 30), dtype=np.uint8), threshold=0.95)
    assert not empty.any()


# --- schema -----------------------------------------------------------------


def test_ml_parameter_defaults_and_bounds() -> None:
    params = MlSegmentParameters()
    assert params.model_name == "cxr_lung"
    assert params.threshold == ml.DEFAULT_THRESHOLD

    with pytest.raises(ValidationError):
        MlSegmentParameters(threshold=0.0)
    with pytest.raises(ValidationError):
        MlSegmentParameters(threshold=1.5)
    with pytest.raises(ValidationError):
        MlSegmentParameters(model_name="other_model")
    with pytest.raises(ValidationError):
        MlSegmentParameters.model_validate({"threshold": 0.5, "device": "cuda"})


def test_schema_literal_matches_supported_models() -> None:
    import typing

    literal_values = set(typing.get_args(MlSegmentParameters.model_fields["model_name"].annotation))
    assert literal_values == set(ml.SUPPORTED_MODELS)


# --- registry + validator integration --------------------------------------


def test_registry_entry_shape() -> None:
    definition = TOOL_REGISTRY["segment_ml"]
    assert definition.input_type == "image"  # no forced grayscale step
    assert definition.output_type == "mask"
    assert definition.parameter_model is MlSegmentParameters


def _ml_plan(steps: list[ToolStep], supported: bool = True) -> ExecutionPlan:
    return ExecutionPlan(goal="g", supported=supported, explanation="e", steps=steps)


def test_validator_accepts_ml_segmentation_pipeline() -> None:
    plan = _ml_plan(
        [
            ToolStep(tool="segment_ml", parameters={"model_name": "cxr_lung"}),
            ToolStep(tool="clean_mask"),
            ToolStep(tool="measure_objects"),
        ]
    )
    result = validate_plan(plan, channels=1)
    assert result.valid, result.errors
    # segment_ml satisfies the mask requirement for downstream tools.
    assert result.normalized_plan.steps[0].tool == "segment_ml"


def test_ml_segmentation_needs_no_grayscale_on_rgb() -> None:
    # segment_ml accepts the image directly, so an RGB image needs no
    # convert_to_grayscale before it (unlike segment_otsu).
    plan = _ml_plan([ToolStep(tool="segment_ml"), ToolStep(tool="measure_objects")])
    assert validate_plan(plan, channels=3).valid


def test_out_of_range_ml_threshold_rejected_by_validator() -> None:
    rogue = ExecutionPlan.model_construct(
        goal="g",
        supported=True,
        explanation="e",
        steps=[ToolStep.model_construct(tool="segment_ml", parameters={"threshold": 5})],
        warnings=[],
    )
    result = validate_plan(rogue, channels=1)
    assert not result.valid
    assert any("threshold" in error for error in result.errors)


# --- demo planner routing ---------------------------------------------------


def test_demo_planner_routes_lung_request_to_ml() -> None:
    plan = generate_demo_plan("Segment the lungs in this chest x-ray and measure them.", channels=1)
    tools = [step.tool for step in plan.steps]
    assert "segment_ml" in tools
    assert "segment_otsu" not in tools


def test_demo_planner_deep_learning_keyword_routes_to_ml() -> None:
    plan = generate_demo_plan("Use deep learning to segment and count the objects.", channels=1)
    assert any(step.tool == "segment_ml" for step in plan.steps)


def test_demo_planner_normal_request_still_uses_otsu() -> None:
    plan = generate_demo_plan("Segment the bright objects and measure them.", channels=1)
    tools = [step.tool for step in plan.steps]
    assert "segment_otsu" in tools
    assert "segment_ml" not in tools


@pytest.mark.parametrize(
    "prompt",
    [
        "Train a neural network on my images.",
        "Use Cellpose to segment the cells.",
        "Use MONAI to segment the organs.",
    ],
)
def test_demo_planner_still_rejects_training_and_unsupported_frameworks(prompt: str) -> None:
    plan = generate_demo_plan(prompt, channels=1)
    assert not plan.supported
    assert not validate_plan(plan, channels=1).valid
