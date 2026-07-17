"""Tests for the plan and parameter schemas (first validation boundary)."""

from __future__ import annotations

import typing

import pytest
from pydantic import ValidationError

from src.agent.schemas import (
    ALLOWED_TOOL_NAMES,
    AllowedToolName,
    CleanMaskParameters,
    ContrastParameters,
    ExecutionPlan,
    MedianDenoiseParameters,
    NoParameters,
    OtsuParameters,
    ToolStep,
)
from src.tool_registry import TOOL_REGISTRY

SPEC_EXAMPLE_PLAN = {
    "goal": "segment_and_measure_objects",
    "supported": True,
    "explanation": "The image will be denoised, segmented, cleaned, and measured.",
    "steps": [
        {"tool": "convert_to_grayscale", "parameters": {}},
        {"tool": "denoise_median", "parameters": {"radius": 2}},
        {"tool": "segment_otsu", "parameters": {}},
        {"tool": "clean_mask", "parameters": {"minimum_object_size": 40}},
        {"tool": "measure_objects", "parameters": {}},
    ],
    "warnings": [],
}


def test_spec_example_plan_parses() -> None:
    plan = ExecutionPlan.model_validate(SPEC_EXAMPLE_PLAN)

    assert plan.supported
    assert [step.tool for step in plan.steps] == [
        "convert_to_grayscale",
        "denoise_median",
        "segment_otsu",
        "clean_mask",
        "measure_objects",
    ]


def test_unknown_tool_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ToolStep.model_validate({"tool": "run_shell_command", "parameters": {}})


def test_extra_plan_fields_are_rejected() -> None:
    payload = dict(SPEC_EXAMPLE_PLAN, injected_field="import os")
    with pytest.raises(ValidationError):
        ExecutionPlan.model_validate(payload)


def test_allowed_names_match_literal_and_registry() -> None:
    literal_names = typing.get_args(AllowedToolName)
    assert set(literal_names) == set(ALLOWED_TOOL_NAMES)
    assert set(TOOL_REGISTRY) == set(ALLOWED_TOOL_NAMES)


def test_registry_parameter_models_match_schemas() -> None:
    assert TOOL_REGISTRY["convert_to_grayscale"].parameter_model is NoParameters
    assert TOOL_REGISTRY["denoise_median"].parameter_model is MedianDenoiseParameters
    assert TOOL_REGISTRY["enhance_contrast"].parameter_model is ContrastParameters
    assert TOOL_REGISTRY["segment_otsu"].parameter_model is OtsuParameters
    assert TOOL_REGISTRY["clean_mask"].parameter_model is CleanMaskParameters


@pytest.mark.parametrize("radius", [1, 3, 5])
def test_denoise_radius_bounds_accept_valid(radius: int) -> None:
    assert MedianDenoiseParameters(radius=radius).radius == radius


@pytest.mark.parametrize("radius", [0, 6, -2, "huge"])
def test_denoise_radius_bounds_reject_invalid(radius) -> None:
    with pytest.raises(ValidationError):
        MedianDenoiseParameters(radius=radius)


def test_denoise_defaults_applied() -> None:
    assert MedianDenoiseParameters().radius == 2


def test_extra_parameters_are_rejected() -> None:
    with pytest.raises(ValidationError):
        MedianDenoiseParameters.model_validate({"radius": 2, "command": "rm -rf /"})
    with pytest.raises(ValidationError):
        NoParameters.model_validate({"anything": 1})


@pytest.mark.parametrize("clip_limit", [0.0, 0.2, -1])
def test_contrast_clip_limit_bounds(clip_limit) -> None:
    with pytest.raises(ValidationError):
        ContrastParameters(clip_limit=clip_limit)


def test_otsu_polarity_is_normalized() -> None:
    assert OtsuParameters.model_validate({"polarity": "  BRIGHT "}).polarity == "bright"
    assert OtsuParameters().polarity == "bright"


def test_otsu_polarity_rejects_unknown_values() -> None:
    with pytest.raises(ValidationError):
        OtsuParameters(polarity="sideways")


def test_clean_mask_parameter_bounds() -> None:
    parameters = CleanMaskParameters(minimum_object_size=40, fill_holes=True)
    assert parameters.minimum_object_size == 40
    assert parameters.fill_holes is True

    with pytest.raises(ValidationError):
        CleanMaskParameters(minimum_object_size=-5)
    with pytest.raises(ValidationError):
        CleanMaskParameters(minimum_object_size=1_000_000)


def test_plan_rejects_excessive_structural_length() -> None:
    steps = [{"tool": "denoise_median", "parameters": {}}] * 25
    payload = dict(SPEC_EXAMPLE_PLAN, steps=steps)
    with pytest.raises(ValidationError):
        ExecutionPlan.model_validate(payload)
