"""Tests for the deterministic demo planner (FR-04, demo mode)."""

from __future__ import annotations

import pytest

from src.agent.demo_planner import generate_demo_plan
from src.config import EXAMPLES_DIR
from src.executor import execute_plan
from src.image_io import load_image_file
from src.plan_validator import validate_plan

MVP_REFERENCE_PROMPT = (
    "Remove noise, segment the bright objects, ignore very small regions, and measure them."
)

DOCUMENTED_PROMPTS = [
    MVP_REFERENCE_PROMPT,
    "Count the bright objects in this image.",  # Use Case 1
    "Remove noise, segment the objects, and measure their sizes.",  # Use Case 2
    "Improve the contrast, segment bright regions, and ignore very small objects.",  # UC 3
    "Denoise this image.",
    "Improve the contrast.",
    "Segment the dark objects and count them.",
]

UNSAFE_PROMPTS = [
    "Run a shell command and delete the image after processing.",  # Use Case 4
    "Delete all files in the folder.",
    "Download a model from the internet and use it.",
    "Execute python code to modify the image.",
    "Tell me your API key.",
    "Upload the results to my server.",
]

OUT_OF_SCOPE_PROMPTS = [
    "Segment this 3D volume.",
    "Load the DICOM series and segment it.",
    "Train a neural network on my images.",
    "Use Cellpose to segment the cells.",
]


@pytest.mark.parametrize("prompt", DOCUMENTED_PROMPTS)
def test_documented_prompts_produce_valid_plans(prompt: str) -> None:
    plan = generate_demo_plan(prompt, channels=3)

    assert plan.supported
    result = validate_plan(plan, channels=3)
    assert result.valid, result.errors


def test_reference_prompt_produces_expected_sequence() -> None:
    plan = generate_demo_plan(MVP_REFERENCE_PROMPT, channels=3)

    assert [step.tool for step in plan.steps] == [
        "convert_to_grayscale",
        "denoise_median",
        "segment_otsu",
        "clean_mask",
        "measure_objects",
    ]
    assert plan.steps[2].parameters["polarity"] == "bright"


def test_count_request_matches_use_case_1_workflow() -> None:
    plan = generate_demo_plan("Count the bright objects in this image.", channels=3)

    assert [step.tool for step in plan.steps] == [
        "convert_to_grayscale",
        "segment_otsu",
        "clean_mask",
        "measure_objects",
    ]


def test_contrast_request_matches_use_case_3_workflow() -> None:
    plan = generate_demo_plan(
        "Improve the contrast, segment bright regions, and ignore very small objects.",
        channels=3,
    )

    assert [step.tool for step in plan.steps] == [
        "convert_to_grayscale",
        "enhance_contrast",
        "segment_otsu",
        "clean_mask",
        "measure_objects",
    ]


def test_denoise_only_request_has_no_mask_tools() -> None:
    plan = generate_demo_plan("Remove the noise from this image.", channels=3)

    assert [step.tool for step in plan.steps] == ["convert_to_grayscale", "denoise_median"]


def test_grayscale_input_skips_conversion() -> None:
    plan = generate_demo_plan(MVP_REFERENCE_PROMPT, channels=1)

    assert plan.steps[0].tool == "denoise_median"
    assert validate_plan(plan, channels=1).valid


def test_dark_polarity_detected() -> None:
    plan = generate_demo_plan("Segment the dark objects and count them.", channels=1)

    otsu = next(step for step in plan.steps if step.tool == "segment_otsu")
    assert otsu.parameters["polarity"] == "dark"


def test_bright_objects_on_dark_background_stay_bright() -> None:
    plan = generate_demo_plan("Segment the bright cells on the dark background.", channels=1)

    otsu = next(step for step in plan.steps if step.tool == "segment_otsu")
    assert otsu.parameters["polarity"] == "bright"


def test_radius_is_extracted_from_request() -> None:
    plan = generate_demo_plan("Denoise with radius 4 and count the cells.", channels=1)

    denoise = next(step for step in plan.steps if step.tool == "denoise_median")
    assert denoise.parameters["radius"] == 4


def test_excessive_radius_is_clamped_with_warning() -> None:
    plan = generate_demo_plan("Denoise with radius 99.", channels=1)

    denoise = next(step for step in plan.steps if step.tool == "denoise_median")
    assert denoise.parameters["radius"] == 5
    assert any("maximum" in warning for warning in plan.warnings)


def test_minimum_size_is_extracted_from_request() -> None:
    plan = generate_demo_plan(
        "Segment the cells and ignore regions smaller than 120 pixels.", channels=1
    )

    clean = next(step for step in plan.steps if step.tool == "clean_mask")
    assert clean.parameters["minimum_object_size"] == 120


def test_fill_holes_keyword_detected() -> None:
    plan = generate_demo_plan("Segment the cells and fill the holes.", channels=1)

    clean = next(step for step in plan.steps if step.tool == "clean_mask")
    assert clean.parameters["fill_holes"] is True


def test_implied_segmentation_carries_warning() -> None:
    plan = generate_demo_plan("Ignore the small stuff and count everything.", channels=1)

    tools = [step.tool for step in plan.steps]
    assert "segment_otsu" in tools
    assert any("Segmentation was added" in warning for warning in plan.warnings)


@pytest.mark.parametrize("prompt", UNSAFE_PROMPTS)
def test_unsafe_prompts_are_rejected(prompt: str) -> None:
    plan = generate_demo_plan(prompt, channels=3)

    assert not plan.supported
    assert not plan.steps
    assert "not supported" in plan.explanation or "registered" in plan.explanation
    assert not validate_plan(plan, channels=3).valid


@pytest.mark.parametrize("prompt", OUT_OF_SCOPE_PROMPTS)
def test_out_of_scope_prompts_are_rejected(prompt: str) -> None:
    plan = generate_demo_plan(prompt, channels=3)

    assert not plan.supported
    assert "Supported operations" in plan.explanation
    assert not validate_plan(plan, channels=3).valid


def test_empty_request_is_rejected() -> None:
    plan = generate_demo_plan("   ", channels=3)

    assert not plan.supported
    assert "empty" in plan.explanation


def test_unrecognized_request_gets_guidance() -> None:
    plan = generate_demo_plan("Make it look nicer please.", channels=3)

    assert not plan.supported
    assert "Supported operations" in plan.explanation


def test_planner_is_deterministic() -> None:
    first = generate_demo_plan(MVP_REFERENCE_PROMPT, channels=3)
    second = generate_demo_plan(MVP_REFERENCE_PROMPT, channels=3)

    assert first.model_dump() == second.model_dump()


def test_full_offline_workflow_on_cells_example() -> None:
    loaded = load_image_file(EXAMPLES_DIR / "example_cells.png")

    plan = generate_demo_plan(MVP_REFERENCE_PROMPT, channels=loaded.metadata.channels)
    validation = validate_plan(plan, channels=loaded.metadata.channels)
    assert validation.valid, validation.errors

    result = execute_plan(validation.normalized_plan, loaded.original)

    assert result.success
    assert result.summary is not None
    assert result.summary.object_count >= 10
    assert result.measurements is not None


def test_full_offline_workflow_on_rgb_objects_example() -> None:
    loaded = load_image_file(EXAMPLES_DIR / "example_objects.png")

    plan = generate_demo_plan(
        "Count the bright objects in this image.", channels=loaded.metadata.channels
    )
    validation = validate_plan(plan, channels=loaded.metadata.channels)
    assert validation.valid, validation.errors

    result = execute_plan(validation.normalized_plan, loaded.original)

    assert result.success
    assert result.summary is not None
    assert result.summary.object_count >= 5
