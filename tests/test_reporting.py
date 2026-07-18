"""Tests for the reproducibility report builders (first version)."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from src.agent.schemas import ExecutionPlan, ToolStep
from src.executor import execute_plan
from src.image_io import load_image_bytes
from src.models import ImageMetadata
from src.plan_validator import validate_plan
from src.reporting import build_report, report_to_json, report_to_markdown


def _loaded_image():
    array = np.full((48, 48), 15, dtype=np.uint8)
    array[10:22, 10:22] = 210
    array[30:38, 28:36] = 190
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return load_image_bytes(buffer.getvalue(), "report_sample.png")


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        goal="segment_and_measure",
        supported=True,
        explanation="Segment and measure the test image.",
        steps=[
            ToolStep(tool="segment_otsu"),
            ToolStep(tool="clean_mask", parameters={"minimum_object_size": 10}),
            ToolStep(tool="measure_objects"),
        ],
    )


@pytest.fixture()
def report() -> dict:
    loaded = _loaded_image()
    validation = validate_plan(_plan(), channels=loaded.metadata.channels)
    assert validation.valid
    execution = execute_plan(validation.normalized_plan, loaded.original)
    return build_report(
        metadata=loaded.metadata,
        request="Segment and measure the objects.",
        planner_mode="demo",
        plan=validation.normalized_plan,
        execution=execution,
    )


def test_report_contains_required_metadata(report: dict) -> None:
    assert report["report_version"] == 1
    assert report["software_version"]
    assert report["generated_at"]
    assert report["input_image"]["filename"] == "report_sample.png"
    assert report["user_request"] == "Segment and measure the objects."
    assert report["planner_mode"] == "demo"
    assert report["plan"]["goal"] == "segment_and_measure"
    assert report["execution"]["success"] is True
    assert len(report["execution"]["steps"]) == 3
    assert all("runtime_seconds" in step for step in report["execution"]["steps"])
    assert report["summary"]["object_count"] == 2
    assert len(report["measurements"]) == 2


def test_report_json_round_trips(report: dict) -> None:
    serialized = report_to_json(report)
    restored = json.loads(serialized)
    assert restored["summary"]["object_count"] == 2


def test_report_markdown_contains_sections(report: dict) -> None:
    markdown = report_to_markdown(report)
    assert "# SciFlow Agent — Analysis Report" in markdown
    assert "## Input image" in markdown
    assert "## Request" in markdown
    assert "## Executed plan" in markdown
    assert "## Summary statistics" in markdown
    assert "## Per-object measurements" in markdown
    assert "segment_otsu" in markdown


def test_report_never_contains_secrets(monkeypatch, report: dict) -> None:
    monkeypatch.setenv("LLM_API_KEY", "top-secret-value")
    serialized = report_to_json(report) + report_to_markdown(report)
    assert "top-secret-value" not in serialized


def test_failed_execution_still_produces_report() -> None:
    loaded = _loaded_image()
    rogue = ExecutionPlan.model_construct(
        goal="g",
        supported=True,
        explanation="e",
        steps=[ToolStep.model_construct(tool="not_a_tool", parameters={})],
        warnings=[],
    )
    execution = execute_plan(rogue, loaded.original)
    assert not execution.success

    report = build_report(
        metadata=loaded.metadata,
        request="anything",
        planner_mode="demo",
        plan=rogue,
        execution=execution,
    )
    assert report["execution"]["success"] is False
    assert report["execution"]["errors"]
    assert report["summary"] is None
    assert report["measurements"] == []
    json.loads(report_to_json(report))  # still serializable


def test_empty_segmentation_report_is_valid() -> None:
    metadata = ImageMetadata(
        filename="blank.png",
        width=32,
        height=32,
        channels=1,
        mode="L",
        dtype="uint8",
        minimum_intensity=128,
        maximum_intensity=128,
    )
    blank = np.full((32, 32), 128, dtype=np.uint8)
    plan = ExecutionPlan(
        goal="segment",
        supported=True,
        explanation="Segment a constant image.",
        steps=[ToolStep(tool="segment_otsu"), ToolStep(tool="measure_objects")],
    )
    validation = validate_plan(plan, channels=1)
    execution = execute_plan(validation.normalized_plan, blank)

    report = build_report(
        metadata=metadata,
        request="segment",
        planner_mode="demo",
        plan=validation.normalized_plan,
        execution=execution,
    )
    assert report["summary"]["object_count"] == 0
    assert report["measurements"] == []
    assert report["execution"]["warnings"]  # empty-mask warning present
    markdown = report_to_markdown(report)
    assert "Warnings" in markdown
