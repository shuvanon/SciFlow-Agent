"""Tests for the reproducibility report builders (FR-13)."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime

import numpy as np
import pytest
from PIL import Image

from src.agent.schemas import ExecutionPlan, ToolStep
from src.executor import ExecutionResult, StepResult, execute_plan
from src.image_io import load_image_bytes
from src.models import ImageMetadata
from src.plan_validator import validate_plan
from src.reporting import (
    MAX_MARKDOWN_MEASUREMENT_ROWS,
    REQUIRED_REPORT_KEYS,
    build_report,
    report_to_json,
    report_to_markdown,
)


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


def test_report_has_exactly_the_schema_keys(report: dict) -> None:
    assert set(report.keys()) == REQUIRED_REPORT_KEYS


def test_report_timestamp_is_valid_iso8601(report: dict) -> None:
    parsed = datetime.fromisoformat(report["generated_at"])
    assert parsed.tzinfo is not None  # explicit UTC offset


def test_report_records_input_file_hash(report: dict) -> None:
    loaded = _loaded_image()
    array = np.full((48, 48), 15, dtype=np.uint8)
    array[10:22, 10:22] = 210
    array[30:38, 28:36] = 190
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    expected = hashlib.sha256(buffer.getvalue()).hexdigest()

    assert loaded.metadata.sha256 == expected
    assert report["input_image"]["sha256"] == expected


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


def test_report_logs_model_metadata_in_json_and_markdown() -> None:
    metadata = ImageMetadata(
        filename="cxr.png",
        width=8,
        height=8,
        channels=1,
        mode="L",
        dtype="uint8",
        minimum_intensity=0,
        maximum_intensity=255,
        sha256="abc123",
    )
    plan = ExecutionPlan(
        goal="segment_lungs",
        supported=True,
        explanation="Segment the lungs.",
        steps=[
            ToolStep(tool="segment_ml", parameters={"model_name": "cxr_lung", "threshold": 0.5})
        ],
    )
    model_meta = {
        "model_name": "cxr_lung",
        "display_name": "Chest X-ray lung segmentation",
        "framework": "torchxrayvision",
        "framework_version": "1.5.2",
        "torch_version": "2.5.1",
        "device": "cuda",
        "weights_sha256": "deadbeefcafe",
    }
    execution = ExecutionResult(
        success=True,
        steps=[
            StepResult(
                tool="segment_ml",
                parameters={"model_name": "cxr_lung", "threshold": 0.5},
                runtime_seconds=0.5,
                metadata=model_meta,
            )
        ],
        total_runtime_seconds=0.5,
    )

    report = build_report(
        metadata=metadata,
        request="Segment the lungs.",
        planner_mode="demo",
        plan=plan,
        execution=execution,
    )

    step = report["execution"]["steps"][0]
    assert step["metadata"]["weights_sha256"] == "deadbeefcafe"
    assert step["metadata"]["framework"] == "torchxrayvision"
    assert set(report.keys()) == REQUIRED_REPORT_KEYS  # top-level schema unchanged

    markdown = report_to_markdown(report)
    assert "## Models used" in markdown
    assert "deadbeefcafe" in markdown
    assert "torchxrayvision" in markdown


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
    assert set(report.keys()) == REQUIRED_REPORT_KEYS  # same schema on failure
    json.loads(report_to_json(report))  # still serializable

    markdown = report_to_markdown(report)
    assert "## Errors" in markdown
    assert "Execution success:** False" in markdown


def test_markdown_caps_measurement_rows() -> None:
    # 15x15 grid of isolated bright pixels -> 225 objects, above the cap.
    image = np.zeros((64, 64), dtype=np.uint8)
    image[2:62:4, 2:62:4] = 220
    plan = ExecutionPlan(
        goal="count_specks",
        supported=True,
        explanation="Segment and measure many tiny objects.",
        steps=[ToolStep(tool="segment_otsu"), ToolStep(tool="measure_objects")],
    )
    validation = validate_plan(plan, channels=1)
    execution = execute_plan(validation.normalized_plan, image)
    assert execution.summary is not None
    assert execution.summary.object_count > MAX_MARKDOWN_MEASUREMENT_ROWS

    metadata = ImageMetadata(
        filename="specks.png",
        width=64,
        height=64,
        channels=1,
        mode="L",
        dtype="uint8",
        minimum_intensity=0,
        maximum_intensity=220,
    )
    report = build_report(
        metadata=metadata,
        request="count specks",
        planner_mode="demo",
        plan=validation.normalized_plan,
        execution=execution,
    )
    markdown = report_to_markdown(report)
    assert f"showing {MAX_MARKDOWN_MEASUREMENT_ROWS} of" in markdown
    # JSON keeps the complete table.
    assert len(report["measurements"]) == execution.summary.object_count


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
