"""Reproducibility reports (FR-13): JSON and Markdown builders.

First version shipped with the UI phase; Phase 7 completes the schema
(input-file hashing, exhaustive field coverage, dedicated tests).

Reports are built only from explicitly passed values — never from the
configuration object — so secrets cannot leak into a report by
construction.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from src import __version__
from src.agent.schemas import ExecutionPlan
from src.executor import ExecutionResult
from src.models import ImageMetadata

REPORT_VERSION = 1

#: Markdown reports include at most this many measurement rows.
MAX_MARKDOWN_MEASUREMENT_ROWS = 200


def build_report(
    *,
    metadata: ImageMetadata,
    request: str,
    planner_mode: str,
    plan: ExecutionPlan,
    execution: ExecutionResult,
) -> dict[str, Any]:
    """Assemble the reproducibility report as a JSON-serializable dictionary."""
    measurements: list[dict[str, Any]] = []
    if execution.measurements is not None and not execution.measurements.empty:
        measurements = execution.measurements.round(3).to_dict(orient="records")

    return {
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "software_version": __version__,
        "input_image": metadata.to_dict(),
        "user_request": request,
        "planner_mode": planner_mode,
        "plan": plan.model_dump(),
        "execution": {
            "success": execution.success,
            "total_runtime_seconds": round(execution.total_runtime_seconds, 4),
            "steps": [
                {
                    "tool": step.tool,
                    "parameters": step.parameters,
                    "runtime_seconds": round(step.runtime_seconds, 4),
                    "warnings": step.warnings,
                }
                for step in execution.steps
            ],
            "warnings": execution.warnings,
            "errors": execution.errors,
        },
        "summary": execution.summary.to_dict() if execution.summary is not None else None,
        "measurements": measurements,
    }


def report_to_json(report: dict[str, Any]) -> str:
    """Serialize a report dictionary to pretty-printed JSON."""
    return json.dumps(report, indent=2)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "*(no rows)*"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |" for row in rows)
    return "\n".join(lines)


def report_to_markdown(report: dict[str, Any]) -> str:
    """Render a report dictionary as a human-readable Markdown document."""
    image = report["input_image"]
    execution = report["execution"]
    summary = report["summary"]

    sections: list[str] = [
        "# SciFlow Agent — Analysis Report",
        "",
        f"- **Generated:** {report['generated_at']}",
        f"- **Software version:** {report['software_version']}",
        f"- **Planner mode:** {report['planner_mode']}",
        f"- **Execution success:** {execution['success']}",
        f"- **Total runtime:** {execution['total_runtime_seconds']} s",
        "",
        "## Input image",
        "",
        _markdown_table([image]),
        "",
        "## Request",
        "",
        f"> {report['user_request']}",
        "",
        "## Executed plan",
        "",
        f"**Goal:** `{report['plan']['goal']}` — {report['plan']['explanation']}",
        "",
    ]

    step_rows = [
        {
            "step": index,
            "tool": step["tool"],
            "parameters": json.dumps(step["parameters"]),
            "runtime_s": step["runtime_seconds"],
            "warnings": "; ".join(step["warnings"]) or "—",
        }
        for index, step in enumerate(execution["steps"], start=1)
    ]
    sections += [_markdown_table(step_rows), ""]

    if summary is not None:
        sections += ["## Summary statistics", "", _markdown_table([summary]), ""]

    measurements = report["measurements"]
    if measurements:
        shown = measurements[:MAX_MARKDOWN_MEASUREMENT_ROWS]
        sections += ["## Per-object measurements", "", _markdown_table(shown)]
        if len(measurements) > len(shown):
            sections.append(
                f"\n*(showing {len(shown)} of {len(measurements)} objects — "
                "see the JSON report for the complete table)*"
            )
        sections.append("")

    if execution["warnings"]:
        sections += ["## Warnings", ""] + [f"- {w}" for w in execution["warnings"]] + [""]
    if execution["errors"]:
        sections += ["## Errors", ""] + [f"- {e}" for e in execution["errors"]] + [""]

    return "\n".join(sections)
