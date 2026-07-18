"""Benchmark the required pipelines on the synthetic dataset (spec 14.2-14.4).

Pipelines compared:

1. ``otsu_only``      — Otsu thresholding, nothing else.
2. ``otsu_cleaned``   — Otsu + mask cleanup (tool defaults).
3. ``planner_demo``   — the plan the demo planner generates for the MVP
   reference request, executed through the real validator and executor.

Metrics per case: IoU, Dice, runtime (seconds), and object-count error
against the known ground truth. All numbers are produced by actually
running the pipelines — never hard-coded.

Run from the repository root:

    python benchmark/run_benchmark.py

Outputs: ``benchmark/results.csv`` and ``benchmark/results.md``.
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import skimage  # noqa: E402

from benchmark.generate_dataset import DATASET_SEED, DatasetCase, generate_cases  # noqa: E402
from benchmark.metrics import dice_score, intersection_over_union  # noqa: E402
from src import __version__  # noqa: E402
from src.agent.demo_planner import generate_demo_plan  # noqa: E402
from src.executor import execute_plan  # noqa: E402
from src.plan_validator import validate_plan  # noqa: E402
from src.tools.measurement import measure_objects  # noqa: E402
from src.tools.segmentation import clean_mask, segment_otsu  # noqa: E402

BENCHMARK_DIR = Path(__file__).resolve().parent

REFERENCE_REQUEST = (
    "Remove noise, segment the bright objects, ignore very small regions, and measure them."
)

PIPELINES = ("otsu_only", "otsu_cleaned", "planner_demo")


def run_pipeline(name: str, case: DatasetCase) -> tuple[np.ndarray, float, int]:
    """Run one pipeline on one case.

    Returns:
        (predicted mask, runtime in seconds, predicted object count).
    """
    if name == "otsu_only":
        start = time.perf_counter()
        mask = segment_otsu(case.image)
        runtime = time.perf_counter() - start
        count = measure_objects(mask)[1].object_count
        return mask, runtime, count

    if name == "otsu_cleaned":
        start = time.perf_counter()
        mask = clean_mask(segment_otsu(case.image))
        runtime = time.perf_counter() - start
        count = measure_objects(mask)[1].object_count
        return mask, runtime, count

    if name == "planner_demo":
        plan = generate_demo_plan(REFERENCE_REQUEST, channels=1)
        validation = validate_plan(plan, channels=1)
        if not validation.valid or validation.normalized_plan is None:
            raise RuntimeError(f"Demo plan failed validation: {validation.errors}")
        result = execute_plan(validation.normalized_plan, case.image)
        if not result.success or result.mask is None or result.summary is None:
            raise RuntimeError(f"Planner pipeline failed: {result.errors}")
        return result.mask, result.total_runtime_seconds, result.summary.object_count

    raise ValueError(f"Unknown pipeline {name!r}")


def run_benchmark(seed: int = DATASET_SEED) -> pd.DataFrame:
    """Run every pipeline on every case and return the per-run results."""
    rows: list[dict[str, object]] = []
    for case in generate_cases(seed):
        for pipeline in PIPELINES:
            mask, runtime, predicted = run_pipeline(pipeline, case)
            rows.append(
                {
                    "case": case.name,
                    "pipeline": pipeline,
                    "iou": round(intersection_over_union(mask, case.ground_truth), 4),
                    "dice": round(dice_score(mask, case.ground_truth), 4),
                    "runtime_seconds": round(runtime, 4),
                    "true_objects": case.true_object_count,
                    "predicted_objects": predicted,
                    "count_error": predicted - case.true_object_count,
                }
            )
    return pd.DataFrame(rows)


def _summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.assign(abs_count_error=results["count_error"].abs())
        .groupby("pipeline", sort=False)
        .agg(
            mean_iou=("iou", "mean"),
            mean_dice=("dice", "mean"),
            mean_runtime_s=("runtime_seconds", "mean"),
            mean_abs_count_error=("abs_count_error", "mean"),
        )
        .round(4)
        .reset_index()
    )
    return summary


def _write_markdown(results: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    def table(frame: pd.DataFrame) -> str:
        headers = list(frame.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        lines.extend(
            "| " + " | ".join(str(value) for value in row) + " |"
            for row in frame.itertuples(index=False)
        )
        return "\n".join(lines)

    content = "\n".join(
        [
            "# SciFlow Agent — Benchmark Results",
            "",
            f"- Generated: {datetime.now(UTC).isoformat(timespec='seconds')}",
            f"- Application version: {__version__}",
            f"- Dataset seed: {DATASET_SEED} (6 synthetic 256×256 cases, known ground truth)",
            f"- numpy {np.__version__}, scikit-image {skimage.__version__}, "
            f"pandas {pd.__version__}",
            f'- Reference request (planner_demo): "{REFERENCE_REQUEST}"',
            "",
            "## Summary (mean over all cases)",
            "",
            table(summary),
            "",
            "## Per-case results",
            "",
            table(results),
            "",
            "Notes: the ground truth excludes the debris specks in the `debris_specks` "
            "case, so cleanup-based pipelines are expected to score a lower count error "
            "there. `planner_demo` runs the full plan (denoise → Otsu → cleanup → "
            "measure) through the validator and controlled executor.",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    """Run the benchmark and write CSV and Markdown results."""
    results = run_benchmark()
    summary = _summarize(results)

    csv_path = BENCHMARK_DIR / "results.csv"
    md_path = BENCHMARK_DIR / "results.md"
    results.to_csv(csv_path, index=False)
    _write_markdown(results, summary, md_path)

    print(summary.to_string(index=False))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
