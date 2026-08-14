"""Benchmark classical vs deep-learning lung segmentation on real chest X-rays.

Compares three pipelines on the Montgomery County CXR set (real images with
manual left/right lung masks), scoring Dice/IoU against the ground truth:

1. ``otsu_bright``  — Otsu thresholding, bright polarity (classical).
2. ``otsu_dark``    — Otsu thresholding, dark polarity (classical).
3. ``ml_cxr_lung``  — the segment_ml deep-learning tool (torchxrayvision).

This complements the synthetic benchmark (run_benchmark.py). It needs the
optional ``[ml]`` dependencies and the Montgomery County chest X-ray set
(with its manual lung masks) unpacked into ``data/montgomery``. All numbers
are produced by running the pipelines — never hard-coded.

    python benchmark/run_cxr_benchmark.py [num_images]

Outputs: ``benchmark/cxr_results.csv`` and ``benchmark/cxr_results.md``.
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.transform import resize

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmark.metrics import dice_score, intersection_over_union  # noqa: E402
from src import __version__  # noqa: E402
from src.tools.ml_segmentation import model_metadata, segment_ml  # noqa: E402
from src.tools.segmentation import segment_otsu  # noqa: E402

BENCHMARK_DIR = Path(__file__).resolve().parent
MONTGOMERY = _PROJECT_ROOT / "data" / "montgomery" / "MontgomerySet"
IMG_DIR = MONTGOMERY / "CXR_png"
LEFT_DIR = MONTGOMERY / "ManualMask" / "leftMask"
RIGHT_DIR = MONTGOMERY / "ManualMask" / "rightMask"
EVAL_SIZE = 512
DEFAULT_NUM_IMAGES = 20


def _load_pair(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an image and its ground-truth lung mask, resized to EVAL_SIZE."""
    image = np.asarray(Image.open(image_path).convert("L"))
    image = resize(image, (EVAL_SIZE, EVAL_SIZE), order=1, preserve_range=True).astype(np.uint8)
    left = np.asarray(Image.open(LEFT_DIR / image_path.name).convert("L")) > 127
    right = np.asarray(Image.open(RIGHT_DIR / image_path.name).convert("L")) > 127
    truth = resize((left | right).astype(float), (EVAL_SIZE, EVAL_SIZE), order=0) > 0.5
    return image, truth


def _run_pipeline(name: str, image: np.ndarray) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    if name == "otsu_bright":
        mask = segment_otsu(image, polarity="bright")
    elif name == "otsu_dark":
        mask = segment_otsu(image, polarity="dark")
    elif name == "ml_cxr_lung":
        mask = segment_ml(image, model_name="cxr_lung")
    else:
        raise ValueError(f"Unknown pipeline {name!r}")
    return mask, time.perf_counter() - start


PIPELINES = ("otsu_bright", "otsu_dark", "ml_cxr_lung")


def main() -> None:
    if not IMG_DIR.is_dir():
        print(
            f"Montgomery dataset not found at {IMG_DIR}.\n"
            "Download the Montgomery County chest X-ray set (images + manual lung "
            "masks) into data/montgomery, then re-run."
        )
        sys.exit(1)

    num_images = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_NUM_IMAGES
    images = sorted(IMG_DIR.glob("*.png"))[:num_images]

    rows: list[dict[str, object]] = []
    for image_path in images:
        image, truth = _load_pair(image_path)
        for pipeline in PIPELINES:
            mask, runtime = _run_pipeline(pipeline, image)
            rows.append(
                {
                    "image": image_path.name,
                    "pipeline": pipeline,
                    "dice": round(dice_score(mask, truth), 4),
                    "iou": round(intersection_over_union(mask, truth), 4),
                    "runtime_seconds": round(runtime, 4),
                }
            )

    _write_outputs(rows, len(images))


def _mean(rows: list[dict[str, object]], pipeline: str, field: str) -> float:
    values = [float(r[field]) for r in rows if r["pipeline"] == pipeline]
    return round(sum(values) / len(values), 4) if values else 0.0


def _write_outputs(rows: list[dict[str, object]], num_images: int) -> None:
    import csv

    csv_path = BENCHMARK_DIR / "cxr_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = [
        {
            "pipeline": pipeline,
            "mean_dice": _mean(rows, pipeline, "dice"),
            "mean_iou": _mean(rows, pipeline, "iou"),
            "mean_runtime_s": _mean(rows, pipeline, "runtime_seconds"),
        }
        for pipeline in PIPELINES
    ]

    def table(items: list[dict[str, object]]) -> str:
        headers = list(items[0].keys())
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        lines += ["| " + " | ".join(str(item[h]) for h in headers) + " |" for item in items]
        return "\n".join(lines)

    meta = model_metadata("cxr_lung")
    md = "\n".join(
        [
            "# SciFlow Agent — CXR Lung Segmentation Benchmark (classical vs ML)",
            "",
            f"- Generated: {datetime.now(UTC).isoformat(timespec='seconds')}",
            f"- Application version: {__version__}",
            f"- Dataset: Montgomery County CXR set, first {num_images} images with "
            "manual lung masks",
            f"- Model: {meta.get('display_name')} "
            f"({meta.get('framework')} {meta.get('framework_version')}, "
            f"torch {meta.get('torch_version')}, device {meta.get('device')})",
            f"- Weights SHA-256: `{meta.get('weights_sha256')}`",
            "",
            "## Summary (mean over all images)",
            "",
            table(summary),
            "",
            "Classical Otsu thresholding can not isolate lung fields (the brightest "
            "pixels are bone, the darkest are air outside the body), so both polarities "
            "score poorly; the deep-learning model learned lung anatomy.",
            "",
        ]
    )
    (BENCHMARK_DIR / "cxr_results.md").write_text(md, encoding="utf-8")

    for item in summary:
        print(
            f"{item['pipeline']:14s} dice={item['mean_dice']:.3f} "
            f"iou={item['mean_iou']:.3f} runtime={item['mean_runtime_s']:.3f}s"
        )
    print(f"\nWrote {csv_path}")
    print(f"Wrote {BENCHMARK_DIR / 'cxr_results.md'}")


if __name__ == "__main__":
    main()
