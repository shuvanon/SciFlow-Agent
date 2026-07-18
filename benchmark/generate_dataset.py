"""Synthetic benchmark dataset with known ground-truth masks (spec 14.1).

Six 256x256 cases produced from one fixed seed, varying noise level, object
size, and foreground intensity. The ground truth is the exact disk mask
*before* blurring and noise. The ``debris_specks`` case additionally draws
tiny 1-2 px bright specks that are deliberately **excluded** from the ground
truth — they model debris that mask cleanup should remove.

Run directly to dump the images and masks as PNGs for inspection:

    python benchmark/generate_dataset.py     # writes benchmark/dataset/
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from skimage import filters  # noqa: E402

DATASET_SEED = 1234
IMAGE_SIZE = 256
BLUR_SIGMA = 1.2


@dataclass(frozen=True)
class DatasetCase:
    """One synthetic test image with its ground truth."""

    name: str
    image: np.ndarray  # (H, W) uint8
    ground_truth: np.ndarray  # (H, W) bool — placed disks only, no debris
    true_object_count: int
    parameters: dict[str, float | int] = field(default_factory=dict)


@dataclass(frozen=True)
class _CaseSpec:
    name: str
    radius_range: tuple[int, int]
    disk_count: int
    foreground: float
    background: float
    noise_sigma: float
    speck_count: int = 0


_CASE_SPECS: tuple[_CaseSpec, ...] = (
    _CaseSpec("large_low_noise", (10, 18), 12, 200, 30, 5),
    _CaseSpec("small_low_noise", (4, 8), 20, 200, 30, 5),
    _CaseSpec("medium_noise", (6, 14), 15, 180, 30, 15),
    _CaseSpec("high_noise", (6, 14), 15, 170, 35, 30),
    _CaseSpec("low_contrast", (8, 14), 12, 95, 60, 8),
    _CaseSpec("debris_specks", (8, 16), 12, 200, 30, 8, speck_count=25),
)


def _place_disks(
    rng: np.random.Generator,
    size: int,
    radii: list[int],
    minimum_gap: int = 4,
) -> list[tuple[int, int, int]]:
    placed: list[tuple[int, int, int]] = []
    for radius in radii:
        for _ in range(300):
            cy = int(rng.integers(radius + 2, size - radius - 2))
            cx = int(rng.integers(radius + 2, size - radius - 2))
            if all(np.hypot(cy - py, cx - px) > radius + pr + minimum_gap for py, px, pr in placed):
                placed.append((cy, cx, radius))
                break
    return placed


def _build_case(spec: _CaseSpec, rng: np.random.Generator) -> DatasetCase:
    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE), float(spec.background))
    ground_truth = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)

    radii = [
        int(r)
        for r in rng.integers(spec.radius_range[0], spec.radius_range[1] + 1, size=spec.disk_count)
    ]
    disks = _place_disks(rng, IMAGE_SIZE, radii)
    for cy, cx, radius in disks:
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        canvas[disk] = spec.foreground
        ground_truth |= disk

    # Debris: bright specks in the image but NOT in the ground truth.
    if spec.speck_count:
        speck_radii = [int(r) for r in rng.integers(1, 3, size=spec.speck_count)]
        for cy, cx, radius in _place_disks(rng, IMAGE_SIZE, speck_radii, minimum_gap=2):
            speck = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
            canvas[speck & ~ground_truth] = spec.foreground

    blurred = filters.gaussian(canvas, sigma=BLUR_SIGMA, preserve_range=True)
    noisy = blurred + rng.normal(0.0, spec.noise_sigma, size=blurred.shape)
    image = np.clip(noisy, 0, 255).astype(np.uint8)

    return DatasetCase(
        name=spec.name,
        image=image,
        ground_truth=ground_truth,
        true_object_count=len(disks),
        parameters={
            "radius_min": spec.radius_range[0],
            "radius_max": spec.radius_range[1],
            "foreground": spec.foreground,
            "background": spec.background,
            "noise_sigma": spec.noise_sigma,
            "speck_count": spec.speck_count,
        },
    )


def generate_cases(seed: int = DATASET_SEED) -> list[DatasetCase]:
    """Generate all benchmark cases deterministically from one seed."""
    rng = np.random.default_rng(seed)
    return [_build_case(spec, rng) for spec in _CASE_SPECS]


def main() -> None:
    """Dump the dataset as PNG files for visual inspection."""
    from PIL import Image

    output_dir = Path(__file__).resolve().parent / "dataset"
    output_dir.mkdir(exist_ok=True)
    for case in generate_cases():
        Image.fromarray(case.image).save(output_dir / f"{case.name}.png")
        Image.fromarray(case.ground_truth.astype(np.uint8) * 255).save(
            output_dir / f"{case.name}_truth.png"
        )
        print(f"{case.name}: {case.true_object_count} objects, params={case.parameters}")
    print(f"Wrote dataset to {output_dir}")


if __name__ == "__main__":
    main()
