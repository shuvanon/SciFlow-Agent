"""Deterministically generate the built-in example images.

Run from the repository root:

    python examples/generate_examples.py

Both images are produced with fixed random seeds, so regenerating them always
yields identical files.

- ``example_cells.png``: grayscale microscopy-like image with bright,
  blurred blobs ("cells") of varying size, tiny specks, and sensor noise.
- ``example_objects.png``: RGB image with bright geometric objects on a dark
  background, used to exercise RGB-to-grayscale conversion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage import filters

EXAMPLES_DIR = Path(__file__).resolve().parent

CELLS_SEED = 42
OBJECTS_SEED = 7


def _place_centers(
    rng: np.random.Generator,
    height: int,
    width: int,
    radii: list[int],
    minimum_gap: int,
) -> list[tuple[int, int, int]]:
    """Place non-overlapping disk centers by rejection sampling."""
    placed: list[tuple[int, int, int]] = []
    for radius in radii:
        for _ in range(200):
            cy = int(rng.integers(radius + 4, height - radius - 4))
            cx = int(rng.integers(radius + 4, width - radius - 4))
            if all(np.hypot(cy - py, cx - px) > radius + pr + minimum_gap for py, px, pr in placed):
                placed.append((cy, cx, radius))
                break
    return placed


def generate_cells_image(seed: int = CELLS_SEED, height: int = 512, width: int = 512) -> np.ndarray:
    """Create a grayscale (H, W) uint8 microscopy-like image of bright blobs."""
    rng = np.random.default_rng(seed)
    canvas = np.full((height, width), 25.0)
    yy, xx = np.mgrid[0:height, 0:width]

    cell_radii = [int(r) for r in rng.integers(6, 20, size=30)]
    for cy, cx, radius in _place_centers(rng, height, width, cell_radii, minimum_gap=6):
        intensity = float(rng.uniform(140, 230))
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        canvas[mask] = intensity

    # Tiny specks: realistic debris that mask cleanup should remove later.
    speck_radii = [int(r) for r in rng.integers(1, 3, size=15)]
    for cy, cx, radius in _place_centers(rng, height, width, speck_radii, minimum_gap=2):
        intensity = float(rng.uniform(120, 190))
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        canvas[mask] = intensity

    blurred = filters.gaussian(canvas, sigma=1.4, preserve_range=True)
    noisy = blurred + rng.normal(0.0, 7.0, size=blurred.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def generate_objects_image(
    seed: int = OBJECTS_SEED, height: int = 480, width: int = 640
) -> np.ndarray:
    """Create an RGB (H, W, 3) uint8 image of bright geometric objects."""
    rng = np.random.default_rng(seed)
    canvas = np.zeros((height, width, 3), dtype=np.float64)
    canvas[..., :] = (40.0, 42.0, 48.0)
    yy, xx = np.mgrid[0:height, 0:width]

    palette = [
        (235, 200, 90),  # yellow
        (110, 200, 235),  # light blue
        (230, 120, 110),  # salmon
        (150, 230, 140),  # green
        (220, 160, 235),  # violet
        (240, 240, 240),  # white
    ]

    disk_radii = [int(r) for r in rng.integers(12, 42, size=10)]
    centers = _place_centers(rng, height, width, disk_radii, minimum_gap=10)
    for index, (cy, cx, radius) in enumerate(centers):
        color = palette[index % len(palette)]
        if index % 3 == 0:  # axis-aligned square
            half = radius
            canvas[cy - half : cy + half, cx - half : cx + half] = color
        elif index % 3 == 1:  # ellipse
            ry, rx = radius, max(6, radius // 2)
            mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
            canvas[mask] = color
        else:  # disk
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
            canvas[mask] = color

    noisy = canvas + rng.normal(0.0, 5.0, size=canvas.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def main() -> None:
    """Generate and save both example images into the examples directory."""
    cells = generate_cells_image()
    objects = generate_objects_image()

    cells_path = EXAMPLES_DIR / "example_cells.png"
    objects_path = EXAMPLES_DIR / "example_objects.png"
    Image.fromarray(cells).save(cells_path)  # 2D uint8 -> mode "L"
    Image.fromarray(objects).save(objects_path)  # 3D uint8 -> mode "RGB"

    print(f"Wrote {cells_path}")
    print(f"Wrote {objects_path}")


if __name__ == "__main__":
    main()
