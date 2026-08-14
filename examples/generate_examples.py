"""Deterministically generate the built-in example images.

Run from the repository root:

    python examples/generate_examples.py

Every image is produced with a fixed random seed, so regenerating them always
yields identical files.

- ``example_cells.png``: grayscale microscopy-like image with bright,
  blurred blobs ("cells") of varying size, tiny specks, and sensor noise.
- ``example_objects.png``: RGB image with bright geometric objects on a dark
  background, used to exercise RGB-to-grayscale conversion.

The ``synthetic_*`` images below each isolate one capability that the real
sample datasets do not show cleanly:

- ``synthetic_rings.png``: annular objects — the ``fill_holes`` option is
  visibly different with and without.
- ``synthetic_low_contrast.png``: faint objects a few levels above the
  background — Otsu alone under-detects, ``enhance_contrast`` rescues it.
- ``synthetic_touching_objects.png``: deliberately overlapping disks, which
  connected-component labelling merges. Demonstrates the known limitation.
- ``synthetic_blank.png``: a single uniform intensity — exercises the "no
  objects detected" path end to end.

These exist because they have a **known correct answer**, which real acquired
data does not: each one pins a specific behaviour to a number that can be
checked. Real medical and scientific examples are fetched separately by
``fetch_example_data.py`` and are what the app should be demonstrated on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage import filters

EXAMPLES_DIR = Path(__file__).resolve().parent

CELLS_SEED = 42
OBJECTS_SEED = 7
RINGS_SEED = 11
LOW_CONTRAST_SEED = 23
TOUCHING_SEED = 5


#: Hole radius as a fraction of the ring's outer radius (see generate_rings_image).
RING_INNER_RATIO = 0.35


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


def generate_rings_image(seed: int = RINGS_SEED, height: int = 512, width: int = 512) -> np.ndarray:
    """Create a grayscale image of bright annuli (rings) on a dark background.

    Each object is a filled disk with its center punched back out to the
    background level, so ``clean_mask(fill_holes=True)`` measurably changes
    the reported areas.
    """
    rng = np.random.default_rng(seed)
    background = 22.0
    canvas = np.full((height, width), background)
    yy, xx = np.mgrid[0:height, 0:width]

    outer_radii = [int(r) for r in rng.integers(22, 38, size=12)]
    for cy, cx, radius in _place_centers(rng, height, width, outer_radii, minimum_gap=10):
        intensity = float(rng.uniform(170, 235))
        distance_squared = (yy - cy) ** 2 + (xx - cx) ** 2
        canvas[distance_squared <= radius**2] = intensity
        # Punch the hole. The ratio is chosen so every hole (154-452 px) stays
        # below a usable minimum_object_size while every annulus (>1300 px)
        # stays above it — clean_mask fills holes smaller than that same
        # threshold, so both have to sit on opposite sides of it.
        inner = max(4, int(radius * RING_INNER_RATIO))
        canvas[distance_squared <= inner**2] = background

    blurred = filters.gaussian(canvas, sigma=1.0, preserve_range=True)
    noisy = blurred + rng.normal(0.0, 4.0, size=blurred.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def generate_low_contrast_image(
    seed: int = LOW_CONTRAST_SEED, height: int = 512, width: int = 512
) -> np.ndarray:
    """Create a faint, unevenly lit image that a global threshold under-detects.

    Objects sit only ~16-28 grey levels above the background and an
    illumination gradient makes one side of the frame darker than the other.
    A single global Otsu threshold therefore cannot suit both sides and finds
    roughly half the objects; CLAHE equalizes locally first and recovers most
    of them.

    Objects are kept small relative to CLAHE's kernel (about an eighth of the
    frame). Objects approaching the kernel size would become their own local
    background and be equalized away — which would defeat the point.
    """
    rng = np.random.default_rng(seed)
    canvas = np.full((height, width), 96.0)
    yy, xx = np.mgrid[0:height, 0:width]

    radii = [int(r) for r in rng.integers(6, 12, size=20)]
    for cy, cx, radius in _place_centers(rng, height, width, radii, minimum_gap=8):
        intensity = float(rng.uniform(112, 124))  # only 16-28 above background
        canvas[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2] = intensity

    # A slow illumination gradient — the classic reason a global threshold
    # fails: the same object is brighter on one side of the frame.
    gradient = np.linspace(-8.0, 8.0, width)[None, :]
    blurred = filters.gaussian(canvas + gradient, sigma=1.2, preserve_range=True)
    noisy = blurred + rng.normal(0.0, 1.0, size=blurred.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def generate_touching_objects_image(
    seed: int = TOUCHING_SEED, height: int = 512, width: int = 512
) -> np.ndarray:
    """Create a grayscale image of deliberately overlapping bright disks.

    Pairs and triplets are placed so their disks intersect. Connected-component
    labelling counts each cluster as one object, which is the limitation
    watershed separation would remove (roadmap v0.2).
    """
    rng = np.random.default_rng(seed)
    canvas = np.full((height, width), 24.0)
    yy, xx = np.mgrid[0:height, 0:width]

    # Cluster anchors are well separated; members of a cluster deliberately are not.
    anchor_radii = [int(r) for r in rng.integers(20, 28, size=7)]
    for cy, cx, radius in _place_centers(rng, height, width, anchor_radii, minimum_gap=80):
        for _ in range(int(rng.integers(2, 4))):  # 2-3 disks per cluster
            angle = float(rng.uniform(0, 2 * np.pi))
            # Offset < 2*radius guarantees the disks intersect.
            offset = float(rng.uniform(0.9, 1.4)) * radius
            oy = int(np.clip(cy + offset * np.sin(angle), radius, height - radius - 1))
            ox = int(np.clip(cx + offset * np.cos(angle), radius, width - radius - 1))
            intensity = float(rng.uniform(180, 230))
            canvas[(yy - oy) ** 2 + (xx - ox) ** 2 <= radius**2] = intensity

    blurred = filters.gaussian(canvas, sigma=1.2, preserve_range=True)
    noisy = blurred + rng.normal(0.0, 5.0, size=blurred.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def generate_blank_image(height: int = 256, width: int = 256, level: int = 128) -> np.ndarray:
    """Create a featureless grayscale image: a single uniform intensity.

    Deliberately noiseless. ``segment_otsu`` short-circuits to an all-background
    mask when an image has one constant intensity, so the run completes with
    zero objects, an all-zero summary, a "segmentation produced an empty mask"
    warning, and a downloadable report — the graceful-empty path end to end.

    (Adding even mild noise defeats this: Otsu would split the noise around the
    mean and report a hundred meaningless "objects". That is a different and
    much less useful demonstration.)
    """
    return np.full((height, width), level, dtype=np.uint8)


def main() -> None:
    """Generate and save every controlled example into the examples directory."""
    images: list[tuple[str, np.ndarray]] = [
        ("example_cells.png", generate_cells_image()),  # 2D uint8 -> mode "L"
        ("example_objects.png", generate_objects_image()),  # 3D uint8 -> "RGB"
        ("synthetic_rings.png", generate_rings_image()),
        ("synthetic_low_contrast.png", generate_low_contrast_image()),
        ("synthetic_touching_objects.png", generate_touching_objects_image()),
        ("synthetic_blank.png", generate_blank_image()),
    ]

    for filename, array in images:
        path = EXAMPLES_DIR / filename
        Image.fromarray(array).save(path)
        print(f"Wrote {path.name}  ({array.shape}, {array.dtype})")


if __name__ == "__main__":
    main()
