"""Run the complete Phase 2 image-processing pipeline from Python, no planner needed.

From the repository root:

    python examples/run_pipeline_demo.py

Loads the built-in cells example, runs grayscale -> denoise -> contrast ->
Otsu segmentation -> mask cleanup -> measurement, prints the summary, and
saves all intermediate images plus the measurements table into
``reports/demo/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PIL import Image  # noqa: E402

from src.config import EXAMPLES_DIR, REPORTS_DIR  # noqa: E402
from src.image_io import load_image_file  # noqa: E402
from src.tools.measurement import label_objects, measure_objects  # noqa: E402
from src.tools.preprocessing import (  # noqa: E402
    convert_to_grayscale,
    denoise_median,
    enhance_contrast,
)
from src.tools.segmentation import clean_mask, segment_otsu  # noqa: E402
from src.visualization import (  # noqa: E402
    create_labelled_image,
    create_overlay,
    to_display_mask,
)


def main() -> None:
    """Execute the reference workflow on the bundled cells example."""
    output_dir = REPORTS_DIR / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_image_file(EXAMPLES_DIR / "example_cells.png")
    print(
        f"Loaded {loaded.metadata.filename}: {loaded.metadata.width}x"
        f"{loaded.metadata.height}, mode {loaded.metadata.mode}"
    )

    grayscale = convert_to_grayscale(loaded.original)
    denoised = denoise_median(grayscale, radius=2)
    enhanced = enhance_contrast(denoised, clip_limit=0.01)
    raw_mask = segment_otsu(enhanced, polarity="bright")
    mask = clean_mask(raw_mask, minimum_object_size=40, fill_holes=True)
    # Intensity is measured on the unenhanced grayscale (the photometric
    # baseline), matching what the executor does — not on `enhanced`.
    measurements, summary = measure_objects(mask, intensity_image=grayscale)
    labels = label_objects(mask)

    Image.fromarray(grayscale).save(output_dir / "01_grayscale.png")
    Image.fromarray(denoised).save(output_dir / "02_denoised.png")
    Image.fromarray(enhanced).save(output_dir / "03_enhanced.png")
    Image.fromarray(to_display_mask(raw_mask)).save(output_dir / "04_raw_mask.png")
    Image.fromarray(to_display_mask(mask)).save(output_dir / "05_clean_mask.png")
    Image.fromarray(create_overlay(loaded.original, mask)).save(output_dir / "06_overlay.png")
    Image.fromarray(create_labelled_image(labels, grayscale)).save(output_dir / "07_labels.png")
    measurements.to_csv(output_dir / "measurements.csv", index=False)

    raw_count = measure_objects(raw_mask)[1].object_count
    print(f"Objects before cleanup: {raw_count}")
    print(f"Objects after cleanup:  {summary.object_count}")
    print(f"Mean area:              {summary.mean_area:.1f} px")
    print(f"Median area:            {summary.median_area:.1f} px")
    print(f"Area range:             {summary.minimum_area:.0f}-{summary.maximum_area:.0f} px")
    print(
        f"Total segmented area:   {summary.total_segmented_area:.0f} px "
        f"({summary.segmented_area_percent:.2f}% of image)"
    )
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
