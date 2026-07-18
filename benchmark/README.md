# Benchmark

Reproducible comparison of the three required pipelines (spec §14) on a synthetic dataset
with known ground truth.

## Dataset

Six 256×256 grayscale images generated deterministically from seed `1234`
([generate_dataset.py](generate_dataset.py)), varying:

- object size (radii 4–18 px),
- noise level (σ 5–30),
- foreground/background contrast (including a deliberately hard low-contrast case),
- debris (`debris_specks`: tiny bright specks present in the image but **excluded** from the
  ground truth — cleanup should remove them).

The ground truth is the exact disk mask before blur and noise.

## Pipelines

| Pipeline | Steps |
|---|---|
| `otsu_only` | Otsu thresholding |
| `otsu_cleaned` | Otsu + `clean_mask` (defaults) |
| `planner_demo` | The demo planner's plan for the MVP reference request, run through the real validator and executor (denoise → Otsu → cleanup → measure) |

## Metrics

IoU, Dice, runtime (seconds), and signed object-count error vs. the known ground truth.
All values come from actually running the pipelines.

## Run it

From the repository root (with the virtual environment active):

```bash
python benchmark/run_benchmark.py
```

Outputs:

- [results.csv](results.csv) — per-case, per-pipeline rows
- [results.md](results.md) — summary table + configuration (seed, versions)

To dump the dataset images for visual inspection (written to `benchmark/dataset/`,
not committed):

```bash
python benchmark/generate_dataset.py
```
