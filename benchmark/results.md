# SciFlow Agent — Benchmark Results

- Generated: 2026-07-18T13:57:15+00:00
- Application version: 0.1.0
- Dataset seed: 1234 (6 synthetic 256×256 cases, known ground truth)
- numpy 2.4.6, scikit-image 0.26.0, pandas 2.3.3
- Reference request (planner_demo): "Remove noise, segment the bright objects, ignore very small regions, and measure them."

## Summary (mean over all cases)

| pipeline | mean_iou | mean_dice | mean_runtime_s | mean_abs_count_error |
| --- | --- | --- | --- | --- |
| otsu_only | 0.8282 | 0.89 | 0.0021 | 912.0 |
| otsu_cleaned | 0.9478 | 0.9728 | 0.0009 | 0.0 |
| planner_demo | 0.9689 | 0.9841 | 0.0217 | 0.0 |

## Per-case results

| case | pipeline | iou | dice | runtime_seconds | true_objects | predicted_objects | count_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| large_low_noise | otsu_only | 0.9931 | 0.9965 | 0.0115 | 12 | 12 | 0 |
| large_low_noise | otsu_cleaned | 0.9931 | 0.9965 | 0.0008 | 12 | 12 | 0 |
| large_low_noise | planner_demo | 0.9922 | 0.9961 | 0.0269 | 12 | 12 | 0 |
| small_low_noise | otsu_only | 0.9654 | 0.9824 | 0.0003 | 20 | 20 | 0 |
| small_low_noise | otsu_cleaned | 0.9654 | 0.9824 | 0.0008 | 20 | 20 | 0 |
| small_low_noise | planner_demo | 0.9574 | 0.9782 | 0.0227 | 20 | 20 | 0 |
| medium_noise | otsu_only | 0.9619 | 0.9806 | 0.0002 | 15 | 16 | 1 |
| medium_noise | otsu_cleaned | 0.9626 | 0.981 | 0.0008 | 15 | 15 | 0 |
| medium_noise | planner_demo | 0.975 | 0.9873 | 0.0217 | 15 | 15 | 0 |
| high_noise | otsu_only | 0.6169 | 0.7631 | 0.0003 | 15 | 2088 | 2073 |
| high_noise | otsu_cleaned | 0.897 | 0.9457 | 0.0011 | 15 | 15 | 0 |
| high_noise | planner_demo | 0.9495 | 0.9741 | 0.0211 | 15 | 15 | 0 |
| low_contrast | otsu_only | 0.4633 | 0.6333 | 0.0002 | 12 | 3397 | 3385 |
| low_contrast | otsu_cleaned | 0.8854 | 0.9392 | 0.0013 | 12 | 12 | 0 |
| low_contrast | planner_demo | 0.9552 | 0.9771 | 0.019 | 12 | 12 | 0 |
| debris_specks | otsu_only | 0.9686 | 0.984 | 0.0003 | 12 | 25 | 13 |
| debris_specks | otsu_cleaned | 0.9835 | 0.9917 | 0.0008 | 12 | 12 | 0 |
| debris_specks | planner_demo | 0.9841 | 0.992 | 0.019 | 12 | 12 | 0 |

Notes: the ground truth excludes the debris specks in the `debris_specks` case, so cleanup-based pipelines are expected to score a lower count error there. `planner_demo` runs the full plan (denoise → Otsu → cleanup → measure) through the validator and controlled executor.
