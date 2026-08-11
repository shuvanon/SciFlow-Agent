# SciFlow Agent — CXR Lung Segmentation Benchmark (classical vs ML)

- Generated: 2026-08-10T22:17:27+00:00
- Application version: 0.1.0
- Dataset: Montgomery County CXR set, first 20 images with manual lung masks
- Model: Chest X-ray lung segmentation (torchxrayvision PSPNet) (torchxrayvision 1.5.2, torch 2.5.1+cu121, device cuda)
- Weights SHA-256: `019b167eac6b729fc1bb92bbbc185fc1730aaa65819f4e3fe718186cadc044fc`

## Summary (mean over all images)

| pipeline | mean_dice | mean_iou | mean_runtime_s |
| --- | --- | --- | --- |
| otsu_bright | 0.1146 | 0.0615 | 0.0031 |
| otsu_dark | 0.5421 | 0.384 | 0.0021 |
| ml_cxr_lung | 0.8696 | 0.7699 | 2.1877 |

Classical Otsu thresholding cannot isolate lung fields (the brightest pixels are bone, the darkest are air outside the body), so both polarities score poorly; the deep-learning model learned lung anatomy.
