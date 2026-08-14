# Example images

Most of these are **real acquired data** — clinical radiology, fluorescence microscopy,
histology, and astronomy — because a scientific-image tool should be judged on scientific
images. A small set of controlled images with known correct answers is kept alongside them
for pinning specific behaviours.

```bash
python examples/fetch_example_data.py    # real data: skimage_*, pydicom_*, montgomery_*
python examples/generate_examples.py     # controlled cases: example_*, synthetic_*
```

Filenames are `<source>_<dataset>.<ext>`, so every image traces back to where it came from.
All of them appear in the app's **Built-in example** dropdown, including the DICOM and TIFF
files.

Every number below came from actually running that request through the demo planner and
executor. None are estimates.

---

## Medical imaging

| File | Size / type | Try this request | Result | What it demonstrates |
|---|---|---|---|---|
| `montgomery_cxr.dcm` | 1024×841 DICOM (DX) | *"Segment the lungs in this DICOM chest X-ray and measure them."* (needs `[ml]`) | 2 lungs | **The flagship medical example.** A real chest radiograph, read as DICOM with no extra dependencies, then segmented by the pretrained deep-learning model. DICOM + DL + measurement in one run |
| `example_chest_xray.png` | 1024×841 grey | *"Segment the lungs in this chest X-ray and measure them."* (needs `[ml]`) | 2 lungs | The same radiograph as PNG, for the ML demo without DICOM |
| `pydicom_ct_small.dcm` | 128×128 DICOM (CT) | *"Segment the bright regions in this DICOM image and measure them."* | 1 region | A real CT slice. Look at Image details: intensity range **−896 to 1167** — genuine Hounsfield units, negative values and all |
| `pydicom_mr_small.dcm` | 64×64 DICOM (MR) | *"Segment the bright regions in this DICOM image and measure them."* | 3 regions | A real MR slice; range **127–2145**. A second modality proves DICOM support is not CT-specific |
| `skimage_brain_mri.tif` | 256×256 **16-bit** TIFF | *"Segment the bright regions and measure them."* | 4 regions | Real MRI at native 16-bit depth, range **0–47089**. Medical data is rarely 8-bit — this is what the loader's rescaling exists for |
| `skimage_retina.png` | 1411×1411 RGB | *"Improve the contrast, segment the dark objects, and count them."* | 42 objects | Fundus photograph with strong uneven illumination — the classic case for contrast enhancement before thresholding |
| `skimage_microaneurysms.png` | 102×102 grey | *"Segment the dark spots and count them."* | 4 lesions | Retinal microaneurysms: small, dark, low contrast |

## Microscopy and histology

| File | Size / type | Try this request | Result | What it demonstrates |
|---|---|---|---|---|
| `skimage_human_mitosis.png` | 512×512 grey | *"Segment the bright nuclei, ignore very small regions, and measure them."* | **285 nuclei** | Fluorescence microscopy of dividing human cells — the single best "this is what the tool is for" example. Bright nuclei on dark, densely packed |
| `skimage_cells3d_nuclei.tif` | 256×256 **16-bit** TIFF | *"Segment the bright nuclei and measure them."* | 23 nuclei | Real confocal microscopy, nuclei channel, native 16-bit (range **1091–58327**). Large touching nuclei, so it also shows the merging limitation on real data |
| `skimage_skin.png` | 1280×960 RGB | *"Segment the dark regions and measure them."* | 753 regions | Histology section — dense, textured, and genuinely hard. A realistic look at what happens on tissue |
| `skimage_immunohistochemistry.png` | 512×512 RGB | *"Segment the dark regions and measure them."* | 75 regions | Immunohistochemistry staining. Add *"ignore regions smaller than 200 pixels"* and it drops to 4 — a vivid lesson in how much the cleanup threshold decides |
| `skimage_cell.png` | 550×660 grey | *"Segment the dark objects and measure them."* | 1 object at **96.8%** | ⚠️ A **failure case on purpose**. 96.8% of the frame segmented means the background was captured, not the cell. The fastest way to teach the "Segmented %" sanity check |

## Astronomy

| File | Size / type | Try this request | Result | What it demonstrates |
|---|---|---|---|---|
| `skimage_hubble_deep_field.png` | 1000×872 RGB | *"Count the bright objects in this image."* | **156 objects** | Hubble deep field. Genuinely *count the bright things*, at only 2.2% coverage, and being RGB it exercises the grayscale step |

## Controlled cases (known ground truth)

Generated with fixed seeds, so they reproduce byte for byte. These exist because they have a
**known correct answer** — real acquired data does not come with one, so these are what pin a
specific behaviour to a checkable number.

| File | Try this request | Result | What it pins |
|---|---|---|---|
| `example_cells.png` | *"Remove noise, segment the bright objects, ignore very small regions, and measure them."* | 30 objects | The reference workflow. The grayscale step is skipped — the planner knows the image is already 1-channel |
| `example_objects.png` | *"Count the bright objects in this image."* | 10 objects | RGB → grayscale conversion appearing in the plan |
| `synthetic_rings.png` | *"Segment the bright objects, ignore regions smaller than 600 pixels, fill the holes, and measure them."* | 12 objects, area **28 245 → 31 655** | The only demonstration of **`fill_holes`**. Run it once without *"fill the holes"* and compare: same 12 objects, **+12% area** |
| `synthetic_low_contrast.png` | *"Improve the contrast, segment the bright objects, and count them."* | **19 of 20** (Otsu alone: **11**) | That contrast enhancement is worth planning. 20 faint objects under an illumination gradient — the truth is known, so the improvement is measurable rather than merely visible |
| `synthetic_touching_objects.png` | *"Count the bright objects in this image."* | **9** (18 disks drawn) | The touching-object limitation, quantified. Real dense nuclei merge too, but here the exact error is known: 18 → 9. Watershed separation is roadmap v0.2 |
| `synthetic_blank.png` | *"Count the bright objects in this image."* | **0 objects** | The graceful-empty path: an empty-mask warning, an all-zero summary, and a **report that still downloads** |

### Note: `fill_holes` shares the size threshold

`clean_mask` fills holes *smaller than `minimum_object_size`* — the same number that removes
small objects. With the default (30 px) nothing visible happens to `synthetic_rings.png`,
whose holes are 154–452 px. The suggested request sets **600**, above every hole and below
every ring (the smallest annulus is ~1 370 px). Real behaviour worth knowing, not a quirk of
the example.

---

## Provenance and licensing

These files are redistributed from third-party sources. **Check the upstream terms before
publishing this repository**, and cite each source as it asks.

| Source | Files | Notes |
|---|---|---|
| [scikit-image sample data](https://scikit-image.org/docs/stable/api/skimage.data.html) | `skimage_*` | Bundled with, or downloaded by, scikit-image. Each dataset has its own provenance and licence — `human_mitosis`, `cells3d`, `brain`, `skin`, and `retina` all originate from named research groups credited in scikit-image's documentation |
| pydicom test data | `pydicom_ct_small.dcm`, `pydicom_mr_small.dcm` | `CT_small.dcm` and `MR_small.dcm`, shipped with pydicom for testing |
| [Montgomery County CXR set](https://openi.nlm.nih.gov/faq) (U.S. National Library of Medicine) | `montgomery_cxr.dcm`, `example_chest_xray.png` | Public research dataset of chest radiographs with radiologist lung masks. Free for research; cite as the source requires |
| Generated here | `example_*`, `synthetic_*` | Produced by `generate_examples.py`; same licence as this project |

### About `montgomery_cxr.dcm`

This file is **derived, not original**: a real Montgomery radiograph, downscaled to fit the
app's size limit and re-wrapped as DICOM by `fetch_example_data.py` so that one example
exercises DICOM reading and deep-learning segmentation together.

It is a demonstration fixture and **not a medical record**. Every patient-identifying field
is an explicit placeholder (`ANONYMOUS^PLACEHOLDER`, `SCIFLOW-EXAMPLE-001`), and the file's
`ImageComments` says so in the metadata itself. It describes no individual and carries no
clinical study.

---

## Related

- [`../docs/results_guide.md`](../docs/results_guide.md) — how to read what comes back
- [`run_pipeline_demo.py`](run_pipeline_demo.py) — the same pipeline as a script, writing
  every intermediate to `reports/demo/`

For **quantitative** evaluation against ground truth, the
[Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/) is the standard free
source: BBBC005 (known cell counts), BBBC039 (nuclei with masks), and BBBC004 (controllable
object overlap, which maps directly onto the touching-object limitation above).
