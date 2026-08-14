# SciFlow Agent — User Guide

A step-by-step guide for using the app. No programming knowledge required. For
installation and configuration details, see the [README](../README.md); for how the
system works internally, see [architecture.md](architecture.md).

## 1. Start the app

```bash
streamlit run app.py          # from the project folder, with the venv active
# or with Docker:
docker run -p 8501:8501 sciflow-agent
```

Open `http://localhost:8501`. The app starts in **demo mode** and needs no
configuration or internet access.

## 2. The interface at a glance

- **Sidebar** — choose your image (built-in example or upload), pick the planner mode
  (Demo / LLM), and see the LLM connection status. "Advanced settings" shows the active
  safety limits.
- **Main page, section 1** — image preview, its metadata, and the request box.
- **Section 2** — the generated plan and its validation verdict (appears after
  *Generate plan*).
- **Section 3** — results (appears after you execute).
- **Section 4** — report downloads.

## 3. Choose an image

**Built-in examples**: `example_cells.png` (grayscale microscopy-like blobs),
`example_objects.png` (RGB geometric shapes), and `example_chest_xray.png` (for the
deep-learning lung demo).

**Upload**: PNG, JPG/JPEG, TIFF, or **DICOM** (`.dcm`/`.dicom`) up to 4096 × 4096 px
(configurable). RGB, RGBA, palette, 16-bit, float, and DICOM images are accepted and
normalized automatically; the metadata panel shows the original mode and intensity range.
Unsupported or corrupt files are rejected with an explanation.

> Switching images clears any existing plan and results, so nothing stale is ever
> shown for a new image.

## 4. Write a request

Describe what you want in plain English. The system supports:

| You want to… | Say, for example |
|---|---|
| Reduce noise | "remove the noise", "denoise", "smooth the image" |
| Improve contrast | "improve the contrast", "equalize" |
| Find objects | "segment the cells", "detect the bright objects", "find the particles" |
| Find the *densest* structures | "segment the bone", "the brightest regions", "the calcifications" |
| Drop small regions | "ignore very small regions", "remove specks/debris" |
| Count / measure | "count them", "measure their sizes", "how many cells" |

Useful extras the planner understands:

- **Numbers**: "denoise with **radius 4**", "ignore regions **smaller than 100** pixels"
  (values outside the safe ranges are clamped, with a visible warning in the plan).
- **Polarity**: "segment the **dark** objects" — the default is bright-on-dark;
  "bright cells on a dark background" stays bright.
- **Holes**: "fill the holes" enables hole filling during cleanup.

Whenever you ask for counting or measuring, segmentation and small-region cleanup are
included automatically — counting a noisy, uncleaned mask would count specks.

### Images with more than two intensity classes

Ordinary thresholding assumes an image has a foreground and a background. A CT slice has
three groups — air, soft tissue, and bone — and a single threshold cuts between the two
largest, so asking for "the bone" would return the whole body.

Name the structures you want and the planner switches to **multi-level thresholding**, which
splits the intensities into several classes and keeps only the extreme one:

> *"Remove noise, segment the bone, ignore very small regions, and measure them."*

Trigger words: **bone**, **calcification**, **the densest structures**, **the brightest
regions**, **the darkest regions**. The plan shows the choice and a warning explaining why.
If your image has very few distinct grey levels the run stops with a message telling you to
ask for fewer classes.

**What gets rejected**: anything outside image analysis (shell commands, file
operations, downloads, credentials) and out-of-scope topics (3D volumes, NIfTI files,
multi-slice DICOM series, model training). You'll get a message listing what *is*
supported. Rejections cannot execute — this is enforced by the validator, not just by
the planner.

> A **single 2D DICOM image is fully supported** — naming DICOM in your request is fine.
> Only a DICOM *series* (a 3D stack) is out of scope.

### Deep-learning lung segmentation

If the optional `[ml]` extra is installed (`pip install -e ".[ml]"`), you can ask for
deep-learning segmentation of **chest X-ray lungs**:

> *"Segment the lungs in this chest X-ray and measure them."*

The planner routes this to the `segment_ml` tool (a pretrained torchxrayvision model), which
runs on your GPU if available. The report records the model name, version, device, and the
weights hash for reproducibility. Try it with the built-in `example_chest_xray.png` or your
own DICOM chest X-ray. Without the `[ml]` extra, the request is planned but execution shows a
clear "install the ML dependencies" message; every classical tool still works.

## 5. Demo mode vs. LLM mode

| | Demo | LLM |
|---|---|---|
| Needs configuration | No | `.env` with endpoint + model |
| Needs network | No | Yes (local server counts) |
| Understanding | Fixed English keyword rules | Free-form language |
| Speed | Instant | Model-dependent (local models can take minutes) |
| Safety | Identical — both modes pass the same validation | |

Switch modes anytime with the sidebar radio. If the LLM fails (connection, timeout,
invalid output after one automatic retry), you get a clear message and demo mode keeps
working. Only your request text and the image *metadata* are sent to the LLM — never
the image itself.

## 6. Review the plan

After *Generate plan* you see:

- **Goal** and a one-sentence **explanation**;
- the **numbered steps** with their actual parameter values (defaults filled in);
- **warnings**, e.g. "Requested radius 99 exceeds the maximum of 5; using 5";
- the **raw JSON** (expander) — exactly what will be executed;
- the **validation verdict**. A green banner means every tool is approved and every
  parameter is within safe bounds.

Your options (nothing runs until you choose):

- **▶ Execute workflow** — run the plan as shown.
- **✖ Discard plan** — throw it away.
- Edit the request text and press *Generate plan* again for a new plan.

If validation fails, the reasons are listed and no execute button exists.

## 7. Read the results

**Images** — *Original* and *Processed* (after denoising/contrast steps), then
*Segmentation mask* (white = detected), *Overlay* (detection painted on the original),
and *Labelled objects* (each object its own color). "Intermediate images" shows the
output of every single step.

**Summary statistics**

| Metric | Meaning |
|---|---|
| Objects | Number of connected regions after cleanup |
| Mean / Median area | Average / middle object size, in pixels |
| Area range | Smallest–largest object |
| Segmented area / % | Total detected pixels, and share of the image |
| Steps / Runtime | Executed step count and total processing time |

**Measurements table** — one row per object:
`label` (matches the labelled image), `area`, `perimeter`,
`major/minor_axis_length` (object shape), `centroid_row/col` (position),
`bbox_*` (bounding box), `mean_intensity` (average brightness inside the object,
measured on the grayscale image *before* denoising or contrast enhancement, so it
describes the sample rather than the preprocessing).

**Warnings** are informational, not failures: an empty segmentation, zero objects
after cleanup, or "cleanup removed every object" still produce a valid run and report.

> **Every panel and metric explained in depth**, including how to diagnose a wrong
> result from the numbers alone: **[results_guide.md](results_guide.md)**.

## 8. Download and use reports

Two formats, same content:

- **JSON** — complete and machine-readable, including every measurement row.
- **Markdown** — human-readable summary (measurement table capped at 200 rows).

Every report records the timestamp, software version, input image metadata **with a
SHA-256 hash of your exact file**, your request, the planner mode, the full executed
plan with all parameters, per-step runtimes, statistics, measurements, warnings, and
errors. Re-running the same file with the same plan and version reproduces the same
results — the report is the recipe. Reports are also produced for failed runs (with
the error recorded), and never contain API keys.

## 9. Recipes and tips

- **Dark objects on a bright background** → say "dark": *"Segment the dark objects and
  count them."*
- **Dusty / noisy images** → raise the cleanup threshold: *"…ignore regions smaller
  than 100 pixels."*
- **Faint objects, uneven lighting** → add contrast: *"Improve the contrast, then
  segment and measure."*
- **Objects with holes** (rings, vacuoles) → *"…and fill the holes."*
- **"No objects detected"?** Check polarity first (try "dark"), then lower the minimum
  size, then add contrast enhancement. The overlay shows you exactly what was (or
  wasn't) detected.
- Touching objects are counted as one — separating them (watershed) is on the roadmap.

## 10. When something goes wrong

Every error message says what happened and what to do next. The most common cases —
LLM connection/timeout issues, port conflicts, rejected uploads — are collected in the
[README troubleshooting table](../README.md#troubleshooting).
