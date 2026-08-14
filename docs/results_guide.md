# Reading the Results

Everything SciFlow Agent shows you after a run, panel by panel, in the order it appears on
screen — what each number means, how it is computed, and what it tells you when it looks
wrong.

For *using* the app, see [user_guide.md](user_guide.md). For the images referenced here, see
[examples/README.md](../examples/README.md).

**Units are pixels, always.** There is no physical calibration — no µm, no mm — even for
DICOM, where the pixel-spacing tag exists but is not applied. An "area" of 1 370 means 1 370
pixels.

---

## Before you run: Image details

This panel appears next to the image preview, before any plan exists. It is easy to skip and
worth ten seconds.

| Field | Meaning | Why it matters |
|---|---|---|
| **Size** | Width × height in pixels | Everything downstream is measured in these units |
| **Channels** | 1 (grayscale) or 3 (RGB) | **This changes the plan.** With 1 channel the planner omits the grayscale-conversion step; with 3 it adds one. The plan adapts to the data, not just to your words |
| **Original mode / dtype** | The PIL mode and NumPy dtype *as found on disk* — `L`, `RGB`, `I;16`, `DICOM (MONOCHROME2)` | Shows what was really loaded, before normalization |
| **Intensity range** | Minimum–maximum of the original data | For 8-bit images this is within 0–255. For `synthetic_cells_16bit.tif` it reads **800–11165**; for `pydicom_ct_small.dcm`, **−896 to 1167** (Hounsfield units). Proof the original range was recorded even though processing happens in 8-bit |

> All processing is done on an 8-bit normalized copy. The original range is preserved in the
> metadata and in the report, never in the pixel values you measure.

---

## Step 1: The plan, before it runs

Not results yet — but the plan is half of what makes a result meaningful, and it is recorded
in the report alongside the numbers.

| Element | What to look at |
|---|---|
| **Goal** | A short machine-readable slug (`denoise_segment_measure`) |
| **Planner** | `demo` or `llm` — which planner produced this |
| **Explanation** | One plain sentence. It is generated *from the plan structure*, not written by the model, so it cannot misdescribe what will run |
| **Numbered steps** | Each tool in execution order with its resolved parameters — defaults filled in, nothing implicit |
| **⚠️ Warnings** | **Never skip these.** This is where the system tells you it changed your intent |
| **Raw plan (JSON)** | The actual validated object. Worth opening once: nothing is hidden behind the rendering |
| **Validation banner** | "N steps, all tools approved, all parameters within safe ranges" |

### Warnings you will actually see

| Warning | What happened |
|---|---|
| *"Segmentation was added because removing small regions or measuring requires detected objects."* | You asked to count or measure without asking to segment. The planner added segmentation and cleanup — counting an uncleaned mask would count noise specks |
| *"Requested median radius 99 exceeds the maximum of 5; using 5."* | Your number was outside the tool's declared bounds and was clamped. The plan runs with 5 |
| *"Requested minimum object size … is below the minimum of 0; using 0."* | Same, for the cleanup threshold |

If the plan is **rejected**, there is no execute button at all — not a disabled one. The
action is unreachable, not merely discouraged.

---

## Step 2: Result images

Five images, then every intermediate.

### Original vs. Processed

**Original** is your input as loaded. **Processed** is the last 2D grayscale image before
segmentation — after denoising and contrast enhancement.

This pair answers "did the preprocessing earn its place?" On `synthetic_low_contrast.png` the
difference is dramatic; on a clean image, near-invisible, which is itself informative.

> Processed is what the segmentation actually saw. It is **not** what `mean_intensity` is
> measured on — see [the intensity note](#the-mean_intensity-rule) below.

### Segmentation mask

Binary. White = detected, black = background. The raw output of thresholding plus cleanup,
with no image underneath to flatter it.

### Overlay — the honest panel

The mask painted over the original in translucent red (45%). **Look at this before you look
at any number.** It is the only view that shows detections against real pixels, so it is
where you see what was missed and what was over-claimed.

If a count looks surprising, the overlay usually explains it in a second.

### Labelled objects

Each connected component in its own colour. This is what "object count" *literally* means:
one colour, one counted object.

It is the single best view for the touching-object limitation. On
`synthetic_touching_objects.png` you can see two or three disks sharing one colour — 18 disks
drawn, **9 counted**. Nothing is wrong with the code; connected-component labelling merges
regions that touch, and separating them (watershed) is roadmap v0.2.

### Intermediate images (expander)

One thumbnail per executed step, in order, keyed `01_convert_to_grayscale`,
`02_denoise_median`, and so on. Every intermediate array is retained.

This is what makes the pipeline inspectable rather than a black box with two ends. If a
result is wrong, this panel localizes *which step* broke it.

---

## Step 3: Summary statistics

Eight metrics. Definitions are exact — these are what the code computes.

| Metric | Exactly what it is | How to read it |
|---|---|---|
| **Objects** | Count of connected components in the cleaned mask, at **8-connectivity** (diagonal neighbours belong to the same object) | The headline number. 8- vs 4-connectivity genuinely changes counts for thin diagonal structures |
| **Mean area** | Arithmetic mean of per-object pixel areas | — |
| **Median area** | Median of per-object pixel areas | **Compare it to the mean.** Far apart ⇒ a few very large objects are dragging the mean up, usually merged/touching objects or background leakage. Close together ⇒ a homogeneous population |
| **Area range** | Smallest–largest object area | If the minimum sits exactly at your cleanup threshold, the threshold is doing real work — try changing it |
| **Segmented area** | Total `True` pixels in the mask | Equals the sum of all object areas; a free internal consistency check |
| **Segmented %** | Segmented area ÷ total pixels × 100 | **The fastest sanity check in the app.** On a "bright objects on dark background" image, anything above ~50% means you segmented the background. `skimage_cell.png` reports **96.8%** — that is not a cell, that is the whole frame |
| **Steps** | Number of steps that executed | Matches the plan unless a step failed |
| **Runtime** | Total wall-clock seconds | Classical pipelines: ~0.02–0.1 s. `segment_ml` on GPU: ~2.2 s. The honest cost of the learned model |

### Diagnosing from these numbers alone

| Symptom | Most likely cause | Fix |
|---|---|---|
| Segmented % > 50 on bright-on-dark | Wrong polarity | Add "dark" (or remove it) from the request |
| Objects in the thousands | Noise thresholded into specks | Add "remove noise", or raise the minimum size |
| Objects = 0 | Empty mask, or cleanup removed everything | Check warnings; lower the minimum size; try the opposite polarity |
| Mean ≫ median | Merged/touching objects | Expected on dense images — see the labelled view |
| Count far below expectation | Faint objects lost to a global threshold | Add "improve the contrast" |

---

## Step 4: Per-object measurements

One row per object. Columns, in table order:

| Column | Definition |
|---|---|
| `label` | Object ID, 1-based. **Matches the colour in the Labelled objects panel** |
| `area` | Pixel count of the object |
| `perimeter` | Boundary length in pixels (scikit-image's perimeter estimate, which corrects for staircase edges — it is not simply the count of boundary pixels) |
| `major_axis_length` | Major axis of the ellipse with the same second moments as the region |
| `minor_axis_length` | Minor axis of that same ellipse |
| `centroid_row`, `centroid_col` | Centre of mass, in (row, column) order — **not** (x, y) |
| `bbox_min_row`, `bbox_min_col` | Top-left corner of the bounding box |
| `bbox_max_row`, `bbox_max_col` | Bottom-right corner (exclusive, following NumPy slicing) |
| `mean_intensity` | Mean pixel value inside the object — **conditional**, see below |

### The `mean_intensity` rule

It is measured on the **photometric baseline**: the first 2D grayscale image in the pipeline —
your input, or the output of `convert_to_grayscale` — and **not** on the denoised or
contrast-enhanced image.

This is deliberate. CLAHE is non-linear and spatially adaptive: the same input value maps to
different outputs depending on its surroundings. Measuring intensity after CLAHE would
produce a number that describes *your plan* rather than *your sample* — change `clip_limit`
and every intensity would move, though the specimen did not. Segmentation still runs on the
fully processed image; only the reported intensity comes from the baseline.

**The column is absent** when no 2D grayscale image ever existed — for example an LLM plan
going `segment_ml → clean_mask → measure_objects` directly on an RGB input. The demo planner
always inserts grayscale conversion for RGB, so it is normally present.

### Measurements that are not here (yet)

Circularity, eccentricity, and solidity are **not** computed. Two are easy to derive from
what is:

- **Elongation** = `major_axis_length / minor_axis_length` (1.0 = round)
- **Circularity** = `4π × area / perimeter²` (1.0 = perfect circle)

Both are roadmap v0.2 items.

---

## Step 5: Step timing

Per-step runtime in milliseconds, plus any per-step warnings.

Worth expanding once on an ML run: `segment_ml` dominates by roughly 100× over every
classical step. Profiling is built into the executor, not bolted on afterwards.

---

## Step 6: The report

The most under-used panel. Two formats, identical content: **JSON** (complete, machine
readable) and **Markdown** (human readable, measurement table capped at 200 rows).

### What a report contains

| Key | Contents |
|---|---|
| `report_version` | Schema version of the report itself — pinned by a test, so downstream parsers do not silently break |
| `generated_at` | UTC timestamp, ISO 8601 |
| `software_version` | The SciFlow Agent version that produced it |
| `input_image` | Filename, dimensions, channels, original mode/dtype, intensity range, and the **SHA-256 of the original file bytes** |
| `user_request` | Your request, verbatim |
| `planner_mode` | `demo` or `llm` |
| `plan` | The validated, **normalized** plan — every default filled in, nothing implicit |
| `execution` | Success flag, total runtime, and per step: tool, resolved parameters, runtime, warnings, metadata |
| `summary` | The aggregate statistics, or `null` if measurement never ran |
| `measurements` | Every per-object row (the full table, not the 200-row Markdown cap) |

### The two hashes

**`input_image.sha256`** — two reports carrying the same hash provably ran on the same bytes.
This is what turns "reproducible" from a claim into something checkable.

**`execution.steps[].metadata.weights_sha256`** — for `segment_ml` only, alongside the model
name, framework and torch versions, and the device. Two identical plans run against different
model weights produce visibly different reports. Model provenance, not just code provenance.
The Markdown report renders this as a **Models used** section.

### Reports for failed runs

A failure produces a report too — same input hash, the steps that did complete, and the
error. A failed run is exactly as reproducible as a successful one. This is worth triggering
once deliberately (`synthetic_blank.png` is the gentle version; an out-of-order plan is the
hard version).

---

## Quick reference: the ten things worth checking

1. **Channels** — explains why the plan has, or lacks, a grayscale step
2. **Plan warnings** — the system announcing it changed your intent
3. **No execute button** on a rejected plan — unreachable, not disabled
4. **Overlay** before any number
5. **Labelled objects** — what "count" literally means
6. **Segmented %** — >50% on bright-on-dark means wrong polarity
7. **Mean vs. median area** — far apart means merged objects
8. **Runtime** — 0.02 s classical vs 2.2 s ML
9. **Input SHA-256**, and **weights SHA-256** on ML runs
10. A **failed run's report** — proof that failures are reproducible too
