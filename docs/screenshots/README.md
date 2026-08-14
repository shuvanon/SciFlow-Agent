# Screenshots

**Shots 01–04 are one continuous walkthrough** on `pydicom_ct_head.dcm` — the image the app
opens with — so the sequence reads as a single analysis from input to report. The image never
changes under the reader.

| File | Shows |
|---|---|
| `01_input_request.png` | The app **as it opens**: a real head CT read from DICOM, its metadata (`MONOCHROME2`, intensity range −3995–1812 — genuine Hounsfield units), and the request suggested for that image |
| `02_plan_review.png` | The plan for that request: the goal, a warning explaining why it switched to multi-level thresholding, the numbered steps, the raw-JSON expander, the green validation banner, and the Execute/Discard buttons |
| `03_results_masks.png` | The same run executed: original and processed, segmentation mask, overlay, labelled objects, and the summary metrics |
| `04_measurements_report.png` | The end of that run: summary statistics, the per-object measurement table, and the JSON/Markdown report downloads |

Shots 05–07 cover the two things a head CT cannot demonstrate.

| File | Example | Shows |
|---|---|---|
| `05_ml_plan_review.png` | `montgomery_cxr.dcm` | The deep-learning tool selected through the same validation path as every classical tool |
| `06_ml_results.png` | `montgomery_cxr.dcm` | **DICOM + deep learning together.** A real chest radiograph read as DICOM, both lungs segmented and labelled separately |
| `07_microscopy_results.png` | `skimage_human_mitosis.png` | **Scale.** 275 nuclei in a fluorescence micrograph — the CT walkthrough segments a single object, which is correct for a skull but says nothing about how the tool handles hundreds |

## Why not one image for everything

Three segmentation paths are mutually exclusive by construction, so no single image can show
all of them:

- `segment_otsu` needs **two** intensity groups — an ordinary bright-on-dark image.
- `segment_threshold` (multi-Otsu) needs **three or more**. If a single threshold worked on
  the image, the tool would have nothing to demonstrate.
- `segment_ml` only applies to **chest X-rays** — that is what the model was trained on.

Hence one walkthrough plus a small number of capability shots, rather than switching images
partway through the walkthrough.

## Regenerating

With the app running (`streamlit run app.py`) and Playwright installed
(`pip install playwright && playwright install chromium` — dev machine only, not a project
dependency):

```bash
python docs/screenshots/capture.py
```

The deep-learning shots need the `[ml]` extra; without it `capture_deep_learning` fails at the
execute step. Everything else works on the base install.

Requests are read from [`src/example_catalogue.py`](../../src/example_catalogue.py) rather
than duplicated in the capture script, so a screenshot always shows the request the app
itself suggests for that image.

Three things the script handles that are easy to get wrong:

- **Restart the app after editing anything under `src/`.** Streamlit caches imported modules,
  so a running server will happily screenshot the previous version of the catalogue.
- **The example selectbox arrives pre-filled** and typing inserts at the caret, so the field
  is cleared before the filter text is typed — otherwise the search matches nothing and the
  *previous* image stays selected with no error at all. A `wait_for_selector` now asserts the
  change landed.
- **`scroll_into_view_if_needed()` leaves its anchor at the bottom of the viewport**, so each
  shot scrolls past the anchor to frame the content that follows it.
