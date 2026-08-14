"""Presentation metadata for the built-in example images.

Which example the app opens on, the order they are listed in, and the request
that best demonstrates each one. This is data about the example set rather than
UI wiring, so it lives here: ``app.py`` imports it, and tests can read it
without executing the Streamlit script.

Adding an image to ``examples/`` without a catalogue entry is fine — it appears
after the catalogued ones and falls back to the generic prompts.
"""

from __future__ import annotations

from pathlib import Path

#: Curated presentation order for the built-in examples: filename mapped to a
#: category label and the request that best demonstrates that image. Medical
#: imaging leads, because that is what the application is for; the controlled
#: synthetic cases sit last, since they exist to pin behaviour rather than to
#: show the tool off. Files not listed here still appear, after these.
EXAMPLE_CATALOGUE: dict[str, tuple[str, str]] = {
    "pydicom_ct_head.dcm": (
        "CT",
        "Remove noise, segment the bone inside, ignore very small regions, and measure them.",
    ),
    "montgomery_cxr.dcm": (
        "X-ray · DICOM",
        "Segment the lungs in this DICOM chest X-ray and measure them.",
    ),
    "example_chest_xray.png": (
        "X-ray",
        "Segment the lungs in this chest X-ray and measure them.",
    ),
    "pydicom_ct_small.dcm": (
        "CT",
        "Segment the bright regions in this DICOM image and measure them.",
    ),
    "pydicom_mr_small.dcm": (
        "MR",
        "Segment the bright regions in this DICOM image and measure them.",
    ),
    "skimage_brain_mri.tif": (
        "MR",
        "Segment the bright regions and measure them.",
    ),
    "skimage_retina.png": (
        "Retina",
        "Improve the contrast, segment the dark objects, and count them.",
    ),
    "skimage_microaneurysms.png": (
        "Retina",
        "Segment the dark spots and count them.",
    ),
    "skimage_skin.png": (
        "Histology",
        "Segment the dark regions and measure them.",
    ),
    "skimage_immunohistochemistry.png": (
        "Histology",
        "Segment the dark regions, ignore regions smaller than 200 pixels, and measure them.",
    ),
    "skimage_human_mitosis.png": (
        "Microscopy",
        "Remove noise, segment the bright nuclei, ignore very small regions, and measure them.",
    ),
    "skimage_cells3d_nuclei.tif": (
        "Microscopy",
        "Segment the bright nuclei and measure them.",
    ),
    "skimage_cell.png": (
        "Microscopy",
        "Segment the dark objects and measure them.",
    ),
    "skimage_hubble_deep_field.png": (
        "Astronomy",
        "Count the bright objects in this image.",
    ),
    "synthetic_rings.png": (
        "Test case",
        "Segment the bright objects, ignore regions smaller than 600 pixels, "
        "fill the holes, and measure them.",
    ),
    "synthetic_low_contrast.png": (
        "Test case",
        "Improve the contrast, segment the bright objects, and count them.",
    ),
    "synthetic_touching_objects.png": (
        "Test case",
        "Count the bright objects in this image.",
    ),
    "synthetic_blank.png": (
        "Test case",
        "Count the bright objects in this image.",
    ),
    "example_cells.png": (
        "Test case",
        "Remove noise, segment the bright objects, ignore very small regions, and measure them.",
    ),
    "example_objects.png": (
        "Test case",
        "Count the bright objects in this image.",
    ),
}

#: Category labels that count as medical imaging, for ordering and for the
#: test that pins them to the front of the list.
MEDICAL_CATEGORIES = frozenset({"CT", "X-ray", "X-ray · DICOM", "MR", "Retina", "Histology"})

#: Opens on a real CT: medical, DICOM, and it needs no optional dependency, so
#: the first thing a new user sees works on the base install.
DEFAULT_EXAMPLE = "pydicom_ct_head.dcm"

#: Shown when the selected image is not in the catalogue (an upload, or a file
#: added to examples/ without a catalogue entry).
GENERIC_PROMPTS = [
    "Remove noise, segment the bright objects, ignore very small regions, and measure them.",
    "Count the bright objects in this image.",
    "Improve the contrast, segment bright regions, and ignore very small objects.",
]


def ordered_examples(paths: list[Path]) -> list[str]:
    """Catalogue order first, then anything else alphabetically.

    Files listed in the catalogue but absent from disk are skipped: the
    Montgomery X-ray and the pooch-downloaded sets are optional.
    """
    names = {path.name for path in paths}
    ordered = [name for name in EXAMPLE_CATALOGUE if name in names]
    return ordered + sorted(names - set(ordered))


def example_label(name: str) -> str:
    entry = EXAMPLE_CATALOGUE.get(name)
    return f"{entry[0]} · {name}" if entry else name
