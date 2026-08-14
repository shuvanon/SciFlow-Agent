"""Materialize real medical and scientific example images.

Run from the repository root:

    python examples/fetch_example_data.py

Every image is real acquired data — clinical radiology, fluorescence microscopy,
histology, or astronomy — because a scientific-image tool should be demonstrated
on scientific images. Files are named ``<source>_<dataset>.<ext>`` so each one
is traceable to where it came from.

Sources, in order of what they need:

1. **Bundled** — scikit-image and pydicom ship these inside the installed
   package. No network, no account.
2. **Downloaded** — scikit-image fetches these on first use via ``pooch``
   (``pip install pooch``). Skipped with a message when pooch is missing.
3. **Local dataset** — the Montgomery County chest X-ray set, if it has already
   been downloaded to ``data/montgomery`` for the CXR benchmark. Skipped
   entirely when absent.

See ``examples/README.md`` for what each image exercises, a suggested request,
and the provenance and licence of every source.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

EXAMPLES_DIR = Path(__file__).resolve().parent
MONTGOMERY_DIR = EXAMPLES_DIR.parent / "data" / "montgomery" / "MontgomerySet" / "CXR_png"

#: scikit-image datasets bundled with the package (no download).
BUNDLED_SKIMAGE_DATASETS = (
    "hubble_deep_field",  # astronomy: many small bright sources
    "microaneurysms",  # retinal lesions, dark polarity
    "cell",  # brightfield microscopy, single cell
    "immunohistochemistry",  # stained tissue section
    "retina",  # fundus photograph, uneven illumination
)

#: pydicom's bundled clinical images, as (test-file name, output name).
PYDICOM_IMAGES = (
    ("CT_small.dcm", "pydicom_ct_small.dcm"),
    ("MR_small.dcm", "pydicom_mr_small.dcm"),
)

#: Longest edge for the downscaled chest X-ray (originals exceed the 4096 cap).
CXR_TARGET_WIDTH = 1024

#: Slice indices chosen to show well-separated structure in the 3D stacks.
BRAIN_SLICE = 5
CELLS3D_SLICE = 30
CELLS3D_NUCLEI_CHANNEL = 1


def _save(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array).save(path)
    print(f"Wrote {path.name}  ({array.shape}, {array.dtype})")


def fetch_bundled_skimage(output_dir: Path = EXAMPLES_DIR) -> list[Path]:
    """Save scikit-image's bundled 2D sample images as PNG."""
    from skimage import data

    written: list[Path] = []
    for name in BUNDLED_SKIMAGE_DATASETS:
        path = output_dir / f"skimage_{name}.png"
        _save(getattr(data, name)(), path)
        written.append(path)
    return written


def fetch_downloaded_skimage(output_dir: Path = EXAMPLES_DIR) -> list[Path]:
    """Save the scikit-image datasets that ``pooch`` downloads on first use.

    Two of these are 3D stacks; a single 2D slice is extracted, since this
    project handles 2D images only. Both are kept at their native 16-bit depth
    (saved as TIFF) because real microscopy and MRI data are rarely 8-bit — the
    app rescales for processing and records the original range in the report.
    """
    from skimage import data

    written: list[Path] = []
    try:
        _save(data.human_mitosis(), output_dir / "skimage_human_mitosis.png")
        written.append(output_dir / "skimage_human_mitosis.png")

        _save(data.skin(), output_dir / "skimage_skin.png")
        written.append(output_dir / "skimage_skin.png")

        # (slices, channels, rows, cols); channel 1 is the nuclei stain.
        nuclei = data.cells3d()[CELLS3D_SLICE, CELLS3D_NUCLEI_CHANNEL]
        _save(nuclei, output_dir / "skimage_cells3d_nuclei.tif")
        written.append(output_dir / "skimage_cells3d_nuclei.tif")

        _save(data.brain()[BRAIN_SLICE], output_dir / "skimage_brain_mri.tif")
        written.append(output_dir / "skimage_brain_mri.tif")
    except ModuleNotFoundError:
        print("Skipped the downloaded datasets — run `pip install pooch` and re-run.")
    return written


def fetch_pydicom_images(output_dir: Path = EXAMPLES_DIR) -> list[Path]:
    """Copy pydicom's bundled clinical CT and MR slices."""
    from pydicom.data import get_testdata_file

    written: list[Path] = []
    for source_name, output_name in PYDICOM_IMAGES:
        source = get_testdata_file(source_name)
        if not source:
            print(f"Skipped {output_name} — pydicom test file {source_name} not found.")
            continue
        destination = output_dir / output_name
        destination.write_bytes(Path(source).read_bytes())
        print(f"Wrote {destination.name}  (DICOM, from pydicom test data)")
        written.append(destination)
    return written


def _load_montgomery_chest_xray() -> np.ndarray | None:
    """Load and downscale one normal chest X-ray from the Montgomery set.

    Originals are about 4892x4020, above the app's 4096 px limit, so the image
    is reduced to ``CXR_TARGET_WIDTH`` on its long edge. Filenames ending in
    ``_0`` are the radiologist-labelled normal cases.
    """
    if not MONTGOMERY_DIR.is_dir():
        return None
    normal = sorted(MONTGOMERY_DIR.glob("*_0.png"))
    if not normal:
        return None
    with Image.open(normal[0]) as image:
        grayscale = image.convert("L")
        scale = CXR_TARGET_WIDTH / max(grayscale.size)
        size = (round(grayscale.width * scale), round(grayscale.height * scale))
        return np.asarray(grayscale.resize(size, Image.LANCZOS))


def build_chest_xray_dicom(output_dir: Path = EXAMPLES_DIR) -> Path | None:
    """Wrap a real chest X-ray as DICOM so one example exercises both paths.

    This is the example that puts the whole medical pipeline together: a DICOM
    input read with no extra dependencies, then deep-learning lung segmentation.

    The pixel data is a real radiograph from the public Montgomery County
    research set; every patient-identifying field is written as an explicit
    placeholder and the image is labelled as derived, non-clinical data. It is
    a demonstration fixture, not a medical record.
    """
    pixels = _load_montgomery_chest_xray()
    if pixels is None:
        print(
            "Skipped montgomery_cxr.dcm — the Montgomery set is not in data/montgomery "
            "(it is only needed for the CXR benchmark)."
        )
        return None

    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    destination = output_dir / "montgomery_cxr.dcm"

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dataset = FileDataset(str(destination), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()

    # Explicit placeholders: this file describes no person.
    dataset.PatientName = "ANONYMOUS^PLACEHOLDER"
    dataset.PatientID = "SCIFLOW-EXAMPLE-001"
    dataset.StudyDescription = "SciFlow Agent example - derived, non-clinical"
    dataset.SeriesDescription = "Montgomery County CXR set, downscaled, re-wrapped as DICOM"
    dataset.ImageComments = (
        "Demonstration fixture for SciFlow Agent. Pixel data derived from the public "
        "Montgomery County chest X-ray research set. Not a clinical study; the patient "
        "fields are placeholders and describe no individual."
    )

    dataset.Modality = "DX"
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.SamplesPerPixel = 1
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.RescaleSlope = 1
    dataset.RescaleIntercept = 0
    dataset.PixelData = pixels.astype(np.uint8).tobytes()

    dataset.save_as(destination, enforce_file_format=True)
    print(f"Wrote {destination.name}  ({pixels.shape}, DICOM DX, derived from Montgomery)")
    return destination


def main() -> None:
    """Write every obtainable real-data example."""
    fetch_bundled_skimage()
    fetch_downloaded_skimage()
    fetch_pydicom_images()
    build_chest_xray_dicom()
    print("\nDone. See examples/README.md for what each image demonstrates.")


if __name__ == "__main__":
    main()
