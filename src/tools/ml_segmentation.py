"""Deep-learning segmentation tool (``segment_ml``).

Wraps a pretrained model as a registered, validated tool that fits the exact
``image -> mask`` slot used by ``segment_otsu`` — so the planner, validator,
and executor treat it like any other approved tool.

Design choices that keep this aligned with the project's safety model:

- **Optional dependency.** torch and torchxrayvision are imported lazily
  inside the tool, never at module import. The base app, demo mode, and the
  registry work with the ``[ml]`` extra absent; only *executing* this tool
  needs it, and then it fails with a clear, actionable message.
- **Fixed model registry.** ``model_name`` is a whitelist; weights come from
  the model library's pinned release, never from plan/LLM content.
- **Device auto-detect.** Uses CUDA when available, else CPU (the "no GPU
  required" guarantee holds — GPU only makes it faster).
- **Model-agnostic seam.** Adding another model (MONAI, Cellpose, …) later
  means one new entry in ``SUPPORTED_MODELS`` plus a loader branch; the
  registry, schema, and executor are untouched.

Currently supported model: ``cxr_lung`` — chest X-ray lung-field segmentation
via torchxrayvision's PSPNet (targets "Left Lung" and "Right Lung").
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

from src.errors import ToolInputError

DEFAULT_MODEL = "cxr_lung"
MIN_THRESHOLD = 0.05
MAX_THRESHOLD = 0.95
DEFAULT_THRESHOLD = 0.5

_MODEL_INPUT_SIZE = 512

#: Whitelist of approved models. ``builder`` returns a ready model given the
#: lazily imported torchxrayvision module; ``targets`` are the class names whose
#: probability maps are unioned into the output mask.
SUPPORTED_MODELS: dict[str, dict[str, Any]] = {
    "cxr_lung": {
        "display_name": "Chest X-ray lung segmentation (torchxrayvision PSPNet)",
        "framework": "torchxrayvision",
        "targets": ("Left Lung", "Right Lung"),
        "builder": lambda xrv: xrv.baseline_models.chestx_det.PSPNet(),
    },
}

#: Cache of loaded models keyed by (model_name, device). Loading weights is
#: expensive, so a workflow reuses one instance.
_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _import_backend() -> tuple[Any, Any]:
    """Import the optional ML backend or raise a clear, actionable error."""
    try:
        import torch
        import torchxrayvision as xrv
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ToolInputError(
            "segment_ml needs the optional ML dependencies, which are not installed. "
            "Install them with `pip install -e .[ml]` (or `pip install -r requirements-ml.txt`)."
        ) from exc
    return torch, xrv


def _get_model(model_name: str, torch: Any, xrv: Any) -> tuple[Any, str]:
    """Load (and cache) the requested model, returning (model, device)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (model_name, device)
    if key not in _MODEL_CACHE:
        model = SUPPORTED_MODELS[model_name]["builder"](xrv)
        _MODEL_CACHE[key] = model.to(device).eval()
    return _MODEL_CACHE[key], device


def _to_grayscale_float(image: np.ndarray) -> np.ndarray:
    """Return a 2D float image in [0, 255] from a grayscale or RGB uint8 image."""
    if image.ndim == 3:
        return rgb2gray(image) * 255.0  # rgb2gray -> [0, 1]
    return image.astype(np.float32)


def _predict_lung_probability(
    image: np.ndarray, model_name: str, torch: Any, xrv: Any
) -> np.ndarray:
    """Run the model and return a lung-probability map at the input resolution.

    Isolated so unit tests can substitute a deterministic map without torch.
    """
    model, device = _get_model(model_name, torch, xrv)
    target_names = SUPPORTED_MODELS[model_name]["targets"]
    indices = [model.targets.index(name) for name in target_names]

    gray = _to_grayscale_float(image)
    normalized = xrv.datasets.normalize(gray.astype(np.float32), 255)  # -> [-1024, 1024]
    small = resize(
        normalized,
        (_MODEL_INPUT_SIZE, _MODEL_INPUT_SIZE),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    )
    tensor = torch.from_numpy(np.ascontiguousarray(small)).float()[None, None].to(device)
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor))[0]
    union = probability[indices[0]]
    for index in indices[1:]:
        union = torch.maximum(union, probability[index])
    prob_small = union.cpu().numpy()
    return resize(prob_small, image.shape[:2], order=1, preserve_range=True, anti_aliasing=True)


def segment_ml(
    image: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """Segment an image with a pretrained deep-learning model.

    Args:
        image: 2D grayscale or (H, W, 3) RGB uint8 image.
        model_name: Approved model key (see ``SUPPORTED_MODELS``).
        threshold: Probability cutoff for the foreground mask (0.05-0.95).

    Returns:
        A 2D boolean mask at the input resolution.

    Raises:
        ToolInputError: If the image or parameters are invalid, the model is
            not approved, or the optional ML dependencies are not installed.
    """
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ToolInputError(
            f"segment_ml requires a 2D or RGB image; got shape {getattr(image, 'shape', None)}."
        )
    if model_name not in SUPPORTED_MODELS:
        approved = ", ".join(sorted(SUPPORTED_MODELS))
        raise ToolInputError(f"segment_ml: unknown model {model_name!r}. Approved: {approved}.")
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise ToolInputError(f"segment_ml: threshold must be a number, got {threshold!r}.")
    if not MIN_THRESHOLD <= threshold <= MAX_THRESHOLD:
        raise ToolInputError(
            f"segment_ml: threshold must be between {MIN_THRESHOLD} and {MAX_THRESHOLD}, "
            f"got {threshold}."
        )

    torch, xrv = _import_backend()
    probability = _predict_lung_probability(image, model_name, torch, xrv)
    return probability >= float(threshold)


_WEIGHTS_HASH_CACHE: dict[str, str] = {}


def _weights_sha256(model: Any) -> str:
    """SHA-256 of the model's local weights file, cached per path ("" if unknown)."""
    path = getattr(model, "weights_filename_local", "")
    if not path or not Path(path).is_file():
        return ""
    key = str(path)
    if key not in _WEIGHTS_HASH_CACHE:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        _WEIGHTS_HASH_CACHE[key] = digest.hexdigest()
    return _WEIGHTS_HASH_CACHE[key]


def model_metadata(model_name: str) -> dict[str, Any]:
    """Provenance for the reproducibility report: model, versions, device, weights hash.

    Requires the ML backend (only called after the tool has run successfully).
    """
    if model_name not in SUPPORTED_MODELS:
        return {"model_name": model_name}
    torch, xrv = _import_backend()
    model, device = _get_model(model_name, torch, xrv)
    info = SUPPORTED_MODELS[model_name]
    return {
        "model_name": model_name,
        "display_name": info["display_name"],
        "framework": info["framework"],
        "framework_version": getattr(xrv, "__version__", "unknown"),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "device": device,
        "weights_sha256": _weights_sha256(model),
    }
