"""Visualization helpers: display-ready masks, overlays, and labelled images.

These functions prepare uint8 images for the UI and reports. They never
change measurement results — visualization stays strictly separate from
measurement logic.
"""

from __future__ import annotations

import numpy as np
from skimage import color, util

from src.errors import ToolInputError

DEFAULT_OVERLAY_COLOR = (255, 70, 70)
DEFAULT_OVERLAY_ALPHA = 0.45


def _require_mask(mask: np.ndarray) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ToolInputError(
            f"Expected a 2D binary mask; got shape {getattr(mask, 'shape', None)}."
        )
    return mask.astype(bool)


def to_display_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a boolean mask into a displayable uint8 image (0 or 255)."""
    return _require_mask(mask).astype(np.uint8) * 255


def _as_rgb(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ToolInputError(
            f"Expected a 2D grayscale or (H, W, 3) RGB image; "
            f"got shape {getattr(image, 'shape', None)}."
        )
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.shape[2] != 3:
        raise ToolInputError(f"Expected 3 color channels, got shape {image.shape}.")
    return image.copy()


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    overlay_color: tuple[int, int, int] = DEFAULT_OVERLAY_COLOR,
    alpha: float = DEFAULT_OVERLAY_ALPHA,
) -> np.ndarray:
    """Blend a translucent color over the mask region of an image.

    Args:
        image: 2D grayscale or (H, W, 3) RGB uint8 image.
        mask: 2D binary mask of the same height and width.
        overlay_color: RGB color painted over segmented pixels.
        alpha: Blend strength in [0, 1]; 0 shows only the image, 1 only the
            color.
    """
    boolean_mask = _require_mask(mask)
    rgb = _as_rgb(image).astype(np.float64)
    if boolean_mask.shape != rgb.shape[:2]:
        raise ToolInputError(
            f"Mask shape {boolean_mask.shape} does not match image shape {rgb.shape[:2]}."
        )
    if not 0.0 <= alpha <= 1.0:
        raise ToolInputError(f"alpha must be between 0 and 1, got {alpha}.")

    color_layer = np.asarray(overlay_color, dtype=np.float64)
    rgb[boolean_mask] = (1.0 - alpha) * rgb[boolean_mask] + alpha * color_layer
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def create_labelled_image(labels: np.ndarray, image: np.ndarray | None = None) -> np.ndarray:
    """Render labelled objects in distinct colors, optionally over an image.

    Args:
        labels: 2D integer label image (0 is background).
        image: Optional grayscale or RGB uint8 image used as the backdrop.
    """
    if not isinstance(labels, np.ndarray) or labels.ndim != 2:
        raise ToolInputError(
            f"Expected a 2D label image; got shape {getattr(labels, 'shape', None)}."
        )
    backdrop = None
    if image is not None:
        backdrop = _as_rgb(image)
        if backdrop.shape[:2] != labels.shape:
            raise ToolInputError(
                f"Image shape {backdrop.shape[:2]} does not match labels shape {labels.shape}."
            )
    rendered = color.label2rgb(labels, image=backdrop, bg_label=0)  # float in [0, 1]
    return util.img_as_ubyte(rendered)
