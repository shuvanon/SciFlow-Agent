"""Prompt construction for the LLM planner (spec section 12.2).

The system prompt is assembled from the tool registry, so tool names,
parameter ranges, and defaults can never drift from the implementation.
Prompting is guidance only, never a security boundary: every response is
parsed and validated exactly like any other untrusted input.

The prompt is deliberately compact with explicit examples, because small
local models follow short, example-driven instructions more reliably
(spec Risk 5).
"""

from __future__ import annotations

import json

from src.models import ImageMetadata
from src.tool_registry import TOOL_REGISTRY

_EXAMPLE_SUPPORTED = {
    "goal": "denoise_segment_measure",
    "supported": True,
    "explanation": (
        "The image will be denoised, segmented with Otsu thresholding, cleaned, and measured."
    ),
    "steps": [
        {"tool": "convert_to_grayscale", "parameters": {}},
        {"tool": "denoise_median", "parameters": {"radius": 2}},
        {"tool": "segment_otsu", "parameters": {"polarity": "bright"}},
        {"tool": "clean_mask", "parameters": {"minimum_object_size": 40, "fill_holes": False}},
        {"tool": "measure_objects", "parameters": {}},
    ],
    "warnings": [],
}

_EXAMPLE_UNSUPPORTED = {
    "goal": "unsupported_request",
    "supported": False,
    "explanation": (
        "Only registered image-analysis tools are available; shell commands cannot be run."
    ),
    "steps": [],
    "warnings": [],
}


def _tool_lines() -> str:
    return "\n".join(
        f"- {name}: {definition.description}" for name, definition in TOOL_REGISTRY.items()
    )


def build_system_prompt(max_steps: int = 8) -> str:
    """Build the planner system prompt from the tool registry."""
    return f"""You are the planning component of SciFlow Agent, a scientific image-analysis \
application. Convert the user's request about a 2D image into a JSON execution plan.

APPROVED TOOLS (the only tools that exist):
{_tool_lines()}

OUTPUT FORMAT — respond with a single JSON object and nothing else (no markdown, no \
commentary):
{{"goal": "<short_snake_case_goal>", "supported": true|false, "explanation": "<one or two \
sentences for the user>", "steps": [{{"tool": "<tool_name>", "parameters": {{}}}}], \
"warnings": []}}

RULES:
1. Use only the approved tools. Never invent tool names or parameters.
2. Step order: convert_to_grayscale (only if the image is RGB) -> denoise_median -> \
enhance_contrast -> segment_otsu -> clean_mask -> measure_objects. Include only the steps \
the request needs.
3. segment_otsu requires a grayscale image. clean_mask and measure_objects require a \
segmentation step (segment_otsu OR segment_ml) earlier in the plan.
4. When the user wants objects counted or measured, include a segmentation step, clean_mask, \
and measure_objects. Use segment_ml (deep learning) when the request asks for it or names \
lungs / chest X-ray; otherwise use segment_otsu.
5. Use at most {max_steps} steps.
6. If the request asks for anything outside these tools (code execution, file or shell \
operations, network access, 3D or DICOM data, model training), set "supported": false with \
an empty "steps" list and explain briefly.
7. Keep parameters within the documented ranges; omit a parameter to use its default.

EXAMPLE 1 — request: "Remove noise, segment the bright objects, ignore very small regions, \
and measure them." (RGB image):
{json.dumps(_EXAMPLE_SUPPORTED)}

EXAMPLE 2 — request: "Run a shell command and delete the image.":
{json.dumps(_EXAMPLE_UNSUPPORTED)}"""


def build_user_prompt(request: str, metadata: ImageMetadata | None = None) -> str:
    """Build the user message: image metadata context plus the request."""
    lines: list[str] = []
    if metadata is not None:
        channel_note = (
            "1 channel (already grayscale)"
            if metadata.channels == 1
            else f"{metadata.channels} channels (RGB color)"
        )
        lines.append(
            f"Image: filename='{metadata.filename}', {metadata.width}x{metadata.height} px, "
            f"{channel_note}, original mode '{metadata.mode}', intensity range "
            f"[{metadata.minimum_intensity:g}, {metadata.maximum_intensity:g}]."
        )
        if metadata.channels == 1:
            lines.append("The image is already grayscale, so convert_to_grayscale is not needed.")
        else:
            lines.append("The image is RGB, so start with convert_to_grayscale.")
    lines.append(f'Request: "{request}"')
    lines.append("Respond with the JSON plan only.")
    return "\n".join(lines)


def build_repair_prompt(error_summary: str) -> str:
    """Build the retry message sent after an invalid response."""
    return (
        f"Your previous response was not a valid plan: {error_summary} "
        "Respond again with ONLY the corrected JSON object — no markdown, no commentary."
    )
