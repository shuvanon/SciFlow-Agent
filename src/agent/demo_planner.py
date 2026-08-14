"""Deterministic demo planner (FR-04, demo mode).

Turns a natural-language request into an :class:`ExecutionPlan` using
keyword and intent rules only — no LLM, no network, no API key. The output
uses the same schema the LLM planner produces and passes the identical
validation path (schemas → plan validator) before execution.

Composition rules (canonical order: convert → denoise → contrast →
segment → clean → measure):

- Unsafe or out-of-scope requests yield ``supported=False`` plans, which the
  validator refuses to execute (Use Case 4).
- Segmentation is implied by cleanup or measurement keywords.
- After any segmentation, mask cleanup and measurement are always included:
  every segmentation use case in the specification ends with cleanup and
  measurement, and counting on an uncleaned mask would count noise specks.
- ``convert_to_grayscale`` is prepended unless the image is known to be
  grayscale already (``channels == 1``).
- Numeric parameters mentioned in the request (median radius, minimum
  object size) are extracted and clamped to the safe ranges, with a plan
  warning whenever clamping changed the requested value.
"""

from __future__ import annotations

import re

from src.agent.schemas import ExecutionPlan, ToolStep
from src.tools import preprocessing, segmentation

SUPPORTED_OPERATIONS_SENTENCE = (
    "Supported operations: noise removal, contrast enhancement, segmentation of "
    "bright or dark objects, removal of small regions, and object counting and "
    "measurement."
)

EXAMPLE_REQUEST = (
    "Remove noise, segment the bright objects, ignore very small regions, and measure them."
)

#: Requests matching any of these are refused outright (reason, pattern).
_UNSAFE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b(?:shell|terminal|cmd|bash|powershell|command(?:\s+line)?|commands)\b"),
        "shell or command execution",
    ),
    (
        re.compile(
            r"\b(?:delete|erase|wipe|destroy)\b|\bremove\s+(?:the\s+)?(?:files?|images?|folders?|director(?:y|ies)|data)\b"
        ),
        "deleting files or data",
    ),
    (
        re.compile(
            r"\b(?:execute|run|eval)\s+(?:some\s+)?(?:code|python|scripts?|programs?)\b|\bexec\s*\(|\beval\s*\("
        ),
        "arbitrary code execution",
    ),
    (
        re.compile(r"\b(?:download|upload|internet|network|email|send)\b|https?://"),
        "network or file-transfer actions",
    ),
    (
        re.compile(r"\b(?:install|pip|conda|npm)\b"),
        "installing software",
    ),
    (
        re.compile(r"\b(?:passwords?|api\s+keys?|secrets?|credentials?|tokens?)\b"),
        "accessing credentials",
    ),
)

#: In-domain but out-of-scope for the MVP (reason, pattern).
_OUT_OF_SCOPE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b(?:3d|three-dimensional|volumetric|z-?stacks?)\b"),
        "3D image processing",
    ),
    (
        # 2D DICOM is a supported input format (FR-01), so the bare word must
        # not be refused. Only NIfTI and multi-slice DICOM stay out of scope —
        # a DICOM *series* is a 3D volume, which the executor cannot process.
        re.compile(r"\bnifti\b|\.nii\b|\bdicom\s+(?:series|stack|volume|study)\b"),
        "NIfTI files or multi-slice DICOM series",
    ),
    (
        # Deep-learning *segmentation* is supported via segment_ml; still reject
        # model training and model frameworks that are not integrated.
        re.compile(
            r"\b(?:train(?:ing)?|fine-?tune|cellpose|"
            r"stardist|nnu-?net|monai|sam|segment\s+anything)\b"
        ),
        "model training or an unsupported model framework",
    ),
)

_DENOISE = re.compile(r"\b(?:de-?noise[a-z]*|noise|noisy|smooth[a-z]*|despeckle|grain[a-z]*)\b")
_CONTRAST = re.compile(r"\b(?:contrast|equalize|equalization|clahe)\b")
_SEGMENT = re.compile(
    r"\b(?:segment[a-z]*|threshold[a-z]*|objects?|cells?|regions?|particles?|blobs?|"
    r"nuclei|nucleus|spots?|detect[a-z]*|find|identify|isolate|bones?|skeletons?|"
    r"calcifications?|structures?|lesions?|tissue)\b"
)
_CLEAN = re.compile(
    r"\b(?:ignore|remove|filter\s+out|discard|drop|exclude)\s+(?:the\s+)?(?:very\s+)?"
    r"(?:small|tiny)\b"
    r"|\bsmall\s+(?:regions?|objects?|particles?|areas?)\b"
    r"|\b(?:specks?|debris|artifacts?|dust)\b"
)
_MEASURE = re.compile(
    r"\b(?:count[a-z]*|measure[a-z]*|sizes?|areas?|quantify|statistics|properties|how\s+many)\b"
)
_FILL_HOLES = re.compile(r"\bfill\s+(?:the\s+)?holes?\b")

#: Route the segmentation step to multi-level thresholding rather than a plain
#: two-class Otsu. These requests name the *extreme* intensity population in an
#: image that has more than two: bone against soft tissue and air in a CT, or an
#: explicit "only the brightest". A single threshold cannot express that — it
#: separates the largest two groups, which on a CT is air from the whole body.
_DENSEST = re.compile(
    r"\b(?:bones?|bony|skeletons?|skeletal|calcifications?|"
    r"densest|dense\s+(?:regions?|structures?|tissue)|"
    r"brightest|darkest|highest\s+intensity|lowest\s+intensity)\b"
)
_DARKEST = re.compile(r"\b(?:darkest|lowest\s+intensity)\b")

#: Intensity classes used for "densest structures" requests. Chosen from a
#: measured comparison on three independent CT slices against Hounsfield-unit
#: ground truth (bone > 300 HU): 4 classes scored a mean Dice of 0.88 against
#: 0.40 for 3 classes and 0.20 for plain Otsu.
DENSEST_CLASSES = 4

#: Route the segmentation step to the deep-learning tool (segment_ml).
_ML = re.compile(
    r"\b(?:deep[- ]?learning|neural(?:\s+network)?|pretrained|torchxrayvision|"
    r"lungs?|chest\s+x-?rays?|cxr|ml\s+(?:model|segmentation)|ai\s+(?:model|segmentation))\b"
)

_DARK_OBJECTS = re.compile(
    r"\bdark(?:er)?\s+(?:objects?|cells?|regions?|particles?|blobs?|spots?|nuclei|areas?)\b"
)
_DARK_BACKGROUND = re.compile(r"\bdark(?:er)?\s+background\b")
_DARK_WORD = re.compile(r"\bdark(?:er)?\b")
_BRIGHT_WORD = re.compile(r"\bbright(?:er)?\b")

_RADIUS = re.compile(r"\bradius\s*(?:of\s*)?[:=]?\s*(\d+)\b")
_MIN_SIZE_PATTERNS = (
    re.compile(r"\b(?:smaller|less)\s+than\s+(\d+)\b"),
    re.compile(r"\b(?:under|below)\s+(\d+)\b"),
    re.compile(r"\bmin(?:imum)?\s+(?:object\s+)?size\s*(?:of\s*)?[:=]?\s*(\d+)\b"),
)


def _unsupported_plan(explanation: str) -> ExecutionPlan:
    return ExecutionPlan(
        goal="unsupported_request",
        supported=False,
        explanation=explanation,
        steps=[],
    )


def _clamp(value: int, low: int, high: int, name: str, warnings: list[str]) -> int:
    if value < low:
        warnings.append(f"Requested {name} {value} is below the minimum of {low}; using {low}.")
        return low
    if value > high:
        warnings.append(f"Requested {name} {value} exceeds the maximum of {high}; using {high}.")
        return high
    return value


def _detect_densest_polarity(text: str) -> str:
    """Which extreme class a 'densest structures' request means.

    Bone and calcification are the bright extreme; only an explicit 'darkest'
    or 'lowest intensity' asks for the other end.
    """
    if _DARKEST.search(text):
        return segmentation.POLARITY_DARK
    return segmentation.POLARITY_BRIGHT


def _detect_polarity(text: str) -> str:
    if _DARK_OBJECTS.search(text):
        return segmentation.POLARITY_DARK
    if _DARK_BACKGROUND.search(text):
        return segmentation.POLARITY_BRIGHT
    if _DARK_WORD.search(text) and not _BRIGHT_WORD.search(text):
        return segmentation.POLARITY_DARK
    return segmentation.POLARITY_BRIGHT


def _extract_radius(text: str, warnings: list[str]) -> int:
    match = _RADIUS.search(text)
    if not match:
        return preprocessing.DEFAULT_MEDIAN_RADIUS
    return _clamp(
        int(match.group(1)),
        preprocessing.MIN_MEDIAN_RADIUS,
        preprocessing.MAX_MEDIAN_RADIUS,
        "median radius",
        warnings,
    )


def _extract_minimum_size(text: str, warnings: list[str]) -> int:
    for pattern in _MIN_SIZE_PATTERNS:
        match = pattern.search(text)
        if match:
            return _clamp(
                int(match.group(1)),
                segmentation.MIN_OBJECT_SIZE_LIMIT,
                segmentation.MAX_OBJECT_SIZE_LIMIT,
                "minimum object size",
                warnings,
            )
    return segmentation.DEFAULT_MINIMUM_OBJECT_SIZE


def _join_phrases(phrases: list[str]) -> str:
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def generate_demo_plan(request: str, *, channels: int | None = None) -> ExecutionPlan:
    """Create a deterministic execution plan from a natural-language request.

    Args:
        request: The user's analysis request.
        channels: Channel count of the input image when known; ``1`` skips
            the grayscale-conversion step.

    Returns:
        An :class:`ExecutionPlan`. Unsafe, out-of-scope, or unrecognized
        requests return a plan with ``supported=False`` and an explanation;
        such plans are refused by the validator and never execute.
    """
    text = " ".join(request.lower().split())
    if not text:
        return _unsupported_plan(
            "The request is empty. Describe the analysis you need, for example: "
            f"'{EXAMPLE_REQUEST}'"
        )

    for pattern, reason in _UNSAFE_RULES:
        if pattern.search(text):
            return _unsupported_plan(
                f"The request appears to ask for {reason}, which is not supported. "
                "Only registered image-analysis tools can run. " + SUPPORTED_OPERATIONS_SENTENCE
            )
    for pattern, reason in _OUT_OF_SCOPE_RULES:
        if pattern.search(text):
            return _unsupported_plan(
                f"This request involves {reason}, which is outside the scope of this "
                "application. " + SUPPORTED_OPERATIONS_SENTENCE
            )

    wants_denoise = bool(_DENOISE.search(text))
    wants_contrast = bool(_CONTRAST.search(text))
    clean_keywords = bool(_CLEAN.search(text))
    measure_keywords = bool(_MEASURE.search(text))
    segment_keywords = bool(_SEGMENT.search(text))
    wants_ml = bool(_ML.search(text))
    wants_densest = bool(_DENSEST.search(text)) and not wants_ml
    wants_segment = (
        segment_keywords or clean_keywords or measure_keywords or wants_ml or wants_densest
    )

    if not (wants_denoise or wants_contrast or wants_segment):
        return _unsupported_plan(
            "The request did not match any supported operation. "
            + SUPPORTED_OPERATIONS_SENTENCE
            + f" Try, for example: '{EXAMPLE_REQUEST}'"
        )

    warnings: list[str] = []
    steps: list[ToolStep] = []
    phrases: list[str] = []
    goal_parts: list[str] = []

    if channels != 1:
        steps.append(ToolStep(tool="convert_to_grayscale"))
        phrases.append("convert the image to grayscale")

    if wants_denoise:
        radius = _extract_radius(text, warnings)
        steps.append(ToolStep(tool="denoise_median", parameters={"radius": radius}))
        phrases.append(f"reduce noise with a median filter (radius {radius})")
        goal_parts.append("denoise")

    if wants_contrast:
        steps.append(ToolStep(tool="enhance_contrast", parameters={}))
        phrases.append("enhance contrast with adaptive histogram equalization")
        goal_parts.append("enhance_contrast")

    if wants_segment:
        if not segment_keywords and not wants_ml:
            warnings.append(
                "Segmentation was added because removing small regions or measuring "
                "requires detected objects."
            )
        if wants_ml:
            steps.append(ToolStep(tool="segment_ml", parameters={"model_name": "cxr_lung"}))
            phrases.append("segment the lungs with a pretrained deep-learning model")
            goal_parts.append("segment_ml")
        elif wants_densest:
            polarity = _detect_densest_polarity(text)
            steps.append(
                ToolStep(
                    tool="segment_threshold",
                    parameters={
                        "method": "multiotsu",
                        "classes": DENSEST_CLASSES,
                        "polarity": polarity,
                    },
                )
            )
            extreme = "brightest" if polarity == segmentation.POLARITY_BRIGHT else "darkest"
            phrases.append(
                f"split the intensities into {DENSEST_CLASSES} classes with multi-level Otsu "
                f"and keep the {extreme} one"
            )
            warnings.append(
                "Multi-level thresholding was chosen because the request names the "
                f"{extreme} structures specifically. A single Otsu threshold separates the two "
                "largest intensity groups, which in a CT slice is air from the whole body."
            )
            goal_parts.append("segment_threshold")
        else:
            polarity = _detect_polarity(text)
            steps.append(ToolStep(tool="segment_otsu", parameters={"polarity": polarity}))
            phrases.append(f"segment {polarity} objects with Otsu thresholding")
            goal_parts.append("segment")

        minimum_size = _extract_minimum_size(text, warnings)
        fill_holes = bool(_FILL_HOLES.search(text))
        steps.append(
            ToolStep(
                tool="clean_mask",
                parameters={"minimum_object_size": minimum_size, "fill_holes": fill_holes},
            )
        )
        clean_phrase = f"remove objects smaller than {minimum_size} pixels"
        if fill_holes:
            clean_phrase += " and fill small holes"
        phrases.append(clean_phrase)

        steps.append(ToolStep(tool="measure_objects", parameters={}))
        phrases.append("count and measure the detected objects")
        goal_parts.append("measure")

    return ExecutionPlan(
        goal="_".join(goal_parts),
        supported=True,
        explanation=f"The workflow will {_join_phrases(phrases)}.",
        steps=steps,
        warnings=warnings,
    )
