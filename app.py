"""SciFlow Agent — Streamlit user interface (Phase 6: complete workflow).

Flow: select or upload an image → describe the analysis → generate a plan
(demo or LLM mode) → review the validated plan → explicitly approve
execution → inspect results → download the reproducibility report.

The UI holds no analysis logic: it only wires together the planner, the
validator, the executor, and the reporting module. Plans never execute
without the user pressing the execute button (FR-07).
"""

from __future__ import annotations

import hashlib
from datetime import datetime

import numpy as np
import streamlit as st

from src import __version__
from src.agent.planner import generate_plan
from src.agent.schemas import ExecutionPlan, ToolStep
from src.config import EXAMPLES_DIR, AppConfig, load_config, setup_logging
from src.errors import ImageValidationError, PlannerError
from src.example_catalogue import (
    DEFAULT_EXAMPLE,
    EXAMPLE_CATALOGUE,
    GENERIC_PROMPTS,
    example_label,
    ordered_examples,
)
from src.executor import ExecutionResult, execute_plan
from src.image_io import SUPPORTED_EXTENSIONS, load_image_bytes, load_image_file
from src.models import LoadedImage
from src.plan_validator import ValidationResult, validate_plan
from src.reporting import build_report, report_to_json, report_to_markdown
from src.visualization import create_labelled_image, create_overlay, to_display_mask

st.set_page_config(page_title="SciFlow Agent", page_icon="🔬", layout="wide")

_STATE_DEFAULTS: dict[str, object] = {
    "source_id": None,
    "plan": None,
    "validation": None,
    "planner_mode_used": "",
    "execution": None,
}


def _init_state() -> None:
    for key, value in _STATE_DEFAULTS.items():
        st.session_state.setdefault(key, value)


def _clear_plan_state() -> None:
    st.session_state["plan"] = None
    st.session_state["validation"] = None
    st.session_state["planner_mode_used"] = ""
    st.session_state["execution"] = None


def _sidebar_input(config: AppConfig) -> LoadedImage | None:
    """Render the sidebar input controls and return the loaded image."""
    with st.sidebar:
        st.header("Input")
        source = st.radio(
            "Image source",
            options=("example", "upload"),
            format_func=lambda s: "Built-in example" if s == "example" else "Upload image",
            key="input_source",
        )

        loaded: LoadedImage | None = None
        source_id: str | None = None
        try:
            if source == "example":
                example_paths = [
                    path
                    for extension in SUPPORTED_EXTENSIONS
                    for path in EXAMPLES_DIR.glob(f"*{extension}")
                ]
                if not example_paths:
                    st.error(
                        "No example images found. Fetch them with "
                        "`python examples/fetch_example_data.py`."
                    )
                    return None
                options = ordered_examples(example_paths)
                selected = st.selectbox(
                    "Example image",
                    options=options,
                    index=options.index(DEFAULT_EXAMPLE) if DEFAULT_EXAMPLE in options else 0,
                    format_func=example_label,
                    key="example_select",
                )
                loaded = load_image_file(
                    EXAMPLES_DIR / selected,
                    max_width=config.max_image_width,
                    max_height=config.max_image_height,
                )
                source_id = f"example:{selected}"
            else:
                uploaded = st.file_uploader(
                    "Upload a scientific image",
                    type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
                    key="file_upload",
                )
                if uploaded is None:
                    st.caption("PNG, JPG/JPEG, TIFF, or DICOM (.dcm/.dicom).")
                    return None
                data = uploaded.getvalue()
                loaded = load_image_bytes(
                    data,
                    uploaded.name,
                    max_width=config.max_image_width,
                    max_height=config.max_image_height,
                )
                source_id = f"upload:{uploaded.name}:{hashlib.md5(data).hexdigest()}"
        except ImageValidationError as exc:
            st.error(str(exc))
            return None

        # A different image invalidates any existing plan and results.
        if source_id != st.session_state["source_id"]:
            st.session_state["source_id"] = source_id
            _clear_plan_state()

        st.divider()
        st.header("Planner")
        st.radio(
            "Planner mode",
            options=("demo", "llm"),
            index=0 if config.planner_mode == "demo" else 1,
            format_func=lambda m: "Demo — offline rules" if m == "demo" else "LLM — via endpoint",
            key="planner_mode",
        )
        if st.session_state["planner_mode"] == "llm":
            if config.llm_base_url and config.llm_model:
                st.caption(f"Endpoint: `{config.llm_base_url}`")
                st.caption(f"Model: `{config.llm_model}` · Timeout: {config.llm_timeout_seconds}s")
                key_state = "configured" if config.llm_api_key else "none (fine for local servers)"
                st.caption(f"API key: {key_state}")
            else:
                st.warning(
                    "LLM mode is not configured. Set LLM_BASE_URL and LLM_MODEL in `.env` "
                    "(see `.env.example`), or use demo mode."
                )

        with st.expander("Advanced settings"):
            st.markdown(
                f"- Max workflow steps: `{config.max_workflow_steps}`\n"
                f"- Max image size: `{config.max_image_width} × {config.max_image_height}` px\n"
                f"- Version: `{__version__}`"
            )
    return loaded


def _describe_step(step: ToolStep) -> str:
    parameters = step.parameters
    if step.tool == "convert_to_grayscale":
        return "Convert the image to grayscale"
    if step.tool == "denoise_median":
        return f"Apply median denoising with radius {parameters.get('radius', 2)}"
    if step.tool == "enhance_contrast":
        return f"Enhance contrast (CLAHE, clip limit {parameters.get('clip_limit', 0.01)})"
    if step.tool == "segment_otsu":
        return f"Segment {parameters.get('polarity', 'bright')} objects using Otsu thresholding"
    if step.tool == "segment_threshold":
        method = parameters.get("method", "otsu")
        polarity = parameters.get("polarity", "bright")
        extreme = "brightest" if polarity == "bright" else "darkest"
        if method == "multiotsu":
            classes = parameters.get("classes", 3)
            return (
                f"Split the intensities into {classes} classes (multi-level Otsu) "
                f"and keep the {extreme} one"
            )
        return f"Segment the {extreme} regions by {method} thresholding"
    if step.tool == "segment_ml":
        model = parameters.get("model_name", "cxr_lung")
        return f"Segment with a pretrained deep-learning model ({model})"
    if step.tool == "clean_mask":
        text = f"Remove objects smaller than {parameters.get('minimum_object_size', 30)} pixels"
        if parameters.get("fill_holes"):
            text += " and fill small holes"
        return text
    if step.tool == "measure_objects":
        return "Label and measure the detected objects"
    return step.tool


def _generate_plan_action(config: AppConfig, loaded: LoadedImage) -> None:
    request = str(st.session_state.get("request_input", "")).strip()
    mode = str(st.session_state["planner_mode"])
    _clear_plan_state()
    if not request:
        st.error("The request is empty. Describe the analysis you need.")
        return
    spinner_text = (
        "Generating plan with the demo planner…"
        if mode == "demo"
        else f"Asking {config.llm_model or 'the LLM'} — local models can take a few minutes…"
    )
    try:
        with st.spinner(spinner_text):
            plan = generate_plan(request, config=config, metadata=loaded.metadata, mode=mode)
    except PlannerError as exc:
        st.error(str(exc))
        return
    st.session_state["plan"] = plan
    st.session_state["planner_mode_used"] = mode
    st.session_state["validation"] = validate_plan(
        plan,
        channels=loaded.metadata.channels,
        max_steps=config.max_workflow_steps,
    )


def _render_plan_review(loaded: LoadedImage) -> None:
    plan: ExecutionPlan = st.session_state["plan"]
    validation: ValidationResult = st.session_state["validation"]
    display_plan = validation.normalized_plan if validation.valid else plan

    st.subheader("2. Review the plan")
    st.markdown(
        f"**Goal:** `{display_plan.goal}` · **Planner:** `{st.session_state['planner_mode_used']}`"
    )
    st.markdown(display_plan.explanation)
    for warning in display_plan.warnings:
        st.warning(warning)

    steps_text = "\n".join(
        f"{index}. {_describe_step(step)}" for index, step in enumerate(display_plan.steps, start=1)
    )
    if steps_text:
        st.markdown(steps_text)
    with st.expander("Raw plan (JSON)"):
        st.json(display_plan.model_dump())

    if validation.valid:
        st.success(
            f"Plan validated: {len(display_plan.steps)} steps, all tools approved, "
            "all parameters within safe ranges."
        )
        execute_col, discard_col = st.columns([1, 1])
        with execute_col:
            if st.button("▶ Execute workflow", type="primary", key="execute_plan"):
                with st.spinner("Executing approved workflow…"):
                    st.session_state["execution"] = execute_plan(
                        validation.normalized_plan, loaded.original
                    )
        with discard_col:
            if st.button("✖ Discard plan", key="discard_plan"):
                _clear_plan_state()
                st.rerun()
        st.caption(
            "Nothing runs until you press Execute. To change the request, edit it above "
            "and generate a new plan."
        )
    else:
        st.error("The plan was rejected by validation and cannot execute:")
        for error in validation.errors:
            st.markdown(f"- {error}")


def _last_processed_image(execution: ExecutionResult) -> np.ndarray | None:
    latest: np.ndarray | None = None
    for array in execution.images.values():
        if array.dtype != np.bool_ and array.ndim == 2:
            latest = array
    return latest


def _render_download_section(loaded: LoadedImage, execution: ExecutionResult) -> None:
    """Offer JSON and Markdown reports — for failed executions as well."""
    st.subheader("4. Download report")
    report = build_report(
        metadata=loaded.metadata,
        request=str(st.session_state.get("request_input", "")).strip(),
        planner_mode=st.session_state["planner_mode_used"],
        plan=st.session_state["validation"].normalized_plan,
        execution=execution,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    download_columns = st.columns([1, 1, 2])
    with download_columns[0]:
        st.download_button(
            "⬇ JSON report",
            data=report_to_json(report),
            file_name=f"sciflow_report_{stamp}.json",
            mime="application/json",
            key="download_json",
        )
    with download_columns[1]:
        st.download_button(
            "⬇ Markdown report",
            data=report_to_markdown(report),
            file_name=f"sciflow_report_{stamp}.md",
            mime="text/markdown",
            key="download_markdown",
        )


def _render_results(loaded: LoadedImage, execution: ExecutionResult) -> None:
    st.subheader("3. Results")

    for warning in execution.warnings:
        st.warning(warning)
    if not execution.success:
        for error in execution.errors:
            st.error(error)
        if execution.steps:
            st.caption(f"{len(execution.steps)} step(s) completed before the failure.")
        if execution.images:
            with st.expander("Intermediate images before the failure"):
                for key, array in execution.images.items():
                    display = to_display_mask(array) if array.dtype == np.bool_ else array
                    st.image(display, caption=key, width=280)
        _render_download_section(loaded, execution)
        return

    processed = _last_processed_image(execution)
    first_row = st.columns(2)
    with first_row[0]:
        st.markdown("**Original image**")
        st.image(loaded.original, width="stretch")
    with first_row[1]:
        st.markdown("**Processed image**")
        st.image(processed if processed is not None else loaded.original, width="stretch")

    if execution.mask is not None:
        second_row = st.columns(3)
        with second_row[0]:
            st.markdown("**Segmentation mask**")
            st.image(to_display_mask(execution.mask), width="stretch")
        with second_row[1]:
            st.markdown("**Overlay**")
            st.image(create_overlay(loaded.original, execution.mask), width="stretch")
        with second_row[2]:
            st.markdown("**Labelled objects**")
            if execution.labels is not None:
                st.image(create_labelled_image(execution.labels, processed), width="stretch")
            else:
                st.caption("No labels available.")

    with st.expander("Intermediate images (per step)"):
        keys = list(execution.images)
        for start in range(0, len(keys), 4):
            columns = st.columns(4)
            for column, key in zip(columns, keys[start : start + 4], strict=False):
                array = execution.images[key]
                display = to_display_mask(array) if array.dtype == np.bool_ else array
                with column:
                    st.image(display, caption=key, width="stretch")

    if execution.summary is not None:
        summary = execution.summary
        st.markdown("**Summary statistics**")
        row1 = st.columns(4)
        row1[0].metric("Objects", summary.object_count)
        row1[1].metric("Mean area", f"{summary.mean_area:.1f} px")
        row1[2].metric("Median area", f"{summary.median_area:.1f} px")
        row1[3].metric("Area range", f"{summary.minimum_area:.0f}–{summary.maximum_area:.0f} px")
        row2 = st.columns(4)
        row2[0].metric("Segmented area", f"{summary.total_segmented_area:.0f} px")
        row2[1].metric("Segmented %", f"{summary.segmented_area_percent:.2f}%")
        row2[2].metric("Steps", len(execution.steps))
        row2[3].metric("Runtime", f"{execution.total_runtime_seconds:.2f} s")

    if execution.measurements is not None:
        st.markdown("**Per-object measurements**")
        if execution.measurements.empty:
            st.info("No objects detected.")
        else:
            st.dataframe(execution.measurements.round(2), width="stretch", height=300)

    with st.expander("Step timing"):
        st.table(
            [
                {
                    "step": index,
                    "tool": step.tool,
                    "runtime (ms)": round(step.runtime_seconds * 1000, 1),
                    "warnings": "; ".join(step.warnings) or "—",
                }
                for index, step in enumerate(execution.steps, start=1)
            ]
        )

    _render_download_section(loaded, execution)


def main() -> None:
    """Render the application."""
    config = load_config()
    setup_logging(config.log_level)
    _init_state()

    st.title("🔬 SciFlow Agent")
    st.caption(
        "Describe a scientific image-analysis task in plain language. SciFlow Agent plans it "
        "with approved tools only, shows you the workflow, and runs it after your approval."
    )

    loaded = _sidebar_input(config)
    if loaded is None:
        st.info("Select a built-in example or upload an image in the sidebar to begin.")
        return

    preview_column, details_column = st.columns([2, 1])
    with preview_column:
        st.subheader("1. Image and request")
        st.image(loaded.original, caption=loaded.metadata.filename, width="stretch")
    with details_column:
        st.markdown("**Image details**")
        metadata = loaded.metadata
        kind = "grayscale" if metadata.channels == 1 else "RGB"
        st.markdown(
            f"- Size: {metadata.width} × {metadata.height} px\n"
            f"- Channels: {metadata.channels} ({kind})\n"
            f"- Original mode: `{metadata.mode}` (`{metadata.dtype}`)\n"
            f"- Intensity range: {metadata.minimum_intensity:g}–{metadata.maximum_intensity:g}"
        )
        catalogue_entry = EXAMPLE_CATALOGUE.get(metadata.filename)
        if catalogue_entry is not None:
            st.markdown("**Suggested request for this image**")
            st.info(catalogue_entry[1])
        else:
            st.markdown("**Example requests**")
            for prompt in GENERIC_PROMPTS:
                st.caption(f"· {prompt}")

    suggested = catalogue_entry[1] if catalogue_entry is not None else GENERIC_PROMPTS[0]
    st.text_area(
        "What should be analyzed?",
        placeholder=suggested,
        key="request_input",
        height=90,
    )
    if st.button("Generate plan", type="primary", key="generate_plan"):
        _generate_plan_action(config, loaded)

    if st.session_state["plan"] is not None:
        st.divider()
        _render_plan_review(loaded)

    if st.session_state["execution"] is not None:
        st.divider()
        _render_results(loaded, st.session_state["execution"])


main()
