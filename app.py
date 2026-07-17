"""SciFlow Agent — Streamlit entry point.

Phase 1 scope: a minimal application shell that loads configuration and
displays the built-in example images. Request input, planning, execution,
and reporting are added in later phases.
"""

from __future__ import annotations

import streamlit as st
from PIL import Image

from src import __version__
from src.config import EXAMPLES_DIR, load_config, setup_logging

st.set_page_config(page_title="SciFlow Agent", page_icon="🔬", layout="wide")


def main() -> None:
    """Render the application page."""
    config = load_config()
    setup_logging(config.log_level)

    st.title("🔬 SciFlow Agent")
    st.caption(
        "Turn plain-language scientific image-analysis requests into validated, "
        "reproducible tool pipelines."
    )

    with st.sidebar:
        st.header("Input")
        example_paths = sorted(EXAMPLES_DIR.glob("*.png"))
        if not example_paths:
            st.error(
                "No example images found. Generate them with "
                "`python examples/generate_examples.py`."
            )
            st.stop()
        selected_name = st.selectbox("Example image", [path.name for path in example_paths])
        st.divider()
        st.subheader("Status")
        st.markdown(f"Planner mode: `{config.planner_mode}`")
        st.markdown(f"Version: `{__version__}`")

    selected_path = EXAMPLES_DIR / selected_name
    image = Image.open(selected_path)

    image_column, details_column = st.columns([2, 1])
    with image_column:
        st.subheader("Original image")
        st.image(image, caption=selected_name, width="stretch")
    with details_column:
        st.subheader("Image details")
        st.markdown(
            f"- **Size:** {image.width} × {image.height} px\n"
            f"- **Mode:** {image.mode}\n"
            f"- **Format:** {image.format}"
        )

    st.info(
        "Phase 1 shell — natural-language requests, plan review, execution, "
        "and reports arrive in the next phases."
    )


main()
