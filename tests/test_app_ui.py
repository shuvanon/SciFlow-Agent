"""UI flow tests using Streamlit's AppTest harness.

These drive the real app.py script: select the example image, enter a
request, generate a plan in demo mode, approve execution, and check the
results state — the full browser workflow without a browser.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

# Absolute path so AppTest works regardless of Streamlit's relative-path
# resolution (older versions resolved against the CWD, newer ones against the
# calling test file) and regardless of the pytest working directory.
APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture()
def app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    # Force demo mode regardless of the developer's local .env.
    monkeypatch.setenv("PLANNER_MODE", "demo")
    test_app = AppTest.from_file(APP_PATH, default_timeout=120)
    test_app.run()
    assert not test_app.exception
    return test_app


def test_app_starts_with_example_image(app: AppTest) -> None:
    assert app.session_state["source_id"] == "example:example_cells.png"
    assert app.session_state["plan"] is None


def test_full_demo_workflow(app: AppTest) -> None:
    app.text_area(key="request_input").input(
        "Remove noise, segment the bright objects, ignore very small regions, and measure them."
    ).run()
    app.button(key="generate_plan").click().run()
    assert not app.exception

    assert app.session_state["plan"] is not None
    validation = app.session_state["validation"]
    assert validation is not None and validation.valid
    # Plan is displayed for review; execution has NOT happened yet (FR-07).
    assert app.session_state["execution"] is None

    app.button(key="execute_plan").click().run()
    assert not app.exception

    execution = app.session_state["execution"]
    assert execution is not None
    assert execution.success
    assert execution.summary is not None
    assert execution.summary.object_count >= 10


def test_unsupported_request_shows_rejection_and_blocks_execution(app: AppTest) -> None:
    app.text_area(key="request_input").input(
        "Run a shell command and delete the image after processing."
    ).run()
    app.button(key="generate_plan").click().run()
    assert not app.exception

    validation = app.session_state["validation"]
    assert validation is not None and not validation.valid
    assert app.session_state["execution"] is None
    # The execute button must not exist for an invalid plan.
    assert not any(button.key == "execute_plan" for button in app.button)


def test_empty_request_shows_error(app: AppTest) -> None:
    app.button(key="generate_plan").click().run()
    assert not app.exception
    assert app.session_state["plan"] is None
    assert any("empty" in str(error.value).lower() for error in app.error)


def test_discard_plan_clears_state(app: AppTest) -> None:
    app.text_area(key="request_input").input("Count the bright objects.").run()
    app.button(key="generate_plan").click().run()
    assert app.session_state["plan"] is not None

    app.button(key="discard_plan").click().run()
    assert not app.exception
    assert app.session_state["plan"] is None
    assert app.session_state["execution"] is None


def test_unconfigured_llm_mode_shows_clear_error(app: AppTest, monkeypatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "")
    monkeypatch.setenv("LLM_MODEL", "")
    app.radio(key="planner_mode").set_value("llm").run()
    app.text_area(key="request_input").input("Count the cells.").run()
    app.button(key="generate_plan").click().run()
    assert not app.exception

    assert app.session_state["plan"] is None
    assert any("not configured" in str(error.value) for error in app.error)


def test_switching_example_image_resets_plan(app: AppTest) -> None:
    app.text_area(key="request_input").input("Count the bright objects.").run()
    app.button(key="generate_plan").click().run()
    assert app.session_state["plan"] is not None

    app.selectbox(key="example_select").set_value("example_objects.png").run()
    assert not app.exception
    assert app.session_state["source_id"] == "example:example_objects.png"
    assert app.session_state["plan"] is None  # stale plan invalidated
