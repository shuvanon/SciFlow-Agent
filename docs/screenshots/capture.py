"""Regenerate the README/documentation screenshots from a running app.

Prerequisites (dev machine only — not project dependencies):

    pip install playwright
    playwright install chromium

Usage: start the app (``streamlit run app.py``; install the ``[ml]`` extra for
the deep-learning shots), then:

    python docs/screenshots/capture.py

Shots 01-04 are a **single continuous walkthrough** on the image the app opens
with, so the sequence reads as one analysis from input to report rather than as
a sampler. Shots 05-07 then cover the two things that image cannot show: the
deep-learning tool, which only applies to chest X-rays, and what a few hundred
objects look like.

Requests are read from the example catalogue rather than repeated here, so a
screenshot always shows the request the app itself suggests for that image.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.example_catalogue import DEFAULT_EXAMPLE, EXAMPLE_CATALOGUE  # noqa: E402

APP_URL = "http://localhost:8501"
OUTPUT_DIR = Path(__file__).resolve().parent

#: The walkthrough runs on whatever the app opens with, so shot 01 and the
#: shots that follow it are the same image by construction.
WALKTHROUGH_EXAMPLE = DEFAULT_EXAMPLE
CXR_EXAMPLE = "montgomery_cxr.dcm"
MICROSCOPY_EXAMPLE = "skimage_human_mitosis.png"

#: scroll_into_view_if_needed() leaves the anchor at the bottom of the viewport,
#: so each shot scrolls past it to frame the content that follows the heading.
PLAN_SCROLL = 520
RESULTS_SCROLL = 60


def _request_for(name: str) -> str:
    return EXAMPLE_CATALOGUE[name][1]


def _shot(page: Page, anchor: str, filename: str, extra_scroll: int = 0) -> None:
    page.locator(f"text={anchor}").first.scroll_into_view_if_needed()
    page.wait_for_timeout(500)
    if extra_scroll:
        page.mouse.wheel(0, extra_scroll)
        page.wait_for_timeout(500)
    page.screenshot(path=str(OUTPUT_DIR / filename))
    print(f"captured {filename}")


def _open(page: Page) -> None:
    page.goto(APP_URL, wait_until="domcontentloaded")
    page.wait_for_selector("text=SciFlow Agent", timeout=45000)
    page.wait_for_timeout(2500)


def _select_example(page: Page, name: str) -> None:
    """Drive the Streamlit example-image selectbox to `name`.

    Type to filter rather than clicking an option: the list holds twenty
    entries, so most are scrolled out of the dropdown's viewport and a direct
    click times out.

    The combobox arrives pre-filled with the current selection and typing
    inserts at the caret, so the field must be cleared first — otherwise the
    query becomes nonsense, the dropdown shows "No results", and the old image
    stays selected without any error. The final wait asserts the change landed,
    because that failure is otherwise invisible until the screenshots are
    reviewed.
    """
    combo = page.get_by_role("combobox").first
    combo.click()
    page.wait_for_timeout(300)
    page.keyboard.press("Control+a")
    page.keyboard.type(name, delay=10)
    page.wait_for_timeout(700)
    page.get_by_role("option").first.click()
    page.wait_for_timeout(2000)  # let the new image load
    page.wait_for_selector(f"text={name}", timeout=15000)


def _run_request(page: Page, request: str) -> None:
    textarea = page.locator("textarea").first
    textarea.click()
    textarea.fill(request)
    page.keyboard.press("Control+Enter")
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="Generate plan").click()
    page.wait_for_selector("text=Plan validated", timeout=60000)
    page.wait_for_timeout(800)


def _execute(page: Page, timeout: int = 120000) -> None:
    page.get_by_role("button", name="Execute workflow").click()
    page.wait_for_selector("text=Summary statistics", timeout=timeout)
    page.wait_for_timeout(2500)


def capture_walkthrough(page: Page) -> None:
    """One image, start to finish: input, plan, approval, results, report."""
    _open(page)
    _shot(page, "1. Image and request", "01_input_request.png")

    _run_request(page, _request_for(WALKTHROUGH_EXAMPLE))
    _shot(page, "2. Review the plan", "02_plan_review.png", extra_scroll=PLAN_SCROLL)

    _execute(page)
    _shot(page, "Segmentation mask", "03_results_masks.png", extra_scroll=RESULTS_SCROLL)
    _shot(page, "4. Download report", "04_measurements_report.png")


def capture_deep_learning(page: Page) -> None:
    """The one thing a CT cannot show: the pretrained chest X-ray model."""
    _open(page)
    _select_example(page, CXR_EXAMPLE)
    _run_request(page, _request_for(CXR_EXAMPLE))
    _shot(page, "2. Review the plan", "05_ml_plan_review.png", extra_scroll=PLAN_SCROLL)

    _execute(page, timeout=180000)
    _shot(page, "Segmentation mask", "06_ml_results.png", extra_scroll=RESULTS_SCROLL)


def capture_many_objects(page: Page) -> None:
    """What a few hundred objects look like — the CT segments one."""
    _open(page)
    _select_example(page, MICROSCOPY_EXAMPLE)
    _run_request(page, _request_for(MICROSCOPY_EXAMPLE))
    _execute(page)
    _shot(page, "Segmentation mask", "07_microscopy_results.png", extra_scroll=RESULTS_SCROLL)


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 950})
        capture_walkthrough(page)
        capture_deep_learning(page)
        capture_many_objects(page)
        browser.close()
    print("done")


if __name__ == "__main__":
    main()
