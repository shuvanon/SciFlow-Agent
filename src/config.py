"""Central application configuration.

All runtime settings are read from environment variables (optionally loaded
from a local ``.env`` file) with safe defaults, so the application starts with
zero configuration in demo mode. Secrets stay in the environment; the API key
is excluded from ``repr()`` so it can never leak through logs or reports.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
REPORTS_DIR = PROJECT_ROOT / "reports"

PLANNER_MODE_DEMO = "demo"
PLANNER_MODE_LLM = "llm"
VALID_PLANNER_MODES = (PLANNER_MODE_DEMO, PLANNER_MODE_LLM)


@dataclass(frozen=True)
class AppConfig:
    """Immutable snapshot of all application settings."""

    planner_mode: str = PLANNER_MODE_DEMO
    llm_base_url: str = ""
    llm_api_key: str = field(default="", repr=False)  # excluded from repr: never log secrets
    llm_model: str = ""
    llm_timeout_seconds: int = 120  # generous: local models can be slow
    max_workflow_steps: int = 8
    max_image_width: int = 4096
    max_image_height: int = 4096
    log_level: str = "INFO"


def _read_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def _read_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"Environment variable {name} must be positive, got {value}.")
    return value


def load_config(*, use_dotenv: bool = True) -> AppConfig:
    """Build an :class:`AppConfig` from the current environment.

    Args:
        use_dotenv: When true, load a local ``.env`` file first (existing
            environment variables are never overridden). Tests disable this
            to stay independent of developer machines.

    Raises:
        ValueError: If a variable holds an invalid value.
    """
    if use_dotenv:
        load_dotenv()

    planner_mode = _read_str("PLANNER_MODE", PLANNER_MODE_DEMO).lower()
    if planner_mode not in VALID_PLANNER_MODES:
        raise ValueError(
            f"PLANNER_MODE must be one of {VALID_PLANNER_MODES}, got {planner_mode!r}."
        )

    return AppConfig(
        planner_mode=planner_mode,
        llm_base_url=_read_str("LLM_BASE_URL", ""),
        llm_api_key=_read_str("LLM_API_KEY", ""),
        llm_model=_read_str("LLM_MODEL", ""),
        llm_timeout_seconds=_read_int("LLM_TIMEOUT_SECONDS", 120),
        max_workflow_steps=_read_int("MAX_WORKFLOW_STEPS", 8),
        max_image_width=_read_int("MAX_IMAGE_WIDTH", 4096),
        max_image_height=_read_int("MAX_IMAGE_HEIGHT", 4096),
        log_level=_read_str("LOG_LEVEL", "INFO").upper(),
    )


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging with a concise, secret-free format."""
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
