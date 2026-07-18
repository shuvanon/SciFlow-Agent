"""Tests for the central configuration module."""

from __future__ import annotations

import pytest

from src.config import AppConfig, load_config

ALL_VARS = [
    "PLANNER_MODE",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_TIMEOUT_SECONDS",
    "MAX_WORKFLOW_STEPS",
    "MAX_IMAGE_WIDTH",
    "MAX_IMAGE_HEIGHT",
    "LOG_LEVEL",
]


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove every SciFlow variable from the environment."""
    for name in ALL_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_defaults_without_environment(clean_env: pytest.MonkeyPatch) -> None:
    config = load_config(use_dotenv=False)
    assert config.planner_mode == "demo"
    assert config.llm_base_url == ""
    assert config.llm_api_key == ""
    assert config.llm_timeout_seconds == 120
    assert config.max_workflow_steps == 8
    assert config.max_image_width == 4096
    assert config.max_image_height == 4096
    assert config.log_level == "INFO"


def test_environment_overrides(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("PLANNER_MODE", "LLM")
    clean_env.setenv("LLM_BASE_URL", "http://localhost:1234/v1")
    clean_env.setenv("MAX_WORKFLOW_STEPS", "5")
    clean_env.setenv("LLM_TIMEOUT_SECONDS", "180")
    clean_env.setenv("LOG_LEVEL", "debug")

    config = load_config(use_dotenv=False)
    assert config.planner_mode == "llm"  # normalized to lowercase
    assert config.llm_base_url == "http://localhost:1234/v1"
    assert config.max_workflow_steps == 5
    assert config.llm_timeout_seconds == 180
    assert config.log_level == "DEBUG"  # normalized to uppercase


def test_invalid_planner_mode_rejected(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("PLANNER_MODE", "autopilot")
    with pytest.raises(ValueError, match="PLANNER_MODE"):
        load_config(use_dotenv=False)


def test_non_integer_limit_rejected(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("MAX_WORKFLOW_STEPS", "many")
    with pytest.raises(ValueError, match="MAX_WORKFLOW_STEPS"):
        load_config(use_dotenv=False)


def test_non_positive_limit_rejected(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("MAX_IMAGE_WIDTH", "0")
    with pytest.raises(ValueError, match="MAX_IMAGE_WIDTH"):
        load_config(use_dotenv=False)


def test_api_key_excluded_from_repr() -> None:
    config = AppConfig(llm_api_key="super-secret-key")
    assert "super-secret-key" not in repr(config)
    assert "super-secret-key" not in str(config)
