"""Tests for the LLM planner: parsing, retry, error handling, secret safety.

No test touches the network — every endpoint interaction goes through an
injected ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from src.agent.planner import generate_llm_plan, generate_plan
from src.agent.schemas import ExecutionPlan
from src.config import AppConfig
from src.errors import PlannerError
from src.models import ImageMetadata
from src.plan_validator import validate_plan

VALID_PLAN_DICT = {
    "goal": "denoise_segment_measure",
    "supported": True,
    "explanation": "Denoise, segment, clean, and measure the image.",
    "steps": [
        {"tool": "convert_to_grayscale", "parameters": {}},
        {"tool": "denoise_median", "parameters": {"radius": 2}},
        {"tool": "segment_otsu", "parameters": {"polarity": "bright"}},
        {"tool": "clean_mask", "parameters": {"minimum_object_size": 40}},
        {"tool": "measure_objects", "parameters": {}},
    ],
    "warnings": [],
}


def _config(**overrides: Any) -> AppConfig:
    settings: dict[str, Any] = {
        "llm_base_url": "http://llm.test/v1",
        "llm_api_key": "",
        "llm_model": "test-model",
        "llm_timeout_seconds": 30,
    }
    settings.update(overrides)
    return AppConfig(**settings)


def _metadata(channels: int = 3) -> ImageMetadata:
    return ImageMetadata(
        filename="sample.png",
        width=64,
        height=48,
        channels=channels,
        mode="RGB" if channels == 3 else "L",
        dtype="uint8",
        minimum_intensity=0,
        maximum_intensity=255,
    )


def _completion(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _transport_returning(*contents: str) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """Mock transport yielding each content once, recording every request."""
    seen: list[httpx.Request] = []
    responses = list(contents)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        content = responses.pop(0) if responses else responses_exhausted()
        return httpx.Response(200, json=_completion(content))

    def responses_exhausted() -> str:
        raise AssertionError("More requests sent than responses configured.")

    return httpx.MockTransport(handler), seen


def test_valid_response_produces_plan() -> None:
    transport, seen = _transport_returning(json.dumps(VALID_PLAN_DICT))

    plan = generate_llm_plan(
        "Remove noise and measure the objects.",
        config=_config(),
        metadata=_metadata(),
        transport=transport,
    )

    assert isinstance(plan, ExecutionPlan)
    assert [step.tool for step in plan.steps] == [
        "convert_to_grayscale",
        "denoise_median",
        "segment_otsu",
        "clean_mask",
        "measure_objects",
    ]
    assert len(seen) == 1
    assert validate_plan(plan, channels=3).valid


def test_request_payload_and_prompt_content() -> None:
    transport, seen = _transport_returning(json.dumps(VALID_PLAN_DICT))

    generate_llm_plan(
        "Count the cells.", config=_config(), metadata=_metadata(channels=1), transport=transport
    )

    request = seen[0]
    assert request.url.path.endswith("/v1/chat/completions")
    payload = json.loads(request.content)
    assert payload["model"] == "test-model"
    assert payload["temperature"] == 0
    system = payload["messages"][0]["content"]
    user = payload["messages"][1]["content"]
    assert "convert_to_grayscale" in system
    assert "supported" in system
    assert "already grayscale" in user  # metadata reached the prompt
    assert "Count the cells." in user


def test_no_authorization_header_without_key() -> None:
    transport, seen = _transport_returning(json.dumps(VALID_PLAN_DICT))

    generate_llm_plan("Count cells.", config=_config(llm_api_key=""), transport=transport)

    assert "authorization" not in seen[0].headers


def test_authorization_header_present_with_key() -> None:
    transport, seen = _transport_returning(json.dumps(VALID_PLAN_DICT))

    generate_llm_plan("Count cells.", config=_config(llm_api_key="sk-test"), transport=transport)

    assert seen[0].headers["authorization"] == "Bearer sk-test"


def test_markdown_fenced_response_is_parsed() -> None:
    fenced = f"```json\n{json.dumps(VALID_PLAN_DICT)}\n```"
    transport, _ = _transport_returning(fenced)

    plan = generate_llm_plan("Count cells.", config=_config(), transport=transport)

    assert plan.supported


def test_response_with_surrounding_prose_is_parsed() -> None:
    chatty = f"Sure! Here is the plan you asked for:\n{json.dumps(VALID_PLAN_DICT)}\nHope it helps."
    transport, _ = _transport_returning(chatty)

    plan = generate_llm_plan("Count cells.", config=_config(), transport=transport)

    assert plan.supported


def test_malformed_then_valid_response_retries_once() -> None:
    transport, seen = _transport_returning("this is not json", json.dumps(VALID_PLAN_DICT))

    plan = generate_llm_plan("Count cells.", config=_config(), transport=transport)

    assert plan.supported
    assert len(seen) == 2
    retry_payload = json.loads(seen[1].content)
    roles = [message["role"] for message in retry_payload["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert "not a valid plan" in retry_payload["messages"][-1]["content"]


def test_persistently_malformed_response_raises_planner_error() -> None:
    transport, seen = _transport_returning("nope", "still nope")

    with pytest.raises(PlannerError, match="demo mode"):
        generate_llm_plan("Count cells.", config=_config(), transport=transport)

    assert len(seen) == 2  # exactly one retry, never more


def test_unknown_tool_in_response_is_rejected() -> None:
    bad_plan = dict(VALID_PLAN_DICT, steps=[{"tool": "run_shell", "parameters": {}}])
    transport, _ = _transport_returning(json.dumps(bad_plan), json.dumps(bad_plan))

    with pytest.raises(PlannerError, match="did not match the plan schema"):
        generate_llm_plan("Count cells.", config=_config(), transport=transport)


def test_out_of_range_parameter_in_response_is_rejected() -> None:
    bad_plan = dict(
        VALID_PLAN_DICT,
        steps=[
            {"tool": "convert_to_grayscale", "parameters": {}},
            {"tool": "denoise_median", "parameters": {"radius": 50}},
        ],
    )
    transport, _ = _transport_returning(json.dumps(bad_plan))
    # Schema-level parse succeeds (parameters stay a dict), but the shared
    # validator must reject the out-of-range radius before execution.
    plan = generate_llm_plan("Denoise.", config=_config(), transport=transport)

    result = validate_plan(plan, channels=3)
    assert not result.valid
    assert any("radius" in error for error in result.errors)


def test_connection_error_gives_clear_guidance() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(PlannerError, match="Could not connect"):
        generate_llm_plan("Count cells.", config=_config(), transport=httpx.MockTransport(handler))


def test_timeout_error_mentions_configured_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(PlannerError, match="within 30s"):
        generate_llm_plan("Count cells.", config=_config(), transport=httpx.MockTransport(handler))


def test_http_error_status_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    with pytest.raises(PlannerError, match="HTTP 500"):
        generate_llm_plan("Count cells.", config=_config(), transport=httpx.MockTransport(handler))


def test_unexpected_response_shape_is_handled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    with pytest.raises(PlannerError, match="unexpected response format"):
        generate_llm_plan("Count cells.", config=_config(), transport=httpx.MockTransport(handler))


def test_missing_configuration_is_reported() -> None:
    with pytest.raises(PlannerError, match="not configured"):
        generate_llm_plan("Count cells.", config=_config(llm_base_url=""))
    with pytest.raises(PlannerError, match="not configured"):
        generate_llm_plan("Count cells.", config=_config(llm_model=""))


def test_empty_request_fails_without_network_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("No network call expected for an empty request.")

    with pytest.raises(PlannerError, match="empty"):
        generate_llm_plan("   ", config=_config(), transport=httpx.MockTransport(handler))


def test_error_messages_never_contain_the_api_key() -> None:
    secret = "sk-super-secret-key"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(PlannerError) as excinfo:
        generate_llm_plan(
            "Count cells.",
            config=_config(llm_api_key=secret),
            transport=httpx.MockTransport(handler),
        )

    assert secret not in str(excinfo.value)


def test_generate_plan_dispatches_to_demo_mode() -> None:
    config = _config(planner_mode="demo")

    plan = generate_plan("Count the bright objects.", config=config, metadata=_metadata())

    assert plan.supported
    assert plan.steps[0].tool == "convert_to_grayscale"


def test_generate_plan_dispatches_to_llm_mode() -> None:
    transport, seen = _transport_returning(json.dumps(VALID_PLAN_DICT))
    config = _config(planner_mode="llm")

    plan = generate_plan("Count cells.", config=config, metadata=_metadata(), transport=transport)

    assert plan.supported
    assert len(seen) == 1


def test_generate_plan_mode_override_beats_config() -> None:
    config = _config(planner_mode="llm")  # would need network without override

    plan = generate_plan("Count cells.", config=config, metadata=_metadata(), mode="demo")

    assert plan.supported  # demo planner answered; no transport was needed


def test_generate_plan_rejects_unknown_mode() -> None:
    with pytest.raises(PlannerError, match="Unknown planner mode"):
        generate_plan("Count cells.", config=_config(), mode="autopilot")


def test_unsupported_llm_plan_passes_through_and_fails_validation() -> None:
    refusal = {
        "goal": "unsupported_request",
        "supported": False,
        "explanation": "Only registered image-analysis tools are available.",
        "steps": [],
        "warnings": [],
    }
    transport, _ = _transport_returning(json.dumps(refusal))

    plan = generate_llm_plan("Delete my files.", config=_config(), transport=transport)

    assert not plan.supported
    assert not validate_plan(plan, channels=3).valid
