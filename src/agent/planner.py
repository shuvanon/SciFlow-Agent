"""Planner front door: demo mode and LLM mode behind one interface.

``generate_plan`` routes a request to the deterministic demo planner or to
the LLM planner. All LLM specifics (HTTP client, JSON extraction, repair
retry, error mapping) live here, isolated from the rest of the system.

Failure philosophy (spec sections 12.3 and 23):

- Malformed model output gets exactly one repair retry with the validation
  feedback; if it fails again, a :class:`PlannerError` is raised. Nothing
  malformed ever reaches the executor.
- Network failures do not retry automatically (a slow local model would
  double the wait); the error message says what happened and suggests
  retrying or switching to demo mode.
- Error messages never contain the API key. The base URL may appear — it
  is the user's own configuration and is needed for debugging.
"""

from __future__ import annotations

import json
import logging
import re

import httpx
from pydantic import ValidationError

from src.agent.demo_planner import generate_demo_plan
from src.agent.prompts import build_repair_prompt, build_system_prompt, build_user_prompt
from src.agent.schemas import ExecutionPlan
from src.config import PLANNER_MODE_DEMO, PLANNER_MODE_LLM, AppConfig
from src.errors import PlannerError
from src.models import ImageMetadata

logger = logging.getLogger(__name__)

_FALLBACK_HINT = "Try again, rephrase the request, or switch to demo mode."
_MAX_COMPLETION_TOKENS = 1024
_CONNECT_TIMEOUT_SECONDS = 10.0

_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json_object(content: str) -> str:
    """Peel markdown fences or surrounding prose off a JSON payload."""
    text = content.strip()
    fenced = _FENCE_PATTERN.search(text)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    return text


def _summarize_validation_error(error: ValidationError) -> str:
    parts = []
    for issue in error.errors()[:5]:
        location = ".".join(str(item) for item in issue["loc"]) or "value"
        parts.append(f"{location}: {issue['msg']}")
    return "; ".join(parts)


def _parse_plan(content: str) -> ExecutionPlan:
    """Parse raw model output into an ExecutionPlan.

    Raises:
        ValueError: With a readable summary when the content is not valid
            JSON or does not match the plan schema.
    """
    try:
        data = json.loads(_extract_json_object(content))
    except json.JSONDecodeError as exc:
        raise ValueError(f"the response was not valid JSON ({exc.msg}).") from exc
    try:
        return ExecutionPlan.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            f"the JSON did not match the plan schema ({_summarize_validation_error(exc)})."
        ) from exc


def _post_chat(
    config: AppConfig,
    messages: list[dict[str, str]],
    transport: httpx.BaseTransport | None,
) -> str:
    """Send one chat-completion request and return the message content."""
    url = config.llm_base_url.rstrip("/") + "/chat/completions"
    headers = {}
    if config.llm_api_key:
        headers["Authorization"] = f"Bearer {config.llm_api_key}"
    payload = {
        "model": config.llm_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": _MAX_COMPLETION_TOKENS,
    }
    timeout = httpx.Timeout(
        float(config.llm_timeout_seconds),
        connect=min(_CONNECT_TIMEOUT_SECONDS, float(config.llm_timeout_seconds)),
    )
    with httpx.Client(timeout=timeout, transport=transport) as client:
        response = client.post(url, json=payload, headers=headers)
    response.raise_for_status()

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise PlannerError(
            "The LLM endpoint returned an unexpected response format. "
            "Check that the URL points to an OpenAI-compatible /v1 endpoint. " + _FALLBACK_HINT
        ) from exc
    if not isinstance(content, str):
        raise PlannerError(
            "The LLM endpoint returned an empty or non-text completion. " + _FALLBACK_HINT
        )
    return content


def generate_llm_plan(
    request: str,
    *,
    config: AppConfig,
    metadata: ImageMetadata | None = None,
    transport: httpx.BaseTransport | None = None,
) -> ExecutionPlan:
    """Generate a plan through the configured OpenAI-compatible endpoint.

    Args:
        request: The user's analysis request.
        config: Application configuration (endpoint, model, timeout).
        metadata: Metadata of the loaded image, added to the prompt context.
        transport: Optional httpx transport, injectable for tests.

    Raises:
        PlannerError: On missing configuration, connection problems, or when
            the model fails to produce a schema-valid plan after one repair
            retry. The message is user-safe and contains fallback guidance.
    """
    if not request.strip():
        raise PlannerError("The request is empty. Describe the analysis to perform.")
    if not config.llm_base_url or not config.llm_model:
        raise PlannerError(
            "LLM mode is not configured. Set LLM_BASE_URL and LLM_MODEL in your .env file "
            "(see .env.example), or switch to demo mode."
        )

    messages = [
        {"role": "system", "content": build_system_prompt(config.max_workflow_steps)},
        {"role": "user", "content": build_user_prompt(request, metadata)},
    ]

    last_error = ""
    for attempt in (1, 2):
        try:
            content = _post_chat(config, messages, transport)
        except httpx.TimeoutException as exc:
            raise PlannerError(
                f"The LLM did not respond within {config.llm_timeout_seconds}s. "
                "Local models can be slow — increase LLM_TIMEOUT_SECONDS, try again, "
                "or switch to demo mode."
            ) from exc
        except httpx.ConnectError as exc:
            raise PlannerError(
                f"Could not connect to the LLM endpoint at {config.llm_base_url}. "
                "Check that the server is running and reachable, or switch to demo mode."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise PlannerError(
                f"The LLM endpoint returned HTTP {exc.response.status_code}. "
                "Check LLM_MODEL and the server logs, or switch to demo mode."
            ) from exc
        except httpx.HTTPError as exc:
            raise PlannerError(
                f"Communication with the LLM endpoint failed ({type(exc).__name__}). "
                + _FALLBACK_HINT
            ) from exc

        try:
            plan = _parse_plan(content)
        except ValueError as exc:
            last_error = str(exc)
            logger.warning("LLM plan attempt %d rejected: %s", attempt, last_error)
            messages = [
                *messages,
                {"role": "assistant", "content": content},
                {"role": "user", "content": build_repair_prompt(last_error)},
            ]
            continue

        logger.info(
            "LLM planner produced a plan on attempt %d (%d steps, supported=%s)",
            attempt,
            len(plan.steps),
            plan.supported,
        )
        return plan

    raise PlannerError(
        f"The model did not produce a valid plan after a retry: {last_error} " + _FALLBACK_HINT
    )


def generate_plan(
    request: str,
    *,
    config: AppConfig,
    metadata: ImageMetadata | None = None,
    mode: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> ExecutionPlan:
    """Generate a plan using the selected planner mode.

    Args:
        request: The user's analysis request.
        config: Application configuration.
        metadata: Metadata of the loaded image, if available.
        mode: ``"demo"`` or ``"llm"``; defaults to ``config.planner_mode``.
        transport: Optional httpx transport, injectable for tests.
    """
    selected = (mode or config.planner_mode).lower()
    if selected == PLANNER_MODE_DEMO:
        channels = metadata.channels if metadata else None
        return generate_demo_plan(request, channels=channels)
    if selected == PLANNER_MODE_LLM:
        return generate_llm_plan(request, config=config, metadata=metadata, transport=transport)
    raise PlannerError(f"Unknown planner mode {selected!r}; use 'demo' or 'llm'.")
