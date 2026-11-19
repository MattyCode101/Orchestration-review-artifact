#!/usr/bin/env python
"""One-shot JSON caller for Gemini, the only provider (DAC 2027 LLM orchestrator).

Reads ONE JSON request on stdin and writes ONE JSON response on stdout, then exits.
It runs in this directory's own virtual environment (uv, Python 3.12, the official
Google Gen AI SDK), so the pinned project environment is never touched.

Request fields: model, prompt, system (optional), temperature, max_output_tokens,
seed, thinking_budget (optional), thinking_level (optional; low, medium or high),
response_json_schema (optional), timeout_ms, json_mode (default true). Test-only
fields, honoured by `--fake` or by the LLM_BRIDGE_FAKE environment variable:
fake_reply, fake_error.

An absent temperature means 0 (the earlier model). An explicit null temperature or
seed is left out of the request: gemini-4-argon ignores sampling parameters and its
documentation says to remove them. thinking_budget and thinking_level are exclusive;
asking for both is a bad request and no call is made.

JSON output is requested with response_mime_type="application/json" plus the SDK's
response_json_schema field (checked against the installed SDK, 2.34.0: that field
and response_schema exist; response_json_schema takes a JSON Schema dict and
excludes response_schema).

The API key is read only by the SDK, from GOOGLE_API_KEY. It is never logged,
echoed or stored here; any occurrence of it in an error text is replaced.

Retries: HTTP 429, 5xx and timeouts are retried after 2, 4 and 8 seconds. The
caller treats a final failure as a transient fault that is never charged to an
episode budget.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any

RETRY_SLEEPS = (2, 4, 8)
RETRYABLE_HTTP = frozenset({408, 429, 500, 502, 503, 504})
KEY_VARIABLES = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
_KEY_SHAPE = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")
_NETWORK_NAMES = (
    "ConnectError", "ConnectionError", "RemoteProtocolError", "ReadError", "WriteError",
    "NetworkError", "TransportError", "ProtocolError", "ConnectionResetError",
)


def scrub(text: Any) -> str:
    """Remove anything that looks like, or equals, an API key."""

    out = str(text)
    for name in KEY_VARIABLES:
        value = os.environ.get(name)
        if value and len(value) >= 8:
            out = out.replace(value, "<redacted>")
    return _KEY_SHAPE.sub("<redacted>", out)[:2000]


def _error(error_type: str, *, message: str = "", status: int | None = None,
           retryable: bool = False, live: bool = True, **extra: Any) -> dict[str, Any]:
    return {
        "ok": False, "live": live, "error_type": error_type, "error_message": scrub(message),
        "http_status": status, "retryable": retryable, **extra,
    }


def classify(exc: BaseException) -> tuple[str, int | None, bool]:
    """Map an SDK or transport exception to (error_type, http_status, retryable)."""

    code = getattr(exc, "code", None)
    if isinstance(code, int) and not isinstance(code, bool):
        if code == 429:
            return "rate_limited", code, True
        if code in RETRYABLE_HTTP or code >= 500:
            return ("timeout" if code == 408 else "server_error"), code, True
        if code == 400:
            return "bad_request", code, False
        if code in (401, 403):
            return "auth", code, False
        return "client_error", code, False
    name = type(exc).__name__
    if "Timeout" in name:
        return "timeout", None, True
    if name in _NETWORK_NAMES:
        return "network", None, True
    return "unexpected", None, False


def build_config(req: dict[str, Any], types: Any) -> Any:
    """Build the SDK generation config; JSON mode always sets the mime type."""

    kwargs: dict[str, Any] = {"max_output_tokens": req.get("max_output_tokens", 1024)}
    temperature = req.get("temperature", 0)
    if temperature is not None:
        kwargs["temperature"] = temperature
    if req.get("seed") is not None:
        kwargs["seed"] = req["seed"]
    if req.get("system"):
        kwargs["system_instruction"] = req["system"]
    schema = req.get("response_json_schema")
    if schema is not None or req.get("json_mode", True):
        kwargs["response_mime_type"] = "application/json"
    if schema is not None:
        kwargs["response_json_schema"] = schema
    budget, level = req.get("thinking_budget"), req.get("thinking_level")
    if budget is not None and level is not None:
        raise ValueError("thinking_budget and thinking_level cannot be combined")
    if budget is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)
    elif level is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level)
    return types.GenerateContentConfig(**kwargs)


def _count(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def call_live(req: dict[str, Any], *, client: Any = None, sleep: Any = time.sleep) -> dict[str, Any]:
    """Call Gemini with the retry schedule; `client` and `sleep` are injectable for tests."""

    types = None
    if client is None:
        if not any(os.environ.get(name) for name in KEY_VARIABLES):
            return _error("no_api_key", message="GOOGLE_API_KEY is not set in the environment")
        from google import genai  # imported late: fake mode and key-less paths never need it
        from google.genai import types as sdk_types

        types = sdk_types
        client = genai.Client(http_options=types.HttpOptions(timeout=int(req.get("timeout_ms", 60000))))
    elif req.get("_types") is not None:
        types = req["_types"]
    if types is None:
        try:
            from google.genai import types as sdk_types

            types = sdk_types
        except ImportError:  # injected test client without the SDK installed
            types = _PlainTypes()
    try:
        config = build_config(req, types)
    except ValueError as exc:
        return _error("bad_request", message=str(exc), retryable=False)
    log: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for attempt in range(len(RETRY_SLEEPS) + 1):
        started = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=req["model"], contents=req["prompt"], config=config,
            )
        except Exception as exc:  # noqa: BLE001 - classified below, never re-raised raw
            kind, status, retryable = classify(exc)
            log.append({"attempt": attempt + 1, "error_type": kind, "http_status": status})
            if retryable and attempt < len(RETRY_SLEEPS):
                sleep(RETRY_SLEEPS[attempt])
                continue
            return _error(kind, message=f"{type(exc).__name__}: {getattr(exc, 'message', exc)}",
                          status=status, retryable=retryable, attempts=attempt + 1, retry_log=log)
        try:
            text = response.text
        except Exception:  # noqa: BLE001
            text = None
        finish = None
        try:
            finish = response.candidates[0].finish_reason.name
        except Exception:  # noqa: BLE001
            pass
        if not text:
            return _error("empty_response", message="the model returned no text", retryable=False,
                          finish_reason=finish, attempts=attempt + 1, retry_log=log)
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = _count(getattr(usage, "prompt_token_count", None))
        completion_tokens = _count(getattr(usage, "candidates_token_count", None))
        thoughts = _count(getattr(usage, "thoughts_token_count", None))
        total = _count(getattr(usage, "total_token_count", None)) or prompt_tokens + completion_tokens + thoughts
        now = time.perf_counter()
        return {
            "ok": True, "live": True, "text": text, "model": req["model"], "finish_reason": finish,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "thoughts_tokens": thoughts, "total_tokens": total,
            "cached_tokens": _count(getattr(usage, "cached_content_token_count", None)),
            "latency_ms": round((now - started) * 1000.0, 1),
            "total_latency_ms": round((now - started_all) * 1000.0, 1),
            "attempts": attempt + 1, "retry_log": log, "error_type": None, "retryable": False,
        }
    raise AssertionError("unreachable")  # the loop always returns


class _PlainTypes:
    """Stand-in config types so an injected client can be used without the SDK installed."""

    class GenerateContentConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class ThinkingConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)


def call_fake(req: dict[str, Any]) -> dict[str, Any]:
    """Deterministic offline reply for tests; never imports the SDK or reads the key."""

    simulated = req.get("fake_error")
    if simulated:
        status = simulated.get("http_status")
        retryable = bool(simulated.get("retryable", status in RETRYABLE_HTTP))
        return _error(simulated.get("error_type", "fake_error"), message="simulated failure",
                      status=status, retryable=retryable, live=False, attempts=1, retry_log=[])
    text = req.get("fake_reply", "{}")
    prompt_tokens = max(1, len(str(req.get("prompt", ""))) // 4)
    completion_tokens = max(1, len(text) // 4)
    return {
        "ok": True, "live": False, "text": text, "model": req.get("model", "fake"),
        "finish_reason": "STOP", "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "thoughts_tokens": 0, "total_tokens": prompt_tokens + completion_tokens, "cached_tokens": 0,
        "latency_ms": 0.0, "total_latency_ms": 0.0, "attempts": 1, "retry_log": [],
        "error_type": None, "retryable": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    fake = "--fake" in args or os.environ.get("LLM_BRIDGE_FAKE") == "1"
    try:
        req = json.loads(sys.stdin.read())
        if not isinstance(req, dict):
            raise ValueError("request must be a JSON object")
        if not fake and not (isinstance(req.get("model"), str) and isinstance(req.get("prompt"), str)):
            raise ValueError("request needs string fields model and prompt")
    except (ValueError, json.JSONDecodeError) as exc:
        out = _error("bad_request", message=str(exc), live=not fake)
    else:
        out = call_fake(req) if fake else call_live(req)
    sys.stdout.write(json.dumps(out, sort_keys=True) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
