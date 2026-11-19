#!/usr/bin/env python
"""POST-FREEZE COPY (Mon 16 Nov 2026) of tools/llm_bridge/call_gateway.py: a request for a Qwen model also sends chat_template_kwargs
enable_thinking false (Qwen3.5 thinks by default: a 60-token prompt used all 800 completion tokens on reasoning and returned no text). Nothing else changes.

One-shot JSON caller for the gateway (an OpenAI-compatible gateway), same protocol as call.py (Gemini).

Reads ONE JSON request on stdin and writes ONE JSON response on stdout, then exits. Same request fields as call.py (model, prompt, system,
temperature, max_output_tokens, seed, response_json_schema, timeout_ms, json_mode) and the same response fields (ok, text, model,
prompt_tokens, completion_tokens, thoughts_tokens, total_tokens, finish_reason, error_type, retryable, ...), so BridgeTransport and the
replay cache work unchanged. A request {"op": "list_models"} returns {"ok": true, "models": [ids]}.

Configuration (environment only; neither value is ever logged, echoed or stored here):
  LLM_GATEWAY_API_KEY    the the gateway API key (Authorization: Bearer ...), set in the login shell
  LLM_GATEWAY_BASE_URL   the OpenAI-compatible base URL that the gateway issues with the key (the OPENAI_BASE_URL of the gateway documentation); the
                 chat endpoint is {LLM_GATEWAY_BASE_URL}/chat/completions and the model list {LLM_GATEWAY_BASE_URL}/models.
STATUS: written against the documented OpenAI chat-completions protocol and NOT YET RUN against the gateway: the base URL is not on the server
and is not in the public gateway pages. Lower confidence until `tr_gateway.py smoke` has passed.

Parameter handling (conservative): the schema is sent as response_format json_schema (strict false) with Gemini-only keys (propertyOrdering)
removed; the output limit is sent as max_completion_tokens; temperature and seed are sent only when given. If the service answers HTTP 400
naming one of temperature, seed, max_completion_tokens or response_format as unsupported, that parameter is dropped and the call is
repeated once (recorded in retry_log). Retries: HTTP 429, 5xx and timeouts after 2, 4 and 8 seconds.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any

RETRY_SLEEPS = (2, 4, 8)
RETRYABLE_HTTP = frozenset({408, 429, 500, 502, 503, 504})
DROPPABLE = ("temperature", "seed", "max_completion_tokens", "response_format")


def scrub(text: Any) -> str:
    out = str(text)
    for name in ("LLM_GATEWAY_API_KEY", "LLM_GATEWAY_BASE_URL"):
        value = os.environ.get(name)
        if value and len(value) >= 8:
            out = out.replace(value, "<redacted>")
    return out[:2000]


def _error(error_type: str, *, message: str = "", status: int | None = None, retryable: bool = False, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "live": True, "error_type": error_type, "error_message": scrub(message), "http_status": status,
            "retryable": retryable, **extra}


def strip_keys(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: strip_keys(v) for k, v in schema.items() if k != "propertyOrdering"}
    if isinstance(schema, list):
        return [strip_keys(v) for v in schema]
    return schema


def _http(method: str, url: str, key: str, payload: dict | None, timeout_s: float) -> tuple[int, dict | str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - a fixed https URL from the environment
            body = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        status, body = exc.code, exc.read().decode("utf-8", "replace")
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body


def _config() -> tuple[str, str] | dict[str, Any]:
    key, base = os.environ.get("LLM_GATEWAY_API_KEY"), os.environ.get("LLM_GATEWAY_BASE_URL")
    if not key:
        return _error("no_api_key", message="LLM_GATEWAY_API_KEY is not set in the environment")
    if not base:
        return _error("no_base_url", message="LLM_GATEWAY_BASE_URL is not set (the OpenAI-compatible base URL issued with the the gateway key)")
    return key, base.rstrip("/")


def list_models(timeout_s: float = 60.0) -> dict[str, Any]:
    cfg = _config()
    if isinstance(cfg, dict):
        return cfg
    key, base = cfg
    try:
        status, body = _http("GET", f"{base}/models", key, None, timeout_s)
    except Exception as exc:  # noqa: BLE001
        return _error("network", message=f"{type(exc).__name__}: {exc}", retryable=True)
    if status != 200 or not isinstance(body, dict):
        return _error("client_error" if status < 500 else "server_error", message=str(body)[:300], status=status)
    items = body.get("data") if isinstance(body.get("data"), list) else body.get("models", [])
    ids = sorted(str(m.get("id") or m.get("name")) for m in items if isinstance(m, dict))
    return {"ok": True, "live": True, "models": ids, "http_status": status}


def call_live(req: dict[str, Any], *, sleep: Any = time.sleep) -> dict[str, Any]:
    cfg = _config()
    if isinstance(cfg, dict):
        return cfg
    key, base = cfg
    messages = ([{"role": "system", "content": req["system"]}] if req.get("system") else []) + [{"role": "user", "content": req["prompt"]}]
    payload: dict[str, Any] = {"model": req["model"], "messages": messages,
                               "max_completion_tokens": int(req.get("max_output_tokens", 1024))}
    if req.get("temperature") is not None:
        payload["temperature"] = req["temperature"]
    if req.get("seed") is not None:
        payload["seed"] = req["seed"]
    if "qwen" in str(req["model"]).lower():
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    schema = req.get("response_json_schema")
    if schema is not None:
        payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "decision", "schema": strip_keys(schema), "strict": False}}
    elif req.get("json_mode", True):
        payload["response_format"] = {"type": "json_object"}
    timeout_s = float(req.get("timeout_ms", 60000)) / 1000.0
    log: list[dict[str, Any]] = []
    started_all, dropped = time.perf_counter(), False
    attempt = 0
    while attempt <= len(RETRY_SLEEPS):
        started = time.perf_counter()
        try:
            status, body = _http("POST", f"{base}/chat/completions", key, payload, timeout_s)
        except Exception as exc:  # noqa: BLE001
            name = type(exc).__name__
            kind, retryable = ("timeout", True) if "Timeout" in name or "timed out" in str(exc) else ("network", True)
            log.append({"attempt": attempt + 1, "error_type": kind})
            if attempt < len(RETRY_SLEEPS):
                sleep(RETRY_SLEEPS[attempt])
                attempt += 1
                continue
            return _error(kind, message=f"{name}: {exc}", retryable=retryable, attempts=attempt + 1, retry_log=log)
        if status == 400 and not dropped:
            text = json.dumps(body)[:600].lower() if not isinstance(body, str) else body[:600].lower()
            names = [p for p in DROPPABLE if re.search(rf"\b{p}\b", text)]
            if names:
                for p in names:
                    payload.pop(p, None)
                dropped = True
                log.append({"attempt": attempt + 1, "dropped_parameters": names, "http_status": 400})
                continue
        if status in RETRYABLE_HTTP and attempt < len(RETRY_SLEEPS):
            log.append({"attempt": attempt + 1, "error_type": "server_error" if status >= 500 else "rate_limited", "http_status": status})
            sleep(RETRY_SLEEPS[attempt])
            attempt += 1
            continue
        if status != 200 or not isinstance(body, dict):
            kind = {400: "bad_request", 401: "auth", 403: "auth", 429: "rate_limited"}.get(status, "server_error" if status >= 500 else "client_error")
            return _error(kind, message=str(body)[:400], status=status, retryable=status in RETRYABLE_HTTP, attempts=attempt + 1, retry_log=log)
        try:
            choice = body["choices"][0]
            text = choice["message"]["content"]
            finish = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError):
            return _error("empty_response", message="no choices in the reply", attempts=attempt + 1, retry_log=log)
        if not text:
            return _error("empty_response", message="the model returned no text", finish_reason=choice.get("finish_reason"),
                          attempts=attempt + 1, retry_log=log)
        usage = body.get("usage") or {}
        p, c = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        total = int(usage.get("total_tokens") or (p + c))
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        now = time.perf_counter()
        return {"ok": True, "live": True, "text": text, "model": req["model"], "finish_reason": finish, "prompt_tokens": p,
                "completion_tokens": c, "thoughts_tokens": 0, "total_tokens": total, "cached_tokens": cached,
                "latency_ms": round((now - started) * 1000.0, 1), "total_latency_ms": round((now - started_all) * 1000.0, 1),
                "attempts": attempt + 1, "retry_log": log, "error_type": None, "retryable": False}
    return _error("server_error", message="retries exhausted", retryable=True, retry_log=log)


def main() -> int:
    try:
        req = json.loads(sys.stdin.read())
        if not isinstance(req, dict):
            raise ValueError("request must be a JSON object")
        if req.get("op") == "list_models":
            out = list_models()
        else:
            if not (isinstance(req.get("model"), str) and isinstance(req.get("prompt"), str)):
                raise ValueError("request needs string fields model and prompt")
            out = call_live(req)
    except (ValueError, json.JSONDecodeError) as exc:
        out = _error("bad_request", message=str(exc))
    sys.stdout.write(json.dumps(out, sort_keys=True) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
