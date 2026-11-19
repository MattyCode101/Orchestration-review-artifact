"""The Gemini bridge: protocol, fake mode, retries, key hygiene. No network, no key needed."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BRIDGE = Path(__file__).resolve().parents[1] / "tools/llm_bridge/call.py"


def _load():
    spec = importlib.util.spec_from_file_location("llm_bridge_call", BRIDGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bridge = _load()


class FakeAPIError(Exception):
    def __init__(self, code: int, message: str = "boom") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ReadTimeout(Exception):  # same class name the HTTP library uses
    pass


def _response(text: str | None = '{"ok": true}', finish: str = "STOP"):
    usage = SimpleNamespace(
        prompt_token_count=11, candidates_token_count=7, thoughts_token_count=None,
        total_token_count=18, cached_content_token_count=None,
    )
    return SimpleNamespace(
        text=text, usage_metadata=usage,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))],
    )


class ScriptedClient:
    """Stands in for genai.Client: ScriptedClient(...).models.generate_content(...)."""

    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0
        self.models = self

    def generate_content(self, *, model, contents, config):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


REQUEST = {"model": "gemini-test", "prompt": "hello", "temperature": 0, "max_output_tokens": 64}


def _run(request, *, client, key_present=True):
    sleeps: list[float] = []
    result = bridge.call_live(request, client=client, sleep=sleeps.append)
    return result, sleeps


def test_fake_mode_returns_the_scripted_reply_with_token_counts() -> None:
    proc = subprocess.run(
        [sys.executable, str(BRIDGE), "--fake"], capture_output=True, text=True, timeout=30,
        input=json.dumps({**REQUEST, "fake_reply": '{"action": "STOP"}'}),
    )
    out = json.loads(proc.stdout)
    assert proc.returncode == 0 and out["ok"] is True and out["live"] is False
    assert out["text"] == '{"action": "STOP"}'
    assert out["prompt_tokens"] > 0 and out["completion_tokens"] > 0
    assert out["total_tokens"] == out["prompt_tokens"] + out["completion_tokens"]


def test_fake_mode_can_simulate_a_transient_failure() -> None:
    proc = subprocess.run(
        [sys.executable, str(BRIDGE), "--fake"], capture_output=True, text=True, timeout=30,
        input=json.dumps({**REQUEST, "fake_error": {"error_type": "rate_limited", "http_status": 429}}),
    )
    out = json.loads(proc.stdout)
    assert out["ok"] is False and out["retryable"] is True and out["http_status"] == 429


def test_fake_mode_does_not_import_the_sdk() -> None:
    code = (
        "import importlib.util, json, sys;"
        f"s=importlib.util.spec_from_file_location('b', {str(BRIDGE)!r});"
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m);"
        "m.call_fake({'model':'x','prompt':'p','fake_reply':'{}'});"
        "print(json.dumps('google' in sys.modules))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert proc.stdout.strip() == "false"


def test_malformed_stdin_is_reported_as_an_error_not_a_crash() -> None:
    proc = subprocess.run(
        [sys.executable, str(BRIDGE), "--fake"], capture_output=True, text=True, timeout=30,
        input="not json",
    )
    out = json.loads(proc.stdout)
    assert out["ok"] is False and out["error_type"] == "bad_request" and out["retryable"] is False


def test_success_reports_usage_and_finish_reason() -> None:
    result, sleeps = _run(REQUEST, client=ScriptedClient([_response()]))
    assert result["ok"] is True and sleeps == []
    assert (result["prompt_tokens"], result["completion_tokens"], result["total_tokens"]) == (11, 7, 18)
    assert result["thoughts_tokens"] == 0 and result["finish_reason"] == "STOP"
    assert result["attempts"] == 1 and result["live"] is True


@pytest.mark.parametrize("code", [429, 500, 503])
def test_rate_limits_and_server_errors_are_retried_at_two_four_eight_seconds(code) -> None:
    client = ScriptedClient([FakeAPIError(code), FakeAPIError(code), FakeAPIError(code), _response()])
    result, sleeps = _run(REQUEST, client=client)
    assert result["ok"] is True and sleeps == [2, 4, 8] and client.calls == 4 and result["attempts"] == 4


def test_retries_stop_after_the_third_wait() -> None:
    client = ScriptedClient([FakeAPIError(429)] * 4)
    result, sleeps = _run(REQUEST, client=client)
    assert result["ok"] is False and result["retryable"] is True and result["http_status"] == 429
    assert sleeps == [2, 4, 8] and client.calls == 4


def test_a_timeout_is_retried() -> None:
    client = ScriptedClient([ReadTimeout("slow"), _response()])
    result, sleeps = _run(REQUEST, client=client)
    assert result["ok"] is True and sleeps == [2]


def test_a_client_error_is_not_retried() -> None:
    client = ScriptedClient([FakeAPIError(400, "bad schema")])
    result, sleeps = _run(REQUEST, client=client)
    assert result["ok"] is False and result["retryable"] is False and sleeps == [] and client.calls == 1
    assert result["error_type"] == "bad_request" and result["http_status"] == 400


def test_empty_text_is_an_error_with_its_finish_reason() -> None:
    result, _ = _run(REQUEST, client=ScriptedClient([_response(text=None, finish="SAFETY")]))
    assert result["ok"] is False and result["error_type"] == "empty_response"
    assert result["finish_reason"] == "SAFETY" and result["retryable"] is False


def test_truncated_text_is_returned_with_max_tokens_so_the_validator_can_judge_it() -> None:
    result, _ = _run(REQUEST, client=ScriptedClient([_response(text='{"a": ', finish="MAX_TOKENS")]))
    assert result["ok"] is True and result["finish_reason"] == "MAX_TOKENS"


def test_key_material_never_appears_in_an_error(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "SUPER-SECRET-VALUE-0123456789")
    leaky = FakeAPIError(403, "denied for key SUPER-SECRET-VALUE-0123456789 and AIza" + "SyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q")
    result, _ = _run(REQUEST, client=ScriptedClient([leaky]))
    text = json.dumps(result)
    assert "SUPER-SECRET-VALUE" not in text and "AIzaSy" not in text and "<redacted>" in text


def test_missing_key_is_reported_without_a_network_call(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out = bridge.call_live(REQUEST)
    assert out["ok"] is False and out["error_type"] == "no_api_key" and out["retryable"] is False


def test_config_requests_json_mime_type_with_the_json_schema_field() -> None:
    seen = {}

    class Config:
        def __init__(self, **kwargs) -> None:
            seen.update(kwargs)

    class Thinking:
        def __init__(self, **kwargs) -> None:
            seen["thinking"] = kwargs

    types = SimpleNamespace(GenerateContentConfig=Config, ThinkingConfig=Thinking)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    bridge.build_config({**REQUEST, "response_json_schema": schema, "seed": 3, "thinking_budget": 0}, types)
    assert seen["response_mime_type"] == "application/json"
    assert seen["response_json_schema"] == schema and "response_schema" not in seen
    assert seen["temperature"] == 0 and seen["seed"] == 3 and seen["max_output_tokens"] == 64
    assert seen["thinking"] == {"thinking_budget": 0}


def _capture_config(request):
    seen: dict = {}

    class Config:
        def __init__(self, **kwargs) -> None:
            seen.update(kwargs)

    class Thinking:
        def __init__(self, **kwargs) -> None:
            seen["thinking"] = kwargs

    bridge.build_config(request, SimpleNamespace(GenerateContentConfig=Config, ThinkingConfig=Thinking))
    return seen


def test_an_explicit_null_temperature_is_left_out_so_the_model_uses_its_documented_default() -> None:
    """gemini-4-argon ignores sampling parameters and its documentation says to remove them."""

    assert "temperature" not in _capture_config({**REQUEST, "temperature": None})


def test_an_absent_temperature_still_defaults_to_zero_as_for_the_earlier_model() -> None:
    request = {key: value for key, value in REQUEST.items() if key != "temperature"}
    assert _capture_config(request)["temperature"] == 0


def test_a_null_seed_is_left_out_and_a_zero_seed_is_kept() -> None:
    assert "seed" not in _capture_config({**REQUEST, "seed": None})
    assert _capture_config({**REQUEST, "seed": 0})["seed"] == 0


def test_thinking_level_is_passed_to_the_thinking_config_without_a_budget() -> None:
    seen = _capture_config({**REQUEST, "thinking_level": "medium"})
    assert seen["thinking"] == {"thinking_level": "medium"}


def test_thinking_budget_and_thinking_level_together_are_a_bad_request_without_a_call() -> None:
    client = ScriptedClient([_response()])
    result = bridge.call_live({**REQUEST, "thinking_budget": 0, "thinking_level": "low"}, client=client)
    assert result["ok"] is False and result["error_type"] == "bad_request" and result["retryable"] is False
    assert client.calls == 0


def test_thinking_tokens_are_reported_apart_from_the_visible_completion() -> None:
    usage = SimpleNamespace(
        prompt_token_count=100, candidates_token_count=30, thoughts_token_count=900,
        total_token_count=1030, cached_content_token_count=None,
    )
    response = SimpleNamespace(
        text="{}", usage_metadata=usage, candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
    )
    result, _ = _run(REQUEST, client=ScriptedClient([response]))
    assert (result["prompt_tokens"], result["completion_tokens"], result["thoughts_tokens"]) == (100, 30, 900)
    assert result["total_tokens"] == 1030
