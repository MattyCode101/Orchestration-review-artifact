"""Replay cache, bridge transport, rate limiter and telemetry (no network, no key)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from orch.policies.llm_client import (
    BridgeTransport, CachedLLM, LLMCallError, LLMParams, LLMReply, LLMUnavailableError,
    RateLimiter, ReplayCache, ReplayMiss, TelemetrySink, cache_key,
)

BRIDGE = Path(__file__).resolve().parents[1] / "tools/llm_bridge/call.py"
PARAMS = LLMParams(model="gemini-test", temperature=0.0, max_output_tokens=256, seed=0)
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


def _reply(text: str = '{"ok": true}', tokens: tuple[int, int] = (50, 10)) -> LLMReply:
    return LLMReply(
        text=text, prompt_tokens=tokens[0], completion_tokens=tokens[1], thoughts_tokens=0,
        total_tokens=sum(tokens), latency_ms=12.5, finish_reason="STOP", model="gemini-test", attempts=1,
    )


class CountingTransport:
    def __init__(self, *replies: LLMReply | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[str] = []

    def complete(self, prompt, params, schema):
        self.calls.append(prompt)
        item = self.replies.pop(0) if self.replies else _reply()
        if isinstance(item, Exception):
            raise item
        return item


def test_the_cache_key_depends_on_model_parameters_prompt_and_schema() -> None:
    base = cache_key(PARAMS, "prompt", SCHEMA)
    assert base == cache_key(PARAMS, "prompt", SCHEMA) and len(base) == 64
    assert base != cache_key(LLMParams(model="other"), "prompt", SCHEMA)
    assert base != cache_key(LLMParams(model="gemini-test", temperature=0.5, max_output_tokens=256, seed=0), "prompt", SCHEMA)
    assert base != cache_key(PARAMS, "prompt two", SCHEMA)
    assert base != cache_key(PARAMS, "prompt", {"type": "object"})
    assert base == cache_key(LLMParams(model="gemini-test", temperature=0.0, max_output_tokens=256, seed=0, timeout_ms=1), "prompt", SCHEMA)


def test_cache_keys_of_the_earlier_model_did_not_move_when_new_parameters_were_added() -> None:
    """Stored Saturday replies stay addressable: thinking_level is part of the key only when set."""

    assert sorted(PARAMS.key_parameters()) == ["max_output_tokens", "model", "seed", "temperature", "thinking_budget"]
    assert cache_key(PARAMS, "prompt", SCHEMA) == "e4438dd3a1e70fd8fd30eb84e04886cb640db7b445cb9a8191a634b5d325097b"
    flash = LLMParams(model="gemini-3.1-flash-lite", temperature=0.0, max_output_tokens=2048, seed=0, thinking_budget=None)
    assert cache_key(flash, "golden prompt", None) == "bded74674f79b70c8dfaa15e3862c910f0f5f145b1198af1279e4ec8f77bc816"


def test_thinking_level_and_a_null_temperature_are_part_of_the_cache_key_when_set() -> None:
    base = cache_key(PARAMS, "p", SCHEMA)
    assert base != cache_key(LLMParams(model="gemini-test", temperature=0.0, max_output_tokens=256, seed=0, thinking_level="medium"), "p", SCHEMA)
    assert base != cache_key(LLMParams(model="gemini-test", temperature=None, max_output_tokens=256, seed=0), "p", SCHEMA)
    low = LLMParams(model="m", temperature=None, seed=None, max_output_tokens=9, thinking_level="low")
    high = LLMParams(model="m", temperature=None, seed=None, max_output_tokens=9, thinking_level="high")
    assert cache_key(low, "p", None) != cache_key(high, "p", None)


def test_a_record_with_a_thinking_level_and_no_temperature_round_trips(tmp_path) -> None:
    params = LLMParams(model="m", temperature=None, seed=None, max_output_tokens=4096, thinking_level="medium")
    cache = ReplayCache(tmp_path, mode="record")
    key = cache_key(params, "p", SCHEMA)
    cache.store(key, params, "p", SCHEMA, _reply('{"v": 1}'))
    assert cache.lookup(key).text == '{"v": 1}'


def test_record_mode_calls_once_then_replays_from_the_store(tmp_path) -> None:
    transport = CountingTransport(_reply('{"ok": true}'))
    llm = CachedLLM(transport, ReplayCache(tmp_path, mode="record"), PARAMS)
    first = llm.complete("p", SCHEMA)
    second = llm.complete("p", SCHEMA)
    assert (first.cache_hit, second.cache_hit) == (False, True)
    assert first.reply.text == second.reply.text and len(transport.calls) == 1
    assert second.reply.prompt_tokens == 50 and second.cache_mode == "record"


def test_a_stored_reply_survives_a_new_process(tmp_path) -> None:
    CachedLLM(CountingTransport(_reply("{}")), ReplayCache(tmp_path, mode="record"), PARAMS).complete("p", None)
    again = CachedLLM(CountingTransport(), ReplayCache(tmp_path, mode="record"), PARAMS)
    assert again.complete("p", None).cache_hit is True and again.transport.calls == []


def test_replay_mode_fails_on_a_miss_and_never_calls_the_transport(tmp_path) -> None:
    transport = CountingTransport()
    llm = CachedLLM(transport, ReplayCache(tmp_path, mode="replay"), PARAMS)
    with pytest.raises(ReplayMiss):
        llm.complete("never seen", SCHEMA)
    assert transport.calls == []
    CachedLLM(CountingTransport(_reply()), ReplayCache(tmp_path, mode="record"), PARAMS).complete("seen", SCHEMA)
    assert llm.complete("seen", SCHEMA).cache_hit is True and transport.calls == []


def test_fresh_mode_ignores_the_main_store_and_keeps_its_own_namespace(tmp_path) -> None:
    CachedLLM(CountingTransport(_reply('{"v": "recorded"}')), ReplayCache(tmp_path, mode="record"), PARAMS).complete("p", None)
    fresh_transport = CountingTransport(_reply('{"v": "fresh one"}'), _reply('{"v": "fresh two"}'))
    run_a = CachedLLM(fresh_transport, ReplayCache(tmp_path, mode="fresh", fresh_tag="rep1"), PARAMS)
    out = run_a.complete("p", None)
    assert out.cache_hit is False and out.reply.text == '{"v": "fresh one"}' and out.cache_mode == "fresh"
    assert run_a.complete("p", None).cache_hit is True                       # resume inside the same repeat
    run_b = CachedLLM(fresh_transport, ReplayCache(tmp_path, mode="fresh", fresh_tag="rep2"), PARAMS)
    assert run_b.complete("p", None).reply.text == '{"v": "fresh two"}'      # a different repeat calls again
    main = CachedLLM(CountingTransport(), ReplayCache(tmp_path, mode="record"), PARAMS).complete("p", None)
    assert main.reply.text == '{"v": "recorded"}' and main.cache_hit is True  # the recorded reply is untouched


def test_fresh_mode_needs_a_tag(tmp_path) -> None:
    with pytest.raises(ValueError):
        ReplayCache(tmp_path, mode="fresh")


def test_errors_are_never_stored(tmp_path) -> None:
    transport = CountingTransport(LLMUnavailableError("busy"), _reply())
    llm = CachedLLM(transport, ReplayCache(tmp_path, mode="record"), PARAMS)
    with pytest.raises(LLMUnavailableError):
        llm.complete("p", None)
    assert llm.complete("p", None).cache_hit is False and len(transport.calls) == 2


def test_the_first_writer_wins_when_two_processes_race(tmp_path) -> None:
    one = ReplayCache(tmp_path, mode="record")
    key = cache_key(PARAMS, "p", None)
    first = one.store(key, PARAMS, "p", None, _reply('{"w": 1}'))
    second = one.store(key, PARAMS, "p", None, _reply('{"w": 2}'))
    assert first.text == second.text == '{"w": 1}'


def test_a_corrupt_record_is_refused_instead_of_trusted(tmp_path) -> None:
    cache = ReplayCache(tmp_path, mode="record")
    key = cache_key(PARAMS, "p", None)
    cache.store(key, PARAMS, "p", None, _reply())
    path = next(tmp_path.rglob(f"{key}.json"))
    record = json.loads(path.read_text())
    record["prompt"] = "tampered"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        cache.lookup(key)


def test_the_bridge_transport_round_trips_through_fake_mode() -> None:
    transport = BridgeTransport(sys.executable, BRIDGE, extra_args=("--fake",),
                                request_extras={"fake_reply": '{"ok": true}'})
    reply = transport.complete("a prompt of some length", PARAMS, SCHEMA)
    assert reply.text == '{"ok": true}' and reply.prompt_tokens > 0 and reply.total_tokens > 0


ECHO = (
    "import json, sys\n"
    "request = json.loads(sys.stdin.read())\n"
    "print(json.dumps({'ok': True, 'text': json.dumps(request, sort_keys=True), 'prompt_tokens': 3,\n"
    "                  'completion_tokens': 2, 'thoughts_tokens': 5, 'total_tokens': 10, 'model': request['model']}))\n"
)


def test_the_bridge_request_carries_thinking_level_and_leaves_out_a_null_temperature_and_seed(tmp_path) -> None:
    script = tmp_path / "echo_bridge.py"
    script.write_text(ECHO, encoding="utf-8")
    transport = BridgeTransport(sys.executable, script)
    params = LLMParams(model="m", temperature=None, seed=None, max_output_tokens=4096, thinking_level="medium")
    sent = json.loads(transport.complete("hello", params, SCHEMA).text)
    assert sent["thinking_level"] == "medium" and "thinking_budget" not in sent
    assert "temperature" not in sent and "seed" not in sent
    assert sent["max_output_tokens"] == 4096 and sent["response_json_schema"] == SCHEMA
    old = json.loads(transport.complete("hello", PARAMS, None).text)
    assert old["temperature"] == 0.0 and old["seed"] == 0 and "thinking_level" not in old


def test_thinking_tokens_reach_the_reply(tmp_path) -> None:
    script = tmp_path / "echo_bridge.py"
    script.write_text(ECHO, encoding="utf-8")
    reply = BridgeTransport(sys.executable, script).complete("hello", PARAMS, None)
    assert (reply.prompt_tokens, reply.completion_tokens, reply.thoughts_tokens, reply.total_tokens) == (3, 2, 5, 10)


def test_a_retryable_bridge_failure_becomes_unavailable_and_a_bad_request_a_call_error() -> None:
    busy = BridgeTransport(sys.executable, BRIDGE, extra_args=("--fake",),
                           request_extras={"fake_error": {"error_type": "rate_limited", "http_status": 429}})
    with pytest.raises(LLMUnavailableError):
        busy.complete("p", PARAMS, None)
    bad = BridgeTransport(sys.executable, BRIDGE, extra_args=("--fake",),
                          request_extras={"fake_error": {"error_type": "bad_request", "http_status": 400}})
    with pytest.raises(LLMCallError):
        bad.complete("p", PARAMS, None)


def test_the_rate_limiter_waits_for_the_window_and_is_shared_through_a_directory(tmp_path) -> None:
    now = [1000.0]
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(tmp_path, per_minute=2, clock=lambda: now[0], sleep=sleep)
    limiter.acquire()
    now[0] += 10
    RateLimiter(tmp_path, per_minute=2, clock=lambda: now[0], sleep=sleep).acquire()   # another process
    assert slept == []
    limiter.acquire()                                                                  # third within a minute
    assert slept and 49.0 <= slept[0] <= 51.0


def test_the_telemetry_sink_appends_one_json_line_per_record(tmp_path) -> None:
    sink = TelemetrySink(tmp_path / "t" / "calls.jsonl")
    sink.append({"a": 1})
    TelemetrySink(tmp_path / "t" / "calls.jsonl").append({"b": 2})
    assert TelemetrySink.read(tmp_path / "t" / "calls.jsonl") == [{"a": 1}, {"b": 2}]
