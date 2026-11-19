"""Model spend ledger: price arithmetic, a hard stop below the cap, sharing across processes."""
from __future__ import annotations

import json
import multiprocessing

import pytest

from orch.policies.llm_client import LLMParams, LLMReply, LLMUnavailableError
from orch.policies.llm_ledger import LedgeredTransport, Price, SpendCapReached, SpendLedger

PRICE = Price(input_per_million=2.0, output_per_million=10.0)
PARAMS = LLMParams(model="gemini-4-argon", temperature=None, seed=None, max_output_tokens=16384, thinking_level="medium")


def _reply(prompt: int = 3000, completion: int = 200, thoughts: int = 3000) -> LLMReply:
    return LLMReply(
        text="{}", prompt_tokens=prompt, completion_tokens=completion, thoughts_tokens=thoughts,
        total_tokens=prompt + completion + thoughts, latency_ms=1.0, finish_reason="STOP",
        model="gemini-4-argon", attempts=1,
    )


class Scripted:
    def __init__(self, *items) -> None:
        self.items = list(items)
        self.calls = 0

    def complete(self, prompt, params, schema):
        self.calls += 1
        item = self.items.pop(0) if self.items else _reply()
        if isinstance(item, Exception):
            raise item
        return item


def test_thinking_tokens_are_charged_at_the_output_price() -> None:
    cost = PRICE.cost(prompt_tokens=3000, completion_tokens=200, thoughts_tokens=3000)
    assert cost == pytest.approx(3000 * 2.0 / 1e6 + 3200 * 10.0 / 1e6)


def test_tokens_in_the_reported_total_but_in_neither_count_are_charged_as_output(tmp_path) -> None:
    """If the provider's total exceeds prompt + completion + thoughts, the surplus is hidden thinking."""

    ledger = SpendLedger(tmp_path / "spend.jsonl", cap_usd=35.0, price=PRICE)
    hidden = LLMReply(text="{}", prompt_tokens=1000, completion_tokens=100, thoughts_tokens=0, total_tokens=1800,
                      latency_ms=1.0, finish_reason="STOP", model="gemini-4-argon", attempts=1)
    usd = ledger.add(source="x", reply=hidden)
    assert usd == pytest.approx(1000 * 2.0 / 1e6 + 800 * 10.0 / 1e6)
    plain = LLMReply(text="{}", prompt_tokens=1000, completion_tokens=100, thoughts_tokens=0, total_tokens=1100,
                     latency_ms=1.0, finish_reason="STOP", model="gemini-4-argon", attempts=1)
    assert ledger.add(source="y", reply=plain) == pytest.approx(PRICE.cost(1000, 100, 0))


def test_the_total_is_the_sum_of_the_recorded_calls_and_survives_a_new_process(tmp_path) -> None:
    ledger = SpendLedger(tmp_path / "spend.jsonl", cap_usd=35.0, price=PRICE)
    ledger.add(source="a", reply=_reply())
    ledger.add(source="b", reply=_reply(prompt=1000, completion=100, thoughts=0))
    expected = PRICE.cost(3000, 200, 3000) + PRICE.cost(1000, 100, 0)
    assert ledger.total_usd() == pytest.approx(expected)
    assert SpendLedger(tmp_path / "spend.jsonl", cap_usd=35.0, price=PRICE).total_usd() == pytest.approx(expected)


def test_a_call_is_refused_once_the_total_reaches_the_cap_less_the_headroom(tmp_path) -> None:
    ledger = SpendLedger(tmp_path / "spend.jsonl", cap_usd=0.10, headroom_usd=0.03, price=PRICE)
    transport = Scripted()
    wrapped = LedgeredTransport(transport, ledger, source="t")
    wrapped.complete("p", PARAMS, None)           # 0.038, the stop is at 0.07
    wrapped.complete("p", PARAMS, None)           # 0.076: the second began below the stop
    with pytest.raises(SpendCapReached, match="US"):
        wrapped.complete("p", PARAMS, None)       # refused before any third call
    assert transport.calls == 2


def test_the_stop_is_checked_before_the_call_not_after(tmp_path) -> None:
    ledger = SpendLedger(tmp_path / "spend.jsonl", cap_usd=0.05, headroom_usd=0.0, price=PRICE)
    ledger.add(source="earlier", reply=_reply())  # 0.038: below the stop
    transport = Scripted()
    wrapped = LedgeredTransport(transport, ledger, source="t")
    wrapped.complete("p", PARAMS, None)           # allowed: the total was below the stop when it began
    assert ledger.total_usd() > 0.05
    with pytest.raises(SpendCapReached):
        wrapped.complete("p", PARAMS, None)


def test_a_failed_call_is_not_charged(tmp_path) -> None:
    ledger = SpendLedger(tmp_path / "spend.jsonl", cap_usd=35.0, price=PRICE)
    wrapped = LedgeredTransport(Scripted(LLMUnavailableError("busy")), ledger, source="t")
    with pytest.raises(LLMUnavailableError):
        wrapped.complete("p", PARAMS, None)
    assert ledger.total_usd() == 0.0


def test_each_record_keeps_the_tokens_the_model_and_the_source(tmp_path) -> None:
    ledger = SpendLedger(tmp_path / "spend.jsonl", cap_usd=35.0, price=PRICE)
    LedgeredTransport(Scripted(_reply(prompt=10, completion=2, thoughts=7)), ledger, source="cell-1").complete("p", PARAMS, None)
    record = json.loads((tmp_path / "spend.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["source"] == "cell-1" and record["model"] == "gemini-4-argon"
    assert (record["prompt_tokens"], record["completion_tokens"], record["thoughts_tokens"]) == (10, 2, 7)
    assert record["usd"] == pytest.approx(PRICE.cost(10, 2, 7))


def _add_many(path: str, count: int) -> None:
    ledger = SpendLedger(path, cap_usd=1e9, price=PRICE)
    for _ in range(count):
        ledger.add(source="p", reply=_reply(prompt=1000, completion=0, thoughts=0))


def test_concurrent_writers_lose_no_record(tmp_path) -> None:
    path = str(tmp_path / "spend.jsonl")
    workers = [multiprocessing.Process(target=_add_many, args=(path, 25)) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    ledger = SpendLedger(path, cap_usd=1e9, price=PRICE)
    assert len((tmp_path / "spend.jsonl").read_text(encoding="utf-8").splitlines()) == 100
    assert ledger.total_usd() == pytest.approx(100 * PRICE.cost(1000, 0, 0))
