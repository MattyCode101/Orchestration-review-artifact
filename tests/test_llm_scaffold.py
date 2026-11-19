"""Synthetic tests for the API-free LLM policy and exact prompt replay."""

from __future__ import annotations

import hashlib
import json

import pytest

from orch.contract.brief_text import load_frozen_brief
from orch.contract.models import MetricName, Objective, WorkerCard
from orch.contract.views import PolicyView, StateSummary
from orch.policies.llm_scaffold import (
    LLMPolicyScaffold,
    LLMReply,
    PromptResponseCache,
    mandate_json_schema,
    prompt_sha256,
    serialize_view,
)


class FakeLLM:
    """A fixed reply queue that performs no network or tool call."""

    def __init__(self, replies: list[LLMReply]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> LLMReply:
        self.prompts.append(prompt)
        return self.replies.pop(0)


def _view(*, budget: int = 6, episode_id: str = "llm-synthetic") -> PolicyView:
    start = StateSummary(
        state_id=hashlib.sha256(b"start").hexdigest(),
        metrics={MetricName.WNS_NS: -1.0, MetricName.PACKED_LUTS: 100.0},
        physically_valid=True,
        brief_valid=True,
    )
    cards = (
        WorkerCard(
            id="timing", version="test", recipe_set_id="timing",
            objective=Objective(metric=MetricName.WNS_NS, direction="max"),
        ),
        WorkerCard(
            id="area", version="test", recipe_set_id="area",
            objective=Objective(metric=MetricName.PACKED_LUTS, direction="min"),
        ),
    )
    return PolicyView(
        brief=load_frozen_brief("T0"),
        budget_left=budget,
        start=start,
        current=start,
        best_valid=start,
        visited=(start,),
        worker_cards=cards,
        episode_id=episode_id,
        fake=True,
    )


def _choice(view: PolicyView) -> dict[str, object]:
    return {
        "worker_id": "timing",
        "base_state_id": view.start.state_id,
        "settings": {"preserve_placement": False, "max_runs": 2},
        "guards": [],
    }


def test_view_serialization_and_mandate_schema_are_stable_and_safe() -> None:
    view = _view()
    prompt = serialize_view(view)
    decoded = json.loads(prompt)
    assert decoded["schema_version"] == "llm-policy-view-v1"
    assert decoded["policy_view"]["episode_id"] == view.episode_id
    assert decoded["policy_view"]["brief"]["id"] == "T0"
    assert isinstance(decoded["brief_text"], str) and decoded["brief_text"]
    assert decoded["mandate_choice_schema"] == mandate_json_schema()
    assert decoded["mandate_choice_schema"]["oneOf"][0] == {"type": "null"}
    assert "$defs" in decoded["mandate_choice_schema"]
    assert prompt == serialize_view(view)
    assert prompt_sha256(prompt) != prompt_sha256(
        serialize_view(_view(episode_id="different"))
    )
    assert "recipes/tcl/" not in prompt and "orch.engine.runner" not in prompt


def test_valid_choice_constructs_fixed_mandate_and_logs_usage(tmp_path) -> None:
    view = _view()
    fake = FakeLLM([LLMReply(
        json.dumps(_choice(view)), prompt_tokens=123,
        completion_tokens=27, latency_ms=4.5,
    )])
    policy = LLMPolicyScaffold(fake, PromptResponseCache(tmp_path))
    mandate = policy.decide(view)
    assert mandate is not None
    assert mandate.worker_id == "timing"
    assert mandate.base_state_id == view.start.state_id
    assert mandate.brief_hash == view.brief.fingerprint()
    assert mandate.context_ref == prompt_sha256(serialize_view(view))
    assert mandate.id.startswith("mandate_")
    assert len(fake.prompts) == 1
    log = policy.attempt_log[0]
    assert (log.prompt_tokens, log.completion_tokens, log.latency_ms) == (
        123, 27, 4.5,
    )
    assert not log.cache_hit and log.validation_error is None


def test_one_retry_then_exact_disk_replay_with_zero_client_calls(tmp_path) -> None:
    view = _view()
    wrong = {**_choice(view), "worker_id": "unknown"}
    fake = FakeLLM([
        LLMReply(json.dumps(wrong), prompt_tokens=10, completion_tokens=2),
        LLMReply(json.dumps(_choice(view)), prompt_tokens=20, completion_tokens=3),
    ])
    cache = PromptResponseCache(tmp_path)
    first = LLMPolicyScaffold(fake, cache)
    mandate = first.decide(view)
    assert mandate is not None
    assert len(fake.prompts) == 2
    assert "unknown_worker" in fake.prompts[1]
    assert [row.validation_error for row in first.attempt_log] == [
        "unknown_worker", None,
    ]
    assert not any(row.cache_hit for row in first.attempt_log)
    assert all(row.latency_ms >= 0 for row in first.attempt_log)

    empty = FakeLLM([])
    replay = LLMPolicyScaffold(empty, PromptResponseCache(tmp_path))
    assert replay.decide(view) == mandate
    assert empty.prompts == []
    assert [row.cache_hit for row in replay.attempt_log] == [True, True]
    assert [row.prompt_tokens for row in replay.attempt_log] == [10, 20]


def test_two_invalid_replies_stop_and_loose_guard_is_rejected(tmp_path) -> None:
    view = _view()
    loose = _choice(view)
    loose["guards"] = [{
        "id": "loose", "metric": "packed_luts",
        "comparison": "<=", "threshold": 101.0,
    }]
    fake = FakeLLM([
        LLMReply("{not json"),
        LLMReply(json.dumps(loose)),
    ])
    policy = LLMPolicyScaffold(fake, PromptResponseCache(tmp_path))
    assert policy.decide(view) is None
    assert len(fake.prompts) == 2
    assert [row.validation_error for row in policy.attempt_log] == [
        "invalid_json", "brief_guard_mismatch",
    ]


def test_unvisited_base_and_oversized_attempts_trigger_retry(tmp_path) -> None:
    view = _view(budget=1)
    wrong_base = {**_choice(view), "base_state_id": "f" * 64}
    fake = FakeLLM([
        LLMReply(json.dumps(wrong_base)),
        LLMReply(json.dumps(_choice(view))),
    ])
    policy = LLMPolicyScaffold(fake, PromptResponseCache(tmp_path))
    assert policy.decide(view) is None
    assert [row.validation_error for row in policy.attempt_log] == [
        "base_not_physically_visited", "attempts_exceed_budget",
    ]


def test_stop_and_zero_budget_do_not_retry_or_call_unnecessarily(tmp_path) -> None:
    view = _view()
    fake = FakeLLM([LLMReply("null")])
    policy = LLMPolicyScaffold(fake, PromptResponseCache(tmp_path))
    assert policy.decide(view) is None
    assert len(fake.prompts) == 1
    assert policy.attempt_log[0].validation_error is None

    empty = FakeLLM([])
    zero = LLMPolicyScaffold(empty, PromptResponseCache(tmp_path / "zero"))
    assert zero.decide(_view(budget=0)) is None
    assert empty.prompts == [] and zero.attempt_log == []


def test_response_cache_rejects_record_under_wrong_prompt(tmp_path) -> None:
    cache = PromptResponseCache(tmp_path)
    prompt = serialize_view(_view())
    cache.write(prompt, LLMReply("null"))
    path = tmp_path / f"{prompt_sha256(prompt)}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["prompt"] = "different"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its prompt"):
        cache.read(prompt)
