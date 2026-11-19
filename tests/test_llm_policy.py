"""LLMPolicy: mapping, validation, repair turns, fallback, telemetry, and whole fake-backend episodes.

Acceptance tests 4 to 8 of the specification (aliases and branching, continue from a violating
state, repair and fallback, replay determinism, crash and resume).
"""
from __future__ import annotations

import json

import pytest

import llm_support as ls
from orch.contract.models import Guard, MetricName
from orch.engine.journal import Journal
from orch.episode import run_episode
from orch.policies.llm_client import ReplayMiss, TelemetrySink
from orch.policies.llm_view import build_fuzzy_view

NOISE = {MetricName.WNS_NS: 0.05, MetricName.PACKED_LUTS: 5.0}
decision = ls.decision


def _tree_view(budget_left: int = 4, calls=None):
    s0 = ls.state(10, -0.347, 1337)
    s1 = ls.state(11, -0.147, 1352, brief_valid=False)
    s2 = ls.state(12, -0.357, 1301)
    calls = calls if calls is not None else (
        ls.call(1, "timing", 10, 11, reason="largest goal move first"),
        ls.call(2, "area", 10, 12, preserve=True, attempts=1),
    )
    steps = (ls.step("timing", 10, 11), ls.step("area", 10, 12, preserve=True, attempts=1))
    return ls.view(visited=[s0, s1, s2], calls=calls, steps=steps, budget_left=budget_left, current=s2, best=s2)


def test_a_valid_reply_becomes_a_mandate_on_the_chosen_alias(tmp_path) -> None:
    policy, transport, _ = ls.make_policy(tmp_path, decision("area", "S1", preserve=True, attempts=2, reason="repair the limit"))
    view = _tree_view()
    mandate = policy.decide(view)
    assert mandate.worker_id == "area" and mandate.base_state_id == ls.sid(11)
    assert mandate.settings.preserve_placement is True and mandate.settings.max_runs == 2
    assert mandate.guards == () and mandate.brief_hash == view.brief.fingerprint()
    ref = json.loads(mandate.context_ref)
    assert ref["s"] == "PUSH" and ref["r"] == "repair the limit" and len(ref["p"]) == 16
    assert len(transport.prompts) == 1


def test_choosing_the_start_alias_after_other_states_exist_resolves_to_the_start_state(tmp_path) -> None:
    policy, _, _ = ls.make_policy(tmp_path, decision("area", "S0", preserve=False, attempts=1))
    mandate = policy.decide(_tree_view())
    assert mandate.base_state_id == ls.sid(10)


def test_respecting_the_brief_limits_becomes_a_guard_equal_to_the_threshold(tmp_path) -> None:
    policy, _, _ = ls.make_policy(tmp_path, decision("timing", "S2", respect=True, attempts=1))
    mandate = policy.decide(_tree_view())
    assert mandate.guards == (
        Guard(id="respect_t0_limit", metric=MetricName.PACKED_LUTS, comparison="<=", threshold=1337.0),
    )
    free, _, _ = ls.make_policy(tmp_path, decision("timing", "S2", respect=False, attempts=1), telemetry="b.jsonl", cache_dir="c2")
    assert free.decide(_tree_view()).guards == ()


@pytest.mark.parametrize(("bad_reply", "expected"), [
    (decision("area", "S9"), "base_state 'S9' is not one of the listed states"),
    (decision("mystery", "S0"), "worker_id 'mystery' is not one of the listed workers"),
    (decision("timing", "S0", attempts=3), "attempts 3 exceeds the remaining budget of 2"),
    (decision("timing", "S0", preserve=False), "already tried"),
    ("this is not json", "not valid JSON"),
    ('{"action": "INVOKE_WORKER"}', "Field required"),
])
def test_a_rejected_reply_is_repaired_on_the_second_turn_with_the_exact_error(tmp_path, bad_reply, expected) -> None:
    good = decision("area", "S1", preserve=True, attempts=1)
    policy, transport, sink = ls.make_policy(tmp_path, bad_reply, good)
    mandate = policy.decide(_tree_view(budget_left=2))
    assert mandate is not None and mandate.base_state_id == ls.sid(11)
    assert len(transport.prompts) == 2
    assert transport.prompts[1].startswith(transport.prompts[0])
    assert expected in transport.prompts[1]
    record = TelemetrySink.read(sink.path)[0]
    assert record["repair_attempts"] == 1 and record["fallback_triggered"] is False


def test_three_bad_replies_stop_and_log_the_fallback(tmp_path) -> None:
    policy, transport, sink = ls.make_policy(tmp_path, "nope", decision("timing", "S9"), "still nope")
    assert policy.decide(_tree_view()) is None
    assert len(transport.prompts) == 3
    record = TelemetrySink.read(sink.path)[0]
    assert record["fallback_triggered"] is True and record["repair_attempts"] == 2 and record["action"] == "STOP"
    assert len(record["calls"]) == 3 and "last_error" in record


def test_a_stop_action_returns_none_without_a_fallback(tmp_path) -> None:
    policy, _, sink = ls.make_policy(tmp_path, decision(action="STOP", reason="nothing untried is likely to help"))
    assert policy.decide(_tree_view()) is None
    record = TelemetrySink.read(sink.path)[0]
    assert record["action"] == "STOP" and record["fallback_triggered"] is False and record["strategy"] == "STOP"


def test_no_model_call_is_made_when_the_budget_is_gone(tmp_path) -> None:
    policy, transport, sink = ls.make_policy(tmp_path)
    assert policy.decide(_tree_view(budget_left=0)) is None
    assert transport.prompts == [] and TelemetrySink.read(sink.path) == []


def test_telemetry_records_everything_the_specification_lists(tmp_path) -> None:
    policy, _, sink = ls.make_policy(tmp_path, decision("area", "S1", preserve=True, attempts=2, strategy="REPAIR", reason="fix the limit"),
                                     lessons="- repairs by the owner of a limit often work.")
    view = _tree_view()
    policy.decide(view)
    record = TelemetrySink.read(sink.path)[0]
    built = build_fuzzy_view(view, mode="fuzzy")
    assert record["model"] == "scripted-model" and record["temperature"] == 0.0 and record["config"] == "F1"
    assert len(record["template_sha256"]) == 64 and record["profile_sha256"] == ls.PROFILE_SHA
    assert len(record["lessons_sha256"]) == 64 and record["view_sha256"] == built.sha256
    assert record["strategy"] == "REPAIR" and record["rationale"] == "fix the limit"
    assert record["diagnosis"] == "diagnosis" and record["conflict_check"] == "no gains to undo"
    assert record["mandate"] == {"worker": "area", "base": "S1", "preserve_placement": True, "attempts": 2, "respect_brief_limits": False}
    assert record["tokens"] == {"prompt": 100, "completion": 20, "thoughts": 0, "total": 120, "fresh_total": 120}
    call = record["calls"][0]
    assert call["cache_hit"] is False and call["cache_mode"] == "record" and call["latency_ms"] == 5.0
    assert record["decision_index"] == 2 and record["repair_attempts"] == 0


def test_profiles_off_sends_no_profile_text_and_records_no_profile_hashes(tmp_path) -> None:
    policy, transport, sink = ls.make_policy(tmp_path, decision("timing", "S0"), profiles_on=False)
    policy.decide(_tree_view(calls=()))
    assert "goal within_noise usually" not in transport.prompts[0] and "Worker timing" in transport.prompts[0]
    assert TelemetrySink.read(sink.path)[0]["profile_sha256"] is None


def test_the_policy_version_changes_with_template_profiles_lessons_and_mode(tmp_path) -> None:
    versions = {
        ls.make_policy(tmp_path, cache_dir="a")[0].version,
        ls.make_policy(tmp_path, profiles_on=False, cache_dir="b")[0].version,
        ls.make_policy(tmp_path, lessons="- one lesson", cache_dir="c")[0].version,
        ls.make_policy(tmp_path, mode="exact", cache_dir="d")[0].version,
    }
    assert len(versions) == 4


def test_the_same_view_gives_the_same_mandate_from_the_replay_cache(tmp_path) -> None:
    policy, transport, _ = ls.make_policy(tmp_path, decision("area", "S1", preserve=True, attempts=1))
    first = policy.decide(_tree_view())
    second = policy.decide(_tree_view())
    assert first == second and len(transport.prompts) == 1


def test_the_prompt_sent_to_the_model_is_anonymous(tmp_path) -> None:
    policy, transport, _ = ls.make_policy(tmp_path, decision("timing", "S2"))
    view = _tree_view()
    policy.decide(view)
    prompt = transport.prompts[0]
    assert ls.DESIGN_SECRET not in prompt and view.episode_id not in prompt and "mandate_" not in prompt
    assert ls.sid(10) not in prompt and "1337" not in prompt


# ------------------------------------------------------------------ whole episodes on the fake backend
def _scenario() -> list[str]:
    return [
        decision("timing", "S0", preserve=False, attempts=2, reason="largest goal move first"),
        decision("area", "S1", preserve=True, attempts=1, strategy="REPAIR", reason="repair the broken limit from the violating state"),
        decision("timing", "S0", preserve=True, attempts=1, strategy="BACKTRACK", reason="try the start state again with placement kept"),
        decision(action="STOP", reason="nothing untried is likely to help"),
    ]


def _run(tmp_path, policy, *, engine, registry, workers, start, episode_id="llm_unit"):
    from orch.contract.brief_text import load_frozen_brief

    return run_episode(
        engine=engine, registry=registry, workers=workers, policy=policy, brief=load_frozen_brief("T0"),
        start_state=start, episode_id=episode_id, noise=NOISE,
    )


def test_branching_and_continuing_from_a_violating_state_run_in_the_real_episode_loop(tmp_path) -> None:
    engine, registry, workers, backend, start = ls.episode_setup(tmp_path)
    policy, transport, sink = ls.make_policy(tmp_path, *_scenario())
    result = _run(tmp_path, policy, engine=engine, registry=registry, workers=workers, start=start)
    events = Journal(engine.store.root).read("llm_unit")
    mandates = [e["mandate"] for e in events if e["type"] == "mandate"]
    assert [m["worker_id"] for m in mandates] == ["timing", "area", "timing"]
    steps = [e["step"] for e in events if e["type"] == "step"]
    first_pick = steps[0]["output_state_id"]                      # t1: the best timing result, which breaks the LUT limit
    assert mandates[1]["base_state_id"] == first_pick             # acceptance 5: a brief-violating base validates and runs
    assert mandates[2]["base_state_id"] == start.id               # acceptance 4: S0 chosen again after S1.. exist
    assert result.stop_reason == "policy_stop" and result.recipe_runs == 4
    assert result.final_verdict.brief_valid and result.final_state.id == steps[2]["output_state_id"]
    assert backend.calls == ["t1", "t2", "a1", "t3"]
    # the third prompt shows the earlier states and what was tried from the start
    assert "S1" in transport.prompts[2] and "tried:" in transport.prompts[2]
    assert "repair the broken limit from the violating state" in transport.prompts[2]
    assert len(TelemetrySink.read(sink.path)) == 4


def test_replay_reproduces_the_same_decisions_and_final_pick(tmp_path) -> None:
    one = tmp_path / "one"
    one.mkdir()
    engine, registry, workers, _, start = ls.episode_setup(one)
    policy, _, _ = ls.make_policy(one, *_scenario(), cache_dir="shared_cache")
    first = _run(one, policy, engine=engine, registry=registry, workers=workers, start=start)
    two = tmp_path / "two"
    two.mkdir()
    engine2, registry2, workers2, _, start2 = ls.episode_setup(two)
    replay, _, _ = ls.make_policy(two, cache_mode="replay", cache_dir="../one/shared_cache")
    second = _run(two, replay, engine=engine2, registry=registry2, workers=workers2, start=start2)

    def trace(engine_, result):
        events = Journal(engine_.store.root).read("llm_unit")
        return [(e["mandate"]["worker_id"], e["mandate"]["base_state_id"], e["mandate"]["settings"]) for e in events if e["type"] == "mandate"]

    assert trace(engine, first) == trace(engine2, second)
    assert first.final_state.id == second.final_state.id and first.recipe_runs == second.recipe_runs
    assert [s.output_state_id for s in first.steps] == [s.output_state_id for s in second.steps]


def test_replay_mode_fails_loudly_when_a_prompt_was_never_recorded(tmp_path) -> None:
    engine, registry, workers, _, start = ls.episode_setup(tmp_path)
    policy, _, _ = ls.make_policy(tmp_path, cache_mode="replay")
    with pytest.raises(ReplayMiss):
        _run(tmp_path, policy, engine=engine, registry=registry, workers=workers, start=start)


def test_crash_and_resume_give_the_same_view_correct_token_totals_and_a_valid_pick(tmp_path) -> None:
    full = tmp_path / "full"
    full.mkdir()
    engine0, registry0, workers0, _, start0 = ls.episode_setup(full)
    policy0, _, sink0 = ls.make_policy(full, *_scenario())
    uninterrupted = _run(full, policy0, engine=engine0, registry=registry0, workers=workers0, start=start0)

    broken = tmp_path / "broken"
    broken.mkdir()
    engine, registry, workers, backend, start = ls.episode_setup(broken, fail_on_call=4)   # dies in the fourth recipe run
    policy, _, sink = ls.make_policy(broken, *_scenario())
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(broken, policy, engine=engine, registry=registry, workers=workers, start=start)
    assert len(TelemetrySink.read(sink.path)) == 3                      # three decisions were made before the crash
    resumed_policy, resumed_transport, _ = ls.make_policy(broken, _scenario()[3], cache_dir="llm_cache")
    resumed = _run(broken, resumed_policy, engine=engine, registry=registry, workers=workers, start=start)
    assert resumed.resumed is True
    records = TelemetrySink.read(sink.path)
    assert len(records) == 4 and sum(r["tokens"]["total"] for r in records) == 4 * 120   # no decision was paid for twice
    assert len(resumed_transport.prompts) == 1                           # only the final decision needed the model again
    full_records = TelemetrySink.read(sink0.path)
    assert records[3]["view_sha256"] == full_records[3]["view_sha256"]   # identical view after resume
    assert resumed.final_state.id == uninterrupted.final_state.id and resumed.final_verdict.brief_valid
    assert resumed.recipe_runs == uninterrupted.recipe_runs == 4
