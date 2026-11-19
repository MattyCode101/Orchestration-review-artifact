"""Campaign harness: cell plan, fail-closed caps, request log, validity rule, pins. No tool, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import campaign_harness as h  # noqa: E402
from orch.contract.models import Evidence, MetricName, MetricValue  # noqa: E402
from orch.policies.llm_client import CachedLLM, LLMParams, LLMReply, ReplayCache, ReplayMiss  # noqa: E402

DESIGNS = tuple(h.settings()["test_designs"])
BRIEFS = ("T0", "A0", "T", "A", "P0")


def test_cell_plan_has_370_cells_and_420_if_not_deterministic() -> None:
    cells = h.make_cells("test", DESIGNS, BRIEFS, repeats=2)
    assert len(cells) == 370 and len({c.episode_id for c in cells}) == 370
    assert len(h.make_cells("test", DESIGNS, BRIEFS, repeats=3)) == 420
    for design in DESIGNS:
        assert sum(c.design == design for c in cells) == 74
        assert sum(c.design == design and c.brief == "P0" for c in cells) == 18
    assert [c.tier for c in cells] == sorted(c.tier for c in cells)           # priority order, ablations last
    tier3 = {c.approach for c in cells if c.tier == 3}
    assert tier3 == {"F1-R1", "GA-R1"} and all(c.tag == "m2" for c in cells if c.tier == 3)
    assert {c.approach for c in cells if c.tier == 2} == {"E0-R1", "F1-R0"}
    assert {c.approach for c in cells if c.brief == "P0" and c.approach.startswith("fixed_")} == {
        f"fixed_{p}" for p in h.PERMS}
    assert {c.approach for c in cells if c.brief == "T0" and c.approach.startswith("fixed_")} == {"fixed_AB", "fixed_BA"}


def test_every_non_llm_approach_builds_a_policy_and_v2_is_g05() -> None:
    for brief in BRIEFS:
        for approach, _tag, _tier in h.approaches(brief):
            if approach not in h.settings()["llm_cells"]:
                assert h.policy_for_approach(approach, brief) is not None
    v2 = h.policy_for_approach("v2", "A0")
    assert (v2.repair_attempts, v2.repair_preserve_placement, v2.gentle_step, v2.after_failed_repair) == (1, False, True, "rollback")
    assert h.policy_for_approach("v1", "A0").variant == (1, True, True, "rollback")


def test_the_model_is_one_value_in_the_dev_yaml_not_in_the_campaign_yaml() -> None:
    assert "model" not in h.settings()
    cfg, _ = h.llm_config("F1-R1")
    assert cfg["model"] and cfg["templates"]["orchestrator"].endswith("orchestrator_v2.txt")
    assert h.llm_config("F1-R0")[0]["templates"]["orchestrator"].endswith("orchestrator_v1.txt")
    assert h.llm_config("GA-R1")[1]["kind"] == "general"


def _caps(tmp_path: Path, **over) -> h.CapLedger:
    caps = {"new_real_runs": 2, "job_hours": 1.0, "tokens": 100, "usd": 1.0, **over}
    return h.CapLedger(tmp_path, caps, "plan-a")


def test_run_cap_fails_closed_and_stays_closed(tmp_path: Path) -> None:
    caps = _caps(tmp_path)
    caps.reserve_run("e1", "run_a", "s", "r")
    caps.reserve_run("e1", "run_b", "s", "r")
    with pytest.raises(h.CapReached):
        caps.reserve_run("e2", "run_c", "s", "r")
    with pytest.raises(h.CapReached):
        caps.reserve_run("e3", "run_d", "s", "r")
    assert caps.used()["new_real_runs"] == 2 and "new_real_runs" in caps.stopped()


def test_job_hour_cap(tmp_path: Path) -> None:
    caps = _caps(tmp_path, new_real_runs=10)
    caps.reserve_run("e1", "run_a", "s", "r")
    caps.add_job_seconds(3600.0)
    with pytest.raises(h.CapReached):
        caps.reserve_run("e1", "run_b", "s", "r")


def test_token_and_dollar_caps_stop_a_fresh_call_before_it_is_made(tmp_path: Path) -> None:
    for which, kw in (("tokens", {"tokens": 10}), ("usd", {"usd": 0.001})):
        caps = _caps(tmp_path / which, **kw)
        caps.add_llm(11, 0.002)
        with pytest.raises(h.CapReached):
            caps.check_llm()
        assert which in caps.stopped()


def test_cap_ledger_refuses_other_caps_or_plan(tmp_path: Path) -> None:
    _caps(tmp_path)
    with pytest.raises(Exception):
        h.CapLedger(tmp_path, {"new_real_runs": 3, "job_hours": 1.0, "tokens": 100, "usd": 1.0}, "plan-a")
    with pytest.raises(Exception):
        h.CapLedger(tmp_path, {"new_real_runs": 2, "job_hours": 1.0, "tokens": 100, "usd": 1.0}, "plan-b")


class _Fake:
    """Transport returning a fixed reply with known token counts."""

    calls = 0

    def complete(self, prompt, params, schema):
        _Fake.calls += 1
        return LLMReply(text='{"ok": true}', prompt_tokens=1000, completion_tokens=100, thoughts_tokens=50,
                        total_tokens=1150, latency_ms=1.0, finish_reason="STOP", model=params.model)


def _llm(tmp_path: Path, mode: str, caps, tag: str | None = "m1") -> h.LedgeredLLM:
    cache = ReplayCache(tmp_path / "cache", mode=mode, fresh_tag=tag if mode == "fresh" else None)
    inner = CachedLLM(_Fake() if mode != "replay" else None, cache, LLMParams(model="unit-model"))
    return h.LedgeredLLM(inner, caps, tmp_path / "calls.jsonl", "ep1", {"input_per_million": 0.25, "output_per_million": 1.5})


def test_fresh_call_is_charged_logged_with_hashes_and_stored(tmp_path: Path) -> None:
    caps = _caps(tmp_path, tokens=10_000, usd=5.0)
    _Fake.calls = 0
    llm = _llm(tmp_path, "fresh", caps)
    first = llm.complete("hello", None)
    assert not first.cache_hit and _Fake.calls == 1
    again = llm.complete("hello", None)                      # a resumed decision reads its own stored reply
    assert again.cache_hit and _Fake.calls == 1
    used = caps.used()
    assert used["tokens"] == 1150 and abs(used["usd"] - (1000 * 0.25 + 150 * 1.5) / 1e6) < 1e-9
    lines = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [x["charged"] for x in lines] == [True, False]
    assert lines[0]["prompt_sha256"] == h.sha("hello") and lines[0]["reply_sha256"] == h.sha('{"ok": true}')
    record = json.loads(Path(lines[0]["record"]).read_text())
    assert record["prompt"] == "hello" and record["reply"]["text"] == '{"ok": true}'    # request and reply stored
    caps.add_llm(10_000, 0.0)
    with pytest.raises(h.CapReached):
        llm.complete("another prompt", None)
    assert _Fake.calls == 1                                    # the capped call was never made


def test_replay_mode_never_calls_and_never_charges(tmp_path: Path) -> None:
    caps = _caps(tmp_path)
    _llm(tmp_path, "record", None).complete("p", None)         # store one reply
    replay = _llm(tmp_path, "replay", caps)
    assert replay.complete("p", None).cache_hit
    with pytest.raises(ReplayMiss):
        replay.complete("not stored", None)
    assert caps.used()["tokens"] == 0


def _evidence(**props) -> Evidence:
    counts = {"n_DONT_TOUCH": 0, "n_IS_LOC_FIXED": 0, "n_IS_BEL_FIXED": 0, **props}
    return Evidence(
        id="e" * 64, state_id="s" * 64, routed=True, route_errors=0, drc_errors=0, constraint_fingerprint="x",
        clock_digest="y", meter_version="m", fake=False, v_props=counts,
        metrics={MetricName.WNS_NS: MetricValue(value=0.0, unit="ns", method="u", basis="routed")},
    )


def test_validity_rule_is_start_relative() -> None:
    h.start_relative_check(_evidence(), _evidence())                                  # zero against zero
    h.start_relative_check(_evidence(n_DONT_TOUCH=5), _evidence(n_DONT_TOUCH=5))       # equal to the start: fine
    h.start_relative_check(_evidence(n_DONT_TOUCH=4), _evidence(n_DONT_TOUCH=5))       # fewer: fine
    with pytest.raises(h.pilot.PilotSafetyStop):
        h.start_relative_check(_evidence(n_DONT_TOUCH=6), _evidence(n_DONT_TOUCH=5))   # more: refused
    with pytest.raises(h.pilot.PilotSafetyStop):
        h.start_relative_check(_evidence(n_IS_BEL_FIXED=1), _evidence())
    with pytest.raises(h.pilot.PilotSafetyStop):
        h.start_relative_check(_evidence().model_copy(update={"v_props": None}), _evidence())


def test_journal_normalisation_ignores_run_ids_but_not_content() -> None:
    a = [{"type": "step", "step": {"run_id": "run_" + "a" * 32, "id": "step_" + "1" * 64}, "x": 1}]
    b = [{"type": "step", "step": {"run_id": "run_" + "b" * 32, "id": "step_" + "2" * 64}, "x": 1}]
    c = [{"type": "step", "step": {"run_id": "run_" + "b" * 32, "id": "step_" + "2" * 64}, "x": 2}]
    assert h.first_difference(h.normalised_events(a), h.normalised_events(b)) is None
    assert h.first_difference(h.normalised_events(a), h.normalised_events(c)) == "[0]/x: 1 vs 2"


def test_pin_changes_name_the_file_that_changed() -> None:
    start = h.imported_hashes()
    assert h.pin_changes(start) == []
    forged = dict(start)
    forged["configs/briefs.yaml"] = "0" * 64
    assert h.pin_changes(forged) == ["configs/briefs.yaml"]
    assert "scripts/campaign_harness.py" in start and "configs/campaign/campaign.yaml" in start
    assert "orch/policies/llm.py" in start and "orch/baselines/general_agent.py" in start
