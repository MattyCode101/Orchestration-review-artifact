"""Tests for the frozen learned policy's entry point (O1, freeze of Friday 13 November 2026). Synthetic views only: no engine, no tool, no design data.

Run with your own base temp:  .venv/bin/python -B -m pytest tests/test_bandit_frozen.py --basetemp=data/bandit_2026-11-11/pytest_frozen -q
"""
from __future__ import annotations

import copy
import json
import pathlib
from pathlib import Path

import pytest
import yaml

from orch.config import load_worker_card
from orch.contract.brief_text import load_frozen_brief
from orch.contract.models import Brief, BriefTerm, Objective
from orch.contract.models import MetricName as M
from orch.contract.views import PolicyView, StateSummary
from orch.policies import bandit_frozen as bf
from orch.policies import bandit_r2 as r2

PROJECT = Path(__file__).resolve().parents[1]
CARDS2 = tuple(load_worker_card(PROJECT / "configs/workers" / f"{n}.yaml") for n in ("timing", "area"))
CARDS3 = tuple(load_worker_card(PROJECT / "configs/workers" / f"{n}.yaml") for n in ("timing", "area", "power"))
NOISE = {M.WNS_NS: 0.1, M.PACKED_LUTS: 10.0, M.DYNAMIC_POWER_W: 0.01}
START = {M.WNS_NS: -0.5, M.PACKED_LUTS: 1000.0, M.DYNAMIC_POWER_W: 1.0, M.TNS_NS: -9.0, M.NETLIST_LUTS: 1200.0, M.FFS: 800.0}
BRIEFS = ("T0", "A0", "T", "A", "L2R", "P0")


def sid(n: int) -> str:
    return f"{n:064x}"


def summary(n: int, *, luts: float = 1000.0, wns: float = -0.5, power: float = 1.0, valid: bool = True) -> StateSummary:
    metrics = dict(START)
    metrics[M.PACKED_LUTS], metrics[M.WNS_NS], metrics[M.DYNAMIC_POWER_W] = luts, wns, power
    return StateSummary(state_id=sid(n), metrics=metrics, physically_valid=True, brief_valid=valid)


def brief_for(brief_id: str) -> Brief:
    if brief_id == "L2R":                       # as the power study defines it (bd_common.l2r_brief)
        terms = (BriefTerm(id="l2r_limit_wns_ns", metric=M.WNS_NS, comparison=">=", factor=1.0, offset=0.0),
                 BriefTerm(id="l2r_limit_dynamic_power_w", metric=M.DYNAMIC_POWER_W, comparison="<=", factor=1.0, offset=0.011))
        return Brief(id="L2R", objective=Objective(metric=M.PACKED_LUTS, direction="min"), terms=terms, budget_runs=6, require_routed=True,
                     max_drc_errors=0, require_constraint_match=True, require_clock_match=True, equivalence_policy="final_only")
    return load_frozen_brief(brief_id).model_copy(update={"budget_runs": 6})


def first_view(brief: Brief, *, budget_left: int = 6) -> PolicyView:
    start = summary(0)
    cards = CARDS3 if brief.id in ("L2R", "P0") else CARDS2
    return PolicyView(brief=brief, budget_left=budget_left, start=start, current=start, best_valid=start, visited=(start,), calls=(),
                      worker_cards=cards, noise_bands=NOISE, episode_id="ep_frozen_test")


@pytest.fixture(autouse=True)
def fresh_cache():
    bf._frozen.cache_clear()
    yield
    bf._frozen.cache_clear()


def test_all_six_briefs_build_and_differ_by_prior():
    versions = set()
    for brief_id in BRIEFS:
        policy = bf.bandit("t2_cordic_s", brief_id, 0)
        assert isinstance(policy, bf.FrozenBandit) and policy.id == r2.POLICY_ID and policy.brief_id == brief_id
        info = policy.provenance()
        assert info["seed_used"] is False and info["prior_sha256"] == policy.prior.sha256 and len(info["fit_designs"]) == 8
        versions.add(policy.version)
    assert len(versions) == 6
    assert set(bf.frozen_briefs()) == set(BRIEFS) and len(bf.frozen_briefs()) == 6


def test_the_prior_is_fitted_on_all_eight_designs_under_the_declared_settings():
    cfg = bf._frozen()["config"]
    assert cfg["policy"]["settings"]["stop_threshold"] == 0.0 and cfg["policy"]["settings"]["repair_feature"] is True
    assert cfg["policy"]["settings"]["observe_empty"] is True
    for brief_id in BRIEFS:
        prior = bf.bandit("x", brief_id).prior
        assert prior.payload["meta"]["fit"] == "all8" and prior.payload["meta"]["held_out"] == "none"
        assert len(prior.payload["meta"]["designs"]) == 8 and prior.sigma >= 0.5
        assert len(prior.payload["arms"]) == 24


def test_w2r_and_unknown_briefs_are_refused():
    with pytest.raises(bf.BanditNotApplicable, match="W2R"):
        bf.bandit("t2_cordic_s", "W2R", 0)
    with pytest.raises(bf.BanditNotApplicable):
        bf.bandit("t2_cordic_s", "vexriscv_re-place", 0)          # a design where the brief belongs
    with pytest.raises(TypeError):
        bf.bandit("", "T0", 0)


def test_design_and_seed_are_inert():
    a, b = bf.bandit("t2_cordic_s", "A0", 0), bf.bandit("rosetta_digit-recognition", "A0", 12345)
    assert a.version == b.version
    view = first_view(brief_for("A0"))
    assert a.decide(view) == b.decide(view) and a.decide(view) is not None


@pytest.mark.parametrize("brief_id", BRIEFS)
def test_decisions_equal_the_unwrapped_policy(brief_id):
    policy = bf.bandit("any", brief_id, 0)
    plain = r2.BanditPolicyR2(prior=policy.prior, settings=policy.settings)
    assert plain.version == policy.version
    for budget in (6, 3, 1, 0):
        view = first_view(brief_for(brief_id), budget_left=budget)
        assert policy.decide(view) == plain.decide(view)


def test_brief_guard():
    policy = bf.bandit("any", "A0", 0)
    with pytest.raises(bf.BriefMismatch, match="frozen for brief A0"):
        policy.decide(first_view(brief_for("T0")))
    changed = brief_for("A0").model_copy(update={"terms": tuple(t.model_copy(update={"comparison": "<="}) for t in brief_for("A0").terms)})
    with pytest.raises(bf.BriefMismatch, match="differs in kind"):
        policy.decide(first_view(changed))
    with pytest.raises(bf.BriefMismatch):
        bf.bandit("any", brief_for("T0").model_copy(update={"id": "A0"}), 0)                 # the object is checked at once
    l2r = bf.bandit("any", "L2R", 0)
    other_band = brief_for("L2R").model_copy(update={"terms": tuple(
        t.model_copy(update={"offset": 0.25}) if t.metric == M.DYNAMIC_POWER_W else t for t in brief_for("L2R").terms)})
    assert l2r.decide(first_view(other_band)) is not None                                    # only the design's power band may differ


def test_hash_checks_fail_closed(tmp_path, monkeypatch):
    cfg = yaml.safe_load(bf.FREEZE_CONFIG.read_text(encoding="utf-8"))
    bad_code = copy.deepcopy(cfg)
    bad_code["code_sha256"]["orch/policies/bandit_r2.py"] = "0" * 64
    path = tmp_path / "bad_code.yaml"
    path.write_text(yaml.safe_dump(bad_code), encoding="utf-8")
    monkeypatch.setattr(bf, "FREEZE_CONFIG", path)
    with pytest.raises(bf.FreezeError, match="bandit_r2.py"):
        bf.bandit("any", "T0", 0)
    bad_prior = copy.deepcopy(cfg)
    bad_prior["briefs"]["T0"]["prior_sha256"] = "f" * 64
    path2 = tmp_path / "bad_prior.yaml"
    path2.write_text(yaml.safe_dump(bad_prior), encoding="utf-8")
    monkeypatch.setattr(bf, "FREEZE_CONFIG", path2)
    bf._frozen.cache_clear()
    with pytest.raises(bf.FreezeError, match="frozen prior"):
        bf.bandit("any", "T0", 0)
    bad_settings = copy.deepcopy(cfg)
    bad_settings["policy"]["settings"]["alpha"] = 2.0
    path3 = tmp_path / "bad_settings.yaml"
    path3.write_text(yaml.safe_dump(bad_settings), encoding="utf-8")
    monkeypatch.setattr(bf, "FREEZE_CONFIG", path3)
    bf._frozen.cache_clear()
    with pytest.raises(bf.FreezeError, match="settings"):
        bf.bandit("any", "T0", 0)


def test_building_the_policy_reads_no_design_data(monkeypatch):
    read: list[str] = []
    real_bytes, real_text = pathlib.Path.read_bytes, pathlib.Path.read_text

    def spy_bytes(self, *a, **k):
        read.append(str(self))
        return real_bytes(self, *a, **k)

    def spy_text(self, *a, **k):
        read.append(str(self))
        return real_text(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "read_bytes", spy_bytes)
    monkeypatch.setattr(pathlib.Path, "read_text", spy_text)
    for brief_id in BRIEFS:
        bf.bandit("rosetta_digit-recognition", brief_id, 0)
    allowed = ("configs/policies/", "orch/policies/", "orch/episode_r2.py", "data/bandit_2026-11-11/priors_frozen/")
    rel = {str(Path(p).resolve().relative_to(PROJECT)) for p in read}
    assert rel and all(r.startswith(allowed) for r in rel), sorted(rel)


def test_run_bandit_episode_uses_both_round2_options(monkeypatch):
    import orch.engine.journal as journal_module
    import orch.episode_r2 as loop

    seen: dict = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return "result"

    class FakeJournal:
        def __init__(self, root):
            pass

        def read(self, episode_id):
            return [{"envelope": {"loop_options": loop.ROUND2.as_json()}}]

    class Engine:
        class store:                      # noqa: N801
            root = Path("/nonexistent")

    monkeypatch.setattr(loop, "run_episode_r2", fake_run)
    monkeypatch.setattr(journal_module, "Journal", FakeJournal)
    brief = brief_for("A0")
    out = bf.run_bandit_episode("any", brief, 3, engine=Engine, registry="reg", workers={}, start_state="s", episode_id="e1", noise=NOISE,
                                validator="v", telemetry_provider=None)
    assert out == "result" and seen["options"] is loop.ROUND2 and isinstance(seen["policy"], bf.FrozenBandit)
    assert seen["brief"] is brief and seen["episode_id"] == "e1" and seen["noise"] == NOISE and seen["policy"].seed == 3
    assert loop.ROUND2.empty_call_observation and loop.ROUND2.stop_rule
    with pytest.raises(TypeError):
        bf.run_bandit_episode("any", "A0", 3, engine=Engine, registry="reg", workers={}, start_state="s", episode_id="e2")

    class WrongJournal(FakeJournal):
        def read(self, episode_id):
            return [{"envelope": {"loop_options": {"empty_call_observation": False, "stop_rule": True}}}]

    monkeypatch.setattr(journal_module, "Journal", WrongJournal)
    with pytest.raises(RuntimeError, match="both round-2 options"):
        bf.run_bandit_episode("any", brief, 3, engine=Engine, registry="reg", workers={}, start_state="s", episode_id="e3", noise=NOISE)


def test_frozen_files_are_the_declared_ones():
    import orch.episode_r2 as loop

    cfg = bf._frozen()["config"]
    assert cfg["not_applicable"].keys() == {"W2R"} and cfg["loop"]["options"] == loop.ROUND2.as_json()
    assert loop.ROUND2.empty_call_observation is True and loop.ROUND2.stop_rule is True
    assert cfg["fit"]["designs"] == sorted(cfg["fit"]["designs"]) and len(cfg["fit"]["designs"]) == 8
    plan = PROJECT / "notes/bandit_test_freeze_plan.md"
    import hashlib
    assert hashlib.sha256(plan.read_bytes()).hexdigest() == cfg["plan_sha256"]
    assert json.loads(json.dumps(cfg["briefs"]["L2R"]["signature"])) == [
        "packed_luts", "min", sorted([["wns_ns", ">=", 1.0, 0.0], ["dynamic_power_w", "<=", 1.0, 0.0]])]
