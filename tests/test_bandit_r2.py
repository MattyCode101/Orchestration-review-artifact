"""Tests for the round-2 contextual-bandit policy (O1, the authors' decisions of 12 Nov). Synthetic views only: no engine, no tool.

Run with your own base temp:  .venv/bin/python -B -m pytest tests/test_bandit_r2.py --basetemp=data/bandit_2026-11-11/pytest_r2 -q
"""
from __future__ import annotations

import dataclasses
import json
import math
import random
import sys
from pathlib import Path

import pytest

from orch.config import load_worker_card
from orch.contract.brief_text import load_frozen_brief
from orch.contract.models import MetricName as M
from orch.contract.models import VerdictStatus
from orch.contract.views import CallSummary, PolicyView, StateSummary
from orch.policies import bandit as r1
from orch.policies import bandit_r2 as r2

PROJECT = Path(__file__).resolve().parents[1]
CARDS = tuple(load_worker_card(PROJECT / "configs/workers" / f"{n}.yaml") for n in ("timing", "area"))
NOISE = {M.WNS_NS: 0.1, M.PACKED_LUTS: 10.0}
START = {M.WNS_NS: -0.5, M.PACKED_LUTS: 1000.0, M.DYNAMIC_POWER_W: 1.0, M.TNS_NS: -9.0, M.NETLIST_LUTS: 1200.0, M.FFS: 800.0}


def sid(n: int) -> str:
    return f"{n:064x}"


def state(n: int, *, luts: float, wns: float, valid: bool | None = None, physical: bool = True) -> StateSummary:
    metrics = dict(START)
    metrics[M.PACKED_LUTS], metrics[M.WNS_NS] = luts, wns
    ok = (wns >= START[M.WNS_NS]) if valid is None else valid          # A0: WNS must not fall below the start
    return StateSummary(state_id=sid(n), metrics=metrics, physically_valid=physical, brief_valid=ok)


def view(visited, current=None, best=None, calls=(), budget_left=6, brief_id="A0") -> PolicyView:
    brief = load_frozen_brief(brief_id)
    ranked = sorted((s for s in visited if s.brief_valid), key=lambda s: s.metrics[M.PACKED_LUTS])
    return PolicyView(
        brief=brief, budget_left=budget_left, start=visited[0], current=current or visited[-1],
        best_valid=best or ranked[0], visited=tuple(visited), calls=tuple(calls), worker_cards=CARDS,
        noise_bands=NOISE, episode_id="ep_test",
    )


def prior2(sigma: float = 1.0, arms: dict | None = None) -> r2.Prior2:
    payload = {"schema": r2.SCHEMA, "features": list(r2.FEATURES), "ridge": 1.0, "sigma": sigma, "arms": arms or {}}
    return r2.Prior2(payload, "0" * 64)


def policy2(**kw) -> r2.BanditPolicyR2:
    prior = kw.pop("prior_", prior2())
    return r2.BanditPolicyR2(prior=prior, settings=r2.R2Settings.for_variant("main", **kw))


def call_of(mandate, *, selected=None, status=None, runs=None) -> CallSummary:
    return CallSummary(
        mandate_id=mandate.id, worker_id=mandate.worker_id, base_state_id=mandate.base_state_id, settings=mandate.settings,
        selected_state_id=selected, selected_verdict_status=status,
        recipe_runs=mandate.settings.max_runs if runs is None else runs, context_ref=mandate.context_ref,
    )


# ------------------------------------------------------------------------------------------ the feature
def test_repair_pending_needs_a_limit_break_with_a_larger_gain_than_the_best_valid_state():
    s0 = state(0, luts=1000.0, wns=-0.5)
    assert not r2.repair_pending(view([s0]))                                               # nothing visited but the start
    broken = state(1, luts=900.0, wns=-0.62)                                              # breaks the A0 limit, gain 10 bands
    assert r2.repair_pending(view([s0, broken]))
    better_valid = state(2, luts=800.0, wns=-0.45)                                        # valid, gain 20 bands
    assert not r2.repair_pending(view([s0, broken, better_valid]))                         # the broken gain is no longer larger
    smaller_valid = state(3, luts=950.0, wns=-0.4)                                         # valid, gain 5 bands
    assert r2.repair_pending(view([s0, broken, smaller_valid]))
    assert not r2.repair_pending(view([s0, state(4, luts=900.0, wns=-0.62, physical=False)]))   # a dead state cannot be repaired
    assert not r2.repair_pending(view([s0, state(5, luts=1005.0, wns=-0.62)]))            # broken, but no gain at all


def test_repair_pending_counts_an_unpicked_attempt_that_never_became_current():
    s0 = state(0, luts=1000.0, wns=-0.5)
    unpicked = state(1, luts=850.0, wns=-0.7)                                              # a second attempt, not selected
    picked = state(2, luts=980.0, wns=-0.5)
    v = view([s0, unpicked, picked], current=picked, best=picked)
    assert r2.repair_pending(v)


def test_features_have_eleven_numbers_and_the_new_one_is_last():
    s0 = state(0, luts=1000.0, wns=-0.5)
    s1 = state(1, luts=900.0, wns=-0.62)
    v = view([s0, s1], current=s1, best=s0, budget_left=4)
    x = r2.features(v, s1, "timing", 2)
    assert len(x) == r2.DIM == 11 and r2.FEATURES[-1] == "repair" and x[-1] == 1.0
    assert x[:10] == r1.features(v, s1, "timing", 2)
    assert r2.features(v, s1, "timing", 2, repair=False)[-1] == 0.0
    assert r2.features(view([s0]), s0, "timing", 1)[-1] == 0.0


# --------------------------------------------------------------------------------------- empty calls
def empty_call(worker: str, preserve: bool, base: str, attempts: int = 1) -> CallSummary:
    from orch.contract.models import WorkerSettings
    return CallSummary(mandate_id=f"m_{worker}_{preserve}", worker_id=worker, base_state_id=base,
                       settings=WorkerSettings(preserve_placement=preserve, max_runs=attempts), selected_state_id=None,
                       selected_verdict_status=None, recipe_runs=0, context_ref="")


def test_an_empty_call_removes_that_worker_and_scope_for_that_state_only():
    s0 = state(0, luts=1000.0, wns=-0.5)
    s1 = state(1, luts=900.0, wns=-0.4)
    v = view([s0, s1], current=s1, best=s1, calls=[empty_call("area", True, sid(1))], budget_left=4)
    left = {(a[1], a[2], a[5].state_id) for a in r2.enumerate_arms(v, "best_or_latest")}
    assert ("area", True, sid(1)) not in left                                              # both attempts are gone
    assert ("area", False, sid(1)) in left and ("timing", True, sid(1)) in left and ("timing", False, sid(1)) in left
    assert ("area", True, sid(1)) in {(a[1], a[2], a[5].state_id) for a in r2.enumerate_arms(v, "best_or_latest", False)}
    other = view([s0, s1], current=s1, best=s1, calls=[empty_call("area", True, sid(0))], budget_left=4)
    assert ("area", True, sid(1)) in {(a[1], a[2], a[5].state_id) for a in r2.enumerate_arms(other, "best_or_latest")}


def test_with_every_scope_empty_the_policy_stops():
    s0 = state(0, luts=1000.0, wns=-0.5)
    calls = [empty_call(w, p, sid(0)) for w in ("timing", "area") for p in (False, True)]
    v = view([s0], calls=calls)
    assert policy2().decide(v) is None
    assert policy2(observe_empty=False).decide(v) is not None                              # round 1 would not see it


def test_the_policy_never_repeats_an_emptied_scope():
    p = policy2()
    s0 = state(0, luts=1000.0, wns=-0.5)
    seen: set = set()
    calls: list[CallSummary] = []
    for _ in range(4):
        m = p.decide(view([s0], calls=calls))
        assert m is not None
        key = (m.worker_id, m.settings.preserve_placement, m.base_state_id)
        assert key not in seen
        seen.add(key)
        calls.append(call_of(m, runs=0))                                                    # the contract answers: no recipes
    assert p.decide(view([s0], calls=calls)) is None


def test_empty_calls_are_not_reward_samples_and_rewards_still_telescope():
    p = policy2()
    s0 = state(0, luts=1000.0, wns=-0.5)
    v0 = view([s0])
    m0 = p.decide(v0)
    s1 = state(1, luts=900.0, wns=-0.62)
    call0 = call_of(m0, selected=s1.state_id, status=VerdictStatus.BRIEF_VIOLATION)
    v1 = view([s0, s1], current=s1, best=s0, calls=[call0], budget_left=6 - m0.settings.max_runs)
    m1 = p.decide(v1)
    assert m1 is not None
    empty1 = call_of(m1, runs=0)                                                            # nothing left for that scope
    v2 = view([s0, s1], current=s1, best=s0, calls=[call0, empty1], budget_left=6 - m0.settings.max_runs)
    assert len(p.rewards(v2)) == 1                                                          # only the call that ran
    m2 = p.decide(v2)
    assert m2 is not None and (m2.worker_id, m2.settings.preserve_placement, m2.base_state_id) != (
        m1.worker_id, m1.settings.preserve_placement, m1.base_state_id)
    s2 = state(2, luts=902.0, wns=-0.40)
    call2 = call_of(m2, selected=s2.state_id, status=VerdictStatus.ACCEPTABLE)
    left = 6 - m0.settings.max_runs - m2.settings.max_runs
    v3 = view([s0, s1, s2], current=s2, best=s2, calls=[call0, empty1, call2], budget_left=left)
    total = sum(r for _, r in p.rewards(v3))
    assert math.isclose(total, r1.credited_gain(v3, 0.5) - r1.credited_gain(v0, 0.5), abs_tol=1e-12)


def test_a_resumed_episode_with_an_empty_call_makes_the_same_choice():
    p, q = policy2(), policy2()
    s0 = state(0, luts=1000.0, wns=-0.5)
    m0 = p.decide(view([s0]))
    v1 = view([s0], calls=[call_of(m0, runs=0)])
    assert p.decide(v1) == q.decide(v1)


def test_a_call_with_a_foreign_context_is_refused():
    s0 = state(0, luts=1000.0, wns=-0.5)
    m0 = policy2().decide(view([s0]))
    foreign = CallSummary(mandate_id="m", worker_id="area", base_state_id=sid(0), settings=m0.settings, selected_state_id=sid(0),
                          recipe_runs=1, context_ref="")
    with pytest.raises(ValueError):
        policy2().decide(view([s0], calls=[foreign], budget_left=5))


# ------------------------------------------------------------------------------------- round-1 equality
def _old_and_padded_priors(seed: int = 7):
    """The same statistics in the round-1 layout (10 features) and padded to the round-2 layout (the 11th feature unused)."""

    rng = random.Random(seed)
    arms10, arms11 = {}, {}
    for key in ("timing|free|1|best", "timing|keep|2|best", "area|free|1|best", "area|keep|1|best", "area|keep|2|latest",
                "timing|free|2|latest", "area|free|2|best", "timing|keep|1|best"):
        a = [[1.0 if i == j else 0.0 for j in range(10)] for i in range(10)]
        b = [0.0] * 10
        n = rng.randint(3, 9)
        for _ in range(n):
            x = [1.0] + [rng.uniform(0.0, 1.5) for _ in range(9)]
            reward = rng.uniform(-1.0, 3.0)
            for i in range(10):
                for j in range(10):
                    a[i][j] += x[i] * x[j]
                b[i] += reward * x[i]
        arms10[key] = {"A": a, "b": b, "n": n}
        a11 = [row + [0.0] for row in a] + [[0.0] * 10 + [1.0]]
        arms11[key] = {"A": a11, "b": b + [0.0], "n": n}
    old = r1.Prior({"schema": r1.SCHEMA, "features": list(r1.FEATURES), "ridge": 1.0, "sigma": 1.7, "arms": arms10}, "1" * 64)
    new = r2.Prior2({"schema": r2.SCHEMA, "features": list(r2.FEATURES), "ridge": 1.0, "sigma": 1.7, "arms": arms11}, "2" * 64)
    return old, new


def _views():
    s0 = state(0, luts=1000.0, wns=-0.5)
    s1 = state(1, luts=900.0, wns=-0.62)
    s2 = state(2, luts=902.0, wns=-0.40)
    yield view([s0])
    yield view([s0], budget_left=1)
    yield view([s0, s1], current=s1, best=s0, budget_left=4)
    yield view([s0, s1, s2], current=s2, best=s2, budget_left=3)
    yield view([s0, s2], current=s2, best=s2, budget_left=2, brief_id="A")
    yield view([s0], brief_id="T0")


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_with_both_switches_off_the_choices_are_round_one(threshold):
    old, new = _old_and_padded_priors()
    p1 = r1.BanditPolicy(prior=old, settings=r1.BanditSettings(stop_threshold=threshold))
    p2 = r2.BanditPolicyR2(prior=new, settings=r2.R2Settings(stop_threshold=threshold, repair_feature=False, observe_empty=False))
    for v in _views():
        rows1, rows2 = p1.scored(v), p2.scored(v)
        assert [(r["key"], r["base_id"]) for r in rows1] == [(r["key"], r["base_id"]) for r in rows2]
        for a, b in zip(rows1, rows2):
            assert (a["mean"], a["bonus"], a["ucb"]) == (b["mean"], b["bonus"], b["ucb"])        # bit for bit
        m1, m2 = p1.decide(v), p2.decide(v)
        assert (m1 is None) == (m2 is None)
        if m1 is not None:
            assert (m1.worker_id, m1.base_state_id, m1.settings) == (m2.worker_id, m2.base_state_id, m2.settings)


def test_the_repair_feature_changes_the_estimate_only_through_its_own_weight():
    old, new = _old_and_padded_priors()
    s0 = state(0, luts=1000.0, wns=-0.5)
    s1 = state(1, luts=900.0, wns=-0.62)
    v = view([s0, s1], current=s1, best=s0, budget_left=4)
    off = r2.BanditPolicyR2(prior=new, settings=r2.R2Settings(repair_feature=False, observe_empty=False))
    on = r2.BanditPolicyR2(prior=new, settings=r2.R2Settings(repair_feature=True, observe_empty=False))
    assert all(r["x"][-1] == 0.0 for r in off.scored(v)) and all(r["x"][-1] == 1.0 for r in on.scored(v))
    assert all(a["x"][:10] == b["x"][:10] for a, b in zip(off.scored(v), on.scored(v)))


# ----------------------------------------------------------------------------- schema and declaration
def test_a_round_one_prior_is_refused_by_the_round_two_policy_and_back():
    old, new = _old_and_padded_priors()
    with pytest.raises(ValueError):
        r2.Prior2(old.payload, "x")
    with pytest.raises(ValueError):
        r1.Prior(new.payload, "x")


def test_the_declared_yaml_equals_the_defaults_and_names_the_four_labels():
    import yaml
    declared = yaml.safe_load((PROJECT / "configs/policies/bandit_r2.yaml").read_text(encoding="utf-8"))
    settings = r2.R2Settings.for_variant("main", **declared["settings"])
    assert settings == r2.R2Settings.for_variant("main")
    assert settings.stop_threshold == 0.0 and settings.repair_feature and settings.observe_empty
    assert (settings.rho, settings.alpha, settings.ridge, settings.flat_penalty) == (0.5, 1.0, 1.0, 3.0)
    assert set(declared["labels"]) == {"learned", "obs_only", "feat_only", "r1_stop0"}
    assert declared["labels"]["learned"] == {"repair_feature": True, "observe_empty": True}
    assert declared["labels"]["r1_stop0"] == {"repair_feature": False, "observe_empty": False}


def test_the_round_two_policy_names_no_recipe():
    s0 = state(0, luts=1000.0, wns=-0.5)
    m = policy2().decide(view([s0]))
    ctx = json.loads(m.context_ref)
    assert ctx["policy"] == r2.POLICY_ID and len(ctx["x"]) == r2.DIM
    assert "LATCH" not in ctx["arm"] and "A_R_" not in ctx["arm"]


# ------------------------------------------------------------------------------------- the harness loop
def _harness():
    for d in (PROJECT / "scripts/power_mediation_2026-11-10", PROJECT / "scripts/power_readiness_2026-11-11",
              PROJECT / "scripts/bandit_2026-11-11"):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    import bd_eval_r2
    return bd_eval_r2


@dataclasses.dataclass(frozen=True)
class _Node:
    calls: tuple
    used: int = 0
    path: tuple = ()

    def final(self, brief):
        return "final"


class _Sim:
    """Answers 'no recipes' for the first mandate on a scope, and runs everything else for one run."""

    class brief:                                                                         # noqa: N801
        budget_runs = 3

    def __init__(self):
        self.applied = []

    def initial(self):
        return _Node(())

    def view(self, node, episode_id):
        return node

    def apply(self, node, mandate):
        self.applied.append(mandate)
        if mandate.worker_id == "area" and mandate.settings.preserve_placement:
            return None, "no_recipe"
        return dataclasses.replace(node, calls=(*node.calls, "ran"), used=node.used + 1), "ok"


class _Scripted:
    id, version = "scripted", "1"

    def __init__(self, mandates):
        self.mandates = list(mandates)
        self.seen = []

    def decide(self, node):
        self.seen.append(node.calls)
        return self.mandates.pop(0) if self.mandates else None


def _mandate(worker, preserve):
    from orch.contract.models import Mandate, WorkerSettings
    return Mandate(id=f"m_{worker}_{preserve}_{random.random()}", worker_id=worker, base_state_id=sid(0),
                   settings=WorkerSettings(preserve_placement=preserve, max_runs=1), guards=(), brief_hash="h", context_ref="{}")


def test_the_harness_shows_an_empty_call_to_the_policy_and_goes_on():
    h = _harness()
    sim, pol = _Sim(), _Scripted([_mandate("area", True), _mandate("timing", False), _mandate("area", False)])
    res, empties = h.simulate_observing(sim, pol, episode_id="e", observe=True)
    assert res.reason == "stop" and res.used == 2 and len(empties) == 1
    assert pol.seen[0] == () and len(pol.seen[1]) == 1 and pol.seen[1][0].recipe_runs == 0           # the policy was shown it
    assert pol.seen[1][0].selected_state_id is None and pol.seen[1][0].worker_id == "area"


def test_the_old_contract_still_ends_the_episode_at_an_empty_call():
    h = _harness()
    sim, pol = _Sim(), _Scripted([_mandate("area", True), _mandate("timing", False)])
    res, empties = h.simulate_observing(sim, pol, episode_id="e", observe=False)
    assert res.reason == "no_recipe" and res.used == 0 and empties == []


def test_a_policy_that_repeats_an_emptied_scope_is_stopped_with_an_error():
    h = _harness()
    sim, pol = _Sim(), _Scripted([_mandate("area", True), _mandate("area", True)])
    with pytest.raises(ValueError):
        h.simulate_observing(sim, pol, episode_id="e", observe=True)
