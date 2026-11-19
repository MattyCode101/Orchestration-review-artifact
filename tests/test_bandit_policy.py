"""Tests for the contextual-bandit policy (O1). Synthetic views only: no engine, no tool, no data root.

Run with your own base temp:  .venv/bin/python -B -m pytest tests/test_bandit_policy.py --basetemp=data/bandit_2026-11-11/pytest_tmp -q
"""
from __future__ import annotations

import json
import math

import pytest

from orch.config import load_worker_card
from orch.contract.brief_text import load_frozen_brief
from orch.contract.models import MetricName as M
from orch.contract.models import VerdictStatus
from orch.contract.views import CallSummary, PolicyView, StateSummary
from orch.policies import bandit as bd

PROJECT = __import__("pathlib").Path(__file__).resolve().parents[1]
CARDS = tuple(load_worker_card(PROJECT / "configs/workers" / f"{n}.yaml") for n in ("timing", "area"))
NOISE = {M.WNS_NS: 0.1, M.PACKED_LUTS: 10.0}
START = {M.WNS_NS: -0.5, M.PACKED_LUTS: 1000.0, M.DYNAMIC_POWER_W: 1.0, M.TNS_NS: -9.0, M.NETLIST_LUTS: 1200.0, M.FFS: 800.0}


def sid(n: int) -> str:
    return f"{n:064x}"


def state(n: int, *, luts: float, wns: float, valid: bool | None = None) -> StateSummary:
    metrics = dict(START)
    metrics[M.PACKED_LUTS], metrics[M.WNS_NS] = luts, wns
    ok = (wns >= START[M.WNS_NS]) if valid is None else valid          # A0: WNS must not fall below the start
    return StateSummary(state_id=sid(n), metrics=metrics, physically_valid=True, brief_valid=ok)


def view(visited, current=None, best=None, calls=(), budget_left=6, brief_id="A0") -> PolicyView:
    brief = load_frozen_brief(brief_id)
    ranked = sorted((s for s in visited if s.brief_valid), key=lambda s: s.metrics[M.PACKED_LUTS])
    return PolicyView(
        brief=brief, budget_left=budget_left, start=visited[0], current=current or visited[-1],
        best_valid=best or ranked[0], visited=tuple(visited), calls=tuple(calls), worker_cards=CARDS,
        noise_bands=NOISE, episode_id="ep_test",
    )


def prior(sigma: float = 1.0, arms: dict | None = None) -> bd.Prior:
    payload = {"schema": bd.SCHEMA, "features": list(bd.FEATURES), "ridge": 1.0, "sigma": sigma, "arms": arms or {}}
    return bd.Prior(payload, "0" * 64)


def policy(variant="main", **kw) -> bd.BanditPolicy:
    return bd.BanditPolicy(prior=kw.pop("prior_", prior()), settings=bd.BanditSettings.for_variant(variant, **kw))


def test_signed_log_is_odd_and_one_at_one_band():
    assert bd.signed_log(0.0) == 0.0
    assert bd.signed_log(1.0) == 1.0
    assert bd.signed_log(-3.0) == -bd.signed_log(3.0)
    assert bd.signed_log(7.0) == 3.0


def test_solve_matches_a_known_system():
    a = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]
    x = bd._solve(a, [1.0, 2.0, 3.0])
    back = [sum(a[i][j] * x[j] for j in range(3)) for i in range(3)]
    assert all(abs(back[i] - v) < 1e-12 for i, v in enumerate([1.0, 2.0, 3.0]))


def test_goal_gain_and_slack_are_in_noise_bands():
    s0 = state(0, luts=1000.0, wns=-0.5)
    s1 = state(1, luts=900.0, wns=-0.62, valid=False)
    v = view([s0, s1], best=s0)
    goal = bd.goal_gain(v, s1)
    assert math.isclose(goal, 10.0)                                   # 100 LUTs saved, band 10
    term = v.brief.terms[0]
    assert math.isclose(bd.term_slack(v, term, s1), -1.2)             # 0.12 ns below the limit, band 0.1


def test_credited_gain_pays_a_broken_gain_only_while_a_run_remains():
    s0 = state(0, luts=1000.0, wns=-0.5)
    s1 = state(1, luts=900.0, wns=-0.62)                              # breaks the A0 limit
    v_open = view([s0, s1], budget_left=4)
    v_shut = view([s0, s1], budget_left=0)
    assert bd.credited_gain(view([s0]), 0.5) == 0.0
    assert math.isclose(bd.credited_gain(v_open, 0.5), 0.5 * bd.signed_log(10.0))
    assert bd.credited_gain(v_shut, 0.5) == 0.0


def test_credited_gain_after_repair_equals_the_banked_gain():
    s0 = state(0, luts=1000.0, wns=-0.5)
    s1 = state(1, luts=900.0, wns=-0.62)
    s2 = state(2, luts=900.0, wns=-0.40)                               # repaired with no loss: valid again
    v = view([s0, s1, s2], budget_left=3)
    assert math.isclose(bd.credited_gain(v, 0.5), bd.signed_log(10.0))      # nothing left to credit
    s3 = state(3, luts=902.0, wns=-0.40)                               # repaired with a small loss
    v = view([s0, s1, s3], budget_left=3)
    extra = 0.5 * (bd.signed_log(10.0) - bd.signed_log(9.8))                 # option value of the broken state
    assert math.isclose(bd.credited_gain(v, 0.5), bd.signed_log(9.8) + extra)


def test_features_are_scale_free_and_finite():
    s0 = state(0, luts=1000.0, wns=-0.5)
    s1 = state(1, luts=900.0, wns=-0.62)
    v = view([s0, s1], current=s1, best=s0, budget_left=4)
    x = bd.features(v, s1, "timing", 2)
    assert len(x) == bd.DIM and all(math.isfinite(t) for t in x)
    names = dict(zip(bd.FEATURES, x))
    assert names["valid"] == 0.0 and names["violation"] > 0.0 and names["slack"] == 0.0
    assert names["is_start"] == 0.0 and math.isclose(names["budget"], 4 / 6)
    assert names["exhausts"] == 0.0
    assert dict(zip(bd.FEATURES, bd.features(v, s0, "timing", 2)))["is_start"] == 1.0


def test_arms_name_no_recipe_and_respect_the_budget():
    s0 = state(0, luts=1000.0, wns=-0.5)
    s1 = state(1, luts=900.0, wns=-0.62)
    v = view([s0, s1], current=s1, best=s0, budget_left=1)
    arms = list(bd.enumerate_arms(v, "best_or_latest"))
    assert arms and all(a[3] == 1 for a in arms)                     # a 2-attempt call does not fit
    assert {a[4] for a in arms} == {"best", "latest"}
    assert all("LATCH" not in a[0] and "A_R_" not in a[0] for a in arms)
    assert {a[4] for a in bd.enumerate_arms(v, "best_only")} == {"best"}


def test_decision_is_deterministic_and_a_valid_mandate():
    s0 = state(0, luts=1000.0, wns=-0.5)
    v = view([s0])
    p = policy()
    m1, m2 = p.decide(v), p.decide(v)
    assert m1 is not None and m1 == m2
    m1.validate_against(v.brief, None)                                 # no guards, so the start evidence is not read
    assert m1.settings.max_runs in (1, 2) and m1.worker_id in {"timing", "area"} and m1.base_state_id == sid(0)
    ctx = json.loads(m1.context_ref)
    assert ctx["policy"] == p.id and ctx["version"] == p.version and len(ctx["x"]) == bd.DIM


def test_cold_prior_explores_and_a_flat_prior_stops():
    s0 = state(0, luts=1000.0, wns=-0.5)
    assert policy(prior_=prior(sigma=1.0)).decide(view([s0])) is not None      # cold arms carry a big bonus
    assert policy(prior_=prior(sigma=0.0)).decide(view([s0])) is None          # no mean, no bonus: nothing beats noise


def test_budget_exhausted_stops():
    s0 = state(0, luts=1000.0, wns=-0.5)
    assert policy().decide(view([s0], budget_left=0)) is None


def test_in_episode_rewards_telescope_to_the_credited_gain():
    p = policy()
    s0 = state(0, luts=1000.0, wns=-0.5)
    v0 = view([s0])
    m0 = p.decide(v0)
    # call 0 returns a limit-breaking gain, call 1 repairs it
    s1 = state(1, luts=900.0, wns=-0.62)
    call0 = CallSummary(mandate_id=m0.id, worker_id=m0.worker_id, base_state_id=m0.base_state_id, settings=m0.settings,
                        selected_state_id=s1.state_id, selected_verdict_status=VerdictStatus.BRIEF_VIOLATION,
                        recipe_runs=m0.settings.max_runs, context_ref=m0.context_ref)
    v1 = view([s0, s1], current=s1, best=s0, calls=[call0], budget_left=6 - m0.settings.max_runs)
    m1 = p.decide(v1)
    assert m1 is not None
    s2 = state(2, luts=902.0, wns=-0.40)
    call1 = CallSummary(mandate_id=m1.id, worker_id=m1.worker_id, base_state_id=m1.base_state_id, settings=m1.settings,
                        selected_state_id=s2.state_id, selected_verdict_status=VerdictStatus.ACCEPTABLE,
                        recipe_runs=m1.settings.max_runs, context_ref=m1.context_ref)
    left = 6 - m0.settings.max_runs - m1.settings.max_runs
    v2 = view([s0, s1, s2], current=s2, best=s2, calls=[call0, call1], budget_left=left)
    rewards = p.rewards(v2)
    total = sum(r for _, r in rewards)
    assert math.isclose(total, bd.credited_gain(v2, 0.5) - bd.credited_gain(v0, 0.5), abs_tol=1e-12)


def test_a_resumed_episode_makes_the_same_choice():
    p, q = policy(), policy()
    s0 = state(0, luts=1000.0, wns=-0.5)
    m0 = p.decide(view([s0]))
    s1 = state(1, luts=900.0, wns=-0.62)
    call0 = CallSummary(mandate_id=m0.id, worker_id=m0.worker_id, base_state_id=m0.base_state_id, settings=m0.settings,
                        selected_state_id=s1.state_id, selected_verdict_status=VerdictStatus.BRIEF_VIOLATION,
                        recipe_runs=m0.settings.max_runs, context_ref=m0.context_ref)
    v1 = view([s0, s1], current=s1, best=s0, calls=[call0], budget_left=6 - m0.settings.max_runs)
    assert p.decide(v1) == q.decide(v1)                                # a fresh policy object, same prior


def test_a_foreign_call_is_refused():
    p = policy()
    s0 = state(0, luts=1000.0, wns=-0.5)
    foreign = CallSummary(mandate_id="m", worker_id="area", base_state_id=sid(0),
                          settings=policy().decide(view([s0])).settings, selected_state_id=None,
                          recipe_runs=1, context_ref="")
    with pytest.raises(ValueError):
        p.decide(view([s0], calls=[foreign], budget_left=5))


def test_literal_reward_is_goal_change_less_a_flat_penalty():
    s0 = state(0, luts=1000.0, wns=-0.5)
    s1 = state(1, luts=900.0, wns=-0.62)
    call = CallSummary(mandate_id="m", worker_id="area", base_state_id=sid(0), settings=policy().decide(view([s0])).settings,
                       selected_state_id=sid(1), recipe_runs=2, context_ref="")
    v = view([s0, s1], current=s1, best=s0, calls=[call], budget_left=4)
    assert math.isclose(bd.flat_reward(v, call, 3.0), bd.signed_log(10.0) - 3.0)


def test_l2r_brief_equals_the_power_study_brief():
    pm_dir = PROJECT / "scripts/power_mediation_2026-11-10"
    bd_dir = PROJECT / "scripts/bandit_2026-11-11"
    import sys
    for d in (pm_dir, bd_dir):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    import bd_common as bc
    import pm_common as pm
    for design in bc.DESIGNS:
        band = bc.power_band_w(design)
        assert bc.l2r_brief(band).fingerprint() == pm.candidate_brief("L2R", 6, band).fingerprint()


def test_variants_are_the_two_by_two_ablation():
    names = {bd.BanditSettings.for_variant(n).variant for n in bd.VARIANTS}
    assert names == {"main", "literal", "base_only", "reward_only"}
    assert {bd.VARIANTS[n] for n in bd.VARIANTS} == {(b, r) for b in bd.BASE_CHOICES for r in bd.REWARDS}
    assert bd.BanditSettings.for_variant("main").sha256() != bd.BanditSettings.for_variant("literal").sha256()


def test_the_flat_reward_variant_updates_from_the_view_alone():
    p = policy("literal")
    s0 = state(0, luts=1000.0, wns=-0.5)
    m0 = p.decide(view([s0]))
    assert m0 is not None
    s1 = state(1, luts=900.0, wns=-0.62)
    call0 = CallSummary(mandate_id=m0.id, worker_id=m0.worker_id, base_state_id=m0.base_state_id, settings=m0.settings,
                        selected_state_id=s1.state_id, selected_verdict_status=VerdictStatus.BRIEF_VIOLATION,
                        recipe_runs=m0.settings.max_runs, context_ref=m0.context_ref)
    v1 = view([s0, s1], current=s1, best=s0, calls=[call0], budget_left=6 - m0.settings.max_runs)
    (ctx, reward), = p.rewards(v1)
    assert math.isclose(reward, bd.signed_log(10.0) - 3.0) and ctx["arm"].endswith("|best")
