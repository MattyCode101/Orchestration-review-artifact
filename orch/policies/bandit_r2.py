"""Contextual-bandit orchestration policy, version 2: The authors' decisions of Thursday 12 November 2026.

The round-1 policy (orch/policies/bandit.py, declared in notes/bandit_policy_plan.md) is untouched. This module reuses its
pure-Python LinUCB and changes exactly three things, declared in notes/bandit_policy_plan_r2.md before any evaluation:

  1. stop threshold 0: the policy stops only when no arm has a non-negative upper bound (round 1: one noise band);
  2. one more context feature, `repair`: 1.0 when a visited state that is not the best valid state breaks a brief limit and
     has a larger goal gain than the best valid state (a gain is awaiting repair), else 0.0;
  3. an empty call (no recipe runs, nothing selected: "no recipes available") is an observation. It removes that worker and
     placement scope from the action set for that base state, it adds no reward sample, and the episode goes on.

Two switches (repair_feature, observe_empty) turn changes 2 and 3 off, so the attribution grid and the round-1 policy can be run
from this code. With both off the choices are those of round 1, bit for bit (tests/test_bandit_r2.py and the 30 cells).

Black-box rule unchanged: the policy reads the PolicyView (the brief, the contract's measurements of the saved states, the
contract's record of this episode's calls) and a frozen prior from the training call record, leave-one-design-out. It never
reads a worker's internal score, a recipe name or a recipe list, and it names no recipe.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from orch.contract.views import CallSummary, PolicyView, StateSummary
from orch.policies import bandit as r1
from orch.policies.bandit import (  # noqa: F401  (re-exported: the prior fit and the replay call them through this module)
    ATTEMPTS,
    BanditSettings,
    Prior,
    _dot,
    _solve,
    arm_key,
    banked_gain,
    credited_gain,
    flat_reward,
    goal_gain,
    signed_log,
    term_slack,
)

SCHEMA = "bandit-prior-v2"
FEATURES = (*r1.FEATURES, "repair")
DIM = len(FEATURES)
ROUND1_DIM = len(r1.FEATURES)
POLICY_ID = "bandit_linucb_r2"


# ------------------------------------------------------------------------------------------ settings
@dataclass(frozen=True)
class R2Settings(BanditSettings):
    """Round-1 settings plus the two switches. The defaults are the learned policy of the authors' decisions."""

    stop_threshold: float = 0.0       # decision of 12 Nov: stop threshold 0 (round 1: 1.0, one noise band)
    repair_feature: bool = True       # the added context feature (False: it is held at 0, as in round 1)
    observe_empty: bool = True        # an empty call is an observation (False: round 1, the episode ends there)


# ------------------------------------------------------------------------------------------- the feature
def repair_pending(view: PolicyView) -> bool:
    """A visited state is not the best valid state, breaks a limit of the brief, and has a larger goal gain than it.

    Same states and same comparison as the repair credit in credited_gain (physically valid, not brief valid, gain strictly
    above the best brief-valid gain), without the refinement of a run remaining: a decision is only asked while one does.
    """

    banked = banked_gain(view)
    return any(s.physically_valid and not s.brief_valid and goal_gain(view, s) > banked for s in view.visited)


def features(view: PolicyView, base: StateSummary, worker_id: str, attempts: int, *, repair: bool = True) -> tuple[float, ...]:
    """The ten round-1 numbers and `repair` (1.0 while a larger gain awaits repair; held at 0.0 when repair is False)."""

    return (*r1.features(view, base, worker_id, attempts), 1.0 if (repair and repair_pending(view)) else 0.0)


# ------------------------------------------------------------------------------------------ empty calls
def is_empty_call(call: CallSummary) -> bool:
    """A call that found no recipe to run: nothing charged, nothing selected."""

    return call.recipe_runs == 0 and call.selected_state_id is None


def exhausted(view: PolicyView) -> set[tuple[str, bool, str]]:
    """(worker, placement scope, base state) of every empty call of this episode: no recipe is left for it."""

    return {(c.worker_id, bool(c.settings.preserve_placement), c.base_state_id) for c in view.calls if is_empty_call(c)}


def enumerate_arms(view: PolicyView, base_choice: str, observe: bool = True) -> Iterator[tuple[str, str, bool, int, str, StateSummary]]:
    """The round-1 arms in the round-1 order, less the (worker, scope, base state) combinations an empty call has removed."""

    gone = exhausted(view) if observe else set()
    for arm in r1.enumerate_arms(view, base_choice):
        _key, worker_id, preserve, _attempts, _kind, base = arm
        if (worker_id, preserve, base.state_id) in gone:
            continue
        yield arm


# ------------------------------------------------------------------------------------------------ prior
class Prior2(Prior):
    """Per-arm ridge statistics over the eleven features (same layout as the round-1 prior)."""

    def __init__(self, payload: dict[str, Any], sha256: str) -> None:
        if payload.get("schema") != SCHEMA or tuple(payload.get("features", ())) != FEATURES:
            raise ValueError("bandit r2 prior has another schema or feature list")
        self.payload = payload
        self.sha256 = sha256
        self.sigma = float(payload["sigma"])
        self.ridge = float(payload["ridge"])
        self.arms = {
            key: ([list(map(float, row)) for row in value["A"]], list(map(float, value["b"])), int(value["n"]))
            for key, value in payload["arms"].items()
        }

    def fresh(self) -> tuple[list[list[float]], list[float], int]:
        return ([[self.ridge if i == j else 0.0 for j in range(DIM)] for i in range(DIM)], [0.0] * DIM, 0)


def _add(stats: dict[str, tuple[list[list[float]], list[float], int]], prior: Prior2, key: str,
         x: Sequence[float], reward: float) -> None:
    if key not in stats:
        a, b, n = prior.arms.get(key) or prior.fresh()
        stats[key] = ([row[:] for row in a], b[:], n)
    a, b, n = stats[key]
    for i in range(DIM):
        for j in range(DIM):
            a[i][j] += x[i] * x[j]
        b[i] += reward * x[i]
    stats[key] = (a, b, n + 1)


# ---------------------------------------------------------------------------------------------- policy
class BanditPolicyR2(r1.BanditPolicy):
    """Round-1 UCB choice with the added feature, stop threshold 0 and empty calls as observations."""

    id = POLICY_ID

    def __init__(self, *, prior: Prior2, settings: R2Settings) -> None:
        self.prior, self.settings = prior, settings
        self.version = f"2.0-{settings.variant}-{settings.sha256()[:8]}-{prior.sha256[:8]}"

    def _context(self, call: CallSummary) -> dict[str, Any]:
        try:
            value = json.loads(call.context_ref)
        except (ValueError, TypeError) as exc:
            raise ValueError("bandit call lacks its journalled context") from exc
        if (
            not isinstance(value, dict) or value.get("policy") != self.id or value.get("version") != self.version
            or not isinstance(value.get("arm"), str) or not isinstance(value.get("x"), list)
            or len(value["x"]) != DIM or not isinstance(value.get("phi"), float)
        ):
            raise ValueError("bandit call has an invalid journalled context")
        return value

    def rewards(self, view: PolicyView) -> list[tuple[dict[str, Any], float]]:
        """(context, reward) of every completed call that ran something. Empty calls are observations, not samples; they
        change neither the state nor the credited gain, so the rewards still telescope across them."""

        calls = [c for c in view.calls if not is_empty_call(c)]
        contexts = [self._context(c) for c in calls]
        now = credited_gain(view, self.settings.rho)
        out = []
        for j, ctx in enumerate(contexts):
            if self.settings.reward == "credited":
                after = contexts[j + 1]["phi"] if j + 1 < len(contexts) else now
                out.append((ctx, after - ctx["phi"]))
            else:
                out.append((ctx, flat_reward(view, calls[j], self.settings.flat_penalty)))
        return out

    def scored(self, view: PolicyView) -> list[dict[str, Any]]:
        stats: dict[str, tuple[list[list[float]], list[float], int]] = {}
        for ctx, reward in self.rewards(view):
            _add(stats, self.prior, ctx["arm"], ctx["x"], reward)
        phi = credited_gain(view, self.settings.rho)
        rows = []
        for order, (key, worker_id, preserve, attempts, kind, base) in enumerate(
            enumerate_arms(view, self.settings.base_choice, self.settings.observe_empty)
        ):
            x = features(view, base, worker_id, attempts, repair=self.settings.repair_feature)
            a, b, n = stats[key] if key in stats else (self.prior.arms.get(key) or self.prior.fresh())
            z = _solve(a, x)
            mean = _dot(b, z)
            bonus = self.settings.alpha * self.prior.sigma * math.sqrt(max(0.0, _dot(x, z)))
            ucb = mean + bonus
            rows.append({
                "order": order, "key": key, "worker": worker_id, "preserve": preserve, "attempts": attempts,
                "base_kind": kind, "base_id": base.state_id, "x": list(x), "phi": phi, "n": n,
                "mean": mean, "bonus": bonus, "ucb": ucb, "per_run": ucb / attempts,
            })
        return rows


class PeekBanditR2(BanditPolicyR2):
    """The choice rule with no in-episode learning (for agreement analyses on another policy's views)."""

    def rewards(self, view: PolicyView):
        return []


def settings_hash(settings: R2Settings) -> str:
    return hashlib.sha256(settings.canonical().encode("utf-8")).hexdigest()
