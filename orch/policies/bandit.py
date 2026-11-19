"""Contextual-bandit orchestration policy: the paper's learned policy (round 2).

Black-box rule. The policy reads only the PolicyView (the brief, the contract's measurements of the saved
states, and the contract's record of this episode's worker calls) and a frozen prior built from the
training call record, leave-one-design-out. It never sees a worker's internal score, a recipe name or a
recipe list, and it names no recipe: a mandate is worker, base state, placement scope and attempts.

Design (declared in notes/bandit_policy_plan.md before any evaluation, not tuned afterwards):
  arm      worker x scope (keep placement or not) x attempts (1 or 2) x base (best valid state, or the latest
           result when that differs); plus STOP.
  reward   change in the episode's credited gain, in signed log2(1 + noise bands). The credited gain is the
           best brief-valid gain plus rho x the extra gain of the best limit-breaking state, while at least one
           run remains to repair it. Rewards therefore sum to the final pick's gain once the budget is spent.
           The two changes from the first design (base, reward) are independent switches, so a 2 x 2 ablation
           measures each: "literal" = base best valid state only + flat reward (goal change of the returned state
           minus a flat penalty when it breaks a limit); "base_only" and "reward_only" switch one at a time.
  model    one ridge-regression model per arm over ten scale-free features; UCB = mean + alpha x sigma x
           sqrt(x' A^-1 x). Deterministic: no random numbers; ties go to the first arm in a fixed order.
  choice   the arm with the highest UCB per run among those whose UCB is at least one noise band; if none,
           STOP.
  update   the prior comes from the training record (other designs only); every completed call of the running
           episode is added to its arm's statistics, rebuilt from the journalled context_ref at each decision,
           so a resumed episode makes the same choices.

Pure Python linear algebra (10 x 10), so decisions are bit-reproducible across machines.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from orch.contract.models import BriefTerm, Mandate, MetricName, WorkerSettings
from orch.contract.views import CallSummary, PolicyView, StateSummary

SCHEMA = "bandit-prior-v1"
FEATURES = (
    "bias", "valid", "violation", "slack", "gain", "banked", "is_start", "tried", "budget", "exhausts",
)
DIM = len(FEATURES)
ATTEMPTS = (1, 2)
BASE_CHOICES = ("best_or_latest", "best_only")
REWARDS = ("credited", "flat")
VARIANTS = {                                    # name -> (base_choice, reward); main is the declared policy
    "main": ("best_or_latest", "credited"),
    "literal": ("best_only", "flat"),
    "base_only": ("best_or_latest", "flat"),
    "reward_only": ("best_only", "credited"),
}


# ------------------------------------------------------------------------------------------ settings
@dataclass(frozen=True)
class BanditSettings:
    """Every setting of the policy. The YAML file configs/policies/bandit.yaml is the declaration."""

    base_choice: str = "best_or_latest"
    reward: str = "credited"
    rho: float = 0.5                  # credit share of an unvalidated (limit-breaking) gain while a run remains
    flat_penalty: float = 3.0         # flat reward only, in log2(1 + band) units
    alpha: float = 1.0                # exploration weight
    ridge: float = 1.0                # lambda of the per-arm ridge regression
    stop_threshold: float = 1.0       # one noise band, in log2(1 + band) units

    @classmethod
    def for_variant(cls, name: str, **common) -> "BanditSettings":
        base_choice, reward = VARIANTS[name]
        return cls(base_choice=base_choice, reward=reward, **common)

    @property
    def variant(self) -> str:
        for name, pair in VARIANTS.items():
            if pair == (self.base_choice, self.reward):
                return name
        return f"{self.base_choice}+{self.reward}"

    def __post_init__(self) -> None:
        if self.base_choice not in BASE_CHOICES or self.reward not in REWARDS:
            raise ValueError(f"unknown bandit base choice or reward: {self.base_choice!r}, {self.reward!r}")
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [0, 1]")
        if self.alpha < 0 or self.ridge <= 0 or self.stop_threshold < 0 or self.flat_penalty < 0:
            raise ValueError("alpha, stop_threshold and flat_penalty must be >= 0 and ridge > 0")

    def canonical(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------------- maths
def signed_log(value: float) -> float:
    """sign(v) x log2(1 + |v|): noise-band units, compressed (gains run from 1 to about 1,000 bands)."""

    return math.copysign(math.log2(1.0 + abs(value)), value)


def _solve(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Gaussian elimination with partial pivoting; the matrices here are symmetric positive definite."""

    n = len(rhs)
    a = [list(map(float, row)) + [float(rhs[i])] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-300:
            raise ValueError("singular bandit matrix")
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        inv = 1.0 / a[col][col]
        for row in range(col + 1, n):
            factor = a[row][col] * inv
            if factor != 0.0:
                for k in range(col, n + 1):
                    a[row][k] -= factor * a[col][k]
    out = [0.0] * n
    for row in range(n - 1, -1, -1):
        total = a[row][n] - sum(a[row][k] * out[k] for k in range(row + 1, n))
        out[row] = total / a[row][row]
    return out


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


# --------------------------------------------------------------------------- view helpers (no recipes)
def _band(view: PolicyView, metric: MetricName) -> float:
    band = view.noise_bands.get(metric)
    if band is None or not math.isfinite(band) or band <= 0:
        raise ValueError(f"the bandit needs a positive noise band for {metric.value}")
    return float(band)


def _metric(summary: StateSummary, metric: MetricName) -> float:
    value = summary.metrics.get(metric)
    if value is None:
        raise ValueError(f"state {summary.state_id[:10]} lacks {metric.value}")
    return value


def goal_gain(view: PolicyView, summary: StateSummary) -> float:
    """Improvement of the brief's goal over the start, in noise bands (positive is better)."""

    objective = view.brief.objective
    delta = _metric(summary, objective.metric) - _metric(view.start, objective.metric)
    if objective.direction == "min":
        delta = -delta
    return delta / _band(view, objective.metric)


def term_slack(view: PolicyView, term: BriefTerm, summary: StateSummary) -> float:
    """Distance to a brief limit in noise bands: >= 0 inside the limit, < 0 beyond it."""

    limit = term.factor * _metric(view.start, term.metric) + term.offset
    value = _metric(summary, term.metric)
    slack = limit - value if term.comparison == "<=" else value - limit
    return slack / _band(view, term.metric)


def banked_gain(view: PolicyView) -> float:
    """Goal gain of the best brief-valid state visited (the start counts, at zero)."""

    gains = [goal_gain(view, s) for s in view.visited if s.brief_valid]
    return max(gains) if gains else 0.0


def credited_gain(view: PolicyView, rho: float) -> float:
    """Best brief-valid gain, plus rho x the extra gain of the best limit-breaking state while a run remains."""

    banked = banked_gain(view)
    value = signed_log(banked)
    broken = [goal_gain(view, s) for s in view.visited if s.physically_valid and not s.brief_valid]
    if broken and view.budget_left >= 1:
        top = max(broken)
        if top > banked:
            value += rho * (signed_log(top) - signed_log(banked))
    return value


def _state(view: PolicyView, state_id: str) -> StateSummary:
    for summary in reversed(view.visited):
        if summary.state_id == state_id:
            return summary
    raise ValueError("a call cites a state absent from the view")


def flat_reward(view: PolicyView, call: CallSummary, penalty: float) -> float:
    """First design, read literally: goal change of the returned state against its base, less a flat penalty."""

    if call.selected_state_id is None:
        return 0.0
    base, chosen = _state(view, call.base_state_id), _state(view, call.selected_state_id)
    value = signed_log(goal_gain(view, chosen) - goal_gain(view, base))
    if not chosen.brief_valid:
        value -= penalty
    return value


def features(view: PolicyView, base: StateSummary, worker_id: str, attempts: int) -> tuple[float, ...]:
    """Ten scale-free numbers, all measurements the PolicyView already carries.

    valid, violation, slack   limit status of the base and its distance to the tightest limit (log bands)
    gain, banked              goal gain at the base, and the best brief-valid gain so far (signed log bands)
    is_start                  the base is the untouched start
    tried                     earlier calls to this worker on this base (0, 0.5 or 1)
    budget, exhausts          runs left as a share of the budget, and whether this call spends the last runs
    """

    slacks = [term_slack(view, term, base) for term in view.brief.terms]
    worst = min(slacks) if slacks else 0.0
    tried = sum(1 for c in view.calls if c.worker_id == worker_id and c.base_state_id == base.state_id)
    return (
        1.0,
        1.0 if base.brief_valid else 0.0,
        math.log2(1.0 + max(0.0, -worst)),
        math.log2(1.0 + max(0.0, worst)),
        signed_log(goal_gain(view, base)),
        signed_log(banked_gain(view)),
        1.0 if base.state_id == view.start.state_id else 0.0,
        min(2, tried) / 2.0,
        view.budget_left / view.brief.budget_runs,
        1.0 if view.budget_left - attempts <= 0 else 0.0,
    )


# ------------------------------------------------------------------------------------------------ arms
def arm_key(worker_id: str, preserve: bool, attempts: int, base_kind: str) -> str:
    return f"{worker_id}|{'keep' if preserve else 'free'}|{attempts}|{base_kind}"


def enumerate_arms(view: PolicyView, base_choice: str) -> Iterator[tuple[str, str, bool, int, str, StateSummary]]:
    """(key, worker, preserve, attempts, base_kind, base) in a fixed order. Nothing here names a recipe."""

    best = view.best_valid or view.start
    bases = [("best", best)]
    if base_choice == "best_or_latest" and view.current.physically_valid and view.current.state_id != best.state_id:
        bases.append(("latest", view.current))
    for worker_id in sorted(card.id for card in view.worker_cards):
        for preserve in (False, True):
            for attempts in ATTEMPTS:
                if attempts > view.budget_left:
                    continue
                for kind, base in bases:
                    yield arm_key(worker_id, preserve, attempts, kind), worker_id, preserve, attempts, kind, base


# ------------------------------------------------------------------------------------------------ prior
class Prior:
    """Per-arm ridge statistics from the training call record (A = lambda I + sum x x', b = sum r x)."""

    def __init__(self, payload: dict[str, Any], sha256: str) -> None:
        if payload.get("schema") != SCHEMA or tuple(payload.get("features", ())) != FEATURES:
            raise ValueError("bandit prior has another schema or feature list")
        self.payload = payload
        self.sha256 = sha256
        self.sigma = float(payload["sigma"])
        self.ridge = float(payload["ridge"])
        self.arms = {
            key: ([list(map(float, row)) for row in value["A"]], list(map(float, value["b"])), int(value["n"]))
            for key, value in payload["arms"].items()
        }

    @classmethod
    def load(cls, path: Path | str) -> "Prior":
        raw = Path(path).read_bytes()
        return cls(json.loads(raw), hashlib.sha256(raw).hexdigest())

    def fresh(self) -> tuple[list[list[float]], list[float], int]:
        return ([[self.ridge if i == j else 0.0 for j in range(DIM)] for i in range(DIM)], [0.0] * DIM, 0)


def _add(stats: dict[str, tuple[list[list[float]], list[float], int]], prior: Prior, key: str,
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
class BanditPolicy:
    """Deterministic UCB choice over (worker, scope, attempts, base) from a frozen prior."""

    id = "bandit_linucb"

    def __init__(self, *, prior: Prior, settings: BanditSettings) -> None:
        self.prior, self.settings = prior, settings
        self.version = f"1.0-{settings.variant}-{settings.sha256()[:8]}-{prior.sha256[:8]}"

    # -- the running episode's own calls, rebuilt from the journalled context --------------------
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
        """(context, reward) of every completed call of this episode, from the view alone."""

        contexts = [self._context(c) for c in view.calls]
        now = credited_gain(view, self.settings.rho)
        out = []
        for j, ctx in enumerate(contexts):
            if self.settings.reward == "credited":
                after = contexts[j + 1]["phi"] if j + 1 < len(contexts) else now
                out.append((ctx, after - ctx["phi"]))
            else:
                out.append((ctx, flat_reward(view, view.calls[j], self.settings.flat_penalty)))
        return out

    def scored(self, view: PolicyView) -> list[dict[str, Any]]:
        """Every feasible arm with its prediction, bonus and index (also used by the agreement analysis)."""

        stats: dict[str, tuple[list[list[float]], list[float], int]] = {}
        for ctx, reward in self.rewards(view):
            _add(stats, self.prior, ctx["arm"], ctx["x"], reward)
        phi = credited_gain(view, self.settings.rho)
        rows = []
        for order, (key, worker_id, preserve, attempts, kind, base) in enumerate(
            enumerate_arms(view, self.settings.base_choice)
        ):
            x = features(view, base, worker_id, attempts)
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

    def decide(self, view: PolicyView) -> Mandate | None:
        if view.budget_left <= 0:
            return None
        rows = [r for r in self.scored(view) if r["ucb"] >= self.settings.stop_threshold]
        if not rows:
            return None
        top = max(round(r["per_run"], 12) for r in rows)
        choice = next(r for r in rows if round(r["per_run"], 12) == top)        # first in the fixed order
        context = json.dumps({
            "policy": self.id, "version": self.version, "arm": choice["key"], "x": choice["x"],
            "phi": choice["phi"], "call": len(view.calls),
        }, sort_keys=True, separators=(",", ":"))
        identity = json.dumps(
            [view.episode_id, self.version, len(view.calls), choice["key"], choice["base_id"]],
            separators=(",", ":"),
        )
        return Mandate(
            id="mandate_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
            worker_id=choice["worker"],
            base_state_id=choice["base_id"],
            settings=WorkerSettings(preserve_placement=choice["preserve"], max_runs=choice["attempts"]),
            guards=(),
            brief_hash=view.brief.fingerprint(),
            context_ref=context,
        )
