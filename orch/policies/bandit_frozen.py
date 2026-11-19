"""The learned policy, frozen for the test runs (O1, Friday 13 November 2026).

What is frozen, declared in notes/bandit_test_freeze_plan.md before the fit:
  * the declared policy, no change: orch/policies/bandit_r2.py (stop threshold 0, the `repair` context feature, an empty call is an observation);
  * a prior fitted on the call records of all 5 training and all 3 development designs, one per brief: T0, A0, T, A, L2R, P0;
  * W2R is not applicable (no prior): bandit() refuses it with BanditNotApplicable and the runner must skip that row.

The weekend runner calls one entry point:

    from orch.policies.bandit_frozen import bandit, run_bandit_episode

    policy = bandit(design, brief, seed)        # brief: its id or the Brief object; returns the policy
    result = run_episode_r2(..., policy=policy, options=ROUND2)           # the runner's own call, both options on
    # or, in one call that also makes sure both options are on:
    result = run_bandit_episode(design, brief, seed, engine=..., registry=..., workers=..., start_state=...,
                                episode_id=..., noise=..., validator=..., telemetry_provider=...)

Both go through orch/episode_r2.py (frozen; used as it is) with ROUND2 = both options on: an empty call is an observation, and a STOP is accepted only
after two calls without gain beyond the goal's noise band or when the budget is spent.

Properties the runner can rely on:
  * `design` is metadata. The policy holds no design data and reads none: its inputs are the brief, the contract's measurements of the saved states, the
    episode's own past calls and the frozen prior. Nothing is opened for any design, test designs included.
  * `seed` has no effect. The policy is deterministic (no random number is drawn) and decide() is a pure function of the view and the prior, so a run that
    is resumed from its journal repeats the same decisions. Repeats of a row keyed by seed are identical; one run per (design, brief) is enough.
  * A component that differs from its recorded hash (policy code, settings, prior, this entry point's config) makes bandit() raise FreezeError.
  * An episode whose brief is not the brief the policy was frozen for (other id, or another kind of objective or limit) raises BriefMismatch at the
    first decision; offsets of the power limit may differ (it is the design's band).
Every decision is the declared policy's: FrozenBandit adds only the brief check (tests/test_bandit_frozen.py compares it with the unwrapped policy).
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from orch.contract.models import Brief
from orch.contract.views import PolicyView
from orch.policies import bandit_r2 as r2

PROJECT = Path(__file__).resolve().parents[2]
FREEZE_CONFIG = PROJECT / "configs/policies/bandit_test_frozen.yaml"
SCHEMA = "bandit-test-freeze-v1"
APPROACH = "bandit"


class FreezeError(RuntimeError):
    """A frozen component differs from the hash recorded in the freeze config."""


class BanditNotApplicable(ValueError):
    """The frozen policy has no prior for this brief (W2R) or for a brief that is not one of the frozen briefs."""


class BriefMismatch(ValueError):
    """The brief of the episode is not the brief this policy was frozen for."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def brief_signature(brief: Brief) -> list[Any]:
    """The kind of the brief: goal metric and direction, and each limit's metric, comparison and factor and offset. The offset of a dynamic-power limit
    is left out: it is the design's band and differs between designs."""

    terms = []
    for term in brief.terms:
        metric = getattr(term.metric, "value", term.metric)
        offset = 0.0 if metric == "dynamic_power_w" else round(float(term.offset), 9)
        terms.append([metric, term.comparison, round(float(term.factor), 9), offset])
    goal = brief.objective
    return [getattr(goal.metric, "value", goal.metric), goal.direction, sorted(terms)]


def check_brief(brief: Brief, brief_id: str, signature: list[Any]) -> None:
    if brief.id != brief_id:
        raise BriefMismatch(f"this policy is frozen for brief {brief_id}, the episode runs brief {brief.id}")
    if json.loads(json.dumps(brief_signature(brief))) != json.loads(json.dumps(signature)):
        raise BriefMismatch(
            f"brief {brief_id} differs in kind from the brief its prior was fitted under: "
            f"{json.dumps(brief_signature(brief))} instead of {json.dumps(signature)}")


@lru_cache(maxsize=1)
def _frozen() -> dict[str, Any]:
    """The freeze config with every code and settings hash verified (once per process)."""

    cfg = yaml.safe_load(FREEZE_CONFIG.read_text(encoding="utf-8"))
    if cfg.get("schema_version") != SCHEMA:
        raise FreezeError(f"{FREEZE_CONFIG.name}: unknown schema {cfg.get('schema_version')!r}")
    for rel, want in cfg["code_sha256"].items():
        if _sha(PROJECT / rel) != want:
            raise FreezeError(f"{rel} differs from the frozen file")
    settings = r2.R2Settings.for_variant("main", **cfg["policy"]["settings"])
    if settings.sha256() != cfg["policy"]["settings_sha256"]:
        raise FreezeError("the policy settings differ from the frozen ones")
    if not (settings.stop_threshold == 0.0 and settings.repair_feature and settings.observe_empty):
        raise FreezeError("the frozen policy is stop threshold 0, repair feature on, empty call as an observation")
    return {"config": cfg, "settings": settings, "config_sha256": _sha(FREEZE_CONFIG)}


class FrozenBandit(r2.BanditPolicyR2):
    """The declared round-2 policy with a prior fitted on all eight training and development designs, and a check that the episode's brief is the
    brief the prior belongs to. decide() is the base class's decide() after that check."""

    def __init__(self, *, prior: r2.Prior2, settings: r2.R2Settings, brief_id: str, signature: list[Any],
                 design: str | None = None, seed: Any = None, freeze: Mapping[str, Any] | None = None) -> None:
        super().__init__(prior=prior, settings=settings)
        self.brief_id, self.signature = brief_id, signature
        self.design, self.seed = design, seed              # metadata only: neither is read by any decision
        self.freeze = dict(freeze or {})

    def decide(self, view: PolicyView):
        check_brief(view.brief, self.brief_id, self.signature)
        return super().decide(view)

    def provenance(self) -> dict[str, Any]:
        """What a result row can record about the policy that produced it."""

        return {
            "approach": APPROACH, "policy_id": self.id, "policy_version": self.version, "brief": self.brief_id,
            "prior_sha256": self.prior.sha256, "settings_sha256": self.settings.sha256(),
            "freeze_config_sha256": self.freeze.get("config_sha256"), "plan_sha256": self.freeze.get("plan_sha256"),
            "fit_designs": self.freeze.get("fit_designs"), "design": self.design, "seed": self.seed, "seed_used": False,
        }


def frozen_briefs() -> tuple[str, ...]:
    return tuple(_frozen()["config"]["briefs"])


def bandit(design: str, brief: "str | Brief", seed: Any = None) -> FrozenBandit:
    """The frozen learned policy for one episode. `brief` is the brief id or the Brief object (then it is checked at once, otherwise at the first
    decision). `design` and `seed` are recorded and read by nothing."""

    if not isinstance(design, str) or not design:
        raise TypeError("design must be the design's name")
    brief_id = brief if isinstance(brief, str) else brief.id
    frozen = _frozen()
    cfg = frozen["config"]
    if brief_id in cfg["not_applicable"]:
        raise BanditNotApplicable(f"brief {brief_id}: {cfg['not_applicable'][brief_id]}")
    if brief_id not in cfg["briefs"]:
        raise BanditNotApplicable(f"no frozen prior for brief {brief_id!r}; frozen briefs: {', '.join(cfg['briefs'])}")
    row = cfg["briefs"][brief_id]
    if isinstance(brief, Brief):
        check_brief(brief, brief_id, row["signature"])
    path = PROJECT / row["prior"]
    if _sha(path) != row["prior_sha256"]:
        raise FreezeError(f"{row['prior']} differs from the frozen prior")
    prior = r2.Prior2.load(path)
    settings = frozen["settings"]
    if prior.payload["settings"] != json.loads(settings.canonical()) or prior.payload["meta"]["brief"] != brief_id:
        raise FreezeError(f"{row['prior']} was not fitted under the frozen settings and brief")
    return FrozenBandit(
        prior=prior, settings=settings, brief_id=brief_id, signature=row["signature"], design=design, seed=seed,
        freeze={"config_sha256": frozen["config_sha256"], "plan_sha256": cfg["plan_sha256"], "fit_designs": cfg["fit"]["designs"]})


def run_bandit_episode(
    design: str,
    brief: Brief,
    seed: Any = None,
    *,
    engine: Any,
    registry: Any,
    workers: Mapping[str, Any],
    start_state: Any,
    episode_id: str,
    noise: Mapping[Any, float] | None = None,
    validator: Callable[..., Any] | None = None,
    telemetry_provider: Callable[..., Mapping[str, object]] | None = None,
):
    """One episode of the frozen policy through orch.episode_r2.run_episode_r2 with ROUND2 (both options on). Returns the loop's EpisodeResult."""

    from orch import episode_r2 as loop
    from orch.engine.journal import Journal

    if not isinstance(brief, Brief):
        raise TypeError("run_bandit_episode needs the Brief object the episode runs under")
    policy = bandit(design, brief, seed)
    result = loop.run_episode_r2(
        engine=engine, registry=registry, workers=workers, policy=policy, brief=brief, start_state=start_state,
        episode_id=episode_id, noise=noise, validator=validator, telemetry_provider=telemetry_provider, options=loop.ROUND2)
    recorded = Journal(engine.store.root).read(episode_id)[0]["envelope"].get("loop_options")
    if recorded != loop.ROUND2.as_json():
        raise RuntimeError(f"episode {episode_id} did not run with both round-2 options on: {recorded}")
    return result
