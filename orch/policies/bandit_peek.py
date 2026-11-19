"""Ask the bandit what it would choose at a given PolicyView, without running it as a policy.

For agreement analyses (for example O2's comparison of the LLM's choices with the bandit and a greedy-experience rule).
The prior is the leave-one-design-out prior of the view's own design and brief; the episode's calls are another policy's,
so no in-episode learning is applied (only `rewards` is overridden). It reads the PolicyView and the prior file only.

    from orch.policies.bandit_peek import ask
    choice = ask(view, brief_id="A0", held_out="rosetta_spam-filter")      # dict, or {"stop": True}

The priors are data/bandit_2026-11-11/priors/<variant>_<brief>_<held_out>.json (held_out = a training design, or "none"
for a prior built from all five designs, which includes the evaluated design and must not be used for held-out claims).
"""
from __future__ import annotations

import json
from pathlib import Path

from orch.contract.views import PolicyView
from orch.policies.bandit import BanditPolicy, BanditSettings, Prior

PROJECT = Path(__file__).resolve().parents[2]
PRIORS = PROJECT / "data/bandit_2026-11-11"
SETTINGS = PROJECT / "configs/policies/bandit.yaml"


class PeekBandit(BanditPolicy):
    """The declared choice rule with no in-episode learning."""

    def rewards(self, view: PolicyView):
        return []


def load(brief_id: str, held_out: str, *, variant: str = "main", priors_dir: str = "priors", **overrides) -> PeekBandit:
    import yaml

    common = dict(yaml.safe_load(SETTINGS.read_text(encoding="utf-8"))["settings"], **overrides)
    settings = BanditSettings.for_variant(variant, **common)
    return PeekBandit(prior=Prior.load(PRIORS / priors_dir / f"{variant}_{brief_id}_{held_out}.json"), settings=settings)


def ask(view: PolicyView, *, brief_id: str, held_out: str, variant: str = "main", **overrides) -> dict:
    """The arm the bandit would call at this view: worker, scope, attempts, base kind, and its bound; or {"stop": True}."""

    policy = load(brief_id, held_out, variant=variant, **overrides)
    mandate = policy.decide(view)
    if mandate is None:
        return {"stop": True}
    arm = json.loads(mandate.context_ref)["arm"]
    worker, scope, attempts, base = arm.split("|")
    row = next(r for r in policy.scored(view) if r["key"] == arm)
    return {"worker": worker, "preserve": scope == "keep", "attempts": int(attempts), "base": base, "arm": arm,
            "ucb": row["ucb"], "mean": row["mean"], "n": row["n"]}
