"""Fixed-order and random baselines over a brief's relevant workers.

For a two-worker brief these reduce to the existing fixed A/B alternation and
uniform-random baselines' structure. For P0 the relevant workers are the
primary (power, C) and one worker per hard limit (timing A for WNS, area B for
LUTs). Every mandate names a worker, base state and limits, never a recipe.
"""
from __future__ import annotations

import hashlib
import json
import random

from orch.contract.models import Mandate, WorkerCard
from orch.contract.views import PolicyView
from orch.policies.training_baselines import _latest, _mandate

_LETTER_TO_WORKER = {"A": "timing", "B": "area", "C": "power"}


def relevant_cards(view: PolicyView) -> tuple[WorkerCard, ...]:
    """The brief's primary worker, then one worker per hard limit, without repeats."""

    cards: list[WorkerCard] = []
    metrics = [view.brief.objective.metric, *[term.metric for term in view.brief.terms]]
    for metric in metrics:
        matches = [card for card in view.worker_cards if card.objective.metric == metric]
        if len(matches) != 1:
            raise ValueError(f"expected one worker card for {metric}")
        if matches[0] not in cards:
            cards.append(matches[0])
    return tuple(cards)


class FixedOrderPolicy:
    """Cycle a fixed order of the three workers on the latest physical result."""

    version = "1.0-frozen"

    def __init__(self, *, order: str) -> None:
        if len(order) != 3 or sorted(order) != ["A", "B", "C"]:
            raise ValueError("fixed order must be a permutation of A, B and C")
        self.order = order
        self.id = f"fixed_{order}"

    def decide(self, view: PolicyView) -> Mandate | None:
        if view.budget_left < 2:
            return None
        worker_id = _LETTER_TO_WORKER[self.order[len(view.calls) % 3]]
        matches = [card for card in view.worker_cards if card.id == worker_id]
        if len(matches) != 1:
            raise ValueError(f"expected one worker card named {worker_id}")
        return _mandate(
            view, policy_id=self.id, version=self.version, card=matches[0],
            base=_latest(view), preserve_placement=False, attempts=2,
        )


class RelevantRandomPolicy:
    """Seeded uniform worker, placement, call size and base over relevant workers."""

    id = "random_uniform"

    def __init__(self, *, seed: int) -> None:
        self.seed = int(seed)
        self.version = f"1.0-frozen-seed-{self.seed}-relevant"

    def decide(self, view: PolicyView) -> Mandate | None:
        if view.budget_left <= 0:
            return None
        cards = relevant_cards(view)
        payload = json.dumps({
            "seed": self.seed, "view": view.model_dump(mode="json"),
        }, sort_keys=True, separators=(",", ":"))
        rng = random.Random(int(hashlib.sha256(payload.encode("utf-8")).hexdigest(), 16))
        card = rng.choice(cards)
        preserve = rng.choice((False, True))
        attempts = rng.choice((1, 2) if view.budget_left >= 2 else (1,))
        base = rng.choice((_latest(view), view.best_valid or view.start))
        return _mandate(
            view, policy_id=self.id, version=self.version, card=card,
            base=base, preserve_placement=preserve, attempts=attempts,
        )
