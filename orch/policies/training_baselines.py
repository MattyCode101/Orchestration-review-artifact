"""Fixed training-pilot baselines over the same PolicyView and Mandate."""
from __future__ import annotations

import hashlib
import json
import random

from orch.contract.models import Mandate, MetricName, WorkerCard, WorkerSettings
from orch.contract.views import PolicyView, StateSummary


def _card_for(view: PolicyView, metric: MetricName) -> WorkerCard:
    matches = [card for card in view.worker_cards if card.objective.metric == metric]
    if len(matches) != 1:
        raise ValueError(f"expected one worker card for {metric}")
    return matches[0]


def _latest(view: PolicyView) -> StateSummary:
    """Use the latest selected physical result, with no brief rollback."""

    if view.current.physically_valid:
        return view.current
    for state in reversed(view.visited):
        if state.physically_valid:
            return state
    return view.start


def _mandate(
    view: PolicyView, *, policy_id: str, version: str,
    card: WorkerCard, base: StateSummary,
    preserve_placement: bool, attempts: int,
) -> Mandate:
    if view.budget_left < attempts:
        raise ValueError("policy attempted more runs than remain")
    action = (
        policy_id, version, len(view.calls), card.id, base.state_id,
        preserve_placement, attempts,
    )
    digest = hashlib.sha256(
        json.dumps(action, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    context = json.dumps({
        "policy": policy_id, "version": version, "call_index": len(view.calls),
        "base_state_id": base.state_id,
    }, sort_keys=True, separators=(",", ":"))
    return Mandate(
        id="mandate_" + digest[:24],
        worker_id=card.id,
        base_state_id=base.state_id,
        settings=WorkerSettings(
            preserve_placement=preserve_placement, max_runs=attempts,
        ),
        guards=(),
        brief_hash=view.brief.fingerprint(),
        context_ref=context,
    )


class SingleSpecialistPolicy:
    """Call only the brief's primary worker from its latest selected result."""

    id = "single_specialist"
    version = "1.0-frozen"

    def decide(self, view: PolicyView) -> Mandate | None:
        if view.budget_left < 2:
            return None
        card = _card_for(view, view.brief.objective.metric)
        return _mandate(
            view, policy_id=self.id, version=self.version, card=card,
            base=_latest(view), preserve_placement=False, attempts=2,
        )


class FixedAlternationPolicy:
    """Alternate timing A and area B, starting with the selected worker."""

    version = "1.0-frozen"

    def __init__(self, *, first: str) -> None:
        if first not in ("A", "B"):
            raise ValueError("fixed order must start with A or B")
        self.first = first
        self.id = "fixed_AB" if first == "A" else "fixed_BA"

    def decide(self, view: PolicyView) -> Mandate | None:
        if view.budget_left < 2:
            return None
        order = ("A", "B") if self.first == "A" else ("B", "A")
        selected = order[len(view.calls) % 2]
        metric = MetricName.WNS_NS if selected == "A" else MetricName.PACKED_LUTS
        card = _card_for(view, metric)
        return _mandate(
            view, policy_id=self.id, version=self.version, card=card,
            base=_latest(view), preserve_placement=False, attempts=2,
        )


class UniformRandomPolicy:
    """Seeded uniform worker, placement, call size and base-state baseline."""

    id = "random_uniform"

    def __init__(self, *, seed: int) -> None:
        self.seed = int(seed)
        self.version = f"1.0-frozen-seed-{self.seed}"

    def decide(self, view: PolicyView) -> Mandate | None:
        if view.budget_left <= 0:
            return None
        cards = (
            _card_for(view, MetricName.WNS_NS),
            _card_for(view, MetricName.PACKED_LUTS),
        )
        payload = json.dumps({
            "seed": self.seed,
            "view": view.model_dump(mode="json"),
        }, sort_keys=True, separators=(",", ":"))
        rng = random.Random(
            int(hashlib.sha256(payload.encode("utf-8")).hexdigest(), 16)
        )
        card = rng.choice(cards)
        preserve = rng.choice((False, True))
        attempts = rng.choice((1, 2) if view.budget_left >= 2 else (1,))
        base = rng.choice((_latest(view), view.best_valid or view.start))
        return _mandate(
            view, policy_id=self.id, version=self.version, card=card,
            base=base, preserve_placement=preserve, attempts=attempts,
        )
