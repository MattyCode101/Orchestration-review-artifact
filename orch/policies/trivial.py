"""A deterministic, deliberately simple policy for engine integration."""

from __future__ import annotations

import hashlib

from orch.contract.models import Mandate, WorkerSettings
from orch.contract.views import PolicyView, StateSummary


def _latest_physical_state(view: PolicyView) -> StateSummary | None:
    """Choose the latest state safe to build on, regardless of brief terms."""
    if view.current.physically_valid:
        return view.current
    for state in reversed(view.visited):
        if state.physically_valid:
            return state
    return view.start if view.start.physically_valid else None


class TrivialPolicy:
    """Keep calling the worker aligned with the brief's objective.

    This is a software integration policy, not the rule-based research policy.
    It has no recipe knowledge and passes no optional guards.
    """

    id = "trivial"
    version = "0.1.0"

    def __init__(
        self,
        *,
        worker_id: str | None = None,
        max_runs: int | None = None,
        preserve_placement: bool = False,
    ) -> None:
        if max_runs is not None and max_runs <= 0:
            raise ValueError("max_runs must be positive")
        self.worker_id = worker_id
        self.max_runs = max_runs
        self.preserve_placement = preserve_placement

    def decide(self, view: PolicyView) -> Mandate | None:
        """Return a mandate from the policy view alone, or stop."""
        if view.budget_left == 0 or not view.worker_cards:
            return None
        base = _latest_physical_state(view)
        if base is None:
            return None
        if self.worker_id is not None:
            cards = [card for card in view.worker_cards if card.id == self.worker_id]
            if not cards:
                raise ValueError(f"worker {self.worker_id} is absent from the view")
            card = cards[0]
        else:
            card = next(
                (
                    candidate
                    for candidate in view.worker_cards
                    if candidate.objective == view.brief.objective
                ),
                view.worker_cards[0],
            )
        context = hashlib.sha256(view.model_dump_json().encode("utf-8")).hexdigest()
        mandate_id = hashlib.sha256(
            f"{self.id}:{self.version}:{context}:{card.id}".encode("utf-8")
        ).hexdigest()
        return Mandate(
            id=f"mandate_{mandate_id[:24]}",
            worker_id=card.id,
            base_state_id=base.state_id,
            settings=WorkerSettings(
                preserve_placement=self.preserve_placement,
                max_runs=min(
                    self.max_runs if self.max_runs is not None else card.default_attempts,
                    view.budget_left,
                ),
            ),
            guards=(),
            brief_hash=view.brief.fingerprint(),
            context_ref=context,
        )

