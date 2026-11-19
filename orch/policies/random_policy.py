"""Seeded random worker and setting choices over the same PolicyView."""

from __future__ import annotations

import hashlib
import json
import random

from orch.contract.models import Mandate, WorkerSettings
from orch.contract.views import PolicyView
from orch.policies.trivial import _latest_physical_state


class RandomPolicy:
    """A reproducible random baseline with no recipe or runner access.

    The RNG is derived afresh from the seed and complete current view. A
    restarted episode therefore makes the same choice at the same view.
    """

    id = "random"

    def __init__(self, *, seed: int, max_runs: int | None = None) -> None:
        if max_runs is not None and max_runs <= 0:
            raise ValueError("max_runs must be positive")
        self.seed = int(seed)
        self.max_runs = max_runs
        self.version = f"0.1.0-seed-{self.seed}"

    def decide(self, view: PolicyView) -> Mandate | None:
        """Choose worker, replacement setting and call size uniformly."""
        if view.budget_left == 0 or not view.worker_cards:
            return None
        base = _latest_physical_state(view)
        if base is None:
            return None
        payload = json.dumps(
            {"seed": self.seed, "view": view.model_dump(mode="json")},
            sort_keys=True,
            separators=(",", ":"),
        )
        context = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        rng = random.Random(int(context, 16))
        card = rng.choice(view.worker_cards)
        preserve_placement = bool(rng.getrandbits(1))
        max_runs = rng.randint(
            1, min(
                self.max_runs if self.max_runs is not None else card.default_attempts,
                view.budget_left,
            ),
        )
        action = f"{card.id}:{preserve_placement}:{max_runs}"
        mandate_id = hashlib.sha256(
            f"{self.id}:{self.version}:{context}:{action}".encode("utf-8")
        ).hexdigest()
        return Mandate(
            id=f"mandate_{mandate_id[:24]}",
            worker_id=card.id,
            base_state_id=base.state_id,
            settings=WorkerSettings(
                preserve_placement=preserve_placement,
                max_runs=max_runs,
            ),
            guards=(),
            brief_hash=view.brief.fingerprint(),
            context_ref=context,
        )

