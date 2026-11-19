"""A worker that runs exactly the one recipe a recipe-level agent asked for.

The general-agent baseline acts on recipes directly, a privilege real black-box workers
would not offer. To keep the engine, journal and gate untouched, such an action is carried
as an ordinary mandate to the recipe's own worker (timing, area, power), with one attempt;
the recipe id travels in the mandate's `context_ref` under "x". This worker plans just that
recipe, and only if it belongs to the worker's set and has not yet been run on the base
state. Selection is the ordinary one (the single result, if physically valid).
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from orch.contract.models import AppliedStep, Mandate, RunStatus
from orch.workers.base import GenericWorker


def requested_recipe(mandate: Mandate) -> str | None:
    try:
        data = json.loads(mandate.context_ref) if mandate.context_ref else {}
    except json.JSONDecodeError:
        return None
    recipe = data.get("x") if isinstance(data, dict) else None
    return recipe if isinstance(recipe, str) and recipe else None


class DirectRecipeWorker(GenericWorker):
    def plan(
        self, state, features: Mapping[str, object] | None, mandate: Mandate,
        history_on_this_state: Sequence[AppliedStep],
    ) -> tuple[str, ...]:
        if mandate.worker_id != self.card.id:
            raise ValueError("mandate names a different worker")
        if state.id != mandate.base_state_id:
            raise ValueError("mandate names a different base state")
        if state.stage != self.card.accepts_stage:
            raise ValueError("worker does not accept this state stage")
        recipe = requested_recipe(mandate)
        if recipe is None or recipe not in self.recipe_ids:
            return ()
        tried = {
            step.outcome.run.recipe_id for step in history_on_this_state
            if step.outcome.run.status != RunStatus.TRANSIENT
        }
        return () if recipe in tried else (recipe,)
