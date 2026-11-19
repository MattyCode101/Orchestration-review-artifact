"""Fail-closed campaign guard outside the frozen AMD validator fingerprint.

The running 4 November backfill pins verification.py and the adapter by hash.
This module tightens campaign commit and resume rules without changing those
validator inputs or the copied AMD scripts.
"""
from __future__ import annotations

from typing import Iterable

from orch.baselines.verification import VerificationLedger, VerifiedStepEngine
from orch.contract.models import DesignState, RunOutcome
from orch.interfaces import RecipeSpec


class FailClosedVerifiedStepEngine(VerifiedStepEngine):
    """Require AMD PASS for every saved output, including meter-error outputs."""

    def execute(self, state: DesignState, recipe: RecipeSpec) -> RunOutcome:
        outcome = super().execute(state, recipe)
        if outcome.state is not None:
            # The base wrapper runs the check for every saved state, but only
            # raises on FAIL/UNKNOWN when the recipe status is OK. A meter can
            # fail after the DCP was saved, so the campaign must check here too.
            self.ledger.require_pass(outcome.state.design, outcome.state.id)
        return outcome


def audit_all_saved(
    journal: Iterable[dict], ledger: VerificationLedger
) -> dict[str, int]:
    """Require PASS for outputs from every journal attempt, not just its latest."""

    edge_attempts = 0
    saved_attempts = 0
    sequences: set[tuple[str, tuple[str, ...]]] = set()
    outputs: set[tuple[str, str]] = set()
    for event in journal:
        if event.get("type") != "edge":
            continue
        edge = event["edge"]
        design = edge["design"]
        if design not in ledger.references:
            raise ValueError("journal contains an unselected design")
        edge_attempts += 1
        sequences.add((design, tuple(edge["sequence"])))
        output_id = edge.get("output_state_id")
        if output_id:
            saved_attempts += 1
            outputs.add((design, output_id))
    for design, output_id in sorted(outputs):
        ledger.require_pass(design, output_id)
    return {
        "journal_edge_attempts": edge_attempts,
        "distinct_sequences": len(sequences),
        "saved_output_attempts": saved_attempts,
        "distinct_verified_dcps": len(outputs),
    }
