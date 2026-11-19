"""Template v2 texts rendered by code: brief (P1), worker cards (P1, P2), vocabulary (P1).

The frozen brief text (orch/contract/brief_text.py) is not touched; P1 renders the same frozen contract with:
  The authors' decision 1  never "B2": the start is "S0, the placed-and-routed checkpoint you were given";
  change 2           the brief's noise line ("flag only; never use it to accept or reject") is dropped; the one
                     noise rule lives in the vocabulary;
  change 10          validity in one line;
  defect 7           the reporting line (Fmax and LUT-cut percentages, never shown in the view) is dropped.
Cards: change 7 and the authors' decision 3 (one description line per worker, three wordings to test, from
configs/prompts/template_v2/worker_descriptions.yaml); change 11 (placement words).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from orch.contract.brief_text import METRIC_TEXT, _limits, load_frozen_brief, load_frozen_spec
from orch.contract.models import Brief, WorkerCard
from orch.policies.llm_view import HIGHER_IS_BETTER, METRIC_NAMES

START_ALIAS = "S0"
FINAL_PICK_FROZEN = (
    "best brief-valid state visited; ties broken by the better non-objective metric, then by fewer runs; "
    "if none, the start (no change)"
)
FINAL_PICK_V2 = (
    "the best saved state that meets every limit; ties are broken by the better other metric, then by fewer "
    "runs; if no state qualifies, the start (no change)"
)
LIMIT_TEXT_V2 = {
    "<= start": "must not exceed its value at S0",
    ">= start": "must not fall below its value at S0",
    "<= 1.01 * start": "must not exceed 1.01 times its value at S0",
    ">= start - 0.100": "must not fall more than 0.100 ns below its value at S0",
}
VALIDITY_V2 = ("the final design must be fully routed, functionally equivalent to the original in simulation, "
               "and keep the original constraints")
_NUMBER_WORDS = {1: "one", 2: "two", 3: "three"}


def render_brief_text_v2(brief: Brief | str) -> str:
    brief_id = brief.id if isinstance(brief, Brief) else brief
    expected = load_frozen_brief(brief_id)
    if isinstance(brief, Brief) and brief.fingerprint() != expected.fingerprint():
        raise ValueError("brief contract differs from the frozen specification")
    spec = load_frozen_spec()
    if spec["common"]["final_pick"] != FINAL_PICK_FROZEN:
        raise ValueError("the frozen final-pick text changed; the v2 wording must be re-derived")
    row = spec["briefs"][brief_id]
    limits = _limits(row)
    head = (
        f"Brief {brief_id} ({row['role']}).\n"
        f"Start from {START_ALIAS}, the placed-and-routed checkpoint you were given. "
        f"Budget: at most {spec['common']['budget']['recipe_runs']} recipe runs.\n"
        f"Objective: {row['objective']['direction'].capitalize()} {METRIC_TEXT[row['objective']['metric']]}.\n"
    )
    limit_lines = "; ".join(f"{METRIC_TEXT[lim['metric']]} {LIMIT_TEXT_V2[lim['rule']]}" for lim in limits)
    label = "Hard limit" if len(limits) == 1 else "Hard limits"
    return head + f"{label}: {limit_lines}.\nFinal pick: {FINAL_PICK_V2}.\nValidity: {VALIDITY_V2}.\n"


def render_cards_v2(cards: Sequence[WorkerCard], *, descriptions: Mapping[str, str] | None,
                    placement_v2: bool) -> str:
    """One block per worker; the description line is absent when `descriptions` is None (round-1 cards)."""

    blocks = []
    for card in cards:
        metric = METRIC_NAMES.get(card.objective.metric, card.objective.metric.value)
        better = "higher" if card.objective.metric in HIGHER_IS_BETTER or card.objective.direction == "max" else "lower"
        default = _NUMBER_WORDS.get(card.default_attempts, str(card.default_attempts))
        settings = ("placement may change or must not change" if placement_v2
                    else "preserve_placement true or false")
        head = f"Worker {card.id}: improves {metric} ({better} is better)."
        lines = [head]
        text = (descriptions or {}).get(card.id)
        if text:
            lines[0] = head + " " + text.strip()
        lines[0] += f" Settings: {settings}; attempts one to three (default {default})."
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_vocabulary(*, exact_only: bool) -> str:
    """The P1 vocabulary block (digits allowed: it is rendered by code, not written in the template)."""

    if exact_only:
        return (
            "Vocabulary\n"
            "- Changes and limits are shown as exact values and exact differences. A limit is either met "
            "(inside the limit by some amount, or at the limit) or broken by some amount.\n"
            "- Measurement noise (run-to-run spread on this design) never affects validity or the final pick, "
            "and a change smaller than the noise is not counted as progress."
        )
    return (
        "Vocabulary\n"
        "- A noise band is the run-to-run spread of a metric on this design. A change is within noise when it is "
        "under 1 band, small for 1 to 3 bands, large for 3 to 30 bands and very large for over 30 bands; it is a "
        "gain or a loss according to the metric's preferred direction.\n"
        "- Limit status of a saved state: at limit (equal to the start); within limit by under one band; within "
        "limit with margin; broken by under one band; broken by more.\n"
        "- Noise never affects validity or the final pick, and a change within noise is not counted as progress."
    )
