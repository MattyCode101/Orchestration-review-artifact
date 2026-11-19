"""The placement-property validity rule, defined once.

An output state is valid only if it has no more DONT_TOUCH, fixed-LOC or fixed-BEL cells
than the start state of its episode (decision of 2026-11-07). The earlier rule demanded zero,
which every training start satisfies but another design's start need not. The start state
is the reference, so a start may carry any non-negative number of such cells.

The counts are the meter's Evidence.v_props. A missing, non-integer or negative count fails
closed: absence means unknown, never zero. The rule is applied per property, so fewer cells
of one kind never offset more of another.
"""

from __future__ import annotations

from orch.contract.models import Evidence

PROPERTY_KEYS = ("n_DONT_TOUCH", "n_IS_LOC_FIXED", "n_IS_BEL_FIXED")


class PropertyRuleError(ValueError):
    """The counts are unusable, or an output exceeds its start state's counts."""


def _counts(evidence: Evidence | None) -> dict[str, int] | None:
    """The three counts as non-negative integers, or None if any is missing or ill-typed."""

    props = evidence.v_props if evidence is not None else None
    if not isinstance(props, dict):
        return None
    counts: dict[str, int] = {}
    for key in PROPERTY_KEYS:
        value = props.get(key)
        if type(value) is not int or value < 0:
            return None
        counts[key] = value
    return counts


def check_start_counts(start: Evidence) -> dict[str, int]:
    """Accept any usable counts on a start state, and return them."""

    counts = _counts(start)
    if counts is None:
        raise PropertyRuleError(f"start state lacks DONT_TOUCH/fixed LOC/BEL counts: {start.state_id}")
    return counts


def check_not_above_start(evidence: Evidence, start: Evidence | None) -> None:
    """Raise unless every count of evidence is no higher than the same count of start."""

    base = _counts(start)
    if base is None:
        name = start.state_id if start is not None else "?"
        raise PropertyRuleError(f"start state lacks DONT_TOUCH/fixed LOC/BEL counts: {name}")
    counts = _counts(evidence)
    if counts is None:
        raise PropertyRuleError(f"missing DONT_TOUCH/fixed LOC/BEL counts: {evidence.state_id}")
    over = [f"{key} {counts[key]} > {base[key]}" for key in PROPERTY_KEYS if counts[key] > base[key]]
    if over:
        raise PropertyRuleError(
            f"more DONT_TOUCH/fixed LOC/BEL cells than the start state: {evidence.state_id} "
            f"({', '.join(over)})"
        )
