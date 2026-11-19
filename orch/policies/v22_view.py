"""Template v2.2, switch H: the compact state list (new file; orch/policies/v2_view.py is untouched).

"As now" (H = full) is orch/policies/v2_view.render_view_v2. The compact form keeps what a decision needs about the
latest situation and drops the history of how each state came about:

  kept     the budget and the two pointers (best state within limits, last selected state); per state: whether it meets
           every limit, the exact value of every goal and limit metric with its category against the start, the other
           metrics against the start, the limit statuses, the goal against the best state within limits, and the list
           of calls tried from the state (with "no options left", which the repeat rule needs)
  dropped  parent, made-by, selected-by-the-worker, every figure against the parent, the exact differences against the
           start, and the trajectory log (the step-by-step record with the policy's own earlier reasons)

Aliases follow the same order as render_view_v2, so a state has the same alias in either form.
"""
from __future__ import annotations

import hashlib

from orch.contract.models import MetricName
from orch.contract.views import CallSummary, PolicyView, StateSummary
from orch.policies.llm_view import HIGHER_IS_BETTER, METRIC_NAMES, SHOWN, effective_bands
from orch.policies.r2_view import brief_metrics, value_text
from orch.policies.v2_view import START_NOTE, BuiltViewV2, unique_states
from orch.policies.v2_vocab import PLACEMENT_V2, categorise_v2, limit_status_v2

NO_BAND = "(no noise band recorded)"


def aliases_of(view: PolicyView) -> dict[str, str]:
    """state id -> alias, in the order every v2 rendering uses."""

    return {s.state_id: f"S{i}" for i, s in enumerate(unique_states(view))}


def attempts_phrase(attempts: int | None) -> str:
    if attempts is None:
        return "default attempts"
    return f"{attempts} attempt" if attempts == 1 else f"{attempts} attempts"


def category_v2(bands: dict[MetricName, float], metric: MetricName, delta: float, *, zero_band: bool = False) -> str | None:
    band = bands.get(metric)
    if not band:
        if not zero_band:
            return None
        band = 0.0
    return categorise_v2(delta, band, higher_is_better=metric in HIGHER_IS_BETTER)


def limit_statuses(view: PolicyView, state: StateSummary, bands: dict[MetricName, float]) -> list[str]:
    """One status string per brief limit, in brief order (same words as the full view)."""

    out = []
    start = view.start
    for term in view.brief.terms:
        reference, value = start.metrics.get(term.metric), state.metrics.get(term.metric)
        if reference is None or value is None:
            continue
        threshold = term.factor * reference + term.offset
        margin = (threshold - value) if term.comparison == "<=" else (value - threshold)
        start_relative = term.factor == 1.0 and term.offset == 0.0
        status = limit_status_v2(margin, bands.get(term.metric, 0.0), start_relative=start_relative)
        out.append(f"{METRIC_NAMES.get(term.metric, term.metric.value)} {status}")
    return out


def state_summary_line(view: PolicyView, state: StateSummary, bands: dict[MetricName, float]) -> str:
    """One line of facts about a state: limits met, goal against the start, limit statuses (no intent words)."""

    start, goal = view.start, view.brief.objective.metric
    parts = [f"meets every limit: {'yes' if state.brief_valid else 'no'}"]
    if state.state_id != start.state_id and goal in state.metrics and goal in start.metrics:
        cat = category_v2(bands, goal, state.metrics[goal] - start.metrics[goal])
        parts.append(f"{METRIC_NAMES.get(goal, goal.value)} {cat or NO_BAND} vs start")
    lim = limit_statuses(view, state, bands)
    if lim:
        parts.append("limits: " + ", ".join(lim))
    return "; ".join(parts)


def render_view_compact(view: PolicyView) -> BuiltViewV2:
    states = unique_states(view)
    aliases = {s.state_id: f"S{i}" for i, s in enumerate(states)}
    start = view.start
    bands = effective_bands(view.noise_bands, start.metrics)
    cards = {c.id: c for c in view.worker_cards}
    goal = view.brief.objective.metric
    best = view.best_valid if view.best_valid is not None else start
    brief_ms = brief_metrics(view)
    brief_names = {METRIC_NAMES.get(m, m.value) for m in brief_ms}
    used = view.brief.budget_runs - view.budget_left

    def attempts_of(worker_id: str, max_runs: int | None) -> int | None:
        if max_runs is not None:
            return max_runs
        card = cards.get(worker_id)
        return card.default_attempts if card is not None else None

    def exhausted(call: CallSummary) -> str:
        requested = attempts_of(call.worker_id, call.settings.max_runs)
        if requested is None or call.recipe_runs >= requested:
            return ""
        return f" (only {call.recipe_runs} of {requested} attempts could run: no options left)"

    def result_of(call: CallSummary) -> str:
        if call.selected_state_id is None:
            return "nothing selectable"
        if call.selected_state_id == call.base_state_id:
            return "unchanged"
        return aliases.get(call.selected_state_id, "unknown")

    tried: dict[str, list[str]] = {}
    for call in view.calls:
        line = (f"{call.worker_id} ({PLACEMENT_V2[call.settings.preserve_placement]}, "
                f"{attempts_phrase(attempts_of(call.worker_id, call.settings.max_runs))}) -> {result_of(call)}"
                f"{exhausted(call)}")
        tried.setdefault(call.base_state_id, []).append(line)

    lines = [
        f"budget: used {used}, remaining {view.budget_left}, total {view.brief.budget_runs}",
        f"best state within limits so far: {aliases.get(best.state_id, 'S0')}",
        f"last selected state: {aliases.get(view.current.state_id, 'S0')}",
        "saved states:",
    ]
    for state in states:
        alias = aliases[state.state_id]
        is_start = state.state_id == start.state_id
        lines.append(f"{alias} ({START_NOTE})" if is_start else alias)
        lines.append(f"  meets every limit: {'yes' if state.brief_valid else 'no'}")
        for metric in brief_ms:
            if metric not in state.metrics:
                continue
            text = f"  {METRIC_NAMES.get(metric, metric.value)}: {value_text(metric, state.metrics[metric])}"
            if not is_start and metric in start.metrics:
                cat = category_v2(bands, metric, state.metrics[metric] - start.metrics[metric])
                text += f" ({cat or NO_BAND} vs start)" if cat else f" {NO_BAND}"
            lines.append(text)
        if not is_start:
            others = []
            for metric in SHOWN:
                name = METRIC_NAMES[metric]
                if name in brief_names or metric not in state.metrics or metric not in start.metrics:
                    continue
                cat = category_v2(bands, metric, state.metrics[metric] - start.metrics[metric])
                if cat is not None:
                    others.append(f"{name} {cat}")
            if others:
                lines.append("  other metrics vs start: " + ", ".join(others))
        lines.append("  limits: " + (", ".join(limit_statuses(view, state, bands)) or "none"))
        if goal in state.metrics and goal in best.metrics:
            gvb = category_v2(bands, goal, state.metrics[goal] - best.metrics[goal], zero_band=True) or "within noise"
        else:
            gvb = "within noise"
        lines.append(f"  goal vs best state within limits: {gvb}")
        if state.state_id in tried:
            lines.append("  tried:")
            lines += [f"    - {entry}" for entry in tried[state.state_id]]
        else:
            lines.append("  tried: nothing yet")
    text = "\n".join(lines) + "\n"
    return BuiltViewV2(text, {a: s for s, a in aliases.items()}, dict(aliases), hashlib.sha256(text.encode("utf-8")).hexdigest())
