"""Pure fuzzy-view builder for the LLM orchestrator (final specification, section 4).

The orchestrator never sees raw numbers, design names, paths, hashes or recipes.
Each change is a category relative to that design's noise band b, and each brief
limit is a status relative to the brief. States are named S0, S1, ... in order of
discovery; the alias-to-state map stays outside the rendered text.

    change category (preferred direction)      limit status
    large_gain   better by more than 3b        satisfied_margin  inside by more than b
    small_gain   better by b to 3b             satisfied_tight   inside by at most b
    within_noise changed by less than b        broken_small      beyond by up to 3b
    small_loss   worse by b to 3b              broken_large      beyond by more than 3b
    large_loss   worse by more than 3b

Edges are inclusive towards the smaller category (exactly b is a small change,
exactly 3b is still small), with a relative tolerance of 1e-9 so that rounding
cannot move a value across an edge. An "exact" mode, used only as an ablation,
adds absolute values and exact deltas to the same structure.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from orch.contract.models import FrozenModel, MetricName
from orch.contract.views import CallSummary, PolicyView, StateSummary, StepSummary

CATEGORIES = ("large_gain", "small_gain", "within_noise", "small_loss", "large_loss")
LIMIT_STATUSES = ("satisfied_margin", "satisfied_tight", "broken_small", "broken_large")
METRIC_NAMES = {
    MetricName.WNS_NS: "wns",
    MetricName.PACKED_LUTS: "packed_luts",
    MetricName.NETLIST_LUTS: "netlist_luts",
    MetricName.FFS: "ffs",
    MetricName.DYNAMIC_POWER_W: "power",
}
SHOWN = tuple(METRIC_NAMES)
HIGHER_IS_BETTER = frozenset({MetricName.WNS_NS})
_REL = 1e-9
_INSIDE_TOLERANCE = 1e-9


def categorise(delta: float, band: float, *, higher_is_better: bool) -> str:
    """Bin one change (candidate minus reference) by the metric's preferred direction."""

    better = delta if higher_is_better else -delta
    size = abs(better)
    if size == 0.0 or size < band * (1.0 - _REL):
        return "within_noise"
    large = size > 3.0 * band * (1.0 + _REL)
    if better > 0:
        return "large_gain" if large else "small_gain"
    return "large_loss" if large else "small_loss"


def limit_status(margin: float, band: float) -> str:
    """Status of a limit; margin is the distance inside the limit (negative when broken)."""

    if margin >= -_INSIDE_TOLERANCE:
        return "satisfied_margin" if margin > band * (1.0 + _REL) else "satisfied_tight"
    return "broken_small" if -margin <= 3.0 * band * (1.0 + _REL) else "broken_large"


def effective_bands(
    noise: Mapping[MetricName, float], start: Mapping[MetricName, float],
) -> dict[MetricName, float]:
    """Absolute noise bands for every metric the view can show.

    Recorded bands are used as given. Netlist LUTs and FFs have no recorded band,
    so they borrow the design's packed-LUT fraction (an assumption, flagged in the
    decision log). Dynamic power is shown only when a recorded band exists.
    """

    bands = {m: float(noise[m]) for m in SHOWN if m in noise}
    packed, packed_start = bands.get(MetricName.PACKED_LUTS), start.get(MetricName.PACKED_LUTS)
    if packed is not None and packed_start:
        fraction = packed / abs(packed_start)
        for metric in (MetricName.NETLIST_LUTS, MetricName.FFS):
            if metric not in bands and metric in start:
                bands[metric] = fraction * abs(start[metric])
    return bands


def placement_phrase(preserve_placement: bool) -> str:
    return "placement preserved" if preserve_placement else "re-placing allowed"


def attempts_phrase(attempts: int | None) -> str:
    if attempts is None:
        return "default attempts"
    return f"{attempts} attempt" if attempts == 1 else f"{attempts} attempts"


class FuzzyState(FrozenModel):
    id: str
    parent: str
    made_by: str
    picked_by_worker: str
    valid: bool = True
    brief_valid: bool
    limits: dict[str, str]
    vs_start: dict[str, str]
    vs_parent: dict[str, str]
    goal_vs_best: str
    tried: tuple[str, ...] = ()
    values: dict[str, float] | None = None
    exact_vs_start: dict[str, float] | None = None
    exact_vs_parent: dict[str, float] | None = None


class FuzzyView(FrozenModel):
    mode: Literal["fuzzy", "exact"]
    budget: dict[str, int]
    best_valid_state: str
    current_state: str
    states: tuple[FuzzyState, ...]
    trajectory_log: tuple[str, ...]


@dataclass(frozen=True)
class BuiltView:
    fuzzy: FuzzyView
    alias_to_state: dict[str, str]
    state_to_alias: dict[str, str]
    text: str
    sha256: str


def _num(value: float) -> str:
    return f"{value:.6g}"


def _signed(value: float) -> str:
    return f"{value:+.6g}"


def _unique_states(view: PolicyView) -> list[StateSummary]:
    seen: dict[str, StateSummary] = {}
    for state in (view.start, *view.visited):
        seen.setdefault(state.state_id, state)
    return list(seen.values())


def _reason(call: CallSummary) -> str:
    try:
        data = json.loads(call.context_ref) if call.context_ref else {}
    except json.JSONDecodeError:
        data = {}
    reason = data.get("r") if isinstance(data, dict) else None
    cleaned = reason.strip().rstrip(".").strip() if isinstance(reason, str) else ""
    return cleaned or "not recorded"


def recipe_of(call: CallSummary) -> str:
    """The recipe a general-agent call ran, recorded in the mandate's context_ref under "x"."""

    try:
        data = json.loads(call.context_ref) if call.context_ref else {}
    except json.JSONDecodeError:
        data = {}
    recipe = data.get("x") if isinstance(data, dict) else None
    return recipe if isinstance(recipe, str) and recipe else "unknown"


def build_fuzzy_view(
    view: PolicyView, *, mode: Literal["fuzzy", "exact"] = "fuzzy", recipe_level: bool = False,
) -> BuiltView:
    """Build the orchestrator's view of one episode from the policy-safe PolicyView.

    With `recipe_level` the same builder serves the general agent: only the description of
    an action changes (a recipe instead of a worker, placement setting and attempt count).
    """

    states = _unique_states(view)
    aliases = {state.state_id: f"S{index}" for index, state in enumerate(states)}
    by_alias = {aliases[state.state_id]: state for state in states}
    start = view.start
    bands = effective_bands(view.noise_bands, start.metrics)
    cards = {card.id: card for card in view.worker_cards}
    goal = view.brief.objective.metric

    def attempts_of(worker_id: str, max_runs: int | None) -> int | None:
        if max_runs is not None:
            return max_runs
        card = cards.get(worker_id)
        return card.default_attempts if card is not None else None

    def exhausted_note(call: CallSummary) -> str:
        """Fewer runs than requested means the worker had nothing more to try on that state."""

        requested = attempts_of(call.worker_id, call.settings.max_runs)
        if recipe_level or requested is None or call.recipe_runs >= requested:
            return ""
        return f" (only {call.recipe_runs} of {requested} attempts could run: nothing more to try there)"

    def change(state: StateSummary, ref: StateSummary) -> dict[str, str]:
        out: dict[str, str] = {}
        for metric in SHOWN:
            if metric in bands and metric in state.metrics and metric in ref.metrics:
                out[METRIC_NAMES[metric]] = categorise(
                    state.metrics[metric] - ref.metrics[metric], bands[metric],
                    higher_is_better=metric in HIGHER_IS_BETTER,
                )
        return out

    def exact_change(state: StateSummary, ref: StateSummary) -> dict[str, float]:
        return {
            METRIC_NAMES[m]: state.metrics[m] - ref.metrics[m]
            for m in SHOWN if m in state.metrics and m in ref.metrics
        }

    def limits_of(state: StateSummary) -> dict[str, str]:
        out: dict[str, str] = {}
        for term in view.brief.terms:
            reference = start.metrics.get(term.metric)
            value = state.metrics.get(term.metric)
            if reference is None or value is None:
                continue
            threshold = term.factor * reference + term.offset
            margin = (threshold - value) if term.comparison == "<=" else (value - threshold)
            out[METRIC_NAMES.get(term.metric, term.metric.value)] = limit_status(
                margin, bands.get(term.metric, 0.0)
            )
        return out

    best = view.best_valid if view.best_valid is not None else start
    producer: dict[str, StepSummary] = {}
    for step in view.steps:
        out_id = step.output_state_id
        if out_id is None or out_id == step.base_state_id or out_id in producer:
            continue
        if out_id == start.state_id or out_id not in aliases:
            continue
        producer[out_id] = step

    def call_line(call: CallSummary) -> str:
        if call.selected_state_id is None:
            result = "no usable state"
        elif call.selected_state_id == call.base_state_id:
            result = "unchanged"
        else:
            result = aliases.get(call.selected_state_id, "unknown")
        if recipe_level:
            return f"recipe {recipe_of(call)} -> {result}"
        attempts = attempts_of(call.worker_id, call.settings.max_runs)
        return (
            f"{call.worker_id} ({placement_phrase(call.settings.preserve_placement)}, "
            f"{attempts_phrase(attempts)}) -> {result}{exhausted_note(call)}"
        )

    tried: dict[str, list[str]] = {}
    for call in view.calls:
        tried.setdefault(call.base_state_id, []).append(call_line(call))

    fuzzy_states: list[FuzzyState] = []
    for state in states:
        alias = aliases[state.state_id]
        step = producer.get(state.state_id)
        if state.state_id == start.state_id:
            parent_alias, made_by, picked = "none", "start", "n/a"
            vs_start, vs_parent = {}, {}
            exact_start = exact_parent = None
        else:
            parent_id = step.base_state_id if step is not None else None
            parent_alias = aliases.get(parent_id, "unknown") if parent_id else "unknown"
            if recipe_level:
                maker = next((c for c in view.calls
                              if c.selected_state_id == state.state_id and c.base_state_id == parent_id), None)
                made_by, picked = (f"recipe {recipe_of(maker)}" if maker is not None else "unknown"), "n/a"
            elif step is not None:
                made_by = (
                    f"{step.worker_id}, {placement_phrase(step.settings.preserve_placement)}, "
                    f"{attempts_phrase(attempts_of(step.worker_id, step.settings.max_runs))}"
                )
                picked = "yes" if any(
                    c.selected_state_id == state.state_id and c.base_state_id == parent_id
                    and c.worker_id == step.worker_id for c in view.calls
                ) else "no"
            else:
                made_by, picked = "unknown", "no"
            vs_start = change(state, start)
            parent_state = by_alias.get(parent_alias)
            vs_parent = change(state, parent_state) if parent_state is not None else {}
            exact_start = exact_change(state, start)
            exact_parent = exact_change(state, parent_state) if parent_state is not None else {}
        goal_gap = categorise(
            state.metrics.get(goal, 0.0) - best.metrics.get(goal, 0.0),
            bands.get(goal, 0.0), higher_is_better=goal in HIGHER_IS_BETTER,
        ) if goal in state.metrics and goal in best.metrics else "within_noise"
        fuzzy_states.append(FuzzyState(
            id=alias, parent=parent_alias, made_by=made_by, picked_by_worker=picked,
            brief_valid=state.brief_valid, limits=limits_of(state), vs_start=vs_start,
            vs_parent=vs_parent, goal_vs_best=goal_gap,
            tried=tuple(tried.get(state.state_id, ())),
            values=({METRIC_NAMES[m]: state.metrics[m] for m in SHOWN if m in state.metrics}
                    if mode == "exact" else None),
            exact_vs_start=exact_start if mode == "exact" else None,
            exact_vs_parent=exact_parent if mode == "exact" else None,
        ))

    log: list[str] = []
    for index, call in enumerate(view.calls, start=1):
        base_alias = aliases.get(call.base_state_id, "unknown")
        attempts = attempts_of(call.worker_id, call.settings.max_runs)
        if call.selected_state_id is None:
            outcome = "no usable state returned"
        elif call.selected_state_id == call.base_state_id:
            outcome = "unchanged"
        else:
            picked_state = next((s for s in states if s.state_id == call.selected_state_id), None)
            base_state = next((s for s in states if s.state_id == call.base_state_id), None)
            parts = [f"{aliases.get(call.selected_state_id, 'unknown')} chosen"]
            if picked_state is not None and base_state is not None and goal in picked_state.metrics:
                parts.append("goal " + categorise(
                    picked_state.metrics[goal] - base_state.metrics.get(goal, picked_state.metrics[goal]),
                    bands.get(goal, 0.0), higher_is_better=goal in HIGHER_IS_BETTER,
                ))
            if picked_state is not None:
                parts += [f"limit {name} {status}" for name, status in limits_of(picked_state).items()]
            outcome = "; ".join(parts)
        action = (
            f"recipe {recipe_of(call)} from {base_alias}" if recipe_level else
            f"{call.worker_id} from {base_alias}, "
            f"{placement_phrase(call.settings.preserve_placement)}, {attempts_phrase(attempts)}"
        )
        log.append(f"Step {index}: {action}. Reason: {_reason(call)}. Outcome: {outcome}{exhausted_note(call)}.")

    fuzzy = FuzzyView(
        mode=mode,
        budget={
            "used": view.brief.budget_runs - view.budget_left, "remaining": view.budget_left,
            "total": view.brief.budget_runs,
        },
        best_valid_state=aliases.get(best.state_id, "S0"),
        current_state=aliases.get(view.current.state_id, "S0"),
        states=tuple(fuzzy_states), trajectory_log=tuple(log),
    )
    text = render_view(fuzzy)
    return BuiltView(
        fuzzy=fuzzy, alias_to_state={alias: state_id for state_id, alias in aliases.items()},
        state_to_alias=dict(aliases), text=text, sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _kv(mapping: Mapping[str, str]) -> str:
    return ", ".join(f"{key} {value}" for key, value in mapping.items()) or "none"


def render_view(fuzzy: FuzzyView) -> str:
    """Deterministic plain-text rendering; it contains no hashes by construction."""

    budget = fuzzy.budget
    lines = [
        f"budget: used {budget['used']}, remaining {budget['remaining']}, total {budget['total']}",
        f"best_valid_state: {fuzzy.best_valid_state}",
        f"current_state: {fuzzy.current_state}",
        "states:",
    ]
    for state in fuzzy.states:
        lines.append(f"{state.id}" + (" (start)" if state.parent == "none" else ""))
        if state.parent != "none":
            lines.append(f"  parent: {state.parent}")
            suffix = "" if state.picked_by_worker == "n/a" else f"; picked by the worker: {state.picked_by_worker}"
            lines.append(f"  made_by: {state.made_by}{suffix}")
        lines.append(f"  brief_valid: {'yes' if state.brief_valid else 'no'}")
        lines.append(f"  limits: {_kv(state.limits)}")
        if state.parent != "none":
            lines.append(f"  vs_start: {_kv(state.vs_start)}")
            lines.append("  vs_parent: " + ("same as vs_start" if state.parent == "S0" else _kv(state.vs_parent)))
        lines.append(f"  goal_vs_best: {state.goal_vs_best}")
        if state.values is not None:
            lines.append("  values: " + ", ".join(f"{k} {_num(v)}" for k, v in state.values.items()))
            if state.exact_vs_start:
                lines.append("  exact_vs_start: " + ", ".join(f"{k} {_signed(v)}" for k, v in state.exact_vs_start.items()))
            if state.exact_vs_parent and state.parent != "S0":
                lines.append("  exact_vs_parent: " + ", ".join(f"{k} {_signed(v)}" for k, v in state.exact_vs_parent.items()))
        if state.tried:
            lines.append("  tried:")
            lines += [f"    - {entry}" for entry in state.tried]
        else:
            lines.append("  tried: nothing yet")
    lines.append("trajectory_log:" + ("" if fuzzy.trajectory_log else " empty"))
    lines += [f"  {entry}" for entry in fuzzy.trajectory_log]
    return "\n".join(lines) + "\n"
