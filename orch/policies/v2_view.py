"""Template v2 state list (the "Current situation" section), switchable by package.

With every switch off this is not used: the base of the ladder is the round-2 view with history
(orch/policies/r2_view.build_r2_view), byte for byte. The switches:

  labels   (P1)  plain labels: "best state within limits so far", "last selected state", "meets every limit",
                 "S0 (start: the placed-and-routed checkpoint you were given)", "no options left" (change 3, 5)
  vocab    (P1)  the v2 categories and limit statuses (orch/policies/v2_vocab.py, change 8)
  placement(P2)  "placement may change" / "placement must not change" (change 11)
  exact_only     the noise-category variant of the authors' decision 5: exact values and exact differences only,
                 limits as exact distances; no category or status word anywhere

Aliases, parents, producers, the tried lists and the trajectory log follow orch/policies/llm_view exactly, so a
state has the same alias in every rendering of a point.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from orch.contract.models import MetricName
from orch.contract.views import CallSummary, PolicyView, StateSummary, StepSummary
from orch.policies.llm_view import HIGHER_IS_BETTER, METRIC_NAMES, SHOWN, _reason, effective_bands
from orch.policies.r2_view import brief_metrics, delta_text, producing_calls, value_text
from orch.policies.v2_vocab import PLACEMENT_R1, PLACEMENT_V2, categorise, limit_status

START_NOTE = "start: the placed-and-routed checkpoint you were given"


@dataclass(frozen=True)
class ViewFlags:
    labels: bool = False
    vocab: bool = False
    placement: bool = False
    exact_only: bool = False


@dataclass(frozen=True)
class BuiltViewV2:
    text: str
    alias_to_state: dict[str, str]
    state_to_alias: dict[str, str]
    sha256: str


def unique_states(view: PolicyView) -> list[StateSummary]:
    seen: dict[str, StateSummary] = {}
    for state in (view.start, *view.visited):
        seen.setdefault(state.state_id, state)
    return list(seen.values())


def _attempts_phrase(attempts: int | None) -> str:
    if attempts is None:
        return "default attempts"
    return f"{attempts} attempt" if attempts == 1 else f"{attempts} attempts"


def render_view_v2(view: PolicyView, flags: ViewFlags) -> BuiltViewV2:
    states = unique_states(view)
    aliases = {s.state_id: f"S{i}" for i, s in enumerate(states)}
    by_id = {s.state_id: s for s in states}
    start = view.start
    bands = effective_bands(view.noise_bands, start.metrics)
    cards = {c.id: c for c in view.worker_cards}
    goal = view.brief.objective.metric
    place = PLACEMENT_V2 if flags.placement else PLACEMENT_R1
    v2 = flags.vocab
    within = "within noise" if v2 else "within_noise"

    def attempts_of(worker_id: str, max_runs: int | None) -> int | None:
        if max_runs is not None:
            return max_runs
        card = cards.get(worker_id)
        return card.default_attempts if card is not None else None

    def exhausted(call: CallSummary) -> str:
        requested = attempts_of(call.worker_id, call.settings.max_runs)
        if requested is None or call.recipe_runs >= requested:
            return ""
        tail = "no options left" if flags.labels else "nothing more to try there"
        return f" (only {call.recipe_runs} of {requested} attempts could run: {tail})"

    def category(metric: MetricName, delta: float, *, zero_band: bool = False) -> str | None:
        """None without a band, except where round 1 binned against a zero band (goal vs best, log)."""

        band = bands.get(metric)
        if not band:
            if not zero_band:
                return None
            band = 0.0
        return categorise(delta, band, higher_is_better=metric in HIGHER_IS_BETTER, v2=v2)

    def limit_text(state: StateSummary) -> list[str]:
        out = []
        for term in view.brief.terms:
            reference, value = start.metrics.get(term.metric), state.metrics.get(term.metric)
            if reference is None or value is None:
                continue
            threshold = term.factor * reference + term.offset
            margin = (threshold - value) if term.comparison == "<=" else (value - threshold)
            name = METRIC_NAMES.get(term.metric, term.metric.value)
            if flags.exact_only:
                size = delta_text(term.metric, abs(margin)).lstrip("+")
                if abs(margin) <= 1e-9:
                    out.append(f"{name} at the limit")
                else:
                    out.append(f"{name} {'inside the limit by' if margin > 0 else 'broken by'} {size}")
                continue
            start_relative = term.factor == 1.0 and term.offset == 0.0
            status = limit_status(margin, bands.get(term.metric, 0.0), start_relative=start_relative, v2=v2)
            out.append(f"{name} {status}")
        return out

    producer: dict[str, StepSummary] = {}
    for step in view.steps:
        out_id = step.output_state_id
        if out_id is None or out_id == step.base_state_id or out_id in producer:
            continue
        if out_id == start.state_id or out_id not in aliases:
            continue
        producer[out_id] = step
    provenance = producing_calls(view)

    def result_of(call: CallSummary) -> str:
        if call.selected_state_id is None:
            return "nothing selectable" if flags.labels else "no usable state"
        if call.selected_state_id == call.base_state_id:
            return "unchanged"
        return aliases.get(call.selected_state_id, "unknown")

    tried: dict[str, list[str]] = {}
    for call in view.calls:
        line = (f"{call.worker_id} ({place[call.settings.preserve_placement]}, "
                f"{_attempts_phrase(attempts_of(call.worker_id, call.settings.max_runs))}) -> {result_of(call)}"
                f"{exhausted(call)}")
        tried.setdefault(call.base_state_id, []).append(line)

    best = view.best_valid if view.best_valid is not None else start
    brief_ms = brief_metrics(view)
    brief_names = {METRIC_NAMES.get(m, m.value) for m in brief_ms}
    used = view.brief.budget_runs - view.budget_left
    lbl = {
        "best": "best state within limits so far" if flags.labels else "best_valid_state",
        "current": "last selected state" if flags.labels else "current_state",
        "states": "saved states:" if flags.labels else "states:",
        "valid": "meets every limit" if flags.labels else "brief_valid",
        "made": "made by" if flags.labels else "made_by",
        "picked": "selected by the worker" if flags.labels else "picked by the worker",
        "gvb": "goal vs best state within limits" if flags.labels else "goal_vs_best",
        "log": "trajectory log:" if flags.labels else "trajectory_log:",
    }
    lines = [
        f"budget: used {used}, remaining {view.budget_left}, total {view.brief.budget_runs}",
        f"{lbl['best']}: {aliases.get(best.state_id, 'S0')}",
        f"{lbl['current']}: {aliases.get(view.current.state_id, 'S0')}",
        lbl["states"],
    ]
    for state in states:
        alias = aliases[state.state_id]
        is_start = state.state_id == start.state_id
        step = producer.get(state.state_id)
        parent_id = None if is_start or step is None else step.base_state_id
        parent_alias = "none" if is_start else (aliases.get(parent_id, "unknown") if parent_id else "unknown")
        parent = by_id.get(parent_id) if parent_id else None
        if is_start:
            lines.append(f"{alias} ({START_NOTE})" if flags.labels else f"{alias} (start)")
        else:
            lines.append(alias)
            lines.append(f"  parent: {parent_alias}")
            if step is not None:
                made = (f"{step.worker_id}, {place[step.settings.preserve_placement]}, "
                        f"{_attempts_phrase(attempts_of(step.worker_id, step.settings.max_runs))}")
                picked = "yes" if any(
                    c.selected_state_id == state.state_id and c.base_state_id == parent_id
                    and c.worker_id == step.worker_id for c in view.calls) else "no"
            else:
                made, picked = "unknown", "no"
            call = provenance.get(state.state_id)
            where = f" (call {call[0]}, attempt {call[1]} of {call[2]})" if call else ""
            lines.append(f"  {lbl['made']}: {made}{where}; {lbl['picked']}: {picked}")
        lines.append(f"  {lbl['valid']}: {'yes' if state.brief_valid else 'no'}")
        for metric in brief_ms:
            if metric not in state.metrics:
                continue
            name = METRIC_NAMES.get(metric, metric.value)
            text = f"  {name}: {value_text(metric, state.metrics[metric])}"
            if not is_start:
                for label, ref in (("vs start", start), ("vs parent", parent)):
                    if label == "vs parent" and parent_alias == "S0":
                        text += "; vs parent same as vs start"
                        continue
                    if ref is None or metric not in ref.metrics:
                        continue
                    delta = state.metrics[metric] - ref.metrics[metric]
                    cat = None if flags.exact_only else category(metric, delta)
                    if flags.exact_only:
                        text += f"; {label} {delta_text(metric, delta)}"
                    else:
                        text += f"; {label} {delta_text(metric, delta)} " + (cat or "(no noise band recorded)")
            lines.append(text)
        if not is_start:
            def others_vs(ref: StateSummary | None) -> list[str]:
                found = []
                if ref is None:
                    return found
                for metric in SHOWN:
                    name = METRIC_NAMES[metric]
                    if name in brief_names or metric not in state.metrics or metric not in ref.metrics:
                        continue
                    delta = state.metrics[metric] - ref.metrics[metric]
                    if flags.exact_only:
                        found.append(f"{name} {delta_text(metric, delta)}")
                    elif (cat := category(metric, delta)) is not None:
                        found.append(f"{name} {cat}")
                return found

            others = others_vs(start)
            if others:
                lines.append("  other metrics vs start: " + ", ".join(others))
                if parent_alias == "S0":
                    lines.append("  other metrics vs parent: same as vs start")
                else:
                    lines.append("  other metrics vs parent: " + (", ".join(others_vs(parent)) or "none"))
        lines.append("  limits: " + (", ".join(limit_text(state)) or "none"))
        if goal in state.metrics and goal in best.metrics:
            delta = state.metrics[goal] - best.metrics[goal]
            if flags.exact_only:
                gvb = delta_text(goal, delta)
            else:
                gvb = category(goal, delta, zero_band=True) or within
        else:
            gvb = delta_text(goal, 0.0) if flags.exact_only else within
        lines.append(f"  {lbl['gvb']}: {gvb}")
        if state.state_id in tried:
            lines.append("  tried:")
            lines += [f"    - {entry}" for entry in tried[state.state_id]]
        else:
            lines.append("  tried: nothing yet")
    log = []
    for index, call in enumerate(view.calls, start=1):
        base_alias = aliases.get(call.base_state_id, "unknown")
        attempts = attempts_of(call.worker_id, call.settings.max_runs)
        if call.selected_state_id is None:
            outcome = "nothing selectable" if flags.labels else "no usable state returned"
        elif call.selected_state_id == call.base_state_id:
            outcome = "unchanged"
        else:
            picked_state, base_state = by_id.get(call.selected_state_id), by_id.get(call.base_state_id)
            parts = [f"{aliases.get(call.selected_state_id, 'unknown')} chosen"]
            if picked_state is not None and base_state is not None and goal in picked_state.metrics:
                delta = picked_state.metrics[goal] - base_state.metrics.get(goal, picked_state.metrics[goal])
                parts.append("goal " + (delta_text(goal, delta) if flags.exact_only
                                        else (category(goal, delta, zero_band=True) or within)))
            if picked_state is not None:
                parts += [f"limit {entry}" for entry in limit_text(picked_state)]
            outcome = "; ".join(parts)
        action = (f"{call.worker_id} from {base_alias}, {place[call.settings.preserve_placement]}, "
                  f"{_attempts_phrase(attempts)}")
        log.append(f"Step {index}: {action}. Reason: {_reason(call)}. Outcome: {outcome}{exhausted(call)}.")
    lines.append(lbl["log"] + ("" if log else " empty"))
    lines += [f"  {entry}" for entry in log]
    text = "\n".join(lines) + "\n"
    return BuiltViewV2(text, {a: s for s, a in aliases.items()}, dict(aliases),
                       hashlib.sha256(text.encode("utf-8")).hexdigest())
