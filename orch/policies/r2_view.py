"""Round-2 view features, each independently switchable (Wednesday-night note, sections 3 and 7).

  history       provenance of every state (producing call, attempt, parent, worker, placement setting) and
                every brief metric with its exact value plus the category against the start and against
                the parent. Additive: every line of the round-1 fuzzy view stays, so a configuration
                differs from F0 only by what this feature adds.
  physical      the physical description of the saved states (orch/policies/r2_physical.py), by group.
  experience    the worker-experience section (orch/policies/r2_experience.py), fuzzy or count form,
                leave-one-design-out or from all designs.

With every feature off the view is the round-1 fuzzy view, byte for byte.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from orch.contract.models import MetricName
from orch.contract.views import PolicyView, StateSummary
from orch.policies.llm_view import METRIC_NAMES, BuiltView, build_fuzzy_view
from orch.policies.r2_physical import GROUPS

EXPERIENCE_FORMS = ("off", "fuzzy", "count")
EXPERIENCE_SOURCES = ("lodo", "all")


@dataclass(frozen=True)
class Features:
    history: bool = False
    physical: tuple[str, ...] = ()
    experience: str = "off"
    experience_source: str = "lodo"

    @classmethod
    def from_config(cls, spec: Mapping[str, Any]) -> "Features":
        asked = tuple(spec.get("physical") or ())
        unknown = sorted(set(asked) - set(GROUPS))
        if unknown:
            raise ValueError(f"unknown physical group {unknown[0]!r}; known: {', '.join(GROUPS)}")
        experience = spec.get("experience", "off")
        source = spec.get("experience_source", "lodo")
        if experience not in EXPERIENCE_FORMS:
            raise ValueError(f"experience must be one of {', '.join(EXPERIENCE_FORMS)}")
        if source not in EXPERIENCE_SOURCES:
            raise ValueError(f"experience_source must be one of {', '.join(EXPERIENCE_SOURCES)}")
        return cls(bool(spec.get("history", False)), tuple(g for g in GROUPS if g in asked), experience, source)

    @property
    def key(self) -> str:
        return "|".join((
            "history" if self.history else "nohistory",
            "physical=" + ("+".join(self.physical) or "none"),
            f"experience={self.experience}:{self.experience_source if self.experience != 'off' else '-'}",
        ))


def brief_metrics(view: PolicyView) -> list[MetricName]:
    """The brief's goal metric first, then each limit metric once: every metric the brief is about."""

    ordered = [view.brief.objective.metric]
    for term in view.brief.terms:
        if term.metric not in ordered:
            ordered.append(term.metric)
    return ordered


def goal_and_limits(view: PolicyView) -> tuple[str, list[str]]:
    goal = METRIC_NAMES.get(view.brief.objective.metric, view.brief.objective.metric.value)
    limits = [METRIC_NAMES.get(m, m.value) for m in brief_metrics(view)[1:]]
    return goal, limits


def value_text(metric: MetricName, value: float) -> str:
    if metric == MetricName.WNS_NS:
        return f"{value:.3f} ns"
    if metric == MetricName.DYNAMIC_POWER_W:
        return f"{value:.3f} W"
    return f"{value:.0f}"


def delta_text(metric: MetricName, delta: float) -> str:
    if metric in (MetricName.WNS_NS, MetricName.DYNAMIC_POWER_W):
        return f"{delta:+.3f}"
    return f"{delta:+.0f}"


def producing_calls(view: PolicyView) -> dict[str, tuple[int, int, int]]:
    """state id -> (call number, attempt within the call, runs the call made) for every produced state.

    A step belongs to the call with the same worker, base state and placement setting; the policy never
    repeats that triple, so the match is unique.
    """

    keys = [(c.worker_id, c.base_state_id, c.settings.preserve_placement) for c in view.calls]
    seen: dict[tuple[str, str, bool], int] = defaultdict(int)
    out: dict[str, tuple[int, int, int]] = {}
    for step in view.steps:
        key = (step.worker_id, step.base_state_id, step.settings.preserve_placement)
        if key not in keys:
            continue
        seen[key] += 1
        if step.output_state_id and step.output_state_id not in out and step.output_state_id != view.start.state_id:
            index = keys.index(key)
            out[step.output_state_id] = (index + 1, seen[key], view.calls[index].recipe_runs)
    return out


def _change_part(metric: MetricName, summary: StateSummary, ref: StateSummary | None, category: str | None) -> str | None:
    if ref is None or metric not in summary.metrics or metric not in ref.metrics:
        return None
    delta = delta_text(metric, summary.metrics[metric] - ref.metrics[metric])
    return f"{delta} {category}" if category else f"{delta} (no noise band recorded)"


def render_history_view(view: PolicyView, base: BuiltView) -> str:
    """The round-1 fuzzy state list plus provenance and exact brief-metric values with their categories."""

    fuzzy = base.fuzzy
    summaries = {s.state_id: s for s in (view.start, *view.visited)}
    provenance = producing_calls(view)
    brief = brief_metrics(view)
    brief_names = {METRIC_NAMES.get(m, m.value) for m in brief}
    budget = fuzzy.budget
    lines = [
        f"budget: used {budget['used']}, remaining {budget['remaining']}, total {budget['total']}",
        f"best_valid_state: {fuzzy.best_valid_state}",
        f"current_state: {fuzzy.current_state}",
        "states:",
    ]
    for state in fuzzy.states:
        state_id = base.alias_to_state[state.id]
        summary = summaries[state_id]
        is_start = state.parent == "none"
        parent = summaries.get(base.alias_to_state.get(state.parent, ""))
        lines.append(state.id + (" (start)" if is_start else ""))
        if not is_start:
            lines.append(f"  parent: {state.parent}")
            call = provenance.get(state_id)
            where = f" (call {call[0]}, attempt {call[1]} of {call[2]})" if call else ""
            picked = "" if state.picked_by_worker == "n/a" else f"; picked by the worker: {state.picked_by_worker}"
            lines.append(f"  made_by: {state.made_by}{where}{picked}")
        lines.append(f"  brief_valid: {'yes' if state.brief_valid else 'no'}")
        for metric in brief:
            if metric not in summary.metrics:
                continue
            name = METRIC_NAMES.get(metric, metric.value)
            text = f"  {name}: {value_text(metric, summary.metrics[metric])}"
            if not is_start:
                against_start = _change_part(metric, summary, view.start, state.vs_start.get(name))
                text += f"; vs start {against_start}" if against_start else ""
                if state.parent == "S0":
                    text += "; vs parent same as vs start"
                else:
                    against_parent = _change_part(metric, summary, parent, state.vs_parent.get(name))
                    text += f"; vs parent {against_parent}" if against_parent else ""
            lines.append(text)
        if not is_start:
            others = {k: v for k, v in state.vs_start.items() if k not in brief_names}
            if others:
                lines.append("  other metrics vs start: " + ", ".join(f"{k} {v}" for k, v in others.items()))
                if state.parent == "S0":
                    lines.append("  other metrics vs parent: same as vs start")
                else:
                    other_parent = {k: v for k, v in state.vs_parent.items() if k not in brief_names}
                    lines.append("  other metrics vs parent: "
                                 + (", ".join(f"{k} {v}" for k, v in other_parent.items()) or "none"))
        lines.append("  limits: " + (", ".join(f"{k} {v}" for k, v in state.limits.items()) or "none"))
        lines.append(f"  goal_vs_best: {state.goal_vs_best}")
        if state.tried:
            lines.append("  tried:")
            lines += [f"    - {entry}" for entry in state.tried]
        else:
            lines.append("  tried: nothing yet")
    lines.append("trajectory_log:" + ("" if fuzzy.trajectory_log else " empty"))
    lines += [f"  {entry}" for entry in fuzzy.trajectory_log]
    return "\n".join(lines) + "\n"


def build_r2_view(view: PolicyView, features: Features) -> BuiltView:
    """The state list the prompt shows: the round-1 fuzzy view, enriched when `history` is on."""

    base = build_fuzzy_view(view, mode="fuzzy")
    if not features.history:
        return base
    text = render_history_view(view, base)
    return replace(base, text=text, sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())


def state_lookup(view: PolicyView) -> dict[str, StateSummary]:
    return {s.state_id: s for s in (view.start, *view.visited)}


def alias_order(aliases: Sequence[str]) -> list[str]:
    return sorted(aliases, key=lambda a: int(a[1:]))
