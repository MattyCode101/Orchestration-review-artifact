"""Template v2.2, switch S' (situation experience): evidence for the situation the current decision is in (new file).

The single-call experience lines (switch E) pool every stored call of a worker and placement setting, so they cannot show
that, for example, a deep limit break is repaired by re-placing and rarely by keeping placement (the diagnosis note of
2026-11-13, Steps 4g and 7). S' keeps only the stored calls made in the same situation as the current decision.

Situations, detected from the saved states of the view:
  opening / start   no call has been made yet (opening), or no saved state has gained a band on the goal and no gained
                    state breaks a limit (start): calls from the start state, per worker and placement setting, in the
                    experience-line words
  repair            saved states that gained at least one noise band on the goal and break a limit: calls from such a state,
                    per worker, per break depth (under half a band, half a band to two bands, over two bands) and
                    placement setting: how often some run met every limit and kept a goal gain of at least one band
  push              the best state within limits has already gained at least one band: calls from a valid state that had
                    gained, per worker and placement setting, in the experience-line words

Evidence: every stored call is one row (configs/experience/situations_v22.json, built by
scripts/v22_2026-11-13/v22_build_situations.py from the TRAINING and VALIDATION episodes and call records), with the changes
in noise bands of the goal and of the limit. One row stands for one (design, brief, base state, worker, placement setting):
the most complete stored call (most runs). A line is built from the rows of the other designs only (leave one design
out) under the current brief, shows "n calls from k designs", and needs at least MIN_DESIGNS designs, otherwise it says
"no evidence for this situation". No design is named. Lines reuse the words and the text of the single-call lines
(orch.policies.v2_experience.render_line, frequency_word). Pure aggregation and text.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orch.contract.models import MetricName
from orch.contract.views import PolicyView, StateSummary
from orch.policies.llm_view import METRIC_NAMES, effective_bands
from orch.policies.v2_experience import SCOPES, LineStats, frequency_word, render_line
from orch.policies.v2_policy import exhausted_combinations
from orch.policies.v2_view import unique_states
from orch.policies.v2_vocab import PLACEMENT_V2, categorise_v2

SCHEMA = "v22-situation-calls-v2"
MIN_DESIGNS = 3
GAIN_BANDS = 1.0                                   # a goal gain counts from one noise band
BUCKETS = (("under half a band", 0.0, 0.5), ("half a band to two bands", 0.5, 2.0), ("over two bands", 2.0, float("inf")))
BRIEFS = ("T0", "A0")
QUIET = "within noise"
GOAL_LIMIT = {"T0": ("wns_ns", "packed_luts"), "A0": ("packed_luts", "wns_ns")}     # brief -> (goal metric, limit metric)
WORKER_OF = {"wns_ns": "timing", "packed_luts": "area", "dynamic_power_w": "power"}  # metric -> the worker aimed at it
SOURCE_ORDER = {"v4": 0, "d6v2": 1, "calls_v2": 2}


def bucket_of(depth_b: float) -> str:
    for name, low, high in BUCKETS:
        if low <= depth_b < high:
            return name
    return BUCKETS[-1][0]


# ------------------------------------------------------------------------------------------------ rows from stored data
def _row(*, design: str, group: str, brief: str, source: str, worker: str, preserve: bool, base_id: str, n_runs: int,
         situation: str, depth_b: float | None, outcome: str, d_goal_b: float | None, d_limit_b: float | None,
         restored: bool | None) -> dict[str, Any]:
    goal, limit = GOAL_LIMIT[brief]
    return {"design": design, "group": group, "brief": brief, "source": source, "worker": worker, "preserve": bool(preserve),
            "base": base_id, "n_runs": n_runs, "situation": situation,
            "depth_b": None if depth_b is None else round(depth_b, 4), "outcome": outcome,
            "d_goal_b": None if d_goal_b is None else round(d_goal_b, 4),
            "d_limit_b": None if d_limit_b is None else round(d_limit_b, 4), "restored": restored,
            "goal": goal, "limit": limit}


def classify_base(gain_b: float, margin_b: float, valid: bool, is_start: bool) -> tuple[str, float | None]:
    """(situation, break depth in bands) of a call's base state."""

    if is_start:
        return "start", None
    if gain_b >= GAIN_BANDS:
        if valid:
            return "gain", None
        if margin_b < 0.0:
            return "break", -margin_b
    return "other", None


def _select(steps: Sequence[Mapping[str, Any]], worker: str, brief: str) -> Mapping[str, Any] | None:
    """The guarded selection of a call (limits respected): the best run inside every limit for the worker's own objective."""

    valid = [s for s in steps if s["valid"]]
    if not valid:
        return None
    goal, _limit = GOAL_LIMIT[brief]
    own_goal = WORKER_OF[goal] == worker
    key = (lambda s: s["gain_b"]) if own_goal else (lambda s: s["margin_b"])
    return max(valid, key=key)


def _call_row(*, design: str, group: str, brief: str, source: str, worker: str, preserve: bool, base_id: str,
              base: Mapping[str, Any], is_start: bool, steps: Sequence[Mapping[str, Any]],
              n_runs: int | None = None) -> dict[str, Any] | None:
    """One row from a call whose base and runs are given in band units: base gain_b, margin_b, valid; steps gain_b, margin_b,
    valid (gain_b is positive for a goal gain, margin_b is positive inside the limit)."""

    situation, depth = classify_base(base["gain_b"], base["margin_b"], base["valid"], is_start)
    if situation == "other":
        return None
    if not steps:
        outcome, d_goal, d_limit = "no_options", None, None
    else:
        chosen = _select(steps, worker, brief)
        if chosen is None:
            outcome, d_goal, d_limit = "nothing_selectable", None, None
        else:
            outcome = "selected"
            d_goal = chosen["gain_b"] - base["gain_b"]
            d_limit = chosen["margin_b"] - base["margin_b"]
    restored = None
    if situation == "break":
        restored = any(s["valid"] and s["gain_b"] >= GAIN_BANDS for s in steps)
    return _row(design=design, group=group, brief=brief, source=source, worker=worker, preserve=preserve, base_id=base_id,
                n_runs=len(steps) if n_runs is None else n_runs, situation=situation, depth_b=depth, outcome=outcome,
                d_goal_b=d_goal, d_limit_b=d_limit, restored=restored)


def rows_from_diagnosis_episode(ep: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rows from one D6-v2 episode as parsed by the diagnosis (data/llm_diagnosis_2026-11-13/episodes.json)."""

    if ep["brief"] not in BRIEFS:
        return []
    group = "TRAINING" if ep["group"] == "TRAINING" else "VALIDATION"
    states = ep["states"]
    rows = []
    for call in ep["calls"]:
        base = states.get(call["base"])
        if base is None:
            continue
        steps = [{"out": s["out"], "gain_b": s["gain_b"], "margin_b": s["margin_b"], "valid": bool(s["valid"])}
                 for s in call["steps"] if s.get("phys", True)]
        row = _call_row(design=ep["design"], group=group, brief=ep["brief"], source="d6v2", worker=call["worker"],
                        preserve=bool(call["preserve"]), base_id=call["base"],
                        base={"gain_b": base["gain_b"], "margin_b": base["margin_b"], "valid": bool(base["valid"])},
                        is_start=base.get("call") == -1, steps=steps)
        if row is not None:
            rows.append(row)
    return rows


def _metrics_in_bands(brief: Any, start: Mapping[str, float], bands: Mapping[str, float],
                      metrics: Mapping[str, float]) -> tuple[float, float] | None:
    """(goal gain in bands, the smallest limit margin in bands) of one state's metrics against the start."""

    goal = brief.objective.metric.value
    if goal not in metrics or goal not in start or not bands.get(goal):
        return None
    sign = 1.0 if brief.objective.direction == "max" else -1.0
    gain = sign * (metrics[goal] - start[goal]) / bands[goal]
    margins = []
    for term in brief.terms:
        m = term.metric.value
        if m not in metrics or m not in start or not bands.get(m):
            return None
        threshold = term.factor * start[m] + term.offset
        margins.append(((threshold - metrics[m]) if term.comparison == "<=" else (metrics[m] - threshold)) / bands[m])
    return gain, min(margins)


def rows_from_v4_episode(ep: Mapping[str, Any], brief: Any, bands: Mapping[str, float], group: str) -> list[dict[str, Any]]:
    """Rows from one stored v4 training episode (data/bandit_2026-11-11/calls/episodes_v4.jsonl)."""

    if ep["brief"] not in BRIEFS:
        return []
    start = ep["start_metrics"]
    start_id = ep["start_state_id"]
    known: dict[str, dict[str, Any]] = {start_id: {"gain_b": 0.0, "margin_b": 0.0, "valid": True}}
    calls = [c for c in ep["calls"] if c.get("completed")]
    for call in calls:                                           # every state a call produced, with its verdict
        for step in call["steps"]:
            if step.get("physically_valid") and step["out_id"] not in known:
                gm = _metrics_in_bands(brief, start, bands, step["metrics"])
                if gm is not None:
                    known[step["out_id"]] = {"gain_b": gm[0], "margin_b": gm[1], "valid": step["status"] == "ACCEPTABLE"}
    rows = []
    for call in calls:
        base = known.get(call["base_id"])
        if base is None:
            continue
        steps = []
        for step in call["steps"]:
            if step.get("physically_valid") and step["out_id"] in known:
                steps.append({"out": step["out_id"], **known[step["out_id"]]})
        row = _call_row(design=ep["design"], group=group, brief=ep["brief"], source="v4", worker=call["worker"],
                        preserve=bool(call["preserve"]), base_id=call["base_id"], base=base,
                        is_start=call["base_id"] == start_id, steps=steps)
        if row is not None:
            rows.append(row)
    return rows


def rows_from_calls_v2(items: Sequence[Mapping[str, Any]], brief_of: Mapping[str, Any],
                       group_of: Mapping[str, str]) -> list[dict[str, Any]]:
    """Rows from the exhaustive call table (configs/experience/calls_v2.json): first-step calls from the start and
    second-step calls from the one-step states, TRAINING designs, selection with the limits respected."""

    starts: dict[tuple[str, str], Mapping[str, float]] = {}
    for it in items:
        if it["base"] == "start":
            starts[(it["design"], it["brief"])] = it["base_values"]
    rows = []
    for it in items:
        brief_id = it["brief"]
        if brief_id not in BRIEFS or (it["design"], brief_id) not in starts:
            continue
        brief = brief_of[brief_id]
        start, bands = starts[(it["design"], brief_id)], it.get("bands") or {}
        base_gm = _metrics_in_bands(brief, start, bands, it["base_values"])
        if base_gm is None:
            continue
        base = {"gain_b": base_gm[0], "margin_b": base_gm[1], "valid": base_gm[1] >= 0.0}
        steps = []
        if it["outcome"] == "selected" and it.get("result_values") is not None:
            res = _metrics_in_bands(brief, start, bands, it["result_values"])
            if res is not None:
                steps = [{"out": "selected", "gain_b": res[0], "margin_b": res[1], "valid": res[1] >= 0.0}]
        elif it["outcome"] == "nothing_selectable":
            steps = [{"out": "none", "gain_b": 0.0, "margin_b": -1.0, "valid": False}]     # runs existed, none inside the limits
        row = _call_row(design=it["design"], group=group_of[it["design"]], brief=brief_id, source="calls_v2",
                        worker=it["worker"], preserve=it["scope"] == SCOPES[0], base_id=it["base_state"], base=base,
                        is_start=it["base"] == "start", steps=steps, n_runs=0)
        if row is not None:
            rows.append(row)
    return rows


def dedupe(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per (design, brief, base state, worker, placement setting), the way the exhaustive call table counts: the
    most complete stored call, that is the one with the most runs; an exhaustive-table row is used only when no episode
    call covers the combination (its single summary result cannot say whether any run restored a limit). Ties go to the
    source order v4, d6v2, calls_v2 (runs are deterministic, so tied calls agree)."""

    def rank(r: Mapping[str, Any]) -> tuple[bool, int, int]:
        return (r["source"] != "calls_v2", r["n_runs"], -SOURCE_ORDER[r["source"]])

    best: dict[tuple, Mapping[str, Any]] = {}
    for r in rows:
        key = (r["design"], r["brief"], r["base"], r["worker"], r["preserve"])
        cur = best.get(key)
        if cur is None or rank(r) > rank(cur):
            best[key] = r
    return [dict(best[k]) for k in sorted(best)]


def table_payload(rows: Sequence[Mapping[str, Any]], *, source: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": SCHEMA, "source": dict(source), "rows": list(rows)}


def load_rows(path: Path | str) -> tuple[list[dict[str, Any]], str]:
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != SCHEMA:
        raise ValueError(f"unknown situation table schema {data.get('schema_version')!r}")
    return list(data["rows"]), hashlib.sha256(raw).hexdigest()


# ------------------------------------------------------------------------------------------------ the current decision
@dataclass(frozen=True)
class Detected:
    opening: bool
    start_lines: bool
    repair: tuple[tuple[str, float, str], ...]      # (alias, break depth in bands, limit metric name), shallowest first
    push_from: str | None                           # alias of the best state within limits when it has gained a band
    gains: Mapping[str, float] = field(default_factory=dict)


def _gain_margin(view: PolicyView, state: StateSummary,
                 bands: Mapping[MetricName, float]) -> tuple[float | None, tuple[float, MetricName] | None]:
    """(goal gain in bands against the start, (smallest limit margin in bands, its metric)) of one saved state."""

    goal = view.brief.objective.metric
    gain = None
    band = bands.get(goal)
    if band and goal in state.metrics and goal in view.start.metrics:
        sign = 1.0 if view.brief.objective.direction == "max" else -1.0
        gain = sign * (state.metrics[goal] - view.start.metrics[goal]) / band
    worst: tuple[float, MetricName] | None = None
    for term in view.brief.terms:
        ref, val, tb = view.start.metrics.get(term.metric), state.metrics.get(term.metric), bands.get(term.metric)
        if ref is None or val is None or not tb:
            continue
        threshold = term.factor * ref + term.offset
        margin = ((threshold - val) if term.comparison == "<=" else (val - threshold)) / tb
        if worst is None or margin < worst[0]:
            worst = (margin, term.metric)
    return gain, worst


def detect(view: PolicyView) -> Detected:
    bands = effective_bands(view.noise_bands, view.start.metrics)
    spent = exhausted_combinations(view)
    worker_ids = [c.id for c in view.worker_cards]
    states = unique_states(view)
    aliases = {s.state_id: f"S{i}" for i, s in enumerate(states)}
    repair: list[tuple[str, float, str]] = []
    gains: dict[str, float] = {}
    for state in states:
        gain, worst = _gain_margin(view, state, bands)
        if gain is not None:
            gains[aliases[state.state_id]] = round(gain, 3)
        if state.state_id == view.start.state_id or not state.physically_valid or state.brief_valid:
            continue
        if gain is None or gain < GAIN_BANDS or worst is None or worst[0] >= 0.0:
            continue
        if all((w, state.state_id, p) in spent for w in worker_ids for p in (True, False)):
            continue                                          # no worker has an option left from this state
        repair.append((aliases[state.state_id], -worst[0], METRIC_NAMES.get(worst[1], worst[1].value)))
    repair.sort(key=lambda t: (t[1], t[0]))
    best = view.best_valid
    push_from = None
    if best is not None and best.state_id != view.start.state_id:
        gain, _ = _gain_margin(view, best, bands)
        if gain is not None and gain >= GAIN_BANDS:
            push_from = aliases[best.state_id]
    opening = len(view.calls) == 0
    return Detected(opening=opening, start_lines=(not opening and not repair and push_from is None),
                    repair=tuple(repair), push_from=push_from, gains=gains)


# ------------------------------------------------------------------------------------------------ text
def _line_stats(rows: Sequence[Mapping[str, Any]], *, worker: str, preserve: bool, goal: str, limit: str) -> LineStats:
    """The same statistics as orch.policies.v2_experience.line_stats, from rows that already hold the changes in noise bands
    (a change of one band or more is a gain or a loss, below one band it is within noise)."""

    mine = [r for r in rows if r["worker"] == worker and r["preserve"] == preserve]
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    observed: dict[str, int] = defaultdict(int)
    usable = nothing = no_change = no_change_base = 0
    for r in mine:
        if r["outcome"] == "nothing_selectable":
            nothing += 1
            continue
        if r["outcome"] != "selected":
            continue
        usable += 1
        cats: dict[str, str] = {}
        for metric, delta in ((goal, r["d_goal_b"]), (limit, r["d_limit_b"])):
            if delta is None:
                continue
            cat = categorise_v2(delta, 1.0, higher_is_better=True)
            cats[metric] = cat
            counts[metric][cat] += 1
            observed[metric] += 1
        if goal in cats:
            no_change_base += 1
            if cats[goal] == QUIET and all(cats[m] == QUIET for m in (limit,) if m in cats):
                no_change += 1
    return LineStats(worker=worker, scope=SCOPES[0] if preserve else SCOPES[1], calls=len(mine),
                     designs=len({r["design"] for r in mine}), usable=usable, nothing_selectable=nothing,
                     no_usable=len(mine) - usable - nothing, counts={m: dict(v) for m, v in counts.items()},
                     observed=dict(observed), no_change=no_change, no_change_base=no_change_base, changes={})


def _bands_text(depth: float) -> str:
    return f"{depth:.1f} noise band" + ("" if round(depth, 1) == 1.0 else "s")


def render_situation(rows: Sequence[Mapping[str, Any]], view: PolicyView, *, design: str | None, texts: Mapping[str, str],
                     workers: Sequence[str], min_designs: int = MIN_DESIGNS) -> tuple[str, dict[str, Any]]:
    """Section text (without the blank lines the prompt puts around it) and what it rests on. Empty text for a brief the
    table does not cover (the grid uses T0 and A0 only)."""

    brief = view.brief.id
    if brief not in GOAL_LIMIT:
        return "", {"shown": [], "omitted": [], "blocks": [], "note": "no situation rows for this brief"}
    goal, limit = GOAL_LIMIT[brief]
    kept = [r for r in rows if r["brief"] == brief and (design is None or r["design"] != design)]
    listed = [w for w in workers if any(r["worker"] == w for r in kept)]      # a worker nobody called gets no line
    found = detect(view)
    no_evidence = texts["no_evidence"]
    lines = [texts["header"], texts["intro"]]
    shown: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    blocks: list[str] = []

    def worker_block(title: str, situation: str) -> None:
        lines.append(title)
        pool = [r for r in kept if r["situation"] == situation]
        for worker in listed:
            lines.append(worker)
            for preserve in (True, False):
                stats = _line_stats(pool, worker=worker, preserve=preserve, goal=goal, limit=limit)
                row = {"block": situation, "worker": worker, "preserve": preserve, "calls": stats.calls, "designs": stats.designs}
                if stats.calls == 0 or stats.designs < min_designs:
                    omitted.append(row)
                    lines.append(f"  {PLACEMENT_V2[preserve]}: {no_evidence}")
                else:
                    shown.append(row)
                    lines.append(render_line(stats, goal=goal, limits=[limit], form="words", v2=True, placement_v2=True))

    if found.opening:
        blocks.append("opening")
        worker_block(texts["opening"], "start")
    elif found.start_lines:
        blocks.append("start")
        worker_block(texts["start_no_gain"], "start")
    if found.repair:
        blocks.append("repair")
        lines.append(texts["repair"].format(states=", ".join(
            f"{alias} ({name} broken by {_bands_text(depth)})" for alias, depth, name in found.repair)))
        pool = [r for r in kept if r["situation"] == "break"]
        for bucket, _low, _high in BUCKETS:
            if not any(bucket_of(depth) == bucket for _a, depth, _n in found.repair):
                continue
            lines.append(texts["repair_bucket"].format(bucket=bucket))
            for worker in listed:
                lines.append(worker)
                for preserve in (True, False):
                    group = [r for r in pool if r["worker"] == worker and r["preserve"] == preserve
                             and bucket_of(r["depth_b"]) == bucket]
                    n, k = len(group), len({r["design"] for r in group})
                    row = {"block": "repair", "bucket": bucket, "worker": worker, "preserve": preserve, "calls": n, "designs": k}
                    if n == 0 or k < min_designs:
                        omitted.append(row)
                        lines.append(f"  {PLACEMENT_V2[preserve]}: {no_evidence}")
                        continue
                    share = sum(bool(r["restored"]) for r in group) / n
                    row["share"] = round(share, 3)
                    shown.append(row)
                    lines.append(f"  {PLACEMENT_V2[preserve]} ({n} calls from {k} designs): restored the limit and kept a goal "
                                 f"gain of at least one band {frequency_word(share)}.")
    if found.push_from is not None:
        blocks.append("push")
        worker_block(texts["push"].format(alias=found.push_from), "gain")
    text = "\n".join(lines)
    meta = {"shown": shown, "omitted": omitted, "blocks": blocks, "repair_states": [list(t) for t in found.repair],
            "push_from": found.push_from, "rows_in_scope": len(kept), "designs_in_scope": sorted({r["design"] for r in kept}),
            "excluded_design": design, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    return text, meta
