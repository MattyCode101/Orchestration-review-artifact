"""Template v2 worker experience: the build-review fixes (defects 1 to 5, and the evidence word of defect 9).

orch/policies/r2_experience.py is left byte-identical as the record of the round-2 ablation (its 618 stored
prompts still rebuild). This module is what template v2 shows instead:

  defect 1  calls are pooled: every call counts once, whatever design it came from. Each line says
            "n calls from k designs", and a line resting on fewer than MIN_DESIGNS (4) designs is left out.
  defect 2  the vocabulary may carry the "very large" tier (over 30 noise bands), so a placement-keeping gain
            of 3 bands and a re-placing gain of 80 bands no longer share one word (orch/policies/v2_vocab.py).
  defect 3  "no change" is counted only over calls that carry a category for the brief's goal metric; a call
            on a design with no band for a metric carries no category for it (it still counts as a call).
  defect 4  the calls are generated per brief with the brief's limits respected (guarded mandates, as 513 of
            516 real round-2 mandates were), and "nothing selectable" is an outcome of its own.
  defect 5  a call that could not run (the worker had no option for that setting) counts as "no usable
            result" instead of being dropped. A call whose outcome is not stored is also counted there, and
            the generator records how many there were.
  defect 9  the evidence wording counts designs and calls ("n calls from k designs"), never calls alone.

The record holds exact values, so the same calls render in either vocabulary (round 1 or v2) and in the
exact-only form (median changes, no categories). Pure aggregation and text; no I/O beyond loading the record.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from orch.contract.models import MetricName
from orch.policies.llm_view import HIGHER_IS_BETTER, METRIC_NAMES
from orch.policies.v2_vocab import CATEGORIES_V2, PLACEMENT_R1, PLACEMENT_V2, categorise

CALLS_SCHEMA_V2 = "v2-experience-calls-v1"
OUTCOMES = ("selected", "nothing_selectable", "no_options", "not_stored")
SCOPES = ("preserve_placement_true", "preserve_placement_false")
MIN_DESIGNS = 4
MENTION_GOAL = 0.05           # a goal category is named when it holds at least this share (round-1 cut-off)
MENTION_LIMIT = 0.15          # and a limit-metric category at this share
R1_ORDER = ("large_gain", "small_gain", "within_noise", "small_loss", "large_loss")
Form = Literal["words", "exact"]


def frequency_word(share: float) -> str:
    """The round-1 words: rarely under 15%, sometimes to 40%, often to 70%, usually above."""

    if share < 0.15:
        return "rarely"
    if share <= 0.40:
        return "sometimes"
    if share <= 0.70:
        return "often"
    return "usually"


@dataclass(frozen=True)
class CallV2:
    design: str
    brief: str
    worker: str
    scope: str
    outcome: str
    base_values: dict[str, float]
    result_values: dict[str, float] | None
    bands: dict[str, float] = field(default_factory=dict)

    @property
    def preserve(self) -> bool:
        return self.scope == SCOPES[0]


def load_calls_v2(path: Path | str) -> tuple[list[CallV2], str]:
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != CALLS_SCHEMA_V2:
        raise ValueError(f"unknown experience call record schema {data.get('schema_version')!r}")
    calls = [calls_from_item(item) for item in data["calls"]]
    return calls, hashlib.sha256(raw).hexdigest()


def calls_from_item(item: dict) -> CallV2:
    if item["scope"] not in SCOPES:
        raise ValueError(f"unknown scope {item['scope']!r}")
    if item["outcome"] not in OUTCOMES:
        raise ValueError(f"unknown outcome {item['outcome']!r}")
    if (item["outcome"] == "selected") != (item.get("result_values") is not None):
        raise ValueError("a selected call needs result values, and only a selected call has them")
    return CallV2(item["design"], item["brief"], item["worker"], item["scope"], item["outcome"],
                  dict(item["base_values"]), None if item.get("result_values") is None else dict(item["result_values"]),
                  {k: float(v) for k, v in (item.get("bands") or {}).items() if v})


@dataclass(frozen=True)
class LineStats:
    worker: str
    scope: str
    calls: int
    designs: int
    usable: int
    nothing_selectable: int
    no_usable: int
    counts: dict[str, dict[str, int]]          # metric value -> category -> calls (usable calls with a band)
    observed: dict[str, int]                   # metric value -> usable calls carrying a category for it
    no_change: int                             # usable calls whose goal and every carried limit are within noise
    no_change_base: int                        # usable calls that carry a goal category
    changes: dict[str, list[float]]            # metric value -> exact changes (ns) or percent of the base value


def _change_value(metric: str, base: float, result: float) -> float | None:
    if metric == MetricName.WNS_NS.value:
        return result - base
    return None if base == 0 else 100.0 * (result - base) / abs(base)


def line_stats(calls: Sequence[CallV2], *, worker: str, scope: str, goal: str, limits: Sequence[str],
               v2: bool) -> LineStats:
    """Pooled statistics of one (worker, placement setting) line; goal and limits are MetricName values."""

    mine = [c for c in calls if c.worker == worker and c.scope == scope]
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    observed: dict[str, int] = defaultdict(int)
    changes: dict[str, list[float]] = defaultdict(list)
    usable = nothing = no_change = no_change_base = 0
    for call in mine:
        if call.outcome == "nothing_selectable":
            nothing += 1
            continue
        if call.outcome != "selected":
            continue
        usable += 1
        assert call.result_values is not None
        cats: dict[str, str] = {}
        for metric in (goal, *limits):
            if metric not in call.base_values or metric not in call.result_values:
                continue
            delta = call.result_values[metric] - call.base_values[metric]
            value = _change_value(metric, call.base_values[metric], call.result_values[metric])
            if value is not None:
                changes[metric].append(value)
            band = call.bands.get(metric)
            if not band:
                continue
            category = categorise(delta, band, higher_is_better=MetricName(metric) in HIGHER_IS_BETTER, v2=v2)
            cats[metric] = category
            counts[metric][category] += 1
            observed[metric] += 1
        if goal in cats:
            no_change_base += 1
            quiet = {"within noise", "within_noise"}
            if cats[goal] in quiet and all(cats[m] in quiet for m in limits if m in cats):
                no_change += 1
    return LineStats(
        worker=worker, scope=scope, calls=len(mine), designs=len({c.design for c in mine}), usable=usable,
        nothing_selectable=nothing, no_usable=len(mine) - usable - nothing,
        counts={m: dict(v) for m, v in counts.items()}, observed=dict(observed), no_change=no_change,
        no_change_base=no_change_base, changes={m: list(v) for m, v in changes.items()},
    )


def _name(metric: str) -> str:
    return METRIC_NAMES.get(MetricName(metric), metric)


def _clause(stats: LineStats, metric: str, threshold: float, v2: bool) -> str:
    n = stats.observed.get(metric, 0)
    if n == 0:
        return "not observed"
    order = CATEGORIES_V2 if v2 else R1_ORDER
    cats = stats.counts.get(metric, {})
    ranked = sorted(((count / n, c) for c, count in cats.items() if count / n >= threshold - 1e-9),
                    key=lambda t: (-t[0], order.index(t[1])))
    return ", ".join(f"{category} {frequency_word(share)}" for share, category in ranked) or "not observed"


def _exact_clause(stats: LineStats, metric: str) -> str:
    values = sorted(stats.changes.get(metric, []))
    if not values:
        return "not observed"
    unit = " ns" if metric == MetricName.WNS_NS.value else "%"
    fmt = (lambda v: f"{v:+.3f}") if metric == MetricName.WNS_NS.value else (lambda v: f"{v:+.1f}")
    if len(values) >= 4:
        q = statistics.quantiles(values, n=4, method="inclusive")
        return f"median {fmt(statistics.median(values))}{unit} (middle half {fmt(q[0])} to {fmt(q[2])}{unit})"
    return f"median {fmt(statistics.median(values))}{unit}"


def render_line(stats: LineStats, *, goal: str, limits: Sequence[str], form: Form, v2: bool,
                placement_v2: bool) -> str:
    phrase = (PLACEMENT_V2 if placement_v2 else PLACEMENT_R1)[stats.scope == SCOPES[0]]
    head = f"  {phrase} ({stats.calls} calls from {stats.designs} designs): "
    n = stats.calls
    if form == "exact":
        parts = [f"goal {_name(goal)}: " + _exact_clause(stats, goal)]
        parts += [f"limit {_name(m)}: " + _exact_clause(stats, m) for m in limits]
        parts.append(f"nothing selectable {stats.nothing_selectable} of {n}")
        parts.append(f"no usable result {stats.no_usable} of {n}")
        return head + "; ".join(parts) + "."
    parts = [f"goal {_name(goal)}: " + _clause(stats, goal, MENTION_GOAL, v2)]
    parts += [f"limit {_name(m)}: " + _clause(stats, m, MENTION_LIMIT, v2) for m in limits]
    if stats.no_change_base:
        parts.append(f"no change {frequency_word(stats.no_change / stats.no_change_base)}")
    parts.append(f"nothing selectable {frequency_word(stats.nothing_selectable / n)}")
    parts.append(f"no usable result {frequency_word(stats.no_usable / n)}")
    return head + "; ".join(parts) + "."


def experience_calls_for(calls: Sequence[CallV2], *, brief: str, exclude_design: str | None) -> list[CallV2]:
    """The calls behind one rendered section: this brief's calls, the evaluated design left out."""

    return [c for c in calls if c.brief == brief and (exclude_design is None or c.design != exclude_design)]


def render_experience_v2(
    calls: Sequence[CallV2], *, brief: str, goal: str, limits: Sequence[str], workers: Sequence[str],
    exclude_design: str | None, header: str, intro: str, empty_text: str, form: Form = "words", v2: bool = True,
    placement_v2: bool = True, min_designs: int = MIN_DESIGNS,
) -> tuple[str, dict]:
    """Section text (without the blank lines the prompt puts around it) and a summary of what it rests on."""

    kept = experience_calls_for(calls, brief=brief, exclude_design=exclude_design)
    lines = [header, intro]
    shown: list[dict] = []
    omitted: list[dict] = []
    for worker in workers:
        block = []
        for scope in SCOPES:
            stats = line_stats(kept, worker=worker, scope=scope, goal=goal, limits=limits, v2=v2)
            if stats.calls == 0:
                continue
            row = {"worker": worker, "scope": scope, "calls": stats.calls, "designs": stats.designs}
            if stats.designs < min_designs:
                omitted.append(row)
                continue
            shown.append(row)
            block.append(render_line(stats, goal=goal, limits=limits, form=form, v2=v2, placement_v2=placement_v2))
        if block:
            lines.append(worker)
            lines += block
    if not shown:
        lines.append(empty_text)
    text = "\n".join(lines)
    return text, {"shown": shown, "omitted": omitted, "calls_in_scope": len(kept),
                  "designs_in_scope": sorted({c.design for c in kept}), "excluded_design": exclude_design,
                  "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
