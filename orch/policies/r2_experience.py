"""Round-2 worker experience for the LLM orchestrator (Wednesday-night note, sections 4 and 7).

Experience is what the contract has already seen a worker do: one call of a worker on a saved state
with one placement setting, binned with the same categories and noise bands as the view. It needs no
worker cooperation, internals or recipe names. This module is pure aggregation, text and hashing; the
call record (configs/experience/calls_ab.json) is built from stored outcomes by
scripts/feature_ablation_2026-11-12/fa_calls.py.

  * leave-one-design-out: naming the evaluated design drops its calls before anything is counted, so
    the card cannot know that design's answers; no name keeps them (the leakage check);
  * unconditional: one summary per (worker, placement setting); nothing is conditioned on the state;
  * design-equal: every design counts the same however many calls it contributed (as the round-1
    profiles did); pooled counts are kept for the count form and for the evidence word;
  * fuzzy form: a frequency word (the round-1 cut-offs) plus an evidence word from the pooled number of
    calls; count form: the pooled counts themselves.

The greedy reference rule at the end is used for agreement analysis only; no policy runs it.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from orch.policies.llm_view import CATEGORIES

CALLS_SCHEMA = "r2-experience-calls-v1"
SCOPES = ("preserve_placement_true", "preserve_placement_false")
SCOPE_PHRASE = {
    "preserve_placement_true": "placement preserved",
    "preserve_placement_false": "re-placing allowed",
}
MENTION_GOAL = 0.05          # a category is named for the goal when it holds at least this share
MENTION_LIMIT = 0.15         # and for a limit metric at this share (the round-1 profile cut-offs)
FEW_BELOW = 10               # evidence words by pooled number of calls: few < 10 <= some < 30 <= many
MANY_FROM = 30
VALUE = {"large_gain": 2.0, "small_gain": 1.0, "within_noise": 0.0, "small_loss": -1.0, "large_loss": -2.0}
LOSSES = ("small_loss", "large_loss")
Form = Literal["fuzzy", "count"]


def scope_of(preserve_placement: bool) -> str:
    return SCOPES[0] if preserve_placement else SCOPES[1]


def frequency_word(share: float) -> str:
    """The round-1 words: rarely under 15%, sometimes to 40%, often to 70%, usually above."""

    if share < 0.15:
        return "rarely"
    if share <= 0.40:
        return "sometimes"
    if share <= 0.70:
        return "often"
    return "usually"


def strength_word(calls: int) -> str:
    if calls < FEW_BELOW:
        return "few observations"
    if calls < MANY_FROM:
        return "some observations"
    return "many observations"


@dataclass(frozen=True)
class Call:
    """One simulated worker call on a saved state of one design."""

    design: str
    worker: str
    scope: str
    valid: bool
    changes: Mapping[str, str]          # metric name -> category against the call's base state


def load_calls(path: Path | str) -> tuple[list[Call], str]:
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != CALLS_SCHEMA:
        raise ValueError(f"unknown experience call record schema {data.get('schema_version')!r}")
    calls: list[Call] = []
    for item in data["calls"]:
        if item["scope"] not in SCOPES:
            raise ValueError(f"unknown scope {item['scope']!r}")
        bad = [c for c in item["changes"].values() if c not in CATEGORIES]
        if bad:
            raise ValueError(f"unknown category {bad[0]!r}")
        calls.append(Call(item["design"], item["worker"], item["scope"], bool(item["valid"]), dict(item["changes"])))
    return calls, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ScopeStats:
    worker: str
    scope: str
    calls: int
    designs: int
    valid_calls: int
    quiet_calls: int
    invalid_rate: float                       # design-equal
    no_change_rate: float                     # design-equal
    shares: Mapping[str, Mapping[str, float]]     # metric -> category -> design-equal share of valid calls
    counts: Mapping[str, Mapping[str, int]]       # metric -> category -> pooled calls
    observed: Mapping[str, int]                   # metric -> valid calls that carry a category for it


def _stats(worker: str, scope: str, calls: Sequence[Call]) -> ScopeStats:
    by_design: dict[str, list[Call]] = defaultdict(list)
    for call in calls:
        by_design[call.design].append(call)
    invalid, quiet = [], []
    per_design: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    observed: dict[str, int] = defaultdict(int)
    valid_total = quiet_total = 0
    for design in sorted(by_design):
        items = by_design[design]
        valid = [c for c in items if c.valid]
        silent = sum(1 for c in valid if c.changes and all(v == "within_noise" for v in c.changes.values()))
        invalid.append((len(items) - len(valid)) / len(items))
        quiet.append(silent / len(items))
        valid_total, quiet_total = valid_total + len(valid), quiet_total + silent
        for metric in sorted({m for c in valid for m in c.changes}):
            seen = [c.changes[metric] for c in valid if metric in c.changes]
            observed[metric] += len(seen)
            for category in CATEGORIES:
                n = seen.count(category)
                per_design[metric][category].append(n / len(seen))
                counts[metric][category] += n
    shares = {
        metric: {c: sum(v) / len(v) for c, v in cats.items() if sum(v) > 0}
        for metric, cats in per_design.items()
    }
    return ScopeStats(
        worker=worker, scope=scope, calls=len(calls), designs=len(by_design), valid_calls=valid_total,
        quiet_calls=quiet_total, invalid_rate=sum(invalid) / len(invalid), no_change_rate=sum(quiet) / len(quiet),
        shares=shares, counts={m: dict(v) for m, v in counts.items()}, observed=dict(observed),
    )


@dataclass(frozen=True)
class ExperienceTable:
    stats: Mapping[tuple[str, str], ScopeStats]
    excluded: str | None
    designs: tuple[str, ...]
    source_sha256: str

    def get(self, worker: str, scope: str) -> ScopeStats | None:
        return self.stats.get((worker, scope))

    def sha256(self) -> str:
        body = {
            "source": self.source_sha256, "excluded": self.excluded, "designs": list(self.designs),
            "stats": {
                f"{w}|{s}": {
                    "calls": st.calls, "designs": st.designs, "invalid": round(st.invalid_rate, 9),
                    "quiet": round(st.no_change_rate, 9),
                    "shares": {m: {c: round(v, 9) for c, v in sorted(cs.items())} for m, cs in sorted(st.shares.items())},
                    "counts": {m: dict(sorted(cs.items())) for m, cs in sorted(st.counts.items())},
                }
                for (w, s), st in sorted(self.stats.items())
            },
        }
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_table(calls: Sequence[Call], source_sha256: str, *, exclude_design: str | None = None) -> ExperienceTable:
    """Leave-one-design-out table when `exclude_design` is given, all designs otherwise."""

    kept = [c for c in calls if exclude_design is None or c.design != exclude_design]
    groups: dict[tuple[str, str], list[Call]] = defaultdict(list)
    for call in kept:
        groups[(call.worker, call.scope)].append(call)
    return ExperienceTable(
        stats={key: _stats(key[0], key[1], items) for key, items in groups.items()},
        excluded=exclude_design, designs=tuple(sorted({c.design for c in kept})), source_sha256=source_sha256,
    )


# ------------------------------------------------------------------ text
def _fuzzy_clause(shares: Mapping[str, float] | None, threshold: float) -> str:
    if not shares:
        return "not observed"
    ranked = sorted(((s, c) for c, s in shares.items() if s >= threshold - 1e-9), key=lambda t: (-t[0], t[1]))
    return ", ".join(f"{category} {frequency_word(share)}" for share, category in ranked) or "not observed"


def _count_clause(counts: Mapping[str, int] | None, observed: int) -> str:
    if not counts or observed == 0:
        return "not observed"
    parts = [f"{category} {counts[category]}" for category in CATEGORIES if counts.get(category)]
    return ", ".join(parts) + f" (of {observed})"


def scope_line(stats: ScopeStats, scope: str, goal: str, limits: Sequence[str], form: Form) -> str:
    phrase = SCOPE_PHRASE[scope]
    if form == "fuzzy":
        head = f"  {phrase} ({strength_word(stats.calls)}): "
        parts = [f"brief goal {goal}: " + _fuzzy_clause(stats.shares.get(goal), MENTION_GOAL)]
        parts += [f"brief limit {m}: " + _fuzzy_clause(stats.shares.get(m), MENTION_LIMIT) for m in limits]
        parts.append(f"no change {frequency_word(stats.no_change_rate)}")
        parts.append(f"no usable result {frequency_word(stats.invalid_rate)}")
        return head + "; ".join(parts) + "."
    head = f"  {phrase} ({stats.calls} calls on {stats.designs} designs): "
    parts = [f"brief goal {goal}: " + _count_clause(stats.counts.get(goal), stats.observed.get(goal, 0))]
    parts += [f"brief limit {m}: " + _count_clause(stats.counts.get(m), stats.observed.get(m, 0)) for m in limits]
    parts.append(f"no change {stats.quiet_calls} of {stats.calls}")
    parts.append(f"no usable result {stats.calls - stats.valid_calls} of {stats.calls}")
    return head + "; ".join(parts) + "."


def render_experience_section(
    table: ExperienceTable, *, goal: str, limits: Sequence[str], workers: Sequence[str], form: Form,
) -> str:
    """The text of the section, without the blank lines the prompt puts around it."""

    intro = (
        "Each line summarises calls of one worker with one placement setting, made on saved states of "
        "earlier designs and judged against the same brief as yours. "
        + ("Frequencies are shares of those calls." if form == "fuzzy" else "Numbers are counts of those calls.")
    )
    lines = ["Past experience with the workers (profiles from earlier designs)", intro]
    for worker in workers:
        block = [(scope, table.get(worker, scope)) for scope in SCOPES]
        block = [(scope, st) for scope, st in block if st is not None]
        if not block:
            continue
        lines.append(worker)
        lines += [scope_line(st, scope, goal, limits, form) for scope, st in block]
    return "\n".join(lines)


# ------------------------------------------------------------------ greedy reference rule (analysis only)
@dataclass(frozen=True)
class GreedyPick:
    worker: str
    preserve_placement: bool
    score: float


def greedy_score(stats: ScopeStats, goal: str, limits: Sequence[str]) -> float:
    """Design-equal expected goal value minus the share of calls that worsen a limit metric.

    A call that returned no usable result counts as value zero. Large gain +2, small gain +1, within
    noise 0, small loss -1, large loss -2; a limit metric's small and large losses each cost one.
    """

    goal_value = sum(share * VALUE[c] for c, share in stats.shares.get(goal, {}).items())
    limit_loss = sum(stats.shares.get(m, {}).get(c, 0.0) for m in limits for c in LOSSES)
    return (1.0 - stats.invalid_rate) * (goal_value - limit_loss)


def greedy_scores(
    table: ExperienceTable, goal: str, limits: Sequence[str], workers: Sequence[str],
) -> dict[tuple[str, bool], float]:
    out: dict[tuple[str, bool], float] = {}
    for worker in workers:
        for preserve in (True, False):
            stats = table.get(worker, scope_of(preserve))
            if stats is not None:
                out[(worker, preserve)] = greedy_score(stats, goal, limits)
    return out


def pick_greedy(
    table: ExperienceTable, goal: str, limits: Sequence[str], workers: Sequence[str],
    tried: Collection[tuple[str, bool]] = (),
) -> GreedyPick | None:
    """The best untried (worker, placement setting) by experience; ties go to the card order, then re-placing."""

    scores = greedy_scores(table, goal, limits, workers)
    candidates = [(key, score) for key, score in scores.items() if key not in set(tried)]
    if not candidates:
        return None
    (worker, preserve), score = min(
        candidates, key=lambda item: (-round(item[1], 12), list(workers).index(item[0][0]), 1 if item[0][1] else 0),
    )
    return GreedyPick(worker, preserve, score)
