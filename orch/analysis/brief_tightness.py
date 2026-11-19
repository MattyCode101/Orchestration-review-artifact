"""Training-only brief-tightness analysis from checked meter Evidence.

D3 includes selected depth-three and depth-four paths. The loader accepts only
the five-design training root and this module launches no hardware tools.
"""
from __future__ import annotations

import argparse
import hashlib
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from orch.analysis.landscape_e1 import (
    PROJECT, TRAIN_ROOT, Metrics, Snapshot, load_training, require_training_root,
)
from orch.analysis.landscape_e1_final import final_rankings

REPORT = PROJECT / "notes/brief_tightness_training.md"
EPS = 1e-10


@dataclass(frozen=True)
class TightBrief:
    """A start-relative brief with an increasing objective score."""

    id: str
    specialist: str | None

    def valid(self, candidate: Metrics, start: Metrics) -> bool:
        if self.id == "T":
            return candidate.packed_luts <= 1.01 * start.packed_luts
        if self.id == "T0":
            return candidate.packed_luts <= start.packed_luts
        if self.id == "A":
            return candidate.wns_ns >= start.wns_ns - 0.1
        if self.id == "A0":
            return candidate.wns_ns >= start.wns_ns
        if self.id == "B":
            return (
                candidate.wns_ns >= start.wns_ns
                and candidate.packed_luts <= start.packed_luts
            )
        raise ValueError(f"unknown brief: {self.id}")

    def value(self, candidate: Metrics, start: Metrics) -> float:
        if self.id in ("T", "T0"):
            return candidate.wns_ns
        if self.id in ("A", "A0"):
            return -candidate.packed_luts
        if self.id == "B":
            if start.wns_ns == 0 or start.packed_luts <= 0:
                raise ValueError("Brief B needs nonzero start WNS and positive LUTs")
            # Negative start WNS is a violation: use its magnitude as denominator.
            wns_gain = 100 * (candidate.wns_ns - start.wns_ns) / abs(start.wns_ns)
            lut_cut = 100 * (start.packed_luts - candidate.packed_luts) / start.packed_luts
            return wns_gain + lut_cut
        raise ValueError(f"unknown brief: {self.id}")


BRIEFS = (
    TightBrief("T", "timing"),
    TightBrief("A", "area"),
    TightBrief("A0", "area"),
    TightBrief("T0", "timing"),
    TightBrief("B", None),
)


@dataclass(frozen=True)
class PathCandidate:
    """One known path to a deduplicated DCP state."""

    metrics: Metrics
    sequence: tuple[str, ...]
    cross_worker: bool
    violating_intermediate: bool


@dataclass(frozen=True)
class Winner:
    """Best observed brief-valid state and conservative path interpretation."""

    candidate: PathCandidate
    score: float
    needs_both: bool
    violation_unavoidable: bool
    complexity_needed: bool


@dataclass(frozen=True)
class Baseline:
    """Best valid visited state for one fixed or specialist baseline."""

    metrics: Metrics
    sequence: tuple[str, ...]
    score: float
    share: float | None
    worker: str | None = None


@dataclass(frozen=True)
class DesignResult:
    design: str
    brief: TightBrief
    winner: Winner
    specialist: Baseline
    a_then_b: Baseline
    b_then_a: Baseline
    singles: Mapping[str, Baseline]


@dataclass(frozen=True)
class RuleOutcome:
    brief_id: str
    complex_designs: int
    mean_specialist_share: float
    keep: bool
    undefined_shares: int


def _state_paths(
    snapshot: Snapshot, design: str,
) -> tuple[
    dict[str, tuple[Metrics, list[tuple[str, ...]]]],
    dict[tuple[str, ...], Metrics],
]:
    """Deduplicate by DCP hash while retaining every checked path."""

    start = snapshot.starts[design]
    grouped: dict[str, tuple[Metrics, list[tuple[str, ...]]]] = {
        start.state_id: (start, [()])
    }
    paths = {(): start}
    for (name, sequence), edge in sorted(snapshot.edges.items()):
        if name != design or not edge.eligible or edge.amd_status != "pass":
            continue
        if edge.output_id is None or edge.metrics is None:
            raise ValueError("AMD-PASS edge lacks measured output")
        if edge.metrics.state_id != edge.output_id:
            raise ValueError("meter state ID differs from output hash")
        previous = grouped.get(edge.output_id)
        if previous is None:
            grouped[edge.output_id] = (edge.metrics, [sequence])
        else:
            if previous[0] != edge.metrics:
                raise ValueError("same DCP hash has inconsistent meter Evidence")
            previous[1].append(sequence)
        paths[sequence] = edge.metrics
    for sequence in paths:
        if sequence and any(sequence[:depth] not in paths for depth in range(1, len(sequence))):
            raise ValueError("AMD-PASS path has a missing checked prefix")
    return grouped, paths


def _path_candidate(
    snapshot: Snapshot, design: str, brief: TightBrief,
    sequence: tuple[str, ...], metrics: Metrics,
    paths: Mapping[tuple[str, ...], Metrics],
) -> PathCandidate:
    start = snapshot.starts[design]
    cross = len({snapshot.worker_of(recipe_id) for recipe_id in sequence}) > 1
    violates = any(
        not brief.valid(paths[sequence[:depth]], start)
        for depth in range(1, len(sequence))
    )
    return PathCandidate(metrics, sequence, cross, violates)


def _winner(
    snapshot: Snapshot, design: str, brief: TightBrief,
    grouped: Mapping[str, tuple[Metrics, list[tuple[str, ...]]]],
    paths: Mapping[tuple[str, ...], Metrics],
) -> Winner:
    start = snapshot.starts[design]
    valid = [
        (metrics, sequences, brief.value(metrics, start))
        for metrics, sequences in grouped.values()
        if brief.valid(metrics, start)
    ]
    if not valid:
        raise ValueError("B2 start is absent from brief-valid states")
    best_score = max(score for _, _, score in valid)
    best_paths = [
        _path_candidate(snapshot, design, brief, sequence, metrics, paths)
        for metrics, sequences, score in valid
        if abs(score - best_score) <= EPS
        for sequence in sequences
    ]
    # For tied best states prefer a path that needs neither a second worker
    # nor a brief violation, so the rule cannot gain evidence from tie choice.
    chosen = min(
        best_paths,
        key=lambda item: (
            item.cross_worker or item.violating_intermediate,
            item.violating_intermediate,
            item.cross_worker,
            len(item.sequence),
            item.sequence,
            item.metrics.state_id,
        ),
    )
    return Winner(
        candidate=chosen,
        score=best_score,
        needs_both=all(item.cross_worker for item in best_paths),
        violation_unavoidable=all(item.violating_intermediate for item in best_paths),
        complexity_needed=all(
            item.cross_worker or item.violating_intermediate for item in best_paths
        ),
    )


def _share(start_score: float, best_score: float, candidate_score: float) -> float | None:
    gain = best_score - start_score
    if gain <= EPS:
        return None
    ratio = (candidate_score - start_score) / gain
    if ratio < -EPS or ratio > 1 + EPS:
        raise ValueError("baseline improvement lies outside observed best")
    return max(0.0, min(1.0, ratio))


def _baseline(
    snapshot: Snapshot, design: str, brief: TightBrief,
    paths: Mapping[tuple[str, ...], Metrics],
    selected_paths: Sequence[tuple[str, ...]],
    best_score: float, *, worker: str | None = None,
) -> Baseline:
    start = snapshot.starts[design]
    candidates = []
    for sequence in dict.fromkeys(((), *selected_paths)):
        if sequence not in paths:
            raise ValueError(f"baseline path absent from AMD-PASS journal: {sequence}")
        metrics = paths[sequence]
        if brief.valid(metrics, start):
            candidates.append((sequence, metrics, brief.value(metrics, start)))
    sequence, metrics, score = min(
        candidates,
        key=lambda row: (-row[2], len(row[0]), row[0], row[1].state_id),
    )
    return Baseline(
        metrics=metrics, sequence=sequence, score=score,
        share=_share(brief.value(start, start), best_score, score),
        worker=worker,
    )


def analyze(
    snapshot: Snapshot,
    rankings: Mapping[tuple[str, str], tuple[str, ...]],
) -> list[DesignResult]:
    """Compare all five briefs on AMD-PASS, state-deduplicated training data."""

    results = []
    for design in snapshot.starts:
        grouped, paths = _state_paths(snapshot, design)
        timing = rankings[(design, "timing")]
        area = rankings[(design, "area")]
        if timing[0] not in snapshot.recipes["timing"] or area[0] not in snapshot.recipes["area"]:
            raise ValueError("fixed top recipe differs from signed worker set")
        all_recipes = snapshot.recipes["timing"] + snapshot.recipes["area"]
        for brief in BRIEFS:
            winner = _winner(snapshot, design, brief, grouped, paths)
            workers = (brief.specialist,) if brief.specialist else ("timing", "area")
            specialists = [
                _baseline(
                    snapshot, design, brief, paths,
                    tuple(
                        sequence for sequence in paths
                        if 1 <= len(sequence) <= 3
                        and all(snapshot.worker_of(recipe_id) == worker for recipe_id in sequence)
                    ),
                    winner.score, worker=worker,
                )
                for worker in workers
            ]
            specialist = min(
                specialists,
                key=lambda row: (-row.score, len(row.sequence), row.sequence, row.worker or ""),
            )
            a_then_b = _baseline(
                snapshot, design, brief, paths,
                ((timing[0],), (timing[0], area[0])),
                winner.score,
            )
            b_then_a = _baseline(
                snapshot, design, brief, paths,
                ((area[0],), (area[0], timing[0])),
                winner.score,
            )
            singles = {
                recipe_id: _baseline(
                    snapshot, design, brief, paths, ((recipe_id,),), winner.score,
                )
                for recipe_id in all_recipes
            }
            results.append(DesignResult(
                design, brief, winner, specialist, a_then_b, b_then_a, singles,
            ))
    return results


def evaluate_rule(brief_id: str, rows: Sequence[DesignResult]) -> RuleOutcome:
    """Apply the two-of-five and strictly-under-90-percent rule."""

    if len(rows) != 5 or any(row.brief.id != brief_id for row in rows):
        raise ValueError("rule requires five results for one brief")
    complex_designs = sum(row.winner.complexity_needed for row in rows)
    undefined = sum(row.specialist.share is None for row in rows)
    # With no improvement from B2, the specialist matched the best (B2).
    shares = [1.0 if row.specialist.share is None else row.specialist.share for row in rows]
    mean = statistics.mean(shares)
    return RuleOutcome(
        brief_id, complex_designs, mean, complex_designs >= 2 and mean < 0.9, undefined,
    )


def _path(sequence: tuple[str, ...]) -> str:
    return "B2" if not sequence else " -> ".join(sequence)


def _share_label(share: float | None) -> str:
    return "n/a" if share is None else f"{100 * share:.1f}%"


def render_report(
    snapshot: Snapshot, results: Sequence[DesignResult],
    rankings: Mapping[tuple[str, str], tuple[str, ...]],
    model_hashes: Mapping[str, str],
) -> str:
    """Render the training result with the decision summary first."""

    by_brief = {
        brief.id: [row for row in results if row.brief.id == brief.id]
        for brief in BRIEFS
    }
    rules = {name: evaluate_rule(name, rows) for name, rows in by_brief.items()}
    distinct = sum(len(_state_paths(snapshot, design)[0]) for design in snapshot.starts)
    repeated_winners = sum(
        any(left == right for left, right in zip(row.winner.candidate.sequence, row.winner.candidate.sequence[1:]))
        for row in results
    )
    lines = [
        "# Brief tightness on AMD-PASS training states",
        "",
        "## Summary",
        "",
        f"**Verified, training only.** {snapshot.journal_edge_count} checked edges across five designs "
        f"yield {distinct} distinct state hashes after deduplication. D3 selected deeper branches; "
        "depth three and four are not exhaustive. A share is the baseline's brief-objective "
        "improvement from B2 divided by the best observed improvement from B2.",
        "",
        "| Brief | Complex best designs / 5 | Mean specialist share | Rule outcome |",
        "|---|---:|---:|---|",
    ]
    for brief in BRIEFS:
        rule = rules[brief.id]
        if brief.id in ("T", "A"):
            label = ("passes" if rule.keep else "fails") + " addition rule (reference brief)"
        else:
            label = "keep" if rule.keep else "do not keep"
        lines.append(
            f"| {brief.id} | {rule.complex_designs}/5 | "
            f"{100 * rule.mean_specialist_share:.1f}% | {label} |"
        )
    lines.extend([
        "",
        "**Rule interpretation.** A complex best has no equally scoring known path that "
        "is both single-worker and brief-valid at every intermediate. The rule requires "
        "at least two such designs and mean specialist share strictly below 90%. "
        "An undefined share (no gain over B2) counts as 100% for this decision. "
        "T and A are existing references; only A0, T0 and B are proposed additions.",
        "",
        f"**Verified path limit.** {repeated_winners}/25 representative best paths immediately "
        "repeat a recipe. These historical outcomes are included in the observed best "
        "and specialist oracle; the current worker no-repeat rule cannot reproduce "
        "those exact sequences.",
        "",
        "**Brief B definition.** WNS gain is 100 x (WNS - start WNS) / abs(start WNS); "
        "LUT cut is 100 x (start LUTs - LUTs) / start LUTs. B maximises their sum "
        "while requiring neither metric worse than B2. All five start WNS values "
        "are nonzero. B has no own worker, so its specialist baseline is the better "
        "of the timing-only and area-only paths of length at most three.",
        "",
        "## Best state and specialist/fixed baselines by design",
        "",
        "A and B in the fixed-order columns denote timing and area workers. "
        "Each order uses that design's top FINAL-scored B2 recipe from each worker; "
        "the best brief-valid visited state, including B2, is counted. "
        "The specialist is an oracle over observed own-worker paths up to length three.",
        "",
        "| Design | Brief | Best state SHA-256 | Best path | WNS ns | Packed LUTs | Both workers needed | Invalid intermediate | Specialist path / share | A then B share | B then A share |",
        "|---|---|---|---|---:|---:|---|---|---|---:|---:|",
    ])
    for row in results:
        c = row.winner.candidate
        lines.append(
            f"| {row.design} | {row.brief.id} | {c.metrics.state_id} | "
            f"{_path(c.sequence)} | {c.metrics.wns_ns:.3f} | "
            f"{c.metrics.packed_luts:.0f} | "
            f"{'yes' if row.winner.needs_both else 'no'} | "
            f"{'yes' if c.violating_intermediate else 'no'} | "
            f"{row.specialist.worker}: {_path(row.specialist.sequence)} / "
            f"{_share_label(row.specialist.share)} | "
            f"{_share_label(row.a_then_b.share)} | "
            f"{_share_label(row.b_then_a.share)} |"
        )
    recipes = snapshot.recipes["timing"] + snapshot.recipes["area"]
    lines.extend([
        "",
        "## Fixed top recipes by training design",
        "",
        "| Design | Timing A top | Area B top |",
        "|---|---|---|",
    ])
    for design in snapshot.starts:
        lines.append(f"| {design} | {rankings[(design, 'timing')][0]} | {rankings[(design, 'area')][0]} |")
    lines.extend([
        "",
        "## Every single-recipe baseline",
        "",
        "Each cell is the share reached by one D1 recipe, with B2 available as final pick.",
        "",
        "| Code | Exact recipe ID |",
        "|---|---|",
    ])
    for index, recipe_id in enumerate(recipes, 1):
        lines.append(f"| R{index} | {recipe_id} |")
    lines.extend([
        "",
        "| Design | Brief | " + " | ".join(f"R{index}" for index in range(1, len(recipes) + 1)) + " |",
        "|---|---|" + "|".join("---:" for _ in recipes) + "|",
    ])
    for row in results:
        cells = [_share_label(row.singles[recipe_id].share) for recipe_id in recipes]
        lines.append(f"| {row.design} | {row.brief.id} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## Provenance and limits",
        "",
        f"- Checked D3 journal prefix SHA-256: {snapshot.journal_prefix_sha256}.",
        f"- FINAL timing model SHA-256: {model_hashes['timing']}; area model SHA-256: {model_hashes['area']}.",
        f"- Analysis source SHA-256: {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}.",
    ])
    lines.extend(
        f"- {name} SHA-256: {digest}."
        for name, digest in sorted(snapshot.source_sha256.items())
    )
    lines.extend([
        "- **Inferred limit:** these selected training paths cannot establish held-out "
        "performance or prove that an unobserved path cannot reach a better state.",
        "",
    ])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=TRAIN_ROOT)
    args = parser.parse_args(argv)
    root = require_training_root(args.data_root)
    snapshot = load_training(root, final=True)
    rankings, model_hashes = final_rankings(snapshot, root)
    results = analyze(snapshot, rankings)
    REPORT.write_text(
        render_report(snapshot, results, rankings, model_hashes), encoding="utf-8",
    )
    print(f"wrote {REPORT} from {snapshot.journal_edge_count} checked training edges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
