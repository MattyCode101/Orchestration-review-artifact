"""FINAL training E1 landscape and fixed-baseline shares.

All inputs are the checked D1-D3 phase journal, saved meter Evidence and
canonical AMD PASS records. The D3 phase includes selected depth-three and
depth-four paths; it is not an exhaustive depth-three/four search.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from orch.analysis.landscape_e1 import (
    PROJECT, SCENARIOS, TRAIN_DESIGNS, TRAIN_ROOT, Choice, Metrics, Scenario,
    Snapshot, _output_inside, _path, _plot_scatter, _repeat_unavailable,
    _write_scatter_csv, best_choice, load_training, noise_flags, scatter_rows,
)
from orch.analysis.report import share_of_best
from orch.config import load_brief, load_recipe_sets, load_worker_card
from orch.contract.models import DesignState, Evidence, Mandate, WorkerSettings
from orch.recipes.registry import load_registry
from orch.workers.fitted import FittedWorker
from orch.workers.telemetry import SavedTelemetryProvider

REPORT_FINAL = PROJECT / "notes/e1_training_landscape_FINAL.md"
SCATTER_FINAL = PROJECT / "notes/e1_scatter_final"


def _start_context(root: Path, design: str, snapshot: Snapshot) -> tuple[DesignState, Evidence]:
    """Read the measured B2 state named by the checked training journal."""

    state_id = snapshot.starts[design].state_id
    state = DesignState.model_validate_json(
        (root / "states" / state_id / "state.json").read_bytes()
    )
    if state.id != state_id or state.design != design or state.parent_id is not None:
        raise ValueError("FINAL B2 state metadata differs from training journal")
    matches = []
    for path in (root / "measurements" / state_id).glob("*/evidence.json"):
        evidence = Evidence.model_validate_json(path.read_bytes())
        if (
            evidence.state_id == state_id
            and evidence.meter_version == snapshot.starts[design].meter_version
            and evidence.metric("wns_ns").value == snapshot.starts[design].wns_ns
            and evidence.metric("packed_luts").value == snapshot.starts[design].packed_luts
        ):
            matches.append(evidence)
    if len(matches) != 1:
        raise ValueError("FINAL B2 start lacks one matching saved Evidence")
    return state, matches[0]


def final_rankings(
    snapshot: Snapshot, root: Path,
) -> tuple[dict[tuple[str, str], tuple[str, ...]], dict[str, str]]:
    """Rank each signed worker on each B2 with the published FINAL JSON."""

    provider = SavedTelemetryProvider(root)
    registry = load_registry(PROJECT)
    sets = load_recipe_sets(PROJECT / "configs/recipe_sets.yaml")
    sets.require_signed_off()
    workers = {
        worker_id: FittedWorker(
            load_worker_card(PROJECT / "configs/workers" / f"{worker_id}.yaml"),
            sets.worker_sets[worker_id], registry,
        )
        for worker_id in ("timing", "area")
    }
    if any(worker.model_label != "FINAL" for worker in workers.values()):
        raise ValueError("FINAL landscape requires both published FINAL models")
    rankings: dict[tuple[str, str], tuple[str, ...]] = {}
    for design in TRAIN_DESIGNS:
        state, evidence = _start_context(root, design, snapshot)
        context = provider(state, evidence, None, None)
        for worker_id, worker in workers.items():
            brief = load_brief(PROJECT / "configs/briefs" / (
                "timing.yaml" if worker_id == "timing" else "area.yaml"
            ))
            mandate = Mandate(
                id=f"landscape-final-{design}-{worker_id}",
                worker_id=worker_id, base_state_id=state.id,
                settings=WorkerSettings(max_runs=5, preserve_placement=False),
                brief_hash=brief.fingerprint(),
            )
            plan = worker.explain_plan(state, context, mandate, ())
            if len(plan.selected_ids) != 5:
                raise ValueError("FINAL ranking omitted a signed recipe")
            rankings[(design, worker_id)] = plan.selected_ids
    return rankings, {
        worker_id: worker.model_sha256 for worker_id, worker in workers.items()
    }


def _baseline_definitions(
    snapshot: Snapshot, design: str,
    rankings: dict[tuple[str, str], tuple[str, ...]],
) -> list[tuple[str, tuple[tuple[str, ...], ...], int]]:
    """Single recipes, both fixed orders and independent top-N B2 tries."""

    timing = rankings[(design, "timing")]
    area = rankings[(design, "area")]
    result = [
        (recipe_id, ((recipe_id,),), 1)
        for recipe_id in snapshot.recipes["timing"] + snapshot.recipes["area"]
    ]
    result.extend((
        ("A then B", ((timing[0],), (timing[0], area[0])), 2),
        ("B then A", ((area[0],), (area[0], timing[0])), 2),
    ))
    for worker_id, order in (("A", timing), ("B", area)):
        for n in (1, 2, 3):
            result.append((
                f"worker {worker_id} alone, {n} tries",
                tuple((recipe_id,) for recipe_id in order[:n]), n,
            ))
    return result


def _best_reached(
    snapshot: Snapshot, design: str, scenario: Scenario,
    paths: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, ...], Metrics]:
    """Final-pick the best brief-valid visited state, including B2."""

    start = snapshot.starts[design]
    choices: dict[tuple[str, ...], Metrics] = {(): start}
    for path in paths:
        for depth in range(1, len(path) + 1):
            prefix = path[:depth]
            edge = snapshot.edges.get((design, prefix))
            if edge is None:
                raise ValueError(f"fixed baseline is absent from D1/D2: {design} {prefix}")
            if not edge.eligible or edge.amd_status != "pass" or edge.metrics is None:
                raise ValueError("FINAL fixed baseline lacks AMD-PASS meter Evidence")
            if scenario.valid(edge.metrics, start):
                choices[prefix] = edge.metrics
    return min(
        choices.items(),
        key=lambda row: (-scenario.value(row[1]), len(row[0]), row[0], row[1].state_id),
    )


def baseline_shares(
    snapshot: Snapshot,
    rankings: dict[tuple[str, str], tuple[str, ...]],
) -> list[dict[str, object]]:
    """Report every fixed baseline against each design/brief's D3-inclusive best."""

    rows: list[dict[str, object]] = []
    for design in TRAIN_DESIGNS:
        start = snapshot.starts[design]
        for scenario in (SCENARIOS[1], SCENARIOS[5]):
            best = best_choice(snapshot, design, scenario)
            for label, paths, tries in _baseline_definitions(snapshot, design, rankings):
                selected_path, candidate = _best_reached(snapshot, design, scenario, paths)
                direction = "max" if scenario.worker == "timing" else "min"
                start_value = start.wns_ns if direction == "max" else start.packed_luts
                best_value = best.metrics.wns_ns if direction == "max" else best.metrics.packed_luts
                candidate_value = (
                    candidate.wns_ns if direction == "max" else candidate.packed_luts
                )
                share = share_of_best(start_value, best_value, candidate_value, direction)
                wn_noise, lut_noise = noise_flags(snapshot, design, candidate)
                rows.append({
                    "design": design,
                    "brief": "T" if direction == "max" else "A",
                    "baseline": label,
                    "tries": tries,
                    "evaluated_paths": "; ".join(_path(path) for path in paths),
                    "picked_sequence": _path(selected_path),
                    "state_sha256": candidate.state_id,
                    "wns_ns": candidate.wns_ns,
                    "packed_luts": candidate.packed_luts,
                    "best_state_sha256": best.metrics.state_id,
                    "share_of_best": share,
                    "within_wns_noise": wn_noise,
                    "within_lut_noise": lut_noise,
                })
    return rows


def _noise_label(snapshot: Snapshot, design: str, choice: Choice) -> str:
    flags = noise_flags(snapshot, design, choice.metrics)
    return "/".join(
        "?" if value is None else ("yes" if value else "no") for value in flags
    )


def _share_label(value: object) -> str:
    return "?" if value is None else f"{100 * float(value):.1f}%"


def _write_baseline_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_final(
    snapshot: Snapshot,
    rankings: dict[tuple[str, str], tuple[str, ...]],
    model_hashes: dict[str, str],
    baselines: list[dict[str, object]],
    scatter_files: dict[str, tuple[str, str | None]],
    baseline_csv: Path,
) -> str:
    """Summary first, followed by all winners and all fixed baseline shares."""

    defaults = []
    grid = []
    cross = violations = no_repeat = deeper = 0
    for design in TRAIN_DESIGNS:
        for scenario in SCENARIOS:
            full = best_choice(snapshot, design, scenario)
            two = best_choice(snapshot, design, scenario, max_depth=2)
            allowed = best_choice(snapshot, design, scenario, allow_repeat=False)
            if "(default)" in scenario.label:
                cross += full.cross_worker_needed
                violations += full.intermediate_violates
                no_repeat += _repeat_unavailable(full.sequence)
                deeper += len(full.sequence) > 2
                defaults.append(
                    f"| {design} | {scenario.worker.upper()} | {_path(two.sequence)} "
                    f"({two.metrics.wns_ns:.3f}, {two.metrics.packed_luts:.0f}) | "
                    f"{_path(full.sequence)} ({full.metrics.wns_ns:.3f}, "
                    f"{full.metrics.packed_luts:.0f}) | "
                    f"{'yes' if full.cross_worker_needed else 'no'} | "
                    f"{'yes' if full.intermediate_violates else 'no'} | "
                    f"{'yes' if _repeat_unavailable(full.sequence) else 'no'} | "
                    f"{_path(allowed.sequence)} | {_noise_label(snapshot, design, full)} |"
                )
            grid.append(
                f"| {design} | {scenario.label} | {_path(full.sequence)} | "
                f"{full.metrics.wns_ns:.3f} | {full.metrics.packed_luts:.0f} | "
                f"{'yes' if full.cross_worker_needed else 'no'} | "
                f"{'yes' if full.intermediate_violates else 'no'} | "
                f"{'yes' if _repeat_unavailable(full.sequence) else 'no'} |"
            )
    depth_counts = Counter(len(edge.sequence) for edge in snapshot.edges.values())
    distinct_outputs = len({
        edge.output_id for edge in snapshot.edges.values() if edge.output_id
    })
    baseline_table = [
        f"| {row['design']} | {row['brief']} | {row['baseline']} | "
        f"{row['tries']} | {row['picked_sequence']} | "
        f"{_share_label(row['share_of_best'])} | "
        f"{'?' if row['within_wns_noise'] is None else row['within_wns_noise']}/"
        f"{'?' if row['within_lut_noise'] is None else row['within_lut_noise']} |"
        for row in baselines
    ]
    sources = "; ".join(
        f"{name} SHA-256 {digest}"
        for name, digest in sorted(snapshot.source_sha256.items())
    )
    scatter_links = [
        f"- {design}: [CSV](e1_scatter_final/{files[0]})"
        + (f", [PNG](e1_scatter_final/{files[1]})" if files[1] else "")
        for design, files in scatter_files.items()
    ]
    rankings_text = [
        f"- {design}: A {', '.join(rankings[(design, 'timing')])}; "
        f"B {', '.join(rankings[(design, 'area')])}."
        for design in TRAIN_DESIGNS
    ]
    lines = [
        "# FINAL E1 training landscape",
        "",
        "## Summary",
        "",
        f"**Verified, FINAL.** The D3-complete training journal contains "
        f"{snapshot.journal_edge_count} attempted edges: "
        f"{depth_counts[1]} depth-1, {depth_counts[2]} depth-2, "
        f"{depth_counts[3]} depth-3 and {depth_counts[4]} depth-4. "
        f"All {len(snapshot.edges)} edge outputs are AMD PASS, representing "
        f"{distinct_outputs} distinct output DCP hashes. "
        f"Under the two default briefs, {deeper}/10 best states need more than "
        f"two steps, {cross}/10 need a cross-worker path, and "
        f"{violations}/10 pass through a brief-violating intermediate. "
        f"{no_repeat}/10 default winners repeat a recipe on the state it produced.",
        "",
        "**Interpretation.** The D3 phase sampled selected deeper branches; its "
        "depth-three and depth-four set is not exhaustive. The two-step column "
        "answers the original E1 question, while the full column uses every "
        "committed AMD-PASS output. A baseline's share is its improvement from "
        "B2 divided by the full winner's improvement, using WNS for Brief T "
        "and packed LUTs for Brief A; undefined means the winner did not improve "
        "on B2. Each worker's 1/2/3 tries are independent top-ranked recipes "
        "on B2, with the best brief-valid visited state selected. A then B and "
        "B then A use each worker's first FINAL-scored B2 recipe. Intermediate "
        "violations are allowed in a sequence; every final pick obeys the "
        "start-relative brief. Noise flags are start-relative WNS/LUT bands.",
        "",
        "## Default brief winners",
        "",
        "| Design | Brief | Best within 2: path (WNS, LUT) | Best all D3: path (WNS, LUT) | Cross needed | Intermediate violation | Own-recipe repeat | Best no-repeat path | WNS/LUT within noise |",
        "|---|---|---|---|---|---|---|---|---|",
        *defaults,
        "",
        "## Constraint grid winners across all committed outputs",
        "",
        "| Design | Scenario | Sequence | WNS ns | Packed LUTs | Cross needed | Intermediate violation | Own-recipe repeat |",
        "|---|---|---|---:|---:|---|---|---|",
        *grid,
        "",
        "## Fixed baseline share of best",
        "",
        f"[Machine-readable CSV]({baseline_csv.name}) contains all "
        f"{len(baselines)} design/brief/baseline rows and exact state hashes.",
        "",
        "| Design | Brief | Baseline | Tries | Best reached | Share of best | WNS/LUT within noise |",
        "|---|---|---|---:|---|---:|---|",
        *baseline_table,
        "",
        "## FINAL B2 recipe rankings",
        "",
        *rankings_text,
        "",
        "## Timing versus area scatter",
        "",
        *scatter_links,
        "",
        "## Provenance",
        "",
        f"- **Verified:** D3 journal prefix SHA-256 {snapshot.journal_prefix_sha256}; "
        f"generated {datetime.now(timezone.utc).isoformat()}.",
        f"- **Verified:** FINAL timing model SHA-256 {model_hashes['timing']}; "
        f"FINAL area model SHA-256 {model_hashes['area']}.",
        f"- **Verified:** {sources}.",
        f"- **Verified analysis code:** landscape_e1.py SHA-256 "
        f"{hashlib.sha256((PROJECT / 'orch/analysis/landscape_e1.py').read_bytes()).hexdigest()}; "
        f"landscape_e1_final.py SHA-256 "
        f"{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}.",
        "- **Limit:** This is a five-design training description, not a held-out "
        "estimate. AMD PASS is the saved finite-vector check, not formal equivalence.",
        "",
    ]
    return "\n".join(lines)


def run_final(
    *, data_root: Path | str = TRAIN_ROOT,
    report: Path | str = REPORT_FINAL,
    scatter_dir: Path | str = SCATTER_FINAL,
) -> Path:
    """Write the FINAL training landscape without reading held-out outcomes."""

    snapshot = load_training(data_root, final=True)
    root = TRAIN_ROOT.resolve(strict=True)
    rankings, hashes = final_rankings(snapshot, root)
    baseline_rows = baseline_shares(snapshot, rankings)
    report_path = _output_inside(Path(report), PROJECT / "notes")
    scatter_path = _output_inside(Path(scatter_dir), PROJECT / "notes")
    scatter_path.mkdir(parents=True, exist_ok=True)
    (PROJECT / "data/scratch_landscape").mkdir(parents=True, exist_ok=True)
    scatter_files = {}
    for design in TRAIN_DESIGNS:
        rows = scatter_rows(snapshot, design)
        csv_name = f"{design}.csv"
        png_name = f"{design}.png"
        _write_scatter_csv(scatter_path / csv_name, rows)
        plotted = _plot_scatter(
            scatter_path / png_name, rows, snapshot.starts[design],
            label="FINAL", max_depth=4,
        )
        scatter_files[design] = (csv_name, png_name if plotted else None)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = report_path.with_name("e1_training_baseline_shares_FINAL.csv")
    _write_baseline_csv(csv_path, baseline_rows)
    report_path.write_text(
        render_final(
            snapshot, rankings, hashes, baseline_rows, scatter_files, csv_path
        ),
        encoding="utf-8",
    )
    return report_path

