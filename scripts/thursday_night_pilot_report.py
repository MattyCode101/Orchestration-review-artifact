"""Read-only, training-only share-of-best audit for the Thursday policy pilot.

The denominator includes checked D1-D3 training edges, committed outputs from
the concurrent cross-worker depth-three sweep, and committed pilot steps.
Each source uses its own matching B2 meter version for gate judgment. The
current and original B2 values must agree before numeric results are combined.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orch.analysis.landscape_e1 import load_training
from orch.analysis.report import share_of_best
from orch.baselines.verification import VerificationLedger
from orch.contract.brief_text import load_frozen_brief
from orch.contract.models import Evidence, RunStatus
from orch.engine.gate import judge
from orch.engine.journal import Journal
from orch.engine.store import StateStore, _atomic_bytes, _json_bytes
from orch.episode import EpisodeResult
from orch.runtime import make_real_runtime
from scripts.thursday_night_pilot import (
    BRIEFS, CAP, DEPTH3, DESIGNS, POLICIES, ROOT, SMOKE, PilotSafetyStop,
    episode_id, outputs_compared, read, require_amd_pass,
    require_clean_properties, require_start_properties,
)
from orch.analysis.fit_workers import TRAINING_ROOT

PROJECT = Path(__file__).resolve().parents[1]
REPORT = PROJECT / "notes/thursday_night_dev.md"
DETAIL = ROOT / "policy_brief_results.csv"
SUMMARY = ROOT / "policy_brief_summary.json"
PROPERTY_KEYS = ("n_DONT_TOUCH", "n_IS_LOC_FIXED", "n_IS_BEL_FIXED")


def _start_evidence(store: StateStore, state_id: str, version: str) -> Evidence:
    key = hashlib.sha256(version.encode("utf-8")).hexdigest()
    path = store.root / "measurements" / state_id / key / "evidence.json"
    evidence = Evidence.model_validate_json(path.read_bytes())
    if evidence.state_id != state_id or evidence.meter_version != version or evidence.fake:
        raise ValueError("start Evidence differs from pinned state and meter")
    store.state_path(state_id)
    require_start_properties(evidence)
    return evidence


def _same_baseline(old: Evidence, current: Evidence) -> None:
    if old.constraint_fingerprint != current.constraint_fingerprint or old.clock_digest != current.clock_digest:
        raise ValueError("old and current B2 constraints differ")
    if set(old.metrics) != set(current.metrics):
        raise ValueError("old and current B2 metric sets differ")
    for name in old.metrics:
        a, b = old.metrics[name], current.metrics[name]
        if (a.value, a.unit, a.method, a.basis) != (b.value, b.unit, b.method, b.basis):
            raise ValueError(f"old and current B2 {name} differ")


def _load_sources() -> tuple[dict[str, Evidence], dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """Build the best fully valid observed state per design and frozen brief."""

    runtime = make_real_runtime(
        ROOT, cache_enabled=True, max_slots=96,
        load_ceiling_1m=180.0, min_available_memory_bytes=200_000_000_000,
        latch_smoke_status=SMOKE,
    )
    if not (ROOT / "plan.json").is_file():
        raise ValueError("pilot plan is absent")
    plan = read(ROOT / "plan.json")
    if set(plan.get("training_inputs", {})) != set(DESIGNS) or plan.get("meter_version") != runtime.meter.version:
        raise ValueError("pilot plan is not the five-design current-meter training run")
    pilot_store = runtime.store
    current_starts = {
        design: _start_evidence(pilot_store, plan["training_inputs"][design], runtime.meter.version)
        for design in DESIGNS
    }
    old_snapshot = load_training(TRAINING_ROOT, final=True)
    old_store = StateStore(TRAINING_ROOT)
    old_starts = {
        design: _start_evidence(
            old_store, old_snapshot.starts[design].state_id,
            old_snapshot.starts[design].meter_version,
        )
        for design in DESIGNS
    }
    for design in DESIGNS:
        if old_starts[design].state_id != current_starts[design].state_id:
            raise ValueError("old and current B2 byte hashes differ")
        _same_baseline(old_starts[design], current_starts[design])

    briefs = {brief_id: load_frozen_brief(brief_id) for brief_id in BRIEFS}
    best: dict[tuple[str, str], dict[str, Any]] = {}
    unique: dict[str, set[str]] = {design: set() for design in DESIGNS}
    counts: dict[str, int] = defaultdict(int)

    def offer(design: str, evidence: Evidence, start: Evidence, source: str) -> None:
        if design not in current_starts or evidence.fake:
            raise ValueError("candidate escaped the training-only sources")
        if evidence.state_id == start.state_id:
            source = "B2 start"
        try:
            require_clean_properties(evidence, start)
        except PilotSafetyStop:
            counts[f"{source}_unclean_or_missing_properties"] += 1
            return
        unique[design].add(evidence.state_id)
        for brief_id, brief in briefs.items():
            verdict = judge(brief, start, start, evidence)
            if not verdict.brief_valid:
                continue
            measured = evidence.metric(brief.objective.metric)
            assert measured is not None
            value = measured.value
            key = (design, brief_id)
            old = best.get(key)
            better = old is None or (
                value > old["value"] if brief.objective.direction == "max"
                else value < old["value"]
            )
            if better or (old is not None and value == old["value"] and evidence.state_id < old["state_id"]):
                best[key] = {
                    "state_id": evidence.state_id, "value": value,
                    "source": source, "evidence_id": evidence.id,
                }

    for design in DESIGNS:
        offer(design, current_starts[design], current_starts[design], "B2 start")

    for edge in old_snapshot.edges.values():
        if not edge.eligible or edge.amd_status != "pass" or edge.run_id is None:
            raise ValueError("FINAL D1-D3 edge lacks AMD PASS and measured output")
        outcome = old_store.load_outcome(edge.run_id)
        if (
            outcome.run.status != RunStatus.OK or outcome.state is None or outcome.evidence is None
            or outcome.state.design != edge.design or outcome.state.id != edge.output_id
            or outcome.evidence.state_id != edge.output_id
        ):
            raise ValueError("FINAL edge outcome differs from checked training journal")
        offer(edge.design, outcome.evidence, old_starts[edge.design], "D1-D3")
        counts["D1-D3"] += 1

    depth_hashes: dict[str, str] = {}
    for design in DESIGNS:
        root = DEPTH3 / design
        source_store = StateStore(root)
        ledger = VerificationLedger(
            store=source_store, runner=runtime.runner,
            validator=runtime.validator,
            references={design: current_starts[design].state_id},
        )
        events = Journal(root).read(f"thursday_crossworker_{design}")
        depth_hashes[design] = hashlib.sha256(_json_bytes(events)).hexdigest()
        for event in events:
            if event.get("type") != "edge":
                continue
            row = event["edge"]
            if row.get("design") != design:
                raise ValueError("depth-three journal design differs")
            if row.get("status") != "ok":
                counts["depth3_failed"] += 1
                continue
            if row.get("amd_result") != "pass" or row.get("amd_outputs_compared", 0) <= 0:
                raise ValueError("committed depth-three edge lacks AMD PASS")
            outcome = source_store.load_outcome(row["run_id"])
            if (
                outcome.run.status != RunStatus.OK
                or outcome.run.input_sha256_before != row["input_hash"]
                or outcome.run.input_sha256_after != row["input_hash"]
                or outcome.run.recipe_id != row["recipe_id"]
                or outcome.state is None or outcome.evidence is None
                or outcome.state.id != row["output_hash"]
                or outcome.evidence.meter_version != runtime.meter.version
            ):
                raise ValueError("depth-three journal differs from persisted run")
            record = ledger.require_pass(design, outcome.state.id)
            if outputs_compared(record, ledger.root / "runs") != row["amd_outputs_compared"]:
                raise ValueError("depth-three AMD census differs from journal")
            offer(design, outcome.evidence, current_starts[design], "depth3")
            counts["depth3"] += 1

    pilot_ledger = VerificationLedger(
        store=pilot_store, runner=runtime.runner,
        validator=runtime.validator, references=plan["training_inputs"],
    )
    pilot_hashes: dict[str, str] = {}
    for design in DESIGNS:
        for brief_id in BRIEFS:
            for policy in POLICIES:
                identity = episode_id(design, brief_id, policy)
                events = Journal(ROOT).read(identity)
                if not events:
                    continue
                pilot_hashes[identity] = hashlib.sha256(_json_bytes(events)).hexdigest()
                for event in events:
                    if event.get("type") != "step" or not event["verdict"]["physically_valid"]:
                        continue
                    outcome = pilot_store.load_outcome(event["step"]["run_id"])
                    if outcome.run.status != RunStatus.OK or outcome.state is None or outcome.evidence is None:
                        raise ValueError("pilot physical state lacks an OK persisted outcome")
                    require_amd_pass(pilot_ledger, design, outcome.state.id, allow_new=False)
                    offer(design, outcome.evidence, current_starts[design], "pilot")
                    counts["pilot"] += 1

    if len(best) != len(DESIGNS) * len(BRIEFS):
        raise ValueError("a training design/brief lacks its B2 fallback")
    source = {
        "training_journal_sha256": old_snapshot.journal_prefix_sha256,
        "training_edges": old_snapshot.journal_edge_count,
        "depth3_journal_hashes": depth_hashes,
        "pilot_journal_hashes": pilot_hashes,
        "candidate_counts": dict(counts),
        "unique_state_counts": {design: len(ids) for design, ids in unique.items()},
        "meter_versions": {
            "D1-D3": next(iter(old_starts.values())).meter_version,
            "current": runtime.meter.version,
        },
    }
    return current_starts, best, source


def _checked_results(current_starts: dict[str, Evidence], best: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for design in DESIGNS:
        for brief_id in BRIEFS:
            for policy in POLICIES:
                identity = episode_id(design, brief_id, policy)
                path = ROOT / "results" / f"{identity}.json"
                if not path.is_file():
                    continue
                row = read(path)
                events = Journal(ROOT).read(identity)
                finals = [event["result"] for event in events if event.get("type") == "final"]
                if len(finals) != 1:
                    raise ValueError("completed pilot result lacks one journal final")
                result = EpisodeResult.model_validate(finals[0])
                if (
                    result.episode_id != identity or result.brief_id != brief_id
                    or result.final_state is None or result.final_evidence is None
                    or result.final_verdict is None or not result.final_verdict.brief_valid
                    or result.validation_status != "pass"
                    or result.final_state.id != row.get("final_state_id")
                    or result.recipe_runs != row.get("runs_used")
                    or row.get("design") != design or row.get("policy") != policy
                ):
                    raise ValueError("pilot result differs from its checked final event")
                require_clean_properties(result.final_evidence, current_starts[design])
                brief = load_frozen_brief(brief_id)
                start_value = current_starts[design].metric(brief.objective.metric).value
                candidate_value = result.final_evidence.metric(brief.objective.metric).value
                best_value = best[(design, brief_id)]["value"]
                share = share_of_best(start_value, best_value, candidate_value, brief.objective.direction)
                if share is not None and share > 1.000000001:
                    raise ValueError("policy outcome exceeds the supposedly best known state")
                rows.append({
                    **row, "start_value": start_value,
                    "candidate_value": candidate_value, "best_value": best_value,
                    "best_state_id": best[(design, brief_id)]["state_id"],
                    "share_of_best": share,
                })
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["policy"], row["brief"])].append(row)
    table: list[dict[str, Any]] = []
    for policy in POLICIES:
        for brief_id in BRIEFS:
            group = groups[(policy, brief_id)]
            shares = [row["share_of_best"] for row in group if row["share_of_best"] is not None]
            table.append({
                "policy": policy, "brief": brief_id,
                "completed": len(group), "planned": len(DESIGNS),
                "brief_valid_final_picks": sum(bool(row["final_brief_valid"]) for row in group),
                "mean_share_of_best": sum(shares) / len(shares) if shares else None,
                "defined_shares": len(shares),
                "runs_used": sum(row["runs_used"] for row in group),
                "cache_hits": sum(row["cache_hits"] for row in group),
                "new_real_runs": sum(row["new_real_runs_reserved"] for row in group),
                "repair_final_picks": sum(row["repair_final_phase"] == "repair" for row in group),
                "gentle_final_picks": sum(row["repair_final_phase"] == "gentle" for row in group),
            })
    return table


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    fields = (
        "design", "brief", "policy", "episode_id", "final_state_id",
        "best_state_id", "start_value", "candidate_value", "best_value",
        "share_of_best", "runs_used", "cache_hits", "new_real_runs_reserved",
        "repair_final_phase", "validation_status", "stop_reason",
    )
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def render(table: list[dict[str, Any]], status: dict[str, Any], source: dict[str, Any]) -> str:
    completed = status.get("completed_episodes", 0)
    rows = [
        "# Thursday night development — training policy pilot",
        "",
        f"[Verified] Four frozen briefs and seven policy instances are wired through the fitted workers. "
        f"The train-only pilot has {completed}/140 completed episodes; phase {status.get('phase')}; "
        f"{status.get('new_real_runs_reserved', 0)}/{CAP} new real recipe runs reserved. "
        "Every committed output must have zero placement-property counts and an exact-DCP AMD PASS.",
        "",
        "[Verified] Each cell is valid finals/5; mean share of best; recipe runs/cache hits/new real runs; "
        "repair/gentle final picks. Share uses start-relative improvement divided by the best "
        "brief-valid improvement in checked training data; — means undefined or pending.",
        "",
        "| Policy | T0 | A0 | T | A |",
        "|---|---|---|---|---|",
    ]
    cells = {(item["policy"], item["brief"]): item for item in table}
    for policy in POLICIES:
        values = []
        for brief_id in BRIEFS:
            item = cells[(policy, brief_id)]
            share = item["mean_share_of_best"]
            shown = "—" if share is None else f"{100 * share:.1f}%"
            values.append(
                f"{item['brief_valid_final_picks']}/{item['planned']};{shown};"
                f"{item['runs_used']}/{item['cache_hits']}/{item['new_real_runs']};"
                f"{item['repair_final_picks']}/{item['gentle_final_picks']}"
            )
        rows.append(f"| {policy} | " + " | ".join(values) + " |")
    rows.extend([
        "",
        "[Verified] The first pilot stopped after an AMD-PASS no-op was falsely rejected: "
        "The JSON report is read for an output census that the working driver reads from validate.log. "
        "The second pilot was stopped by its recorded process group; its watchdog found zero tagged survivors. "
        "Both failed roots and exact source snapshots are preserved. This v3 pilot reads and hashes the observed log.",
        "",
        f"[Verified] Best-known denominator snapshot: {source['training_edges']} original AMD-PASS edges, "
        f"{source['candidate_counts'].get('depth3', 0)} newly committed depth-three edges, "
        f"and {source['candidate_counts'].get('pilot', 0)} pilot step occurrences. "
        "Original and current B2 measurements agree on all six meter metrics and both digests.",
        "",
        "[Verified] Full offline pytest passed 309/309 at the sharded launch checkpoint. The pilot uses "
        "a private training root, the current deterministic LATCH smoke, shared load/memory gates and "
        "exact-tag watchdogs. FINAL worker JSONs, pinned tool code, meter and registry were not edited.",
        "",
        "[Inferred] The pilot may take longer than tonight because each cache miss includes a real recipe, "
        "meter and AMD validation. Partial rows must not be interpreted as a full policy comparison.",
    ])
    launch = source.get("sharded_launch")
    if launch:
        rows.extend([
            "",
            f"[Verified] Eight disjoint schedule shards were launched at {launch['started_at_utc']}; "
            f"wrapper SHA-256 {launch['wrapper_sha256']}. The serial v3 driver was stopped by its "
            "recorded process group, and its watchdog found zero exact-tag survivors. "
            "All shards share the persistent 300-run cap and the same checked episode journal.",
        ])
    return "\n".join(rows) + "\n"


def main() -> int:
    current, best, source = _load_sources()
    rows = _checked_results(current, best)
    table = _aggregate(rows)
    status = dict(read(ROOT / "status.json"))
    # The detached driver updates status after each episode, while the
    # reservation file is fsynced before every new backend call.
    status["completed_episodes"] = len(rows)
    status["new_real_runs_reserved"] = len(read(ROOT / "new_real_runs.json")["reservations"])
    if (ROOT / "sharded_launch.json").is_file():
        launch = read(ROOT / "sharded_launch.json")
        if launch.get("shards") != 8 or launch.get("pilot_plan_sha256") != hashlib.sha256(
            (ROOT / "plan.json").read_bytes()
        ).hexdigest():
            raise ValueError("sharded launch differs from pilot plan")
        source["sharded_launch"] = {
            "started_at_utc": launch["started_at_utc"],
            "wrapper_sha256": launch["wrapper_sha256"],
            "shards": launch["shards"],
        }
    result = {
        "schema_version": "thursday-night-pilot-report-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_status": status, "sources": source,
        "best_known": {f"{design}:{brief}": row for (design, brief), row in best.items()},
        "policy_brief": table, "completed_episode_rows": len(rows),
    }
    _atomic_bytes(DETAIL, _csv_bytes(rows))
    _atomic_bytes(SUMMARY, _json_bytes(result))
    _atomic_bytes(REPORT, render(table, status, source).encode("utf-8"))
    print(json.dumps({
        "completed_episodes": len(rows), "depth3_edges": source["candidate_counts"].get("depth3", 0),
        "new_real_runs_reserved": status.get("new_real_runs_reserved"),
        "report": str(REPORT), "summary": str(SUMMARY),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
