"""Audited, train-only cache aliases for the restarted Thursday pilot.

Old stores are read-only. Admission requires an exact recipe key, independent
routed Evidence, clean placement properties, saved AMD PASS, and a frozen
meter-equivalence audit. This module starts no hardware or validator jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orch.contract.models import DesignState, Evidence, RunOutcome, RunStatus
from orch.engine.cache import RunCache
from orch.engine.dcp_parser import _parse_utilization_report
from orch.engine.meter_equivalence import (
    CURRENT_METER, MeterEquivalence, canonical_evidence_id, promote_evidence,
)
from orch.engine.store import (
    StateStore, _atomic_bytes, _file_lock, _fsync_dir, _json_bytes, _sha256_file,
)
from orch.recipes.registry import load_registry

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
DEST = PROJECT / "data/pilots/thursday_overnight_train_v4_2026-11-05"
V3 = PROJECT / "data/pilots/thursday_night_train_v3_2026-11-05"
FIT = PROJECT / "data/sweeps/worker_fitting_train"
OLD_DEPTH3_ROOT = PROJECT / "data/sweeps/overnight_priority2_2026-11-04/depth3"
CROSS = PROJECT / "data/sweeps/thursday_crossworker_depth3_2026-11-05"
AUDIT_ROOT = PROJECT / "data/scratch_thursday_night_2026-11-05/training_meter_equivalence"
ROOT = PROJECT / "data/scratch_thursday_night_2026-11-05/training_cache_promotion"
REJECTED_DRAFT = ROOT / "rejected_plan_1260_rows.json"
REJECTED_DRAFT_SHA256 = "564e2d5b28592af4231cabf64057a357a5345c32e0caea758bf2566a405e8bfe"
EXPECTED_AMBIGUOUS_KEYS_SHA256 = "ca8e6ec7f203fb58da86d42ad0520c44c80d0a59b8309b7ad96457f310793942"
CONFLICT_AUDIT = ROOT / "conflict_audit.json"
OLD_FIT = "e11abbe86bd48e7cfd070dd6d9adeda1424517642cd6c1e09c7e29a727947a8f"
OLD_DEPTH3_METER = "2ee8ee9efb9416d84afbef9f1e37350c7b2ea0b054637efd8cff02dbed784435"
DESIGNS = (
    "vexriscv_re-place", "logicnets_jscl", "rosetta_spam-filter",
    "rosetta_3d-rendering", "finn_radioml",
)
RECIPES = {
    "LATCH.phys_opt", "LATCH.retime", "LATCH.auto_pblock",
    "LATCH.hardblock_relocate", "LATCH.memory_path_opt",
    "A_R_EA", "A_R_RSA_SRLM", "A_R_RSA_AREM", "A_R_RSA_MED",
    "A_E_DUPL_Q1000",
}
PROPERTIES = ("n_DONT_TOUCH", "n_IS_LOC_FIXED", "n_IS_BEL_FIXED")
EXPECTED_V3_MISSING_AMD = {
    "353f6ea1e89c8821ce3aef9bdfa754e79336af04598afd8383d965a4f184406a",
    "3d1a3d8a4dd6e49438dfd9d644465e8296fa66ea816a6b43595aff33c7c65a61",
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
RUN_ID = re.compile(r"run_[0-9a-f]{32}\Z")


@dataclass(frozen=True)
class Source:
    """Explicitly allowed training cache, in precedence order."""

    name: str
    root: Path
    meter_version: str
    pointers: int
    plan_name: str


def sources() -> tuple[Source, ...]:
    return (
        Source("stopped_v3", V3, CURRENT_METER, 56, "plan.json"),
        Source("cross_vexriscv_re-place", CROSS / "vexriscv_re-place", CURRENT_METER, 130, "plan.json"),
        Source("cross_logicnets_jscl", CROSS / "logicnets_jscl", CURRENT_METER, 90, "plan.json"),
        Source("cross_rosetta_spam-filter", CROSS / "rosetta_spam-filter", CURRENT_METER, 120, "plan.json"),
        Source("cross_rosetta_3d-rendering", CROSS / "rosetta_3d-rendering", CURRENT_METER, 120, "plan.json"),
        Source("cross_finn_radioml", CROSS / "finn_radioml", CURRENT_METER, 110, "plan.json"),
        Source("fit_D1_D3", FIT, OLD_FIT, 610, "launch_manifest.json"),
        Source("old_vex_depth3", OLD_DEPTH3_ROOT, OLD_DEPTH3_METER, 185, "plan.json"),
    )


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def stable_sha256(value: Any) -> str:
    """Hash a JSON value independently of file formatting or key order."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rejected_draft() -> dict[str, Any]:
    """Require the byte-identical rejected plan before replacing the active draft."""
    if not REJECTED_DRAFT.is_file() or _sha256_file(REJECTED_DRAFT) != REJECTED_DRAFT_SHA256:
        raise ValueError("rejected 1260-row draft is absent or changed")
    draft = read(REJECTED_DRAFT)
    keys = sorted(row["key"] for row in draft.get("conflicts", []))
    if (len(draft.get("rows", [])) != 1260 or len(keys) != 94
            or stable_sha256(keys) != EXPECTED_AMBIGUOUS_KEYS_SHA256):
        raise ValueError("rejected draft does not pin the expected conflict set")
    return draft


def same_output_evidence_checks(
    alternatives: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Reject a reused DCP hash with conflicting current-meter Evidence."""
    checks: list[dict[str, Any]] = []
    for key, rows in sorted(alternatives.items()):
        by_output: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_output[row["output_id"]].append(row)
        for output_id, matches in sorted(by_output.items()):
            if len(matches) < 2:
                continue
            evidence_ids = {row["new_evidence_id"] for row in matches}
            if len(evidence_ids) != 1:
                raise ValueError(f"same-output duplicate Evidence differs: {key} {output_id}")
            checks.append({
                "key": key, "output_id": output_id,
                "current_evidence_id": next(iter(evidence_ids)),
                "sources": sorted(row["source"] for row in matches),
            })
    return checks


def select_unambiguous(
    alternatives: dict[str, list[dict[str, Any]]],
    selected: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Do not choose one DCP byte identity from an ambiguous cache key."""
    conflicts = sorted((
        {
            "key": key,
            "chosen_source": selected[key]["source"] if key in selected else "existing_v4",
            "source_output_hashes": {row["source"]: row["output_id"] for row in rows},
        }
        for key, rows in alternatives.items()
        if len({row["output_id"] for row in rows}) > 1
    ), key=lambda row: row["key"])
    unsafe = {row["key"] for row in conflicts}
    return {key: row for key, row in selected.items() if key not in unsafe}, conflicts


def inside(path: Path, root: Path) -> Path:
    """Reject external paths before touching the named training artifact."""
    if not path.is_absolute() or not path.is_relative_to(root):
        raise ValueError("artifact path escapes named training root")
    actual = path.resolve(strict=True)
    if not actual.is_relative_to(root.resolve(strict=True)) or not actual.is_file():
        raise ValueError("artifact resolves outside named training root")
    return actual


def approvals(audit_sha: str) -> dict[str, MeterEquivalence]:
    """Approve only ten identical training DCP remeasurements and exact tags."""
    if not HEX64.fullmatch(audit_sha):
        raise ValueError("approval needs an exact SHA-256")
    audit_path, plan_path = AUDIT_ROOT / "audit.json", AUDIT_ROOT / "plan.json"
    if _sha256_file(audit_path) != audit_sha:
        raise ValueError("training meter audit SHA-256 differs")
    audit, plan = read(audit_path), read(plan_path)
    versions = audit.get("versions")
    if (
        audit.get("schema_version") != "training-meter-equivalence-audit-v1"
        or audit.get("plan_sha256") != _sha256_file(plan_path)
        or not isinstance(versions, dict) or set(versions) != {OLD_FIT, OLD_DEPTH3_METER}
        or audit.get("normalized_differences") != []
        or audit.get("live_exact_tags") != []
        or audit.get("tagged_runner_jobs") != 20
        or plan.get("current_meter_version", plan.get("current_version")) != CURRENT_METER
    ):
        raise ValueError("training meter audit is not approved")
    for version in (OLD_FIT, OLD_DEPTH3_METER):
        row = versions[version]
        if not isinstance(row, dict) or (
            row.get("samples"), row.get("identical"), row.get("different")
        ) != (5, 5, 0):
            raise ValueError("one old training meter failed comparison")
    return {
        version: MeterEquivalence(version, CURRENT_METER, audit_sha)
        for version in (OLD_FIT, OLD_DEPTH3_METER)
    }


def stopped_v3() -> None:
    """Require the authors' exact-tag stop audit before using v3 as a source."""
    done = (PROJECT / "notes/pilot_stopped.done").read_text().strip()
    record = read(V3 / "pilot_requested_stop_audit.json")
    if not done or record.get("stopped_at_utc") != done or record.get("survivors") != {} or record.get("remaining_recorded_groups") != {}:
        raise ValueError("old pilot is not proven stopped")
    from scripts import thursday_night_pilot as pilot
    from scripts import thursday_night_pilot_shards as shards
    launch = read(V3 / "sharded_launch.json")
    if any(shards._alive(int(pid), int(shard)) for shard, pid in launch["driver_pids"].items()):
        raise ValueError("old pilot shard still runs")
    if pilot._targets()[0]:
        raise ValueError("old pilot still has exact tagged jobs")


def destination() -> tuple[dict[str, Any], str, dict[str, Evidence]]:
    """Verify the frozen v4 plan, selected recipe versions and B2 inputs."""
    frozen = read(DEST / "plan.json")
    if (
        frozen.get("schema_version") != "thursday-overnight-pilot-v4"
        or frozen.get("training_only") is not True
        or frozen.get("data_root") != str(DEST)
        or frozen.get("meter_version") != CURRENT_METER
        or set(frozen.get("training_inputs", {})) != set(DESIGNS)
        or set(frozen.get("recipe_versions", {})) != RECIPES
        or frozen.get("max_new_real_recipe_runs") != 600
    ):
        raise ValueError("v4 plan is not the frozen five-design training pilot")
    hashes = frozen.get("source_hashes")
    expected = {
        "registry": PROJECT / "recipes/registry.yaml",
        "A_fit": PROJECT / "configs/workers/A_fit.json",
        "B_fit": PROJECT / "configs/workers/B_fit.json",
        "briefs": PROJECT / "configs/briefs.yaml",
        "worker_features": PROJECT / "configs/worker_features.yaml",
    }
    if not isinstance(hashes, dict) or any(hashes.get(name) != _sha256_file(path) for name, path in expected.items()):
        raise ValueError("v4 code, registry or FINAL worker hash differs")
    registry = load_registry(PROJECT)
    if any(registry.get(recipe).version != version for recipe, version in frozen["recipe_versions"].items()):
        raise ValueError("v4 recipe versions differ")
    if (DEST / "launch.json").exists() or (DEST / "sharded_launch.json").exists():
        raise ValueError("v4 pilot was launched during cache import")
    store = StateStore(DEST)
    starts: dict[str, Evidence] = {}
    version_key = hashlib.sha256(CURRENT_METER.encode("ascii")).hexdigest()
    for design, state_id in frozen["training_inputs"].items():
        checkpoint = store.state_path(state_id)
        state = DesignState.model_validate_json((checkpoint.parent / "state.json").read_bytes())
        path = DEST / "measurements" / state_id / version_key / "evidence.json"
        evidence = Evidence.model_validate_json(path.read_bytes())
        if (state.design != design or state.parent_id is not None or state.fake
                or evidence.state_id != state_id or evidence.fake
                or evidence.meter_version != CURRENT_METER):
            raise ValueError("v4 B2 start or Evidence differs")
        starts[design] = evidence
    return frozen, _sha256_file(DEST / "plan.json"), starts


class MissingProof(ValueError):
    """Saved source state has no usable AMD output-census PASS."""


def pass_proof(source: Source, design: str, output_id: str, reference: str) -> str:
    """Read the retained source AMD report and nonzero compared-output log."""
    root = source.root
    verifier = read(root / "verification/plan.json")
    if verifier.get("references", {}).get(design) != reference:
        raise ValueError("source AMD reference differs from frozen B2")
    fingerprint = verifier.get("fingerprint")
    if not isinstance(fingerprint, str) or not HEX64.fullmatch(fingerprint):
        raise ValueError("source verifier plan lacks fingerprint")
    path = root / "verification/records" / fingerprint / design / f"{output_id}.json"
    if not path.is_file():
        raise MissingProof("saved AMD PASS record missing")
    record = read(path)
    latest = root / "verification/latest" / design / f"{output_id}.json"
    if (
        record.get("schema_version") != "worker-fitting-equivalence-v1"
        or record.get("design") != design
        or record.get("reference_sha256") != reference
        or record.get("output_sha256") != output_id
        or record.get("validator_fingerprint") != fingerprint
    ):
        raise ValueError("saved AMD record identity differs")
    if record.get("result") != "pass":
        raise MissingProof("saved AMD result is not PASS")
    if (
        record.get("vectors") != 200 or record.get("precheck_vectors") != 100
        or not latest.is_file() or latest.read_bytes() != _json_bytes(record)
    ):
        raise ValueError("saved AMD PASS/latest record differs")
    report = inside(Path(str(record.get("raw_report_path"))), root / "verification/runs")
    if _sha256_file(report) != record.get("raw_report_sha256"):
        raise ValueError("saved AMD raw report hash differs")
    raw = read(report)
    sim = raw.get("simulation_report") or {}
    pre, full = sim.get("precheck_report") or {}, sim.get("full_report") or {}
    if (
        raw.get("overall_passed") is not True
        or raw.get("phase1_passed") is not True
        or raw.get("phase2_passed") is not True
        or raw.get("infrastructure_failure") is not False
        or raw.get("num_vectors") != 200
        or raw.get("precheck_vectors") != 100
        or any(
            part.get("passed") is not True
            or part.get("cycles_simulated") != cycles
            or part.get("mismatch_count") != 0
            or part.get("protocol_mismatch_count") != 0
            for part, cycles in ((pre, 100), (full, 200))
        )
    ):
        raise ValueError("saved AMD raw report does not substantiate PASS")
    attempts = record.get("attempts") or []
    if not attempts:
        raise MissingProof("saved AMD PASS lacks a validator attempt")
    log = inside(Path(str(attempts[-1].get("run_dir"))) / "validate.log",
                 root / "verification/runs")
    counts = [int(x) for x in re.findall(r"Outputs\s*\((\d+)\):",
                                        log.read_text(errors="replace"))]
    if not counts or max(counts) <= 0:
        raise MissingProof("saved AMD PASS compared no outputs")
    return _sha256_file(path)


def local_report(source: Source, run_id: str, evidence: Evidence) -> tuple[str | None, str | None]:
    """Find independent utilization inside one training root only."""
    root = source.root
    candidates = [root / "runs" / run_id / "reports/final_utilization.rpt"]
    raw_dir = Path(evidence.raw_report_path) if evidence.raw_report_path else None
    if raw_dir is not None and raw_dir.is_absolute() and raw_dir.is_relative_to(root):
        candidates.insert(0, raw_dir / "final_utilization.rpt")
    for path in candidates:
        if not path.is_file():
            continue
        report = inside(path, root)
        summary = _parse_utilization_report(report.read_text(errors="replace"))
        packed = summary["utilization_summary"].get("CLB LUTs", {}).get("used")
        measured = evidence.metric("packed_luts")
        if packed is None or measured is None or float(packed) != measured.value:
            raise ValueError("saved utilization disagrees with independent Evidence")
        return str(report.relative_to(root)), _sha256_file(report)
    return None, None


def candidate(source: Source, pointer: Path, frozen: dict[str, Any],
              starts: dict[str, Evidence],
              allowed: dict[str, MeterEquivalence]) -> dict[str, Any]:
    """Validate one source pointer, DCP pair, Evidence and AMD provenance."""
    root, key = source.root, pointer.stem
    entry = read(pointer)
    if entry.get("key") != key or entry.get("retryable") is not False:
        raise ValueError("malformed or retryable source cache pointer")
    cache, store = RunCache(root), StateStore(root)
    run_id = cache.get(key)
    if run_id != entry.get("run_id") or not RUN_ID.fullmatch(run_id):
        raise ValueError("source cache run ID differs")
    original = store.load_outcome(run_id)
    run, state, evidence = original.run, original.state, original.evidence
    if (
        run.status != RunStatus.OK or run.fake or state is None or evidence is None
        or state.fake or evidence.fake or evidence.state_id != state.id
        or state.design not in starts
        or run.input_sha256_before is None
        or run.input_sha256_before != run.input_sha256_after
        or state.parent_id != run.input_sha256_before
        or RunCache.key(run.input_sha256_before, run.recipe_id, run.recipe_version) != key
        or frozen["recipe_versions"].get(run.recipe_id) != run.recipe_version
        or evidence.meter_version != source.meter_version
        or evidence.routed is not True or evidence.route_errors != 0 or evidence.drc_errors != 0
        or evidence.constraint_fingerprint != starts[state.design].constraint_fingerprint
        or evidence.clock_digest != starts[state.design].clock_digest
        or not isinstance(evidence.v_props, dict)
        or any(type(evidence.v_props.get(name)) is not int or evidence.v_props[name] != 0
               for name in PROPERTIES)
    ):
        raise ValueError(f"training source failed key or physical gate: {source.name} {key}")
    store.state_path(run.input_sha256_before)
    store.state_path(state.id)
    approval = allowed.get(source.meter_version)
    if source.meter_version != CURRENT_METER and approval is None:
        raise ValueError("source meter has no approved equivalence")
    promoted = promote_evidence(evidence, approval) if approval else evidence
    if promoted.id != canonical_evidence_id(promoted):
        raise ValueError("source Evidence ID is noncanonical")
    proof = pass_proof(source, state.design, state.id,
                       frozen["training_inputs"][state.design])
    report_rel, report_sha = local_report(source, run_id, evidence)
    return {
        "key": key, "source": source.name, "source_run_id": run_id,
        "source_pointer_sha256": _sha256_file(pointer),
        "source_complete_sha256": _sha256_file(root / "runs" / run_id / "complete.json"),
        "source_amd_record_sha256": proof,
        "design": state.design, "input_id": run.input_sha256_before,
        "output_id": state.id, "recipe_id": run.recipe_id,
        "recipe_version": run.recipe_version, "old_evidence_id": evidence.id,
        "new_evidence_id": promoted.id, "source_meter_version": source.meter_version,
        # Exclude state identity as well, so different DCP bytes can be
        # compared on measured fields without claiming the states equivalent.
        "measurement_content_sha256": stable_sha256(promoted.model_dump(
            mode="json", exclude={"id", "state_id", "raw_report_path"}
        )),
        "utilization_report_relative": report_rel,
        "utilization_report_sha256": report_sha,
    }


def snapshots(specs: tuple[Source, ...]) -> dict[str, Any]:
    """Pin named training stores, never a generic data root."""
    result: dict[str, Any] = {}
    for source in specs:
        if not source.root.is_dir() or not (source.root / "cache").is_dir():
            raise ValueError(f"named training source missing: {source.name}")
        if source.name.startswith("cross_") or source.name == "old_vex_depth3":
            if read(source.root / "status.json").get("phase") != "complete":
                raise ValueError(f"source not complete: {source.name}")
        pointers = list((source.root / "cache").glob("*.json"))
        if len(pointers) != source.pointers:
            raise ValueError(f"source pointer count changed: {source.name}")
        result[source.name] = {
            "root": str(source.root), "meter_version": source.meter_version,
            "pointer_count": len(pointers),
            "plan_sha256": _sha256_file(source.root / source.plan_name),
            "verification_plan_sha256": _sha256_file(source.root / "verification/plan.json"),
        }
    return result


def build_conflict_audit(
    alternatives: dict[str, list[dict[str, Any]]],
    conflicts: list[dict[str, Any]],
    specs: tuple[Source, ...],
    draft: dict[str, Any],
    audit_sha: str,
) -> dict[str, Any]:
    """Record every ambiguous DCP and how each byte identity affects lineage."""
    outputs = {
        row["output_id"] for conflict in conflicts
        for row in alternatives[conflict["key"]]
    }
    descendants: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for source in specs:
        for path in (source.root / "runs").glob("*/output_state.json"):
            state = read(path)
            if state.get("parent_id") in outputs:
                descendants[state["parent_id"]].append((source.name, state["id"]))
    telemetry = PROJECT / "data/exp_evening_2026-11-04/training_telemetry_v3/states"
    draft_choices = {row["key"]: row["chosen_source"] for row in draft["conflicts"]}
    records: list[dict[str, Any]] = []
    for conflict in conflicts:
        key = conflict["key"]
        rows = alternatives[key]
        if len({(row["design"], row["input_id"], row["recipe_id"],
                 row["recipe_version"]) for row in rows}) != 1:
            raise ValueError(f"ambiguous key carries different recipe or input: {key}")
        if len({row["measurement_content_sha256"] for row in rows}) != 1:
            raise ValueError(f"ambiguous DCPs have different measured content: {key}")
        candidates = []
        for row in sorted(rows, key=lambda item: item["source"]):
            edges = sorted(descendants[row["output_id"]])
            telemetry_path = telemetry / row["output_id"] / "dcp_telemetry.json"
            candidates.append({
                "source": row["source"], "output_id": row["output_id"],
                "source_pointer_sha256": row["source_pointer_sha256"],
                "source_complete_sha256": row["source_complete_sha256"],
                "source_amd_record_sha256": row["source_amd_record_sha256"],
                "new_evidence_id": row["new_evidence_id"],
                "measurement_content_sha256": row["measurement_content_sha256"],
                "descendant_count": len(edges),
                "descendant_set_sha256": stable_sha256(edges),
                "training_v3_telemetry_sha256": (
                    _sha256_file(telemetry_path) if telemetry_path.is_file() else None
                ),
            })
        records.append({
            "key": key, "design": rows[0]["design"],
            "input_id": rows[0]["input_id"], "recipe_id": rows[0]["recipe_id"],
            "draft_chosen_source": draft_choices[key],
            "source_output_hashes": conflict["source_output_hashes"],
            "candidates": candidates,
        })
    b2_inputs = set(read(DEST / "plan.json")["training_inputs"].values())
    first_step = {
        row["key"] for rows in alternatives.values() for row in rows
        if row["input_id"] in b2_inputs
    }
    differing_lineage = sum(
        len({candidate["descendant_set_sha256"] for candidate in row["candidates"]}) > 1
        for row in records
    )
    differing_presence = sum(
        len({candidate["descendant_count"] > 0 for candidate in row["candidates"]}) > 1
        for row in records
    )
    draft_loses_lineage = sum(
        not any(candidate["descendant_count"] for candidate in row["candidates"]
                if candidate["source"] == row["draft_chosen_source"])
        and any(candidate["descendant_count"] for candidate in row["candidates"]
                if candidate["source"] != row["draft_chosen_source"])
        for row in records
    )
    return {
        "schema_version": "training-cache-conflict-audit-v1",
        "rejected_draft_sha256": REJECTED_DRAFT_SHA256,
        "approval_audit_sha256": audit_sha,
        "source_script_sha256": _sha256_file(SCRIPT),
        "ambiguous_keys_excluded": len(records),
        "ambiguous_key_set_sha256": stable_sha256(sorted(row["key"] for row in records)),
        "b2_first_step_ambiguous": sum(row["key"] in first_step for row in records),
        "differing_descendant_sets": differing_lineage,
        "differing_descendant_presence": differing_presence,
        "draft_choice_loses_existing_descendants": draft_loses_lineage,
        "conflicts": records,
        "tool_jobs_launched": 0,
    }


def build_plan(audit_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze only unambiguous source aliases and separately hash exclusions."""
    draft = rejected_draft()
    allowed = approvals(audit_sha)
    stopped_v3()
    frozen, pilot_sha, starts = destination()
    specs = sources()
    source_pins = snapshots(specs)
    cache, store = RunCache(DEST), StateStore(DEST)
    existing: dict[str, str] = {}
    for pointer in sorted(cache.cache_dir.glob("*.json")):
        run_id = cache.get(pointer.stem)
        outcome = store.load_outcome(run_id)
        if outcome.evidence is None or outcome.evidence.meter_version != CURRENT_METER:
            raise ValueError("existing v4 cache pointer is not current")
        existing[pointer.stem] = _sha256_file(pointer)
    selected: dict[str, dict[str, Any]] = {}
    alternatives: dict[str, list[dict[str, Any]]] = defaultdict(list)
    omitted: list[dict[str, str]] = []
    for source in specs:
        for pointer in sorted((source.root / "cache").glob("*.json")):
            try:
                row = candidate(source, pointer, frozen, starts, allowed)
            except MissingProof as exc:
                if source.name != "stopped_v3" or pointer.stem not in EXPECTED_V3_MISSING_AMD:
                    raise ValueError("unexpected missing AMD proof in training source") from exc
                omitted.append({"source": source.name, "key": pointer.stem, "reason": str(exc)})
                continue
            alternatives[row["key"]].append(row)
            if row["key"] not in existing and row["key"] not in selected:
                selected[row["key"]] = row
    if {row["key"] for row in omitted} != EXPECTED_V3_MISSING_AMD:
        raise ValueError("interrupted v3 AMD omission key set changed")
    duplicate_evidence_checks = same_output_evidence_checks(alternatives)
    selected, conflicts = select_unambiguous(alternatives, selected)
    ambiguous_keys = sorted(row["key"] for row in conflicts)
    if len(ambiguous_keys) != 94 or stable_sha256(ambiguous_keys) != EXPECTED_AMBIGUOUS_KEYS_SHA256:
        raise ValueError("ambiguous training cache key set changed")
    draft_hashes = {row["key"]: row["source_output_hashes"] for row in draft["conflicts"]}
    if {row["key"]: row["source_output_hashes"] for row in conflicts} != draft_hashes:
        raise ValueError("ambiguous source output hashes differ from rejected draft")
    conflict_audit = build_conflict_audit(alternatives, conflicts, specs, draft, audit_sha)
    for row in selected.values():
        row["alias_run_id"] = "run_" + hashlib.sha256(
            (row["key"] + row["source"] + audit_sha).encode("ascii")
        ).hexdigest()[:32]
    results = {
        path.name: _sha256_file(path)
        for path in sorted((DEST / "results").glob("*.json"))
    } if (DEST / "results").is_dir() else {}
    plan = {
        "schema_version": "training-cache-promotion-plan-v1",
        "created_at_utc": utc(), "destination": str(DEST),
        "pilot_plan_sha256": pilot_sha, "audit_sha256": audit_sha,
        "source_script_sha256": _sha256_file(SCRIPT),
        "rejected_draft_sha256": REJECTED_DRAFT_SHA256,
        "conflict_audit_sha256": hashlib.sha256(_json_bytes(conflict_audit)).hexdigest(),
        "ambiguous_keys_excluded": len(ambiguous_keys),
        "ambiguous_key_set_sha256": EXPECTED_AMBIGUOUS_KEYS_SHA256,
        "omitted_ambiguous_keys": ambiguous_keys,
        "duplicate_same_output_evidence_checks": duplicate_evidence_checks,
        "source_snapshots": source_pins,
        "existing_pointers": existing, "existing_results": results,
        "rows": [selected[key] for key in sorted(selected)],
        "conflicts": conflicts, "omitted_missing_amd_proof": omitted,
        "source_candidate_counts": {source.name: source.pointers for source in specs},
        "tool_jobs_launched": 0,
    }
    return plan, conflict_audit


def check_plan(plan: dict[str, Any], *, deep: bool = False) -> tuple[dict[str, MeterEquivalence], dict[str, Source]]:
    """Recheck the exact audit, v4 code/registry/models and named source plans."""
    if plan.get("schema_version") != "training-cache-promotion-plan-v1" or plan.get("destination") != str(DEST):
        raise ValueError("alias plan targets a different root")
    rejected_draft()
    ambiguous = plan.get("omitted_ambiguous_keys")
    if (not isinstance(ambiguous, list) or len(ambiguous) != 94
            or ambiguous != sorted(set(ambiguous))
            or stable_sha256(ambiguous) != EXPECTED_AMBIGUOUS_KEYS_SHA256
            or plan.get("ambiguous_keys_excluded") != 94
            or plan.get("ambiguous_key_set_sha256") != EXPECTED_AMBIGUOUS_KEYS_SHA256
            or plan.get("rejected_draft_sha256") != REJECTED_DRAFT_SHA256
            or any(row["key"] in set(ambiguous) for row in plan.get("rows", []))):
        raise ValueError("ambiguous DCP aliases were not excluded exactly")
    if (not CONFLICT_AUDIT.is_file()
            or _sha256_file(CONFLICT_AUDIT) != plan.get("conflict_audit_sha256")):
        raise ValueError("independent conflict audit is absent or changed")
    conflict_audit = read(CONFLICT_AUDIT)
    if (
        conflict_audit.get("schema_version") != "training-cache-conflict-audit-v1"
        or conflict_audit.get("rejected_draft_sha256") != REJECTED_DRAFT_SHA256
        or conflict_audit.get("ambiguous_keys_excluded") != 94
        or conflict_audit.get("ambiguous_key_set_sha256") != EXPECTED_AMBIGUOUS_KEYS_SHA256
        or conflict_audit.get("source_script_sha256") != plan.get("source_script_sha256")
        or conflict_audit.get("approval_audit_sha256") != plan.get("audit_sha256")
        or [row["key"] for row in conflict_audit.get("conflicts", [])] != ambiguous
        or {row["key"]: row["source_output_hashes"] for row in conflict_audit["conflicts"]}
        != {row["key"]: row["source_output_hashes"] for row in plan.get("conflicts", [])}
    ):
        raise ValueError("conflict audit does not match excluded plan keys")
    allowed = approvals(plan["audit_sha256"])
    stopped_v3()
    _, pilot_sha, _ = destination()
    if pilot_sha != plan.get("pilot_plan_sha256") or _sha256_file(SCRIPT) != plan.get("source_script_sha256"):
        raise ValueError("alias script or v4 plan changed")
    specs = {source.name: source for source in sources()}
    if snapshots(tuple(specs.values())) != plan.get("source_snapshots"):
        raise ValueError("training source plans changed")
    if deep:
        # Recheck every same-output duplicate at the migration boundary. A
        # shared DCP hash with different Evidence would make alias selection
        # dependent on source precedence even without a DCP-byte conflict.
        for check in plan.get("duplicate_same_output_evidence_checks", []):
            for name in check["sources"]:
                source = specs[name]
                run_id = RunCache(source.root).get(check["key"])
                original = StateStore(source.root).load_outcome(run_id)
                if original.state is None or original.evidence is None:
                    raise ValueError("same-output duplicate lost its saved outcome")
                approval = allowed.get(source.meter_version)
                evidence = promote_evidence(original.evidence, approval) if approval else original.evidence
                if (original.state.id != check["output_id"]
                        or evidence.id != check["current_evidence_id"]):
                    raise ValueError("same-output duplicate Evidence changed")
        for conflict in conflict_audit["conflicts"]:
            for candidate_record in conflict["candidates"]:
                source = specs[candidate_record["source"]]
                pointer = source.root / "cache" / f"{conflict['key']}.json"
                run_id = RunCache(source.root).get(conflict["key"])
                if (_sha256_file(pointer) != candidate_record["source_pointer_sha256"]
                        or _sha256_file(source.root / "runs" / run_id / "complete.json")
                        != candidate_record["source_complete_sha256"]):
                    raise ValueError("excluded ambiguous source changed")
    for name, digest in plan.get("existing_results", {}).items():
        if _sha256_file(DEST / "results" / name) != digest:
            raise ValueError("existing v4 result changed")
    return allowed, specs


def link_file(source: Path, target: Path, digest: str) -> None:
    """Publish a checked immutable hardlink; never modify the source inode."""
    if source.is_symlink() or _sha256_file(source) != digest:
        raise ValueError("training source artifact changed")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if target.is_symlink() or _sha256_file(target) != digest:
            raise ValueError("destination artifact differs")
        return
    temporary = target.with_name(f".{target.name}.{os.getpid()}.link")
    try:
        os.link(source, temporary)
        if _sha256_file(temporary) != digest:
            raise ValueError("hardlinked artifact hash differs")
        os.replace(temporary, target)
        _fsync_dir(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def link_state(source_store: StateStore, target_store: StateStore,
               state_id: str, design: str) -> Path:
    """Register input and output DCPs in v4 with per-state OS locks."""
    source = source_store.state_path(state_id)
    metadata = DesignState.model_validate_json((source.parent / "state.json").read_bytes())
    if metadata.id != state_id or metadata.design != design or metadata.fake:
        raise ValueError("source state metadata differs")
    target = target_store.states_dir / state_id / "design.dcp"
    with _file_lock(target_store.states_dir / f"{state_id}.lock"):
        link_file(source, target, state_id)
        meta_path = target.parent / "state.json"
        if meta_path.is_file():
            saved = DesignState.model_validate_json(meta_path.read_bytes())
            if (
                saved.id != state_id or saved.design != design or saved.fake
                or saved.path != str(target)
            ):
                raise ValueError("v4 state metadata differs")
        else:
            relinked = metadata.model_copy(update={"path": str(target)})
            _atomic_bytes(meta_path, _json_bytes(relinked.model_dump(mode="json")))
    return target_store.state_path(state_id)


def alias_outcome(original: RunOutcome, target_state: DesignState,
                  approval: MeterEquivalence | None, alias_id: str) -> RunOutcome:
    """Keep source cost/provenance and mark the new outcome as a cache alias."""
    if original.evidence is None:
        raise ValueError("source run lacks Evidence")
    evidence = promote_evidence(original.evidence, approval) if approval else original.evidence
    run = original.run.model_copy(update={
        "id": alias_id, "cache_hit": True, "original_run_id": original.run.id,
        "attempts": (), "process_group_kills": 0, "tag_sweep_kills": 0,
    })
    claims = original.claims
    if claims is not None:
        claims = claims.model_copy(update={"id": f"{alias_id}:claims", "run_id": alias_id})
    return RunOutcome(run=run, state=target_state, evidence=evidence, claims=claims)


def copy_one(row: dict[str, Any], source: Source, target_store: StateStore,
             target_cache: RunCache, approval: MeterEquivalence | None,
             audit_sha: str) -> str:
    """Publish one checked alias, with the cache pointer committed last."""
    key = row["key"]
    source_store, source_cache = StateStore(source.root), RunCache(source.root)
    pointer = source_cache.cache_dir / f"{key}.json"
    if (
        _sha256_file(pointer) != row["source_pointer_sha256"]
        or source_cache.get(key) != row["source_run_id"]
        or _sha256_file(source.root / "runs" / row["source_run_id"] / "complete.json")
        != row["source_complete_sha256"]
    ):
        raise ValueError("source cache pointer or completed run changed")
    original = source_store.load_outcome(row["source_run_id"])
    if original.state is None or original.evidence is None or (
        original.state.id, original.run.input_sha256_before, original.evidence.id
    ) != (row["output_id"], row["input_id"], row["old_evidence_id"]):
        raise ValueError("source run identity changed")
    reference = read(DEST / "plan.json")["training_inputs"][row["design"]]
    if pass_proof(source, row["design"], row["output_id"], reference) != row["source_amd_record_sha256"]:
        raise ValueError("saved AMD proof changed")
    report_rel, report_sha = local_report(source, row["source_run_id"], original.evidence)
    if (report_rel, report_sha) != (
        row["utilization_report_relative"], row["utilization_report_sha256"]
    ):
        raise ValueError("source utilization report changed")
    with target_cache.locked(key):
        current = target_cache.get(key)
        if current is not None:
            if current != row["alias_run_id"]:
                raise ValueError("v4 cache pointer changed during promotion")
            alias = target_store.load_outcome(current)
            if alias.evidence is None or alias.evidence.id != row["new_evidence_id"]:
                raise ValueError("existing alias Evidence differs")
            folder = target_store.runs_dir / current
            sidecar = read(folder / "training_cache_alias.json")
            if (
                sidecar.get("audit_sha256") != audit_sha
                or sidecar.get("source") != source.name
                or sidecar.get("cache_key") != key
                or sidecar.get("source_run_id") != row["source_run_id"]
                or sidecar.get("source_amd_record_sha256") != row["source_amd_record_sha256"]
                or sidecar.get("utilization_report_sha256") != report_sha
            ):
                raise ValueError("existing alias provenance differs")
            linked_report = folder / "reports/final_utilization.rpt"
            if report_sha is not None:
                if not linked_report.is_file() or _sha256_file(linked_report) != report_sha:
                    raise ValueError("existing alias utilization report differs")
            elif linked_report.exists():
                raise ValueError("unexpected alias utilization report")
            return "already"
        link_state(source_store, target_store, row["input_id"], row["design"])
        output_path = link_state(source_store, target_store, row["output_id"], row["design"])
        target_state = original.state.model_copy(update={"path": str(output_path)})
        alias = alias_outcome(original, target_state, approval, row["alias_run_id"])
        if alias.evidence is None or alias.evidence.id != row["new_evidence_id"]:
            raise ValueError("alias Evidence differs from frozen plan")
        folder = target_store.runs_dir / row["alias_run_id"]
        folder.mkdir(parents=True, exist_ok=True)
        sidecar = {
            "schema_version": "training-cache-alias-v1",
            "audit_sha256": audit_sha, "source": row["source"],
            "source_root": str(source.root), "source_run_id": row["source_run_id"],
            "source_complete_sha256": row["source_complete_sha256"],
            "source_amd_record_sha256": row["source_amd_record_sha256"],
            "source_evidence_id": row["old_evidence_id"],
            "alias_evidence_id": row["new_evidence_id"],
            "input_id": row["input_id"], "output_id": row["output_id"],
            "cache_key": key,
            "utilization_report_relative": report_rel,
            "utilization_report_sha256": report_sha,
            "missing_local_utilization_report": report_rel is None,
        }
        sidecar_path = folder / "training_cache_alias.json"
        content = _json_bytes(sidecar)
        if sidecar_path.is_file() and sidecar_path.read_bytes() != content:
            raise ValueError("alias provenance sidecar changed")
        _atomic_bytes(sidecar_path, content)
        if report_rel is not None:
            report = inside(source.root / report_rel, source.root)
            link_file(report, folder / "reports/final_utilization.rpt", report_sha)
        target_store.save_outcome(alias)
        target_cache.put(key, alias.run.id)
        return "imported"


def status(phase: str, planned: int, done: int, error: str | None = None) -> None:
    _atomic_bytes(ROOT / "status.json", _json_bytes({
        "schema_version": "training-cache-promotion-status-v1",
        "phase": phase, "planned": planned, "imported_or_already": done,
        "error": error, "updated_at_utc": utc(), "tool_jobs_launched": 0,
    }))


def migrate(plan: dict[str, Any]) -> dict[str, Any]:
    """Import frozen aliases idempotently; source stores remain read-only."""
    allowed, specs = check_plan(plan, deep=True)
    target_store, target_cache = StateStore(DEST), RunCache(DEST)
    for key, digest in plan["existing_pointers"].items():
        if _sha256_file(target_cache.cache_dir / f"{key}.json") != digest:
            raise ValueError("preexisting v4 pointer changed")
    count = 0
    status("running", len(plan["rows"]), 0)
    for row in plan["rows"]:
        if count % 100 == 0:
            check_plan(plan)
        source = specs[row["source"]]
        copy_one(row, source, target_store, target_cache,
                 allowed.get(source.meter_version), plan["audit_sha256"])
        count += 1
        if count % 25 == 0 or count == len(plan["rows"]):
            status("running", len(plan["rows"]), count)
    for row in plan["rows"]:
        if target_cache.get(row["key"]) != row["alias_run_id"]:
            raise ValueError("imported training pointer missing")
        alias = target_store.load_outcome(row["alias_run_id"])
        if alias.evidence is None or alias.evidence.meter_version != CURRENT_METER:
            raise ValueError("imported Evidence is not current")
        original = StateStore(specs[row["source"]].root).load_outcome(row["source_run_id"])
        for field in (
            "wall_s", "cpu_user_s", "cpu_sys_s", "peak_memory_bytes",
            "load_1m_start", "load_1m_end", "busy_slots_start",
            "backpressure_wait_s", "backpressure_events",
        ):
            if getattr(alias.run, field) != getattr(original.run, field):
                raise ValueError("alias changed source cost")
    for key, digest in plan["existing_pointers"].items():
        if _sha256_file(target_cache.cache_dir / f"{key}.json") != digest:
            raise ValueError("preexisting v4 pointer was rewritten")
    audit = {
        "schema_version": "training-cache-promotion-audit-v1",
        "checked_at_utc": utc(), "plan_sha256": _sha256_file(ROOT / "plan.json"),
        "approval_audit_sha256": plan["audit_sha256"],
        "approved_versions": [OLD_FIT, OLD_DEPTH3_METER],
        "destination_root": str(DEST),
        "pilot_plan_sha256": plan["pilot_plan_sha256"],
        "promoted_pointers": count,
        "source_candidate_counts": plan["source_candidate_counts"],
        "existing_v4_pointers": len(plan["existing_pointers"]),
        "imported_pointers": count,
        "conflicting_output_hash_keys": len(plan["conflicts"]),
        "ambiguous_keys_excluded": len(plan["omitted_ambiguous_keys"]),
        "omitted_ambiguous_keys": plan["omitted_ambiguous_keys"],
        "ambiguous_key_set_sha256": plan["ambiguous_key_set_sha256"],
        "conflict_audit_sha256": plan["conflict_audit_sha256"],
        "rejected_draft_sha256": plan["rejected_draft_sha256"],
        "same_output_duplicate_evidence_groups_checked": len(
            plan["duplicate_same_output_evidence_checks"]
        ),
        "omitted_missing_amd_proof": len(plan["omitted_missing_amd_proof"]),
        "omitted_amd_keys": sorted(row["key"] for row in plan["omitted_missing_amd_proof"]),
        "missing_local_utilization_reports": sum(
            row["utilization_report_relative"] is None for row in plan["rows"]
        ),
        "cost_fields_equal_to_source": True,
        "preexisting_v4_pointers_unchanged": True,
        "tool_jobs_launched": 0,
    }
    _atomic_bytes(ROOT / "audit.json", _json_bytes(audit))
    status("complete", len(plan["rows"]), count)
    return audit


def launch_plan(audit_sha: str) -> dict[str, Any]:
    """Build the read-only source plan in its own recorded process group."""
    rejected_draft()
    approvals(audit_sha)
    stopped_v3()
    destination()
    ROOT.mkdir(parents=True, exist_ok=True)
    with _file_lock(ROOT / "plan_launch.lock"):
        manifest = ROOT / "replan_launch.json"
        active = ROOT / "plan.json"
        if manifest.is_file() or not active.is_file() or _sha256_file(active) != REJECTED_DRAFT_SHA256:
            raise FileExistsError("rejected draft changed or safe replan already launched")
        with (ROOT / "replan_driver.stdout").open("wb") as out, (
            ROOT / "replan_driver.stderr"
        ).open("wb") as err:
            driver = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--plan", "--approval-sha256", audit_sha],
                cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                start_new_session=True, close_fds=True,
            )
        result = {
            "schema_version": "training-cache-replan-launch-v1",
            "driver_pid": driver.pid, "driver_pgid": driver.pid,
            "started_at_utc": utc(), "approval_audit_sha256": audit_sha,
            "source_script_sha256": _sha256_file(SCRIPT),
            "rejected_draft_sha256": REJECTED_DRAFT_SHA256,
            "v4_plan_sha256": _sha256_file(DEST / "plan.json"),
            "tool_jobs_launched": 0,
        }
        _atomic_bytes(manifest, _json_bytes(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "plan", "plan-launch", "run", "launch", "status"):
        mode.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--approval-sha256")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(read(ROOT / "status.json") if (ROOT / "status.json").is_file()
                         else {"phase": "not_started"}, sort_keys=True))
        return 0
    if args.plan_launch:
        if not args.approval_sha256:
            raise ValueError("exact meter audit SHA-256 is required")
        print(json.dumps(launch_plan(args.approval_sha256), sort_keys=True))
        return 0
    if args.dry_run or args.plan:
        if not args.approval_sha256:
            raise ValueError("exact meter audit SHA-256 is required")
        if args.plan:
            status("planning", 0, 0)
        try:
            plan, conflict_audit = build_plan(args.approval_sha256)
        except Exception as exc:
            if args.plan:
                status("plan_failed", 0, 0, f"{type(exc).__name__}: {exc}")
            raise
        if args.plan:
            ROOT.mkdir(parents=True, exist_ok=True)
            with _file_lock(ROOT / "plan.lock"):
                rejected_draft()
                audit_content = _json_bytes(conflict_audit)
                if hashlib.sha256(audit_content).hexdigest() != plan["conflict_audit_sha256"]:
                    raise ValueError("conflict audit content hash differs from safe plan")
                if CONFLICT_AUDIT.is_file() and CONFLICT_AUDIT.read_bytes() != audit_content:
                    raise ValueError("frozen conflict audit differs")
                if not CONFLICT_AUDIT.is_file():
                    _atomic_bytes(CONFLICT_AUDIT, audit_content)
                path, content = ROOT / "plan.json", _json_bytes(plan)
                if path.is_file() and _sha256_file(path) == REJECTED_DRAFT_SHA256:
                    # The old draft survives byte-identically at REJECTED_DRAFT.
                    _atomic_bytes(path, content)
                elif path.is_file() and path.read_bytes() != content:
                    raise ValueError("frozen safe alias plan differs")
                elif not path.is_file():
                    _atomic_bytes(path, content)
            check_plan(plan, deep=True)
            status("plan_ready", len(plan["rows"]), 0)
        print(json.dumps({
            "selected_aliases": len(plan["rows"]),
            "source_candidates": plan["source_candidate_counts"],
            "existing_v4": len(plan["existing_pointers"]),
            "conflicting_output_hash_keys": len(plan["conflicts"]),
            "ambiguous_keys_excluded": len(plan["omitted_ambiguous_keys"]),
            "conflict_audit_sha256": plan["conflict_audit_sha256"],
            "same_output_duplicate_evidence_groups_checked": len(
                plan["duplicate_same_output_evidence_checks"]
            ),
            "omitted_missing_amd_proof": len(plan["omitted_missing_amd_proof"]),
            "missing_local_utilization_reports": sum(
                row["utilization_report_relative"] is None for row in plan["rows"]
            ),
            "tool_jobs_launched": 0,
        }, sort_keys=True))
        return 0
    if args.run:
        with _file_lock(ROOT / "driver.lock"):
            plan = read(ROOT / "plan.json")
            if (ROOT / "launch.json").is_file() and _sha256_file(ROOT / "plan.json") != read(ROOT / "launch.json").get("plan_sha256"):
                raise ValueError("frozen alias plan changed after launch")
            try:
                migrate(plan)
            except Exception as exc:
                status("stopped", len(plan["rows"]),
                       read(ROOT / "status.json").get("imported_or_already", 0),
                       f"{type(exc).__name__}: {exc}")
                raise
        return 0
    plan = read(ROOT / "plan.json")
    check_plan(plan, deep=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    with _file_lock(ROOT / "launch.lock"):
        if (ROOT / "launch.json").is_file():
            raise FileExistsError("training alias import was already launched")
        with (ROOT / "driver.stdout").open("wb") as out, (ROOT / "driver.stderr").open("wb") as err:
            driver = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--run"], cwd=PROJECT,
                stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                start_new_session=True, close_fds=True,
            )
        launch = {
            "schema_version": "training-cache-promotion-launch-v1",
            "driver_pid": driver.pid, "started_at_utc": utc(),
            "plan_sha256": _sha256_file(ROOT / "plan.json"),
            "source_script_sha256": _sha256_file(SCRIPT),
            "tool_jobs_launched": 0,
        }
        _atomic_bytes(ROOT / "launch.json", _json_bytes(launch))
    print(json.dumps(launch, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
