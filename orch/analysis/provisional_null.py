"""Training-only provisional beta rankings from saved meter Evidence.

This never launches tools. All output is PROVISIONAL, even when a saved AMD
check passes. It reads only the explicit five-design training data root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import yaml

from orch.contract.models import Evidence
from orch.engine.journal import Journal

PROJECT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = PROJECT / "data/sweeps/worker_fitting_train"
RECIPE_SETS = PROJECT / "configs/recipe_sets.yaml"
FEATURE_SPEC = PROJECT / "configs/worker_features.yaml"
NOTE = PROJECT / "notes/provisional_null_ranking.md"
NULL_CONFIGS = {
    "timing": PROJECT / "configs/workers/A_null_PROVISIONAL.json",
    "area": PROJECT / "configs/workers/B_null_PROVISIONAL.json",
}
TRAIN_DESIGNS = (
    "vexriscv_re-place", "rosetta_spam-filter", "rosetta_3d-rendering",
    "logicnets_jscl", "finn_radioml",
)
# notes/training_data_audit.md:36 checked the first Requirement line in
# all 708 evidence-linked meter reports against each B2 report.
REQUIREMENT_NS = {
    "vexriscv_re-place": 1.570,
    "rosetta_spam-filter": 1.600,
    "rosetta_3d-rendering": 1.538,
    "logicnets_jscl": 1.500,
    "finn_radioml": 1.600,
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SAFE_RUN = re.compile(r"run_[0-9a-f]{32}\Z")


@dataclass(frozen=True)
class Observation:
    """One recipe's realised gain on one input checkpoint."""

    design: str
    sequence: tuple[str, ...]
    input_state_id: str
    recipe_id: str
    output_state_id: str | None
    gain_pct: float
    zero_reason: str | None
    verification: str


def require_training_root(data_root: Path | str | None = None) -> Path:
    """Refuse any data root except the fixed five-design training sweep."""
    expected = TRAIN_ROOT.resolve()
    proposed = TRAIN_ROOT if data_root is None else Path(data_root).expanduser()
    if not proposed.is_absolute():
        proposed = PROJECT / proposed
    actual = proposed.resolve()
    if actual != expected:
        raise ValueError("refusing data root outside the training allow-list")
    if not actual.is_dir():
        raise FileNotFoundError(actual)
    return actual


def _hex(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def _read_json(root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("training record escaped its allow-listed root")
    content = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"expected JSON object: {path}")
    return content


def _metric(evidence: Evidence, name: str) -> float:
    measured = evidence.metric(name)
    if measured is None or not math.isfinite(measured.value):
        raise ValueError(f"meter Evidence lacks finite {name}: {evidence.id}")
    return measured.value


def gain_pct(
    worker: str, parent: Evidence, child: Evidence, requirement_ns: float
) -> float:
    """Apply the frozen meter-only target formula to a valid changed edge."""
    if worker == "timing":
        parent_period = requirement_ns - _metric(parent, "wns_ns")
        child_period = requirement_ns - _metric(child, "wns_ns")
        if parent_period <= 0 or child_period <= 0:
            raise ValueError("nonpositive Fmax period")
        return 100.0 * (parent_period / child_period - 1.0)
    if worker == "area":
        before = _metric(parent, "packed_luts")
        after = _metric(child, "packed_luts")
        if before <= 0:
            raise ValueError("nonpositive parent packed LUT count")
        return 100.0 * (before - after) / before
    raise ValueError(f"unknown worker: {worker}")


def zero_gain_reason(
    run_status: str | None, physically_valid: bool, input_state_id: str,
    output_state_id: str | None, verification: str,
) -> str | None:
    """Classify the frozen zero-gain outcomes before reading child metrics."""
    if run_status != "ok" or not physically_valid:
        return "invalid_or_failed"
    if output_state_id is None:
        return "no_output"
    if output_state_id == input_state_id:
        return "unchanged_output"
    if verification in {"fail", "unknown", "no_record"}:
        return "amd_nonpass"
    return None


def deduplicate(rows: Sequence[Observation]) -> tuple[tuple[Observation, ...], int]:
    """Choose a stable sequence per input and recipe, rejecting conflicts."""
    selected: dict[tuple[str, str], Observation] = {}
    duplicates = 0
    for row in sorted(rows, key=lambda item: (item.design, item.sequence)):
        key = (row.input_state_id, row.recipe_id)
        prior = selected.get(key)
        if prior is None:
            selected[key] = row
            continue
        duplicates += 1
        if (
            prior.output_state_id != row.output_state_id
            or not math.isclose(prior.gain_pct, row.gain_pct, abs_tol=1e-9)
            or prior.zero_reason != row.zero_reason
        ):
            raise ValueError(f"conflicting duplicate training edge: {key}")
    return tuple(selected.values()), duplicates


def rank_observations(
    rows: Sequence[Observation], recipe_sets: Mapping[str, Sequence[str]]
) -> dict[str, list[dict[str, Any]]]:
    """Give each design equal weight, regardless of its number of states."""
    unique, _ = deduplicate(rows)
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    for row in unique:
        if row.design not in TRAIN_DESIGNS:
            raise ValueError("non-training design in observations")
        if not math.isfinite(row.gain_pct):
            raise ValueError("nonfinite target")
        buckets[(row.recipe_id, row.design)].append(row.gain_pct)
        if row.zero_reason is not None:
            reasons[row.recipe_id][row.zero_reason] += 1
    ranked: dict[str, list[dict[str, Any]]] = {}
    for worker in ("timing", "area"):
        recipes = tuple(recipe_sets[worker])
        entries = []
        for recipe in recipes:
            by_design = {
                design: mean(buckets[(recipe, design)])
                for design in TRAIN_DESIGNS
                if buckets[(recipe, design)]
            }
            if set(by_design) != set(TRAIN_DESIGNS):
                raise ValueError(f"missing training design for {recipe}")
            entries.append({
                "recipe_id": recipe,
                "beta_pct": mean(by_design.values()),
                "design_means_pct": by_design,
                "unique_edges": sum(len(buckets[(recipe, d)]) for d in TRAIN_DESIGNS),
                "zero_reasons": dict(reasons[recipe]),
            })
        ranked[worker] = sorted(
            entries, key=lambda row: (-row["beta_pct"], row["recipe_id"])
        )
    return ranked


def _checked_run(root: Path, run_id: str) -> tuple[dict[str, Any], Evidence | None, str | None]:
    if SAFE_RUN.fullmatch(run_id) is None:
        raise ValueError(f"unsafe run ID: {run_id}")
    run_dir = root / "runs" / run_id
    marker = _read_json(root, run_dir / "complete.json")
    if marker.get("run_id") != run_id:
        raise ValueError("run completion marker ID mismatch")
    files = marker.get("files")
    if not isinstance(files, dict) or "runinfo.json" not in files:
        raise ValueError("run completion marker lacks runinfo")
    raw: dict[str, bytes] = {}
    for name, digest in files.items():
        if name not in {"runinfo.json", "output_state.json", "evidence.json", "claims.json"}:
            raise ValueError("unexpected run artifact")
        _hex(digest, "artifact digest")
        path = (run_dir / name).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("run artifact escaped training root")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"run artifact hash mismatch: {path}")
        raw[name] = content
    run = json.loads(raw["runinfo.json"])
    if run.get("id") != run_id or run.get("fake") is not False:
        raise ValueError("run identity or origin mismatch")
    evidence = (
        Evidence.model_validate_json(raw["evidence.json"])
        if "evidence.json" in raw else None
    )
    state_id = None
    if "output_state.json" in raw:
        state = json.loads(raw["output_state.json"])
        state_id = _hex(state.get("id"), "output state ID")
        if state.get("fake") is not False:
            raise ValueError("output state origin mismatch")
    if evidence is not None and evidence.state_id != state_id:
        raise ValueError("Evidence/output state mismatch")
    return run, evidence, state_id


def _start_evidence(root: Path, design: str, state_id: str, evidence_id: str) -> Evidence:
    _hex(state_id, "start state ID")
    _hex(evidence_id, "start Evidence ID")
    found: list[Evidence] = []
    for path in (root / "measurements" / state_id).glob("*/evidence.json"):
        record = Evidence.model_validate(_read_json(root, path))
        if record.id == evidence_id:
            found.append(record)
    if len(found) != 1 or found[0].state_id != state_id or found[0].fake:
        raise ValueError(f"missing or ambiguous B2 Evidence for {design}")
    return found[0]


def _verification(
    root: Path, fingerprint: str, design: str, output_id: str,
    start_id: str,
) -> str:
    path = root / "verification/records" / fingerprint / design / f"{output_id}.json"
    if not path.exists():
        return "no_record"
    record = _read_json(root, path)
    if (
        record.get("validator_fingerprint") != fingerprint
        or record.get("design") != design
        or record.get("output_sha256") != output_id
        or record.get("reference_sha256") != start_id
    ):
        raise ValueError("AMD verification record identity mismatch")
    result = record.get("result")
    if result not in {"pass", "fail", "unknown"}:
        raise ValueError("invalid AMD verification result")
    return result


def load_observations(
    data_root: Path | str | None = None,
) -> tuple[list[Observation], dict[str, Any]]:
    """Read a checksummed live journal snapshot and saved meter records."""
    root = require_training_root(data_root)
    spec = yaml.safe_load(RECIPE_SETS.read_text(encoding="utf-8"))
    if spec.get("signed_off") is not True:
        raise ValueError("recipe sets are not signed")
    recipe_sets = {
        worker: tuple(spec["worker_sets"][worker]) for worker in ("timing", "area")
    }
    if any(len(ids) != 5 or len(set(ids)) != 5 for ids in recipe_sets.values()):
        raise ValueError("expected five unique recipes per worker")
    recipe_worker = {
        recipe: worker for worker, ids in recipe_sets.items() for recipe in ids
    }
    if len(recipe_worker) != 10:
        raise ValueError("recipe appears in both workers")
    events = Journal(root).read("worker_fitting_train")
    if not events or events[0].get("type") != "configuration":
        raise ValueError("training journal has no configuration")
    configuration = events[0]["configuration"]
    if configuration.get("fake") is not False:
        raise ValueError("fake training journal")
    starts = configuration["designs"]
    if set(starts) != set(TRAIN_DESIGNS):
        raise ValueError("training journal design set changed")
    for worker, ids in recipe_sets.items():
        if tuple(row["id"] for row in configuration["recipe_sets"][worker]) != ids:
            raise ValueError("journal recipe set differs from signed configuration")
    completed = {event.get("phase") for event in events if event.get("type") == "phase_complete"}
    if not {"D1", "D2"}.issubset(completed):
        raise ValueError("D1 and D2 must be complete")
    plan = _read_json(root, root / "verification/plan.json")
    fingerprint = _hex(plan.get("fingerprint"), "validator fingerprint")
    start_evidence = {
        design: _start_evidence(
            root, design, starts[design]["start_state_id"],
            starts[design]["start_evidence_id"],
        )
        for design in TRAIN_DESIGNS
    }
    expected_meter_version = configuration.get("meter_version")
    if not isinstance(expected_meter_version, str) or not expected_meter_version:
        raise ValueError("journal lacks a meter version")
    if any(record.meter_version != expected_meter_version for record in start_evidence.values()):
        raise ValueError("B2 Evidence meter version differs from journal")
    state_evidence = {record.state_id: record for record in start_evidence.values()}
    latest: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for event in events[1:]:
        if event.get("type") != "edge":
            continue
        edge = event["edge"]
        if edge.get("first_phase") not in {"D1", "D2", "D3"}:
            continue
        design = edge.get("design")
        sequence = tuple(edge.get("sequence") or ())
        if design not in TRAIN_DESIGNS or not sequence or sequence[-1] not in recipe_worker:
            raise ValueError("non-training edge in journal")
        latest[(design, sequence)] = edge
    outcomes: dict[tuple[str, tuple[str, ...]], tuple[dict[str, Any], Evidence | None, str | None]] = {}
    for key, edge in latest.items():
        run_id = edge.get("run_id")
        if run_id is None:
            continue
        run, evidence, state_id = _checked_run(root, run_id)
        if (
            run.get("recipe_id") != key[1][-1]
            or run.get("status") != edge.get("run_status")
            or run.get("input_sha256_before") != edge.get("input_state_id")
            or (
                run.get("input_sha256_after") is not None
                and run["input_sha256_after"] != edge.get("input_state_id")
            )
            or state_id != edge.get("output_state_id")
        ):
            raise ValueError("journal edge differs from run record")
        if evidence is not None and evidence.meter_version != expected_meter_version:
            raise ValueError("edge Evidence meter version differs from journal")
        outcomes[key] = (run, evidence, state_id)
        if evidence is not None and edge.get("run_status") == "ok" and edge.get("physically_valid"):
            old = state_evidence.get(evidence.state_id)
            if old is not None and (
                _metric(old, "wns_ns") != _metric(evidence, "wns_ns")
                or _metric(old, "packed_luts") != _metric(evidence, "packed_luts")
                or old.constraint_fingerprint != evidence.constraint_fingerprint
                or old.clock_digest != evidence.clock_digest
            ):
                raise ValueError("same state has conflicting meter Evidence")
            state_evidence[evidence.state_id] = evidence
    rows: list[Observation] = []
    verification_counts: Counter[str] = Counter()
    for key, edge in latest.items():
        design, sequence = key
        input_id = edge.get("input_state_id")
        if input_id is None:
            continue  # An invalid parent was never a recipe decision on a state.
        _hex(input_id, "input state ID")
        output_id = edge.get("output_state_id")
        if output_id is not None:
            _hex(output_id, "output state ID")
        recipe = sequence[-1]
        parent = state_evidence.get(input_id)
        if parent is None:
            raise ValueError(f"missing input meter Evidence: {input_id}")
        start = start_evidence[design]
        if (
            parent.constraint_fingerprint != start.constraint_fingerprint
            or parent.clock_digest != start.clock_digest
        ):
            raise ValueError("input constraints or clock differ from B2")
        status = (
            _verification(
                root, fingerprint, design, output_id,
                starts[design]["start_state_id"],
            )
            if output_id is not None else "no_output"
        )
        verification_counts[status] += 1
        zero_reason = zero_gain_reason(
            edge.get("run_status"), bool(edge.get("physically_valid")),
            input_id, output_id, status,
        )
        if zero_reason is None:
            child = state_evidence.get(output_id)
            if child is None:
                raise ValueError("valid output lacks meter Evidence")
            if (
                child.constraint_fingerprint != start.constraint_fingerprint
                or child.clock_digest != start.clock_digest
            ):
                raise ValueError("output constraints or clock differ from B2")
        gain = (
            0.0 if zero_reason is not None else
            gain_pct(recipe_worker[recipe], parent, state_evidence[output_id],
                     REQUIREMENT_NS[design])
        )
        rows.append(Observation(
            design=design, sequence=sequence, input_state_id=input_id,
            recipe_id=recipe, output_state_id=output_id,
            gain_pct=gain, zero_reason=zero_reason, verification=status,
        ))
    snapshot = hashlib.sha256(
        json.dumps(events, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    metadata = {
        "journal_event_count": len(events),
        "journal_snapshot_sha256": snapshot,
        "latest_edges": len(latest),
        "raw_observations": len(rows),
        "verification_counts": dict(verification_counts),
        "recipe_sets": recipe_sets,
        "feature_spec_sha256": hashlib.sha256(FEATURE_SPEC.read_bytes()).hexdigest(),
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_phases": sorted(completed),
    }
    return rows, metadata


def format_report(
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    metadata: Mapping[str, Any],
    duplicate_count: int,
) -> str:
    """Render a short, fully labelled training snapshot."""
    lines = [
        "# PROVISIONAL null rankings from training meter Evidence",
        "",
        f"**PROVISIONAL snapshot:** {metadata['observed_at_utc']}. "
        f"{metadata['journal_event_count']} checksummed journal events; "
        f"{metadata['latest_edges']} latest D1/D2/D3 edges; "
        f"{duplicate_count} repeated (input hash, recipe) pairs removed. "
        "D3 may still be running.",
        "",
        "Each beta is the mean of five design means, including zero gain for "
        "invalid, AMD non-PASS, missing or unchanged outputs. "
        "Timing uses Fmax = 1000 / (B2 requirement - meter WNS); area uses "
        "meter packed LUT cut. The B2 requirements (ns) are "
        + ", ".join(f"{name} {value:.3f}" for name, value in REQUIREMENT_NS.items())
        + ". notes/training_data_audit.md:36 verified these against the first "
        "708 meter timing reports. For later D3 edges, equal requirement is "
        "inferred from matching constraint/clock digests; their report lines "
        "were not checked here. Every Evidence meter version is checked "
        "against the journal. No raw report path outside the training root "
        "is followed.",
        "",
    ]
    for worker, label in (("timing", "A, Fmax gain"), ("area", "B, packed LUT cut")):
        lines.extend([
            f"## Worker {label}",
            "",
            "| Rank | Recipe | Beta gain (%) | Unique inputs | Zero gains |",
            "|---:|---|---:|---:|---:|",
        ])
        for index, row in enumerate(rankings[worker], 1):
            lines.append(
                f"| {index} | {row['recipe_id']} | {row['beta_pct']:+.4f} | "
                f"{row['unique_edges']} | {sum(row['zero_reasons'].values())} |"
            )
        lines.extend([
            "",
            "Per-design means, in configured design order:",
            "",
        ])
        for row in rankings[worker]:
            values = ", ".join(
                f"{design} {row['design_means_pct'][design]:+.3f}%"
                for design in TRAIN_DESIGNS
            )
            lines.append(f"- {row['recipe_id']}: {values}.")
        lines.append("")
    lines.extend([
        f"Verification records in the snapshot: {metadata['verification_counts']}. "
        "All ranking outputs remain labelled PROVISIONAL.",
        "",
        f"Journal event snapshot SHA-256: {metadata['journal_snapshot_sha256']}. "
        f"Frozen feature YAML SHA-256: {metadata['feature_spec_sha256']}.",
        "",
    ])
    return "\n".join(lines)


def null_payload(
    worker: str, rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Make a machine-readable null model that cannot look production-ready."""
    if worker not in {"timing", "area"}:
        raise ValueError("unknown worker")
    return {
        "schema_version": "worker-null-v1-provisional",
        "status": "PROVISIONAL",
        "worker": worker,
        "target": "fmax_gain_pct" if worker == "timing" else "packed_lut_cut_pct",
        "beta_pct_by_recipe": {
            row["recipe_id"]: row["beta_pct"] for row in rankings[worker]
        },
        "selected_recipe_id": rankings[worker][0]["recipe_id"],
        "training_designs": list(TRAIN_DESIGNS),
        "journal_event_count": metadata["journal_event_count"],
        "journal_snapshot_sha256": metadata["journal_snapshot_sha256"],
        "feature_spec_sha256": metadata["feature_spec_sha256"],
        "created_at_utc": metadata["observed_at_utc"],
    }


def _write_atomic(path: Path, content: str) -> None:
    """Publish each generated artifact without exposing a partial file."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=TRAIN_ROOT)
    arguments = parser.parse_args()
    rows, metadata = load_observations(arguments.data_root)
    unique, duplicates = deduplicate(rows)
    ranked = rank_observations(unique, metadata["recipe_sets"])
    _write_atomic(NOTE, format_report(ranked, metadata, duplicates))
    for worker in ("timing", "area"):
        payload = null_payload(worker, ranked, metadata)
        _write_atomic(
            NULL_CONFIGS[worker], json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        best = ranked[worker][0]
        print(f"PROVISIONAL {worker}: {best['recipe_id']} beta={best['beta_pct']:+.4f}%")


if __name__ == "__main__":
    main()

