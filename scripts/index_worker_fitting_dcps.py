"""Create readable aliases and a CSV for every committed training DCP.

Canonical DCPs remain content addressed and read only. Aliases are relative
symlinks; the checked campaign journal supplies their sequence and run ID.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from orch.baselines.worker_fitting import (
    AREA_IDS, CAMPAIGN_ID, TIMING_IDS, TRAIN_DESIGNS,
)
from orch.engine.journal import Journal
from orch.engine.store import StateStore, _atomic_bytes, _json_bytes, _sha256_file

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "data" / "sweeps" / "worker_fitting_train"
ALIASES = ROOT / "by_design"
SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")
COLUMNS = (
    "design", "depth", "sequence", "run_id", "output_sha256", "dcp_alias",
    "canonical_dcp", "run_status", "physically_valid", "timing_brief_valid",
    "area_brief_valid", "equivalence_status", "verification_record",
    "alias_role", "attempt",
)


def _component(value: str) -> str:
    """Keep human names readable without allowing path traversal."""
    if value in (".", "..") or not SAFE.fullmatch(value):
        raise ValueError(f"unsafe alias component: {value!r}")
    return value


def _alias(path: Path, target: Path) -> None:
    """Publish or update a relative symlink atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not path.is_symlink():
        raise ValueError(f"alias is not a symlink: {path}")
    if path.is_symlink() and path.resolve(strict=True) == target:
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    relative = os.path.relpath(target, path.parent)
    try:
        temporary.symlink_to(relative)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verification(design: str, state_id: str, start_id: str) -> tuple[str, str]:
    """Label a DCP only from its pinned record and retained raw AMD report."""
    root = ROOT / "verification"
    latest = root / "latest" / design / f"{state_id}.json"
    if not latest.is_file():
        return "UNVERIFIED", ""
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    fingerprint = plan["fingerprint"]
    canonical = root / "records" / fingerprint / design / f"{state_id}.json"
    if not canonical.is_file() or latest.read_bytes() != canonical.read_bytes():
        raise ValueError(f"verification pointer lacks its canonical record: {latest}")
    data = json.loads(latest.read_text(encoding="utf-8"))
    if (
        data.get("design") != design
        or data.get("reference_sha256") != start_id
        or data.get("output_sha256") != state_id
        or data.get("validator_fingerprint") != fingerprint
        or data.get("result") not in ("pass", "fail", "unknown")
        or plan.get("references", {}).get(design) != start_id
    ):
        raise ValueError(f"verification pointer differs from pinned DCP: {latest}")
    raw_path = data.get("raw_report_path")
    raw_sha = data.get("raw_report_sha256")
    if bool(raw_path) != bool(raw_sha):
        raise ValueError(f"validator report path/hash pair is incomplete: {latest}")
    if raw_path:
        raw = Path(raw_path).resolve(strict=True)
        if not raw.is_relative_to((root / "runs").resolve()):
            raise ValueError(f"validator report escaped private runs: {raw}")
        if _sha256_file(raw) != raw_sha:
            raise ValueError(f"validator raw report changed: {raw}")
        if data["result"] == "pass":
            report = json.loads(raw.read_text(encoding="utf-8"))
            simulation = report.get("simulation_report") or {}
            precheck = simulation.get("precheck_report") or {}
            full = simulation.get("full_report") or {}
            if not (
                report.get("overall_passed") is True
                and report.get("phase1_passed") is True
                and report.get("phase2_passed") is True
                and report.get("infrastructure_failure") is False
                and report.get("num_vectors") == plan["vectors"]
                and report.get("precheck_vectors") == plan["precheck_vectors"]
                and precheck.get("passed") is True
                and precheck.get("cycles_simulated") == plan["precheck_vectors"]
                and precheck.get("mismatch_count") == 0
                and precheck.get("protocol_mismatch_count") == 0
                and full.get("passed") is True
                and full.get("cycles_simulated") == plan["vectors"]
                and full.get("mismatch_count") == 0
                and full.get("protocol_mismatch_count") == 0
            ):
                raise ValueError(f"PASS pointer lacks a passing AMD report: {latest}")
    elif data["result"] == "pass":
        raise ValueError(f"PASS pointer lacks a raw validator report: {latest}")
    return str(data["result"]), str(latest)


def build(checked_cache: dict[str, Path] | None = None) -> dict[str, object]:
    """Refresh aliases and index from the checksummed train-only journal."""
    manifest = json.loads((ROOT / "launch_manifest.json").read_text(encoding="utf-8"))
    inputs = {row["design"]: row for row in manifest["train_inputs"]}
    if tuple(inputs) != TRAIN_DESIGNS:
        raise ValueError("launch manifest is not exactly the five training designs")
    selected = {row["id"] for worker in manifest["recipe_sets"].values() for row in worker}
    if selected != set(TIMING_IDS + AREA_IDS):
        raise ValueError("launch manifest recipe set differs from selected ten")
    latest: dict[tuple[str, tuple[str, ...]], dict] = {}
    histories: dict[tuple[str, tuple[str, ...]], list[dict]] = {}
    for event in Journal(ROOT).read(CAMPAIGN_ID):
        if event.get("type") != "edge":
            continue
        edge = event["edge"]
        design = edge["design"]
        sequence = tuple(edge["sequence"])
        if design not in inputs or not 1 <= len(sequence) <= 4 or any(
            recipe not in selected for recipe in sequence
        ):
            raise ValueError("journal edge exceeds train-only alias scope")
        key = (design, sequence)
        histories.setdefault(key, []).append(edge)
        latest[key] = edge
    store = StateStore(ROOT)
    checked: dict[str, Path] = checked_cache if checked_cache is not None else {}

    def canonical(state_id: str) -> Path:
        if state_id not in checked:
            checked[state_id] = store.state_path(state_id)
        return checked[state_id]

    rows: list[dict[str, object]] = []
    for design, start in inputs.items():
        name = _component(design)
        start_path = canonical(start["sha256"])
        alias = ALIASES / name / "start_B2.dcp"
        _alias(alias, start_path)
        rows.append({
            "design": design, "depth": 0, "sequence": "B2", "run_id": "",
            "output_sha256": start["sha256"], "dcp_alias": str(alias),
            "canonical_dcp": str(start_path), "run_status": "start",
            "physically_valid": True, "timing_brief_valid": "",
            "area_brief_valid": "", "equivalence_status": "reference",
            "verification_record": "", "alias_role": "reference", "attempt": "",
        })
    for (design, sequence), edge in sorted(latest.items()):
        components = [_component(recipe) for recipe in sequence]
        filename = "__".join(components) + ".dcp"
        if len(filename.encode("utf-8")) > 240:
            raise ValueError("sequence alias name exceeds safe filename length")
        alias = ALIASES / _component(design) / f"depth{len(sequence)}" / filename
        output_id = edge.get("output_state_id")
        if not output_id:
            # A retry can supersede a saved DCP with no output. Keep its
            # attempt alias, but remove only the now-stale latest symlink.
            if alias.is_symlink():
                alias.unlink()
            elif alias.exists():
                raise ValueError(f"latest alias is not a symlink: {alias}")
            continue
        path = canonical(output_id)
        _alias(alias, path)
        equivalence, record = _verification(
            design, output_id, inputs[design]["sha256"]
        )
        rows.append({
            "design": design, "depth": len(sequence),
            "sequence": " > ".join(sequence), "run_id": edge["run_id"] or "",
            "output_sha256": output_id, "dcp_alias": str(alias),
            "canonical_dcp": str(path), "run_status": edge["run_status"] or "",
            "physically_valid": edge["physically_valid"],
            "timing_brief_valid": edge["brief_valid"].get("brief_t", ""),
            "area_brief_valid": edge["brief_valid"].get("brief_a", ""),
            "equivalence_status": equivalence,
            "verification_record": record,
            "alias_role": "latest", "attempt": edge["attempt"],
        })
    history_aliases = 0
    saved_attempt_outputs = 0
    for (design, sequence), history in sorted(histories.items()):
        saved_attempt_outputs += sum(bool(edge.get("output_state_id")) for edge in history)
        if len(history) < 2:
            continue
        seen_attempts: set[int] = set()
        components = [_component(recipe) for recipe in sequence]
        for edge in history:
            attempt = edge["attempt"]
            if type(attempt) is not int or attempt < 1 or attempt in seen_attempts:
                raise ValueError("journal retry attempt is missing or duplicated")
            seen_attempts.add(attempt)
            output_id = edge.get("output_state_id")
            if not output_id:
                continue
            filename = "__".join(components) + f"__attempt{attempt}.dcp"
            if len(filename.encode("utf-8")) > 240:
                raise ValueError("attempt alias name exceeds safe filename length")
            path = canonical(output_id)
            alias = (
                ALIASES / _component(design) / "attempts"
                / f"depth{len(sequence)}" / filename
            )
            _alias(alias, path)
            equivalence, record = _verification(
                design, output_id, inputs[design]["sha256"]
            )
            rows.append({
                "design": design, "depth": len(sequence),
                "sequence": " > ".join(sequence), "run_id": edge["run_id"] or "",
                "output_sha256": output_id, "dcp_alias": str(alias),
                "canonical_dcp": str(path), "run_status": edge["run_status"] or "",
                "physically_valid": edge["physically_valid"],
                "timing_brief_valid": edge["brief_valid"].get("brief_t", ""),
                "area_brief_valid": edge["brief_valid"].get("brief_a", ""),
                "equivalence_status": equivalence,
                "verification_record": record,
                "alias_role": "history", "attempt": attempt,
            })
            history_aliases += 1
    ALIASES.mkdir(parents=True, exist_ok=True)
    readme = (
        "# Training DCP aliases\n\n"
        "Each design has start_B2.dcp and depthN/<recipe sequence>.dcp. "
        "The depthN alias points to the latest attempt; retried sequences also "
        "have attempts/depthN/<recipe sequence>__attemptK.dcp aliases for "
        "every saved attempt. "
        "Aliases are relative symlinks to read-only content-addressed "
        "states/<SHA-256>/design.dcp. Copy a DCP before editing it. "
        "index.csv maps every alias to its sequence, run, hash, validity, "
        "and functional-equivalence status. UNVERIFIED is never a pass.\n"
    )
    _atomic_bytes(ALIASES / "README.md", readme.encode("utf-8"))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(ALIASES / "index.csv", output.getvalue().encode("utf-8"))
    summary = {
        "schema_version": "worker-fitting-dcp-index-v1",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_designs": list(inputs),
        "committed_edges": len(latest),
        "saved_attempt_outputs": saved_attempt_outputs,
        "history_aliases": history_aliases,
        "aliased_outputs": len(rows) - len(inputs),
        "unique_output_states": len({
            row["output_sha256"] for row in rows if row["depth"] != 0
        }),
        "unverified_aliases": sum(
            row["equivalence_status"] == "UNVERIFIED" for row in rows
        ),
        "index_csv": str(ALIASES / "index.csv"),
    }
    _atomic_bytes(ALIASES / "status.json", _json_bytes(summary))
    return summary


def _driver_alive(pid: int) -> bool:
    """Identify only the recorded resumed campaign process."""
    try:
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\x00")
    except OSError:
        return False
    return (
        b"--run" in args
        and str(PROJECT / "scripts/launch_worker_fitting_train.py").encode() in args
    )


def watch(pid: int) -> dict[str, object]:
    """Keep readable aliases current while the verified driver runs."""
    deadline = time.monotonic() + 36 * 3600
    checked: dict[str, Path] = {}
    result = build(checked)
    while time.monotonic() < deadline:
        if not _driver_alive(pid):
            # A final fresh hash pass catches any damaged canonical checkpoint.
            return build()
        time.sleep(45)
        result = build(checked)
    raise RuntimeError("DCP index watcher lifetime ended while driver remained alive")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()
    if args.watch:
        if args.pid is None:
            parser.error("--watch requires --pid")
        result = watch(args.pid)
    else:
        result = build()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
