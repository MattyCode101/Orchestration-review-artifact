"""Keep every saved retry DCP discoverable by a readable name."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from orch.engine.journal import Journal
from orch.engine.store import StateStore
from scripts import index_worker_fitting_dcps as indexer


def test_index_keeps_both_saved_attempts_and_latest_alias(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "campaign"
    store = StateStore(root)
    files = []
    for name in ("start", "first", "second"):
        path = tmp_path / f"{name}.dcp"
        path.write_bytes(name.encode("ascii"))
        files.append(path)
    start, first, second = (
        store.import_state(path, design="train") for path in files
    )
    monkeypatch.setattr(indexer, "ROOT", root)
    monkeypatch.setattr(indexer, "ALIASES", root / "by_design")
    monkeypatch.setattr(indexer, "TRAIN_DESIGNS", ("train",))
    monkeypatch.setattr(indexer, "TIMING_IDS", ("R",))
    monkeypatch.setattr(indexer, "AREA_IDS", ())
    monkeypatch.setattr(indexer, "CAMPAIGN_ID", "train_test")
    (root / "launch_manifest.json").write_text(json.dumps({
        "train_inputs": [{"design": "train", "sha256": start.id}],
        "recipe_sets": {"timing": [{"id": "R"}], "area": []},
    }), encoding="utf-8")
    journal = Journal(root)
    for attempt, state in ((1, first), (2, second)):
        journal.append("train_test", {
            "id": f"edge-{attempt}", "type": "edge",
            "edge": {
                "design": "train", "sequence": ["R"],
                "run_id": f"run-{attempt}", "output_state_id": state.id,
                "run_status": "meter_error" if attempt == 1 else "ok",
                "physically_valid": attempt == 2,
                "brief_valid": {}, "attempt": attempt,
            },
        })
    result = indexer.build()
    assert result["committed_edges"] == 1
    assert result["saved_attempt_outputs"] == 2
    assert result["history_aliases"] == 2
    aliases = root / "by_design" / "train"
    assert (aliases / "depth1" / "R.dcp").resolve() == store.state_path(second.id)
    assert (
        aliases / "attempts" / "depth1" / "R__attempt1.dcp"
    ).resolve() == store.state_path(first.id)
    assert (
        aliases / "attempts" / "depth1" / "R__attempt2.dcp"
    ).resolve() == store.state_path(second.id)
    with (root / "by_design" / "index.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["alias_role"] for row in rows].count("history") == 2
    assert {row["attempt"] for row in rows if row["alias_role"] == "history"} == {"1", "2"}
    assert {row["equivalence_status"] for row in rows if row["alias_role"] == "history"} == {"UNVERIFIED"}
    # A later retry with no output must not leave the prior latest symlink.
    journal.append("train_test", {
        "id": "edge-3", "type": "edge",
        "edge": {
            "design": "train", "sequence": ["R"],
            "run_id": "run-3", "output_state_id": None,
            "run_status": "timeout", "physically_valid": False,
            "brief_valid": {}, "attempt": 3,
        },
    })
    second_index = indexer.build()
    assert second_index["history_aliases"] == 2
    assert not (aliases / "depth1" / "R.dcp").is_symlink()
    assert (
        aliases / "attempts" / "depth1" / "R__attempt1.dcp"
    ).resolve() == store.state_path(first.id)


def test_pass_index_label_requires_pinned_record_and_raw_report(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "campaign"
    monkeypatch.setattr(indexer, "ROOT", root)
    verification = root / "verification"
    fingerprint = "f" * 64
    reference = "a" * 64
    output = "b" * 64
    (verification / "plan.json").parent.mkdir(parents=True)
    (verification / "plan.json").write_text(json.dumps({
        "fingerprint": fingerprint,
        "references": {"train": reference},
        "vectors": 200, "precheck_vectors": 100,
    }), encoding="utf-8")
    raw = verification / "runs" / "verify-1" / "validator_work" / "validation_report.json"
    raw.parent.mkdir(parents=True)
    report = {
        "overall_passed": True, "phase1_passed": True,
        "phase2_passed": True, "infrastructure_failure": False,
        "num_vectors": 200, "precheck_vectors": 100,
        "simulation_report": {
            "precheck_report": {
                "passed": True, "cycles_simulated": 100,
                "mismatch_count": 0, "protocol_mismatch_count": 0,
            },
            "full_report": {
                "passed": True, "cycles_simulated": 200,
                "mismatch_count": 0, "protocol_mismatch_count": 0,
            },
        },
    }
    raw.write_text(json.dumps(report), encoding="utf-8")
    record = {
        "design": "train", "reference_sha256": reference,
        "output_sha256": output, "validator_fingerprint": fingerprint,
        "result": "pass", "raw_report_path": str(raw),
        "raw_report_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
    }
    canonical = verification / "records" / fingerprint / "train" / f"{output}.json"
    latest = verification / "latest" / "train" / f"{output}.json"
    for path in (canonical, latest):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(record), encoding="utf-8")
    assert indexer._verification("train", output, reference)[0] == "pass"
    raw.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="raw report changed"):
        indexer._verification("train", output, reference)
    report["simulation_report"]["full_report"]["mismatch_count"] = 1
    raw.write_text(json.dumps(report), encoding="utf-8")
    record["raw_report_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    for path in (canonical, latest):
        path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks a passing AMD report"):
        indexer._verification("train", output, reference)
    report["simulation_report"]["full_report"]["mismatch_count"] = 0
    raw.write_text(json.dumps(report), encoding="utf-8")
    record["raw_report_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    record["validator_fingerprint"] = "e" * 64
    for path in (canonical, latest):
        path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from pinned DCP"):
        indexer._verification("train", output, reference)
