"""The read-through AMD ledger accepts an earlier root's PASS record, never writes that root, and refuses stale ones."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orch.baselines.verification import SCHEMA, VerificationFailure, VerificationLedger
from orch.engine.read_through_ledger import ReadThroughLedger
from orch.engine.store import StateStore, _json_bytes, _sha256_file

PROJECT = Path(__file__).resolve().parents[1]


def _validator(version: str = "v1") -> SimpleNamespace:
    return SimpleNamespace(env_script=PROJECT / "recipes/support/env.sh", python="py", rapidwright_root="rw", version=version)


def _state(store: StateStore, tmp: Path, name: str):
    f = tmp / name
    f.write_bytes(name.encode())
    return store.import_state(f, design="d", fake=True)


def _tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): _sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def _build(tmp_path: Path, *, source_version: str = "v1"):
    own_store = StateStore(tmp_path / "own")
    ref = _state(own_store, tmp_path, "reference")
    out = _state(own_store, tmp_path, "output")
    src_store = StateStore(tmp_path / "src")
    src_ref = _state(src_store, tmp_path, "reference")
    assert src_ref.id == ref.id
    source = VerificationLedger(store=src_store, runner=None, validator=_validator(source_version),
                                references={"d": ref.id, "other": ref.id})
    report = tmp_path / "raw_report.json"
    report.write_text("{}", encoding="utf-8")
    record = {
        "schema_version": SCHEMA, "design": "d", "reference_sha256": ref.id, "output_sha256": out.id,
        "validator_fingerprint": source.fingerprint, "result": "pass", "attempts": [],
        "raw_report_path": str(report), "raw_report_sha256": _sha256_file(report),
    }
    path = source._record_path("d", out.id)
    path.parent.mkdir(parents=True)
    path.write_bytes(_json_bytes(record))
    ledger = ReadThroughLedger(source_roots=[tmp_path / "src"], store=own_store, runner=None,
                               validator=_validator(), references={"d": ref.id})
    return ledger, out, report, tmp_path / "src"


def test_a_source_pass_is_accepted_and_the_source_root_is_not_written(tmp_path: Path) -> None:
    ledger, out, _report, src = _build(tmp_path)
    assert ledger.equivalent_sources and not ledger.rejected_sources
    before = _tree(src)
    assert ledger.require_pass("d", out.id)["result"] == "pass"
    assert ledger.verify("d", out.id)["result"] == "pass"           # no validator run: runner is None
    assert _tree(src) == before
    assert ledger.accepted_from_sources[("d", out.id)].startswith(str(src))


def test_a_changed_raw_report_is_refused(tmp_path: Path) -> None:
    ledger, out, report, _src = _build(tmp_path)
    report.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(VerificationFailure):
        ledger.require_pass("d", out.id)


def test_a_source_with_another_validator_setup_is_not_trusted(tmp_path: Path) -> None:
    ledger, out, _report, _src = _build(tmp_path, source_version="v-other")
    assert not ledger.equivalent_sources and ledger.rejected_sources
    with pytest.raises(VerificationFailure):
        ledger.require_pass("d", out.id)
