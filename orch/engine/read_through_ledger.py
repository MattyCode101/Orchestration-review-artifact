"""AMD verification ledger that also reads earlier roots' PASS records, never writing them.

The campaign's own root is the only place this ledger writes. A stored output already has a
validator record in the root that produced it; this ledger accepts such a record when
  * the source root's verification plan differs from ours only in its design reference set
    (the same validator code, adapter, runner, job tagging, environment and RapidWright), and
  * the record names the same design, reference B2 and output hash, is a pass, and its
    retained raw report still has the recorded hash.
Outputs without an accepted record are validated as the base ledger does, into our own root.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from orch.baselines.verification import SCHEMA, VerificationFailure, VerificationLedger
from orch.engine.store import _json_bytes, _sha256_file

_COMPARED = (
    "schema_version", "vectors", "precheck_vectors", "adapter_sha256", "verification_code_sha256",
    "runner_sha256", "jobtag_sha256", "env_script_sha256", "validator_sources", "validator_python",
    "rapidwright_root", "validator_adapter_version",
)


class ReadThroughLedger(VerificationLedger):
    def __init__(self, *, source_roots: Sequence[Path], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.equivalent_sources: list[Path] = []
        self.rejected_sources: dict[str, str] = {}
        for root in source_roots:
            plan_path = Path(root) / "verification" / "plan.json"
            if not plan_path.is_file():
                self.rejected_sources[str(root)] = "no verification plan"
                continue
            theirs = json.loads(plan_path.read_text(encoding="utf-8"))
            differs = [k for k in _COMPARED if theirs.get(k) != self.plan.get(k)]
            if differs:
                self.rejected_sources[str(root)] = "validator setup differs: " + ",".join(differs)
            else:
                self.equivalent_sources.append(Path(root) / "verification")
        self.accepted_from_sources: dict[tuple[str, str], str] = {}

    def _source_record(self, design: str, output_id: str) -> tuple[dict[str, Any], Path] | None:
        for base in self.equivalent_sources:
            for path in sorted((base / "records").glob(f"*/{design}/{output_id}.json")):
                record = json.loads(path.read_text(encoding="utf-8"))
                if (
                    record.get("schema_version") != SCHEMA or record.get("design") != design
                    or record.get("reference_sha256") != self.references[design]
                    or record.get("output_sha256") != output_id or record.get("result") != "pass"
                ):
                    continue
                report, digest = record.get("raw_report_path"), record.get("raw_report_sha256")
                if not report or not digest or not Path(report).is_file() or _sha256_file(Path(report)) != digest:
                    continue
                return record, path
        return None

    def record_location(self, design: str, output_id: str) -> Path | None:
        own = self._record_path(design, output_id)
        if own.is_file():
            return own
        found = self._source_record(design, output_id)
        return found[1] if found else None

    def require_pass(self, design: str, output_id: str) -> dict:
        if design not in self.references:
            raise ValueError(f"design is outside verification references: {design}")
        self.store.state_path(output_id)
        if self._record_path(design, output_id).is_file():
            return super().require_pass(design, output_id)
        found = self._source_record(design, output_id)
        if found is None:
            raise VerificationFailure(f"missing validator record: {design} {output_id}")
        self.accepted_from_sources[(design, output_id)] = str(found[1])
        return found[0]

    def verify(self, design: str, output_id: str) -> dict:
        if design not in self.references:
            raise ValueError(f"design is outside verification references: {design}")
        self.store.state_path(output_id)
        if not self._record_path(design, output_id).is_file():
            found = self._source_record(design, output_id)
            if found is not None:
                self.accepted_from_sources[(design, output_id)] = str(found[1])
                return found[0]
        return super().verify(design, output_id)


def census_proof(ledger: ReadThroughLedger, design: str, output_id: str, record: Mapping[str, Any],
                 outputs_compared: int, record_path: Path) -> dict[str, Any]:
    """The AMD output census, written under our own root only."""

    log_path = Path(record["attempts"][-1]["run_dir"]) / "validate.log"
    return {
        "schema_version": "thursday-night-amd-census-v1", "design": design, "state_sha256": output_id,
        "validator_record_sha256": _sha256_file(record_path), "validate_log_sha256": _sha256_file(log_path),
        "outputs_compared": outputs_compared,
    }


__all__ = ["ReadThroughLedger", "census_proof", "_json_bytes"]
