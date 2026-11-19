"""Explicit Evidence promotion for one audited, nonfunctional meter hash change.

The approved pair is loaded only when the immutable 20-DCP audit and its
frozen plan match their recorded SHA-256 values. Original Evidence stays in
its source run; a promoted cache hit retains a link to that run.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from orch.contract.models import Evidence

PROJECT = Path(__file__).resolve().parents[2]
OLD_TEST_METER = "2ee8ee9efb9416d84afbef9f1e37350c7b2ea0b054637efd8cff02dbed784435"
CURRENT_METER = "0dd0cf9424bbb7ab20dc5cf981bc35068c4ba08612ecd0f6c46098e6bf880bbe"
AUDIT_SHA256 = "36bb1fd15675f0e04f898223f8fc43047c7241d9f25a17d82bbb8ea9660299cc"
AUDIT = PROJECT / "data/scratch_thursday_night_2026-11-05/meter_equivalence/audit.json"


@dataclass(frozen=True)
class MeterEquivalence:
    """One exact old/current pair and the immutable audit that approved it."""

    old_version: str
    new_version: str
    audit_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_evidence_id(evidence: Evidence) -> str:
    """Recompute the ID using the existing meter parser's canonical payload."""

    payload = evidence.model_dump(
        mode="json", exclude={"id", "raw_report_path"}
    )
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def promote_evidence(
    evidence: Evidence, approval: MeterEquivalence
) -> Evidence:
    """Represent audited old Evidence under the current meter, retaining fields."""

    if evidence.meter_version != approval.old_version:
        raise ValueError("Evidence is not from the approved old meter")
    if evidence.id != canonical_evidence_id(evidence):
        raise ValueError("old Evidence ID does not match its measured content")
    data = evidence.model_dump(mode="json")
    data["meter_version"] = approval.new_version
    data["id"] = "pending"
    promoted = Evidence.model_validate(data)
    data["id"] = canonical_evidence_id(promoted)
    return Evidence.model_validate(data)


def approved_test_equivalence() -> MeterEquivalence:
    """Fail closed if the saved comparison or its frozen sample plan changed."""

    if not AUDIT.is_file() or _sha256(AUDIT) != AUDIT_SHA256:
        raise ValueError("meter equivalence audit is absent or changed")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    plan_path = AUDIT.parent / "plan.json"
    if not plan_path.is_file() or _sha256(plan_path) != audit.get("plan_sha256"):
        raise ValueError("meter equivalence sample plan changed")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        audit.get("training_samples") != 10
        or audit.get("test_samples") != 10
        or audit.get("identical") != 20
        or audit.get("different") != 0
        or audit.get("normalized_differences") != []
        or audit.get("live_exact_tags") != []
        or plan.get("old_test_version") != OLD_TEST_METER
        or plan.get("current_version") != CURRENT_METER
    ):
        raise ValueError("meter equivalence audit failed its approval terms")
    return MeterEquivalence(
        old_version=OLD_TEST_METER,
        new_version=CURRENT_METER,
        audit_sha256=AUDIT_SHA256,
    )
