"""Adversarial tests for fail-closed sweep verification at journal boundaries."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from orch.baselines.verification import VerificationFailure
from orch.baselines.verification_guard import (
    FailClosedVerifiedStepEngine, audit_all_saved,
)
from orch.contract.models import RunStatus


class Ledger:
    """Minimal record ledger that exposes which exact DCPs were checked."""

    references = {"train": "a" * 64}

    def __init__(self, result: str = "pass") -> None:
        self.result = result
        self.checked: list[tuple[str, str]] = []
        self.passing: set[tuple[str, str]] = set()

    def verify(self, design: str, output_id: str) -> dict[str, str]:
        self.checked.append((design, output_id))
        if self.result == "pass":
            self.passing.add((design, output_id))
        return {"result": self.result}

    def require_pass(self, design: str, output_id: str) -> dict[str, str]:
        self.checked.append((design, output_id))
        if (design, output_id) not in self.passing:
            raise VerificationFailure(f"missing PASS for {design} {output_id}")
        return {"result": "pass"}


class Engine:
    """Return a saved state even when its independent meter failed."""

    store = object()
    meter_version = "test-meter"

    def __init__(self, status: RunStatus) -> None:
        self.status = status

    def execute(self, state, recipe):
        return SimpleNamespace(
            state=SimpleNamespace(design="train", id="b" * 64),
            run=SimpleNamespace(status=self.status),
        )


@pytest.mark.parametrize("status", [RunStatus.METER_ERROR, RunStatus.INVALID])
@pytest.mark.parametrize("result", ["fail", "unknown"])
def test_saved_non_ok_output_cannot_commit_without_pass(status, result) -> None:
    ledger = Ledger(result)
    wrapped = FailClosedVerifiedStepEngine(Engine(status), ledger)
    with pytest.raises(VerificationFailure, match="missing PASS"):
        wrapped.execute(object(), object())
    assert ledger.checked == [("train", "b" * 64), ("train", "b" * 64)]


def test_saved_meter_error_can_commit_when_validator_passes() -> None:
    ledger = Ledger("pass")
    wrapped = FailClosedVerifiedStepEngine(Engine(RunStatus.METER_ERROR), ledger)
    outcome = wrapped.execute(object(), object())
    assert outcome.state.id == "b" * 64
    assert ledger.checked == [("train", "b" * 64), ("train", "b" * 64)]


def test_audit_includes_superseded_saved_attempt() -> None:
    ledger = Ledger()
    old = ("train", "b" * 64)
    new = ("train", "c" * 64)
    ledger.passing.add(new)
    events = [
        {"type": "edge", "edge": {
            "design": "train", "sequence": ["recipe"], "output_state_id": old[1],
        }},
        {"type": "edge", "edge": {
            "design": "train", "sequence": ["recipe"], "output_state_id": new[1],
        }},
    ]
    with pytest.raises(VerificationFailure, match=old[1]):
        audit_all_saved(events, ledger)
    ledger.passing.add(old)
    assert audit_all_saved(events, ledger) == {
        "journal_edge_attempts": 2,
        "distinct_sequences": 1,
        "saved_output_attempts": 2,
        "distinct_verified_dcps": 2,
    }


def test_audit_rejects_held_out_design() -> None:
    ledger = Ledger()
    events = [{"type": "edge", "edge": {
        "design": "held_out", "sequence": ["recipe"],
        "output_state_id": "d" * 64,
    }}]
    with pytest.raises(ValueError, match="unselected design"):
        audit_all_saved(events, ledger)
