"""Safety gates for detached resume after AMD DCP backfill."""
from __future__ import annotations

import pytest

from scripts import continue_worker_fitting_train as continuation

from scripts.continue_worker_fitting_train import FINGERPRINT, gate


def _state() -> dict:
    return {
        "mode": "backfill", "validator_fingerprint": FINGERPRINT,
        "total_distinct_dcps": 460, "phase": "running",
        "completed": 25, "statuses": {"pass": 25}, "errors": [],
    }


def test_controller_waits_for_incomplete_pass_only_backfill() -> None:
    assert gate(_state()) == "wait"


def test_controller_waits_while_final_backfill_audit_is_running() -> None:
    state = _state()
    state.update(phase="running", completed=460, statuses={"pass": 460})
    assert gate(state) == "wait"


def test_controller_accepts_only_all_460_pass() -> None:
    state = _state()
    state.update(phase="complete", completed=460, statuses={"pass": 460})
    assert gate(state) == "complete"


@pytest.mark.parametrize("bad", ["fail", "unknown", "error"])
def test_controller_stops_on_any_non_pass(bad: str) -> None:
    state = _state()
    state.update(completed=26, statuses={"pass": 25, bad: 1})
    with pytest.raises(RuntimeError, match="non-PASS"):
        gate(state)


def test_controller_stops_on_infrastructure_error_or_scope_drift() -> None:
    state = _state()
    state["errors"] = ["validator timeout"]
    with pytest.raises(RuntimeError, match="non-PASS"):
        gate(state)
    state = _state()
    state["validator_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint changed"):
        gate(state)


def test_controller_rejects_incomplete_complete_claim() -> None:
    state = _state()
    state.update(phase="complete", completed=459, statuses={"pass": 459})
    with pytest.raises(RuntimeError, match="incomplete or failed"):
        gate(state)


def test_backfill_stop_targets_only_its_exact_private_group(monkeypatch) -> None:
    sent = []
    monkeypatch.setattr(continuation, "_driver_alive", lambda pid: pid == 123)
    monkeypatch.setattr(continuation.os, "getpgid", lambda pid: 999)
    monkeypatch.setattr(continuation.os, "getpgrp", lambda: 777)
    monkeypatch.setattr(continuation.os, "killpg", lambda group, sig: sent.append(group))
    with pytest.raises(RuntimeError, match="unsafe backfill process group"):
        continuation._signal_backfill(123)
    assert sent == []
    monkeypatch.setattr(continuation.os, "getpgid", lambda pid: pid)
    assert continuation._signal_backfill(123) is True
    assert sent == [123]
    assert continuation._signal_backfill(456) is False


def test_resume_monitor_check_requires_driver_watchdog_and_indexer(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(continuation, "ROOT", tmp_path)
    (tmp_path / "resume_manifest.json").write_text(
        '{"driver_pid": 11, "watchdog_pid": 22, "indexer_pid": 33}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        continuation, "_exact_script_alive",
        lambda pid, script, flag: pid in (11, 22),
    )
    assert continuation._resume_monitors() == {
        "driver": True, "watchdog": True, "indexer": False,
    }


def test_timeout_escalates_exact_backfill_group_and_audits_tags(
    tmp_path, monkeypatch
) -> None:
    from scripts import watch_worker_fitting_verification as watcher

    monkeypatch.setattr(continuation, "VERIFY", tmp_path)
    monkeypatch.setattr(watcher, "ROOT", tmp_path)
    monkeypatch.setattr(watcher, "_cleanup", lambda mode: {"survivors": {}})
    monkeypatch.setattr(watcher, "_targets", lambda: (set(), set()))
    monkeypatch.setattr(continuation.time, "sleep", lambda seconds: None)
    calls = {"alive": 0}
    signals = []

    def alive(pid):
        calls["alive"] += 1
        return calls["alive"] <= 16

    monkeypatch.setattr(continuation, "_driver_alive", alive)
    monkeypatch.setattr(
        continuation, "_signal_backfill",
        lambda pid, sig: signals.append(sig) or True,
    )
    result = continuation._stop_backfill_and_audit(123)
    assert signals == [continuation.signal.SIGTERM, continuation.signal.SIGKILL]
    assert result["survivors"] == {}
