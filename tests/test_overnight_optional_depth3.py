"""Safety gates for the isolated optional VexRiscv depth-three study."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import overnight_optional_depth3 as study


def test_scope_is_exactly_within_worker_depth_three() -> None:
    assert len(study.TASKS) == 250
    assert len(set(study.TASKS)) == 250
    assert sum(all(name in study.TIMING for name in seq) for seq in study.TASKS) == 125
    assert sum(all(name in study.AREA for name in seq) for seq in study.TASKS) == 125
    assert all(len(seq) == 3 for seq in study.TASKS)


def test_launch_gate_refuses_incomplete_test_baseline(tmp_path, monkeypatch) -> None:
    plan_bytes = b"frozen test plan"
    (tmp_path / "plan.json").write_bytes(plan_bytes)
    (tmp_path / "summary.json").write_text(json.dumps({
        "phase": "one_step_and_c0", "successful_edges_by_phase": {"C0": 7, "D1": 70, "D2": 0},
    }))
    (tmp_path / "status.json").write_text(json.dumps({"phase": "one_step_and_c0"}))
    monkeypatch.setattr(study, "TEST", tmp_path)
    monkeypatch.setattr(study, "TEST_PLAN_SHA", hashlib.sha256(plan_bytes).hexdigest())
    with pytest.raises(ValueError, match="not complete and valid"):
        study.test_baseline_ready()


def test_watchdog_scope_excludes_prefix_sibling(tmp_path, monkeypatch) -> None:
    root = tmp_path / "depth3"
    root.mkdir()
    monkeypatch.setattr(study, "ROOT", root)
    monkeypatch.setattr(study, "_own_pids", lambda: (101, 102))
    monkeypatch.setattr(
        study, "_stat",
        lambda pid: SimpleNamespace(state="S", pgrp=10001 if pid == 101 else 10002),
    )
    monkeypatch.setattr(study.os, "readlink", lambda path:
                        str(root / "run") if path == "/proc/101/cwd" else str(tmp_path / "depth3-sibling"))
    monkeypatch.setattr(study, "_tag_of", lambda pid: "a" * 32 if pid == 101 else "b" * 32)
    tags, groups = study.targets()
    assert tags == {"a" * 32}
    assert groups == {10001}
