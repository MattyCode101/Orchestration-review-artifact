"""CLI integration through the labelled fake engine, episodes and report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orch.cli import main
from orch.runtime import _require_deterministic_smoke, _require_latch_venv_smoke


def test_fake_cli_sweep_episodes_and_report(tmp_path: Path, capsys) -> None:
    """One shared fake store serves every CLI decider and a comparison table."""

    base = ["--backend", "fake", "--data-root", str(tmp_path)]
    assert main(["sweep", *base, "--design", "vexriscv_re-place",
                 "--id", "cli_fake_depth2", "--depth", "2"]) == 0
    sweep = json.loads(capsys.readouterr().out)
    assert sweep["fake"] is True
    assert sweep["recipe_set"] == "FAKE_TEST_ONLY_unsigned"
    assert sweep["entries"] == 20
    assert sweep["actual_backend_runs"] == 20

    cases = (
        ("cli_trivial_t", "timing", "trivial"),
        ("cli_random_t", "timing", "random"),
        ("cli_trivial_a", "area", "trivial"),
        ("cli_random_a", "area", "random"),
    )
    for episode_id, brief, policy in cases:
        assert main(["episode", *base, "--design", "vexriscv_re-place",
                     "--id", episode_id, "--brief", brief,
                     "--policy", policy, "--seed", "17"]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["fake"] is True
        assert result["validation_status"] == "skipped_fake"
        assert result["selected_set_signed_off"] is False
        assert 0 <= result["charged_recipe_runs"] <= 6
        assert result["final_state_id"]

    report_args = ["report", *base, "--id", "cli_fake_depth2"]
    for episode_id, _, _ in cases:
        report_args.extend(["--episode-id", episode_id])
    assert main(report_args) == 0
    table = capsys.readouterr().out
    assert table.count("| vexriscv_re-place |") == 6
    assert table.count("| true |") == 6
    assert "| brief_t | trivial |" in table
    assert "| brief_a | random |" in table

    assert main(["run-recipe", *base, "--design", "vexriscv_re-place",
                 "--recipe", "A_R_EA", "--brief", "area"]) == 0
    run = json.loads(capsys.readouterr().out)
    assert run["fake"] is True
    assert run["cache_hit"] is True

    assert main(["measure", *base, "--design", "vexriscv_re-place"]) == 0
    measured = json.loads(capsys.readouterr().out)
    assert measured["evidence"]["fake"] is True

    assert main(["status", *base]) == 0
    status = json.loads(capsys.readouterr().out)
    assert len(status["saved_episodes"]) == 4
    assert "sweep-cli_fake_depth2" in status["saved_sweeps"]


def test_real_cache_gate_requires_complete_deterministic_smoke(tmp_path: Path) -> None:
    """An incomplete or nondeterministic smoke cannot enable real caching."""

    status_path = tmp_path / "smoke.json"
    status_path.write_text(json.dumps({
        "phase": "recipe_running",
        "baseline_match": True,
        "deterministic": None,
        "runs": [],
        "determinism": {},
        "tags": {"tagged_survivor_pids": []},
    }), encoding="utf-8")
    try:
        _require_deterministic_smoke(status_path)
    except RuntimeError:
        pass
    else:
        raise AssertionError("incomplete smoke enabled real caching")

    complete = {
        "phase": "complete",
        "baseline_match": True,
        "deterministic": True,
        "runs": [{}] * 6,
        "determinism": {
            recipe: {"identical_metrics": True}
            for recipe in ("C0_reroute", "LATCH.phys_opt", "A_R_EA")
        },
        "tags": {"tagged_survivor_pids": [12345]},
    }
    status_path.write_text(json.dumps(complete), encoding="utf-8")
    try:
        _require_deterministic_smoke(status_path)
    except RuntimeError:
        pass
    else:
        raise AssertionError("tag survivor enabled real caching")

    complete["tags"]["tagged_survivor_pids"] = []
    status_path.write_text(json.dumps(complete), encoding="utf-8")
    _require_deterministic_smoke(status_path)


def test_real_cache_requires_current_latch_venv_smoke(tmp_path: Path) -> None:
    """A stale or mislabeled venv check cannot enable real run caching."""

    path = tmp_path / "latch_status.json"
    interpreter = "/verified/venv/bin/python"
    row = {
        "run_status": "ok",
        "cache_hit": False,
        "physically_valid": True,
        "metrics": {"wns_ns": -0.785, "tns_ns": -498.982, "packed_luts": 1895, "netlist_luts": 1953},
    }
    status = {
        "phase": "complete",
        "fake": False,
        "cache_enabled": False,
        "latch_python": interpreter,
        "latch_prefix": "/verified/venv",
        "recipe_version": "current-recipe",
        "meter_version": "current-meter",
        "deterministic": True,
        "runs": [{**row, "repeat": 1}, {**row, "repeat": 2}],
        "tags": {"runner_jobs_checked": 8, "tagged_survivor_pids": []},
    }

    def check() -> None:
        path.write_text(json.dumps(status), encoding="utf-8")
        _require_latch_venv_smoke(
            path,
            recipe_version="current-recipe",
            meter_version="current-meter",
            interpreter=interpreter,
        )

    check()
    complete_metrics = dict(row["metrics"])
    for incomplete_or_invalid in (
        {"wns_ns": -0.785},
        {name: None for name in complete_metrics},
        {**complete_metrics, "tns_ns": float("inf")},
        {**complete_metrics, "packed_luts": True},
    ):
        status["runs"][0]["metrics"] = incomplete_or_invalid
        status["runs"][1]["metrics"] = incomplete_or_invalid
        with pytest.raises(RuntimeError, match="corrected LATCH"):
            check()
    status["runs"][0]["metrics"] = complete_metrics
    status["runs"][1]["metrics"] = complete_metrics
    check()

    status["recipe_version"] = "stale-recipe"
    with pytest.raises(RuntimeError, match="corrected LATCH"):
        check()
    status["recipe_version"] = "current-recipe"
    status["latch_python"] = "/usr/bin/python3.12"
    with pytest.raises(RuntimeError, match="corrected LATCH"):
        check()
    status["latch_python"] = interpreter
    status["runs"][1]["metrics"] = {**complete_metrics, "wns_ns": -0.700}
    with pytest.raises(RuntimeError, match="corrected LATCH"):
        check()
