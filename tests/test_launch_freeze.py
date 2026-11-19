"""Gate G8: the test scope launches only the frozen configuration. No tool, no network, nothing is launched."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import campaign_run as run  # noqa: E402
import test_campaign as tc_run  # noqa: E402  (the module campaign_run imports)
from scripts import test_campaign as tc  # noqa: E402

ARGS = dict(name="unit", scope="test", freeze_approved=None, designs=None, briefs=None, repeats=None, approaches=None,
            shards=None, cap_runs=None, cap_job_hours=None, cap_tokens=None, cap_usd=None, cap_extraction_jobs=None,
            telemetry=None)


@pytest.fixture()
def frozen(tmp_path: Path, monkeypatch) -> Path:
    record = tc.freeze_record()
    record["status"] = "FINAL: unit-test copy of the draft"
    path = tmp_path / "freeze_final.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(tc, "FINAL_FREEZE", path)
    monkeypatch.setattr(tc_run, "FINAL_FREEZE", path)
    return path


def _args(**over) -> SimpleNamespace:
    return SimpleNamespace(**{**ARGS, **over})


def test_the_test_scope_cannot_be_planned_without_a_freeze() -> None:
    with pytest.raises(run.pilot.PilotSafetyStop, match="approved freeze"):
        run.build_plan(_args(), Path("/nonexistent"))


@pytest.mark.parametrize("flag", [{"cap_runs": 900}, {"cap_usd": 100.0}, {"approaches": "v2"}, {"repeats": 3},
                                  {"designs": "rosetta_digit-recognition"}, {"briefs": "T0"}, {"telemetry": "saved"},
                                  {"cap_job_hours": 500.0}, {"cap_tokens": 10**9}, {"cap_extraction_jobs": 5}])
def test_the_test_scope_refuses_every_flag_that_would_change_the_configuration(frozen, flag) -> None:
    with pytest.raises(run.pilot.PilotSafetyStop, match="only from the freeze"):
        run.build_plan(_args(freeze_approved=str(frozen), **flag), Path("/nonexistent"))


def test_the_plan_a_freeze_produces_is_accepted_and_every_deviation_is_refused(frozen) -> None:
    plan = run.build_plan(_args(freeze_approved=str(frozen)), Path("/nonexistent"))
    assert plan["telemetry"] == "live" and plan["extraction_jobs"] == tc.EXTRACTION_JOBS
    assert plan["caps"] == {"new_real_runs": 800, "job_hours": 240.0, "tokens": tc.caps()["tokens"], "usd": 50.0}
    assert len(plan["cells"]) == 370
    tc.verify_plan_against_freeze(frozen, plan)
    mutations = {
        "caps": lambda p: p["caps"].__setitem__("new_real_runs", 801),
        "cells": lambda p: p["cells"].pop(),
        "cell order": lambda p: p["cells"].reverse(),
        "telemetry": lambda p: p.__setitem__("telemetry", "saved"),
        "model": lambda p: p.__setitem__("model", "gemini-4-argon"),
        "manifest": lambda p: p.__setitem__("manifest_sha256", "0" * 64),
        "extraction cap": lambda p: p.__setitem__("extraction_jobs", 5),
        "scope": lambda p: p.__setitem__("scope", "train"),
        "repeats": lambda p: p.__setitem__("repeats", 3),
    }
    for label, mutate in mutations.items():
        altered = copy.deepcopy(plan)
        mutate(altered)
        with pytest.raises(RuntimeError, match="not the frozen configuration"):
            tc.verify_plan_against_freeze(frozen, altered)


def test_a_changed_frozen_file_or_a_draft_or_another_path_refuses(frozen, tmp_path, monkeypatch) -> None:
    saved = json.loads(frozen.read_text())
    saved["files"]["configs/briefs.yaml"] = "0" * 64
    frozen.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(RuntimeError, match="freeze check fails"):
        tc.frozen_settings(frozen)
    saved["files"]["configs/briefs.yaml"] = tc._sha256_file(tc.PROJECT / "configs/briefs.yaml")
    saved["status"] = "DRAFT, not signed"
    frozen.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(RuntimeError, match="draft"):
        tc.frozen_settings(frozen)
    other = tmp_path / "copy.json"
    other.write_text(frozen.read_text(), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not the final freeze file"):
        tc.frozen_settings(other)


def test_the_launch_re_verifies_the_freeze_for_the_test_scope(frozen, tmp_path, monkeypatch) -> None:
    plan = run.build_plan(_args(freeze_approved=str(frozen)), Path("/nonexistent"))
    monkeypatch.setattr(run, "load_plan", lambda name: plan)
    with pytest.raises(run.pilot.PilotSafetyStop, match="approved freeze"):
        run.launch("unit", resume=False, freeze_ok=False)
    with pytest.raises(run.pilot.PilotSafetyStop, match="approved freeze"):
        run.launch("unit", resume=False, freeze_ok=True, freeze_path=None)
    altered = copy.deepcopy(plan)
    altered["caps"]["usd"] = 51.0
    monkeypatch.setattr(run, "load_plan", lambda name: altered)
    with pytest.raises(RuntimeError, match="not the frozen configuration"):
        run.launch("unit", resume=False, freeze_ok=True, freeze_path=str(frozen))


def test_no_stored_telemetry_is_ever_used_on_test() -> None:
    plan = {"scope": "test", "telemetry": "saved", "manifest_sha256": run._sha256_file(run.TEST_MANIFEST),
            "caps": {}, "extraction_jobs": 1}
    with pytest.raises(run.pilot.PilotSafetyStop, match="no stored telemetry"):
        run.scope_for(plan, Path("/nonexistent"))
