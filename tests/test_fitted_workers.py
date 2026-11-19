"""Synthetic behavior checks for frozen scored workers."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orch.analysis.fit_workers import FROZEN_SHA256, load_feature_spec
from orch.contract.models import DesignState, Mandate, MetricName, Objective, WorkerCard, WorkerSettings
from orch.workers.fitted import FittedWorker

TIMING = (
    "LATCH.phys_opt", "LATCH.retime", "LATCH.auto_pblock",
    "LATCH.hardblock_relocate", "LATCH.memory_path_opt",
)
START = DesignState(id="a" * 64, design="synthetic", path="synthetic://start")


class StubRegistry:
    """Only owner and replace flag are needed for planning."""

    def __init__(self, replacing: set[str] | None = None) -> None:
        self.replacing = replacing or set()

    def get(self, recipe_id: str):
        return SimpleNamespace(worker="timing", replaces=recipe_id in self.replacing)


def card() -> WorkerCard:
    return WorkerCard(
        id="timing", version="test", recipe_set_id="timing",
        objective=Objective(metric=MetricName.WNS_NS, direction="max"),
    )


def mandate(state: DesignState = START, *, replace: bool = True, runs: int = 2) -> Mandate:
    return Mandate(
        id="test", worker_id="timing", base_state_id=state.id,
        settings=WorkerSettings(preserve_placement=not replace, max_runs=runs),
        brief_hash="synthetic",
    )


def write_null(directory: Path) -> None:
    payload = {
        "schema_version": "worker-null-v1-provisional",
        "status": "PROVISIONAL", "worker": "timing",
        "feature_spec_sha256": FROZEN_SHA256,
        "beta_pct_by_recipe": {recipe_id: 1.0 for recipe_id in TIMING},
    }
    (directory / "A_null_PROVISIONAL.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )


def fit_payload() -> dict:
    spec = load_feature_spec()
    recipes = {
        recipe_id: {
            "feature": spec["recipes"][recipe_id]["feature"],
            "beta": 0.0, "w": 0.0, "feature_available": False,
            "feature_mean": None, "feature_sd": None,
            "z_min": None, "z_max": None,
        }
        for recipe_id in TIMING
    }
    recipes["LATCH.retime"].update({
        "feature_available": True, "w": 2.0,
        "feature_mean": 2.0, "feature_sd": 1.0,
        "z_min": -1.0, "z_max": 1.0,
    })
    recipes["LATCH.phys_opt"].update({
        "feature_available": True, "beta": 1.0, "w": 3.0,
        "feature_mean": 0.5, "feature_sd": 0.5,
        "z_min": -1.0, "z_max": 1.0,
    })
    return {
        "schema_version": "worker-fit-v1", "worker": "timing",
        "label": "FINAL", "feature_spec_sha256": FROZEN_SHA256,
        "keep_features": True, "recipes": recipes,
    }


def test_null_filters_replace_before_top_runs_and_breaks_ties(tmp_path: Path) -> None:
    write_null(tmp_path)
    worker = FittedWorker(
        card(), TIMING, StubRegistry({"LATCH.auto_pblock"}), model_dir=tmp_path,
    )
    trace = worker.explain_plan(START, None, mandate(replace=False), ())
    assert trace.model_label == "PROVISIONAL_NULL"
    assert trace.excluded_replacing == ("LATCH.auto_pblock",)
    assert trace.selected_ids == ("LATCH.hardblock_relocate", "LATCH.memory_path_opt")
    assert worker.plan(START, None, mandate(replace=False), ()) == trace.selected_ids


def test_producer_no_repeat_and_missing_producer_fail_closed(tmp_path: Path) -> None:
    write_null(tmp_path)
    worker = FittedWorker(card(), TIMING, StubRegistry(), model_dir=tmp_path)
    child = DesignState(
        id="b" * 64, design="synthetic", parent_id=START.id,
        made_by_run_id="run_producer", path="synthetic://child",
    )
    with pytest.raises(ValueError, match="producer recipe"):
        worker.plan(child, None, mandate(child), ())
    trace = worker.explain_plan(
        child, {"produced_by_recipe_id": "LATCH.hardblock_relocate"},
        mandate(child), (),
    )
    assert trace.excluded_producer == ("LATCH.hardblock_relocate",)
    assert "LATCH.hardblock_relocate" not in trace.selected_ids


def test_final_fit_clips_and_missing_feature_uses_beta_with_flag(tmp_path: Path) -> None:
    write_null(tmp_path)
    (tmp_path / "A_fit.json").write_text(json.dumps(fit_payload()), encoding="utf-8")
    worker = FittedWorker(card(), TIMING, StubRegistry(), model_dir=tmp_path)
    telemetry = {"state_sha256": START.id, "wpath": {"logic_levels": 4}}
    trace = worker.explain_plan(START, {"telemetry": telemetry}, mandate(), ())
    scores = {row.recipe_id: row for row in trace.ranked}
    assert trace.model_label == "FINAL"
    assert trace.selected_ids == ("LATCH.retime", "LATCH.phys_opt")
    assert scores["LATCH.retime"].score_pct == pytest.approx(2.0)
    assert scores["LATCH.retime"].feature_value == 4.0
    assert scores["LATCH.phys_opt"].score_pct == pytest.approx(1.0)
    assert scores["LATCH.phys_opt"].incomplete_measurement
    assert trace.incomplete_measurement
    with pytest.raises(ValueError, match="different input state"):
        worker.plan(START, {"telemetry": {"state_sha256": "b" * 64}}, mandate(), ())


def test_provisional_fit_uses_null_and_bad_final_hash_fails(tmp_path: Path) -> None:
    write_null(tmp_path)
    fit = fit_payload()
    fit["label"] = "PROVISIONAL"
    (tmp_path / "A_fit.json").write_text(json.dumps(fit), encoding="utf-8")
    worker = FittedWorker(card(), TIMING, StubRegistry(), model_dir=tmp_path)
    assert worker.model_label == "PROVISIONAL_NULL"
    fit["label"] = "FINAL"
    fit["feature_spec_sha256"] = "0" * 64
    (tmp_path / "A_fit.json").write_text(json.dumps(fit), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen worker"):
        FittedWorker(card(), TIMING, StubRegistry(), model_dir=tmp_path)

def test_project_null_models_load_for_both_signed_workers(tmp_path: Path) -> None:
    """Exercise saved null models even after final fits have been published."""

    from orch.recipes.registry import load_registry

    saved_models = Path(__file__).resolve().parents[1] / "configs/workers"
    for name in ("A_null_PROVISIONAL.json", "B_null_PROVISIONAL.json"):
        (tmp_path / name).write_bytes((saved_models / name).read_bytes())
    registry = load_registry()
    area_ids = (
        "A_R_EA", "A_R_RSA_SRLM", "A_R_RSA_AREM",
        "A_R_RSA_MED", "A_E_DUPL_Q1000",
    )
    for worker_id, ids, metric, direction, expected in (
        ("timing", TIMING, MetricName.WNS_NS, "max", "LATCH.hardblock_relocate"),
        ("area", area_ids, MetricName.PACKED_LUTS, "min", "A_R_RSA_SRLM"),
    ):
        worker_card = WorkerCard(
            id=worker_id, version="test", recipe_set_id=worker_id,
            objective=Objective(metric=metric, direction=direction),
        )
        worker = FittedWorker(worker_card, ids, registry, model_dir=tmp_path)
        worker_mandate = Mandate(
            id="test", worker_id=worker_id, base_state_id=START.id,
            settings=WorkerSettings(preserve_placement=False, max_runs=1),
            brief_hash="synthetic",
        )
        trace = worker.explain_plan(START, None, worker_mandate, ())
        assert trace.model_label == "PROVISIONAL_NULL"
        assert trace.selected_ids == (expected,)
