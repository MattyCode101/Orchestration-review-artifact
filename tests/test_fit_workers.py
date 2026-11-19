"""Synthetic checks for the frozen five-design worker estimator."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace

import pytest

from orch.analysis import fit_workers as fit


def _ids(worker: str) -> tuple[str, ...]:
    spec = fit.load_feature_spec()
    return tuple(sorted(key for key, value in spec["recipes"].items()
                        if value["worker"] == worker))


def _rows(
    worker: str,
    *,
    states_per_design: tuple[int, ...] = (2, 2, 2, 2, 2),
    gains=None,
) -> list[fit.TrainingRow]:
    recipe_ids = _ids(worker)
    rows: list[fit.TrainingRow] = []
    for design_index, count in enumerate(states_per_design):
        design = f"design_{design_index}"
        for state_index in range(count):
            x = -1.0 if state_index % 2 == 0 else 1.0
            state = f"state_{design_index}_{state_index}"
            features = {recipe_id: x for recipe_id in recipe_ids}
            for recipe_index, recipe_id in enumerate(recipe_ids):
                gain = gains(design_index, state_index, recipe_index, x) if gains else 0.0
                rows.append(fit.TrainingRow(
                    design=design,
                    input_state_id=state,
                    recipe_id=recipe_id,
                    sequence=(state, recipe_id),
                    output_state_id=f"out_{state}_{recipe_index}",
                    run_id=f"run_{state}_{recipe_index}",
                    phase="D2",
                    timing_gain_pct=gain,
                    area_gain_pct=gain,
                    features=features,
                ))
    return rows


def _predictive_gain(_, __, recipe_index: int, x: float) -> float:
    if recipe_index == 0:
        return 5.0 + 3.0 * x
    if recipe_index == 1:
        return 5.0 - 3.0 * x
    return 0.0


def test_predictive_feature_kept_and_clipped() -> None:
    model = fit.fit_worker(_rows("timing", gains=_predictive_gain), "timing")
    recipe_id = _ids("timing")[0]
    assert model["keep_features"] is True
    assert model["improved_folds"] == 5
    assert model["regret_feature"] < model["regret_null"]
    assert model["recipes"][recipe_id]["w"] > 0
    upper, _ = fit.score_recipe(model, recipe_id, 1.0)
    beyond, _ = fit.score_recipe(model, recipe_id, 100.0)
    assert beyond == pytest.approx(upper)
    assert model["recipes"][recipe_id]["z_min"] < model["recipes"][recipe_id]["z_max"]


def test_pure_noise_dropped_when_null_regret_is_already_zero() -> None:
    def constant_gain(_, __, recipe_index: int, ___: float) -> float:
        return 10.0 if recipe_index == 0 else 9.0 if recipe_index == 1 else 0.0

    model = fit.fit_worker(_rows("area", gains=constant_gain), "area")
    assert model["keep_features"] is False
    assert model["regret_null"] == 0
    assert model["regret_feature"] == 0
    assert all(recipe["w"] == 0 for recipe in model["recipes"].values())


def test_duplicate_inputs_are_removed_and_conflicts_refused() -> None:
    rows = _rows("area", states_per_design=(1, 1, 1, 1, 1))
    duplicate = replace(rows[0], sequence=("other_parent", rows[0].recipe_id),
                        run_id="second_run")
    unique, counts = fit.deduplicate_rows(rows + [duplicate])
    assert len(unique) == len(rows)
    assert counts["unique_input_recipe_rows"] == len(rows)
    conflicting = replace(duplicate, output_state_id="different_output")
    with pytest.raises(ValueError, match="conflicting duplicate"):
        fit.deduplicate_rows(rows + [conflicting])


def test_each_design_has_equal_total_weight() -> None:
    def one_design_gain(design_index: int, _, recipe_index: int, __: float) -> float:
        return 10.0 if design_index == 0 and recipe_index == 0 else 0.0

    rows = _rows("area", states_per_design=(20, 1, 1, 1, 1), gains=one_design_gain)
    first = _ids("area")[0]
    rows = [replace(row, features={**row.features,
                                   first: 10.0 if row.design == "design_0" else 0.0})
            for row in rows]
    model = fit.fit_worker(rows, "area")
    assert model["recipes"][first]["beta"] == pytest.approx(2.0)
    assert model["recipes"][first]["feature_mean"] == pytest.approx(2.0)
    assert model["states_by_design"]["design_0"] == 20


def test_feature_missing_on_more_than_ten_percent_of_states_uses_beta() -> None:
    rows = _rows("area")
    first = _ids("area")[0]
    missing_states = {"state_0_0", "state_1_0"}
    rows = [replace(row, features={**row.features, first: None})
            if row.input_state_id in missing_states else row for row in rows]
    model = fit.fit_worker(rows, "area")
    params = model["recipes"][first]
    assert params["feature_available"] is False
    assert params["feature_missing_fraction"] == pytest.approx(0.2)
    assert params["feature_drop_reason"] == "more_than_10_percent_missing"
    assert params["w"] == 0


def test_ties_break_alphabetically_and_missing_feature_uses_beta() -> None:
    model = fit.fit_worker(_rows("timing", gains=_predictive_gain), "timing")
    first = _ids("timing")[0]
    score, incomplete = fit.score_recipe(model, first, None)
    assert score == pytest.approx(model["recipes"][first]["beta"])
    assert incomplete is True
    tie = {
        "schema_version": fit.SCHEMA,
        "feature_spec_sha256": fit.FROZEN_SHA256,
        "keep_features": False,
        "recipes": {
            "z_recipe": {"beta": 1.0, "feature_available": False},
            "a_recipe": {"beta": 1.0, "feature_available": False},
        },
    }
    ranking, incomplete = fit.rank_recipes(tie, {"z_recipe": 1.0, "a_recipe": 1.0})
    assert ranking == ("a_recipe", "z_recipe")
    assert incomplete is False


def test_invalid_fail_no_output_and_unchanged_gain_are_zero() -> None:
    parent = {"metrics": {"wns_ns": {"value": -0.5}, "packed_luts": {"value": 100}}}
    child = {"metrics": {"wns_ns": {"value": -0.3}, "packed_luts": {"value": 90}}}
    expected = fit.target_gain(parent, child, 1.5, physically_valid=True,
                               amd_result="pass", output_identical_to_input=False)
    assert expected["timing"] == pytest.approx(11.1111111111)
    assert expected["area"] == pytest.approx(10.0)
    for physically_valid, amd_result, output_identical, candidate in (
        (False, "pass", False, child),
        (True, "fail", False, child),
        (True, "pass", True, child),
        (True, "pass", False, None),
    ):
        assert fit.target_gain(parent, candidate, 1.5,
                               physically_valid=physically_valid,
                               amd_result=amd_result,
                               output_identical_to_input=output_identical) == {
                                   "timing": 0.0, "area": 0.0,
                               }
    assert fit.target_gain(parent, child, 1.5, physically_valid=True,
                           amd_result="pass", output_identical_to_input=False,
                           no_output=True) == {"timing": 0.0, "area": 0.0}


def test_incomplete_five_recipe_table_is_refused() -> None:
    rows = _rows("timing")
    with pytest.raises(ValueError, match="incomplete five-recipe"):
        fit.fit_worker(rows[1:], "timing")


def test_feature_paths_and_missing_values_use_frozen_spec() -> None:
    spec = fit.load_feature_spec()
    record = {
        "metrics": {
            "wns_ns": {"value": -0.5},
            "packed_luts": {"value": 100},
            "netlist_luts": {"value": 120},
        },
        "wpath": {
            "requirement_ns": 1.5,
            "logic_levels": 3,
            "tile_span": {"x_max": 20, "x_min": 16, "y_max": 80, "y_min": 74},
            "net_delay_ns": 1.0,
            "data_path_delay_ns": 2.0,
        },
        "probe": {
            "pblock_sizing_resources": {"luts": 60},
            "utilization_summary": {
                "Block RAM Tile": {"used": 2},
                "DSPs": {"used": 1},
                "URAM": {"used": 0},
                "CLB Registers": {"used": 30},
            },
            "lut_as_shift_register": 10,
            "lut_primitive_counts": {
                "LUT1": 1, "LUT2": 2, "LUT3": 3,
                "LUT4": 4, "LUT5": 5, "LUT6": 5,
            },
            "dup_driver_count": 6,
        },
    }
    assert fit.extract_feature(record, "LATCH.phys_opt", spec) == pytest.approx(1 / 3)
    assert fit.extract_feature(record, "LATCH.retime", spec) == 3
    assert fit.extract_feature(record, "LATCH.auto_pblock", spec) == 10
    assert fit.extract_feature(record, "LATCH.hardblock_relocate", spec) == 0.5
    assert fit.extract_feature(record, "LATCH.memory_path_opt", spec) == 20
    assert fit.extract_feature(record, "A_R_EA", spec) == 2
    assert fit.extract_feature(record, "A_R_RSA_SRLM", spec) == 0.1
    assert fit.extract_feature(record, "A_R_RSA_AREM", spec) == 0.3
    assert fit.extract_feature(record, "A_R_RSA_MED", spec) == 0.05
    assert fit.extract_feature(record, "A_E_DUPL_Q1000", spec) == 0.05
    del record["probe"]["dup_driver_count"]
    assert fit.extract_feature(record, "A_R_RSA_MED", spec) is None


def test_yaml_hash_and_training_root_are_fail_closed(tmp_path, monkeypatch) -> None:
    rows = _rows("timing")
    changed = tmp_path / "worker_features.yaml"
    changed.write_bytes(fit.FEATURE_FILE.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="freeze SHA-256 differs"):
        fit.load_feature_spec(changed)
    monkeypatch.setattr(fit, "FEATURE_FILE", changed)
    with pytest.raises(ValueError, match="freeze SHA-256 differs"):
        fit.fit_worker(rows, "timing")
    with pytest.raises(ValueError, match="outside the training allow-list"):
        fit._guard_training_root(tmp_path)


def test_provisional_unknown_is_allowed_but_fail_still_zero() -> None:
    assert fit._target_amd_result(None, provisional=True) == "provisional"
    assert fit._target_amd_result("unknown", provisional=True) == "provisional"
    assert fit._target_amd_result("fail", provisional=True) == "fail"
    with pytest.raises(ValueError, match="AMD PASS required"):
        fit._target_amd_result("unknown", provisional=False)


def test_data_path_symlink_cannot_escape_training_root(tmp_path) -> None:
    root = tmp_path / "training"
    root.mkdir()
    outside = tmp_path / "held_out.json"
    outside.write_text("{}", encoding="utf-8")
    pointer = root / "pointer.json"
    pointer.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes allow-list"):
        fit._read_json(pointer, root)


def test_d3_requires_phase_complete_before_fitting() -> None:
    events = [
        {"type": "edge", "edge": {"first_phase": "D1"}},
        {"type": "edge", "edge": {"first_phase": "D2"}},
        {"type": "edge", "edge": {"first_phase": "D3"}},
        {"type": "phase_complete", "phase": "D1"},
        {"type": "phase_complete", "phase": "D2"},
    ]
    with pytest.raises(ValueError, match="D3"):
        fit._require_phase_completion(events)
    events.append({"type": "phase_complete", "phase": "D3"})
    fit._require_phase_completion(events)


def _training_v3_fixture(tmp_path, monkeypatch):
    """Build metadata-only source and training stores with five fake names."""

    source = tmp_path / "named_training_v3"
    training = tmp_path / "training"
    source.mkdir()
    training.mkdir()
    monkeypatch.setattr(fit, "TRAINING_V3_SOURCE", source)
    starts = {design: f"{index + 1:064x}"
              for index, design in enumerate(fit.TRAINING_DESIGNS)}
    outputs = {(design, f"{index + 101:064x}")
               for index, design in enumerate(fit.TRAINING_DESIGNS)}
    logical = ([{"kind": "start", "design": design, "state_sha256": state_id}
                for design, state_id in starts.items()]
               + [{"kind": "output", "design": design, "state_sha256": state_id}
                  for design, state_id in sorted(outputs)])
    states = []
    for _, design, state_id in ((row["kind"], row["design"], row["state_sha256"])
                                for row in logical):
        dcp = training / "states" / state_id / "design.dcp"
        dcp.parent.mkdir(parents=True)
        dcp.write_bytes(b"synthetic checkpoint")
        evidence = training / "evidence" / f"{state_id}.json"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_bytes(b"{}")
        states.append({
            "design": design,
            "state_sha256": state_id,
            "dcp_path": str(dcp),
            "evidence_path": str(evidence),
            "evidence_sha256": hashlib.sha256(b"{}").hexdigest(),
        })
    plan = {
        "schema_version": "exp-evening-training-telemetry-v1",
        "source_campaign": "worker_fitting_train",
        "source_first_edge_count": 708,
        "meter_version": "meter-v1",
        "parser_sha256": fit.TRAINING_V3_PARSER_SHA256,
        "distinct_output_dcps": len(outputs),
        "logical_rows": len(logical),
        "unique_state_dcps": len(states),
        "logical_entries": logical,
        "states": states,
    }
    (source / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    index_path = source / "logical_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "design", "state_sha256",
                                                   "telemetry_json"])
        writer.writeheader()
        for row in logical:
            writer.writerow({**row, "telemetry_json": str(source / "states" /
                                                              row["state_sha256"] /
                                                              "dcp_telemetry.json")})
    return source, training, starts, outputs, plan, index_path


def test_named_v3_source_accepts_exact_training_plan_and_index(tmp_path, monkeypatch) -> None:
    source, training, starts, outputs, _, _ = _training_v3_fixture(tmp_path, monkeypatch)
    required = set(starts.items()) | outputs
    record_dir, hashes = fit._validate_training_v3_source(
        source, training, starts, outputs, required, "meter-v1"
    )
    assert record_dir == source / "states"
    assert hashes["telemetry_source_kind"] == "training-v3"
    assert len(hashes["telemetry_source_plan_sha256"]) == 64


def test_named_v3_source_pins_extractor_hash(tmp_path, monkeypatch) -> None:
    source, training, starts, outputs, plan, _ = _training_v3_fixture(
        tmp_path, monkeypatch
    )
    plan["parser_sha256"] = "f" * 64
    (source / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    required = set(starts.items()) | outputs
    with pytest.raises(ValueError, match="scope or source version"):
        fit._validate_training_v3_source(source, training, starts, outputs,
                                         required, "meter-v1")


def test_named_v3_source_refuses_held_out_design_and_rogue_index(tmp_path, monkeypatch) -> None:
    source, training, starts, outputs, plan, index_path = _training_v3_fixture(
        tmp_path, monkeypatch
    )
    required = set(starts.items()) | outputs
    plan["logical_entries"][0]["design"] = "held_out_design"
    (source / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="five-design scope"):
        fit._validate_training_v3_source(source, training, starts, outputs,
                                         required, "meter-v1")
    plan["logical_entries"][0]["design"] = next(iter(starts))
    (source / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    rows = list(csv.DictReader(index_path.open(encoding="utf-8", newline="")))
    rows[0]["telemetry_json"] = str(tmp_path / "held_out_design" / "dcp_telemetry.json")
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "design", "state_sha256",
                                                   "telemetry_json"])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="named source layout"):
        fit._validate_training_v3_source(source, training, starts, outputs,
                                         required, "meter-v1")


def test_named_v3_source_requires_all_fit_inputs_and_no_symlink(tmp_path, monkeypatch) -> None:
    source, training, starts, outputs, _, _ = _training_v3_fixture(tmp_path, monkeypatch)
    required = set(starts.items()) | outputs | {(next(iter(starts)), f"{999:064x}")}
    with pytest.raises(ValueError, match="lacks an expected fit input"):
        fit._validate_training_v3_source(source, training, starts, outputs,
                                         required, "meter-v1")
    alias = tmp_path / "source_alias"
    alias.symlink_to(source, target_is_directory=True)
    monkeypatch.setattr(fit, "TRAINING_V3_SOURCE", alias)
    with pytest.raises(ValueError, match="symlinked"):
        fit._validate_training_v3_source(alias, training, starts, outputs,
                                         set(starts.items()), "meter-v1")


def _synthetic_training_v3_pair() -> tuple[str, fit.Evidence, dict]:
    state_id = "a" * 64
    meter = fit.Evidence.model_validate({
        "id": "evidence", "state_id": state_id, "routed": True,
        "route_errors": 0, "drc_errors": 0,
        "meter_version": "meter-v1", "constraint_fingerprint": "constraint-a",
        "clock_digest": "clock-a",
        "metrics": {
            "wns_ns": {"value": -0.5, "unit": "ns", "method": "meter"},
            "packed_luts": {"value": 100, "unit": "count", "method": "meter"},
        },
    })
    telemetry = {
        "metrics": {"wns_ns": {"value": -0.5}, "packed_luts": {"value": 100}},
        "provenance": {"extractor_sha256": "f" * 64},
        "validity": {
            "constraint_fingerprint": "constraint-a",
            "clock_digest": "clock-a",
            "routed": True,
            "route_errors": 0,
            "drc_errors": 0,
        },
    }
    return state_id, meter, telemetry


def test_training_v3_telemetry_validity_matches_meter() -> None:
    state_id, meter, telemetry = _synthetic_training_v3_pair()
    fit._check_telemetry_meter(telemetry, meter, state_id, check_validity=True,
                               expected_extractor_sha256="f" * 64)
    telemetry["provenance"]["extractor_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="extractor hash differs"):
        fit._check_telemetry_meter(telemetry, meter, state_id, check_validity=True,
                                   expected_extractor_sha256="f" * 64)


@pytest.mark.parametrize("field", (
    "constraint_fingerprint", "clock_digest", "routed", "route_errors", "drc_errors",
))
def test_training_v3_telemetry_refuses_missing_validity_field(field: str) -> None:
    state_id, meter, telemetry = _synthetic_training_v3_pair()
    del telemetry["validity"][field]
    with pytest.raises(ValueError, match=field):
        fit._check_telemetry_meter(telemetry, meter, state_id, check_validity=True,
                                   expected_extractor_sha256="f" * 64)


@pytest.mark.parametrize(("field", "wrong"), (
    ("constraint_fingerprint", None),
    ("clock_digest", None),
    ("clock_digest", "different-clock"),
    ("routed", False),
    ("route_errors", 1),
    ("drc_errors", 1),
    ("route_errors", False),
))
def test_training_v3_telemetry_refuses_validity_mismatch(
    field: str, wrong: object
) -> None:
    state_id, meter, telemetry = _synthetic_training_v3_pair()
    telemetry["validity"][field] = wrong
    with pytest.raises(ValueError, match=field):
        fit._check_telemetry_meter(telemetry, meter, state_id, check_validity=True,
                                   expected_extractor_sha256="f" * 64)


def test_required_telemetry_is_complete_and_cannot_escape(tmp_path) -> None:
    directory = tmp_path / "telemetry"
    first, second = "a" * 64, "b" * 64
    first_path = directory / first / "dcp_telemetry.json"
    first_path.parent.mkdir(parents=True)
    first_path.write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing 1/2"):
        fit._require_telemetry_files(directory, tmp_path, {first, second})
    outside = tmp_path.parent / f"external_{tmp_path.name}.json"
    outside.write_text("{}", encoding="utf-8")
    second_path = directory / second / "dcp_telemetry.json"
    second_path.parent.mkdir(parents=True)
    second_path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes allow-list"):
        fit._require_telemetry_files(directory, tmp_path, {first, second})
