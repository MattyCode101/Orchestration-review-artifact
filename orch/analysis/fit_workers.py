"""Frozen, training-only fits for the two single-objective workers.

The estimator is pure once supplied TrainingRows. The CLI reads only the
allow-listed five-design campaign and publishes models after every input gate
has passed. It never starts a hardware job or follows a raw report path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from orch.contract.models import Evidence, RunStatus
from orch.engine.journal import Journal
from orch.engine.store import StateStore, _atomic_bytes, _json_bytes


PROJECT = Path(__file__).resolve().parents[2]
FEATURE_FILE = PROJECT / "configs/worker_features.yaml"
TRAINING_ROOT = PROJECT / "data/sweeps/worker_fitting_train"
TRAINING_V3_SOURCE = PROJECT / "data/exp_evening_2026-11-04/training_telemetry_v3"
TRAINING_V3_PARSER_SHA256 = "6a0f1f0c287a710bbd565a377826b9007a0ee39ce29a7a307934629705827112"
FROZEN_SHA256 = "d2f47ed3c9ac708c8f50c1b68fd66254118b04af02d564fc409d900d48d8086d"
SCHEMA = "worker-fit-v1"
_HASH64 = re.compile(r"^[0-9a-f]{64}$")
TRAINING_DESIGNS = (
    "finn_radioml",
    "logicnets_jscl",
    "rosetta_3d-rendering",
    "rosetta_spam-filter",
    "vexriscv_re-place",
)
# Audited from each B2 Evidence-linked final_timing_summary.rpt, with matching
# constraint and clock digests on all 708 audited edges. This is a fixed clock
# input; measured WNS and LUT outcomes still come only from Evidence.
# Source: notes/training_data_audit.md, Metric and constraint completeness.
B2_REQUIREMENTS_NS = {
    "vexriscv_re-place": 1.570,
    "rosetta_spam-filter": 1.600,
    "rosetta_3d-rendering": 1.538,
    "logicnets_jscl": 1.500,
    "finn_radioml": 1.600,
}


@dataclass(frozen=True)
class TrainingRow:
    """One unique input/recipe attempt, with targets from meter Evidence."""

    design: str
    input_state_id: str
    recipe_id: str
    sequence: tuple[str, ...]
    output_state_id: str | None
    run_id: str
    phase: str
    timing_gain_pct: float
    area_gain_pct: float
    features: Mapping[str, float | None]
    physically_valid: bool = True
    amd_result: str = "pass"

    def gain(self, worker: str) -> float:
        """Return this worker's registered meter target."""

        if worker == "timing":
            return self.timing_gain_pct
        if worker == "area":
            return self.area_gain_pct
        raise ValueError(f"unknown worker {worker}")


@dataclass(frozen=True)
class TrainingDataset:
    """Checked journal snapshot and provenance hashes for one fit invocation."""

    rows: tuple[TrainingRow, ...]
    data_hashes: Mapping[str, str]
    counts: Mapping[str, int]
    provisional: bool


def _sha256(path: Path) -> str:
    """Hash a retained source without interpreting its contents."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_pairs(pairs: Sequence[tuple[str, str]]) -> str:
    """Compact, deterministic provenance for a checked collection of files."""

    return hashlib.sha256(_json_bytes(sorted(pairs))).hexdigest()


def load_feature_spec(path: Path | None = None) -> dict[str, Any]:
    """Read the preregistered YAML only when its exact bytes match the freeze."""

    source = path or FEATURE_FILE
    raw = source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != FROZEN_SHA256:
        raise ValueError(f"worker feature freeze SHA-256 differs: {actual}")
    spec = yaml.safe_load(raw)
    if not isinstance(spec, dict) or spec.get("schema_version") != "worker-features-v1":
        raise ValueError("worker feature YAML has an unsupported schema")
    return spec


def _number(value: object) -> float | None:
    """Accept only finite JSON numbers, never booleans or numeric text."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _at(record: Mapping[str, Any], path: Sequence[str]) -> object:
    current: object = record
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def extract_feature(
    telemetry_record: Mapping[str, Any], recipe_id: str, spec: Mapping[str, Any]
) -> float | None:
    """Extract one frozen input-state feature, or None when it is unavailable.

    This pure helper performs no file or data-root access. The caller must use
    load_feature_spec() for the frozen YAML, and check telemetry state identity.
    """

    recipe = spec["recipes"][recipe_id]
    feature = recipe["feature"]
    inputs = {
        name: _number(_at(telemetry_record, path))
        for name, path in recipe["inputs"].items()
    }
    if feature == "violation_depth":
        wns = inputs["wns_ns"]
        if wns is None:
            return None
        if wns >= 0:
            return 0.0
        requirement = inputs["requirement_ns"]
        return -wns / requirement if requirement is not None and requirement > 0 else None
    if any(value is None for value in inputs.values()):
        return None
    value: float
    if feature == "worst_path_logic_levels":
        value = inputs["logic_levels"]
    elif feature == "worst_path_tile_span":
        value = (inputs["x_max"] - inputs["x_min"]
                 + inputs["y_max"] - inputs["y_min"])
    elif feature == "worst_path_route_share":
        denominator = inputs["data_path_delay_ns"]
        value = inputs["net_delay_ns"] / denominator if denominator > 0 else math.nan
    elif feature == "soft_luts_per_hardblock":
        denominator = max(1.0, inputs["bram_tiles"] + inputs["dsps"] + inputs["urams"])
        value = inputs["logic_luts"] / denominator
    elif feature == "luts_per_ff":
        denominator = inputs["clb_registers"]
        value = inputs["logic_luts"] / denominator if denominator > 0 else math.nan
    elif feature == "srl_share":
        denominator = inputs["packed_clb_luts"]
        value = inputs["lut_as_shift_register"] / denominator if denominator > 0 else math.nan
    elif feature == "small_lut_share":
        denominator = sum(inputs[f"LUT{index}"] for index in range(1, 7))
        value = (sum(inputs[f"LUT{index}"] for index in range(1, 4)) / denominator
                 if denominator > 0 else math.nan)
    elif feature == "dup_driver_share":
        denominator = inputs["netlist_lut_cells"]
        value = inputs["dup_driver_count"] / denominator if denominator > 0 else math.nan
    else:
        raise ValueError(f"unregistered frozen feature {feature}")
    return value if math.isfinite(value) and value >= 0 else None


def extract_features(
    telemetry_record: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, float | None]:
    """Return feature values keyed by exact signed recipe ID."""

    return {
        recipe_id: extract_feature(telemetry_record, recipe_id, spec)
        for recipe_id in spec["recipes"]
    }


def target_gain(
    parent: Mapping[str, Any],
    child: Mapping[str, Any] | None,
    requirement_ns: float,
    *,
    physically_valid: bool,
    amd_result: str,
    output_identical_to_input: bool,
    no_output: bool = False,
) -> dict[str, float]:
    """Compute both registered gains from meter Evidence only.

    The audited clock requirement is a fixed denominator after digest checks.
    Invalid, AMD FAIL, no-output and unchanged-output attempts get zero gain.
    """

    if (not physically_valid or child is None or no_output or amd_result == "fail"
            or output_identical_to_input):
        return {"timing": 0.0, "area": 0.0}
    if amd_result not in {"pass", "provisional"}:
        raise ValueError(f"unverified AMD result {amd_result}")
    requirement = _number(requirement_ns)
    parent_wns = _number(_at(parent, ("metrics", "wns_ns", "value")))
    child_wns = _number(_at(child, ("metrics", "wns_ns", "value")))
    parent_luts = _number(_at(parent, ("metrics", "packed_luts", "value")))
    child_luts = _number(_at(child, ("metrics", "packed_luts", "value")))
    if None in (requirement, parent_wns, child_wns, parent_luts, child_luts):
        raise ValueError("meter Evidence lacks a required target number")
    parent_period = requirement - parent_wns
    child_period = requirement - child_wns
    if parent_period <= 0 or child_period <= 0 or parent_luts <= 0 or child_luts < 0:
        raise ValueError("meter Evidence contains an invalid Fmax or LUT denominator")
    parent_fmax = 1000.0 / parent_period
    child_fmax = 1000.0 / child_period
    return {
        "timing": 100.0 * (child_fmax - parent_fmax) / parent_fmax,
        "area": 100.0 * (parent_luts - child_luts) / parent_luts,
    }


def deduplicate_rows(
    rows: Sequence[TrainingRow],
) -> tuple[tuple[TrainingRow, ...], dict[str, int]]:
    """Keep latest sequence attempts, then one stable input/recipe outcome.

    Repeated input/recipe outcomes must agree, otherwise the training target
    is ambiguous and the fit stops rather than selecting a favorable repeat.
    """

    latest: dict[tuple[str, tuple[str, ...]], TrainingRow] = {}
    for row in rows:
        latest[(row.design, row.sequence)] = row
    unique: dict[tuple[str, str], TrainingRow] = {}
    for row in sorted(latest.values(), key=lambda r: (r.design, r.sequence)):
        key = (row.input_state_id, row.recipe_id)
        prior = unique.get(key)
        if prior is None:
            unique[key] = row
            continue
        same = (
            prior.design == row.design
            and prior.output_state_id == row.output_state_id
            and prior.physically_valid == row.physically_valid
            and prior.amd_result == row.amd_result
            and math.isclose(prior.timing_gain_pct, row.timing_gain_pct, abs_tol=1e-12)
            and math.isclose(prior.area_gain_pct, row.area_gain_pct, abs_tol=1e-12)
            and dict(prior.features) == dict(row.features)
        )
        if not same:
            raise ValueError(f"conflicting duplicate input/recipe outcome: {key}")
    ordered = tuple(sorted(unique.values(), key=lambda r: (r.design, r.input_state_id, r.recipe_id)))
    return ordered, {
        "attempt_rows": len(rows),
        "latest_sequence_rows": len(latest),
        "unique_input_recipe_rows": len(ordered),
    }


def _table(
    rows: Sequence[TrainingRow], recipe_ids: Sequence[str]
) -> dict[tuple[str, str], dict[str, TrainingRow]]:
    """Require all five alternatives on every input state used for regret."""

    wanted = set(recipe_ids)
    table: dict[tuple[str, str], dict[str, TrainingRow]] = {}
    for row in rows:
        if row.recipe_id not in wanted:
            continue
        table.setdefault((row.design, row.input_state_id), {})[row.recipe_id] = row
    if not table:
        raise ValueError("no training rows for worker")
    for key, alternatives in table.items():
        missing = wanted - alternatives.keys()
        if missing:
            raise ValueError(f"incomplete five-recipe alternatives at {key}: {sorted(missing)}")
        feature_maps = [dict(item.features) for item in alternatives.values()]
        if any(features != feature_maps[0] for features in feature_maps[1:]):
            raise ValueError(f"input telemetry varies across alternatives: {key}")
    return table


def _design_mean(values: Sequence[tuple[str, float]]) -> float:
    """Give each design equal total weight, regardless of state count."""

    grouped: dict[str, list[float]] = {}
    for design, value in values:
        grouped.setdefault(design, []).append(value)
    return float(np.mean([np.mean(group) for _, group in sorted(grouped.items())]))


def _fit_parameters(
    table: Mapping[tuple[str, str], Mapping[str, TrainingRow]],
    training_designs: Sequence[str],
    recipe_ids: Sequence[str],
    worker: str,
    spec: Mapping[str, Any],
    penalty: float,
) -> dict[str, dict[str, Any]]:
    """Fit fixed β and one ridge slope per recipe on selected designs."""

    selected = [(key, alternatives) for key, alternatives in table.items()
                if key[0] in training_designs]
    if not selected:
        raise ValueError("no selected training states")
    maximum_missing = float(spec["fitting"]["feature_availability"]["maximum_missing_fraction"])
    result: dict[str, dict[str, Any]] = {}
    for recipe_id in recipe_ids:
        gains = [(key[0], alternatives[recipe_id].gain(worker)) for key, alternatives in selected]
        beta = _design_mean(gains)
        observed: dict[str, list[tuple[float, float]]] = {}
        missing = 0
        raw_values: list[float] = []
        for key, alternatives in selected:
            value = _number(alternatives[recipe_id].features.get(recipe_id))
            if value is None:
                missing += 1
            else:
                raw_values.append(value)
                observed.setdefault(key[0], []).append((value, alternatives[recipe_id].gain(worker)))
        missing_fraction = missing / len(selected)
        drop_reason: str | None = None
        if missing_fraction > maximum_missing:
            drop_reason = "more_than_10_percent_missing"
        elif not raw_values or min(raw_values) == max(raw_values):
            drop_reason = "constant_or_absent"
        if drop_reason is not None:
            result[recipe_id] = {
                "feature": spec["recipes"][recipe_id]["feature"],
                "expected_sign": spec["recipes"][recipe_id]["expected_sign"],
                "beta": beta, "w_fitted": 0.0, "feature_available": False,
                "feature_missing_fraction": missing_fraction,
                "feature_drop_reason": drop_reason,
                "feature_mean": None, "feature_sd": None,
                "raw_min": None, "raw_max": None,
                "z_min": None, "z_max": None,
            }
            continue
        # For each observed design, each state gets 1/n_design_states weight.
        # Normalize over represented designs after missing feature values drop.
        weighted = [(x, y, 1.0 / (len(observed) * len(group)))
                    for group in observed.values() for x, y in group]
        mean = sum(x * weight for x, _, weight in weighted)
        variance = sum(weight * (x - mean) ** 2 for x, _, weight in weighted)
        sd = math.sqrt(variance)
        if not math.isfinite(sd) or sd <= 0:
            raise ValueError(f"feature variance cannot be standardized: {recipe_id}")
        numerator = sum(weight * ((x - mean) / sd) * (y - beta)
                        for x, y, weight in weighted)
        denominator = sum(weight * ((x - mean) / sd) ** 2
                          for x, _, weight in weighted) + penalty
        slope = numerator / denominator
        raw_min, raw_max = min(raw_values), max(raw_values)
        result[recipe_id] = {
            "feature": spec["recipes"][recipe_id]["feature"],
            "expected_sign": spec["recipes"][recipe_id]["expected_sign"],
            "beta": beta, "w_fitted": slope, "feature_available": True,
            "feature_missing_fraction": missing_fraction,
            "feature_drop_reason": None,
            "feature_mean": mean, "feature_sd": sd,
            "raw_min": raw_min, "raw_max": raw_max,
            "z_min": (raw_min - mean) / sd,
            "z_max": (raw_max - mean) / sd,
        }
    return result


def _score_parameter(
    parameters: Mapping[str, Any], value: float | None, *, use_terms: bool
) -> float:
    beta = float(parameters["beta"])
    if not use_terms or not parameters["feature_available"] or value is None:
        return beta
    z = (value - parameters["feature_mean"]) / parameters["feature_sd"]
    z = max(parameters["z_min"], min(parameters["z_max"], z))
    return beta + parameters["w_fitted"] * z


def _fold_mse(
    table: Mapping[tuple[str, str], Mapping[str, TrainingRow]],
    held_design: str,
    recipe_ids: Sequence[str],
    worker: str,
    parameters: Mapping[str, Mapping[str, Any]],
) -> float:
    errors = [
        (_score_parameter(parameters[recipe_id], alternatives[recipe_id].features.get(recipe_id),
                          use_terms=True) - alternatives[recipe_id].gain(worker)) ** 2
        for (design, _), alternatives in table.items() if design == held_design
        for recipe_id in recipe_ids
    ]
    return float(np.mean(errors))


def _fold_regret(
    table: Mapping[tuple[str, str], Mapping[str, TrainingRow]],
    held_design: str,
    recipe_ids: Sequence[str],
    worker: str,
    parameters: Mapping[str, Mapping[str, Any]],
    *,
    use_terms: bool,
) -> float:
    regrets: list[float] = []
    for (design, _), alternatives in table.items():
        if design != held_design:
            continue
        gains = {recipe_id: alternatives[recipe_id].gain(worker) for recipe_id in recipe_ids}
        chosen = min(
            recipe_ids,
            key=lambda recipe_id: (
                -_score_parameter(parameters[recipe_id],
                                  alternatives[recipe_id].features.get(recipe_id),
                                  use_terms=use_terms),
                recipe_id,
            ),
        )
        regrets.append(max(gains.values()) - gains[chosen])
    return float(np.mean(regrets))


def fit_worker(
    rows: Sequence[TrainingRow],
    worker: str,
    *,
    data_hashes: Mapping[str, str] | None = None,
    provisional: bool = False,
) -> dict[str, Any]:
    """Fit one frozen worker and decide whether to ship its feature terms.

    A single ridge penalty is chosen per worker by five-fold, design-weighted
    leave-one-design-out MSE across all five recipe outcomes. β stays fixed at
    each recipe's design-weighted mean; the penalty applies to w alone.
    """

    spec = load_feature_spec()  # Public fit boundary enforces the byte freeze.
    if worker not in {"timing", "area"}:
        raise ValueError(f"worker {worker} is not fitted")
    recipe_ids = tuple(sorted(recipe_id for recipe_id, definition in spec["recipes"].items()
                              if definition["worker"] == worker))
    if len(recipe_ids) != 5:
        raise ValueError("frozen worker must have exactly five recipes")
    unique_rows, dedup_counts = deduplicate_rows(rows)
    table = _table(unique_rows, recipe_ids)
    designs = tuple(sorted({design for design, _ in table}))
    if len(designs) != 5:
        raise ValueError("frozen keep/drop rule requires exactly five designs")
    penalty_spec = spec["fitting"]["penalty_grid"]
    grid = np.logspace(
        math.log10(float(penalty_spec["min"])),
        math.log10(float(penalty_spec["max"])),
        int(penalty_spec["count"]),
    )
    candidates: list[dict[str, float]] = []
    for candidate in grid:
        fold_mses = []
        for held in designs:
            training = tuple(design for design in designs if design != held)
            parameters = _fit_parameters(table, training, recipe_ids, worker, spec,
                                         float(candidate))
            fold_mses.append(_fold_mse(table, held, recipe_ids, worker, parameters))
        candidates.append({"penalty": float(candidate), "lodo_mse": float(np.mean(fold_mses))})
    selected = min(candidates, key=lambda item: (item["lodo_mse"], item["penalty"]))
    penalty = selected["penalty"]
    folds: list[dict[str, Any]] = []
    for held in designs:
        training = tuple(design for design in designs if design != held)
        parameters = _fit_parameters(table, training, recipe_ids, worker, spec, penalty)
        feature_regret = _fold_regret(table, held, recipe_ids, worker, parameters,
                                      use_terms=True)
        null_regret = _fold_regret(table, held, recipe_ids, worker, parameters,
                                   use_terms=False)
        folds.append({
            "held_design": held,
            "feature_regret": feature_regret,
            "null_regret": null_regret,
            "feature_better": feature_regret < null_regret,
            "feature_mse": _fold_mse(table, held, recipe_ids, worker, parameters),
        })
    feature_regret = float(np.mean([fold["feature_regret"] for fold in folds]))
    null_regret = float(np.mean([fold["null_regret"] for fold in folds]))
    improved_folds = sum(fold["feature_better"] for fold in folds)
    keep = feature_regret < null_regret and improved_folds >= 3
    final_params = _fit_parameters(table, designs, recipe_ids, worker, spec, penalty)
    recipes: dict[str, dict[str, Any]] = {}
    for recipe_id, parameters in final_params.items():
        fitted = float(parameters["w_fitted"])
        recipes[recipe_id] = {
            **parameters,
            "w": fitted if keep and parameters["feature_available"] else 0.0,
            "observed_sign": ("positive" if fitted > 0 else "negative" if fitted < 0 else "zero"),
        }
    return {
        "schema_version": SCHEMA,
        "worker": worker,
        "label": "PROVISIONAL" if provisional else "FINAL",
        "feature_spec_sha256": FROZEN_SHA256,
        "score_expression": "beta + w * clip((feature_value - feature_mean) / feature_sd, z_min, z_max)",
        "penalty": penalty,
        "penalty_lodo_mse": candidates,
        "keep_features": keep,
        "keep_or_drop": "keep" if keep else "drop",
        "regret_feature": feature_regret,
        "regret_null": null_regret,
        "improved_folds": improved_folds,
        "folds": folds,
        "recipes": recipes,
        "designs": list(designs),
        "states_by_design": {
            design: sum(key[0] == design for key in table)
            for design in designs
        },
        "deduplication": dedup_counts,
        "data_hashes": dict(data_hashes or {}),
    }


def score_recipe(
    model: Mapping[str, Any], recipe_id: str, feature_value: float | None
) -> tuple[float, bool]:
    """Return (score, incomplete_measurement) from a published fit JSON."""

    if model.get("schema_version") != SCHEMA or model.get("feature_spec_sha256") != FROZEN_SHA256:
        raise ValueError("worker model schema or feature freeze differs")
    parameters = model["recipes"][recipe_id]
    value = _number(feature_value)
    incomplete = value is None
    if value is None or not model["keep_features"] or not parameters["feature_available"]:
        return float(parameters["beta"]), incomplete
    z = (value - parameters["feature_mean"]) / parameters["feature_sd"]
    z = max(parameters["z_min"], min(parameters["z_max"], z))
    return float(parameters["beta"] + parameters["w"] * z), False


def rank_recipes(
    model: Mapping[str, Any], feature_values: Mapping[str, float | None]
) -> tuple[tuple[str, ...], bool]:
    """Rank exact recipe IDs by score, breaking equal scores alphabetically."""

    scored = {
        recipe_id: score_recipe(model, recipe_id, feature_values.get(recipe_id))
        for recipe_id in model["recipes"]
    }
    return (
        tuple(sorted(scored, key=lambda recipe_id: (-scored[recipe_id][0], recipe_id))),
        any(incomplete for _, incomplete in scored.values()),
    )


def _guard_training_root(path: Path) -> Path:
    """Refuse every non-allow-listed data root, including a sibling test set."""

    allowed = TRAINING_ROOT.resolve()
    candidate = path.expanduser().resolve()
    if candidate != allowed or not candidate.is_dir():
        raise ValueError(f"data root is outside the training allow-list: {path}")
    return candidate


def _checked_data_path(path: Path, root: Path) -> Path:
    """Reject symlink traversal out of the one training data root."""

    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"training data path escapes allow-list: {path}")
    return path


def _read_json(path: Path, root: Path | None = None) -> dict[str, Any]:
    if root is not None:
        _checked_data_path(path, root)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _evidence_from_json(path: Path, state_id: str, root: Path) -> Evidence:
    _checked_data_path(path, root)
    evidence = Evidence.model_validate_json(path.read_bytes())
    if evidence.state_id != state_id or evidence.fake:
        raise ValueError(f"Evidence state identity or origin differs: {path}")
    return evidence


def _same_measurement(first: Evidence, second: Evidence) -> bool:
    return (
        first.metrics == second.metrics
        and first.routed == second.routed
        and first.route_errors == second.route_errors
        and first.drc_errors == second.drc_errors
        and first.constraint_fingerprint == second.constraint_fingerprint
        and first.clock_digest == second.clock_digest
        and first.meter_version == second.meter_version
    )


def _check_digests(evidence: Evidence, start: Evidence) -> None:
    if (not evidence.constraint_fingerprint or not evidence.clock_digest
            or evidence.constraint_fingerprint != start.constraint_fingerprint
            or evidence.clock_digest != start.clock_digest):
        raise ValueError(f"Evidence constraint or clock digest differs: {evidence.state_id}")


def _check_run_tree(root: Path, run_id: str) -> None:
    """Check all paths Store.load_outcome will open before calling it."""

    run_dir = root / "runs" / run_id
    marker = _read_json(run_dir / "complete.json", root)
    files = marker.get("files")
    if marker.get("run_id") != run_id or not isinstance(files, dict):
        raise ValueError(f"invalid run completion marker: {run_id}")
    for name in files:
        if name not in {"runinfo.json", "output_state.json", "evidence.json", "claims.json"}:
            raise ValueError(f"unexpected run artifact {name}")
        _checked_data_path(run_dir / name, root)
    if "output_state.json" in files:
        state = _read_json(run_dir / "output_state.json", root)
        state_id = state.get("id")
        if not isinstance(state_id, str) or not _HASH64.fullmatch(state_id):
            raise ValueError(f"invalid output state ID in {run_id}")
        _checked_data_path(root / "states" / state_id / "state.json", root)
        _checked_data_path(root / "states" / state_id / "design.dcp", root)


def _target_amd_result(recorded: str | None, *, provisional: bool) -> str:
    """Allow an unknown check only when every output is labelled provisional."""

    if recorded == "fail":
        return "fail"
    if recorded == "pass":
        return "provisional" if provisional else "pass"
    if provisional and recorded in {None, "unknown"}:
        return "provisional"
    raise ValueError(f"AMD PASS required for final fit: {recorded}")


def _require_phase_completion(events: Sequence[Mapping[str, Any]]) -> None:
    """Do not fit a live, partially populated D1/D2/D3 generation."""

    present = {event["edge"]["first_phase"] for event in events
               if event.get("type") == "edge"
               and event["edge"].get("first_phase") in {"D1", "D2", "D3"}}
    completed = {event.get("phase") for event in events
                 if event.get("type") == "phase_complete"}
    required = {"D1", "D2"} | ({"D3"} if "D3" in present else set())
    if not required <= completed:
        raise ValueError(f"training phase not complete: {sorted(required - completed)}")


def _validate_training_v3_source(
    source: Path,
    training_root: Path,
    starts: Mapping[str, str],
    first_output_pairs: set[tuple[str, str]],
    required_input_pairs: set[tuple[str, str]],
    meter_version: str,
) -> tuple[Path, dict[str, str]]:
    """Validate the one named training telemetry tree before reading records.

    The v3 extractor planned 708 early edges. Its 465 logical index rows are
    exactly five B2 starts plus 460 distinct saved output hashes. The fit may
    use only input states from that set; later D3 outputs are not fit inputs.
    """

    fixed = TRAINING_V3_SOURCE.absolute()
    if source.absolute() != fixed:
        raise ValueError("telemetry source is not the named training-v3 tree")
    resolved = source.resolve(strict=True)
    if resolved != fixed or not resolved.is_dir():
        raise ValueError("named training-v3 source is symlinked or not a directory")
    plan_path = resolved / "plan.json"
    index_path = resolved / "logical_index.csv"
    plan = _read_json(plan_path, resolved)
    _checked_data_path(index_path, resolved)
    expected_starts = {(design, state_id) for design, state_id in starts.items()}
    if (plan.get("schema_version") != "exp-evening-training-telemetry-v1"
            or plan.get("source_campaign") != "worker_fitting_train"
            or plan.get("source_first_edge_count") != 708
            or plan.get("meter_version") != meter_version
            or plan.get("parser_sha256") != TRAINING_V3_PARSER_SHA256
            or plan.get("distinct_output_dcps") != len(first_output_pairs)
            or plan.get("logical_rows") != len(expected_starts) + len(first_output_pairs)):
        raise ValueError("training-v3 plan scope or source version differs")
    if any(design not in starts or not isinstance(state_id, str)
           or not _HASH64.fullmatch(state_id)
           for design, state_id in first_output_pairs | required_input_pairs):
        raise ValueError("training-v3 expected hashes escape five-design scope")
    logical_entries = plan.get("logical_entries")
    states = plan.get("states")
    if not isinstance(logical_entries, list) or not isinstance(states, list):
        raise ValueError("training-v3 plan lacks logical entries or state entries")
    logical: list[tuple[str, str, str]] = []
    for item in logical_entries:
        if not isinstance(item, dict):
            raise ValueError("training-v3 logical entry is not an object")
        kind, design, state_id = item.get("kind"), item.get("design"), item.get("state_sha256")
        if (kind not in {"start", "output"} or design not in starts
                or not isinstance(state_id, str) or not _HASH64.fullmatch(state_id)):
            raise ValueError("training-v3 logical entry escapes five-design scope")
        logical.append((kind, design, state_id))
    logical_starts = {(design, state_id) for kind, design, state_id in logical
                      if kind == "start"}
    logical_outputs = {(design, state_id) for kind, design, state_id in logical
                       if kind == "output"}
    if (len(logical) != plan["logical_rows"]
            or len(set(logical)) != len(logical)
            or logical_starts != expected_starts
            or logical_outputs != first_output_pairs):
        raise ValueError("training-v3 logical entries differ from the training journal")
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["kind", "design", "state_sha256", "telemetry_json"]:
            raise ValueError("training-v3 index has unexpected columns")
        index_rows = list(reader)
    indexed: list[tuple[str, str, str]] = []
    for row in index_rows:
        kind, design, state_id = row["kind"], row["design"], row["state_sha256"]
        if (kind not in {"start", "output"} or design not in starts
                or not _HASH64.fullmatch(state_id)):
            raise ValueError("training-v3 index row escapes five-design scope")
        expected_path = resolved / "states" / state_id / "dcp_telemetry.json"
        if row["telemetry_json"] != str(expected_path):
            raise ValueError("training-v3 index points outside the named source layout")
        indexed.append((kind, design, state_id))
    if Counter(indexed) != Counter(logical):
        raise ValueError("training-v3 index differs from its plan")
    expected_state_pairs = expected_starts | first_output_pairs
    state_pairs: set[tuple[str, str]] = set()
    state_hashes: set[str] = set()
    for item in states:
        if not isinstance(item, dict):
            raise ValueError("training-v3 state entry is not an object")
        design, state_id = item.get("design"), item.get("state_sha256")
        if (design not in starts or not isinstance(state_id, str)
                or not _HASH64.fullmatch(state_id)
                or (design, state_id) not in expected_state_pairs
                or state_id in state_hashes):
            raise ValueError("training-v3 state entry escapes journal state set")
        expected_dcp = training_root / "states" / state_id / "design.dcp"
        if item.get("dcp_path") != str(expected_dcp):
            raise ValueError("training-v3 DCP path differs from canonical training store")
        _checked_data_path(expected_dcp, training_root)
        evidence_path = Path(item.get("evidence_path", ""))
        _checked_data_path(evidence_path, training_root)
        if _sha256(evidence_path) != item.get("evidence_sha256"):
            raise ValueError("training-v3 plan Evidence hash differs")
        state_pairs.add((design, state_id))
        state_hashes.add(state_id)
    if (state_pairs != expected_state_pairs
            or plan.get("unique_state_dcps") != len(state_hashes)
            or not required_input_pairs <= state_pairs):
        raise ValueError("training-v3 plan lacks an expected fit input state")
    return resolved / "states", {
        "telemetry_source_kind": "training-v3",
        "telemetry_source_plan_sha256": _sha256(plan_path),
        "telemetry_source_index_sha256": _sha256(index_path),
        "telemetry_source_parser_sha256": plan["parser_sha256"],
    }


def _check_telemetry_meter(
    record: Mapping[str, Any], meter: Evidence, state_id: str, *,
    check_validity: bool, expected_extractor_sha256: str | None = None,
) -> None:
    """Reject feature records measured for a different meter state."""

    for metric in ("wns_ns", "packed_luts"):
        measured = meter.metric(metric)
        telemetered = _number(_at(record, ("metrics", metric, "value")))
        tolerance = 0.001 if metric == "wns_ns" else 0.0
        if measured is None or telemetered is None or abs(measured.value - telemetered) > tolerance:
            raise ValueError(f"telemetry disagrees with meter {metric}: {state_id}")
    if check_validity:
        validity = record.get("validity")
        if not isinstance(validity, Mapping):
            raise ValueError(f"training-v3 telemetry lacks validity: {state_id}")
        for key in ("constraint_fingerprint", "clock_digest", "routed",
                    "route_errors", "drc_errors"):
            if key not in validity:
                raise ValueError(f"training-v3 telemetry lacks {key}: {state_id}")
            observed = validity[key]
            expected = getattr(meter, key)
            if (key in {"constraint_fingerprint", "clock_digest"} and not observed):
                raise ValueError(f"training-v3 telemetry lacks {key}: {state_id}")
            if type(observed) is not type(expected) or observed != expected:
                raise ValueError(f"telemetry disagrees with meter {key}: {state_id}")
    if expected_extractor_sha256 is not None:
        observed = _at(record, ("provenance", "extractor_sha256"))
        if observed != expected_extractor_sha256:
            raise ValueError(f"training-v3 telemetry extractor hash differs: {state_id}")


def _require_telemetry_files(
    directory: Path, guard_root: Path, state_ids: set[str]
) -> None:
    """Fail quickly and with a count until every fit input has a safe record."""

    missing = []
    for state_id in sorted(state_ids):
        if not _HASH64.fullmatch(state_id):
            raise ValueError("fit input state is not a DCP hash")
        path = directory / state_id / "dcp_telemetry.json"
        if path.is_file():
            _checked_data_path(path, guard_root)
        else:
            missing.append(state_id)
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)}/{len(state_ids)} input-state DCP telemetry records"
        )


def load_training_rows(
    data_root: Path,
    *,
    provisional: bool = False,
    telemetry_root: Path | None = None,
    telemetry_source: str = "sweep-local",
) -> TrainingDataset:
    """Load only checked training edges and telemetry, without tool execution.

    A sweep-local source uses telemetry/<sha>/dcp_telemetry.json under the
    allow-listed data root. The named training-v3 source uses the exact
    experiment tree after its plan/index are checked against this journal.
    """

    spec = load_feature_spec()  # Public data-loading boundary checks the freeze.
    root = _guard_training_root(data_root)
    if telemetry_source not in {"sweep-local", "training-v3"}:
        raise ValueError("unknown telemetry source")
    if telemetry_source == "training-v3" and telemetry_root is not None:
        raise ValueError("named training-v3 source cannot take an arbitrary telemetry root")
    manifest_path = root / "launch_manifest.json"
    manifest = _read_json(manifest_path, root)
    inputs = {item["design"]: item["sha256"] for item in manifest["train_inputs"]}
    if (manifest.get("schema_version") != "worker-fitting-launch-v1"
            or set(inputs) != set(TRAINING_DESIGNS)
            or len(set(inputs.values())) != 5
            or any(not isinstance(value, str) or not _HASH64.fullmatch(value)
                   for value in inputs.values())
            or manifest.get("data_root") != str(root)):
        raise ValueError("launch manifest differs from the five-design training scope")
    journal = Journal(root)
    _checked_data_path(journal.path("worker_fitting_train"), root)
    journal_lock = journal.path("worker_fitting_train").parent / "journal.lock"
    if journal_lock.exists() and not journal_lock.resolve().is_relative_to(root):
        raise ValueError("journal lock escapes training allow-list")
    events = journal.read("worker_fitting_train")
    _require_phase_completion(events)
    configurations = [event for event in events if event.get("type") == "configuration"]
    if len(configurations) != 1:
        raise ValueError("training journal lacks one configuration event")
    configuration = configurations[0]["configuration"]
    if (configuration.get("schema_version") != "worker-fitting-train-v1"
            or configuration.get("fake") is not False):
        raise ValueError("journal configuration is not the real training campaign")
    if {design: item["start_state_id"] for design, item in configuration["designs"].items()} != inputs:
        raise ValueError("journal start states differ from launch manifest")
    meter_version = configuration["meter_version"]
    recipe_ids = set(spec["recipes"])
    configured_recipes = {item["id"] for worker in ("timing", "area")
                          for item in configuration["recipe_sets"][worker]}
    if configured_recipes != recipe_ids:
        raise ValueError("journal recipe set differs from frozen feature spec")
    latest: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    skipped_edges = 0
    for event in events:
        if event.get("type") != "edge":
            continue
        edge = event["edge"]
        if edge.get("first_phase") not in {"D1", "D2", "D3"}:
            continue
        design = edge["design"]
        sequence = tuple(edge["sequence"])
        if (design not in inputs or not sequence or sequence[-1] not in recipe_ids
                or edge.get("fake") is not False):
            raise ValueError("journal edge escapes frozen training scope")
        if edge.get("skipped_reason") is not None:
            if edge.get("run_id") is not None or edge.get("input_state_id") is not None:
                raise ValueError("skipped edge falsely claims a run or input state")
            skipped_edges += 1
            continue
        if edge.get("run_id") is None or edge.get("input_state_id") is None:
            raise ValueError("attempted edge lacks run or input state")
        if (not isinstance(edge["input_state_id"], str)
                or not _HASH64.fullmatch(edge["input_state_id"])):
            raise ValueError("attempted edge input is not a DCP hash")
        output_id = edge.get("output_state_id")
        if output_id is not None and (not isinstance(output_id, str)
                                      or not _HASH64.fullmatch(output_id)):
            raise ValueError("attempted edge output is not a DCP hash")
        latest[(design, sequence)] = edge
    edges = [latest[key] for key in sorted(latest)]
    if not edges:
        raise ValueError("training journal has no D1-D3 edges")
    source_hashes: dict[str, str]
    if telemetry_source == "training-v3":
        attempted = [event["edge"] for event in events
                     if event.get("type") == "edge" and event["edge"].get("run_id")]
        if len(attempted) < 708:
            raise ValueError("training-v3 source needs the first 708 attempted journal edges")
        first_output_pairs = {
            (edge["design"], edge["output_state_id"])
            for edge in attempted[:708] if edge.get("output_state_id")
        }
        required_input_pairs = {(edge["design"], edge["input_state_id"]) for edge in edges}
        telemetry_dir, source_hashes = _validate_training_v3_source(
            TRAINING_V3_SOURCE, root, inputs, first_output_pairs,
            required_input_pairs, meter_version,
        )
        telemetry_guard_root = TRAINING_V3_SOURCE.resolve(strict=True)
    else:
        telemetry_dir = (telemetry_root or root / "telemetry").expanduser().resolve()
        if not telemetry_dir.is_relative_to(root):
            raise ValueError("telemetry root escapes the training allow-list")
        telemetry_guard_root = root
        source_hashes = {"telemetry_source_kind": "sweep-local"}
    required_input_ids = {edge["input_state_id"] for edge in edges}
    _require_telemetry_files(telemetry_dir, telemetry_guard_root, required_input_ids)
    store = StateStore(root)
    start_evidence: dict[str, Evidence] = {}
    evidence_by_state: dict[str, Evidence] = {}
    evidence_hashes: list[tuple[str, str]] = []
    for design, state_id in sorted(inputs.items()):
        candidates = sorted((root / "measurements" / state_id).glob("*/evidence.json"))
        matching: list[tuple[Path, Evidence]] = []
        for path in candidates:
            evidence = _evidence_from_json(path, state_id, root)
            if evidence.meter_version == meter_version:
                matching.append((path, evidence))
        if len(matching) != 1:
            raise ValueError(f"expected one B2 meter Evidence for {design}")
        path, evidence = matching[0]
        start_evidence[design] = evidence
        evidence_by_state[state_id] = evidence
        evidence_hashes.append((str(path.relative_to(root)), _sha256(path)))
    outcomes: dict[str, Any] = {}
    run_hashes: list[tuple[str, str]] = []
    for edge in edges:
        run_id = edge["run_id"]
        if run_id in outcomes:
            continue
        _check_run_tree(root, run_id)
        outcome = store.load_outcome(run_id)
        outcomes[run_id] = outcome
        run_dir = root / "runs" / run_id
        run_hashes.append((run_id, _sha256(run_dir / "complete.json")))
        if outcome.evidence is not None:
            if outcome.evidence.meter_version != meter_version:
                raise ValueError(f"meter version differs in {run_id}")
            prior = evidence_by_state.get(outcome.evidence.state_id)
            if prior is not None and not _same_measurement(prior, outcome.evidence):
                raise ValueError(f"conflicting meter Evidence for state {outcome.evidence.state_id}")
            evidence_by_state[outcome.evidence.state_id] = outcome.evidence
            evidence_hashes.append((f"runs/{run_id}/evidence.json",
                                    _sha256(run_dir / "evidence.json")))
    plan_path = root / "verification/plan.json"
    plan = _read_json(plan_path, root)
    fingerprint = plan["fingerprint"]
    if plan.get("references") != inputs:
        raise ValueError("AMD verification references differ from B2 starts")
    if not provisional:
        summary = _read_json(root / "verification/backfill_summary.json", root)
        if (summary.get("phase") != "complete" or summary.get("completed") != 460
                or summary.get("statuses") != {"pass": 460}
                or summary.get("validator_fingerprint") != fingerprint):
            raise ValueError("460/460 AMD PASS backfill gate is incomplete")
    verification_hashes: list[tuple[str, str]] = []
    telemetry_hashes: list[tuple[str, str]] = []
    telemetry_by_state: dict[str, dict[str, Any]] = {}
    for input_id in sorted(required_input_ids):
        path = telemetry_dir / input_id / "dcp_telemetry.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing input-state DCP telemetry: {path}")
        record = _read_json(path, telemetry_guard_root)
        if (record.get("state_sha256") != input_id
                or record.get("schema_version") != "dcp-telemetry-v1"
                or record.get("fake") is not False):
            raise ValueError(f"DCP telemetry identity or origin differs: {path}")
        if input_id not in evidence_by_state:
            raise ValueError(f"input state lacks meter Evidence: {input_id}")
        meter = evidence_by_state[input_id]
        _check_telemetry_meter(record, meter, input_id,
                               check_validity=telemetry_source == "training-v3",
                               expected_extractor_sha256=(
                                   source_hashes["telemetry_source_parser_sha256"]
                                   if telemetry_source == "training-v3" else None
                               ))
        telemetry_by_state[input_id] = record
        telemetry_hashes.append((str(path.relative_to(PROJECT)), _sha256(path)))
    rows: list[TrainingRow] = []
    for edge in edges:
        design = edge["design"]
        input_id = edge["input_state_id"]
        output_id = edge.get("output_state_id")
        recipe_id = edge["sequence"][-1]
        outcome = outcomes[edge["run_id"]]
        if (outcome.run.recipe_id != recipe_id
                or outcome.run.status.value != edge["run_status"]
                or outcome.run.input_sha256_before != input_id
                or (outcome.state.id if outcome.state else None) != output_id):
            raise ValueError(f"journal and run outcome disagree: {edge['run_id']}")
        parent = evidence_by_state.get(input_id)
        if parent is None:
            raise ValueError(f"missing parent meter Evidence: {input_id}")
        _check_digests(parent, start_evidence[design])
        child = outcome.evidence
        if child is not None:
            _check_digests(child, start_evidence[design])
        if edge["physically_valid"] and (
            child is None or child.routed is not True
            or child.route_errors != 0 or child.drc_errors != 0
        ):
            raise ValueError(f"journal claims physical validity without meter proof: {edge['run_id']}")
        if output_id is None and edge["physically_valid"]:
            raise ValueError(f"journal claims physical validity without an output: {edge['run_id']}")
        recorded_amd: str | None = None
        if output_id is not None:
            path = root / "verification/records" / fingerprint / design / f"{output_id}.json"
            if path.is_file():
                record = _read_json(path, root)
                if (record.get("design") != design or record.get("output_sha256") != output_id
                        or record.get("reference_sha256") != inputs[design]
                        or record.get("validator_fingerprint") != fingerprint
                        or record.get("result") not in {"pass", "fail", "unknown"}):
                    raise ValueError(f"AMD record identity differs: {path}")
                recorded_amd = record["result"]
                raw_report = record.get("raw_report_path")
                raw_sha256 = record.get("raw_report_sha256")
                if raw_report and raw_sha256:
                    raw_path = Path(raw_report)
                    _checked_data_path(raw_path, root)
                    if _sha256(raw_path) != raw_sha256:
                        raise ValueError(f"AMD report hash differs: {raw_path}")
                elif recorded_amd == "pass":
                    raise ValueError(f"AMD PASS lacks a retained report: {path}")
                verification_hashes.append((str(path.relative_to(root)), _sha256(path)))
            elif not provisional:
                raise ValueError(f"missing AMD verification record: {output_id}")
        amd_result = _target_amd_result(recorded_amd, provisional=provisional) if output_id else "provisional"
        if edge["physically_valid"] is not True and edge["physically_valid"] is not False:
            raise ValueError("journal physical-validity field is not boolean")
        gains = target_gain(
            parent.model_dump(mode="json"),
            child.model_dump(mode="json") if child is not None else None,
            B2_REQUIREMENTS_NS[design],
            physically_valid=edge["physically_valid"] and outcome.run.status == RunStatus.OK,
            amd_result=amd_result,
            output_identical_to_input=output_id == input_id,
            no_output=output_id is None,
        )
        rows.append(TrainingRow(
            design=design, input_state_id=input_id, recipe_id=recipe_id,
            sequence=tuple(edge["sequence"]), output_state_id=output_id,
            run_id=edge["run_id"], phase=edge["first_phase"],
            timing_gain_pct=gains["timing"], area_gain_pct=gains["area"],
            features=extract_features(telemetry_by_state[input_id], spec),
            physically_valid=edge["physically_valid"],
            amd_result=recorded_amd or "not_recorded",
        ))
    hashes = {
        "fit_code_sha256": _sha256(Path(__file__)),
        "feature_spec_sha256": FROZEN_SHA256,
        "audited_b2_requirements_sha256": hashlib.sha256(
            _json_bytes(B2_REQUIREMENTS_NS)
        ).hexdigest(),
        "launch_manifest_sha256": _sha256(manifest_path),
        "journal_snapshot_sha256": hashlib.sha256(_json_bytes(events)).hexdigest(),
        "verification_plan_sha256": _sha256(plan_path),
        "audited_requirements_note_sha256": _sha256(PROJECT / "notes/training_data_audit.md"),
        "run_completion_digest": _digest_pairs(run_hashes),
        "meter_evidence_digest": _digest_pairs(evidence_hashes),
        "amd_record_digest": _digest_pairs(verification_hashes),
        "input_telemetry_digest": _digest_pairs(telemetry_hashes),
    }
    hashes.update(source_hashes)
    if telemetry_source == "training-v3":
        plan_again = _checked_data_path(telemetry_guard_root / "plan.json",
                                        telemetry_guard_root)
        index_again = _checked_data_path(telemetry_guard_root / "logical_index.csv",
                                         telemetry_guard_root)
        if (_sha256(plan_again)
                != source_hashes["telemetry_source_plan_sha256"]
                or _sha256(index_again)
                != source_hashes["telemetry_source_index_sha256"]):
            raise ValueError("training-v3 plan or index changed during fit loading")
    return TrainingDataset(
        rows=tuple(rows), data_hashes=hashes,
        counts={"journal_edges": len(edges), "journal_skips": skipped_edges,
                "distinct_input_states": len(telemetry_by_state),
                "distinct_runs": len(outcomes), "amd_records": len(set(verification_hashes))},
        provisional=provisional,
    )


def _report(models: Mapping[str, Mapping[str, Any]], dataset: TrainingDataset) -> str:
    """Create a short auditable report from the checked model objects."""

    label = "PROVISIONAL" if dataset.provisional else "FINAL"
    lines = [
        f"# Worker fit ({label})",
        "",
        f"{label}: {dataset.counts['journal_edges']} latest D1-D3 edges and "
        f"{dataset.counts['distinct_input_states']} distinct input states were checked.",
        "",
        "The timing target uses only meter WNS and the audited, fixed B2 clock requirements "
        "(Vex 1.570, spam 1.600, 3D 1.538, LogicNets 1.500, Finn 1.600 ns). "
        "Every parent/child Evidence constraint and clock digest must match its B2. "
        "The area target uses only meter packed LUTs. Telemetry supplies input features only.",
        "",
        "One ridge penalty per worker is chosen by five-fold, design-weighted "
        "leave-one-design-out MSE across its five recipes. β is fixed at each "
        "recipe's equal-design mean; only w is penalized. The same five folds "
        "evaluate regret, as specified by the freeze.",
        "",
    ]
    for worker, model in models.items():
        lines.extend([
            f"## {worker}", "",
            f"{label}: feature terms **{model['keep_or_drop']}**; penalty "
            f"{model['penalty']:.6g}; feature regret {model['regret_feature']:.6g} "
            f"versus null {model['regret_null']:.6g}; improved folds "
            f"{model['improved_folds']}/5.",
            "",
            "| Recipe | β gain % | fitted w | shipped w | missing | status |",
            "|---|---:|---:|---:|---:|---|",
        ])
        for recipe_id, params in model["recipes"].items():
            status = params["feature_drop_reason"] or params["observed_sign"]
            lines.append(
                f"| {recipe_id} | {params['beta']:.5g} | {params['w_fitted']:.5g} "
                f"| {params['w']:.5g} | {params['feature_missing_fraction']:.1%} "
                f"| {status} |"
            )
        lines.append("")
    lines.extend([
        "Model JSON records raw/z clip bounds, standardization, per-fold regrets, "
        "the selected penalty grid and all input hashes. The expected feature "
        "signs are hypotheses; the frozen ridge rule imposes no sign constraint.",
        "",
    ])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Fit and atomically publish both workers only after all gates pass."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=TRAINING_ROOT)
    parser.add_argument("--telemetry-root", type=Path)
    parser.add_argument("--telemetry-source", choices=("sweep-local", "training-v3"),
                        default="sweep-local")
    parser.add_argument("--provisional", action="store_true")
    args = parser.parse_args(argv)
    load_feature_spec()  # CLI boundary checks before any data read or output.
    dataset = load_training_rows(args.data_root, provisional=args.provisional,
                                 telemetry_root=args.telemetry_root,
                                 telemetry_source=args.telemetry_source)
    models = {
        worker: fit_worker(dataset.rows, worker, data_hashes=dataset.data_hashes,
                           provisional=args.provisional)
        for worker in ("timing", "area")
    }
    output_dir = PROJECT / "configs/workers"
    # Provisional exploration must never replace a previously published final fit.
    if args.provisional:
        model_names = ("A_fit_PROVISIONAL.json", "B_fit_PROVISIONAL.json")
        report_name = "fit_report_PROVISIONAL.md"
    else:
        model_names = ("A_fit.json", "B_fit.json")
        report_name = "fit_report.md"
    _atomic_bytes(output_dir / model_names[0], _json_bytes(models["timing"]))
    _atomic_bytes(output_dir / model_names[1], _json_bytes(models["area"]))
    _atomic_bytes(PROJECT / "notes" / report_name, _report(models, dataset).encode("utf-8"))
    print(json.dumps({"label": "PROVISIONAL" if args.provisional else "FINAL",
                      "timing": models["timing"]["keep_or_drop"],
                      "area": models["area"]["keep_or_drop"],
                      "data_hashes": dataset.data_hashes}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
