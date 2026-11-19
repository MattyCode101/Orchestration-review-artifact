"""EXPLORATORY, training-only worker choice comparisons.

Every outer fold holds out one of the five training designs. The published
FINAL keep/drop and penalty are treated as frozen choices and its coefficients
are re-fit on the other four designs. V2 and V3 select their shared complexity
by an inner leave-one-design-out MSE. No model is published or switched here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from orch.analysis.fit_workers import (
    FROZEN_SHA256, TRAINING_DESIGNS, TRAINING_ROOT, TRAINING_V3_SOURCE,
    TrainingRow, _fit_parameters, _score_parameter, _table,
    deduplicate_rows, load_feature_spec, load_training_rows,
)
from orch.engine.store import _atomic_bytes, _json_bytes

PROJECT = Path(__file__).resolve().parents[2]
REPORT = PROJECT / "notes/exploratory_worker_analysis.md"
DETAIL = PROJECT / "notes/exploratory_worker_analysis.json"
SHRINKAGE_PSEUDOSTATES = 5.0
RIDGE_GRID = tuple(float(x) for x in np.logspace(-3, 3, 13))
METHODS = ("null", "FINAL", "oracle", "V1", "V2", "V3")
TRIALS = (1, 2, 3)


@dataclass(frozen=True)
class StateRows:
    """Five realised alternatives for one content-addressed input state."""

    design: str
    state_id: str
    context: str
    features: Mapping[str, float | None]
    gains: Mapping[str, float]


@dataclass(frozen=True)
class LinearModel:
    """One shared feature basis and one coefficient vector per recipe."""

    recipes: tuple[str, ...]
    columns: tuple[str, ...]
    norms: Mapping[str, tuple[float, float, float, float]]
    betas: np.ndarray
    slopes: np.ndarray


def _number(value: object) -> float | None:
    if type(value) not in (int, float) or not math.isfinite(value):
        return None
    return float(value)


def _at(record: Mapping[str, Any], path: Sequence[str]) -> object:
    current: object = record
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _diagnostic(state_id: str, spec: Mapping[str, Any]) -> float | None:
    """Read only this input's named v3 record for the frozen diagnostic."""

    if len(state_id) != 64 or any(char not in "0123456789abcdef" for char in state_id):
        raise ValueError("input state is not a SHA-256")
    root = TRAINING_V3_SOURCE.resolve(strict=True)
    path = root / "states" / state_id / "dcp_telemetry.json"
    if not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("diagnostic telemetry escapes named v3 source")
    record = json.loads(path.read_bytes())
    if record.get("state_sha256") != state_id:
        raise ValueError("diagnostic telemetry names another input")
    inputs = spec["diagnostics"]["dont_touch_share"]["inputs"]
    numerator = _number(_at(record, inputs["n_DONT_TOUCH"]))
    denominator = _number(_at(record, inputs["primitive_cell_count"]))
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _context(
    alternatives: Mapping[str, TrainingRow], spec: Mapping[str, Any],
    state_id: str, start_ids: set[str],
) -> str:
    """Give canonical B2 priority; preserve ambiguous derived lineages."""

    if state_id in start_ids:
        return "B2"
    contexts = set()
    for row in alternatives.values():
        if len(row.sequence) == 1:
            contexts.add("B2")
        else:
            producer = row.sequence[-2]
            worker = spec["recipes"][producer]["worker"]
            contexts.add("A" if worker == "timing" else "B")
    return next(iter(contexts)) if len(contexts) == 1 else "ambiguous"


def _feature_names(spec: Mapping[str, Any]) -> dict[str, str]:
    """One representative recipe per frozen feature, plus the diagnostic."""

    result = {}
    for recipe_id, definition in sorted(spec["recipes"].items()):
        result.setdefault(definition["feature"], recipe_id)
    return result


def _states(
    table: Mapping[tuple[str, str], Mapping[str, TrainingRow]],
    recipes: Sequence[str],
    spec: Mapping[str, Any],
    diagnostics: Mapping[str, float | None],
    start_ids: set[str],
) -> list[StateRows]:
    representatives = _feature_names(spec)
    result = []
    for (design, state_id), alternatives in sorted(table.items()):
        first = alternatives[recipes[0]]
        features = {
            name: _number(first.features.get(recipe_id))
            for name, recipe_id in representatives.items()
        }
        features["dont_touch_share"] = diagnostics[state_id]
        result.append(StateRows(
            design, state_id, _context(alternatives, spec, state_id, start_ids), features,
            {recipe_id: alternatives[recipe_id].gain(spec["recipes"][recipe_id]["worker"])
             for recipe_id in recipes},
        ))
    return result


def _weights(states: Sequence[StateRows]) -> np.ndarray:
    counts = Counter(state.design for state in states)
    if not counts:
        raise ValueError("model has no training states")
    return np.array([
        1.0 / (len(counts) * counts[state.design]) for state in states
    ], dtype=float)


def _design_mean(states: Sequence[StateRows], recipe_id: str) -> float:
    groups: dict[str, list[float]] = defaultdict(list)
    for state in states:
        groups[state.design].append(state.gains[recipe_id])
    return float(np.mean([np.mean(values) for _, values in sorted(groups.items())]))


def _feature_basis(
    states: Sequence[StateRows],
) -> tuple[tuple[str, ...], dict[str, tuple[float, float, float, float]], dict[str, str]]:
    """Use every available nonconstant frozen feature; drop absent census."""

    names = sorted(states[0].features)
    columns = []
    norms = {}
    dropped = {}
    for name in names:
        present = [state for state in states if _number(state.features.get(name)) is not None]
        missing = 1.0 - len(present) / len(states)
        if missing > 0.10:
            dropped[name] = f"missing {missing:.1%}"
            continue
        values = [float(state.features[name]) for state in present]
        if min(values) == max(values):
            dropped[name] = "constant"
            continue
        weight = _weights(present)
        mean = float(np.dot(weight, values))
        variance = float(np.dot(weight, (np.asarray(values) - mean) ** 2))
        sd = math.sqrt(variance)
        if sd <= 0 or not math.isfinite(sd):
            dropped[name] = "zero weighted variance"
            continue
        columns.append(name)
        norms[name] = (mean, sd, min(values), max(values))
    return tuple(columns), norms, dropped


def _matrix(states: Sequence[StateRows], model: LinearModel) -> np.ndarray:
    """Impute a missing input at the train mean and clip to train range."""

    matrix = np.zeros((len(states), len(model.columns)), dtype=float)
    for i, state in enumerate(states):
        for j, name in enumerate(model.columns):
            mean, sd, low, high = model.norms[name]
            value = _number(state.features.get(name))
            if value is not None:
                matrix[i, j] = (min(high, max(low, value)) - mean) / sd
    return matrix


def _pls_slope(
    x: np.ndarray, y: np.ndarray, weight: np.ndarray, components: int,
) -> np.ndarray:
    """Weighted one-response PLS1 with at most two deflated components."""

    if x.shape[1] == 0:
        return np.zeros(0)
    residual_x = x.copy()
    residual_y = y.copy()
    w_columns = []
    p_columns = []
    q_values = []
    for _ in range(min(components, x.shape[1])):
        direction = residual_x.T @ (weight * residual_y)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            break
        direction /= norm
        score = residual_x @ direction
        denominator = float(np.dot(weight, score * score))
        if denominator <= 1e-12:
            break
        loading = residual_x.T @ (weight * score) / denominator
        response_loading = float(np.dot(weight, score * residual_y) / denominator)
        w_columns.append(direction)
        p_columns.append(loading)
        q_values.append(response_loading)
        residual_x -= np.outer(score, loading)
        residual_y -= score * response_loading
    if not w_columns:
        return np.zeros(x.shape[1])
    w = np.column_stack(w_columns)
    p = np.column_stack(p_columns)
    q = np.asarray(q_values)
    return w @ np.linalg.pinv(p.T @ w) @ q


def _linear_fit(
    states: Sequence[StateRows], recipes: tuple[str, ...],
    *, method: str, setting: float | int,
) -> LinearModel:
    columns, norms, _ = _feature_basis(states)
    beta = np.array([_design_mean(states, recipe_id) for recipe_id in recipes])
    empty = LinearModel(recipes, columns, norms, beta, np.zeros((len(columns), len(recipes))))
    x = _matrix(states, empty)
    if not columns:
        return empty
    weight = _weights(states)
    y = np.array([[state.gains[recipe_id] for recipe_id in recipes] for state in states])
    centered = y - beta
    if method == "V2":
        penalty = float(setting)
        matrix = x.T @ (weight[:, None] * x) + penalty * np.eye(x.shape[1])
        slopes = np.linalg.solve(matrix, x.T @ (weight[:, None] * centered))
    elif method == "V3":
        slopes = np.column_stack([
            _pls_slope(x, centered[:, index], weight, int(setting))
            for index in range(len(recipes))
        ])
    else:
        raise ValueError("unknown all-feature model")
    return LinearModel(recipes, columns, norms, beta, slopes)


def _linear_scores(model: LinearModel, states: Sequence[StateRows]) -> np.ndarray:
    return _matrix(states, model) @ model.slopes + model.betas


def _choose_setting(
    states: Sequence[StateRows], recipes: tuple[str, ...], method: str,
) -> float | int:
    """Inner design-held-out MSE chooses one shared complexity value."""

    grid: Sequence[float | int] = RIDGE_GRID if method == "V2" else (1, 2)
    designs = sorted({state.design for state in states})
    best = None
    for setting in grid:
        errors = []
        for held in designs:
            train = [state for state in states if state.design != held]
            test = [state for state in states if state.design == held]
            model = _linear_fit(train, recipes, method=method, setting=setting)
            prediction = _linear_scores(model, test)
            realised = np.array([
                [state.gains[recipe_id] for recipe_id in recipes] for state in test
            ])
            errors.append(float(np.mean((prediction - realised) ** 2)))
        mse = float(np.mean(errors))
        candidate = (mse, float(setting))
        if best is None or candidate < best[0]:
            best = candidate, setting
    assert best is not None
    return best[1]


def _v1_fit(
    states: Sequence[StateRows], recipes: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Shrink each B2/A/B context mean toward its overall recipe beta."""

    result = {}
    for recipe_id in recipes:
        beta = _design_mean(states, recipe_id)
        scores = {"overall": beta}
        for context in ("B2", "A", "B"):
            subset = [state for state in states if state.context == context]
            if not subset:
                scores[context] = beta
                continue
            conditioned = _design_mean(subset, recipe_id)
            strength = len(subset) / (len(subset) + SHRINKAGE_PSEUDOSTATES)
            scores[context] = beta + strength * (conditioned - beta)
        result[recipe_id] = scores
    return result


def _rank(scores: Mapping[str, float]) -> tuple[str, ...]:
    return tuple(sorted(scores, key=lambda recipe_id: (-scores[recipe_id], recipe_id)))


def _fold_metrics(
    states: Sequence[StateRows], recipes: tuple[str, ...],
    predicted: Mapping[str, Sequence[Mapping[str, float]]],
) -> tuple[dict[str, dict[int, float]], dict[str, set[str]]]:
    regrets = {}
    first_choices = {}
    for method, score_rows in predicted.items():
        if len(score_rows) != len(states):
            raise ValueError("score rows do not match held-out states")
        first_choices[method] = set()
        values = {trials: [] for trials in TRIALS}
        for state, scores in zip(states, score_rows):
            order = _rank(scores)
            first_choices[method].add(order[0])
            oracle = max(state.gains.values())
            for trials in TRIALS:
                achieved = max(state.gains[recipe_id] for recipe_id in order[:trials])
                values[trials].append(oracle - achieved)
        regrets[method] = {
            trials: float(np.mean(values[trials])) for trials in TRIALS
        }
    return regrets, first_choices


def switch_rule(
    final_folds: Sequence[float], variant_folds: Sequence[float],
) -> dict[str, object]:
    """Apply the pre-declared 10% average and four-of-five-fold rule."""

    if len(final_folds) != 5 or len(variant_folds) != 5:
        raise ValueError("switch rule requires five held-design folds")
    final_mean = float(np.mean(final_folds))
    variant_mean = float(np.mean(variant_folds))
    if not all(math.isfinite(value) and value >= 0
               for value in (*final_folds, *variant_folds)):
        raise ValueError("switch rule requires finite nonnegative regrets")
    improved = sum(variant < final for final, variant in zip(final_folds, variant_folds))
    lower_by_ten = final_mean > 0 and variant_mean <= 0.9 * final_mean
    return {
        "mean_regret_10pct_lower": lower_by_ten,
        "folds_strictly_lower": improved,
        "eligible_for_author_review": lower_by_ten and improved >= 4,
    }


def evaluate_worker(
    rows: Sequence[TrainingRow], worker: str,
    diagnostics: Mapping[str, float | None],
    spec: Mapping[str, Any],
    published: Mapping[str, Any],
    start_ids: set[str],
) -> dict[str, Any]:
    """Run five outer held-design folds without using held outcomes for fits."""

    recipes = tuple(sorted(
        recipe_id for recipe_id, definition in spec["recipes"].items()
        if definition["worker"] == worker
    ))
    table = _table(rows, recipes)
    states = _states(table, recipes, spec, diagnostics, start_ids)
    designs = tuple(sorted({state.design for state in states}))
    if designs != tuple(sorted(TRAINING_DESIGNS)):
        raise ValueError("exploratory fit needs exactly the five training designs")
    if (
        published.get("label") != "FINAL"
        or published.get("worker") != worker
        or published.get("feature_spec_sha256") != FROZEN_SHA256
        or set(published.get("recipes", {})) != set(recipes)
    ):
        raise ValueError("published FINAL model differs from the frozen worker")
    penalty = float(published["penalty"])
    keep = bool(published["keep_features"])
    folds = []
    choices: dict[str, set[str]] = {name: set() for name in METHODS}
    selected_settings: dict[str, list[object]] = {"V2": [], "V3": []}
    for held in designs:
        train = [state for state in states if state.design != held]
        test = [state for state in states if state.design == held]
        train_designs = [design for design in designs if design != held]
        frozen = _fit_parameters(table, train_designs, recipes, worker, spec, penalty)
        v1 = _v1_fit(train, recipes)
        selected = {
            method: _choose_setting(train, recipes, method) for method in ("V2", "V3")
        }
        for method, setting in selected.items():
            selected_settings[method].append(setting)
        linear = {
            method: _linear_fit(train, recipes, method=method, setting=setting)
            for method, setting in selected.items()
        }
        predicted: dict[str, list[dict[str, float]]] = {
            name: [] for name in METHODS
        }
        matrices = {
            method: _linear_scores(model, test)
            for method, model in linear.items()
        }
        for index, state in enumerate(test):
            source = next(iter(table[(state.design, state.state_id)].values()))
            features = source.features
            predicted["null"].append({
                recipe_id: float(frozen[recipe_id]["beta"]) for recipe_id in recipes
            })
            predicted["FINAL"].append({
                recipe_id: _score_parameter(
                    frozen[recipe_id], features.get(recipe_id), use_terms=keep
                )
                for recipe_id in recipes
            })
            predicted["oracle"].append(dict(state.gains))
            predicted["V1"].append({
                recipe_id: v1[recipe_id].get(state.context, v1[recipe_id]["overall"])
                for recipe_id in recipes
            })
            for method in ("V2", "V3"):
                predicted[method].append({
                    recipe_id: float(matrices[method][index, recipe_index])
                    for recipe_index, recipe_id in enumerate(recipes)
                })
        regrets, first = _fold_metrics(test, recipes, predicted)
        for method, recipe_ids in first.items():
            choices[method].update(recipe_ids)
        folds.append({
            "held_design": held,
            "input_states": len(test),
            "regret_pct_points": {
                method: {str(trials): regrets[method][trials] for trials in TRIALS}
                for method in METHODS
            },
            "distinct_first_choices": {
                method: sorted(first[method]) for method in METHODS
            },
            "selected_setting": selected,
        })
    averages = {
        method: {
            str(trials): float(np.mean([
                fold["regret_pct_points"][method][str(trials)] for fold in folds
            ]))
            for trials in TRIALS
        }
        for method in METHODS
    }
    switch = {}
    final_fold_regrets = [
        fold["regret_pct_points"]["FINAL"]["1"] for fold in folds
    ]
    for variant in ("V1", "V2", "V3"):
        switch[variant] = switch_rule(
            final_fold_regrets,
            [fold["regret_pct_points"][variant]["1"] for fold in folds],
        )
    _, _, dropped = _feature_basis(states)
    return {
        "worker": worker,
        "recipes": recipes,
        "input_states": len(states),
        "contexts": dict(Counter(state.context for state in states)),
        "all_feature_columns": sorted(states[0].features),
        "all_feature_drops_on_full_training": dropped,
        "published_final_penalty": penalty,
        "published_final_keep_features": keep,
        "folds": folds,
        "average_regret_pct_points": averages,
        "first_choice_diversity": {
            method: sorted(choices[method]) for method in METHODS
        },
        "variant_settings_by_fold": selected_settings,
        "switch_rule": switch,
    }


def _read_published(name: str, worker: str) -> tuple[dict[str, Any], str]:
    path = PROJECT / "configs/workers" / name
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("worker") != worker or payload.get("label") != "FINAL":
        raise ValueError("exploratory comparison requires FINAL worker JSON")
    return payload, hashlib.sha256(raw).hexdigest()


def _render(result: Mapping[str, Any]) -> str:
    workers = result["workers"]
    table = []
    fold_rows = []
    switch_rows = []
    for worker_id in ("timing", "area"):
        worker = workers[worker_id]
        for method in METHODS:
            scores = worker["average_regret_pct_points"][method]
            diversity = worker["first_choice_diversity"][method]
            table.append(
                f"| {worker_id} | {method} | {scores['1']:.3f} | "
                f"{scores['2']:.3f} | {scores['3']:.3f} | "
                f"{len(diversity)} ({', '.join(diversity)}) |"
            )
        for fold in worker["folds"]:
            fold_rows.append(
                f"| {worker_id} | {fold['held_design']} | "
                + " | ".join(
                    f"{fold['regret_pct_points'][method]['1']:.3f}"
                    for method in ("null", "FINAL", "V1", "V2", "V3")
                )
                + " |"
            )
        for variant, rule in worker["switch_rule"].items():
            switch_rows.append(
                f"| {worker_id} | {variant} | "
                f"{'yes' if rule['mean_regret_10pct_lower'] else 'no'} | "
                f"{rule['folds_strictly_lower']}/5 | "
                f"{'eligible for the authors review' if rule['eligible_for_author_review'] else 'FINAL stands'} |"
            )
    statuses = [
        f"{worker_id}: " + ", ".join(
            f"{variant} {'eligible' if row['eligible_for_author_review'] else 'fails'}"
            for variant, row in workers[worker_id]["switch_rule"].items()
        )
        for worker_id in ("timing", "area")
    ]
    lines = [
        "# EXPLORATORY training-only worker analysis",
        "",
        "## Summary",
        "",
        "**Verified, EXPLORATORY.** " + "; ".join(statuses)
        + ". No FINAL JSON was changed. Regret is the percentage-point gap "
        "between the best realised recipe gain and the best realised gain "
        "among the top 1, 2 or 3 predicted choices, averaged within each "
        "held-out design and then equally across five designs.",
        "",
        "**Method.** Each outer fold holds one training design out. Null and "
        "FINAL coefficients use the other four designs; FINAL keeps its "
        "already-published keep/drop choice and ridge penalty. Those two "
        "settings were selected from all five designs before this analysis, "
        "so the FINAL cross-validation estimate is conditional and can be "
        "optimistic. V1 shrinks the B2/A/B producer-context mean toward the "
        "design-weighted recipe mean with five pseudo-states. A hash reachable "
        "through multiple producer classes is labelled ambiguous and uses "
        "the overall beta. V2 fits every available nonconstant frozen feature "
        "with one shared ridge penalty. V3 uses weighted PLS1 with at most two "
        "components per recipe on that same feature set. V2/V3 choose "
        "complexity by inner design-held-out MSE; all scaling and clipping "
        "are fitted inside each training fold. The absent duplicate-driver "
        "census and constant or unavailable diagnostics are dropped.",
        "",
        "## Equal-design mean regret and first-choice diversity",
        "",
        "| Worker | Method | 1 try | 2 tries | 3 tries | Distinct first choices across held states |",
        "|---|---|---:|---:|---:|---|",
        *table,
        "",
        "## One-try regret by held-out design",
        "",
        "| Worker | Held design | Null | FINAL | V1 | V2 | V3 |",
        "|---|---|---:|---:|---:|---:|---:|",
        *fold_rows,
        "",
        "## Pre-declared switching rule",
        "",
        "A variant qualifies for the authors' review only if mean one-try regret is "
        "at least 10% below FINAL and strictly lower in at least four of five "
        "outer folds. This analysis never switches a model.",
        "",
        "| Worker | Variant | Mean at least 10% lower | Improved folds | Outcome |",
        "|---|---|---|---:|---|",
        *switch_rows,
        "",
        "## Feature coverage and provenance",
        "",
        *[
            f"- {worker_id}: {workers[worker_id]['input_states']} distinct input states; "
            f"contexts {workers[worker_id]['contexts']}; full-training dropped "
            f"columns {workers[worker_id]['all_feature_drops_on_full_training']}."
            for worker_id in ("timing", "area")
        ],
        f"- Analysis code SHA-256 {result['analysis_code_sha256']}; "
        f"frozen feature YAML SHA-256 {result['feature_spec_sha256']}; "
        f"FINAL A JSON SHA-256 {result['final_model_sha256']['timing']}; "
        f"FINAL B JSON SHA-256 {result['final_model_sha256']['area']}.",
        f"- Checked training data hashes: {result['training_data_hashes']}.",
        "- Limit: D3 expansion selected branches; these folds describe the "
        "five training designs and do not measure held-out performance. "
        "The published FINAL models remain unchanged pending the authors' Thursday decision.",
        "",
    ]
    return "\n".join(lines)


def run(data_root: Path = TRAINING_ROOT) -> dict[str, Any]:
    """Load only the pinned training source, evaluate and write labelled files."""

    dataset = load_training_rows(data_root, telemetry_source="training-v3")
    if dataset.provisional:
        raise ValueError("exploratory comparison refuses provisional training rows")
    rows, counts = deduplicate_rows(dataset.rows)
    spec = load_feature_spec()
    state_ids = {row.input_state_id for row in rows}
    start_ids = {
        row.input_state_id for row in dataset.rows if len(row.sequence) == 1
    }
    if len(start_ids) != 5:
        raise ValueError("training D1 does not identify five B2 starts")
    diagnostics = {
        state_id: _diagnostic(state_id, spec) for state_id in sorted(state_ids)
    }
    models = {
        "timing": _read_published("A_fit.json", "timing"),
        "area": _read_published("B_fit.json", "area"),
    }
    workers = {
        worker: evaluate_worker(
            rows, worker, diagnostics, spec, models[worker][0], start_ids
        )
        for worker in ("timing", "area")
    }
    result = {
        "label": "EXPLORATORY",
        "schema_version": "worker-exploration-v1",
        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_spec_sha256": FROZEN_SHA256,
        "training_data_hashes": dict(dataset.data_hashes),
        "dataset_counts": dict(dataset.counts),
        "deduplication": counts,
        "final_model_sha256": {
            worker: models[worker][1] for worker in models
        },
        "v1_shrinkage_pseudostates": SHRINKAGE_PSEUDOSTATES,
        "v2_ridge_penalty_grid": RIDGE_GRID,
        "v3_components_grid": (1, 2),
        "workers": workers,
        "model_switch_performed": False,
    }
    _atomic_bytes(DETAIL, _json_bytes(result))
    REPORT.write_text(_render(result), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=TRAINING_ROOT)
    args = parser.parse_args(argv)
    result = run(args.data_root)
    for worker, row in result["workers"].items():
        print(worker, {name: row["average_regret_pct_points"][name]["1"]
                       for name in METHODS})
        print("switch", row["switch_rule"])
    print(REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

