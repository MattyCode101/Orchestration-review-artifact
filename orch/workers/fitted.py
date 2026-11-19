"""Score frozen timing and area recipes from input-state telemetry.

This worker reads saved model JSON only. It runs no hardware tools and exposes
a score trace so missing measurements remain visible to the caller.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from orch.analysis.fit_workers import FROZEN_SHA256, extract_feature, load_feature_spec, score_recipe
from orch.contract.models import AppliedStep, DesignState, Mandate, WorkerCard
from orch.recipes.registry import Registry
from orch.workers.base import GenericWorker

PROJECT = Path(__file__).resolve().parents[2]
MODEL_NAMES = {
    "timing": ("A_fit.json", "A_null_PROVISIONAL.json"),
    "area": ("B_fit.json", "B_null_PROVISIONAL.json"),
}


@dataclass(frozen=True)
class RecipeScore:
    """One recipe's fitted or beta-fallback score."""

    recipe_id: str
    score_pct: float
    beta_pct: float
    feature_value: float | None
    incomplete_measurement: bool


@dataclass(frozen=True)
class ScoredPlan:
    """Ranked choices and exclusions for one worker call."""

    selected_ids: tuple[str, ...]
    ranked: tuple[RecipeScore, ...]
    excluded_history: tuple[str, ...]
    excluded_replacing: tuple[str, ...]
    excluded_producer: tuple[str, ...]
    incomplete_measurement: bool
    model_label: str
    model_sha256: str


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _read_model(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"worker model is not an object: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


class FittedWorker(GenericWorker):
    """Use a final fit when present, otherwise an explicitly provisional null.

    DesignState lacks its producer recipe. A derived state therefore needs
    produced_by_recipe_id in the supplied context to enforce no reapplication.
    """

    def __init__(
        self, card: WorkerCard, recipe_ids: Sequence[str], registry: Registry,
        *, model_dir: Path | None = None,
    ) -> None:
        super().__init__(card, recipe_ids)
        if card.id not in MODEL_NAMES:
            raise ValueError("fitted scoring is defined only for timing and area")
        self.registry = registry
        self.feature_spec = load_feature_spec()
        expected = {
            recipe_id for recipe_id, definition in self.feature_spec["recipes"].items()
            if definition["worker"] == card.id
        }
        if set(self.recipe_ids) != expected or len(self.recipe_ids) != 5:
            raise ValueError("recipe set differs from frozen feature spec")
        if any(registry.get(recipe_id).worker != card.id for recipe_id in self.recipe_ids):
            raise ValueError("worker recipe set contains a foreign recipe")
        directory = Path(model_dir) if model_dir is not None else PROJECT / "configs/workers"
        fit_name, null_name = MODEL_NAMES[card.id]
        fit_path = directory / fit_name
        self.fit_model: dict[str, Any] | None = None
        if fit_path.is_file():
            candidate, digest = _read_model(fit_path)
            if candidate.get("label") == "FINAL":
                self._validate_fit(candidate)
                self.fit_model = candidate
                self.model_label, self.model_sha256 = "FINAL", digest
        if self.fit_model is None:
            null_path = directory / null_name
            if not null_path.is_file():
                raise FileNotFoundError(f"no final fit or labelled null model: {directory}")
            candidate, digest = _read_model(null_path)
            self._validate_null(candidate)
            self.null_betas = {
                recipe_id: _finite(beta, f"{recipe_id} beta")
                for recipe_id, beta in candidate["beta_pct_by_recipe"].items()
            }
            self.model_label, self.model_sha256 = "PROVISIONAL_NULL", digest
        else:
            self.null_betas = {}

    def _validate_fit(self, model: Mapping[str, Any]) -> None:
        if (
            model.get("schema_version") != "worker-fit-v1"
            or model.get("worker") != self.card.id
            or model.get("feature_spec_sha256") != FROZEN_SHA256
            or model.get("label") != "FINAL"
            or not isinstance(model.get("keep_features"), bool)
            or not isinstance(model.get("recipes"), dict)
            or set(model["recipes"]) != set(self.recipe_ids)
        ):
            raise ValueError("final fit differs from the frozen worker")
        for recipe_id in self.recipe_ids:
            row = model["recipes"][recipe_id]
            if (
                not isinstance(row, dict)
                or row.get("feature") != self.feature_spec["recipes"][recipe_id]["feature"]
                or not isinstance(row.get("feature_available"), bool)
            ):
                raise ValueError(f"invalid model row for {recipe_id}")
            _finite(row.get("beta"), f"{recipe_id} beta")
            _finite(row.get("w"), f"{recipe_id} slope")
            if row["feature_available"]:
                _finite(row.get("feature_mean"), f"{recipe_id} mean")
                sd = _finite(row.get("feature_sd"), f"{recipe_id} sd")
                low = _finite(row.get("z_min"), f"{recipe_id} z_min")
                high = _finite(row.get("z_max"), f"{recipe_id} z_max")
                if sd <= 0 or low > high:
                    raise ValueError(f"invalid standardization for {recipe_id}")

    def _validate_null(self, model: Mapping[str, Any]) -> None:
        if (
            model.get("schema_version") != "worker-null-v1-provisional"
            or model.get("status") != "PROVISIONAL"
            or model.get("worker") != self.card.id
            or model.get("feature_spec_sha256") != FROZEN_SHA256
            or not isinstance(model.get("beta_pct_by_recipe"), dict)
            or set(model["beta_pct_by_recipe"]) != set(self.recipe_ids)
        ):
            raise ValueError("null model differs from the frozen worker")
        for recipe_id, beta in model["beta_pct_by_recipe"].items():
            _finite(beta, f"{recipe_id} beta")

    @staticmethod
    def _context(
        state: DesignState, features: Mapping[str, object] | None,
    ) -> tuple[Mapping[str, Any] | None, str | None]:
        """Verify telemetry identity and obtain the producer recipe."""

        if features is None:
            telemetry, producer = None, None
        elif "telemetry" in features:
            telemetry = features["telemetry"]
            producer = features.get("produced_by_recipe_id")
        else:
            telemetry = features if "state_sha256" in features else None
            producer = features.get("produced_by_recipe_id")
        if telemetry is not None and (
            not isinstance(telemetry, Mapping) or telemetry.get("state_sha256") != state.id
        ):
            raise ValueError("telemetry names a different input state")
        if producer is not None and (not isinstance(producer, str) or not producer):
            raise ValueError("producer recipe ID must be a nonempty string")
        if state.parent_id is not None and producer is None:
            raise ValueError("derived state needs producer recipe ID")
        return telemetry, producer

    def explain_plan(
        self, state: DesignState, features: Mapping[str, object] | None,
        mandate: Mandate, history_on_this_state: Sequence[AppliedStep],
    ) -> ScoredPlan:
        """Filter before taking the top max_runs, then expose every score."""

        remaining = super().plan(state, features, mandate, history_on_this_state)
        excluded_history = tuple(recipe_id for recipe_id in self.recipe_ids if recipe_id not in remaining)
        telemetry, producer = self._context(state, features)
        excluded_replacing: list[str] = []
        excluded_producer: list[str] = []
        scores: list[RecipeScore] = []
        for recipe_id in remaining:
            if self.registry.get(recipe_id).replaces and mandate.settings.preserve_placement:
                excluded_replacing.append(recipe_id)
                continue
            if recipe_id == producer:
                excluded_producer.append(recipe_id)
                continue
            if self.fit_model is None:
                beta, value = self.null_betas[recipe_id], None
                score, incomplete = beta, False
            else:
                beta = _finite(self.fit_model["recipes"][recipe_id]["beta"], f"{recipe_id} beta")
                value = extract_feature(telemetry, recipe_id, self.feature_spec) if telemetry else None
                score, incomplete = score_recipe(self.fit_model, recipe_id, value)
            scores.append(RecipeScore(recipe_id, score, beta, value, incomplete))
        ranked = tuple(sorted(scores, key=lambda row: (-row.score_pct, row.recipe_id)))
        return ScoredPlan(
            selected_ids=tuple(row.recipe_id for row in ranked[:mandate.settings.resolved_max_runs(self.card)]),
            ranked=ranked,
            excluded_history=excluded_history,
            excluded_replacing=tuple(excluded_replacing),
            excluded_producer=tuple(excluded_producer),
            incomplete_measurement=any(row.incomplete_measurement for row in ranked),
            model_label=self.model_label, model_sha256=self.model_sha256,
        )

    def plan(
        self, state: DesignState, features: Mapping[str, object] | None,
        mandate: Mandate, history_on_this_state: Sequence[AppliedStep],
    ) -> tuple[str, ...]:
        """Return the top eligible recipe IDs with alphabetical tie breaking."""

        return self.explain_plan(state, features, mandate, history_on_this_state).selected_ids
