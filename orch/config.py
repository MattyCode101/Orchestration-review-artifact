"""Validated YAML configuration for briefs, recipe sets and noise bands."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field, model_validator

from orch.contract.models import Brief, Evidence, FrozenModel, MetricName, WorkerCard


class RecipeSetsConfig(FrozenModel):
    """Scientific recipe choices remain unsigned until the authors review them."""

    signed_off: bool = False
    worker_sets: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    selection_note: str = ""

    @model_validator(mode="after")
    def unique_recipe_ids(self) -> "RecipeSetsConfig":
        for worker, recipes in self.worker_sets.items():
            if len(set(recipes)) != len(recipes):
                raise ValueError(f"duplicate recipe in {worker} set")
        return self

    def require_signed_off(self) -> None:
        """Prevent an unsigned candidate set from becoming a research run."""

        if not self.signed_off:
            raise ValueError("recipe sets require the authors' scientific sign-off")


class NoiseBand(FrozenModel):
    """Absolute band or fraction of the start metric's magnitude."""

    value: float = Field(ge=0)
    basis: str
    relative: bool = False
    samples: int = Field(ge=1)
    source: str

    def absolute(self, start_value: float) -> float:
        return self.value * abs(start_value) if self.relative else self.value


class NoiseConfig(FrozenModel):
    bands: dict[str, dict[MetricName, NoiseBand]] = Field(default_factory=dict)

    def for_start(self, design: str, start: Evidence) -> dict[MetricName, float]:
        """Return absolute bands that the pure gate can compare with deltas."""

        result: dict[MetricName, float] = {}
        for name, band in self.bands.get(design, {}).items():
            metric = start.metric(name)
            if metric is not None:
                result[name] = band.absolute(metric.value)
        return result


def read_yaml(path: Path) -> object:
    """Load only a named YAML file; no environment files are consulted."""

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_brief(path: Path) -> Brief:
    return Brief.model_validate(read_yaml(path))


def load_recipe_sets(path: Path) -> RecipeSetsConfig:
    return RecipeSetsConfig.model_validate(read_yaml(path))


def load_noise(path: Path) -> NoiseConfig:
    return NoiseConfig.model_validate(read_yaml(path))


def load_worker_card(path: Path) -> WorkerCard:
    return WorkerCard.model_validate(read_yaml(path))

