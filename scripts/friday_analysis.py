"""Shared read-only helpers for the Friday notes (training designs only).

Denominators for "share of best known" are the checked values saved with the
v4 pilot (its best-known pool already holds the D1-D3 edges, the cross-worker
depth-three campaign and every v3/v4 pilot step), so every share here is
directly comparable with the published pilot table. Nothing here launches a
tool job or opens a test-design path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from orch.analysis.report import share_of_best
from orch.config import load_noise
from orch.contract.models import Evidence

PROJECT = Path(__file__).resolve().parents[1]
V4 = PROJECT / "data/pilots/thursday_overnight_train_v4_2026-11-05"
SWEEP = PROJECT / "data/pilots/friday_repair_sweep_2026-11-06"
P0ROOT = PROJECT / "data/pilots/friday_p0_pilot_2026-11-06"
DESIGNS = (
    "vexriscv_re-place", "logicnets_jscl", "rosetta_spam-filter",
    "rosetta_3d-rendering", "finn_radioml",
)
POLICIES = (
    "repair_loop", "single_specialist", "fixed_AB", "fixed_BA",
    "random_seed_0", "random_seed_1", "random_seed_2",
)
BRIEFS = ("T0", "A0", "T", "A")
OBJECTIVE = {
    "T0": ("wns_ns", "max"), "A0": ("packed_luts", "min"),
    "T": ("wns_ns", "max"), "A": ("packed_luts", "min"),
    "P0": ("dynamic_power_w", "min"),
}
LIMIT_METRIC = {"T0": "packed_luts", "A0": "wns_ns", "T": "packed_luts", "A": "wns_ns"}


def read(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def start_evidence() -> dict[str, Evidence]:
    plan = read(V4 / "plan.json")
    version_key = hashlib.sha256(plan["meter_version"].encode()).hexdigest()
    return {
        design: Evidence.model_validate_json(
            (V4 / "measurements" / plan["training_inputs"][design] / version_key
             / "evidence.json").read_bytes()
        )
        for design in DESIGNS
    }


def start_metrics() -> dict[str, dict[str, float]]:
    return {
        design: {name.value: metric.value for name, metric in ev.metrics.items()}
        for design, ev in start_evidence().items()
    }


def noise_bands() -> dict[str, dict[str, float]]:
    """Absolute per-design bands (metric name -> value) from configs/noise.yaml."""

    config = load_noise(PROJECT / "configs/noise.yaml")
    return {
        design: {name.value: band for name, band in config.for_start(design, ev).items()}
        for design, ev in start_evidence().items()
    }


def best_known() -> dict[tuple[str, str], dict[str, Any]]:
    summary = read(V4 / "policy_brief_summary.json")["best_known"]
    result = {}
    for key, value in summary.items():
        design, brief = key.split(":")
        result[(design, brief)] = value
    return result


def share(brief: str, design: str, value: float,
          starts: dict[str, dict[str, float]], best: dict[tuple[str, str], dict[str, Any]]) -> float | None:
    metric, direction = OBJECTIVE[brief]
    return share_of_best(starts[design][metric], best[(design, brief)]["value"], value, direction)


def improvement(brief: str, design: str, value: float,
                starts: dict[str, dict[str, float]]) -> float:
    """Signed gain over the B2 start in the brief's own objective direction."""

    metric, direction = OBJECTIVE[brief]
    delta = value - starts[design][metric]
    return delta if direction == "max" else -delta


def result_rows(root: Path) -> list[dict[str, Any]]:
    folder = root / "results"
    return [read(p) for p in sorted(folder.glob("*.json"))] if folder.is_dir() else []
