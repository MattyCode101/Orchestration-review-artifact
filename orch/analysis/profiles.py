"""Behaviour-profile generator ("onboarding"), final specification section 5.

A profile tells the orchestrator, in frequency words and without numbers or recipe
names, what a worker (or, for the general agent, a recipe) tends to do. It is built
from training designs only, from stored outcomes only:

  * every observation is one simulated worker call (or one recipe application) on a
    training state that has full stored coverage, binned with the same categories and
    noise bands as the orchestrator's view (orch.policies.llm_view);
  * each design counts equally, however many observations it contributed;
  * frequencies become words: rarely under 15%, sometimes up to 40%, often up to 70%,
    usually above 70%.

This module is pure aggregation, text and hashing. The scripts that read the stores
turn their outcomes into `Observation`s. The same function onboards worker C and any
later worker. Output files are deterministic (no timestamps) so their hashes can be
frozen; only manifest.json records when it was written.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from orch.contract.models import MetricName
from orch.policies.llm_view import (
    CATEGORIES, HIGHER_IS_BETTER, METRIC_NAMES, SHOWN, categorise, effective_bands,
)

SCHEMA = "behaviour-profile-v1"
MENTION_GOAL = 0.05
MENTION_OTHER = 0.15
SETTING_ORDER = ("preserve_placement_true", "preserve_placement_false")
SETTING_LABEL = {"preserve_placement_true": "preserve_placement=true",
                 "preserve_placement_false": "preserve_placement=false"}


def frequency_word(share: float) -> str:
    if share < 0.15:
        return "rarely"
    if share <= 0.40:
        return "sometimes"
    if share <= 0.70:
        return "often"
    return "usually"


@dataclass(frozen=True)
class Observation:
    design: str
    setting: str
    valid: bool
    changes: Mapping[str, str]
    no_change: bool


def observation_from_metrics(
    design: str, setting: str, base: Mapping[MetricName, float],
    result: Mapping[MetricName, float] | None, *,
    start_metrics: Mapping[MetricName, float], noise: Mapping[MetricName, float],
) -> Observation:
    """Bin one result against its base state; `result=None` means no usable state came back."""

    if result is None:
        return Observation(design, setting, False, {}, False)
    bands = effective_bands(noise, start_metrics)
    changes: dict[str, str] = {}
    for metric in SHOWN:
        if metric in bands and metric in base and metric in result:
            changes[METRIC_NAMES[metric]] = categorise(
                result[metric] - base[metric], bands[metric], higher_is_better=metric in HIGHER_IS_BETTER,
            )
    quiet = bool(changes) and all(category == "within_noise" for category in changes.values())
    return Observation(design, setting, True, changes, quiet)


@dataclass(frozen=True)
class SettingStats:
    calls: int
    designs: int
    invalid_rate: float
    no_change_rate: float
    metrics: dict[str, dict[str, float]]


def summarise(observations: Sequence[Observation]) -> dict[str, SettingStats]:
    """Per setting, average per-design frequencies so every design weighs the same."""

    grouped: dict[str, dict[str, list[Observation]]] = defaultdict(lambda: defaultdict(list))
    for obs in observations:
        grouped[obs.setting][obs.design].append(obs)
    out: dict[str, SettingStats] = {}
    for setting, by_design in grouped.items():
        invalid, quiet = [], []
        shares: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for items in by_design.values():
            total = len(items)
            valid = [o for o in items if o.valid]
            invalid.append((total - len(valid)) / total)
            quiet.append(sum(o.no_change for o in valid) / total)
            if not valid:
                continue
            for metric in {m for o in valid for m in o.changes}:
                observed = [o.changes[metric] for o in valid if metric in o.changes]
                for category in CATEGORIES:
                    shares[metric][category].append(observed.count(category) / len(observed))
        metrics = {
            metric: {c: sum(v) / len(v) for c, v in categories.items() if sum(v) > 0}
            for metric, categories in shares.items()
        }
        out[setting] = SettingStats(
            calls=sum(len(v) for v in by_design.values()), designs=len(by_design),
            invalid_rate=sum(invalid) / len(invalid), no_change_rate=sum(quiet) / len(quiet),
            metrics=metrics,
        )
    return out


def _clause(metric: str, shares: Mapping[str, float], threshold: float, label: str | None = None) -> str | None:
    ranked = sorted(((s, c) for c, s in shares.items() if s >= threshold - 1e-9), key=lambda t: (-t[0], t[1]))
    if not ranked:
        return None
    return f"{label or metric} " + ", ".join(f"{category} {frequency_word(share)}" for share, category in ranked)


def _setting_line(stats: SettingStats, goal: str | None) -> str:
    clauses = []
    if goal is not None and goal in stats.metrics:
        clause = _clause(goal, stats.metrics[goal], MENTION_GOAL, label="goal")
        if clause:
            clauses.append(clause)
    for metric in sorted(stats.metrics):
        if metric == goal:
            continue
        clause = _clause(metric, stats.metrics[metric], MENTION_OTHER)
        if clause:
            clauses.append(clause)
    clauses.append(f"no change {frequency_word(stats.no_change_rate)}")
    clauses.append(f"invalid {frequency_word(stats.invalid_rate)}")
    return "; ".join(clauses) + "."


def render_worker_text(stats: Mapping[str, SettingStats], goal: str) -> str:
    return "\n".join(
        f"  {SETTING_LABEL[setting]}: {_setting_line(stats[setting], goal)}"
        for setting in SETTING_ORDER if setting in stats
    )


def render_recipe_text(recipe_id: str, stats: SettingStats, *, replaces: bool, targets: str) -> str:
    placement = "re-places the design" if replaces else "keeps the placement"
    clauses = []
    for metric in sorted(stats.metrics):
        clause = _clause(metric, stats.metrics[metric], MENTION_OTHER)
        if clause:
            clauses.append(clause)
    clauses += [f"no change {frequency_word(stats.no_change_rate)}", f"invalid {frequency_word(stats.invalid_rate)}"]
    return f"{recipe_id}: targets {targets}; {placement}. Observed: " + "; ".join(clauses) + "."


def _stats_dict(stats: SettingStats) -> dict[str, Any]:
    return {
        "calls": stats.calls, "designs": stats.designs,
        "invalid_rate": round(stats.invalid_rate, 4), "no_change_rate": round(stats.no_change_rate, 4),
        "metrics": {m: {c: round(s, 4) for c, s in sorted(v.items())} for m, v in sorted(stats.metrics.items())},
    }


def _dump(data: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(dict(data), sort_keys=True, default_flow_style=False, allow_unicode=True).encode("utf-8")


def _sha(data: bytes | str) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()


def write_profiles(
    out_dir: Path | str, workers: Sequence[Mapping[str, Any]], recipes: Mapping[str, Mapping[str, Any]],
    *, provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one file per worker, one recipes file and a manifest of their SHA-256 values."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for worker in workers:
        text = render_worker_text(worker["stats"], worker["objective"])
        data = {
            "schema_version": SCHEMA, "kind": "worker", "worker": worker["worker"],
            "objective": worker["objective"], "generator": "orch/analysis/profiles.py",
            "provenance": dict(worker.get("provenance", {})), "common": dict(provenance),
            "settings": {s: _stats_dict(v) for s, v in sorted(worker["stats"].items())},
            "text": text, "text_sha256": _sha(text),
        }
        payload = _dump(data)
        (out / f"{worker['worker']}.yaml").write_bytes(payload)
        files[f"{worker['worker']}.yaml"] = _sha(payload)
    if recipes:
        entries, texts = {}, {}
        for recipe_id, item in sorted(recipes.items()):
            targets = item.get("targets") or item["worker"]
            text = render_recipe_text(recipe_id, item["stats"], replaces=bool(item["replaces"]), targets=targets)
            entries[recipe_id] = {
                "worker": item["worker"], "replaces": bool(item["replaces"]), "targets": targets,
                "stats": _stats_dict(item["stats"]), "text": text, "text_sha256": _sha(text),
            }
            texts[recipe_id] = text
        data = {"schema_version": SCHEMA, "kind": "recipes", "generator": "orch/analysis/profiles.py",
                "common": dict(provenance), "recipes": entries}
        payload = _dump(data)
        (out / "recipes.yaml").write_bytes(payload)
        files["recipes.yaml"] = _sha(payload)
    manifest = {
        "schema_version": SCHEMA, "files": files, "provenance": dict(provenance),
        "written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return manifest
