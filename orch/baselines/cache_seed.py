"""Strict, train-scoped import of completed first-step cache entries.

The old campaign remains read-only. Only requested start hashes and selected
recipe IDs are inspected; every imported checkpoint is copied into the new
content-addressed store and every run record keeps its old provenance paths.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

from orch.contract.models import DesignState, Evidence, RunOutcome, RunStatus
from orch.engine.apply import StepEngine
from orch.engine.cache import RunCache
from orch.engine.store import StateStore, _atomic_bytes, _json_bytes
from orch.interfaces import RecipeSpec


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_first_steps(
    *,
    engine: StepEngine,
    source_root: Path,
    starts: Mapping[str, DesignState],
    recipes: Sequence[RecipeSpec],
) -> dict[str, object]:
    """Copy only current-version, complete, independently measured cache hits."""

    if not engine.cache_enabled:
        raise ValueError("cache seeding requires a cache-enabled destination")
    source = Path(source_root).resolve(strict=True)
    if source == engine.store.root:
        raise ValueError("source and destination cache roots must differ")
    if not (source / "cache").is_dir() or not (source / "runs").is_dir():
        raise ValueError("source lacks completed cache and run directories")
    old_cache = RunCache(source)
    old_store = StateStore(source)
    rows: list[dict[str, object]] = []
    starts_seeded: list[str] = []
    for design, start in starts.items():
        engine.store.state_path(start.id)
        version_key = hashlib.sha256(engine.meter_version.encode("utf-8")).hexdigest()
        old_measure = (
            source / "measurements" / start.id / version_key / "evidence.json"
        )
        if old_measure.is_file():
            old_store.state_path(start.id)
            evidence = Evidence.model_validate_json(old_measure.read_bytes())
            if (
                evidence.state_id != start.id
                or evidence.meter_version != engine.meter_version
                or evidence.fake != start.fake
            ):
                raise ValueError("old start Evidence differs from exact start and meter")
            target = (
                engine.store.root / "measurements" / start.id
                / version_key / "evidence.json"
            )
            data = _json_bytes(evidence.model_dump(mode="json"))
            if target.is_file() and target.read_bytes() != data:
                raise ValueError("destination start Evidence differs from source")
            if not target.exists():
                _atomic_bytes(target, data)
            starts_seeded.append(design)
        for recipe in recipes:
            key = engine.cache.key(start.id, recipe.id, recipe.version)
            origin_id = old_cache.get(key)
            if origin_id is None:
                continue
            original = old_store.load_outcome(origin_id)
            if (
                original.run.status != RunStatus.OK
                or original.run.cache_hit
                or original.run.recipe_id != recipe.id
                or original.run.recipe_version != recipe.version
                or original.run.input_sha256_before != start.id
                or original.run.input_sha256_after != start.id
                or original.run.fake != start.fake
                or original.state is None
                or original.state.design != design
                or original.state.parent_id != start.id
                or original.evidence is None
                or original.evidence.state_id != original.state.id
                or original.evidence.meter_version != engine.meter_version
                or original.evidence.fake != start.fake
            ):
                continue
            original_path = Path(original.state.path).resolve(strict=True)
            if not original_path.is_relative_to(source / "states"):
                raise ValueError("old cached output escapes source state store")
            copied = engine.store.import_state(
                original_path,
                design=design,
                parent_id=start.id,
                made_by_run_id=origin_id,
                stage=original.state.stage,
                fake=start.fake,
            )
            if copied.id != original.state.id:
                raise ValueError("copied output checkpoint hash changed")
            transplanted = RunOutcome(
                run=original.run,
                state=copied,
                evidence=original.evidence,
                claims=original.claims,
            )
            engine.store.save_outcome(transplanted)
            with engine.cache.locked(key):
                existing = engine.cache.get(key)
                if existing not in (None, origin_id):
                    raise ValueError("destination cache contains another run")
                if existing is None:
                    engine.cache.put(key, origin_id)
            rows.append({
                "design": design,
                "recipe_id": recipe.id,
                "recipe_version": recipe.version,
                "cache_key": key,
                "source_run_id": origin_id,
                "source_complete_sha256": _sha256(
                    source / "runs" / origin_id / "complete.json"
                ),
                "output_state_id": copied.id,
            })
    return {
        "source_root": str(source),
        "destination_root": str(engine.store.root),
        "meter_version": engine.meter_version,
        "seeded_start_designs": starts_seeded,
        "seeded_run_count": len(rows),
        "seeded_runs": rows,
    }

