"""One shared step operation for real and labelled fake backends.

Only the run and its independent measurement are cached. The gate is called
again for every brief and guard set, including on a cache hit.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Mapping, Sequence

from orch.contract.models import (
    AppliedStep,
    Brief,
    Claims,
    DesignState,
    Evidence,
    Guard,
    MetricName,
    RunOutcome,
    RunStatus,
)
from orch.engine.cache import RunCache
from orch.engine.gate import judge
from orch.engine.store import StateStore, _atomic_bytes, _file_lock, _json_bytes
from orch.interfaces import Backend, RecipeSpec


_RETRYABLE = {RunStatus.TIMEOUT, RunStatus.TRANSIENT, RunStatus.METER_ERROR}


class StepEngine:
    """Run, measure, cache, judge and record one recipe without choosing it."""

    def __init__(
        self,
        *,
        store: StateStore,
        cache: RunCache,
        backend: Backend,
        meter_version: str,
        cache_enabled: bool = False,
    ) -> None:
        self.store = store
        self.cache = cache
        self.backend = backend
        self.meter_version = meter_version
        self.cache_enabled = cache_enabled

    def _check_input_state(self, state: DesignState) -> None:
        """Validate both the registered bytes and the path passed to a backend."""

        canonical = self.store.state_path(state.id)
        if Path(state.path).resolve(strict=True) != canonical:
            raise ValueError("input state path differs from registered checkpoint")

    def measure_start(self, state: DesignState) -> Evidence:
        """Measure a start state through the backend, keyed by state and meter.

        An episode never asks its policy to measure. The evidence file is
        published atomically, and raw meter jobs stay in an attempt directory.
        """

        self._check_input_state(state)
        version_key = hashlib.sha256(self.meter_version.encode("utf-8")).hexdigest()
        folder = self.store.root / "measurements" / state.id / version_key
        folder.mkdir(parents=True, exist_ok=True)
        evidence_path = folder / "evidence.json"
        with _file_lock(folder / "evidence.lock"):
            if self.cache_enabled and evidence_path.is_file():
                evidence = Evidence.model_validate_json(evidence_path.read_bytes())
                if (
                    evidence.state_id == state.id
                    and evidence.meter_version == self.meter_version
                    and evidence.fake == state.fake
                ):
                    return evidence
                raise ValueError("cached start evidence does not match state or meter")
            run_dir = folder / "attempts" / uuid.uuid4().hex
            run_dir.mkdir(parents=True)
            evidence = self.backend.measure_start(state, run_dir=run_dir)
            if (
                evidence.state_id != state.id
                or evidence.meter_version != self.meter_version
                or evidence.fake != state.fake
            ):
                raise ValueError("start meter returned evidence for a different state or version")
            _atomic_bytes(
                evidence_path, _json_bytes(evidence.model_dump(mode="json"))
            )
            return evidence

    def _execute(self, state: DesignState, recipe: RecipeSpec) -> RunOutcome:
        """Call the backend and persist a complete outcome before cache publish."""

        run_id = f"run_{uuid.uuid4().hex}"
        run_dir = self.store.root / "runs" / run_id
        run_dir.mkdir(parents=True)
        outcome = self.backend.execute(
            state, recipe, run_id=run_id, run_dir=run_dir
        )
        if outcome.run.id != run_id:
            raise ValueError("backend returned the wrong run id")
        if (
            outcome.run.recipe_id != recipe.id
            or outcome.run.recipe_version != recipe.version
        ):
            raise ValueError("backend returned the wrong recipe identity")
        if (
            outcome.run.input_sha256_before is not None
            and outcome.run.input_sha256_before != state.id
        ):
            raise ValueError("backend ran on the wrong input checkpoint")
        if (
            outcome.run.status != RunStatus.INPUT_MUTATED
            and outcome.run.input_sha256_after is not None
            and outcome.run.input_sha256_after != state.id
        ):
            raise ValueError("input checkpoint changed during execution")
        if outcome.state is not None and outcome.state.parent_id != state.id:
            raise ValueError("output state does not cite its actual parent")
        if outcome.evidence is not None and outcome.evidence.meter_version != self.meter_version:
            raise ValueError("backend returned evidence from the wrong meter version")
        self.store.save_outcome(outcome)
        return outcome

    def _cached_call(self, original: RunOutcome) -> RunOutcome:
        """Record a cache hit as its own run, charging the original wall time."""

        run_id = f"run_{uuid.uuid4().hex}"
        run = original.run.model_copy(
            update={
                "id": run_id,
                "cache_hit": True,
                "original_run_id": original.run.id,
                "attempts": (),
                "process_group_kills": 0,
                "tag_sweep_kills": 0,
            }
        )
        claims = original.claims
        if claims is not None:
            claims = claims.model_copy(
                update={"id": f"{run_id}:claims", "run_id": run_id}
            )
        outcome = RunOutcome(
            run=run,
            state=original.state,
            evidence=original.evidence,
            claims=claims,
        )
        self.store.save_outcome(outcome)
        return outcome

    def execute(self, state: DesignState, recipe: RecipeSpec) -> RunOutcome:
        """Return a measured outcome, using a terminal cache hit when safe."""

        self._check_input_state(state)
        if not self.cache_enabled:
            return self._execute(state, recipe)
        key = self.cache.key(state.id, recipe.id, recipe.version)
        with self.cache.locked(key):
            origin_id = self.cache.get(key)
            if origin_id is not None:
                original = self.store.load_outcome(origin_id)
                if (
                    original.run.recipe_id != recipe.id
                    or original.run.recipe_version != recipe.version
                    or original.run.fake != state.fake
                    or (
                        original.state is not None
                        and original.state.parent_id != state.id
                    )
                ):
                    raise ValueError("cached outcome identity differs from cache key")
                if (
                    original.run.input_sha256_before != state.id
                    or original.run.input_sha256_after != state.id
                    or original.run.status == RunStatus.INPUT_MUTATED
                ):
                    raise ValueError("cached outcome input hash or mutation differs")
                if (
                    original.evidence is None
                    or original.evidence.meter_version == self.meter_version
                ):
                    return self._cached_call(original)
                # The recipe key is unchanged, but evidence from an older
                # meter cannot be trusted. Re-execute and replace the pointer.
            outcome = self._execute(state, recipe)
            if outcome.run.status != RunStatus.INPUT_MUTATED:
                self.cache.put(
                    key,
                    outcome.run.id,
                    retryable=outcome.run.status in _RETRYABLE,
                )
            return outcome

    def apply(
        self,
        state: DesignState,
        recipe: RecipeSpec,
        *,
        brief: Brief,
        start: Evidence,
        parent: Evidence,
        guards: Sequence[Guard] = (),
        noise: Mapping[MetricName, float] | None = None,
    ) -> AppliedStep:
        """Execute a recipe once, then judge it against this fixed episode brief."""

        if parent.state_id != state.id:
            raise ValueError("parent evidence is not for the input state")
        outcome = self.execute(state, recipe)
        verdict = judge(
            brief=brief,
            start=start,
            parent=parent,
            candidate=outcome.evidence,
            guards=guards,
            noise=noise,
            run_status=outcome.run.status,
        )
        # A run may be judged for multiple briefs or guards. Store each
        # judgment separately without changing the cached run outcome.
        file_key = hashlib.sha256(verdict.id.encode("utf-8")).hexdigest()
        verdict_path = (
            self.store.root / "runs" / outcome.run.id / "verdicts" / f"{file_key}.json"
        )
        wrapper = {
            "brief_hash": brief.fingerprint(),
            "start_evidence_id": start.id,
            "parent_evidence_id": parent.id,
            "guards": [guard.model_dump(mode="json") for guard in guards],
            "verdict": verdict.model_dump(mode="json"),
        }
        _atomic_bytes(verdict_path, _json_bytes(wrapper))
        return AppliedStep(outcome=outcome, verdict=verdict)

