"""Read-only cache lookup through a manifest over existing run roots.

The test campaign must reuse outcomes stored in 11 earlier roots without copying
or linking any record. A CacheManifest maps each step key to the existing run
record (root, run id, completion-marker hash, original cost fields, meter version,
approval). ManifestRunCache and ManifestStateStore subclass the pinned engine
classes: the campaign's own root wins, then the manifest answers, and nothing in a
source root is ever written. A manifest hit is charged exactly like any cache hit,
the original cost of the stored run, and starts no tool.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from orch.contract.models import (
    Claims,
    DesignState,
    Evidence,
    RunOutcome,
    RunRecord,
)
from orch.engine.apply import StepEngine
from orch.engine.cache import RunCache
from orch.engine.store import (
    StateStore,
    _HEX64,
    _atomic_bytes,
    _json_bytes,
    _safe_name,
    _sha256_file,
)

SCHEMA = "test-cache-manifest-v1"
COST_FIELDS = (
    "wall_s", "cpu_user_s", "cpu_sys_s", "peak_memory_bytes", "load_1m_start",
    "load_1m_end", "busy_slots_start", "backpressure_wait_s",
    "backpressure_events",
)
_ARTIFACTS = {"runinfo.json", "output_state.json", "evidence.json", "claims.json"}


def manifest_hash(body: Mapping[str, Any]) -> str:
    """Hash the canonical JSON of the whole manifest body."""

    return hashlib.sha256(_json_bytes(body)).hexdigest()


class CacheManifest:
    """A verified, immutable view of a manifest body."""

    def __init__(self, body: Mapping[str, Any], *, expected_hash: str | None = None) -> None:
        if body.get("schema_version") != SCHEMA:
            raise ValueError("unknown cache manifest schema")
        self.body = dict(body)
        self.hash = manifest_hash(self.body)
        if expected_hash is not None and expected_hash != self.hash:
            raise ValueError("cache manifest hash differs from the expected hash")
        self.roots = {name: Path(path) for name, path in self.body["roots"].items()}
        self.entries: dict[str, dict[str, Any]] = self.body["entries"]
        self._run_to_key: dict[str, str] = {}
        for key, entry in self.entries.items():
            if entry["root"] not in self.roots:
                raise ValueError("manifest entry names an unknown root")
            if entry["run_id"] in self._run_to_key:
                raise ValueError("manifest run id is not unique")
            self._run_to_key[entry["run_id"]] = key

    @classmethod
    def load(cls, path: Path | str, *, expected_hash: str | None = None) -> "CacheManifest":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), expected_hash=expected_hash)

    def entry(self, key: str) -> dict[str, Any] | None:
        return self.entries.get(key)

    def run_dir(self, run_id: str) -> Path | None:
        key = self._run_to_key.get(run_id)
        if key is None:
            return None
        return self.roots[self.entries[key]["root"]] / "runs" / run_id

    def complete_sha256(self, run_id: str) -> str:
        return str(self.entries[self._run_to_key[run_id]]["complete_sha256"])

    def state_locations(self, state_id: str) -> tuple[Path, ...]:
        return tuple(Path(p) for p in self.body["states"].get(state_id, ()))

    def start_evidence(self, state_id: str, meter_version: str) -> Path | None:
        item = self.body.get("start_evidence", {}).get(state_id)
        if item and item["meter_version"] == meter_version:
            return Path(item["path"])
        return None


def _read_run(run_dir: Path, run_id: str) -> tuple[RunRecord, DesignState | None, Evidence | None, Claims | None]:
    """The checksum-verified read of StateStore.load_outcome, without the path rule."""

    complete = run_dir / "complete.json"
    if not complete.is_file():
        raise FileNotFoundError(f"incomplete or missing run: {run_id}")
    marker = json.loads(complete.read_text(encoding="utf-8"))
    if marker.get("run_id") != run_id or not isinstance(marker.get("files"), dict):
        raise ValueError(f"invalid completion marker: {complete}")
    raw: dict[str, bytes] = {}
    for name, checksum in marker["files"].items():
        if name not in _ARTIFACTS:
            raise ValueError(f"unexpected run artifact: {name}")
        content = (run_dir / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != checksum:
            raise ValueError(f"run artifact hash mismatch: {name}")
        raw[name] = content
    if "runinfo.json" not in raw:
        raise ValueError("completion marker lacks runinfo.json")
    run = RunRecord.model_validate_json(raw["runinfo.json"])
    if run.id != run_id:
        raise ValueError("runinfo id differs from completion marker")
    state = DesignState.model_validate_json(raw["output_state.json"]) if "output_state.json" in raw else None
    evidence = Evidence.model_validate_json(raw["evidence.json"]) if "evidence.json" in raw else None
    claims = Claims.model_validate_json(raw["claims.json"]) if "claims.json" in raw else None
    return run, state, evidence, claims


class ManifestRunCache(RunCache):
    """Own-root pointers first, then the manifest; never writes a source root."""

    def __init__(self, root: Path | str, manifest: CacheManifest, *, meter_version: str) -> None:
        super().__init__(root)
        self.manifest = manifest
        self.meter_version = meter_version

    def get(self, key: str) -> str | None:
        own = self._entry(key)
        if own is not None and not own["retryable"]:
            return super().get(key)
        entry = self.manifest.entry(self._check_key(key))
        if entry is None or entry["meter_version"] not in (None, self.meter_version):
            return None
        return str(entry["run_id"])


class ManifestStateStore(StateStore):
    """Resolve checkpoints and run records from this root, then the manifest.

    A state keeps one deterministic location: this root if it holds the bytes,
    otherwise the first manifest location whose content hash matches. Outcomes
    loaded from a source root are returned with their state path normalised to
    that location, so the engine's input-path check holds.
    """

    def __init__(self, root: Path | str, manifest: CacheManifest) -> None:
        super().__init__(root)
        self.manifest = manifest

    def _verified(self, path: Path, state_id: str) -> Path:
        if not path.is_file() or not (path.parent / "state.json").is_file():
            raise FileNotFoundError(f"incomplete or missing state: {state_id}")
        if path.is_symlink():
            raise ValueError(f"canonical state is a symlink: {path}")
        metadata = DesignState.model_validate_json((path.parent / "state.json").read_bytes())
        if metadata.id != state_id:
            raise ValueError(f"state metadata id mismatch: {path}")
        if _sha256_file(path) != state_id:
            raise ValueError(f"state content hash mismatch: {path}")
        return path

    def state_path(self, state_id: str) -> Path:
        if not _HEX64.fullmatch(state_id):
            raise ValueError(f"invalid state id: {state_id!r}")
        # One deterministic location for the whole campaign: the manifest's first (a byte-identical
        # output imported later into this root must not move a state under a running episode).
        for location in self.manifest.state_locations(state_id):
            if location.is_file():
                return self._verified(location, state_id)
        if (self.states_dir / state_id / "design.dcp").exists():
            return super().state_path(state_id)
        raise FileNotFoundError(f"incomplete or missing state: {state_id}")

    def _normalised(self, state: DesignState | None) -> DesignState | None:
        if state is None:
            return None
        canonical = str(self.state_path(state.id))
        if state.path == canonical:
            return state
        return state.model_copy(update={"path": canonical})

    def save_outcome(self, outcome: RunOutcome) -> None:
        state = self._normalised(outcome.state)
        if state is not outcome.state:
            outcome = outcome.model_copy(update={"state": state})
        super().save_outcome(outcome)

    def load_outcome(self, run_id: str) -> RunOutcome:
        run_id = _safe_name(run_id, what="run id")
        own = self.runs_dir / run_id
        if (own / "complete.json").is_file():
            run_dir = own
        else:
            run_dir = self.manifest.run_dir(run_id)
            if run_dir is None:
                raise FileNotFoundError(f"incomplete or missing run: {run_id}")
            if _sha256_file(run_dir / "complete.json") != self.manifest.complete_sha256(run_id):
                raise ValueError(f"stored run changed since the manifest was built: {run_id}")
        run, state, evidence, claims = _read_run(run_dir, run_id)
        return RunOutcome(run=run, state=self._normalised(state), evidence=evidence, claims=claims)


class ManifestStepEngine(StepEngine):
    """StepEngine whose start measurement can also come from the manifest.

    With require_zero_tool, a start with neither own nor manifest evidence raises
    instead of starting the meter.
    """

    def __init__(self, *, manifest: CacheManifest, require_zero_tool: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.manifest = manifest
        self.require_zero_tool = require_zero_tool

    def measure_start(self, state: DesignState) -> Evidence:
        version_key = hashlib.sha256(self.meter_version.encode("utf-8")).hexdigest()
        own = self.store.root / "measurements" / state.id / version_key / "evidence.json"
        if not own.is_file():
            path = self.manifest.start_evidence(state.id, self.meter_version)
            if path is not None and path.is_file():
                evidence = Evidence.model_validate_json(path.read_bytes())
                if (
                    evidence.state_id == state.id
                    and evidence.meter_version == self.meter_version
                    and evidence.fake == state.fake
                ):
                    self._check_input_state(state)
                    # An episode resumes against its own root's persisted start measurement.
                    _atomic_bytes(own, _json_bytes(evidence.model_dump(mode="json")))
                    return evidence
                raise ValueError("manifest start evidence does not match state or meter")
            if self.require_zero_tool:
                raise RuntimeError("start measurement would start a tool; none is allowed")
        return super().measure_start(state)
