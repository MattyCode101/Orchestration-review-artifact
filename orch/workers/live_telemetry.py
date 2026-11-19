"""Live input-state telemetry for fitted workers on any design (Saturday night, item 2).

FittedWorker scores its recipes from telemetry of the INPUT state of each call: the start state for a
call at depth zero, the produced state for a later call. For the training designs that telemetry was
stored beforehand (orch/workers/telemetry.py: a pinned v3 extraction plus, for later states, the
utilization report of the producing run). Held-out designs have none. This provider computes the same
telemetry for any real state by running the very extraction that made the training records
(DcpTelemetryExtractor: the copied LATCH probe and worst-path Tcl, no Tcl written here, Vivado 2025.1)
and returns it in the shape FittedWorker already reads.

Rules, fail closed:
  * a state whose extraction fails, or whose record lacks a feature worker B ranks on, is REFUSED:
    TelemetryUnavailableError is raised before anything is journalled, the refusal is logged with its
    reason, and the worker is never scored from the recipe means in its place;
  * features are cached by state hash (the record and its cost), so a state is extracted once for
    every approach and every repeat; the cost of an extraction is stored apart from any recipe cost;
  * a persisted counter caps the number of extraction jobs (each job is one DcpTelemetryExtractor
    call: three Vivado launches), and a cached state never needs a job;
  * telemetry_source holds only content-addressed fields, so a journalled plan can be rebuilt on
    resume whether the record was just extracted or read from the cache.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orch.analysis.fit_workers import FROZEN_SHA256, extract_features, load_feature_spec
from orch.contract.models import DesignState, Evidence
from orch.engine.store import _atomic_bytes, _file_lock, _json_bytes

SCHEMA = "live-dcp-telemetry-v1"
PROJECT = Path(__file__).resolve().parents[2]
_RUN = re.compile(r"run_[0-9a-f]{32}\Z")
_LOG_LOCK = threading.Lock()


class TelemetryUnavailableError(RuntimeError):
    """Telemetry for this state cannot be produced or is incomplete: the worker refuses the call."""


class ExtractionCapReached(TelemetryUnavailableError):
    """The extraction-job cap is used up; no further state is extracted."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExtractionBudget:
    """A persisted cap on extraction jobs, shared by every process through a file lock."""

    def __init__(self, path: Path | str, cap: int) -> None:
        self.path, self.cap = Path(path), int(cap)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {"reservations": []}

    def reserve(self, state_id: str) -> None:
        with _file_lock(self.path.with_suffix(".lock")):
            value = self._load()
            if len(value["reservations"]) >= self.cap:
                raise ExtractionCapReached(f"{self.cap} extraction jobs are already used")
            value["reservations"].append({"state_sha256": state_id, "at_utc": _utc()})
            value["cap"] = self.cap
            _atomic_bytes(self.path, _json_bytes(value))

    def count(self) -> int:
        return len(self._load()["reservations"])


def default_required_recipes(project: Path = PROJECT) -> tuple[str, ...]:
    """The recipes worker B ranks on its fitted score: those whose FINAL fit row has feature_available."""

    spec = load_feature_spec()
    model = json.loads((project / "configs/workers/B_fit.json").read_text(encoding="utf-8"))
    return tuple(sorted(
        recipe_id for recipe_id, row in model["recipes"].items()
        if row["feature_available"] and spec["recipes"][recipe_id]["worker"] == "area"
    ))


class LiveTelemetryProvider:
    """Callable with the signature FittedWorker's episode machinery uses for a telemetry provider."""

    def __init__(
        self, cache_root: Path | str, *, extractor: Any = None, runner: Any = None,
        budget: ExtractionBudget | None = None, log_path: Path | str | None = None, store: Any = None,
        required_recipes: Sequence[str] | None = None, env_script: Path | None = None,
        tool_version: str = "2025.1", timeout_s: float = 1800, project: Path = PROJECT,
    ) -> None:
        self.root = Path(cache_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.budget, self.store = budget, store
        self.log_path = Path(log_path) if log_path is not None else None
        self.spec = load_feature_spec()
        self.required_recipes = tuple(required_recipes) if required_recipes else default_required_recipes(project)
        if extractor is None:
            from orch.engine.dcp_parser import DcpTelemetryExtractor
            from orch.engine.runner import Runner

            runner = runner or Runner(project / "data/slots", max_slots=48)
            extractor = DcpTelemetryExtractor(
                runner=runner, env_script=env_script or project / "recipes/support/env.sh",
                tool_version=tool_version, meter=None, timeout_s=timeout_s,
            )
        self.extractor = extractor

    # ------------------------------------------------------------------ identity and layout
    def identity(self) -> dict[str, Any]:
        """Binds a journal's resume to this extraction code, tool version and required-feature set."""

        value: dict[str, Any] = {
            "kind": SCHEMA, "extractor_sha256": self.extractor.extractor_sha256,
            "env_script_sha256": self.extractor.env_script_sha256, "tool_version": self.extractor.tool_version,
            "feature_spec_sha256": FROZEN_SHA256, "required_recipes": sorted(self.required_recipes),
            "provider_sha256": _sha(Path(__file__)),
        }
        try:
            from orch.engine import dcp_parser

            value.update({
                "probe_template_sha256": dcp_parser.PROBE_TEMPLATE_SHA256,
                "wpath_sha256": dcp_parser.WPATH_SHA256, "wpath2_sha256": dcp_parser.WPATH2_SHA256,
            })
        except (ImportError, AttributeError):
            pass
        return value

    def _dir(self, state_id: str) -> Path:
        return self.root / state_id[:2] / state_id

    def _log(self, event: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        line = json.dumps({"at_utc": _utc(), **event}, sort_keys=True) + "\n"
        with _LOG_LOCK:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with _file_lock(self.log_path.with_suffix(".lock")), open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(line)

    def _refuse(self, state_id: str, reason: str, *, cap: bool = False) -> TelemetryUnavailableError:
        self._log({"event": "refused", "state_sha256": state_id, "reason": reason})
        return (ExtractionCapReached if cap else TelemetryUnavailableError)(f"telemetry refused for {state_id[:12]}: {reason}")

    # ------------------------------------------------------------------ accounting
    def cost_of(self, state_id: str) -> dict[str, Any] | None:
        path = self._dir(state_id) / "extraction.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    # ------------------------------------------------------------------ the provider call
    def __call__(
        self, state: DesignState, evidence: Evidence, producer_recipe_id: str | None, run_id: str | None,
    ) -> dict[str, object]:
        self._check_inputs(state, evidence, producer_recipe_id, run_id)
        directory = self._dir(state.id)
        directory.mkdir(parents=True, exist_ok=True)
        with _file_lock(directory / "state.lock"):
            path = directory / "features.json"
            if path.is_file():
                raw = path.read_bytes()
            else:
                raw = self._extract(state, evidence, directory)
            record = json.loads(raw)
            self._validate(record, state)
        return {
            "telemetry": record, "produced_by_recipe_id": producer_recipe_id,
            "telemetry_source": {
                "kind": SCHEMA, "state_sha256": state.id, "record_sha256": hashlib.sha256(raw).hexdigest(),
                "extractor_sha256": self.extractor.extractor_sha256,
            },
        }

    def _check_inputs(self, state: DesignState, evidence: Evidence, producer: str | None, run_id: str | None) -> None:
        if state.fake or evidence.fake:
            raise self._refuse(state.id, "live telemetry needs a real state and real meter Evidence")
        if evidence.state_id != state.id:
            raise self._refuse(state.id, "Evidence names a different state")
        if state.parent_id is None:
            if producer is not None or run_id is not None:
                raise self._refuse(state.id, "a start state cannot have a producer")
        else:
            if not producer or not run_id or not _RUN.fullmatch(run_id):
                raise self._refuse(state.id, "a derived state needs its producing recipe and run (producer missing)")
            if self.store is not None:
                produced = self.store.load_outcome(run_id)
                if (
                    produced.state != state or produced.evidence != evidence
                    or produced.run.recipe_id != producer or produced.run.input_sha256_before != state.parent_id
                ):
                    raise self._refuse(state.id, "producer identity differs from the persisted run outcome")
        if not Path(state.path).is_file():
            raise self._refuse(state.id, "the state's checkpoint file is missing")

    def _extract(self, state: DesignState, evidence: Evidence, directory: Path) -> bytes:
        if self.budget is not None:
            try:
                self.budget.reserve(state.id)
            except ExtractionCapReached as exc:
                raise self._refuse(state.id, str(exc), cap=True) from exc
        attempt = directory / "attempts" / uuid.uuid4().hex
        started_at, started = _utc(), time.perf_counter()
        try:
            payload = self.extractor.extract(
                state.path, run_dir=attempt, evidence=evidence, design=state.design,
                run_id=f"live-telemetry-{state.id[:12]}",
            )
        except Exception as exc:  # noqa: BLE001 - any failure refuses the call; none falls back to the means
            raise self._refuse(state.id, f"extraction failed: {type(exc).__name__}: {str(exc)[:300]}") from exc
        elapsed = time.perf_counter() - started
        raw = _json_bytes(payload)
        runners = [payload["provenance"]["runner"], payload["provenance"]["wpath"]["runner"],
                   payload["provenance"]["wpath2"]["runner"]]
        cost = {
            "schema_version": "live-extraction-cost-v1", "state_sha256": state.id, "design": state.design,
            "started_at_utc": started_at, "finished_at_utc": _utc(), "elapsed_s": elapsed,
            "vivado_launches": len(runners), "runner_wall_s": [float(r["wall_s"]) for r in runners],
            "wall_s_total": sum(float(r["wall_s"]) for r in runners),
            "peak_memory_bytes": max(int(r["peak_memory_bytes"]) for r in runners),
            "extractor_sha256": self.extractor.extractor_sha256, "record_sha256": hashlib.sha256(raw).hexdigest(),
        }
        _atomic_bytes(directory / "features.json", raw)
        _atomic_bytes(directory / "extraction.json", _json_bytes(cost))
        self._log({"event": "extracted", "state_sha256": state.id, "design": state.design, "elapsed_s": elapsed,
                   "wall_s_total": cost["wall_s_total"], "peak_memory_bytes": cost["peak_memory_bytes"]})
        return raw

    def _validate(self, record: Mapping[str, Any], state: DesignState) -> None:
        if record.get("state_sha256") != state.id or record.get("fake") is not False:
            raise self._refuse(state.id, "the stored record does not describe this real state")
        if record.get("provenance", {}).get("extractor_sha256") != self.extractor.extractor_sha256:
            raise self._refuse(state.id, "the stored record was made by a different extractor version")
        values = extract_features(record, self.spec)
        missing = sorted(recipe for recipe in self.required_recipes if values.get(recipe) is None)
        if missing:
            raise self._refuse(state.id, "the record lacks features worker B ranks on: " + ", ".join(missing))


def total_extraction_cost(cache_root: Path | str, state_ids: Sequence[str]) -> dict[str, Any]:
    """Sum the stored extraction costs of these states: what an approach that called worker B on them paid."""

    root = Path(cache_root)
    costs = []
    for state_id in dict.fromkeys(state_ids):
        path = root / state_id[:2] / state_id / "extraction.json"
        if path.is_file():
            costs.append(json.loads(path.read_text(encoding="utf-8")))
    return {
        "states_requested": len(set(state_ids)), "states_with_cost": len(costs),
        "wall_s_total": sum(c["wall_s_total"] for c in costs), "elapsed_s_total": sum(c["elapsed_s"] for c in costs),
        "peak_memory_bytes_max": max((c["peak_memory_bytes"] for c in costs), default=0),
    }
