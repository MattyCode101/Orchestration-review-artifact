"""Campaign harness: one code path for the training replay, the training rehearsal and the test
campaign. Only the scope (design list, manifest, noise source), the cache root and the episode list
differ between them.

What it does
  * Reads stored outcomes through a hash-pinned read-only manifest (orch/engine/manifest_cache.py);
    new outputs go to its own overlay root. No source root is ever written.
  * Imports the LLM orchestrator, the general agent, the profiles and the bridge from orch/policies,
    orch/baselines and tools/llm_bridge. Nothing is copied. The policy's own repair and fallback
    rules are used unchanged; the harness adds no second layer.
  * Four caps fail closed: new real runs, job-hours, tokens, US dollars. Every model request and reply
    is stored with its hash (the replay cache record holds the prompt and the reply) and logged.
  * Cells are picked by name from configs/campaign/campaign.yaml, which names every configuration and
    its files; hashes of every imported project file are taken at the start and the end of a run.
No Vivado or RapidWright command is written here: tool runs go through the unchanged engine backend.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT))

import thursday_night_pilot as pilot  # noqa: E402
import saturday_llm as dev_llm  # noqa: E402  (imported: noise_for for training, no campaign state is used)

from orch.baselines.direct_recipe_worker import DirectRecipeWorker  # noqa: E402
from orch.baselines.general_agent import build_general_agent  # noqa: E402
from orch.config import load_recipe_sets, load_worker_card  # noqa: E402
from orch.contract.brief_text import load_frozen_brief  # noqa: E402
from orch.contract.models import (  # noqa: E402
    AppliedStep, DesignState, EquivalenceCheck, Evidence, MetricName, RunOutcome, RunStatus,
)
from orch.engine.apply import StepEngine  # noqa: E402
from orch.engine.journal import Journal  # noqa: E402
from orch.engine.manifest_cache import (  # noqa: E402
    COST_FIELDS, CacheManifest, ManifestRunCache, ManifestStateStore, ManifestStepEngine,
)
from orch.engine.meter import VivadoMeter  # noqa: E402
from orch.engine.real_backend import RealBackend  # noqa: E402
from orch.engine.read_through_ledger import ReadThroughLedger  # noqa: E402
from orch.engine.runner import Runner  # noqa: E402
from orch.engine.store import _atomic_bytes, _file_lock, _json_bytes, _sha256_file  # noqa: E402
from orch.episode import FinalValidation, run_episode  # noqa: E402
from orch.policies.llm import LLMPolicy, LLMPolicyConfig  # noqa: E402
from orch.policies.llm_client import (  # noqa: E402
    BridgeTransport, CachedLLM, LLMCallError, LLMParams, LLMResult, LLMUnavailableError, RateLimiter, ReplayCache,
    TelemetrySink,
)
from orch.policies.llm_prompt import load_template  # noqa: E402
from orch.policies.relevant_workers import FixedOrderPolicy, RelevantRandomPolicy  # noqa: E402
from orch.policies.repair_loop import RepairLoopPolicy  # noqa: E402
from orch.policies.training_baselines import (  # noqa: E402
    FixedAlternationPolicy, SingleSpecialistPolicy, UniformRandomPolicy,
)
from orch.recipes.adapters.latch import LatchAdapter  # noqa: E402
from orch.recipes.adapters.tcl import TclAdapter  # noqa: E402
from orch.recipes.adapters.validator import ValidatorAdapter  # noqa: E402
from orch.recipes.registry import load_registry  # noqa: E402
from orch.runtime import LATCH_VENV_STATUS, _require_deterministic_smoke, _require_latch_venv_smoke  # noqa: E402
from orch.workers.base import GenericWorker  # noqa: E402
from orch.workers.fitted import FittedWorker  # noqa: E402
from orch.workers.live_telemetry import (  # noqa: E402
    ExtractionBudget, ExtractionCapReached, LiveTelemetryProvider, TelemetryUnavailableError,
)
from orch.workers.telemetry import SavedTelemetryProvider  # noqa: E402

CAMPAIGN_YAML = PROJECT / "configs/campaign/campaign.yaml"
RANKING = PROJECT / "data/scratch_friday_2026-11-06/worker_c_ranking.json"
DEV_SHARED = PROJECT / "data/pilots/saturday_llm_shared"
V4 = PROJECT / "data/pilots/thursday_overnight_train_v4_2026-11-05"
RUN_ID = re.compile(r"run_[0-9a-f]{32}")
STEP_ID = re.compile(r"step_[0-9a-f]{64}")


def infrastructure_kind(exc: BaseException) -> str | None:
    """'extraction' or 'api' if this error (or its cause) is an infrastructure failure, else None.

    Decision (d), 2026-11-08: a failed worker B telemetry extraction or a persistent model-API failure ends
    the episode as an infrastructure failure. It is never an orchestration decision: no STOP, no schema
    repair, no fallback, no charged run. An exhausted extraction-job cap is a cap, not a failure.
    """

    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ExtractionCapReached):
            return None
        if isinstance(exc, TelemetryUnavailableError):
            return "extraction"
        if isinstance(exc, (LLMUnavailableError, LLMCallError)):
            return "api"
        exc = exc.__cause__ or exc.__context__
    return None


class CapReached(RuntimeError):
    """A cap would be exceeded; the campaign stops starting new work of that kind."""


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def settings() -> dict[str, Any]:
    return yaml.safe_load(CAMPAIGN_YAML.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------ scope
@dataclass
class Scope:
    name: str                     # train | test
    designs: tuple[str, ...]
    manifest: CacheManifest
    work_root: Path
    starts: dict[str, str]        # design -> start state id (B2)
    noise: Callable[[str, Evidence], dict[MetricName, float]]
    provider_for: Callable[[ManifestStateStore], Any] | None

    @property
    def engine_root(self) -> Path:
        return self.work_root / "engine"


def _starts_from_manifest(manifest: CacheManifest) -> dict[str, str]:
    return {item["design"]: state for state, item in manifest.body["start_evidence"].items()}


def train_provider(store: ManifestStateStore) -> "ScopeTelemetryProvider":
    return ScopeTelemetryProvider(V4, store, [V4])


def train_scope(work_root: Path, manifest_path: Path, expected_sha: str | None = None) -> Scope:
    manifest = CacheManifest.load(manifest_path, expected_hash=expected_sha)
    starts = _starts_from_manifest(manifest)
    if set(starts) != set(pilot.DESIGNS):
        raise ValueError("training manifest does not hold the five training starts")
    return Scope("train", tuple(pilot.DESIGNS), manifest, work_root, starts, dev_llm.noise_for, train_provider)


def test_noise(design: str, start: Evidence) -> dict[MetricName, float]:
    bands = json.loads((PROJECT / settings()["test_noise_file"]).read_text(encoding="utf-8"))["bands"][design]
    power = json.loads((PROJECT / settings()["test_power_noise_file"]).read_text(encoding="utf-8"))["bands"][design]
    return {
        MetricName.WNS_NS: float(bands["wns_ns"]), MetricName.PACKED_LUTS: float(bands["packed_luts"]),
        MetricName.DYNAMIC_POWER_W: float(power["dynamic_power_w_band"]),
    }


def test_scope(work_root: Path, manifest_path: Path, expected_sha: str | None = None,
               provider_for: Callable[[ManifestStateStore], Any] | None = None) -> Scope:
    manifest = CacheManifest.load(manifest_path, expected_hash=expected_sha)
    starts = _starts_from_manifest(manifest)
    designs = tuple(settings()["test_designs"])
    if set(starts) != set(designs):
        raise ValueError("test manifest does not hold the five test starts")
    return Scope("test", designs, manifest, work_root, starts, test_noise, provider_for)


# ------------------------------------------------------------------------------ telemetry shim
class ScopeTelemetryProvider(SavedTelemetryProvider):
    """The saved training telemetry, reading through an overlay store.

    The pinned provider looks producer runs and reports up in one data root. With an overlay root,
    producer runs (including cache-hit copies) live in the overlay and the reports in the stored root,
    so the lookup goes through the overlay store and searches the stored roots for the report. The
    features it returns are the pinned provider's own. This is a stand-in until the development
    agent's live provider is READY.
    """

    def __init__(self, data_root: Path, store: ManifestStateStore, report_roots: list[Path]) -> None:
        super().__init__(data_root)
        self.store = store
        self.report_roots = [Path(r).resolve() for r in report_roots]

    def _report(self, state, evidence, run_id):
        candidates: list[Path] = []
        if run_id is not None:
            outcome = self.store.load_outcome(run_id)
            if outcome.state is None or outcome.state.id != state.id:
                raise ValueError("producer outcome differs from input state")
            ids = [run_id] + ([outcome.run.original_run_id] if outcome.run.original_run_id else [])
            for item in ids:
                for root in [self.store.root, *self.report_roots]:
                    candidates.append(Path(root) / "runs" / item / "reports" / "final_utilization.rpt")
        if evidence.raw_report_path:
            candidates.append(Path(evidence.raw_report_path) / "final_utilization.rpt")
        allowed = [self.store.root.resolve(), *self.report_roots]
        for path in candidates:
            if path.is_file():
                resolved = path.resolve(strict=True)
                if any(resolved.is_relative_to(r) for r in allowed):
                    return resolved
        return None

    def __call__(self, state, evidence, producer_recipe_id, run_id):
        original_root = self.root
        # The pinned __call__ reports the report path relative to self.root; stored reports are in
        # self.root (the stored training root), overlay reports are not, so give it the shared parent.
        try:
            return super().__call__(state, evidence, producer_recipe_id, run_id)
        except ValueError as exc:
            if "is not in the subpath" not in str(exc):
                raise
            self.root = Path(os.path.commonpath([str(original_root), str(self.store.root)]))
            try:
                return super().__call__(state, evidence, producer_recipe_id, run_id)
            finally:
                self.root = original_root


# ------------------------------------------------------------------------------ live worker B telemetry
class AccountedTelemetry:
    """The development run's live provider, with the campaign's accounting around it.

    Extraction wall time is added to the job-hour cap once per state (it is a tool run, reported apart from
    recipe runs); every use is logged per episode. Features are cached by state hash by the inner provider,
    so a state is extracted once for every approach and repeat. A state that needs an extraction after the
    job-hour cap is used up is refused as a cap, not as a failure.
    """

    def __init__(self, inner: LiveTelemetryProvider, caps: CapLedger | None, work_root: Path) -> None:
        self.inner, self.caps, self.work = inner, caps, work_root
        self.active_episode: str | None = None

    def identity(self) -> dict[str, Any]:
        return self.inner.identity()

    def __call__(self, state, evidence, producer_recipe_id, run_id):
        if self.caps is not None and self.inner.cost_of(state.id) is None:
            self.caps.check_job_hours()
        out = self.inner(state, evidence, producer_recipe_id, run_id)
        cost = self.inner.cost_of(state.id)
        if cost is not None:
            self._account_once(state, cost)
        with _file_lock(self.work / "extraction_use.lock"):
            with (self.work / "extraction_use.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"episode_id": self.active_episode, "state_sha256": state.id,
                                         "design": state.design, "at_utc": pilot.utc()}, sort_keys=True) + "\n")
        return out

    def _account_once(self, state, cost: dict[str, Any]) -> None:
        path = self.work / "extraction_accounted.json"
        with _file_lock(self.work / "extraction_accounted.lock"):
            done = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            if state.id in done:
                return
            if self.caps is not None:
                self.caps.add_job_seconds(float(cost["wall_s_total"]))      # a crash here over-counts: the safe side
            done[state.id] = {"design": state.design, "wall_s_total": cost["wall_s_total"],
                              "elapsed_s": cost["elapsed_s"], "vivado_launches": cost["vivado_launches"],
                              "at_utc": pilot.utc()}
            _atomic_bytes(path, _json_bytes(done))


def live_provider_factory(work_root: Path, caps_plan: dict[str, Any], plan_sha256: str, extraction_jobs: int):
    """provider_for(store) for a scope: live worker B telemetry, a persisted extraction-job cap, own cache."""

    def make(store: ManifestStateStore) -> AccountedTelemetry:
        runner = Runner(PROJECT / "data/slots", max_slots=pilot.MAX_SLOTS, load_ceiling_1m=pilot.LOAD_CEILING,
                        min_available_memory_bytes=pilot.MEMORY_FLOOR)
        inner = LiveTelemetryProvider(
            work_root / "telemetry_cache", runner=runner, store=store,
            budget=ExtractionBudget(work_root / "extraction_budget.json", extraction_jobs),
            log_path=work_root / "extraction.jsonl")
        return AccountedTelemetry(inner, CapLedger(work_root, caps_plan, plan_sha256), work_root)
    return make


# ------------------------------------------------------------------------------ caps
class CapLedger:
    """Shared, file-locked counters for the four caps. Fail closed on any inconsistency."""

    def __init__(self, work_root: Path, caps: dict[str, float], plan_sha256: str) -> None:
        self.path = work_root / "caps.json"
        self.lock = work_root / "caps.lock"
        self.caps = dict(caps)
        work_root.mkdir(parents=True, exist_ok=True)
        with _file_lock(self.lock):
            if self.path.is_file():
                self._load(plan_sha256)
            else:
                _atomic_bytes(self.path, _json_bytes({
                    "schema_version": "campaign-caps-v1", "plan_sha256": plan_sha256, "caps": self.caps,
                    "used": {"new_real_runs": 0, "job_seconds": 0.0, "tokens": 0, "usd": 0.0},
                    "reservations": [], "stopped": {},
                }))
        self.plan_sha256 = plan_sha256

    def _load(self, plan_sha256: str | None = None) -> dict[str, Any]:
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if value.get("schema_version") != "campaign-caps-v1" or value["caps"] != self.caps:
            raise pilot.PilotSafetyStop("cap ledger belongs to other caps")
        if plan_sha256 is not None and value["plan_sha256"] != plan_sha256:
            raise pilot.PilotSafetyStop("cap ledger belongs to another plan")
        return value

    def _update(self, fn: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        with _file_lock(self.lock):
            value = self._load(self.plan_sha256)
            fn(value)
            _atomic_bytes(self.path, _json_bytes(value))
            return value

    def used(self) -> dict[str, Any]:
        return self._load(self.plan_sha256)["used"]

    def stopped(self) -> dict[str, str]:
        return self._load(self.plan_sha256)["stopped"]

    def _stop(self, value: dict[str, Any], which: str, why: str) -> None:
        value["stopped"].setdefault(which, why)

    def reserve_run(self, episode: str, run_id: str, state_id: str, recipe_id: str) -> None:
        def go(value: dict[str, Any]) -> None:
            used, caps = value["used"], self.caps
            if used["new_real_runs"] >= caps["new_real_runs"]:
                self._stop(value, "new_real_runs", f"reached {caps['new_real_runs']} new real runs")
            elif used["job_seconds"] >= caps["job_hours"] * 3600.0:
                self._stop(value, "job_hours", f"reached {caps['job_hours']} job-hours")
            else:
                used["new_real_runs"] += 1
                value["reservations"].append({"episode_id": episode, "run_id": run_id, "input_state_id": state_id,
                                              "recipe_id": recipe_id, "at_utc": pilot.utc()})
        value = self._update(go)
        if value["stopped"].get("new_real_runs") or value["stopped"].get("job_hours"):
            raise CapReached(value["stopped"].get("new_real_runs") or value["stopped"].get("job_hours"))

    def check_job_hours(self) -> None:
        """Refuse new tool work (an extraction) once the job-hour cap is used."""

        def go(value: dict[str, Any]) -> None:
            if value["used"]["job_seconds"] >= self.caps["job_hours"] * 3600.0:
                self._stop(value, "job_hours", f"reached {self.caps['job_hours']} job-hours")
        value = self._update(go)
        if value["stopped"].get("job_hours"):
            raise CapReached(value["stopped"]["job_hours"])

    def add_job_seconds(self, seconds: float) -> None:
        self._update(lambda v: v["used"].__setitem__("job_seconds", v["used"]["job_seconds"] + float(seconds)))

    def check_llm(self) -> None:
        def go(value: dict[str, Any]) -> None:
            used, caps = value["used"], self.caps
            if used["tokens"] >= caps["tokens"]:
                self._stop(value, "tokens", f"reached the token cap {caps['tokens']}")
            if used["usd"] >= caps["usd"]:
                self._stop(value, "usd", f"reached the US$ cap {caps['usd']}")
        value = self._update(go)
        for which in ("tokens", "usd"):
            if value["stopped"].get(which):
                raise CapReached(value["stopped"][which])

    def add_llm(self, tokens: int, usd: float) -> None:
        def go(value: dict[str, Any]) -> None:
            value["used"]["tokens"] += int(tokens)
            value["used"]["usd"] = round(value["used"]["usd"] + float(usd), 6)
        self._update(go)


class LedgeredLLM(CachedLLM):
    """CachedLLM that checks the token and US$ caps before a fresh call and logs every request.

    The stored request and reply are the replay-cache record named by `key` (prompt, schema, parameters
    and reply, keyed by their hash); the log line adds the hashes of the prompt and of the reply text.
    """

    def __init__(self, inner: CachedLLM, caps: CapLedger | None, log: Path, episode_id: str,
                 price: dict[str, float]) -> None:
        super().__init__(inner.transport, inner.cache, inner.params)
        self.caps, self.log, self.episode_id, self.price = caps, log, episode_id, price

    def complete(self, prompt: str, schema) -> LLMResult:
        live = self.cache.mode != "replay"
        if live and self.caps is not None:
            self.caps.check_llm()
        result = super().complete(prompt, schema)
        reply = result.reply
        usd = (reply.prompt_tokens * self.price["input_per_million"]
               + (reply.completion_tokens + reply.thoughts_tokens) * self.price["output_per_million"]) / 1e6
        if live and not result.cache_hit and self.caps is not None:
            self.caps.add_llm(reply.total_tokens, usd)
        line = {
            "episode_id": self.episode_id, "key": result.key, "mode": result.cache_mode, "cache_hit": result.cache_hit,
            "prompt_sha256": sha(prompt), "reply_sha256": sha(reply.text), "model": reply.model,
            "prompt_tokens": reply.prompt_tokens, "completion_tokens": reply.completion_tokens,
            "thoughts_tokens": reply.thoughts_tokens, "total_tokens": reply.total_tokens,
            "usd": round(usd, 6), "charged": bool(live and not result.cache_hit),
            "record": str(self.cache._path(result.key)), "at_utc": pilot.utc(),
        }
        with _file_lock(self.log.with_suffix(".lock")):
            with self.log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line, sort_keys=True) + "\n")
        return result


# ------------------------------------------------------------------------------ backends
class NoToolBackend:
    """Replay: any tool run is a bug."""

    calls = 0

    def execute(self, *a, **k):
        NoToolBackend.calls += 1
        raise pilot.PilotSafetyStop("replay would start a tool run")

    def measure_start(self, *a, **k):
        NoToolBackend.calls += 1
        raise pilot.PilotSafetyStop("replay would start a start measurement")


class CampaignBackend:
    """Reserve a slot against the run and job-hour caps before any real recipe launch."""

    def __init__(self, backend, caps: CapLedger, check_pins: Callable[[], None]) -> None:
        self.backend, self.caps, self.check_pins = backend, caps, check_pins
        self.active_episode: str | None = None

    def measure_start(self, *a, **k):
        raise pilot.PilotSafetyStop("a start was not found in the manifest; none is measured")

    def execute(self, state, recipe, *, run_id, run_dir) -> RunOutcome:
        self.check_pins()
        if self.active_episode is None:
            raise pilot.PilotSafetyStop("real recipe launch lacks an episode identity")
        self.caps.reserve_run(self.active_episode, run_id, state.id, recipe.id)
        outcome = self.backend.execute(state, recipe, run_id=run_id, run_dir=run_dir)
        self.caps.add_job_seconds(outcome.run.wall_s or 0.0)
        return outcome


# ------------------------------------------------------------------------------ validity and AMD
def start_relative_check(evidence: Evidence, start: Evidence | None) -> None:
    """The authors' rule (2026-11-07): no more DONT_TOUCH / fixed LOC / fixed BEL cells than the start state.

    Defined once, by the development run, in orch/engine/start_properties.py; the pilot layer turns its error
    into PilotSafetyStop. The harness calls that and holds no second copy of the rule.
    """

    pilot.require_clean_properties(evidence, start)


def amd_pass(ledger: ReadThroughLedger, design: str, state_id: str, *, allow_new: bool) -> dict[str, Any]:
    record = ledger.verify(design, state_id) if allow_new else ledger.require_pass(design, state_id)
    attempts = record.get("attempts") or []
    log_path = Path(str(attempts[-1]["run_dir"])) / "validate.log" if attempts else None
    content = log_path.read_text(encoding="utf-8", errors="replace") if log_path and log_path.is_file() else ""
    count = max((int(x) for x in re.findall(r"Outputs\s*\((\d+)\):", content)), default=0)
    if record["result"] != "pass" or count <= 0:
        raise pilot.PilotSafetyStop(f"AMD simulation did not PASS with compared outputs: {design} {state_id}")
    location = ledger.record_location(design, state_id)
    proof = {
        "schema_version": "thursday-night-amd-census-v1", "design": design, "state_sha256": state_id,
        "validator_record_sha256": _sha256_file(location) if location else None,
        "validate_log_sha256": _sha256_file(log_path), "outputs_compared": count,
    }
    path = ledger.root / "output_census" / design / f"{state_id}.json"
    with _file_lock(path.with_suffix(".lock")):
        if path.is_file():
            if path.read_bytes() != _json_bytes(proof):
                raise pilot.PilotSafetyStop("AMD output census proof changed")
        else:
            _atomic_bytes(path, _json_bytes(proof))
    return record


class CampaignEpisodeEngine:
    """Permit a gate verdict into the episode only after the start-relative and AMD checks."""

    def __init__(self, engine: StepEngine, ledger: ReadThroughLedger, *, allow_new_verification: bool,
                 property_check: Callable[[Evidence, Evidence | None], None] = start_relative_check) -> None:
        self.engine, self.ledger, self.store = engine, ledger, engine.store
        self.meter_version, self.cache_enabled = engine.meter_version, engine.cache_enabled
        self.allow_new, self.property_check = allow_new_verification, property_check

    def measure_start(self, state: DesignState) -> Evidence:
        evidence = self.engine.measure_start(state)
        self.property_check(evidence, evidence)
        return evidence

    def apply(self, state, recipe, *, brief, start, parent, guards=(), noise=None) -> AppliedStep:
        applied = self.engine.apply(state, recipe, brief=brief, start=start, parent=parent, guards=guards, noise=noise)
        outcome = applied.outcome
        if outcome.state is not None:
            if outcome.evidence is None:
                raise pilot.PilotSafetyStop("saved output lacks independent meter Evidence")
            self.property_check(outcome.evidence, start)
            amd_pass(self.ledger, outcome.state.design, outcome.state.id, allow_new=self.allow_new)
        return applied


def final_validator(ledger: ReadThroughLedger, brief):
    def saved(golden: DesignState, revised: DesignState) -> FinalValidation:
        if revised.id == golden.id:
            return FinalValidation(check=EquivalenceCheck(
                result="pass", method="identical B2 checkpoint", vectors=brief.equivalence_vectors,
                reference_state_id=golden.id), note="No-change start is byte-identical to the reference B2")
        record = amd_pass(ledger, revised.design, revised.id, allow_new=False)
        return FinalValidation(check=EquivalenceCheck(
            result="pass", method="saved AMD 100+200-vector simulation", vectors=brief.equivalence_vectors,
            reference_state_id=golden.id), artifact_dir=str(record["raw_report_path"]),
            note="Exact-DCP verified output; no final-pick tool rerun")
    return saved


# ------------------------------------------------------------------------------ runtime assembly
@dataclass
class Context:
    scope: Scope
    store: ManifestStateStore
    registry: Any
    engine: CampaignEpisodeEngine
    base_engine: ManifestStepEngine
    ledger: ReadThroughLedger
    backend: Any
    starts: dict[str, DesignState]
    caps: CapLedger | None
    meter_version: str
    mode: str                      # replay | run
    provider: Any = None
    notes: dict[str, Any] = field(default_factory=dict)


def build_context(scope: Scope, *, mode: str, caps: CapLedger | None = None,
                  check_pins: Callable[[], None] = lambda: None) -> Context:
    if mode not in ("replay", "run"):
        raise ValueError(mode)
    root = scope.engine_root
    root.mkdir(parents=True, exist_ok=True)
    if not root.resolve().is_relative_to((PROJECT / "data").resolve()):
        raise ValueError("campaign root must stay inside the project data folder")
    meter_version = scope.manifest.body["meter_version"]
    store = ManifestStateStore(root, scope.manifest)
    cache = ManifestRunCache(root, scope.manifest, meter_version=meter_version)
    runner = Runner(PROJECT / "data/slots", max_slots=pilot.MAX_SLOTS, load_ceiling_1m=pilot.LOAD_CEILING,
                    min_available_memory_bytes=pilot.MEMORY_FLOOR)
    meter = VivadoMeter(runner=runner, env_script=PROJECT / "recipes/support/env.sh", tool_version="2025.1")
    if meter.version != meter_version:
        raise pilot.PilotSafetyStop("the meter version differs from the manifest's")
    registry = load_registry(PROJECT)
    latch = LatchAdapter(PROJECT)
    _require_deterministic_smoke()
    _require_latch_venv_smoke(pilot.SMOKE, recipe_version=registry.get("LATCH.phys_opt").version,
                              meter_version=meter.version, interpreter=str(latch.latch_python))
    if mode == "replay":
        backend: Any = NoToolBackend()
    else:
        assert caps is not None
        real = RealBackend(store=store, runner=runner, meter=meter,
                           adapters={"tcl": TclAdapter(PROJECT), "latch": latch})
        backend = CampaignBackend(real, caps, check_pins)
    base = ManifestStepEngine(manifest=scope.manifest, require_zero_tool=(mode == "replay"), store=store, cache=cache,
                              backend=backend, meter_version=meter_version, cache_enabled=True)
    ledger = ReadThroughLedger(
        source_roots=list(scope.manifest.roots.values()), store=store, runner=runner,
        validator=ValidatorAdapter(PROJECT), references=scope.starts)
    engine = CampaignEpisodeEngine(base, ledger, allow_new_verification=(mode == "run"))
    starts: dict[str, DesignState] = {}
    for design, state_id in scope.starts.items():
        path = store.state_path(state_id)
        starts[design] = DesignState.model_validate_json((path.parent / "state.json").read_bytes())
        if starts[design].id != state_id or str(starts[design].path) != str(path):
            raise pilot.PilotSafetyStop(f"start state record differs from its location: {design}")
    provider = scope.provider_for(store) if scope.provider_for else None
    return Context(scope, store, registry, engine, base, ledger, backend, starts, caps, meter_version, mode, provider)


# ------------------------------------------------------------------------------ cells
@dataclass(frozen=True)
class Cell:
    design: str
    brief: str
    approach: str          # v2 v1 single fixed_AB random_0 ... or an LLM cell name such as F1-R1
    tag: str               # m1 m2 (fresh repeat label) or '-' for deterministic approaches
    tier: int
    episode_id: str
    cache: str = "fresh"   # LLM cache mode: fresh | replay

    @property
    def is_llm(self) -> bool:
        return self.approach in settings()["llm_cells"]

    @property
    def workers(self) -> str:
        return "abc" if self.brief == "P0" else "ab"


def policy_for_approach(approach: str, brief: str):
    if approach == "v1":
        return RepairLoopPolicy()
    if approach == "v2":
        return RepairLoopPolicy(**settings()["v2_parameters"])
    if approach == "single":
        return SingleSpecialistPolicy()
    if approach.startswith("fixed_"):
        order = approach.split("_", 1)[1]
        return FixedOrderPolicy(order=order) if brief == "P0" else FixedAlternationPolicy(first=order[0])
    if approach.startswith("random_"):
        seed = int(approach.split("_", 1)[1])
        return RelevantRandomPolicy(seed=seed) if brief == "P0" else UniformRandomPolicy(seed=seed)
    raise ValueError(f"no policy builder for {approach}")


def llm_config(approach: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The development run's llm.yaml with this cell's template; returns (cfg, cell spec)."""

    camp = settings()
    spec = camp["llm_cells"][approach]
    cfg = yaml.safe_load((PROJECT / camp["llm_yaml"]).read_text(encoding="utf-8"))
    if spec["kind"] == "orchestrator":
        cfg["templates"] = dict(cfg["templates"], orchestrator=camp["prompt_revisions"][spec["revision"]])
    return cfg, spec


def make_workers(ctx: Context, cell: Cell) -> dict[str, Any]:
    sets = load_recipe_sets(PROJECT / "configs/recipe_sets.yaml")
    sets.require_signed_off()
    workers: dict[str, Any] = {
        wid: FittedWorker(load_worker_card(PROJECT / "configs/workers" / f"{wid}.yaml"), sets.worker_sets[wid], ctx.registry)
        for wid in ("timing", "area")
    }
    if any(w.model_label != "FINAL" for w in workers.values()):
        raise pilot.PilotSafetyStop("requires both frozen FINAL workers")
    if cell.workers == "abc":
        power = load_recipe_sets(PROJECT / "configs/power_recipe_set.yaml")
        card = load_worker_card(PROJECT / "configs/workers/power.yaml")
        if card.id != "power" or list(power.worker_sets["power"]) != json.loads(RANKING.read_text())["ranking"]:
            raise pilot.PilotSafetyStop("power worker card or order differs from worker C's ranking")
        workers["power"] = GenericWorker(card, power.worker_sets["power"])
    if cell.is_llm and settings()["llm_cells"][cell.approach]["kind"] == "general":
        workers = {wid: DirectRecipeWorker(w.card, w.recipe_ids) for wid, w in workers.items()}
    return workers


def make_policy(ctx: Context, cell: Cell, worker_ids: list[str]):
    if not cell.is_llm:
        return policy_for_approach(cell.approach, cell.brief)
    cfg, spec = llm_config(cell.approach)
    work = ctx.scope.work_root
    if cell.cache == "replay":
        root = Path(getattr(ctx, "notes", {}).get("replay_cache_root") or DEV_SHARED)      # the stage's own store
        cache = ReplayCache(root / "llm_cache", mode="replay")
        transport = None
    else:
        cache = ReplayCache(work / "llm_cache", mode="fresh", fresh_tag=cell.tag)
        transport = getattr(ctx, "notes", {}).get("fake_transport") or BridgeTransport(
            PROJECT / cfg["bridge"]["python"], PROJECT / cfg["bridge"]["script"],
            rate_limiter=RateLimiter(work / "ratelimit", cfg["rate_limit_per_minute"]))
    params = LLMParams(model=cfg["model"], temperature=cfg["temperature"], max_output_tokens=cfg["max_output_tokens"],
                       seed=cfg["seed"], thinking_budget=cfg["thinking_budget"], timeout_ms=cfg["timeout_ms"])
    llm = LedgeredLLM(CachedLLM(transport, cache, params), ctx.caps, work / "llm_calls.jsonl", cell.episode_id,
                      settings()["price"])
    sink = TelemetrySink(work / "telemetry.jsonl")
    if spec["kind"] == "general":
        return build_general_agent(PROJECT, cfg, llm, sink, worker_ids)
    conf = cfg["configurations"][spec["config"]]
    manifest = json.loads((PROJECT / cfg["profiles_dir"] / "manifest.json").read_text(encoding="utf-8"))
    texts, hashes = {}, {}
    if conf["profiles"]:
        for wid in worker_ids:
            data = yaml.safe_load((PROJECT / cfg["profiles_dir"] / f"{wid}.yaml").read_text(encoding="utf-8"))
            texts[wid], hashes[wid] = data["text"], manifest["files"][f"{wid}.yaml"]
    return LLMPolicy(LLMPolicyConfig(
        name=spec["config"], mode=conf["mode"], profiles_on=conf["profiles"],
        template=load_template(PROJECT / cfg["templates"]["orchestrator"]), profile_texts=texts,
        profile_sha256=hashes, lessons=None, max_repairs=cfg["max_repair_turns"]), llm, sink)


# ------------------------------------------------------------------------------ one episode
def run_cell(ctx: Context, cell: Cell, result_dir: Path, *, run_no: int = 0) -> tuple[str, dict[str, Any] | None]:
    from orch.baselines.verification import VerificationFailure  # noqa: F401

    brief = load_frozen_brief(cell.brief)
    start = ctx.starts[cell.design]
    start_evidence = ctx.engine.measure_start(start)
    workers = make_workers(ctx, cell)
    policy = make_policy(ctx, cell, sorted(workers))
    fitted = any(isinstance(w, FittedWorker) for w in workers.values())
    if fitted and ctx.provider is None:
        raise pilot.PilotSafetyStop("no telemetry provider for this scope (worker B needs one)")
    if hasattr(ctx.backend, "active_episode"):
        ctx.backend.active_episode = cell.episode_id
    if hasattr(ctx.provider, "active_episode"):
        ctx.provider.active_episode = cell.episode_id
    try:
        result = run_episode(
            engine=ctx.engine, registry=ctx.registry, workers=workers, policy=policy, brief=brief, start_state=start,
            episode_id=cell.episode_id, noise=ctx.scope.noise(cell.design, start_evidence),
            telemetry_provider=ctx.provider if fitted else None, validator=final_validator(ctx.ledger, brief),
        )
    except (CapReached, ExtractionCapReached) as exc:
        return "capped", {"episode_id": cell.episode_id, "reason": str(exc)}
    except Exception as exc:
        kind = infrastructure_kind(exc)
        if kind is None:
            raise
        # Nothing was journalled for the failed call: the episode resumes from its journal when re-queued.
        return "infrastructure_failure", {"episode_id": cell.episode_id, "kind": kind,
                                          "reason": f"{type(exc).__name__}: {str(exc)[:300]}"}
    finally:
        if hasattr(ctx.backend, "active_episode"):
            ctx.backend.active_episode = None
        if hasattr(ctx.provider, "active_episode"):
            ctx.provider.active_episode = None
    row = result_row(ctx, cell, result, policy, run_no)
    _atomic_bytes(result_dir / f"{cell.episode_id}.json", _json_bytes(row))
    return "done", row


def llm_identity(cell: Cell) -> dict[str, Any] | None:
    """What an LLM cell was given, for the pre-registered checks: the template hash, the workers it saw and the
    profile hash of each (the profile of worker C is its only description beyond its card)."""

    if not cell.is_llm:
        return None
    cfg, spec = llm_config(cell.approach)
    key = "general_agent" if spec["kind"] == "general" else "orchestrator"
    workers = ["timing", "area", "power"] if cell.workers == "abc" else ["timing", "area"]
    identity: dict[str, Any] = {"kind": spec["kind"], "template": cfg["templates"][key],
                                "template_sha256": _sha256_file(PROJECT / cfg["templates"][key]),
                                "model": cfg["model"], "workers_seen": sorted(workers)}
    if spec["kind"] == "orchestrator" and cfg["configurations"][spec["config"]]["profiles"]:
        manifest = json.loads((PROJECT / cfg["profiles_dir"] / "manifest.json").read_text(encoding="utf-8"))
        identity["profile_sha256"] = {w: manifest["files"][f"{w}.yaml"] for w in sorted(workers)}
    return identity


def result_row(ctx: Context, cell: Cell, result, policy, run_no: int) -> dict[str, Any]:
    if (result.validation_status != "pass" or result.final_state is None or result.final_evidence is None
            or result.final_verdict is None or not result.final_verdict.brief_valid):
        raise pilot.PilotSafetyStop("completed episode lacks a verified brief-valid final pick")
    events = Journal(ctx.store.root).read(cell.episode_id)
    runs = [ctx.store.load_outcome(e["step"]["run_id"]).run for e in events if e.get("type") == "step"]
    reserved = []
    if ctx.caps is not None:
        reserved = [r for r in ctx.caps._load(ctx.caps.plan_sha256)["reservations"] if r["episode_id"] == cell.episode_id]
    return {
        "schema_version": "campaign-result-v1", "episode_id": cell.episode_id, "scope": ctx.scope.name,
        "design": cell.design, "brief": cell.brief, "approach": cell.approach, "tag": cell.tag, "tier": cell.tier,
        "run_no": run_no, "final_state_id": result.final_state.id,
        "final_metrics": {n.value: m.value for n, m in result.final_evidence.metrics.items()},
        "final_brief_valid": True, "validation_status": result.validation_status,
        "runs_used": result.recipe_runs, "cache_hits": sum(bool(r.cache_hit) for r in runs),
        "new_real_runs": len(reserved), "stop_reason": result.stop_reason,
        "charged_wall_s": sum((r.wall_s or 0.0) for r in runs),
        "charged_cpu_s": sum((r.cpu_user_s or 0.0) + (r.cpu_sys_s or 0.0) for r in runs),
        "policy_version": getattr(policy, "version", None), "llm": llm_identity(cell),
        "completed_at_utc": pilot.utc(),
    }


# ------------------------------------------------------------------------------ journal traces
def normalised_events(events: list[dict[str, Any]]) -> list[Any]:
    """Journal events with run ids and step ids (new in every replay) replaced by a fixed token."""

    text = json.dumps(events, sort_keys=True)
    text = STEP_ID.sub("<step>", RUN_ID.sub("<run>", text))
    value = json.loads(text)
    for event in value:                                  # the journal's own location is not an outcome
        if isinstance(event.get("result"), dict):
            event["result"].pop("journal_path", None)
    return value


def first_difference(a: Any, b: Any, path: str = "") -> str | None:
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}/{k}: present on one side only"
            d = first_difference(a[k], b[k], f"{path}/{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {str(a)[:80]} vs {str(b)[:80]}"


# ------------------------------------------------------------------------------ pins
CONFIG_PINS = (
    "configs/campaign/campaign.yaml", "configs/policies/llm.yaml", "configs/policies/llm_r1.yaml",
    "configs/prompts/orchestrator_v1.txt", "configs/prompts/orchestrator_v2.txt", "configs/prompts/general_agent_v1.txt",
    "configs/profiles/manifest.json", "configs/profiles/timing.yaml", "configs/profiles/area.yaml",
    "configs/profiles/power.yaml", "configs/profiles/recipes.yaml", "configs/briefs.yaml", "configs/noise.yaml",
    "configs/recipe_sets.yaml", "configs/power_recipe_set.yaml", "configs/workers/timing.yaml",
    "configs/workers/area.yaml", "configs/workers/power.yaml", "configs/workers/A_fit.json",
    "configs/workers/B_fit.json", "configs/worker_features.yaml", "recipes/registry.yaml", "recipes/support/env.sh",
    "tools/llm_bridge/call.py", "tools/llm_bridge/requirements.lock",
)


def imported_hashes(extra: tuple[str, ...] = ()) -> dict[str, str]:
    """SHA-256 of every project file this process has imported, plus the named configuration files."""

    found: dict[str, str] = {}
    root = str(PROJECT) + os.sep
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and path.startswith(root) and "/.venv/" not in path and "/data/" not in path and path.endswith(".py"):
            found[path[len(root):]] = _sha256_file(Path(path))
    for relative in (*CONFIG_PINS, *extra):
        if (PROJECT / relative).is_file():
            found[relative] = _sha256_file(PROJECT / relative)
    return dict(sorted(found.items()))


def pin_changes(start: dict[str, str]) -> list[str]:
    now = imported_hashes()
    return sorted(k for k, v in start.items() if now.get(k) != v)


# ------------------------------------------------------------------------------ the cell plan
PERMS = ("ABC", "ACB", "BAC", "BCA", "CAB", "CBA")
BRIEF_ORDER = ("A0", "T0", "T", "A", "P0")        # the primary briefs first


def approaches(brief: str, repeats: int = 2) -> list[tuple[str, str, int]]:
    """(approach, tag, tier) in priority order for one design and brief.

    Tier 1: v2, v1, single specialist, fixed orders, random seeds, F1-R1, GA-R1. Tier 2: E0-R1 and
    F1-R0. Tier 3: the fresh repeats of F1-R1 and GA-R1 (one repeat if the model is deterministic on
    training, two if not). Deterministic approaches carry the tag x.
    """

    fixed = [f"fixed_{p}" for p in (PERMS if brief == "P0" else ("AB", "BA"))]
    tier1 = [(a, "x", 1) for a in ("v2", "v1", "single", *fixed, "random_0", "random_1", "random_2")]
    tier1 += [("F1-R1", "m1", 1), ("GA-R1", "m1", 1)]
    tier2 = [("E0-R1", "m1", 2), ("F1-R0", "m1", 2)]
    tier3 = [(a, f"m{i}", 3) for i in range(2, repeats + 1) for a in ("F1-R1", "GA-R1")]
    return tier1 + tier2 + tier3


def make_cells(scope_name: str, designs: tuple[str, ...], briefs: tuple[str, ...], repeats: int = 2) -> list[Cell]:
    rows: list[Cell] = []
    for tier in (1, 2, 3):
        for brief in sorted(briefs, key=BRIEF_ORDER.index):
            for approach, tag, t in approaches(brief, repeats):
                if t != tier:
                    continue
                for design in designs:
                    rows.append(Cell(design, brief, approach, tag, tier, f"{scope_name}_{design}_{brief}_{approach}_{tag}"))
    if len({c.episode_id for c in rows}) != len(rows):
        raise ValueError("duplicate cell")
    return rows
