"""D6-v2 (Thu 12 Nov 2026, night): episode evaluation of template v2.1 under the round-2 loop options.

Shared constants, scopes, builders and readers. Read-only on every stored root: stored outcomes are read through
hash-pinned union manifests (scripts/campaign_manifest.py build_generic, imported); new outputs, the cap ledger, the
model cache, the telemetry cache and every log go only to data/d6v2_2026-11-12/. Imported, not edited: the campaign
harness (scripts/campaign_harness.py), the template-v2 code and the round-2 loop and harness modules.

Never opened: the five round-1 test designs and t2_or1200.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(os.environ.get("ORCH_PROJECT_ROOT") or Path(__file__).resolve().parents[2])
for _p in (str(PROJECT), str(PROJECT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# D6_WORK and D6_SMOKE=1 are for the smoke test only: a throwaway folder under the main one, caps that allow no new
# real run and US$1 of model calls.
WORK = Path(os.environ.get("D6_WORK") or (PROJECT / "data/d6v2_2026-11-12"))
SMOKE = os.environ.get("D6_SMOKE") == "1"
TRAIN_WORK = WORK / "train"
DEV_WORK = WORK / "dev"
PLAN = WORK / "plan"
LOGS = WORK / "logs"
RESULTS = WORK / "results"
JOURNAL = WORK / "journal.jsonl"
STATUS = WORK / "status.json"
PIDFILE = WORK / "queue.pid.json"
STOPFILE = WORK / "STOP"
CONFIG = WORK / "queue_config.json"
MANIFEST_TRAIN = WORK / "manifest_train"
MANIFEST_DEV = WORK / "manifest_dev"
TELEMETRY_CACHE = WORK / "telemetry_cache"
LLM_CACHE = WORK / "llm_cache"
LLM_CALLS = WORK / "llm_calls.jsonl"
TELEMETRY = WORK / "telemetry.jsonl"
PLAN_NOTE = PROJECT / "notes/d6v2_plan.md"

TRAINING = ("vexriscv_re-place", "logicnets_jscl", "rosetta_spam-filter", "rosetta_3d-rendering", "finn_radioml")
DEVELOPMENT = ("t2_cordic_s", "vtr_mcml_v2", "amd_mini-isp")
FORBIDDEN = (
    "t2_or1200", "rosetta_digit-recognition", "corescore_500_mod", "vexriscv_re-place_v2",
    "fir_symmetric_systolic", "fir_systolic_transposed",
)
BRIEFS = ("T0", "A0")
LLM_APPROACHES = ("v21", "F1")
BASELINES = ("v2", "fixed_BA", "random_0", "random_1", "random_2")
SEEDS = {"r1": 0, "r2": 1, "r3": 2}
TEMPERATURE = 0.3
CAPS = ({"new_real_runs": 0, "job_hours": 1, "tokens": 2_000_000, "usd": 1.0} if SMOKE else
        {"new_real_runs": 450, "job_hours": 100, "tokens": 40_000_000, "usd": 15.0})
EXTRACTION_JOBS = 300
LOAD_GUARD = 160.0
PRICE = {"input_per_million": 0.25, "output_per_million": 1.50}
NOISE_FILE = PROJECT / "configs/noise_round2.yaml"
V21_DEFINITION = PROJECT / "configs/policies/template_v2.1.yaml"
READ_ONLY_TELEMETRY = (
    PROJECT / "data/development_designs_2026-11-12/telemetry_cache",
    PROJECT / "data/feature_ablation_2026-11-12/shared/live_telemetry",
)
DEV_ENGINE_D3 = PROJECT / "data/development_designs_2026-11-12/engine"
BX = PROJECT.parent / "benchmark_expansion_2026-11-10/data/engine"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def jread(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jwrite(path: Path | str, payload: Any, *, indent: int | None = 1) -> None:
    """Atomic JSON write (same-directory temp file, fsync, replace)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def guard_design(design: str) -> str:
    if design in FORBIDDEN or design not in TRAINING + DEVELOPMENT:
        raise SystemExit(f"refused: {design} is not a training or development design")
    return design


def group_of(design: str) -> str:
    return "TRAINING" if guard_design(design) in TRAINING else "DEVELOPMENT"


def plan_sha() -> str:
    import hashlib

    return hashlib.sha256(PLAN_NOTE.read_bytes()).hexdigest()


def episode_id(design: str, brief: str, approach: str, tag: str) -> str:
    return f"d6_{design}_{brief}_{approach}_{tag}"


# ---------------------------------------------------------------------------------------------------- noise
_NOISE = None


def noise_for(design: str, start_evidence) -> dict:
    global _NOISE
    from orch.config import load_noise

    if _NOISE is None:
        _NOISE = load_noise(NOISE_FILE)
    return _NOISE.for_start(guard_design(design), start_evidence)


# ---------------------------------------------------------------------------------------------------- scopes
def _manifest(folder: Path):
    from orch.engine.manifest_cache import CacheManifest

    expected = (folder / "manifest.sha256").read_text(encoding="utf-8").split()[0]
    return CacheManifest.load(folder / "manifest.json", expected_hash=expected)


def _charging_ledger_class():
    from orch.engine.read_through_ledger import ReadThroughLedger

    class ChargingLedger(ReadThroughLedger):
        """ReadThroughLedger that adds the wall time of a NEW AMD validation to the job-second cap."""

        caps = None

        def verify(self, design: str, output_id: str) -> dict:
            known = self.record_location(design, output_id) is not None
            record = super().verify(design, output_id)
            if not known and self.caps is not None:
                self.caps.add_job_seconds(sum(float(a["wall_s"]) for a in record.get("attempts", [])))
            return record

    return ChargingLedger


def live_telemetry(store, caps):
    """Live worker B telemetry (and the physical description's extractor): own cache, persisted job cap, accounting."""

    import campaign_harness as h
    import thursday_night_pilot as pilot
    from orch.engine.runner import Runner
    from orch.workers.live_telemetry import ExtractionBudget, LiveTelemetryProvider

    runner = Runner(PROJECT / "data/slots", max_slots=pilot.MAX_SLOTS, load_ceiling_1m=LOAD_GUARD,
                    min_available_memory_bytes=pilot.MEMORY_FLOOR)
    inner = LiveTelemetryProvider(
        TELEMETRY_CACHE, runner=runner, store=store,
        budget=ExtractionBudget(WORK / "extraction_budget.json", EXTRACTION_JOBS), log_path=WORK / "extraction.jsonl")
    return h.AccountedTelemetry(inner, caps, WORK)


def build(scope_name: str):
    """Campaign-harness Context for 'train' or 'dev', real runs against the shared D6 cap ledger."""

    import campaign_harness as h
    import thursday_night_pilot as pilot

    pilot.LOAD_CEILING = LOAD_GUARD                 # every tool job of this process waits while load1 > 160
    caps = h.CapLedger(WORK, CAPS, plan_sha())
    h.ReadThroughLedger = _charging_ledger_class()
    if scope_name == "train":
        manifest = _manifest(MANIFEST_TRAIN)
        starts = h._starts_from_manifest(manifest)
        if set(starts) != set(TRAINING):
            raise SystemExit("training manifest does not hold exactly the five training starts")
        roots = list(manifest.roots.values())
        provider_for = lambda store: h.ScopeTelemetryProvider(h.V4, store, roots)  # noqa: E731
        scope = h.Scope("train", TRAINING, manifest, TRAIN_WORK, starts, noise_for, provider_for)
    elif scope_name == "dev":
        manifest = _manifest(MANIFEST_DEV)
        starts = h._starts_from_manifest(manifest)
        if set(starts) != set(DEVELOPMENT):
            raise SystemExit("development manifest does not hold exactly the three development starts")
        scope = h.Scope("development", DEVELOPMENT, manifest, DEV_WORK, starts, noise_for,
                        lambda store: live_telemetry(store, caps))
    else:
        raise ValueError(scope_name)
    ctx = h.build_context(scope, mode="run", caps=caps)
    ctx.ledger.caps = caps
    return ctx


# ---------------------------------------------------------------------------------------------------- descriptions
class DescribeSource:
    """state id -> dcp-telemetry-v1 record for the physical description (template v2.1, P3).

    Order: the pinned v3 training extraction (free); this task's telemetry cache (records copied at setup from the
    development and feature-ablation caches when made by the current extractor, plus this task's extractions);
    otherwise a live extraction through the same accounted provider worker B uses (job-hours, extraction cap).
    Anything that cannot be produced raises DescriptionUnavailableError (an infrastructure failure, never a decision).
    """

    def __init__(self, ctx, live) -> None:
        self.ctx, self.live = ctx, live
        self.memo: dict[str, dict] = {}
        self._overlay_index: dict[str, list[str]] | None = None
        self._manifest_index: dict[str, list[tuple[str, str]]] | None = None
        self.served = {"v3": 0, "cache": 0, "live": 0}

    def _v3(self, state_id: str):
        from orch.analysis.fit_workers import TRAINING_V3_PARSER_SHA256, TRAINING_V3_SOURCE
        from orch.policies.v2_policy import DescriptionUnavailableError

        path = TRAINING_V3_SOURCE / "states" / state_id / "dcp_telemetry.json"
        if not path.is_file():
            return None
        record = jread(path)
        if (record.get("state_sha256") != state_id or record.get("fake") is not False
                or record.get("provenance", {}).get("extractor_sha256") != TRAINING_V3_PARSER_SHA256):
            raise DescriptionUnavailableError(f"stored telemetry for {state_id[:12]} has the wrong identity")
        return record

    def _cached(self, state_id: str):
        path = TELEMETRY_CACHE / state_id[:2] / state_id / "features.json"
        if not path.is_file():
            return None
        record = jread(path)
        if record.get("state_sha256") != state_id or record.get("fake") is not False:
            return None
        return record

    def _producer(self, state_id: str):
        """(state, evidence, recipe id, run id) of a run that produced this state: own overlay first, then the manifest."""

        store = self.ctx.store
        if self._overlay_index is None or state_id not in self._overlay_index:
            index: dict[str, list[str]] = {}
            if store.runs_dir.is_dir():
                for run_dir in sorted(store.runs_dir.iterdir()):
                    out = run_dir / "output_state.json"
                    if (run_dir / "complete.json").is_file() and out.is_file():
                        try:
                            index.setdefault(jread(out)["id"], []).append(run_dir.name)
                        except (OSError, ValueError, KeyError):
                            continue
            self._overlay_index = index
        if self._manifest_index is None:
            mi: dict[str, list[tuple[str, str]]] = {}
            for entry in self.ctx.scope.manifest.body["entries"].values():
                if entry.get("output_state_id") and entry.get("status") == "ok":
                    mi.setdefault(entry["output_state_id"], []).append((entry["run_id"], entry["recipe_id"]))
            self._manifest_index = mi
        candidates = list(self._overlay_index.get(state_id, [])) + [r for r, _ in self._manifest_index.get(state_id, [])]
        for run_id in candidates:
            try:
                outcome = store.load_outcome(run_id)
            except (FileNotFoundError, ValueError):
                continue
            if outcome.state is not None and outcome.state.id == state_id and outcome.evidence is not None:
                return outcome.state, outcome.evidence, outcome.run.recipe_id, run_id
        return None

    def __call__(self, state_id: str) -> dict:
        from orch.policies.v2_policy import DescriptionUnavailableError

        if state_id in self.memo:
            return self.memo[state_id]
        record = self._v3(state_id)
        if record is not None:
            self.served["v3"] += 1
        else:
            record = self._cached(state_id)
            if record is not None:
                self.served["cache"] += 1
        if record is None:
            start = next((s for s in self.ctx.starts.values() if s.id == state_id), None)
            try:
                if start is not None:
                    served = self.live(start, self.ctx.engine.measure_start(start), None, None)
                else:
                    found = self._producer(state_id)
                    if found is None:
                        raise DescriptionUnavailableError(f"no producing run found for state {state_id[:12]}")
                    served = self.live(*found)
            except DescriptionUnavailableError:
                raise
            except Exception as exc:  # noqa: BLE001 - never a decision on missing evidence
                raise DescriptionUnavailableError(f"{type(exc).__name__}: {str(exc)[:300]}") from exc
            record = served["telemetry"]
            self.served["live"] += 1
        self.memo[state_id] = record
        return record


def seed_telemetry_cache() -> dict[str, int]:
    """Copy features.json records made by the current extractor from the read-only caches (no extraction.json, so
    no job-hours are charged for work done by another task). Source caches are only read."""

    from orch.engine.dcp_parser import DcpTelemetryExtractor
    from orch.engine.runner import Runner

    current = DcpTelemetryExtractor(runner=Runner(PROJECT / "data/slots", max_slots=1),
                                    env_script=PROJECT / "recipes/support/env.sh", tool_version="2025.1",
                                    meter=None, timeout_s=1800).extractor_sha256
    counts = {"copied": 0, "present": 0, "other_extractor": 0, "unreadable": 0}
    for root in READ_ONLY_TELEMETRY:
        for path in sorted(root.glob("*/*/features.json")):
            state_id = path.parent.name
            target = TELEMETRY_CACHE / state_id[:2] / state_id / "features.json"
            try:
                record = jread(path)
            except Exception:  # noqa: BLE001
                counts["unreadable"] += 1
                continue
            if record.get("provenance", {}).get("extractor_sha256") != current or record.get("state_sha256") != state_id:
                counts["other_extractor"] += 1
                continue
            if target.is_file():
                counts["present"] += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            counts["copied"] += 1
    return counts


# ---------------------------------------------------------------------------------------------------- policies
def _llm_settings() -> dict:
    import yaml

    return yaml.safe_load((PROJECT / "configs/policies/llm.yaml").read_text(encoding="utf-8"))


def make_llm(task: dict, caps):
    import campaign_harness as h
    from orch.policies.llm_client import BridgeTransport, CachedLLM, LLMParams, RateLimiter, ReplayCache

    cfg = _llm_settings()
    transport = BridgeTransport(PROJECT / cfg["bridge"]["python"], PROJECT / cfg["bridge"]["script"],
                                rate_limiter=RateLimiter(WORK / "ratelimit", cfg["rate_limit_per_minute"]))
    cache = ReplayCache(LLM_CACHE, mode="fresh", fresh_tag=f"d6v2-{task['approach']}-{task['tag']}")
    params = LLMParams(model=cfg["model"], temperature=TEMPERATURE, max_output_tokens=cfg["max_output_tokens"],
                       seed=SEEDS[task["tag"]], thinking_budget=cfg["thinking_budget"], timeout_ms=cfg["timeout_ms"])
    return h.LedgeredLLM(CachedLLM(transport, cache, params), caps, LLM_CALLS, task["episode_id"], PRICE)


def make_policy(ctx, task: dict, worker_ids: list[str], describe=None):
    import campaign_harness as h
    import yaml
    from orch.policies.llm import LLMPolicyConfig
    from orch.policies.llm_client import TelemetrySink
    from orch.policies.llm_prompt import load_template
    from orch.policies.relevant_workers import FixedOrderPolicy  # noqa: F401  (three-worker briefs are not run)
    from orch.policies.repair_loop import RepairLoopPolicy
    from orch.policies.round2_harness import F1Policy, V21Policy
    from orch.policies.training_baselines import FixedAlternationPolicy, UniformRandomPolicy
    from orch.policies.v2_experience import load_calls_v2
    from orch.policies.v2_prompt import Variant, load_assets

    approach = task["approach"]
    if approach == "v2":
        return RepairLoopPolicy(**h.settings()["v2_parameters"])
    if approach == "fixed_BA":
        return FixedAlternationPolicy(first="B")
    if approach.startswith("random_"):
        return UniformRandomPolicy(seed=int(approach.split("_", 1)[1]))
    llm = make_llm(task, ctx.caps)
    sink = TelemetrySink(TELEMETRY)
    cfg = _llm_settings()
    if approach == "F1":
        conf = cfg["configurations"]["F1"]
        manifest = jread(PROJECT / cfg["profiles_dir"] / "manifest.json")
        texts, hashes = {}, {}
        for wid in worker_ids:
            data = yaml.safe_load((PROJECT / cfg["profiles_dir"] / f"{wid}.yaml").read_text(encoding="utf-8"))
            texts[wid], hashes[wid] = data["text"], manifest["files"][f"{wid}.yaml"]
        template = h.settings()["prompt_revisions"]["R1"]
        return F1Policy(LLMPolicyConfig(
            name="F1", mode=conf["mode"], profiles_on=conf["profiles"], template=load_template(PROJECT / template),
            profile_texts=texts, profile_sha256=hashes, lessons=None, max_repairs=cfg["max_repair_turns"]), llm, sink)
    if approach == "v21":
        definition = yaml.safe_load(V21_DEFINITION.read_text(encoding="utf-8"))
        variant = Variant(definition["name"], tuple(definition["packages"]), wording=definition["worker_wording"])
        assets = load_assets(PROJECT)
        if assets.templates[variant.combo].sha256 != definition["template_sha256"]:
            raise SystemExit("the v2.1 template differs from its frozen definition")
        calls, _sha = load_calls_v2(PROJECT / definition["experience_calls"])
        config = LLMPolicyConfig(name="template_v2.1", mode="fuzzy", profiles_on=False,
                                 template=assets.templates["base"], max_repairs=cfg["max_repair_turns"])
        return V21Policy(config, llm, sink, variant=variant, assets=assets, design=task["design"], describe=describe,
                         r2_experience=None, calls_v2=calls)
    raise ValueError(f"no policy for {approach}")


# ---------------------------------------------------------------------------------------------------- readers
def read_journal() -> dict[str, dict]:
    """Latest row per task key (a torn last line after a hard kill is ignored)."""

    rows: dict[str, dict] = {}
    if JOURNAL.is_file():
        for line in JOURNAL.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[row["task"]] = row
    return rows


def tasks() -> list[dict]:
    return jread(PLAN / "tasks.json")


def depth3_running() -> bool:
    """True while the exhaustive depth-3 queue (scripts/exhaustive_d3_2026-11-12) is alive."""

    pidfile = PROJECT / "data/exhaustive_d3_2026-11-12/queue.pid.json"
    if not pidfile.is_file():
        return False
    try:
        info = jread(pidfile)
        cmd = Path(f"/proc/{info['pid']}/cmdline").read_bytes().decode(errors="replace")
        stat = Path(f"/proc/{info['pid']}/stat").read_text()
        return "ed_queue.py" in cmd and stat[stat.rindex(")") + 2:].split()[0] != "Z"
    except (OSError, KeyError, ValueError):
        return False
