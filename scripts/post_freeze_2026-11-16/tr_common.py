"""POST-FREEZE COPY (Mon 16 Nov 2026) of scripts/test_r2_2026-11-13/tr_common.py: queue slot "pf" only, changes listed in pf_make_driver.py.
Test-run driver (Fri 13 Nov 2026): shared constants, scopes, noise, load rule, model and policy plug-ins.

Pattern of scripts/d6v2_2026-11-12 (copied, not edited). One queue runs any policy row on the five round-1 TEST designs and the seven
round-2 briefs under orch/episode_r2.py with both options on (empty call = observation, stop accepted only after two calls without gain
beyond noise or when the budget is spent). Stored outcomes are read through a hash-pinned union manifest over every stored root (including
data/exhaustive_d3_2026-11-12/engine); only missing (state, recipe) pairs run. Frozen files are imported, never edited.

Folders. TR_DRY unset: everything of the real run, TEST designs only, goes to data/test_r2_2026-11-13/sealed/ (mode 700). TR_DRY=1: the dry
run, VALIDATION designs only, replay mode (no tool job can start), goes to data/test_r2_2026-11-13/dryrun/. Nothing else is written, except
counts-only lines in notes/progress.md.

Never opened: t2_or1200; the TEST designs from a dry run; the training designs.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(os.environ.get("ORCH_PROJECT_ROOT") or Path(__file__).resolve().parents[2])
for _p in (str(PROJECT), str(PROJECT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DRY = os.environ.get("TR_DRY") == "1"        # VALIDATION designs, replay mode, folder dryrun/
SMOKE = os.environ.get("TR_SMOKE") == "1"    # TEST designs, replay mode (no tool job can start), folder sealed/smoke/
REPLAY = DRY or SMOKE
# TR_QUEUE (added 2026-11-13 evening): '' = the baseline queue (sealed/), 'llm1' = TEST LLM rows, own caps and manifest (sealed/llm1/),
# 'power' = TRAINING and VALIDATION power review (power_review/). Sat 14 Nov: 'llm3' = TEST timing and area rows of the v2.2 variants V3 and V6
# (sealed/llm3/), 'llm4' = TEST power rows F1-R1, v2.1 and GA-R1 on W2R, L2R, P0 (sealed/llm4/); each with its own caps and manifest.
QUEUE = os.environ.get("TR_QUEUE", "")
if QUEUE != "pf":
    raise SystemExit("refused: the post-freeze driver runs only with TR_QUEUE=pf (the earlier queues belong to scripts/test_r2_2026-11-13)")
_QROOT = "post_freeze_2026-11-16"
LLM_QUEUES = ("pf",)
BASE = PROJECT / "data/test_r2_2026-11-13"
ROOT = BASE / (f"{_QROOT}/dry_validation" if DRY else f"{_QROOT}/smoke" if SMOKE else _QROOT)
TEST_WORK = ROOT / "test"
DEV_WORK = ROOT / "dev"
TRAIN_WORK = ROOT / "train"
PLAN = ROOT / "plan"
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
JOURNAL = ROOT / "journal.jsonl"
STATUS = ROOT / "status.json"
PIDFILE = ROOT / "queue.pid.json"
STOPFILE = ROOT / "STOP"
CONFIG = ROOT / "queue_config.json"
OVERRIDE = ROOT / "override.json"
TOKENS = ROOT / "override_tokens"
MANIFEST_TEST = BASE / (f"{_QROOT}/manifest_test" if QUEUE in LLM_QUEUES else "sealed/manifest_test")
MANIFEST_DEV = PROJECT / "data/d6v2_2026-11-12/manifest_dev" if DRY else BASE / _QROOT / "manifest_dev"
MANIFEST_TRAIN = BASE / _QROOT / "manifest_train"
TELEMETRY_CACHE = ROOT / "telemetry_cache"
LLM_CACHE = ROOT / "llm_cache"
LLM_CALLS = ROOT / "llm_calls.jsonl"
TELEMETRY = ROOT / "telemetry.jsonl"
PRICES = BASE / _QROOT / "prices.json"
PLAN_NOTE = PROJECT / "notes/test_r2_post_freeze_plan.md"

TEST = (
    "rosetta_digit-recognition", "corescore_500_mod", "vexriscv_re-place_v2", "fir_symmetric_systolic",
    "fir_systolic_transposed",
)
DEVELOPMENT = ("t2_cordic_s", "vtr_mcml_v2", "amd_mini-isp")
TRAINING = ("vexriscv_re-place", "logicnets_jscl", "rosetta_spam-filter", "rosetta_3d-rendering", "finn_radioml")
NEVER = ("t2_or1200",)
BRIEFS = ("T0", "A0", "T", "A", "W2R", "L2R", "P0")
THREE_WORKER = ("W2R", "L2R", "P0")
SEEDS = {"r1": 0, "r2": 1, "r3": 2}
# LLM rows: the three of the first evening plus the template v2.2 variants of the development run's grid (revision 2, V2 to V8; V0 is F1-R1 and
# V1 is v2.1 and run as those rows). A variant row runs orch.policies.v22_policy.V22Policy with the variant's switches.
V22_ROWS = ("V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11")      # V9 and V10: tier 3 of the development grid (Sat 14 Nov); V11: ablation row (V9 with P on)
LLM_APPROACHES = ("F1-R1", "v21", "GA-R1") + V22_ROWS
TEMPERATURE_LLM = 0.3
QUEUE_CAPS = {"pf": {"new_real_runs": 80, "job_hours": 40, "tokens": 20_000_000, "usd": 8.0}}      # decision of Mon 16 Nov: 80 runs, 40 job-hours, US$8 for Gemini and ChatGPT together
QWEN_CAPS = {"new_real_runs": 1_000_000, "job_hours": 1_000_000, "tokens": 1_500_000, "usd": 1_000_000.0}      # Qwen's own token ledger: only its 1.5 million tokens bind
QWEN_LEDGER_ROOT = BASE / _QROOT / "qwen_tokens"
CAPS = ({"new_real_runs": 0, "job_hours": 1, "tokens": 2_000_000, "usd": 0.5} if REPLAY else QUEUE_CAPS[QUEUE])
EXTRACTION_JOBS = 0 if REPLAY else 400
LOAD_GUARD = 160.0
MAX_TOOL_JOBS = 24
OVERRIDE_JOBS = 8
STALL_OVERRIDE_MIN = 15
GEMINI = "gemini-3.1-flash-lite"
PRICE_GEMINI = {"input_per_million": 0.25, "output_per_million": 1.50}
NOISE_TEST = PROJECT / "configs/noise_round2_test.yaml"
NOISE_BASE = PROJECT / "configs/noise_round2.yaml"
NOISE_TRAINPOWER = PROJECT / "configs/noise_round2_trainpower.yaml"     # power review: the 5 TRAINING + 3 VALIDATION designs, all three bands
V21_DEFINITION = PROJECT / "configs/policies/template_v2.1.yaml"
FIXED_ORDERS = PROJECT / "notes/round2_fixed_orders.json"       # declared before launch: the best fixed order per brief, TRAINING only
GRID_PRIORS = {"ab": PROJECT / "data/beam_2026-11-13/priors_ab.json", "abc": PROJECT / "data/beam_2026-11-13/priors_abc.json"}
_DEV_TELEMETRY = [PROJECT / "data/d6v2_2026-11-12/telemetry_cache", PROJECT / "data/development_designs_2026-11-12/telemetry_cache",
                  PROJECT / "data/feature_ablation_2026-11-12/shared/live_telemetry", PROJECT / "data/bandit_2026-11-11/dev/telemetry_cache"]
_TEST_TELEMETRY = [PROJECT / "data/campaign_test_final/telemetry_cache", PROJECT / "data/telemetry_live/test_starts_2026-11-07"]
# the telemetry caches of earlier TEST queues are read-only seeds for the later ones (their states are in the later manifests too)
READ_ONLY_TELEMETRY = (_TEST_TELEMETRY + _DEV_TELEMETRY + [BASE / "sealed/telemetry_cache", BASE / "power_review/telemetry_cache"]
                       + [BASE / f"sealed/{q}/telemetry_cache" for q in ("llm1", "llm3", "llm4", "llm5", "llm6", "llm7")])


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
    if design in NEVER:
        raise SystemExit(f"refused: {design}")
    if DRY:
        allowed, what = DEVELOPMENT, "VALIDATION (smoke test)"
    else:
        allowed, what = TEST + TRAINING + DEVELOPMENT, "TEST, TRAINING or VALIDATION (post-freeze rows)"
    if design not in allowed:
        raise SystemExit(f"refused: {design} is not a {what} design for this run folder")
    return design


def scope_of(design: str) -> str:
    design = guard_design(design)
    return "dev" if design in DEVELOPMENT else "train" if design in TRAINING else "test"


def group_of(design: str) -> str:
    design = guard_design(design)
    return "VALIDATION" if design in DEVELOPMENT else "TRAINING" if design in TRAINING else "TEST"


def plan_sha() -> str:
    import hashlib

    return hashlib.sha256(PLAN_NOTE.read_bytes()).hexdigest()


def episode_id(design: str, brief: str, approach: str, tag: str, model: str | None = None, budget: int | None = None) -> str:
    m = "" if not model or model == GEMINI else "_" + model.replace(":", "-").replace("/", "-")
    b = "" if budget in (None, 6) else f"_b{int(budget)}"            # budget sensitivity rows (3 or 10); the tag stays r1.. so seeds still resolve
    return f"tr_{design}_{brief}_{approach}{m}{b}_{tag}"


# ---------------------------------------------------------------------------------------------------- load rule
def loadavg() -> float:
    return float(Path("/proc/loadavg").read_text().split()[0])


_ov = {"read_at": 0.0, "active": False}


def override_active() -> bool:
    now = time.time()
    if now - _ov["read_at"] > 5.0:
        _ov["read_at"] = now
        try:
            _ov["active"] = bool(jread(OVERRIDE).get("active"))
        except (OSError, ValueError):
            _ov["active"] = False
    return _ov["active"]


def _try_token() -> int | None:
    """One of OVERRIDE_JOBS private flock tokens (held until this process releases it)."""

    import fcntl

    TOKENS.mkdir(parents=True, exist_ok=True)
    for i in range(OVERRIDE_JOBS):
        fd = os.open(str(TOKENS / f"tok{i}"), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            os.close(fd)
    return None


def install_load_rule() -> None:
    """Runner load ceiling = 160 normally. While the queue's stall override is active (admission stalled for more than 30 minutes) a tool
    job may start regardless of load, but only while it holds one of 8 private tokens: at most 8 jobs of this driver run past the guard.
    Patched on the class in this process only; no frozen file is edited."""

    from orch.engine import runner as R

    if getattr(R.Runner, "_tr_patched", False):
        return

    def ceiling(self):
        if not override_active():
            return LOAD_GUARD
        if getattr(self, "_tr_token", None) is None:
            self._tr_token = _try_token()
        return float("inf") if self._tr_token is not None else LOAD_GUARD

    R.Runner.load_ceiling_1m = property(ceiling, lambda self, value: None)
    original_run = R.Runner.run

    def run(self, spec, *, run_id, run_dir):
        try:
            return original_run(self, spec, run_id=run_id, run_dir=run_dir)
        finally:
            tok = getattr(self, "_tr_token", None)
            if tok is not None:
                with suppress(OSError):
                    os.close(tok)
                self._tr_token = None

    R.Runner.run = run
    R.Runner._tr_patched = True


install_load_rule()


# ---------------------------------------------------------------------------------------------------- noise and briefs
_NOISE: dict[str, Any] = {}


def noise_for(design: str, start_evidence) -> dict:
    from orch.config import load_noise

    path = NOISE_TEST if design in TEST else NOISE_TRAINPOWER
    if str(path) not in _NOISE:
        _NOISE[str(path)] = load_noise(path)
    return _NOISE[str(path)].for_start(guard_design(design), start_evidence)


def power_band_tp(design: str) -> float:
    """dynamic_power_w band in W of a TRAINING or VALIDATION design, from configs/noise_round2_trainpower.yaml (never invented)."""

    from orch.config import load_noise
    from orch.contract.models import MetricName as M

    bands = load_noise(NOISE_TRAINPOWER).bands.get(design, {})
    if M.DYNAMIC_POWER_W not in bands:
        raise SystemExit(f"refused: no dynamic_power_w band for {design} in {NOISE_TRAINPOWER.name}")
    band = bands[M.DYNAMIC_POWER_W]
    if band.relative:
        raise SystemExit("refused: a relative power band is not supported here")
    return float(band.value)


def brief_for(design: str, brief_id: str, budget: int | None = None):
    """The round-2 brief, with its recipe-run budget replaced when `budget` is given (queue 5 block 4, budget sensitivity, decision of Sat 14 Nov
    20:30). The replacement is a copy of the frozen brief with only `budget_runs` changed (same limits, objective and validity terms); the
    frozen loader is not edited. The default (None, or the spec's own number) returns the frozen brief unchanged."""

    base = _brief_for(design, brief_id)
    if budget is None or int(budget) == base.budget_runs:
        return base
    if int(budget) not in (3, 10):
        raise ValueError(f"budget {budget}: only the declared sensitivity budgets 3 and 10 are allowed")
    return base.model_copy(update={"budget_runs": int(budget)})


def _brief_for(design: str, brief_id: str):
    """Round-2 brief. TEST designs: orch.contract.briefs_round2 (frozen loader). TRAINING and VALIDATION designs: the frozen briefs come
    from the same loader (design-independent); W2R and L2R are built exactly as that loader builds them, with the power band of the
    trainpower noise file (the frozen loader reads only the TEST and round-2 noise files)."""

    import orch.contract.briefs_round2 as br
    from orch.contract.models import MetricName as M

    design = guard_design(design)
    if design in TEST or brief_id in br.FROZEN_IDS:
        return br.brief_for(design, brief_id)
    if brief_id not in br.BANDED:
        raise ValueError(f"not a round-2 brief: {brief_id}")
    budget = int(br.load_spec()["common"]["budget"]["recipe_runs"])
    band = power_band_tp(design)
    if brief_id == "L2R":
        return br._brief("L2R", M.PACKED_LUTS, "min", [br._term("L2R", M.WNS_NS, ">="), br._term("L2R", M.DYNAMIC_POWER_W, "<=", band)], budget)
    return br._brief("W2R", M.WNS_NS, "max", [br._term("W2R", M.PACKED_LUTS, "<="), br._term("W2R", M.DYNAMIC_POWER_W, "<=", band)], budget)


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
        budget=ExtractionBudget(ROOT / "extraction_budget.json", EXTRACTION_JOBS),
        log_path=ROOT / "extraction.jsonl")
    return h.AccountedTelemetry(inner, caps, ROOT)


def build(scope_name: str):
    """Campaign-harness Context for 'test' (real runs against the shared cap ledger) or 'dev' (dry run: replay, no tool job)."""

    import campaign_harness as h
    import thursday_night_pilot as pilot

    pilot.LOAD_CEILING = LOAD_GUARD
    caps = h.CapLedger(ROOT, CAPS, plan_sha())
    h.ReadThroughLedger = _charging_ledger_class()
    if scope_name == "test":
        if DRY:
            raise SystemExit("refused: a dry run never opens the TEST designs")
        manifest, designs, work = _manifest(MANIFEST_TEST), TEST, TEST_WORK
    elif scope_name == "dev":
        manifest, designs, work = _manifest(MANIFEST_DEV), DEVELOPMENT, DEV_WORK
    elif scope_name == "train":
        if QUEUE != "pf":
            raise SystemExit("refused: the TRAINING scope is opened only by the post-freeze queue")
        manifest, designs, work = _manifest(MANIFEST_TRAIN), TRAINING, TRAIN_WORK
    else:
        raise ValueError(scope_name)
    starts = h._starts_from_manifest(manifest)
    if set(starts) != set(designs):
        raise SystemExit(f"manifest {scope_name} does not hold exactly its {len(designs)} starts")
    if scope_name == "train":
        roots = list(manifest.roots.values())
        provider_for = lambda store: h.ScopeTelemetryProvider(h.V4, store, roots)  # noqa: E731 - stored v4 telemetry, as D6-v2
    else:
        provider_for = lambda store: live_telemetry(store, caps)  # noqa: E731
    scope = h.Scope(scope_name, designs, manifest, work, starts, noise_for, provider_for)
    ctx = h.build_context(scope, mode="replay" if REPLAY else "run", caps=caps)
    ctx.ledger.caps = caps
    return ctx


# ---------------------------------------------------------------------------------------------------- descriptions
class DescribeSource:
    """state id -> dcp-telemetry-v1 record for the physical description (template v2.1, P3): this run's telemetry cache first, then a
    live extraction through the accounted provider; anything else raises DescriptionUnavailableError (infrastructure, never a decision)."""

    def __init__(self, ctx, live) -> None:
        self.ctx, self.live = ctx, live
        self.memo: dict[str, dict] = {}
        self._overlay_index: dict[str, list[str]] | None = None
        self._manifest_index: dict[str, list[tuple[str, str]]] | None = None
        self.served = {"v3": 0, "cache": 0, "live": 0}

    def _v3(self, state_id: str):
        """The pinned v3 TRAINING extraction (free), as in D6-v2: only states of the five TRAINING designs have one."""

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
    """Copy features.json records made by the current extractor from the read-only caches (sources only read; no job-hours charged)."""

    from orch.engine.dcp_parser import DcpTelemetryExtractor
    from orch.engine.runner import Runner

    current = DcpTelemetryExtractor(runner=Runner(PROJECT / "data/slots", max_slots=1),
                                    env_script=PROJECT / "recipes/support/env.sh", tool_version="2025.1",
                                    meter=None, timeout_s=1800).extractor_sha256
    counts = {"copied": 0, "present": 0, "other_extractor": 0, "unreadable": 0}
    for root in READ_ONLY_TELEMETRY:
        for path in sorted(Path(root).glob("*/*/features.json")) + sorted(Path(root).glob("*/*/*/features.json")):
            state_id = path.parent.name
            if len(state_id) != 64:
                continue
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


# ---------------------------------------------------------------------------------------------------- models (plug-ins)
def llm_settings() -> dict:
    import yaml

    return yaml.safe_load((PROJECT / "configs/policies/llm.yaml").read_text(encoding="utf-8"))


def model_spec(model: str | None) -> dict:
    """{'name', 'script', 'price'}: 'gemini-3.1-flash-lite' (the default) or 'gateway:<exact name from the gateway model list>'."""

    cfg = llm_settings()
    model = model or GEMINI
    if model.startswith("gateway:"):
        prices = jread(PRICES) if PRICES.is_file() else {}
        if model not in prices:
            raise SystemExit(f"refused: no price recorded for {model} in {PRICES.name} (the US$ cap needs one)")
        script = Path(__file__).resolve().parent / "call_gateway_qwen.py" if model.startswith("gateway:Qwen/") else PROJECT / "tools/llm_bridge/call_gateway.py"
        return {"name": model[4:], "script": script, "price": prices[model], "label": model}
    if model != GEMINI:
        raise SystemExit(f"refused: unknown model plug-in {model}")
    return {"name": GEMINI, "script": PROJECT / cfg["bridge"]["script"], "price": PRICE_GEMINI, "label": GEMINI}


def llm_caps(task: dict, caps):
    """Qwen calls are counted in their own token ledger (1.5 million tokens, smoke test included); every other model in the shared ledger."""

    import campaign_harness as h

    if str(task.get("model") or "").startswith("gateway:Qwen/"):
        return h.CapLedger(QWEN_LEDGER_ROOT, QWEN_CAPS, plan_sha())
    return caps


def make_llm(task: dict, caps):
    import campaign_harness as h
    from orch.policies.llm_client import BridgeTransport, CachedLLM, LLMParams, RateLimiter, ReplayCache

    cfg = llm_settings()
    spec = model_spec(task.get("model"))
    transport = BridgeTransport(PROJECT / cfg["bridge"]["python"], spec["script"],
                                rate_limiter=RateLimiter(ROOT / "ratelimit", cfg["rate_limit_per_minute"]))
    label = re.sub(r"[^A-Za-z0-9_.-]", "-", spec["label"])        # ReplayCache allows only [A-Za-z0-9_.-] in a fresh tag ('gateway:x' had ':')
    cache = ReplayCache(LLM_CACHE, mode="fresh", fresh_tag=f"tr-{task['approach']}-{label}-{task['tag']}")
    temperature = task.get("temperature")
    params = LLMParams(model=spec["name"], temperature=TEMPERATURE_LLM if temperature is None else temperature,
                       max_output_tokens=cfg["max_output_tokens"], seed=SEEDS.get(task["tag"], 0),
                       thinking_budget=cfg["thinking_budget"], timeout_ms=cfg["timeout_ms"])
    return h.LedgeredLLM(CachedLLM(transport, cache, params), llm_caps(task, caps), LLM_CALLS, task["episode_id"], spec["price"])


# ---------------------------------------------------------------------------------------------------- readers
def read_journal() -> dict[str, dict]:
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
    path = PLAN / "tasks.json"
    return jread(path) if path.is_file() else []
