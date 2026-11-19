"""Saturday LLM campaigns on the training designs (spec section 10): orchestrator configurations, F2
folds, variance repeats, the general agent and the unseen-worker smoke.

Each stage is one campaign with its own directory and a frozen, hashed work file (a list of cells),
so a prompt revision or a new configuration is a new stage. All stages share
  * the 400 new-real-run cap (data/pilots/saturday_llm_shared/new_real_runs.json),
  * the replay cache of model replies, the rate limiter and the telemetry log (same directory).
The engine root is the sealed v4 training cache used in place, additions only, exactly as in the
Friday sweep and P0 campaigns; every output is checked by the verified episode engine (clean
properties and an AMD simulation, run on demand for outputs never simulated before).

The model is called through the bridge in tools/llm_bridge (its own venv); the key is only ever in
the environment of the bridge process. Training designs only; no test-design path is opened.

Pinned (shards stop fail-closed if any changes): engine, workers, meter, registry and fitted models
as in the earlier campaigns, plus the LLM policy modules, template, profiles, lessons, configuration,
bridge files and this stage's work file. This driver itself is hashed into the launch manifest but is
not pinned, so it can be extended while a stage runs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

import thursday_overnight_power as campaign  # noqa: E402

from orch.analysis.lessons import lessons_block, load_lessons  # noqa: E402
from orch.baselines.verification import VerificationLedger  # noqa: E402
from orch.config import load_noise, load_recipe_sets, load_worker_card  # noqa: E402
from orch.contract.brief_text import load_frozen_brief  # noqa: E402
from orch.contract.models import MetricName  # noqa: E402
from orch.engine.apply import StepEngine  # noqa: E402
from orch.engine.runner import Runner  # noqa: E402
from orch.engine.store import _atomic_bytes, _file_lock, _json_bytes, _sha256_file  # noqa: E402
from orch.episode import run_episode  # noqa: E402
from orch.policies.llm import LLMPolicy, LLMPolicyConfig  # noqa: E402
from orch.policies.llm_client import (  # noqa: E402
    BridgeTransport, CachedLLM, LLMParams, LLMUnavailableError, RateLimiter, ReplayCache, TelemetrySink,
)
from orch.policies.llm_ledger import LedgeredTransport, Price, SpendCapReached, SpendLedger  # noqa: E402
from orch.policies.llm_prompt import load_template  # noqa: E402
from orch.runtime import make_real_runtime  # noqa: E402
from orch.workers.base import GenericWorker  # noqa: E402
from orch.workers.fitted import FittedWorker  # noqa: E402
from orch.workers.telemetry import SavedTelemetryProvider  # noqa: E402
from scripts import friday_sweep as sweep  # noqa: E402
from scripts import thursday_night_pilot as pilot  # noqa: E402

SCRIPT = Path(__file__).resolve()
BASE = PROJECT / "data/pilots"
# Saturday-night stages (the stronger-model test) use their own shared directory, run cap and spend cap:
#   SATURDAY_LLM_SHARED     directory name under data/pilots (default saturday_llm_shared)
#   SATURDAY_LLM_RUN_CAP    new real recipe runs allowed across the stages sharing that directory (default 400)
#   SATURDAY_LLM_USD_CAP    model spend cap in US dollars (unset: no spend ledger), with
#   SATURDAY_LLM_PRICE_IN / SATURDAY_LLM_PRICE_OUT  US$ per million input / output tokens (thinking is output)
#   SATURDAY_LLM_USD_HEADROOM  reserved below the cap for calls already in flight (default 1.5)
# They are recorded in each stage plan, so a stage cannot be resumed under different values.
SHARED = BASE / (os.environ.get("SATURDAY_LLM_SHARED") or "saturday_llm_shared")
ENGINE_ROOT = sweep.ENGINE_ROOT
SOURCE_ROOT = sweep.SOURCE_ROOT
CONFIG = Path(os.environ.get("SATURDAY_LLM_CONFIG") or "configs/policies/llm.yaml")
if not CONFIG.is_absolute():
    CONFIG = PROJECT / CONFIG
RANKING = PROJECT / "data/scratch_friday_2026-11-06/worker_c_ranking.json"
CAP = int(os.environ.get("SATURDAY_LLM_RUN_CAP") or 400)
SHARED_PLAN_ID = "saturday-llm-shared-v1"
DATE = "2026-11-06"
DESIGNS = tuple(pilot.DESIGNS)
CRITERION_BRIEFS = ("A0", "T0")
ALL_BRIEFS = ("T0", "A0", "T", "A")
_hash = _sha256_file
_BRIEFS: dict[str, Any] = {}
_BOUND = False
STAGE: str | None = None
ROOT: Path = BASE / "saturday_llm_unset"
SHARD_COUNT = 12


@dataclass(frozen=True)
class Cell:
    config: str          # F0 F1 E0 E1 F2 or GA (general agent)
    design: str
    brief: str
    tag: str             # r0 for the first record-mode run, v1 v2 ... for fresh repeats
    cache: str           # record | fresh
    fold: str | None     # held-out design whose lessons are excluded (F2), else None
    kind: str            # orchestrator | general
    workers: str         # ab | abc

    def episode_id(self, stage: str) -> str:
        fold = f"_f-{self.fold}" if self.fold else ""
        return f"llm_{stage}_{self.config}{fold}_{self.design}_{self.brief}_{self.tag}"


# ------------------------------------------------------------------ stage, work and configuration
def set_stage(stage: str) -> None:
    global STAGE, ROOT
    if not re.fullmatch(r"[a-z0-9]+", stage):
        raise ValueError("a stage name is lowercase letters and digits")
    STAGE, ROOT = stage, BASE / f"saturday_llm_{stage}_{DATE}"


def settings() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def spend_settings() -> dict[str, Any] | None:
    """The model-spend ledger of this stage family from the environment (see the header), or None."""

    cap = os.environ.get("SATURDAY_LLM_USD_CAP")
    if not cap:
        return None
    try:
        return {
            "cap_usd": float(cap), "headroom_usd": float(os.environ.get("SATURDAY_LLM_USD_HEADROOM") or 1.5),
            "price_in_per_million": float(os.environ["SATURDAY_LLM_PRICE_IN"]),
            "price_out_per_million": float(os.environ["SATURDAY_LLM_PRICE_OUT"]),
        }
    except KeyError as exc:
        raise pilot.PilotSafetyStop("a spend cap needs SATURDAY_LLM_PRICE_IN and SATURDAY_LLM_PRICE_OUT") from exc


def spend_ledger() -> SpendLedger | None:
    spend = spend_settings()
    if spend is None:
        return None
    return SpendLedger(
        SHARED / "spend.jsonl", cap_usd=spend["cap_usd"], headroom_usd=spend["headroom_usd"],
        price=Price(spend["price_in_per_million"], spend["price_out_per_million"]),
    )


def work_path() -> Path:
    return ROOT / "work.json"


def rows() -> list[Cell]:
    data = json.loads(work_path().read_text(encoding="utf-8"))
    if data.get("stage") != STAGE:
        raise ValueError("work file belongs to another stage")
    return [Cell(**item) for item in data["cells"]]


def make_work(stage: str, configs: list[str], *, tag: str, cache: str, fold_mode: str, kind: str,
              workers: str, briefs: list[str], designs: list[str]) -> list[Cell]:
    defined = settings()["configurations"]
    for config in configs:
        if defined.get(config, {}).get("dropped"):
            raise ValueError(f"configuration {config} is dropped (decision of 2026-11-07) and cannot be planned")
    cells: list[Cell] = []
    for config in configs:
        if fold_mode == "lodo":                     # every (held-out fold, design) pair on A0/T0, own fold on T/A
            for fold in designs:
                for design in designs:
                    for brief in briefs:
                        if brief in CRITERION_BRIEFS or design == fold:
                            cells.append(Cell(config, design, brief, tag, cache, fold, kind, workers))
            continue
        for design in designs:
            for brief in briefs:
                fold = design if fold_mode == "own" else None
                cells.append(Cell(config, design, brief, tag, cache, fold, kind, workers))
    # run own-fold cells first when folds are generated, so the headline table fills before the selection folds
    cells.sort(key=lambda c: (configs.index(c.config), 0 if c.fold in (None, c.design) else 1))
    if len({c.episode_id(stage) for c in cells}) != len(cells):
        raise ValueError("duplicate cells")
    return cells


def _brief(brief_id: str):
    if brief_id not in _BRIEFS:
        _BRIEFS[brief_id] = load_frozen_brief(brief_id)
    return _BRIEFS[brief_id]


def _write_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, _json_bytes(value))


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with _file_lock(path.with_suffix(path.suffix + ".lock")):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")


# ------------------------------------------------------------------ pins and plan
def source_files(cells: list[Cell]) -> dict[str, str]:
    files = {k: v for k, v in sweep.SOURCE_FILES.items() if k not in ("sweep_driver", "repair_loop")}
    files.update({
        "pilot_script": "scripts/thursday_night_pilot.py", "brief_text": "orch/contract/brief_text.py",
        "briefs": "configs/briefs.yaml", "noise": "configs/noise.yaml",
        "recipe_sets": "configs/recipe_sets.yaml",
        "start_properties": "orch/engine/start_properties.py", "llm_ledger": "orch/policies/llm_ledger.py",
        "llm": "orch/policies/llm.py", "llm_client": "orch/policies/llm_client.py",
        "llm_prompt": "orch/policies/llm_prompt.py", "llm_schemas": "orch/policies/llm_schemas.py",
        "llm_view": "orch/policies/llm_view.py", "lessons_module": "orch/analysis/lessons.py",
        "llm_yaml": str(CONFIG.relative_to(PROJECT)), "bridge_call": "tools/llm_bridge/call.py",
        "bridge_lock": "tools/llm_bridge/requirements.lock", "template": settings()["templates"]["orchestrator"],
        "profile_manifest": "configs/profiles/manifest.json", "profile_timing": "configs/profiles/timing.yaml",
        "profile_area": "configs/profiles/area.yaml", "profile_power": "configs/profiles/power.yaml",
        "profile_recipes": "configs/profiles/recipes.yaml",
        "work": str(work_path().relative_to(PROJECT)),
    })
    if any(c.config == "F2" for c in cells):
        files["lessons_file"] = settings()["lessons"]
    if any(c.kind == "general" for c in cells):
        files.update({"general_agent": "orch/baselines/general_agent.py",
                      "direct_worker": "orch/baselines/direct_recipe_worker.py",
                      "ga_template": settings()["templates"]["general_agent"]})
    if any(c.workers == "abc" for c in cells):
        files.update({"power_card": "configs/workers/power.yaml", "power_set": "configs/power_recipe_set.yaml",
                      "power_ranking": str(RANKING.relative_to(PROJECT)), "relevant_workers": "orch/policies/relevant_workers.py"})
    return files


def _make_plan(runtime) -> dict[str, Any]:
    cells = rows()
    sets = load_recipe_sets(PROJECT / "configs/recipe_sets.yaml")
    sets.require_signed_off()
    references = pilot._training_inputs()
    source_plan = pilot.read(SOURCE_ROOT / "plan.json")
    briefs = {b: _brief(b).fingerprint() for b in sorted({c.brief for c in cells})}
    for b, fp in briefs.items():
        if b in source_plan["brief_hashes"] and fp != source_plan["brief_hashes"][b]:
            raise pilot.PilotSafetyStop(f"brief {b} differs from the pilot's frozen brief")
    cfg = settings()
    return {
        "schema_version": "saturday-llm-plan-v1", "stage": STAGE, "training_only": True,
        "bookkeeping_root": str(ROOT), "shared_root": str(SHARED), "engine_root": str(ENGINE_ROOT),
        "engine_root_plan_sha256": _hash(SOURCE_ROOT / "plan.json"),
        "training_inputs": references,
        "training_launch_manifest_sha256": _hash(pilot.TRAINING_ROOT / "launch_manifest.json"),
        "current_b2_sources": pilot._start_source_hashes(runtime.meter.version, references),
        "cells": [asdict(c) for c in cells], "brief_hashes": briefs,
        "worker_sets": {"timing": list(sets.worker_sets["timing"]), "area": list(sets.worker_sets["area"])},
        "recipe_versions": {rid: runtime.registry.get(rid).version for ids in (sets.worker_sets["timing"], sets.worker_sets["area"]) for rid in ids},
        "meter_version": runtime.meter.version, "smoke_sha256": _hash(pilot.SMOKE),
        "model": cfg["model"], "configurations": cfg["configurations"], **({"config_path": str(CONFIG.relative_to(PROJECT))} if os.environ.get("SATURDAY_LLM_CONFIG") else {}),
        **({"thinking_level": cfg.get("thinking_level"), "temperature": cfg.get("temperature"),
            "spend": spend_settings()} if (cfg.get("thinking_level") or spend_settings()) else {}),
        "source_hashes": pilot.source_hashes(), "max_new_real_recipe_runs": CAP,
        "shared_slots": pilot.MAX_SLOTS, "load_ceiling_1m": pilot.LOAD_CEILING,
        "available_memory_floor_bytes": pilot.MEMORY_FLOOR,
    }


def _require_plan(runtime) -> dict[str, Any]:
    frozen = _make_plan(runtime)
    path = ROOT / "plan.json"
    expected = _json_bytes(frozen)
    ROOT.mkdir(parents=True, exist_ok=True)
    with _file_lock(ROOT / "plan.lock"):
        if path.is_file():
            if path.read_bytes() != expected:
                raise pilot.PilotSafetyStop("stage plan differs from frozen source, input or work file")
        else:
            _atomic_bytes(path, expected)
    return frozen


def _counter() -> dict[str, Any]:
    path = SHARED / "new_real_runs.json"
    return pilot.read(path) if path.is_file() else {"reservations": []}


def _stage_reservations() -> list[dict[str, Any]]:
    return [r for r in _counter()["reservations"] if r["episode_id"].startswith(f"llm_{STAGE}_")]


def _status(phase: str, *, error: str | None = None) -> dict[str, Any]:
    done = [pilot.read(p) for p in (ROOT / "results").glob("*.json")] if (ROOT / "results").is_dir() else []
    value = {
        "schema_version": "saturday-llm-status-v1", "stage": STAGE, "phase": phase,
        "updated_at_utc": pilot.utc(), "planned_cells": len(rows()), "completed_cells": len(done),
        "stage_new_real_runs": len(_stage_reservations()), "all_stages_new_real_runs": len(_counter()["reservations"]),
        "cap": CAP, "cache_hits": sum(r["cache_hits"] for r in done),
        "recipe_runs_used": sum(r["runs_used"] for r in done), "error": error,
    }
    ledger = spend_ledger()
    if ledger is not None:
        value["model_spend"] = ledger.summary()
    _write_json(ROOT / "status.json", value)
    return value


# ------------------------------------------------------------------ cleanup of this stage's tool jobs only
def _targets() -> tuple[set[str], set[int]]:
    mine = {r["run_id"] for r in _stage_reservations()}
    prefix = str(ENGINE_ROOT / "runs").encode() + b"/"
    tags: set[str] = set()
    groups: set[int] = set()
    for pid in pilot._own_pids():
        info = pilot._stat(pid)
        if info is None or info.state == "Z":
            continue
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd").encode()
        except OSError:
            continue
        if not cwd.startswith(prefix):
            continue
        run_id = cwd[len(prefix):].split(b"/", 1)[0].decode()
        tag = pilot._tag_of(pid)
        if run_id in mine and tag and re.fullmatch(r"[0-9a-f]{32}", tag):
            tags.add(tag)
            groups.add(info.pgrp)
    groups.discard(os.getpgrp())
    return tags, groups


def _cleanup() -> dict[str, Any]:
    tags, groups = _targets()
    result: dict[str, Any] = {"schema_version": "saturday-llm-cleanup-v1", "exact_tags": sorted(tags),
                              "groups": sorted(groups), "started_at_utc": pilot.utc()}
    _write_json(ROOT / "watchdog_cleanup_audit.json", result)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for group in sorted(groups):
            if any(pilot._tag_of(m.pid) in tags for m in pilot.group_members(group)):
                try:
                    os.killpg(group, sig)
                except ProcessLookupError:
                    pass
        if sig == signal.SIGTERM:
            time.sleep(3)
    result["tag_sweeps"] = {t: list(pilot.sweep_tag(t, grace_s=5).alive) for t in sorted(tags)}
    result["survivors"] = {t: [p.pid for p in pilot.tagged(t)] for t in sorted(tags) if pilot.tagged(t)}
    result["finished_at_utc"] = pilot.utc()
    _write_json(ROOT / "watchdog_cleanup_audit.json", result)
    return result


def _bind() -> None:
    global _BOUND
    if _BOUND:
        return
    for brief_id in sorted({c.brief for c in rows()}):
        _brief(brief_id)
    if pilot.read(SOURCE_ROOT / "status.json").get("phase") != "complete":
        raise pilot.PilotSafetyStop("the approved cache root is not sealed complete")
    pilot.ROOT = ENGINE_ROOT
    pilot.CAP = CAP
    pilot.SOURCE_FILES = source_files(rows())
    pilot.make_plan = _make_plan
    pilot.require_plan = _require_plan
    pilot._status = _status
    pilot.cleanup = _cleanup
    pilot.episode_schedule = lambda: tuple(rows())
    _BOUND = True


# ------------------------------------------------------------------ workers, policies, noise
_POWER_BANDS: dict[str, float] = {}


def noise_for(design: str, start_evidence) -> dict[MetricName, float]:
    noise = load_noise(PROJECT / "configs/noise.yaml").for_start(design, start_evidence)
    if not _POWER_BANDS:
        for d, item in campaign.selected()["by_design"].items():
            _POWER_BANDS[d] = item["noise_power_pct"]
    power = start_evidence.metric(MetricName.DYNAMIC_POWER_W)
    if design in _POWER_BANDS and power is not None:
        noise[MetricName.DYNAMIC_POWER_W] = _POWER_BANDS[design] / 100.0 * power.value
    return noise


def make_workers(runtime, cell: Cell) -> dict[str, Any]:
    workers: dict[str, Any] = dict(pilot._workers(runtime))
    if cell.workers == "abc":
        power = load_recipe_sets(PROJECT / "configs/power_recipe_set.yaml")
        card = load_worker_card(PROJECT / "configs/workers/power.yaml")
        if card.id != "power" or list(power.worker_sets["power"]) != json.loads(RANKING.read_text())["ranking"]:
            raise pilot.PilotSafetyStop("power worker card or order differs from worker C's ranking")
        workers["power"] = GenericWorker(card, power.worker_sets["power"])
    if cell.kind == "general":
        from orch.baselines.direct_recipe_worker import DirectRecipeWorker

        workers = {wid: DirectRecipeWorker(w.card, w.recipe_ids) for wid, w in workers.items()}
    return workers


def profile_material(worker_ids: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    cfg = settings()
    manifest = json.loads((PROJECT / cfg["profiles_dir"] / "manifest.json").read_text(encoding="utf-8"))
    texts, hashes = {}, {}
    for worker in worker_ids:
        data = yaml.safe_load((PROJECT / cfg["profiles_dir"] / f"{worker}.yaml").read_text(encoding="utf-8"))
        texts[worker], hashes[worker] = data["text"], manifest["files"][f"{worker}.yaml"]
    return texts, hashes


def make_llm(cell: Cell) -> CachedLLM:
    cfg = settings()
    transport: Any = BridgeTransport(
        PROJECT / cfg["bridge"]["python"], PROJECT / cfg["bridge"]["script"],
        rate_limiter=RateLimiter(SHARED / "ratelimit", cfg["rate_limit_per_minute"]),
    )
    ledger = spend_ledger()
    if ledger is not None:            # a call is refused before it starts once the spend reaches its stop
        transport = LedgeredTransport(transport, ledger, source=cell.episode_id(STAGE))
    cache = ReplayCache(SHARED / "llm_cache", mode=cell.cache, fresh_tag=cell.tag if cell.cache == "fresh" else None)
    params = LLMParams(
        model=cfg["model"], temperature=cfg["temperature"], max_output_tokens=cfg["max_output_tokens"],
        seed=cfg["seed"], thinking_budget=cfg["thinking_budget"], timeout_ms=cfg["timeout_ms"],
        thinking_level=cfg.get("thinking_level"),
    )
    return CachedLLM(transport, cache, params)


def make_policy(cell: Cell, worker_ids: list[str]):
    cfg = settings()
    llm = make_llm(cell)
    sink = TelemetrySink(SHARED / "telemetry.jsonl")
    if cell.kind == "general":
        from orch.baselines.general_agent import build_general_agent

        return build_general_agent(PROJECT, cfg, llm, sink, worker_ids)
    conf = cfg["configurations"][cell.config]
    texts, hashes = profile_material(worker_ids) if conf["profiles"] else ({}, {})
    lessons = None
    if conf["lessons"]:
        lessons = lessons_block(load_lessons(PROJECT / cfg["lessons"]), exclude_design=cell.fold)
    return LLMPolicy(
        LLMPolicyConfig(
            name=cell.config, mode=conf["mode"], profiles_on=conf["profiles"],
            template=load_template(PROJECT / cfg["templates"]["orchestrator"]),
            profile_texts=texts, profile_sha256=hashes, lessons=lessons, max_repairs=cfg["max_repair_turns"],
        ), llm, sink,
    )


# ------------------------------------------------------------------ one cell
@dataclass
class _Context:
    runtime: Any
    starts: dict[str, Any]
    provider: SavedTelemetryProvider
    ledger: VerificationLedger
    backend: pilot.CountedBackend
    engine: pilot.VerifiedEpisodeEngine


def _runtime():
    return make_real_runtime(
        ENGINE_ROOT, cache_enabled=True, max_slots=pilot.MAX_SLOTS, load_ceiling_1m=pilot.LOAD_CEILING,
        min_available_memory_bytes=pilot.MEMORY_FLOOR, latch_smoke_status=pilot.SMOKE,
    )


def _context() -> _Context:
    runtime = _runtime()
    frozen = _require_plan(runtime)
    starts = pilot._seed_starts(runtime, frozen["training_inputs"])
    ledger = VerificationLedger(store=runtime.store, runner=runtime.runner, validator=runtime.validator,
                                references=frozen["training_inputs"])
    SHARED.mkdir(parents=True, exist_ok=True)
    backend = pilot.CountedBackend(runtime.backend, SHARED, SHARED_PLAN_ID, lambda: _require_plan(runtime), cap=CAP)
    engine = pilot.VerifiedEpisodeEngine(
        StepEngine(store=runtime.store, cache=runtime.cache, backend=backend,
                   meter_version=runtime.meter.version, cache_enabled=True), ledger)
    (ROOT / "results").mkdir(exist_ok=True)
    return _Context(runtime, starts, SavedTelemetryProvider(ENGINE_ROOT), ledger, backend, engine)


def llm_stats(episode: str) -> dict[str, Any]:
    records = [r for r in TelemetrySink.read(SHARED / "telemetry.jsonl") if r["episode_id"] == episode]
    keep: dict[int, dict[str, Any]] = {}
    for record in records:                      # a resumed decision may appear twice; the last record wins
        keep[record["decision_index"]] = record
    chosen = [keep[i] for i in sorted(keep)]
    tokens = {k: sum(r["tokens"][k] for r in records) for k in ("prompt", "completion", "thoughts", "total", "fresh_total")}
    return {
        "decisions": len(chosen), "calls": sum(len(r["calls"]) for r in records),
        "repairs": sum(r["repair_attempts"] for r in chosen), "fallbacks": sum(r["fallback_triggered"] for r in chosen),
        "tokens": tokens, "policy_version": chosen[0]["policy_version"] if chosen else None,
        "choices": [{"strategy": r["strategy"], "action": r["action"], "mandate": r["mandate"]} for r in chosen],
    }


def run_cell(ctx: _Context, cell: Cell) -> str:
    episode = cell.episode_id(STAGE)
    result_path = ROOT / "results" / f"{episode}.json"
    if result_path.is_file():
        saved = pilot.read(result_path)
        if saved.get("episode_id") != episode or saved.get("final_brief_valid") is not True:
            raise pilot.PilotSafetyStop("saved result identity differs")
        pilot._audit_episode(episode, ctx.runtime.store, ctx.ledger)
        return "skipped"
    pilot._audit_episode(episode, ctx.runtime.store, ctx.ledger)
    brief, start = _brief(cell.brief), ctx.starts[cell.design]
    start_evidence = ctx.engine.measure_start(start)
    workers = make_workers(ctx.runtime, cell)
    policy = make_policy(cell, sorted(workers))
    provider = ctx.provider if any(isinstance(w, FittedWorker) for w in workers.values()) else None
    ctx.backend.active_episode = episode
    try:
        result = run_episode(
            engine=ctx.engine, registry=ctx.runtime.registry, workers=workers, policy=policy, brief=brief,
            start_state=start, episode_id=episode, noise=noise_for(cell.design, start_evidence),
            telemetry_provider=provider, validator=pilot._validator(ctx.ledger, brief),
        )
    except (pilot.PilotRunCap, SpendCapReached) as exc:
        _append_jsonl(ROOT / "capped_cells.jsonl", {"episode_id": episode, "reason": str(exc), "at_utc": pilot.utc()})
        return "capped"
    except LLMUnavailableError as exc:
        _append_jsonl(ROOT / "llm_unavailable.jsonl", {"episode_id": episode, "reason": str(exc)[:300], "at_utc": pilot.utc()})
        return "llm_unavailable"
    except pilot.PilotSafetyStop as exc:
        message = str(exc)
        if message.startswith(("AMD simulation did not PASS", "missing DONT_TOUCH", "more DONT_TOUCH")):
            _append_jsonl(ROOT / "invalid_output_cells.jsonl", {"episode_id": episode, "reason": message[:300], "at_utc": pilot.utc()})
            return "invalid_output"
        raise
    finally:
        ctx.backend.active_episode = None
    pilot._audit_episode(episode, ctx.runtime.store, ctx.ledger)
    row = pilot._result_row(result, ctx.runtime.store, ctx.backend, cell.design, cell.brief, cell.config)
    row.update({
        "campaign": "saturday-llm", "stage": STAGE, "cell": asdict(cell), "policy_version": policy.version,
        "llm": llm_stats(episode),
    })
    _atomic_bytes(result_path, _json_bytes(row))
    _status("running")
    return "done"


# ------------------------------------------------------------------ shards, controller, launch
def prepare() -> dict[str, Any]:
    _bind()
    ROOT.mkdir(parents=True, exist_ok=True)
    runtime = _runtime()
    frozen = _require_plan(runtime)
    starts = pilot._seed_starts(runtime, frozen["training_inputs"])
    for cell in rows()[:1]:
        make_workers(runtime, cell)
    SHARED.mkdir(parents=True, exist_ok=True)
    pilot.CountedBackend(runtime.backend, SHARED, SHARED_PLAN_ID, lambda: _require_plan(runtime), cap=CAP)
    result = {"schema_version": "saturday-llm-prepared-v1", "stage": STAGE, "prepared_at_utc": pilot.utc(),
              "plan_sha256": _hash(ROOT / "plan.json"), "meter_version": runtime.meter.version,
              "start_states": len(starts), "cells": len(rows()), "tool_jobs_launched": 0,
              "model": settings()["model"], "pinned_files": len(pilot.source_hashes())}
    _write_json(ROOT / "prepared.json", result)
    return result


def shard_count() -> int:
    """The launch manifest is the single source of the shard count for controller and shards."""

    launch_file = ROOT / "llm_launch.json"
    return int(pilot.read(launch_file)["shards"]) if launch_file.is_file() else SHARD_COUNT


def _write_shard(shard: int, state: str, counts: dict[str, int], planned: int, error: str | None = None) -> None:
    _write_json(ROOT / f"llm_shard_{shard}.json", {"shard": shard, "state": state, "planned": planned,
                                                   "counts": counts, "error": error, "updated_at_utc": pilot.utc()})


def run_shard(shard: int) -> None:
    _bind()
    ctx = _context()
    mine = rows()[shard::shard_count()]
    counts = {"done": 0, "skipped": 0, "capped": 0, "llm_unavailable": 0, "invalid_output": 0}
    _write_shard(shard, "running", counts, len(mine))
    try:
        pending = list(mine)
        for attempt in range(4):
            retry = []
            for cell in pending:
                _require_plan(ctx.runtime)
                outcome = run_cell(ctx, cell)
                if outcome == "llm_unavailable":
                    retry.append(cell)
                else:
                    counts[outcome] += 1
                _write_shard(shard, "running", counts, len(mine))
            pending = retry
            if not pending:
                break
            time.sleep(120 * (attempt + 1))
        counts["llm_unavailable"] = len(pending)
    except BaseException as exc:
        _write_shard(shard, "failed", counts, len(mine), f"{type(exc).__name__}: {exc}")
        raise
    _write_shard(shard, "complete", counts, len(mine))


def _ctl(state: str, **extra: Any) -> None:
    _write_json(ROOT / "llm_control.json", {"state": state, "controller_pid": os.getpid(),
                                           "updated_at_utc": pilot.utc(), **extra})


def _spawn_shard(shard: int) -> subprocess.Popen:
    with (ROOT / f"llm_s{shard}.stdout").open("ab") as out, (ROOT / f"llm_s{shard}.stderr").open("ab") as err:
        return subprocess.Popen(
            [sys.executable, str(SCRIPT), "--stage", STAGE, "--shard", str(shard)], cwd=PROJECT,
            stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True, close_fds=True,
        )


def control() -> None:
    _bind()
    count = shard_count()
    attempts = {s: 1 for s in range(count)}
    live = {s: _spawn_shard(s) for s in attempts}
    final: dict[int, int] = {}
    _ctl("running", pids={str(s): p.pid for s, p in live.items()})
    while live:
        time.sleep(15)
        for shard, proc in list(live.items()):
            code = proc.poll()
            if code is None:
                continue
            del live[shard]
            if code != 0 and attempts[shard] < 3:
                time.sleep(30)
                attempts[shard] += 1
                live[shard] = _spawn_shard(shard)
            else:
                final[shard] = code
        _ctl("running", live=sorted(live), finished={str(s): c for s, c in final.items()})
    _write_json(ROOT / "llm_summary.json", {"exit_codes": {str(s): c for s, c in final.items()},
                                            "attempts": {str(s): a for s, a in attempts.items()}})
    _ctl("complete" if all(c == 0 for c in final.values()) else "complete_with_failures")


def _alive(pid: int, mode: bytes) -> bool:
    info = pilot._stat(pid)
    if info is None or info.state == "Z":
        return False
    try:
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return str(SCRIPT).encode() in args and mode in args


def watch() -> None:
    _bind()
    pid = int(pilot.read(ROOT / "llm_launch.json")["controller_pid"])
    deadline = time.monotonic() + 96 * 3600
    control_file = ROOT / "llm_control.json"
    while time.monotonic() < deadline:
        if not _alive(pid, b"--control"):
            state = pilot.read(control_file) if control_file.is_file() else {}
            terminal = state.get("state") in ("complete", "complete_with_failures")
            tags, _ = _targets()
            if not terminal or tags:
                _write_json(ROOT / "llm_watchdog_cleanup.json", _cleanup())
                if not terminal:
                    _ctl("failed", error="controller exited early; watchdog swept exact tags")
            return
        time.sleep(15)
    _write_json(ROOT / "llm_watchdog_cleanup.json", _cleanup())
    _ctl("failed", error="watchdog lifetime expired")


def launch() -> dict[str, Any]:
    _bind()
    prepared = prepare()
    load, available = Runner._resource_snapshot()
    if load > pilot.LOAD_CEILING or available < pilot.MEMORY_FLOOR:
        raise pilot.PilotSafetyStop("load or available memory blocks the launch")
    if not os.environ.get("GOOGLE_API_KEY"):
        raise pilot.PilotSafetyStop("GOOGLE_API_KEY is not set in this shell (use a login shell)")
    with _file_lock(ROOT / "llm_launch.lock"):
        launch_file = ROOT / "llm_launch.json"
        if launch_file.exists():
            raise FileExistsError("this stage was already launched")
        manifest: dict[str, Any] = {
            "schema_version": "saturday-llm-launch-v1", "stage": STAGE, "started_at_utc": pilot.utc(),
            "plan_sha256": prepared["plan_sha256"], "driver_sha256": _hash(SCRIPT), "cap": CAP,
            "shards": min(SHARD_COUNT, len(rows())), "load_1m_at_launch": load,
            "available_memory_bytes_at_launch": available,
        }
        _write_json(launch_file, manifest)
        for name, mode in (("control", "--control"), ("watchdog", "--watch")):
            with (ROOT / f"llm_{name}.stdout").open("wb") as out, (ROOT / f"llm_{name}.stderr").open("wb") as err:
                proc = subprocess.Popen([sys.executable, str(SCRIPT), "--stage", STAGE, mode], cwd=PROJECT,
                                        stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                        start_new_session=True, close_fds=True)
            manifest[f"{name}_pid"] = proc.pid
            _write_json(launch_file, manifest)
        return manifest


def status() -> dict[str, Any]:
    _bind()
    shards = {p.stem: pilot.read(p) for p in sorted(ROOT.glob("llm_shard_*.json"))}
    results = list((ROOT / "results").glob("*.json")) if (ROOT / "results").is_dir() else []
    control_file = ROOT / "llm_control.json"
    return {
        "stage": STAGE, "control": pilot.read(control_file) if control_file.is_file() else None,
        "planned": len(rows()), "completed_cells": len(results),
        "stage_new_real_runs": len(_stage_reservations()), "all_stages_new_real_runs": len(_counter()["reservations"]),
        "cap": CAP, "shards": {k: {"state": v["state"], "counts": v["counts"], "error": v["error"]} for k, v in shards.items()},
    }


def main() -> int:
    global SHARD_COUNT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "launch", "control", "watch", "status", "summary"):
        modes.add_argument(f"--{name}", action="store_true")
    modes.add_argument("--shard", type=int)
    modes.add_argument("--make-work", action="store_true")
    parser.add_argument("--configs", default="F1,F0,E1,E0")
    parser.add_argument("--tag", default="r0")
    parser.add_argument("--cache", choices=("record", "fresh"), default="record")
    parser.add_argument("--fold-mode", choices=("none", "own", "lodo"), default="none")
    parser.add_argument("--kind", choices=("orchestrator", "general"), default="orchestrator")
    parser.add_argument("--workers", choices=("ab", "abc"), default="ab")
    parser.add_argument("--briefs", default=",".join(ALL_BRIEFS))
    parser.add_argument("--designs", default=",".join(DESIGNS))
    parser.add_argument("--shards", type=int, default=SHARD_COUNT)
    args = parser.parse_args()
    SHARD_COUNT = args.shards
    set_stage(args.stage)
    if args.make_work:
        ROOT.mkdir(parents=True, exist_ok=True)
        if work_path().exists():
            raise FileExistsError("this stage already has a work file")
        cells = make_work(STAGE, args.configs.split(","), tag=args.tag, cache=args.cache, fold_mode=args.fold_mode,
                          kind=args.kind, workers=args.workers, briefs=args.briefs.split(","),
                          designs=args.designs.split(","))
        _write_json(work_path(), {"stage": STAGE, "cells": [asdict(c) for c in cells]})
        print(json.dumps({"stage": STAGE, "cells": len(cells), "work_sha256": _hash(work_path())}))
    elif args.prepare:
        print(json.dumps(prepare(), sort_keys=True))
    elif args.launch:
        print(json.dumps(launch(), sort_keys=True))
    elif args.control:
        control()
    elif args.watch:
        watch()
    elif args.status:
        print(json.dumps(status(), sort_keys=True))
    elif args.summary:
        print(json.dumps({"cells": len(rows()), "first": [asdict(c) for c in rows()[:2]]}, indent=1))
    else:
        run_shard(args.shard)
    return 0


if __name__ == "__main__":
    sys.exit(main())
