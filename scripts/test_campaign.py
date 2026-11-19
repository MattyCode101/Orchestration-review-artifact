"""Test-campaign plan, freeze list and dry run. It cannot launch.

--write-freeze-draft  record the hashes the campaign will pin (a DRAFT until the authors sign on Sunday)
--dry-run             verify every hash and the manifest, look up real stored records through the manifest with
                      zero tool runs, walk one fake episode per approach (all twelve kinds, the LLM ones through
                      the real policy and a scripted transport), and report whether the campaign would launch
--token-cap           print the token cap: 3 times the expected use from training per-episode means
No hardware tool starts, nothing is written to a stored root, and no test outcome value is printed.
Launching is scripts/campaign_run.py --scope test, which refuses without an approved freeze file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

import campaign_harness as h  # noqa: E402
from orch.contract.brief_text import load_frozen_brief  # noqa: E402
from orch.contract.models import (  # noqa: E402
    DesignState, Evidence, MetricName, MetricValue, RunOutcome, RunRecord, RunStatus,
)
from orch.engine.apply import StepEngine  # noqa: E402
from orch.engine.cache import RunCache  # noqa: E402
from orch.engine.manifest_cache import (  # noqa: E402
    CacheManifest, ManifestRunCache, ManifestStateStore, ManifestStepEngine,
)
from orch.engine.store import StateStore, _atomic_bytes, _json_bytes, _sha256_file  # noqa: E402
from orch.episode import run_episode  # noqa: E402
from orch.policies.llm_client import LLMReply  # noqa: E402
from orch.recipes.registry import load_registry  # noqa: E402

PREP = PROJECT / "data/test_campaign_prep_2026-11-06"
MANIFEST = PREP / "cache_manifest.json"
FREEZE = PREP / "freeze_draft.json"
FINAL_FREEZE = PREP / "freeze_final.json"
DEV_CANDIDATE = PROJECT / "notes/freeze_candidate_sunday.sha256"
DEV_TELEMETRY = PROJECT / "data/pilots/saturday_llm_shared/telemetry.jsonl"
DESIGNS = tuple(h.settings()["test_designs"])
FOUR = ("T0", "A0", "T", "A")
BRIEFS = FOUR + ("P0",)
HARD_CAPS = {"new_real_runs": 800, "job_hours": 240, "usd": 50}
REPEATS = 2                      # one fresh repeat of F1-R1 and GA-R1; the model is deterministic (decision of 2026-11-08)
# A circuit-breaker on live extraction jobs, not a decided cap: the 240 job-hour cap bounds extraction time (about
# 180 states expected). Set well above the expectation so it never turns an extraction into a dropped cell.
EXTRACTION_JOBS = 500

FREEZE_FILES = (
    # briefs, gate and contract
    "configs/briefs.yaml", "orch/contract/brief_text.py", "orch/contract/models.py", "orch/contract/views.py",
    "configs/noise.yaml",
    # rule-based policies
    "orch/policies/base.py", "orch/policies/repair_loop.py", "orch/policies/relevant_workers.py",
    "orch/policies/training_baselines.py", "orch/policies/random_policy.py", "orch/policies/trivial.py",
    # LLM orchestrator and general agent (developed by the development run; imported, never copied)
    "orch/policies/llm.py", "orch/policies/llm_client.py", "orch/policies/llm_prompt.py",
    "orch/policies/llm_schemas.py", "orch/policies/llm_view.py", "orch/policies/llm_scaffold.py",
    "orch/analysis/profiles.py", "orch/baselines/general_agent.py", "orch/baselines/direct_recipe_worker.py",
    # LLM templates, configurations, profiles, model configuration
    "configs/prompts/orchestrator_v1.txt", "configs/prompts/orchestrator_v2.txt", "configs/prompts/general_agent_v1.txt",
    "configs/policies/llm.yaml", "configs/policies/llm_r1.yaml", "configs/campaign/campaign.yaml",
    "configs/profiles/manifest.json", "configs/profiles/timing.yaml", "configs/profiles/area.yaml",
    "configs/profiles/power.yaml", "configs/profiles/recipes.yaml",
    # bridge and SDK versions
    "tools/llm_bridge/call.py", "tools/llm_bridge/requirements.lock",
    # episode machinery, workers, engine, meter, recipes
    "orch/episode.py", "orch/workers/base.py", "orch/workers/fitted.py", "orch/workers/telemetry.py",
    "configs/recipe_sets.yaml", "configs/power_recipe_set.yaml", "configs/workers/timing.yaml",
    "configs/workers/area.yaml", "configs/workers/power.yaml", "configs/workers/A_fit.json",
    "configs/workers/B_fit.json", "configs/worker_features.yaml", "recipes/registry.yaml", "recipes/support/env.sh",
    "orch/engine/apply.py", "orch/engine/cache.py", "orch/engine/store.py", "orch/engine/gate.py",
    "orch/engine/journal.py", "orch/engine/meter.py", "orch/engine/meter_parsers.py", "orch/engine/real_backend.py",
    "orch/engine/runner.py", "orch/engine/jobtag.py", "orch/runtime.py", "orch/recipes/registry.py",
    "orch/recipes/adapters/tcl.py", "orch/recipes/adapters/latch.py", "orch/recipes/adapters/validator.py",
    "orch/baselines/verification.py",
    # manifest layer, harness, runner, launcher, analysis
    "orch/engine/manifest_cache.py", "orch/engine/read_through_ledger.py", "scripts/campaign_harness.py",
    "scripts/campaign_run.py", "scripts/campaign_manifest.py", "scripts/test_campaign.py",
    "scripts/test_cache_manifest.py", "scripts/campaign_analysis.py", "notes/analysis_plan.md",
    "scripts/campaign_audit.py", "scripts/campaign_replay.py",
    # test inputs: selection, noise bands, manifest
    "data/sweeps/friday_test_power_2026-11-06/selection.json",
    "data/exp_evening_2026-11-04/test_b2/overnight_baseline_2026-11-04/noise_bands_2026-11-04.json",
    "data/exp_evening_2026-11-04/test_b2/overnight_baseline_2026-11-04/admitted_test_designs_2026-11-04.json",
    "data/scratch_saturday_2026-11-07/test_power_noise_bands.json",
    "data/test_campaign_prep_2026-11-06/cache_manifest.json",
)
# Still open on Sunday 2026-11-08 (the earlier items - validity rule, live worker B provider, named variants, model
# decision, repeats - are done and listed in notes/decisions.md).
PENDING = (
    "the development run's FINAL line in notes/progress.md, after which nothing in the freeze may change",
    "the final freeze (union of both lists, hashed from disk) and the development run's independent recomputation",
    "replay equivalence and the second rehearsal on the final code",
    "The authors' signature: matching hashes plus all gates passing (decision (f))",
)


def episodes(repeats: int = REPEATS) -> list[dict[str, Any]]:
    return [{"episode_id": c.episode_id, "design": c.design, "brief": c.brief, "approach": c.approach,
             "tag": c.tag, "tier": c.tier} for c in h.make_cells("test", DESIGNS, BRIEFS, repeats)]


def episodes_sha256(repeats: int = REPEATS) -> str:
    return hashlib.sha256(_json_bytes(episodes(repeats))).hexdigest()


def training_token_means() -> dict[str, tuple[float, int]]:
    """Per-episode total tokens of each LLM cell on training (what a fresh run costs), from stored telemetry."""

    want = {("grid1", "F1"): "F1-R0", ("grid2", "F1"): "F1-R1", ("grid2", "E0"): "E0-R1", ("ga1", "GA"): "GA-R1"}
    seen: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
    for line in DEV_TELEMETRY.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        parts = r["episode_id"].split("_")
        if not r["episode_id"].startswith("llm_") or (parts[1], parts[2]) not in want:
            continue
        seen[(want[(parts[1], parts[2])], r["episode_id"])][r["decision_index"]] = r["tokens"]["total"]
    means: dict[str, tuple[float, int]] = {}
    for key in set(k for k, _ in seen):
        totals = [sum(v.values()) for (k, _e), v in seen.items() if k == key]
        means[key] = (statistics.mean(totals), len(totals))
    return means


def expected_tokens(repeats: int = REPEATS) -> float:
    means = training_token_means()
    runs = {"F1-R1": repeats, "GA-R1": repeats, "E0-R1": 1, "F1-R0": 1}
    return sum(means[k][0] * runs[k] * len(DESIGNS) * len(BRIEFS) for k in runs)


def token_cap(repeats: int = REPEATS) -> int:
    return int(-(-3 * expected_tokens(repeats) // 100_000) * 100_000)      # 3 x expected, rounded up to 100k


def caps(repeats: int = REPEATS) -> dict[str, Any]:
    return {**HARD_CAPS, "tokens": token_cap(repeats)}


def ready_lines() -> list[str]:
    path = PROJECT / "notes/progress.md"
    return [line[:200] for line in path.read_text(encoding="utf-8").splitlines() if re.search(r"\bREADY:", line)]


# ----------------------------------------------------------------- freeze checks
def dev_candidate_files() -> dict[str, str]:
    """The development run's sha256sum-style freeze candidate: path -> hash it lists (comment lines skipped)."""

    out: dict[str, str] = {}
    for line in DEV_CANDIDATE.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, path = line.split(None, 1)
        out[path.strip()] = digest
    return out


def freeze_record(final: bool = False) -> dict[str, Any]:
    missing = [p for p in FREEZE_FILES if not (PROJECT / p).is_file()]
    if missing:
        raise FileNotFoundError(f"freeze file missing: {missing}")
    extra: dict[str, str] = {}
    if final:                                   # decision (f): the union of the two lists, hashed from disk here
        listed = dev_candidate_files()
        missing = [p for p in listed if not (PROJECT / p).is_file()]
        if missing:
            raise FileNotFoundError(f"development freeze file missing: {missing}")
        extra = {p: _sha256_file(PROJECT / p) for p in listed if p not in FREEZE_FILES}
        disagree = [p for p, d in listed.items() if _sha256_file(PROJECT / p) != d]
        if disagree:
            raise ValueError(f"the development candidate lists other hashes than disk for: {disagree}")
    lock = (PROJECT / "tools/llm_bridge/requirements.lock").read_text(encoding="utf-8")
    sdk = re.findall(r"^google-genai==\S+", lock, flags=re.M)
    llm = json.loads(json.dumps(__import__("yaml").safe_load((PROJECT / "configs/policies/llm.yaml").read_text())))
    return {
        "schema_version": "test-campaign-freeze-draft-v2",
        "status": ("FINAL: union of the experimentation draft and the development candidate; valid only with matching "
                   "independent recomputation") if final else "DRAFT, not signed",
        "files": {**{p: _sha256_file(PROJECT / p) for p in FREEZE_FILES}, **extra},
        "dev_candidate_sha256": _sha256_file(DEV_CANDIDATE) if final else None, "extraction_jobs": EXTRACTION_JOBS,
        "model": llm["model"], "temperature": llm["temperature"], "seed": llm["seed"],
        "sdk": sdk, "episodes": len(episodes()), "episodes_sha256": episodes_sha256(),
        "manifest_sha256": _sha256_file(MANIFEST), "caps": caps(), "repeats": REPEATS,
        "pending_items": list(PENDING),
    }


def check_freeze(path: Path | None = None) -> dict[str, Any]:
    saved = json.loads((path or FREEZE).read_text(encoding="utf-8"))
    changed = [p for p, hv in saved["files"].items() if not (PROJECT / p).is_file() or _sha256_file(PROJECT / p) != hv]
    llm = __import__("yaml").safe_load((PROJECT / "configs/policies/llm.yaml").read_text())
    ok_parts = {
        "episodes_hash_matches": saved["episodes_sha256"] == episodes_sha256(saved["repeats"]),
        "manifest_hash_matches": saved["manifest_sha256"] == _sha256_file(MANIFEST),
        "caps_recorded": saved["caps"] == caps(saved["repeats"]),
        "model_recorded": saved["model"] == llm["model"],
        "extraction_cap_recorded": saved.get("extraction_jobs") == EXTRACTION_JOBS,
        "dev_candidate_unchanged": saved.get("dev_candidate_sha256") in (None, _sha256_file(DEV_CANDIDATE)),
    }
    return {"files_checked": len(saved["files"]), "files_changed": changed, **ok_parts,
            "ok": not changed and all(ok_parts.values())}


def pins_against_freeze(name: str, freeze_path: Path | None = None) -> dict[str, Any]:
    """Did a finished campaign run on the frozen code? Compare the hashes it recorded for every project file it
    imported, at the start and at the end of each run, with the final freeze (paths in both only)."""

    saved = json.loads((freeze_path or FINAL_FREEZE).read_text(encoding="utf-8"))["files"]
    work = PROJECT / f"data/campaign_{name}"
    runs = {}
    for run_dir in sorted((work / "runs").glob("[0-9]*"), key=lambda q: int(q.name)):
        for label, path in (("start", run_dir / "pins_start.json"), ("end", run_dir / "pins_end.json")):
            if not path.is_file():
                runs[f"run {run_dir.name} {label}"] = {"recorded": False}
                continue
            files = json.loads(path.read_text(encoding="utf-8"))["files"]
            common = sorted(set(files) & set(saved))
            runs[f"run {run_dir.name} {label}"] = {
                "recorded": True, "files_recorded": len(files), "in_the_freeze": len(common),
                "differ_from_the_freeze": [f for f in common if files[f] != saved[f]],
                "recorded_but_not_in_the_freeze": sorted(set(files) - set(saved))}
    ok = bool(runs) and all(v["recorded"] and not v["differ_from_the_freeze"] and not v["recorded_but_not_in_the_freeze"]
                            for v in runs.values())
    return {"campaign": name, "ok": ok, "runs": runs}


def frozen_settings(path: Path) -> dict[str, Any]:
    """The settings of an approved FINAL freeze, or a refusal. Only the project's own final freeze file counts."""

    path = Path(path).resolve()
    if path != FINAL_FREEZE.resolve():
        raise RuntimeError(f"not the final freeze file: {path}")
    result = check_freeze(path)
    if not result["ok"]:
        raise RuntimeError("the freeze check fails: " + json.dumps({k: v for k, v in result.items() if v is not True}))
    saved = json.loads(path.read_text(encoding="utf-8"))
    if not str(saved["status"]).startswith("FINAL"):
        raise RuntimeError("the freeze file is a draft")
    return saved


def verify_plan_against_freeze(path: Path, plan: dict[str, Any]) -> None:
    """Refuse a plan that differs from the frozen configuration in any respect (gate G8)."""

    saved = frozen_settings(path)
    expected = episodes(saved["repeats"])
    got = [{k: c[k] for k in ("episode_id", "design", "brief", "approach", "tag", "tier")} for c in plan["cells"]]
    problems = []
    if got != expected:
        problems.append("cells differ from the frozen episode list")
    if plan["caps"] != saved["caps"]:
        problems.append("caps differ")
    for key, value in (("scope", "test"), ("telemetry", "live"), ("repeats", saved["repeats"]),
                       ("model", saved["model"]), ("manifest_sha256", saved["manifest_sha256"]),
                       ("extraction_jobs", saved["extraction_jobs"]), ("designs", list(DESIGNS)), ("briefs", list(BRIEFS))):
        if plan.get(key) != value:
            problems.append(f"plan {key} differs")
    if problems:
        raise RuntimeError("the plan is not the frozen configuration: " + "; ".join(problems))


# ------------------------------------------------- real-record lookups, zero tools
class NoToolBackend:
    calls = 0

    def execute(self, *a, **k):
        NoToolBackend.calls += 1
        raise RuntimeError("tool run attempted")

    def measure_start(self, *a, **k):
        NoToolBackend.calls += 1
        raise RuntimeError("start measurement attempted")


def check_real_lookups(manifest: CacheManifest) -> dict[str, Any]:
    registry = load_registry(PROJECT)
    changed = [k[:12] for k, e in manifest.entries.items()
               if _sha256_file(manifest.roots[e["root"]] / "runs" / e["run_id"] / "complete.json") != e["complete_sha256"]]
    version_mismatch = []
    for k, e in manifest.entries.items():
        try:
            if registry.get(e["recipe_id"]).version != e["recipe_version"]:
                version_mismatch.append(k[:12])
        except Exception:
            version_mismatch.append(k[:12])
    picks: dict[str, str] = {}
    for key, e in manifest.entries.items():
        kind = "alias" if e["approval"]["kind"] == "meter_equivalence_alias" else e["root"].split(":")[0]
        picks.setdefault(kind, key)
    scratch = Path(tempfile.mkdtemp(prefix="lookup_", dir=PREP))
    store = ManifestStateStore(scratch, manifest)
    NoToolBackend.calls = 0
    engine = ManifestStepEngine(
        manifest=manifest, require_zero_tool=True, store=store,
        cache=ManifestRunCache(scratch, manifest, meter_version=manifest.body["meter_version"]),
        backend=NoToolBackend(), meter_version=manifest.body["meter_version"], cache_enabled=True)
    results = {}
    for kind, key in picks.items():
        e = manifest.entries[key]
        before = _sha256_file(manifest.roots[e["root"]] / "runs" / e["run_id"] / "complete.json")
        state = DesignState(id=e["input_state_id"], design="stored", stage="routed", fake=False,
                            path=str(store.state_path(e["input_state_id"])))
        outcome = engine.execute(state, registry.get(e["recipe_id"]))
        results[kind] = {
            "cache_hit": outcome.run.cache_hit is True,
            "original_cost_charged": all(getattr(outcome.run, f) == e["cost"][f] for f in e["cost"]),
            "output_state_matches": outcome.state is not None and outcome.state.id == e["output_state_id"],
            "stored_record_unchanged": before == _sha256_file(manifest.roots[e["root"]] / "runs" / e["run_id"] / "complete.json"),
        }
    return {"entries": len(manifest.entries), "markers_changed": changed, "recipe_version_mismatch": version_mismatch,
            "sampled_lookups": results, "tool_runs": NoToolBackend.calls,
            "ok": not changed and not version_mismatch and NoToolBackend.calls == 0
            and all(all(v.values()) for v in results.values())}


# ------------------------------------------------------- fake walk, one per approach
class DryRunBackend:
    """Labelled fake backend: deterministic, any recipe, three metrics, no tool."""

    meter_version = "dry-run-fake-v1"

    def __init__(self, store: StateStore) -> None:
        self.store, self.metrics, self.calls = store, {}, 0

    def create_start(self, folder: Path) -> DesignState:
        source = folder / "start.dcp"
        source.write_bytes(b"dry-run-start")
        state = self.store.import_state(source, design="dry_run_fake", fake=True)
        self.metrics[state.id] = (-1.0, 100.0, 2.0)
        return state

    def _evidence(self, state: DesignState) -> Evidence:
        wns, luts, power = self.metrics[state.id]
        return Evidence(
            id=hashlib.sha256(f"{state.id}:{wns}:{luts}:{power}".encode()).hexdigest(), state_id=state.id,
            routed=True, route_errors=0, drc_errors=0, constraint_fingerprint="dry", clock_digest="dry",
            meter_version=self.meter_version, fake=True,
            metrics={
                MetricName.WNS_NS: MetricValue(value=wns, unit="ns", method="dry-run", basis="routed"),
                MetricName.PACKED_LUTS: MetricValue(value=luts, unit="count", method="dry-run", basis="routed"),
                MetricName.DYNAMIC_POWER_W: MetricValue(value=power, unit="W", method="dry-run", basis="routed"),
            })

    def measure_start(self, state: DesignState, *, run_dir: Path) -> Evidence:
        return self._evidence(state)

    def execute(self, state, recipe, *, run_id, run_dir) -> RunOutcome:
        self.calls += 1
        v = int(hashlib.sha256(recipe.id.encode()).hexdigest()[:8], 16)
        wns, luts, power = self.metrics[state.id]
        new = (wns + ((v % 16) - 4) / 100.0, luts + ((v >> 4) % 7) - 3, power + (((v >> 8) % 9) - 4) / 100.0)
        out = run_dir / "out.dcp"
        out.write_bytes(f"{state.id}:{recipe.id}".encode())
        succ = self.store.import_state(out, design=state.design, parent_id=state.id, made_by_run_id=run_id, fake=True)
        self.metrics[succ.id] = new
        run = RunRecord(id=run_id, recipe_id=recipe.id, recipe_version=recipe.version, status=RunStatus.OK, exit_code=0,
                        wall_s=1.0, input_sha256_before=state.id, input_sha256_after=state.id, fake=True)
        return RunOutcome(run=run, state=succ, evidence=self._evidence(succ))


class ScriptedTransport:
    """Stands in for the model: a valid decision for each schema, then STOP. Exercises the real policy."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, params, schema) -> LLMReply:
        self.calls += 1
        props = schema["properties"]
        i = self.calls
        base = {"state_diagnosis": "dry run", "strategy": "EXPLORE", "conflict_check": "none", "rationale_summary": "dry run"}
        if i > 3:
            body = {**base, "strategy": "STOP", "action": "STOP", "mandate": None}
            if "recipe_action" in props:
                body = {**base, "strategy": "STOP", "action": "STOP", "recipe_action": None}
        elif "recipe_action" in props:
            rec = props["recipe_action"]["anyOf"][1]["properties"]
            body = {**base, "action": "RUN_RECIPE", "recipe_action": {
                "recipe_id": rec["recipe_id"]["enum"][(i - 1) % len(rec["recipe_id"]["enum"])],
                "base_state": rec["base_state"]["enum"][0]}}
        else:
            m = props["mandate"]["anyOf"][1]["properties"]
            body = {**base, "action": "INVOKE_WORKER", "mandate": {
                "worker_id": m["worker_id"]["enum"][(i - 1) % len(m["worker_id"]["enum"])],
                "base_state": m["base_state"]["enum"][0], "preserve_placement": bool(i % 2), "attempts": 1,
                "respect_brief_limits": True}}
        text = json.dumps(body)
        return LLMReply(text=text, prompt_tokens=400, completion_tokens=60, thoughts_tokens=0, total_tokens=460,
                        latency_ms=1.0, finish_reason="STOP", model=params.model)


def walk_fake() -> list[dict[str, Any]]:
    registry = load_registry(PROJECT)
    plan = [("v2", "T0"), ("v1", "A0"), ("single", "A0"), ("fixed_AB", "T"), ("fixed_BA", "A"), ("fixed_ACB", "P0"),
            ("random_0", "T0"), ("random_1", "P0"), ("F1-R1", "T0"), ("F1-R0", "A0"), ("E0-R1", "A0"), ("GA-R1", "P0")]
    out = []
    for approach, brief in plan:
        root = Path(tempfile.mkdtemp(prefix="fake_", dir=PREP))
        store = StateStore(root)
        backend = DryRunBackend(store)
        start = backend.create_start(root)
        engine = StepEngine(store=store, cache=RunCache(root), backend=backend,
                            meter_version=backend.meter_version, cache_enabled=False)
        cell = h.Cell("dry_run_fake", brief, approach, "m1", 1, f"dryrun_{brief}_{approach}")
        scope = SimpleNamespace(work_root=root)
        transport = ScriptedTransport()
        ctx = SimpleNamespace(scope=scope, caps=None, notes={"fake_transport": transport})
        workers = fake_workers(brief, general=cell.is_llm and h.settings()["llm_cells"][approach]["kind"] == "general")
        policy = h.make_policy(ctx, cell, sorted(workers))
        result = run_episode(engine=engine, registry=registry, workers=workers, policy=policy,
                             brief=load_frozen_brief(brief), start_state=start, episode_id=cell.episode_id)
        out.append({"approach": approach, "brief": brief, "fake": result.fake, "recipe_runs": result.recipe_runs,
                    "stop_reason": result.stop_reason,
                    "final_brief_valid": bool(result.final_verdict and result.final_verdict.brief_valid),
                    "fake_tool_calls": backend.calls, "model_calls": transport.calls})
    return out


def fake_workers(brief: str, general: bool):
    from orch.baselines.direct_recipe_worker import DirectRecipeWorker
    from orch.config import load_recipe_sets, load_worker_card
    from orch.workers.base import GenericWorker

    sets = load_recipe_sets(PROJECT / "configs/recipe_sets.yaml")
    power = load_recipe_sets(PROJECT / "configs/power_recipe_set.yaml")
    ids = {"timing": tuple(sets.worker_sets["timing"]), "area": tuple(sets.worker_sets["area"]),
           "power": tuple(power.worker_sets["power"])}
    names = ("timing", "area", "power") if brief == "P0" else ("timing", "area")
    cls = DirectRecipeWorker if general else GenericWorker
    return {n: cls(load_worker_card(PROJECT / f"configs/workers/{n}.yaml"), ids[n]) for n in names}


def launcher_refuses_test_scope_without_freeze() -> bool:
    import campaign_run as run

    try:
        run.launch("__nonexistent__", resume=False, freeze_ok=False)
    except Exception:
        pass
    source = (PROJECT / "scripts/campaign_run.py").read_text(encoding="utf-8")
    return "the test scope needs an approved freeze check" in source


def dry_run() -> dict[str, Any]:
    PREP.mkdir(parents=True, exist_ok=True)
    manifest = CacheManifest.load(MANIFEST)
    freeze = check_freeze(FINAL_FREEZE if FINAL_FREEZE.exists() else None)
    freeze["file"] = "freeze_final.json" if FINAL_FREEZE.exists() else "freeze_draft.json"
    lookups = check_real_lookups(manifest)
    walks = walk_fake()
    walked_ok = all(w["final_brief_valid"] and w["fake"] for w in walks)
    rows = episodes()
    by_tier = {t: sum(r["tier"] == t for r in rows) for t in (1, 2, 3)}
    report = {
        "schema_version": "test-campaign-dry-run-v2", "manifest_sha256": manifest.hash, "freeze": freeze,
        "real_lookups": lookups, "fake_walks": walks, "episodes": len(rows), "episodes_by_tier": by_tier,
        "episodes_if_not_deterministic": len(episodes(3)), "caps": caps(), "caps_if_not_deterministic": caps(3),
        "ready_lines_in_progress_md": ready_lines(), "pending_items": list(PENDING),
        "launcher_refuses_test_scope_without_freeze": launcher_refuses_test_scope_without_freeze(),
    }
    ready = freeze["ok"] and lookups["ok"] and walked_ok and report["launcher_refuses_test_scope_without_freeze"]
    report["checks_pass"] = bool(ready)
    report["would_launch_full_campaign"] = bool(ready and FINAL_FREEZE.exists())
    report["launch_blocked_by"] = [] if report["would_launch_full_campaign"] else list(PENDING)
    _atomic_bytes(PREP / "dry_run_report.json", _json_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pins-against-freeze", metavar="CAMPAIGN",
                      help="did this finished campaign run on the files of the final freeze?")
    mode.add_argument("--write-freeze-draft", action="store_true")
    mode.add_argument("--write-freeze-final", action="store_true",
                      help="after the development run's FINAL line: union of both lists, hashed from disk")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--token-cap", action="store_true")
    args = parser.parse_args()
    if args.token_cap:
        print(json.dumps({"means": training_token_means(), "expected_tokens": expected_tokens(),
                          "token_cap": token_cap(), "token_cap_if_not_deterministic": token_cap(3)}, default=str))
        return 0
    if args.pins_against_freeze:
        report = pins_against_freeze(args.pins_against_freeze)
        print(json.dumps(report, indent=1, sort_keys=True))
        return 0 if report["ok"] else 1
    if args.write_freeze_final:
        record = freeze_record(final=True)
        if FINAL_FREEZE.exists():
            raise FileExistsError(f"{FINAL_FREEZE} exists; a final freeze is written once")
        _atomic_bytes(FINAL_FREEZE, _json_bytes(record))
        print(json.dumps({"freeze_final_sha256": _sha256_file(FINAL_FREEZE), "files": len(record["files"]),
                          "episodes": record["episodes"], "caps": record["caps"]}, sort_keys=True))
        return 0
    if args.write_freeze_draft:
        record = freeze_record()
        _atomic_bytes(FREEZE, _json_bytes(record))
        print(json.dumps({"freeze_draft_sha256": _sha256_file(FREEZE), "files": len(record["files"]),
                          "episodes": record["episodes"], "caps": record["caps"]}, sort_keys=True))
        return 0
    report = dry_run()
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if report["checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
