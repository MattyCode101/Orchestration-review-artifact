"""Power-mediation study (Tuesday 2026-11-10): shared setup. Training designs only.

Nothing pinned is edited. The pinned ceiling code, gate, workers and policies are imported and used as
they are (scripts/friday_ceiling_exact.py is imported so the memoised gate is active, exactly as in the
Friday exact ceilings). What is new here:

  1. UnionIndex / UnionStore: a read-only view of the training run graph over the sealed v4 root, the ten
     overnight power roots (phase a and b, five designs) and, once it exists, this study's own engine root.
     Nothing in an old root is written (no StateStore is opened on an old root).
  2. UnionOracle: the same acceptance rules as friday_ceiling_exact.LiveCacheOracle (identity, current
     meter, clean start-relative properties, AMD PASS with compared outputs), reading the union. It also
     logs every uncached (state, recipe) pair a mandate asked for, with the cost at which it was asked.
  3. UnionTelemetryProvider: SavedTelemetryProvider with the producing run's utilization report taken from
     whichever root holds it (same rule the Friday ceiling used for states made after the v3 plan).
  4. The candidate briefs (P0 is the frozen brief; L2 and W2 are new, defined here only).
"""
from __future__ import annotations
import os

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT = Path(os.environ.get("ORCH_PROJECT_ROOT") or Path(__file__).resolve().parents[2])
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from orch.analysis import orchestrator_ceiling as old  # noqa: E402
from orch.analysis.orchestrator_ceiling import (  # noqa: E402
    Simulator, SimNode, _best_key, _improvement, _signature, simulate_policy,
)
from orch.config import load_noise, load_recipe_sets, load_worker_card  # noqa: E402
from orch.contract.brief_text import load_frozen_brief  # noqa: E402
from orch.contract.models import (  # noqa: E402
    Brief, BriefTerm, DesignState, Evidence, MetricName, Objective, RunOutcome, RunStatus,
)
from orch.engine.cache import RunCache  # noqa: E402
from orch.engine.manifest_cache import _read_run  # noqa: E402
from orch.recipes.registry import load_registry  # noqa: E402
from orch.workers.base import GenericWorker  # noqa: E402
from orch.workers.fitted import FittedWorker  # noqa: E402
from orch.workers.telemetry import SavedTelemetryProvider  # noqa: E402
from scripts import friday_analysis as fa  # noqa: E402
from scripts import friday_ceiling_exact as fce  # noqa: E402  (activates the memoised gate)
from scripts import thursday_night_pilot as pilot  # noqa: E402

V4 = PROJECT / "data/pilots/thursday_overnight_train_v4_2026-11-05"
POWER_TOP = PROJECT / "data/sweeps/thursday_overnight_power_2026-11-05"
WORK = PROJECT / "data/power_mediation_2026-11-10"
NEW_ROOT = WORK / "engine"
DESIGNS = tuple(pilot.DESIGNS)
_RUN = re.compile(r"run_[0-9a-f]{32}\Z")


def jread(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def union_roots(include_new: bool = True) -> dict[str, Path]:
    """Order matters: for a duplicated key the first root wins (the sealed v4 root is first)."""

    roots: dict[str, Path] = {"v4": V4}
    for phase in ("a", "b"):
        for design in DESIGNS:
            roots[f"power_{phase}:{design}"] = POWER_TOP / f"phase_{phase}" / design
    if include_new and (NEW_ROOT / "cache").is_dir():
        roots["new"] = NEW_ROOT
    return roots


class UnionIndex:
    """Cache pointers over several engine roots (non-retryable, completed runs only)."""

    def __init__(self, roots: dict[str, Path]) -> None:
        self.roots = {name: Path(root).resolve() for name, root in roots.items()}
        self.key_to_run: dict[str, tuple[str, str]] = {}
        self.run_to_root: dict[str, str] = {}
        self.duplicate_keys: list[tuple[str, str, str]] = []
        self.skipped = Counter()
        for name, root in self.roots.items():
            for pointer in sorted((root / "cache").glob("*.json")):
                data = jread(pointer)
                if data.get("retryable"):
                    self.skipped["retryable"] += 1
                    continue
                key, run_id = data["key"], data["run_id"]
                if not (root / "runs" / run_id / "complete.json").is_file():
                    self.skipped["dangling"] += 1
                    continue
                self.run_to_root.setdefault(run_id, name)
                if key in self.key_to_run:
                    self.duplicate_keys.append((key, self.key_to_run[key][0], name))
                    continue
                self.key_to_run[key] = (name, run_id)

    def digest(self) -> str:
        lines = sorted(f"{k}:{r}:{i}" for k, (r, i) in self.key_to_run.items())
        return hashlib.sha256("\n".join(lines).encode()).hexdigest()


class UnionStore:
    """Read-only access to completed runs and to one canonical state location per state."""

    def __init__(self, index: UnionIndex) -> None:
        self.index = index
        self._paths: dict[str, Path] = {}

    def run_dir(self, run_id: str) -> Path | None:
        name = self.index.run_to_root.get(run_id)
        return None if name is None else self.index.roots[name] / "runs" / run_id

    def state_path(self, state_id: str) -> Path:
        if state_id not in self._paths:
            for root in self.index.roots.values():
                path = root / "states" / state_id / "design.dcp"
                if path.is_file():
                    self._paths[state_id] = path
                    break
            else:
                raise FileNotFoundError(f"state is in no union root: {state_id}")
        return self._paths[state_id]

    def normalised(self, state: DesignState | None) -> DesignState | None:
        if state is None:
            return None
        canonical = str(self.state_path(state.id))
        return state if state.path == canonical else state.model_copy(update={"path": canonical})

    def load_outcome(self, run_id: str) -> RunOutcome:
        run_dir = self.run_dir(run_id)
        if run_dir is None:
            raise FileNotFoundError(f"run is in no union root: {run_id}")
        run, state, evidence, claims = _read_run(run_dir, run_id)
        return RunOutcome(run=run, state=self.normalised(state), evidence=evidence, claims=claims)


class UnionAmd:
    """AMD PASS with compared outputs, read from whichever root holds the verification record."""

    def __init__(self, index: UnionIndex, b2: dict[str, str], lenient: bool = False) -> None:
        self.index, self.b2 = index, b2
        self.lenient = lenient
        self.memo: dict[tuple[str, str], bool] = {}
        self.unverified_accepted: set[tuple[str, str]] = set()

    def has_record(self, design: str, state_id: str) -> bool:
        return any((root / "verification" / "latest" / design / f"{state_id}.json").is_file()
                   for root in self.index.roots.values())

    def passed(self, design: str, state_id: str) -> bool:
        ident = design, state_id
        if ident in self.memo:
            return self.memo[ident]
        ok = False
        for root in self.index.roots.values():
            latest = root / "verification" / "latest" / design / f"{state_id}.json"
            if not latest.is_file():
                continue
            try:
                record = jread(latest)
                if (
                    record.get("result") != "pass" or record.get("design") != design
                    or record.get("output_sha256") != state_id
                    or record.get("reference_sha256") != self.b2[design]
                ):
                    continue
                report_path, report_sha = record.get("raw_report_path"), record.get("raw_report_sha256")
                if not report_path or not report_sha:
                    continue
                if hashlib.sha256(Path(report_path).read_bytes()).hexdigest() != report_sha:
                    continue
                if pilot.outputs_compared(record, root / "verification" / "runs") <= 0:
                    continue
                ok = True
                break
            except (OSError, ValueError, KeyError, pilot.PilotSafetyStop):
                continue
        if not ok and self.lenient and not self.has_record(design, state_id):
            # lenient mode: a state with no AMD record at all is accepted and remembered; a recorded
            # failure or unknown is still refused
            ok = True
            self.unverified_accepted.add(ident)
        self.memo[ident] = ok
        return ok


class TwinAlias:
    """Stored runs by (input state class, recipe, version), where a class holds byte-different twins.

    Used only in the twin-aware variant of the oracle (an INFERRED extension: twins have identical WNS, LUTs
    and power in every class checked, and the real replicate runs of this study test the assumption)."""

    def __init__(self, uf, runs) -> None:
        from collections import defaultdict
        self.uf = uf
        self.by_edge: dict[tuple, list[str]] = defaultdict(list)
        for r in sorted(runs, key=lambda x: x["run"]):
            self.by_edge[(uf.find(r["in"]), r["recipe"], r["ver"])].append(r["run"])


class UnionOracle:
    """Acceptance rules of friday_ceiling_exact.LiveCacheOracle, reading the union of roots."""

    def __init__(self, registry, meter_version: str, index: UnionIndex, store: UnionStore,
                 amd: UnionAmd, start_evidence: dict[str, Evidence], alias: "TwinAlias | None" = None) -> None:
        self.registry, self.meter_version = registry, meter_version
        self.index, self.store, self.amd = index, store, amd
        self.alias = alias
        self.alias_used: set[tuple[str, str, str]] = set()
        self.start_evidence = start_evidence
        self.memo: dict[tuple[str, str], RunOutcome | None] = {}
        self.miss_reasons: Counter[str] = Counter()
        self.miss_log: dict[tuple[str, str], dict[str, Any]] = {}
        self.accepted_roots: Counter[str] = Counter()
        self.ctx: tuple[int, int | None, str | None] = (0, None, None)

    def _miss(self, base: DesignState, recipe_id: str, reason: str):
        self.memo[(base.id, recipe_id)] = None
        self.miss_reasons[reason] += 1
        self.miss_log[(base.id, recipe_id)] = {
            "design": base.design, "reason": reason, "requests": 0,
            "min_cost": None, "workers": set(),
        }
        return None

    def _note(self, base: DesignState, recipe_id: str) -> None:
        row = self.miss_log[(base.id, recipe_id)]
        cost, _attempts, worker = self.ctx
        row["requests"] += 1
        row["min_cost"] = cost if row["min_cost"] is None else min(row["min_cost"], cost)
        if worker:
            row["workers"].add(worker)

    def get(self, base: DesignState, recipe_id: str) -> RunOutcome | None:
        ident = base.id, recipe_id
        if ident in self.memo:
            if self.memo[ident] is None:
                self._note(base, recipe_id)
            return self.memo[ident]
        recipe = self.registry.get(recipe_id)
        key = RunCache.key(base.id, recipe.id, recipe.version)
        hit = self.index.key_to_run.get(key)
        aliased = False
        if hit is None and self.alias is not None:
            rids = self.alias.by_edge.get((self.alias.uf.find(base.id), recipe.id, recipe.version))
            if rids:
                hit, aliased = (self.index.run_to_root[rids[0]], rids[0]), True
        if hit is None:
            self._miss(base, recipe_id, "not_in_union")
            self._note(base, recipe_id)
            return None
        root_name, run_id = hit
        outcome = self.store.load_outcome(run_id)
        run = outcome.run
        same_input = (
            self.alias.uf.find(run.input_sha256_before) == self.alias.uf.find(base.id)
            and self.alias.uf.find(run.input_sha256_after) == self.alias.uf.find(base.id)
        ) if aliased else (run.input_sha256_before == base.id and run.input_sha256_after == base.id)
        if aliased and same_input:
            self.alias_used.add((base.design, base.id, recipe_id))
        if (
            run.recipe_id != recipe.id or run.recipe_version != recipe.version
            or not same_input or run.fake
        ):
            self._miss(base, recipe_id, "identity_mismatch")
            self._note(base, recipe_id)
            return None
        if run.status == RunStatus.OK:
            if outcome.state is None or outcome.evidence is None:
                self._miss(base, recipe_id, "ok_without_output")
                self._note(base, recipe_id)
                return None
            if outcome.evidence.meter_version != self.meter_version:
                self._miss(base, recipe_id, "stale_meter")
                self._note(base, recipe_id)
                return None
            try:
                pilot.require_clean_properties(outcome.evidence, self.start_evidence[base.design])
            except pilot.PilotSafetyStop:
                self._miss(base, recipe_id, "unclean_properties")
                self._note(base, recipe_id)
                return None
            if not self.amd.passed(base.design, outcome.state.id):
                self._miss(base, recipe_id, "no_amd_pass")
                self._note(base, recipe_id)
                return None
        elif run.status == RunStatus.TRANSIENT:
            self._miss(base, recipe_id, "transient")
            self._note(base, recipe_id)
            return None
        self.accepted_roots[root_name.split(":")[0]] += 1
        self.memo[ident] = outcome
        return outcome


class UnionTelemetryProvider(SavedTelemetryProvider):
    """SavedTelemetryProvider whose producing-run lookups and report search span the union roots."""

    def __init__(self, store: UnionStore) -> None:
        super().__init__(V4)
        self.store = store
        self.root = PROJECT / "data"  # all roots live under data/; used for the containment check
        self.failures: Counter[str] = Counter()

    def _report(self, state: DesignState, evidence: Evidence, run_id: str | None) -> Path | None:
        candidates: list[Path] = []
        if run_id is not None:
            if not _RUN.fullmatch(run_id):
                raise ValueError("producer run ID is invalid")
            outcome = self.store.load_outcome(run_id)
            if outcome.state is None or outcome.state.id != state.id:
                raise ValueError("producer outcome differs from input state")
            ids = [run_id]
            if outcome.run.original_run_id is not None:
                ids.append(outcome.run.original_run_id)
            for item in ids:
                if not _RUN.fullmatch(item):
                    raise ValueError("original run ID is invalid")
                run_dir = self.store.run_dir(item)
                if run_dir is not None:
                    candidates.append(run_dir / "reports" / "final_utilization.rpt")
        if evidence.raw_report_path:
            candidates.append(Path(evidence.raw_report_path) / "final_utilization.rpt")
        for path in candidates:
            if path.is_file():
                resolved = path.resolve(strict=True)
                if resolved.is_relative_to(self.root):
                    return resolved
        return None


# ---------------------------------------------------------------------------------------------- briefs
def _term(brief_id: str, metric: MetricName, comparison: str) -> BriefTerm:
    return BriefTerm(
        id=f"{brief_id.lower()}_limit_{metric.value}", metric=metric,
        comparison=comparison, factor=1.0, offset=0.0,
    )


def _brief(brief_id: str, metric: MetricName, direction: str, terms, budget: int) -> Brief:
    return Brief(
        id=brief_id, objective=Objective(metric=metric, direction=direction), terms=tuple(terms),
        budget_runs=budget, require_routed=True, max_drc_errors=0, require_constraint_match=True,
        require_clock_match=True, equivalence_policy="final_only",
    )


BRIEF_IDS = ("P0", "L2", "W2")
# Tolerance variants of the two new briefs (suffix K, F, R): the power limit becomes "<= start + tolerance"
# instead of "<= start". K = matched placement-keeping band (C1 envelope), F = matched re-placing band (C2
# envelope), R = recorded design-wide band; each floored at the 1 mW reporting step. Used only to measure how
# much of the strict-limit result is power measurement noise; none of them is proposed as a frozen brief.
TOL_KINDS = ("K", "F", "R")
BRIEF_TEXT = {
    "P0": "minimise dynamic power; keep WNS and packed LUTs at the start (frozen brief P0)",
    "L2": "minimise packed LUTs; keep WNS and dynamic power at the start",
    "W2": "maximise WNS; keep packed LUTs and dynamic power at the start",
}


def split_cell_brief(cell_brief: str) -> tuple[str, str]:
    """'L2K' -> ('L2', 'K'); 'P0' -> ('P0', '')."""

    base, kind = cell_brief[:2], cell_brief[2:]
    if base not in BRIEF_IDS or (kind and (base == "P0" or kind not in TOL_KINDS)):
        raise ValueError(f"unknown candidate brief {cell_brief}")
    return base, kind


def power_tolerance_w(design: str, kind: str, noise_table: dict) -> float:
    if not kind:
        return 0.0
    row = noise_table["designs"][design]
    b2, floor = row["b2_power_w"], 0.001
    if kind == "R":
        return row["recorded_N_pct"] / 100.0 * b2
    return max(row["C1" if kind == "K" else "C2"]["envelope_pct"] / 100.0 * b2, floor)


def candidate_brief(cell_brief: str, budget: int = 6, power_tol_w: float = 0.0) -> Brief:
    """P0 is the frozen brief; L2 and W2 use the frozen start-relative limits (factor 1) with offset 0 unless a
    power tolerance (watts) is given."""

    base, kind = split_cell_brief(cell_brief)
    M = MetricName

    def power_term() -> BriefTerm:
        return BriefTerm(id=f"{cell_brief.lower()}_limit_{M.DYNAMIC_POWER_W.value}", metric=M.DYNAMIC_POWER_W,
                         comparison="<=", factor=1.0, offset=float(power_tol_w))

    if base == "P0":
        return load_frozen_brief("P0").model_copy(update={"budget_runs": budget})
    if base == "L2":
        return _brief(cell_brief, M.PACKED_LUTS, "min", [_term(cell_brief, M.WNS_NS, ">="), power_term()], budget)
    return _brief(cell_brief, M.WNS_NS, "max", [_term(cell_brief, M.PACKED_LUTS, "<="), power_term()], budget)


# ------------------------------------------------------------------------------------------------ context
def build_workers(registry) -> dict[str, Any]:
    sets = load_recipe_sets(PROJECT / "configs/recipe_sets.yaml")
    workers: dict[str, Any] = {
        wid: FittedWorker(
            load_worker_card(PROJECT / "configs/workers" / f"{wid}.yaml"), sets.worker_sets[wid], registry,
        )
        for wid in ("timing", "area")
    }
    if any(w.model_label != "FINAL" for w in workers.values()):
        raise ValueError("the study requires both published FINAL workers")
    power = load_recipe_sets(PROJECT / "configs/power_recipe_set.yaml")
    workers["power"] = GenericWorker(
        load_worker_card(PROJECT / "configs/workers/power.yaml"), power.worker_sets["power"],
    )
    return workers


def build_context(include_new: bool = True, roots: dict[str, Path] | None = None,
                  amd_lenient: bool = False, twin_alias: bool = False) -> dict[str, Any]:
    plan = jread(V4 / "plan.json")
    meter = plan["meter_version"]
    b2 = dict(plan["training_inputs"])
    registry = load_registry(PROJECT)
    index = UnionIndex(roots if roots is not None else union_roots(include_new))
    store = UnionStore(index)
    amd = UnionAmd(index, b2, lenient=amd_lenient)
    starts_evidence = fa.start_evidence()
    alias = None
    if twin_alias:
        import pm_twin_classes as tc  # local import: pm_twin_classes imports this module
        uf, runs = tc.build({"index": index, "b2": b2})
        alias = TwinAlias(uf, runs)
    oracle = UnionOracle(registry, meter, index, store, amd, starts_evidence, alias)
    provider = UnionTelemetryProvider(store)
    starts = {}
    for design in DESIGNS:
        sid = b2[design]
        state = DesignState.model_validate_json((V4 / "states" / sid / "state.json").read_bytes())
        if state.id != sid or state.design != design or state.parent_id is not None or state.fake:
            raise ValueError(f"B2 identity differs: {design}")
        starts[design] = (store.normalised(state), starts_evidence[design])
    return {
        "plan": plan, "meter": meter, "b2": b2, "registry": registry, "workers": build_workers(registry),
        "index": index, "store": store, "amd": amd, "oracle": oracle, "provider": provider,
        "starts": starts, "noise": load_noise(PROJECT / "configs/noise.yaml"),
    }


class TracingSimulator(Simulator):
    """Simulator that tells the oracle the cost and worker of the mandate being applied."""

    def apply(self, node: SimNode, mandate):
        oracle = self.oracle
        if hasattr(oracle, "ctx"):
            oracle.ctx = (node.used, mandate.settings.max_runs, mandate.worker_id)
        return super().apply(node, mandate)


def make_sim(ctx: dict[str, Any], design: str, brief: Brief, workers: dict[str, Any] | None = None) -> Simulator:
    state, evidence = ctx["starts"][design]
    noise = ctx["noise"].for_start(design, evidence)
    return TracingSimulator(
        brief=brief, start_state=state, start_evidence=evidence,
        workers=workers if workers is not None else ctx["workers"], registry=ctx["registry"],
        oracle=ctx["oracle"], provider=ctx["provider"], noise=noise,
    )
