"""Development designs (D3, Thu 12 Nov 2026): shared helpers. Read-only on every stored root; nothing frozen is edited.

Stored outcomes come from the benchmark screen's per-design engine roots (benchmark_expansion_2026-11-10/data/engine/<design>),
imported through a hash-pinned manifest built with the campaign's own generic builder (scripts/campaign_manifest.py,
imported, not edited). New outputs go only to this task's own root (data/development_designs_2026-11-12/).

Never touched: t2_or1200 (sealed as a possible fresh test design), the five round-1 test designs, the sealed test power data.
"""
from __future__ import annotations
import os

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT = Path(os.environ.get("ORCH_PROJECT_ROOT") or Path(__file__).resolve().parents[2])
BX = PROJECT.parent / "benchmark_expansion_2026-11-10"
BXD = BX / "data"
for _p in (str(PROJECT), str(PROJECT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from orch.config import load_recipe_sets  # noqa: E402
from orch.contract.brief_text import load_frozen_brief  # noqa: E402
from orch.contract.models import Brief, BriefTerm, Evidence, MetricName, Objective  # noqa: E402
from orch.engine.gate import judge  # noqa: E402
from orch.engine.manifest_cache import CacheManifest, _read_run  # noqa: E402
from orch.engine.start_properties import PropertyRuleError, check_not_above_start  # noqa: E402

M = MetricName
WORK = PROJECT / "data/development_designs_2026-11-12"
CANDIDATES = ("t2_aes", "t2_cordic_s", "vtr_mcml_v2", "rosetta_optical-flow", "amd_mini-isp", "t2_raygentop")
FORBIDDEN = (
    "t2_or1200", "rosetta_digit-recognition", "corescore_500_mod", "vexriscv_re-place_v2",
    "fir_symmetric_systolic", "fir_systolic_transposed",
)
V4_PLAN = PROJECT / "data/pilots/thursday_overnight_train_v4_2026-11-05/plan.json"

_SETS = load_recipe_sets(PROJECT / "configs/recipe_sets.yaml")
_POWER = load_recipe_sets(PROJECT / "configs/power_recipe_set.yaml")
TIMING = tuple(_SETS.worker_sets["timing"])
AREA = tuple(_SETS.worker_sets["area"])
POWER = tuple(_POWER.worker_sets["power"])
RECIPES2 = TIMING + AREA                  # the two-worker pool (timing, area): 10 recipes
RECIPES3 = TIMING + AREA + POWER          # the three-worker pool: 14 recipes
WORKER_OF = {**{r: "timing" for r in TIMING}, **{r: "area" for r in AREA}, **{r: "power" for r in POWER}}

BRIEFS = ("T0", "A0", "W2R", "L2R", "P0")
TWO_WORKER = ("T0", "A0")
THREE_WORKER = ("W2R", "L2R", "P0")
OBJECTIVE = {
    "T0": ("wns_ns", "max"), "W2R": ("wns_ns", "max"),
    "A0": ("packed_luts", "min"), "L2R": ("packed_luts", "min"),
    "P0": ("dynamic_power_w", "min"),
}
UNIT = {"wns_ns": "ns", "packed_luts": "LUT", "dynamic_power_w": "W"}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def jread(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def guard_design(design: str) -> str:
    if design in FORBIDDEN:
        raise SystemExit(f"refused: {design} is a sealed or round-1 test design")
    return design


# ------------------------------------------------------------------------------ starts, bands, roots
def meter_version() -> str:
    return str(jread(V4_PLAN)["meter_version"])


def start_info(design: str) -> dict[str, Any]:
    return jread(BXD / "b2" / guard_design(design) / "start.json")


def start_id(design: str) -> str:
    return str(start_info(design)["b2_state_id"])


def screen_root(design: str) -> Path:
    return BXD / "engine" / guard_design(design)


def screen_bands(design: str) -> dict[str, float]:
    """The screen's effective band: max absolute difference of five controls to B2 (raised to the metric's resolution)."""

    b = jread(BXD / "noise" / guard_design(design) / "bands.json")["band_effective"]
    return {"wns_ns": float(b["wns_ns"]), "packed_luts": float(b["packed_luts"]), "dynamic_power_w": float(b["dynamic_power_w"])}


def power_offset(design: str) -> float:
    """Power limit offset in W for the banded briefs: the band, whole mW (the meter reports power to 1 mW)."""

    return round(screen_bands(design)["dynamic_power_w"], 3)


# ------------------------------------------------------------------------------ briefs
def _term(brief_id: str, metric: MetricName, comparison: str, offset: float = 0.0) -> BriefTerm:
    return BriefTerm(id=f"{brief_id.lower()}_limit_{metric.value}", metric=metric, comparison=comparison, factor=1.0, offset=float(offset))


def _brief(brief_id: str, metric: MetricName, direction: str, terms: Iterable[BriefTerm], budget: int) -> Brief:
    return Brief(
        id=brief_id, objective=Objective(metric=metric, direction=direction), terms=tuple(terms), budget_runs=budget,
        require_routed=True, max_drc_errors=0, require_constraint_match=True, require_clock_match=True,
        equivalence_policy="final_only",
    )


def make_brief(brief_id: str, power_band_w: float = 0.0, budget: int = 6) -> Brief:
    """T0, A0 and P0 are the frozen briefs. W2R and L2R are built exactly as the power study defines them
    (scripts/power_mediation_2026-11-10/pm_common.candidate_brief, tolerance kind R): the WNS or LUT limit stays strict,
    dynamic power at most start + the design's power band (inclusive)."""

    if brief_id in ("T0", "A0", "P0"):
        return load_frozen_brief(brief_id).model_copy(update={"budget_runs": budget})
    if brief_id == "L2R":
        return _brief("L2R", M.PACKED_LUTS, "min",
                      [_term("L2R", M.WNS_NS, ">="), _term("L2R", M.DYNAMIC_POWER_W, "<=", power_band_w)], budget)
    if brief_id == "W2R":
        return _brief("W2R", M.WNS_NS, "max",
                      [_term("W2R", M.PACKED_LUTS, "<="), _term("W2R", M.DYNAMIC_POWER_W, "<=", power_band_w)], budget)
    raise ValueError(brief_id)


def briefs_for(design: str) -> dict[str, Brief]:
    off = power_offset(design)
    return {b: make_brief(b, off) for b in BRIEFS}


# ------------------------------------------------------------------------------ manifest and graph
def manifest_roots(designs: Iterable[str]) -> dict[str, Path]:
    return {f"bx_{guard_design(d)}": screen_root(d) for d in designs}


def build_manifest(designs: Iterable[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    import campaign_manifest as cm  # noqa: E402  (imported, not edited)

    designs = tuple(designs)
    roots = manifest_roots(designs)
    starts = {d: start_id(d) for d in designs}
    return cm.build_generic(roots, meter_version(), starts, dict(roots))


class Graph:
    """The stored run graph over a manifest body: (input state, recipe) -> entry; outputs read checksum-verified."""

    def __init__(self, body: dict[str, Any], extra_roots: dict[str, Path] | None = None) -> None:
        self.body = body
        self.roots = {k: Path(v) for k, v in body["roots"].items()}
        self.by_in: dict[tuple[str, str], dict[str, Any]] = {}
        for e in body["entries"].values():
            key = (e["input_state_id"], e["recipe_id"])
            if key in self.by_in and self.by_in[key]["run_id"] != e["run_id"]:
                raise SystemExit(f"two runs for one (state, recipe) key: {key}")
            self.by_in[key] = e
        self._ev: dict[str, tuple[Any, Any]] = {}
        self._start: dict[str, Evidence] = {}

    def start_evidence(self, sid: str) -> Evidence:
        if sid not in self._start:
            item = self.body["start_evidence"][sid]
            self._start[sid] = Evidence.model_validate_json(Path(item["path"]).read_bytes())
        return self._start[sid]

    def outcome(self, entry: dict[str, Any]):
        rid = entry["run_id"]
        if rid not in self._ev:
            run, state, ev, _ = _read_run(self.roots[entry["root"]] / "runs" / rid, rid)
            self._ev[rid] = (run, state, ev)
        return self._ev[rid]

    def evidence(self, entry: dict[str, Any]) -> Evidence:
        return self.outcome(entry)[2]


def amd_records(roots: dict[str, Path], reference_of: dict[str, str]) -> dict[tuple[str, str], str]:
    """(design, output state) -> 'pass' | 'fail' | 'unknown' from the roots' verification records, only where the
    record's reference is the design's B2 (the contract's start). Where two records exist a pass wins."""

    out: dict[tuple[str, str], str] = {}
    rank = {"pass": 2, "fail": 1, "unknown": 0}
    for root in roots.values():
        for path in (Path(root) / "verification" / "records").glob("*/*/*.json"):
            rec = jread(path)
            d = rec.get("design")
            if d not in reference_of or rec.get("reference_sha256") != reference_of[d]:
                continue
            key = (d, rec["output_sha256"])
            if key not in out or rank[rec["result"]] > rank[out[key]]:
                out[key] = rec["result"]
    return out


# ------------------------------------------------------------------------------ validity, gains, classes
def is_valid(brief: Brief, ev: Evidence, start: Evidence) -> bool:
    try:
        check_not_above_start(ev, start)
    except PropertyRuleError:
        return False
    return bool(judge(brief, start, start, ev).brief_valid)


def metrics_of(ev: Evidence) -> dict[str, float]:
    return {n.value: m.value for n, m in ev.metrics.items()}


def gain_of(brief_id: str, start: Evidence, ev: Evidence) -> float:
    metric, direction = OBJECTIVE[brief_id]
    s, v = start.metric(M(metric)).value, ev.metric(M(metric)).value
    return (v - s) if direction == "max" else (s - v)


def head_class(bands_gain: float) -> str:
    """none: under 1 band; low: 1 to 3 bands; high: over 3 bands (Thursday plan, section 2)."""

    if bands_gain < 1.0:
        return "none"
    return "low" if bands_gain <= 3.0 else "high"


def reachable(g: Graph, sid: str, first: Iterable[str], second: Iterable[str]):
    """Distinct states within two steps of the start over the given recipe pools, using stored runs only.

    Returns (states, missing, failed): states[state_id] = {path, steps, entry}; a recipe whose output is a state already
    seen is not expanded again (its outputs are the same runs). missing lists (input state, recipe) pairs with no stored
    run; failed lists pairs whose stored run has no output."""

    first, second = tuple(first), tuple(second)
    states: dict[str, dict[str, Any]] = {sid: {"path": [], "steps": 0, "entry": None}}
    missing: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []
    frontier = [sid]
    for depth, pool in ((1, first), (2, second)):
        nxt: list[str] = []
        for parent in frontier:
            for r in pool:
                e = g.by_in.get((parent, r))
                if e is None:
                    missing.append((parent, r))
                    continue
                out = e.get("output_state_id")
                if e["status"] != "ok" or not out:
                    failed.append((parent, r))
                    continue
                if out in states:
                    continue
                states[out] = {"path": states[parent]["path"] + [r], "steps": depth, "entry": e, "parent": parent}
                nxt.append(out)
        frontier = nxt
    return states, missing, failed
