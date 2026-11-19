"""Power readiness (Wed 2026-11-11): shared setup. Training designs only; no test data; nothing frozen edited.

Imports the power-mediation study's pm_common / pm_tables / pm_search read-only (scripts/power_mediation_2026-11-10
is not edited). What is new here:
  1. this task's own engine root (new runs only) and the union that adds it to the study's roots;
  2. the power-band rule: L2R = minimise packed LUTs; WNS not below start; dynamic power at most start + the
     design's recorded power noise band (selection.json noise_power_pct / 100 x B2 power). It is the study's
     tolerance kind R. WNS and LUT limits stay strict. P0 is the frozen brief (no power limit);
  3. a cap ledger (100 runs, 30 job-hours) kept apart from the study's.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PM_DIR = Path(__file__).resolve().parents[1] / "power_mediation_2026-11-10"
if str(PM_DIR) not in sys.path:
    sys.path.insert(0, str(PM_DIR))

import pm_common as pm  # noqa: E402
import pm_tables  # noqa: E402

PROJECT = pm.PROJECT
WORK = PROJECT / "data/power_readiness_2026-11-11"
NEW_ROOT = WORK / "engine"
DESIGNS = tuple(pm.DESIGNS)
M = pm.MetricName
BUDGET = 6
CELL_BRIEFS = ("L2R", "P0", "W2R")   # L2R and P0 are the task; W2R only answers "does training show room for a timing brief"
CAP_RUNS = 100
CAP_SECONDS = 30 * 3600
SHORT = {"vexriscv_re-place": "vexriscv", "logicnets_jscl": "logicnets", "rosetta_spam-filter": "spam-filter",
         "rosetta_3d-rendering": "3D-rendering", "finn_radioml": "finn"}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def jread(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def roots() -> dict:
    """The study's union (sealed v4, ten power roots, the study's own engine) plus this task's engine root."""

    r = pm.union_roots(include_new=True)
    if (NEW_ROOT / "cache").is_dir():
        r["new_pr"] = NEW_ROOT
    return r


def build_ctx(twin: bool = True, lenient: bool = True) -> dict:
    return pm.build_context(include_new=True, roots=roots(), amd_lenient=lenient, twin_alias=twin)


_NT = None


def noise_table() -> dict:
    global _NT
    if _NT is None:
        _NT = pm.jread(pm.WORK / "noise_table.json")
    return _NT


def tol_w(design: str) -> float:
    """Recorded power noise band in watts: noise_power_pct / 100 x B2 dynamic power (kind R of the study)."""

    return pm.power_tolerance_w(design, "R", noise_table())


_BANDS = None


def bands() -> dict:
    global _BANDS
    if _BANDS is None:
        _BANDS = pm_tables.objective_bands(noise_table())
    return _BANDS


def band(brief_id: str, design: str) -> float:
    """Noise band of the brief's objective, in the objective's unit (LUTs, W, ns)."""

    b = bands()[design]
    return {"L2R": b["packed_luts"], "L2": b["packed_luts"], "P0": b["power_recorded_w"],
            "W2R": b["wns_ns"], "W2": b["wns_ns"]}[brief_id]


TOL_DELTA_W = 0.0   # sensitivity only: shifts the power tolerance (e.g. -0.0005 makes the limit exclusive of the band itself)


def make_brief(brief_id: str, design: str, budget: int = BUDGET):
    """L2R / W2R carry the banded power limit (offset = recorded band, in W); P0 is the frozen brief."""

    return pm.candidate_brief(brief_id, budget, 0.0 if brief_id == "P0" else tol_w(design) + TOL_DELTA_W)


def unit(brief_id: str) -> tuple[str, float]:
    """(unit label, factor from the stored unit to the label's unit)."""

    return {"L2R": ("LUT", 1.0), "P0": ("mW", 1000.0), "W2R": ("ps", 1000.0)}[brief_id]
