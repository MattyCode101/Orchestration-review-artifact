"""Round-2 brief loader (written 2026-11-13, test-run preparation). New file; no frozen file is edited.

Reads configs/briefs_round2_final.yaml and builds the seven round-2 briefs (T0, A0, T, A, W2R, L2R, P0) as contract Brief objects.

  * T0, A0, T, A and P0 are the frozen briefs themselves (orch.contract.brief_text.load_frozen_brief): same objects, same
    fingerprints. Their blocks in the round-2 file must equal the frozen YAML's blocks, or loading is refused.
  * W2R and L2R are built exactly as scripts/development_designs_2026-11-12/dd_common.make_brief builds them (the power study's
    candidate brief, tolerance kind R): the WNS or LUT limit stays strict and dynamic power is limited to `<= start + band`,
    a '<=' term with factor 1.0 and an offset equal to the design's dynamic_power_w band in W (inclusive).
  * The round-2 file and the TEST noise file are hash-checked against notes/round2_briefs_freeze.sha256; no freeze list, no briefs.

usage (library): brief_for(design, brief_id) -> Brief ; power_band_w(design) -> float ; briefs_for(design) -> dict
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from orch.config import load_noise
from orch.contract.brief_text import load_frozen_brief, load_frozen_spec
from orch.contract.models import Brief, BriefTerm, MetricName as M, Objective

PROJECT = Path(__file__).resolve().parents[2]
FINAL = PROJECT / "configs/briefs_round2_final.yaml"
FREEZE = PROJECT / "notes/round2_briefs_freeze.sha256"
NOISE_TEST = PROJECT / "configs/noise_round2_test.yaml"
NOISE_BASE = PROJECT / "configs/noise_round2.yaml"
BRIEF_IDS = ("T0", "A0", "T", "A", "W2R", "L2R", "P0")
FROZEN_IDS = ("T0", "A0", "T", "A", "P0")
BANDED = ("W2R", "L2R")
THREE_WORKER = ("W2R", "L2R", "P0")
TEST_DESIGNS = (
    "rosetta_digit-recognition", "corescore_500_mod", "vexriscv_re-place_v2", "fir_symmetric_systolic",
    "fir_systolic_transposed",
)
OBJECTIVE = {"T0": ("wns_ns", "max"), "T": ("wns_ns", "max"), "W2R": ("wns_ns", "max"),
             "A0": ("packed_luts", "min"), "A": ("packed_luts", "min"), "L2R": ("packed_luts", "min"),
             "P0": ("dynamic_power_w", "min")}
_cache: dict[str, Any] = {}


class BriefFreezeError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze_entries() -> dict[str, str]:
    if not FREEZE.is_file():
        raise BriefFreezeError("notes/round2_briefs_freeze.sha256 does not exist: the round-2 briefs are not frozen")
    out = {}
    for line in FREEZE.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[1]] = parts[0]
    return out


def check_freeze() -> dict[str, str]:
    """Every file named in the freeze list equals its recorded hash (refuses otherwise)."""

    entries = _freeze_entries()
    for rel in ("configs/briefs_round2_final.yaml", "configs/noise_round2_test.yaml"):
        if rel not in entries:
            raise BriefFreezeError(f"{rel} is not in the freeze list")
    for rel, digest in entries.items():
        if _sha(PROJECT / rel) != digest:
            raise BriefFreezeError(f"{rel} differs from its frozen hash")
    return entries


def load_spec() -> dict[str, Any]:
    if "spec" not in _cache:
        check_freeze()
        spec = yaml.safe_load(FINAL.read_text(encoding="utf-8"))
        frozen = load_frozen_spec()
        if tuple(spec["briefs"]) != BRIEF_IDS:
            raise BriefFreezeError("round-2 brief ids or order differ from the freeze")
        if spec["common"] != frozen["common"]:
            raise BriefFreezeError("the common block differs from configs/briefs.yaml")
        for brief_id in FROZEN_IDS:
            if spec["briefs"][brief_id] != frozen["briefs"][brief_id]:
                raise BriefFreezeError(f"{brief_id} differs from its frozen block in configs/briefs.yaml")
        _cache["spec"] = spec
    return _cache["spec"]


def power_band_w(design: str) -> float:
    """The design's dynamic_power_w band in W: the TEST noise file for TEST designs, noise_round2.yaml otherwise."""

    path = NOISE_TEST if design in TEST_DESIGNS else NOISE_BASE
    bands = load_noise(path).bands.get(design, {})
    if M.DYNAMIC_POWER_W not in bands:
        raise BriefFreezeError(f"no dynamic_power_w band for {design} in {path.name}")
    band = bands[M.DYNAMIC_POWER_W]
    if band.relative:
        raise BriefFreezeError("a relative power band is not supported here")
    return float(band.value)


def _term(brief_id: str, metric: M, comparison: str, offset: float = 0.0) -> BriefTerm:
    return BriefTerm(id=f"{brief_id.lower()}_limit_{metric.value}", metric=metric, comparison=comparison, factor=1.0,
                     offset=float(offset))


def _brief(brief_id: str, metric: M, direction: str, terms: list[BriefTerm], budget: int) -> Brief:
    return Brief(
        id=brief_id, objective=Objective(metric=metric, direction=direction), terms=tuple(terms), budget_runs=budget,
        require_routed=True, max_drc_errors=0, require_constraint_match=True, require_clock_match=True,
        equivalence_policy="final_only",
    )


def brief_for(design: str, brief_id: str) -> Brief:
    """The round-2 brief for one design (the design matters only for the power band of W2R and L2R)."""

    spec = load_spec()
    if brief_id not in BRIEF_IDS:
        raise ValueError(f"not a round-2 brief: {brief_id}")
    budget = int(spec["common"]["budget"]["recipe_runs"])
    if brief_id in FROZEN_IDS:
        return load_frozen_brief(brief_id).model_copy(update={"budget_runs": budget})
    band = power_band_w(design)
    if brief_id == "L2R":
        return _brief("L2R", M.PACKED_LUTS, "min",
                      [_term("L2R", M.WNS_NS, ">="), _term("L2R", M.DYNAMIC_POWER_W, "<=", band)], budget)
    return _brief("W2R", M.WNS_NS, "max",
                  [_term("W2R", M.PACKED_LUTS, "<="), _term("W2R", M.DYNAMIC_POWER_W, "<=", band)], budget)


def briefs_for(design: str) -> dict[str, Brief]:
    return {b: brief_for(design, b) for b in BRIEF_IDS}


def p0_gain_counts(design: str, start_power_w: float, final_power_w: float) -> bool:
    """A P0 power gain counts only if it is larger than the design's power band (inside the band = no change)."""

    return (float(start_power_w) - float(final_power_w)) > power_band_w(design)
