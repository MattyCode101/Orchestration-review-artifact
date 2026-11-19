"""Bandit (O1): shared paths, briefs, noise bands and worker cards. Training designs only.

Nothing frozen is edited. The L2R brief is rebuilt here exactly as the power study defines it (checked equal by
fingerprint in tests/test_bandit_policy.py); P0, T0, A0, T and A come from the frozen loader.
"""
from __future__ import annotations
import os

import json
import sys
from pathlib import Path

PROJECT = Path(os.environ.get("ORCH_PROJECT_ROOT") or Path(__file__).resolve().parents[2])
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from orch.config import load_worker_card  # noqa: E402
from orch.contract.brief_text import load_frozen_brief  # noqa: E402
from orch.contract.models import Brief, BriefTerm, MetricName, Objective, WorkerCard  # noqa: E402

M = MetricName
DATA = PROJECT / "data/bandit_2026-11-11"
CALLS = DATA / "calls" / "episodes_v4.jsonl"
PRIORS = DATA / "priors"
NOISE_TABLE = PROJECT / "data/power_mediation_2026-11-10/noise_table.json"
DESIGNS = ("vexriscv_re-place", "rosetta_spam-filter", "rosetta_3d-rendering", "logicnets_jscl", "finn_radioml")
BRIEFS = ("T0", "A0", "T", "A", "L2R", "P0")
FOUR = ("T0", "A0", "T", "A")
VARIANTS = ("main", "literal", "base_only", "reward_only")       # main = declared policy; the others are the 2 x 2 ablation
BUDGET = 6


def jread(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def power_band_w(design: str) -> float:
    """Recorded design-wide power noise band in watts: recorded_N_pct / 100 x B2 dynamic power (power study, kind R)."""

    row = jread(NOISE_TABLE)["designs"][design]
    return row["recorded_N_pct"] / 100.0 * row["b2_power_w"]


def l2r_brief(band_w: float, budget: int = BUDGET) -> Brief:
    """Minimise packed LUTs; WNS not below start; dynamic power at most start + the recorded band (inclusive)."""

    terms = (
        BriefTerm(id="l2r_limit_wns_ns", metric=M.WNS_NS, comparison=">=", factor=1.0, offset=0.0),
        BriefTerm(id="l2r_limit_dynamic_power_w", metric=M.DYNAMIC_POWER_W, comparison="<=", factor=1.0,
                  offset=float(band_w)),
    )
    return Brief(
        id="L2R", objective=Objective(metric=M.PACKED_LUTS, direction="min"), terms=terms, budget_runs=budget,
        require_routed=True, max_drc_errors=0, require_constraint_match=True, require_clock_match=True,
        equivalence_policy="final_only",
    )


def make_brief(brief_id: str, design: str) -> Brief:
    if brief_id == "L2R":
        return l2r_brief(power_band_w(design))
    return load_frozen_brief(brief_id).model_copy(update={"budget_runs": BUDGET})


def cards_for(brief_id: str) -> tuple[WorkerCard, ...]:
    """Two workers for the four timing and area briefs, three for the power briefs (as in the campaign harness)."""

    names = ("timing", "area", "power") if brief_id in ("L2R", "P0") else ("timing", "area")
    return tuple(load_worker_card(PROJECT / "configs/workers" / f"{name}.yaml") for name in names)


def design_noise(records: list[dict] | None = None) -> dict[str, dict[MetricName, float]]:
    """Noise bands per design: the power study's (data/bandit_2026-11-11/design_bands.json).

    configs/noise.yaml has no logicnets entry, so the journals record none for it and the frozen training frame has a
    zero band there; the power study fills it from the run-3 Family N range. Wherever a journal does record a band, it
    must equal this table (checked when records are given)."""

    table = jread(DATA / "design_bands.json")["bands"]
    out = {
        design: {M.WNS_NS: float(row["wns_ns"]), M.PACKED_LUTS: float(row["packed_luts"]),
                 M.DYNAMIC_POWER_W: float(row["power_recorded_w"])}
        for design, row in table.items()
    }
    for rec in records or ():
        for key, value in rec["noise"].items():
            if key in {m.value for m in M} and M(key) in out[rec["design"]]:
                want = out[rec["design"]][M(key)]
                if abs(want - value) > 1e-6 * max(1.0, abs(want)):
                    raise ValueError(f"journal band {key} of {rec['design']} differs from the power study's table")
    return out
