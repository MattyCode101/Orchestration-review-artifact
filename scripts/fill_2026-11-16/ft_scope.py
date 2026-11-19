"""Tier 2 (item d) scope: the five TRAINING designs for the development designs' exhaustive two-step procedure (scripts/development_designs_2026-11-12/dd_*.py,
imported and copied with substitutions by ft_make_scripts.py; the originals are only read).

What is different from dd_scope.py, and nothing else:
  * designs: the 5 TRAINING designs; start states and the stored graph: the fill-run TRAINING manifest (fill_2026-11-16/manifest_train: the D6-v2 roots, the power
    review and post-freeze engines), the same one the tier-1 queue reads;
  * noise: scripts/test_r2 ... tr_common.noise_for (configs/noise_round2_trainpower.yaml, as every TRAINING episode of this folder);
  * briefs: tr_common.brief_for (the frozen T0, A0, P0 and the banded W2R, L2R of the TRAINING power band);
  * run folder: data/test_r2_2026-11-13/fill_2026-11-16/tier2_exhaustive/ (new outputs only there); caps 600 new runs, 150 job-hours, no tokens, no US$ (item d),
    bound to notes/test_r2_fill_tier2_plan.md; a new AMD validation is charged to the job-second cap (as dd_scope.ChargingLedger).
No TEST design is opened, t2_or1200 never.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

os.environ["TR_QUEUE"] = "fl"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tr_common as tc  # noqa: E402  (the fill-run copy)

PROJECT = tc.PROJECT
sys.path.insert(0, str(PROJECT / "scripts/development_designs_2026-11-12"))
import dd_common as dd  # noqa: E402
import campaign_harness as h  # noqa: E402
from orch.contract.models import DesignState  # noqa: E402

TIER2 = tc.ROOT / "tier2_exhaustive"
PLAN_NOTE = PROJECT / "notes/test_r2_fill_tier2_plan.md"
CAPS = {"new_real_runs": 600, "job_hours": 150, "tokens": 0, "usd": 0.0}
DESIGNS = tuple(tc.TRAINING)
BRIEFS = ("W2R", "L2R", "P0")


def plan_sha() -> str:
    return hashlib.sha256(PLAN_NOTE.read_bytes()).hexdigest()


MANIFEST_T2 = tc.ROOT / "manifest_train_t2"      # manifest_train plus the tier-1 TRAINING engine (fl_manifest2.py train_t2), built 16 Nov 17:2x UTC, so no pair tier 1 ran is run again


def manifest():
    return tc._manifest(MANIFEST_T2)


def start_ids() -> dict[str, str]:
    body = manifest().body
    return {it["design"]: sid for sid, it in body["start_evidence"].items() if it["design"] in DESIGNS}


def start_id(design: str) -> str:
    return start_ids()[tc.guard_design(design)]


# the development designs' helpers read their start states and run folder from dd_common: point them at the TRAINING scope (this process only)
dd.start_id = start_id
dd.WORK = TIER2


def build(mode: str, with_provider: bool = False) -> h.Context:
    """mode 'run': real runs against the caps (as dd_scope.build)."""

    TIER2.mkdir(parents=True, exist_ok=True)
    caps = h.CapLedger(TIER2, CAPS, plan_sha()) if mode == "run" else None
    man = manifest()
    starts = h._starts_from_manifest(man)
    if set(starts) != set(tc.TRAINING):
        raise SystemExit("the TRAINING manifest does not hold exactly its 5 starts")
    scope = h.Scope("train", tc.TRAINING, man, TIER2, starts, tc.noise_for, None)
    h.ReadThroughLedger = tc._charging_ledger_class()
    ctx = h.build_context(scope, mode=mode, caps=caps)
    ctx.ledger.caps = caps
    return ctx


def load_state(ctx: h.Context, state_id: str) -> DesignState:
    path = ctx.store.state_path(state_id)
    state = DesignState.model_validate_json((path.parent / "state.json").read_bytes())
    if state.id != state_id or str(state.path) != str(path):
        raise SystemExit(f"state record differs from its location: {state_id}")
    return state
