"""Template v2.2 feature grid (planned, NOT launched): shared constants and the policy factory.

The pinned D6-v2 driver (scripts/d6v2_2026-11-12/d6_common.py) is imported as a library and never edited. Before it is
imported, its work folder is pointed at this grid's own folder (data/v22_2026-11-13/grid, or .../smoke when V22_SMOKE=1);
its caps, plan note, manifests and telemetry seeds are then set from here, so every path, ledger, cache and log of the grid
is separate from D6-v2's. Training and validation designs only; the guard of d6_common refuses any other design.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT = Path(os.environ.get("ORCH_PROJECT_ROOT") or Path(__file__).resolve().parents[2])
D6_ROOT = PROJECT / "data/d6v2_2026-11-12"
IS_SMOKE = os.environ.get("V22_SMOKE") == "1"
GRID_ROOT = PROJECT / "data/v22_2026-11-13"
os.environ.setdefault("D6_WORK", str(GRID_ROOT / ("smoke" if IS_SMOKE else "grid")))
os.environ.pop("D6_SMOKE", None)                    # the grid sets its own caps below
sys.path.insert(0, str(PROJECT / "scripts/d6v2_2026-11-12"))

import d6_common as _d6  # noqa: E402
from d6_common import *  # noqa: E402,F401,F403  (re-exported: paths, readers, scopes, DescribeSource, ...)

from orch.policies.v22_switches import GRID_VARIANTS, VARIANTS  # noqa: E402

LLM_APPROACHES = tuple(GRID_VARIANTS)
# The authors' decision of 2026-11-13 21:45: T0 and A0 only; power plays no part in choosing a version (the power briefs, and the
# blocked W2R and L2R rows of the first plan, are not in this grid).
BRIEFS_MAIN = ("T0", "A0")
CARRYOVER_APPROACH = {"V0": "F1", "V1": "v21"}          # D6-v2 rows with the same policy, loop, seeds and designs
PLAN_NOTE = PROJECT / ("notes/v22_smoke_plan.md" if IS_SMOKE else "notes/v22_grid_plan_r2.md")
# Caps of the decisions message: 300 new runs, 120 job-hours, US$ 5 (tokens: far above what US$ 5 buys, so US$ binds first).
CAPS = ({"new_real_runs": int(os.environ.get("V22_SMOKE_RUNS", "0")), "job_hours": 2, "tokens": 2_000_000, "usd": 0.05}
        if IS_SMOKE else {"new_real_runs": 300, "job_hours": 120, "tokens": 30_000_000, "usd": 5.0})
MAX_TOOL_JOBS = 15      # an episode runs its tool jobs one at a time, so at most 15 episodes in flight = at most 15 tool jobs
_d6.PLAN_NOTE, _d6.CAPS = PLAN_NOTE, CAPS
# The union manifests of D6-v2 are read, never rebuilt or written. D6-v2's own 91 new runs sit in its overlay engines and are
# not in them, so the grid re-runs those states if it needs them (the estimate assumes this; see notes/v22_grid_plan.md).
_d6.MANIFEST_TRAIN, _d6.MANIFEST_DEV = D6_ROOT / "manifest_train", D6_ROOT / "manifest_dev"
_d6.READ_ONLY_TELEMETRY = (*_d6.READ_ONLY_TELEMETRY, D6_ROOT / "telemetry_cache")
MANIFEST_TRAIN, MANIFEST_DEV = _d6.MANIFEST_TRAIN, _d6.MANIFEST_DEV


SITUATION_TABLE = "configs/experience/situations_v22.json"


def episode_id(design: str, brief: str, approach: str, tag: str) -> str:
    return f"v22_{design}_{brief}_{approach}_{tag}"


def make_policy_v22(ctx, task: dict, worker_ids: list[str], describe=None):
    """V0: the D6-v2 F1 policy; V1: the D6-v2 template v2.1 policy; V2 to V8: V22Policy with the variant's switches."""

    import yaml
    from orch.policies.llm import LLMPolicyConfig
    from orch.policies.llm_client import TelemetrySink
    from orch.policies.v2_experience import load_calls_v2
    from orch.policies.v2_prompt import Variant, load_assets
    from orch.policies.v22_policy import V22Policy
    from orch.policies.v22_prompt import V22Context
    from orch.policies.v22_situation import load_rows
    from orch.policies.v22_switches import load_switch_texts

    approach = task["approach"]
    if approach == "V0":
        return _d6.make_policy(ctx, dict(task, approach="F1"), worker_ids, describe=None)
    if approach == "V1":
        return _d6.make_policy(ctx, dict(task, approach="v21"), worker_ids, describe=describe)
    switches = VARIANTS[approach]
    llm = _d6.make_llm(task, ctx.caps)
    sink = TelemetrySink(_d6.TELEMETRY)
    cfg = _d6._llm_settings()
    definition = yaml.safe_load(_d6.V21_DEFINITION.read_text(encoding="utf-8"))
    variant = Variant(definition["name"], tuple(definition["packages"]), wording=definition["worker_wording"])
    assets = load_assets(PROJECT)
    if assets.templates[variant.combo].sha256 != definition["template_sha256"]:
        raise SystemExit("the v2.1 template differs from its frozen definition")
    calls, _sha = load_calls_v2(PROJECT / definition["experience_calls"])
    situations, _sit_sha = load_rows(PROJECT / SITUATION_TABLE)
    context = V22Context(load_switch_texts(PROJECT), [], candidate_seed=_d6.SEEDS[task["tag"]], situations=situations)
    config = LLMPolicyConfig(name=f"template_v2.2_{approach}", mode="fuzzy", profiles_on=False,
                             template=assets.templates["base"], max_repairs=cfg["max_repair_turns"])
    return V22Policy(config, llm, sink, variant=variant, assets=assets, design=task["design"], describe=describe,
                     r2_experience=None, calls_v2=calls, switches=switches, context=context)
