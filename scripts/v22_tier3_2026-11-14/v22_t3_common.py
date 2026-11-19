"""Template v2.2 tier 3 (decisions of 2026-11-14 11:30): shared constants and the policy factory (new file).

The grid's own library (scripts/v22_2026-11-13/v22_common.py, launched, never edited) is imported after the work folder is
pointed at tier 3's own folder (data/v22_2026-11-14/tier3), so every path, ledger, cache, journal and log of tier 3 is separate
from the grid's and from D6-v2's (both stay read-only). Its caps, plan note and seeds are then replaced by tier 3's.
TRAINING and VALIDATION designs only; the guard of d6_common refuses any other design.

Rows of tier 3 (16 cells, 3 repeats each, tags t1 to t3 = seeds 3, 4, 5; D6-v2 and the grid used seeds 0, 1, 2):
  V0   F1-R1                      same-day re-run (the D6-v2 policy, loop and wording)
  V1   template v2.1              same-day re-run, through the V22Policy path with the default switches (current renderer path)
  V9   v2.1 without P, with U and S'2
  V10  V9 with E off
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[1]
os.environ["D6_WORK"] = str(_PROJECT / "data/v22_2026-11-14" / "tier3")      # forced: tier 3 never writes in the grid's folder
os.environ.pop("V22_SMOKE", None)
os.environ.pop("D6_SMOKE", None)
sys.path.insert(0, str(_PROJECT / "scripts/v22_2026-11-13"))

import v22_common as _g  # noqa: E402  (the grid's library: launched, read-only)
from v22_common import *  # noqa: E402,F401,F403

_d6 = _g._d6
PROJECT = _PROJECT
GRID_ROOT = PROJECT / "data/v22_2026-11-13"
TIER3_ROOT = PROJECT / "data/v22_2026-11-14"
if Path(_d6.WORK) != TIER3_ROOT / "tier3":
    raise SystemExit(f"the work folder is {_d6.WORK}, not tier 3's own folder")

from orch.policies.v22_switches import VARIANTS  # noqa: E402
from orch.policies.v22_tier3 import SITUATION_TABLE_LLM, T3_VARIANTS, TIER3_SEEDS  # noqa: E402

VARIANTS_T3 = ("V0", "V1", "V9", "V10")
TAGS_T3 = tuple(TIER3_SEEDS)
LLM_APPROACHES = VARIANTS_T3
PLAN_NOTE = PROJECT / "notes/v22_tier3_plan.md"
# Caps of the decisions message of 2026-11-14 11:30: 200 episodes, 150 new runs, 60 job-hours, US$ 3.
CAPS = {"new_real_runs": 150, "job_hours": 60, "tokens": 30_000_000, "usd": 3.0}
EPISODE_CAP = 200
MAX_TOOL_JOBS = 15
_d6.PLAN_NOTE, _d6.CAPS = PLAN_NOTE, CAPS
_d6.SEEDS.update(TIER3_SEEDS)
# The finished grid's extraction records are read, never written (same rule as D6-v2's records for the grid).
_d6.READ_ONLY_TELEMETRY = (*_d6.READ_ONLY_TELEMETRY, GRID_ROOT / "grid" / "telemetry_cache")


def episode_id(design: str, brief: str, approach: str, tag: str) -> str:
    return f"v22t3_{design}_{brief}_{approach}_{tag}"


def make_policy_v22(ctx, task: dict, worker_ids: list[str], describe=None):
    """V0: the D6-v2 F1 policy. V1 (template v2.1), V9 and V10: V22Policy with the variant's switches, which is the current
    renderer path (with the default switches its prompt equals the frozen v2.1 render byte for byte). V9 and V10 read the S'2
    table and state its population in one line; no other variant reads it."""

    import yaml
    from orch.policies.llm import LLMPolicyConfig
    from orch.policies.llm_client import TelemetrySink
    from orch.policies.v2_experience import load_calls_v2
    from orch.policies.v2_prompt import Variant, load_assets
    from orch.policies.v22_policy import V22Policy
    from orch.policies.v22_prompt import V22Context
    from orch.policies.v22_situation import load_rows
    from orch.policies.v22_switches import load_switch_texts
    from orch.policies.v22_tier3 import situation_texts_llm

    approach = task["approach"]
    if approach == "V0":
        return _d6.make_policy(ctx, dict(task, approach="F1"), worker_ids, describe=None)
    if approach not in VARIANTS_T3:
        raise SystemExit(f"tier 3 has no variant {approach}")
    switches = {**VARIANTS, **T3_VARIANTS}[approach]
    llm = _d6.make_llm(task, ctx.caps)
    sink = TelemetrySink(_d6.TELEMETRY)
    cfg = _d6._llm_settings()
    definition = yaml.safe_load(_d6.V21_DEFINITION.read_text(encoding="utf-8"))
    variant = Variant(definition["name"], tuple(definition["packages"]), wording=definition["worker_wording"])
    assets = load_assets(PROJECT)
    if assets.templates[variant.combo].sha256 != definition["template_sha256"]:
        raise SystemExit("the v2.1 template differs from its frozen definition")
    calls, _sha = load_calls_v2(PROJECT / definition["experience_calls"])
    texts = load_switch_texts(PROJECT)
    if approach in T3_VARIANTS:
        situations, _sit_sha = load_rows(PROJECT / SITUATION_TABLE_LLM)
        texts = situation_texts_llm(texts, PROJECT)
    else:
        situations, _sit_sha = load_rows(PROJECT / _g.SITUATION_TABLE)
    context = V22Context(texts, [], candidate_seed=_d6.SEEDS[task["tag"]], situations=situations)
    config = LLMPolicyConfig(name=f"template_v2.2_{approach}", mode="fuzzy", profiles_on=False,
                             template=assets.templates["base"], max_repairs=cfg["max_repair_turns"])
    return V22Policy(config, llm, sink, variant=variant, assets=assets, design=task["design"], describe=describe,
                     r2_experience=None, calls_v2=calls, switches=switches, context=context)
