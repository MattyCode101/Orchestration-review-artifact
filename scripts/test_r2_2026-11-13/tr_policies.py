"""Test-run driver: policy rows (plug-ins) and the default brief wording for W2R and L2R.

A row is a name in the task list (approach). make_policy(ctx, task, worker_ids, describe) returns the policy object the round-2 loop runs.
  v1, v2            goal-repair loop (orch.policies.repair_loop; v2 = the round-1 selected parameters, g05)
  fixed             best fixed order chosen on TRAINING only (task['fixed_order'], declared in notes/round2_fixed_orders.json):
                    two-worker briefs: alternate timing and area ("BA" = area first); power briefs: a permutation of A, B, C
  random_0..2       seeded uniform random (relevant workers for the power briefs)
  grid              six single recipes ranked by mean TRAINING gain for the brief's goal, run once each from the start, as direct
                    recipe mandates (orch.baselines.direct_recipe_worker); declared priors data/beam_2026-11-13/priors_{ab,abc}.json
  bandit            the frozen learned policy (orch.policies.bandit_frozen); W2R is not applicable
  F1-R1, v21, GA-R1 the LLM rows (Gemini flash-lite or an the gateway model through the task's `model` field)
  V2..V8            template v2.2 variants of the development run's grid (orch.policies.v22_policy.V22Policy with the variant's switches;
                    configs/policies/template_v2.2_r2/V*.yaml); same loop, model plug-in and replay cache as v21
The beam is a planner with its own simulator over the stored graph (scripts/beam_2026-11-13) and is not a row of this loop.
"""
from __future__ import annotations

import json
import hashlib

import tr_common as tc

import campaign_harness as h
import yaml
from orch.config import load_recipe_sets
from orch.contract.models import Brief, Mandate, MetricName, WorkerSettings
from orch.contract.views import PolicyView

ROWS = ("v1", "v2", "fixed", "grid", "random_0", "random_1", "random_2", "bandit") + tc.LLM_APPROACHES
LLM_ROWS = tc.LLM_APPROACHES                       # F1-R1, v21, GA-R1 and the v2.2 variants V2 to V8 (Sat 14 Nov)
SITUATION_TABLE = "configs/experience/situations_v22.json"
VARIANT_DIR = "configs/policies/template_v2.2_r2"
GOAL = {"T0": "T0", "T": "T0", "W2R": "T0", "A0": "A0", "A": "A0", "L2R": "A0", "P0": "P0"}


def worker_of_recipe() -> dict[str, str]:
    sets = load_recipe_sets(tc.PROJECT / "configs/recipe_sets.yaml")
    power = load_recipe_sets(tc.PROJECT / "configs/power_recipe_set.yaml")
    out = {r: "timing" for r in sets.worker_sets["timing"]}
    out.update({r: "area" for r in sets.worker_sets["area"]})
    out.update({r: "power" for r in power.worker_sets["power"]})
    return out


def grid_recipes(brief: str) -> list[str]:
    """The six highest-prior recipes for the brief's goal (declared TRAINING priors; ties by recipe id)."""

    key = "abc" if brief in tc.THREE_WORKER else "ab"
    priors = tc.jread(tc.GRID_PRIORS[key])["goals"][GOAL[brief]]
    return list(priors["ranking"])[:6]


class GridPolicy:
    """Run six given single recipes once each from the start (a recipe-level plan carried as ordinary one-attempt mandates)."""

    id = "grid6"
    version = "1.0-r2"

    def __init__(self, recipes: list[str], worker_of: dict[str, str]) -> None:
        self.recipes, self.worker_of = list(recipes), dict(worker_of)

    def decide(self, view: PolicyView) -> Mandate | None:
        n = len(view.calls)
        if n >= len(self.recipes) or view.budget_left < 1:
            return None
        recipe = self.recipes[n]
        card = next((c for c in view.worker_cards if c.id == self.worker_of[recipe]), None)
        if card is None:
            raise ValueError(f"no worker card for {recipe}")
        digest = hashlib.sha256(json.dumps([self.id, self.version, n, recipe, view.start.state_id],
                                           separators=(",", ":")).encode("utf-8")).hexdigest()
        return Mandate(
            id="mandate_" + digest[:24], worker_id=card.id, base_state_id=view.start.state_id,
            settings=WorkerSettings(preserve_placement=False, max_runs=1), guards=(),
            brief_hash=view.brief.fingerprint(),
            context_ref=json.dumps({"x": recipe, "policy": self.id, "call_index": n}, sort_keys=True, separators=(",", ":")),
        )


def direct_workers(workers: dict) -> dict:
    from orch.baselines.direct_recipe_worker import DirectRecipeWorker

    return {wid: DirectRecipeWorker(w.card, w.recipe_ids) for wid, w in workers.items()}


def make_policy(ctx, task: dict, worker_ids: list[str], describe=None):
    from orch.policies.relevant_workers import FixedOrderPolicy, RelevantRandomPolicy
    from orch.policies.repair_loop import RepairLoopPolicy
    from orch.policies.training_baselines import FixedAlternationPolicy, UniformRandomPolicy

    approach, brief = task["approach"], task["brief"]
    three = brief in tc.THREE_WORKER
    if approach == "v1":
        return RepairLoopPolicy()
    if approach == "v2":
        return RepairLoopPolicy(**h.settings()["v2_parameters"])
    if approach == "fixed":
        order = task["fixed_order"]
        return FixedOrderPolicy(order=order) if three else FixedAlternationPolicy(first=order[0])
    if approach.startswith("random_"):
        seed = int(approach.split("_", 1)[1])
        return RelevantRandomPolicy(seed=seed) if three else UniformRandomPolicy(seed=seed)
    if approach == "grid":
        return GridPolicy(grid_recipes(brief), worker_of_recipe())
    if approach == "bandit":
        from orch.policies.bandit_frozen import bandit

        return bandit(task["design"], brief, 0)
    if approach in LLM_ROWS:
        return make_llm_policy(ctx, task, worker_ids, describe)
    raise ValueError(f"no policy for {approach}")


def make_llm_policy(ctx, task: dict, worker_ids: list[str], describe):
    from orch.baselines.general_agent import build_general_agent
    from orch.policies.llm import LLMPolicyConfig
    from orch.policies.llm_client import TelemetrySink
    from orch.policies.llm_prompt import load_template
    from orch.policies.round2_harness import F1Policy, V21Policy
    from orch.policies.v2_experience import load_calls_v2
    from orch.policies.v2_prompt import Variant, load_assets

    install_brief_text()
    approach = task["approach"]
    llm = tc.make_llm(task, ctx.caps)
    sink = TelemetrySink(tc.TELEMETRY)
    cfg = tc.llm_settings()
    if approach == "GA-R1":
        gcfg, _spec = h.llm_config("GA-R1")
        return build_general_agent(tc.PROJECT, gcfg, llm, sink, worker_ids)
    if approach == "F1-R1":
        conf = cfg["configurations"]["F1"]
        manifest = tc.jread(tc.PROJECT / cfg["profiles_dir"] / "manifest.json")
        texts, hashes = {}, {}
        for wid in worker_ids:
            data = yaml.safe_load((tc.PROJECT / cfg["profiles_dir"] / f"{wid}.yaml").read_text(encoding="utf-8"))
            texts[wid], hashes[wid] = data["text"], manifest["files"][f"{wid}.yaml"]
        template = h.settings()["prompt_revisions"]["R1"]
        return F1Policy(LLMPolicyConfig(
            name="F1", mode=conf["mode"], profiles_on=conf["profiles"], template=load_template(tc.PROJECT / template),
            profile_texts=texts, profile_sha256=hashes, lessons=None, max_repairs=cfg["max_repair_turns"]), llm, sink)
    if approach == "v21":
        definition = yaml.safe_load(tc.V21_DEFINITION.read_text(encoding="utf-8"))
        variant = Variant(definition["name"], tuple(definition["packages"]), wording=definition["worker_wording"])
        assets = load_assets(tc.PROJECT)
        if assets.templates[variant.combo].sha256 != definition["template_sha256"]:
            raise SystemExit("the v2.1 template differs from its frozen definition")
        calls, _sha = load_calls_v2(tc.PROJECT / definition["experience_calls"])
        config = LLMPolicyConfig(name="template_v2.1", mode="fuzzy", profiles_on=False,
                                 template=assets.templates["base"], max_repairs=cfg["max_repair_turns"])
        return V21Policy(config, llm, sink, variant=variant, assets=assets, design=task["design"], describe=describe,
                         r2_experience=None, calls_v2=calls)
    if approach in tc.V22_ROWS:
        return make_v22_policy(approach, task, llm, sink, cfg, describe)
    raise ValueError(approach)


def make_v22_policy(approach: str, task: dict, llm, sink, cfg: dict, describe):
    """V2..V8: built as scripts/v22_2026-11-13/v22_common.make_policy_v22 builds them (the development run's renderer and policy, imported,
    not edited); only the model plug-in, replay cache, seeds and telemetry sink are this driver's. The stored variant file must equal the
    registry's switches, or the row refuses to run."""

    from orch.policies.llm import LLMPolicyConfig
    from orch.policies.v2_experience import load_calls_v2
    from orch.policies.v2_prompt import Variant, load_assets
    from orch.policies.v22_policy import V22Policy
    from orch.policies.v22_prompt import V22Context
    from orch.policies.v22_situation import load_rows
    from orch.policies.v22_ablation import ABLATION_VARIANTS            # V11 (decision of Sat 14 Nov 18:00; READY: V11 config, 18:29 UTC)
    from orch.policies.v22_switches import VARIANTS, load_switch_texts, load_variant
    from orch.policies.v22_tier3 import SITUATION_TABLE_LLM, T3_VARIANTS, situation_texts_llm

    name, switches, _data = load_variant(tc.PROJECT / VARIANT_DIR / f"{approach}.yaml")
    if name != approach or switches != {**VARIANTS, **T3_VARIANTS, **ABLATION_VARIANTS}[approach]:
        raise SystemExit(f"refused: {VARIANT_DIR}/{approach}.yaml does not match the variant registry")
    definition = yaml.safe_load(tc.V21_DEFINITION.read_text(encoding="utf-8"))
    variant = Variant(definition["name"], tuple(definition["packages"]), wording=definition["worker_wording"])
    assets = load_assets(tc.PROJECT)
    if assets.templates[variant.combo].sha256 != definition["template_sha256"]:
        raise SystemExit("the v2.1 template differs from its frozen definition")
    calls, _sha = load_calls_v2(tc.PROJECT / definition["experience_calls"])
    texts = load_switch_texts(tc.PROJECT)
    if approach in T3_VARIANTS or approach in ABLATION_VARIANTS:     # V9, V10, V11: the S'2 table (LLM calls only) and its one-line population text, as tier 3 builds them
        situations, _sit_sha = load_rows(tc.PROJECT / SITUATION_TABLE_LLM)
        texts = situation_texts_llm(texts, tc.PROJECT)
    else:
        situations, _sit_sha = load_rows(tc.PROJECT / SITUATION_TABLE)
    context = V22Context(texts, [], candidate_seed=tc.SEEDS[task["tag"]], situations=situations)
    config = LLMPolicyConfig(name=f"template_v2.2_{approach}", mode="fuzzy", profiles_on=False,
                             template=assets.templates["base"], max_repairs=cfg["max_repair_turns"])
    return V22Policy(config, llm, sink, variant=variant, assets=assets, design=task["design"], describe=describe,
                     r2_experience=None, calls_v2=calls, switches=switches, context=context)


# ---------------------------------------------------------------------------------------------------- default wording of W2R and L2R
_METRIC = {MetricName.WNS_NS: "worst negative slack (WNS, ns)", MetricName.PACKED_LUTS: "packed CLB LUT count",
           MetricName.DYNAMIC_POWER_W: "dynamic power (W, Vivado vectorless estimate)"}
_installed = {"done": False}
WORDING_FILE = "configs/briefs_round2_power_wording.yaml"        # the approved sentence; hash-listed in notes/round2_briefs_freeze.sha256


def power_sentence() -> str:
    """The sentence the authors approved on Sat 14 Nov 11:30 (added after the hard limits of W2R and L2R, in both layouts). The file is read only
    if its hash equals its entry in notes/round2_briefs_freeze.sha256."""

    import hashlib

    import orch.contract.briefs_round2 as br

    entries = br._freeze_entries()
    path = tc.PROJECT / WORDING_FILE
    if WORDING_FILE not in entries:
        raise SystemExit(f"refused: {WORDING_FILE} is not in notes/round2_briefs_freeze.sha256")
    if hashlib.sha256(path.read_bytes()).hexdigest() != entries[WORDING_FILE]:
        raise SystemExit(f"refused: {WORDING_FILE} differs from its frozen hash")
    sentence = str(yaml.safe_load(path.read_text(encoding="utf-8"))["sentence_after_hard_limits"]).strip()
    if not sentence or "\n" in sentence:
        raise SystemExit("refused: the approved sentence must be one non-empty line")
    return sentence


def _limit_text(term, ref: str) -> str:
    """ref: 'B2 value' (round-1 wording) or 'value at S0' (v2 wording)."""

    metric = _METRIC[term.metric]
    if term.comparison == "<=" and term.offset > 0:
        return f"{metric} must not exceed its {ref} by more than {term.offset:g} W (the design's power noise band)"
    if term.comparison == "<=":
        return f"{metric} must not exceed its {ref}"
    return f"{metric} must not fall below its {ref}"


def render_banded(brief: Brief, *, v2: bool) -> str:
    """Default wording for W2R and L2R (not the authors' text): the round-1 or v2 layout of orch.contract.brief_text / v2_text, with the power
    limit stated as 'start value plus the design's band'. Generated from the Brief object, so the stated band is the one enforced."""

    from orch.contract.brief_text import load_frozen_spec

    spec = load_frozen_spec()
    runs = brief.budget_runs
    direction = "Maximise" if brief.objective.direction == "max" else "Minimise"
    obj = _METRIC[brief.objective.metric]
    if v2:
        from orch.policies.v2_text import FINAL_PICK_V2, START_ALIAS, VALIDITY_V2

        limits = "; ".join(_limit_text(t, "value at S0") for t in brief.terms)
        return (f"Brief {brief.id} (secondary).\nStart from {START_ALIAS}, the placed-and-routed checkpoint you were given. "
                f"Budget: at most {runs} recipe runs.\nObjective: {direction} {obj}.\nHard limits: {limits}.\n{power_sentence()}\n"
                f"Final pick: {FINAL_PICK_V2}.\nValidity: {VALIDITY_V2}.\n")
    from orch.contract.brief_text import TEMPLATE_MULTI_LIMIT

    limits = "; ".join(_limit_text(t, "B2 value") for t in brief.terms)
    text = TEMPLATE_MULTI_LIMIT.format(brief_id=brief.id, role="secondary", runs=runs, direction=direction, objective_metric=obj,
                                       limits_text=limits, final_pick=spec["common"]["final_pick"])
    needle = "\nFinal pick:"
    if text.count(needle) != 1:
        raise ValueError("the round-1 layout changed: cannot place the approved sentence after the hard limits")
    return text.replace(needle, "\n" + power_sentence() + needle)


def install_brief_text() -> None:
    """Make the LLM prompt builders render W2R and L2R (in this process only; every other brief goes through the frozen functions)."""

    if _installed["done"]:
        return
    import orch.contract.brief_text as bt
    import orch.policies.llm as m_llm
    import orch.policies.llm_scaffold as m_scaffold
    import orch.policies.r2_policy as m_r2
    import orch.policies.round2_harness as m_harness
    import orch.policies.v2_prompt as m_v2prompt
    import orch.policies.v2_text as m_v2text
    import orch.policies.v22_prompt as m_v22prompt

    banded = ("W2R", "L2R")
    orig1, orig2 = bt.render_brief_text, m_v2text.render_brief_text_v2
    frozen_runs = int(bt.load_frozen_spec()["common"]["budget"]["recipe_runs"])

    def with_budget(orig, brief):
        """The frozen renderers refuse a brief whose fingerprint differs from the frozen spec, and the budget is part of it. A budget-sensitivity
        brief (queue 5 block 4: budget 3 or 10) differs only in budget_runs: render its budget-6 twin with the frozen function and put the
        budget number in afterwards, checked to be in exactly one place."""

        if not isinstance(brief, Brief) or brief.budget_runs == frozen_runs:
            return orig(brief)
        text = orig(brief.model_copy(update={"budget_runs": frozen_runs}))
        old = f"at most {frozen_runs} recipe runs"
        if text.count(old) != 1:
            raise ValueError("the frozen brief text no longer carries the budget in exactly one place; the budget rows cannot be rendered")
        return text.replace(old, f"at most {brief.budget_runs} recipe runs")

    def r1(brief):
        bid = brief.id if isinstance(brief, Brief) else brief
        return render_banded(brief, v2=False) if bid in banded and isinstance(brief, Brief) else with_budget(orig1, brief)

    def r2(brief):
        bid = brief.id if isinstance(brief, Brief) else brief
        return render_banded(brief, v2=True) if bid in banded and isinstance(brief, Brief) else with_budget(orig2, brief)

    for mod in (m_llm, m_scaffold, m_r2, m_harness, m_v2prompt):
        mod.render_brief_text = r1
    m_v2prompt.render_brief_text_v2 = r2
    m_v2text.render_brief_text_v2 = r2
    m_v22prompt.render_brief_text_v2 = r2
    _installed["done"] = True
