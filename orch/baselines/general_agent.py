"""General-agent baseline (final specification, section 9): the same model, brief text, budget,
fuzzy view and final-pick rule as the LLM orchestrator, but acting on recipes directly.

It shares the orchestrator's whole decision loop (LLMPolicy.decide: view, prompt, replay cache,
repair turns, fallback, telemetry) and overrides only the hooks that differ: the action is
`run_recipe(base_state, recipe_id)` or stop, the catalog lists recipes with a one-line generic
description (what each targets and whether it re-places) and its recipe-level behaviour
profile, and a validated action becomes an ordinary one-attempt mandate to the recipe's own
worker. The recipe id travels in the mandate's `context_ref` under "x" (see
orch/baselines/direct_recipe_worker.py). Its recipe-level access is a privilege real
black-box workers would not offer; the paper frames it as the white-box comparison.

Policy-layer rule: this module imports no engine, runner, recipe or worker code. The recipe
catalog arrives as plain data.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError, model_validator

from orch.contract.models import FrozenModel, Mandate, WorkerSettings
from orch.contract.views import PolicyView
from orch.policies.llm import DecisionRejected, LLMPolicy, LLMPolicyConfig, _canonical, _sha
from orch.policies.llm_client import CachedLLM, TelemetrySink
from orch.policies.llm_prompt import load_template
from orch.policies.llm_schemas import MAX_RATIONALE_CHARS, STRATEGIES, DecisionFormatError
from orch.policies.llm_view import BuiltView, recipe_of

ACTIONS = ("RUN_RECIPE", "STOP")
_ORDER = ("state_diagnosis", "strategy", "conflict_check", "rationale_summary", "action", "recipe_action")


class RecipeAction(FrozenModel):
    recipe_id: str
    base_state: str


class GeneralDecisionOutput(FrozenModel):
    state_diagnosis: str
    strategy: str
    conflict_check: str
    rationale_summary: str
    action: str
    recipe_action: RecipeAction | None

    @model_validator(mode="after")
    def consistent(self) -> "GeneralDecisionOutput":
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {', '.join(STRATEGIES)}")
        if self.action not in ACTIONS:
            raise ValueError(f"action must be one of {', '.join(ACTIONS)}")
        if self.action == "RUN_RECIPE" and self.recipe_action is None:
            raise ValueError("recipe_action is required when the action is RUN_RECIPE")
        if self.action == "STOP" and self.recipe_action is not None:
            raise ValueError("recipe_action must be null when the action is STOP")
        return self


def parse_general_decision(text: str) -> GeneralDecisionOutput:
    body = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, flags=re.DOTALL)
    if fenced:
        body = fenced.group(1)
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise DecisionFormatError(f"the reply is not valid JSON ({exc.msg})") from exc
    if not isinstance(data, dict):
        raise DecisionFormatError("the reply must be one JSON object")
    try:
        return GeneralDecisionOutput.model_validate(data)
    except ValidationError as exc:
        problems = [f"{'.'.join(str(p) for p in e['loc']) or 'reply'}: {e['msg']}" for e in exc.errors(include_url=False)]
        raise DecisionFormatError("; ".join(problems[:4])) from exc


def general_json_schema(recipe_ids: Sequence[str], aliases: Sequence[str]) -> dict[str, Any]:
    action = {
        "type": "object",
        "properties": {
            "recipe_id": {"type": "string", "enum": list(recipe_ids)},
            "base_state": {"type": "string", "enum": list(aliases)},
        },
        "required": ["recipe_id", "base_state"], "propertyOrdering": ["recipe_id", "base_state"],
    }
    return {
        "type": "object",
        "properties": {
            "state_diagnosis": {"type": "string"},
            "strategy": {"type": "string", "enum": list(STRATEGIES)},
            "conflict_check": {"type": "string"},
            "rationale_summary": {"type": "string"},
            "action": {"type": "string", "enum": list(ACTIONS)},
            "recipe_action": {"anyOf": [{"type": "null"}, action]},
        },
        "required": list(_ORDER), "propertyOrdering": list(_ORDER),
    }


@dataclass(frozen=True)
class RecipeInfo:
    id: str
    worker: str
    text: str            # one-line generic description plus the recipe's behaviour profile


class GeneralAgentPolicy(LLMPolicy):
    id = "general_agent"
    recipe_level = True

    def __init__(self, config: LLMPolicyConfig, llm: CachedLLM, telemetry: TelemetrySink | None,
                 recipes: Sequence[RecipeInfo]) -> None:
        super().__init__(config, llm, telemetry)
        self.recipes = {r.id: r for r in recipes}
        if len(self.recipes) != len(recipes):
            raise ValueError("duplicate recipe in the catalog")
        self.version = f"{self.version}-cat{_sha(_canonical([[r.id, r.worker, r.text] for r in recipes]))[:10]}"

    def _available(self, view: PolicyView) -> list[RecipeInfo]:
        workers = {card.id for card in view.worker_cards}
        return [r for r in self.recipes.values() if r.worker in workers]

    def _schema(self, view: PolicyView, built: BuiltView, max_attempts: int) -> dict[str, Any]:
        return general_json_schema([r.id for r in self._available(view)], list(built.alias_to_state))

    def _cards_text(self, view: PolicyView) -> str:
        return "\n".join(r.text for r in self._available(view))

    def _parse(self, text: str) -> GeneralDecisionOutput:
        return parse_general_decision(text)

    @staticmethod
    def _invokes(output: GeneralDecisionOutput) -> bool:
        return output.action == "RUN_RECIPE"

    @staticmethod
    def _choice_summary(output: GeneralDecisionOutput) -> dict[str, Any] | None:
        if output.recipe_action is None:
            return None
        return {"recipe": output.recipe_action.recipe_id, "base": output.recipe_action.base_state}

    def _validate(self, output: GeneralDecisionOutput, view: PolicyView, built: BuiltView) -> None:
        choice = output.recipe_action
        if choice is None:
            return
        available = {r.id for r in self._available(view)}
        if choice.recipe_id not in available:
            raise DecisionRejected(f"recipe_id {choice.recipe_id!r} is not one of the listed recipes")
        if choice.base_state not in built.alias_to_state:
            raise DecisionRejected(
                f"base_state {choice.base_state!r} is not one of the listed states ({', '.join(built.alias_to_state)})"
            )
        if view.budget_left < 1:
            raise DecisionRejected("no budget is left")
        base_id = built.alias_to_state[choice.base_state]
        for call in view.calls:
            if call.base_state_id == base_id and recipe_of(call) == choice.recipe_id:
                raise DecisionRejected("that recipe was already run on that state; choose another recipe or state, or stop")

    def _mandate(self, output: GeneralDecisionOutput, view: PolicyView, built: BuiltView, prompt_sha: str) -> Mandate:
        choice = output.recipe_action
        assert choice is not None
        identity = _sha(f"{view.episode_id}|{len(view.calls)}|{prompt_sha}|{_canonical(choice.model_dump())}")
        reason = output.rationale_summary.strip()[:MAX_RATIONALE_CHARS]
        return Mandate(
            id=f"mandate_{identity[:24]}", worker_id=self.recipes[choice.recipe_id].worker,
            base_state_id=built.alias_to_state[choice.base_state],
            settings=WorkerSettings(preserve_placement=False, max_runs=1), guards=(),
            brief_hash=view.brief.fingerprint(),
            context_ref=json.dumps({"s": output.strategy, "r": reason, "p": prompt_sha[:16], "x": choice.recipe_id},
                                   sort_keys=True, separators=(",", ":")),
        )


def build_general_agent(project: Path, cfg: dict[str, Any], llm: CachedLLM, sink: TelemetrySink,
                        worker_ids: Sequence[str]) -> GeneralAgentPolicy:
    """The agent for one episode: catalog and profiles restricted to the episode's workers."""

    profiles_dir = project / cfg["profiles_dir"]
    manifest = json.loads((profiles_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = yaml.safe_load((profiles_dir / "recipes.yaml").read_text(encoding="utf-8"))["recipes"]
    recipes = [
        RecipeInfo(rid, entry["worker"], entry["text"])
        for rid, entry in sorted(entries.items(), key=lambda kv: (kv[1]["worker"], kv[0]))
        if entry["worker"] in worker_ids
    ]
    config = LLMPolicyConfig(
        name="GA", mode="fuzzy", profiles_on=True,
        template=load_template(project / cfg["templates"]["general_agent"]),
        profile_texts={}, profile_sha256={"recipes": manifest["files"]["recipes.yaml"]},
        lessons=None, max_repairs=cfg["max_repair_turns"],
    )
    return GeneralAgentPolicy(config, llm, sink, recipes)
