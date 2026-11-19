"""R2Policy: the LLM orchestrator with switchable round-2 evidence (history, physical description, experience).

It is the round-1 LLMPolicy with a different way of building the prompt: the state list may carry
provenance and exact brief metrics (history), and up to two evidence sections are placed before the
decision request (physical description of the saved states; worker experience from earlier designs).
The output schema, the validation, the repair turns, the stop rule and the mandate construction are the
round-1 ones, unchanged, because the decision loop below is the round-1 loop.

Each decision also logs, for the agreement analysis, what the greedy-experience reference rule would
have taken on the same view. That rule is never run as a policy and never shown to the model.

A state without a description is an infrastructure failure (DescriptionUnavailableError, a kind of
LLMUnavailableError): the cell is retried, and no decision is made on missing evidence.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import Any

from orch.contract.brief_text import render_brief_text
from orch.contract.models import Mandate, MetricName
from orch.contract.views import PolicyView
from orch.policies.llm import DecisionRejected, LLMPolicy, LLMPolicyConfig
from orch.policies.llm_client import CachedLLM, LLMUnavailableError, TelemetrySink
from orch.policies.llm_prompt import repair_prompt
from orch.policies.llm_schemas import MAX_ATTEMPTS, DecisionFormatError
from orch.policies.llm_view import BuiltView
from orch.policies.r2_experience import (
    ExperienceTable, greedy_scores, pick_greedy, render_experience_section,
)
from orch.policies.r2_physical import facts_from_record, render_physical_section
from orch.policies.r2_prompt import render_prompt_r2
from orch.policies.r2_view import (
    Features, alias_order, build_r2_view, goal_and_limits, state_lookup,
)

Describe = Callable[[str], Mapping[str, Any]]


class DescriptionUnavailableError(LLMUnavailableError):
    """No telemetry could be produced for a saved state; never an orchestration decision."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class R2Policy(LLMPolicy):
    id = "llm_orchestrator_r2"

    def __init__(
        self, config: LLMPolicyConfig, llm: CachedLLM, telemetry: TelemetrySink | None = None, *,
        features: Features, experience: ExperienceTable | None = None, greedy: ExperienceTable | None = None,
        describe: Describe | None = None,
    ) -> None:
        super().__init__(config, llm, telemetry)
        if features.physical and describe is None:
            raise ValueError("a physical description needs a describe function")
        if features.experience != "off" and experience is None:
            raise ValueError("an experience section needs an experience table")
        self.features, self.experience, self.greedy, self.describe = features, experience, greedy, describe
        tag = "|".join((
            features.key, experience.sha256() if features.experience != "off" and experience else "noexp",
            greedy.sha256() if greedy else "nogreedy",
        ))
        self.version = f"{self.version}-r2-{_sha(tag)[:12]}"
        self._pending: dict[str, Any] = {}

    # ------------------------------------------------------------------ evidence sections
    def _physical(self, view: PolicyView, built: BuiltView) -> tuple[str, dict[str, Any]]:
        groups = self.features.physical
        if not groups:
            return "", {}
        assert self.describe is not None
        lookup = state_lookup(view)
        entries = []
        for alias in alias_order(list(built.alias_to_state)):
            state_id = built.alias_to_state[alias]
            facts = facts_from_record(self.describe(state_id))
            facts["dynamic_power_w"] = lookup[state_id].metrics.get(MetricName.DYNAMIC_POWER_W)
            entries.append((alias, alias == "S0", facts))
        text = render_physical_section(entries, groups, power_band=view.noise_bands.get(MetricName.DYNAMIC_POWER_W))
        return text, {"groups": list(groups), "states": len(entries), "chars": len(text), "sha256": _sha(text)}

    def _experience_text(self, view: PolicyView) -> tuple[str, dict[str, Any]]:
        if self.features.experience == "off" or self.experience is None:
            return "", {}
        goal, limits = goal_and_limits(view)
        text = render_experience_section(
            self.experience, goal=goal, limits=limits, workers=[card.id for card in view.worker_cards],
            form=self.features.experience,  # type: ignore[arg-type]
        )
        return text, {
            "form": self.features.experience, "source": self.features.experience_source,
            "excluded_design_in_table": self.experience.excluded is not None, "chars": len(text),
            "sha256": _sha(text), "table_sha256": self.experience.sha256(),
        }

    # ------------------------------------------------------------------ decision (the round-1 loop)
    def decide(self, view: PolicyView) -> Mandate | None:
        if view.budget_left <= 0:
            return None
        built = build_r2_view(view, self.features)
        physical_text, physical_meta = self._physical(view, built)
        experience_text, experience_meta = self._experience_text(view)
        max_attempts = min(MAX_ATTEMPTS, view.budget_left)
        schema = self._schema(view, built, max_attempts)
        base_prompt = render_prompt_r2(
            self.config.template, brief_text=render_brief_text(view.brief), cards_text=self._cards_text(view),
            view_text=built.text, schema=schema, max_attempts=max_attempts,
            physical_text=physical_text, experience_text=experience_text,
        )
        base_sha = _sha(base_prompt)
        self._pending = {
            "features": self.features.key, "view_chars": len(built.text), "prompt_chars": len(base_prompt),
            "physical": physical_meta, "experience": experience_meta,
        }
        prompt, calls = base_prompt, []
        output: Any = None
        last_error: str | None = None
        for _turn in range(1 + max(0, self.config.max_repairs)):
            result = self.llm.complete(prompt, schema)
            error: str | None = None
            try:
                output = self._parse(result.reply.text)
                self._validate(output, view, built)
            except (DecisionFormatError, DecisionRejected) as exc:
                error, output = str(exc), None
            calls.append(self._call_record(prompt, result, error))
            if error is None:
                break
            last_error = error
            prompt = repair_prompt(base_prompt, result.reply.text, error)
        mandate: Mandate | None = None
        fallback = output is None
        if output is not None and self._invokes(output):
            mandate = self._mandate(output, view, built, base_sha)
        self._log(view, built, base_sha, calls, output, fallback, last_error if fallback else None)
        return mandate

    # ------------------------------------------------------------------ telemetry
    def _greedy_record(self, view: PolicyView, built: BuiltView, output: Any) -> dict[str, Any] | None:
        if self.greedy is None:
            return None
        goal, limits = goal_and_limits(view)
        workers = [card.id for card in view.worker_cards]
        choice = None if output is None else output.mandate
        base_alias = choice.base_state if choice is not None else built.fuzzy.best_valid_state
        base_id = built.alias_to_state.get(base_alias)
        tried = {(c.worker_id, c.settings.preserve_placement) for c in view.calls if c.base_state_id == base_id}
        pick = pick_greedy(self.greedy, goal, limits, workers, tried)
        scores = greedy_scores(self.greedy, goal, limits, workers)
        record: dict[str, Any] = {
            "base": base_alias, "tried_at_base": sorted(f"{w}|{'keep' if p else 'free'}" for w, p in tried),
            "scores": {f"{w}|{'keep' if p else 'free'}": round(s, 4) for (w, p), s in sorted(scores.items())},
            "pick": None if pick is None else {
                "worker": pick.worker, "preserve_placement": pick.preserve_placement, "score": round(pick.score, 4),
            },
            "table_sha256": self.greedy.sha256(), "table_excludes_design": self.greedy.excluded is not None,
        }
        if choice is None or pick is None:
            record["agree"] = record["agree_worker"] = None
        else:
            record["agree"] = (pick.worker, pick.preserve_placement) == (choice.worker_id, choice.preserve_placement)
            record["agree_worker"] = pick.worker == choice.worker_id
        return record

    def _log(self, view: PolicyView, built: BuiltView, prompt_sha: str, calls: list[dict[str, Any]],
             output: Any, fallback: bool, last_error: str | None) -> None:
        sink, self.telemetry = self.telemetry, None      # let the round-1 logger build the record, then extend it
        try:
            super()._log(view, built, prompt_sha, calls, output, fallback, last_error)
        finally:
            self.telemetry = sink
        record = self.last_record
        assert record is not None
        record["r2"] = {**self._pending, "greedy": self._greedy_record(view, built, output)}
        if sink is not None:
            sink.append(record)
