"""LLMPolicy: one stateless model call per decision (final specification, sections 2, 6 to 8).

The policy sees only the PolicyView, rebuilds the fuzzy view from it each time, asks
the model for one structured decision, validates it against the live view and returns
a Mandate (or None to stop). The model never sees recipes, design names, paths, hashes
or raw numbers (fuzzy mode); worker-specific knowledge arrives only through the cards
and their generated profiles. Invalid replies get up to `max_repairs` repair turns that
carry the exact error; after that the policy stops and logs `fallback_triggered`.

A mandate's `context_ref` carries the model's one-line reason, so the next stateless
decision can rebuild the trajectory log from the episode journal alone.

The decision loop is shared with the general-agent baseline (orch/baselines/general_agent.py),
which overrides only the small hooks below: schema, cards, parsing, validation, mandate
construction and the telemetry summary of a choice.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from orch.contract.brief_text import render_brief_text
from orch.contract.models import Guard, Mandate, WorkerSettings
from orch.contract.views import PolicyView
from orch.policies.llm_client import CachedLLM, LLMResult, TelemetrySink
from orch.policies.llm_prompt import Template, render_cards, render_prompt, repair_prompt
from orch.policies.llm_schemas import (
    MAX_ATTEMPTS, MAX_CONFLICT_WORDS, MAX_DIAGNOSIS_WORDS, MAX_RATIONALE_CHARS,
    DecisionFormatError, LLMDecisionOutput, clip_words, decision_json_schema, parse_decision,
)
from orch.policies.llm_view import BuiltView, build_fuzzy_view


class DecisionRejected(ValueError):
    """The reply parsed but breaks a rule that depends on the live view."""


@dataclass(frozen=True)
class LLMPolicyConfig:
    name: str
    mode: Literal["fuzzy", "exact"]
    profiles_on: bool
    template: Template
    profile_texts: Mapping[str, str] = field(default_factory=dict)
    profile_sha256: Mapping[str, str] = field(default_factory=dict)
    lessons: str | None = None
    max_repairs: int = 2


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class LLMPolicy:
    id = "llm_orchestrator"
    recipe_level = False

    def __init__(self, config: LLMPolicyConfig, llm: CachedLLM, telemetry: TelemetrySink | None = None) -> None:
        self.config, self.llm, self.telemetry = config, llm, telemetry
        self.profile_hashes = dict(config.profile_sha256) if config.profiles_on else None
        self.lessons_sha256 = _sha(config.lessons) if config.lessons and config.lessons.strip() else None
        parts = (
            config.name, config.mode, config.template.sha256[:10],
            (_sha(_canonical(self.profile_hashes))[:10] if self.profile_hashes else "noprof"),
            (self.lessons_sha256[:10] if self.lessons_sha256 else "nolessons"), llm.params.model,
        )
        self.version = "-".join(parts)
        self.totals = {"decisions": 0, "calls": 0, "fallbacks": 0, "prompt": 0, "completion": 0,
                       "thoughts": 0, "total": 0, "fresh_total": 0}
        self.last_record: dict[str, Any] | None = None

    # ------------------------------------------------------------------ hooks (overridden by the general agent)
    def _schema(self, view: PolicyView, built: BuiltView, max_attempts: int) -> dict[str, Any]:
        return decision_json_schema([card.id for card in view.worker_cards], list(built.alias_to_state),
                                    max_attempts=max_attempts)

    def _cards_text(self, view: PolicyView) -> str:
        profiles = self.config.profile_texts if self.config.profiles_on else None
        return render_cards(view.worker_cards, profiles)

    def _parse(self, text: str) -> Any:
        return parse_decision(text)

    @staticmethod
    def _invokes(output: Any) -> bool:
        return output.action == "INVOKE_WORKER"

    @staticmethod
    def _choice_summary(output: Any) -> dict[str, Any] | None:
        choice = output.mandate
        if choice is None:
            return None
        return {
            "worker": choice.worker_id, "base": choice.base_state,
            "preserve_placement": choice.preserve_placement, "attempts": choice.attempts,
            "respect_brief_limits": choice.respect_brief_limits,
        }

    def _validate(self, output: LLMDecisionOutput, view: PolicyView, built: BuiltView) -> None:
        mandate = output.mandate
        if mandate is None:
            return
        workers = [card.id for card in view.worker_cards]
        if mandate.worker_id not in workers:
            raise DecisionRejected(
                f"worker_id {mandate.worker_id!r} is not one of the listed workers ({', '.join(workers)})"
            )
        if mandate.base_state not in built.alias_to_state:
            raise DecisionRejected(
                f"base_state {mandate.base_state!r} is not one of the listed states "
                f"({', '.join(built.alias_to_state)})"
            )
        if mandate.attempts > view.budget_left:
            raise DecisionRejected(
                f"attempts {mandate.attempts} exceeds the remaining budget of {view.budget_left}"
            )
        base_id = built.alias_to_state[mandate.base_state]
        for call in view.calls:
            if (call.worker_id, call.base_state_id, call.settings.preserve_placement) == (
                mandate.worker_id, base_id, mandate.preserve_placement,
            ):
                raise DecisionRejected(
                    "that combination of worker, state and placement setting was already tried; "
                    "choose a different combination or stop"
                )

    def _mandate(self, output: LLMDecisionOutput, view: PolicyView, built: BuiltView, prompt_sha: str) -> Mandate:
        choice = output.mandate
        assert choice is not None
        guards: tuple[Guard, ...] = ()
        if choice.respect_brief_limits:
            guards = tuple(
                Guard(
                    id=f"respect_{term.id}", metric=term.metric, comparison=term.comparison,
                    threshold=term.factor * view.start.metrics[term.metric] + term.offset,
                )
                for term in view.brief.terms if term.metric in view.start.metrics
            )
        identity = _sha(f"{view.episode_id}|{len(view.calls)}|{prompt_sha}|{_canonical(choice.model_dump())}")
        reason = output.rationale_summary.strip()[:MAX_RATIONALE_CHARS]
        return Mandate(
            id=f"mandate_{identity[:24]}", worker_id=choice.worker_id,
            base_state_id=built.alias_to_state[choice.base_state],
            settings=WorkerSettings(preserve_placement=choice.preserve_placement, max_runs=choice.attempts),
            guards=guards, brief_hash=view.brief.fingerprint(),
            context_ref=json.dumps({"s": output.strategy, "r": reason, "p": prompt_sha[:16]},
                                   sort_keys=True, separators=(",", ":")),
        )

    # ------------------------------------------------------------------ decision
    def decide(self, view: PolicyView) -> Mandate | None:
        if view.budget_left <= 0:
            return None
        built = build_fuzzy_view(view, mode=self.config.mode, recipe_level=self.recipe_level)
        max_attempts = min(MAX_ATTEMPTS, view.budget_left)
        schema = self._schema(view, built, max_attempts)
        base_prompt = render_prompt(
            self.config.template, brief_text=render_brief_text(view.brief),
            cards_text=self._cards_text(view), lessons=self.config.lessons,
            view_text=built.text, schema=schema, max_attempts=max_attempts,
        )
        base_sha = _sha(base_prompt)
        prompt, calls = base_prompt, []
        output: Any = None
        last_error: str | None = None
        for turn in range(1 + max(0, self.config.max_repairs)):
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
    @staticmethod
    def _call_record(prompt: str, result: LLMResult, error: str | None) -> dict[str, Any]:
        reply = result.reply
        return {
            "prompt_sha256": _sha(prompt), "cache_key": result.key, "cache_mode": result.cache_mode,
            "cache_hit": result.cache_hit, "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens, "thoughts_tokens": reply.thoughts_tokens,
            "total_tokens": reply.total_tokens, "latency_ms": reply.latency_ms,
            "finish_reason": reply.finish_reason, "error": error,
        }

    def _log(self, view: PolicyView, built: BuiltView, prompt_sha: str, calls: list[dict[str, Any]],
             output: Any, fallback: bool, last_error: str | None) -> None:
        tokens = {
            "prompt": sum(c["prompt_tokens"] for c in calls),
            "completion": sum(c["completion_tokens"] for c in calls),
            "thoughts": sum(c["thoughts_tokens"] for c in calls),
            "total": sum(c["total_tokens"] for c in calls),
            "fresh_total": sum(c["total_tokens"] for c in calls if not c["cache_hit"]),
        }
        record: dict[str, Any] = {
            "episode_id": view.episode_id, "decision_index": len(view.calls), "config": self.config.name,
            "mode": self.config.mode, "profiles_on": self.config.profiles_on, "policy_version": self.version,
            "model": self.llm.params.model, "temperature": self.llm.params.temperature,
            "seed": self.llm.params.seed, "thinking_level": self.llm.params.thinking_level,
            "template_sha256": self.config.template.sha256,
            "profile_sha256": self.profile_hashes, "lessons_sha256": self.lessons_sha256,
            "view_sha256": built.sha256, "prompt_sha256": prompt_sha, "calls": calls,
            "repair_attempts": len(calls) - 1, "fallback_triggered": fallback,
            "strategy": "STOP" if output is None else output.strategy,
            "diagnosis": "" if output is None else clip_words(output.state_diagnosis, MAX_DIAGNOSIS_WORDS),
            "conflict_check": "" if output is None else clip_words(output.conflict_check, MAX_CONFLICT_WORDS),
            "rationale": (f"fallback: {last_error}" if output is None else output.rationale_summary)[:MAX_RATIONALE_CHARS],
            "action": "STOP" if output is None else output.action,
            "mandate": None if output is None else self._choice_summary(output),
            "tokens": tokens, "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if last_error is not None:
            record["last_error"] = last_error
        self.last_record = record
        self.totals["decisions"] += 1
        self.totals["calls"] += len(calls)
        self.totals["fallbacks"] += int(fallback)
        for name, value in tokens.items():
            self.totals[name] += value
        if self.telemetry is not None:
            self.telemetry.append(record)
