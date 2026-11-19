"""V2Policy: the LLM orchestrator with a template-v2 variant, for real or simulated episodes.

The prompt is orch/policies/v2_prompt.render (the same renderer the decision bank evaluates), the reply is parsed by
orch/policies/v2_schemas.parse_reply, and the decision loop is the round-1 loop (stateless call, up to
`max_repairs` repair turns with the exact error, then the fallback STOP). Harness rules enforced here, never left
to the wording:
  - the worker and the base state must be listed, and attempts may not exceed the remaining budget;
  - with P1 (one repeat rule, change 3), repeating a worker, state and placement setting is legal unless the view
    already shows that combination ran out of options; without P1 the round-1 rule (never repeat) stays;
  - respect limits on: the guards are the brief's limits relative to the start, as in round 1;
  - the final pick, validity and the budget stay with the episode engine.
A state without a description is an infrastructure failure (DescriptionUnavailableError), never a decision.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from orch.contract.models import Guard, Mandate, WorkerSettings
from orch.contract.views import PolicyView
from orch.policies.llm import LLMPolicy, LLMPolicyConfig
from orch.policies.llm_client import CachedLLM, LLMUnavailableError, TelemetrySink
from orch.policies.llm_prompt import repair_prompt
from orch.policies.llm_schemas import MAX_RATIONALE_CHARS
from orch.policies.llm_view import METRIC_NAMES
from orch.policies.r2_experience import ExperienceTable
from orch.policies.v2_experience import CallV2
from orch.policies.v2_prompt import Assets, Situation, Variant, render
from orch.policies.v2_schemas import ReplyError, parse_reply


class DescriptionUnavailableError(LLMUnavailableError):
    """No telemetry could be produced for a saved state; never an orchestration decision."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def exhausted_combinations(view: PolicyView) -> set[tuple[str, str, bool]]:
    """(worker, base, placement kept) whose last call ran fewer runs than it asked for: no options left."""

    cards = {c.id: c for c in view.worker_cards}
    out = set()
    for call in view.calls:
        asked = call.settings.max_runs if call.settings.max_runs is not None else cards[call.worker_id].default_attempts
        if call.recipe_runs < asked:
            out.add((call.worker_id, call.base_state_id, call.settings.preserve_placement))
    return out


class V2Policy(LLMPolicy):
    id = "llm_orchestrator_v2"

    def __init__(self, config: LLMPolicyConfig, llm: CachedLLM, telemetry: TelemetrySink | None = None, *,
                 variant: Variant, assets: Assets, design: str, describe: Callable[[str], Mapping[str, Any]],
                 r2_experience: ExperienceTable | None, calls_v2: Sequence[CallV2]) -> None:
        super().__init__(config, llm, telemetry)
        self.variant, self.assets, self.design = variant, assets, design
        self.describe, self.r2_experience, self.calls_v2 = describe, r2_experience, list(calls_v2)
        template = assets.templates["base" if not variant.packages and not variant.exact_only else variant.combo]
        tag = json.dumps({"variant": variant.name, "packages": variant.packages, "wording": variant.wording,
                          "exact_only": variant.exact_only, "template": template.sha256}, sort_keys=True)
        self.version = f"{config.name}-v2-{_sha(tag)[:12]}-{llm.params.model}"

    def _describe(self, state_id: str) -> Mapping[str, Any]:
        try:
            return self.describe(state_id)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise DescriptionUnavailableError(f"no description for state {state_id[:12]}") from exc

    def _check(self, parsed: dict[str, Any], view: PolicyView, aliases: Mapping[str, str]) -> None:
        m = parsed["mandate"]
        if m is None:
            return
        if m["attempts"] > view.budget_left:
            raise ReplyError(f"attempts {m['attempts']} exceeds the remaining budget of {view.budget_left}")
        base_id = aliases[m["base"]]
        key = (m["worker"], base_id, m["preserve"])
        tried = {(c.worker_id, c.base_state_id, c.settings.preserve_placement) for c in view.calls}
        if self.variant.p1:
            if key in exhausted_combinations(view):
                raise ReplyError("that worker has no options left on that state with that placement setting")
        elif key in tried:
            raise ReplyError("that combination of worker, state and placement setting was already tried; "
                             "choose a different combination or stop")

    def decide(self, view: PolicyView) -> Mandate | None:
        if view.budget_left <= 0:
            return None
        sit = Situation(view=view, design=self.design, describe=self._describe, r2_experience=self.r2_experience,
                        calls_v2=self.calls_v2)
        base_prompt, schema, meta = render(sit, self.variant, self.assets)
        aliases = meta["aliases"]
        workers = [c.id for c in view.worker_cards]
        limits = [METRIC_NAMES[t.metric] for t in view.brief.terms]
        flags = self.variant.schema_flags()
        base_sha = _sha(base_prompt)
        prompt, calls, parsed, last_error = base_prompt, [], None, None
        for _turn in range(1 + max(0, self.config.max_repairs)):
            result = self.llm.complete(prompt, schema)
            error = None
            try:
                parsed = parse_reply(result.reply.text, flags, workers=workers, aliases=list(aliases), limit_names=limits)
                self._check(parsed, view, aliases)
            except ReplyError as exc:
                error, parsed = str(exc), None
            calls.append(self._call_record(prompt, result, error))
            if error is None:
                break
            last_error = error
            prompt = repair_prompt(base_prompt, result.reply.text, error)
        mandate = None
        if parsed is not None and parsed["action"] == "INVOKE_WORKER":
            m = parsed["mandate"]
            guards: tuple[Guard, ...] = ()
            if m["respect"]:
                guards = tuple(Guard(id=f"respect_{t.id}", metric=t.metric, comparison=t.comparison,
                                     threshold=t.factor * view.start.metrics[t.metric] + t.offset)
                               for t in view.brief.terms if t.metric in view.start.metrics)
            reason = str(parsed.get("rationale", "")).strip()[:MAX_RATIONALE_CHARS]
            identity = _sha(f"{view.episode_id}|{len(view.calls)}|{base_sha}|{json.dumps(m, sort_keys=True)}")
            mandate = Mandate(
                id=f"mandate_{identity[:24]}", worker_id=m["worker"], base_state_id=aliases[m["base"]],
                settings=WorkerSettings(preserve_placement=m["preserve"], max_runs=m["attempts"]), guards=guards,
                brief_hash=view.brief.fingerprint(),
                context_ref=json.dumps({"s": parsed.get("strategy") or "-", "r": reason, "p": base_sha[:16]},
                                       sort_keys=True, separators=(",", ":")),
            )
        self._log_v2(view, meta, base_sha, calls, parsed, last_error)
        return mandate

    def _log_v2(self, view: PolicyView, meta: dict[str, Any], prompt_sha: str, calls: list[dict[str, Any]],
                parsed: dict[str, Any] | None, last_error: str | None) -> None:
        fallback = parsed is None
        tokens = {k: sum(c[f"{k}_tokens"] for c in calls) for k in ("prompt", "completion", "thoughts", "total")}
        record = {
            "episode_id": view.episode_id, "decision_index": len(view.calls), "config": self.config.name,
            "policy_version": self.version, "variant": self.variant.name, "model": self.llm.params.model,
            "temperature": self.llm.params.temperature, "seed": self.llm.params.seed,
            "template_sha256": meta["template_sha256"], "prompt_sha256": prompt_sha, "calls": calls,
            "repair_attempts": len(calls) - 1, "fallback_triggered": fallback,
            "action": "STOP" if parsed is None else parsed["action"],
            "mandate": None if parsed is None else parsed["mandate"],
            "forecast": None if parsed is None else parsed.get("forecast"),
            "rationale": (f"fallback: {last_error}" if parsed is None else str(parsed.get("rationale", "")))[:MAX_RATIONALE_CHARS],
            "experience": meta.get("experience"), "tokens": tokens,
        }
        if parsed is not None and "reasoning" in parsed:
            record["reasoning"] = parsed["reasoning"]
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
