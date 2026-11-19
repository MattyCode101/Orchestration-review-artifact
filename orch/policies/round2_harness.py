"""Round-2 harness rules on the LLM policies (D6-v2; the authors' decisions 4 and 5, 2026-11-12). New file; nothing frozen
is edited.

  V21Policy   template v2.1 = V2Policy with Variant P1 + P2 + P3 + P4r (the frozen v2 code renders and parses),
              plus two rules in code:
                - the stop note: when the episode loop sets `harness_note`, that one line is added as the last line
                  of the view (the "Current situation" block) for that ask only;
                - every free-text field of the reply is truncated (TEXT_LIMITS) before it is used or stored: the
                  mandate's context_ref, which later prompts show, and the telemetry record.
  F1Policy    the round-1 LLMPolicy (configuration F1, wording R1) with the same stop note. Its free-text fields are
              already truncated in code where they are used or stored (diagnosis 60 words, conflict check 40 words,
              rationale 200 characters; the strategy is an enum), so decision 5 needs no change there.

The decision loop of each class is the parent's, copied with only these changes; with no note set and replies inside
the limits, prompts, mandates and telemetry fields equal the parent's (tests/test_round2_harness.py).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from orch.contract.brief_text import render_brief_text
from orch.contract.models import Guard, Mandate
from orch.contract.views import PolicyView
from orch.contract.models import WorkerSettings
from orch.policies.llm import DecisionRejected, LLMPolicy
from orch.policies.llm_prompt import render_prompt, repair_prompt
from orch.policies.llm_schemas import (
    MAX_ATTEMPTS, MAX_CONFLICT_WORDS, MAX_DIAGNOSIS_WORDS, MAX_RATIONALE_CHARS, DecisionFormatError, clip_words,
)
from orch.policies.llm_view import METRIC_NAMES, build_fuzzy_view
from orch.policies.v2_policy import V2Policy
from orch.policies.v2_prompt import Situation, render
from orch.policies.v2_schemas import MAX_REASONING_WORDS, ReplyError, parse_reply
from orch.policies.v2_view import ViewFlags, render_view_v2

HARNESS_VERSION = "r2h1"
VIEW_HEADER = "Current situation\n"

# Stated in the v2.1 prompt: reasoning under 80 words, rationale under 200 characters (and, without P2, diagnosis
# 60 words, conflict check 40 words). Not stated there, set here as harness limits: physical facts 80 words; at most
# 6 evidence items (label 60 characters, fact 40 words) and 6 options (40 words each part).
TEXT_LIMITS = {
    "reasoning_words": MAX_REASONING_WORDS, "rationale_chars": MAX_RATIONALE_CHARS,
    "diagnosis_words": MAX_DIAGNOSIS_WORDS, "conflict_words": MAX_CONFLICT_WORDS,
    "physical_facts_words": 80, "evidence_items": 6, "evidence_label_chars": 60, "evidence_fact_words": 40,
    "options_items": 6, "option_words": 40, "effect_words": 40,
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _words(value: Any, limit: int) -> str:
    return clip_words(str(value if value is not None else ""), limit)


def truncate_reply(parsed: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Every free-text field of a parsed v2 reply cut to TEXT_LIMITS; returns (reply, names of the cut fields)."""

    out = dict(parsed)
    cut: list[str] = []

    def put(key: str, value: str, original: Any) -> None:
        if value != str(original if original is not None else "").strip():
            cut.append(key)
        out[key] = value

    for key, limit in (("reasoning", TEXT_LIMITS["reasoning_words"]), ("diagnosis", TEXT_LIMITS["diagnosis_words"]),
                       ("physical_facts", TEXT_LIMITS["physical_facts_words"]),
                       ("conflict_check", TEXT_LIMITS["conflict_words"])):
        if key in out and out[key] is not None:
            put(key, _words(out[key], limit), out[key])
    if "rationale" in out and out["rationale"] is not None:
        put("rationale", str(out["rationale"]).strip()[:TEXT_LIMITS["rationale_chars"]], out["rationale"])
    if isinstance(out.get("evidence"), list):
        items = []
        for item in out["evidence"][:TEXT_LIMITS["evidence_items"]]:
            label = str(item.get("label", "")).strip()[:TEXT_LIMITS["evidence_label_chars"]]
            fact = _words(item.get("fact", ""), TEXT_LIMITS["evidence_fact_words"])
            source = item.get("source")
            new = {"source": source, "label": label, "fact": fact}
            if label != str(item.get("label", "")).strip() or fact != str(item.get("fact", "")).strip():
                cut.append("evidence")
            items.append(new)
        if len(out["evidence"]) > TEXT_LIMITS["evidence_items"]:
            cut.append("evidence")
        out["evidence"] = items
    if isinstance(out.get("options"), list):
        items = []
        for item in out["options"][:TEXT_LIMITS["options_items"]]:
            option = _words(item.get("option", ""), TEXT_LIMITS["option_words"])
            effect = _words(item.get("expected_effect", ""), TEXT_LIMITS["effect_words"])
            if option != str(item.get("option", "")).strip() or effect != str(item.get("expected_effect", "")).strip():
                cut.append("options")
            items.append({"option": option, "expected_effect": effect})
        if len(out["options"]) > TEXT_LIMITS["options_items"]:
            cut.append("options")
        out["options"] = items
    return out, sorted(set(cut))


def insert_note(prompt: str, view_text: str, note: str | None) -> str:
    """Add `note` as the last line of the view block (after 'Current situation' and the view text)."""

    if not note:
        return prompt
    anchor = VIEW_HEADER + view_text.rstrip("\n") + "\n"
    if prompt.count(anchor) != 1:
        raise ValueError("the view block is not found exactly once in the rendered prompt")
    return prompt.replace(anchor, anchor + note + "\n", 1)


class V21Policy(V2Policy):
    """Template v2.1 in episodes: V2Policy plus the stop note and free-text truncation in code."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not (self.variant.p1 and self.variant.p2):
            raise ValueError("V21Policy renders the v2 view (packages P1 and P2 required)")
        self.version = f"{self.version}-{HARNESS_VERSION}"
        self.harness_note: str | None = None

    def view_text(self, view: PolicyView) -> str:
        flags = ViewFlags(labels=self.variant.p1, vocab=self.variant.p1, placement=self.variant.p2,
                          exact_only=self.variant.exact_only)
        return render_view_v2(view, flags).text

    def decide(self, view: PolicyView) -> Mandate | None:
        if view.budget_left <= 0:
            return None
        note = self.harness_note
        sit = Situation(view=view, design=self.design, describe=self._describe, r2_experience=self.r2_experience,
                        calls_v2=self.calls_v2)
        rendered, schema, meta = render(sit, self.variant, self.assets)
        base_prompt = insert_note(rendered, self.view_text(view), note)
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
        cut: list[str] = []
        if parsed is not None:
            parsed, cut = truncate_reply(parsed)
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
        self._log_v21(view, meta, base_sha, _sha(rendered), calls, parsed, last_error, note, cut, mandate)
        return mandate

    def _log_v21(self, view: PolicyView, meta: dict[str, Any], prompt_sha: str, rendered_sha: str,
                 calls: list[dict[str, Any]], parsed: dict[str, Any] | None, last_error: str | None,
                 note: str | None, cut: list[str], mandate: Mandate | None) -> None:
        fallback = parsed is None
        tokens = {k: sum(c[f"{k}_tokens"] for c in calls) for k in ("prompt", "completion", "thoughts", "total")}
        record: dict[str, Any] = {
            "episode_id": view.episode_id, "decision_index": len(view.calls), "config": self.config.name,
            "policy_version": self.version, "variant": self.variant.name, "packages": list(self.variant.packages),
            "model": self.llm.params.model, "temperature": self.llm.params.temperature, "seed": self.llm.params.seed,
            "template_sha256": meta["template_sha256"], "prompt_sha256": prompt_sha,
            "rendered_prompt_sha256": rendered_sha, "harness_note": note, "calls": calls,
            "repair_attempts": len(calls) - 1, "fallback_triggered": fallback,
            "action": "STOP" if parsed is None else parsed["action"],
            "mandate": None if parsed is None else parsed["mandate"],
            "mandate_id": mandate.id if mandate is not None else None,
            "aliases": meta.get("aliases"), "budget_left": view.budget_left,
            "forecast": None if parsed is None else parsed.get("forecast"),
            "rationale": (f"fallback: {last_error}" if parsed is None else str(parsed.get("rationale", "")))[:MAX_RATIONALE_CHARS],
            "experience": meta.get("experience"), "tokens": tokens, "truncated_fields": cut,
            "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if parsed is not None:
            for key in ("reasoning", "physical_facts", "evidence", "options", "diagnosis", "strategy"):
                if key in parsed:
                    record[key] = parsed[key]
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


class F1Policy(LLMPolicy):
    """The round-1 LLMPolicy (F1, wording R1) with the stop note as the last line of its view."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.version = f"{self.version}-{HARNESS_VERSION}"
        self.harness_note: str | None = None
        self._note_used: str | None = None

    def decide(self, view: PolicyView) -> Mandate | None:
        if view.budget_left <= 0:
            return None
        note = self.harness_note
        built = build_fuzzy_view(view, mode=self.config.mode, recipe_level=self.recipe_level)
        max_attempts = min(MAX_ATTEMPTS, view.budget_left)
        schema = self._schema(view, built, max_attempts)
        view_text = built.text if not note else built.text.rstrip("\n") + "\n" + note
        base_prompt = render_prompt(
            self.config.template, brief_text=render_brief_text(view.brief),
            cards_text=self._cards_text(view), lessons=self.config.lessons,
            view_text=view_text, schema=schema, max_attempts=max_attempts,
        )
        base_sha = _sha(base_prompt)
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
        self._note_used = note
        self._log(view, built, base_sha, calls, output, fallback, last_error if fallback else None)
        return mandate

    def _log(self, view, built, prompt_sha, calls, output, fallback, last_error) -> None:
        sink, self.telemetry = self.telemetry, None
        try:
            super()._log(view, built, prompt_sha, calls, output, fallback, last_error)
        finally:
            self.telemetry = sink
        record = dict(self.last_record or {})
        record["harness_note"] = self._note_used
        record["policy_version"] = self.version
        self.last_record = record
        if self.telemetry is not None:
            self.telemetry.append(record)


__all__ = ["F1Policy", "HARNESS_VERSION", "TEXT_LIMITS", "V21Policy", "insert_note", "truncate_reply"]
