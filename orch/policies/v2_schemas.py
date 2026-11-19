"""Template v2 reply schemas and one tolerant parser for every package combination.

  base, P1          the round-1 schema (orch/policies/llm_schemas.decision_json_schema), unchanged
  P2                no strategy label and no conflict_check (changes 12, 14); placement as "may change" /
                    "must not change" (change 11)
  P3 without P4     adds physical_facts after state_diagnosis (change 19)
  P4                reasoning first, evidence, options, rationale, action, mandate, forecast (change 20, the authors'
                    decision 8); sent only as the provider's response schema, never written into the prompt

The schema keywords are those the bridge already uses (type, enum, minimum, maximum, anyOf, properties, required,
propertyOrdering) plus description and minItems, which the Gemini structured-output documentation lists.
`parse_reply` turns any of these replies into one normalised dict; a reply it cannot use raises ReplyError.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from orch.policies.llm_schemas import (
    ACTIONS, MAX_ATTEMPTS, MAX_CONFLICT_WORDS, MAX_DIAGNOSIS_WORDS, MAX_RATIONALE_CHARS, STRATEGIES,
    decision_json_schema,
)
from orch.policies.v2_vocab import CATEGORIES_V2, NOTHING_SELECTABLE

R1_FORECAST = ("large_gain", "small_gain", "within_noise", "small_loss", "large_loss")
EXACT_FORECAST = ("gain", "no change", "loss")
PLACEMENT_ENUM = ("may change", "must not change")
EVIDENCE_SOURCES = ("state", "physical", "experience", "brief")
MAX_REASONING_WORDS = 80


class ReplyError(ValueError):
    """The reply cannot be used as a decision (not JSON, missing or ill-typed field, unknown value)."""


@dataclass(frozen=True)
class SchemaFlags:
    p2: bool = False
    p3: bool = False
    p4: bool = False
    vocab_v2: bool = False
    exact_only: bool = False
    p4r: bool = False          # P4 revision 1: P3's physical_facts field kept in the structured reply


def forecast_values(flags: SchemaFlags) -> tuple[str, ...]:
    if flags.exact_only:
        return (*EXACT_FORECAST, NOTHING_SELECTABLE)
    return (*(CATEGORIES_V2 if flags.vocab_v2 else R1_FORECAST), NOTHING_SELECTABLE)


def _mandate_schema(workers: Sequence[str], aliases: Sequence[str], max_attempts: int, p2: bool) -> dict[str, Any]:
    placement_key = "placement" if p2 else "preserve_placement"
    placement = {"type": "string", "enum": list(PLACEMENT_ENUM)} if p2 else {"type": "boolean"}
    order = ["worker_id", "base_state", placement_key, "attempts", "respect_brief_limits"]
    return {
        "type": "object",
        "properties": {
            "worker_id": {"type": "string", "enum": list(workers)},
            "base_state": {"type": "string", "enum": list(aliases)},
            placement_key: placement,
            "attempts": {"type": "integer", "minimum": 1, "maximum": max(1, min(MAX_ATTEMPTS, max_attempts))},
            "respect_brief_limits": {"type": "boolean"},
        },
        "required": order, "propertyOrdering": order,
    }


def reply_schema(flags: SchemaFlags, *, workers: Sequence[str], aliases: Sequence[str], max_attempts: int,
                 limit_names: Sequence[str], texts: Mapping[str, str] | None = None) -> dict[str, Any]:
    if not (flags.p2 or flags.p3 or flags.p4):
        return decision_json_schema(list(workers), list(aliases), max_attempts=max_attempts)
    mandate = _mandate_schema(workers, aliases, max_attempts, flags.p2)
    if not flags.p4:
        props: dict[str, Any] = {"state_diagnosis": {"type": "string"}}
        if flags.p3:
            props["physical_facts"] = {"type": "string"}
        if not flags.p2:
            props["strategy"] = {"type": "string", "enum": list(STRATEGIES)}
            props["conflict_check"] = {"type": "string"}
        props["rationale_summary"] = {"type": "string"}
        props["action"] = {"type": "string", "enum": list(ACTIONS)}
        props["mandate"] = {"anyOf": [{"type": "null"}, mandate]}
        order = list(props)
        return {"type": "object", "properties": props, "required": order, "propertyOrdering": order}
    t = dict(texts or {})
    values = list(forecast_values(flags))
    forecast_props = {"goal": {"type": "string", "enum": values}}
    for name in limit_names:
        forecast_props[name] = {"type": "string", "enum": values}
    forecast = {"type": "object", "properties": forecast_props, "required": list(forecast_props),
                "propertyOrdering": list(forecast_props)}
    evidence_text = t.get("evidence", "") + (t.get("evidence_physical", "") if flags.p3 else "")
    props = {
        "reasoning": {"type": "string", "description": t.get("reasoning", "")},
        "evidence": {
            "type": "array", "description": evidence_text,
            "items": {"type": "object", "properties": {
                "source": {"type": "string", "enum": list(EVIDENCE_SOURCES)},
                "label": {"type": "string"}, "fact": {"type": "string"}},
                "required": ["source", "label", "fact"], "propertyOrdering": ["source", "label", "fact"]},
        },
    }
    if flags.p4r and flags.p3:
        props["physical_facts"] = {"type": "string", "description": t.get("physical_facts", "")}
    props |= {
        "options": {
            "type": "array", "minItems": 2, "description": t.get("options", ""),
            "items": {"type": "object", "properties": {
                "option": {"type": "string"}, "expected_effect": {"type": "string"}},
                "required": ["option", "expected_effect"], "propertyOrdering": ["option", "expected_effect"]},
        },
    }
    if not flags.p2:
        props["strategy"] = {"type": "string", "enum": list(STRATEGIES), "description": t.get("strategy", "")}
        props["conflict_check"] = {"type": "string", "description": t.get("conflict_check", "")}
    props["rationale"] = {"type": "string", "description": t.get("rationale", "")}
    props["action"] = {"type": "string", "enum": list(ACTIONS), "description": t.get("action", "")}
    props["mandate"] = {"anyOf": [{"type": "null"}, mandate], "description": t.get("mandate", "")}
    props["forecast"] = {"anyOf": [{"type": "null"}, forecast], "description": t.get("forecast", "")}
    order = list(props)
    return {"type": "object", "properties": props, "required": order, "propertyOrdering": order}


def reply_fields_text(schema: Mapping[str, Any], *, flags: SchemaFlags) -> str:
    """The P4 field list written into the prompt (names, order and one instruction each; no JSON)."""

    lines = []
    for name in schema["propertyOrdering"]:
        spec = schema["properties"][name]
        text = spec.get("description", "")
        if name == "strategy":
            text += " (" + ", ".join(STRATEGIES) + ")"
        if name == "forecast":
            text += "; values: " + ", ".join(forecast_values(flags))
        if name == "mandate":
            inner = spec["anyOf"][1]["propertyOrdering"]
            text += " (fields " + ", ".join(inner) + ")"
        lines.append(f"- {name}: {text}")
    return "\n".join(lines)


def _body(text: str) -> Any:
    body = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, flags=re.DOTALL)
    if fenced:
        body = fenced.group(1)
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ReplyError(f"the reply is not valid JSON ({exc.msg})") from exc


def parse_reply(text: str, flags: SchemaFlags, *, workers: Sequence[str], aliases: Sequence[str],
                limit_names: Sequence[str]) -> dict[str, Any]:
    data = _body(text)
    if not isinstance(data, dict):
        raise ReplyError("the reply must be one JSON object")
    action = data.get("action")
    if action not in ACTIONS:
        raise ReplyError("action must be INVOKE_WORKER or STOP")
    raw = data.get("mandate")
    if action == "INVOKE_WORKER" and not isinstance(raw, dict):
        raise ReplyError("mandate is required when the action is INVOKE_WORKER")
    mandate = None
    if action == "INVOKE_WORKER":
        assert isinstance(raw, dict)
        if raw.get("worker_id") not in workers:
            raise ReplyError(f"worker_id {raw.get('worker_id')!r} is not a listed worker")
        if raw.get("base_state") not in aliases:
            raise ReplyError(f"base_state {raw.get('base_state')!r} is not a listed state")
        if flags.p2:
            if raw.get("placement") not in PLACEMENT_ENUM:
                raise ReplyError("placement must be 'may change' or 'must not change'")
            preserve = raw["placement"] == "must not change"
        else:
            if not isinstance(raw.get("preserve_placement"), bool):
                raise ReplyError("preserve_placement must be true or false")
            preserve = raw["preserve_placement"]
        attempts = raw.get("attempts")
        if not isinstance(attempts, int) or isinstance(attempts, bool) or not 1 <= attempts <= MAX_ATTEMPTS:
            raise ReplyError("attempts must be an integer from 1 to 3")
        if not isinstance(raw.get("respect_brief_limits"), bool):
            raise ReplyError("respect_brief_limits must be true or false")
        mandate = {"worker": raw["worker_id"], "base": raw["base_state"], "preserve": preserve,
                   "attempts": attempts, "respect": raw["respect_brief_limits"]}
    out: dict[str, Any] = {"action": action, "mandate": mandate}
    if flags.p4:
        out["reasoning"] = str(data.get("reasoning", ""))
        if "physical_facts" in data:
            out["physical_facts"] = str(data.get("physical_facts", ""))
        out["evidence"] = [e for e in data.get("evidence") or [] if isinstance(e, dict)]
        out["options"] = [o for o in data.get("options") or [] if isinstance(o, dict)]
        out["rationale"] = str(data.get("rationale", ""))
        forecast = data.get("forecast")
        allowed = set(forecast_values(flags))
        if action == "INVOKE_WORKER" and isinstance(forecast, dict):
            out["forecast"] = {k: v for k, v in forecast.items()
                               if k in ("goal", *limit_names) and isinstance(v, str) and v in allowed}
        else:
            out["forecast"] = None
    else:
        out["diagnosis"] = str(data.get("state_diagnosis", ""))
        out["rationale"] = str(data.get("rationale_summary", ""))
        if flags.p3:
            out["physical_facts"] = str(data.get("physical_facts", ""))
    if "strategy" in data:
        out["strategy"] = data.get("strategy")
    if "conflict_check" in data:
        out["conflict_check"] = str(data.get("conflict_check", ""))
    return out


def reply_text_fields(parsed: Mapping[str, Any]) -> str:
    """Every free-text field of a parsed reply, joined (for the citation check)."""

    parts = [str(parsed.get(k, "")) for k in ("reasoning", "diagnosis", "physical_facts", "conflict_check", "rationale")]
    for item in parsed.get("evidence") or []:
        parts.append(f"{item.get('label', '')}: {item.get('fact', '')}")
    for item in parsed.get("options") or []:
        parts.append(f"{item.get('option', '')}: {item.get('expected_effect', '')}")
    return "\n".join(p for p in parts if p)


__all__ = ["MAX_CONFLICT_WORDS", "MAX_DIAGNOSIS_WORDS", "MAX_RATIONALE_CHARS", "ReplyError", "SchemaFlags",
           "forecast_values", "parse_reply", "reply_fields_text", "reply_schema", "reply_text_fields"]
