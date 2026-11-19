"""Structured output of the LLM orchestrator (final specification, section 7).

`LLMDecisionOutput` is validated here and again against the live view by the policy.
`decision_json_schema` builds the JSON Schema sent to the model; it uses only the
keywords the Gemini SDK documents as supported (type, enum, minimum, maximum, anyOf,
properties, required, additionalProperties, propertyOrdering), because word and
length limits are not enforced by the service and are checked in our own code.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from pydantic import Field, ValidationError, model_validator

from orch.contract.models import FrozenModel

STRATEGIES = ("EXPLORE", "PUSH", "REPAIR", "CONTINUE_FROM_VIOLATION", "BACKTRACK", "STOP")
ACTIONS = ("INVOKE_WORKER", "STOP")
MAX_ATTEMPTS = 3
MAX_DIAGNOSIS_WORDS = 60
MAX_CONFLICT_WORDS = 40
MAX_RATIONALE_CHARS = 200
_ORDER = ("state_diagnosis", "strategy", "conflict_check", "rationale_summary", "action", "mandate")
_MANDATE_ORDER = ("worker_id", "base_state", "preserve_placement", "attempts", "respect_brief_limits")


class MandatePayload(FrozenModel):
    worker_id: str
    base_state: str
    preserve_placement: bool
    attempts: int = Field(ge=1, le=MAX_ATTEMPTS)
    respect_brief_limits: bool


class LLMDecisionOutput(FrozenModel):
    state_diagnosis: str
    strategy: str
    conflict_check: str
    rationale_summary: str
    action: str
    mandate: MandatePayload | None

    @model_validator(mode="after")
    def consistent(self) -> "LLMDecisionOutput":
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {', '.join(STRATEGIES)}")
        if self.action not in ACTIONS:
            raise ValueError(f"action must be one of {', '.join(ACTIONS)}")
        if self.action == "INVOKE_WORKER" and self.mandate is None:
            raise ValueError("mandate is required when the action is INVOKE_WORKER")
        if self.action == "STOP" and self.mandate is not None:
            raise ValueError("mandate must be null when the action is STOP")
        return self


class DecisionFormatError(ValueError):
    """The reply is not usable JSON for the decision schema; the message goes to a repair turn."""


def parse_decision(text: str) -> LLMDecisionOutput:
    """Parse a raw model reply; tolerate a markdown fence but nothing else."""

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
        return LLMDecisionOutput.model_validate(data)
    except ValidationError as exc:
        problems = []
        for error in exc.errors(include_url=False):
            where = ".".join(str(part) for part in error["loc"]) or "reply"
            problems.append(f"{where}: {error['msg']}")
        raise DecisionFormatError("; ".join(problems[:4])) from exc


def clip_words(text: str, limit: int) -> str:
    words = text.split()
    return text.strip() if len(words) <= limit else " ".join(words[:limit])


def decision_json_schema(
    worker_ids: Sequence[str], aliases: Sequence[str], *, max_attempts: int = MAX_ATTEMPTS,
) -> dict[str, Any]:
    """JSON Schema of one decision, with this episode's workers, aliases and attempt cap."""

    mandate = {
        "type": "object",
        "properties": {
            "worker_id": {"type": "string", "enum": list(worker_ids)},
            "base_state": {"type": "string", "enum": list(aliases)},
            "preserve_placement": {"type": "boolean"},
            "attempts": {"type": "integer", "minimum": 1, "maximum": max(1, min(MAX_ATTEMPTS, max_attempts))},
            "respect_brief_limits": {"type": "boolean"},
        },
        "required": list(_MANDATE_ORDER),
        "propertyOrdering": list(_MANDATE_ORDER),
    }
    return {
        "type": "object",
        "properties": {
            "state_diagnosis": {"type": "string"},
            "strategy": {"type": "string", "enum": list(STRATEGIES)},
            "conflict_check": {"type": "string"},
            "rationale_summary": {"type": "string"},
            "action": {"type": "string", "enum": list(ACTIONS)},
            "mandate": {"anyOf": [{"type": "null"}, mandate]},
        },
        "required": list(_ORDER),
        "propertyOrdering": list(_ORDER),
    }
