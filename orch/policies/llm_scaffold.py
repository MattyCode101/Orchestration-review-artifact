"""Replay-only LLM policy scaffold over the safe PolicyView contract.

This module has no API client. A caller supplies a synchronous client protocol;
tests use a fake client. The model chooses only the mandate fields it controls.
The fixed brief hash, context reference, and mandate ID are assigned here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

from pydantic import Field, ValidationError

from orch.contract.brief_text import render_brief_text
from orch.contract.models import (
    FrozenModel,
    Guard,
    Mandate,
    MetricName,
    MetricValue,
    WorkerSettings,
)
from orch.contract.views import PolicyView

PROMPT_SCHEMA_VERSION = "llm-policy-view-v1"
CACHE_SCHEMA_VERSION = "llm-response-cache-v1"


def _canonical(value: object) -> str:
    """Use one stable JSON representation for prompts, IDs, and cache records."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def prompt_sha256(prompt: str) -> str:
    """Name an exact prompt by its UTF-8 bytes."""

    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class MandateChoice(FrozenModel):
    """Only the fields an LLM may choose; the engine supplies fixed identities."""

    worker_id: str
    base_state_id: str
    settings: WorkerSettings = Field(default_factory=WorkerSettings)
    guards: tuple[Guard, ...] = ()


def mandate_json_schema() -> dict[str, object]:
    """Return a JSON schema for either stop or an LLM-controlled mandate choice."""

    choice = MandateChoice.model_json_schema()
    definitions = choice.pop("$defs", {})
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Mandate choice or stop",
        "oneOf": [{"type": "null"}, choice],
        "$defs": definitions,
    }


def serialize_view(view: PolicyView) -> str:
    """Serialize only policy-safe data and the frozen natural-language brief."""

    return _canonical({
        "schema_version": PROMPT_SCHEMA_VERSION,
        "instruction": (
            "Return JSON null to stop, or one object matching mandate_choice_schema. "
            "Choose a worker and a physically valid visited base state. "
            "Do not name a recipe. Return JSON only."
        ),
        "brief_text": render_brief_text(view.brief),
        "policy_view": view.model_dump(mode="json"),
        "mandate_choice_schema": mandate_json_schema(),
    })


def _retry_prompt(original: str, error_code: str) -> str:
    """Give one bounded, deterministic retry with the original view and schema."""

    return _canonical({
        "schema_version": "llm-policy-retry-v1",
        "retry_of_sha256": prompt_sha256(original),
        "validation_error": error_code,
        "instruction": "One retry remains. Return JSON null or a valid mandate choice.",
        "original_prompt": json.loads(original),
    })


@dataclass(frozen=True)
class LLMReply:
    """A supplied client reply and optional usage/latency observations."""

    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise ValueError("LLM reply content must be text")
        for name in ("prompt_tokens", "completion_tokens"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer")
        latency = self.latency_ms
        if latency is not None and (
            type(latency) not in (int, float)
            or not math.isfinite(latency)
            or latency < 0
        ):
            raise ValueError("latency_ms must be finite and nonnegative")


class LLMClient(Protocol):
    """Injected completion boundary; no network implementation lives here."""

    def complete(self, prompt: str) -> LLMReply:
        """Return one raw JSON reply plus optional token and latency counts."""


class PromptResponseCache:
    """Create one immutable response record per exact prompt hash.

    A hard-link create makes the first completed record authoritative even if
    two callers race. Temporary files remain in this directory, so a crash
    cannot expose a partially written response under its prompt key.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.root = self.root.resolve(strict=True)

    def _path(self, prompt: str) -> Path:
        return self.root / f"{prompt_sha256(prompt)}.json"

    def read(self, prompt: str) -> LLMReply | None:
        """Replay the exact recorded reply or fail on a corrupt key."""

        path = self._path(prompt)
        if not path.is_file():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != CACHE_SCHEMA_VERSION
            or record.get("prompt_sha256") != prompt_sha256(prompt)
            or record.get("prompt") != prompt
            or not isinstance(record.get("reply"), dict)
        ):
            raise ValueError("response cache record differs from its prompt")
        try:
            return LLMReply(**record["reply"])
        except (TypeError, ValueError) as exc:
            raise ValueError("response cache reply is invalid") from exc

    def write(self, prompt: str, reply: LLMReply) -> tuple[LLMReply, bool]:
        """Publish atomically; a concurrent winner is replayed as authoritative."""

        existing = self.read(prompt)
        if existing is not None:
            return existing, True
        path = self._path(prompt)
        payload = (_canonical({
            "schema_version": CACHE_SCHEMA_VERSION,
            "prompt_sha256": prompt_sha256(prompt),
            "prompt": prompt,
            "reply": asdict(reply),
        }) + "\n").encode("utf-8")
        fd, temporary = tempfile.mkstemp(
            prefix=f".{path.stem}.", suffix=".tmp", dir=self.root
        )
        created = False
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
                created = True
            except FileExistsError:
                pass
        finally:
            os.unlink(temporary)
        saved = self.read(prompt)
        if saved is None:
            raise ValueError("response cache publication failed")
        return saved, not created


@dataclass(frozen=True)
class LLMAttemptLog:
    """Fields an episode journal can record without storing client internals."""

    prompt_sha256: str
    response_sha256: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float
    cache_hit: bool
    validation_error: str | None


class ResponseValidationError(ValueError):
    """A bounded error code suitable for one retry prompt and a journal."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _StartMetricProxy:
    """Supply only the metric lookup used by Mandate.validate_against."""

    def __init__(self, values: dict[MetricName, float]) -> None:
        self.values = values

    def metric(self, name: MetricName) -> MetricValue | None:
        value = self.values.get(name)
        if value is None:
            return None
        return MetricValue(value=value, unit="policy-view", method="policy-view")


def _validate_response(
    content: str, view: PolicyView, original_prompt_hash: str,
) -> Mandate | None:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ResponseValidationError("invalid_json") from exc
    if value is None:
        return None
    try:
        choice = MandateChoice.model_validate_json(content, strict=True)
    except ValidationError as exc:
        raise ResponseValidationError("invalid_mandate_schema") from exc
    card = next(
        (item for item in view.worker_cards if item.id == choice.worker_id),
        None,
    )
    if card is None:
        raise ResponseValidationError("unknown_worker")
    valid_bases = {
        state.state_id
        for state in (view.start, view.current, *view.visited)
        if state.physically_valid
    }
    if choice.base_state_id not in valid_bases:
        raise ResponseValidationError("base_not_physically_visited")
    if choice.settings.resolved_max_runs(card) > view.budget_left:
        raise ResponseValidationError("attempts_exceed_budget")
    choice_json = _canonical(choice.model_dump(mode="json"))
    identity = hashlib.sha256(
        f"{original_prompt_hash}:{choice_json}".encode("utf-8")
    ).hexdigest()
    mandate = Mandate(
        id=f"mandate_{identity[:24]}",
        worker_id=choice.worker_id,
        base_state_id=choice.base_state_id,
        settings=choice.settings,
        guards=choice.guards,
        brief_hash=view.brief.fingerprint(),
        context_ref=original_prompt_hash,
    )
    try:
        # The contract checks each hard limit using the episode's start metrics.
        mandate.validate_against(view.brief, _StartMetricProxy(view.start.metrics))
    except ValueError as exc:
        raise ResponseValidationError("brief_guard_mismatch") from exc
    return mandate


class LLMPolicyScaffold:
    """Policy boundary with exact replay, one validation retry, then stop."""

    id = "llm_scaffold"
    version = "0.1.0"

    def __init__(self, client: LLMClient, cache: PromptResponseCache) -> None:
        self.client = client
        self.cache = cache
        self.attempt_log: list[LLMAttemptLog] = []

    def _reply(self, prompt: str) -> tuple[LLMReply, bool]:
        cached = self.cache.read(prompt)
        if cached is not None:
            return cached, True
        started = time.perf_counter()
        reply = self.client.complete(prompt)
        if not isinstance(reply, LLMReply):
            raise TypeError("injected LLM client must return LLMReply")
        if reply.latency_ms is None:
            reply = replace(
                reply, latency_ms=(time.perf_counter() - started) * 1000.0
            )
        return self.cache.write(prompt, reply)

    def decide(self, view: PolicyView) -> Mandate | None:
        """Return a validated mandate, or stop after two invalid responses."""

        if view.budget_left == 0:
            return None
        original = serialize_view(view)
        original_hash = prompt_sha256(original)
        prompt = original
        for retry_index in range(2):
            reply, cache_hit = self._reply(prompt)
            error_code: str | None = None
            try:
                mandate = _validate_response(reply.content, view, original_hash)
            except ResponseValidationError as exc:
                mandate = None
                error_code = exc.code
            self.attempt_log.append(LLMAttemptLog(
                prompt_sha256=prompt_sha256(prompt),
                response_sha256=hashlib.sha256(
                    reply.content.encode("utf-8")
                ).hexdigest(),
                prompt_tokens=reply.prompt_tokens,
                completion_tokens=reply.completion_tokens,
                latency_ms=float(reply.latency_ms or 0.0),
                cache_hit=cache_hit,
                validation_error=error_code,
            ))
            if error_code is None:
                return mandate
            if retry_index == 0:
                prompt = _retry_prompt(original, error_code)
        return None
