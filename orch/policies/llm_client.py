"""Provider-agnostic LLM client: bridge transport, replay cache, rate limit, telemetry.

Final specification, section 8. A call is a pure function of (model, parameters,
prompt, schema); the replay cache is keyed by their SHA-256 so any episode can be
replayed exactly. Modes:
  replay  fail on a miss (never calls the network);
  record  use a stored reply if there is one, otherwise call and store;
  fresh   ignore the main store; call, and store in a private namespace per repeat
          so that a crashed repeat resumes on its own replies (variance runs).
Transport failures (HTTP 429, 5xx, timeouts) are retried inside the bridge and, if
they persist, raised as LLMUnavailableError; they are never cached and never
charged to an episode's recipe-run budget. The API key is read only by the bridge
process from GOOGLE_API_KEY; this module neither reads nor logs it.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

CACHE_SCHEMA = "llm-replay-cache-v2"
_KEY_SHAPE = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")


class LLMUnavailableError(RuntimeError):
    """The service stayed unavailable after the bridge's retries; safe to resume later."""


class LLMCallError(RuntimeError):
    """A non-retryable failure (bad request, authentication, empty or malformed bridge output)."""


class ReplayMiss(LookupError):
    """Replay mode found no stored reply for this prompt."""


@dataclass(frozen=True)
class LLMParams:
    """Everything about a call except the prompt and the schema.

    temperature None and seed None are left out of the request (gemini-4-argon ignores sampling
    parameters and its documentation says to remove them). thinking_level (low, medium or high)
    is the documented thinking control of that model; thinking_budget is the earlier one. The
    two are exclusive.
    """

    model: str
    temperature: float | None = 0.0
    max_output_tokens: int = 1024
    seed: int | None = 0
    thinking_budget: int | None = None
    timeout_ms: int = 60000
    thinking_level: str | None = None

    def key_parameters(self) -> dict[str, Any]:
        """The cache-key parameters. thinking_level joins the key only when set, so every key
        of the earlier model (stored Saturday replies) is unchanged."""

        data = asdict(self)
        data.pop("timeout_ms")
        if data.get("thinking_level") is None:
            data.pop("thinking_level", None)
        return data


@dataclass(frozen=True)
class LLMReply:
    text: str
    prompt_tokens: int
    completion_tokens: int
    thoughts_tokens: int
    total_tokens: int
    latency_ms: float
    finish_reason: str | None
    model: str
    attempts: int = 1


@dataclass(frozen=True)
class LLMResult:
    reply: LLMReply
    cache_hit: bool
    cache_mode: str
    key: str


class Transport(Protocol):
    def complete(self, prompt: str, params: LLMParams, schema: Mapping[str, Any] | None) -> LLMReply:
        """One model call; raises LLMUnavailableError or LLMCallError."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_key(params: LLMParams, prompt: str, schema: Mapping[str, Any] | None) -> str:
    return _sha(_canonical({
        "parameters": params.key_parameters(), "prompt": prompt,
        "schema": None if schema is None else _sha(_canonical(schema)),
    }))


def _scrub(text: str) -> str:
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        value = os.environ.get(name)
        if value and len(value) >= 8:
            text = text.replace(value, "<redacted>")
    return _KEY_SHAPE.sub("<redacted>", text)


class RateLimiter:
    """At most `per_minute` calls per sliding minute across every process sharing a directory."""

    def __init__(self, directory: Path | str, per_minute: int, *,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep) -> None:
        if per_minute < 1:
            raise ValueError("per_minute must be at least one")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.per_minute, self.clock, self.sleep = per_minute, clock, sleep

    def acquire(self) -> None:
        with open(self.directory / "rate.lock", "a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger = self.directory / "rate.json"
            stamps = json.loads(ledger.read_text()) if ledger.is_file() else []
            now = self.clock()
            stamps = [t for t in stamps if now - t < 60.0]
            if len(stamps) >= self.per_minute:
                self.sleep(max(0.0, 60.0 - (now - stamps[0])) + 0.05)
                now = self.clock()
                stamps = [t for t in stamps if now - t < 60.0]
            stamps.append(now)
            ledger.write_text(json.dumps(stamps))


class BridgeTransport:
    """Calls Gemini through tools/llm_bridge/call.py in its own virtual environment."""

    def __init__(self, python: Path | str, script: Path | str, *, extra_args: Sequence[str] = (),
                 request_extras: Mapping[str, Any] | None = None, rate_limiter: RateLimiter | None = None) -> None:
        self.python, self.script = str(python), str(script)
        self.extra_args = tuple(extra_args)
        self.request_extras = dict(request_extras or {})
        self.rate_limiter = rate_limiter

    def complete(self, prompt: str, params: LLMParams, schema: Mapping[str, Any] | None) -> LLMReply:
        request: dict[str, Any] = {
            "model": params.model, "prompt": prompt,
            "max_output_tokens": params.max_output_tokens, "timeout_ms": params.timeout_ms,
        }
        if params.temperature is not None:
            request["temperature"] = params.temperature
        if params.seed is not None:
            request["seed"] = params.seed
        if params.thinking_budget is not None:
            request["thinking_budget"] = params.thinking_budget
        if params.thinking_level is not None:
            request["thinking_level"] = params.thinking_level
        if schema is not None:
            request["response_json_schema"] = dict(schema)
        request.update(self.request_extras)
        if self.rate_limiter is not None:
            self.rate_limiter.acquire()
        try:
            proc = subprocess.run(
                [self.python, self.script, *self.extra_args], input=json.dumps(request),
                capture_output=True, text=True, timeout=params.timeout_ms / 1000.0 * 4 + 60,
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMUnavailableError("the bridge process timed out") from exc
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        try:
            data = json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise LLMCallError(f"the bridge returned no JSON (exit {proc.returncode}): {_scrub(proc.stderr)[-300:]}") from exc
        if not data.get("ok"):
            message = f"{data.get('error_type')}: {_scrub(str(data.get('error_message', '')))[:300]}"
            if data.get("retryable"):
                raise LLMUnavailableError(message)
            raise LLMCallError(message)
        return LLMReply(
            text=data["text"], prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            thoughts_tokens=int(data.get("thoughts_tokens", 0)), total_tokens=int(data.get("total_tokens", 0)),
            latency_ms=float(data.get("latency_ms", 0.0)), finish_reason=data.get("finish_reason"),
            model=str(data.get("model", params.model)), attempts=int(data.get("attempts", 1)),
        )


class ReplayCache:
    """Immutable one-file-per-key store; the first completed record is authoritative."""

    def __init__(self, root: Path | str, *, mode: Literal["replay", "record", "fresh"],
                 fresh_tag: str | None = None) -> None:
        if mode not in ("replay", "record", "fresh"):
            raise ValueError(f"unknown cache mode {mode!r}")
        if mode == "fresh" and not (fresh_tag and re.fullmatch(r"[A-Za-z0-9_.-]+", fresh_tag)):
            raise ValueError("fresh mode needs a simple tag naming the repeat")
        self.mode, self.fresh_tag = mode, fresh_tag
        base = Path(root)
        self.root = base / "fresh" / fresh_tag if mode == "fresh" else base
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def lookup(self, key: str) -> LLMReply | None:
        path = self._path(key)
        if not path.is_file():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        parameters = record.get("parameters", {})
        try:
            params = LLMParams(**parameters)
        except TypeError as exc:
            raise ValueError("replay record has unknown parameters") from exc
        schema = None if record.get("schema") is None else record["schema"]
        if (
            record.get("schema_version") != CACHE_SCHEMA or record.get("key") != key
            or cache_key(params, record.get("prompt", ""), schema) != key
        ):
            raise ValueError(f"replay record {key[:12]} does not match its own key")
        return LLMReply(**record["reply"])

    def store(self, key: str, params: LLMParams, prompt: str, schema: Mapping[str, Any] | None,
              reply: LLMReply) -> LLMReply:
        if self.mode == "replay":
            raise ValueError("replay mode never stores")
        existing = self.lookup(key)
        if existing is not None:
            return existing
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({
            "schema_version": CACHE_SCHEMA, "key": key, "parameters": params.key_parameters(),
            "schema": None if schema is None else dict(schema), "prompt": prompt, "reply": asdict(reply),
            "stored_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, sort_keys=True).encode("utf-8")
        fd, temporary = tempfile.mkstemp(prefix=f".{key[:12]}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            os.unlink(temporary)
        stored = self.lookup(key)
        if stored is None:
            raise ValueError("replay record publication failed")
        return stored


class CachedLLM:
    """The only object the policy talks to: transport behind a replay cache."""

    def __init__(self, transport: Transport | None, cache: ReplayCache, params: LLMParams) -> None:
        self.transport, self.cache, self.params = transport, cache, params

    def complete(self, prompt: str, schema: Mapping[str, Any] | None) -> LLMResult:
        key = cache_key(self.params, prompt, schema)
        stored = self.cache.lookup(key)
        if stored is not None:
            return LLMResult(stored, True, self.cache.mode, key)
        if self.cache.mode == "replay":
            raise ReplayMiss(f"no stored reply for prompt {key[:12]}")
        if self.transport is None:
            raise LLMCallError("no transport configured")
        reply = self.transport.complete(prompt, self.params, schema)
        return LLMResult(self.cache.store(key, self.params, prompt, schema, reply), False, self.cache.mode, key)


class TelemetrySink:
    """Append-only JSON-lines log shared by every process of a campaign."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True) + "\n"
        with open(self.path, "a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(line)
            handle.flush()

    @staticmethod
    def read(path: Path | str) -> list[dict[str, Any]]:
        p = Path(path)
        if not p.is_file():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
