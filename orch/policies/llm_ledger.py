"""Model spend ledger with a hard stop, shared by every process of a campaign.

Thinking tokens are charged at the output price (the provider's pricing page for
gemini-4-argon labels the output rate "including thinking tokens"). A call is refused
before it starts once the recorded total has reached the cap less a headroom that covers
the calls already in flight, so the total ends at or below the cap. Failed calls are not
charged; replays from the reply cache never reach the transport and cost nothing.
"""
from __future__ import annotations

import fcntl
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orch.policies.llm_client import LLMParams, LLMReply, Transport


class SpendCapReached(RuntimeError):
    """The recorded model spend has reached its stop; no further call is made."""


@dataclass(frozen=True)
class Price:
    """US dollars per million tokens. Output covers visible and thinking tokens."""

    input_per_million: float
    output_per_million: float

    def cost(self, prompt_tokens: int, completion_tokens: int, thoughts_tokens: int) -> float:
        return (
            prompt_tokens * self.input_per_million
            + (completion_tokens + thoughts_tokens) * self.output_per_million
        ) / 1_000_000.0


class SpendLedger:
    """Append-only JSON lines, one per charged call, guarded by a file lock."""

    def __init__(self, path: Path | str, *, cap_usd: float, price: Price, headroom_usd: float = 0.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cap_usd, self.price, self.headroom_usd = cap_usd, price, headroom_usd

    @property
    def stop_usd(self) -> float:
        return self.cap_usd - self.headroom_usd

    def _records(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def total_usd(self) -> float:
        return sum(float(record["usd"]) for record in self._records())

    def check(self) -> None:
        total = self.total_usd()
        if total >= self.stop_usd:
            raise SpendCapReached(
                f"model spend US${total:.2f} has reached the stop US${self.stop_usd:.2f} (cap US${self.cap_usd:.2f})"
            )

    def add(self, *, source: str, reply: LLMReply) -> float:
        # Output tokens are the larger of what the counts say and what the provider's total says,
        # so thinking tokens that were billed but not itemised are never missed.
        output = max(reply.completion_tokens + reply.thoughts_tokens, reply.total_tokens - reply.prompt_tokens)
        usd = self.price.cost(reply.prompt_tokens, output, 0)
        record = {
            "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": source,
            "model": reply.model, "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens, "thoughts_tokens": reply.thoughts_tokens,
            "total_tokens": reply.total_tokens, "billed_output_tokens": output, "usd": usd,
        }
        with open(self.path, "a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
        return usd

    def summary(self) -> dict[str, Any]:
        records = self._records()
        return {
            "calls": len(records), "usd": sum(float(r["usd"]) for r in records), "cap_usd": self.cap_usd,
            "stop_usd": self.stop_usd,
            "prompt_tokens": sum(int(r["prompt_tokens"]) for r in records),
            "completion_tokens": sum(int(r["completion_tokens"]) for r in records),
            "thoughts_tokens": sum(int(r["thoughts_tokens"]) for r in records),
        }


class LedgeredTransport:
    """A transport that checks the stop before each call and records its cost after it."""

    def __init__(self, transport: Transport, ledger: SpendLedger, *, source: str) -> None:
        self.transport, self.ledger, self.source = transport, ledger, source

    def complete(self, prompt: str, params: LLMParams, schema: Mapping[str, Any] | None) -> LLMReply:
        self.ledger.check()
        reply = self.transport.complete(prompt, params, schema)
        self.ledger.add(source=self.source, reply=reply)
        return reply
