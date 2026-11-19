"""Durable per-episode JSONL journal with crash-tail recovery."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel

from orch.engine.store import _fsync_dir, _json_bytes, _safe_name


class Journal:
    """Append one checksummed event per line and resume after a torn tail."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.episodes_dir = self.root / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

    def path(self, episode_id: str) -> Path:
        """The journal path for a safe episode identifier."""

        episode_id = _safe_name(episode_id, what="episode id")
        return self.episodes_dir / episode_id / "journal.jsonl"

    @staticmethod
    def _parse(data: bytes) -> tuple[list[dict[str, Any]], int]:
        """Return complete events and the byte offset before an incomplete tail."""

        events: list[dict[str, Any]] = []
        complete_length = 0
        for line in data.splitlines(keepends=True):
            if not line.endswith(b"\n"):
                break
            try:
                wrapper = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("corrupt complete journal line") from exc
            if not isinstance(wrapper, dict) or not isinstance(wrapper.get("event"), dict):
                raise ValueError("journal line lacks an event object")
            event = wrapper["event"]
            digest = hashlib.sha256(_json_bytes(event)).hexdigest()
            if wrapper.get("sha256") != digest:
                raise ValueError("journal line checksum mismatch")
            events.append(event)
            complete_length += len(line)
        return events, complete_length

    def read(self, episode_id: str) -> list[dict[str, Any]]:
        """Read committed events; ignore an incomplete final line."""

        path = self.path(episode_id)
        if not path.is_file():
            return []
        lock = path.parent / "journal.lock"
        fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            events, _ = self._parse(path.read_bytes())
            return events
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def append(
        self, episode_id: str, record: Mapping[str, Any] | BaseModel
    ) -> None:
        """Fsync one event; truncate a torn tail before the next append."""

        event = (
            record.model_dump(mode="json")
            if isinstance(record, BaseModel)
            else dict(record)
        )
        event_bytes = _json_bytes(event)
        wrapper = {
            "event": event,
            "sha256": hashlib.sha256(event_bytes).hexdigest(),
        }
        line = _json_bytes(wrapper)
        path = self.path(episode_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.parent / "journal.lock"
        lock_fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            file_fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                existing = path.read_bytes()
                events, complete_length = self._parse(existing)
                if len(existing) != complete_length:
                    os.ftruncate(file_fd, complete_length)
                    os.fsync(file_fd)
                event_id = event.get("id")
                if event_id is not None:
                    for old in events:
                        if old.get("id") == event_id:
                            if old == event:
                                return
                            raise ValueError(f"journal event id reused: {event_id}")
                os.lseek(file_fd, 0, os.SEEK_END)
                remaining = memoryview(line)
                while remaining:
                    written = os.write(file_fd, remaining)
                    remaining = remaining[written:]
                os.fsync(file_fd)
            finally:
                os.close(file_fd)
            _fsync_dir(path.parent)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

