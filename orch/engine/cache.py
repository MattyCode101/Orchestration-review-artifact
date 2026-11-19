"""Per-run cache keyed only by input bytes, recipe ID and recipe version.

The caller holds locked(key) across lookup, execution, outcome persistence and
put. A retryable entry records the earlier attempt but is not a terminal hit.
The pure gate always runs again for the current brief.
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from orch.engine.store import _atomic_bytes, _file_lock, _json_bytes, _safe_name


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class RunCache:
    """Atomic cache pointers with one OS file lock per exact step key."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.cache_dir = self.root / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(state_id: str, recipe_id: str, recipe_version: str) -> str:
        """Avoid ambiguous concatenation by hashing a canonical JSON tuple."""

        if not _HEX64.fullmatch(state_id):
            raise ValueError("state_id must be a SHA-256 hex digest")
        if not recipe_id or not recipe_version:
            raise ValueError("recipe id and version are required")
        payload = _json_bytes([state_id, recipe_id, recipe_version])
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _check_key(key: str) -> str:
        if not _HEX64.fullmatch(key):
            raise ValueError("cache key must be a SHA-256 hex digest")
        return key

    @contextmanager
    def locked(self, key: str) -> Iterator[None]:
        """Wait on a per-key flock; a crash releases it automatically."""

        key = self._check_key(key)
        with _file_lock(self.cache_dir / f"{key}.lock"):
            yield

    def _entry(self, key: str) -> dict[str, object] | None:
        key = self._check_key(key)
        path = self.cache_dir / f"{key}.json"
        if not path.is_file():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"corrupt cache entry: {path}") from exc
        if (
            not isinstance(entry, dict)
            or entry.get("key") != key
            or not isinstance(entry.get("run_id"), str)
            or not isinstance(entry.get("retryable"), bool)
        ):
            raise ValueError(f"invalid cache entry: {path}")
        _safe_name(entry["run_id"], what="cached run id")
        return entry

    def get(self, key: str) -> str | None:
        """Return a terminal run ID, or None on miss or retryable attempt."""

        entry = self._entry(key)
        if entry is None or entry["retryable"]:
            return None
        run_id = str(entry["run_id"])
        if not (self.root / "runs" / run_id / "complete.json").is_file():
            raise ValueError(f"dangling terminal cache pointer: {run_id}")
        return run_id

    def put(self, key: str, run_id: str, *, retryable: bool = False) -> None:
        """Publish a pointer only after the run outcome has been persisted.

        The caller should hold locked(key). Replacing an older entry allows a
        corrected meter version to be measured and cached without changing the
        run key specified by the architecture.
        """

        key = self._check_key(key)
        run_id = _safe_name(run_id, what="cached run id")
        if not (self.root / "runs" / run_id / "complete.json").is_file():
            raise FileNotFoundError(f"run must be complete before caching: {run_id}")
        payload = {"key": key, "run_id": run_id, "retryable": bool(retryable)}
        _atomic_bytes(self.cache_dir / f"{key}.json", _json_bytes(payload))

    def last_attempt(self, key: str) -> str | None:
        """Inspect a retryable record without treating it as a cache hit."""

        entry = self._entry(key)
        return str(entry["run_id"]) if entry is not None else None

