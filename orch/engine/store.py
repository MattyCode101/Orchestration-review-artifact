"""Content-addressed checkpoint and run-record storage.

A canonical checkpoint is named by its bytes. The first complete import owns
states/<sha>/state.json; later occurrences keep their actual parent in their
RunOutcome and Step records. External inputs are copied before any hard link is
made, so the shared B2 benchmark inode is never linked to a private path.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from orch.contract.models import (
    Claims,
    DesignState,
    Evidence,
    RunOutcome,
    RunRecord,
)


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _safe_name(value: str, *, what: str) -> str:
    """Keep identifiers from escaping the data directory."""

    if value in {".", ".."} or not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"unsafe {what}: {value!r}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    """Durably publish a rename or new directory entry on Linux."""

    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _json_bytes(payload: Any) -> bytes:
    """Use a stable, inspectable JSON representation for file hashes."""

    return (
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_bytes(path: Path, content: bytes) -> None:
    """Write and fsync a same-directory temporary file, then replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_dir(path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """OS flock releases this lock automatically if the holder dies."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class StateStore:
    """Plain-file truth for checkpoint identity and completed run outcomes."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.states_dir = self.root / "states"
        self.runs_dir = self.root / "runs"
        self.states_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def import_state(
        self,
        source: Path,
        *,
        design: str,
        parent_id: str | None = None,
        made_by_run_id: str | None = None,
        stage: str = "routed",
        fake: bool = False,
    ) -> DesignState:
        """Register one exact file and preserve this occurrence's lineage.

        Private finished run outputs are hard-linked to the canonical file.
        Inputs outside data/ are copied, even when hard-linking would work,
        because shared benchmark files must remain independent of our inode.
        """

        src = Path(source).expanduser().resolve(strict=True)
        if not src.is_file():
            raise ValueError(f"state source is not a file: {src}")
        digest = _sha256_file(src)
        state_dir = self.states_dir / digest
        canonical = state_dir / "design.dcp"
        private_source = src.is_relative_to(self.root)
        state = DesignState(
            id=digest,
            design=design,
            stage=stage,
            parent_id=parent_id,
            made_by_run_id=made_by_run_id,
            path=str(canonical),
            fake=fake,
        )

        with _file_lock(self.states_dir / f"{digest}.lock"):
            state_dir.mkdir(parents=True, exist_ok=True)
            if canonical.exists():
                if _sha256_file(canonical) != digest:
                    raise ValueError(f"corrupt canonical checkpoint: {canonical}")
            else:
                temporary = state_dir / f".design.{uuid.uuid4().hex}.tmp"
                try:
                    if private_source:
                        os.link(src, temporary)
                    else:
                        # Copying rather than linking is essential for shared B2 inputs.
                        with src.open("rb") as input_handle, temporary.open("xb") as output_handle:
                            shutil.copyfileobj(input_handle, output_handle, length=8 * 1024 * 1024)
                            output_handle.flush()
                            os.fsync(output_handle.fileno())
                    if _sha256_file(temporary) != digest:
                        raise ValueError("state source changed while it was imported")
                    os.chmod(temporary, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
                    os.replace(temporary, canonical)
                    _fsync_dir(state_dir)
                finally:
                    temporary.unlink(missing_ok=True)

            metadata = state_dir / "state.json"
            if not metadata.exists():
                _atomic_bytes(metadata, _json_bytes(state.model_dump(mode="json")))

            if private_source and src != canonical and not os.path.samefile(src, canonical):
                # Deduplicate a repeated private output without altering its path.
                temporary_link = src.parent / f".{src.name}.{uuid.uuid4().hex}.link"
                try:
                    os.link(canonical, temporary_link)
                    os.replace(temporary_link, src)
                    _fsync_dir(src.parent)
                finally:
                    temporary_link.unlink(missing_ok=True)

        return state

    def state_path(self, state_id: str) -> Path:
        """Return a registered file only if its content still matches its ID."""

        if not _HEX64.fullmatch(state_id):
            raise ValueError(f"invalid state id: {state_id!r}")
        path = self.states_dir / state_id / "design.dcp"
        if not path.is_file() or not (path.parent / "state.json").is_file():
            raise FileNotFoundError(f"incomplete or missing state: {state_id}")
        if path.is_symlink():
            raise ValueError(f"canonical state is a symlink: {path}")
        metadata = DesignState.model_validate_json((path.parent / "state.json").read_bytes())
        if metadata.id != state_id:
            raise ValueError(f"state metadata id mismatch: {path}")
        if _sha256_file(path) != state_id:
            raise ValueError(f"state content hash mismatch: {path}")
        return path

    def save_outcome(self, outcome: RunOutcome) -> None:
        """Commit run files atomically, with complete.json published last."""

        run_id = _safe_name(outcome.run.id, what="run id")
        if outcome.state is not None:
            registered = self.state_path(outcome.state.id)
            if Path(outcome.state.path).resolve() != registered:
                raise ValueError("outcome state path does not name the registered checkpoint")
        if outcome.claims is not None and outcome.claims.run_id != run_id:
            raise ValueError("claims refer to a different run")
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        with _file_lock(run_dir / ".record.lock"):
            complete = run_dir / "complete.json"
            if complete.exists():
                if self.load_outcome(run_id) == outcome:
                    return
                raise FileExistsError(f"completed run id already exists: {run_id}")

            artifacts: dict[str, bytes] = {
                "runinfo.json": _json_bytes(outcome.run.model_dump(mode="json"))
            }
            if outcome.state is not None:
                artifacts["output_state.json"] = _json_bytes(
                    outcome.state.model_dump(mode="json")
                )
            if outcome.evidence is not None:
                artifacts["evidence.json"] = _json_bytes(
                    outcome.evidence.model_dump(mode="json")
                )
            if outcome.claims is not None:
                artifacts["claims.json"] = _json_bytes(
                    outcome.claims.model_dump(mode="json")
                )
            for name, content in artifacts.items():
                _atomic_bytes(run_dir / name, content)
            manifest = {
                "run_id": run_id,
                "files": {
                    name: hashlib.sha256(content).hexdigest()
                    for name, content in artifacts.items()
                },
            }
            _atomic_bytes(complete, _json_bytes(manifest))

    def load_outcome(self, run_id: str) -> RunOutcome:
        """Load only a complete, checksum-matching run outcome."""

        run_id = _safe_name(run_id, what="run id")
        run_dir = self.runs_dir / run_id
        complete = run_dir / "complete.json"
        if not complete.is_file():
            raise FileNotFoundError(f"incomplete or missing run: {run_id}")
        manifest = json.loads(complete.read_text(encoding="utf-8"))
        if manifest.get("run_id") != run_id or not isinstance(manifest.get("files"), dict):
            raise ValueError(f"invalid completion marker: {complete}")
        expected = manifest["files"]
        if "runinfo.json" not in expected:
            raise ValueError("completion marker lacks runinfo.json")
        raw: dict[str, bytes] = {}
        for name, checksum in expected.items():
            if name not in {
                "runinfo.json", "output_state.json", "evidence.json", "claims.json"
            }:
                raise ValueError(f"unexpected run artifact: {name}")
            content = (run_dir / name).read_bytes()
            if hashlib.sha256(content).hexdigest() != checksum:
                raise ValueError(f"run artifact hash mismatch: {name}")
            raw[name] = content
        run = RunRecord.model_validate_json(raw["runinfo.json"])
        if run.id != run_id:
            raise ValueError("runinfo id differs from completion marker")
        state = (
            DesignState.model_validate_json(raw["output_state.json"])
            if "output_state.json" in raw else None
        )
        evidence = (
            Evidence.model_validate_json(raw["evidence.json"])
            if "evidence.json" in raw else None
        )
        claims = (
            Claims.model_validate_json(raw["claims.json"])
            if "claims.json" in raw else None
        )
        outcome = RunOutcome(run=run, state=state, evidence=evidence, claims=claims)
        if state is not None:
            registered = self.state_path(state.id)
            if Path(state.path).resolve() != registered:
                raise ValueError("loaded outcome state path differs from canonical checkpoint")
        return outcome

