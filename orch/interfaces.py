"""Stable component boundaries for the shared step engine.

Adapters translate recipes into launch specifications. The runner owns all
process lifetime control. The meter alone writes Evidence. The backend is the
small seam used by both real tools and the labelled fake backend.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Mapping, Protocol, Literal

from pydantic import Field, model_validator

from orch.contract.models import (
    Claims,
    DesignState,
    Evidence,
    FrozenModel,
    Mandate,
    RunAttempt,
    RunOutcome,
    RunStatus,
)
from orch.contract.views import PolicyView


class RecipeSpec(FrozenModel):
    """A registry entry after content-hash version resolution."""

    id: str
    worker: str
    kind: Literal["tcl", "latch"]
    script: str | None = None
    flow: str | None = None
    replaces: bool
    cause: str
    source: str
    source_sha256: str
    timeout_s: int = Field(gt=0)
    version: str

    @model_validator(mode="after")
    def source_for_kind(self) -> "RecipeSpec":
        if self.kind == "tcl" and not self.script:
            raise ValueError("tcl recipe requires script")
        if self.kind == "latch" and not self.flow:
            raise ValueError("latch recipe requires flow")
        return self


class LaunchSpec(FrozenModel):
    """The complete launch request; env_overrides are not logged verbatim."""

    argv: tuple[str, ...]
    cwd: str
    env_overrides: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = Field(gt=0)
    output_hint: str | None = None
    input_copy_path: str | None = None
    script_sha256: str | None = None
    tool_versions: dict[str, str] = Field(default_factory=dict)
    java_version: str | None = None


class ProcessResult(FrozenModel):
    """Observed process outcome, including every transient retry attempt."""

    run_id: str
    status: RunStatus
    exit_code: int | None
    wall_s: float = Field(ge=0)
    cpu_user_s: float | None = Field(default=None, ge=0)
    cpu_sys_s: float | None = Field(default=None, ge=0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    load_1m_start: float | None = Field(default=None, ge=0)
    load_1m_end: float | None = Field(default=None, ge=0)
    busy_slots_start: int | None = Field(default=None, ge=0)
    backpressure_wait_s: float = Field(default=0, ge=0)
    backpressure_events: int = Field(default=0, ge=0)
    process_group_kills: int = Field(ge=0)
    tag_sweep_kills: int = Field(ge=0)
    tag: str
    stdout_path: str
    stderr_path: str
    attempts: tuple[RunAttempt, ...] = ()


class Runner(Protocol):
    def run(self, spec: LaunchSpec, *, run_id: str, run_dir: Path) -> ProcessResult:
        """Run one tagged tool job under a slot, deadline and cleanup."""


class RecipeAdapter(Protocol):
    kind: str

    def build_launch(
        self, recipe: RecipeSpec, *, input_path: Path, run_dir: Path, run_id: str
    ) -> LaunchSpec:
        """Translate a recipe and a private input copy into a native command."""

    def find_output(self, recipe: RecipeSpec, *, run_dir: Path) -> Path | None:
        """Return only a completed checkpoint, if one exists."""

    def classify(
        self, recipe: RecipeSpec, *, run_dir: Path, process: ProcessResult
    ) -> RunStatus:
        """Inspect native status files as well as the process exit code."""

    def parse_claims(
        self, recipe: RecipeSpec, *, run_dir: Path, run_id: str
    ) -> Claims:
        """Parse worker claims without creating Evidence."""


class Meter(Protocol):
    version: str

    def measure(
        self, state: DesignState, *, run_dir: Path, start: Evidence | None = None
    ) -> Evidence:
        """Re-open a checkpoint and return independent measured evidence."""


class Backend(Protocol):
    def execute(
        self,
        state: DesignState,
        recipe: RecipeSpec,
        *,
        run_id: str,
        run_dir: Path,
    ) -> RunOutcome:
        """Run and measure a recipe; neither brief nor guards are inputs."""

    def measure_start(self, state: DesignState, *, run_dir: Path) -> Evidence:
        """Measure the episode start without charging a recipe run."""


class StateStore(Protocol):
    root: Path

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
        """Register a file by its SHA-256 and hard-link it into the store."""

    def state_path(self, state_id: str) -> Path:
        """Return the canonical checkpoint path, checking that it exists."""

    def save_outcome(self, outcome: RunOutcome) -> None:
        """Persist a run, its claims and independent evidence."""

    def load_outcome(self, run_id: str) -> RunOutcome:
        """Load a persisted run outcome with model validation."""


class RunCache(Protocol):
    def key(self, state_id: str, recipe_id: str, recipe_version: str) -> str:
        """Hash the input state and exact recipe version."""

    def locked(self, key: str) -> AbstractContextManager[None]:
        """Hold an OS file lock for one cache key."""

    def get(self, key: str) -> str | None:
        """Return the original run id on a terminal hit."""

    def put(self, key: str, run_id: str, *, retryable: bool = False) -> None:
        """Atomically publish a cache entry after the run is persisted."""


class Policy(Protocol):
    id: str
    version: str

    def decide(self, view: PolicyView) -> Mandate | None:
        """Choose a worker, base state, settings and guards, or stop."""
