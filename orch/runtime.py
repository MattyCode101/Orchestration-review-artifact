"""Compose the observed real-tool components under one project-local data root.

UNTESTED REAL TOOL PATH until the detached smoke reports complete. No hardware
command is invented here; the adapters and copied run-3/4 scripts own commands.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from orch.engine.apply import StepEngine
from orch.engine.cache import RunCache
from orch.engine.meter import VivadoMeter
from orch.engine.real_backend import RealBackend
from orch.engine.runner import Runner
from orch.engine.store import StateStore
from orch.recipes.adapters.latch import LatchAdapter
from orch.recipes.adapters.tcl import TclAdapter
from orch.recipes.adapters.validator import ValidatorAdapter
from orch.recipes.registry import Registry, load_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_STATUS = PROJECT_ROOT / "data" / "smoke_vexriscv_b2" / "status.json"
LATCH_VENV_STATUS = PROJECT_ROOT / "data" / "smoke_latch_venv" / "status.json"


@dataclass(frozen=True)
class RealRuntime:
    """Bound real engine, registry and final-only validator."""

    data_root: Path
    store: StateStore
    cache: RunCache
    runner: Runner
    meter: VivadoMeter
    backend: RealBackend
    engine: StepEngine
    registry: Registry
    validator: ValidatorAdapter


def _require_deterministic_smoke(path: Path = SMOKE_STATUS) -> None:
    """Permit real caching only after all six uncached smoke outputs pass."""

    if not path.is_file():
        raise RuntimeError("real cache is disabled until the VexRiscv smoke passes")
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("smoke status cannot prove real-cache safety") from exc
    checks = status.get("determinism", {})
    if (
        status.get("phase") != "complete"
        or status.get("baseline_match") is not True
        or status.get("deterministic") is not True
        or len(status.get("runs", ())) != 6
        or not all(
            checks.get(recipe, {}).get("identical_metrics") is True
            for recipe in ("C0_reroute", "LATCH.phys_opt", "A_R_EA")
        )
        or status.get("tags", {}).get("tagged_survivor_pids") != []
    ):
        raise RuntimeError("smoke has not proved safe real caching")


_REQUIRED_SMOKE_METRICS = ("wns_ns", "tns_ns", "packed_luts", "netlist_luts")


def _has_complete_smoke_metrics(metrics: object) -> bool:
    """Require all four repeated measurements as finite JSON numbers."""

    if not isinstance(metrics, dict):
        return False
    for name in _REQUIRED_SMOKE_METRICS:
        value = metrics.get(name)
        if type(value) not in (int, float):
            return False
        try:
            if not math.isfinite(value):
                return False
        except OverflowError:
            return False
    return True


def _require_latch_venv_smoke(
    path: Path,
    *,
    recipe_version: str,
    meter_version: str,
    interpreter: str,
) -> None:
    """Require two current-version LATCH runs through the pinned venv path."""

    if not path.is_file():
        raise RuntimeError("real cache is disabled until the corrected LATCH venv smoke passes")
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("corrected LATCH smoke status cannot be read") from exc
    runs = status.get("runs")
    expected_prefix = str(Path(interpreter).parent.parent)
    if (
        status.get("phase") != "complete"
        or status.get("fake") is not False
        or status.get("cache_enabled") is not False
        or status.get("latch_python") != interpreter
        or status.get("latch_prefix") != expected_prefix
        or status.get("recipe_version") != recipe_version
        or status.get("meter_version") != meter_version
        or status.get("deterministic") is not True
        or not isinstance(runs, list)
        or len(runs) != 2
        or [row.get("repeat") for row in runs] != [1, 2]
        or any(
            not isinstance(row, dict)
            or row.get("run_status") != "ok"
            or row.get("cache_hit") is not False
            or row.get("physically_valid") is not True
            or not _has_complete_smoke_metrics(row.get("metrics"))
            for row in runs
        )
        or runs[0]["metrics"] != runs[1]["metrics"]
        or status.get("tags", {}).get("tagged_survivor_pids") != []
        or status.get("tags", {}).get("runner_jobs_checked", 0) < 1
    ):
        raise RuntimeError("corrected LATCH venv smoke has not proved safe real caching")


def make_real_runtime(
    data_root: Path | None = None,
    *,
    cache_enabled: bool = False,
    max_slots: int = 24,
    load_ceiling_1m: float | None = None,
    min_available_memory_bytes: int | None = None,
    latch_smoke_status: Path | None = None,
) -> RealRuntime:
    """Build a real engine; reject writes outside this Development project."""

    root = (data_root or PROJECT_ROOT / "data" / "real").expanduser().resolve()
    if not root.is_relative_to(PROJECT_ROOT / "data"):
        raise ValueError("real data root must stay inside this Development project")
    if cache_enabled:
        _require_deterministic_smoke()
    store = StateStore(root)
    cache = RunCache(root)
    runner = Runner(
        PROJECT_ROOT / "data" / "slots", max_slots=max_slots,
        load_ceiling_1m=load_ceiling_1m,
        min_available_memory_bytes=min_available_memory_bytes,
    )
    meter = VivadoMeter(
        runner=runner,
        env_script=PROJECT_ROOT / "recipes" / "support" / "env.sh",
        tool_version="2025.1",
    )
    latch_adapter = LatchAdapter(PROJECT_ROOT)
    backend = RealBackend(
        store=store,
        runner=runner,
        meter=meter,
        adapters={
            "tcl": TclAdapter(PROJECT_ROOT),
            "latch": latch_adapter,
        },
    )
    registry = load_registry(PROJECT_ROOT)
    if cache_enabled:
        smoke_status = (
            Path(latch_smoke_status).expanduser().resolve()
            if latch_smoke_status is not None else LATCH_VENV_STATUS
        )
        if not smoke_status.is_relative_to(PROJECT_ROOT / "data"):
            raise ValueError("cache smoke status must stay inside Development data")
        _require_latch_venv_smoke(
            smoke_status,
            recipe_version=registry.get("LATCH.phys_opt").version,
            meter_version=meter.version,
            interpreter=str(latch_adapter.latch_python),
        )
    engine = StepEngine(
        store=store,
        cache=cache,
        backend=backend,
        meter_version=meter.version,
        cache_enabled=cache_enabled,
    )
    return RealRuntime(
        data_root=root,
        store=store,
        cache=cache,
        runner=runner,
        meter=meter,
        backend=backend,
        engine=engine,
        registry=registry,
        validator=ValidatorAdapter(PROJECT_ROOT),
    )
