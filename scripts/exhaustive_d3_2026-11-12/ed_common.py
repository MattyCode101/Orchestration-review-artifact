"""Exhaustive depth-3 reference on the five TEST designs (2026-11-12): shared constants and readers.

Read-only helpers over the stored run graph. Nothing here edits an existing root. Power recipes are never
read: only the ten timing and area recipes are accepted into the graph (the sealed test power roots stay closed).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT = Path(os.environ.get("ORCH_PROJECT_ROOT") or Path(__file__).resolve().parents[2])
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

WORK = PROJECT / "data/exhaustive_d3_2026-11-12"
ENGINE = WORK / "engine"          # this task's own engine root (states, runs, cache, verification)
PLAN = WORK / "plan"
LOGS = WORK / "logs"
JOURNAL = WORK / "journal.jsonl"
SMOKE_JOURNAL = WORK / "journal_smoke.jsonl"
STATUS = WORK / "status.json"
STATUS_HISTORY = WORK / "status_history.jsonl"
PIDFILE = WORK / "queue.pid.json"
STOPFILE = WORK / "STOP"
CONFIG = WORK / "queue_config.json"

DESIGNS = (
    "rosetta_digit-recognition", "corescore_500_mod", "vexriscv_re-place_v2",
    "fir_symmetric_systolic", "fir_systolic_transposed",
)
B2 = {
    "rosetta_digit-recognition": "57a8689b62581982847e4140f76f53f731925c664bc7a8fc9a585a3789bd19c4",
    "corescore_500_mod": "85b7470eff50093d10c3b06e3e458f10488effe9ef1f9e563948edabea1b258f",
    "vexriscv_re-place_v2": "8e5c5a6968f01a06aef124e2d17fffe920f559f76227ff77ba5c99b4525b8272",
    "fir_symmetric_systolic": "a8cbdb23cdcc3c71e26e03e2c74a1aa43c42d9ee79eef37ae8fbe374e976e9ec",
    "fir_systolic_transposed": "20329fc34fdfb2da036cf221da9c27af33a7c735bb5421fc0e93ce6744f49020",
}
TIMING = ("LATCH.phys_opt", "LATCH.retime", "LATCH.auto_pblock", "LATCH.hardblock_relocate", "LATCH.memory_path_opt")
AREA = ("A_R_EA", "A_R_RSA_SRLM", "A_R_RSA_AREM", "A_R_RSA_MED", "A_E_DUPL_Q1000")
REC10 = TIMING + AREA
POWER4 = ("P_ROUTE_HIGHDELAY", "P_POST_ALL", "P_POST_BRAM", "P_PRE_ALL")
CUR_METER = "0dd0cf9424bbb7ab20dc5cf981bc35068c4ba08612ecd0f6c46098e6bf880bbe"

# Roots that may hold timing/area runs on the five test designs. The sealed power roots are NOT listed.
SRC_ROOTS = {
    "test_base": PROJECT / "data/exp_evening_2026-11-04/test_b2/overnight_baseline_2026-11-04",
    "campaign": PROJECT / "data/campaign_test_final/engine",
    "repeats_c0": PROJECT / "data/exp_evening_2026-11-04/repeats_c0",
}


def jread(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jwrite(path: Path | str, payload: Any, *, indent: int | None = 1) -> None:
    """Atomic JSON write (same-directory temp file, fsync, replace)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class UF:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra
            return True
        return False


def versions() -> dict[str, str]:
    from orch.recipes.registry import load_registry

    reg = load_registry(PROJECT)
    return {r: reg.get(r).version for r in REC10}


def load_runs(roots: dict[str, Path], vers: dict[str, str]) -> tuple[list[dict], dict]:
    """Every completed OK run of the ten recipes under the current meter and current recipe versions."""

    runs: list[dict] = []
    skipped: dict[str, int] = defaultdict(int)
    for name, root in roots.items():
        rd = root / "runs"
        if not rd.is_dir():
            continue
        for d in os.scandir(rd):
            if not d.is_dir():
                continue
            p = Path(d.path)
            try:
                if not (p / "complete.json").is_file():
                    skipped[f"{name}:incomplete"] += 1
                    continue
                info = jread(p / "runinfo.json")
            except Exception:
                skipped[f"{name}:unreadable"] += 1
                continue
            rec = info.get("recipe_id")
            if rec not in vers:
                skipped[f"{name}:other_recipe"] += 1          # includes every power recipe: never read further
                continue
            if info.get("status") != "ok":
                skipped[f"{name}:status_{info.get('status')}"] += 1
                continue
            if not (p / "output_state.json").is_file() or not (p / "evidence.json").is_file():
                skipped[f"{name}:no_output_or_evidence"] += 1
                continue
            ev = jread(p / "evidence.json")
            if ev.get("meter_version") != CUR_METER:
                skipped[f"{name}:old_meter"] += 1
                continue
            if info.get("recipe_version") != vers[rec]:
                skipped[f"{name}:other_recipe_version"] += 1
                continue
            m = ev["metrics"]
            runs.append({
                "root": name, "run": d.name, "in": info["input_sha256_before"], "recipe": rec,
                "out": jread(p / "output_state.json")["id"],
                "hit": bool(info.get("cache_hit")), "orig": info.get("original_run_id"),
                "wall": float(info.get("wall_s") or 0.0), "cu": info.get("cpu_user_s"), "cs": info.get("cpu_sys_s"),
                "peak": info.get("peak_memory_bytes"), "load0": info.get("load_1m_start"), "busy0": info.get("busy_slots_start"),
                "wns": m["wns_ns"]["value"], "luts": m["packed_luts"]["value"], "pow": m["dynamic_power_w"]["value"],
                "routed": ev.get("routed"), "rerr": ev.get("route_errors"), "drc": ev.get("drc_errors"),
            })
    return runs, dict(skipped)


def twin_classes(runs: list[dict]) -> tuple[UF, dict[str, set[str]], dict[str, set[tuple]]]:
    """Power-study rule: outputs of runs that share (input class, recipe, version) are one class; to fixpoint."""

    uf = UF()
    for r in runs:
        uf.find(r["in"]); uf.find(r["out"])
    while True:
        groups: dict[tuple, set[str]] = defaultdict(set)
        for r in runs:
            groups[(uf.find(r["in"]), r["recipe"])].add(r["out"])
        changed = False
        for outs in groups.values():
            outs = sorted(outs)
            for o in outs[1:]:
                changed |= uf.union(outs[0], o)
        if not changed:
            break
    members: dict[str, set[str]] = defaultdict(set)
    metrics: dict[str, set[tuple]] = defaultdict(set)
    for r in runs:
        members[uf.find(r["out"])].add(r["out"])
        members[uf.find(r["in"])].add(r["in"])
        metrics[uf.find(r["out"])].add((r["wns"], r["luts"], r["pow"]))
    return uf, members, metrics


def state_source(state_id: str, roots: dict[str, Path]) -> tuple[str | None, dict | None]:
    """First root holding the state's checkpoint file, with its state.json."""

    for name, root in roots.items():
        f = root / "states" / state_id / "design.dcp"
        if f.is_file():
            try:
                meta = jread(root / "states" / state_id / "state.json")
            except Exception:
                meta = {}
            return str(f), meta
    return None, None
