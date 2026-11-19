"""Policy table (Wed 2026-11-11): shared read-only helpers.

Scope: the round-1 test campaign (data/campaign_test_final) and its frozen analysis frame, briefs T0, A0, T and A
only. Power briefs (P0) on test designs and the test power baseline (data/sweeps/friday_test_power_*) are sealed:
this module never opens a P0 result file, a P0 episode journal or a power root. Nothing is written outside
data/policy_table_2026-11-11. Journals are read as plain files (the Journal class takes a lock file and may create
directories, so it is not used on old roots).
"""
from __future__ import annotations
import os

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT = Path(os.environ.get("ORCH_PROJECT_ROOT") or Path(__file__).resolve().parents[2])
for _p in (PROJECT, PROJECT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

WORK = PROJECT / "data/policy_table_2026-11-11"
TEST = PROJECT / "data/campaign_test_final"
ENGINE = TEST / "engine"
FOUR = ("T0", "A0", "T", "A")
MAIN = ("T0", "A0")
OBJECTIVE = {"T0": ("wns_ns", "max"), "A0": ("packed_luts", "min"), "T": ("wns_ns", "max"), "A": ("packed_luts", "min")}
LIMIT_METRIC = {"T0": "packed_luts", "A0": "wns_ns", "T": "packed_luts", "A": "wns_ns"}
DESIGNS = (
    "rosetta_digit-recognition", "corescore_500_mod", "vexriscv_re-place_v2", "fir_symmetric_systolic",
    "fir_systolic_transposed",
)
SHORT = {"rosetta_digit-recognition": "rosetta_digit", "corescore_500_mod": "corescore", "vexriscv_re-place_v2": "vexriscv_v2",
         "fir_symmetric_systolic": "fir_symmetric", "fir_systolic_transposed": "fir_transposed"}
# approach order for the tables: (key, label)
ROWS = (
    ("fixed_BA", "Best fixed order (area, then timing)"),
    ("v2", "Goal-repair, rule-based v2 (g05)"),
    ("v1", "Goal-repair v1 (untuned)"),
    ("single", "Single specialist"),
    ("F1-R1", "LLM orchestrator (round 1, F1)"),
    ("F1-R0", "LLM orchestrator, earlier wording (F1-R0)"),
    ("E0-R1", "LLM orchestrator, exact numbers (E0-R1)"),
    ("GA-R1", "LLM controlling recipes directly (general agent)"),
    ("fixed_AB", "Fixed order (timing, then area)"),
    ("random", "Random (3 seeds)"),
)


def jload(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


# ------------------------------------------------------------------------------ campaign results (no P0)
def void_runs(work: Path = TEST) -> set[int]:
    runs = work / "runs"
    return {int(p.parent.name) for p in runs.glob("*/VOID.json")} if runs.is_dir() else set()


def load_results(work: Path = TEST, briefs: tuple[str, ...] = FOUR) -> list[dict[str, Any]]:
    """Result rows of the finished campaign for the four A/B briefs. P0 files are skipped by name and never opened."""

    voids = void_runs(work)
    rows = []
    for p in sorted((work / "results").glob("*.json")):
        if "_P0_" in p.name:
            continue
        r = jload(p)
        if r["brief"] not in briefs or r.get("run_no") in voids:
            continue
        rows.append(r)
    return rows


def main_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Main run of each LLM cell (tag m1); repeats (m2) are kept out of every mean, as in the frozen analysis."""

    return [r for r in rows if r["approach"] not in ("F1-R1", "GA-R1") or r.get("tag") in ("m1", "r0")]


def bucket(rows: list[dict[str, Any]]) -> dict[str, dict[tuple[str, str], list[dict[str, Any]]]]:
    out: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[r["approach"]][(r["design"], r["brief"])].append(r)
    for key in {k for i in range(3) for k in out.get(f"random_{i}", {})}:
        out["random"][key] = [r for i in range(3) for r in out.get(f"random_{i}", {}).get(key, [])]
    return out


# ------------------------------------------------------------------------------ frozen frame (non-power rows only)
def load_frame() -> dict[tuple[str, str], dict[str, Any]]:
    """(design, brief) -> start, best_known, best_gain, band, no_room for the four A/B briefs, from the frozen
    frame as stored in data/test_final_analysis/detail/detail.json. Power rows are dropped on load, unprinted."""

    d = jload(PROJECT / "data/test_final_analysis/detail/detail.json")
    return {(r["design"], r["brief"]): r for r in d["frame"] if r["brief"] in FOUR}


# ------------------------------------------------------------------------------ journals and run records
def read_journal(root: Path, episode_id: str) -> list[dict[str, Any]]:
    path = root / "episodes" / episode_id / "journal.jsonl"
    events = []
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        events.append(json.loads(line)["event"])
    return events


_RUNINFO: dict[tuple[str, str], dict[str, Any]] = {}


def runinfo(root: Path, run_id: str) -> dict[str, Any]:
    key = (str(root), run_id)
    if key not in _RUNINFO:
        _RUNINFO[key] = jload(root / "runs" / run_id / "runinfo.json")
    return _RUNINFO[key]


def start_metrics_of(root: Path, episode_id: str) -> dict[str, float]:
    ev = read_journal(root, episode_id)[0]
    assert ev["type"] == "start"
    return {k: v["value"] for k, v in ev["start_evidence"]["metrics"].items()}


def episode_steps(root: Path, episode_id: str) -> list[dict[str, Any]]:
    """One record per recipe run of the episode, in order: run id, recipe, charged times, mandate worker and rank."""

    events = read_journal(root, episode_id)
    mand = {e["index"]: e for e in events if e["type"] == "mandate"}
    out: list[dict[str, Any]] = []
    for sel in (e for e in events if e["type"] == "selection"):
        m = mand[sel["index"]]
        sp = (m.get("worker_plan") or {}).get("score_plan") or {}
        ranked = [r["recipe_id"] for r in sp.get("ranked", [])]
        planned = sp.get("selected_ids") or []
        for k, rid in enumerate(sel["attempt_run_ids"], 1):
            info = runinfo(root, rid)
            recipe = info["recipe_id"]
            out.append({
                "run_id": rid, "recipe": recipe, "worker": m["mandate"]["worker_id"], "mandate": sel["index"],
                "attempt": k, "rank": (ranked.index(recipe) + 1) if recipe in ranked else None,
                "planned_pos": (planned.index(recipe) + 1) if recipe in planned else None,
                "preserve_placement": m["mandate"]["settings"].get("preserve_placement"),
                "wall_s": float(info.get("wall_s") or 0.0),
                "cpu_s": float(info.get("cpu_user_s") or 0.0) + float(info.get("cpu_sys_s") or 0.0),
                "cache_hit": bool(info.get("cache_hit")), "status": info.get("status"),
            })
    return out


def llm_usage(work: Path = TEST) -> dict[str, dict[str, float]]:
    """Per episode: tokens (all logged calls), US$ as logged, calls, calls served without a charge, charged tokens."""

    out: dict[str, dict[str, float]] = defaultdict(lambda: {"tokens": 0, "usd": 0.0, "calls": 0, "uncharged": 0,
                                                              "tokens_charged": 0, "usd_charged": 0.0})
    for line in (work / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        e = c["episode_id"]
        if "_P0_" in e:
            continue
        u = out[e]
        u["tokens"] += c["total_tokens"]
        u["usd"] += c["usd"]
        u["calls"] += 1
        if c["charged"]:
            u["tokens_charged"] += c["total_tokens"]
            u["usd_charged"] += c["usd"]
        else:
            u["uncharged"] += 1
    return dict(out)


def share(start: float, best: float | None, value: float, direction: str) -> float | None:
    from orch.analysis.report import share_of_best

    return share_of_best(start, best, value, direction)


def gain(brief: str, start: float, value: float) -> float:
    _, direction = OBJECTIVE[brief]
    return (value - start) if direction == "max" else (start - value)
