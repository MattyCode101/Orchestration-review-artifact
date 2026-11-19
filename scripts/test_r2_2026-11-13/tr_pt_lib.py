"""Paper tables: shared readers and helpers. NEW file. Reads only released TEST data: unsealed_ab, unsealed_rest, unsealed_ledger and the files this task wrote
under data/paper_tables_2026-11-15/. No tool run, no model call. Result files are matched by episode id (the dot-in-filename fix, as tr_review2.py).

Gain definitions (all positive = better):  T0, T, W2R: WNS gain in ns;  A0, A, L2R: packed-LUT cut;  P0: dynamic-power cut in W (compared in whole mW).
A gain "beyond noise" exceeds the design's round-2 noise band of that metric (power: whole mW, the band itself is no change).
"""
from __future__ import annotations
import os

import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT = Path(os.environ.get("ORCH_PROJECT_ROOT") or Path(__file__).resolve().parents[2])
BASE = PROJECT / "data/test_r2_2026-11-13"
OUT = PROJECT / "data/paper_tables_2026-11-15"
AB_SRC = ("baseline", "llm1", "llm3", "llm6")
REST_SRC = ("baseline", "llm4", "llm5", "llm7")
DESIGNS = ("rosetta_digit-recognition", "corescore_500_mod", "vexriscv_re-place_v2", "fir_symmetric_systolic", "fir_systolic_transposed")
SHORT = {"rosetta_digit-recognition": "rosetta_digit", "corescore_500_mod": "corescore", "vexriscv_re-place_v2": "vexriscv_v2",
         "fir_symmetric_systolic": "fir_symmetric", "fir_systolic_transposed": "fir_transposed"}
GOAL = {"T0": ("wns_ns", "max"), "T": ("wns_ns", "max"), "W2R": ("wns_ns", "max"), "A0": ("packed_luts", "min"), "A": ("packed_luts", "min"),
        "L2R": ("packed_luts", "min"), "P0": ("dynamic_power_w", "min")}
AB_BRIEFS = ("T0", "A0", "T", "A")
POWER_BRIEFS = ("W2R", "L2R", "P0")
GEMINI = "gemini-3.1-flash-lite"


def jread(p: Path | str):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def mw(x: float) -> int:
    return round(float(x) * 1000.0)


def gain_of(brief: str, start: dict, final: dict) -> float:
    m, d = GOAL[brief]
    return (final[m] - start[m]) if d == "max" else (start[m] - final[m])


def beyond(brief: str, g: float, band: float) -> bool:
    """A gain beyond one noise band (power briefs compared in whole mW)."""
    return mw(g) > mw(band) if GOAL[brief][0] == "dynamic_power_w" else g > band + 1e-12


def tie_diff(brief: str, a: float, b: float, band: float) -> int:
    """+1 if a exceeds b by more than the band, -1 if b exceeds a by more than the band, else 0."""
    if GOAL[brief][0] == "dynamic_power_w":
        d, bd = mw(a) - mw(b), mw(band)
    else:
        d, bd = a - b, band + 1e-12
    return 1 if d > bd else (-1 if -d > bd else 0)


def head_class(ratio: float) -> str:
    return "none" if ratio < 1.0 else ("low" if ratio <= 3.0 else "high")


# ------------------------------------------------------------------------------------------------ episodes
def row_key(task: dict) -> dict:
    """Policy row of an episode: name, family, budget, seed/repeat index."""
    a = task["approach"]
    model = task.get("model")
    gateway = bool(model and model != GEMINI)
    budget = int(task["budget"]) if task.get("budget") else 6
    seed = None
    if a.startswith("random_"):
        seed, a = int(a.split("_")[1]), "random"
    name = {"v21": "v2.1"}.get(a, a)
    if a == "fixed":
        name = "fixed"
    return {"row": name, "gateway": gateway, "budget": budget, "seed": seed, "order": task.get("fixed_order"), "tag": task.get("tag")}


def load_episodes() -> list[dict]:
    """Every released TEST result (not the amd checks), with the fields the tables need and the dollar cost from the model-call lines."""
    eps = []
    for tree, srcs in (("unsealed_ab", AB_SRC), ("unsealed_rest", REST_SRC)):
        for src in srcs:
            root = BASE / tree / src
            usd = defaultdict(float)
            calls = defaultdict(int)
            lf = root / "llm_calls.jsonl"
            if lf.is_file():
                for line in lf.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        x = json.loads(line)
                        usd[x["episode_id"]] += float(x.get("usd") or 0.0)
                        calls[x["episode_id"]] += 1
            for p in sorted((root / "results").glob("tr_*.json")):
                r = jread(p)
                t = r.get("task") or {}
                eid = t.get("episode_id") or r.get("episode_id")
                if p.name != f"{eid}.json" or t.get("approach") == "amd" or r.get("status") != "done":
                    continue
                k = row_key(t)
                ls = r.get("llm_stats") or {}
                lp = r.get("loop") or {}
                eps.append({
                    "eid": eid, "tree": tree, "src": src, "design": r["design"], "brief": r["brief"], "approach": t["approach"], **k,
                    "start": r["start_metrics"], "final": r["final_metrics"], "noise": r["noise"], "final_state": r["final_state_id"],
                    "valid": bool(r.get("final_brief_valid")), "validation": r.get("validation_status"), "stop": r.get("stop_reason"),
                    "runs_used": r["runs_used"], "new_runs": r["new_real_runs"], "wall_s": r["charged_wall_s"], "cpu_s": r["charged_cpu_s"],
                    "tokens": (ls.get("tokens") or {}).get("total", 0) if ls else 0, "usd": usd.get(eid, 0.0), "model_calls": ls.get("model_calls", 0) if ls else 0,
                    "decisions": ls.get("decision_records", 0) if ls else 0, "repairs": ls.get("repairs", 0) if ls else 0, "fallbacks": ls.get("fallbacks", 0) if ls else 0,
                    "truncated": ls.get("truncated_decisions", 0) if ls else 0,
                    "calls": lp.get("calls"), "mandates": lp.get("mandates"), "kept": lp.get("placement_kept"), "empty_calls": lp.get("empty_calls"),
                    "nothing": lp.get("nothing_selected_calls"), "workers": lp.get("workers") or [], "respect": lp.get("respect_limits"),
                    "stop_rule_firings": lp.get("stop_rule_firings"),
                })
    for e in eps:
        e["gain"] = gain_of(e["brief"], e["start"], e["final"])
        e["band"] = float(e["noise"][GOAL[e["brief"]][0]])
    return eps


def policy_label(e: dict) -> str:
    """Display name of the row an episode belongs to (budget and model included)."""
    n = e["row"]
    if e["gateway"]:
        n += " (ChatGPT)"
    if e["budget"] != 6:
        n += f" b{e['budget']}"
    return n


# ------------------------------------------------------------------------------------------------ small stats
def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def sd(xs):
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) > 1 else None


def boot_ci(units: list[list[float]], stat=None, n: int = 10000, seed: int = 20261116, alpha: float = 0.05):
    """Two-level cluster bootstrap. `units` = one list of values per cell (the values are the repeats of that cell). Cells are resampled with replacement and,
    within a drawn cell, its repeats are resampled with replacement; the statistic is the mean of the cell means (or `stat(list of cell-value-lists)`).
    Returns (point, lo, hi, share of resamples above zero)."""
    rng = random.Random(seed)
    units = [u for u in units if u]
    if not units:
        return None, None, None, None
    f = stat or (lambda us: sum(sum(u) / len(u) for u in us) / len(us))
    point = f(units)
    draws = []
    for _ in range(n):
        us = []
        for _c in range(len(units)):
            u = units[rng.randrange(len(units))]
            us.append([u[rng.randrange(len(u))] for _ in range(len(u))])
        draws.append(f(us))
    draws.sort()
    lo, hi = draws[int(alpha / 2 * n)], draws[int((1 - alpha / 2) * n) - 1]
    return point, lo, hi, sum(1 for d in draws if d > 0) / n


def sign_wtl(diffs: list[int]) -> tuple[int, int, int]:
    return sum(1 for d in diffs if d > 0), sum(1 for d in diffs if d == 0), sum(1 for d in diffs if d < 0)


def fmt(x, nd: int = 1, sign: bool = False, none: str = "-") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return none
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
