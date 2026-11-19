#!/usr/bin/env python3
"""Independent check of the four final results tables (offline). It does NOT import final_data, final_tables, final_tex, names, figstyle or build_breakdown_inputs.

    python final/final_verify.py

For every displayed number of the six tables (the values csv of out/final/) it recomputes the number from the raw stored inputs with its own code and compares
  the value, the displayed text (own rounding and formatting), the bold flag and the italic flag.
Differences in method from the build: TEST gains are recomputed from the start and final metrics of every episode (episodes.json), not from the stored gain field;
fill-run gains are read from the fill EPISODES csv (the build reads the fill CELLS csv); headroom is recomputed from ref_known and the noise band.
It then adds three columns (check_value, check_shown, check_match) to each values csv, compares the TEST policy gains with the earlier figures
(Figures_16-11 values csvs, in out/), checks the previews (no colour, labels, caption length) and writes out/final/verify_report.txt and out/final/compare_earlier.csv.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / os.environ.get("FINAL_OUT", "out/final")
INP = ROOT / "inputs"
PB = INP / "per_benchmark"
FI = INP / "final"
MODEL = os.environ.get("FINAL_LLM_MODEL", "gemini")
REPORT: list[str] = []


def say(s: str) -> None:
    REPORT.append(s)
    print(s)


def rows_of(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def num(x) -> float | None:
    return None if x in ("", None, "nan") else float(x)


def rhu(x: float, nd: int) -> float:
    v = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return float(v)


def show(x: float, nd: int) -> str:
    v = rhu(x, nd)
    if v == 0:
        v = 0.0
    s = f"{abs(v):,.{nd}f}"
    return ("\u2212" + s) if v < 0 else s


GROUPS = ("TRAINING", "VALIDATION", "TEST")
DESIGNS = {"TRAINING": ["vexriscv_re-place", "logicnets_jscl", "rosetta_spam-filter", "rosetta_3d-rendering", "finn_radioml"],
           "VALIDATION": ["t2_cordic_s", "vtr_mcml_v2", "amd_mini-isp"],
           "TEST": ["corescore_500_mod", "fir_symmetric_systolic", "fir_systolic_transposed", "rosetta_digit-recognition", "vexriscv_re-place_v2"]}
SETTING = {"2worker": ("T0", "A0"), "3worker": ("W2R", "L2R", "P0")}
MAIN = ("T0", "A0")
WHAT = {"T0": "wns_ns", "W2R": "wns_ns", "A0": "packed_luts", "L2R": "packed_luts", "P0": "dynamic_power_w"}
SC = {"wns_ns": 1000.0, "packed_luts": 1.0, "dynamic_power_w": 1000.0}
ND = {"T0": 0, "W2R": 0, "A0": 0, "L2R": 0, "P0": 1}
TEST_LABEL = {"direct": "GA-R1", "orch": "V9", "rule": "v2", "fixed": "fixed", "bandit": "bandit", "grid": "grid"}
SHORT = {"corescore": "corescore_500_mod", "fir_symmetric": "fir_symmetric_systolic", "fir_transposed": "fir_systolic_transposed", "rosetta_digit": "rosetta_digit-recognition", "vexriscv_v2": "vexriscv_re-place_v2"}


def goal_gain(b: str, start: dict, final: dict) -> float:
    k = WHAT[b]
    if k == "wns_ns":
        return (final[k] - start[k]) * 1000.0
    if k == "packed_luts":
        return start[k] - final[k]
    return (start[k] - final[k]) * 1000.0


# ------------------------------------------------------------------------------------------------------------------------------ raw sources
EPI = json.load(open(PB / "episodes.json", encoding="utf-8"))
CELLS = json.load(open(PB / "cells.json", encoding="utf-8"))
GEN = json.load(open(PB / "gen_collect.json", encoding="utf-8"))
REFS = {(r["group"], r["design"], r["brief"]): r for r in rows_of(PB / "Gen_references_per_cell.csv")}
PTC = {(r["design"], r["brief"]): r for r in rows_of(INP / "Paper_Tables_Cells.csv")}
FILL_EPS = rows_of(FI / "Fill_Episodes.csv")
FILL_CELLS = rows_of(FI / "Fill_Cells.csv")
START_TEST = {SHORT[r["design"]]: r for r in rows_of(INP / "Eng_TEST_start_metrics.csv")}
COST = {1: {r["key"]: r for r in rows_of(INP / "Paper_Tables_Table1_cost_per_task.csv")}, 2: {r["key"]: r for r in rows_of(INP / "Paper_Tables_Table2_cost_per_task.csv")}}

TEST_EPS = defaultdict(list)
for e in EPI:
    if e["budget"] == 6 and not e["gateway"] and "gateway-" not in e["eid"]:       # the 6-run budget of the paper; budget-3 and budget-10 variants and the alternate-model (ChatGPT) episodes are other rows
        TEST_EPS[(e["row"], e["design"], e["brief"])].append(e)
GEN_BY = defaultdict(list)
for r in GEN["records"]:
    GEN_BY[(r["design"], r["brief"])].append(r)


def ref_row(g, d, b):
    return REFS[(g, d, b)]


def headroom(g, d, b) -> bool:
    r = ref_row(g, d, b)
    ref, band = float(r["ref_known"]), float(r["band"])
    if WHAT[b] == "dynamic_power_w":
        a, c = round(ref * 1000.0), round(band * 1000.0)
        ok = (a / c >= 1.0) if c > 0 else a > 0
    else:
        ok = (ref / band >= 1.0) if band > 0 else ref > 0
    flag = r["head_known"] if g == "TEST" else r["head"]
    if str(flag) != str(ok):
        raise SystemExit(f"headroom flag disagrees: {g} {d} {b}: csv {flag}, recomputed {ok}")
    return ok


def start_of(g, d) -> dict:
    if g == "TEST":
        r = START_TEST[d]
        return {"wns": float(r["wns_ps"]), "luts": float(r["packed_luts"]), "mw": float(r["dynamic_power_mw"])}
    s = GEN["starts"][d]
    return {"wns": round(s["wns_ns"] * 1000.0), "luts": float(s["packed_luts"]), "mw": float(round(s["dynamic_power_w"] * 1000.0))}


def reps_of(g, d, b, col):
    """Independent per-repeat gains (display units) of a policy cell, or None when no result exists."""
    if g == "TEST":
        es = TEST_EPS.get((TEST_LABEL[col], d, b))
        if not es:
            return None
        es = sorted(es, key=lambda e: ((e["seed"] if e.get("seed") is not None else -1), str(e["tag"])))
        return [goal_gain(b, e["start"], e["final"]) for e in es]
    kind_main = b in MAIN
    sc = SC[WHAT[b]]
    if col in ("rule", "fixed"):
        name = "v2" if col == "rule" else "fixed"
        rs = [r for r in GEN_BY[(d, b)] if r["row"] == name and r["src"] == ("d6v2" if kind_main else "power_review")]
        return [r["gain"] * sc for r in sorted(rs, key=lambda r: str(r["eid"]))] or None
    if col == "orch":
        if kind_main:
            rs = sorted([r for r in GEN_BY[(d, b)] if r["row"] == "V9" and r["src"] == "v22t3"], key=lambda r: (r["rep"], str(r["eid"])))
            return [r["gain"] * sc for r in rs] or None
        r1 = [r["gain"] * sc for r in GEN_BY[(d, b)] if r["row"] == "V9" and r["src"] == "post_freeze"]
        f = sorted([x for x in FILL_EPS if x["data_scope"] == g and x["design"] == d and x["brief"] == b and x["row"] == "V9, Gemini"], key=lambda x: x["repeat"])
        return (r1 + [float(x["goal_gain"]) for x in f]) if len(r1) == 1 and len(f) == 2 else None
    if col == "direct":
        f = sorted([x for x in FILL_EPS if x["data_scope"] == g and x["design"] == d and x["brief"] == b and x["row"] == "GA-R1, Gemini"], key=lambda x: x["repeat"])
        return [float(x["goal_gain"]) for x in f] or None
    return None


def exh_of(g, d, b):
    """(display value, marker) of the Exhaustive Search cell, straight from the raw files."""
    sc = SC[WHAT[b]]
    r = ref_row(g, d, b)
    if g == "TEST":
        return float(PTC[(d, b)]["reference_gain"]) * sc, ""
    if g == "VALIDATION":
        return float(r["val_exhaustive_2step"]) * sc, "2"
    if b in MAIN:
        return float(r["stored_best_known_amd_pass_checked"]) * sc, ""
    f = [x for x in FILL_CELLS if x["data_scope"] == g and x["design"] == d and x["brief"] == b and str(x["row"]).startswith("exhaustive 2-step")]
    if f:
        return float(f[0]["goal_gain"]), ""
    raw = float(r["train_ceiling_amd_pass_checked"])
    return (raw if b == "W2R" else raw * 1000.0 if b == "P0" else raw), "\u2020"


def mean(xs):
    return sum(xs) / len(xs)


def sdv(xs):
    return statistics.stdev(xs) if len(xs) > 1 else None


def pooled(setting, b, col):
    """Summary cell: (mean over headroom designs, sd over repeats of the pooled value, n repeats) or None."""
    heads = [d for d in DESIGNS["TEST"] if headroom("TEST", d, b)]
    if col == "exh":
        return mean([exh_of("TEST", d, b)[0] for d in heads]), None, 1
    cs = [reps_of("TEST", d, b, col) for d in heads]
    if any(c is None for c in cs):
        return None
    n = len(cs[0])
    if any(len(c) != n for c in cs):
        raise SystemExit(f"unequal repeats {col} {b}")
    per = [mean([c[i] for c in cs]) for i in range(n)]
    return mean([mean(c) for c in cs]), (sdv(per) if n > 1 else None), n


# ------------------------------------------------------------------------------------------------------------------------------ checks per table
def verify_table(path: Path) -> tuple[list[dict], list[str], int]:
    stem = path.stem.replace("values_", "")
    rows = rows_of(path)
    bad, n = [], 0
    summ = stem.startswith("tab_summary")
    setting = "2worker" if "2worker" in stem else "3worker"
    briefs = SETTING[setting]
    out = []
    # independent expected values, keyed like the csv
    exp: dict[tuple, tuple] = {}     # (row_key, col_key, brief, group) -> (value, sd, n, marker, text_main, nd)
    if summ:
        for b in briefs:
            for col in ("exh", "direct", "orch", "rule", "fixed", "bandit", "grid"):
                p = pooled(setting, b, col)
                if p is None and setting == "3worker" and col == "bandit" and b == "W2R":
                    from pathlib import Path as _P
                    import json as _json
                    _f = _P(__file__).resolve().parent.parent / "inputs" / "bandit_w2r_cells.json"
                    if _f.exists():
                        _g = [float(c["gain_ps"]) for c in _json.loads(_f.read_text(encoding="utf-8"))["cells"] if c["head"]]
                        _v = sum(_g) / len(_g)
                        exp[(f"gain_{b}", col, b, "TEST")] = (_v, None, 1, "d", show(_v, ND[b]), ND[b])
                        continue
                exp[(f"gain_{b}", col, b, "TEST")] = None if p is None else (p[0], p[1], p[2], "", show(p[0], ND[b]), ND[b])
        ck = 1 if setting == "2worker" else 2
        keyrow = {"exh": "ref3" if setting == "2worker" else "ref2p", "direct": "GA-R1", "orch": "V9", "rule": "v2", "fixed": "fixed", "bandit": "bandit", "grid": "grid"}
        for col, k in keyrow.items():
            r = COST[ck][k]
            jh = num(r["job_hours_per_task"])
            exp[("jobh", col, "", "TEST")] = None if jh is None else (jh, None, int(r["tasks"]), "", show(jh, 2), 2)
            tk = num(r["llm_tokens_per_task"])
            exp[("tokens", col, "", "TEST")] = (tk, None, int(r["tasks"]), "", show(tk, 0), 0) if col in ("direct", "orch") else "dash"
    else:
        for g in GROUPS:
            for d in DESIGNS[g]:
                st = start_of(g, d)
                for k in ("wns", "luts", "mw"):
                    exp[(d, f"start_{k}", "", g)] = (st[k], None, 1, "", show(st[k], 0), 0)
                for b in briefs:
                    for col in ("exh", "direct", "orch", "rule", "fixed"):
                        if col == "exh":
                            v, mk = exh_of(g, d, b)
                            exp[(d, col, b, g)] = (v, None, 1, mk, show(v, ND[b]), ND[b])
                        else:
                            rp = reps_of(g, d, b, col)
                            exp[(d, col, b, g)] = None if rp is None else (mean(rp), sdv(rp), len(rp), "", show(mean(rp), ND[b]), ND[b])
        for k in ("wns", "luts", "mw"):
            exp[("total", f"start_{k}", "", "ALL")] = (sum(start_of(g, d)[k] for g in GROUPS for d in DESIGNS[g]), None, 13, "", None, 0)
        for b in briefs:
            for col in ("exh", "direct", "orch", "rule", "fixed"):
                if col == "exh":
                    vals = [exh_of(g, d, b)[0] for g in GROUPS for d in DESIGNS[g]]
                    exp[("total", col, b, "ALL")] = (sum(vals), None, 1, "\u2020" if setting == "3worker" and any(exh_of(g, d, b)[1] == "\u2020" for g in GROUPS for d in DESIGNS[g]) else "", None, ND[b])
                    continue
                rps = [reps_of(g, d, b, col) for g in GROUPS for d in DESIGNS[g]]
                have = [r for r in rps if r is not None]
                if not have:
                    exp[("total", col, b, "ALL")] = None
                    continue
                ns = {len(r) for r in have}
                sd_ = None
                if len(have) == 13 and len(ns) == 1 and next(iter(ns)) > 1:
                    nn = next(iter(ns))
                    sd_ = sdv([sum(r[i] for r in have) for i in range(nn)])
                exp[("total", col, b, "ALL")] = (sum(mean(r) for r in have), sd_, len(have), "", None, ND[b])
    # compare
    for r in rows:
        key = (r["row_key"], r["col_key"], r["brief"], r["group"])
        e = exp.get(key, "missing")
        chk_val, chk_shown, match = "", "", 1
        if e == "missing":
            bad.append(f"{stem}: no independent value for {key}")
            match = 0
        elif e is None:
            chk_shown = "not run" if r["state"] == "not_run" else "n/a"
            if r["state"] not in ("not_run", "na"):
                match = 0
                bad.append(f"{stem}: {key} shown as a value but the raw sources hold no result")
        elif e == "dash":
            chk_shown = "\u2014"
            match = int(r["state"] == "dash")
        else:
            val, sd_, nrep, mk, txt, nd = e
            if not summ and not (mk == "†" or (mk == "2" and setting == "2worker")):
                mk = ""                      # markers shown: the dagger (pending exhaustive) and, on two workers only, the 2 of the 2-step VALIDATION search
            chk_val = repr(float(val))
            ndp = nd
            main_txt = show(val, ndp)
            if r["line"] == "main":
                mk_shown = mk if mk else ("a" if (summ and r["marker"] == "a") else "")
                chk_shown = main_txt + mk_shown
                # Eng_TEST stores the pooled means to 3 or 4 decimals, so summary gains agree to that precision; the displayed text must agree exactly
                tol = 6e-4 if (summ and r["row_key"].startswith("gain_")) else 1e-6 * max(1.0, abs(val)) + 1e-9
                same = abs(float(r["value"]) - val) <= tol and r["shown"].replace("\u2212", "-") == chk_shown.replace("\u2212", "-")
                if r["marker"] != mk and not (summ and r["marker"] in ("a",)):
                    same = False
                if r["marker"] == "2" and mk != "2":
                    same = False
                match = int(same)
                if not same:
                    bad.append(f"{stem}: {key} csv {r['shown']} ({r['value']}) vs independent {chk_shown} ({val})")
            else:
                if sd_ is None:
                    chk_shown = ""
                    k_txt = re.fullmatch(r"(\d+) of 13", r["shown"])
                    match = int(bool(k_txt))
                    if not match:
                        bad.append(f"{stem}: {key} second line {r['shown']} but no sd expected")
                else:
                    chk_val = repr(float(sd_))
                    chk_shown = "\u00b1" + show(abs(sd_), ndp)
                    match = int(r["shown"] == chk_shown)
                    if not match:
                        bad.append(f"{stem}: {key} second line {r['shown']} vs independent {chk_shown}")
        n += 1
        out.append(dict(r, check_value=chk_val, check_shown=chk_shown, check_match=match))
    # bold and italic recomputed from the independent values
    flags = {}
    for r in rows:
        if r["line"] == "main":
            flags[(r["row_key"], r["col_key"], r["brief"], r["group"])] = r
    def policy_shown(key):
        e = exp.get(key)
        return None if (e is None or e == "dash" or e == "missing") else rhu(e[0], e[5])
    nb = 0
    if summ:
        for b in briefs:
            sv = {c: policy_shown((f"gain_{b}", c, b, "TEST")) for c in ("direct", "orch", "rule", "fixed", "bandit", "grid")}
            sv = {c: v for c, v in sv.items() if v is not None}
            mx = max(sv.values())
            want = {c for c, v in sv.items() if mx > 0 and v == mx}
            for c in ("exh", "direct", "orch", "rule", "fixed", "bandit", "grid"):
                f = flags[(f"gain_{b}", c, b, "TEST")]
                nb += 1
                if (int(f["bold"]) == 1) != (c in want):
                    bad.append(f"{stem}: bold flag {b} {c}: csv {f['bold']} vs independent {c in want}")
        for m in ("jobh", "tokens"):
            for c in ("exh", "direct", "orch", "rule", "fixed", "bandit", "grid"):
                if int(flags[(m, c, "", "TEST")]["bold"]) != 0:
                    bad.append(f"{stem}: cost cell bold {m} {c}")
    else:
        for g in GROUPS:
            for d in DESIGNS[g]:
                for b in briefs:
                    sv = {c: policy_shown((d, c, b, g)) for c in ("direct", "orch", "rule", "fixed")}
                    sv = {c: v for c, v in sv.items() if v is not None}
                    mx = max(sv.values()) if sv else None
                    want = {c for c, v in sv.items() if mx is not None and mx > 0 and v == mx}
                    head = headroom(g, d, b)
                    for c in ("exh", "direct", "orch", "rule", "fixed"):
                        f = flags.get((d, c, b, g))
                        if f is None:
                            continue          # design not shown in the representative version
                        nb += 1
                        if (int(f["bold"]) == 1) != (c in want):
                            bad.append(f"{stem}: bold flag {g} {d} {b} {c}: csv {f['bold']} vs independent {c in want}")
                        if (int(f["italic"]) == 1) != (not head):
                            bad.append(f"{stem}: italic flag {g} {d} {b} {c}: csv {f['italic']} vs headroom {head}")
        for b in briefs:
            tot = {c: exp[("total", c, b, "ALL")] for c in ("direct", "orch", "rule", "fixed")}
            comparable = all(t is not None and t[2] == 13 for t in tot.values())
            want = set()
            if comparable:
                sv = {c: rhu(t[0], t[5]) for c, t in tot.items()}
                mx = max(sv.values())
                want = {c for c, v in sv.items() if mx > 0 and v == mx}
            for c in ("exh", "direct", "orch", "rule", "fixed"):
                f = flags[("total", c, b, "ALL")]
                nb += 1
                if (int(f["bold"]) == 1) != (c in want):
                    bad.append(f"{stem}: bold flag total {b} {c}: csv {f['bold']} vs independent {c in want}")
        # the shown totals and the totals of the independent sums
    return out, bad, n + nb


def write_with_checks(path: Path, rows: list[dict]) -> None:
    fields = [f for f in rows[0].keys()]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------------------------------------------------------------------ check values of the prompt
PROMPT_CHECKS = [
    ("tab_summary_2worker_TEST", "gain_T0", {"exh": 400, "direct": 141, "orch": 166, "rule": 167, "fixed": 178, "bandit": 66, "grid": 144}, 0),
    ("tab_summary_2worker_TEST", "gain_A0", {"exh": 419, "direct": 277, "orch": 352, "rule": 334, "fixed": 333, "bandit": 343, "grid": 120}, 0),
    ("tab_summary_2worker_TEST", "jobh", {"exh": 147.16, "direct": 1.14, "orch": 1.16, "rule": 1.11, "fixed": 1.29, "bandit": 1.19, "grid": 1.06}, 2),
    ("tab_summary_2worker_TEST", "tokens", {"direct": 17083, "orch": 10941}, 0),
    ("tab_summary_3worker_TEST", "gain_W2R", {"exh": 279, "direct": 128, "orch": 166, "rule": 167, "fixed": 135, "grid": 144}, 0),
    ("tab_summary_3worker_TEST", "gain_L2R", {"exh": 505, "direct": 365, "orch": 232, "rule": 228, "fixed": 261, "bandit": 0, "grid": 141}, 0),
    ("tab_summary_3worker_TEST", "gain_P0", {"exh": 44.7, "direct": 43.2, "orch": 41.3, "rule": 41.3, "fixed": 43.0, "bandit": 0.0, "grid": 41.3}, 1),
    ("tab_summary_3worker_TEST", "jobh", {"direct": 1.07, "orch": 1.03, "rule": 0.98, "fixed": 1.12, "bandit": 0.89, "grid": 1.00}, 2),
    ("tab_summary_3worker_TEST", "tokens", {"direct": 19717, "orch": 8992}, 0),
]


def check_prompt_values(all_rows: dict[str, list[dict]]) -> int:
    nbad = 0
    n = 0
    for stem, rk, d, nd in PROMPT_CHECKS:
        for col, ev in d.items():
            m = [r for r in all_rows[stem] if r["row_key"] == rk and r["col_key"] == col and r["line"] == "main"]
            n += 1
            if len(m) != 1 or m[0]["check_value"] == "" or rhu(float(m[0]["check_value"]), nd) != rhu(ev, nd):
                nbad += 1
                say(f"PROMPT CHECK VALUE DIFFERS: {stem} {rk} {col}: expected {ev}, independent {m[0]['check_value'] if m else None}")
    wb = [("corescore_500_mod", "A0", {"exh": 742, "direct": 429, "orch": 543, "rule": 544, "fixed": 543}, "tab_breakdown_2worker_all"),
          ("fir_systolic_transposed", "T0", {"exh": 167, "direct": 103, "orch": 103, "rule": 103, "fixed": 100}, "tab_breakdown_2worker_all"),
          ("vtr_mcml_v2", "A0", {"exh": 1595, "orch": 1121, "rule": 1086, "fixed": 742}, "tab_breakdown_2worker_all"),
          ("finn_radioml", "A0", {"exh": 7393, "orch": 1597, "rule": 7095, "fixed": 7047, "direct": 4812}, "tab_breakdown_2worker_all"),
          ("corescore_500_mod", "L2R", {"exh": 741, "direct": 428, "orch": 561, "rule": 544, "fixed": 16}, "tab_breakdown_3worker_all")]
    for d, b, vals, stem in wb:
        for col, ev in vals.items():
            m = [r for r in all_rows[stem] if r["row_key"] == d and r["col_key"] == col and r["brief"] == b and r["line"] == "main"]
            n += 1
            if len(m) != 1 or m[0]["check_value"] == "" or rhu(float(m[0]["check_value"]), 0) != rhu(ev, 0):
                nbad += 1
                say(f"PROMPT CHECK VALUE DIFFERS: {stem} {d} {b} {col}: expected {ev}, independent {m[0]['check_value'] if m else None}")
    for stem in ("tab_breakdown_2worker_all", "tab_breakdown_3worker_all"):
        for k, ev in (("start_wns", -1204), ("start_luts", 105957), ("start_mw", 13364)):
            m = [r for r in all_rows[stem] if r["row_key"] == "corescore_500_mod" and r["col_key"] == k]
            if m:
                n += 1
                if rhu(float(m[0]["check_value"]), 0) != ev:
                    nbad += 1
                    say(f"PROMPT CHECK VALUE DIFFERS: {stem} corescore start {k}: expected {ev}, independent {m[0]['check_value']}")
    say(f"prompt check values re-evaluated from the independent values: {n} compared, {nbad} different")
    return nbad


# ------------------------------------------------------------------------------------------------------------------------------ comparison with the earlier figures
def compare_earlier(all_rows: dict[str, list[dict]]) -> None:
    old_bd = rows_of(ROOT / "out" / "values_per_benchmark_breakdown.csv")
    oldc = {}
    for r in old_bd:
        if r["row_type"] == "design" and r["kind"] in ("value", "no_headroom") and r["value"] != "" and r["block"] in ("T0", "A0", "W2R", "L2R", "P0") and r["column_key"] in ("gemini", "direct", "rule", "fixed"):
            oldc[(r["group"], r["design"], r["block"], {"gemini": "orch"}.get(r["column_key"], r["column_key"]))] = (float(r["value"]), r["n_repeats"], r["note"])
    cause = {"TEST": "same definition"}
    diffs, n_same, n_cmp = [], 0, 0
    new = {}
    for stem in ("tab_breakdown_2worker_all", "tab_breakdown_3worker_all"):
        for r in all_rows[stem]:
            if r["line"] == "main" and r["group"] != "ALL" and r["col_key"] in ("direct", "orch", "rule", "fixed") and r["value"] != "":
                new[(r["group"], r["row_key"], r["brief"], r["col_key"])] = (float(r["value"]), r["n_repeats"])
    with open(OUT / "compare_earlier.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "design", "brief", "policy", "earlier_value", "earlier_repeats", "new_value", "new_repeats", "difference", "status", "cause"])
        for k in sorted(set(new) | set(oldc)):
            o, nw = oldc.get(k), new.get(k)
            g, d, b, p = k
            if o is None and nw is None:
                continue
            if o is None:
                w.writerow([g, d, b, p, "", "", f"{nw[0]:.6g}", nw[1], "", "new cell", "earlier figure: not run; filled by the fill run" if p == "direct" else "no earlier value"])
                continue
            if nw is None:
                w.writerow([g, d, b, p, f"{o[0]:.6g}", o[1], "", "", "", "no new value", ""])
                continue
            n_cmp += 1
            dif = nw[0] - o[0]
            if abs(dif) <= 1e-6 * max(1.0, abs(o[0])) and str(o[1]) == str(nw[1]):
                n_same += 1
                continue
            if abs(dif) <= 1e-6 * max(1.0, abs(o[0])):
                why = f"value unchanged, repeats {o[1]} -> {nw[1]}"
            elif g == "TEST":
                why = "DIFFERENT ON TEST (definition unchanged): to be explained"
            elif p == "direct" and b in MAIN:
                why = "pilot (1 repeat, temperature 0.0, Saturday driver) replaced by 3 fresh fill repeats"
            elif p == "orch" and b not in MAIN:
                why = "repeats 2 and 3 added by the fill run (earlier: repeat 1 only)"
            elif p == "direct":
                why = "fill run (earlier: not run)"
            else:
                why = "DIFFERENT (definition unchanged): to be explained"
            diffs.append((k, why))
            w.writerow([g, d, b, p, f"{o[0]:.6g}", o[1], f"{nw[0]:.6g}", nw[1], f"{dif:.6g}", "changed", why])
    bad_test = [d for d in diffs if "to be explained" in d[1]]
    say(f"comparison with the earlier per-benchmark figure (policy columns, all designs): {n_cmp} cells present in both, {n_same} identical in value and repeats, {len(diffs)} different (see compare_earlier.csv); unexplained differences: {len(bad_test)}")
    for k, why in diffs[:0]:
        pass
    # TEST summary gains against the earlier summary figures
    nsame = ndiff = 0
    for setting in ("2worker", "3worker"):
        old = rows_of(ROOT / "out" / f"values_summary_{setting}_TEST.csv")
        oldv = {}
        for r in old:
            if r["col"] in ("g_t", "g_a", "g_p") and r["line"] == "main" and r["state"] == "ok" and r["value"] != "":
                oldv[(r["row_key"], r["col"])] = float(r["value"])
        mp = {"gain_T0": "g_t", "gain_W2R": "g_t", "gain_A0": "g_a", "gain_L2R": "g_a", "gain_P0": "g_p"}
        rk = {"exh": "ref3" if setting == "2worker" else "ref2p", "direct": "GA-R1", "orch": "V9", "rule": "v2", "fixed": "fixed", "bandit": "bandit", "grid": "grid"}
        for r in all_rows[f"tab_summary_{setting}_TEST"]:
            if r["line"] == "main" and r["row_key"] in mp and r["value"] != "":
                o = oldv.get((rk[r["col_key"]], mp[r["row_key"]]))
                if o is None:
                    continue
                if abs(o - float(r["value"])) <= 1e-3 * max(1.0, abs(o)):
                    nsame += 1
                else:
                    ndiff += 1
                    say(f"  TEST summary gain differs from the earlier figure: {setting} {r['row_key']} {r['col_key']}: earlier {o} vs now {r['value']}")
    say(f"TEST summary gains (definition unchanged) against the earlier summary figures: {nsame} identical, {ndiff} different")


# ------------------------------------------------------------------------------------------------------------------------------ previews
def check_previews(stems: list[str]) -> int:
    from PIL import Image
    nbad = 0
    for s in stems:
        im = Image.open(OUT / f"{s}.png").convert("RGB")
        import numpy as np
        a = np.asarray(im, dtype=int)
        coloured = int(((a.max(axis=2) - a.min(axis=2)) > 0).sum())
        cap = re.search(r"\\caption\{(.*?)\}\n", (OUT / f"{s}.tex").read_text(encoding="ascii"))
        words = len(cap.group(1).split()) if cap else 999
        txt = subprocess.run(["pdftotext", "-raw", str(OUT / f"{s}.pdf"), "-"], capture_output=True, text=True).stdout
        labels = [g for g in GROUPS if g in txt]
        need = GROUPS if "breakdown" in s else ("TEST",)
        ok = coloured == 0 and words <= 40 and all(g in labels for g in need) and "\\textcolor" not in (OUT / f"{s}.tex").read_text(encoding="ascii") and "colortbl" not in (OUT / f"{s}.tex").read_text(encoding="ascii")
        nbad += 0 if ok else 1
        say(f"preview {s}: coloured pixels {coloured}; caption {words} words; group labels in the pdf text {labels}; size {im.size[0]/300:.2f} x {im.size[1]/300:.2f} in -> {'ok' if ok else 'PROBLEM'}")
    return nbad


def repeat_counts() -> None:
    """Distinct repeat counts per group, setting and policy column, from the raw sources (the notes of the tables state them)."""
    for setting, bs in SETTING.items():
        for g in GROUPS:
            parts = []
            for col in ("direct", "orch", "rule", "fixed"):
                ns = sorted({len(reps_of(g, d, b, col)) for d in DESIGNS[g] for b in bs if reps_of(g, d, b, col) is not None})
                miss = sum(1 for d in DESIGNS[g] for b in bs if reps_of(g, d, b, col) is None)
                parts.append(f"{col} {'/'.join(map(str, ns)) or 'none'} rep" + (f" ({miss} cells without data)" if miss else ""))
            say(f"repeats, {setting} {g}: " + "; ".join(parts) + "; exhaustive 1")


def coverage(all_rows) -> None:
    for s, rs in all_rows.items():
        gaps = sorted({(r["row_key"], r["col_key"], r["brief"], r["state"]) for r in rs if r["line"] == "main" and r["state"] not in ("ok",) })
        pend = sorted({(r["row_key"], r["brief"]) for r in rs if r["line"] == "main" and r["marker"] == "†"})
        say(f"coverage {s}: cells without a value: {[(g[0], g[1], g[2], g[3]) for g in gaps if g[3] != 'dash']}; dagger (exhaustive pending) cells: {len(pend)}")


def main() -> int:
    stems = ["tab_summary_2worker_TEST", "tab_summary_3worker_TEST", "tab_breakdown_2worker_all", "tab_breakdown_2worker_representative", "tab_breakdown_3worker_all", "tab_breakdown_3worker_representative"]
    all_rows: dict[str, list[dict]] = {}
    tot_bad, tot_n = [], 0
    for s in stems:
        p = OUT / f"values_{s}.csv"
        out, bad, n = verify_table(p)
        all_rows[s] = out
        write_with_checks(p, out)
        tot_bad += bad
        tot_n += n
        say(f"{s}: {len([r for r in out if r['line'] == 'main'])} displayed cells ({n} checks incl. second lines and flags), disagreements with the independent recomputation: {len(bad)}")
    for b in tot_bad[:40]:
        say("  - " + b)
    repeat_counts()
    coverage(all_rows)
    nb = check_prompt_values(all_rows)
    if (ROOT / "out" / "values_per_benchmark_breakdown.csv").exists():      # comparison with an earlier figure set, not part of this release
        compare_earlier(all_rows)
    npv = check_previews(stems)
    n_eps = sum(1 for k, v in TEST_EPS.items() for _ in v)
    say(f"independent method: TEST gains recomputed from the start and final metrics of {n_eps} stored episodes (not the stored gain field); fill gains from the fill episodes csv; headroom from ref_known and the noise band")
    say(f"OVERALL: disagreements {len(tot_bad)}, prompt check values different {nb}, preview problems {npv}")
    (OUT / "verify_report.txt").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    return 0 if (not tot_bad and nb == 0 and npv == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
