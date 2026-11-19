#!/usr/bin/env python3
"""Per-benchmark breakdown figure, step 1: assemble every number the figure shows. Offline and read-only: no tool run, no model call.

Reads only the files in inputs/per_benchmark/ (copies of the stored results and CSVs, hashes in MANIFEST.sha256) and writes
  out/per_benchmark_cells.json         all values with their sources (display units: ps, LUTs, mW, % of best-known gain)
  out/per_benchmark_checks.txt         cross-checks, check values, best-known raises, not-run cells, repeat counts
  out/per_benchmark_bestknown_raises.csv
  out/per_benchmark_notrun.csv

Definitions (as in the style guide)
  gain          = final pick versus the start checkpoint, positive = better: WNS gain in ps (T0, W2R), packed LUTs removed (A0, L2R), dynamic power saved in mW (P0).
  task          = one design under one brief. Value of a cell = mean over the policy's repeats; sd = sample sd over repeats (none for one repeat).
  best known    = ref_known of Gen_references_per_cell.csv, raised where an LLM Orchestrator final (Gemini, GPT-5.6 Terra, Qwen3.5-397B; any repeat; brief-valid and AMD pass)
                  beats it (Prompt 1 step D). The raise is applied to every group.
  headroom task = best-known gain at least one noise band (power compared in whole mW), as tr_gen_analyse.ratio().
  % of best-known gain of a task = 100 x cell value / best-known gain; pooled = mean over the headroom tasks (tasks weighted equally).
Row selection for TRAINING and VALIDATION mirrors tr_gen_analyse.row_cells (same sources per row); TEST rows are the rows of cells.json (the released TEST results)
plus the post-freeze GPT repeats 2 and 3 and the Qwen row (episodes csv of the post-freeze task).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
INP = HERE / "inputs" / "per_benchmark"
OUT = HERE / "out"

GROUPS = ("TRAINING", "VALIDATION", "TEST")
DESIGNS = {
    "TRAINING": ["vexriscv_re-place", "logicnets_jscl", "rosetta_spam-filter", "rosetta_3d-rendering", "finn_radioml"],
    "VALIDATION": ["t2_cordic_s", "vtr_mcml_v2", "amd_mini-isp"],
    "TEST": ["corescore_500_mod", "fir_symmetric_systolic", "fir_systolic_transposed", "rosetta_digit-recognition", "vexriscv_re-place_v2"],
}
GROUP_OF = {d: g for g, ds in DESIGNS.items() for d in ds}
BRIEFS = ("T0", "A0", "W2R", "L2R", "P0")
MAIN, POWER = ("T0", "A0"), ("W2R", "L2R", "P0")
METRIC = {"T0": "wns_ns", "A0": "packed_luts", "W2R": "wns_ns", "L2R": "packed_luts", "P0": "dynamic_power_w"}
SCALE = {"wns_ns": 1000.0, "packed_luts": 1.0, "dynamic_power_w": 1000.0}     # native -> display
UNIT = {"wns_ns": "ps", "packed_luts": "LUTs", "dynamic_power_w": "mW"}
POLICIES = ("best", "gemini", "gpt", "qwen", "direct", "rule", "fixed", "bandit", "beam", "grid", "random")
LLM_ORCH = ("gemini", "gpt", "qwen")
SHORT_TO_FULL = {"corescore": "corescore_500_mod", "fir_symmetric": "fir_symmetric_systolic", "fir_transposed": "fir_systolic_transposed",
                 "rosetta_digit": "rosetta_digit-recognition", "vexriscv_v2": "vexriscv_re-place_v2"}
TEST_ROW = {"gemini": "V9", "direct": "GA-R1", "rule": "v2", "fixed": "fixed", "bandit": "bandit", "beam": "beam", "grid": "grid", "random": "random"}
GEN_ROW = {"gemini": "V9", "rule": "v2", "fixed": "fixed", "bandit": "bandit", "direct": "GA", "random": "random"}   # row names of Gen_share_per_cell.csv
PERCELL_ROW = {"gemini": "V9", "direct": "GA-R1", "rule": "v2", "fixed": "fixed", "bandit": "bandit", "beam": "beam", "grid": "grid", "random": "random"}
F_SRC = "inputs/per_benchmark/"


def verify_manifest() -> None:
    bad = []
    for line in (INP / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        h, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        if hashlib.sha256((INP / name).read_bytes()).hexdigest() != h:
            bad.append(name)
    if bad:
        raise SystemExit(f"STOP: input files differ from MANIFEST.sha256: {bad}")


def read_csv(name: str) -> pd.DataFrame:
    p = INP / name
    n = 0
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                n += 1
            else:
                break
    return pd.read_csv(p, skiprows=n)


def as_bool(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, str):
        return x.strip() == "True"
    return bool(x)


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    if len(xs) < 2:
        return None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ratio_head(brief: str, ref_native: float, band_native: float) -> float:
    """Best-known gain over one noise band, as tr_gen_analyse.ratio (power compared in whole mW). Both arguments in the source units (ns, LUTs, W)."""
    if METRIC[brief] == "dynamic_power_w":
        r, b = round(ref_native * 1000.0), round(band_native * 1000.0)
        return (r / b) if b > 0 else (math.inf if r > 0 else 0.0)
    return (ref_native / band_native) if band_native > 0 else (math.inf if ref_native > 0 else 0.0)


def main() -> int:
    verify_manifest()
    OUT.mkdir(exist_ok=True)
    CELLS = json.loads((INP / "cells.json").read_text(encoding="utf-8"))
    GEN = json.loads((INP / "gen_collect.json").read_text(encoding="utf-8"))
    EPS = {e["eid"]: e for e in json.loads((INP / "episodes.json").read_text(encoding="utf-8"))}
    ref_df, gshare, pcell = read_csv("Gen_references_per_cell.csv"), read_csv("Gen_share_per_cell.csv"), read_csv("Paper_Tables_PerCell.csv")
    pf_cells = read_csv("PostFreeze_Cells.csv")
    pf_eps = read_csv("PostFreeze_Episodes.csv")
    start_csv = read_csv("Eng_TEST_start_metrics.csv")
    notes: list[str] = []
    mism: list[str] = []          # cross-check mismatches: any entry means STOP and report

    # ------------------------------------------------------------------------------------------------------------------- start values
    start: dict[str, dict] = {}
    for _, r in start_csv.iterrows():
        d = SHORT_TO_FULL[r["design"]]
        start[d] = {"wns_ps": float(r["wns_ps"]), "luts": float(r["packed_luts"]), "mw": float(r["dynamic_power_mw"]),
                    "src": f"{F_SRC}Eng_TEST_start_metrics.csv | row design={r['design']} | columns wns_ps, packed_luts, dynamic_power_mw"}
        s = CELLS["starts"][d]
        for k, v, sc in (("wns_ps", s["wns_ns"], 1000.0), ("luts", s["packed_luts"], 1.0), ("mw", s["dynamic_power_w"], 1000.0)):
            if abs(start[d][k] - v * sc) > 0.51:
                mism.append(f"start {d} {k}: Eng_TEST_start_metrics {start[d][k]} vs cells.json {v * sc}")
    for d in DESIGNS["TRAINING"] + DESIGNS["VALIDATION"]:
        s = GEN["starts"][d]
        start[d] = {"wns_ps": float(round(s["wns_ns"] * 1000.0)), "luts": float(s["packed_luts"]), "mw": float(round(s["dynamic_power_w"] * 1000.0)),
                    "src": f"{F_SRC}gen_collect.json | starts.{d} | wns_ns x1000, packed_luts, dynamic_power_w x1000 (B2 start state stored with the episodes)"}
    order = {g: sorted(DESIGNS[g], key=lambda d: start[d]["luts"]) for g in GROUPS}

    # ------------------------------------------------------------------------------------------------------------------- per-cell policy values
    # one entry per (group, design, brief, policy): reps in display units, plus raise candidates (LLM Orchestrator finals) with their validity
    CELLV: dict[tuple, dict] = {}
    cand: dict[tuple, list] = defaultdict(list)      # (g, d, b) -> [(policy, display gain, valid, source text)]
    recs = defaultdict(list)
    for r in GEN["records"]:
        recs[(r["design"], r["brief"])].append(r)

    def put(g, d, b, p, reps, src, keys, extra=None):
        sc = SCALE[METRIC[b]]
        reps_d = [x * sc for x in reps]
        CELLV[(g, d, b, p)] = {"status": "value", "reps": reps_d, "n": len(reps_d), "mean": mean(reps_d), "sd": sd(reps_d), "src": src, "keys": keys, **(extra or {})}

    # TEST ---------------------------------------------------------------------------------------------------------------
    rows = CELLS["rows"]
    for d in DESIGNS["TEST"]:
        for b in BRIEFS:
            k = f"{d}|{b}"
            for p, lab in TEST_ROW.items():
                ents = rows.get(lab, {}).get(k)
                if not ents:
                    continue
                ents = sorted(ents, key=lambda e: (e["seed"] if e.get("seed") is not None else -1, str(e.get("tag"))))
                put("TEST", d, b, p, [e["gain"] for e in ents], f"{F_SRC}cells.json | rows.{lab}.{k} (all repeats) | gain (source unit to {UNIT[METRIC[b]]}: x{SCALE[METRIC[b]]:g})", [e["eid"] for e in ents])
                if p == "gemini":
                    for e in ents:
                        ep = EPS.get(e["eid"])
                        cand[("TEST", d, b)].append(("gemini", e["gain"] * SCALE[METRIC[b]], bool(ep and ep["valid"] and ep["validation"] == "pass"), e["eid"]))
            # GPT-5.6 Terra: repeat 1 = released TEST result (cells.json row 'V9 (ChatGPT)'), repeats 2 and 3 = post-freeze episodes
            g1 = rows.get("V9 (ChatGPT)", {}).get(k)
            sub = pf_eps[(pf_eps["data_scope"] == "TEST") & (pf_eps["row"] == "V9") & (pf_eps["design"] == d) & (pf_eps["brief"] == b)]
            gpt_sub = sub[sub["model"].str.startswith("ChatGPT")].sort_values("repeat")
            qwen_sub = sub[sub["model"].str.startswith("Qwen")].sort_values("repeat")
            if g1 and len(gpt_sub) == 2:
                reps = [g1[0]["gain"] * SCALE[METRIC[b]]]
                keys = [g1[0]["eid"]]
                valid = [bool(EPS[g1[0]["eid"]]["valid"] and EPS[g1[0]["eid"]]["validation"] == "pass")]
                for _, e in gpt_sub.iterrows():
                    reps.append(float(e["goal_gain"]))
                    keys.append(e["episode_id"])
                    valid.append(bool(as_bool(e["final_pick_brief_valid"]) and e["amd_check"] == "pass"))
                    if e["gain_unit"] != UNIT[METRIC[b]]:
                        mism.append(f"unit {e['episode_id']}: {e['gain_unit']}")
                put("TEST", d, b, "gpt", [x / SCALE[METRIC[b]] for x in reps], f"{F_SRC}cells.json (repeat 1, released) + {F_SRC}PostFreeze_Episodes.csv (repeats 2, 3) | rows.V9 (ChatGPT).{k} (repeat 1) + the episode_id rows listed in source_ids (repeats 2, 3) | gain (cells.json) and goal_gain ({UNIT[METRIC[b]]}, episodes csv)", keys)
                for x, v, kk in zip(reps, valid, keys):
                    cand[("TEST", d, b)].append(("gpt", x, v, kk))
            if len(qwen_sub) == 1:
                e = qwen_sub.iloc[0]
                put("TEST", d, b, "qwen", [float(e["goal_gain"]) / SCALE[METRIC[b]]], f"{F_SRC}PostFreeze_Episodes.csv | row episode_id={e['episode_id']} | goal_gain ({UNIT[METRIC[b]]})", [e["episode_id"]])
                cand[("TEST", d, b)].append(("qwen", float(e["goal_gain"]), bool(as_bool(e["final_pick_brief_valid"]) and e["amd_check"] == "pass"), e["episode_id"]))

    # TRAINING and VALIDATION (row selection as tr_gen_analyse.row_cells) ---------------------------------------------------
    def tv_select(g, d, b, p):
        kind = "main" if b in MAIN else "power"
        rs = recs[(d, b)]
        if p == "bandit":
            src = GEN["bandit_train"] if g == "TRAINING" else GEN["bandit_val"]
            e = src.get(f"{d}|{b}")
            return ([e["gain"]], f"{F_SRC}gen_collect.json | {'bandit_train' if g == 'TRAINING' else 'bandit_val'}.{d}|{b} | gain (source unit to {UNIT[METRIC[b]]}: x{SCALE[METRIC[b]]:g})", [f"{d}|{b}"]) if e is not None else None
        if p == "gemini":
            sel = [r for r in rs if r["row"] == "V9" and r["src"] == ("v22t3" if kind == "main" else "post_freeze")]
        elif p == "direct":
            sel = [r for r in rs if kind == "main" and g == "TRAINING" and r["row"] == "GA-trainingstage" and r["src"] == "ga3"]
        elif p in ("rule", "fixed"):
            name = "v2" if p == "rule" else "fixed"
            sel = [r for r in rs if r["row"] == name and r["src"] == ("d6v2" if kind == "main" else "power_review")]
        elif p == "random":
            sel = [r for r in rs if r["row"] == "random" and (r["src"] == "d6v2" if kind == "main" else r["src"] in ("dev_baselines", "p0_pilot"))]
        else:
            return None
        if not sel:
            return None
        sel = sorted(sel, key=lambda r: (r["rep"] if r["rep"] is not None else -1, str(r["eid"])))
        srcs = sorted({r["src"] for r in sel})
        return ([r["gain"] for r in sel], f"{F_SRC}gen_collect.json | records with design={d}, brief={b}, row={sel[0]['row']}, src={'/'.join(srcs)} (all repeats) | gain (source unit to {UNIT[METRIC[b]]}: x{SCALE[METRIC[b]]:g})", [r["eid"] for r in sel])

    for g in ("TRAINING", "VALIDATION"):
        for d in DESIGNS[g]:
            for b in BRIEFS:
                for p in ("gemini", "direct", "rule", "fixed", "bandit", "random"):
                    sel = tv_select(g, d, b, p)
                    if sel is None:
                        continue
                    put(g, d, b, p, sel[0], sel[1], sel[2])
                    if p == "gemini":
                        for x, kk in zip(sel[0], sel[2]):
                            cand[(g, d, b)].append(("gemini", x * SCALE[METRIC[b]], True, kk))     # kept records are brief-valid with AMD pass by construction (tr_gen_collect)
    for g in GROUPS:
        for d in DESIGNS[g]:
            for b in BRIEFS:
                for p in POLICIES:
                    if p != "best" and (g, d, b, p) not in CELLV:
                        CELLV[(g, d, b, p)] = {"status": "notrun"}

    # ------------------------------------------------------------------------------------------------------------------- best known per task
    REF = {r["key"]: r for _, r in ref_df.iterrows()}
    TASK: dict[tuple, dict] = {}
    raises: list[dict] = []
    for g in GROUPS:
        for d in DESIGNS[g]:
            for b in BRIEFS:
                key = f"TEST:{d}|{b}" if g == "TEST" else f"{d}|{b}"
                r = REF[key]
                sc = SCALE[METRIC[b]]
                band = float(r["band"])
                ref_native = float(r["ref_known"])
                ref_d = ref_native * sc
                head_csv = as_bool(r["head_known"] if g == "TEST" else r["head"])
                head_before = ratio_head(b, ref_native, band) >= 1.0
                if head_csv is not None and head_csv != head_before:
                    mism.append(f"headroom flag {key}: csv {head_csv} vs recomputed {head_before}")
                bk, by = ref_d, None
                for (pol, gain, valid, eid) in cand.get((g, d, b), []):
                    if valid and gain > bk + 1e-9:
                        bk, by = gain, (pol, eid)
                head = head_before if by is None else (ratio_head(b, bk / sc, band) >= 1.0)
                TASK[(g, d, b)] = {"band": band, "band_disp": band * sc, "ref_known": ref_d, "bk": bk, "head": head, "head_before": head_before, "unit": UNIT[METRIC[b]], "key": key,
                                   "raised_by": (f"{by[0]}:{by[1]}" if by else None)}
                if by:
                    raises.append({"group": g, "design": d, "brief": b, "unit": UNIT[METRIC[b]], "ref_known_before": ref_d, "best_known_after": bk, "raised_by_policy": by[0], "episode": by[1],
                                   "headroom_before": head_before, "headroom_after": head})
                put_best = {"status": "value", "reps": [bk], "n": 1, "mean": bk, "sd": None,
                            "src": f"{F_SRC}Gen_references_per_cell.csv | row key={key} | ref_known (source unit to {UNIT[METRIC[b]]}: x{sc:g})" + (f"; raised to {by[0]} final {by[1]} (Prompt 1 step D)" if by else ""),
                            "keys": [key]}
                CELLV[(g, d, b, "best")] = put_best
    # LLM Orchestrator rows that exist but whose non-V9 post-freeze siblings (V3, V6, V11) would beat the best known: listed only
    sib = []
    for _, e in pf_eps[(pf_eps["data_scope"] == "TEST") & (pf_eps["row"].isin(["V3", "V6", "V11"]))].iterrows():
        b, d = e["brief"], e["design"]
        if b in BRIEFS and as_bool(e["final_pick_brief_valid"]) and e["amd_check"] == "pass" and float(e["goal_gain"]) > TASK[("TEST", d, b)]["bk"] + 1e-9:
            sib.append(f"{e['row']} {e['episode_id']}: {float(e['goal_gain'])} {e['gain_unit']} > best known {TASK[('TEST', d, b)]['bk']}")

    # ------------------------------------------------------------------------------------------------------------------- cross-checks
    n_chk = defaultdict(int)
    matched: dict[tuple, set] = defaultdict(set)      # (g, d, b, policy) -> names of the csv files that hold a matching row

    def near(a, b_, _unused=None):
        return abs(a - b_) <= 1e-5 * max(1.0, abs(b_))

    # (1) Gen_share_per_cell.csv: every row (headroom tasks, TRAINING / VALIDATION / TEST_known) against my cells
    inv_gen = {v: k for k, v in GEN_ROW.items()}
    for _, r in gshare.iterrows():
        gv, row, b, d = r["group_version"], r["row"], r["brief"], r["design"]
        if gv == "TEST_ref" or row not in inv_gen or b not in BRIEFS:
            continue
        g = "TEST" if gv == "TEST_known" else gv
        c = CELLV.get((g, d, b, inv_gen[row]))
        n_chk["Gen_share_per_cell"] += 1
        if not c or c["status"] != "value":
            mism.append(f"Gen_share_per_cell {gv} {row} {b} {d}: csv has mean_gain {r['mean_gain']}, my cell is not run")
            continue
        mg = c["mean"] / SCALE[METRIC[b]]
        if not near(mg, float(r["mean_gain"]), 1.0) or int(r["repeats"]) != c["n"]:
            mism.append(f"Gen_share_per_cell {gv} {row} {b} {d}: csv mean_gain {r['mean_gain']} repeats {r['repeats']} vs mine {mg} n={c['n']}")
        t = TASK[(g, d, b)]
        if not near(float(r["reference_gain"]), t["ref_known"] / SCALE[METRIC[b]], 1.0):
            mism.append(f"Gen_share_per_cell {gv} {row} {b} {d}: csv reference_gain {r['reference_gain']} vs ref_known {t['ref_known'] / SCALE[METRIC[b]]}")
        elif near(mg, float(r["mean_gain"])) and int(r["repeats"]) == c["n"]:
            matched[(g, d, b, inv_gen[row])].add("Gen_share_per_cell.csv")
    # (2) Paper_Tables_PerCell.csv (TEST): rows by design, brief, row
    inv_pc = {v: k for k, v in PERCELL_ROW.items()}
    for _, r in pcell.iterrows():
        row, b, d = r["row"], r["brief"], r["design"]
        if b not in BRIEFS:
            continue
        if row == "V9 (ChatGPT)":
            c = CELLV.get(("TEST", d, b, "gpt"))
            mine = c["reps"][0] / SCALE[METRIC[b]] if c and c["status"] == "value" else None     # the PerCell row holds the released repeat 1 only
            nrep = 1
        elif row in inv_pc:
            c = CELLV.get(("TEST", d, b, inv_pc[row]))
            mine = c["mean"] / SCALE[METRIC[b]] if c and c["status"] == "value" else None
            nrep = c["n"] if c and c["status"] == "value" else None
        else:
            continue
        n_chk["Paper_Tables_PerCell"] += 1
        if int(r["repeats"]) == 0:           # the csv lists the row without a result (Learned Bandit, W2R): mine must be not run as well
            if mine is not None:
                mism.append(f"Paper_Tables_PerCell {row} {b} {d}: csv has no result, mine {mine}")
            continue
        if mine is None or not near(mine, float(r["mean_gain"]), 1.0) or nrep != int(r["repeats"]):
            mism.append(f"Paper_Tables_PerCell {row} {b} {d}: csv mean_gain {r['mean_gain']} repeats {r['repeats']} vs mine {mine} n={nrep}")
        elif row != "V9 (ChatGPT)":              # the GPT row holds repeat 1 only, so it does not count as a full match of the 3-repeat cell
            matched[("TEST", d, b, inv_pc[row])].add("Paper_Tables_PerCell.csv")
    # (3) PostFreeze Cells csv: GPT (3 repeats), Qwen, V9 Gemini on TRAINING / VALIDATION power briefs
    lab_map = {("TEST", "V9, ChatGPT, repeats 1 to 3 (d)"): "gpt", ("TEST", "V9, Qwen3.5-397B (a)"): "qwen", ("TRAINING", "V9, Gemini (this run)"): "gemini",
               ("VALIDATION", "V9, Gemini (this run)"): "gemini", ("TEST", "V9, Gemini, mean of 3 repeats"): "gemini"}
    seen = defaultdict(dict)
    for _, r in pf_cells.iterrows():
        pol = lab_map.get((r["data_scope"], r["row"]))
        if pol is None or r["brief"] not in BRIEFS:
            continue
        seen[(r["data_scope"], r["design"], r["brief"], pol)][int(r["repeat"])] = float(r["goal_gain"])
    for (g, d, b, pol), by_rep in seen.items():
        c = CELLV.get((g, d, b, pol))
        n_chk["PostFreeze_Cells"] += 1
        got = sorted(by_rep.items())
        if not c or c["status"] != "value" or len(got) != c["n"] or any(abs(v - c["reps"][i]) > 1e-6 * max(1, abs(v)) for i, (_, v) in enumerate(got)):
            mism.append(f"PostFreeze_Cells {g} {d} {b} {pol}: csv {got} vs mine {c.get('reps') if c else None}")
        else:
            matched[(g, d, b, pol)].add("PostFreeze_Cells.csv")
    unmatched = [(g, d, b, p) for g in GROUPS for d in order[g] for b in BRIEFS for p in POLICIES[1:]
                 if TASK[(g, d, b)]["head"] and CELLV[(g, d, b, p)]["status"] == "value" and not matched.get((g, d, b, p))]

    # ------------------------------------------------------------------------------------------------------------------- pooled values and totals
    def share_of(g, d, b, p):
        c, t = CELLV[(g, d, b, p)], TASK[(g, d, b)]
        return 100.0 * c["mean"] / t["bk"]

    def pooled(task_list, p):
        """task_list = [(g, d, b)] headroom tasks. Returns dict with mean % over the covered tasks, sd by quadrature when every covered task has >= 2 repeats."""
        cov = [(g, d, b) for (g, d, b) in task_list if CELLV[(g, d, b, p)]["status"] == "value"]
        if not task_list:
            return {"status": "na"}
        if not cov:
            return {"status": "notrun", "tasks_head": len(task_list), "tasks_cov": 0}
        sh = [share_of(g, d, b, p) for (g, d, b) in cov]
        sds = [CELLV[(g, d, b, p)]["sd"] for (g, d, b) in cov]
        sdv = math.sqrt(sum((100.0 * s / TASK[(g, d, b)]["bk"]) ** 2 for s, (g, d, b) in zip(sds, cov))) / len(cov) if all(s is not None for s in sds) else None
        return {"status": "value", "mean": mean(sh), "sd": sdv, "tasks_head": len(task_list), "tasks_cov": len(cov),
                "designs_cov": len({d for (_, d, _) in cov}), "n_rep_min": min(CELLV[(g, d, b, p)]["n"] for (g, d, b) in cov), "shares": sh}

    OVERALL, TOTAL = {}, {}
    scopes = {"all": [(g, d) for g in GROUPS for d in order[g]], "TEST": [("TEST", d) for d in order["TEST"]]}
    for g in GROUPS:
        for d in DESIGNS[g]:
            heads = [(g, d, b) for b in BRIEFS if TASK[(g, d, b)]["head"]]
            for p in POLICIES[1:]:
                OVERALL[(g, d, p)] = pooled(heads, p)
    for sc_name, ds in scopes.items():
        N = len(ds)
        for b in BRIEFS:
            for p in POLICIES:
                have = [(g, d) for (g, d) in ds if CELLV[(g, d, b, p)]["status"] == "value"]
                if not have:
                    TOTAL[(sc_name, b, p)] = {"status": "notrun", "k": 0, "N": N}
                    continue
                vals = [CELLV[(g, d, b, p)]["mean"] for (g, d) in have]
                sds = [CELLV[(g, d, b, p)]["sd"] for (g, d) in have]
                bkc = sum(TASK[(g, d, b)]["bk"] for (g, d) in have)
                tot = sum(vals)
                TOTAL[(sc_name, b, p)] = {"status": "value", "mean": tot, "k": len(have), "N": N, "shade": (tot / bkc if bkc > 0 else None),
                                          "sd": (math.sqrt(sum(s * s for s in sds)) if all(s is not None for s in sds) else None)}
        for p in POLICIES[1:]:
            heads = [(g, d, b) for (g, d) in ds for b in BRIEFS if TASK[(g, d, b)]["head"]]
            res = pooled(heads, p)
            nh = len({d for (_, d, _) in heads})
            res.update({"k": res.get("designs_cov", 0), "N": nh})
            TOTAL[(sc_name, "overall", p)] = res
        for m, key in (("wns", "wns_ps"), ("luts", "luts"), ("mw", "mw")):
            TOTAL[(sc_name, "start", m)] = {"status": "value", "mean": sum(start[d][key] for (_, d) in ds), "k": N, "N": N}

    # ------------------------------------------------------------------------------------------------------------------- check values, counts, listings
    out_lines: list[str] = []
    w = out_lines.append
    cs = start["corescore_500_mod"]
    w("CHECK VALUES")
    w(f"  TEST corescore_500_mod start: {cs['wns_ps']:.0f} ps, {cs['luts']:.0f} LUTs, {cs['mw']:.0f} mW   (expected -1204 ps, 105957 LUTs, 13364 mW)")
    fx = CELLV[("TEST", "corescore_500_mod", "A0", "fixed")]["mean"]
    bkc = TASK[("TEST", "corescore_500_mod", "A0")]
    w(f"  TEST corescore_500_mod A0 (area-first, two workers): Fixed Worker Order {fx:.0f} LUTs removed, best known {bkc['bk']:.0f}   (expected 543 and 742)")
    qw = CELLV[("TEST", "corescore_500_mod", "L2R", "qwen")]["mean"]
    w(f"  TEST corescore_500_mod L2R (area-first, three workers): LLM Orchestrator, Qwen {qw:.0f} LUTs removed   (expected 464)")
    ok = (cs["wns_ps"] == -1204 and cs["luts"] == 105957 and cs["mw"] == 13364 and round(fx) == 543 and round(bkc["bk"]) == 742 and round(qw) == 464)
    w(f"  check values reproduced: {ok}")
    if not ok:
        mism.append("check values not reproduced")
    w("")

    def pooled_set(g, p, bs):
        tasks = [(g, d, b) for d in DESIGNS[g] for b in bs if TASK[(g, d, b)]["head"] and CELLV[(g, d, b, p)]["status"] == "value"]
        return (mean([share_of(gg, d, b, p) for (gg, d, b) in tasks]) if tasks else None), len(tasks)

    w("CONSISTENCY WITH THE PROMPT 1 CHECK VALUES (% of best-known gain, pooled over headroom tasks; informational, not part of this figure)")
    p1 = [("TEST", "gemini", MAIN, 56.9, (44.8, 66.0)), ("TEST", "fixed", MAIN, 56.8, (46.8, 64.2)), ("TEST", "gemini", POWER, 46.0, (49.5, 58.9, 29.5)),
          ("TEST", "fixed", POWER, 52.0, (37.1, 61.7, 57.3)), ("TRAINING", "gemini", MAIN, 40.6, (37.3, 43.2)), ("TRAINING", "fixed", MAIN, 56.3, (44.4, 65.9)),
          ("VALIDATION", "gemini", MAIN, 62.5, (52.0, 73.0)), ("VALIDATION", "fixed", MAIN, 62.8, (65.2, 60.5)), ("VALIDATION", "fixed", POWER, 57.1, (44.9, 58.7, 67.6))]
    p1_diff = 0
    for g, p, bs, e_pool, e_each in p1:
        m_pool, n_t = pooled_set(g, p, bs)
        m_each = [pooled_set(g, p, (b,))[0] for b in bs]
        good = abs(m_pool - e_pool) <= 0.051 and all(abs(m - e) <= 0.051 for m, e in zip(m_each, e_each))
        p1_diff += 0 if good else 1
        w(f"  {g:10s} {p:7s} {'+'.join(bs):12s} pooled {m_pool:5.2f} ({n_t} tasks) vs {e_pool}; per brief " + " / ".join(f"{m:.2f}" for m in m_each) + f" vs {e_each}   {'ok' if good else 'DIFFERENT'}")
    w(f"  Prompt 1 check values that differ: {p1_diff} of {len(p1)}")
    w("")
    w(f"CROSS-CHECKS (rows compared): {dict(n_chk)}")
    w(f"  mismatches: {len(mism)}")
    for m in mism:
        w("   - " + m)
    n_head_cells = sum(1 for g in GROUPS for d in DESIGNS[g] for b in BRIEFS for p in POLICIES[1:] if TASK[(g, d, b)]["head"] and CELLV[(g, d, b, p)]["status"] == "value")
    w(f"  headroom-task cells with a value: {n_head_cells}; matched to a row of at least one of the three sources: {n_head_cells - len(unmatched)}; without any source row to compare: {len(unmatched)}")
    for (g, d, b, p) in unmatched:
        w(f"   - no source row: {g} {d} {b} {p}")
    w("")
    w(f"BEST-KNOWN RAISES (Prompt 1 step D: an LLM Orchestrator final above ref_known): {len(raises)}")
    for r in raises:
        w(f"   - {r['group']} {r['design']} {r['brief']}: {r['ref_known_before']:.4g} -> {r['best_known_after']:.4g} {r['unit']} by {r['raised_by_policy']} ({r['episode']}); headroom before {r['headroom_before']}, after {r['headroom_after']}")
    w(f"  post-freeze non-V9 variants (V3, V6, V11) above the (raised) best known, not used for the raise: {len(sib)}")
    for s in sib:
        w("   - " + s)
    w("")
    n_head = {g: sum(1 for d in DESIGNS[g] for b in BRIEFS if TASK[(g, d, b)]["head"]) for g in GROUPS}
    w(f"HEADROOM TASKS per group after the raise (TRAINING and TEST have 25 tasks each, VALIDATION 15): {n_head}")
    w("")
    w("REPEATS PER POLICY, GROUP AND BRIEF (distinct n over the designs with data)")
    for p in POLICIES[1:]:
        for g in GROUPS:
            s = {}
            for b in BRIEFS:
                ns = sorted({CELLV[(g, d, b, p)]["n"] for d in DESIGNS[g] if CELLV[(g, d, b, p)]["status"] == "value"})
                cnt = sum(1 for d in DESIGNS[g] if CELLV[(g, d, b, p)]["status"] == "value")
                s[b] = f"{cnt}/{len(DESIGNS[g])} designs x {'/'.join(map(str, ns))} rep" if cnt else "not run"
            w(f"  {p:7s} {g:10s} " + "  ".join(f"{b}={s[b]}" for b in BRIEFS))
    notrun_rows = []
    for g in GROUPS:
        for d in order[g]:
            for b in BRIEFS:
                for p in POLICIES[1:]:
                    if CELLV[(g, d, b, p)]["status"] == "notrun":
                        notrun_rows.append((g, d, b, p, TASK[(g, d, b)]["head"]))
    w("")
    w(f"NOT-RUN CELLS: {len(notrun_rows)} (design x brief x policy); of them on headroom tasks: {sum(1 for r in notrun_rows if r[4])}")
    (OUT / "per_benchmark_checks.txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    with open(OUT / "per_benchmark_notrun.csv", "w", newline="", encoding="utf-8") as fh:
        cw = csv.writer(fh)
        cw.writerow(["group", "design", "brief", "policy", "headroom_task"])
        cw.writerows(notrun_rows)
    with open(OUT / "per_benchmark_bestknown_raises.csv", "w", newline="", encoding="utf-8") as fh:
        cw = csv.DictWriter(fh, fieldnames=["group", "design", "brief", "unit", "ref_known_before", "best_known_after", "raised_by_policy", "episode", "headroom_before", "headroom_after"])
        cw.writeheader()
        cw.writerows(raises)

    # ------------------------------------------------------------------------------------------------------------------- json
    def jk(t):
        return "|".join(t)

    dump = {
        "schema": "per-benchmark-cells-v1", "groups": list(GROUPS), "briefs": list(BRIEFS), "policies": list(POLICIES), "order": order, "start": start,
        "tasks": {jk((g, d, b)): v for (g, d, b), v in TASK.items()},
        "cells": {jk((g, d, b, p)): v for (g, d, b, p), v in CELLV.items()},
        "overall": {jk(k): v for k, v in OVERALL.items()},
        "total": {jk(k): v for k, v in TOTAL.items()},
        "mismatches": mism, "n_cross_checks": dict(n_chk), "raises": raises,
        "inputs": {l.split()[1].lstrip("*"): l.split()[0] for l in (INP / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines() if l.strip()},
    }
    (OUT / "per_benchmark_cells.json").write_text(json.dumps(dump, indent=1, default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else str(o)), encoding="utf-8")
    print("\n".join(out_lines))
    return 1 if mism else 0


if __name__ == "__main__":
    sys.exit(main())
