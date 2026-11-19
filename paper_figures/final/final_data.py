#!/usr/bin/env python3
"""Final results tables, step 1: every number the four tables show, with its source. Offline and read-only: no tool run, no model call.

Reads copies of stored results only (inputs/ and inputs/per_benchmark/ of paper_figures/, hashes in their MANIFEST.sha256; inputs/final/ holds the fill-run
files, hashes in inputs/final/MANIFEST.sha256). The per-benchmark readers of the earlier script are reused: its constants and helpers are imported from
build_breakdown_inputs.py; the row selection of the stored episodes is adapted from its main() (cells.json for TEST, gen_collect.json for TRAINING and
VALIDATION) and extended with the fill runs.

Definitions
  gain          final pick against the starting checkpoint, positive = better, in display units: WNS gain in ps (T0, W2R), LUTs removed (A0, L2R),
                dynamic power saved in mW (P0).
  cell          one design under one brief; value = mean over the policy's repeats, sd = sample sd (n-1) over repeats (none for one repeat).
  headroom      head_known of Gen_references_per_cell.csv (TEST) / head (TRAINING and VALIDATION), recomputed as best-known gain >= one noise band
                (power compared in whole mW) and compared.
  summary cell  mean over the headroom designs of the brief of the cell values; sd = sample sd over repeats of that pooled value (repeat r of every design).
  % of best-known gain (model choice only, never shown): 100 x cell value / ref_known per headroom task, then the mean over the tasks of the setting.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import build_breakdown_inputs as BB  # noqa: E402  (constants and helpers only; its main() is not run)
import names as N  # noqa: E402
import names_per_benchmark as NPB  # noqa: E402

import final_config as CFG  # noqa: E402

GROUPS, DESIGNS, BRIEFS, MAIN, POWER = BB.GROUPS, BB.DESIGNS, BB.BRIEFS, BB.MAIN, BB.POWER
METRIC, SCALE, UNIT, SHORT_TO_FULL, TEST_ROW = BB.METRIC, BB.SCALE, BB.UNIT, BB.SHORT_TO_FULL, BB.TEST_ROW
SETTING_BRIEFS = {"2worker": ("T0", "A0"), "3worker": ("W2R", "L2R", "P0")}
SETTING_OF = {b: s for s, bs in SETTING_BRIEFS.items() for b in bs}
assert N.FINAL_DESIGN_SHORT == NPB.DESIGN_SHORT
EPS = 1e-9


def rnd(x: float, nd: int = 0) -> float:
    """Round half up on the decimal representation (the displayed digits of every table)."""
    q = Decimal(1).scaleb(-nd)
    return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def sd(xs):
    xs = list(xs)
    if len(xs) < 2:
        return None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rcsv(path: Path) -> pd.DataFrame:
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                n += 1
            else:
                break
    return pd.read_csv(path, skiprows=n)


def verify_inputs() -> list[str]:
    """Hash check of every input file this module reads. A difference stops the build."""
    bad, n_ok = [], 0
    for folder, manifest, only in ((ROOT / "inputs" / "per_benchmark", "MANIFEST.sha256", None), (ROOT / "inputs" / "final", "MANIFEST.sha256", None),
                                   (ROOT / "inputs", "MANIFEST.sha256", {"Eng_TEST_goal_and_protected_by_row_and_brief.csv", "Paper_Tables_Cells.csv", "Paper_Tables_PerCell.csv",
                                                                         "Paper_Tables_Table1_cost_per_task.csv", "Paper_Tables_Table2_cost_per_task.csv", "Eng_TEST_start_metrics.csv"})):
        mp = folder / manifest
        if not mp.exists():
            bad.append(f"missing manifest {mp}")
            continue
        for line in mp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            h, name = line.split(None, 1)
            name = name.strip().lstrip("*")
            if only is not None and name not in only:
                continue
            p = folder / name
            if not p.exists():
                bad.append(f"missing {p}")
            elif sha(p) != h:
                bad.append(f"hash differs {p}")
            else:
                n_ok += 1
    if bad:
        raise SystemExit("STOP: input files differ from their manifests: " + "; ".join(bad))
    return [f"input files hash-checked: {n_ok}"]


def disp(b: str, native: float) -> float:
    return native * SCALE[METRIC[b]]


def build() -> dict:
    log: list[str] = []
    mism: list[str] = []
    log += verify_inputs()
    INPP = ROOT / "inputs" / "per_benchmark"
    INPF = ROOT / "inputs" / "final"
    INP = ROOT / "inputs"
    CELLS = json.loads((INPP / "cells.json").read_text(encoding="utf-8"))
    BW2R_FILE = INP / "bandit_w2r_cells.json"      # post-freeze Learned Bandit timing-first cells (stored episode results)
    BW2R = json.loads(BW2R_FILE.read_text(encoding="utf-8")) if BW2R_FILE.exists() else None
    GEN = json.loads((INPP / "gen_collect.json").read_text(encoding="utf-8"))
    ref_df = rcsv(INPP / "Gen_references_per_cell.csv")
    pf_eps = rcsv(INPP / "PostFreeze_Episodes.csv")
    pf_cells = rcsv(INPP / "PostFreeze_Cells.csv")
    start_csv = rcsv(INPP / "Eng_TEST_start_metrics.csv")
    gshare = rcsv(INPP / "Gen_share_per_cell.csv")
    eng = rcsv(INP / "Eng_TEST_goal_and_protected_by_row_and_brief.csv")
    percell = rcsv(INP / "Paper_Tables_PerCell.csv")
    ptcells = rcsv(INP / "Paper_Tables_Cells.csv")
    cost1, cost2 = rcsv(INP / "Paper_Tables_Table1_cost_per_task.csv"), rcsv(INP / "Paper_Tables_Table2_cost_per_task.csv")
    fill_cells = rcsv(INPF / "Fill_Cells.csv")
    fill_eps = rcsv(INPF / "Fill_Episodes.csv")
    fill_notrun = rcsv(INPF / "Fill_Not_run.csv")
    FP = "inputs/final/"
    PB = "inputs/per_benchmark/"
    IN = "inputs/"

    # ------------------------------------------------------------------------------------------------------------- starts
    start: dict[str, dict] = {}
    for _, r in start_csv.iterrows():
        d = SHORT_TO_FULL[r["design"]]
        start[d] = {"wns": float(r["wns_ps"]), "luts": float(r["packed_luts"]), "mw": float(r["dynamic_power_mw"]),
                    "src": f"{PB}Eng_TEST_start_metrics.csv | design={r['design']} | wns_ps, packed_luts, dynamic_power_mw"}
        s = CELLS["starts"][d]
        for k, v, sc in (("wns", s["wns_ns"], 1000.0), ("luts", s["packed_luts"], 1.0), ("mw", s["dynamic_power_w"], 1000.0)):
            if abs(start[d][k] - v * sc) > 0.51:
                mism.append(f"start {d} {k}: Eng_TEST_start_metrics {start[d][k]} vs cells.json {v * sc}")
    for d in DESIGNS["TRAINING"] + DESIGNS["VALIDATION"]:
        s = GEN["starts"][d]
        start[d] = {"wns": float(rnd(s["wns_ns"] * 1000.0)), "luts": float(s["packed_luts"]), "mw": float(rnd(s["dynamic_power_w"] * 1000.0)),
                    "src": f"{PB}gen_collect.json | starts.{d} | wns_ns x1000, packed_luts, dynamic_power_w x1000"}
    order = {g: sorted(DESIGNS[g], key=lambda d: start[d]["luts"]) for g in GROUPS}

    # ------------------------------------------------------------------------------------------------------------- references and headroom
    REF = {r["key"]: r for _, r in ref_df.iterrows()}
    TASK: dict[tuple, dict] = {}
    for g in GROUPS:
        for d in DESIGNS[g]:
            for b in BRIEFS:
                key = f"TEST:{d}|{b}" if g == "TEST" else f"{d}|{b}"
                r = REF[key]
                head_csv = BB.as_bool(r["head_known"] if g == "TEST" else r["head"])
                ref_native, band = float(r["ref_known"]), float(r["band"])
                head_re = BB.ratio_head(b, ref_native, band) >= 1.0
                if head_csv != head_re:
                    mism.append(f"headroom flag {key}: csv {head_csv} vs recomputed {head_re}")
                TASK[(g, d, b)] = {"key": key, "ref_known": disp(b, ref_native), "band": disp(b, band), "head": bool(head_csv), "unit": UNIT[METRIC[b]],
                                   "src": f"{PB}Gen_references_per_cell.csv | key={key} | ref_known (x{SCALE[METRIC[b]]:g}), {'head_known' if g == 'TEST' else 'head'}"}

    # ------------------------------------------------------------------------------------------------------------- policy cells
    V: dict[tuple, dict] = {}
    recs: dict = {}
    for r in GEN["records"]:
        recs.setdefault((r["design"], r["brief"]), []).append(r)

    def put(g, d, b, p, reps, src, keys):
        V[(g, d, b, p)] = {"reps": list(reps), "n": len(reps), "mean": mean(reps), "sd": sd(reps), "src": src, "keys": keys}

    EPS_ = {e["eid"]: e for e in json.loads((INPP / "episodes.json").read_text(encoding="utf-8"))}
    for d in DESIGNS["TEST"]:
        for b in BRIEFS:
            k = f"{d}|{b}"
            for p, lab in TEST_ROW.items():
                ents = CELLS["rows"].get(lab, {}).get(k)
                if not ents:
                    continue
                ents = sorted(ents, key=lambda e: (e["seed"] if e.get("seed") is not None else -1, str(e.get("tag"))))
                put("TEST", d, b, p, [disp(b, e["gain"]) for e in ents], f"{PB}cells.json | rows.{lab}.{k} | gain (x{SCALE[METRIC[b]]:g})", [e["eid"] for e in ents])
            g1 = CELLS["rows"].get("V9 (ChatGPT)", {}).get(k)
            sub = pf_eps[(pf_eps["data_scope"] == "TEST") & (pf_eps["row"] == "V9") & (pf_eps["design"] == d) & (pf_eps["brief"] == b)]
            gpt_sub, qwen_sub = sub[sub["model"].str.startswith("ChatGPT")].sort_values("repeat"), sub[sub["model"].str.startswith("Qwen")].sort_values("repeat")
            if g1 and len(gpt_sub) == 2:
                reps = [disp(b, g1[0]["gain"])] + [float(e["goal_gain"]) for _, e in gpt_sub.iterrows()]
                put("TEST", d, b, "gpt", reps, f"{PB}cells.json rows.V9 (ChatGPT).{k} (repeat 1, gain x{SCALE[METRIC[b]]:g}) + {PB}PostFreeze_Episodes.csv (repeats 2, 3, goal_gain)",
                    [g1[0]["eid"]] + list(gpt_sub["episode_id"]))
            if len(qwen_sub) == 1:
                e = qwen_sub.iloc[0]
                put("TEST", d, b, "qwen", [float(e["goal_gain"])], f"{PB}PostFreeze_Episodes.csv | episode_id={e['episode_id']} | goal_gain", [e["episode_id"]])

    fc = fill_cells.copy()
    fc["rep"] = fc["repeat"].astype(int)

    def fill_rows(g, d, b, row):
        s = fc[(fc["data_scope"] == g) & (fc["design"] == d) & (fc["brief"] == b) & (fc["row"] == row)].sort_values("rep")
        return s

    for g in ("TRAINING", "VALIDATION"):
        for d in DESIGNS[g]:
            for b in BRIEFS:
                kind = "main" if b in MAIN else "power"
                rs = recs[(d, b)]
                sc = SCALE[METRIC[b]]
                # rule-based policy and fixed worker order: one deterministic run each
                for p, name in (("rule", "v2"), ("fixed", "fixed")):
                    sel = [r for r in rs if r["row"] == name and r["src"] == ("d6v2" if kind == "main" else "power_review")]
                    if sel:
                        sel = sorted(sel, key=lambda r: str(r["eid"]))
                        put(g, d, b, p, [r["gain"] * sc for r in sel], f"{PB}gen_collect.json | records design={d}, brief={b}, row={name}, src={sel[0]['src']} | gain (x{sc:g})", [r["eid"] for r in sel])
                # LLM-Orchestrator, Gemini: two workers = v22t3 (3 repeats); three workers = post-freeze repeat 1 + fill repeats 2, 3
                if kind == "main":
                    sel = sorted([r for r in rs if r["row"] == "V9" and r["src"] == "v22t3"], key=lambda r: (r["rep"], str(r["eid"])))
                    if sel:
                        put(g, d, b, "gemini", [r["gain"] * sc for r in sel], f"{PB}gen_collect.json | records design={d}, brief={b}, row=V9, src=v22t3 (all repeats) | gain (x{sc:g})", [r["eid"] for r in sel])
                else:
                    r1 = [r for r in rs if r["row"] == "V9" and r["src"] == "post_freeze"]
                    f23 = fill_rows(g, d, b, "V9, Gemini")
                    if r1 and len(f23) == 2 and list(f23["rep"]) == [2, 3]:
                        put(g, d, b, "gemini", [r1[0]["gain"] * sc] + [float(x) for x in f23["goal_gain"]],
                            f"{PB}gen_collect.json (repeat 1: records design={d}, brief={b}, row=V9, src=post_freeze, gain x{sc:g}) + {FP}Fill_Cells.csv (repeats 2, 3: row 'V9, Gemini' | goal_gain)",
                            [r1[0]["eid"]] + [f"fill:{g}:{d}:{b}:V9:r{x}" for x in f23["rep"]])
                    f1 = fill_rows(g, d, b, "V9, ChatGPT gpt-5.6-terra")
                    if len(f1) == 1:
                        put(g, d, b, "gpt", [float(f1.iloc[0]["goal_gain"])], f"{FP}Fill_Cells.csv | row 'V9, ChatGPT gpt-5.6-terra', repeat 1 | goal_gain", [f"fill:{g}:{d}:{b}:gpt:r1"])
                # Direct LLM Tool Control: three fresh repeats of the fill run on every TRAINING and VALIDATION cell
                fd = fill_rows(g, d, b, "GA-R1, Gemini")
                if len(fd) == 3 and list(fd["rep"]) == [1, 2, 3]:
                    put(g, d, b, "direct", [float(x) for x in fd["goal_gain"]], f"{FP}Fill_Cells.csv | row 'GA-R1, Gemini' | goal_gain (repeats 1 to 3)", [f"fill:{g}:{d}:{b}:GA-R1:r{x}" for x in fd["rep"]])

    # ------------------------------------------------------------------------------------------------------------- fill cross-checks
    fe = fill_eps.copy()
    fe["rep"] = fe["repeat"].astype(str).str.lstrip("r").astype(int)
    n_fe = 0
    for _, r in fc.iterrows():
        if str(r["row"]).startswith("exhaustive"):
            continue
        m = fe[(fe["data_scope"] == r["data_scope"]) & (fe["design"] == r["design"]) & (fe["brief"] == r["brief"]) & (fe["row"] == r["row"]) & (fe["rep"] == r["rep"])]
        n_fe += 1
        if len(m) != 1 or abs(float(m.iloc[0]["goal_gain"]) - float(r["goal_gain"])) > 1e-6 * max(1.0, abs(float(r["goal_gain"]))):
            mism.append(f"fill cells vs fill episodes: {r['data_scope']} {r['design']} {r['brief']} {r['row']} r{r['rep']}")
        elif str(m.iloc[0]["final_pick_brief_valid"]) != "True" or m.iloc[0]["amd_check"] != "pass" or int(m.iloc[0]["limit_failures"]) != 0:
            mism.append(f"fill episode not valid/pass/0 limit failures: {m.iloc[0]['episode_id']}")
    log.append(f"fill rows cross-checked against the fill episodes (goal gain, brief-valid, AMD pass, 0 limit failures): {n_fe}")

    # Repeat 1 of the Gemini orchestrator on the power briefs against the post-freeze cells csv
    n_pf = 0
    for _, r in pf_cells[pf_cells["row"] == "V9, Gemini (this run)"].iterrows():
        c = V.get((r["data_scope"], r["design"], r["brief"], "gemini"))
        n_pf += 1
        if not c or abs(c["reps"][0] - float(r["goal_gain"])) > 1e-6 * max(1.0, abs(float(r["goal_gain"]))):
            mism.append(f"post-freeze repeat 1 {r['data_scope']} {r['design']} {r['brief']}: csv {r['goal_gain']} vs mine {c['reps'][0] if c else None}")
    log.append(f"V9 Gemini repeat 1 (TRAINING, VALIDATION power briefs) against the post-freeze cells csv: {n_pf} cells")

    # unchanged cells against Gen_share_per_cell.csv (rule, fixed on TRAINING/VALIDATION; Gemini two workers)
    inv = {"V9": "gemini", "v2": "rule", "fixed": "fixed"}
    n_gs = 0
    for _, r in gshare.iterrows():
        gv, row, b, d = r["group_version"], r["row"], r["brief"], r["design"]
        if gv not in ("TRAINING", "VALIDATION") or row not in inv or b not in BRIEFS:
            continue
        p = inv[row]
        if p == "gemini" and b in POWER:
            continue                                   # three repeats now; the csv holds repeat 1 only
        c = V.get((gv, d, b, p))
        n_gs += 1
        if not c or abs(c["mean"] / SCALE[METRIC[b]] - float(r["mean_gain"])) > 1e-5 * max(1.0, abs(float(r["mean_gain"]))) or int(r["repeats"]) != c["n"]:
            mism.append(f"Gen_share_per_cell {gv} {row} {b} {d}: csv {r['mean_gain']} x{r['repeats']} vs mine {c['mean'] / SCALE[METRIC[b]] if c else None} x{c['n'] if c else None}")
    log.append(f"unchanged TRAINING/VALIDATION cells against Gen_share_per_cell.csv: {n_gs}")

    # ------------------------------------------------------------------------------------------------------------- exhaustive column
    PTC = {(r["design"], r["brief"]): r for _, r in ptcells.iterrows()}
    fill_exh = fc[fc["row"].astype(str).str.startswith("exhaustive 2-step")]
    X: dict[tuple, dict] = {}
    for g in GROUPS:
        for d in DESIGNS[g]:
            for b in BRIEFS:
                key = TASK[(g, d, b)]["key"]
                rr = REF[key]
                sc = SCALE[METRIC[b]]
                if g == "TEST":
                    r = PTC[(d, b)]
                    X[(g, d, b)] = {"val": float(r["reference_gain"]) * sc, "kind": "TEST3" if b in MAIN else "TEST2p", "mark": "",
                                    "src": f"{IN}Paper_Tables_Cells.csv | design={d}, brief={b} | reference_gain (x{sc:g})"}
                elif g == "VALIDATION":
                    X[(g, d, b)] = {"val": float(rr["val_exhaustive_2step"]) * sc, "kind": "VAL2", "mark": "2",
                                    "src": f"{PB}Gen_references_per_cell.csv | key={key} | val_exhaustive_2step (x{sc:g})"}
                elif b in MAIN:
                    X[(g, d, b)] = {"val": float(rr["stored_best_known_amd_pass_checked"]) * sc, "kind": "TRAIN3", "mark": "",
                                    "src": f"{PB}Gen_references_per_cell.csv | key={key} | stored_best_known_amd_pass_checked (x{sc:g})"}
                else:
                    f = fill_exh[(fill_exh["data_scope"] == g) & (fill_exh["design"] == d) & (fill_exh["brief"] == b)]
                    if len(f) == 1:
                        X[(g, d, b)] = {"val": float(f.iloc[0]["goal_gain"]), "kind": "TRAIN2p", "mark": "",
                                        "src": f"{FP}Fill_Cells.csv | row 'exhaustive 2-step (14 recipes)', design={d}, brief={b} | goal_gain"}
                    else:
                        raw = float(rr["train_ceiling_amd_pass_checked"])
                        val = raw if b == "W2R" else (raw * 1000.0 if b == "P0" else raw)       # this column holds W2R in ps, P0 in W, L2R in LUTs
                        if abs(val - TASK[(g, d, b)]["ref_known"]) > 0.51:
                            mism.append(f"train_ceiling {key}: {val} vs ref_known {TASK[(g, d, b)]['ref_known']}")
                        X[(g, d, b)] = {"val": val, "kind": "TRAIN_PENDING", "mark": "†",
                                        "src": f"{PB}Gen_references_per_cell.csv | key={key} | train_ceiling_amd_pass_checked (six-run hindsight best; exhaustive 2-step pending; W2R in ps, P0 x1000 from W)"}
    for (g, d, b), x in X.items():
        t = TASK[(g, d, b)]
        if g == "TEST" and abs(x["val"] - t["ref_known"]) > 1e-6 and x["val"] > t["ref_known"] + 1e-6:
            mism.append(f"exhaustive above best known: {g} {d} {b}")
    n_train_pending = sum(1 for x in X.values() if x["kind"] == "TRAIN_PENDING")
    log.append(f"TRAINING three-worker exhaustive cells still pending (shown as the six-run hindsight best with a dagger): {n_train_pending} of 15")

    # ------------------------------------------------------------------------------------------------------------- raises of the best known by any stored policy result
    raises = []
    for (g, d, b, p), c in V.items():
        if p in ("bandit", "beam", "random"):
            continue
        t = TASK[(g, d, b)]
        for i, x in enumerate(c["reps"]):
            if x > t["ref_known"] + 1e-6:
                raises.append((g, d, b, p, i + 1, x, t["ref_known"]))
    log.append(f"policy finals above the best known gain (any repeat; validity not re-checked here): {len(raises)}")

    # ------------------------------------------------------------------------------------------------------------- summary tables (TEST)
    ENG = {(r["table"], r["row_key"], r["brief"]): r for _, r in eng.iterrows()}
    PC = {(r["design"], r["brief"], r["row"]): r for _, r in percell.iterrows()}
    cost = {1: {r["key"]: r for _, r in cost1.iterrows()}, 2: {r["key"]: r for _, r in cost2.iterrows()}}
    summ: dict[str, dict] = {}

    def percell_reps(d, b, row):
        r = PC.get((d, b, row))
        if r is None or int(r["repeats"]) == 0 or pd.isna(r["gains_by_repeat"]):
            return None
        return [float(x) * SCALE[METRIC[b]] for x in str(r["gains_by_repeat"]).split(";")]

    cols = ("exh", "direct", "orch", "rule", "fixed", "bandit", "grid")
    for setting in ("2worker", "3worker"):
        briefs = SETTING_BRIEFS[setting]
        tab = "1a" if setting == "2worker" else "2a"
        ctab = 1 if setting == "2worker" else 2
        ekey = {"exh": "ref3" if setting == "2worker" else "ref2p", "direct": "GA-R1", "orch": "V9", "rule": "v2", "fixed": "fixed", "bandit": "bandit", "grid": "grid"}
        heads = {b: [d for d in order["TEST"] if TASK[("TEST", d, b)]["head"]] for b in briefs}
        S = {"briefs": briefs, "heads": heads, "gain": {}, "cost": {}}
        for c_ in cols:
            for b in briefs:
                e = ENG.get((tab, ekey[c_], b))
                cell = {"state": "ok"}
                if e is None and c_ == "bandit" and b == "W2R" and setting == "3worker" and BW2R is not None:
                    gains = {r["design"]: float(r["gain_ps"]) for r in BW2R["cells"]}
                    hd = heads[b]
                    if sorted(hd) != sorted(BW2R["summary"]["head_designs"]) or not all(d in gains for d in hd):
                        mism.append("summary 3worker bandit W2R: headroom designs differ between the stored cells and the table inputs")
                    S["gain"][(c_, b)] = {"state": "ok", "value": mean(gains[d] for d in hd), "n": 1, "sd": None, "reps": None, "post_freeze": True,
                                          "scope": "post-freeze, prior fitted 17 Nov from TRAINING and VALIDATION only",
                                          "src": f"{IN}bandit_w2r_cells.json | brief=W2R, row=bandit (post-freeze), 5 TEST designs, one deterministic episode each | gain_ps ({len(hd)} headroom of 5 designs, 1 repeat)"}
                    continue
                if e is None:
                    cell = {"state": "notrun", "note": "no episode: the frozen bandit has no W2R prior" if c_ == "bandit" else "not stored"}
                    S["gain"][(c_, b)] = cell
                    continue
                val = float(e["gain_mean_head"])
                n_rep = int(e["repeats"])
                cell.update({"value": val, "n": n_rep, "sd": None, "reps": None,
                             "src": f"{IN}Eng_TEST_goal_and_protected_by_row_and_brief.csv | table={tab}, row_key={ekey[c_]}, brief={b} | gain_mean_head ({int(e['cells_head'])} headroom of {int(e['cells_all'])} designs, {n_rep} repeat{'s' if n_rep > 1 else ''})"})
                # recompute from the per-design cells for the cross-check
                if c_ == "exh":
                    recomputed = mean(X[("TEST", d, b)]["val"] for d in heads[b])
                else:
                    pol = {"direct": "direct", "orch": "gemini", "rule": "rule", "fixed": "fixed", "bandit": "bandit", "grid": "grid"}[c_]
                    cs = [V.get(("TEST", d, b, pol)) for d in heads[b]]
                    recomputed = mean(c["mean"] for c in cs) if all(cs) else None
                if recomputed is None or abs(recomputed - val) > 1e-3 * max(1.0, abs(val)) + 1e-3:
                    mism.append(f"summary {setting} {c_} {b}: Eng_TEST {val} vs recomputed {recomputed}")
                if n_rep > 1:
                    pol = {"direct": "GA-R1", "orch": "V9"}[c_]
                    per = [percell_reps(d, b, pol) for d in heads[b]]
                    cs = [V[("TEST", d, b, {"GA-R1": "direct", "V9": "gemini"}[pol])] for d in heads[b]]
                    if any(p_ is None for p_ in per) or any(len(p_) != n_rep for p_ in per):
                        mism.append(f"summary {setting} {c_} {b}: PerCell repeats missing")
                    else:
                        for p_, c in zip(per, cs):
                            if any(abs(a - bb) > 1e-6 * max(1.0, abs(a)) for a, bb in zip(p_, c["reps"])):
                                mism.append(f"summary {setting} {c_} {b}: PerCell gains_by_repeat differ from cells.json {c['keys']}")
                        by_rep = [mean(p_[i] for p_ in per) for i in range(n_rep)]
                        cell["reps"], cell["sd"] = by_rep, sd(by_rep)
                        cell["src"] += f"; sd from Paper_Tables_PerCell.csv gains_by_repeat (row {pol}, headroom designs)"
                S["gain"][(c_, b)] = cell
        # costs
        crow = {"exh": "ref3" if setting == "2worker" else "ref2p", "direct": "GA-R1", "orch": "V9", "rule": "v2", "fixed": "fixed", "bandit": "bandit", "grid": "grid"}
        cfile = "Paper_Tables_Table1_cost_per_task.csv" if setting == "2worker" else "Paper_Tables_Table2_cost_per_task.csv"
        for c_ in cols:
            r = cost[ctab][crow[c_]]
            jh = None if pd.isna(r["job_hours_per_task"]) else float(r["job_hours_per_task"])
            S["cost"][(c_, "jobh")] = {"state": "ok" if jh is not None else "na", "value": jh, "tasks": int(r["tasks"]),
                                       "src": f"{IN}{cfile} | key={crow[c_]} | job_hours_per_task (tasks={int(r['tasks'])})"}
            tk = float(r["llm_tokens_per_task"]) if not pd.isna(r["llm_tokens_per_task"]) else None
            S["cost"][(c_, "tokens")] = {"state": "ok" if c_ in ("direct", "orch") else "dash", "value": tk if c_ in ("direct", "orch") else None, "tasks": int(r["tasks"]),
                                         "src": f"{IN}{cfile} | key={crow[c_]} | llm_tokens_per_task (tasks={int(r['tasks'])})"}
        rg = cost[ctab]["V9 (ChatGPT)"]
        S["cost_gpt"] = {"jobh": {"value": float(rg["job_hours_per_task"]), "tasks": int(rg["tasks"])}, "tokens": {"value": float(rg["llm_tokens_per_task"]), "tasks": int(rg["tasks"])}}
        summ[setting] = S

    # best-known counts of the exhaustive column
    n_known = {}
    for setting, bs in SETTING_BRIEFS.items():
        tasks = [("TEST", d, b) for d in DESIGNS["TEST"] for b in bs if TASK[("TEST", d, b)]["head"]]
        n_known[setting] = (sum(1 for (g, d, b) in tasks if abs(X[(g, d, b)]["val"] - TASK[(g, d, b)]["ref_known"]) < 0.51 or X[(g, d, b)]["val"] >= TASK[(g, d, b)]["ref_known"]), len(tasks))
        if setting == "3worker":
            beaten = [(d, b) for (g, d, b) in tasks if X[(g, d, b)]["val"] < TASK[(g, d, b)]["ref_known"] - 0.51]
            n_known["beaten_3worker"] = beaten

    # ------------------------------------------------------------------------------------------------------------- model choice (computed, not shown)
    def pooled_pct(setting, model):
        bs = SETTING_BRIEFS[setting]
        tasks = [(d, b) for d in DESIGNS["TEST"] for b in bs if TASK[("TEST", d, b)]["head"]]
        if not all(V.get(("TEST", d, b, model)) for d, b in tasks):
            return None
        return mean(100.0 * V[("TEST", d, b, model)]["mean"] / TASK[("TEST", d, b)]["ref_known"] for d, b in tasks)

    pct = {s: {m: pooled_pct(s, m) for m in ("gemini", "gpt", "qwen")} for s in SETTING_BRIEFS}
    best_model = {s: max((m for m in pct[s] if pct[s][m] is not None), key=lambda m: pct[s][m]) for s in pct}
    use_model = {s: ("gemini" if CFG.LLM_MODEL == "gemini" else best_model[s]) for s in pct}

    # orchestrator column of the selected model in every group
    for (g, d, b, p), c in list(V.items()):
        pass
    for g in GROUPS:
        for d in DESIGNS[g]:
            for b in BRIEFS:
                m = use_model[SETTING_OF[b]]
                if (g, d, b, m) in V:
                    V[(g, d, b, "orch")] = dict(V[(g, d, b, m)], model=m)

    notrun_pairs = sorted({(r["data_scope"], r["design"], r["brief"], r["row"]) for _, r in fill_notrun.iterrows()})
    tier2_done = bool(len(fill_exh) > 0)
    return {"log": log, "mism": mism, "start": start, "order": order, "TASK": TASK, "V": V, "X": X, "summ": summ, "raises": raises, "pct": pct, "best_model": best_model,
            "use_model": use_model, "n_known": n_known, "notrun_fill": notrun_pairs, "tier2_rows": int(len(fill_exh)), "tier2_done": tier2_done, "fill_eps": fe, "pf_eps": pf_eps}


if __name__ == "__main__":
    D = build()
    print("\n".join(D["log"]))
    print("mismatches:", len(D["mism"]))
    for m in D["mism"]:
        print("  -", m)
    print("pooled % of best-known gain, TEST:", {s: {m: (round(v, 2) if v is not None else None) for m, v in d.items()} for s, d in D["pct"].items()})
    print("best model:", D["best_model"], "used:", D["use_model"])
    print("exhaustive best-known counts:", D["n_known"])
    print("raises:", D["raises"][:5])
