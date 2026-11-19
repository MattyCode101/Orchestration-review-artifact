"""X2 part 2, step 2 (local, offline): shares of the reference per policy, design group and brief, one definition throughout.

Definition: share of a cell = mean over the row's repeats of its goal gain, divided by the cell's REFERENCE = the best brief-valid state with an AMD PASS known for the cell:
  - every stored episode result of the cell with final_brief_valid and validation_status pass (all rows, repeats and prompt variants of every source), and
  - VALIDATION: the exhaustive two-step AMD-checked pick; TEST: the Table 1a/2a reference (3-step, or 2-step over 14 recipes) and the recomputed strict ceiling.
Headroom cells only (reference at least one noise band; power in whole mW), cells weighted equally, pooled = mean over the cells of the briefs named.
TEST is given twice: BEST KNOWN (this definition) and REF (the 3-step or 2-step reference of Tables 1a and 2a).
"""
from __future__ import annotations

import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

D = Path(__file__).parent / "din"
G = json.load(open(D / "gen_collect.json"))
CJ = json.load(open(D / "cells.json"))
EPS = json.load(open(D / "episodes.json"))
CELLS, ROWS = CJ["cells"], CJ["rows"]
CEILCSV = {}
with open(sys.argv[1] if len(sys.argv) > 1 else "Paper_Tables_Cells.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        CEILCSV[f"{r['design']}|{r['brief']}"] = r

TRAIN = ("vexriscv_re-place", "logicnets_jscl", "rosetta_spam-filter", "rosetta_3d-rendering", "finn_radioml")
VAL = ("t2_cordic_s", "vtr_mcml_v2", "amd_mini-isp")
TEST = tuple(sorted({k.split("|")[0] for k in CELLS}))
GOALM = {"T0": "wns_ns", "T": "wns_ns", "W2R": "wns_ns", "A0": "packed_luts", "A": "packed_luts", "L2R": "packed_luts", "P0": "dynamic_power_w"}
MAIN, POWER = ("T0", "A0"), ("W2R", "L2R", "P0")
R2SRC = {"d6v2", "v22grid", "v22t3", "power_review", "dev_baselines", "post_freeze"}
rec_by_cell = defaultdict(list)
for r in G["records"]:
    rec_by_cell[(r["design"], r["brief"])].append(r)


def mw(x):
    return round(x * 1000.0)


def band_of(d, b):
    v = G["bands"].get(f"{d}|{GOALM[b]}")
    return v[0] if v else None


def ratio(b, ref, band):
    return (mw(ref) / mw(band)) if GOALM[b] == "dynamic_power_w" else ref / band


# ---------------------------------------------------------------------------------------------------------------- references
def ref_train_val(d, b, flavor="all"):
    """flavor 'all' = the reference of this note: every AMD-validated episode final of the cell, plus the stored AMD-PASS states (TRAINING: the stored best-known state of T0 and A0 and the
    six-run power ceiling state, both with a PASS record against the design's B2 reference in the ledgers of the worker-fitting sweep, the cross-worker depth-3 job and the power-readiness
    study; VALIDATION: the exhaustive two-step AMD-checked pick), plus the bandit's own final state where it exists (its states passed the strict AMD check in the bandit study).
    flavor 'obs' = episode finals (and the VALIDATION exhaustive pick) only, without the stored TRAINING states (a sensitivity check)."""
    cands = [r["gain"] for r in rec_by_cell[(d, b)]]
    k = f"{d}|{b}"
    ve = G["val_exhaustive"].get(k)
    if d in VAL and ve:
        cands.append(ve["gain"])
    if flavor == "all":
        if d in TRAIN:
            sb = G["stored_best_known_train"].get(k)
            if sb and b in ("T0", "A0"):
                cands.append(sb["gain"])
            pc = G["train_power_ceiling"].get(k)
            if pc and b in POWER:
                cands.append(pc["ceiling"] * (0.001 if pc["unit"] in ("mW", "ps") else 1.0))
        bt = (G["bandit_train"] if d in TRAIN else G["bandit_val"]).get(k)
        if bt:
            cands.append(bt["gain"])
    return max(cands + [0.0])


def test_cell_gains(d, b):
    return [e["gain"] for e in EPS if e["design"] == d and e["brief"] == b and e.get("valid") and e.get("validation") == "pass"]


def ref_test_known(d, b):
    k = f"{d}|{b}"
    base = CELLS[k]["ref"]
    ce = CEILCSV.get(k, {})
    cg = ce.get("ceiling_recomputed_gain")
    cands = [base] + test_cell_gains(d, b)
    if cg not in (None, ""):
        cands.append(float(cg))
    return max(cands)


# ---------------------------------------------------------------------------------------------------------------- row gains per cell
def row_cells(group, row, brief):
    """{design: [gains over repeats]} of a row for a group and brief."""
    out = defaultdict(list)
    if group == "TEST":
        lab = {"GA": "GA-R1"}.get(row, row)
        for k, c in CELLS.items():
            if c["brief"] != brief:
                continue
            for x in ROWS.get(lab, {}).get(k, []):
                out[k.split("|")[0]].append(x["gain"])
        return out
    designs = TRAIN if group == "TRAINING" else VAL
    if row == "bandit":
        src = G["bandit_train"] if group == "TRAINING" else G["bandit_val"]
        for d in designs:
            e = src.get(f"{d}|{brief}")
            if e is not None:
                out[d].append(e["gain"])
        return out
    allowed = {("fixed", "main"): "d6v2", ("v2", "main"): "d6v2", ("F1-R1", "main"): "d6v2", ("v2.1", "main"): "d6v2", ("random", "main"): "d6v2", ("V9", "main"): "v22t3", ("GA", "main"): "ga3"}
    kind = "main" if brief in MAIN else "power"
    for d in designs:
        for r in rec_by_cell[(d, brief)]:
            if kind == "main":
                if r["row"] == {"GA": "GA-trainingstage"}.get(row, row) and r["src"] == allowed.get((row, "main")):
                    out[d].append(r["gain"])
            else:
                if row in ("fixed", "v2", "F1-R1", "v2.1") and r["row"] == row and r["src"] == "power_review":
                    out[d].append(r["gain"])
                elif row == "random" and r["row"] == "random" and r["src"] in ("dev_baselines", "p0_pilot"):
                    out[d].append(r["gain"])
                elif row == "V9" and r["row"] == "V9" and r["src"] == "post_freeze":
                    out[d].append(r["gain"])
    if kind == "power" and row == "V9" and not v9_power_complete(group):
        return defaultdict(list)   # post-freeze V9 power rows count only once every design of the group has an episode on all three power briefs
    return out


def v9_power_complete(group):
    designs = TRAIN if group == "TRAINING" else VAL
    for b in POWER:
        have = {r["design"] for d in designs for r in rec_by_cell[(d, b)] if r["row"] == "V9" and r["src"] == "post_freeze"}
        if len(have) < len(designs):
            return False
    return True


def headroom_cells(group, briefs, version):
    """List of (design, brief, ref, band) with reference at least one band."""
    out = []
    designs = TRAIN if group == "TRAINING" else (VAL if group == "VALIDATION" else TEST)
    for b in briefs:
        for d in designs:
            if group == "TEST":
                k = f"{d}|{b}"
                if k not in CELLS:
                    continue
                ref = ref_test_known(d, b) if version == "known" else CELLS[k]["ref"]
                band = CELLS[k]["band"]
            else:
                band = band_of(d, b)
                if band is None:
                    continue
                ref = ref_train_val(d, b, "all" if version == "known" else "obs")
            if ref > 0 and ratio(b, ref, band) >= 1.0:
                out.append((d, b, ref, band))
    return out


ROWSET = ["fixed", "bandit", "v2", "V9", "F1-R1", "v2.1", "GA", "random"]


def cell_shares(group, row, briefs, version):
    hc = headroom_cells(group, briefs, version)
    per = []
    for d, b, ref, band in hc:
        g = row_cells(group, row, b).get(d)
        if g:
            per.append({"design": d, "brief": b, "ref": ref, "band": band, "share": 100.0 * (sum(g) / len(g)) / ref, "reps": len(g), "gain": sum(g) / len(g)})
    return hc, per


def summarise(group, row, briefs, version):
    hc, per = cell_shares(group, row, briefs, version)
    if not per:
        return {"n_head": len(hc), "n": 0, "share": None, "reps": 0}
    return {"n_head": len(hc), "n": len(per), "share": sum(p["share"] for p in per) / len(per), "reps": max(p["reps"] for p in per), "cells": per}


def boot_diff(a, b, n=10000, seed=20261116):
    """95% interval of mean(a) - mean(b), cells of each group resampled independently."""
    rng = random.Random(seed)
    if not a or not b:
        return None
    ds = []
    for _ in range(n):
        ra = [a[rng.randrange(len(a))] for _ in a]
        rb = [b[rng.randrange(len(b))] for _ in b]
        ds.append(sum(ra) / len(ra) - sum(rb) / len(rb))
    ds.sort()
    return ds[int(0.025 * n)], ds[int(0.975 * n) - 1]


def main():
    res = {"rows": {}, "refs": {}}
    # --- check: the declared bandit against the stored best known / six-run ceiling reproduces the bandit report's TRAINING shares (67.2, 48.0, 50.4, 70.8, 30.4, 9.6) -----------------------------------------
    chk = {}
    for b in ("T0", "A0", "T", "A", "L2R", "P0"):
        sh = []
        for d in TRAIN:
            k = f"{d}|{b}"
            e = G["bandit_train"].get(k)
            if e is None:
                continue
            if b in ("T0", "A0", "T", "A"):
                ref = G["stored_best_known_train"][k]["gain"]
                band = band_of(d, b)
            else:
                dk = G["train_power_ceiling"].get(k)
                if dk is None:
                    continue
                unit = dk["unit"]
                ref = dk["ceiling"] * (0.001 if unit == "mW" else 1.0)
                band = dk["band"] * (0.001 if unit == "mW" else 1.0)
            if ref > 0 and (b in ("T0", "A0", "T", "A") or ref >= band):
                sh.append(100.0 * max(e["gain"], 0.0) / ref)
        chk[b] = (round(sum(sh) / len(sh), 1) if sh else None, len(sh))
    res["bandit_check"] = chk
    print("bandit TRAINING check vs the bandit report (67.2, 48.0, 50.4, 70.8, 30.4, 9.6):", chk)
    # --- main tables ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    groups = ("TRAINING", "VALIDATION", "TEST")
    for tag, briefs in (("T0+A0", MAIN), ("W2R+L2R+P0", POWER), ("T0", ("T0",)), ("A0", ("A0",)), ("W2R", ("W2R",)), ("L2R", ("L2R",)), ("P0", ("P0",))):
        res["rows"][tag] = {}
        for row in ROWSET:
            cur = {}
            for g in groups:
                if g == "TEST":
                    cur["TEST_known"] = summarise("TEST", row, briefs, "known")
                    cur["TEST_ref"] = summarise("TEST", row, briefs, "ref")
                else:
                    cur[g] = summarise(g, row, briefs, "known")
                    cur[g + "_obs"] = summarise(g, row, briefs, "obs")
            res["rows"][tag][row] = cur
    # --- drops with bootstrap ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    res["drops"] = {}
    for tag in ("T0+A0", "W2R+L2R+P0"):
        res["drops"][tag] = {}
        for row in ROWSET:
            c = res["rows"][tag][row]
            t = c["TEST_known"]
            for src in ("TRAINING", "VALIDATION"):
                s = c[src]
                if s["n"] and t["n"]:
                    # compare on the same footing: both groups' cell shares for this row
                    ci = boot_diff([p["share"] for p in s["cells"]], [p["share"] for p in t["cells"]])
                    res["drops"][tag].setdefault(row, {})[src] = {"drop": s["share"] - t["share"], "lo": ci[0], "hi": ci[1], "cells_from": s["n"], "cells_test": t["n"]}
    # --- references ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    for g, designs in (("TRAINING", TRAIN), ("VALIDATION", VAL)):
        for b in ("T0", "A0", "W2R", "L2R", "P0"):
            for d in designs:
                band = band_of(d, b)
                ra, rr = ref_train_val(d, b, "all"), ref_train_val(d, b, "obs")
                k = f"{d}|{b}"
                ent = {"group": g, "design": d, "brief": b, "band": band, "ref_known": ra, "ref_obs": rr, "n_candidates": len(rec_by_cell[(d, b)]), "head": ratio(b, ra, band) >= 1.0 if ra > 0 else False}
                if k in G["stored_best_known_train"] and b in ("T0", "A0"):
                    ent["stored_best_known_unchecked"] = G["stored_best_known_train"][k]["gain"]
                if k in G["train_power_ceiling"]:
                    dk = G["train_power_ceiling"][k]
                    ent["train_ceiling_unchecked"] = dk["ceiling"] * (0.001 if dk["unit"] == "mW" else 1.0)
                if g == "VALIDATION" and k in G["val_exhaustive"]:
                    ent["val_exhaustive_2step"] = G["val_exhaustive"][k]["gain"]
                res["refs"][k] = ent
    for b in ("T0", "A0", "W2R", "L2R", "P0"):
        for d in TEST:
            k = f"{d}|{b}"
            if k not in CELLS:
                continue
            res["refs"]["TEST:" + k] = {"group": "TEST", "design": d, "brief": b, "band": CELLS[k]["band"], "ref_table": CELLS[k]["ref"], "ref_known": ref_test_known(d, b), "head_table": CELLS[k]["head"],
                                         "head_known": ratio(b, ref_test_known(d, b), CELLS[k]["band"]) >= 1.0}
    Path("gen_result.json").write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8")
    # --- print the pooled tables -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    for tag in ("T0+A0", "W2R+L2R+P0"):
        print("\n##", tag)
        print(f"{'row':8s} {'TRAIN':>14s} {'VAL':>14s} {'TEST known':>14s} {'TEST ref':>14s}  drop T->Test  drop V->Test")
        for row in ROWSET:
            c = res["rows"][tag][row]

            def f(s):
                return "-" if s["n"] == 0 else f"{s['share']:5.1f} ({s['n']}/{s['n_head']},{s['reps']}r)"
            dr = res["drops"][tag].get(row, {})
            dt = dr.get("TRAINING"); dv = dr.get("VALIDATION")
            print(f"{row:8s} {f(c['TRAINING']):>14s} {f(c['VALIDATION']):>14s} {f(c['TEST_known']):>14s} {f(c['TEST_ref']):>14s}  " + (f"{dt['drop']:+6.1f} ({dt['lo']:+.1f},{dt['hi']:+.1f})" if dt else "-") + "  " + (f"{dv['drop']:+6.1f} ({dv['lo']:+.1f},{dv['hi']:+.1f})" if dv else "-"))
    print("\nreference differences (known vs r2only) cells:", sum(1 for e in res["refs"].values() if e["group"] != "TEST" and abs(e["ref_known"] - e["ref_obs"]) > 1e-12), "of", sum(1 for e in res["refs"].values() if e["group"] != "TEST"))


if __name__ == "__main__":
    main()
