#!/usr/bin/env python3
"""Compute the numbers that the paper text states (not in a table) from the generated tables and the cost files.

Run from the release root after `python paper_figures/final/final_tables.py`:
    python paper_figures/paper_claims.py [<folder with the generated values files>]

Each line shows the claim, the value stated in the text, the value computed here and whether they agree at the stated precision.
The claims that cannot be computed from the shipped inputs are listed at the end.
"""
import csv
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "paper_figures/out/final"
INP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs")


def rows(name):
    with open(os.path.join(OUT, name), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


B2 = [r for r in rows("values_tab_breakdown_2worker_all.csv") if r["line"] == "main"]
B3 = [r for r in rows("values_tab_breakdown_3worker_all.csv") if r["line"] == "main"]
S2 = [r for r in rows("values_tab_summary_2worker_TEST.csv") if r["line"] == "main"]
S3 = [r for r in rows("values_tab_summary_3worker_TEST.csv") if r["line"] == "main"]


def cell(tab, design, col, brief="", group=None):
    for r in tab:
        if r["row_key"] == design and r["col_key"] == col and r["brief"] == brief and (group is None or r["group"] == group):
            return r
    raise KeyError((design, col, brief))


def val(tab, design, col, brief=""):
    return float(cell(tab, design, col, brief)["value"])


def short(d):
    return {"fir_systolic_transposed": "fir_systolic_transposed", "corescore": "corescore_500_mod", "digit-recognition": "rosetta_digit-recognition"}.get(d, d)


def names(tab):
    out = []
    for r in tab:
        if r["col_key"] == "start_wns" and r["row_key"] != "total":
            out.append((r["group"], r["row_key"]))
    return out


def design_key(tab, short_name):
    for g, d in names(tab):
        if d.startswith(short_name) or short_name in d:
            return d
    raise KeyError(short_name)


TEST2 = [d for g, d in names(B2) if g == "TEST"]
claims = []


def claim(label, stated, computed, nd=0, unit="", below=False):
    if below:
        ok = computed < stated
    else:
        ok = round(computed, nd) == round(stated, nd) if nd else int(round(computed)) == int(round(stated))
    claims.append((label, stated, computed, ok, unit))


# --- Introduction: gains on the TEST designs where the exhaustive search has headroom (two workers)
def head(tab, d, brief):
    return cell(tab, d, "exh", brief)["italic"] == "0"


for pol, w_stated, l_stated in (("orch", 22, 5), ("direct", 19, 4)):
    ws = [100.0 * val(B2, d, pol, "T0") / abs(val(B2, d, "start_wns")) for d in TEST2 if head(B2, d, "T0")]
    ls = [100.0 * val(B2, d, pol, "A0") / val(B2, d, "start_luts") for d in TEST2 if head(B2, d, "A0")]
    nm = "LLM Orchestration" if pol == "orch" else "Direct LLM Control"
    claim(f"{nm}: mean share of the initial WNS recovered, TEST designs with headroom ({len(ws)} designs)", w_stated, sum(ws) / len(ws), 0, "%")
    claim(f"{nm}: mean share of the starting LUTs removed, TEST designs with headroom ({len(ls)} designs)", l_stated, sum(ls) / len(ls), 0, "%")


def srow(tab, rk, col):
    return float(cell(tab, rk, col, rk[5:] if rk.startswith("gain_") else "")["value"])


claim("LLM Orchestration: Vivado job-h per task, two workers (stated 'within 1.2')", 1.2, srow(S2, "jobh", "orch"), 1, "job-h")
claim("tokens per task, LLM Orchestration, two workers", 10941, srow(S2, "tokens", "orch"), 0, "tokens")
claim("tokens per task, Direct LLM Control, two workers", 17083, srow(S2, "tokens", "direct"), 0, "tokens")
claim("tokens per task, LLM Orchestration, three workers", 8992, srow(S3, "tokens", "orch"), 0, "tokens")
claim("tokens per task, Direct LLM Control, three workers", 19717, srow(S3, "tokens", "direct"), 0, "tokens")
claim("LLM Orchestration token change with the power worker", -18, 100.0 * (srow(S3, "tokens", "orch") / srow(S2, "tokens", "orch") - 1), 0, "%")
claim("Direct LLM Control token change with the power worker", 15, 100.0 * (srow(S3, "tokens", "direct") / srow(S2, "tokens", "direct") - 1), 0, "%")
claim("Direct LLM Control tokens relative to LLM Orchestration, two workers", 56, 100.0 * (srow(S2, "tokens", "direct") / srow(S2, "tokens", "orch") - 1), 0, "%")
claim("Direct minus LLM Orchestration, WNS gain (ps, two workers)", 25, srow(S2, "gain_T0", "orch") - srow(S2, "gain_T0", "direct"), 0, "ps")
claim("Direct minus LLM Orchestration, LUTs removed (two workers)", 75, srow(S2, "gain_A0", "orch") - srow(S2, "gain_A0", "direct"), 0, "LUTs")
six = [srow(S2, "jobh", c) for c in ("grid", "bandit", "fixed", "rule", "orch", "direct")]
claim("six-run policies' job-h as a share of Exhaustive Search's (stated 'under 1%'; highest)", 1, 100.0 * max(six) / srow(S2, "jobh", "exh"), 0, "%")
claim("lowest six-run job-h per task, two workers", 1.06, min(six), 2, "job-h")
claim("highest six-run job-h per task, two workers", 1.29, max(six), 2, "job-h")
six3 = [srow(S3, "jobh", c) for c in ("grid", "fixed", "rule", "orch", "direct")]
# Exhaustive Search, three workers: no per-run record exists for the 14-recipe search; make_tables.py scales the measured ten-recipe
# 2-step search (82 runs, 17.97 job-h per design over the TEST designs): x (14*15)/(10*11). The two measured constants are not shipped as a file.
EXH3_RUNS, EXH3_JOBH = 82 * (14 * 15) / (10 * 11), 17.97 * (14 * 15) / (10 * 11)
claim("Exhaustive Search, three workers: runs per design (estimate, 82 x 210/110)", 157, EXH3_RUNS, 0, "runs")
claim("Exhaustive Search, three workers: job-h per design (estimate, 17.97 x 210/110)", 34.3, EXH3_JOBH, 1, "job-h")
claim("Exhaustive Search job-h over the highest six-run job-h, three workers (stated 'over 30 times')", 30, int(EXH3_JOBH / max(six3)), 0, "x")
claim("Rule-Based Orchestration: job-h per task, three workers", 0.98, srow(S3, "jobh", "rule"), 2, "job-h")

# --- Results, two workers: comparison with Rule-Based Orchestration
n_head = n_close = 0
gaps = []
for g, d in names(B2):
    for brief in ("T0", "A0"):
        if cell(B2, d, "exh", brief, g)["italic"] != "0":
            continue
        n_head += 1
        a, b = float(cell(B2, d, "orch", brief, g)["value"]), float(cell(B2, d, "rule", brief, g)["value"])
        n_close += int(abs(round(a) - round(b)) <= 13)       # compared as displayed in Tables IV (integers)
        gaps.append((abs(a - b), d, brief, a, b))
claim("design and brief combinations where Exhaustive Search gains beyond run-to-run variation (two workers)", 22, n_head, 0)
claim("of which LLM Orchestration is within 13 ps or 13 LUTs of Rule-Based Orchestration (displayed integers)", 16, n_close, 0)
d_fir, d_core, d_dig = design_key(B2, "fir_systolic_transposed"), design_key(B2, "corescore"), design_key(B2, "digit-recognition")
claim("fir transposed timing-first, LLM Orchestration (ps)", 103, val(B2, d_fir, "orch", "T0"), 0, "ps")
claim("fir transposed timing-first, Rule-Based (ps)", 103, val(B2, d_fir, "rule", "T0"), 0, "ps")
claim("corescore area-first, LLM Orchestration (LUTs)", 543, val(B2, d_core, "orch", "A0"), 0, "LUTs")
claim("corescore area-first, Rule-Based (LUTs)", 544, val(B2, d_core, "rule", "A0"), 0, "LUTs")
claim("fir transposed: LUTs removed as a share of its 603 starting LUTs", 17, 100.0 * val(B2, d_fir, "exh", "A0") / val(B2, d_fir, "start_luts"), 0, "%")
claim("corescore: 742 LUTs as a share of its starting LUTs (stated 'under 1%')", 1, 100.0 * val(B2, d_core, "exh", "A0") / val(B2, d_core, "start_luts"), 0, "%", below=True)
claim("digit-recognition: WNS after the 561 ps of Exhaustive Search (ps)", -534, val(B2, d_dig, "start_wns") + val(B2, d_dig, "exh", "T0"), 0, "ps")
d3 = design_key(B2, "3d-rendering")
claim("3d-rendering timing-first, LLM Orchestration (ps)", 406, val(B2, d3, "orch", "T0"), 0, "ps")
claim("3d-rendering timing-first, Rule-Based (ps)", 84, val(B2, d3, "rule", "T0"), 0, "ps")
claim("3d-rendering timing-first, Direct LLM Control (ps)", 34, val(B2, d3, "direct", "T0"), 0, "ps")
claim("3d-rendering area-first, Direct LLM Control (LUTs)", 526, val(B2, d3, "direct", "A0"), 0, "LUTs")
claim("3d-rendering area-first, LLM Orchestration (LUTs)", 151, val(B2, d3, "orch", "A0"), 0, "LUTs")
for col, st in (("orch", 767), ("fixed", 687), ("rule", 688), ("direct", 610)):
    claim(f"digit-recognition area-first, {col} (LUTs)", st, val(B2, d_dig, col, "A0"), 0, "LUTs")
sp, cd = design_key(B2, "spam-filter"), design_key(B2, "cordic")
claim("Direct LLM Control, spam-filter timing-first (ps)", 0, val(B2, sp, "direct", "T0"), 0, "ps")
claim("Direct LLM Control, spam-filter area-first (LUTs)", 0, val(B2, sp, "direct", "A0"), 0, "LUTs")
claim("Direct LLM Control, cordic timing-first (ps)", 0, val(B2, cd, "direct", "T0"), 0, "ps")

# --- Results, three workers
claim("Fixed Worker Order timing-first, two workers (ps)", 178, srow(S2, "gain_T0", "fixed"), 0, "ps")
claim("Fixed Worker Order timing-first, three workers (ps)", 135, srow(S3, "gain_W2R", "fixed"), 0, "ps")
D3 = {k: design_key(B3, k) for k in ("digit-recognition", "corescore", "fir_systolic_transposed", "3d-rendering", "mcml", "cordic", "finn")}
claim("digit-recognition area-first: Fixed Worker Order and Exhaustive Search (LUTs)", 670, val(B3, D3["digit-recognition"], "fixed", "L2R"), 0, "LUTs")
claim("digit-recognition area-first: Direct LLM Control (LUTs)", 599, val(B3, D3["digit-recognition"], "direct", "L2R"), 0, "LUTs")
claim("digit-recognition area-first: Rule-Based (LUTs)", 36, val(B3, D3["digit-recognition"], "rule", "L2R"), 0, "LUTs")
claim("corescore area-first: Fixed Worker Order (LUTs)", 16, val(B3, D3["corescore"], "fixed", "L2R"), 0, "LUTs")
claim("corescore area-first: Rule-Based (LUTs)", 544, val(B3, D3["corescore"], "rule", "L2R"), 0, "LUTs")
claim("corescore area-first: LLM Orchestration (LUTs)", 561, val(B3, D3["corescore"], "orch", "L2R"), 0, "LUTs")
claim("finn area-first: LLM Orchestration (LUTs)", 4851, val(B3, D3["finn"], "orch", "L2R"), 0, "LUTs")
claim("corescore power-first: Rule-Based (mW)", 124.0, val(B3, D3["corescore"], "rule", "P0"), 1, "mW")
claim("3d-rendering area-first: Rule-Based (LUTs)", 0, val(B3, D3["3d-rendering"], "rule", "L2R"), 0, "LUTs")
claim("3d-rendering area-first: Exhaustive Search (LUTs)", 453, val(B3, D3["3d-rendering"], "exh", "L2R"), 0, "LUTs")
claim("mcml power-first: Rule-Based (mW)", 0.0, val(B3, D3["mcml"], "rule", "P0"), 1, "mW")
claim("mcml power-first: Exhaustive Search (mW)", 1280.0, val(B3, D3["mcml"], "exh", "P0"), 1, "mW")
claim("cordic timing-first: Rule-Based (ps)", 223, val(B3, D3["cordic"], "rule", "W2R"), 0, "ps")
claim("3d-rendering timing-first: Rule-Based (ps)", 17, val(B3, D3["3d-rendering"], "rule", "W2R"), 0, "ps")
claim("3d-rendering timing-first: Fixed Worker Order (ps)", 324, val(B3, D3["3d-rendering"], "fixed", "W2R"), 0, "ps")
n_best = sum(1 for g, d in names(B3) if cell(B3, d, "orch", "L2R", g)["bold"] == "1")
claim("designs on which LLM Orchestration has the best area-first result (three workers; stated '6 of the 13')", 6, n_best, 0)

w = max(len(c[0]) for c in claims)
bad = 0
print(f"{'claim':{w}s}  stated   computed  agree")
for label, stated, computed, ok, unit in claims:
    bad += 0 if ok else 1
    print(f"{label:{w}s}  {stated:>7}  {computed:>8.2f}  {'yes' if ok else 'NO'}  {unit}")
print()
print(f"{len(claims)} claims computed, {bad} disagree with the stated value")
print()
print("LLM Orchestration minus Rule-Based, the 22 combinations ordered by absolute gap (ps or LUTs):")
print("  " + "; ".join(f"{g:.1f} ({d[:14]} {b})" for g, d, b, a_, c_ in sorted(gaps)[::-1][:9]))
print("  combinations within 13: %d; within 14: %d; within 15: %d; within 20: %d" % tuple(sum(1 for g in gaps if g[0] <= t + 1e-9) for t in (13, 14, 15, 20)))
print()
print("Not computable from the shipped inputs:")
print("  - 'in 10 of the 16 exhaustive search runs on test designs a limit is broken on purpose before a later worker repairs it': the shipped inputs hold the recipe ids of each")
print("    exhaustive path (Paper_Tables_Cells.csv, reference_path) but not the metrics of its intermediate states, which live in the stored search records.")
sys.exit(1 if bad else 0)
