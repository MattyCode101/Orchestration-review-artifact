"""X2 parts 1 and 3 (Mon 16 Nov): engineering units and balance across briefs, TEST. NEW file, offline, read-only. No tool run, no model call.

Reads data/paper_tables_2026-11-15/{cells,episodes,refs,graph4}.json and the recomputed-ceiling csv of the tables note. Writes data/paper_tables_2026-11-15/eng/eng_test.json.

Definitions
  cell value  = mean over the repeats (seeds) of that design-and-brief cell; deterministic rows have one repeat.
  head        = the cells with headroom (reference gain at least one noise band, as Tables 1 to 4); all = the five designs of the brief (a row with no episode on a design is left out).
  goal gain   = WNS in ps (T0, T, W2R), packed LUTs removed (A0, A, L2R), dynamic power saved in mW (P0); "% of start" = gain / the design's start value, averaged over the designs.
  protected   = the change final minus start of the metric a brief protects: packed-LUT change in % of start, WNS change in ps, dynamic-power change in mW. Median and range (min to max)
                are over the cell values of the designs (all five, and the headroom ones separately).
"""
from __future__ import annotations
import os

import csv
import json
import statistics
import sys
from pathlib import Path

PROJECT = Path(os.environ.get("ORCH_PROJECT_ROOT") or Path(__file__).resolve().parents[2])
OUT = PROJECT / "data/paper_tables_2026-11-15"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_pt_lib as L  # noqa: E402

C = L.jread(OUT / "cells.json")
CELLS, ROWS, STARTS = C["cells"], C["rows"], C["starts"]
EPS = {e["eid"]: e for e in L.jread(OUT / "episodes.json")}
REFS = L.jread(OUT / "refs.json")["cells"]
G4 = L.jread(OUT / "graph4.json")["designs"]
CEIL = {}
with open(OUT / "stored_inputs/Paper_Tables_Cells.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        CEIL[f"{r['design']}|{r['brief']}"] = r

BRIEFS = ("T0", "A0", "T", "A", "W2R", "L2R", "P0")
GOALM = {b: L.GOAL[b][0] for b in BRIEFS}
UNIT = {"wns_ns": "ps", "packed_luts": "LUTs", "dynamic_power_w": "mW"}
# protected metrics per brief (what the brief's limits hold): LUT % on T0, T, W2R; WNS ps on A0, A, L2R, P0; power mW on T0, A0, W2R, L2R (also computed for the others)
PROT = {"T0": ["packed_luts", "dynamic_power_w"], "T": ["packed_luts", "dynamic_power_w"], "W2R": ["packed_luts", "dynamic_power_w"],
        "A0": ["wns_ns", "dynamic_power_w"], "A": ["wns_ns", "dynamic_power_w"], "L2R": ["wns_ns", "dynamic_power_w"], "P0": ["wns_ns", "packed_luts"]}
T1 = ["ref3", "ceiling", "ref2", "fixed", "grid", "beam", "v1", "v2", "bandit", "random", "V9", "F1-R1", "v2.1", "V3", "V6", "GA-R1", "V9 (ChatGPT)"]
T2 = ["ref2p", "ceiling", "fixed", "grid", "beam", "v1", "v2", "bandit", "random", "V9", "F1-R1", "v2.1", "GA-R1", "V9 (ChatGPT)"]
T3 = [("V0", "F1-R1"), ("V1", "v2.1"), ("V2", "V2"), ("V3", "V3"), ("V4", "V4"), ("V5", "V5"), ("V6", "V6"), ("V7", "V7"), ("V8", "V8"), ("V9", "V9"), ("V10", "V10"), ("V11", "V11"), ("ChatGPT-V9", "V9 (ChatGPT)")]


def to_unit(metric: str, x: float) -> float:
    return x * 1000.0 if metric in ("wns_ns", "dynamic_power_w") else x


def cells_of(brief: str) -> list[str]:
    return sorted((k for k, c in CELLS.items() if c["brief"] == brief), key=lambda k: k)


def ref_state_metrics(key: str, which: str):
    """Metrics of the reference state (3-step, 2-step) of a timing or area cell from the stored graph; None for power briefs."""
    c = REFS.get(key)
    if not c:
        return None
    r = c.get(which)
    d = key.split("|")[0]
    if r is None:                      # no gain at all: the reference is the start state
        return STARTS[d]
    return G4[d]["nodes"][r["state"]]["m"]


def cell_record(label: str, key: str) -> dict | None:
    """One cell of a row: mean goal gain (native units), per-metric mean change final-start, and the number of repeats."""
    d, b = key.split("|")
    c = CELLS[key]
    st = STARTS[d]
    gm = GOALM[b]
    if label in ("ref3", "ref2"):
        r = REFS.get(key)
        if r is None:
            return None
        gain = (r.get(label) or {}).get("gain", 0.0)
        m = ref_state_metrics(key, label)
        return {"gain": gain, "n": 1, "chg": {mm: m[mm] - st[mm] for mm in ("packed_luts", "wns_ns", "dynamic_power_w")}}
    if label == "ref2p":
        ents = ROWS.get("ref2p", {}).get(key)
        if not ents:
            return None
        return {"gain": ents[0]["gain"], "n": 1, "chg": None}
    if label == "ceiling":
        r = CEIL.get(key)
        if r is None:
            return None
        v = r["ceiling_recomputed_gain"] if r.get("ceiling_recomputed_gain") not in ("", None) else r.get("ceiling_section7_gain")
        v = float(v) if v not in ("", None) else None
        if v is None:
            return None
        src = "recomputed" if r.get("ceiling_recomputed_gain") not in ("", None) else "section7"
        return {"gain": v, "n": 1, "chg": None, "ceiling_src": src}
    ents = ROWS.get(label, {}).get(key)
    if not ents:
        return None
    gains = [x["gain"] for x in ents]
    eps = [EPS[x["eid"]] for x in ents if x["eid"] in EPS]
    chg = None
    if len(eps) == len(ents):
        chg = {mm: L.mean(e["final"][mm] - e["start"][mm] for e in eps) for mm in ("packed_luts", "wns_ns", "dynamic_power_w")}
    return {"gain": L.mean(gains), "n": len(ents), "chg": chg}


def summarise(label: str, brief: str) -> dict | None:
    gm = GOALM[brief]
    recs = {}
    for k in cells_of(brief):
        r = cell_record(label, k)
        if r is not None:
            recs[k] = r
    if not recs:
        return None
    head = [k for k in recs if CELLS[k]["head"]]
    out = {"brief": brief, "goal_metric": gm, "unit": UNIT[gm], "cells_all": len(recs), "cells_head": len(head), "reps": sorted({r["n"] for r in recs.values()})}

    def gstats(keys):
        if not keys:
            return {"gain": None, "pct": None}
        g = [to_unit(gm, recs[k]["gain"]) for k in keys]
        pct = [100.0 * recs[k]["gain"] / STARTS[k.split("|")[0]][gm] for k in keys] if gm != "wns_ns" else None
        return {"gain": sum(g) / len(g), "pct": (sum(pct) / len(pct) if pct else None), "per_cell": {k.split("|")[0]: to_unit(gm, recs[k]["gain"]) for k in keys}}

    out["head"], out["all"] = gstats(head), gstats(list(recs))
    out["ceiling_src"] = sorted({r.get("ceiling_src") for r in recs.values() if r.get("ceiling_src")})
    prot = {}
    if all(r["chg"] is not None for r in recs.values()):
        for pm in ("packed_luts", "wns_ns", "dynamic_power_w"):
            def vals(keys):
                v = []
                for k in keys:
                    ch = recs[k]["chg"][pm]
                    if pm == "packed_luts":
                        ch = 100.0 * ch / STARTS[k.split("|")[0]][pm]
                    else:
                        ch = ch * 1000.0
                    v.append(ch)
                return v

            for tag, keys in (("all", list(recs)), ("head", head)):
                v = vals(keys)
                prot.setdefault(pm, {})[tag] = {"median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v)} if v else None
        out["protected"] = prot
    else:
        out["protected"] = None
    return out


def share_cells(label: str, brief_set, ref: str = "ref") -> dict:
    """Share per headroom cell (mean over repeats of gain over the cell's reference), for the balance table."""
    res = {}
    for b in brief_set:
        for k in cells_of(b):
            c = CELLS[k]
            if not c["head"]:
                continue
            ents = ROWS.get(label, {}).get(k)
            if ents:
                res[k] = 100.0 * (sum(x["gain"] for x in ents) / len(ents)) / c[ref]
    return res


def main() -> int:
    out = {"schema": "tr-eng-test-v1", "starts": {L.SHORT[d]: STARTS[d] for d in L.DESIGNS}, "tables": {}, "balance": {}, "headroom": {}}
    for b in BRIEFS:
        out["headroom"][b] = [L.SHORT[k.split("|")[0]] for k in cells_of(b) if CELLS[k]["head"]]
    for name, labels in (("1a", T1), ("2a", T2)):
        out["tables"][name] = {}
        for lab in labels:
            brs = ("T0", "A0") if name == "1a" else ("W2R", "L2R", "P0")
            out["tables"][name][lab] = {b: summarise(lab, b) for b in brs}
    out["tables"]["1b"] = {lab: {b: summarise(lab, b) for b in ("T", "A")} for lab in T1}
    out["tables"]["3"] = {nm: {b: summarise(lab, b) for b in ("T0", "A0")} for nm, lab in T3}
    # balance across briefs (shares of the 3-step or 2-step reference, the Table 1a and 2a definition)
    bal_rows = ["fixed", "grid", "beam", "v1", "v2", "bandit", "random", "V9", "F1-R1", "v2.1", "GA-R1", "V9 (ChatGPT)"]
    for lab in bal_rows:
        per = {}
        for b in ("T0", "A0", "W2R", "L2R", "P0"):
            sh = share_cells(lab, (b,))
            per[b] = (sum(sh.values()) / len(sh)) if sh else None
        have = {b: v for b, v in per.items() if v is not None}
        out["balance"][lab] = {"per_brief": per, "n_briefs": len(have), "mean": sum(have.values()) / len(have) if have else None, "min": min(have.values()) if have else None,
                               "argmin": min(have, key=have.get) if have else None, "max": max(have.values()) if have else None, "spread": (max(have.values()) - min(have.values())) if have else None,
                               "sd": statistics.stdev(have.values()) if len(have) > 1 else None}
    (OUT / "eng").mkdir(exist_ok=True)
    (OUT / "eng/eng_test.json").write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    print("wrote", OUT / "eng/eng_test.json")
    for lab in ("V9", "fixed", "ref3", "ceiling", "bandit"):
        s = out["tables"]["1a"][lab]
        print(lab, {b: (round(v["head"]["gain"], 1), v["cells_head"], v["cells_all"]) for b, v in s.items() if v})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
