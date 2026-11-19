"""Paper tables, final consolidation (Sun 15 Nov evening), step 1: the 'share of ceiling' columns recomputed against the strict ceiling of the mechanism note.

Input (read-only copy, hashed): data/paper_tables_2026-11-15/inputs/Gap_Split_TEST.csv  (written by the mechanism analysis; pinned strict search over every stored root
plus the seven test-run engines; effective ceiling = larger of that and the best AMD-validated episode of any row in the cell, which changes corescore P0 only).
Covers the briefs T0, A0, W2R, L2R and P0 (25 cells). It has NO rows for the control briefs T and A, so Table 1b cannot be recomputed from it.

Definitions used here (the same as the mechanism note's pools):
  cell share of ceiling = 100 * min(100, row share of reference) / min(100, ceiling share of reference), over headroom cells whose ceiling share is above 0; each cell is the mean over the row's repeats;
  pooled = mean over those cells (T0 and A0: 7 cells, ceilings above 0 in 7; W2R, L2R and P0: 9 headroom cells, ceiling above 0 in 8).
  The cap at 100 only changes power cells (the power reference is a two-step floor that a six-run policy can pass).
  Ceiling row = the capped effective ceiling share per cell, mean and median over the headroom cells (T0 67.7, A0 99.1, pooled 85.7; W2R 94.5, L2R 100.0, P0 66.7, pooled 87.1).
Self-check: the same formula with the Section 7 ceilings and no cap must give the pooled 'ceil_share' of tables.json (V9 77.4 on T0 and A0).

    .venv/bin/python scripts/test_r2_2026-11-13/tr_pt_ceiling2.py
Writes data/paper_tables_2026-11-15/ceiling2.json. Offline, read-only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_pt_lib as L  # noqa: E402

INPUT = L.OUT / "inputs" / "Gap_Split_TEST.csv"
AB = ("T0", "A0")
PW = ("W2R", "L2R", "P0")


def fl(x):
    return None if x in ("", None) else float(x)


def main() -> int:
    C = L.jread(L.OUT / "cells.json")
    T = L.jread(L.OUT / "tables.json")
    cells, rows = C["cells"], C["rows"]
    raw = list(csv.DictReader(open(INPUT, encoding="utf-8")))
    gs = {f"{r['design']}|{r['brief']}": r for r in raw if r["row_type"] == "cell"}
    problems = []
    cc: dict[str, dict] = {}
    for k, r in gs.items():
        if k not in cells:
            problems.append(f"csv cell not in cells.json: {k}")
            continue
        c = cells[k]
        ref_csv = fl(r["ref_gain"])
        if abs(ref_csv - c["ref"]) > 1e-9 * max(1.0, abs(c["ref"])):
            problems.append(f"reference differs {k}: csv {ref_csv} cells.json {c['ref']}")
        cc[k] = {"brief": c["brief"], "head": c["head"], "in_pool_csv": r["in_pool"] == "True", "ref": c["ref"], "eff_gain": fl(r["ceiling_effective_gain"]), "eff_share": fl(r["ceiling_effective_share_pct"]),
                 "eff_share_capped": fl(r["ceiling_effective_share_capped_pct"]), "strict_gain": fl(r["ceiling_strict_gain"]), "strict_share": fl(r["ceiling_strict_share_pct"]), "lenient_share": fl(r["ceiling_lenient_share_pct"]),
                 "best_observed_share": fl(r["best_observed_share_pct"]), "s7frame_strict_gain": fl(r["ceiling_s7frame_strict_gain"]), "s7_strict_share_cells": (100.0 * c["ceil_strict"] / c["ref"]) if c.get("ceil_strict") is not None and c["ref"] else None}
        if bool(c["head"]) != cc[k]["in_pool_csv"]:
            problems.append(f"headroom flag differs {k}: cells.json {c['head']} csv {r['in_pool']}")
    briefs_in_csv = sorted({v["brief"] for v in cc.values()})

    def cell_ratio(lab: str, k: str, capped: bool = True):
        g = [x["gain"] for x in rows.get(lab, {}).get(k, [])]
        if not g or not cells[k]["head"] or k not in cc:
            return None
        share = 100.0 * (sum(g) / len(g)) / cells[k]["ref"]
        ce = cc[k]["eff_share_capped"] if capped else cc[k]["eff_share"]
        if ce is None or ce <= 0:
            return None
        return 100.0 * (min(100.0, share) if capped else share) / ce

    def pool(lab: str, briefs):
        ks = sorted(k for k in cc if cells[k]["brief"] in briefs and cells[k]["head"])
        per = {k: cell_ratio(lab, k) for k in ks}
        vals = [v for v in per.values() if v is not None]
        by = {}
        for b in briefs:
            vb = [v for k, v in per.items() if cells[k]["brief"] == b and v is not None]
            by[b] = L.mean(vb) if vb else None
        return {"pooled": L.mean(vals) if vals else None, "n": len(vals), "by_brief": by, "cells": {k: v for k, v in per.items()}}

    out: dict = {"schema": "tr-pt-ceiling2-v1", "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(), "briefs_in_csv": briefs_in_csv, "problems": problems, "rows": {}, "ceiling_row": {}, "cells": cc}
    for lab in sorted(rows):
        if lab in ("ref3", "ref2", "ref2p", "ceiling", "ceiling (unchecked accepted)") or lab.endswith("(unchecked)"):
            continue
        out["rows"][lab] = {"ab": pool(lab, AB), "power": pool(lab, PW)}
    # ceiling pseudo-rows: the capped effective ceiling share per headroom cell
    for grp, briefs in (("ab", AB), ("power", PW)):
        per = {k: cc[k]["eff_share_capped"] for k in cc if cells[k]["brief"] in briefs and cells[k]["head"]}
        d = {"pooled_mean": L.mean(list(per.values())), "pooled_median": L.median(list(per.values())), "n": len(per), "by_brief": {}}
        for b in briefs:
            vb = [v for k, v in per.items() if cells[k]["brief"] == b]
            d["by_brief"][b] = {"mean": L.mean(vb), "median": L.median(vb), "n": len(vb)}
        d["cells"] = per
        out["ceiling_row"][grp] = d
    # budget rows against the recomputed ceiling (V9 at 6 = the V9 row)
    out["budget_v9"] = {lab: pool(lab, AB)["pooled"] for lab in ("V9", "V9 b3", "V9 b10") if lab in rows}
    # cells where an observed episode is above the pinned strict ceiling
    out["above_strict"] = [{"cell": k, "best_observed_share": v["best_observed_share"], "strict_share": v["strict_share"]} for k, v in cc.items()
                           if v["best_observed_share"] is not None and v["strict_share"] is not None and v["best_observed_share"] > v["strict_share"] + 1e-9]
    # lenient pools (headroom cells; T0 and A0 uncapped, power capped)
    for grp, briefs in (("ab", AB), ("power", PW)):
        ks = [k for k in cc if cells[k]["brief"] in briefs and cells[k]["head"]]
        out.setdefault("lenient_row", {})[grp] = {"strict_pooled": L.mean([min(100.0, cc[k]["strict_share"]) for k in ks]), "lenient_pooled": L.mean([min(100.0, cc[k]["lenient_share"]) for k in ks]),
                                                  "by_brief": {b: {"strict": L.mean([min(100.0, cc[k]["strict_share"]) for k in ks if cells[k]["brief"] == b]), "lenient": L.mean([min(100.0, cc[k]["lenient_share"]) for k in ks if cells[k]["brief"] == b])} for b in briefs}}
    # self-check: old formula with Section 7 ceilings and no cap against tables.json
    chk = {}
    for r in T["t1_main"]:
        if r["key"] in ("V9", "v2", "fixed", "GA-R1", "F1-R1"):
            ks = [k for k in cells if cells[k]["table"] == "ab" and cells[k]["brief"] in AB and cells[k]["head"] and cells[k].get("ceil_strict", 0) > 0]
            vals = []
            for k in ks:
                g = [x["gain"] for x in rows[r["key"]].get(k, [])]
                if g:
                    vals.append(100.0 * (sum(g) / len(g)) / cells[k]["ceil_strict"])
            chk[r["key"]] = {"old_tables_json": r["pooled"]["ceil_share"], "recomputed_here": L.mean(vals), "diff": abs(r["pooled"]["ceil_share"] - L.mean(vals))}
    out["old_formula_check"] = chk
    # print
    print("csv cells", len(cc), "briefs", briefs_in_csv, "problems", problems)
    print("ceiling row  ab:", {k: round(v, 2) if isinstance(v, float) else v for k, v in out["ceiling_row"]["ab"].items() if k != "cells"})
    print("ceiling row  power:", {k: round(v, 2) if isinstance(v, float) else v for k, v in out["ceiling_row"]["power"].items() if k != "cells"})
    for lab in ("V9", "v2", "fixed", "GA-R1", "F1-R1", "v2.1", "V3", "V6", "random", "beam", "bandit", "grid", "v1", "V9 (ChatGPT)"):
        d = out["rows"].get(lab)
        if d:
            print(f"  {lab:14s} share of recomputed ceiling  T0+A0 {d['ab']['pooled'] if d['ab']['pooled'] is None else round(d['ab']['pooled'], 1)} (n {d['ab']['n']}) | power {d['power']['pooled'] if d['power']['pooled'] is None else round(d['power']['pooled'], 1)} (n {d['power']['n']})")
    print("old-formula check:", {k: round(v["diff"], 4) for k, v in chk.items()})
    print("budget V9 vs recomputed ceiling:", out["budget_v9"], "| above strict:", out["above_strict"])
    (L.OUT / "ceiling2.json").write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
