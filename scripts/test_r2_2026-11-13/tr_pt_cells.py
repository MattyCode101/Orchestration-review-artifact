"""Paper tables, step 3: the cell table (reference, band, class, ceiling) and the per-row, per-cell gains for every row of every table. Offline, read-only.

Reads: data/paper_tables_2026-11-15/{graph4,refs}.json, the released results (tr_pt_lib.load_episodes), the beam tables (unsealed_ab/beam/cells.json,
unsealed_rest/beam/cells_{power,b3,b10}.json), the power references (unsealed_rest/power_ref), the Section 7 ceiling cells (data/policy_table_section7_2026-11-13/ceiling).
Writes data/paper_tables_2026-11-15/cells.json: {cells: {design|brief: {...}}, rows: {rowlabel: {design|brief: [{tag, gain, ...} per repeat]}}, notes}.

    .venv/bin/python scripts/test_r2_2026-11-13/tr_pt_cells.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_pt_lib as L  # noqa: E402

CEIL = L.PROJECT / "data/policy_table_section7_2026-11-13/ceiling"
UNS_AB, UNS_REST = L.BASE / "unsealed_ab", L.BASE / "unsealed_rest"


def start_of(eps):
    s = {}
    for e in eps:
        s.setdefault(e["design"], e["start"])
    return s


def main() -> int:
    eps = L.load_episodes()
    starts = start_of(eps)
    refs = L.jread(L.OUT / "refs.json")["cells"]
    graph = L.jread(L.OUT / "graph4.json")
    cells: dict[str, dict] = {}
    # ---- timing and area cells: strict three-step reference, noise band, class, Section 7 ceilings
    for key, c in refs.items():
        d, b = key.split("|")
        cell = {"design": d, "brief": b, "table": "ab", "band": c["band"], "ref": c["ref3"]["gain"] if c["ref3"] else 0.0, "ref_steps": c["ref3"]["steps"] if c["ref3"] else 0,
                "ref_path": c["ref3"]["path"] if c["ref3"] else [], "ref2": c["ref2"]["gain"] if c["ref2"] else 0.0, "ref_lenient": c["ref3_lenient"]["gain"] if c["ref3_lenient"] else 0.0,
                "ref2_lenient": c["ref2_lenient"]["gain"] if c["ref2_lenient"] else 0.0, "ratio": c["ratio"], "class": c["class"], "head": c["class"] != "none",
                "section7_ref": c["section7_ref3_gain"], "unverified_better": c["unverified_better"]}
        for tag, name in (("strict", "new_strict_f"), ("lenient", "new_lenient_f")):
            j = L.jread(CEIL / name / f"cell_{d}_{b}.json")
            r6 = j["budgets"]["6"]
            cell["ceil_" + tag] = max(0.0, float(r6["gain"]))
            cell["ceil_" + tag + "_amd_s7"] = r6["final_amd"]
            cell["ceil_" + tag + "_state"] = r6.get("final_state_id")
            if tag == "lenient":
                node = graph["designs"][d]["nodes"].get(r6.get("final_state_id") or "", None)
                cell["ceil_lenient_amd_now"] = node["amd"] if node else ("start" if not r6.get("final_state_id") or r6.get("final_state_id") == graph["designs"][d]["start"] else "(outside the depth-3 graph)")
            cell["ceil_pruned_mandates"] = j["plan_pruned_mandates"]
        cell["ceil"] = cell["ceil_strict"]
        cells[key] = cell
    # ---- power cells: two-step reference over 14 recipes = the AMD-checked winner of the stored graph
    pr = UNS_REST / "power_ref"
    cand, win = L.jread(pr / "candidates.json"), L.jread(pr / "winners.json")
    for d in L.DESIGNS:
        for b in L.POWER_BRIEFS:
            w = win[f"{d}|{b}"]
            st = w.get("winner")
            ent = next((x for x in cand[d]["briefs"][b] if x["state"] == st), None)
            sm = cand[d]["start_metrics"]
            g = L.gain_of(b, sm, ent["metrics"]) if ent else 0.0
            band = float(next(e["noise"][L.GOAL[b][0]] for e in eps if e["design"] == d))
            ratio = (L.mw(g) / L.mw(band)) if L.GOAL[b][0] == "dynamic_power_w" else g / band
            cells[f"{d}|{b}"] = {"design": d, "brief": b, "table": "power", "band": band, "ref": max(g, 0.0), "ref_steps": ent["steps"] if ent else 0,
                                 "ref_path": [p.split(".")[-1] for p in ent["path"]] if ent else [], "ratio": ratio, "class": L.head_class(ratio), "head": L.head_class(ratio) != "none",
                                 "ref_why": w.get("why"), "ref_state": st, "ref_rank": w.get("rank")}
    # ---- rows
    rows: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    meta: dict[str, dict] = {}
    for e in eps:
        lab = L.policy_label(e)
        rows[lab][f"{e['design']}|{e['brief']}"].append({"tag": e["tag"] if e["seed"] is None else f"s{e['seed']}", "gain": e["gain"], "eid": e["eid"], "seed": e["seed"]})
        meta[lab] = {"row": e["row"], "gateway": e["gateway"], "budget": e["budget"], "approach": e["approach"]}
    # ---- beam rows (one simulated episode per cell; strict pick = AMD-passed; optimistic = best valid visited state)
    def beam_rows(path: Path, label: str, opt_label: str):
        j = L.jread(path)
        for key, c in j.items():
            d, b = c["design"], c["brief"]
            sm = starts[d]
            gs = L.gain_of(b, sm, c["pick_amd_passed"]["metrics"])
            go = L.gain_of(b, sm, c["pick_optimistic"]["metrics"])
            rows[label][key].append({"tag": "b", "gain": gs, "eid": "beam|" + key, "seed": None, "runs": c["runs"], "wall_h": c["wall_h"], "censored": c["censored"],
                                     "pick_amd": c["pick_amd_passed"]["amd"], "opt_amd": c["pick_optimistic"]["amd"]})
            rows[opt_label][key].append({"tag": "b", "gain": go, "eid": "beam|" + key, "seed": None})
        meta[label] = {"row": "beam", "gateway": False, "budget": 6, "approach": "beam"}
        meta[opt_label] = {"row": "beam (unchecked accepted)", "gateway": False, "budget": 6, "approach": "beam"}

    beam_rows(UNS_AB / "beam/cells.json", "beam", "beam (unchecked)")
    j = L.jread(UNS_REST / "beam/cells_power.json")
    for key, c in j.items():
        d, b = c["design"], c["brief"]
        sm = starts[d]
        rows["beam"][key].append({"tag": "b", "gain": L.gain_of(b, sm, c["pick_amd_passed"]["metrics"]), "eid": "beam|" + key, "seed": None, "runs": c["runs"], "wall_h": c["wall_h"],
                                  "censored": c["censored"], "pick_amd": c["pick_amd_passed"]["amd"], "opt_amd": c["pick_optimistic"]["amd"]})
        rows["beam (unchecked)"][key].append({"tag": "b", "gain": L.gain_of(b, sm, c["pick_optimistic"]["metrics"]), "eid": "beam|" + key, "seed": None})
    for bud, fn in ((3, "cells_b3.json"), (10, "cells_b10.json")):
        j = L.jread(UNS_REST / "beam" / fn)
        for key, c in j.items():
            d, b = c["design"], c["brief"]
            sm = starts[d]
            rows[f"beam b{bud}"][key].append({"tag": "b", "gain": L.gain_of(b, sm, c["pick_amd_passed"]["metrics"]), "eid": f"beam{bud}|" + key, "seed": None, "runs": c["runs"],
                                              "wall_h": c["wall_h"], "censored": c["censored"], "pick_amd": c["pick_amd_passed"]["amd"], "opt_amd": c["pick_optimistic"]["amd"]})
            rows[f"beam b{bud} (unchecked)"][key].append({"tag": "b", "gain": L.gain_of(b, sm, c["pick_optimistic"]["metrics"]), "eid": f"beam{bud}|" + key, "seed": None})
        meta[f"beam b{bud}"] = {"row": "beam", "gateway": False, "budget": bud, "approach": "beam"}
    # ---- reference rows as pseudo-rows (one value per cell)
    for key, c in cells.items():
        rows["ref3"][key].append({"tag": "-", "gain": c["ref"], "eid": "ref", "seed": None}) if c["table"] == "ab" else rows["ref2p"][key].append({"tag": "-", "gain": c["ref"], "eid": "ref", "seed": None})
        if c["table"] == "ab":
            rows["ref2"][key].append({"tag": "-", "gain": c["ref2"], "eid": "ref", "seed": None})
            rows["ceiling"][key].append({"tag": "-", "gain": c["ceil_strict"], "eid": "ceil", "seed": None})
            rows["ceiling (unchecked accepted)"][key].append({"tag": "-", "gain": c["ceil_lenient"], "eid": "ceil", "seed": None})
    # order repeats
    for lab in rows:
        for key in rows[lab]:
            rows[lab][key].sort(key=lambda x: x["tag"])
    out = {"schema": "tr-pt-cells-v1", "cells": cells, "rows": {k: dict(v) for k, v in rows.items()}, "meta": meta,
           "starts": starts, "n_episodes": len(eps)}
    (L.OUT / "cells.json").write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    (L.OUT / "episodes.json").write_text(json.dumps(eps, sort_keys=True), encoding="utf-8")
    # printouts
    print("episodes", len(eps), "| rows", len(rows))
    cnt = defaultdict(int)
    for lab, v in rows.items():
        cnt[lab] = sum(len(x) for x in v.values())
    for lab in sorted(cnt):
        print(f"  {lab:34s} episodes {cnt[lab]:4d}  cells {len(rows[lab])}")
    print("power cells (class):", {k.split('|')[0][:10] + '|' + k.split('|')[1]: (round(c['ref'], 4), c['class']) for k, c in cells.items() if c["table"] == "power"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
