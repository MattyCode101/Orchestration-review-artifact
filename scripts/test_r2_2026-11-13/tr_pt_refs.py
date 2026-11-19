"""Paper tables, step 2: references per cell for the four timing and area briefs, from graph4.json (tr_pt_graph.py). Offline, read-only.

Per (design, brief): the best brief-valid state within three steps (and within two) of B2 over the ten timing and area recipes, STRICT = AMD PASS against B2 in any ledger
(campaign, depth-3 job, test-run engines), LENIENT = not recorded as failed. Rank: goal gain, then the limit metric (better first), then fewer steps, then state id (as
s7_refs.py). Band = the design's round-2 noise band of the goal metric, from the TEST result files. Headroom class = three-step strict gain over the band: none under 1,
low 1 to 3, high over 3. Cells where a better-ranked state than the strict reference exists but has no AMD record are listed (they are the only states whose AMD check
could change a cell's reference).

    .venv/bin/python scripts/test_r2_2026-11-13/tr_pt_refs.py
Writes data/paper_tables_2026-11-15/refs.json and prints the cell table and the comparison with Section 7 (refs.json of 13 Nov) and the D1 report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_pt_lib as L  # noqa: E402

LIMIT = {"T0": "packed_luts", "T": "packed_luts", "A0": "wns_ns", "A": "wns_ns"}


def path_of(nodes: dict, sid: str) -> list[str]:
    out = []
    while nodes[sid]["parent"] is not None:
        out.append(nodes[sid]["recipe"].split(".")[-1])
        sid = nodes[sid]["parent"]
    return out[::-1]


def main() -> int:
    g = L.jread(L.OUT / "graph4.json")
    eps = L.load_episodes()
    band = {}
    for e in eps:
        for m, v in e["noise"].items():
            band.setdefault((e["design"], m), set()).add(round(float(v), 9))
    clash = {k: v for k, v in band.items() if len(v) > 1}
    s7 = L.jread(L.PROJECT / "data/policy_table_section7_2026-11-13/refs.json")
    out = {"schema": "tr-pt-refs-v1", "amd_sets": g["amd_sets"], "band_clashes": {f"{k[0]}|{k[1]}": sorted(v) for k, v in clash.items()}, "cells": {}}
    print("noise band clashes between episodes:", len(clash))
    hdr = f"{'cell':26s} {'band':>8s} {'ref3 strict':>12s} {'bands':>7s} {'class':>5s} {'ref2 strict':>11s} {'ref3 lenient':>12s} {'S7 ref3':>9s} unverified-better (best first)"
    print(hdr)
    for d in L.DESIGNS:
        nodes = g["designs"][d]["nodes"]
        for b in L.AB_BRIEFS:
            m, direction = L.GOAL[b]
            bd = float(next(iter(band[(d, m)])))
            cands = [(sid, n) for sid, n in nodes.items() if n["d"] > 0 and n["v" + b]]

            def key(c):
                sid, n = c
                o = n["m"][LIMIT[b]]
                return (-n["g" + b], (-o if LIMIT[b] == "wns_ns" else o), n["d"], sid)

            cands.sort(key=key)

            def pick(max_d: int, ok):
                for sid, n in cands:
                    if n["d"] <= max_d and ok(n):
                        return None if n["g" + b] <= 0 else {"state": sid, "gain": n["g" + b], "steps": n["d"], "path": path_of(nodes, sid), "amd": n["amd"], "amd_src": n["amd_src"]}
                return None

            strict = lambda n: n["amd"] == "PASS vs B2"  # noqa: E731
            lenient = lambda n: n["amd"] != "FAIL vs B2"  # noqa: E731
            r3, r2, l3, l2 = pick(3, strict), pick(2, strict), pick(3, lenient), pick(2, lenient)
            g3 = r3["gain"] if r3 else 0.0
            ub = [{"state": sid, "gain": n["g" + b], "steps": n["d"], "path": path_of(nodes, sid), "amd": n["amd"]} for sid, n in cands
                  if n["g" + b] > g3 + 1e-12 and n["amd"] != "PASS vs B2" and n["amd"] != "FAIL vs B2"][:6]
            ratio = g3 / bd
            s7r = (s7["designs"][d]["briefs"][b]["ref3"] or {}).get("gain", 0.0)
            out["cells"][f"{d}|{b}"] = {"design": d, "brief": b, "metric": m, "band": bd, "ref3": r3, "ref2": r2, "ref3_lenient": l3, "ref2_lenient": l2, "ratio": ratio,
                                        "class": L.head_class(ratio), "unverified_better": ub, "section7_ref3_gain": s7r, "section7_class": s7["designs"][d]["briefs"][b]["headroom"],
                                        "start_metrics": g["designs"][d]["start_metrics"]}
            ubs = "; ".join(f"{u['gain']:.4g} ({u['steps']}st, {u['amd']})" for u in ub[:3])
            print(f"{L.SHORT[d]+' '+b:26s} {bd:8.4g} {g3:12.4g} {ratio:7.2f} {L.head_class(ratio):>5s} {(r2['gain'] if r2 else 0):11.4g} {(l3['gain'] if l3 else 0):12.4g} {s7r:9.4g} {ubs}")
    (L.OUT / "refs.json").write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    ncheck = sum(1 for c in out["cells"].values() if c["unverified_better"])
    print("cells whose best valid state lacks an AMD record (check candidates):", ncheck, "of", len(out["cells"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
