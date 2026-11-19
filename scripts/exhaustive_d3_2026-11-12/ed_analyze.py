"""Depth-3 reference from stored outcomes: for each test design and each of the four timing/area briefs (T0, A0, T, A), the best valid
state within two steps and within three steps, with the AMD status of each pick. Read-only; can run on partial data (a design is
marked incomplete while any depth-3 pair is missing). The frozen best-known frame is NOT used (it also holds sealed power rows);
gains are against the B2 start and are shown in objective units and in the recorded noise bands of the test baseline.

Validity as in round 1 (pt_search.is_valid): the brief's limits against the start (orch.engine.gate.judge) and no more DONT_TOUCH /
fixed LOC / fixed BEL cells than the start. Ranking as pt_search.pick: objective, then the other metric, then fewer steps.
No power value is read or printed.

usage: ed_analyze.py [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ed_common as ec  # noqa: E402

sys.path.insert(0, str(ec.PROJECT / "scripts/policy_table_2026-11-11"))
import pt_common as pt  # noqa: E402
import pt_search as ps  # noqa: E402
from orch.contract.models import Evidence, MetricName  # noqa: E402

BANDS = ec.PROJECT / "data/exp_evening_2026-11-04/test_b2/overnight_baseline_2026-11-04/noise_bands_2026-11-04.json"


def failed_pairs() -> set[tuple[str, str]]:
    """(state, recipe) pairs whose run ended in a terminal recipe-level failure (journalled failed_terminal): finished, but no output state."""

    import json

    out = set()
    for jp in (ec.SMOKE_JOURNAL, ec.JOURNAL):
        if jp.is_file():
            for line in jp.read_text().splitlines():
                if line.strip():
                    j = json.loads(line)
                    if j.get("kind") == "run" and j.get("status") == "failed_terminal":
                        out.add((j["state"], j["recipe"]))
    return out


def amd_b2_sets() -> tuple[set, set, set]:
    """(campaign-ledger passes against B2, this task's ledger passes against B2, baseline passes against the contest original)."""

    camp, base = ps.amd_sets()
    new = set()
    for p in (ec.ENGINE / "verification" / "records").glob("*/*/*.json"):
        r = ec.jread(p)
        if r.get("result") == "pass" and r.get("reference_sha256") == ec.B2.get(r.get("design")):
            new.add((r["design"], r["output_sha256"]))
    return camp, new, base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ec.WORK / "analysis")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    vers = ec.versions()
    roots = dict(ec.SRC_ROOTS)
    roots["new"] = ec.ENGINE
    runs, _ = ec.load_runs(roots, vers)
    by_in: dict[tuple[str, str], list[dict]] = {}
    for r in runs:
        by_in.setdefault((r["in"], r["recipe"]), []).append(r)
    bands = ec.jread(BANDS)["bands"]
    camp, new, base = amd_b2_sets()
    failed = failed_pairs()
    g = ps.Graph(ps.TEST_MANIFEST, "test_base", ps.RECIPES)
    out = {"designs": {}, "meter_version": ec.CUR_METER, "note": "no power value used; frame not used"}
    lines = []
    for d in ec.DESIGNS:
        s0 = ec.B2[d]
        start_ev = g.start[d]
        states = {s0: {"steps": 0, "path": [], "ev": start_ev}}
        frontier = [s0]
        missing = []
        n_failed = 0
        for depth in (1, 2, 3):
            nxt = []
            for s in frontier:
                for rec in ec.REC10:
                    L = by_in.get((s, rec))
                    if not L:
                        if (s, rec) in failed:
                            n_failed += 1                    # finished with a terminal recipe failure: no output state, not missing
                        else:
                            missing.append((depth, s, rec))
                        continue
                    r = sorted(L, key=lambda x: (x["hit"], x["run"]))[0]
                    o = r["out"]
                    if o in states:
                        continue
                    ev = Evidence.model_validate_json((roots[r["root"]] / "runs" / r["run"] / "evidence.json").read_bytes())
                    states[o] = {"steps": depth, "path": states[s]["path"] + [rec], "ev": ev}
                    nxt.append(o)
            frontier = nxt
        miss3 = [m for m in missing if m[0] == 3]
        row = {"states_by_depth": {k: sum(1 for v in states.values() if v["steps"] == k) for k in range(4)},
               "depth3_pairs_missing": len(miss3), "depth12_pairs_missing": len(missing) - len(miss3), "pairs_failed_terminal": n_failed,
               "depth3_complete": not miss3 and not (len(missing) - len(miss3)), "briefs": {}}
        for b in pt.FOUR:
            metric, direction = pt.OBJECTIVE[b]
            other = pt.LIMIT_METRIC[b]
            other_dir = "max" if other == "wns_ns" else "min"
            s_val = start_ev.metric(MetricName(metric)).value
            cands = []
            for sid, info in states.items():
                if info["steps"] == 0:
                    continue
                ev = info["ev"]
                if not ps.is_valid(b, ev, start_ev):
                    continue
                v = ev.metric(MetricName(metric)).value
                o = ev.metric(MetricName(other)).value
                verified = (d, sid) in camp or (d, sid) in new
                key = ((-v if direction == "max" else v), (-o if other_dir == "max" else o), info["steps"], 0 if verified else 1, sid)
                cands.append((key, sid, v, info))
            cands.sort(key=lambda c: c[0])

            def best(max_depth, only_verified=False):
                for key, sid, v, info in cands:
                    if info["steps"] > max_depth:
                        continue
                    ok = (d, sid) in camp or (d, sid) in new
                    if only_verified and not ok:
                        continue
                    gain = (v - s_val) if direction == "max" else (s_val - v)
                    if gain <= 0:
                        return None
                    status = "PASS vs B2" if ok else ("PASS vs contest original only" if (d, sid) in base else "no AMD record")
                    return {"state": sid, "path": info["path"], "steps": info["steps"], "value": v, "gain": gain,
                            "gain_bands": gain / bands[d][metric], "amd": status}
                return None

            row["briefs"][b] = {
                "start": s_val, "band": bands[d][metric], "valid_states_depth<=2": sum(1 for c in cands if c[3]["steps"] <= 2),
                "valid_states_depth<=3": len(cands),
                "best_depth2": best(2), "best_depth3": best(3),
                "best_depth2_amd_b2": best(2, True), "best_depth3_amd_b2": best(3, True),
            }
            r2, r3, v2, v3 = (row["briefs"][b][k] for k in ("best_depth2", "best_depth3", "best_depth2_amd_b2", "best_depth3_amd_b2"))
            f = lambda x: "none" if x is None else f"{x['gain']:+.4g} ({x['gain_bands']:.2f} bands, {'>'.join(p.split('.')[-1][:7] for p in x['path'])}, {x['amd']})"
            lines.append(f"{d[:22]:22s} {b:3s} depth2 {f(r2)} | depth3 {f(r3)} | B2-verified depth3 {f(v3)}")
        out["designs"][d] = row
    ec.jwrite(args.out / "depth3_best.json", out)
    (args.out / "depth3_best.txt").write_text("\n".join(lines) + "\n")
    for d, row in out["designs"].items():
        print(d, row["states_by_depth"], "depth-3 pairs missing:", row["depth3_pairs_missing"], "terminal recipe failures:", row["pairs_failed_terminal"],
              "complete" if row["depth3_complete"] else "INCOMPLETE")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
