"""Tier 2 (item d), step 5b for the TRAINING designs: the exhaustive two-step reference of each design on W2R, L2R and P0, AMD-checked on the chosen path.

The procedure of scripts/development_designs_2026-11-12/dd_exhaustive.py (imported: candidates_for, choose, amd_status are used as they are), on the TRAINING scope of
ft_scope.py. Graph: the fill-run TRAINING manifest's roots plus this task's new runs (tier2_exhaustive/engine), joined by the campaign's generic builder. Pool: all 14
recipes. Candidates: the start and every state within two steps that is brief-valid (gate + start-relative property rule) with a positive gain; ordered by objective gain, then
fewer steps, then state id. Lazy exact AMD: the first candidate whose chain (state and, at depth 2, its depth-1 parent) has no failing record is taken; chain states without a
record are validated (a new validation is charged to the job-second cap); a failed candidate is skipped and the next one tried. The pick is the best valid state with an
AMD PASS on every state of its path. Not done here (VALIDATION did it): the grid-search rows.
usage: ft_exhaustive.py [--out PATH] [--no-verify]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ft_scope as ds  # noqa: E402  (first: it points dd_common at the TRAINING scope)
import dd_common as dd  # noqa: E402
import dd_exhaustive as dx  # noqa: E402  (candidates_for, choose, amd_status, unchanged)
import tr_common as tc  # noqa: E402

PROJECT = ds.PROJECT


def union_graph():
    sys.path.insert(0, str(PROJECT / "scripts/d6v2_2026-11-12"))
    import d6_manifests as dm  # noqa: E402

    roots = {k: Path(v) for k, v in ds.manifest().body["roots"].items()}
    overlay = ds.TIER2 / "engine"
    if (overlay / "cache").is_dir():
        roots["fill_tier2_train"] = overlay
    meter, train_starts = dm.cm.train_inputs()
    body, report = dm.cm.build_generic(roots, meter, train_starts, {"v4": dm.cm.V4})
    if report["problems"]:
        raise SystemExit(f"union manifest problems: {report['problems'][:5]}")
    return dd.Graph(body), body, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ds.TIER2 / "exhaustive.json")
    ap.add_argument("--no-verify", action="store_true", help="do not start any AMD validation (report what is unverified)")
    args = ap.parse_args()
    g, body, report = union_graph()
    roots = {k: Path(v) for k, v in body["roots"].items()}
    roots["fill_tier2_train"] = ds.TIER2 / "engine"      # new AMD records are written here, so scan it even before it holds any run
    ref = ds.start_ids()
    ctx = None if args.no_verify else ds.build("run", with_provider=False)
    graphs, cands = {}, {}
    for d in ds.DESIGNS:
        sid = ref[d]
        start = g.start_evidence(sid)
        for b in ds.BRIEFS:
            brief = tc.brief_for(d, b)
            states, missing, failed = dd.reachable(g, sid, dd.RECIPES3, dd.RECIPES3)
            graphs[(d, b)] = {"states": len(states) - 1, "missing": [list(m) for m in missing], "failed": [list(m) for m in failed], "pool": len(dd.RECIPES3)}
            cands[(d, b)] = dx.candidates_for(g, d, b, brief, start, states)
    verified_new = []
    for rnd in range(8):
        amd = dd.amd_records(roots, ref)
        need = set()
        for (d, b), cs in cands.items():
            c, todo = dx.choose(cs, amd, d, ref[d])
            need.update((d, x) for x in todo)
        need = sorted(need)
        if not need or ctx is None:
            break
        print(f"round {rnd}: validating {len(need)} states", flush=True)

        def verify(item):
            d, sid = item
            rec = ctx.ledger.verify(d, sid)
            return {"design": d, "state": sid, "result": rec["result"], "wall_s": sum(a["wall_s"] for a in rec["attempts"])}

        with ThreadPoolExecutor(max_workers=min(len(need), 12)) as pool_:
            for r in pool_.map(verify, need):
                verified_new.append(r)
                print("   ", r["design"], r["state"][:12], r["result"], round(r["wall_s"]), "s", flush=True)
    amd = dd.amd_records(roots, ref)
    out = {"scope": "TRAINING", "manifest_sha256": report["manifest_sha256"], "new_amd_validations": verified_new, "designs": {}}
    for d in ds.DESIGNS:
        sid = ref[d]
        start = g.start_evidence(sid)
        row = {"start": dd.metrics_of(start), "briefs": {}}
        for b in ds.BRIEFS:
            metric, _direction = dd.OBJECTIVE[b]
            c, todo = dx.choose(cands[(d, b)], amd, d, sid)
            gr = graphs[(d, b)]
            n_skipped = 0
            for x in cands[(d, b)]:
                if c is not None and x["state"] == c["state"]:
                    break
                n_skipped += 1
            if c is None:
                pick = {"state": sid, "path": [], "steps": 0, "gain": 0.0, "metrics": dd.metrics_of(start), "amd": "start"}
            else:
                pick = {**c, "amd": "pass" if not todo else "UNVERIFIED", "unverified_states": todo}
            row["briefs"][b] = {"pick": pick, "complete": not gr["missing"], "missing_pairs": len(gr["missing"]), "failed_pairs": len(gr["failed"]),
                                "states_in_graph": gr["states"], "valid_candidates": len(cands[(d, b)]), "better_candidates_failing_amd": n_skipped,
                                "unit": dd.UNIT[metric], "pool": gr["pool"]}
        out["designs"][d] = row
    args.out.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    for d in ds.DESIGNS:
        print("==", d)
        for b in ds.BRIEFS:
            r = out["designs"][d]["briefs"][b]
            p = r["pick"]
            print(f"  {b}: exhaustive {p['gain']:+.4g} {r['unit']} {'>'.join(x.split('.')[-1] for x in p['path']) or 'start'} amd {p['amd']} | "
                  f"complete {r['complete']} (missing {r['missing_pairs']}) valid {r['valid_candidates']} skipped-for-AMD {r['better_candidates_failing_amd']}")
    print("written", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
