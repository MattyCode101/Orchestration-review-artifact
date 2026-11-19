"""Step 5b: the exhaustive two-step reference of each development design and brief, AMD-checked on the chosen path, plus the
grid-search rows (read from stored depth-one outcomes). No recipe run; only AMD validations of picked paths (job-seconds cap).

Graph: the screen's stored runs plus this task's new runs (a union manifest built in memory with the campaign's generic builder).
For each brief the pool is the 10 timing and area recipes (T0, A0) or all 14 (W2R, L2R, P0); candidates are the start and every
state within two steps. Candidates that are valid for the brief (gate + start-relative property rule) are ordered by objective gain,
then fewer steps, then state id. Lazy exact AMD: the first candidate whose chain (state and, at depth 2, its depth-1 parent) has no
failing record is taken; chain states without a record are validated (existing PASS records of the screen are accepted read-through);
a failed candidate is skipped and the next one is tried. So the pick is the best valid state with an AMD PASS on every state of its path.

usage: dd_exhaustive.py [--out PATH] [--no-verify]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dd_common as dd  # noqa: E402
import dd_scope as ds  # noqa: E402
import campaign_manifest as cm  # noqa: E402

OVERLAY = dd.WORK / "engine"


def union_graph() -> tuple[dd.Graph, dict, dict]:
    roots = dd.manifest_roots(ds.DESIGNS)
    if (OVERLAY / "cache").is_dir():
        roots["dev_new"] = OVERLAY
    starts = {d: dd.start_id(d) for d in ds.DESIGNS}
    body, report = cm.build_generic(roots, dd.meter_version(), starts, dict(dd.manifest_roots(ds.DESIGNS)))
    if report["problems"]:
        raise SystemExit(f"union manifest problems: {report['problems'][:5]}")
    return dd.Graph(body), body, report


def candidates_for(g: dd.Graph, design: str, brief_id: str, brief, start, states) -> list[dict]:
    out = []
    for sid, info in states.items():
        ev = start if info["steps"] == 0 else g.evidence(info["entry"])
        if info["steps"] and not dd.is_valid(brief, ev, start):
            continue
        gain = dd.gain_of(brief_id, start, ev)
        if info["steps"] and gain <= 0:
            continue
        out.append({"state": sid, "steps": info["steps"], "path": info["path"], "parent": info.get("parent"), "gain": gain,
                    "metrics": dd.metrics_of(ev)})
    out.sort(key=lambda c: (-c["gain"], c["steps"], c["state"]))
    return out


def amd_status(amd: dict, design: str, sid: str, start_id: str):
    if sid == start_id:
        return "start"
    return amd.get((design, sid))


def choose(cands: list[dict], amd: dict, design: str, start_id: str):
    for c in cands:
        chain = [c["state"]] + ([c["parent"]] if c["steps"] == 2 else [])
        st = [amd_status(amd, design, x, start_id) for x in chain]
        if any(s in ("fail", "unknown") for s in st):
            continue
        if all(s in ("pass", "start") for s in st):
            return c, []
        return c, [x for x, s in zip(chain, st) if s is None]
    return None, []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=dd.WORK / "exhaustive.json")
    ap.add_argument("--no-verify", action="store_true", help="do not start any AMD validation (report what is unverified)")
    args = ap.parse_args()
    g, body, report = union_graph()
    roots = {k: Path(v) for k, v in body["roots"].items()}
    roots["dev_new"] = OVERLAY            # new AMD records are written here, so scan it even before it holds any run
    ref = {d: dd.start_id(d) for d in ds.DESIGNS}
    ctx = None if args.no_verify else ds.build("run", with_provider=False)
    graphs, cands = {}, {}
    for d in ds.DESIGNS:
        sid = ref[d]
        start = g.start_evidence(sid)
        briefs = dd.briefs_for(d)
        for b in dd.BRIEFS:
            pool = dd.RECIPES2 if b in dd.TWO_WORKER else dd.RECIPES3
            states, missing, failed = dd.reachable(g, sid, pool, pool)
            graphs[(d, b)] = {"states": len(states) - 1, "missing": [list(m) for m in missing], "failed": [list(m) for m in failed], "pool": len(pool)}
            cands[(d, b)] = candidates_for(g, d, b, briefs[b], start, states)
    verified_new = []
    for rnd in range(8):
        amd = dd.amd_records(roots, ref)
        need = set()
        for (d, b), cs in cands.items():
            c, todo = choose(cs, amd, d, ref[d])
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
    out = {"manifest_sha256": report["manifest_sha256"], "new_amd_validations": verified_new, "designs": {}}
    for d in ds.DESIGNS:
        sid = ref[d]
        start = g.start_evidence(sid)
        bands = dd.screen_bands(d)
        row = {"start": dd.metrics_of(start), "bands": bands, "briefs": {}}
        for b in dd.BRIEFS:
            metric, direction = dd.OBJECTIVE[b]
            c, todo = choose(cands[(d, b)], amd, d, sid)
            gr = graphs[(d, b)]
            n_valid = len(cands[(d, b)])
            n_skipped = 0
            for x in cands[(d, b)]:
                if c is not None and x["state"] == c["state"]:
                    break
                n_skipped += 1
            if c is None:
                pick = {"state": sid, "path": [], "steps": 0, "gain": 0.0, "gain_bands": 0.0, "metrics": dd.metrics_of(start), "amd": "start"}
            else:
                pick = {**c, "gain_bands": c["gain"] / bands[metric], "amd": "pass" if not todo else "UNVERIFIED",
                        "unverified_states": todo}
            pick["class"] = dd.head_class(pick["gain_bands"])
            row["briefs"][b] = {"pick": pick, "complete": not gr["missing"], "missing_pairs": len(gr["missing"]), "failed_pairs": len(gr["failed"]),
                                "states_in_graph": gr["states"], "valid_candidates": n_valid, "better_candidates_failing_amd": n_skipped,
                                "unit": dd.UNIT[metric], "pool": gr["pool"]}
        out["designs"][d] = row
    # grid search with 6 runs: the training top-6 single recipes, best valid state with an AMD pass, from stored depth-one outcomes
    train = dd.jread(dd.WORK / "train_choices.json")
    out["grid"] = {}
    for d in ds.DESIGNS:
        sid, start, briefs = ref[d], g.start_evidence(ref[d]), dd.briefs_for(d)
        bands = dd.screen_bands(d)
        out["grid"][d] = {}
        for b in dd.BRIEFS:
            metric, _ = dd.OBJECTIVE[b]
            six = train["grid"]["top6"][b]
            best, steps = None, []
            for r in six:
                e = g.by_in.get((sid, r))
                if e is None or e["status"] != "ok" or not e.get("output_state_id"):
                    steps.append({"recipe": r, "found": False})
                    continue
                ev = g.evidence(e)
                run = g.outcome(e)[0]
                steps.append({"recipe": r, "found": True, "wall_s": float(run.wall_s or 0.0), "state": e["output_state_id"]})
                if not dd.is_valid(briefs[b], ev, start) or amd.get((d, e["output_state_id"])) != "pass":
                    continue
                gain = dd.gain_of(b, start, ev)
                if gain > 0 and (best is None or (gain, e["output_state_id"]) > (best["gain"], best["state"])):
                    best = {"state": e["output_state_id"], "path": [r], "steps": 1, "gain": gain, "metrics": dd.metrics_of(ev)}
            if best is None:
                best = {"state": sid, "path": [], "steps": 0, "gain": 0.0, "metrics": dd.metrics_of(start)}
            best["gain_bands"] = best["gain"] / bands[metric]
            out["grid"][d][b] = {"recipes": six, "pick": best, "steps_found": sum(s["found"] for s in steps), "wall_s_sum": sum(s.get("wall_s", 0.0) for s in steps),
                                 "wall_s_max": max((s.get("wall_s", 0.0) for s in steps), default=0.0)}
    args.out.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    for d in ds.DESIGNS:
        print("==", d)
        for b in dd.BRIEFS:
            r = out["designs"][d]["briefs"][b]
            p = r["pick"]
            print(f"  {b}: exhaustive {p['gain']:+.4g} {r['unit']} = {p['gain_bands']:.2f} bands [{p['class']}] {'>'.join(x.split('.')[-1] for x in p['path']) or 'start'} "
                  f"amd {p['amd']} | complete {r['complete']} (missing {r['missing_pairs']}) valid {r['valid_candidates']} skipped-for-AMD {r['better_candidates_failing_amd']}")
    print("written", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
