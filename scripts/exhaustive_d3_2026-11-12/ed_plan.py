"""Plan the depth-3 queue for timing and area (phase 1): read-only over the stored graph; writes only to data/exhaustive_d3_2026-11-12/plan/.

For each test design: start B2 (depth 0), the distinct one-step states, the distinct two-step states (twin classes under the
power-study rule), the ten timing/area recipes on each two-step class, minus every outcome already stored (exact key or twin class).
Includes states that break a limit (no pruning). Prints the estimate (runs and job-hours) from stored run records.

usage: ed_plan.py [--out DIR]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ed_common as ec  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ec.PLAN)
    args = ap.parse_args()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    vers = ec.versions()
    roots = dict(ec.SRC_ROOTS)
    if (ec.ENGINE / "runs").is_dir():
        roots["new"] = ec.ENGINE
    runs, skipped = ec.load_runs(roots, vers)
    uf, members, metrics = ec.twin_classes(runs)

    # --- twin merges (rule: same input class, recipe, version; identical metrics checked, never assumed)
    merges = []
    for cls, mem in sorted(members.items()):
        if len(mem) > 1:
            merges.append({"class": cls, "members": sorted(mem), "metric_triples": sorted(map(list, metrics[cls])),
                           "identical_metrics": len(metrics[cls]) == 1})
    by_in: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_cls_in: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in runs:
        by_in[(r["in"], r["recipe"])].append(r)
        by_cls_in[(uf.find(r["in"]), r["recipe"])].append(r)

    def pick(lst):
        return sorted(lst, key=lambda r: (r["hit"], r["root"] != "test_base", r["run"]))[0]

    # --- design of every state reachable from its B2 (all depths in the stored graph)
    edges_from: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        edges_from[r["in"]].append(r)
    design_of: dict[str, str] = {}
    for d, s0 in ec.B2.items():
        stack = [s0]
        design_of[s0] = d
        while stack:
            s = stack.pop()
            for r in edges_from.get(s, []):
                if r["out"] not in design_of:
                    design_of[r["out"]] = d
                    stack.append(r["out"])

    # --- cost model from stored runs (deduplicated by origin run; wall and cpu are copied into cache-hit records)
    uniq: dict[str, dict] = {}
    for r in runs:
        key = r["orig"] or r["run"]
        if key not in uniq or (uniq[key]["hit"] and not r["hit"]):
            uniq[key] = r
    mean_wall: dict[tuple[str, str], float] = {}
    by_dr: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_r: dict[str, list[float]] = defaultdict(list)
    for r in uniq.values():
        d = design_of.get(r["in"])
        if d is None or r["wall"] <= 0:
            continue
        by_dr[(d, r["recipe"])].append(r["wall"])
        by_r[r["recipe"]].append(r["wall"])
    for d in ec.DESIGNS:
        for rec in ec.REC10:
            L = by_dr.get((d, rec)) or by_r[rec]
            mean_wall[(d, rec)] = statistics.fmean(L)

    # --- enumerate
    per_design = {}
    tasks_by_design: dict[str, list[dict]] = {}
    reuse_rows = []
    for d in ec.DESIGNS:
        s0 = ec.B2[d]
        level1: dict[str, list[str]] = {}
        missing_edges = []
        noop1 = []
        for r1 in ec.REC10:
            L = by_in.get((s0, r1))
            if not L:
                missing_edges.append([s0[:10], r1]); continue
            o = pick(L)["out"]
            level1.setdefault(o, []).append(r1)
            if o == s0:
                noop1.append(r1)
        level2: dict[str, list[tuple[str, str]]] = {}
        for s1 in level1:
            if s1 == s0:
                continue                      # a no-op step: its children are the depth-1 states already
            for r2 in ec.REC10:
                L = by_in.get((s1, r2))
                if not L:
                    missing_edges.append([s1[:10], r2]); continue
                o = pick(L)["out"]
                level2.setdefault(o, []).append((s1, r2))
        seen_lo = set(level1) | {s0}
        two_step = {k: v for k, v in level2.items() if k not in seen_lo}   # distinct states reachable only at depth 2
        classes: dict[str, list[str]] = defaultdict(list)
        for k in sorted(two_step):
            classes[uf.find(k)].append(k)
        tasks, n_have, n_class_merged = [], 0, 0
        for cls, mem in sorted(classes.items()):
            rep = None
            for m in mem:                    # representative: a member whose checkpoint exists in some root
                src, meta = ec.state_source(m, ec.SRC_ROOTS)
                if src:
                    rep, rep_src, rep_meta = m, src, meta
                    break
            if rep is None:
                raise SystemExit(f"no checkpoint file for class {cls[:12]} of {d}")
            if len(mem) > 1:
                n_class_merged += len(mem) - 1
            for rec in ec.REC10:
                stored = by_cls_in.get((cls, rec))
                if stored:
                    r = pick(stored)
                    n_have += 1
                    reuse_rows.append({"design": d, "state": rep, "class": cls, "recipe": rec, "stored_input": r["in"],
                                       "exact": r["in"] == rep, "root": r["root"], "run": r["run"], "orig": r["orig"],
                                       "output": r["out"], "wns_ns": r["wns"], "packed_luts": r["luts"],
                                       "wall_s": r["wall"]})
                    continue
                tasks.append({
                    "phase": 1, "kind": "run", "design": d, "state": rep, "class": cls, "recipe": rec, "src": rep_src,
                    "parent_id": (rep_meta or {}).get("parent_id"), "made_by_run_id": (rep_meta or {}).get("made_by_run_id"),
                    "est_wall_s": round(mean_wall[(d, rec)], 1),
                })
        tasks.sort(key=lambda t: (-t["est_wall_s"], t["state"], t["recipe"]))     # longest first inside a design
        tasks_by_design[d] = tasks
        per_design[d] = {
            "depth1_distinct_states": len(level1), "depth1_noop_recipes": noop1,
            "depth2_edges_found": sum(len(v) for v in level2.values()), "depth2_missing_edges": missing_edges,
            "depth2_distinct_states": len(level2), "two_step_only_states": len(two_step), "two_step_classes": len(classes),
            "twin_members_merged": n_class_merged,
            "pairs_total": len(classes) * len(ec.REC10), "pairs_stored": n_have, "pairs_to_run": len(tasks),
            "est_job_hours": round(sum(t["est_wall_s"] for t in tasks) / 3600, 1),
        }

    order = sorted(ec.DESIGNS, key=lambda d: per_design[d]["est_job_hours"])
    phase1 = [t for d in order for t in tasks_by_design[d]]
    for i, t in enumerate(phase1):
        t["seq"] = i

    # --- stored-run telemetry for the estimate: load and busy slots at start, cpu per wall
    loads = [r["load0"] for r in uniq.values() if r.get("load0") is not None]
    busy = [r["busy0"] for r in uniq.values() if r.get("busy0") is not None]
    cpu_ratio = [((r["cu"] or 0) + (r["cs"] or 0)) / r["wall"] for r in uniq.values()
                 if r["recipe"].startswith("A_") and r["cu"] is not None and r["wall"] > 0]
    wall_all = [r["wall"] for r in uniq.values() if r["wall"] > 0]
    stats = {
        "unique_stored_runs": len(uniq), "mean_wall_s": statistics.fmean(wall_all),
        "wall_p50": pct(wall_all, .5), "wall_p90": pct(wall_all, .9), "wall_max": max(wall_all),
        "load_1m_at_start": {"mean": statistics.fmean(loads), "p50": pct(loads, .5), "p90": pct(loads, .9), "max": max(loads)},
        "busy_slots_at_start": {"mean": statistics.fmean(busy), "p50": pct(busy, .5), "p90": pct(busy, .9), "max": max(busy)},
        "area_recipe_cpu_per_wall": {"mean": statistics.fmean(cpu_ratio), "p90": pct(cpu_ratio, .9), "n": len(cpu_ratio)},
        "mean_wall_by_recipe_s": {rec: round(statistics.fmean(by_r[rec]), 1) for rec in ec.REC10},
        "mean_wall_by_design_recipe_s": {f"{d}|{rec}": round(mean_wall[(d, rec)], 1) for d in ec.DESIGNS for rec in ec.REC10},
    }
    ec.jwrite(out_dir / "tasks_phase1.json", phase1, indent=None)
    ec.jwrite(out_dir / "reuse_phase1.json", reuse_rows)
    ec.jwrite(out_dir / "merges.json", merges)
    ec.jwrite(out_dir / "summary.json", {
        "design_order_cheapest_first": order, "per_design": per_design, "skipped_runs_by_reason": skipped,
        "accepted_runs": len(runs), "twin_classes_with_merges": len(merges), "stored_run_stats": stats,
        "recipe_versions": vers, "meter_version": ec.CUR_METER, "roots": {k: str(v) for k, v in roots.items()},
    })
    print("design order (cheapest first):", order)
    tot_runs = tot_h = 0
    for d in order:
        p = per_design[d]
        tot_runs += p["pairs_to_run"]; tot_h += p["est_job_hours"]
        print(f"{d:28s} d1={p['depth1_distinct_states']:2d} (noop {len(p['depth1_noop_recipes'])}) d2states={p['depth2_distinct_states']:3d} "
              f"two-step-only={p['two_step_only_states']:3d} classes={p['two_step_classes']:3d} pairs={p['pairs_total']:4d} "
              f"stored={p['pairs_stored']:3d} to_run={p['pairs_to_run']:4d} est_h={p['est_job_hours']:6.1f} missing_d2={len(p['depth2_missing_edges'])}")
    print(f"TOTAL to run {tot_runs}, est job-hours {tot_h:.1f}; twin merges {len(merges)}; accepted runs {len(runs)}; skipped {skipped}")
    print(json.dumps({k: v for k, v in stats.items() if not k.startswith("mean_wall_by")}, indent=1, default=str) if (json := __import__("json")) else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
