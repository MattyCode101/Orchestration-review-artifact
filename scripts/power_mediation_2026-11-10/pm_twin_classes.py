"""Twin classes (read-only): states that are the outputs of repeated runs of the same logical edge.

Two runs of the same recipe version on the same (class of) input state can give byte-different outputs. The
engine keys everything by byte hash, so such 'twins' split the graph. This script merges, by union-find, the
outputs of every group of runs that share (input class, recipe, version), repeatedly until stable, then
checks that every class has a single (WNS, packed LUTs, dynamic power) triple. It changes nothing; it only
counts, for the status audit and for deciding whether the stage-1 tasks duplicate an existing twin edge.
usage: pm_twin_classes.py <out.json> [--no-new]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_common as pm  # noqa: E402

M = pm.MetricName


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra
            return True
        return False


def all_runs(roots):
    """Every completed OK run in every root (not only the pointer winners)."""

    runs = []
    for name, root in roots.items():
        runs_dir = root / "runs"
        for ptr in sorted((root / "cache").glob("*.json")):
            p = pm.jread(ptr)
            if p.get("retryable"):
                continue
            rd = runs_dir / p["run_id"]
            if not (rd / "complete.json").is_file():
                continue
            info = pm.jread(rd / "runinfo.json")
            if info["status"] != "ok" or not (rd / "output_state.json").is_file():
                continue
            out = pm.jread(rd / "output_state.json")["id"]
            met = None
            if (rd / "evidence.json").is_file():
                ev = pm.Evidence.model_validate_json((rd / "evidence.json").read_bytes())
                met = (ev.metric(M.WNS_NS).value, ev.metric(M.PACKED_LUTS).value, ev.metric(M.DYNAMIC_POWER_W).value)
            runs.append({"root": name, "run": p["run_id"], "in": info["input_sha256_before"], "recipe": info["recipe_id"],
                         "ver": info["recipe_version"], "out": out, "metrics": met})
    return runs


def _fixpoint(uf, runs):
    while True:
        groups = defaultdict(set)
        for r in runs:
            groups[(uf.find(r["in"]), r["recipe"], r["ver"])].add(r["out"])
        changed = False
        for outs in groups.values():
            outs = sorted(outs)
            for o in outs[1:]:
                changed |= uf.union(outs[0], o)
        if not changed:
            return


def build(ctx, link_selection: bool = True):
    """Union-find over twins: returns (uf, runs).

    Runs that repeat the same logical edge (same input class, recipe, version) have their outputs merged.
    With link_selection, the depth-1 base states the power campaign selected (training power selection.json,
    role D1, one source recipe) are also linked to the output of that recipe on the same B2 in the sealed
    graph: their producing runs sit in an older root that is not part of the union, and the 19 pairs checked
    earlier are metric-identical (WNS, LUTs, power) in all cases. That link is an interpretation, flagged as such.
    """

    runs = all_runs(ctx["index"].roots)
    uf = UF()
    for r in runs:
        uf.find(r["in"])
        uf.find(r["out"])
    _fixpoint(uf, runs)
    if link_selection:
        sel = pm.jread(pm.POWER_TOP / "selection.json")["by_design"]
        for design, b2 in ctx["b2"].items():
            for st in sel[design]["states"]:
                seq = st.get("source_sequences") or []
                if st["role"] != "D1" or len(seq) != 1:
                    continue
                outs = {r["out"] for r in runs if uf.find(r["in"]) == uf.find(b2) and r["recipe"] == seq[0]}
                for o in outs:
                    uf.union(o, st["hash"])
        _fixpoint(uf, runs)
    return uf, runs


def flag_tasks(src: Path, dst: Path) -> None:
    """Rewrite a task list: AMD first, then new logical edges (power recipes first), then twin duplicates last."""

    ctx = pm.build_context(include_new=False)
    uf, runs = build(ctx)
    have = {(uf.find(r["in"]), r["recipe"]) for r in runs}
    tasks = json.loads(src.read_text())
    for t in tasks:
        if t["kind"] == "run":
            t["twin_duplicate"] = (uf.find(t["state"]), t["recipe"]) in have
    rank = lambda t: (0 if t["kind"] == "amd" else (2 if t.get("twin_duplicate") else 1),
                      0 if (t.get("recipe") or "").startswith("P_") else 1, t["design"], t["state"], t.get("recipe") or "")
    tasks.sort(key=rank)
    dst.write_text(json.dumps(tasks, indent=1))
    print(Counter((t["kind"], t.get("twin_duplicate")) for t in tasks))


def main() -> int:
    if sys.argv[1] == "--flag-tasks":
        flag_tasks(Path(sys.argv[2]), Path(sys.argv[3]))
        return 0
    out_path = Path(sys.argv[1])
    ctx = pm.build_context(include_new="--no-new" not in sys.argv)
    uf, runs = build(ctx)
    rounds = 0
    # metric identity per class
    classes = defaultdict(set)
    mets = defaultdict(set)
    for r in runs:
        classes[uf.find(r["out"])].add(r["out"])
        if r["metrics"] is not None:
            mets[uf.find(r["out"])].add(r["metrics"])
    multi = {c: s for c, s in classes.items() if len(s) > 1}
    bad = [c for c in multi if len(mets[c]) > 1]
    print("runs", len(runs), "| union-find rounds", rounds)
    print("states with more than one byte version (twin classes):", len(multi), "| states inside them:", sum(len(s) for s in multi.values()))
    print("twin classes whose members disagree on (WNS, LUTs, power):", len(bad))
    for c in bad[:5]:
        print("   ", [tuple(round(x, 4) for x in m) for m in sorted(mets[c])])
    # logical edges
    logical = {(uf.find(r["in"]), r["recipe"], r["ver"]) for r in runs}
    b2_classes = {uf.find(s): d for d, s in ctx["b2"].items()}
    reach, queue = set(b2_classes), list(b2_classes)
    edges_by_in = defaultdict(list)
    for r in runs:
        edges_by_in[uf.find(r["in"])].append(r)
    seen_logical_reach = set()
    while queue:
        c = queue.pop()
        for r in edges_by_in.get(c, []):
            seen_logical_reach.add((c, r["recipe"], r["ver"]))
            o = uf.find(r["out"])
            if o not in reach:
                reach.add(o); queue.append(o)
    print("logical edges (distinct input class, recipe, version):", len(logical), "| reachable from B2 by classes:", len(seen_logical_reach))
    # byte-level reachability for comparison (state ids)
    key_edges = {(r["in"], r["recipe"], r["ver"]) for r in runs}
    print("distinct byte-level (input state, recipe, version) edges:", len(key_edges))
    # do the stage-1 tasks duplicate an existing twin edge?
    tasks_path = pm.WORK / "tasks_stage1.json"
    dup = Counter()
    if tasks_path.is_file():
        reg = ctx["registry"]
        have = {(uf.find(r["in"]), r["recipe"]) for r in runs}
        for t in json.loads(tasks_path.read_text()):
            if t["kind"] == "run":
                key = (uf.find(t["state"]), t["recipe"])
                dup["stage-1 run tasks"] += 1
                dup["of which a twin edge already exists"] += key in have
    print(dict(dup))
    out_path.write_text(json.dumps({"runs": len(runs), "twin_classes": len(multi), "bad_classes": len(bad),
                                    "logical_edges": len(logical), "logical_edges_reachable": len(seen_logical_reach),
                                    "byte_edges": len(key_edges), "stage1": dict(dup)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
