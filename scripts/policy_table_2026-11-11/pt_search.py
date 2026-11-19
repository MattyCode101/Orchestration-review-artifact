"""Search rows for Table 1, from stored outcomes only (no run, no model call). Read-only.

(a) exhaustive two-step: every one- and two-step sequence of the timing and area recipes from the start
    (5 timing + 5 area recipes: 10 one-step and 100 two-step sequences per design), best valid state visited.
(b) grid search with 6 runs: the 6 single recipes ranked highest by mean training gain for the brief's goal
    (training depth-one outcomes on the five training designs), run once each from the start, best valid state kept.

Outcomes come from the hash-pinned test cache manifest (root test_base only: the power roots are never opened) and the
training cache manifest (root v4). Each run is charged its stored original cost. A sequence whose stored outcome is
missing is reported as missing and the row is marked incomplete; nothing is filled in.
Validity as in the campaign: the brief's limits against the start (orch.engine.gate.judge), no more DONT_TOUCH / fixed
LOC / fixed BEL cells than the start, fully routed and clean; AMD PASS on top. Two AMD readings are reported:
  campaign   = PASS record in the campaign ledger (reference = the campaign start state): the frame's own rule
  any_record = also a PASS record in the baseline root (made against another reference hash; weaker evidence)
  none       = metrics only (AMD not checked)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pt_common as pt  # noqa: E402

from orch.contract.brief_text import load_frozen_brief  # noqa: E402
from orch.contract.models import Evidence, MetricName  # noqa: E402
from orch.engine.gate import judge  # noqa: E402
from orch.engine.manifest_cache import _read_run  # noqa: E402
from orch.engine.start_properties import PropertyRuleError, check_not_above_start  # noqa: E402

TIMING = ("LATCH.phys_opt", "LATCH.retime", "LATCH.auto_pblock", "LATCH.hardblock_relocate", "LATCH.memory_path_opt")
AREA = ("A_R_EA", "A_R_RSA_SRLM", "A_R_RSA_AREM", "A_R_RSA_MED", "A_E_DUPL_Q1000")
RECIPES = TIMING + AREA
TEST_MANIFEST = pt.PROJECT / "data/test_campaign_prep_2026-11-06/cache_manifest.json"
TRAIN_MANIFEST = pt.PROJECT / "data/campaign_prep_2026-11-07/train_cache_manifest.json"
BASE_ROOT = pt.PROJECT / "data/exp_evening_2026-11-04/test_b2/overnight_baseline_2026-11-04"
BRIEFS = {b: load_frozen_brief(b) for b in pt.FOUR}


class Graph:
    """The stored run graph of one manifest root (A/B recipes only)."""

    def __init__(self, manifest_path: Path, root_name: str, recipes: tuple[str, ...]) -> None:
        man = pt.jload(manifest_path)
        self.root = Path(man["roots"][root_name])
        self.by_in: dict[tuple[str, str], dict[str, Any]] = {}
        for e in man["entries"].values():
            if e["root"] != root_name or e["recipe_id"] not in recipes or e["status"] != "ok":
                continue
            key = (e["input_state_id"], e["recipe_id"])
            if key in self.by_in:
                raise SystemExit(f"duplicate key {key}")
            self.by_in[key] = e
        self.start = {}
        self.start_id = {}
        for sid, v in man["start_evidence"].items():
            if v["root"] == root_name:
                self.start[v["design"]] = Evidence.model_validate_json(Path(v["path"]).read_bytes())
                self.start_id[v["design"]] = sid
        self._ev: dict[str, tuple[Evidence, dict[str, Any]]] = {}

    def evidence(self, entry: dict[str, Any]) -> Evidence:
        rid = entry["run_id"]
        if rid not in self._ev:
            run, state, ev, _ = _read_run(self.root / "runs" / rid, rid)
            assert state is not None and ev is not None and state.id == entry["output_state_id"], rid
            self._ev[rid] = (ev, {"wall_s": float(run.wall_s or 0.0), "cpu_s": float((run.cpu_user_s or 0.0) + (run.cpu_sys_s or 0.0))})
        return self._ev[rid][0]

    def cost(self, entry: dict[str, Any]) -> dict[str, float]:
        c = entry["cost"]
        return {"wall_s": float(c.get("wall_s") or 0.0), "cpu_s": float((c.get("cpu_user_s") or 0.0) + (c.get("cpu_sys_s") or 0.0))}


def amd_sets() -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    camp, base = set(), set()
    for p in (pt.ENGINE / "verification" / "records").glob("*/*/*.json"):
        r = pt.jload(p)
        if r.get("result") == "pass":
            camp.add((r["design"], r["output_sha256"]))
    for p in (BASE_ROOT / "verification" / "records").glob("*/*/*.json"):
        r = pt.jload(p)
        if r.get("result") == "pass":
            base.add((r["design"], r["output_sha256"]))
    return camp, base


def is_valid(brief_id: str, ev: Evidence, start: Evidence) -> bool:
    try:
        check_not_above_start(ev, start)
    except PropertyRuleError:
        return False
    return bool(judge(BRIEFS[brief_id], start, start, ev).brief_valid)


def pick(brief_id: str, cands: list[dict[str, Any]], start_ev: Evidence, amd_mode: str, camp, base, design: str):
    """Best valid candidate: objective first, then the better non-objective metric, then fewer steps."""

    metric, direction = pt.OBJECTIVE[brief_id]
    other = pt.LIMIT_METRIC[brief_id]
    other_dir = "max" if other == "wns_ns" else "min"
    best = None
    for c in cands:
        ev = c["ev"]
        if not c["valid"][brief_id]:
            continue
        if amd_mode == "campaign" and (design, c["state"]) not in camp and c["state"] != c["start_id"]:
            continue
        if amd_mode == "any_record" and (design, c["state"]) not in camp and (design, c["state"]) not in base and c["state"] != c["start_id"]:
            continue
        v = ev.metric(MetricName(metric)).value
        o = ev.metric(MetricName(other)).value
        key = ((-v if direction == "max" else v), (-o if other_dir == "max" else o), c["steps"])
        if best is None or key < best[0]:
            best = (key, c)
    return best[1] if best else None


def metrics_of(ev: Evidence) -> dict[str, float]:
    return {n.value: m.value for n, m in ev.metrics.items()}


def candidates_for_design(g: Graph, design: str, recipes_first: tuple[str, ...], recipes_second: tuple[str, ...], camp, base):
    """Start + depth-one outputs (+ depth-two outputs if recipes_second). Returns (cands, steps, missing)."""

    sid = g.start_id[design]
    start_ev = g.start[design]
    cands = [{"state": sid, "start_id": sid, "ev": start_ev, "steps": 0, "path": [], "valid": {b: True for b in pt.FOUR}}]
    steps, missing = [], []
    for r1 in recipes_first:
        e1 = g.by_in.get((sid, r1))
        if e1 is None:
            missing.append([r1])
            continue
        ev1 = g.evidence(e1)
        steps.append({"seq": [r1], **g.cost(e1), "run_id": e1["run_id"], "state": e1["output_state_id"]})
        cands.append({"state": e1["output_state_id"], "start_id": sid, "ev": ev1, "steps": 1, "path": [r1],
                      "valid": {b: is_valid(b, ev1, start_ev) for b in pt.FOUR}})
        for r2 in recipes_second:
            e2 = g.by_in.get((e1["output_state_id"], r2))
            if e2 is None:
                missing.append([r1, r2])
                continue
            ev2 = g.evidence(e2)
            steps.append({"seq": [r1, r2], **g.cost(e2), "run_id": e2["run_id"], "state": e2["output_state_id"]})
            cands.append({"state": e2["output_state_id"], "start_id": sid, "ev": ev2, "steps": 2, "path": [r1, r2],
                          "valid": {b: is_valid(b, ev2, start_ev) for b in pt.FOUR}})
    return cands, steps, missing


def row_for(design: str, brief_id: str, c: dict[str, Any] | None, frame, starts, start_ev: Evidence):
    f = frame[(design, brief_id)]
    metric, direction = pt.OBJECTIVE[brief_id]
    if c is None:
        return None
    m = metrics_of(c["ev"])
    return {"state": c["state"], "path": c["path"], "metrics": m,
            "share": None if f["no_room"] else pt.share(f["start"], f["best_known"], m[metric], direction),
            "gain": pt.gain(brief_id, f["start"], m[metric])}


def train_ranking() -> dict[str, Any]:
    """Mean training gain per single recipe for the goals of T0 (WNS up) and A0 (packed LUTs down)."""

    g = Graph(TRAIN_MANIFEST, "v4", RECIPES)
    designs = sorted(g.start)
    table: dict[str, dict[str, Any]] = {r: {"wns_gain_ns": {}, "lut_cut_pct": {}} for r in RECIPES}
    missing = []
    for d in designs:
        s = g.start[d]
        sw, sl = s.metric(MetricName("wns_ns")).value, \
            s.metric(MetricName("packed_luts")).value
        for r in RECIPES:
            e = g.by_in.get((g.start_id[d], r))
            if e is None:
                missing.append([d, r])
                continue
            m = metrics_of(g.evidence(e))
            table[r]["wns_gain_ns"][d] = m["wns_ns"] - sw
            table[r]["lut_cut_pct"][d] = 100.0 * (sl - m["packed_luts"]) / sl
    out: dict[str, Any] = {"designs": designs, "missing": missing, "table": table}
    for goal, key in (("T0", "wns_gain_ns"), ("A0", "lut_cut_pct")):
        means = {r: pt.mean(table[r][key].values()) for r in RECIPES}
        counts = {r: len(table[r][key]) for r in RECIPES}
        order = sorted(RECIPES, key=lambda r: (-(means[r] if means[r] is not None else -1e18), r))
        out[goal] = {"mean_gain": means, "n_designs": counts, "ranking": order, "top6": order[:6]}
    # sensitivity: band-normalised ranking (gain / noise band of the design's metric)
    try:
        from scripts import friday_analysis as fa  # noqa: E402

        bands = fa.noise_bands()
        starts = fa.start_metrics()
        for goal, key, metric in (("T0", "wns_gain_ns", "wns_ns"), ("A0", "lut_cut_pct", "packed_luts")):
            norm = {}
            for r in RECIPES:
                vals = []
                for d, v in table[r][key].items():
                    band = bands.get(d, {}).get(metric)
                    if not band:
                        continue                      # a design without a recorded band (logicnets) is left out of this check
                    gain_abs = v if key == "wns_gain_ns" else v / 100.0 * starts[d]["packed_luts"]
                    vals.append(gain_abs / band)
                norm[r] = pt.mean(vals)
            order = sorted(RECIPES, key=lambda r: (-(norm[r] if norm[r] is not None else -1e18), r))
            out[goal]["band_normalised_mean"] = norm
            out[goal]["band_normalised_top6"] = order[:6]
            out[goal]["top6_same_set"] = set(order[:6]) == set(out[goal]["top6"])
    except Exception as exc:  # sensitivity only
        out["sensitivity_error"] = f"{type(exc).__name__}: {exc}"
    return out


def run() -> dict[str, Any]:
    camp, base = amd_sets()
    frame = pt.load_frame()
    g = Graph(TEST_MANIFEST, "test_base", RECIPES)
    rank = train_ranking()
    starts = {}
    for d in pt.DESIGNS:
        starts[d] = metrics_of(g.start[d])
    out: dict[str, Any] = {"train_ranking": rank, "amd": {"campaign_pass_records": len(camp), "baseline_pass_records": len(base)},
                           "exhaustive": {}, "grid": {}}
    for d in pt.DESIGNS:
        start_ev = g.start[d]
        # ---- exhaustive two-step
        cands, steps, missing = candidates_for_design(g, d, RECIPES, RECIPES, camp, base)
        distinct = {s["run_id"] for s in steps}
        rec = {"steps_expected": 10 + 100, "steps_found": len(steps), "distinct_runs": len(distinct), "missing": missing,
               "wall_s_sum": sum(s["wall_s"] for s in steps), "wall_s_max": max((s["wall_s"] for s in steps), default=0.0),
               "cpu_s_sum": sum(s["cpu_s"] for s in steps),
               "wall_s_sum_distinct": sum(s["wall_s"] for s in {x["run_id"]: x for x in steps}.values()),
               "valid_counts": {b: sum(c["valid"][b] for c in cands[1:]) for b in pt.FOUR}, "picks": {}}
        for b in pt.FOUR:
            rec["picks"][b] = {}
            for mode in ("none", "any_record", "campaign"):
                c = pick(b, cands, start_ev, mode, camp, base, d)
                rec["picks"][b][mode] = row_for(d, b, c, frame, starts, start_ev)
                if rec["picks"][b][mode]:
                    rec["picks"][b][mode]["amd"] = ("start" if c["state"] == c["start_id"] else
                                                    "campaign" if (d, c["state"]) in camp else "baseline" if (d, c["state"]) in base else "no record")
        out["exhaustive"][d] = rec
        # ---- grid search with 6 runs, per brief (T0 and T share the WNS ranking; A0 and A the LUT ranking)
        out["grid"][d] = {}
        for b in pt.FOUR:
            goal = "T0" if pt.OBJECTIVE[b][0] == "wns_ns" else "A0"
            six = tuple(rank[goal]["top6"])
            cands6, steps6, missing6 = candidates_for_design(g, d, six, (), camp, base)
            r6 = {"recipes": list(six), "steps_found": len(steps6), "missing": missing6,
                  "wall_s_sum": sum(s["wall_s"] for s in steps6), "wall_s_max": max((s["wall_s"] for s in steps6), default=0.0),
                  "cpu_s_sum": sum(s["cpu_s"] for s in steps6), "picks": {}}
            for mode in ("none", "any_record", "campaign"):
                c = pick(b, cands6, start_ev, mode, camp, base, d)
                r6["picks"][mode] = row_for(d, b, c, frame, starts, start_ev)
                if r6["picks"][mode]:
                    r6["picks"][mode]["amd"] = ("start" if c["state"] == c["start_id"] else
                                                "campaign" if (d, c["state"]) in camp else "baseline" if (d, c["state"]) in base else "no record")
            out["grid"][d][b] = r6
        # sensitivity: the same grid with the six recipes ranked by band-normalised training gain
        out.setdefault("grid_bn", {})[d] = {}
        for b in pt.FOUR:
            goal = "T0" if pt.OBJECTIVE[b][0] == "wns_ns" else "A0"
            six = tuple(rank[goal].get("band_normalised_top6", rank[goal]["top6"]))
            cands6, steps6, missing6 = candidates_for_design(g, d, six, (), camp, base)
            r6 = {"recipes": list(six), "steps_found": len(steps6), "missing": missing6,
                  "wall_s_sum": sum(s["wall_s"] for s in steps6), "wall_s_max": max((s["wall_s"] for s in steps6), default=0.0),
                  "cpu_s_sum": sum(s["cpu_s"] for s in steps6), "picks": {}}
            for mode in ("none", "any_record", "campaign"):
                c = pick(b, cands6, start_ev, mode, camp, base, d)
                r6["picks"][mode] = row_for(d, b, c, frame, starts, start_ev)
                if r6["picks"][mode]:
                    r6["picks"][mode]["amd"] = ("start" if c["state"] == c["start_id"] else
                                                "campaign" if (d, c["state"]) in camp else "baseline" if (d, c["state"]) in base else "no record")
            out["grid_bn"][d][b] = r6
    # why a single-step grid fails A0: how many LUT-cutting single area steps keep WNS at or above the start
    cut = keep = 0
    for d in pt.DESIGNS:
        s0, sid = metrics_of(g.start[d]), g.start_id[d]
        for r in AREA:
            e = g.by_in.get((sid, r))
            if e is None:
                continue
            m = metrics_of(g.evidence(e))
            if s0["packed_luts"] - m["packed_luts"] > 0:
                cut += 1
                keep += int(m["wns_ns"] - s0["wns_ns"] >= -1e-12)
    out["single_area_limit_check"] = {"lut_cutting_steps": cut, "keep_wns_at_start": keep}
    out["starts"] = starts
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=pt.WORK / "search_rows.json")
    args = ap.parse_args()
    out = run()
    pt.WORK.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    r = out["train_ranking"]
    print("training ranking T0:", r["T0"]["ranking"], {k: round(v, 3) for k, v in r["T0"]["mean_gain"].items()})
    print("training ranking A0:", r["A0"]["ranking"], {k: round(v, 3) for k, v in r["A0"]["mean_gain"].items()})
    print("top6 same under band-normalisation:", {g: r[g].get("top6_same_set") for g in ("T0", "A0")}, r.get("sensitivity_error"))
    print("training missing:", r["missing"])
    print("AMD PASS records:", out["amd"])
    for d, rec in out["exhaustive"].items():
        print(f"{d}: steps {rec['steps_found']}/{rec['steps_expected']} distinct {rec['distinct_runs']} missing {len(rec['missing'])} "
              f"wall_sum {rec['wall_s_sum'] / 3600:.1f}h max {rec['wall_s_max'] / 3600:.2f}h valid {rec['valid_counts']}")
        for b in pt.FOUR:
            row = []
            for mode in ("none", "any_record", "campaign"):
                p = rec["picks"][b][mode]
                row.append("-" if p is None else f"{mode}:{'' if p['share'] is None else round(100 * p['share'], 1)}({p['amd']},{'>'.join(x.split('.')[-1][:8] for x in p['path']) or 'start'})")
            print("   ", b, " | ".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
