"""COPY of tr_review.py with the dot-in-filename fix (Sun 15 Nov 2026; the original is pinned and untouched). See make_review2.py for the four changes.

Power review (queue 2): the declared measures from data/test_r2_2026-11-13/power_review/results, read-only. Evaluation only; changes no policy.

Cells: 5 TRAINING + 3 VALIDATION designs x W2R, L2R, P0. Rows: goal-repair v2, the declared fixed order (ABC for W2R and L2R, CBA for P0, chosen on
TRAINING, so in-sample there), F1-R1 and v2.1 (one repeat, temperature 0.3, Gemini flash-lite).

Gain: W2R = WNS gain in ns; L2R = packed-LUT cut; P0 = dynamic-power cut in W. A gain counts as an improvement only beyond the design's noise band
(configs/noise_round2_trainpower.yaml); power values are compared in whole milliwatts (the unit of the recorded bands), so a gain of exactly the
band is no change (the P0 reporting rule; same rule as scripts/test_r2_2026-11-13/tr_rules.py).
References (labelled in every table):
  VALIDATION  exhaustive two-step over the 14 recipes, best brief-valid state with an AMD PASS (data/development_designs_2026-11-12/exhaustive.json).
  TRAINING    the stored six-run ceiling: the exact search over the stored training graph of the power-readiness study (data/power_readiness_2026-11-11/
              final_A4.json), a hindsight bound on an incomplete graph; W2R ceilings are in ps, P0 in mW (converted here).
Share of reference = mean over the cells whose reference gain is above one band of 100 x gain / reference gain (a cell with no gain counts 0).
W/T/L of A against B in one cell: W if A's gain exceeds B's by more than the band, L if B's exceeds A's by more than the band, else T.
usage: tr_review.py [--out DIR]
"""
from __future__ import annotations

import collections
import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("TR_QUEUE", "power")
PROJECT = Path(__file__).resolve().parents[2]
for p in (PROJECT, PROJECT / "scripts", PROJECT / "scripts/test_r2_2026-11-13"):
    sys.path.insert(0, str(p))
import tr_common as tc  # noqa: E402

GOAL = {"W2R": ("wns_ns", "max"), "L2R": ("packed_luts", "min"), "P0": ("dynamic_power_w", "min")}
BRIEFS = ("W2R", "L2R", "P0")
ROWS = ("v21", "F1-R1", "v2", "fixed")
LLM = ("v21", "F1-R1")
GROUPS = ("TRAINING", "VALIDATION")
OUT = Path(os.environ["TR_REVIEW_OUT"]) if os.environ.get("TR_REVIEW_OUT") else tc.ROOT / "report2"


def gain(brief: str, start: float, final: float) -> float:
    return final - start if GOAL[brief][1] == "max" else start - final


def mw(x: float) -> int:
    return round(float(x) * 1000.0)


def beyond(brief: str, g: float, band: float) -> bool:
    """A gain beyond one band; power in whole mW."""
    return mw(g) > mw(band) if GOAL[brief][0] == "dynamic_power_w" else g > band


def diff_beyond(brief: str, a: float, b: float, band: float) -> int:
    d = (mw(a) - mw(b), mw(band)) if GOAL[brief][0] == "dynamic_power_w" else (a - b, band)
    return 1 if d[0] > d[1] else (-1 if -d[0] > d[1] else 0)


def load() -> tuple[list[dict], dict]:
    rows, status = [], collections.Counter()
    tasks = {t["episode_id"]: t for t in tc.tasks()}
    for path in sorted(tc.RESULTS.glob("tr_*.json")):
        r = tc.jread(path)
        t = r.get("task") or {}
        if path.name != f"{t.get('episode_id') or r.get('episode_id')}.json":      # capped / failed attempt files (name test by episode id, not by dots)
            status[(r.get("status"), t.get("approach"))] += 1
            continue
        if r.get("status") != "done":
            continue
        brief = t["brief"]
        metric = GOAL[brief][0]
        s, f = r["start_metrics"][metric], r["final_metrics"][metric]
        rows.append({**t, "start": s, "final": f, "gain": gain(brief, s, f), "band": float(r["noise"].get(metric) or 0.0),
                     "r": r, "row": t["approach"]})
    return rows, {"tasks": len(tasks), "not_done_files": dict(status)}


def references() -> dict[tuple[str, str], dict]:
    ex = tc.jread(tc.PROJECT / "data/development_designs_2026-11-12/exhaustive.json")["designs"]
    a4 = tc.jread(tc.PROJECT / "data/power_readiness_2026-11-11/final_A4.json")["briefs"]
    refs = {}
    for d in tc.DEVELOPMENT:
        for b in BRIEFS:
            pick = ex[d]["briefs"][b]["pick"]
            refs[(d, b)] = {"gain": float(pick["gain"]), "label": "exhaustive two-step (AMD PASS)"}
    for b in BRIEFS:
        for e in a4[b]:
            unit = {"ps": 1e-3, "mW": 1e-3, "LUT": 1.0}[e["unit"]]
            refs[(e["design"], b)] = {"gain": float(e["ceiling"]) * unit, "label": "stored six-run ceiling (hindsight)"}
    return refs


def limit_failures(row: dict) -> list[str]:
    """Final pick against the brief's own limits (the brief object the episode used), recomputed from the stored start and final metrics."""

    brief = tc.brief_for(row["design"], row["brief"])
    start, final = row["r"]["start_metrics"], row["r"]["final_metrics"]
    bad = []
    for term in brief.terms:
        m = term.metric.value
        thr = term.factor * start[m] + term.offset
        v = final[m]
        if m == "dynamic_power_w":
            ok = mw(v) <= mw(thr) if term.comparison == "<=" else mw(v) >= mw(thr)
        else:
            ok = v <= thr + 1e-9 if term.comparison == "<=" else v >= thr - 1e-9
        if not ok:
            bad.append(f"{m} {v:g} vs limit {thr:g}")
    return bad


def fmt(x: float | None, nd: int = 1) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def main() -> int:
    rows, meta = load()
    refs = references()
    for r in rows:
        ref = refs[(r["design"], r["brief"])]
        r["ref"], r["ref_label"] = ref["gain"], ref["label"]
        r["share"] = max(0.0, 100.0 * r["gain"] / ref["gain"]) if beyond(r["brief"], ref["gain"], r["band"]) else None
        r["improved"] = beyond(r["brief"], r["gain"], r["band"])
        r["fails"] = limit_failures(r)
    by = {(r["design"], r["brief"], r["row"]): r for r in rows}
    n_tasks = meta["tasks"]
    L = [f"# Power review tables ({len(rows)} finished episodes of {n_tasks} planned)", ""]
    rep: dict = {"episodes_done": len(rows), "tasks": n_tasks, "meta": meta, "tables": {}}

    # Table 1: share, improved, runs
    L += ["## 1. Share of the reference, cells improved beyond noise", "",
          "| Group | Brief | Row | Cells | Cells with a reference gain above one band | Share of reference (mean, %) | Improved beyond noise | New runs per episode |",
          "|---|---|---|---|---|---|---|---|"]
    t1 = []
    for g in GROUPS:
        for b in BRIEFS:
            for row in ROWS:
                sel = [r for r in rows if r["group"] == g and r["brief"] == b and r["row"] == row]
                if not sel:
                    continue
                shares = [r["share"] for r in sel if r["share"] is not None]
                e = {"group": g, "brief": b, "row": row, "cells": len(sel), "ref_cells": len(shares),
                     "share": statistics.mean(shares) if shares else None, "improved": sum(r["improved"] for r in sel),
                     "new_runs": statistics.mean(r["r"]["new_real_runs"] for r in sel)}
                t1.append(e)
                L.append(f"| {g} | {b} | {row} | {len(sel)} | {len(shares)} | {fmt(e['share'])} | {e['improved']} of {len(sel)} | {e['new_runs']:.1f} |")
    rep["tables"]["shares"] = t1

    # Table 2: W/T/L
    L += ["", "## 2. W/T/L of each LLM row against goal-repair v2 and against the fixed order (per cell, one band as the tie margin)", "",
          "| Group | Brief | Row | vs goal-repair v2 (W/T/L) | vs fixed order (W/T/L) | Cells compared |", "|---|---|---|---|---|---|"]
    t2 = []
    for g in GROUPS:
        for b in BRIEFS:
            for row in LLM:
                c = {"v2": collections.Counter(), "fixed": collections.Counter()}
                n = 0
                for d in (tc.TRAINING if g == "TRAINING" else tc.DEVELOPMENT):
                    a = by.get((d, b, row))
                    if a is None:
                        continue
                    n += 1
                    for base in ("v2", "fixed"):
                        o = by.get((d, b, base))
                        if o is not None:
                            c[base]["WTL"[{1: 0, 0: 1, -1: 2}[diff_beyond(b, a["gain"], o["gain"], a["band"])]]] += 1
                t2.append({"group": g, "brief": b, "row": row, "vs_v2": dict(c["v2"]), "vs_fixed": dict(c["fixed"]), "cells": n})
                if n:
                    L.append(f"| {g} | {b} | {row} | {c['v2']['W']}/{c['v2']['T']}/{c['v2']['L']} | {c['fixed']['W']}/{c['fixed']['T']}/{c['fixed']['L']} | {n} |")
    rep["tables"]["wtl"] = t2

    # Table 3: limits, worker C, parse failures
    L += ["", "## 3. Final picks failing a limit (must be 0), worker C calls, parse failures", "",
          "| Group | Brief | Row | Episodes | Final picks failing a limit | Final pick not flagged brief-valid | AMD check not PASS | Mandates | Worker C (power) calls | Episodes calling worker C | Decisions | Repair turns | Fallbacks (reply unusable) | Truncated decisions |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    t3 = []
    for g in GROUPS:
        for b in BRIEFS:
            for row in ROWS:
                sel = [r for r in rows if r["group"] == g and r["brief"] == b and r["row"] == row]
                if not sel:
                    continue
                mand = sum(r["r"]["loop"]["mandates"] for r in sel)
                cc = sum(sum(1 for w in r["r"]["loop"]["workers"] if w == "power") for r in sel)
                cep = sum(1 for r in sel if "power" in r["r"]["loop"]["workers"])
                ls = [r["r"].get("llm_stats") or {} for r in sel]
                e = {"group": g, "brief": b, "row": row, "episodes": len(sel), "fail_limit": sum(1 for r in sel if r["fails"]),
                     "not_valid_flag": sum(1 for r in sel if not r["r"].get("final_brief_valid")),
                     "amd_not_pass": sum(1 for r in sel if r["r"].get("validation_status") != "pass"),
                     "mandates": mand, "worker_c_calls": cc, "worker_c_episodes": cep,
                     "decisions": sum(x.get("decision_records", 0) for x in ls), "repairs": sum(x.get("repairs", 0) for x in ls),
                     "fallbacks": sum(x.get("fallbacks", 0) for x in ls), "truncated": sum(x.get("truncated_decisions", 0) for x in ls)}
                t3.append(e)
                llm = row in LLM
                L.append(f"| {g} | {b} | {row} | {len(sel)} | {e['fail_limit']} | {e['not_valid_flag']} | {e['amd_not_pass']} | {mand} | {cc} | {cep} of {len(sel)} | "
                         f"{e['decisions'] if llm else '-'} | {e['repairs'] if llm else '-'} | {e['fallbacks'] if llm else '-'} | {e['truncated'] if llm else '-'} |")
    rep["tables"]["limits_workers_parse"] = t3
    failing = [(r["episode_id"] if "episode_id" in r else f"{r['design']} {r['brief']} {r['row']}", r["fails"]) for r in rows if r["fails"]]
    L += ["", "Final picks failing a limit: " + ("none" if not failing else "; ".join(f"{k}: {', '.join(v)}" for k, v in failing))]
    notdone = meta["not_done_files"]
    L += ["", "Episodes without a finished result (attempt files): " + (", ".join(f"{k[0]} {k[1]} {v}" for k, v in sorted(notdone.items(), key=str)) or "none")]

    # Table 4: gain per cell in bands
    L += ["", "## 4. Gain per cell in noise bands (reference in bands)", "",
          "| Group | Design | Brief | Band | Reference | Reference label | v2.1 | F1-R1 | goal-repair v2 | fixed order |", "|---|---|---|---|---|---|---|---|---|---|"]
    cells = []
    for g, designs in (("TRAINING", tc.TRAINING), ("VALIDATION", tc.DEVELOPMENT)):
        for d in designs:
            for b in BRIEFS:
                got = {row: by.get((d, b, row)) for row in ROWS}
                any_r = next((x for x in got.values() if x), None)
                if any_r is None:
                    continue
                band = any_r["band"]
                unit = 1000.0 if GOAL[b][0] == "dynamic_power_w" else 1.0
                bu = band * unit

                def cellfmt(x):
                    return "-" if x is None else f"{x['gain'] / band:.1f}"
                cells.append({"group": g, "design": d, "brief": b, "band": band, "ref_gain": any_r["ref"],
                              "gains": {row: (got[row]["gain"] if got[row] else None) for row in ROWS}})
                L.append(f"| {g} | {d} | {b} | {bu:g} {'mW' if unit == 1000.0 else GOAL[b][0]} | {any_r['ref'] / band:.1f} | {any_r['ref_label']} | "
                         f"{cellfmt(got['v21'])} | {cellfmt(got['F1-R1'])} | {cellfmt(got['v2'])} | {cellfmt(got['fixed'])} |")
    rep["cells"] = cells

    # context: stop reasons, runs, tokens
    L += ["", "## 5. Context", ""]
    for row in ROWS:
        sel = [r for r in rows if r["row"] == row]
        if sel:
            stops = collections.Counter(r["r"]["stop_reason"] for r in sel)
            tok = [r["r"]["llm_stats"]["tokens"]["total"] for r in sel if r["r"].get("llm_stats")]
            L.append(f"- {row}: {len(sel)} episodes; stop reasons {dict(stops)}; runs used per episode {statistics.mean(r['r']['runs_used'] for r in sel):.1f}; "
                     f"new real runs {sum(r['r']['new_real_runs'] for r in sel)}" + (f"; tokens per episode {statistics.mean(tok):,.0f}" if tok else ""))
    caps = tc.jread(tc.ROOT / "caps.json")
    u = caps["used"]
    L += ["", f"Caps used: {u['new_real_runs']} of {tc.CAPS['new_real_runs']} new runs, {u['job_seconds'] / 3600:.1f} of {tc.CAPS['job_hours']} job-hours, "
              f"US${u['usd']:.2f} of {tc.CAPS['usd']}; stopped: {caps['stopped'] or 'none'}"]
    rep["caps"] = u
    rep["episodes"] = [{k: r[k] for k in ("design", "group", "brief", "row", "start", "final", "gain", "band", "ref", "share", "improved", "fails")}
                       for r in rows]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    tc.jwrite(OUT / "report.json", rep)
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
