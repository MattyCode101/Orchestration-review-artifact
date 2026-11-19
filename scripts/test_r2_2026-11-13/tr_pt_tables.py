"""Paper tables, step 4: every number of Tables 1 to 3, the budget table, the hypothesis tests and the noise footnotes, from cells.json and episodes.json
(tr_pt_cells.py). Offline, read-only; no tool run. Writes data/paper_tables_2026-11-15/tables.json and csv/*.csv.

Definitions (repeated under each table in the note):
  share (of the 3-step reference, or of the 2-step reference over 14 recipes for power): gain / reference gain x 100, per cell with headroom (reference gain at least one
      noise band); each cell is averaged over its repeats (or seeds) first. Mean and median are over those cells. "+- sd" is the standard deviation, over repeats, of the
      pooled share (the pooled share of repeat j is the mean over the cells of that cell's share at repeat j).
  share of the ceiling: gain / strict budget-matched ceiling gain (Section 7), over the cells with headroom whose ceiling is above zero; the ceiling is a lower bound.
  improved beyond noise: cells with headroom where the cell's mean gain exceeds the design's noise band (power: whole mW).
  W/T/L vs V9: per cell with headroom, mean gain of the row against mean gain of V9, a tie being a difference within the cell's noise band; "matched" compares repeat j of
      the row with repeat j of V9 in every cell (a row with one repeat, or a deterministic row, is held against each V9 repeat; ChatGPT has repeat 1 only and is held against V9
      repeat 1).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_pt_lib as L  # noqa: E402

C = L.jread(L.OUT / "cells.json")
CELLS, ROWS, META, STARTS = C["cells"], C["rows"], C["meta"], C["starts"]
EPS = L.jread(L.OUT / "episodes.json")
MAIN, CONTROL, POWER = ("T0", "A0"), ("T", "A"), ("W2R", "L2R", "P0")
D1_CHECK = {"V9": 56.9, "fixed": 56.8, "v2": 56.4, "v1": 55.0, "F1-R1": 49.6, "V3": 49.4, "GA-R1": 46.1, "v2.1": 45.5, "V6": 39.5, "beam": 36.7, "bandit": 34.3, "random": 29.7, "grid": 23.4}


# ------------------------------------------------------------------------------------------------ cell helpers
def hc(briefs) -> list[str]:
    """Cells with headroom for the briefs (sorted by brief then design)."""
    return sorted((k for k, c in CELLS.items() if c["brief"] in briefs and c["head"]), key=lambda k: (CELLS[k]["brief"], k))


def gains(label: str, key: str) -> list[float]:
    return [x["gain"] for x in ROWS.get(label, {}).get(key, [])]


def cell_share(label: str, key: str, ref: str = "ref") -> float | None:
    g = gains(label, key)
    return None if not g else 100.0 * (sum(g) / len(g)) / CELLS[key][ref]


def reps(label: str, briefs) -> int:
    ns = {len(gains(label, k)) for k in hc(briefs)}
    return max(ns) if ns else 0


def pooled_by_repeat(label: str, briefs, ref: str = "ref") -> list[float]:
    """Pooled share of each repeat j (mean over the headroom cells of the cell's share at repeat j). Rows with one repeat give a list of one value."""
    keys = hc(briefs)
    n = reps(label, briefs)
    out = []
    for j in range(n):
        vals = []
        for k in keys:
            g = gains(label, k)
            if g:
                vals.append(100.0 * g[min(j, len(g) - 1)] / CELLS[k][ref])
        out.append(sum(vals) / len(vals))
    return out


def stats(label: str, briefs, ref: str = "ref") -> dict:
    keys = hc(briefs)
    sh = [cell_share(label, k, ref) for k in keys if gains(label, k)]
    if not sh:
        return {"n_head": len(keys), "n": 0}
    pr = pooled_by_repeat(label, briefs, ref)
    cs = [k for k in keys if gains(label, k) and CELLS[k].get("ceil_strict", 0) > 0]
    ceil_sh = [100.0 * (sum(gains(label, k)) / len(gains(label, k))) / CELLS[k]["ceil_strict"] for k in cs] if all("ceil_strict" in CELLS[k] for k in keys) else []
    imp = sum(1 for k in keys if gains(label, k) and L.beyond(CELLS[k]["brief"], sum(gains(label, k)) / len(gains(label, k)), CELLS[k]["band"]))
    return {"n_head": len(keys), "n": len(sh), "mean": L.mean(sh), "median": L.median(sh), "sd_rep": L.sd(pr) if len(pr) > 1 else None, "n_rep": len(pr), "pooled_by_repeat": pr,
            "ceil_share": L.mean(ceil_sh) if ceil_sh else None, "ceil_n": len(ceil_sh), "improved": imp}


def wtl(label: str, briefs, vs: str = "V9", matched: bool = False) -> tuple[int, int, int] | None:
    out = []
    for k in hc(briefs):
        a, b = gains(label, k), gains(vs, k)
        if not a or not b:
            continue
        c = CELLS[k]
        if matched:
            nrep = len(b) if len(a) > 1 or label.endswith("(ChatGPT)") is False else len(b)
            if label.endswith("(ChatGPT)"):
                nrep = 1
            for j in range(nrep):
                out.append(L.tie_diff(c["brief"], a[min(j, len(a) - 1)], b[j], c["band"]))
        else:
            out.append(L.tie_diff(c["brief"], sum(a) / len(a), sum(b) / len(b), c["band"]))
    return L.sign_wtl(out) if out else None


def cost(label: str, briefs) -> dict:
    es = [e for e in EPS if L.policy_label(e) == label and e["brief"] in briefs]
    if not es:
        brs = [x for key, v in ROWS.get(label, {}).items() if CELLS.get(key, {}).get("brief") in briefs for x in v if "wall_h" in x]
        if brs:
            return {"n": len(brs), "runs": L.mean(x["runs"] for x in brs), "job_h": L.mean(x["wall_h"] for x in brs), "tokens": 0.0, "usd": 0.0}
        return {"n": 0}
    return {"n": len(es), "runs": L.mean(e["runs_used"] for e in es), "job_h": L.mean(e["wall_s"] / 3600.0 for e in es), "tokens": L.mean(e["tokens"] for e in es),
            "usd": L.mean(e["usd"] for e in es), "new_runs": L.mean(e["new_runs"] for e in es)}


# ------------------------------------------------------------------------------------------------ limit failures, worker C, power breaks
def limit_failures() -> dict[str, list[str]]:
    import os

    os.environ["TR_QUEUE"] = "llm7"        # tr_review2 defaults to the power review (TRAINING and VALIDATION designs only); the TEST designs need an LLM queue's scope. Nothing is written.
    import tr_review2 as rv

    out = {}
    for e in EPS:
        row = {"design": e["design"], "brief": e["brief"], "r": {"start_metrics": e["start"], "final_metrics": e["final"]}}
        bad = rv.limit_failures(row)
        if bad:
            out[e["eid"]] = bad
    return out


def power_break_stats() -> dict[str, dict]:
    """Per episode of W2R and L2R: states (physically valid outputs) that break the power limit while gaining a band on the goal, and how many of them were never used as the
    base of a later mandate (left unrepaired: no repair was tried from them)."""
    out = {}
    for e in EPS:
        if e["brief"] not in ("W2R", "L2R"):
            continue
        p = L.BASE / "unsealed_rest" / e["src"] / "episodes" / e["eid"] / "journal.jsonl"
        if not p.is_file():
            out[e["eid"]] = {"missing_journal": True}
            continue
        ev = [json.loads(x)["event"] for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        band = e["band"]
        term = f"{e['brief'].lower()}_limit_dynamic_power_w"
        bases = [(x["index"], x["mandate"]["base_state_id"]) for x in ev if x["type"] == "mandate"]
        broken = {}
        for x in ev:
            if x["type"] != "step":
                continue
            v = x["verdict"]
            if not v.get("physically_valid", True):
                continue
            t = (v.get("terms") or {}).get(term)
            if not t or t.get("status") != "fail":
                continue
            ds = v["delta_from_start"]
            g = ds["wns_ns"] if e["brief"] == "W2R" else -ds["packed_luts"]
            if g > band + 1e-12:
                broken.setdefault(x["step"]["output_state_id"], x["decision_index"])
        based = {sid for _i, sid in bases}
        later = {sid: any(i > di and s == sid for i, s in bases) for sid, di in broken.items()}
        out[e["eid"]] = {"break_states": len(broken), "never_based": sum(1 for s in broken if not later[s]), "based_somewhere": sum(1 for s in broken if s in based)}
    return out


# ------------------------------------------------------------------------------------------------ builders
def row_line(label: str, briefs_list: list, ref_rows: bool = False) -> dict:
    d = {"label": label, "reps": reps(label, tuple(sum(([b] for b in briefs_list), [])) if briefs_list else MAIN)}
    return d


def table_ab(main_briefs, order, label_of):
    """Table 1 block for a brief pair: per brief and pooled."""
    t = []
    for lab in order:
        if lab not in ROWS:
            continue
        r = {"label": label_of.get(lab, lab), "key": lab, "reps": max(reps(lab, (b,)) for b in main_briefs)}
        for b in main_briefs:
            r[b] = stats(lab, (b,))
            r[b]["wtl"] = wtl(lab, (b,)) if lab not in ("ref3",) else None
            r[b]["wtl_matched"] = wtl(lab, (b,), matched=True) if lab not in ("ref3",) else None
        r["pooled"] = stats(lab, main_briefs)
        r["pooled"]["wtl"] = wtl(lab, main_briefs)
        r["pooled"]["wtl_matched"] = wtl(lab, main_briefs, matched=True)
        r["cost"] = cost(lab, main_briefs)
        t.append(r)
    return t


def main() -> int:
    OUTC = L.OUT / "csv"
    OUTC.mkdir(parents=True, exist_ok=True)
    res: dict = {"schema": "tr-pt-tables-v1"}
    # ---------------- Table 1: timing and area
    t1_order = ["ref3", "ceiling", "ref2", "fixed", "grid", "beam", "v1", "v2", "bandit", "random", "V9", "F1-R1", "v2.1", "V3", "V6", "GA-R1", "V9 (ChatGPT)"]
    label_of = {"ref3": "Exhaustive 3-step (reference, AMD-passed)", "ceiling": "Budget-matched ceiling, 6 runs (lower bound)", "ref2": "Exhaustive 2-step", "fixed": "Fixed order BA",
                "grid": "Grid", "beam": "Beam (A+B)", "v1": "Goal-repair v1", "v2": "Goal-repair v2", "bandit": "Bandit", "random": "Random (3 seeds)", "V9": "LLM V9 (main)",
                "F1-R1": "LLM F1-R1", "v2.1": "LLM v2.1", "V3": "LLM V3", "V6": "LLM V6", "GA-R1": "LLM GA-R1 (direct control)", "V9 (ChatGPT)": "ChatGPT, V9 template"}
    res["label_of"] = label_of
    res["t1_main"] = table_ab(MAIN, t1_order, label_of)
    res["t1_control"] = table_ab(CONTROL, t1_order, label_of)
    # cost per task for main briefs: exhaustive rows from Section 7's per-design costs
    s7 = L.jread(L.PROJECT / "data/policy_table_section7_2026-11-13/refs.json")["designs"]
    for key, d in (("ref3", "depth3"), ("ref2", "depth2")):
        cs = [s7[x]["cost"][d] for x in L.DESIGNS]
        for tab in (res["t1_main"], res["t1_control"]):
            for r in tab:
                if r["key"] == key:
                    r["cost"] = {"n": 5, "runs": L.mean(c["runs"] for c in cs), "job_h": L.mean(c["wall_h"] for c in cs), "tokens": 0.0, "usd": 0.0}
    # regression against the D1 report (final shares pooled over the 7 cells)
    chk = {}
    for lab, v in D1_CHECK.items():
        s = stats(lab, MAIN)
        chk[lab] = {"d1": v, "here": s["mean"], "diff": s["mean"] - v}
    res["d1_check"] = chk
    # ---------------- Table 2: power
    t2_order = ["ref2p", "fixed", "grid", "beam", "v1", "v2", "bandit", "random", "V9", "F1-R1", "v2.1", "GA-R1", "V9 (ChatGPT)"]
    label2 = dict(label_of, ref2p="Exhaustive 2-step over 14 recipes (reference, AMD-checked winner)", fixed="Fixed order (ABC for W2R and L2R, CBA for P0)", beam="Beam (A+B+C adapter)")
    res["label_of_power"] = label2
    t2 = []
    for lab in t2_order:
        if lab not in ROWS:
            continue
        r = {"label": label2.get(lab, lab), "key": lab, "reps": max(reps(lab, (b,)) for b in POWER)}
        for b in POWER:
            r[b] = stats(lab, (b,), ref="ref")
            r[b]["wtl"] = wtl(lab, (b,))
            r[b]["wtl_matched"] = wtl(lab, (b,), matched=True)
        r["pooled"] = stats(lab, POWER)
        r["pooled"]["wtl"] = wtl(lab, POWER)
        r["pooled"]["wtl_matched"] = wtl(lab, POWER, matched=True)
        r["cost"] = cost(lab, POWER)
        es = [e for e in EPS if L.policy_label(e) == lab and e["brief"] in POWER]
        r["episodes"] = len(es)
        r["workerC_calls_per_episode"] = (sum(sum(1 for w in e["workers"] if w == "power") for e in es) / len(es)) if es else None
        r["workerC_episodes"] = sum(1 for e in es if "power" in e["workers"]) if es else None
        r["workerC_by_brief"] = {b: (L.mean(sum(1 for w in e["workers"] if w == "power") for e in es if e["brief"] == b) if any(e["brief"] == b for e in es) else None) for b in POWER}
        t2.append(r)
    res["t2"] = t2
    lf = limit_failures()
    res["limit_failures"] = {"n_episodes_checked": len(EPS), "failing": {k: v for k, v in lf.items()}}
    pb = power_break_stats()
    res["power_breaks"] = {"n_episodes": len(pb), "missing_journals": sum(1 for v in pb.values() if v.get("missing_journal"))}
    by = defaultdict(lambda: Counter())
    eps_by_id = {e["eid"]: e for e in EPS}
    for eid, v in pb.items():
        if v.get("missing_journal"):
            continue
        e = eps_by_id[eid]
        k = (L.policy_label(e), e["brief"])
        by[k]["episodes"] += 1
        by[k]["episodes_with_break_state"] += 1 if v["break_states"] else 0
        by[k]["break_states"] += v["break_states"]
        by[k]["never_based"] += v["never_based"]
    res["power_breaks"]["by_row_brief"] = {f"{k[0]}|{k[1]}": dict(v) for k, v in by.items()}
    res["fail_by_row_power"] = {lab: sum(1 for e in EPS if L.policy_label(e) == lab and e["brief"] in POWER and e["eid"] in lf) for lab in t2_order if lab in ROWS}
    res["fail_by_row_ab"] = {lab: sum(1 for e in EPS if L.policy_label(e) == lab and e["brief"] in L.AB_BRIEFS and e["eid"] in lf) for lab in ROWS}
    # ---------------- Table 3: LLM view and model (T0, A0)
    v_rows = [("V0", "F1-R1"), ("V1", "v2.1"), ("V2", "V2"), ("V3", "V3"), ("V4", "V4"), ("V5", "V5"), ("V6", "V6"), ("V7", "V7"), ("V8", "V8"), ("V9", "V9"), ("V10", "V10"), ("V11", "V11"),
              ("ChatGPT-V9", "V9 (ChatGPT)")]
    t3 = []
    for vname, lab in v_rows:
        es = [e for e in EPS if L.policy_label(e) == lab and e["brief"] in MAIN]
        es1 = [e for e in es if e["tag"] == "r1"]
        keys = hc(MAIN)
        # repeat 1 only
        r1 = []
        for k in keys:
            g = [x for x in ROWS[lab].get(k, []) if x["tag"] == "r1"]
            if g:
                r1.append(100.0 * g[0]["gain"] / CELLS[k]["ref"])
        full = stats(lab, MAIN)
        wr = []
        for k in keys:
            a = [x for x in ROWS[lab].get(k, []) if x["tag"] == "r1"]
            b = [x for x in ROWS["V9"].get(k, []) if x["tag"] == "r1"]
            if a and b:
                wr.append(L.tie_diff(CELLS[k]["brief"], a[0]["gain"], b[0]["gain"], CELLS[k]["band"]))
        t3.append({"name": vname, "label": lab, "episodes": len(es), "repeats": reps(lab, MAIN), "share_r1": L.mean(r1), "share_mean": full["mean"], "share_sd": full["sd_rep"], "n_head": len(r1),
                   "wtl_r1": L.sign_wtl(wr), "kept": (sum(e["kept"] or 0 for e in es) / sum(e["mandates"] or 0 for e in es)) if es else None, "kept_n": (sum(e["kept"] or 0 for e in es), sum(e["mandates"] or 0 for e in es)),
                   "early_stop": (sum(1 for e in es if e["stop"] != "budget_exhausted") / len(es)) if es else None, "early_n": (sum(1 for e in es if e["stop"] != "budget_exhausted"), len(es)),
                   "tokens": L.mean(e["tokens"] for e in es), "usd": L.mean(e["usd"] for e in es), "usd_total": sum(e["usd"] for e in es), "repair_turns": sum(e["repairs"] for e in es),
                   "fallbacks": sum(e["fallbacks"] for e in es), "truncated": sum(e["truncated"] for e in es), "model": "gateway:gpt-5.6-terra" if "ChatGPT" in lab else "gemini-3.1-flash-lite"})
    res["t3"] = t3
    # ---------------- budget sensitivity (T0, A0)
    bud = []
    for name, labs in (("V9", {3: "V9 b3", 6: "V9", 10: "V9 b10"}), ("Goal-repair v2", {3: "v2 b3", 6: "v2", 10: "v2 b10"}), ("Fixed order BA", {3: "fixed b3", 6: "fixed", 10: "fixed b10"}),
                       ("Random, seed 0", {3: "random b3", 6: "random", 10: "random b10"}), ("Beam (A+B)", {3: "beam b3", 6: "beam", 10: "beam b10"})):
        r = {"name": name}
        for b, lab in labs.items():
            if name == "Random, seed 0":
                # only seed 0 was run at budgets 3 and 10: compare like with like at budget 6 as well
                keys = hc(MAIN)
                sh = []
                for k in keys:
                    xs = [x for x in ROWS[lab].get(k, []) if (b != 6 or x["seed"] == 0)]
                    if xs:
                        sh.append(100.0 * (sum(x["gain"] for x in xs) / len(xs)) / CELLS[k]["ref"])
                r[b] = {"mean": L.mean(sh), "n": len(sh), "T0": None, "A0": None}
                for br in MAIN:
                    shb = []
                    for k in hc((br,)):
                        xs = [x for x in ROWS[lab].get(k, []) if (b != 6 or x["seed"] == 0)]
                        if xs:
                            shb.append(100.0 * (sum(x["gain"] for x in xs) / len(xs)) / CELLS[k]["ref"])
                    r[b][br] = L.mean(shb)
                r[b]["runs"] = L.mean(e["runs_used"] for e in EPS if L.policy_label(e) == lab and e["brief"] in MAIN and (b != 6 or e["seed"] == 0))
            else:
                s = stats(lab, MAIN)
                r[b] = {"mean": s.get("mean"), "n": s.get("n"), "sd_rep": s.get("sd_rep"), "T0": stats(lab, ("T0",)).get("mean"), "A0": stats(lab, ("A0",)).get("mean")}
                c = cost(lab, MAIN)
                r[b]["runs"] = c.get("runs")
                r[b]["job_h"] = c.get("job_h")
                r[b]["improved"] = s.get("improved")
        bud.append(r)
    res["budget"] = bud
    # ---------------- headroom classes and noise footnotes
    cls = {}
    for b in L.AB_BRIEFS + POWER:
        ks = sorted(k for k, c in CELLS.items() if c["brief"] == b)
        cls[b] = {"cells": len(ks), "none": sum(1 for k in ks if CELLS[k]["class"] == "none"), "low": sum(1 for k in ks if CELLS[k]["class"] == "low"), "high": sum(1 for k in ks if CELLS[k]["class"] == "high"),
                  "head": sum(1 for k in ks if CELLS[k]["head"])}
    res["classes"] = cls
    res["cells"] = {k: {kk: vv for kk, vv in c.items() if kk not in ("unverified_better",)} for k, c in CELLS.items()}
    # sd over repeats per LLM cell
    sdt = {}
    for lab in ("V9", "F1-R1", "v2.1", "V3", "V6", "GA-R1", "V11", "random"):
        for key in sorted(CELLS):
            g = gains(lab, key)
            if len(g) >= 2:
                sdt.setdefault(lab, {})[key] = {"n": len(g), "sd_gain": L.sd(g), "sd_bands": L.sd(g) / CELLS[key]["band"] if CELLS[key]["band"] else None,
                                                "sd_share": (100.0 * L.sd(g) / CELLS[key]["ref"]) if CELLS[key]["ref"] > 0 else None, "gains": g}
    res["sd_cells"] = sdt
    # inside-noise counts: head cells where the row's mean gain does not exceed the band
    inside = {}
    for lab in ("fixed", "v2", "V9", "F1-R1", "v2.1", "V3", "V6", "GA-R1", "V9 (ChatGPT)", "random", "grid", "bandit", "beam", "v1"):
        inside[lab] = {}
        for b in L.AB_BRIEFS + POWER:
            ks = hc((b,))
            ks = [k for k in ks if gains(lab, k)]
            inside[lab][b] = {"head_cells": len(ks), "inside_noise": sum(1 for k in ks if not L.beyond(b, sum(gains(lab, k)) / len(gains(lab, k)), CELLS[k]["band"]))}
    res["inside_noise"] = inside
    def strkeys(o):
        if isinstance(o, dict):
            return {str(k): strkeys(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [strkeys(v) for v in o]
        return o

    (L.OUT / "tables.json").write_text(json.dumps(strkeys(res), indent=1, sort_keys=True, default=str), encoding="utf-8")
    # ---------------- terminal summary (numbers only; TEST)
    print("D1 regression (pooled share over the 7 headroom cells of T0 and A0), here vs D1 report:")
    for lab, v in chk.items():
        print(f"   {lab:8s} D1 {v['d1']:5.1f}  here {v['here']:6.2f}  diff {v['diff']:+.2f}")
    print("limit failures among", len(EPS), "episodes:", len(lf), list(lf.items())[:3])
    print("power-break episode journals:", res["power_breaks"]["n_episodes"], "missing:", res["power_breaks"]["missing_journals"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
