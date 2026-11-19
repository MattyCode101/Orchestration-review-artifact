#!/usr/bin/env python3
"""Final results tables, step 2: build the four tables (six files), their LaTeX, preview and values csv.

    python final/final_tables.py            # all six tables
    python final/final_tables.py summary    # the two summary tables only (sent first)

Reads only the numbers of final_data.build(); no number is typed here. The check values of the prompt are compared with the displayed text at the end
(out/final/check_values.txt). Switches: final_config.py (LLM_MODEL, BOLD_COST_ROWS, OUT_DIR).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import final_config as CFG  # noqa: E402
import final_data as FD  # noqa: E402
import final_tex as TX  # noqa: E402
import names as N  # noqa: E402
from final_tex import Cell, Table  # noqa: E402

POL_COLS = ("exh", "direct", "orch", "rule", "fixed")                # breakdown columns
SUM_COLS = ("exh", "direct", "orch", "rule", "fixed", "bandit", "grid")
POLICIES_ONLY = ("direct", "orch", "rule", "fixed", "bandit", "grid")    # bold candidates (the exhaustive column is a reference)
MINUS = "−"


def nd_of(b: str) -> int:
    return 1 if FD.METRIC[b] == "dynamic_power_w" else 0


def fmt(x: float, nd: int = 0) -> str:
    v = FD.rnd(x, nd)
    if v == 0:
        v = 0.0
    s = f"{abs(v):,.{nd}f}"
    return (MINUS + s) if v < 0 else s


def pm(sdv: float, nd: int) -> str:
    return "±" + fmt(abs(sdv), nd)


def split_src(src: str) -> tuple[str, str, str]:
    p = [x.strip() for x in src.split(" | ")]
    return p[0], (p[1] if len(p) > 1 else ""), (p[2] if len(p) > 2 else "")


def rec(meta: dict) -> dict:
    return meta


# ======================================================================================================================================================
# summary tables (Table 1 and Table 3)
# ======================================================================================================================================================
def orch_alt(D: dict, setting: str, model: str) -> dict:
    """Gain cells of the orchestrator column for model 'gpt' or 'qwen' (TEST), computed from the per-design cells (best-model switch only)."""
    S = D["summ"][setting]
    out = {}
    for b in S["briefs"]:
        cs = [D["V"].get(("TEST", d, b, model)) for d in S["heads"][b]]
        if not all(cs):
            out[b] = {"state": "notrun"}
            continue
        n = cs[0]["n"]
        reps = [FD.mean(c["reps"][i] for c in cs) for i in range(n)] if all(c["n"] == n for c in cs) else None
        out[b] = {"state": "ok", "value": FD.mean(c["mean"] for c in cs), "n": n, "reps": reps, "sd": FD.sd(reps) if reps and n > 1 else None,
                  "src": f"inputs/per_benchmark/cells.json / PostFreeze episodes csv (per-design cells of the {FD.N.FINAL_LLM_MODELS[model]} rows) | mean over the headroom designs"}
    return out


def build_summary(D: dict, setting: str, model: str | None = None, font: tuple | None = None) -> Table:
    model = model or D["use_model"][setting]
    S = D["summ"][setting]
    briefs = S["briefs"]
    fm_, fs = font or CFG.SUMMARY_FONT
    gain_cells = {k: dict(v) for k, v in S["gain"].items()}
    cost_cells = {k: dict(v) for k, v in S["cost"].items()}
    if model != "gemini":
        alt = orch_alt(D, setting, model)
        for b in briefs:
            gain_cells[("orch", b)] = alt[b]
        if model == "gpt":
            ck = {1: "Paper_Tables_Table1_cost_per_task.csv", 2: "Paper_Tables_Table2_cost_per_task.csv"}[1 if setting == "2worker" else 2]
            for m, nm in (("jobh", "job_hours_per_task"), ("tokens", "llm_tokens_per_task")):
                v = S["cost_gpt"][m]
                cost_cells[("orch", m)] = {"state": "ok", "value": v["value"], "tasks": v["tasks"], "src": f"inputs/{ck} | key=V9 (ChatGPT) | {nm} (tasks={v['tasks']})"}
        else:
            for m in ("jobh", "tokens"):
                cost_cells[("orch", m)] = {"state": "na", "value": None, "tasks": 0, "src": "no cost row for this model"}
    n_tasks_cost = {"2worker": 10, "3worker": 15}[setting]
    # header cells
    nk = D["n_known"][setting]
    exh_lines = (["Exhaustive Search", "(3 steps,", "best known)"] if setting == "2worker" else ["Exhaustive Search", f"(2 steps, best known", f"on {nk[0]} of {nk[1]} tasks)"])
    fixed_lines = ["Fixed Worker", "Order", f"({N.final_fixed_order(setting, briefs[0])})"]
    fixed_sup = ""
    if setting == "3worker":
        fixed_sup = "a"
    heads = [
        Cell(lines=exh_lines, align="c"),
        Cell(lines=["Direct LLM", "Tool Control", "(no orchestration)"], align="c"),
        Cell(lines=["LLM-Orchestrator", "(" + N.FINAL_LLM_MODELS[model].replace(" Flash-Lite", ""), "Flash-Lite)"] if model == "gemini" else ["LLM-Orchestrator", "(" + N.FINAL_LLM_MODELS[model] + ")"], align="c"),
        Cell(lines=["Rule-Based", "Policy"], align="c"),
        Cell(lines=fixed_lines[:2] + [fixed_lines[2] + ("ᵃ" if fixed_sup else "")], align="c"),
        Cell(lines=["Learned", "Bandit"], align="c"),
        Cell(lines=["Grid", "Search"], align="c"),
    ]
    items: list = [("top",), ("row", [Cell(lines=[""], align="l")] + heads), ("mid",)]
    bold_best: dict = {}
    # bold: per gain row, among the policy columns, ties at displayed precision all bold, none when nobody gained
    for b in briefs:
        shown = {}
        for c_ in POLICIES_ONLY:
            g = gain_cells.get((c_, b))
            if g and g.get("state") == "ok" and not g.get("post_freeze"):
                shown[c_] = FD.rnd(g["value"], nd_of(b))
        mx = max(shown.values()) if shown else None
        bold_best[b] = {c_ for c_, v in shown.items() if mx is not None and mx > 0 and v == mx}
    records = []
    metric_word = {"T0": "WNS gain (ps)", "W2R": "WNS gain (ps)", "A0": "LUTs removed", "L2R": "LUTs removed", "P0": "Power saved (mW)"}
    for b in briefs:
        n_head = len(S["heads"][b])
        label = Cell(main=metric_word[b], sub=f"{N.FINAL_BRIEF_WORD[b]}, {n_head} designs", align="l", two=True)
        row = [label]
        for c_ in SUM_COLS:
            g = gain_cells.get((c_, b))
            if g is None or g.get("state") == "notrun":
                row.append(Cell(main=N.FINAL_NOT_RUN, two=True, italic=False, meta={"table": "", "row_key": f"gain_{b}", "col_key": c_, "brief": b, "state": "not_run", "value": None, "sd": None, "n": 0,
                                                                                    "src": "", "derivation": (g or {}).get("note", "no result"), "unit": FD.UNIT[FD.METRIC[b]]}))
                continue
            txt = fmt(g["value"], nd_of(b))
            sub = pm(g["sd"], nd_of(b)) if g.get("sd") is not None else ""
            c = Cell(main=txt, sub=sub, bold=(c_ in bold_best[b]), two=True, sup=("a" if (setting == "3worker" and c_ == "fixed" and b == "P0") else ("d" if g.get("post_freeze") else "")),
                     meta={"row_key": f"gain_{b}", "col_key": c_, "brief": b, "state": "ok", "value": g["value"], "sd": g.get("sd"), "n": g.get("n", 1), "unit": FD.UNIT[FD.METRIC[b]], "src": g["src"], "scope": g.get("scope", ""),
                           "derivation": f"mean over the {n_head} headroom designs of the cell means" + ("; sd = sample sd over repeats of that pooled value" if g.get("sd") is not None else "") + ("; post-freeze row: prior fitted 17 Nov, after the freeze" if g.get("post_freeze") else "")})
            row.append(c)
        items.append(("row", row))
    items.append(("mid",))
    for m, lab, sub, nd in (("jobh", "Vivado job-h / task", f"all {n_tasks_cost} TEST tasks", 2), ("tokens", "LLM tokens / task", "LLM policies only", 0)):
        row = [Cell(main=lab, sub=sub, align="l", two=True)]
        for c_ in SUM_COLS:
            g = cost_cells[(c_, m)]
            if g["state"] == "dash":
                row.append(Cell(main=N.FINAL_DASH, two=True, meta={"row_key": m, "col_key": c_, "brief": "", "state": "dash", "value": None, "sd": None, "n": 0, "src": g["src"], "derivation": "not an LLM policy", "unit": "tokens"}))
            elif g["state"] == "na" or g["value"] is None:
                row.append(Cell(main=N.FINAL_NA, two=True, meta={"row_key": m, "col_key": c_, "brief": "", "state": "na", "value": None, "sd": None, "n": 0, "src": g["src"], "derivation": "blank in the cost file; see the note", "unit": "job-h"}))
            else:
                row.append(Cell(main=fmt(g["value"], nd), two=True, bold=False,
                                meta={"row_key": m, "col_key": c_, "brief": "", "state": "ok", "value": g["value"], "sd": None, "n": g["tasks"], "unit": "job-h" if m == "jobh" else "tokens", "src": g["src"], "scope": ("post-freeze, prior fitted 17 Nov from TRAINING and VALIDATION only" if (setting == "3worker" and c_ == "bandit" and m == "jobh") else ""),
                                      "derivation": f"mean per task over {g['tasks']} tasks (cost file)"}))
        items.append(("row", row))
    items.append(("bottom",))
    # note
    grp = "TEST (5 designs)"
    wk = {"2worker": "two workers", "3worker": "three workers"}[setting]
    nh = "; ".join(f"{len(S['heads'][b])} designs {N.FINAL_BRIEF_WORD[b]}" for b in briefs)
    exh_txt = ("Exhaustive Search is the 3-step search, best known on all 7 headroom tasks; it is a reference and is never bold." if setting == "2worker" else
               f"Exhaustive Search is the 2-step search over 14 recipes, AMD-checked; the best known result is higher on {nk[1] - nk[0]} of the {nk[1]} headroom tasks, where a six-run result beats it ("
               + ", ".join(f"{FD.N.FINAL_DESIGN_SHORT[d]} {N.FINAL_BRIEF_WORD[b]}" for d, b in D["n_known"]["beaten_3worker"]) + "); it is a reference and is never bold.")
    rep_txt = {"direct": 3, "orch": 3}
    note = (f"{grp}, {wk}, budget of 6 tool runs per task. Gain = improvement of the final pick over the starting checkpoint (WNS in ps, LUTs removed" + (", dynamic power saved in mW" if setting == "3worker" else "")
            + f"). Each gain cell is the mean over the TEST headroom designs, where the best known result improves beyond one noise band ({nh}), after averaging each design's repeats. "
            + exh_txt + " Direct LLM Tool Control and LLM-Orchestrator: 3 repeats per design; ± is the sample sd over repeats of the pooled mean; all other policies are deterministic or single runs (no ±). "
            + f"Vivado job-h / task: charged job-hours per task over all {n_tasks_cost} TEST tasks ({'both briefs' if setting == '2worker' else 'all three briefs'})"
            + ("; Exhaustive Search is counted per design, because the search is shared by both briefs" if setting == "2worker" else "")
            + "; LLM tokens / task: LLM policies only. Bold = best policy value of a row (ties at displayed precision all bold; none where no policy gained); cost rows are not bolded.")
    if setting == "3worker":
        bn = S["gain"][("bandit", briefs[0])]
        note += (f" Learned Bandit: the frozen bandit had no timing-first prior (freeze of 13 Nov); its timing-first cell is a post-freeze row (d); its job-h is the mean over its {S['cost'][('bandit', 'jobh')]['tasks']} tasks."
                 + (" Exhaustive Search job-h: n/a, the stored 2-step search over 14 recipes has no per-run charged-hours record." if S["cost"][("exh", "jobh")]["state"] == "na" else "")
                 + " a: Fixed Worker Order runs power → area → timing on the power-first brief.")
    if model != "gemini":
        note += f" LLM-Orchestrator shown with {N.FINAL_LLM_MODELS[model]} (LLM_MODEL = best)."
    cap = (f"Two workers, TEST designs: gain of each policy over the starting checkpoint under the timing-first and area-first briefs, and cost per task." if setting == "2worker" else
           f"Three workers, TEST designs: gain of each policy over the starting checkpoint under the timing-first, area-first and power-first briefs, and cost per task.")
    num = "Table 1." if setting == "2worker" else "Table 3."
    return Table(stem=f"tab_summary_{setting}_TEST", caption=cap, label=f"tab:summary_{setting}_TEST", ncols=8, aligns=["l"] + ["c"] * 7, items=items, note=note, width_in=CFG.FULL_WIDTH_IN,
                 font_main=fm_, font_sub=fs, title_no=num)


# ======================================================================================================================================================
# per-benchmark tables (Table 2 and Table 4)
# ======================================================================================================================================================
def build_breakdown(D: dict, setting: str, variant: str) -> Table:
    briefs = FD.SETTING_BRIEFS[setting]
    nb = len(briefs)
    start_keys = ("wns", "luts") + (("mw",) if setting == "3worker" else ())
    nst = len(start_keys)
    V, X, TASK, start, order = D["V"], D["X"], D["TASK"], D["start"], D["order"]
    pol_of = {"direct": "direct", "orch": "orch", "rule": "rule", "fixed": "fixed"}
    all_designs = [(g, d) for g in FD.GROUPS for d in order[g]]
    assert len(all_designs) == 13
    shown_designs = [(g, d) for g, d in all_designs if not (variant == "representative" and d in N.REPRESENTATIVE_DROP)]
    ncols = 2 + nst + 5 * nb
    model = D["use_model"][setting]
    for (fm_, fs) in CFG.BREAKDOWN_FONTS:
        T = _breakdown_table(D, setting, variant, briefs, start_keys, shown_designs, all_designs, ncols, fm_, fs, model)
        L = TX.layout(T)
        if L["extra"] >= 0:
            return T
    return T


def cell_values(D, g, d, b, col):
    """(mean, sd, n, reps, marker, src, state) of one cell of the breakdown."""
    if col == "exh":
        x = D["X"][(g, d, b)]
        return {"state": "ok", "mean": x["val"], "sd": None, "n": 1, "reps": [x["val"]], "mark": x["mark"], "src": x["src"], "kind": x["kind"]}
    c = D["V"].get((g, d, b, col))
    if not c:
        return {"state": "notrun"}
    return {"state": "ok", "mean": c["mean"], "sd": c["sd"], "n": c["n"], "reps": c["reps"], "mark": "", "src": c["src"], "kind": ""}


def _breakdown_table(D, setting, variant, briefs, start_keys, shown_designs, all_designs, ncols, fm_, fs, model) -> Table:
    nb, nst = len(briefs), len(start_keys)
    TASK, start = D["TASK"], D["start"]
    items: list = [("top",)]
    unit_b = {b: FD.UNIT[FD.METRIC[b]] for b in briefs}
    # header row 1: spanners
    r1 = [Cell(main="", span=2, covered=False)]
    r1.append(Cell(main="Start values", span=nst, small=False, align="c"))
    cm = [(3, 2 + nst)]
    j = 3 + nst
    for b in briefs:
        word = N.FINAL_BRIEF_WORD[b].capitalize()
        met = N.FINAL_BRIEF_METRIC[b]
        met = met if met[:2].isupper() else met[0].lower() + met[1:]
        r1.append(Cell(main=f"{word}: {met}", span=5, align="c"))
        cm.append((j, j + 4))
        j += 5
    items.append(("row", r1))
    items.append(("cmid", cm))
    # header row 2: rotated names
    sh = {"wns": ["WNS (ps)"], "luts": ["Packed LUTs"], "mw": ["Dynamic power (mW)"]}
    r2 = [Cell(lines=[""]), Cell(lines=[""])]
    for k in start_keys:
        r2.append(Cell(lines=sh[k], rot=True, align="c"))
    for b in briefs:
        r2.append(Cell(lines=["Exhaustive Search"], rot=True))
        r2.append(Cell(lines=["Direct LLM Tool Control", "(no orchestration)"], rot=True))
        r2.append(Cell(lines=["LLM-Orchestrator", "(" + N.FINAL_LLM_MODELS[model] + ")"], rot=True))
        r2.append(Cell(lines=["Rule-Based Policy"], rot=True))
        r2.append(Cell(lines=["Fixed Worker Order", f"({N.final_fixed_order(setting, b)})"], rot=True))
    items.append(("row", r2))
    items.append(("mid",))
    # bold per row within a brief block among the four policy columns, at displayed precision
    def bold_set(vals: dict, b):
        sh_ = {c: FD.rnd(v, nd_of(b)) for c, v in vals.items() if v is not None}
        mx = max(sh_.values()) if sh_ else None
        return {c for c, v in sh_.items() if mx is not None and mx > 0 and v == mx}

    # body
    first_of_group = True
    gi = 0
    for g in FD.GROUPS:
        ds = [d for (gg, d) in shown_designs if gg == g]
        for k, d in enumerate(ds):
            row = []
            if k == 0:
                row.append(Cell(main=g, rot=True, rowspan=len(ds), align="c"))
            else:
                row.append(Cell(covered=True))
            row.append(Cell(main=N.FINAL_DESIGN_SHORT[d], align="l", two=True))
            for sk in start_keys:
                v = start[d][sk]
                row.append(Cell(main=fmt(v, 0), align="c", two=True, meta={"group": g, "row_key": d, "col_key": f"start_{sk}", "brief": "", "state": "ok", "value": v, "sd": None, "n": 1,
                                                                          "unit": {"wns": "ps", "luts": "LUTs", "mw": "mW"}[sk], "src": start[d]["src"], "derivation": "start value of the B2 checkpoint"}))
            for b in briefs:
                head = TASK[(g, d, b)]["head"]
                cv = {c: cell_values(D, g, d, b, c) for c in POL_COLS}
                bs = bold_set({c: cv[c]["mean"] for c in ("direct", "orch", "rule", "fixed") if cv[c]["state"] == "ok"}, b)
                for c in POL_COLS:
                    x = cv[c]
                    meta = {"group": g, "row_key": d, "col_key": c, "brief": b, "unit": unit_b[b], "headroom": head}
                    if x["state"] != "ok":
                        row.append(Cell(main=N.FINAL_NOT_RUN, italic=True, two=True, meta=dict(meta, state="not_run", value=None, sd=None, n=0, src="", derivation="no episode")))
                        continue
                    mark = x["mark"] if (setting == "2worker" and c == "exh") else ("†" if (c == "exh" and x["mark"] == "†") else "")
                    sub = pm(x["sd"], nd_of(b)) if x["n"] > 1 and x["sd"] is not None else ""
                    row.append(Cell(main=fmt(x["mean"], nd_of(b)), sub=sub, sup=mark, bold=(c in bs), italic=(not head), two=True,
                                    meta=dict(meta, state="ok", value=x["mean"], sd=x["sd"], n=x["n"], src=x["src"], derivation=("mean over repeats" if x["n"] > 1 else "single stored result") + (f" ({x['kind']})" if x["kind"] else ""),
                                              marker=mark)))
            items.append(("row", row))
        items.append(("mid",))
        gi += 1
    # total row over all 13 designs
    row = [Cell(main="Total, all 13 designs", span=2, align="l", two=True)]
    for sk in start_keys:
        s_ = sum(start[d][sk] for _, d in all_designs)
        row.append(Cell(main=fmt(s_, 0), two=True, meta={"group": "ALL", "row_key": "total", "col_key": f"start_{sk}", "brief": "", "state": "ok", "value": s_, "sd": None, "n": 13,
                                                         "unit": {"wns": "ps", "luts": "LUTs", "mw": "mW"}[sk], "src": "sum of the start values of the 13 designs", "derivation": "sum over 13 designs"}))
    for b in briefs:
        tot = {}
        for c in POL_COLS:
            xs = [cell_values(D, g, d, b, c) for g, d in all_designs]
            have = [x for x in xs if x["state"] == "ok"]
            if not have:
                tot[c] = None
                continue
            s_ = sum(x["mean"] for x in have)
            ns = {x["n"] for x in have}
            sdv = None
            if len(have) == 13 and len(ns) == 1 and next(iter(ns)) > 1:
                n = next(iter(ns))
                per = [sum(x["reps"][i] for x in have) for i in range(n)]
                sdv = FD.sd(per)
            tot[c] = {"sum": s_, "k": len(have), "sd": sdv, "n": (next(iter(ns)) if len(ns) == 1 else 0), "pending": any(x.get("mark") == "†" for x in have)}
        comparable = all(tot[c] and tot[c]["k"] == 13 for c in ("direct", "orch", "rule", "fixed"))
        bs = bold_set({c: tot[c]["sum"] for c in ("direct", "orch", "rule", "fixed")}, b) if comparable else set()
        for c in POL_COLS:
            t = tot[c]
            meta = {"group": "ALL", "row_key": "total", "col_key": c, "brief": b, "unit": unit_b[b]}
            if t is None:
                row.append(Cell(main=N.FINAL_NOT_RUN, italic=True, two=True, meta=dict(meta, state="not_run", value=None, sd=None, n=0, src="", derivation="no episode in any design")))
                continue
            sub = f"{t['k']} of 13" if t["k"] < 13 else (pm(t["sd"], nd_of(b)) if t["sd"] is not None else "")
            row.append(Cell(main=fmt(t["sum"], nd_of(b)), sub=sub, sup=("†" if t["pending"] and setting == "3worker" and c == "exh" else ""), bold=(c in bs), two=True,
                            meta=dict(meta, state="ok", value=t["sum"], sd=t["sd"], n=t["n"], src="sum over the designs of the cell values of this column", derivation=f"sum over {t['k']} of 13 designs; sd = sample sd over repeats of the per-repeat sum" if t["sd"] is not None else f"sum over {t['k']} of 13 designs")))
    items.append(("row", row))
    items.append(("bottom",))
    # note
    nh = {g: sum(1 for d in FD.DESIGNS[g] for b in briefs if TASK[(g, d, b)]["head"]) for g in FD.GROUPS}
    wk = {"2worker": "two workers", "3worker": "three workers"}[setting]
    reps = "LLM-Orchestrator and Direct LLM Tool Control: 3 repeats per design (every group); Rule-Based Policy, Fixed Worker Order and Exhaustive Search: 1 (deterministic)."
    gpt_note = ""
    if model != "gemini":
        gpt_note = f" LLM-Orchestrator shown with {N.FINAL_LLM_MODELS[model]} (LLM_MODEL = best): 3 repeats on TEST, 1 repeat on TRAINING and VALIDATION three-worker (fill item e); not run on two workers."
        reps = "Direct LLM Tool Control: 3 repeats per design (every group); Rule-Based Policy, Fixed Worker Order and Exhaustive Search: 1 (deterministic)."
    if setting == "2worker":
        exh_txt = ("Exhaustive Search: TEST and TRAINING, the 3-step search (TRAINING: the stored depth-3 state); VALIDATION " + "(marked 2): the 2-step search only, no 3-step search exists; on mcml area-first a policy run reached 1,625 LUTs above its 1,595. "
                   "Totals add these values as they are.")
    else:
        exh_txt = ("Exhaustive Search: 2 steps over 14 recipes on TEST and VALIDATION; on TRAINING the exhaustive search is "
                   + ("pending (tier 2 of the fill run): † marks the six-run hindsight best, a lower bound, in its place (the total carries †)." if D["tier2_rows"] == 0 else "the fill run's 2-step search over 14 recipes."))
    note = (f"{wk.capitalize()}; TRAINING (5 designs), VALIDATION (3) and TEST (5); budget of 6 tool runs per task; designs ordered by start LUTs within a group. Cell = gain of the final pick over the starting checkpoint "
            f"({', '.join(FD.UNIT[FD.METRIC[b]] for b in briefs)}), mean over repeats; ± = sample sd over repeats (only where there are several). {reps} {exh_txt} "
            f"Italic: no headroom (the best known result is within one noise band of the start). Bold = best policy per row within a brief block (ties at displayed precision all bold; none where no policy gained; the Exhaustive column is a reference and never bold). "
            f"Total = sum over all 13 designs, including italic ones, with ± from the per-repeat sums; start values are those of the B2 checkpoint." + gpt_note)
    if variant == "representative":
        note += " A representative set of benchmarks is shown; totals include all 13 designs."
    cap = (f"{wk.capitalize()}, per design, TRAINING, VALIDATION and TEST: gain of each policy over the starting checkpoint "
           f"({'timing-first and area-first' if setting == '2worker' else 'timing-first, area-first and power-first'}).")
    stem = f"tab_breakdown_{setting}_{variant}"
    aligns = ["c", "l"] + ["c"] * nst + ["c"] * (5 * nb)
    num = "Table 2." if setting == "2worker" else "Table 4."
    return Table(stem=stem, caption=cap, label=f"tab:breakdown_{setting}_{variant}", ncols=ncols, aligns=aligns, items=items, note=note, width_in=CFG.FULL_WIDTH_IN,
                 font_main=fm_, font_sub=fs, font_note=6.0, title_no=num)


# ======================================================================================================================================================
# values csv, checks
# ======================================================================================================================================================
VALUE_FIELDS = ["table", "group", "row_key", "col_key", "brief", "line", "shown", "value", "sd", "n_repeats", "state", "bold", "italic", "marker", "unit", "headroom",
                "source_file", "source_key", "source_column", "derivation", "scope"]


def write_values(T: Table, outdir: Path) -> list[dict]:
    rows = []
    for it in T.items:
        if it[0] != "row":
            continue
        for c in it[1]:
            if c.meta is None or c.covered:
                continue
            m = c.meta
            f, k, col = split_src(m.get("src", ""))
            base = {"table": T.stem, "group": m.get("group", "TEST"), "row_key": m.get("row_key", ""), "col_key": m.get("col_key", ""), "brief": m.get("brief", ""), "state": m.get("state", ""),
                    "bold": int(c.bold), "italic": int(c.italic), "marker": m.get("marker", c.sup), "unit": m.get("unit", ""), "headroom": m.get("headroom", ""), "source_file": f, "source_key": k,
                    "source_column": col, "derivation": m.get("derivation", ""), "scope": m.get("scope", "")}
            rows.append(dict(base, line="main", shown=c.main + (c.sup or ""), value=("" if m.get("value") is None else repr(float(m["value"]))), sd="" if m.get("sd") is None else repr(float(m["sd"])),
                             n_repeats=m.get("n", "")))
            if c.sub:
                rows.append(dict(base, line="sub", shown=c.sub, value="" if m.get("sd") is None else repr(float(m["sd"])), sd="", n_repeats=m.get("n", "")))
    with open(outdir / f"values_{T.stem}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=VALUE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


# expected (check values of the prompt): (table stem, row_key, col_key, brief or '', expected number, decimals)
CHECKS = []


def _add(stem, row_key, col, brief, val, nd=0):
    CHECKS.append((stem, row_key, col, brief, val, nd))


for col, vals in {"exh": (400, 419, 147.16), "direct": (141, 277, 1.14, 17083), "orch": (166, 352, 1.16, 10941), "rule": (167, 334, 1.11), "fixed": (178, 333, 1.29), "bandit": (66, 343, 1.19), "grid": (144, 120, 1.06)}.items():
    s = "tab_summary_2worker_TEST"
    _add(s, "gain_T0", col, "T0", vals[0]); _add(s, "gain_A0", col, "A0", vals[1]); _add(s, "jobh", col, "", vals[2], 2)
    if len(vals) > 3:
        _add(s, "tokens", col, "", vals[3])
for col, vals in {"exh": (279, 505, 44.7), "direct": (128, 365, 43.2, 1.07, 19717), "orch": (166, 232, 41.3, 1.03, 8992), "rule": (167, 228, 41.3, 0.98), "fixed": (135, 261, 43.0, 1.12), "grid": (144, 141, 41.3, 1.00)}.items():
    s = "tab_summary_3worker_TEST"
    _add(s, "gain_W2R", col, "W2R", vals[0]); _add(s, "gain_L2R", col, "L2R", vals[1]); _add(s, "gain_P0", col, "P0", vals[2], 1)
    if len(vals) > 3:
        _add(s, "jobh", col, "", vals[3], 2)
    if len(vals) > 4:
        _add(s, "tokens", col, "", vals[4])
_add("tab_summary_3worker_TEST", "gain_L2R", "bandit", "L2R", 0); _add("tab_summary_3worker_TEST", "gain_P0", "bandit", "P0", 0.0, 1); _add("tab_summary_3worker_TEST", "jobh", "bandit", "", 0.89, 2)
for stem in ("tab_breakdown_2worker_all", "tab_breakdown_2worker_representative", "tab_breakdown_3worker_all", "tab_breakdown_3worker_representative"):
    _add(stem, "corescore_500_mod", "start_wns", "", -1204); _add(stem, "corescore_500_mod", "start_luts", "", 105957)
for stem in ("tab_breakdown_3worker_all", "tab_breakdown_3worker_representative"):
    _add(stem, "corescore_500_mod", "start_mw", "", 13364)
for stem in ("tab_breakdown_2worker_all", "tab_breakdown_2worker_representative"):
    for col, v in zip(POL_COLS, (742, 429, 543, 544, 543)):
        _add(stem, "corescore_500_mod", col, "A0", v)
    for col, v in zip(POL_COLS, (167, 103, 103, 103, 100)):
        _add(stem, "fir_systolic_transposed", col, "T0", v)
    for col, v in (("exh", 1595), ("orch", 1121), ("rule", 1086), ("fixed", 742)):
        _add(stem, "vtr_mcml_v2", col, "A0", v)
    for col, v in (("exh", 7393), ("orch", 1597), ("rule", 7095), ("fixed", 7047), ("direct", 4812)):
        _add(stem, "finn_radioml", col, "A0", v)
for stem in ("tab_breakdown_3worker_all", "tab_breakdown_3worker_representative"):
    for col, v in zip(POL_COLS, (741, 428, 561, 544, 16)):
        _add(stem, "corescore_500_mod", col, "L2R", v)


def run_checks(records: dict[str, list[dict]]) -> list[str]:
    lines, ok_all = [], True
    for stem, row_key, col, brief, exp, nd in CHECKS:
        recs = records.get(stem)
        if recs is None:
            continue
        got = [r for r in recs if r["line"] == "main" and r["row_key"] == row_key and r["col_key"] == col and (brief == "" or r["brief"] == brief)]
        if len(got) != 1 or got[0]["value"] == "":
            lines.append(f"NO CELL   {stem} {row_key} {col} {brief}: expected {exp}")
            ok_all = False
            continue
        v = float(got[0]["value"])
        same = FD.rnd(v, nd) == FD.rnd(exp, nd)
        ok_all &= same
        lines.append(f"{'yes' if same else 'NO '}  {stem:40s} {row_key:24s} {col:7s} {brief:4s} expected {exp:>10} displayed {got[0]['shown']:>9}")
    lines.append(f"all check values match: {ok_all}")
    return lines


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    D = FD.build()
    print("\n".join(D["log"]))
    if D["mism"]:
        print("STOP: cross-check mismatches:")
        for m in D["mism"]:
            print("  -", m)
        return 1
    outdir = ROOT / CFG.OUT_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    tables = [build_summary(D, "2worker"), build_summary(D, "3worker")]
    if which == "all":
        tables += [build_breakdown(D, s, v) for s in ("2worker", "3worker") for v in ("all", "representative")]
    allrec: dict[str, list[dict]] = {}
    report = []
    for setting in ("2worker", "3worker"):
        t7 = build_summary(D, setting, font=(7.0, 7.0))
        nat = TX.layout(t7)["natural"] / 72.0
        report.append(f"single-column test, {setting} summary table: natural width {nat:.2f} in at 7 pt main and 7 pt second line, against {CFG.SINGLE_COLUMN_IN} in available -> "
                      f"{'fits' if nat <= CFG.SINGLE_COLUMN_IN else 'does not fit; full width (7.0 in) is used'}")
    for T in tables:
        s = TX.emit_tex(T)
        bad = TX.lint_tex(s, T)
        with open(outdir / f"{T.stem}.tex", "w", encoding="ascii", newline="\n") as fh:
            fh.write(s)
        pv = TX.preview(T, str(outdir))
        allrec[T.stem] = write_values(T, outdir)
        report.append(f"{T.stem}: preview {pv['width_in']:.2f} x {pv['height_in']:.2f} in, natural width {pv['natural_width_in']:.2f} in, extra {pv['extra_pt']:.1f} pt, text sizes {T.font_main:g}/{T.font_sub:g}/{T.font_note:g} pt, "
                      f"TeX lint: {'ok' if not bad else bad}")
    chk = run_checks(allrec)
    (outdir / "check_values.txt").write_text("\n".join(chk) + "\n", encoding="utf-8")
    print("\n".join(report))
    print("\n".join(chk[-1:]))
    n_no = sum(1 for c in chk if c.startswith("NO"))
    print(f"check values not matching: {n_no}")
    return 0 if n_no == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
