"""X2 part 2 (Mon 16 Nov), step 1: collect every stored TRAINING and VALIDATION episode result into one normalised list of (group, design, brief, row, repeat, gain), plus the cell bands, the
VALIDATION exhaustive two-step references, the stored TRAINING best-known values and the TRAINING power ceilings. NEW file, offline, read-only on stored results. No tool run, no model call.

Writes data/paper_tables_2026-11-15/eng/gen_collect.json.

Gain definitions as tr_pt_lib.gain_of: WNS gain in ns (T0, T, W2R), packed-LUT cut (A0, A, L2R), dynamic-power cut in W (P0). Start metrics come from the result itself when it has them (round-2 harness) and
otherwise from the start of the same design in a round-2-harness result (all start from the same B2 checkpoint). Only results with validation_status == pass and final_brief_valid are kept as reference candidates
(the engine accepts a final state only after an AMD PASS); every kept record carries its source.
"""
from __future__ import annotations
import os

import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(os.environ.get("ORCH_PROJECT_ROOT") or Path(__file__).resolve().parents[2])
OUT = PROJECT / "data/paper_tables_2026-11-15"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_pt_lib as L  # noqa: E402

TRAIN = ("vexriscv_re-place", "logicnets_jscl", "rosetta_spam-filter", "rosetta_3d-rendering", "finn_radioml")
VAL = ("t2_cordic_s", "vtr_mcml_v2", "amd_mini-isp")
GOALM = {b: L.GOAL[b][0] for b in L.GOAL}


def group_of(d: str) -> str:
    return "TRAINING" if d in TRAIN else ("VALIDATION" if d in VAL else "TEST")


SRC = {  # name: (glob, harness)  harness r2 = round-2 result format with start_metrics and noise; old = pilot format
    "d6v2": ("data/d6v2_2026-11-12/results/*.json", "r2"),
    "v22grid": ("data/v22_2026-11-13/grid/results/*.json", "r2"),
    "v22t3": ("data/v22_2026-11-14/tier3/results/*.json", "r2"),
    "power_review": ("data/test_r2_2026-11-13/power_review/results/*.json", "r2"),
    "dev_baselines": ("data/development_designs_2026-11-12/results/*.json", "old"),
    "v4_pilots": ("data/pilots/thursday_overnight_train_v4_2026-11-05/results/*.json", "old"),
    "sweep_fri": ("data/pilots/friday_repair_sweep_2026-11-06/results/*.json", "old"),
    "sweep_sat": ("data/pilots/saturday_sweep_ta_2026-11-06/results/*.json", "old"),
    "p0_pilot": ("data/pilots/friday_p0_pilot_2026-11-06/results/*.json", "old"),
    "llm_f2r1": ("data/pilots/saturday_llm_f2r1_2026-11-06/results/*.json", "old"),
    "llm_var1": ("data/pilots/saturday_llm_var1_2026-11-06/results/*.json", "old"),
    "llm_var2": ("data/pilots/saturday_llm_var2_2026-11-06/results/*.json", "old"),
    "llm_grid1": ("data/pilots/saturday_llm_grid1_2026-11-06/results/*.json", "old"),
    "llm_grid2": ("data/pilots/saturday_llm_grid2_2026-11-06/results/*.json", "old"),
    "ga1": ("data/pilots/saturday_llm_ga1_2026-11-06/results/*.json", "old"),
    "ga3": ("data/pilots/saturday_llm_ga3_2026-11-06/results/*.json", "old"),
    # post-freeze exploratory rows (X1, Mon 16 Nov); empty until the test run has written them
    "post_freeze": ("data/test_r2_2026-11-13/post_freeze_2026-11-16/**/results/tr_*.json", "r2"),
}


def row_of(src: str, r: dict) -> tuple[str, int | None]:
    """Standard row name and repeat index of a stored result."""
    a = r.get("approach") or r.get("policy") or ""
    tag = str(r.get("tag") or "")
    model = ((r.get("task") or {}).get("model")) or ((r.get("llm") or {}).get("model"))
    if src == "post_freeze":
        if model and model != L.GEMINI:
            return f"post_freeze:{model}:{a}", None
        m0 = re.match(r"^[rt](\d+)$", tag)
        return (a if a in ("V9", "V3", "V6", "V11") else f"post_freeze:{a}"), (int(m0.group(1)) if m0 else None)
    rep = None
    m = re.match(r"^[rt](\d+)$", tag)
    if m:
        rep = int(m.group(1))
    if src == "d6v2":
        if a == "F1":
            return "F1-R1", rep
        if a == "v21":
            return "v2.1", rep
        if a == "v2":
            return "v2", None
        if a == "fixed_BA":
            return "fixed", None
        if a.startswith("random_"):
            return "random", int(a.split("_")[1])
    if src == "power_review":
        return {"F1-R1": "F1-R1", "v21": "v2.1", "v2": "v2", "fixed": "fixed"}.get(a, a), None
    if src == "v22t3" and a == "V9":
        return "V9", rep
    if src in ("v22t3", "v22grid"):
        return a, rep                        # other prompt variants: reference candidates only
    if src == "dev_baselines":
        if a.startswith("fixed"):
            return "fixed", None
        if a.startswith("random_"):
            return "random", int(a.split("_")[1])
        if a == "v2":
            return "v2", None
    if src in ("ga1", "ga3"):
        return "GA-trainingstage", None
    if src == "p0_pilot" and a.startswith("random_seed_"):
        return "random", int(a.split("_")[-1])
    return f"{src}:{a}", None


def main() -> int:
    recs, starts, bands = [], {}, defaultdict(dict)
    loaded = {}
    # pass 1: starts and bands from round-2 harness results
    raw = {}
    for name, (pat, kind) in SRC.items():
        fs = sorted(glob.glob(str(PROJECT / pat), recursive=True))
        raw[name] = []
        for f in fs:
            if "dry_validation" in f or f.endswith(".cache_miss.json"):
                continue                         # dry runs and cache-miss stubs are not results
            r = json.loads(Path(f).read_text(encoding="utf-8"))
            if "design" not in r or "final_metrics" not in r or r.get("status") not in (None, "done"):
                continue
            raw[name].append(r)
        loaded[name] = len(raw[name])
    for name, rs in raw.items():
        for r in rs:
            if SRC[name][1] == "r2":
                starts.setdefault(r["design"], r["start_metrics"])
                for m, v in (r.get("noise") or {}).items():
                    bands[(r["design"], m)].setdefault(round(float(v), 9), 0)
                    bands[(r["design"], m)][round(float(v), 9)] += 1
    # VALIDATION exhaustive references and bands
    ex = L.jread(PROJECT / "data/development_designs_2026-11-12/exhaustive.json")
    val_ref = {}
    for d, dd in ex["designs"].items():
        starts.setdefault(d, dd["start"])
        for m, v in dd["bands"].items():
            bands[(d, m)].setdefault(round(float(v), 9), 0)
            bands[(d, m)][round(float(v), 9)] += 1
        for b, br in dd["briefs"].items():
            pick = br.get("pick")
            val_ref[f"{d}|{b}"] = {"gain": (pick or {}).get("gain", 0.0) if pick else 0.0, "steps": (pick or {}).get("steps"), "path": (pick or {}).get("path"), "state": (pick or {}).get("state")}
    # pass 2: records
    n_skip = defaultdict(int)
    for name, rs in raw.items():
        for r in rs:
            d, b = r["design"], r["brief"]
            if d not in TRAIN and d not in VAL:
                n_skip[name] += 1
                continue
            if b not in GOALM:
                n_skip[name] += 1
                continue
            if r.get("final_brief_valid") is not True or r.get("validation_status") != "pass":
                n_skip[name + ":not valid/pass"] += 1
                continue
            st = r.get("start_metrics") or starts.get(d)
            if st is None:
                n_skip[name + ":no start"] += 1
                continue
            g = L.gain_of(b, st, r["final_metrics"])
            row, rep = row_of(name, r)
            recs.append({"group": group_of(d), "design": d, "brief": b, "src": name, "row": row, "rep": rep, "gain": g, "eid": r.get("episode_id"), "runs": r.get("runs_used")})
    # stored TRAINING best-known (not AMD-checked; kept for the comparison only)
    bk = L.jread(PROJECT / "data/pilots/thursday_overnight_train_v4_2026-11-05/policy_brief_summary.json")["best_known"]
    stored_bk = {}
    for k, v in bk.items():
        d, b = k.split(":")
        m, direction = L.GOAL[b]
        s0 = starts[d][m]
        stored_bk[f"{d}|{b}"] = {"gain": (v["value"] - s0) if direction == "max" else (s0 - v["value"]), "source": v["source"], "state": v["state_id"]}
    # TRAINING power ceilings (six-run hindsight bound on the stored graph, not AMD-checked)
    pr = L.jread(PROJECT / "data/power_readiness_2026-11-11/final_A4.json")["briefs"]
    train_ceil = {}
    for b, lst in pr.items():
        for e in lst:
            d = e["design"]
            d_full = next((x for x in TRAIN if x == d or x.startswith(d) or d in x), d)
            train_ceil[f"{d}|{b}"] = {"ceiling": e["ceiling"], "unit": e["unit"], "band": e["band"], "strict_F2": (e.get("strict_twin_aware_F2") or None), "byte_exact_F1": e.get("byte_exact_F1"), "best_fixed": e.get("best_fixed")}
    # bandit TRAINING (leave-one-design-out, frozen settings = 'learned') from the stored simulation tables
    t = L.jread(PROJECT / "data/bandit_2026-11-11/tables_s_3.json")
    bandit_train = {}
    for k, v in t["percell"].items():
        e = v.get("learned")
        if e:
            bandit_train[k] = {"gain": e["improvement"], "censored": e.get("censored"), "used": e.get("used")}
    # bandit VALIDATION (development designs) from the development note, table of 12 cells
    note = (PROJECT / "notes/bandit_policy_dev.md").read_text(encoding="utf-8")
    bandit_val = {}
    unit = {"ps": ("wns_ns", 1e-3), "LUT": ("packed_luts", 1.0), "mW": ("dynamic_power_w", 1e-3)}
    for line in note.splitlines():
        m = re.match(r"^\| (t2_cordic_s|vtr_mcml_v2|amd_mini-isp) \| (T0|A0|L2R|P0) \| \w+ \| [^|]+\| \+(\d+) (ps|LUT|mW) \((\d+)%\)(\*?) \|", line)
        if m:
            d, b, val, u, pct, cut = m.groups()
            bandit_val[f"{d}|{b}"] = {"gain": float(val) * unit[u][1], "censored": bool(cut), "share_note_pct": float(pct)}
    out = {"schema": "tr-gen-collect-v1", "loaded": loaded, "skipped": dict(n_skip), "n_records": len(recs), "starts": starts,
           "bands": {f"{d}|{m}": sorted(v) for (d, m), v in bands.items()}, "records": recs, "val_exhaustive": val_ref, "stored_best_known_train": stored_bk,
           "train_power_ceiling": train_ceil, "bandit_train": bandit_train, "bandit_val": bandit_val}
    (OUT / "eng").mkdir(exist_ok=True)
    (OUT / "eng/gen_collect.json").write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    print("loaded", loaded)
    print("skipped", dict(n_skip))
    print("records", len(recs), "bandit train cells", len(bandit_train), "bandit val cells", len(bandit_val), "val ref cells", len(val_ref))
    clash = {k: v for k, v in bands.items() if len(v) > 1}
    print("band clashes", {f"{d}|{m}": sorted(v) for (d, m), v in clash.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
