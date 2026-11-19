"""Pre-registered analysis of the test campaign (see notes/analysis_plan.md), built and tested on training.

Uses the training definitions exactly as implemented (scripts/friday_analysis.py: share of best known,
signed gain, noise bands; scripts/saturday_llm_report.py: win / tie / loss by gain difference inside the
design's noise band). The only addition for test is the no-room rule: a (design, brief) cell whose
best-known gain is inside the noise band has no share and is left out of every mean.

Modes
  --training-table   robustness table on the training records (no new runs, no model calls)
  --test             the test analysis from a campaign work root (needs the finished campaign)
Reads result rows and logs only; starts no tool.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT))

from orch.analysis.report import share_of_best  # noqa: E402
from scripts import friday_analysis as fa  # noqa: E402

CRITERION = ("A0", "T0")
FOUR = ("T0", "A0", "T", "A")
ALL_BRIEFS = FOUR + ("P0",)
NON_LLM = ("v1", "v2", "single", "fixed_AB", "fixed_BA", "random")
LLM = ("F1-R1", "F1-R0", "E0-R1", "GA-R1")
OBJECTIVE = fa.OBJECTIVE
RowKey = tuple[str, str]          # (design, brief)


def mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def sign_test_p(wins: int, losses: int) -> float | None:
    """Exact two-sided sign test on wins against losses (ties dropped)."""

    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


@dataclass
class Frame:
    """Everything a share needs: starts, noise bands, best known; the training definitions."""

    designs: tuple[str, ...]
    starts: dict[str, dict[str, float]]
    bands: dict[str, dict[str, float]]
    best: dict[RowKey, float | None]
    room_rule: bool

    def metric(self, brief: str) -> str:
        return OBJECTIVE[brief][0]

    def value(self, row: dict[str, Any]) -> float:
        return row["final_metrics"][self.metric(row["brief"])]

    def band(self, design: str, brief: str) -> float:
        return self.bands.get(design, {}).get(self.metric(brief), 0.0)

    def gain(self, row: dict[str, Any]) -> float:
        return fa.improvement(row["brief"], row["design"], self.value(row), self.starts)

    def best_gain(self, design: str, brief: str) -> float | None:
        value = self.best.get((design, brief))
        if value is None:
            return None
        metric, direction = OBJECTIVE[brief]
        delta = value - self.starts[design][metric]
        return delta if direction == "max" else -delta

    def no_room(self, design: str, brief: str) -> bool:
        if not self.room_rule:
            return False
        g = self.best_gain(design, brief)
        return g is None or g <= self.band(design, brief) + 1e-12

    def share(self, row: dict[str, Any]) -> float | None:
        if self.no_room(row["design"], row["brief"]):
            return None
        best = self.best.get((row["design"], row["brief"]))
        metric, direction = OBJECTIVE[row["brief"]]
        return share_of_best(self.starts[row["design"]][metric], best, self.value(row), direction)

    def outcome_vs(self, row: dict[str, Any], other: dict[str, Any]) -> str:
        diff = self.gain(row) - self.gain(other)
        return "tie" if abs(diff) <= self.band(row["design"], row["brief"]) + 1e-12 else ("win" if diff > 0 else "loss")


Rows = dict[str, dict[RowKey, list[dict[str, Any]]]]      # approach -> (design, brief) -> rows (runs)


def per_cell(rows: Rows, approach: str, key: RowKey, *, run_tag: str | None = None) -> dict[str, Any] | None:
    items = rows.get(approach, {}).get(key, [])
    if run_tag is not None:
        items = [r for r in items if r.get("tag") == run_tag]
    return items[0] if items else None


def cell_share(frame: Frame, rows: Rows, approach: str, key: RowKey, run_tag: str | None = None) -> tuple[bool, float | None]:
    """(present, share) of one cell. 'random' is the mean over its seeds; every other approach is one row."""

    items = rows.get(approach, {}).get(key, [])
    if run_tag is not None:
        items = [r for r in items if r.get("tag") == run_tag]
    if not items:
        return False, None
    if approach == "random":
        return True, mean(frame.share(r) for r in items)
    return True, frame.share(items[0])


def brief_means(frame: Frame, rows: Rows, approach: str, briefs: Iterable[str], run_tag: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for brief in briefs:
        shares, n = [], 0
        for design in frame.designs:
            present, share = cell_share(frame, rows, approach, (design, brief), run_tag)
            if present:
                n += 1
                shares.append(share)
        out[brief] = {"n": n, "mean_share": mean(shares), "defined": sum(s is not None for s in shares)}
    return out


def robustness(frame: Frame, rows: Rows, approaches: Iterable[str], briefs: Iterable[str] = ALL_BRIEFS) -> list[dict[str, Any]]:
    table = []
    for a in approaches:
        bm = brief_means(frame, rows, a, briefs)
        crit = mean(frame_share_list(frame, rows, a, CRITERION))
        four = mean(bm[b]["mean_share"] for b in FOUR if b in bm and bm[b]["mean_share"] is not None) \
            if all(bm.get(b, {}).get("mean_share") is not None for b in FOUR) else None
        worst = min((bm[b]["mean_share"] for b in FOUR if bm.get(b, {}).get("mean_share") is not None), default=None) \
            if all(bm.get(b, {}).get("mean_share") is not None for b in FOUR) else None
        table.append({"approach": a, "by_brief": bm, "a0_t0": crit, "four_brief_mean": four, "worst_brief": worst})
    return table


def frame_share_list(frame: Frame, rows: Rows, approach: str, briefs: Iterable[str]) -> list[float | None]:
    out = []
    for brief in briefs:
        for design in frame.designs:
            present, share = cell_share(frame, rows, approach, (design, brief))
            if present:
                out.append(share)
    return out


def compare(frame: Frame, rows: Rows, a: str, b: str, briefs: Iterable[str], *, tag_a: str | None = None,
            tag_b: str | None = None) -> dict[str, Any]:
    """a against b on the named briefs: mean shares, wins / ties / losses by noise band, exact sign test."""

    tally = {"win": 0, "tie": 0, "loss": 0}
    shares_a, shares_b, cells = [], [], []
    for brief in briefs:
        for design in frame.designs:
            ra = per_cell(rows, a, (design, brief), run_tag=tag_a)
            rb = per_cell(rows, b, (design, brief), run_tag=tag_b)
            if ra is None or rb is None:
                cells.append({"design": design, "brief": brief, "status": "missing"})
                continue
            if frame.no_room(design, brief):
                cells.append({"design": design, "brief": brief, "status": "no-room"})
                continue
            outcome = frame.outcome_vs(ra, rb)
            tally[outcome] += 1
            shares_a.append(frame.share(ra))
            shares_b.append(frame.share(rb))
            cells.append({"design": design, "brief": brief, "status": outcome})
    ma, mb = mean(shares_a), mean(shares_b)
    return {
        "a": a, "b": b, "briefs": list(briefs), "cells": cells, "cells_compared": sum(tally.values()),
        "no_room_cells": sum(c["status"] == "no-room" for c in cells), "missing_cells": sum(c["status"] == "missing" for c in cells),
        "mean_share_a": ma, "mean_share_b": mb,
        "difference_points": None if ma is None or mb is None else (ma - mb) * 100.0,
        "wins": tally["win"], "ties": tally["tie"], "losses": tally["loss"],
        "sign_test_p": sign_test_p(tally["win"], tally["loss"]),
    }


BASELINE_CANDIDATES = ("v1", "v2", "single", "fixed_AB", "fixed_BA")     # deterministic policies only


def choose_baselines(frame: Frame, rows: Rows, candidates: Iterable[str] = BASELINE_CANDIDATES) -> dict[str, Any]:
    """Two analysis-only baselines, chosen on the training records and then frozen by name.

    Random is a mean over seeds, not a policy one could pick, so it is not a candidate.
    """

    table = {r["approach"]: r for r in robustness(frame, rows, [c for c in candidates if c in rows], FOUR)}
    ranked = [(a, t["four_brief_mean"]) for a, t in table.items() if t["four_brief_mean"] is not None]
    single_best = max(ranked, key=lambda x: (x[1], x[0]))[0] if ranked else None
    per_brief = {}
    for brief in FOUR:
        scored = [(a, t["by_brief"][brief]["mean_share"]) for a, t in table.items() if t["by_brief"][brief]["mean_share"] is not None]
        per_brief[brief] = max(scored, key=lambda x: (x[1], x[0]))[0] if scored else None
    return {"single_best_non_llm": single_best, "per_brief_best_non_llm": per_brief}


# ------------------------------------------------------------------------------ pre-registered claims
MAIN_TAGS = ("m1", "r0")


def main_run_rows(rows: Rows) -> Rows:
    """The main run of each LLM cell (m1 on test, r0 on training); fresh repeats are kept out of every mean."""

    out: Rows = defaultdict(lambda: defaultdict(list))
    for a, cells in rows.items():
        for key, items in cells.items():
            out[a][key] = [r for r in items if a not in ("F1-R1", "GA-R1") or r.get("tag") in MAIN_TAGS]
    return out


def playbook_entry(frame: Frame, rows: Rows, per_brief: dict[str, str | None]) -> dict[str, Any]:
    by_brief = {b: brief_means(frame, rows, per_brief[b], [b])[b] for b in FOUR if per_brief.get(b)}
    return {"approach": "playbook (per-brief best non-LLM)", "by_brief": by_brief,
            "a0_t0": mean(by_brief[b]["mean_share"] for b in CRITERION if b in by_brief),
            "four_brief_mean": mean(by_brief[b]["mean_share"] for b in FOUR if b in by_brief),
            "worst_brief": min((by_brief[b]["mean_share"] for b in FOUR if b in by_brief and by_brief[b]["mean_share"] is not None),
                               default=None)}


EPS = 1e-12


def _verdict(ok: bool) -> str:
    return "supported" if ok else "not supported"


def claims(frame: Frame, rows_all: Rows, baselines: dict[str, Any], *, manifest_profiles: dict[str, str] | None = None) -> dict[str, Any]:
    """C1 to C5 of notes/analysis_plan.md, decided mechanically. Test designs only; no-room cells are already
    excluded by Frame.share and compare(). The operational definitions are written in the plan."""

    rows = main_run_rows(rows_all)
    approaches = [a for a in (*NON_LLM, *LLM) if a in rows]
    table = {t["approach"]: t for t in robustness(frame, rows, approaches)}
    single = baselines["single_best_non_llm"]
    table[f"best single non-LLM ({single})"] = table[single] if single in table else None
    table["playbook"] = playbook_entry(frame, rows, baselines["per_brief_best_non_llm"])
    worst = {a: (t["worst_brief"] if t else None) for a, t in table.items()}

    # C1: highest worst-brief share over T0, A0, T, A among all approaches and ablations and both baselines
    f1 = worst.get("F1-R1")
    others = {a: w for a, w in worst.items() if a != "F1-R1" and w is not None}
    best_other = max(others.values()) if others else None
    c1_ok = f1 is not None and all(w <= f1 + EPS for w in others.values())
    c1 = {"verdict": _verdict(c1_ok), "F1-R1 worst brief": f1, "best other worst brief": best_other,
          "highest other": sorted(a for a, w in others.items() if best_other is not None and abs(w - best_other) <= EPS),
          "worst brief by approach": worst, "undefined (a brief without cells)": sorted(a for a, w in worst.items() if w is None)}

    # C2: mean share over A0 and T0 at least v2's
    c2cmp = compare(frame, rows, "F1-R1", "v2", CRITERION)
    c2_ok = c2cmp["mean_share_a"] is not None and c2cmp["mean_share_b"] is not None \
        and c2cmp["mean_share_a"] >= c2cmp["mean_share_b"] - EPS
    c2 = {"verdict": _verdict(c2_ok), "mean_share_F1-R1": c2cmp["mean_share_a"], "mean_share_v2": c2cmp["mean_share_b"],
          "difference_points": c2cmp["difference_points"], "wins": c2cmp["wins"], "ties": c2cmp["ties"],
          "losses": c2cmp["losses"], "sign_test_p (reported, not in the verdict)": c2cmp["sign_test_p"],
          "cells_compared": c2cmp["cells_compared"], "no_room_cells": c2cmp["no_room_cells"], "missing_cells": c2cmp["missing_cells"]}

    # C3: unseen worker, P0
    p0 = compare(frame, rows, "F1-R1", "v2", ("P0",))
    p0_rows = [r for cells in [rows.get("F1-R1", {})] for (d, b), items in cells.items() if b == "P0" for r in items]
    other_rows = [r for (d, b), items in rows.get("F1-R1", {}).items() if b != "P0" for r in items]
    ident = [r.get("llm") if isinstance(r.get("llm"), dict) else None for r in p0_rows + other_rows]
    hashes = {i.get("template_sha256") for i in ident if i}
    same_template = bool(p0_rows) and bool(other_rows) and all(i and i.get("template_sha256") for i in ident) and len(hashes) == 1
    sees = bool(p0_rows) and all(
        isinstance(r.get("llm"), dict) and r["llm"].get("workers_seen") == ["area", "power", "timing"]
        and (manifest_profiles is None or r["llm"].get("profile_sha256", {}).get("power") == manifest_profiles.get("power"))
        for r in p0_rows)
    room = p0["cells_compared"] > 0
    c3_ok = same_template and sees and room and p0["losses"] == 0 and p0["missing_cells"] == 0
    c3 = {"verdict": _verdict(c3_ok), "same template hash on P0 and the other briefs": same_template,
          "template hashes": sorted(hashes), "worker C seen only through its card and profile": sees,
          "P0 cells with room compared": p0["cells_compared"], "wins": p0["wins"], "ties": p0["ties"],
          "losses beyond the power noise band": p0["losses"], "no_room_cells": p0["no_room_cells"],
          "missing_cells": p0["missing_cells"]}

    # C4: contract over direct control
    c4cmp = compare(frame, rows, "F1-R1", "GA-R1", CRITERION)
    c4_ok = c4cmp["mean_share_a"] is not None and c4cmp["mean_share_b"] is not None \
        and c4cmp["mean_share_a"] > c4cmp["mean_share_b"] + EPS
    c4 = {"verdict": _verdict(c4_ok), "mean_share_F1-R1": c4cmp["mean_share_a"], "mean_share_GA-R1": c4cmp["mean_share_b"],
          "difference_points": c4cmp["difference_points"], "cells_compared": c4cmp["cells_compared"],
          "missing_cells": c4cmp["missing_cells"]}
    return {"C1 robustness": c1, "C2 versus rule-based": c2, "C3 unseen worker": c3, "C4 contract over direct control": c4,
            "C5 cost": "no verdict: see the cost table", "table": table}


def no_verdict_reports(frame: Frame, rows_all: Rows, baselines: dict[str, Any]) -> dict[str, Any]:
    rows = main_run_rows(rows_all)
    out: dict[str, Any] = {
        "wording sensitivity: F1-R1 vs F1-R0 (all five briefs)": compare(frame, rows, "F1-R1", "F1-R0", ALL_BRIEFS),
        "ablation: F1-R1 vs E0-R1 (all five briefs)": compare(frame, rows, "F1-R1", "E0-R1", ALL_BRIEFS),
    }
    repeat = {}
    for a in ("F1-R1", "GA-R1"):
        for key, items in rows_all.get(a, {}).items():
            tags = {r["tag"]: r["final_state_id"] for r in items}
            if len(tags) > 1:
                repeat[f"{a}:{key[0]}:{key[1]}"] = len(set(tags.values())) == 1
    out["fresh repeats reproduce the final state (determinism)"] = {
        "cells": len(repeat), "identical": sum(repeat.values()), "different": sorted(k for k, v in repeat.items() if not v)}
    single = baselines["single_best_non_llm"]
    out["best single policy and playbook on every brief"] = {
        "best single": {b: brief_means(frame, rows, single, [b])[b] for b in ALL_BRIEFS},
        "playbook (P0 has no entry)": playbook_entry(frame, rows, baselines["per_brief_best_non_llm"])["by_brief"]}
    return out


def print_verdicts(result: dict[str, Any]) -> str:
    lines = [f"{k}: {v['verdict']}" if isinstance(v, dict) else f"{k}: {v}" for k, v in result.items() if k != "table"]
    text = "\n".join(lines)
    print(text)
    return text


# ------------------------------------------------------------------------------ training records
def _bucket(rows: list[dict[str, Any]], approach_of: Callable[[dict[str, Any]], str | None]) -> Rows:
    out: Rows = defaultdict(lambda: defaultdict(list))
    for r in rows:
        a = approach_of(r)
        if a:
            out[a][(r["design"], r["brief"])].append(r)
    return out


def training_frame() -> Frame:
    import thursday_overnight_power as campaign

    starts = fa.start_metrics()
    bands = {d: dict(v) for d, v in fa.noise_bands().items()}
    selected = campaign.selected()["by_design"]
    for d in fa.DESIGNS:
        bands[d]["dynamic_power_w"] = selected[d]["noise_power_pct"] / 100.0 * starts[d]["dynamic_power_w"]
    best: dict[RowKey, float | None] = {(d, b): v["value"] for (d, b), v in fa.best_known().items()}
    for d in fa.DESIGNS:
        best[(d, "P0")] = p0_best_known().get(d)
    return Frame(fa.DESIGNS, starts, bands, best, room_rule=False)


def p0_best_known() -> dict[str, float]:
    """Lowest P0-valid dynamic power in the training power campaign and the P0 pilot (as friday_p0_report)."""

    import thursday_overnight_power as campaign
    import thursday_overnight_power_report as powerrep
    from orch.contract.brief_text import load_frozen_brief
    from orch.contract.models import MetricName
    from orch.engine.gate import judge
    from orch.engine.journal import Journal
    from orch.engine.store import StateStore
    from scripts import friday_p0 as p0

    brief = load_frozen_brief("P0")
    starts = fa.start_evidence()
    store = StateStore(fa.V4)
    power = MetricName.DYNAMIC_POWER_W
    best: dict[str, float] = {}

    def offer(design, evidence):
        if judge(brief, starts[design], starts[design], evidence).brief_valid:
            value = evidence.metric(power).value
            if design not in best or value < best[design]:
                best[design] = value

    for design in fa.DESIGNS:
        offer(design, starts[design])
        for phase in ("a", "b"):
            committed = powerrep.committed(phase, design)
            for evidence in powerrep.completed_evidence(phase, design, committed).values():
                offer(design, evidence)
    for design, brief_id, key in p0.rows():
        for event in Journal(fa.V4).read(p0.episode_id(design, brief_id, key)):
            if event.get("type") == "step" and event["verdict"]["physically_valid"]:
                outcome = store.load_outcome(event["step"]["run_id"])
                if outcome.evidence is not None:
                    offer(design, outcome.evidence)
    return best


def training_rows() -> Rows:
    from scripts import saturday_llm_report as rep

    legacy = {"repair_loop": "v1", "single_specialist": "single", "fixed_AB": "fixed_AB", "fixed_BA": "fixed_BA",
              "random_seed_0": "random_0", "random_seed_1": "random_1", "random_seed_2": "random_2"}
    pool = [dict(r, approach=legacy[r["policy"]]) for r in fa.result_rows(fa.V4)
            if r["policy"] in legacy and r["brief"] in FOUR]
    pool += [dict(r, approach="v2") for r in rep.reference_rows()["v2"].values()]
    for stage, config, name in (("grid2", "F1", "F1-R1"), ("grid1", "F1", "F1-R0"), ("grid2", "E0", "E0-R1"), ("ga1", "GA", "GA-R1")):
        pool += [dict(r, approach=name, tag="r0") for r in rep.stage_rows(stage) if r["policy"] == config and r["cell"]["tag"] == "r0"]
    p0_map = {"repair_loop": "v1", "single_specialist": "single", **{f"fixed_{p}": f"fixed_{p}" for p in
              ("ABC", "ACB", "BAC", "BCA", "CAB", "CBA")},
              **{f"random_seed_{i}": f"random_{i}" for i in range(3)}}
    pool += [dict(r, approach=p0_map[r["policy"]]) for r in fa.result_rows(fa.P0ROOT) if r["policy"] in p0_map]
    rows = _bucket(pool, lambda r: r["approach"])
    # 'random' is the mean over its three seeds: one pseudo-approach with one row per seed
    for key in {k for i in range(3) for k in rows.get(f"random_{i}", {})}:
        rows["random"][key] = [r for i in range(3) for r in rows.get(f"random_{i}", {}).get(key, [])]
    return rows


def pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.1f}"


def render_table(table: list[dict[str, Any]], title: str) -> str:
    lines = [f"### {title}", "", "| approach | T0 | A0 | T | A | P0 | A0+T0 | four-brief mean | worst brief |", "|---|---|---|---|---|---|---|---|---|"]
    for t in table:
        b = t["by_brief"]
        lines.append(f"| {t['approach']} | " + " | ".join(pct(b.get(x, {}).get("mean_share")) for x in ALL_BRIEFS)
                     + f" | {pct(t['a0_t0'])} | {pct(t['four_brief_mean'])} | {pct(t['worst_brief'])} |")
    return "\n".join(lines) + "\n"


def mean_over_runs(rows: Rows) -> Rows:
    return rows


def training_table(out: Path) -> dict[str, Any]:
    frame, rows = training_frame(), training_rows()
    approaches = [a for a in (*NON_LLM, *LLM) if a in rows]
    table = robustness(frame, rows, approaches)
    base = choose_baselines(frame, rows)
    single, per_brief = base["single_best_non_llm"], base["per_brief_best_non_llm"]

    def playbook() -> dict[str, Any]:
        return playbook_entry(frame, rows, per_brief)

    extra = [t for t in robustness(frame, rows, [single]) if single] if single else []
    for t in extra:
        t["approach"] = f"best single non-LLM ({single})"
    table += extra + [playbook()]
    compares = {
        "F1-R1 vs v2 (A0+T0)": compare(frame, rows, "F1-R1", "v2", CRITERION),
        "F1-R1 vs F1-R0": compare(frame, rows, "F1-R1", "F1-R0", CRITERION),
        "F1-R1 vs E0-R1": compare(frame, rows, "F1-R1", "E0-R1", CRITERION),
        "GA-R1 vs F1-R1": compare(frame, rows, "GA-R1", "F1-R1", CRITERION),
    }
    verdicts = claims(frame, rows, base)
    result = {"schema_version": "campaign-training-table-v1", "baselines": base, "table": table, "compares": compares,
              "claims_on_training": {k: v for k, v in verdicts.items() if k != "table"},
              "definitions": "scripts/friday_analysis.py share/improvement; win/tie/loss by gain difference vs the design's band; no-room rule off on training"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "training_robustness.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    md = render_table(table, "Training robustness table: share of best known (%), mean over five designs")
    md += "\n" + "\n".join(f"* {k}: wins/ties/losses {c['wins']}/{c['ties']}/{c['losses']}, sign-test p {c['sign_test_p']}, "
                            f"mean difference {c['difference_points']:+.1f} points ({c['cells_compared']} cells)"
                            if c['difference_points'] is not None else f"* {k}: no cells" for k, c in compares.items()) + "\n"
    md += f"\nAnalysis-only baselines (chosen on training): single best = {single}; per-brief best = {per_brief}.\n"
    md += "\n### Pre-registered claims applied to the training records (five training designs; mechanical)\n\n"
    md += "\n".join(f"* {k}: {v['verdict']}" if isinstance(v, dict) else f"* {k}: {v}"
                    for k, v in verdicts.items() if k != "table") + "\n"
    (out / "training_robustness.md").write_text(md, encoding="utf-8")
    (out / "training_robustness.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(md)
    return result


# ------------------------------------------------------------------------------ campaign records
def void_runs(work: Path) -> set[int]:
    return {int(p.parent.name) for p in (work / "runs").glob("*/VOID.json")} if (work / "runs").is_dir() else set()


def campaign_rows(work: Path) -> Rows:
    voids = void_runs(work)
    pool = []
    for p in sorted((work / "results").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("run_no") in voids:
            continue
        pool.append(r)
    rows = _bucket(pool, lambda r: r["approach"])
    for key in {k for i in range(3) for k in rows.get(f"random_{i}", {})}:
        rows["random"][key] = [r for i in range(3) for r in rows.get(f"random_{i}", {}).get(key, [])]
    return rows


def pool_best(manifest, designs, starts_ev: dict[str, Any], finals: list[dict[str, Any]],
              amd_ok: Callable[[str, str], bool], briefs: Iterable[str] = ALL_BRIEFS) -> dict[RowKey, float | None]:
    """Best brief-valid state in the stored cache or among campaign finals, per (design, brief).

    Valid = the contract's brief check (judge, relative to the start), no more DONT_TOUCH / fixed cells
    than the start, and an AMD PASS record for stored states. Reads stored outcome values, so it runs only
    on a finished campaign (or on training records).
    """

    from orch.contract.brief_text import load_frozen_brief
    from orch.engine.gate import judge
    from orch.engine.manifest_cache import _read_run
    import campaign_harness as h

    frozen = {b: load_frozen_brief(b) for b in briefs}
    cands: dict[RowKey, list[tuple[float, str]]] = defaultdict(list)
    for entry in manifest.entries.values():
        if entry["status"] != "ok" or not entry["output_state_id"]:
            continue
        run, state, evidence, _ = _read_run(manifest.roots[entry["root"]] / "runs" / entry["run_id"], entry["run_id"])
        if state is None or evidence is None or evidence.fake or state.design not in designs:
            continue
        start = starts_ev[state.design]
        try:
            h.start_relative_check(evidence, start)
        except Exception:
            continue
        for b, brief in frozen.items():
            if judge(brief, start, start, evidence).brief_valid:
                cands[(state.design, b)].append((evidence.metric(brief.objective.metric).value, state.id))
    best: dict[RowKey, float | None] = {}
    for design in designs:
        start = starts_ev[design]
        for b, brief in frozen.items():
            metric, direction = OBJECTIVE[b]
            options = [start.metric(brief.objective.metric).value] if judge(brief, start, start, start).brief_valid else []
            options += [r["final_metrics"][metric] for r in finals
                        if r["design"] == design and r["brief"] == b and r.get("final_brief_valid")]
            for value, state_id in sorted(cands[(design, b)], reverse=(direction == "max")):
                if amd_ok(design, state_id):
                    options.append(value)
                    break
            best[(design, b)] = (max if direction == "max" else min)(options) if options else None
    return best


def cost_table(work: Path, rows: Rows) -> dict[str, Any]:
    from collections import Counter

    episodes = {r["episode_id"]: a for a, cells in rows.items() if a != "random" for items in cells.values() for r in items}
    tokens, usd, fresh_tokens = Counter(), Counter(), Counter()
    log = work / "llm_calls.jsonl"
    if log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            a = episodes.get(c["episode_id"])
            if a is None:
                continue
            tokens[a] += c["total_tokens"]
            if c["charged"]:
                fresh_tokens[a] += c["total_tokens"]
                usd[a] += c["usd"]
    accounted = json.loads((work / "extraction_accounted.json").read_text()) if (work / "extraction_accounted.json").is_file() else {}
    used_states: dict[str, set[str]] = defaultdict(set)
    if (work / "extraction_use.jsonl").is_file():
        for line in (work / "extraction_use.jsonl").read_text(encoding="utf-8").splitlines():
            u = json.loads(line)
            used_states[u["episode_id"]].add(u["state_sha256"])
    reserved: dict[str, float] = {}
    caps = work / "caps.json"
    if caps.is_file():
        for res in json.loads(caps.read_text())["reservations"]:
            info = work / "engine" / "runs" / res["run_id"] / "runinfo.json"
            if info.is_file():
                d = json.loads(info.read_text())
                reserved[res["episode_id"]] = reserved.get(res["episode_id"], 0.0) + (d.get("cpu_user_s") or 0) + (d.get("cpu_sys_s") or 0)
    out: dict[str, Any] = {}
    for a, cells in rows.items():
        if a == "random":
            continue
        items = [r for rs in cells.values() for r in rs]
        n = max(len(items), 1)
        eps = {r["episode_id"] for r in items}
        states = {s for e in eps for s in used_states.get(e, ())}
        out[a] = {
            "episodes": len(items), "new_real_runs": sum(r.get("new_real_runs", 0) for r in items),
            "recipe_runs_per_episode": round(sum(r.get("runs_used", 0) for r in items) / n, 3),
            "new_cpu_hours": round(sum(reserved.get(r["episode_id"], 0.0) for r in items) / 3600.0, 3),
            "charged_cpu_hours": round(sum(r.get("charged_cpu_s", 0.0) for r in items) / 3600.0, 3),
            "extraction_states_touched": len(states),
            "extraction_job_hours_of_states_touched (worker B telemetry, apart from recipe runs)": round(
                sum(accounted.get(s, {}).get("wall_s_total", 0.0) for s in states) / 3600.0, 3),
            "tokens_all_logged_incl_cache_hits": tokens[a], "tokens_fresh": fresh_tokens[a],
            "tokens_fresh_per_episode": round(fresh_tokens[a] / n, 1),
            "usd": round(usd[a], 4), "usd_per_episode": round(usd[a] / n, 5),
        }
    requested: dict[str, set[str]] = defaultdict(set)
    from orch.engine.journal import Journal
    journal = Journal(work / "engine")
    for cells in rows.values():
        for items in cells.values():
            for r in items:
                for e in journal.read(r["episode_id"]):
                    if e.get("type") == "mandate":
                        requested[e["mandate"]["base_state_id"]].add(e["mandate"]["worker_id"])

    def hours(states) -> float:
        return round(sum(accounted.get(s, {}).get("wall_s_total", 0.0) for s in states) / 3600.0, 3)

    area_states = [s for s in accounted if "area" in requested.get(s, ())]
    timing_only = [s for s in accounted if requested.get(s) == {"timing"}]
    out["__extraction_total__"] = {
        "by_worker": {"area (worker B: ranks on the features)": {"states": len(area_states), "job_hours": hours(area_states)},
                      "timing (worker A) only: asked and ignored": {"states": len(timing_only), "job_hours": hours(timing_only)}},
        "states_extracted": len(accounted), "vivado_launches": sum(v.get("vivado_launches", 0) for v in accounted.values()),
        "wall_job_hours": round(sum(v["wall_s_total"] for v in accounted.values()) / 3600.0, 3),
        "elapsed_hours": round(sum(v["elapsed_s"] for v in accounted.values()) / 3600.0, 3)}
    return out


def infra_report(work: Path) -> dict[str, Any]:
    """Cells that ended as infrastructure failures: re-queued once; a second failure drops the cell from the
    comparisons it is part of (it shows as a missing cell there) and is never substituted."""

    path = work / "infra_failures.jsonl"
    entries = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.is_file() else []
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        by_cell[e["episode_id"]].append(e)
    finished = {p.stem for p in (work / "results").glob("*.json")} if (work / "results").is_dir() else set()
    return {
        "failed_once_then_completed": sorted(c for c, v in by_cell.items() if c in finished),
        "dropped_after_a_second_failure": [
            {"episode": c, "approach": v[-1]["approach"], "design": v[-1]["design"], "brief": v[-1]["brief"],
             "kinds": [x["kind"] for x in v]}
            for c, v in sorted(by_cell.items()) if len(v) >= 2 and c not in finished],
        "by_kind": {k: sum(e["kind"] == k for e in entries) for k in ("extraction", "api")},
        "failure_events": len(entries),
        # not an orchestration decision either, but outside decision (d): the engine's own uncharged stop when every
        # recipe attempt stayed a transient tool failure. Counted and listed here, not re-queued.
        "episodes_ended_by_transient_unresolved": sorted(
            r["episode_id"] for r in (json.loads(x.read_text(encoding="utf-8")) for x in (work / "results").glob("*.json"))
            if r.get("stop_reason") in ("transient_unresolved", "no_recipes_available")) if (work / "results").is_dir() else []}


def analyse(work: Path, scope: str, out: Path, frame: Frame | None = None) -> dict[str, Any]:
    rows = campaign_rows(work)
    if frame is None:
        frame = training_frame() if scope == "train" else None
    if frame is None:
        raise SystemExit("the test frame needs the finished campaign: build it with pool_best() and pass it in")
    if scope == "train":                                  # keep to the campaign's own designs
        designs = tuple(sorted({d for cells in rows.values() for d, _ in cells}))
        frame = Frame(designs, frame.starts, frame.bands, frame.best, frame.room_rule)
    frozen = json.loads((PROJECT / "data/campaign_prep_2026-11-07/analysis/training_robustness.json").read_text())["baselines"]
    approaches = [a for a in (*NON_LLM, *LLM) if a in rows]
    table = robustness(frame, rows, approaches)
    primary = compare(frame, rows, "F1-R1", "v2", CRITERION, tag_a="m1")
    secondary: dict[str, Any] = {
        "F1-R1 vs v2, all five briefs": compare(frame, rows, "F1-R1", "v2", ALL_BRIEFS, tag_a="m1"),
        "P0 (unseen worker): F1-R1 vs v2": compare(frame, rows, "F1-R1", "v2", ("P0",), tag_a="m1"),
        "wording: F1-R1 vs F1-R0": compare(frame, rows, "F1-R1", "F1-R0", ALL_BRIEFS, tag_a="m1"),
        "ablation: F1-R1 vs E0-R1": compare(frame, rows, "F1-R1", "E0-R1", ALL_BRIEFS, tag_a="m1"),
        "general agent: GA-R1 vs F1-R1": compare(frame, rows, "GA-R1", "F1-R1", ALL_BRIEFS, tag_a="m1", tag_b="m1"),
        "general agent: GA-R1 vs v2": compare(frame, rows, "GA-R1", "v2", ALL_BRIEFS, tag_a="m1"),
    }
    if "F1-R1" in rows:
        repeat = {}
        for key, items in rows["F1-R1"].items():
            tags = {r["tag"]: r["final_state_id"] for r in items}
            if len(tags) > 1:
                repeat[f"{key[0]}:{key[1]}"] = len(set(tags.values())) == 1
        secondary["F1-R1 repeats reproduce the final state"] = repeat
    result = {"schema_version": "campaign-analysis-v1", "scope": scope, "baselines_frozen_on_training": frozen,
              "primary": primary, "robustness": table, "secondary": secondary, "cost": cost_table(work, rows),
              "no_room_cells": [f"{d}:{b}" for d in frame.designs for b in ALL_BRIEFS if frame.no_room(d, b)]}
    profiles = json.loads((PROJECT / "configs/profiles/manifest.json").read_text(encoding="utf-8"))["files"]
    verdicts = claims(frame, rows, frozen, manifest_profiles={w: profiles[f"{w}.yaml"] for w in ("timing", "area", "power")})
    result["claims"] = {k: v for k, v in verdicts.items() if k != "table"}
    result["reported_without_verdict"] = no_verdict_reports(frame, rows, frozen)
    result["infrastructure_failures"] = infra_report(work)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"analysis_{work.name}.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    keys = ("wins", "ties", "losses", "sign_test_p", "mean_share_a", "mean_share_b", "difference_points", "no_room_cells")
    md = render_table(table, f"Robustness: share of best known (%), {work.name}")
    md += f"\nPrimary (A0+T0, F1-R1 against v2): {json.dumps({k: primary[k] for k in keys}, default=str)}\n"
    md += f"\nCost: {json.dumps(result['cost'], default=str)}\n"
    md += "\nPre-registered claims (mechanical):\n" + "\n".join(
        f"* {k}: {v['verdict']}" if isinstance(v, dict) else f"* {k}: {v}" for k, v in result["claims"].items()) + "\n"
    md += f"\nInfrastructure failures: {json.dumps(result['infrastructure_failures'], default=str)}\n"
    (out / f"analysis_{work.name}.md").write_text(md, encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--training-table", action="store_true")
    modes.add_argument("--campaign", default=None, help="campaign work name, e.g. rehearsal1")
    parser.add_argument("--scope", choices=("train", "test"), default="train")
    parser.add_argument("--out", type=Path, default=PROJECT / "data/campaign_prep_2026-11-07/analysis")
    args = parser.parse_args()
    if args.training_table:
        training_table(args.out)
    else:
        result = analyse(PROJECT / f"data/campaign_{args.campaign}", args.scope, args.out)
        print((args.out / f"analysis_campaign_{args.campaign}.md").read_text(encoding="utf-8"))
        print_verdicts(result["claims"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
