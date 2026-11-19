"""Tables for the Saturday LLM report (training designs only; read-only).

Reads the result rows written by scripts/saturday_llm.py and the rule-based reference results:
  * v1 = the pilot's repair loop, v2 = the variant chosen by the authors' Saturday rule
    (data/pilots/saturday_sweep_ta_2026-11-06/selection_saturday.json).
Shares use the pilot's fixed best-known denominators (scripts/friday_analysis.py). A "win", "tie" or
"loss" compares a configuration with the rule-based v2 on one design and brief: a tie when the
difference in the brief's objective is inside the design's noise band.

Selection (spec section 10, step 3): leave one design out, choose the configuration with the best mean
share over A0 and T0 on the four other designs, record its held-out result; F1 stays the main
configuration unless another beats it on the held-out mean by at least 5 points without lower choice
diversity (Shannon entropy of the (worker, setting, strategy) choices on A0 and T0).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts import friday_analysis as fa
from scripts import friday_sweep as sweep
from scripts import saturday_sweep_ta as ext

PROJECT = fa.PROJECT
BASE = PROJECT / "data/pilots"
DATE = "2026-11-06"
CRITERION = ("A0", "T0")
ORDER = ("F1", "F0", "E1", "E0", "F2")           # tie-break order: the intended main configuration first
MARGIN = 5.0


def stage_rows(stage: str) -> list[dict[str, Any]]:
    folder = BASE / f"saturday_llm_{stage}_{DATE}" / "results"
    return [fa.read(p) for p in sorted(folder.glob("*.json"))] if folder.is_dir() else []


def reference_rows() -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    v2_id = fa.read(ext.ROOT / "selection_saturday.json")["v2"]["grid_id"]
    v2: dict[tuple[str, str], dict[str, Any]] = {}
    for root in (sweep.ROOT, ext.ROOT):
        for row in fa.result_rows(root):
            if row.get("grid_id") == v2_id:
                v2[(row["design"], row["brief"])] = row
    v1 = {(r["design"], r["brief"]): r for r in fa.result_rows(fa.V4) if r["policy"] == "repair_loop"}
    return {"v1": v1, "v2": v2, "v2_id": {("id", "id"): {"grid_id": v2_id}}}


class Scorer:
    def __init__(self) -> None:
        self.starts, self.best, self.bands = fa.start_metrics(), fa.best_known(), fa.noise_bands()

    def value(self, row: dict[str, Any]) -> float:
        return row["final_metrics"][fa.OBJECTIVE[row["brief"]][0]]

    def share(self, row: dict[str, Any]) -> float | None:
        return fa.share(row["brief"], row["design"], self.value(row), self.starts, self.best)

    def gain(self, row: dict[str, Any]) -> float:
        return fa.improvement(row["brief"], row["design"], self.value(row), self.starts)

    def band(self, design: str, brief: str) -> float:
        return self.bands[design].get(fa.OBJECTIVE[brief][0], 0.0)

    def outcome_vs(self, row: dict[str, Any], other: dict[str, Any]) -> str:
        diff = self.gain(row) - self.gain(other)
        band = self.band(row["design"], row["brief"])
        return "tie" if abs(diff) <= band + 1e-12 else ("win" if diff > 0 else "loss")


def mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    return -sum(c / total * math.log2(c / total) for c in counter.values()) if total else 0.0


def choice_key(choice: dict[str, Any]) -> tuple:
    mandate = choice.get("mandate") or {}
    if choice["action"] == "STOP":
        return ("STOP",)
    if "recipe" in mandate:
        return (mandate["recipe"], choice["strategy"])
    return (mandate.get("worker"), mandate.get("preserve_placement"), choice["strategy"])


def config_rows(rows: list[dict[str, Any]], config: str, *, tag: str = "r0") -> list[dict[str, Any]]:
    return [r for r in rows if r["policy"] == config and r["cell"]["tag"] == tag]


def own_fold(row: dict[str, Any]) -> bool:
    fold = row["cell"]["fold"]
    return fold is None or fold == row["design"]


def per_config(rows, config, sc: Scorer, ref, *, tag="r0") -> dict[str, Any]:
    mine = [r for r in config_rows(rows, config, tag=tag) if own_fold(r)]
    by_brief: dict[str, dict[str, Any]] = {}
    for brief in fa.BRIEFS:
        cells = [r for r in mine if r["brief"] == brief]
        by_brief[brief] = {
            "n": len(cells), "share": mean([sc.share(r) for r in cells]),
            "gain": mean([sc.gain(r) for r in cells]),
        }
    wtl = Counter()
    for r in mine:
        other = ref["v2"].get((r["design"], r["brief"]))
        if other is not None:
            wtl[sc.outcome_vs(r, other)] += 1
    crit = [r for r in mine if r["brief"] in CRITERION]
    choices = Counter(choice_key(c) for r in crit for c in r["llm"]["choices"])
    allrows = config_rows(rows, config, tag=tag)
    return {
        "config": config, "cells": len(mine), "by_brief": by_brief,
        "share_a0_t0": mean([sc.share(r) for r in crit]),
        "share_all": mean([sc.share(r) for r in mine]),
        "win_tie_loss": [wtl["win"], wtl["tie"], wtl["loss"]],
        "mean_runs": mean([r["runs_used"] for r in mine]),
        "new_real_runs": sum(r["new_real_runs_reserved"] for r in allrows),
        "tokens_total": sum(r["llm"]["tokens"]["total"] for r in allrows),
        "tokens_per_episode": mean([r["llm"]["tokens"]["total"] for r in mine]),
        "decisions": sum(r["llm"]["decisions"] for r in mine),
        "repair_turns": sum(r["llm"]["repairs"] for r in mine),
        "fallbacks": sum(r["llm"]["fallbacks"] for r in mine),
        "stalls": sum(r["stop_reason"] == "no_recipes_available" for r in mine),
        "diversity_bits": entropy(choices), "distinct_choices": len(choices),
    }


def share_for(rows, sc, config, design, brief, fold=None) -> float | None:
    for r in rows:
        if r["policy"] == config and r["design"] == design and r["brief"] == brief and r["cell"]["tag"] == "r0":
            if config != "F2" or r["cell"]["fold"] == fold:
                return sc.share(r)
    return None


def lodo(rows, sc: Scorer, configs) -> dict[str, Any]:
    folds, held_results = [], []
    for held in fa.DESIGNS:
        train = [d for d in fa.DESIGNS if d != held]
        scores = {}
        for c in configs:
            vals = [share_for(rows, sc, c, d, b, fold=held if c == "F2" else None) for d in train for b in CRITERION]
            scores[c] = None if any(v is None for v in vals) else sum(vals) / len(vals)
        valid = {c: v for c, v in scores.items() if v is not None}
        chosen = max(valid, key=lambda c: (valid[c], -ORDER.index(c))) if valid else None
        result = None
        if chosen is not None:
            vals = [share_for(rows, sc, chosen, held, b, fold=held if chosen == "F2" else None) for b in CRITERION]
            result = None if any(v is None for v in vals) else sum(vals) / len(vals)
        if result is not None:
            held_results.append(result)
        folds.append({"held_out": held, "chosen": chosen, "train_scores": scores, "held_out_result": result})
    held_mean = {}
    for c in configs:
        vals = [share_for(rows, sc, c, d, b, fold=d if c == "F2" else None) for d in fa.DESIGNS for b in CRITERION]
        held_mean[c] = None if any(v is None for v in vals) else sum(vals) / len(vals)
    return {"folds": folds, "procedure_mean_held_out": mean(held_results), "held_out_mean_by_config": held_mean}


def decide_main(tables: dict[str, dict[str, Any]], held_mean: dict[str, float | None]) -> dict[str, Any]:
    base = held_mean.get("F1")
    rivals = []
    for c, value in held_mean.items():
        if c == "F1" or value is None or base is None:
            continue
        gain = (value - base) * 100.0
        diversity_ok = tables[c]["diversity_bits"] >= tables["F1"]["diversity_bits"] - 1e-9
        rivals.append({"config": c, "gain_points": gain, "diversity_ok": diversity_ok,
                       "qualifies": gain >= MARGIN and diversity_ok})
    winners = [r for r in rivals if r["qualifies"]]
    main = max(winners, key=lambda r: r["gain_points"])["config"] if winners else "F1"
    return {"main": main, "rivals": rivals, "f1_held_out_mean": base}


def variance(rows, config: str, sc: Scorer) -> dict[str, Any]:
    runs: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        if r["policy"] == config and own_fold(r):
            runs[(r["design"], r["brief"])][r["cell"]["tag"]] = r
    spreads, identical_final, identical_path, n = [], 0, 0, 0
    for cell, tags in runs.items():
        if len(tags) < 2:
            continue
        shares = [sc.share(t) for t in tags.values() if sc.share(t) is not None]
        if len(shares) >= 2:
            m = sum(shares) / len(shares)
            spreads.append(math.sqrt(sum((s - m) ** 2 for s in shares) / (len(shares) - 1)))
        n += 1
        finals = {t["final_state_id"] for t in tags.values()}
        paths = {json.dumps([choice_key(c) for c in t["llm"]["choices"]], default=str) for t in tags.values()}
        identical_final += len(finals) == 1
        identical_path += len(paths) == 1
    return {"cells_with_repeats": n, "mean_share_std_points": (mean(spreads) or 0.0) * 100.0 if spreads else None,
            "identical_final_cells": identical_final, "identical_decision_paths": identical_path}


def pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def render(rows, sc, ref, configs) -> tuple[str, dict[str, Any]]:
    tables = {c: per_config(rows, c, sc, ref) for c in configs if config_rows(rows, c)}
    sel = lodo(rows, sc, [c for c in ORDER if c in tables])
    decision = decide_main(tables, sel["held_out_mean_by_config"]) if "F1" in tables else None
    lines = ["# Saturday LLM tables (training designs)", "",
             f"Reference: rule-based v2 = {ref['v2_id'][('id', 'id')]['grid_id']}; shares use the fixed best-known pool.", "",
             "## Per configuration", "",
             "| config | cells | share A0 | share T0 | share T | share A | A0+T0 | gain A0 (LUT) | gain T0 (ns) | W/T/L vs v2 | mean runs | new real runs | tokens/episode | repairs | fallbacks | stalls | diversity (bits) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, t in tables.items():
        b = t["by_brief"]
        g = lambda brief: "-" if b[brief]["gain"] is None else f"{b[brief]['gain']:.3g}"  # noqa: E731
        lines.append(
            f"| {c} | {t['cells']} | {pct(b['A0']['share'])} | {pct(b['T0']['share'])} | {pct(b['T']['share'])} | {pct(b['A']['share'])} | "
            f"{pct(t['share_a0_t0'])} | {g('A0')} | {g('T0')} | {t['win_tie_loss'][0]}/{t['win_tie_loss'][1]}/{t['win_tie_loss'][2]} | "
            f"{t['mean_runs']:.2f} | {t['new_real_runs']} | {t['tokens_per_episode']:.0f} | {t['repair_turns']} | {t['fallbacks']} | {t['stalls']} | {t['diversity_bits']:.2f} |")
    refs = {k: [r for r in ref[k].values()] for k in ("v1", "v2")}
    for k, items in refs.items():
        crit = [sc.share(r) for r in items if r["brief"] in CRITERION]
        per = {b: [r for r in items if r["brief"] == b] for b in fa.BRIEFS}
        sh = {b: mean([sc.share(r) for r in per[b]]) for b in fa.BRIEFS}
        gn = {b: mean([sc.gain(r) for r in per[b]]) for b in fa.BRIEFS}
        lines.append(
            f"| rule-based {k} | {len(items)} | {pct(sh['A0'])} | {pct(sh['T0'])} | {pct(sh['T'])} | {pct(sh['A'])} | "
            f"{pct(mean(crit))} | {gn['A0']:.3g} | {gn['T0']:.3g} | | | | | | | | |")
    lines += ["", "Mean share over all four briefs: " + ", ".join(
        f"{c} {pct(t['share_all'])}" for c, t in tables.items()) + ".", ""]
    lines += ["", "## Leave one design out (A0 and T0)", "",
              "| held-out design | chosen | held-out result |", "|---|---|---|"]
    for f in sel["folds"]:
        lines.append(f"| {f['held_out']} | {f['chosen']} | {pct(f['held_out_result'])} |")
    lines += ["", f"Procedure mean held-out share: {pct(sel['procedure_mean_held_out'])}. "
              "Held-out mean by configuration (mean over the five designs): "
              + ", ".join(f"{c} {pct(v)}" for c, v in sel["held_out_mean_by_config"].items()) + ".", ""]
    if decision:
        lines += ["## Main configuration", "",
                  f"Main = **{decision['main']}** (F1 unless another beats it by at least {MARGIN:.0f} points with no lower diversity).",
                  *[f"* {r['config']}: {r['gain_points']:+.1f} points against F1; diversity {'ok' if r['diversity_ok'] else 'LOWER'}; "
                    f"{'qualifies' if r['qualifies'] else 'does not qualify'}." for r in decision["rivals"]], ""]
    return "\n".join(lines) + "\n", {"tables": tables, "lodo": sel, "decision": decision}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="grid1")
    parser.add_argument("--configs", default=",".join(ORDER))
    parser.add_argument("--variance-config", default=None)
    parser.add_argument("--out", type=Path, default=PROJECT / "data/scratch_saturday_2026-11-06/llm_tables")
    args = parser.parse_args()
    rows = [r for stage in args.stages.split(",") for r in stage_rows(stage)]
    sc, ref = Scorer(), reference_rows()
    text, data = render(rows, sc, ref, args.configs.split(","))
    if args.variance_config:
        data["variance"] = variance(rows, args.variance_config, sc)
        text += f"\n## Variance of {args.variance_config}\n\n{json.dumps(data['variance'], indent=1)}\n"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tables.md").write_text(text, encoding="utf-8")
    (args.out / "tables.json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
