"""Tables, determinism check and the declared rule for the stronger-model test (item 4; training only).

Reads the result rows written by scripts/saturday_llm.py for the stages of the stronger-model test and for the
earlier flash-lite stages, with the same shares, noise-band ties and fixed best-known pools as the Saturday report
(scripts/saturday_llm_report.py), and applies the rule the authors declared on 2026-11-07:

  switch to the stronger model only if BOTH hold
    (i)  its F1 mean A0+T0 share, averaging R0 and R1 (R1 averaged over its runs first), is at least 5 points
         above flash-lite's on the same basis;
    (ii) its F1-R1 mean over the four briefs is at least 55.0 (per cent).

The script reports the outcome; it changes no configuration.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from orch.policies.llm_client import TelemetrySink
from scripts import friday_analysis as fa
from scripts import saturday_llm_report as sat

PROJECT = fa.PROJECT
BASE = PROJECT / "data/pilots"
CRITERION = ("A0", "T0")
GAIN_POINTS = 5.0
FOUR_BRIEF_MIN = 55.0
EPS = 1e-9

Cell = tuple[str, str]


def cell_means(runs: list[dict[Cell, float]]) -> dict[Cell, float]:
    """Average each (design, brief) cell over the runs that cover it."""

    values: dict[Cell, list[float]] = defaultdict(list)
    for run in runs:
        for cell, value in run.items():
            if value is not None:
                values[cell].append(value)
    return {cell: sum(v) / len(v) for cell, v in values.items()}


def _criterion_mean(cells: dict[Cell, float]) -> float:
    picked = [v for (design, brief), v in cells.items() if brief in CRITERION]
    return sum(picked) / len(picked)


def basis_a0_t0(r0_cells: dict[Cell, float], r1_runs: list[dict[Cell, float]]) -> float:
    """Mean of the R0 A0+T0 share and the R1 A0+T0 share, the R1 value averaged over its runs first."""

    return (_criterion_mean(r0_cells) + _criterion_mean(cell_means(r1_runs))) / 2.0


def four_brief_mean(r1_runs: list[dict[Cell, float]]) -> float:
    means = cell_means(r1_runs)
    return sum(means.values()) / len(means)


def apply_rule(flash_r0, flash_r1_runs, other_r0, other_r1_runs) -> dict[str, Any]:
    """The declared rule, with flash-lite as the reference. Shares are fractions."""

    flash_basis = basis_a0_t0(flash_r0, flash_r1_runs)
    other_basis = basis_a0_t0(other_r0, other_r1_runs)
    gain = (other_basis - flash_basis) * 100.0
    four = four_brief_mean(other_r1_runs) * 100.0
    cond_i, cond_ii = gain >= GAIN_POINTS - EPS, four >= FOUR_BRIEF_MIN - EPS
    return {
        "flash_basis_percent": flash_basis * 100.0, "stronger_basis_percent": other_basis * 100.0,
        "gain_points": gain, "condition_i": cond_i,
        "stronger_f1_r1_four_brief_percent": four, "condition_ii": cond_ii, "switch": cond_i and cond_ii,
    }


def compare_replies(recorded: dict[str, str], fresh: dict[str, str]) -> dict[str, Any]:
    """Compare reply texts for prompts (cache keys) present in both namespaces."""

    both = sorted(set(recorded) & set(fresh))
    identical = sum(recorded[k] == fresh[k] for k in both)
    return {
        "prompts_in_both": len(both), "identical": identical, "differing": len(both) - identical,
        "recorded_only": len(set(recorded) - set(fresh)), "fresh_only": len(set(fresh) - set(recorded)),
        "any_differs": identical != len(both),
    }


def spend_by_stage(records: list[dict[str, Any]]) -> dict[str, float]:
    """US dollars per stage from ledger records whose source is an episode id llm_<stage>_..."""

    totals: dict[str, float] = defaultdict(float)
    for record in records:
        source = record["source"]
        stage = source.split("_")[1] if source.startswith("llm_") else source
        totals[stage] += float(record["usd"])
    return dict(totals)


# ------------------------------------------------------------------ data access (server only)
def share_cells(rows: list[dict[str, Any]], sc: sat.Scorer, config: str, tag: str) -> dict[Cell, float]:
    return {(r["design"], r["brief"]): sc.share(r) for r in rows
            if r["policy"] == config and r["cell"]["tag"] == tag and sat.own_fold(r)}


def reply_texts(cache_dir: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    for path in cache_dir.glob("*/*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        texts[record["key"]] = record["reply"]["text"]
    return texts


def ledger_records(shared: Path) -> list[dict[str, Any]]:
    path = shared / "spend.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []


def telemetry_cost(shared: Path, episodes: set[str], price_in: float, price_out: float) -> float:
    """List-price estimate for stages that ran without a ledger: non-replayed calls of these episodes."""

    total = 0.0
    for record in TelemetrySink.read(shared / "telemetry.jsonl"):
        if record["episode_id"] in episodes:
            for call in record["calls"]:
                if not call["cache_hit"]:
                    total += (call["prompt_tokens"] * price_in + (call["completion_tokens"] + call["thoughts_tokens"]) * price_out) / 1e6
    return total


def pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.1f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--argon-shared", default="saturday_llm_argon_shared")
    parser.add_argument("--flash-r1-stages", default="grid2,var1,var2")
    parser.add_argument("--flash-r0-stage", default="grid1")
    parser.add_argument("--flash-ga-stage", default="ga3")
    parser.add_argument("--flash-ga-r0-stage", default="ga1")
    parser.add_argument("--argon-r1-stage", default="ar1f")
    parser.add_argument("--argon-r0-stage", default="ar0f")
    parser.add_argument("--argon-e0-stage", default="ar1e")
    parser.add_argument("--argon-ga-stage", default="ar1g")
    parser.add_argument("--argon-extra-f1-stages", default="ar1v,ar1x,ar1y")
    args = parser.parse_args()
    sc, ref = sat.Scorer(), sat.reference_rows()
    shared_argon, shared_flash = BASE / args.argon_shared, BASE / "saturday_llm_shared"
    stage = sat.stage_rows
    split = lambda text: [s for s in text.split(",") if s]  # noqa: E731
    flash_r1_rows = [r for s in split(args.flash_r1_stages) for r in stage(s)]
    argon_extra = [r for s in split(args.argon_extra_f1_stages) for r in stage(s)]
    sections = {
        "F1-R1": ("flash-lite", flash_r1_rows, "F1", "r0"), "F1-R0": ("flash-lite", stage(args.flash_r0_stage), "F1", "r0"),
        "E0-R1": ("flash-lite", flash_r1_rows, "E0", "r0"), "GA-R1": ("flash-lite", stage(args.flash_ga_stage), "GA", "r0"),
    }
    argon_sections = {
        "F1-R1": ("gemini-4-argon", stage(args.argon_r1_stage), "F1", "r0"),
        "F1-R0": ("gemini-4-argon", stage(args.argon_r0_stage), "F1", "r0"),
        "E0-R1": ("gemini-4-argon", stage(args.argon_e0_stage), "E0", "r0"),
        "GA-R1": ("gemini-4-argon", stage(args.argon_ga_stage), "GA", "r0"),
    }
    ledger = ledger_records(shared_argon)
    by_stage = spend_by_stage(ledger)
    lines = ["# Stronger-model test on the training designs (item 4)", "",
             "Shares are per cent of the fixed best-known pool, mean over five designs, 20 cells per row; W/T/L is against rule-based v2 "
             "with noise-band ties; runs = recipe runs per episode; US$ is the total for the row's cells (list price).", "",
             "| model | cell | A0 | T0 | T | A | A0+T0 | W/T/L | runs | tokens/ep | US$ |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    table: dict[str, Any] = {}
    for label in ("F1-R1", "F1-R0", "E0-R1", "GA-R1"):
        for model, (_, rows, config, tag) in (("flash-lite", sections[label]), ("gemini-4-argon", argon_sections[label])):
            if not sat.config_rows(rows, config, tag=tag):
                lines.append(f"| {model} | {label} | not run | | | | | | | | |")
                continue
            t = sat.per_config(rows, config, sc, ref, tag=tag)
            episodes = {f"llm_{r['stage']}_{config}_{r['design']}_{r['brief']}_{tag}" for r in sat.config_rows(rows, config, tag=tag)}
            stages = {r["stage"] for r in sat.config_rows(rows, config, tag=tag)}
            if stages and all(s in by_stage for s in stages):          # a stage of this test: the ledger is exact
                usd = sum(by_stage[s] for s in stages)
            else:                                                       # a Saturday flash-lite stage: list-price estimate
                usd = telemetry_cost(shared_flash, episodes, 0.25, 1.50)
            b = t["by_brief"]
            table[f"{model}/{label}"] = {**t, "usd": usd}
            lines.append(
                f"| {model} | {label} | {pct(b['A0']['share'])} | {pct(b['T0']['share'])} | {pct(b['T']['share'])} | {pct(b['A']['share'])} | "
                f"{pct(t['share_a0_t0'])} | {t['win_tie_loss'][0]}/{t['win_tie_loss'][1]}/{t['win_tie_loss'][2]} | {t['mean_runs']:.2f} | "
                f"{t['tokens_per_episode']:.0f} | {usd:.2f} |")
    # determinism of the stronger model on F1-R1
    recorded = reply_texts(shared_argon / "llm_cache")
    determinism: dict[str, Any] = {}
    for tag_dir in sorted((shared_argon / "llm_cache" / "fresh").glob("*")) if (shared_argon / "llm_cache" / "fresh").is_dir() else []:
        determinism[tag_dir.name] = compare_replies(recorded, reply_texts(tag_dir))
    lines += ["", "## Determinism of gemini-4-argon (replies to identical prompts, recorded run against fresh repeats)", "",
              json.dumps(determinism, indent=1), ""]
    # the rule
    flash_r0 = share_cells(sections["F1-R0"][1], sc, "F1", "r0")
    flash_r1_runs = [share_cells(flash_r1_rows, sc, "F1", t) for t in ("r0", "v1", "v2")]
    flash_r1_runs = [run for run in flash_r1_runs if run]
    argon_r0 = share_cells(argon_sections["F1-R0"][1], sc, "F1", "r0")
    argon_rows = argon_sections["F1-R1"][1] + argon_extra
    argon_r1_runs = [share_cells(argon_rows, sc, "F1", t) for t in ("r0", "v1", "v2", "v3")]
    argon_r1_runs = [run for run in argon_r1_runs if run]
    if flash_r0 and flash_r1_runs and argon_r0 and argon_r1_runs:
        rule = apply_rule(flash_r0, flash_r1_runs, argon_r0, argon_r1_runs)
        rule["argon_f1_r1_runs"] = [len(r) for r in argon_r1_runs]
        lines += ["## The declared rule", "",
                  f"* (i) F1 A0+T0 basis (R0 and R1 averaged, R1 over its runs first): flash-lite {rule['flash_basis_percent']:.2f}, "
                  f"gemini-4-argon {rule['stronger_basis_percent']:.2f}; gain {rule['gain_points']:+.2f} points (needs at least +5.00): "
                  f"{'met' if rule['condition_i'] else 'not met'}.",
                  f"* (ii) gemini-4-argon F1-R1 mean over the four briefs {rule['stronger_f1_r1_four_brief_percent']:.2f} (needs at least 55.0): "
                  f"{'met' if rule['condition_ii'] else 'not met'}.",
                  f"* Outcome: **{'switch to gemini-4-argon' if rule['switch'] else 'keep gemini-3.1-flash-lite'}** "
                  "(reported only; configs/policies/llm.yaml is not changed).", ""]
    else:
        rule = {"incomplete": True}
        lines += ["## The declared rule", "", "Not computable yet: a stage is missing.", ""]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "satnight_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "satnight_tables.json").write_text(json.dumps(
        {"table": table, "determinism": determinism, "rule": rule, "spend_by_stage": by_stage}, indent=1, default=str), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
