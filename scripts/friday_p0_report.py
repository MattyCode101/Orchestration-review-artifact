"""Write notes/p0_pilot_training.md from the P0 pilot results (read-only).

Best-known P0 state = lowest dynamic power among P0-brief-valid states observed
in the training power campaign (phase a and b outputs) and in this pilot's own
journals (a lower bound on the true best). Noise = the run-4 dynamic-power band
per design. Ties = differences inside that band.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

import thursday_overnight_power as campaign  # noqa: E402
import thursday_overnight_power_report as powerrep  # noqa: E402

from orch.contract.brief_text import load_frozen_brief  # noqa: E402
from orch.contract.models import Evidence, MetricName  # noqa: E402
from orch.engine.gate import judge  # noqa: E402
from orch.engine.journal import Journal  # noqa: E402
from orch.engine.store import StateStore  # noqa: E402
from scripts import friday_analysis as fa  # noqa: E402
from scripts import friday_p0 as p0  # noqa: E402

OUTPUT = PROJECT / "notes/p0_pilot_training.md"
POWER = MetricName.DYNAMIC_POWER_W


def pct(v):
    return "-" if v is None else f"{v * 100:.1f}%"


def main() -> int:
    brief = load_frozen_brief("P0")
    starts = fa.start_evidence()
    inputs = campaign.selected()
    store = StateStore(fa.V4)
    rows = fa.result_rows(fa.P0ROOT)
    best: dict[str, float] = {}
    best_src: dict[str, str] = {}

    def offer(design: str, evidence: Evidence, source: str) -> None:
        if not judge(brief, starts[design], starts[design], evidence).brief_valid:
            return
        value = evidence.metric(POWER).value
        if design not in best or value < best[design]:
            best[design], best_src[design] = value, source

    for design in fa.DESIGNS:
        offer(design, starts[design], "B2 start")
        for phase in ("a", "b"):
            committed = powerrep.committed(phase, design)
            for evidence in powerrep.completed_evidence(phase, design, committed).values():
                offer(design, evidence, f"power campaign phase {phase}")
    episodes = {}
    for design, brief_id, key in p0.rows():
        events = Journal(fa.V4).read(p0.episode_id(design, brief_id, key))
        episodes[(design, key)] = events
        for event in events:
            if event.get("type") == "step" and event["verdict"]["physically_valid"]:
                outcome = store.load_outcome(event["step"]["run_id"])
                if outcome.evidence is not None:
                    offer(design, outcome.evidence, "P0 pilot")
    band_pct = {d: inputs["by_design"][d]["noise_power_pct"] for d in fa.DESIGNS}
    done = {(r["design"], r["policy"]): r for r in rows}

    def saving(design, row):
        start = starts[design].metric(POWER).value
        return (start - row["final_metrics"]["dynamic_power_w"]) / start * 100.0

    L = ["# P0 pilot on the training designs (minimise dynamic power)", "",
         f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC from `{fa.P0ROOT.relative_to(PROJECT)}`. "
         "Brief P0: WNS >= start and packed LUTs <= start. Power is Vivado's vectorless estimate.", "",
         "## Summary", ""]
    total = len(p0.rows())
    L.append(f"* {len(rows)}/{total} cells complete; new real runs reserved "
             f"{len(fa.read(fa.P0ROOT / 'new_real_runs.json')['reservations'])} (cap {p0.CAP}).")
    means = {}
    for key in p0.POLICIES:
        vals = []
        for design in fa.DESIGNS:
            r = done.get((design, key))
            if r is None or design not in best:
                continue
            start = starts[design].metric(POWER).value
            den = start - best[design]
            if den > 0:
                vals.append((start - r["final_metrics"]["dynamic_power_w"]) / den)
        means[key] = (sum(vals) / len(vals), len(vals)) if vals else (None, 0)
    ranked = sorted((k for k in means if means[k][0] is not None), key=lambda k: -means[k][0])
    if ranked:
        L.append("* Mean share of best known (higher is better): " + ", ".join(
            f"{k} {pct(means[k][0])} ({means[k][1]}/5)" for k in ranked) + ".")
    L += ["", "## Per policy and design: power saving, WNS and LUT change, runs", "",
          "| policy | design | saving % | dWNS ns | dLUT | runs | new | share | vs repair_loop |",
          "|---|---|---|---|---|---|---|---|---|"]
    tally = defaultdict(lambda: [0, 0, 0])
    for key in p0.POLICIES:
        for design in fa.DESIGNS:
            r = done.get((design, key))
            if r is None:
                L.append(f"| {key} | {design} | not complete | | | | | | |")
                continue
            fm, st = r["final_metrics"], fa.start_metrics()[design]
            start_p = st["dynamic_power_w"]
            den = start_p - best[design]
            share = (start_p - fm["dynamic_power_w"]) / den if den > 0 else None
            ref = done.get((design, "repair_loop"))
            verdict = "-"
            if key != "repair_loop" and ref is not None:
                diff = saving(design, r) - saving(design, ref)
                band = band_pct[design]
                verdict = "tie" if abs(diff) <= band else ("win" if diff > 0 else "loss")
                tally[key][("win", "tie", "loss").index(verdict)] += 1
            L.append(f"| {key} | {design} | {saving(design, r):+.2f} | {fm['wns_ns'] - st['wns_ns']:+.3f} | "
                     f"{fm['packed_luts'] - st['packed_luts']:+.0f} | {r['runs_used']} | "
                     f"{r['new_real_runs_reserved']} | {pct(share)} | {verdict} |")
    L += ["", "## Win / tie / loss against repair_loop (ties inside the design's power noise band)", "",
          "| policy | wins | ties | losses |", "|---|---|---|---|"]
    for key, (w, t, l) in tally.items():
        L.append(f"| {key} | {w} | {t} | {l} |")
    L += ["", "## Best-known P0 state per design (observed pool)", "",
          "| design | start power W | best power W | saving % | source | noise band % |", "|---|---|---|---|---|---|"]
    for design in fa.DESIGNS:
        if design in best:
            s = starts[design].metric(POWER).value
            L.append(f"| {design} | {s:.3f} | {best[design]:.3f} | {(s - best[design]) / s * 100:.2f} | "
                     f"{best_src[design]} | {band_pct[design]:.2f} |")
    L += ["", "## Caveats", "",
          "* The best-known pool is the training power campaign plus this pilot only (a lower bound on the true best); "
          "the four-brief pool does not cover P0.",
          "* Worker C ranking (inferred from training data, no features): P_ROUTE_HIGHDELAY > P_POST_ALL > P_POST_BRAM > P_PRE_ALL; "
          "P_PRE_ALL is treated as re-placing.",
          "* Power is a vectorless estimate; savings of a few percent are often inside the design's noise band."]
    OUTPUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
