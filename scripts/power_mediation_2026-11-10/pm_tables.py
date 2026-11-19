"""Power-mediation study: tables from a pass folder (cell_<design>_<brief>.json) and the noise table.

Everything is arithmetic on stored cell files; no run, no model. Output: <WORK>/<tag>/tables.json and
<WORK>/<tag>/tables.md.

Bands (absolute, in the objective's unit):
  WNS      configs/noise.yaml band (ns); logicnets_jscl has none there, so the run-3 Family-N range
           (results/prior/run3_an1_noise_bands.csv, same source and convention) is used and flagged.
  LUTs     configs/noise.yaml fraction x start packed LUTs; logicnets_jscl: run-3 range_pct_lut, flagged.
  power    recorded: noise_power_pct of the training power selection (design-wide Family N);
           proposed: C1 envelope for placement-keeping lineages, C2 envelope for lineages that contain a
           re-placing recipe, both floored at the 1 mW reporting step (noise_table.json).
usage: pm_tables.py <tag> [--noise noise_table.json]
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_common as pm  # noqa: E402

ORDERS = tuple("fixed_" + "".join(p) for p in ("ABC", "ACB", "BAC", "BCA", "CAB", "CBA"))
R4_PRIOR = Path("external/weekend_run4_power_2026-11-01/results/prior")
OBJ = {"P0": ("dynamic_power_w", "mW", 1000.0), "L2": ("packed_luts", "LUT", 1.0), "W2": ("wns_ns", "ps", 1000.0)}


def objective_bands(noise_table: dict) -> dict[str, dict[str, float]]:
    """Per design: wns_ns, packed_luts (absolute) and the three power bands (W)."""

    ctx_noise = pm.load_noise(pm.PROJECT / "configs/noise.yaml")
    start_ev = pm.fa.start_evidence()
    fallback = {}
    with open(R4_PRIOR / "run3_an1_noise_bands.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            fallback[row["design"]] = row
    out: dict[str, dict] = {}
    for design in pm.DESIGNS:
        ev = start_ev[design]
        bands = ctx_noise.for_start(design, ev)
        row: dict = {"flags": []}
        if pm.MetricName.WNS_NS in bands:
            row["wns_ns"] = bands[pm.MetricName.WNS_NS]
        else:
            row["wns_ns"] = float(fallback[design]["range_wns"])
            row["flags"].append("wns band from run-3 Family N range (absent from configs/noise.yaml)")
        if pm.MetricName.PACKED_LUTS in bands:
            row["packed_luts"] = bands[pm.MetricName.PACKED_LUTS]
        else:
            start_luts = ev.metric(pm.MetricName.PACKED_LUTS).value
            row["packed_luts"] = float(fallback[design]["range_pct_lut"]) / 100.0 * start_luts
            row["flags"].append("LUT band from run-3 Family N range_pct_lut (absent from configs/noise.yaml)")
        b2 = ev.metric(pm.MetricName.DYNAMIC_POWER_W).value
        nz = noise_table["designs"][design]
        res_w = 0.001
        row["power_recorded_w"] = nz["recorded_N_pct"] / 100.0 * b2
        row["power_K_w"] = max(nz["C1"]["envelope_pct"] / 100.0 * b2, res_w)
        row["power_F_w"] = max(nz["C2"]["envelope_pct"] / 100.0 * b2, res_w)
        row["b2_power_w"] = b2
        out[design] = row
    return out


def lineage_is_replacing(lineage: list[str], replaces: dict[str, bool]) -> bool:
    return any(replaces.get(r, True) for r in lineage)


def build(tag: str, noise_table: dict) -> dict:
    folder = pm.WORK / tag
    bands = objective_bands(noise_table)
    replaces = noise_table["recipes_replaces"]
    rows = []
    for f in sorted(folder.glob("cell_*_*.json")):
        cell = json.loads(f.read_text())
        design, brief = cell["design"], cell["brief"]
        metric, unit, scale = OBJ[brief] if brief in OBJ else OBJ[brief[:2]]
        band_w = bands[design]
        obj_band = {"dynamic_power_w": band_w["power_recorded_w"], "packed_luts": band_w["packed_luts"], "wns_ns": band_w["wns_ns"]}[metric]
        for budget, entry in cell["budgets"].items():
            c = entry["ceiling"]
            fixed = {n: entry["policies"][n]["improvement"] for n in ORDERS}
            best = max(fixed.values())
            winners = [n.replace("fixed_", "") for n, v in fixed.items() if best > 0 and abs(v - best) < 1e-12]
            v1, v2 = entry["policies"]["rule_v1"], entry["policies"]["rule_v2"]
            rows.append({
                "design": design, "brief": brief, "budget": int(budget), "unit": unit, "scale": scale,
                "start": cell["start"]["metrics"], "objective_band": obj_band,
                "ceiling": c["improvement"], "ceiling_used": c["used"], "ceiling_lineage": c.get("lineage"),
                "ceiling_metrics": c["metrics"],
                "ceiling_replacing": lineage_is_replacing(c.get("lineage") or [], replaces) if c.get("lineage") is not None else None,
                "best_fixed": best, "winning_orders": winners, "fixed_all": fixed,
                "rule_v1": v1["improvement"], "rule_v2": v2["improvement"],
                "rule_v1_censored": v1["censored"], "rule_v2_censored": v2["censored"],
                "fixed_censored": sum(entry["policies"][n]["censored"] for n in ORDERS),
                "headroom": c["improvement"] - best,
                "headroom_bands": (c["improvement"] - best) / obj_band if obj_band > 0 else None,
                "ceiling_bands": c["improvement"] / obj_band if obj_band > 0 else None,
                "flags": band_w["flags"],
            })
    return {"tag": tag, "bands": bands, "rows": rows}


def render(t: dict) -> str:
    rows = [r for r in t["rows"] if r["budget"] == 6]
    lines = ["| brief | design | ceiling | best fixed (orders) | rule v1 | rule v2 | headroom | headroom in bands | ceiling in bands |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["brief"], x["design"])):
        s = r["scale"]
        fb = f"{r['best_fixed'] * s:.1f} ({','.join(r['winning_orders']) or '-'})"
        hb = "n/a" if r["headroom_bands"] is None else f"{r['headroom_bands']:.1f}"
        cb = "n/a" if r["ceiling_bands"] is None else f"{r['ceiling_bands']:.1f}"
        lines.append(f"| {r['brief']} | {r['design']} | {r['ceiling'] * s:.1f} {r['unit']} | {fb} | {r['rule_v1'] * s:.1f} | {r['rule_v2'] * s:.1f} | {r['headroom'] * s:.1f} | {hb} | {cb} |")
    return "\n".join(lines)


def main() -> int:
    tag = sys.argv[1]
    noise_path = pm.WORK / "noise_table.json"
    noise_table = json.loads(noise_path.read_text())
    t = build(tag, noise_table)
    folder = pm.WORK / tag
    (folder / "tables.json").write_text(json.dumps(t, indent=1, sort_keys=True))
    md = render(t)
    (folder / "tables.md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
