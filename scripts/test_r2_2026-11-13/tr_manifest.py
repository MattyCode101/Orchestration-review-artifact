"""Test-run preparation: the read-only union manifest over every stored root of the five TEST designs.

Built with the campaign's generic builder (scripts/campaign_manifest.py build_generic, imported, not edited). Nothing is written in a
source root. Roots, lowest priority first (the builder keeps the LAST root of a duplicated key, so the roots that carry AMD records
against B2 come last): the ten round-1 test power roots (phase a and b), test_base (the round-1 timing and area baseline), the
overnight exhaustive depth-3 engine root (phases 0 to 2, all 14 recipes), the round-1 campaign root. repeats_c0 (repeat controls of the
B2 start, same keys with other bytes) is left out.
Output (sealed folder, catalogue only: run identities, costs, file hashes): data/test_r2_2026-11-13/sealed/manifest_test/
usage: tr_manifest.py [--write]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
import campaign_manifest as cm  # noqa: E402
from orch.engine.store import _atomic_bytes, _json_bytes  # noqa: E402

OUT = PROJECT / "data/test_r2_2026-11-13/sealed/manifest_test"
OUT_LLM1 = PROJECT / "data/test_r2_2026-11-13/sealed/llm1/manifest_test"      # queue llm1 (13 Nov evening): the same roots plus the baseline queue engine
OUT_QUEUE = {"llm1": OUT_LLM1,
             "llm3": PROJECT / "data/test_r2_2026-11-13/sealed/llm3/manifest_test",     # 14 Nov: the roots of llm1 plus the llm1 queue engine
             "llm4": PROJECT / "data/test_r2_2026-11-13/sealed/llm4/manifest_test",
             "llm5": PROJECT / "data/test_r2_2026-11-13/sealed/llm5/manifest_test",
             "llm6": PROJECT / "data/test_r2_2026-11-13/sealed/llm6/manifest_test",
             "llm7": PROJECT / "data/test_r2_2026-11-13/sealed/llm7/manifest_test"}
BASELINE_ENGINE = PROJECT / "data/test_r2_2026-11-13/sealed/test/engine"
LLM1_ENGINE = PROJECT / "data/test_r2_2026-11-13/sealed/llm1/test/engine"
METER = "0dd0cf9424bbb7ab20dc5cf981bc35068c4ba08612ecd0f6c46098e6bf880bbe"
TEST = ("rosetta_digit-recognition", "corescore_500_mod", "vexriscv_re-place_v2", "fir_symmetric_systolic", "fir_systolic_transposed")
B2 = {
    "rosetta_digit-recognition": "57a8689b62581982847e4140f76f53f731925c664bc7a8fc9a585a3789bd19c4",
    "corescore_500_mod": "85b7470eff50093d10c3b06e3e458f10488effe9ef1f9e563948edabea1b258f",
    "vexriscv_re-place_v2": "8e5c5a6968f01a06aef124e2d17fffe920f559f76227ff77ba5c99b4525b8272",
    "fir_symmetric_systolic": "a8cbdb23cdcc3c71e26e03e2c74a1aa43c42d9ee79eef37ae8fbe374e976e9ec",
    "fir_systolic_transposed": "20329fc34fdfb2da036cf221da9c27af33a7c735bb5421fc0e93ce6744f49020",
}


def roots(queue: str = "") -> dict[str, Path]:
    out: dict[str, Path] = {}
    for phase in ("a", "b"):
        for d in TEST:
            out[f"power_{phase}:{d}"] = PROJECT / f"data/sweeps/friday_test_power_2026-11-06/phase_{phase}/{d}"
    out["test_base"] = PROJECT / "data/exp_evening_2026-11-04/test_b2/overnight_baseline_2026-11-04"
    out["exhaustive_d3"] = PROJECT / "data/exhaustive_d3_2026-11-12/engine"
    out["campaign"] = PROJECT / "data/campaign_test_final/engine"
    if queue in ("llm1", "llm3", "llm4", "llm5", "llm6", "llm7"):
        out["baseline_queue"] = BASELINE_ENGINE          # it carries the AMD records against B2 of Friday evening's baseline queue
    if queue in ("llm3", "llm4", "llm5", "llm6", "llm7"):
        out["llm1_queue"] = LLM1_ENGINE                  # last: queue 1's own new runs and their AMD records (read-only, sealed folder)
    if queue in ("llm5", "llm6", "llm7"):
        for q in ("llm3", "llm4"):
            out[f"{q}_queue"] = PROJECT / f"data/test_r2_2026-11-13/sealed/{q}/test/engine"
    if queue in ("llm5", "llm7"):
        out["llm6_queue"] = PROJECT / "data/test_r2_2026-11-13/sealed/llm6/test/engine"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--queue", choices=("", "llm1", "llm3", "llm4", "llm5", "llm6", "llm7"), default="")
    args = ap.parse_args()
    rs = roots(args.queue)
    out_dir = OUT_QUEUE.get(args.queue, OUT)
    missing = [k for k, v in rs.items() if not (v / "cache").is_dir()]
    if missing:
        print(json.dumps({"roots_without_cache_folder": missing}))
    rs = {k: v for k, v in rs.items() if (v / "cache").is_dir()}
    ev_roots = {"test_base": rs["test_base"], "campaign": rs["campaign"], "exhaustive_d3": rs["exhaustive_d3"]}
    body, report = cm.build_generic(rs, METER, B2, ev_roots)
    summary = {k: v for k, v in report.items() if k != "problems"}
    summary["roots"] = len(rs)
    summary["problems_count"] = len(report["problems"])
    summary["problems_first"] = report["problems"][:5]
    summary["entries_by_root"] = {}
    for e in body["entries"].values():
        summary["entries_by_root"][e["root"].split(":")[0]] = summary["entries_by_root"].get(e["root"].split(":")[0], 0) + 1
    print(json.dumps(summary, indent=1, sort_keys=True, default=str))
    if args.write:
        if report["problems"]:
            print("refused: the manifest has problems")
            return 1
        if (out_dir / "manifest.json").exists():
            print("refused: manifest exists (written once)")
            return 2
        out_dir.mkdir(parents=True, exist_ok=True)
        _atomic_bytes(out_dir / "manifest.json", _json_bytes(body))
        _atomic_bytes(out_dir / "manifest.sha256", (report["manifest_sha256"] + "\n").encode())
        _atomic_bytes(out_dir / "report.json", _json_bytes(report))
        print("written", out_dir)
    return 0 if not report["problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
