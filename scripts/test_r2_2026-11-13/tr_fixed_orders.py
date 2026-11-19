"""Declare the fixed worker order of each brief, chosen on TRAINING only (nothing on a TEST design is opened).

T0, A0, T, A: the order of D6-v2 (fixed_BA: area first, then timing, alternating), as the task says ("as in D6-v2").
W2R, L2R, P0: the best of the six frozen worker orders (ABC ACB BAC BCA CAB CBA) on the five TRAINING designs, from the power-readiness
pass A4 (data/power_readiness_2026-11-11/A4/cell_<design>_<brief>.json, policies fixed_<order>): the order with the highest mean over
the five designs of max(improvement, 0) / ceiling improvement (the measure of scripts/power_readiness_2026-11-11/pr_prechosen.py);
ties by the order name. Censored cells keep their recorded lower-bound values. Writes notes/round2_fixed_orders.json once (then 0444).
usage: tr_fixed_orders.py [--write]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SRC = PROJECT / "data/power_readiness_2026-11-11/A4"
OUT = PROJECT / "notes/round2_fixed_orders.json"
TRAINING = ("vexriscv_re-place", "logicnets_jscl", "rosetta_spam-filter", "rosetta_3d-rendering", "finn_radioml")
ORDERS = ("ABC", "ACB", "BAC", "BCA", "CAB", "CBA")


def main() -> int:
    table, files = {}, {}
    for brief in ("W2R", "L2R", "P0"):
        ratio = {o: [] for o in ORDERS}
        for d in TRAINING:
            path = SRC / f"cell_{d}_{brief}.json"
            files[str(path.relative_to(PROJECT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            cell = json.loads(path.read_text(encoding="utf-8"))
            ceiling = float(cell["ceiling"]["improvement"])
            for o in ORDERS:
                imp = float(cell["policies"][f"fixed_{o}"]["improvement"])
                ratio[o].append(max(imp, 0.0) / ceiling if ceiling > 0 else 0.0)
        mean = {o: sum(v) / len(v) for o, v in ratio.items()}
        best = sorted(ORDERS, key=lambda o: (-mean[o], o))[0]
        table[brief] = {"mean_fraction_of_ceiling_TRAINING": {o: round(mean[o], 4) for o in ORDERS}, "chosen": best}
    chosen = {"T0": "BA", "A0": "BA", "T": "BA", "A": "BA", **{b: table[b]["chosen"] for b in table}}
    doc = {
        "schema_version": "round2-fixed-orders-v1",
        "declared_before_launch": True,
        "rule": "T0, A0, T, A: fixed_BA as in D6-v2. W2R, L2R, P0: the best of the six frozen worker orders on the five TRAINING designs "
                "by mean (max(improvement, 0) / ceiling improvement), power-readiness pass A4; ties by order name.",
        "chosen": chosen,
        "evidence_TRAINING": table,
        "source_file_sha256": files,
    }
    print(json.dumps(doc, indent=1, sort_keys=True))
    if "--write" in sys.argv:
        if OUT.exists():
            print("refused: exists (written once)")
            return 2
        OUT.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        OUT.chmod(0o444)
        print("written", OUT, hashlib.sha256(OUT.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
