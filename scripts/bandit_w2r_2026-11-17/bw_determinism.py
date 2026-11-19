"""Determinism of the W2R bandit: the smoke replay (own store, own process, no tool run) and the real run decide the same calls.
For every cell: the mandate sequences (worker, base state, placement scope, attempts, arm) of smoke/ and of the real folder must agree on their common prefix.
Also: the policy draws no random number (no 'random' import in the policy modules) and the seed is carried as metadata only.
usage (project root, venv): python scripts/bandit_w2r_2026-11-17/bw_determinism.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts/bandit_2026-11-11"))
import bd_extract as be  # noqa: E402

RUN = PROJECT / "data/test_r2_2026-11-13/bandit_w2r_2026-11-17"
ORDER = ("rosetta_digit-recognition", "corescore_500_mod", "vexriscv_re-place_v2", "fir_symmetric_systolic", "fir_systolic_transposed")


def mandates(folder: Path, eid: str) -> list[tuple]:
    j = folder / "test/engine/episodes" / eid / "journal.jsonl"
    if not j.is_file():
        return []
    out = []
    for e in be.read_events(j):
        if e.get("type") == "mandate":
            m = e["mandate"]
            out.append((m["worker_id"], m["base_state_id"], bool(m["settings"]["preserve_placement"]), m["settings"].get("max_runs"), json.loads(m["context_ref"])["arm"]))
    return out


def main() -> int:
    ok = True
    for d in ORDER:
        eid = f"tr_{d}_W2R_bandit_pf1"
        a, b = mandates(RUN / "smoke", eid), mandates(RUN, eid)
        n = min(len(a), len(b))
        same = a[:n] == b[:n]
        ok &= same and n > 0
        print(f"{d}: smoke calls {len(a)}, real calls {len(b)}, common prefix {n}, identical: {same}")
    src = [p for p in (PROJECT / "orch/policies").glob("bandit*.py")]
    uses_random = [p.name for p in src if re.search(r"^\s*(import random|from random|import numpy)", p.read_text(encoding="utf-8"), re.M)]
    print("policy modules that import a random-number source:", uses_random or "none")
    print("DETERMINISM:", "confirmed on the common prefixes" if ok and not uses_random else "NOT confirmed")
    return 0 if ok and not uses_random else 1


if __name__ == "__main__":
    raise SystemExit(main())
