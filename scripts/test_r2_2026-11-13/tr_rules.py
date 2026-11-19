"""Reporting rules of the round-2 briefs that are not part of the frozen loader (new file, 2026-11-13).

p0_gain_counts_mw: the P0 reporting rule "a power gain inside the design's band counts as no change", compared in whole milliwatts so that float
noise cannot decide the exact-edge case. The frozen loader orch/contract/briefs_round2.py has the same rule as p0_gain_counts but compares
raw floats: a gain of exactly the band can come out as "counts" (1.0 - (1.0 - 0.054) = 0.05400000000000005). That file is in
notes/round2_briefs_freeze.sha256 and is not edited; reports use this function instead (provisional until the authors confirm the rounding).
"""
from __future__ import annotations

from orch.contract.briefs_round2 import power_band_w


def p0_gain_counts_mw(design: str, start_power_w: float, final_power_w: float) -> bool:
    return round((float(start_power_w) - float(final_power_w)) * 1000.0) > round(power_band_w(design) * 1000.0)
