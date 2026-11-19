"""Tests of the round-2 brief freeze (2026-11-13): loader, fingerprints, power-band offsets, TEST noise file, freeze list."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from orch.config import load_noise
from orch.contract.brief_text import FOUR_BRIEF_FINGERPRINTS, load_frozen_brief
from orch.contract.briefs_round2 import (
    BRIEF_IDS, FREEZE, NOISE_TEST, TEST_DESIGNS, BriefFreezeError, brief_for, briefs_for, check_freeze, p0_gain_counts, power_band_w,
)
from orch.contract.models import MetricName as M
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts/test_r2_2026-11-13"))
from tr_rules import p0_gain_counts_mw  # noqa: E402


def test_seven_briefs_in_order():
    assert BRIEF_IDS == ("T0", "A0", "T", "A", "W2R", "L2R", "P0")
    assert tuple(briefs_for("corescore_500_mod")) == BRIEF_IDS


@pytest.mark.parametrize("design", TEST_DESIGNS)
def test_frozen_briefs_are_unchanged(design):
    for b in ("T0", "A0", "T", "A"):
        assert brief_for(design, b).fingerprint() == FOUR_BRIEF_FINGERPRINTS[b]
    assert brief_for(design, "P0").fingerprint() == load_frozen_brief("P0").fingerprint()


@pytest.mark.parametrize("design", TEST_DESIGNS)
def test_power_limit_is_start_plus_band_inclusive(design):
    band = power_band_w(design)
    assert band > 0
    for b, strict_metric, strict_cmp in (("W2R", M.PACKED_LUTS, "<="), ("L2R", M.WNS_NS, ">=")):
        terms = {t.metric: t for t in brief_for(design, b).terms}
        assert set(terms) == {strict_metric, M.DYNAMIC_POWER_W}
        assert (terms[strict_metric].comparison, terms[strict_metric].factor, terms[strict_metric].offset) == (strict_cmp, 1.0, 0.0)
        p = terms[M.DYNAMIC_POWER_W]
        assert (p.comparison, p.factor, p.offset) == ("<=", 1.0, band)


def test_objectives_and_directions():
    br = briefs_for("fir_symmetric_systolic")
    assert (br["W2R"].objective.metric, br["W2R"].objective.direction) == (M.WNS_NS, "max")
    assert (br["L2R"].objective.metric, br["L2R"].objective.direction) == (M.PACKED_LUTS, "min")
    assert (br["P0"].objective.metric, br["P0"].objective.direction) == (M.DYNAMIC_POWER_W, "min")
    assert all(b.budget_runs == 6 for b in br.values())


def test_every_test_design_has_three_bands():
    cfg = load_noise(NOISE_TEST)
    for d in TEST_DESIGNS:
        assert {k.value for k in cfg.bands[d]} == {"wns_ns", "packed_luts", "dynamic_power_w"}


def test_supplement_copies_noise_round2_where_it_has_entries():
    base = load_noise(PROJECT / "configs/noise_round2.yaml")
    sup = load_noise(NOISE_TEST)
    for d in ("rosetta_digit-recognition", "corescore_500_mod", "vexriscv_re-place_v2"):
        for m in (M.WNS_NS, M.PACKED_LUTS):
            assert sup.bands[d][m] == base.bands[d][m]


def test_freeze_list_matches_files():
    entries = check_freeze()
    assert "configs/briefs_round2_final.yaml" in entries and "configs/noise_round2_test.yaml" in entries
    for rel, digest in entries.items():
        assert hashlib.sha256((PROJECT / rel).read_bytes()).hexdigest() == digest


def test_p0_gain_inside_band_counts_as_no_change():
    d = "corescore_500_mod"
    band = power_band_w(d)
    # frozen helper: clear cases only (its exact-edge case compares raw floats, see tr_rules)
    assert not p0_gain_counts(d, 1.0, 1.0 - band / 2)
    assert p0_gain_counts(d, 1.0, 1.0 - 2 * band)
    # reporting helper in whole mW: exactly the band is no change, one mW more counts
    assert not p0_gain_counts_mw(d, 1.0, 1.0 - band)
    assert not p0_gain_counts_mw(d, 1.0, 1.0)
    assert p0_gain_counts_mw(d, 1.0, 1.0 - band - 0.001)


def test_unknown_brief_is_refused():
    with pytest.raises(ValueError):
        brief_for("corescore_500_mod", "L2")
    assert FREEZE.is_file() and issubclass(BriefFreezeError, RuntimeError)
