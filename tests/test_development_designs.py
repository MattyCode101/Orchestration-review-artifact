"""Tests for the development-designs helpers (D3, Thu 12 Nov 2026). No tool, no run, no model call."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts" / "development_designs_2026-11-12"))
sys.path.insert(0, str(PROJECT / "scripts" / "power_mediation_2026-11-10"))

import dd_common as dd  # noqa: E402


def test_headroom_class_boundaries():
    assert dd.head_class(0.0) == "none"
    assert dd.head_class(0.999) == "none"
    assert dd.head_class(1.0) == "low"
    assert dd.head_class(3.0) == "low"
    assert dd.head_class(3.001) == "high"


def test_recipe_pools():
    assert len(dd.RECIPES2) == 10 and len(dd.RECIPES3) == 14 and len(set(dd.RECIPES3)) == 14
    assert set(dd.POWER) == {"P_ROUTE_HIGHDELAY", "P_POST_ALL", "P_POST_BRAM", "P_PRE_ALL"}


@pytest.mark.parametrize("brief_id", ["L2R", "W2R"])
def test_banded_briefs_equal_the_power_studys(brief_id):
    pm = pytest.importorskip("pm_common")
    for band in (0.0, 0.008, 0.263):
        ours = dd.make_brief(brief_id, band)
        theirs = pm.candidate_brief(brief_id, 6, band)
        assert ours.fingerprint() == theirs.fingerprint()


def test_frozen_briefs_are_loaded_unchanged():
    for b in ("T0", "A0", "P0"):
        assert dd.make_brief(b).fingerprint() == dd.load_frozen_brief(b).model_copy(update={"budget_runs": 6}).fingerprint()


def test_forbidden_designs_are_refused():
    for d in ("t2_or1200", "rosetta_digit-recognition", "corescore_500_mod", "vexriscv_re-place_v2",
              "fir_symmetric_systolic", "fir_systolic_transposed"):
        with pytest.raises(SystemExit):
            dd.guard_design(d)
    assert dd.guard_design("t2_cordic_s") == "t2_cordic_s"


class _FakeGraph:
    def __init__(self, by_in):
        self.by_in = by_in


def _entry(out, status="ok"):
    return {"status": status, "output_state_id": out}


def test_reachable_shares_steps_of_identical_outputs_and_lists_missing_pairs():
    # start S; recipes a and b leave the state unchanged (output S); c gives X; d fails; e gives X again (a twin output)
    by_in = {("S", "a"): _entry("S"), ("S", "b"): _entry("S"), ("S", "c"): _entry("X"), ("S", "d"): _entry(None, "failed"),
             ("S", "e"): _entry("X"), ("X", "a"): _entry("Y"), ("X", "b"): _entry("X")}
    states, missing, failed = dd.reachable(_FakeGraph(by_in), "S", ("a", "b", "c", "d", "e"), ("a", "b", "c"))
    assert set(states) == {"S", "X", "Y"}
    assert states["Y"]["path"] == ["c", "a"] and states["Y"]["steps"] == 2
    assert ("S", "d") in failed
    assert ("X", "c") in missing and ("X", "a") not in missing
    assert all(p != "S" for p, _ in missing)          # the start is not expanded a second time


def test_gain_signs():
    class _M:
        def __init__(self, v):
            self.value = v

    class _Ev:
        def __init__(self, **m):
            self.m = m

        def metric(self, name):
            return _M(self.m[name.value])

    s, e = _Ev(wns_ns=-0.5, packed_luts=1000.0, dynamic_power_w=1.0), _Ev(wns_ns=-0.2, packed_luts=950.0, dynamic_power_w=0.9)
    assert dd.gain_of("T0", s, e) == pytest.approx(0.3)
    assert dd.gain_of("W2R", s, e) == pytest.approx(0.3)
    assert dd.gain_of("A0", s, e) == pytest.approx(50.0)
    assert dd.gain_of("L2R", s, e) == pytest.approx(50.0)
    assert dd.gain_of("P0", s, e) == pytest.approx(0.1)


def test_development_configs_agree_with_the_screen():
    import yaml

    split = yaml.safe_load((PROJECT / "configs/development/split.yaml").read_text())
    noise = yaml.safe_load((PROJECT / "configs/noise_development.yaml").read_text())["bands"]
    assert set(split["designs"]) <= set(noise) and "logicnets_jscl" in noise
    for d, info in split["designs"].items():
        assert d not in dd.FORBIDDEN
        assert info["b2_state_id"] == dd.start_id(d)
        b = dd.screen_bands(d)
        assert noise[d]["wns_ns"]["value"] == pytest.approx(b["wns_ns"])
        assert noise[d]["packed_luts"]["value"] == pytest.approx(b["packed_luts"])
        assert noise[d]["dynamic_power_w"]["value"] == pytest.approx(b["dynamic_power_w"])
        assert info["power_band_offset_w"] == pytest.approx(dd.power_offset(d))


def test_the_pinned_noise_config_is_untouched():
    freeze = json.loads((PROJECT / "data/test_campaign_prep_2026-11-06/freeze_final.json").read_text())["files"]
    import hashlib

    for rel in ("configs/noise.yaml", "configs/campaign/campaign.yaml", "configs/briefs.yaml"):
        assert hashlib.sha256((PROJECT / rel).read_bytes()).hexdigest() == freeze[rel]


def test_manifest_holds_only_registered_designs():
    manifest = PROJECT / "data/development_designs_2026-11-12/manifest/manifest.json"
    if not manifest.is_file():
        pytest.skip("manifest not built")
    body = json.loads(manifest.read_text())
    import yaml

    designs = set(yaml.safe_load((PROJECT / "configs/development/split.yaml").read_text())["designs"])
    assert {item["design"] for item in body["start_evidence"].values()} == designs
    assert all(root.startswith("bx_") and root[3:] in designs for root in body["roots"])
