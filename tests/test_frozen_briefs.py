"""Frozen brief file, renderer and start-relative gate boundary tests.

T0, A0, T and A were frozen on 2026-11-05; P0 is a dated 2026-11-06 addition
that must leave those four byte-for-byte and contract-for-contract unchanged.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from orch.contract.brief_text import (
    BRIEFS_PATH, FOUR_BRIEF_FINGERPRINTS, FROZEN_BRIEFS_SHA256,
    FROZEN_BRIEFS_SHA256_2026_11_05, FROZEN_TEMPLATE_SHA256, TEMPLATE,
    TEMPLATE_SHA256, load_frozen_brief, load_frozen_spec, render_brief_text,
)
from orch.contract.models import DesignState, Evidence, MetricName, MetricValue
from orch.engine.gate import judge
from orch.episode import _Visit, _rank


def _evidence(label: str, *, wns: float, luts: float, power: float | None = None) -> Evidence:
    metrics = {
        MetricName.WNS_NS: MetricValue(value=wns, unit="ns", method="test", basis="routed"),
        MetricName.PACKED_LUTS: MetricValue(
            value=luts, unit="count", method="test", basis="routed",
        ),
    }
    if power is not None:
        metrics[MetricName.DYNAMIC_POWER_W] = MetricValue(
            value=power, unit="W", method="test", basis="routed",
        )
    return Evidence(
        id=label, state_id=hashlib.sha256(label.encode()).hexdigest(),
        metrics=metrics,
        routed=True, route_errors=0, drc_errors=0,
        constraint_fingerprint="xdc", clock_digest="clock",
        meter_version="frozen-brief-test", fake=True,
    )


def test_frozen_file_loading_and_hashes() -> None:
    spec = load_frozen_spec()
    assert hashlib.sha256(BRIEFS_PATH.read_bytes()).hexdigest() == FROZEN_BRIEFS_SHA256
    assert tuple(spec["briefs"]) == ("T0", "A0", "T", "A", "P0")
    assert spec["common"]["budget"] == {"recipe_runs": 6}
    assert TEMPLATE_SHA256 == FROZEN_TEMPLATE_SHA256
    assert hashlib.sha256(TEMPLATE.encode("utf-8")).hexdigest() == TEMPLATE_SHA256
    briefs = [load_frozen_brief(name) for name in spec["briefs"]]
    assert len({brief.fingerprint() for brief in briefs}) == 5
    assert all(brief.budget_runs == 6 for brief in briefs)
    with pytest.raises(ValueError, match="not in the frozen set"):
        load_frozen_brief("B")


def test_p0_is_a_dated_addition_and_the_four_earlier_briefs_are_unchanged() -> None:
    assert FROZEN_BRIEFS_SHA256 != FROZEN_BRIEFS_SHA256_2026_11_05
    assert FROZEN_BRIEFS_SHA256_2026_11_05 == (
        "9087784cf612d00b73b5535b98469df16dba7425df530b892e47bccdf2de4894"
    )
    assert set(FOUR_BRIEF_FINGERPRINTS) == {"T0", "A0", "T", "A"}
    for name, fingerprint in FOUR_BRIEF_FINGERPRINTS.items():
        assert load_frozen_brief(name).fingerprint() == fingerprint
    p0 = load_frozen_brief("P0")
    assert p0.objective.metric == MetricName.DYNAMIC_POWER_W
    assert p0.objective.direction == "min"
    assert [(t.metric, t.comparison, t.factor, t.offset) for t in p0.terms] == [
        (MetricName.WNS_NS, ">=", 1.0, 0.0),
        (MetricName.PACKED_LUTS, "<=", 1.0, 0.0),
    ]
    assert p0.budget_runs == 6 and p0.equivalence_policy == "final_only"
    assert load_frozen_spec()["briefs"]["P0"]["role"] == "secondary"


def test_edited_frozen_file_is_refused(tmp_path: Path) -> None:
    altered = tmp_path / "briefs.yaml"
    altered.write_bytes(BRIEFS_PATH.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash differs"):
        load_frozen_spec(altered)


@pytest.mark.parametrize(
    ("name", "pass_wns", "pass_luts", "fail_wns", "fail_luts", "threshold"),
    [
        ("T0", -0.4, 100.0, -0.4, 100.001, 100.0),
        ("A0", -0.5, 90.0, -0.500001, 90.0, -0.5),
        ("T", -0.4, 101.0, -0.4, 101.001, 101.0),
        ("A", -0.6, 90.0, -0.600001, 90.0, -0.6),
    ],
)
def test_gate_judges_each_frozen_limit_against_b2(
    name: str, pass_wns: float, pass_luts: float,
    fail_wns: float, fail_luts: float, threshold: float,
) -> None:
    start = _evidence("start", wns=-0.5, luts=100.0)
    brief = load_frozen_brief(name)
    assert brief.terms[0].threshold(start) == pytest.approx(threshold)
    passed = judge(
        brief, start, start,
        _evidence("pass-" + name, wns=pass_wns, luts=pass_luts),
    )
    failed = judge(
        brief, start, start,
        _evidence("fail-" + name, wns=fail_wns, luts=fail_luts),
    )
    assert passed.brief_valid
    assert not failed.brief_valid
    assert failed.terms[brief.terms[0].id].status == "fail"


def test_gate_judges_both_p0_limits_against_b2() -> None:
    start = _evidence("p0-start", wns=-0.5, luts=100.0, power=1.0)
    brief = load_frozen_brief("P0")
    ok = judge(brief, start, start,
               _evidence("p0-ok", wns=-0.5, luts=100.0, power=0.9))
    wns_bad = judge(brief, start, start,
                    _evidence("p0-wns", wns=-0.500001, luts=100.0, power=0.9))
    luts_bad = judge(brief, start, start,
                     _evidence("p0-luts", wns=-0.4, luts=100.001, power=0.9))
    assert ok.brief_valid and not wns_bad.brief_valid and not luts_bad.brief_valid
    assert wns_bad.terms["p0_limit_wns_ns"].status == "fail"
    assert wns_bad.terms["p0_limit_packed_luts"].status != "fail"
    assert luts_bad.terms["p0_limit_packed_luts"].status == "fail"


def test_renderer_gives_exact_shared_text_and_rejects_changed_contract() -> None:
    expected = """Brief T0 (main).
Start from the B2 design checkpoint. Budget: at most 6 recipe runs.
Objective: Maximise worst negative slack (WNS, ns).
Hard limit: packed CLB LUT count must not exceed its B2 value; compare against the B2 start measurement.
Final pick: best brief-valid state visited; ties broken by the better non-objective metric, then by fewer runs; if none, the start (no change).
Validity: fully routed; zero route and DRC errors; matching B2 constraint fingerprint and clock digest; no leftover DONT_TOUCH or fixed LOC/BEL; AMD simulation PASS.
Reporting: timing gain as Fmax percent and area gain as packed CLB LUT-cut percent; also TNS, netlist LUT cells, and FFs.
Noise: flag only; never use it to accept or reject.
"""
    brief = load_frozen_brief("T0")
    assert render_brief_text("T0") == expected
    assert render_brief_text(brief) == expected
    with pytest.raises(ValueError, match="differs from the frozen"):
        render_brief_text(brief.model_copy(update={"budget_runs": 7}))


def test_p0_renderer_lists_both_hard_limits() -> None:
    expected = """Brief P0 (secondary).
Start from the B2 design checkpoint. Budget: at most 6 recipe runs.
Objective: Minimise dynamic power (W, Vivado vectorless estimate).
Hard limits: worst negative slack (WNS, ns) must not fall below its B2 value; packed CLB LUT count must not exceed its B2 value; compare against the B2 start measurement.
Final pick: best brief-valid state visited; ties broken by the better non-objective metric, then by fewer runs; if none, the start (no change).
Validity: fully routed; zero route and DRC errors; matching B2 constraint fingerprint and clock digest; no leftover DONT_TOUCH or fixed LOC/BEL; AMD simulation PASS.
Reporting: timing gain as Fmax percent, area gain as packed CLB LUT-cut percent and power gain as dynamic-power percent; also TNS, netlist LUT cells, and FFs.
Noise: flag only; never use it to accept or reject.
"""
    assert render_brief_text("P0") == expected
    assert render_brief_text(load_frozen_brief("P0")) == expected


def test_final_pick_breaks_ties_by_secondary_then_fewer_lineage_runs() -> None:
    brief = load_frozen_brief("T0")
    start = _evidence("tie-start", wns=-0.5, luts=100.0)

    def visit(label: str, luts: float, depth: int) -> _Visit:
        evidence = _evidence(label, wns=-0.4, luts=luts)
        state = DesignState(
            id=evidence.state_id, design="synthetic", path="fake://" + label,
            fake=True,
        )
        verdict = judge(brief, start, start, evidence)
        assert verdict.brief_valid
        return _Visit(state, evidence, verdict, "run_" + label, "synthetic", depth)

    long = visit("long", 100.0, 3)
    short = visit("short", 100.0, 1)
    better_secondary = visit("better", 99.0, 3)
    assert _rank(brief, [long, short, better_secondary]) == [
        better_secondary, short, long,
    ]
