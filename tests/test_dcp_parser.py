"""Saved-report checks for LATCH DCP telemetry; no hardware job in the normal suite."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from orch.contract.models import MetricName
from orch.engine import dcp_parser as parser_module
from orch.engine.dcp_parser import (
    DcpParserError,
    DcpTelemetryExtractor,
    PROBE_TEMPLATE,
    PROBE_TEMPLATE_SHA256,
    WPATH_SHA256,
    WPATH2_SHA256,
    _safe_tcl_path,
    _require_consistent_timing,
    parse_wpath_report,
    parse_wpath2_report,
    parse_pblock_utilization_text,
    parse_probe_report,
    parse_timing_summary_static,
)
from orch.engine.meter_parsers import parse_reports

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LATCH_REPORT = PROJECT_ROOT / "tests/fixtures/dcp_parser/latch_vexriscv_probe.rpt"
B2_REPORT = PROJECT_ROOT / "tests/fixtures/dcp_parser/b2_vexriscv_probe.rpt"
FINN_REPORT = PROJECT_ROOT / "tests/fixtures/dcp_parser/finn_fractional_bram_probe.rpt"
FINN_WPATH2 = PROJECT_ROOT / "tests/fixtures/dcp_parser/finn_fractional_bram_wpath2.tsv"
LOGICNETS_REPORT = PROJECT_ROOT / "tests/fixtures/dcp_parser/logicnets_no_primary_srl_probe.rpt"
B2_METER = PROJECT_ROOT / "tests/fixtures/meter"


def test_saved_latch_probe_extracts_timing_path_area_and_fanout() -> None:
    """Use LATCH's observed Vivado output, not a fabricated report layout."""

    text = LATCH_REPORT.read_text(encoding="utf-8")
    result = parse_probe_report(text)
    assert result["timing"] == {
        "wns_ns": -1.654,
        "tns_ns": -671.139,
        "failing_endpoints": 1937,
    }
    assert parse_timing_summary_static(text) == {
        "wns": -1.654,
        "tns": -671.139,
        "failing_endpoints": 1937,
    }
    path = result["critical_path_features"]
    assert path["data_path_delay_ns"] == 3.148
    assert path["logic_delay_ns"] == 0.507
    assert path["route_delay_ns"] == 2.641
    assert path["route_delay_ratio"] == pytest.approx(0.838945)
    assert path["max_logic_levels"] == 5
    assert path["dominant_path_type"] == "routing_dominated"
    assert result["fanout"] == {"num_high_fanout_nets": 20, "max_fanout": 266}
    assert result["diagnostic_missing_reasons"] == {}


def test_utilization_skips_observed_prohibited_column() -> None:
    """Regression: LATCH used to mistake Prohibited for Available."""

    rows = parse_probe_report(LATCH_REPORT.read_text())["utilization_summary"]
    assert rows["CLB LUTs"] == {
        "used": 1906,
        "fixed": 0,
        "available": 394080,
        "utilization_pct": 0.48,
    }
    assert rows["CLB Registers"]["available"] == 788160
    assert rows["Block RAM Tile"]["used"] == 6
    assert rows["DSPs"]["used"] == 4
    assert rows["URAM"]["used"] == 0
    resources = parse_pblock_utilization_text(LATCH_REPORT.read_text())
    assert resources["luts"] == 1866  # LUT as Logic, not packed CLB LUTs.
    assert resources["ffs"] == 1199
    assert resources["dsps"] == 4
    assert resources["brams"] == 6
    assert resources["urams"] == 0
    assert resources["luts_1_5x"] == 2799


def test_saved_finn_probe_preserves_fractional_block_ram_tiles() -> None:
    """The copied LATCH probes both report 24.5 for this saved Finn DCP."""

    assert hashlib.sha256(FINN_REPORT.read_bytes()).hexdigest() == (
        "e60e9d381144b5acddaf15f6dea52840ead25e04f4421ef1e2f3896c34445316"
    )
    assert hashlib.sha256(FINN_WPATH2.read_bytes()).hexdigest() == (
        "f4b82587a9b3b7e4b1ebd82a1a36934790ad7b217e9867a115dc81dbf54eb92d"
    )
    parsed = parse_probe_report(FINN_REPORT.read_text(encoding="utf-8"))
    wpath2 = parse_wpath2_report(FINN_WPATH2.read_text(encoding="utf-8"))
    assert parsed["utilization_summary"]["Block RAM Tile"]["used"] == 24.5
    assert wpath2["utilization_counts"]["brams"] == 24.5


def test_saved_logicnets_probe_reads_srl_from_distribution_when_primary_omits_it() -> None:
    """A real LogicNets report names the zero SRL count only in its distribution table."""

    assert hashlib.sha256(LOGICNETS_REPORT.read_bytes()).hexdigest() == (
        "1ba1e4a24aa15083beadd6a6412e51986c14d8991e6c62cca88a416430e3ec20"
    )
    text = LOGICNETS_REPORT.read_text(encoding="utf-8")
    primary = parser_module._utilization_section(
        text, "1. CLB Logic", "1.1 Summary of Registers by Type"
    )
    assert primary is not None
    assert parser_module._utilization_used(primary, "LUT as Shift Register") is None
    parsed = parse_probe_report(text)
    assert parsed["lut_as_shift_register"] == 0
    assert "lut_as_shift_register" not in parsed["diagnostic_missing_reasons"]


def test_missing_fanout_is_explicit_and_empty_report_fails() -> None:
    text = LATCH_REPORT.read_text()
    without_fanout = text.split("High Fan-out Nets Information", 1)[0]
    result = parse_probe_report(without_fanout)
    assert result["fanout"] == {}
    assert result["diagnostic_missing_reasons"]["fanout"]
    with pytest.raises(DcpParserError, match="empty"):
        parse_probe_report("")


def test_copied_latch_tcl_remains_byte_identical() -> None:
    assert hashlib.sha256(PROBE_TEMPLATE.read_bytes()).hexdigest() == PROBE_TEMPLATE_SHA256
    assert PROBE_TEMPLATE.read_text().count("report_timing -delay_type max") == 1


def test_unquoted_original_tcl_rejects_unsafe_interpolated_paths() -> None:
    with pytest.raises(DcpParserError, match="interpolated"):
        _safe_tcl_path(Path("/tmp/path with spaces/input.dcp"))
    with pytest.raises(DcpParserError, match="interpolated"):
        _safe_tcl_path(Path("/tmp/{unsafe}/input.dcp"))


def test_power_and_validity_are_from_independent_meter_fixture() -> None:
    probe = parse_probe_report(LATCH_REPORT.read_text())
    assert "dynamic_power_w" not in probe
    evidence = parse_reports(
        B2_METER,
        tag="base",
        state_id="a" * 64,
        meter_version="fixture-v1",
        diag_stdout=(B2_METER / "diag_stdout.txt").read_text(),
    )
    assert evidence.metric(MetricName.DYNAMIC_POWER_W).value == 0.263
    assert evidence.metric(MetricName.NETLIST_LUTS).value == 1953
    assert evidence.metric(MetricName.PACKED_LUTS).value == 1895
    assert evidence.routed and evidence.route_errors == 0 and evidence.drc_errors == 0


def test_real_vexriscv_b2_probe_covers_all_copied_parser_groups() -> None:
    """A successful tagged B2 job produced this exact saved report."""

    assert hashlib.sha256(B2_REPORT.read_bytes()).hexdigest() == (
        "0650a35d98e46ade891471ee53df81b251c2b603bd734203b69b687848b9e8ff"
    )
    text = B2_REPORT.read_text(encoding="utf-8")
    parsed = parse_probe_report(text)
    assert parsed["timing"] == {
        "wns_ns": -0.785,
        "tns_ns": -498.982,
        "failing_endpoints": 1875,
    }
    assert parse_timing_summary_static(text)["wns"] == -0.785
    path = parsed["critical_path_features"]
    assert path["data_path_delay_ns"] == 2.290
    assert path["logic_delay_ns"] == 1.158
    assert path["route_delay_ns"] == 1.132
    assert path["route_delay_ratio"] == pytest.approx(0.494323)
    assert path["max_logic_levels"] == 2
    assert path["dominant_path_type"] == "mixed"
    assert parsed["utilization_summary"]["CLB LUTs"]["used"] == 1895
    assert parsed["utilization_summary"]["CLB LUTs"]["available"] == 394080
    assert parsed["utilization_summary"]["CLB Registers"]["used"] == 1231
    assert parsed["utilization_summary"]["Block RAM Tile"]["used"] == 6
    assert parsed["utilization_summary"]["DSPs"]["used"] == 4
    assert parsed["utilization_summary"]["URAM"]["used"] == 0
    assert parsed["fanout"] == {"num_high_fanout_nets": 20, "max_fanout": 266}
    assert parsed["pblock_sizing_resources"]["luts"] == 1855
    assert parsed["pblock_sizing_resources"]["ffs"] == 1231
    assert parsed["diagnostic_missing_reasons"] == {}


def test_real_b2_worst_path_probes_cover_structure_and_skew() -> None:
    """These tab records came from separate successful, tagged LATCH Tcl runs."""

    first = PROJECT_ROOT / "tests/fixtures/dcp_parser/b2_wpath.tsv"
    second = PROJECT_ROOT / "tests/fixtures/dcp_parser/b2_wpath2.tsv"
    assert hashlib.sha256(first.read_bytes()).hexdigest() == (
        "e0f252299c25571d1d005aba1c24f58cf98cac30aa9946e601c4ec6b93e3e31a"
    )
    assert hashlib.sha256(second.read_bytes()).hexdigest() == (
        "9370a03d40ff8008c13d983d1ed129c42ccde378d5429b52d1c106a862db3292"
    )
    a = parse_wpath_report(first.read_text())
    b = parse_wpath2_report(second.read_text())
    for key, expected in {
        "slack_ns": -0.785,
        "requirement_ns": 1.570,
        "data_path_delay_ns": 2.290,
        "logic_delay_ns": 1.158,
        "net_delay_ns": 1.132,
        "clock_skew_ns": -0.055,
        "logic_levels": 2,
    }.items():
        assert a[key] == b[key] == expected
    assert a["startpoint_pin"] == b["startpoint_pin"]
    assert a["endpoint_pin"] == b["endpoint_pin"]
    assert a["startpoint_clock"] == "clk_fpl26contest"
    assert a["path_cells"] == b["path_cells"] == {
        "count": 4,
        "composition": {"RAMB36E2": 1, "LUT3": 1, "RAMD32": 1, "FDRE": 1},
    }
    assert a["tile_span"] == {
        "x_min": 16, "x_max": 19, "y_min": 74, "y_max": 80, "dx": 3, "dy": 6
    }
    assert b["tile_span"]["x_min"] == 16 and b["tile_span"]["y_max"] == 80
    assert a["pblocks"] == b["pblocks"] == []
    assert a["primitive_cell_count"] == 3405
    assert b["utilization_counts"] == {
        "luts": 1895, "ffs": 1231, "brams": 6, "dsps": 4, "urams": 0
    }
    assert a["unparsed_records"] == b["unparsed_records"] == []


def test_worst_path_adapter_rejects_truncated_or_inconsistent_records() -> None:
    report = (PROJECT_ROOT / "tests/fixtures/dcp_parser/b2_wpath2.tsv").read_text()
    with pytest.raises(DcpParserError, match="omitted expected records"):
        parse_wpath2_report(report.replace("UTIL_DSP\t4\n", ""))
    with pytest.raises(DcpParserError, match="does not sum"):
        parse_wpath2_report(report.replace("CELLS\t4\t", "CELLS\t5\t"))
    with pytest.raises(DcpParserError, match="repeats SLACK"):
        parse_wpath2_report(report + "SLACK\t-0.785\n")


def test_worst_path_tcl_copies_have_exact_latch_hashes() -> None:
    support = PROJECT_ROOT / "recipes/support"
    assert hashlib.sha256((support / "wpath.tcl").read_bytes()).hexdigest() == WPATH_SHA256
    assert hashlib.sha256((support / "wpath2.tcl").read_bytes()).hexdigest() == WPATH2_SHA256


def test_starred_clb_lut_row_is_accepted_as_in_latch_wpath2() -> None:
    """Pure parser compatibility; no real starred B2 row was observed."""

    report = B2_REPORT.read_text()
    modified = report.replace("CLB LUTs                   |", "CLB LUTs*                  |", 1)
    assert modified != report
    assert parse_probe_report(modified)["utilization_summary"]["CLB LUTs"]["used"] == 1895


def test_conflicting_timing_sources_are_rejected() -> None:
    probe = parse_probe_report(B2_REPORT.read_text())
    wpath = parse_wpath_report(
        (PROJECT_ROOT / "tests/fixtures/dcp_parser/b2_wpath.tsv").read_text()
    )
    evidence = parse_reports(
        B2_METER,
        tag="base",
        state_id="a" * 64,
        meter_version="fixture-v1",
        diag_stdout=(B2_METER / "diag_stdout.txt").read_text(),
    )
    _require_consistent_timing(probe, wpath, evidence)
    wrong_probe = {
        **probe,
        "timing": {**probe["timing"], "wns_ns": -0.700},
    }
    with pytest.raises(DcpParserError, match="wns_ns"):
        _require_consistent_timing(wrong_probe, wpath, evidence)
    with pytest.raises(DcpParserError, match="route_delay_ns"):
        _require_consistent_timing(
            probe, {**wpath, "net_delay_ns": 0.900}, evidence
        )


def test_code_and_environment_hash_snapshots_detect_later_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extractor = DcpTelemetryExtractor(
        runner=object(),
        env_script=PROJECT_ROOT / "recipes/support/env.sh",
        tool_version="2025.1",
    )
    original = parser_module._sha256_file

    def changed_code(path: Path) -> str:
        if Path(path) == Path(parser_module.__file__):
            return "0" * 64
        return original(Path(path))

    monkeypatch.setattr(parser_module, "_sha256_file", changed_code)
    with pytest.raises(DcpParserError, match="extractor Python changed"):
        extractor._verify_sources()

    def changed_env(path: Path) -> str:
        if Path(path) == extractor.env_script:
            return "0" * 64
        return original(Path(path))

    monkeypatch.setattr(parser_module, "_sha256_file", changed_env)
    with pytest.raises(DcpParserError, match="environment script changed"):
        extractor._verify_sources()


def test_saved_real_probe_reports_expose_srl_and_lut_primitive_counts() -> None:
    """Both immutable Vivado reports contain the observed source rows."""
    expected = {
        'LUT1': 9, 'LUT2': 179, 'LUT3': 280,
        'LUT4': 353, 'LUT5': 326, 'LUT6': 806,
    }
    for report in (B2_REPORT, LATCH_REPORT):
        probe = parse_probe_report(report.read_text(encoding='utf-8'))
        assert probe['lut_as_shift_register'] == 0
        assert probe['lut_primitive_counts'] == expected
        assert 'lut_as_shift_register' not in probe['diagnostic_missing_reasons']
        assert 'lut_primitive_counts' not in probe['diagnostic_missing_reasons']


def test_srl_primary_row_missing_and_conflicting_repeat() -> None:
    """The distribution row fills an omitted primary row; both must agree."""
    report = B2_REPORT.read_text(encoding='utf-8')
    primary = '|     LUT as Shift Register  |    0 |'
    secondary = '|   LUT as Shift Register                    |    0 |'
    assert primary in report and secondary in report
    fallback = report.replace(primary, '|     LUT as Shift Storage   |    0 |', 1)
    parsed = parse_probe_report(fallback)
    assert parsed['lut_as_shift_register'] == 0
    assert 'lut_as_shift_register' not in parsed['diagnostic_missing_reasons']
    missing = fallback.replace(
        secondary, '|   LUT as Shift Storage                     |    0 |', 1
    )
    assert parse_probe_report(missing)['lut_as_shift_register'] is None
    conflict = report.replace(
        secondary, '|   LUT as Shift Register                    |    1 |', 1
    )
    with pytest.raises(DcpParserError, match='rows disagree'):
        parse_probe_report(conflict)


def test_primitive_counts_require_real_table_and_zero_absent_types() -> None:
    """No table means unknown; only an observed table licenses absent LUT types as zero."""
    report = B2_REPORT.read_text(encoding='utf-8')
    heading = '\n9. Primitives\n-------------\n'
    lut1 = '| LUT1     |    9 |                 CLB |\n'
    assert heading in report and lut1 in report
    without_table = report.replace(heading, '\n9. Primitive Summary\n--------------------\n', 1)
    parsed = parse_probe_report(without_table)
    assert parsed['lut_primitive_counts'] is None
    assert 'lut_primitive_counts' in parsed['diagnostic_missing_reasons']
    without_lut1 = report.replace(lut1, '', 1)
    counts = parse_probe_report(without_lut1)['lut_primitive_counts']
    assert counts['LUT1'] == 0
    assert counts['LUT6'] == 806
    with pytest.raises(DcpParserError, match='duplicate LUT1'):
        parse_probe_report(report.replace(lut1, lut1 + lut1, 1))
