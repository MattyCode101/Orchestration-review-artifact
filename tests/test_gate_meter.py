"""Gate examples and parser checks against copied, observed run-3 reports."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from orch.config import load_brief
from orch.contract.models import (
    Evidence,
    Guard,
    MetricName,
    MetricValue,
    RunStatus,
    VerdictStatus,
)
from orch.engine.gate import judge
from orch.engine.meter_parsers import (
    MeterError,
    _drc_errors,
    clocks_digest,
    parse_reports,
    xdc_digest,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "meter"
BRIEFS = ROOT / "configs" / "briefs"


def evidence(
    id_: str,
    *,
    wns: float,
    packed: int,
    routed: bool = True,
    drc: int = 0,
    xdc: str = "same-xdc",
    clock: str = "same-clock",
    method: str = "unit-meter",
) -> Evidence:
    return Evidence(
        id=id_,
        state_id=(id_[0] if id_ else "a") * 64,
        metrics={
            MetricName.WNS_NS: MetricValue(
                value=wns, unit="ns", method=method, basis="routed"
            ),
            MetricName.PACKED_LUTS: MetricValue(
                value=packed, unit="count", method=method, basis="routed"
            ),
        },
        routed=routed,
        route_errors=0,
        drc_errors=drc,
        constraint_fingerprint=xdc,
        clock_digest=clock,
        meter_version="unit-meter-v1",
    )


def test_example_1_brief_a_compares_each_step_with_start() -> None:
    brief = load_brief(BRIEFS / "area.yaml")
    start = evidence("s", wns=-0.785, packed=10_000)
    first = evidence("f", wns=-0.865, packed=9_600)
    second = evidence("g", wns=-0.945, packed=9_400)
    first_result = judge(brief, start, start, first)
    second_result = judge(brief, start, first, second)
    assert first_result.status == VerdictStatus.ACCEPTABLE
    assert second_result.status == VerdictStatus.BRIEF_VIOLATION
    assert second_result.physically_valid
    assert second_result.terms["wns_floor"].status == "fail"
    assert second_result.delta_from_parent[MetricName.WNS_NS] == pytest.approx(-0.08)
    assert second_result.delta_from_start[MetricName.WNS_NS] == pytest.approx(-0.16)


def test_example_2_brief_invalid_state_can_be_a_recovery_base() -> None:
    brief = load_brief(BRIEFS / "timing.yaml")
    start = evidence("s", wns=-0.785, packed=10_000)
    timing = evidence("t", wns=-0.725, packed=10_290)
    area1 = evidence("u", wns=-0.730, packed=10_140)
    area2 = evidence("v", wns=-0.730, packed=10_080)
    timing_result = judge(brief, start, start, timing)
    area1_result = judge(brief, start, timing, area1)
    area2_result = judge(brief, start, area1, area2)
    assert timing_result.status == VerdictStatus.BRIEF_VIOLATION
    assert area1_result.status == VerdictStatus.BRIEF_VIOLATION
    assert area2_result.status == VerdictStatus.ACCEPTABLE
    assert all(
        result.physically_valid
        for result in (timing_result, area1_result, area2_result)
    )
    assert area2_result.delta_from_start[MetricName.WNS_NS] == pytest.approx(0.055)


def test_example_3_noise_is_a_flag_not_a_validity_rule() -> None:
    brief = load_brief(BRIEFS / "timing.yaml")
    start = evidence("s", wns=-0.785, packed=10_000)
    improved = evidence("n", wns=-0.735, packed=10_000)
    result = judge(
        brief,
        start,
        start,
        improved,
        noise={MetricName.WNS_NS: 0.12},
    )
    assert result.status == VerdictStatus.ACCEPTABLE
    assert result.brief_valid
    assert result.within_noise[MetricName.WNS_NS] is True
    assert result.delta_from_start[MetricName.WNS_NS] == pytest.approx(0.05)


def test_guard_failure_does_not_rewrite_brief_validity() -> None:
    brief = load_brief(BRIEFS / "area.yaml")
    start = evidence("s", wns=-0.785, packed=10_000)
    candidate = evidence("c", wns=-0.83, packed=9_700)
    result = judge(
        brief,
        start,
        start,
        candidate,
        guards=(
            Guard(
                id="protect_timing",
                metric=MetricName.WNS_NS,
                comparison=">=",
                threshold=-0.80,
            ),
        ),
    )
    assert result.brief_valid
    assert result.status == VerdictStatus.GUARD_VIOLATION
    assert result.guards["protect_timing"].status == "fail"


@pytest.mark.parametrize(
    ("change", "check"),
    [
        ({"routed": False}, "routed"),
        ({"drc_errors": 1}, "drc_errors"),
        ({"constraint_fingerprint": "loosened"}, "timing_constraints"),
        ({"clock_digest": "changed"}, "clocks"),
    ],
)
def test_hard_validity_rejects_bad_state(change: dict, check: str) -> None:
    brief = load_brief(BRIEFS / "timing.yaml")
    start = evidence("s", wns=-0.785, packed=10_000)
    candidate = evidence("c", wns=-0.7, packed=9_900).model_copy(update=change)
    result = judge(brief, start, start, candidate)
    assert result.status == VerdictStatus.INVALID
    assert not result.physically_valid
    assert result.validity[check].status == "fail"


def test_missing_or_incomparable_objective_is_invalid_not_pass() -> None:
    brief = load_brief(BRIEFS / "timing.yaml")
    start = evidence("s", wns=-0.785, packed=10_000)
    wrong_method = evidence("w", wns=-0.7, packed=9_900, method="other")
    result = judge(brief, start, start, wrong_method)
    assert result.status == VerdictStatus.INVALID
    assert not result.brief_valid


def test_failed_run_has_no_admissible_state() -> None:
    brief = load_brief(BRIEFS / "area.yaml")
    start = evidence("s", wns=-0.785, packed=10_000)
    result = judge(
        brief,
        start,
        start,
        None,
        run_status=RunStatus.NO_OUTPUT,
    )
    assert result.status == VerdictStatus.RUN_FAILED
    assert not result.physically_valid


def test_gate_is_pure_on_identical_inputs() -> None:
    brief = load_brief(BRIEFS / "area.yaml")
    start = evidence("s", wns=-0.785, packed=10_000)
    candidate = evidence("c", wns=-0.80, packed=9_900)
    first = judge(brief, start, start, candidate)
    second = judge(brief, start, start, candidate)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_observed_vex_b2_reports_reproduce_all_four_smoke_numbers() -> None:
    result = parse_reports(
        FIXTURE,
        tag="base",
        state_id="a" * 64,
        meter_version="fixture-v1",
        diag_stdout=(FIXTURE / "diag_stdout.txt").read_text(),
    )
    assert result.metric(MetricName.WNS_NS).value == -0.785
    assert result.metric(MetricName.TNS_NS).value == -498.982
    assert result.metric(MetricName.PACKED_LUTS).value == 1895
    assert result.metric(MetricName.NETLIST_LUTS).value == 1953
    assert result.metric(MetricName.FFS).value == 1231
    assert result.metric(MetricName.DYNAMIC_POWER_W).value == 0.263
    assert result.routed and result.route_errors == 0 and result.drc_errors == 0
    assert result.v_props == {"n_DONT_TOUCH": 0, "n_IS_LOC_FIXED": 0}


def test_missing_report_or_logged_error_is_not_complete_evidence(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    shutil.copytree(FIXTURE, reports)
    (reports / "base_utilization.rpt").unlink()
    with pytest.raises(MeterError, match="missing meter report"):
        parse_reports(
            reports,
            tag="base",
            state_id="a" * 64,
            meter_version="fixture-v1",
            diag_stdout=(reports / "diag_stdout.txt").read_text(),
        )
    with pytest.raises(MeterError, match="MEASURE_ERROR"):
        parse_reports(
            FIXTURE,
            tag="base",
            state_id="a" * 64,
            meter_version="fixture-v1",
            diag_stdout=(FIXTURE / "diag_stdout.txt").read_text(),
            measurement_log="MEASURE_ERROR timing_summary: simulated",
        )


def test_drc_warnings_do_not_count_as_errors() -> None:
    report = (FIXTURE / "base_drc.rpt").read_text()
    assert _drc_errors(report) == 0
    modified = report.replace("| Warning  |", "| Error    |", 1)
    assert modified != report
    assert _drc_errors(modified) == 8


def test_valid3_digests_ignore_effective_zero_uncertainty_only() -> None:
    base = "create_clock -period 1.570 -name clk [get_ports clk]"
    modified = (
        base
        + "\nset_power_opt -include_registers\n"
        + "set_clock_uncertainty -setup 0.002 [get_clocks]\n"
        + "set_clock_uncertainty -setup 0.000 [get_clocks]\n"
    )
    assert xdc_digest(base) == xdc_digest(modified)
    assert xdc_digest(base) != xdc_digest(
        base + "\nset_clock_uncertainty -setup 0.002 [get_clocks]\n"
    )
    clock_base = "Clock Report\n\nAttributes\n"
    assert clocks_digest(clock_base) == clocks_digest(
        clock_base + "clk clk 0.000 rr -> ff\n"
    )
    assert clocks_digest(clock_base) != clocks_digest(
        clock_base + "clk clk 0.002 rr -> ff\n"
    )



def test_meter_wraps_copied_tcl_and_uses_two_tagged_runner_jobs(
    tmp_path: Path,
) -> None:
    """A stub runner exercises the wrapper; real Vivado remains UNTESTED."""

    import hashlib

    from orch.contract.models import DesignState
    from orch.engine.meter import VivadoMeter
    from orch.interfaces import ProcessResult

    class StubRunner:
        def __init__(self) -> None:
            self.calls = []

        def run(self, spec, *, run_id: str, run_dir: Path) -> ProcessResult:
            self.calls.append(spec)
            run_dir.mkdir(parents=True, exist_ok=True)
            stdout = run_dir / "stdout.txt"
            stderr = run_dir / "stderr.txt"
            stderr.write_text("")
            if len(self.calls) == 1:
                reports = run_dir / "reports"
                reports.mkdir()
                for source in FIXTURE.glob("base_*"):
                    shutil.copyfile(
                        source, reports / source.name.replace("base_", "final_")
                    )
                stdout.write_text("MEASURE_TOTAL final 42 ms\n")
            else:
                stdout.write_text((FIXTURE / "diag_stdout.txt").read_text())
            return ProcessResult(
                run_id=run_id,
                status=RunStatus.OK,
                exit_code=0,
                wall_s=0.1,
                process_group_kills=0,
                tag_sweep_kills=0,
                tag="stub-tag",
                stdout_path=str(stdout),
                stderr_path=str(stderr),
            )

    checkpoint = tmp_path / "stub.dcp"
    checkpoint.write_bytes(b"test-only checkpoint; no Vivado")
    state = DesignState(
        id=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        design="vexriscv_re-place",
        path=str(checkpoint),
    )
    runner = StubRunner()
    meter = VivadoMeter(
        runner=runner,
        env_script=FIXTURE / "empty_env.sh",
        tool_version="2025.1",
    )
    result = meter.measure(state, run_dir=tmp_path / "run")
    assert result.metric(MetricName.WNS_NS).value == -0.785
    assert result.metric(MetricName.NETLIST_LUTS).value == 1953
    assert len(runner.calls) == 2
    assert "measure2.tcl -tclargs" in runner.calls[0].argv[2]
    assert "diag_lut.tcl -tclargs" in runner.calls[1].argv[2]
    assert (tmp_path / "run" / "meter" / "provenance.json").is_file()



def test_verdict_identifier_includes_brief_and_start_context() -> None:
    brief = load_brief(BRIEFS / "area.yaml")
    another_brief = brief.model_copy(update={"id": "same_terms_different_brief"})
    start = evidence("s", wns=-0.785, packed=10_000)
    candidate = evidence("c", wns=-0.80, packed=9_900)
    first = judge(brief, start, start, candidate)
    second = judge(another_brief, start, start, candidate)
    assert first.status == second.status
    assert first.id != second.id


def test_missing_objective_does_not_relabel_physical_validity() -> None:
    brief = load_brief(BRIEFS / "timing.yaml")
    start = evidence("s", wns=-0.785, packed=10_000)
    candidate = evidence("c", wns=-0.70, packed=9_900)
    candidate = candidate.model_copy(
        update={
            "metrics": {
                MetricName.PACKED_LUTS: candidate.metric(MetricName.PACKED_LUTS)
            }
        }
    )
    result = judge(brief, start, start, candidate)
    assert result.physically_valid
    assert not result.brief_valid
    assert result.status == VerdictStatus.INVALID



def test_echoed_tcl_source_is_not_an_emitted_measure_error() -> None:
    """Smoke log echoed rl_try's puts source; only a real puts line is fatal."""

    from orch.engine.meter_parsers import has_measure_error

    echoed = '##     puts "MEASURE_ERROR $name: $err"\n'
    real_error = "MEASURE_ERROR timing_summary: simulated report failure\n"
    diag = (FIXTURE / "diag_stdout.txt").read_text()
    assert not has_measure_error(echoed)
    assert has_measure_error(echoed + real_error)

    accepted = parse_reports(
        FIXTURE,
        tag="base",
        state_id="a" * 64,
        meter_version="fixture-v1",
        diag_stdout=echoed + diag,
        measurement_log=echoed,
    )
    assert accepted.metric(MetricName.WNS_NS).value == -0.785

    with pytest.raises(MeterError, match="MEASURE_ERROR"):
        parse_reports(
            FIXTURE,
            tag="base",
            state_id="a" * 64,
            meter_version="fixture-v1",
            diag_stdout=diag,
            measurement_log=echoed + real_error,
        )
    with pytest.raises(MeterError, match="MEASURE_ERROR"):
        parse_reports(
            FIXTURE,
            tag="base",
            state_id="a" * 64,
            meter_version="fixture-v1",
            diag_stdout=diag + real_error,
            measurement_log=echoed,
        )

