"""Parsers for the exact run-3 measurement reports copied into this project.

Sources: R3/code/run1/parse_results.py:22-98, R3/code/collect.py:33-42,
and R3/code/collect3.py:20-40. The two digest functions below preserve the
run-3 valid3 normalization, including the effective-zero uncertainty rule.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from orch.contract.models import Evidence, MetricName, MetricValue


PARSER_VERSION = "0.1.0"
NUM = r"[-+]?\d+(?:\.\d+)?"
UNC_LINE = re.compile(r"^set_clock_uncertainty -setup ([0-9.]+) \[get_clocks\]$")
ZERO_UNC_ROW = re.compile(r"^\S+\s+\S+\s+0\.000\s+[rf]+ -> [rf]+\s*$")
DRC_ROW = re.compile(
    r"^\|\s*(\S+)\s*\|\s*(Error|Critical Warning|Warning|Advisory)"
    r"\s*\|[^|]*\|\s*(\d+)\s*\|",
    re.M,
)
PROP_KEYS = ("n_DONT_TOUCH", "n_IS_LOC_FIXED", "n_IS_BEL_FIXED")
# Vivado echoes Tcl source with a leading "##"; only an emitted puts line starts here.
EMITTED_MEASURE_ERROR = re.compile(r"^MEASURE_ERROR(?: |$)", re.M)


class MeterError(RuntimeError):
    """A tool run or report cannot support complete independent Evidence."""


def has_measure_error(log: str) -> bool:
    """Detect an emitted run-3 rl_try error, excluding echoed Tcl source."""

    return EMITTED_MEASURE_ERROR.search(log) is not None


def xdc_digest(text: str) -> str:
    """Copy of the normalization in R3/code/collect3.py:24-31."""

    lines = [
        line.strip()
        for line in (text or "").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    lines = [line for line in lines if not line.startswith("set_power_opt")]
    unc = [UNC_LINE.match(line) for line in lines]
    vals = [match.group(1) for match in unc if match]
    if vals and float(vals[-1]) == 0.0:
        lines = [line for line, match in zip(lines, unc) if not match]
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


def clocks_digest(text: str) -> str:
    """Copy of the normalization in R3/code/collect3.py:34-40."""

    t = text or ""
    i = t.find("Clock Report")
    body = [line.rstrip() for line in t[i:].splitlines() if line.strip()] if i >= 0 else []
    body = [
        line
        for line in body
        if not ZERO_UNC_ROW.match(line) and not line.startswith("From Clock")
    ]
    return hashlib.sha256("\n".join(body).encode()).hexdigest()


def _required(report_dir: Path, tag: str, suffix: str) -> str:
    path = report_dir / f"{tag}_{suffix}"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise MeterError(f"missing meter report: {path}") from exc
    if not text.strip():
        raise MeterError(f"empty meter report: {path}")
    return text


def _timing(text: str) -> tuple[float, float]:
    """Use run-1's summary-row anchor, not a first-number guess."""

    match = re.search(
        r"\| Design Timing Summary.*?\n-+\n\n\s+WNS\(ns\).*?"
        r"\n\s*-+.*?\n\s*(.*?)\n",
        text,
        re.S,
    )
    if match is None:
        raise MeterError("timing summary has no WNS/TNS row")
    values = match.group(1).split()
    if len(values) < 2 or not re.fullmatch(NUM, values[0]) or not re.fullmatch(NUM, values[1]):
        raise MeterError("timing summary WNS/TNS is incomplete")
    return float(values[0]), float(values[1])


def _utilization(text: str) -> tuple[int, int]:
    """Parse packed CLB LUTs, distinct from logical LUT cells."""

    found: dict[str, int] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(cells) < 2 or not re.fullmatch(NUM, cells[1]):
            continue
        if cells[0] in ("CLB LUTs", "CLB LUTs*"):
            found["packed_luts"] = int(float(cells[1]))
        elif cells[0] == "CLB Registers":
            found["ffs"] = int(float(cells[1]))
    if "packed_luts" not in found or "ffs" not in found:
        raise MeterError("utilization report lacks CLB LUTs or CLB Registers")
    return found["packed_luts"], found["ffs"]


def _route(text: str) -> tuple[bool, int]:
    """Match R3/code/collect.py:150-152's routed predicate."""

    def field(label: str, required: bool = True) -> int | None:
        match = re.search(re.escape(label) + r"\.*\s*:\s*(\d+)", text)
        if match is None:
            if required:
                raise MeterError(f"route report lacks {label}")
            return None
        return int(match.group(1))

    routable = field("# of routable nets")
    fully = field("# of fully routed nets")
    errors = field("# of nets with routing errors")
    unrouted = field("# of unrouted nets", required=False)
    partial = field("# of partially routed nets", required=False)
    assert routable is not None and fully is not None and errors is not None
    routed = (
        errors == 0
        and fully == routable
        and (unrouted is None or unrouted == 0)
        and (partial is None or partial == 0)
    )
    return routed, errors


def _drc_errors(text: str) -> int:
    """Count only severity Error, not all violations or warnings.

    R3/code/collect.py:33-42 parses the severity table. The summary total
    guards against a changed or truncated table silently appearing error-free.
    """

    total_match = re.search(r"Checks found:\s*(\d+)", text)
    if total_match is None:
        raise MeterError("DRC report lacks Checks found")
    total = int(total_match.group(1))
    rows = list(DRC_ROW.finditer(text))
    counted = sum(int(match.group(3)) for match in rows)
    if counted != total:
        raise MeterError(
            f"DRC severity table counts {counted} checks but summary says {total}"
        )
    return sum(int(match.group(3)) for match in rows if match.group(2) == "Error")


def _dynamic_power(text: str) -> float:
    match = re.search(r"\|\s*Dynamic \(W\)\s*\|\s*(" + NUM + r")", text)
    if match is None:
        raise MeterError("power report lacks Dynamic (W)")
    return float(match.group(1))


def _netlist_luts(diag_stdout: str) -> int:
    """Run-3 diag_lut.tcl:7-11 emits the independent logical LUT census."""

    if has_measure_error(diag_stdout):
        raise MeterError("LUT diagnostic reported MEASURE_ERROR")
    matches = re.findall(r"^RL_DIAG n_lut_cells (\d+)\s*$", diag_stdout, re.M)
    if len(matches) != 1:
        raise MeterError("LUT diagnostic lacks one RL_DIAG n_lut_cells line")
    return int(matches[0])


def _props(text: str | None) -> dict[str, int] | None:
    """Keep observed property counts; absence is unknown rather than zero."""

    if text is None:
        return None
    props: dict[str, int] = {}
    for line in text.splitlines():
        name, sep, raw = line.partition("\t")
        if sep and name in PROP_KEYS and raw.strip().isdigit():
            props[name] = int(raw.strip())
    return props or None


def parse_reports(
    report_dir: Path,
    *,
    tag: str,
    state_id: str,
    meter_version: str,
    diag_stdout: str,
    measurement_log: str = "",
    fake: bool = False,
) -> Evidence:
    """Turn re-opened run-3 reports into complete, traceable Evidence.

    This parser is tested on copied R3 VexRiscv B2 reports. It checks files
    individually because lib_measure.tcl:9-15 catches report errors and Vivado
    can still return zero.
    """

    if has_measure_error(measurement_log):
        raise MeterError("measurement log contains MEASURE_ERROR")
    timing = _required(report_dir, tag, "timing_summary.rpt")
    util = _required(report_dir, tag, "utilization.rpt")
    route = _required(report_dir, tag, "route_status.rpt")
    drc = _required(report_dir, tag, "drc.rpt")
    power = _required(report_dir, tag, "power.rpt")
    xdc = _required(report_dir, tag, "timing.xdc")
    clocks = _required(report_dir, tag, "report_clocks.rpt")
    if "Clock Report" not in clocks:
        raise MeterError("clock report lacks Clock Report section")

    wns, tns = _timing(timing)
    packed_luts, ffs = _utilization(util)
    routed, route_errors = _route(route)
    drc_errors = _drc_errors(drc)
    dynamic_power = _dynamic_power(power)
    netlist_luts = _netlist_luts(diag_stdout)

    props_path = report_dir / f"{tag}_props.txt"
    props_text = (
        props_path.read_text(encoding="utf-8", errors="replace")
        if props_path.exists()
        else None
    )
    metrics = {
        MetricName.WNS_NS: MetricValue(
            value=wns, unit="ns", method="run3_timing_summary", basis="routed"
        ),
        MetricName.TNS_NS: MetricValue(
            value=tns, unit="ns", method="run3_timing_summary", basis="routed"
        ),
        MetricName.PACKED_LUTS: MetricValue(
            value=packed_luts,
            unit="count",
            method="run3_report_utilization",
            basis="routed",
        ),
        MetricName.NETLIST_LUTS: MetricValue(
            value=netlist_luts,
            unit="count",
            method="run3_diag_lut_census",
            basis="routed",
        ),
        MetricName.FFS: MetricValue(
            value=ffs, unit="count", method="run3_report_utilization", basis="routed"
        ),
        MetricName.DYNAMIC_POWER_W: MetricValue(
            value=dynamic_power,
            unit="W",
            method="run3_vectorless_report_power",
            basis="routed_estimated",
        ),
    }
    provisional = Evidence(
        id="pending",
        state_id=state_id,
        metrics=metrics,
        routed=routed,
        route_errors=route_errors,
        drc_errors=drc_errors,
        constraint_fingerprint=xdc_digest(xdc),
        clock_digest=clocks_digest(clocks),
        v_props=_props(props_text),
        meter_version=meter_version,
        raw_report_path=str(report_dir),
        fake=fake,
    )
    # Paths differ across runs; content-based identity covers only measured data.
    payload = provisional.model_dump(
        mode="json", exclude={"id", "raw_report_path"}
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return provisional.model_copy(
        update={"id": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}
    )

