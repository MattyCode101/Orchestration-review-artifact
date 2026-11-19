"""Read-only DCP telemetry assembled from exact LATCH parsers and run-3 Evidence.

LATCH source commit a722b1ecacfe10f549d66efd7f56a0e35c608a97.
Pure parsers are copied from:
  optimizer_pipeline/analysis/report_probe_backends.py:247-391
  optimizer_pipeline/analysis_schema.py:35-48,103-110
  optimizer_pipeline/execution/vivado_utilization.py:17-96,139-165
  dcp_optimizer.py:151-202
The utilization parser skips Vivado's observed Prohibited column and permits
the starred CLB LUT label already accepted by LATCH wpath2.tcl. These are
Python-only changes; no Vivado-facing Tcl has been changed.

The LATCH path probe Tcl is copied byte for byte in recipes/support.
The base probe, meter and both worst-path scripts were run on VexRiscv B2.
Other designs and nonempty pblocks remain UNTESTED real-tool paths; see
notes/dcp_telemetry.md for the exact run and remaining limits.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from orch.contract.models import DesignState, Evidence, MetricName, RunStatus
from orch.interfaces import LaunchSpec, Meter, Runner

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_TEMPLATE = PROJECT_ROOT / "recipes" / "support" / "latch_path_probe.tcl.template"
PROBE_TEMPLATE_SHA256 = "926ebb426ae29b4ba90b135f1fb561f95dcb76da7831306d04dc5cf3d3044e55"
LATCH_COMMIT = "a722b1ecacfe10f549d66efd7f56a0e35c608a97"
LATCH_REPORT_PARSER_SHA256 = "ff3e71b796b3855155107151f71a38acb5bd509635839ab9bdd7bfbb8fd8ae51"
LATCH_UTIL_PARSER_SHA256 = "cb07384aeacb1334c0b2fc80767c919a034755d79bd7d3b2ca0b024240d02428"
LATCH_TIMING_PARSER_SHA256 = "87c2762d5a5092b09bb42d6e996ba27df6ac2755b296d7172d56667586f87b4c"
SAFE_TCL_PATH = re.compile(r"^[A-Za-z0-9_./-]+$")


class DcpParserError(RuntimeError):
    """A checkpoint probe or source report cannot support traceable telemetry."""


# The following definitions are copied from the cited LATCH Python sources.
# Only _parse_utilization_report's row regex differs, as documented below.


DOMINANT_PATH_ROUTE_THRESHOLD = 0.65

DOMINANT_PATH_LOGIC_THRESHOLD = 0.35

def as_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

def classify_dominant_path_type(route_delay_ratio: Any) -> str:
    """Route-vs-logic label for a route/(route+logic) delay ratio."""
    ratio = as_float(route_delay_ratio)
    if ratio is None:
        return "unknown"
    if ratio >= DOMINANT_PATH_ROUTE_THRESHOLD:
        return "routing_dominated"
    if ratio <= DOMINANT_PATH_LOGIC_THRESHOLD:
        return "logic_dominated"
    return "mixed"

_NUMERIC_TOKEN = re.compile(r"^[-+]?\d+(?:\.\d+)?$")

def _parse_design_timing_summary(text: str) -> Dict[str, Any]:
    """Read the Design Timing Summary table.

    Vivado prints WNS/TNS/failing endpoints as a column-aligned table -- header
    row, dashes row, then the values -- so each value sits two lines below its
    own heading rather than beside it. Only the design-level table is read: the
    per-clock tables further down start with a label column instead.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("WNS(ns)") or "TNS(ns)" not in stripped:
            continue
        for data_line in lines[index + 1 : index + 4]:
            tokens = data_line.split()
            if len(tokens) < 3 or not _NUMERIC_TOKEN.match(tokens[0]):
                continue
            return {
                "wns_ns": float(tokens[0]),
                "tns_ns": float(tokens[1]) if _NUMERIC_TOKEN.match(tokens[1]) else None,
                "failing_endpoints": int(tokens[2]) if tokens[2].isdigit() else None,
            }
    return {}

def _parse_timing_report(text: str) -> Dict[str, Any]:
    summary = _parse_design_timing_summary(text)
    wns = summary.get("wns_ns")
    tns = summary.get("tns_ns")
    failing = summary.get("failing_endpoints")
    if wns is None:
        wns = _first_float(r"\bWNS\(ns\)\s+([-+]?\d+(?:\.\d+)?)", text)
    if tns is None:
        tns = _first_float(r"\bTNS\(ns\)\s+([-+]?\d+(?:\.\d+)?)", text)
    if failing is None:
        failing = _first_int(r"\bFailing Endpoints\s+(\d+)", text)
    data_delay = _first_float(r"\bData Path Delay:\s*([-+]?\d+(?:\.\d+)?)", text)
    logic_delay = _first_float(r"\blogic\s+([-+]?\d+(?:\.\d+)?)", text)
    route_delay = _first_float(r"\broute\s+([-+]?\d+(?:\.\d+)?)", text)
    levels = _first_int(r"\bLogic Levels:\s*(\d+)", text)
    route_ratio = None
    if route_delay is not None and logic_delay is not None and route_delay + logic_delay > 0:
        route_ratio = round(route_delay / (route_delay + logic_delay), 6)
    # Canonical classifier only -- this used to be a second, incompatible
    # vocabulary (route/logic at 0.55/0.45). See analysis_schema.
    dominant = classify_dominant_path_type(route_ratio) if route_ratio is not None else None
    return {
        "timing": {
            "wns_ns": wns,
            "tns_ns": tns,
            "failing_endpoints": failing,
        },
        "critical_path_features": {
            "data_path_delay_ns": data_delay,
            "logic_delay_ns": logic_delay,
            "route_delay_ns": route_delay,
            "route_delay_ratio": route_ratio,
            "dominant_path_type": dominant,
            "max_logic_levels": levels,
        },
        "route_vs_logic_delay_hint": dominant,
        "diagnostic_missing_reasons": _missing_report_reasons({
            "critical_path_features": any(v is not None for v in (data_delay, logic_delay, route_delay, route_ratio, levels)),
            "timing": any(v is not None for v in (wns, tns, failing)),
        }),
    }

def _parse_utilization_report(text: str) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for resource in ("CLB LUTs", "CLB Registers", "Block RAM Tile", "DSPs", "URAM", "BUFG"):
        # UNTESTED real starred row; LATCH wpath2.tcl:33 and the run-3 meter permit it.
        label = re.escape(resource) + (r"\*?" if resource == "CLB LUTs" else "")
        # A saved Finn report uses 24.5 Block RAM Tiles; keep that measured
        # fraction instead of silently omitting the row.
        used_pattern = r"([\d,]+(?:\.\d+)?)" if resource == "Block RAM Tile" else r"([\d,]+)"
        pattern = rf"{label}\s*\|\s*{used_pattern}\s*\|\s*([\d,]+)\s*\|\s*[\d,]*\s*\|\s*([\d,]+)\s*\|\s*([-+]?\d+(?:\.\d+)?)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            used_raw = match.group(1).replace(",", "")
            rows[resource] = {
                "used": float(used_raw) if "." in used_raw else int(used_raw),
                "fixed": _int_from_text(match.group(2)),
                "available": _int_from_text(match.group(3)),
                "utilization_pct": float(match.group(4)),
            }
    fanout = _parse_high_fanout_table(text)
    hardblock_pressure = {
        key: value for key, value in rows.items()
        if any(token in key.lower() for token in ("ram", "dsp", "uram", "bufg"))
    }
    return {
        "utilization_summary": rows,
        "fanout": fanout,
        "hardblock_resource_pressure": hardblock_pressure,
        "diagnostic_missing_reasons": _missing_report_reasons({
            "utilization_summary": bool(rows),
            "fanout": bool(fanout),
            "hardblock_resource_pressure": bool(hardblock_pressure),
        }),
    }

# UNTESTED on fresh real-tool output after this addition; both helpers are
# exercised against saved, checksum-pinned LATCH and B2 probe reports.
def _utilization_section(report_text: str, title: str, next_title: str) -> str | None:
    """Find an actual underlined utilization section, not its contents entry."""

    def heading(name: str) -> re.Pattern[str]:
        return re.compile(rf"(?m)^{re.escape(name)}[ \t]*\r?\n-+[ \t]*\r?$")

    start = heading(title).search(report_text)
    if start is None:
        return None
    rest = report_text[start.end():]
    end = heading(next_title).search(rest)
    return rest[:end.start()] if end is not None else rest


def _utilization_used(section: str, label: str) -> int | None:
    """Read the Used column of one named utilization row."""

    row = re.search(
        rf"(?m)^\|[ \t]*{re.escape(label)}[ \t]*\|[ \t]*([^|]*)\|",
        section,
    )
    if row is None:
        return None
    raw = row.group(1).strip()
    if not re.fullmatch(r"[0-9][0-9,]*", raw):
        raise DcpParserError(f"invalid Used count for {label}: {raw!r}")
    return int(raw.replace(",", ""))


def _parse_lut_as_shift_register(report_text: str) -> int | None:
    """Use either observed SRL row; require agreement when both tables give one."""

    primary = _utilization_section(
        report_text, "1. CLB Logic", "1.1 Summary of Registers by Type"
    )
    used = _utilization_used(primary, "LUT as Shift Register") if primary is not None else None
    distribution = _utilization_section(
        report_text, "2. CLB Logic Distribution", "3. BLOCKRAM"
    )
    repeated = (
        _utilization_used(distribution, "LUT as Shift Register")
        if distribution is not None else None
    )
    if used is not None and repeated is not None and repeated != used:
        raise DcpParserError("LUT as Shift Register rows disagree")
    return used if used is not None else repeated


def _parse_lut_primitive_counts(report_text: str) -> dict[str, int] | None:
    """Count LUT1..LUT6 from the real Primitives table; missing types mean zero."""

    section = _utilization_section(report_text, "9. Primitives", "10. Black Boxes")
    if section is None or re.search(
        r"(?m)^\|[ \t]*Ref Name[ \t]*\|[ \t]*Used[ \t]*\|"
        r"[ \t]*Functional Category[ \t]*\|",
        section,
    ) is None:
        return None
    counts = {f"LUT{index}": 0 for index in range(1, 7)}
    seen: set[str] = set()
    for line in section.splitlines():
        columns = line.split("|")
        if len(columns) < 4:
            continue
        name = columns[1].strip()
        if name not in counts:
            continue
        if name in seen:
            raise DcpParserError(f"duplicate {name} primitive row")
        raw = columns[2].strip()
        if not re.fullmatch(r"[0-9][0-9,]*", raw):
            raise DcpParserError(f"invalid Used count for {name}: {raw!r}")
        counts[name] = int(raw.replace(",", ""))
        seen.add(name)
    return counts

_HIGH_FANOUT_ROW = re.compile(r"^\|\s*(\S.*?)\s*\|\s*(\d+)\s*\|\s*(\S+)\s*\|$", re.M)

def _parse_high_fanout_table(text: str) -> Dict[str, Any]:
    """Parse report_high_fanout_nets' summary table.

    Scoped to the section after the "High Fan-out Nets" heading. That matters
    because report_utilization runs in the same session and writes into the
    same text: its resource tables use an identical three-column pipe layout,
    so an unscoped match reads *available LUT counts* as net fanouts.

    Vivado hyphenates the heading ("Fan-out"), which is why the original
    "High Fanout" regex never matched anything at all.
    """
    marker = re.search(r"High\s+Fan-?out\s+Nets", text or "", re.I)
    if not marker:
        return {}
    section = text[marker.end():]
    rows = [
        (m.group(1), int(m.group(2)))
        for m in _HIGH_FANOUT_ROW.finditer(section)
        if not m.group(1).lower().startswith("net name")
    ]
    if not rows:
        return {}
    return {"num_high_fanout_nets": len(rows), "max_fanout": max(f for _n, f in rows)}

def _missing_report_reasons(present: Dict[str, bool]) -> Dict[str, str]:
    return {key: "not found in report preview" for key, ok in present.items() if not ok}

def _first_float(pattern: str, text: str) -> Optional[float]:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except (TypeError, ValueError):
        return None

def _first_int(pattern: str, text: str) -> Optional[int]:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return _int_from_text(match.group(1))

def _int_from_text(value: str) -> Optional[int]:
    try:
        return int(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None

RESOURCE_KEYS = ("luts", "ffs", "dsps", "brams", "urams")

def int_count_or_none(value: Any, *, ceil_fraction: bool = True) -> Optional[int]:
    """Convert a Vivado count-like value to an integer resource count."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().replace(",", "")
        if not text:
            return None
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        try:
            number = float(match.group(0))
        except ValueError:
            return None
    if number <= 0:
        return 0
    return int(math.ceil(number) if ceil_fraction else round(number))

def empty_utilization() -> Dict[str, Optional[int]]:
    data: Dict[str, Optional[int]] = {}
    for key in RESOURCE_KEYS:
        data[key] = None
        data[f"{key}_1_5x"] = None
    return data

def parse_pblock_utilization_text(report: str) -> Dict[str, Optional[int]]:
    """Parse Vivado pblock-utilization text or JSON into stable lower-case keys."""
    parsed = empty_utilization()
    if not report:
        return parsed

    try:
        payload = json.loads(report)
        if isinstance(payload, dict):
            aliases = {
                "luts": ["luts", "lut", "LUT", "LUTs"],
                "ffs": ["ffs", "ff", "FF", "FFs"],
                "dsps": ["dsps", "dsp", "DSP", "DSPs"],
                "brams": ["brams", "bram", "BRAM", "BRAMs"],
                "urams": ["urams", "uram", "URAM", "URAMs"],
                "luts_1_5x": ["luts_1_5x", "lut_1_5x"],
                "ffs_1_5x": ["ffs_1_5x", "ff_1_5x"],
                "dsps_1_5x": ["dsps_1_5x", "dsp_1_5x"],
                "brams_1_5x": ["brams_1_5x", "bram_1_5x"],
                "urams_1_5x": ["urams_1_5x", "uram_1_5x"],
            }
            for key, candidates in aliases.items():
                for candidate in candidates:
                    if candidate in payload:
                        parsed[key] = int_count_or_none(payload.get(candidate))
                        break
    except (json.JSONDecodeError, TypeError):
        pass

    lower_report = report.lower()
    split_index = lower_report.find("1.5x")
    base_block = report[:split_index] if split_index >= 0 else report
    scaled_block = report[split_index:] if split_index >= 0 else ""
    base_values = _parse_resource_block(base_block)
    scaled_values = _parse_resource_block(scaled_block)

    for key in RESOURCE_KEYS:
        if parsed[key] is None:
            parsed[key] = base_values.get(key)
        scaled_key = f"{key}_1_5x"
        if parsed[scaled_key] is None:
            parsed[scaled_key] = scaled_values.get(key)
        if parsed[scaled_key] is None and parsed[key] is not None:
            parsed[scaled_key] = int(round(parsed[key] * 1.5))
    return parsed

def _parse_resource_block(block: str) -> Dict[str, Optional[int]]:
    values: Dict[str, Optional[int]] = {key: None for key in RESOURCE_KEYS}
    patterns = {
        "luts": (r"\|\s*Slice LUTs\s*\|([^|\n]+)", r"\|\s*LUT as Logic\s*\|([^|\n]+)", r"LUTs\s*:\s*([0-9,]+(?:\.\d+)?)"),
        "ffs": (
            r"\|\s*Register as Flip Flop\s*\|([^|\n]+)",
            r"\|\s*Slice Registers\s*\|([^|\n]+)",
            r"\|\s*CLB Registers\s*\|([^|\n]+)",
            r"FFs\s*:\s*([0-9,]+(?:\.\d+)?)",
        ),
        "dsps": (r"\|\s*DSPs\s*\|([^|\n]+)", r"DSPs\s*:\s*([0-9,]+(?:\.\d+)?)"),
        "brams": (r"\|\s*Block RAM Tile\s*\|([^|\n]+)", r"BRAMs\s*:\s*([0-9,]+(?:\.\d+)?)"),
        "urams": (r"\|\s*URAM\s*\|([^|\n]+)", r"URAMs\s*:\s*([0-9,]+(?:\.\d+)?)"),
    }
    for key, candidates in patterns.items():
        for pattern in candidates:
            match = re.search(pattern, block, re.IGNORECASE)
            if match:
                values[key] = int_count_or_none(match.group(1))
                break
    return values

def parse_timing_summary_static(timing_report: str) -> dict:
    """
    Parse timing summary report to extract WNS, TNS, and failing endpoints.
    Returns dict with keys: wns, tns, failing_endpoints
    
    Parses the Design Timing Summary table:
        WNS(ns)      TNS(ns)  TNS Failing Endpoints  ...
        -------      -------  ---------------------  ...
         -0.099       -1.449                     42  ...
    
    This is a shared utility function used by both FPGAOptimizer and FPGAOptimizerTest.
    """
    result = {
        "wns": None,
        "tns": None,
        "failing_endpoints": None
    }
    
    lines = timing_report.split('\n')
    
    # Find the line with "WNS(ns)" header
    header_idx = -1
    for i, line in enumerate(lines):
        if 'WNS(ns)' in line and 'TNS(ns)' in line:
            header_idx = i
            break
    
    if header_idx == -1:
        return result
    
    # The data line should be 2 lines after the header (skipping the dashes line)
    # Format: whitespace + values separated by whitespace
    data_idx = header_idx + 2
    if data_idx >= len(lines):
        return result
    
    data_line = lines[data_idx].strip()
    if not data_line:
        return result
    
    # Split by whitespace and extract first 3 values: WNS, TNS, TNS Failing Endpoints
    parts = data_line.split()
    if len(parts) >= 3:
        try:
            result["wns"] = float(parts[0])
            result["tns"] = float(parts[1])
            result["failing_endpoints"] = int(parts[2])
        except (ValueError, IndexError):
            # If parsing fails, leave as None
            pass
    
    return result



def _sha256_file(path: Path) -> str:
    """Hash checkpoint, exact rendered Tcl, or report bytes without loading all into RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Capture the file seen by this Python import, not its possibly edited later bytes.
_IMPORTED_EXTRACTOR_SHA256 = _sha256_file(Path(__file__))


def _safe_tcl_path(path: Path) -> str:
    """Reject Tcl metacharacters because the original LATCH template is unquoted."""

    value = str(path)
    if not SAFE_TCL_PATH.fullmatch(value):
        raise DcpParserError(f"path cannot be interpolated into copied LATCH Tcl: {path}")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish telemetry only after all reports and hashes have been checked."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def parse_probe_report(report_text: str) -> dict[str, Any]:
    """Parse LATCH's timing, path, fanout, and utilization reports from one probe."""

    if not report_text or not report_text.strip():
        raise DcpParserError("LATCH path probe report is empty")
    timing = _parse_timing_report(report_text)
    active_timing = parse_timing_summary_static(report_text)
    for active_key, probe_key in (
        ("wns", "wns_ns"),
        ("tns", "tns_ns"),
        ("failing_endpoints", "failing_endpoints"),
    ):
        lhs = active_timing[active_key]
        rhs = timing["timing"][probe_key]
        if lhs is not None and rhs is not None and not math.isclose(lhs, rhs, abs_tol=1e-6):
            raise DcpParserError(f"LATCH timing parsers disagree on {probe_key}")
    utilization = _parse_utilization_report(report_text)
    lut_as_shift_register = _parse_lut_as_shift_register(report_text)
    lut_primitive_counts = _parse_lut_primitive_counts(report_text)
    reasons = {
        **timing["diagnostic_missing_reasons"],
        **utilization["diagnostic_missing_reasons"],
    }
    if lut_as_shift_register is None:
        reasons["lut_as_shift_register"] = "CLB Logic row absent"
    if lut_primitive_counts is None:
        reasons["lut_primitive_counts"] = "9. Primitives table absent"
    return {
        "timing": timing["timing"],
        "critical_path_features": timing["critical_path_features"],
        "route_vs_logic_delay_hint": timing["route_vs_logic_delay_hint"],
        "utilization_summary": utilization["utilization_summary"],
        "lut_as_shift_register": lut_as_shift_register,
        "lut_primitive_counts": lut_primitive_counts,
        "fanout": utilization["fanout"],
        "hardblock_resource_pressure": utilization["hardblock_resource_pressure"],
        "pblock_sizing_resources": parse_pblock_utilization_text(report_text),
        "diagnostic_missing_reasons": reasons,
    }



WPATH_SHA256 = "0215cf6d819878cc24841d4667a56120e6399ace93d739e9e3490d17ac53299a"
WPATH2_SHA256 = "341f9f813143e74e891d5c18692d2b1d2b8865feda990ce85b9ee7d55a6ee0b2"
WPATH_FLOAT_FIELDS = {
    "SLACK": "slack_ns",
    "REQUIREMENT": "requirement_ns",
    "DATAPATH_DELAY": "data_path_delay_ns",
    "DATAPATH_LOGIC_DELAY": "logic_delay_ns",
    "DATAPATH_NET_DELAY": "net_delay_ns",
    "SKEW": "clock_skew_ns",
}
WPATH_UTIL_FIELDS = {
    "UTIL_LUT": "luts",
    "UTIL_FF": "ffs",
    "UTIL_BRAM": "brams",
    "UTIL_DSP": "dsps",
    "UTIL_URAM": "urams",
}


def _wpath_number(raw: str, key: str) -> float | None:
    """Parse a value emitted by LATCH's Tcl, preserving a missing value as None."""

    if raw.strip().lower() in ("", "na"):
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise DcpParserError(f"{key} is not numeric in LATCH worst-path output") from exc
    if not math.isfinite(value):
        raise DcpParserError(f"{key} is nonfinite in LATCH worst-path output")
    return value


def _parse_wpath_records(text: str, variant: str) -> dict[str, Any]:
    """Adapt existing wpath*.tcl tab records; no Vivado command is generated here."""

    if variant not in ("wpath", "wpath2"):
        raise ValueError("unknown LATCH worst-path variant")
    if not text.strip():
        raise DcpParserError(f"{variant} output is empty")
    result: dict[str, Any] = {
        **{field: None for field in WPATH_FLOAT_FIELDS.values()},
        "logic_levels": None,
        "startpoint_pin": None,
        "endpoint_pin": None,
        "startpoint_clock": None,
        "path_cells": None,
        "tile_span": None,
        "pblocks": [],
        "primitive_cell_count": None,
        "utilization_counts": {},
        "unparsed_records": [],
    }
    seen: set[str] = set()
    for line in text.splitlines():
        fields = line.split("\t")
        key = fields[0]
        if key != "PBLOCK" and key in seen:
            raise DcpParserError(f"{variant} repeats {key}")
        seen.add(key)
        if key in WPATH_FLOAT_FIELDS:
            if len(fields) != 2:
                raise DcpParserError(f"{variant} malformed {key} row")
            result[WPATH_FLOAT_FIELDS[key]] = _wpath_number(fields[1], key)
        elif key in ("LOGIC_LEVELS", "N_CELLS"):
            if len(fields) != 2:
                raise DcpParserError(f"{variant} malformed {key} row")
            value = _wpath_number(fields[1], key)
            if value is not None and not value.is_integer():
                raise DcpParserError(f"{variant} noninteger {key}")
            result["logic_levels" if key == "LOGIC_LEVELS" else "primitive_cell_count"] = (
                int(value) if value is not None else None
            )
        elif key in ("STARTPOINT_PIN", "ENDPOINT_PIN", "STARTPOINT_CLOCK"):
            if len(fields) != 2:
                raise DcpParserError(f"{variant} malformed {key} row")
            result[key.lower()] = fields[1] or None
        elif key == "CELLS":
            if len(fields) != 3 or not fields[1].isdigit():
                raise DcpParserError(f"{variant} malformed CELLS row")
            tokens = fields[2].split()
            if len(tokens) % 2 or not all(n.isdigit() for n in tokens[1::2]):
                raise DcpParserError(f"{variant} malformed cell-composition dictionary")
            composition = {
                name: int(count) for name, count in zip(tokens[::2], tokens[1::2])
            }
            count = int(fields[1])
            if sum(composition.values()) != count:
                raise DcpParserError(f"{variant} cell composition does not sum to CELLS")
            result["path_cells"] = {"count": count, "composition": composition}
        elif key == "TILE_SPAN":
            if len(fields) not in (3, 5):
                raise DcpParserError(f"{variant} malformed TILE_SPAN row")
            x = re.fullmatch(r"X (\d+)-(\d+)", fields[1])
            y = re.fullmatch(r"Y (\d+)-(\d+)", fields[2])
            if x is None or y is None:
                raise DcpParserError(f"{variant} malformed tile coordinates")
            span: dict[str, int | None] = {
                "x_min": int(x.group(1)), "x_max": int(x.group(2)),
                "y_min": int(y.group(1)), "y_max": int(y.group(2)),
                "dx": None, "dy": None,
            }
            if len(fields) == 5:
                dx = re.fullmatch(r"dx (\d+)", fields[3])
                dy = re.fullmatch(r"dy (\d+)", fields[4])
                if dx is None or dy is None:
                    raise DcpParserError(f"{variant} malformed tile-span deltas")
                span["dx"], span["dy"] = int(dx.group(1)), int(dy.group(1))
            result["tile_span"] = span
        elif key == "PBLOCK":
            if len(fields) < 2:
                raise DcpParserError(f"{variant} malformed PBLOCK row")
            if fields[1] != "none":
                # UNTESTED with nonempty pblocks; retain LATCH's original fields.
                result["pblocks"].append({"name": fields[1], "detail_fields": fields[2:]})
        elif key in WPATH_UTIL_FIELDS:
            if len(fields) != 2:
                raise DcpParserError(f"{variant} malformed {key} row")
            value = _wpath_number(fields[1], key)
            result["utilization_counts"][WPATH_UTIL_FIELDS[key]] = (
                int(value) if value is not None and value.is_integer() else value
            )
        else:
            result["unparsed_records"].append(line)
    required = set(WPATH_FLOAT_FIELDS) | {
        "LOGIC_LEVELS", "STARTPOINT_PIN", "ENDPOINT_PIN", "CELLS", "PBLOCK"
    }
    required |= {"STARTPOINT_CLOCK", "N_CELLS"} if variant == "wpath" else set(WPATH_UTIL_FIELDS)
    absent = sorted(required - seen)
    if absent:
        raise DcpParserError(f"{variant} omitted expected records: {', '.join(absent)}")
    return result


def parse_wpath_report(text: str) -> dict[str, Any]:
    """Parse the exact tab output of LATCH paper_analysis/scripts/wpath.tcl:1-40."""

    return _parse_wpath_records(text, "wpath")


def parse_wpath2_report(text: str) -> dict[str, Any]:
    """Parse the exact tab output of LATCH paper_analysis/scripts/wpath2.tcl:1-40."""

    return _parse_wpath_records(text, "wpath2")


def _require_consistent_timing(
    probe: dict[str, Any], wpath: dict[str, Any], evidence: Evidence
) -> None:
    """Reject conflicting timing from three independent reads of one DCP."""

    checks = (
        ("wns_ns", probe["timing"]["wns_ns"], evidence.metric(MetricName.WNS_NS).value),
        ("tns_ns", probe["timing"]["tns_ns"], evidence.metric(MetricName.TNS_NS).value),
        ("worst_slack_ns", probe["timing"]["wns_ns"], wpath["slack_ns"]),
        ("data_path_delay_ns", probe["critical_path_features"]["data_path_delay_ns"], wpath["data_path_delay_ns"]),
        ("logic_delay_ns", probe["critical_path_features"]["logic_delay_ns"], wpath["logic_delay_ns"]),
        ("route_delay_ns", probe["critical_path_features"]["route_delay_ns"], wpath["net_delay_ns"]),
        ("logic_levels", probe["critical_path_features"]["max_logic_levels"], wpath["logic_levels"]),
    )
    for name, left, right in checks:
        if (
            left is None or right is None
            or not math.isclose(left, right, rel_tol=0, abs_tol=0.001)
        ):
            raise DcpParserError(f"timing sources disagree on {name}: {left} versus {right}")


class DcpTelemetryExtractor:
    """Run the copied LATCH probe under Runner and join independent meter Evidence."""

    def __init__(
        self,
        *,
        runner: Runner,
        env_script: Path,
        tool_version: str,
        meter: Meter | None = None,
        timeout_s: float = 1800,
    ) -> None:
        self.runner = runner
        self.meter = meter
        self.tool_version = tool_version
        self.timeout_s = timeout_s
        self.env_script = Path(env_script).expanduser().resolve(strict=True)
        if not self.env_script.is_relative_to(PROJECT_ROOT):
            raise DcpParserError("environment script must be inside this Development project")
        self.extractor_sha256 = _IMPORTED_EXTRACTOR_SHA256
        self.env_script_sha256 = _sha256_file(self.env_script)
        try:
            template_bytes = PROBE_TEMPLATE.read_bytes()
        except OSError as exc:
            raise DcpParserError("copied LATCH path probe Tcl is missing") from exc
        if hashlib.sha256(template_bytes).hexdigest() != PROBE_TEMPLATE_SHA256:
            raise DcpParserError("copied LATCH path probe Tcl has changed")
        self.probe_template = template_bytes.decode("utf-8")
        self._verify_sources()

    def _verify_sources(self) -> None:
        """Refuse a record if code, settings or the copied probe changed mid-run."""

        if _sha256_file(Path(__file__)) != self.extractor_sha256:
            raise DcpParserError("extractor Python changed during telemetry run")
        if _sha256_file(self.env_script) != self.env_script_sha256:
            raise DcpParserError("Vivado environment script changed during telemetry run")
        if _sha256_file(PROBE_TEMPLATE) != PROBE_TEMPLATE_SHA256:
            raise DcpParserError("copied LATCH path probe Tcl changed during telemetry run")

    def _launch(self, probe_dir: Path, script: Path, run_id: str) -> LaunchSpec:
        """Source the observed run-3 environment, then use LATCH's exact batch flags."""

        local_uv_cache = probe_dir / ".uv_cache"
        local_mpl_config = probe_dir / ".mplconfig"
        local_uv_cache.mkdir()
        local_mpl_config.mkdir()
        # LATCH real_analysis_provider.py:130-137 supplies the Vivado flags.
        # run-3 env.sh:10-13 supplies the observed Vivado settings script.
        # Keep caches here instead of in the read-only historical run directory.
        inner = (
            f". {shlex.quote(str(self.env_script))} && "
            f"export UV_CACHE_DIR={shlex.quote(str(local_uv_cache))} && "
            f"export MPLCONFIGDIR={shlex.quote(str(local_mpl_config))} && "
            "vivado -mode batch "
            f"-source {shlex.quote(str(script))} -nojournal "
            f"-log {shlex.quote(str(probe_dir / 'probe.log'))}"
        )
        return LaunchSpec(
            argv=("/bin/bash", "-c", inner),
            cwd=str(probe_dir),
            env_overrides={
                "UV_CACHE_DIR": str(local_uv_cache),
                "MPLCONFIGDIR": str(local_mpl_config),
            },
            timeout_s=self.timeout_s,
            script_sha256=_sha256_file(script),
            tool_versions={"vivado": self.tool_version},
        )

    def _run_wpath(
        self,
        *,
        variant: str,
        private: Path,
        root: Path,
        correlation_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Execute a byte-identical LATCH worst-path script under the shared runner."""

        expected = WPATH_SHA256 if variant == "wpath" else WPATH2_SHA256
        support = PROJECT_ROOT / "recipes" / "support" / f"{variant}.tcl"
        if not support.is_file() or _sha256_file(support) != expected:
            raise DcpParserError(f"copied LATCH {variant}.tcl hash changed")
        work = root / variant
        work.mkdir()
        copied = work / f"{variant}.tcl"
        shutil.copyfile(support, copied)
        if _sha256_file(copied) != expected:
            raise DcpParserError(f"private {variant}.tcl copy changed")
        output = work / f"{variant}.tsv"
        local_uv_cache = work / ".uv_cache"
        local_mpl_config = work / ".mplconfig"
        local_uv_cache.mkdir()
        local_mpl_config.mkdir()
        # LATCH paper_analysis/scripts/wpath*.tcl:2 gives the exact two
        # -tclargs; all Vivado-facing Tcl remains byte-identical to LATCH.
        inner = (
            f". {shlex.quote(str(self.env_script))} && "
            f"export UV_CACHE_DIR={shlex.quote(str(local_uv_cache))} && "
            f"export MPLCONFIGDIR={shlex.quote(str(local_mpl_config))} && "
            "vivado -mode batch -nojournal -log vivado.log "
            f"-source {shlex.quote(str(copied))} "
            f"-tclargs {shlex.quote(str(private))} {shlex.quote(str(output))}"
        )
        process = self.runner.run(
            LaunchSpec(
                argv=("/bin/bash", "-c", inner),
                cwd=str(work),
                env_overrides={
                    "UV_CACHE_DIR": str(local_uv_cache),
                    "MPLCONFIGDIR": str(local_mpl_config),
                },
                timeout_s=self.timeout_s,
                script_sha256=expected,
                tool_versions={"vivado": self.tool_version},
            ),
            run_id=f"{correlation_id}-{variant}",
            run_dir=work,
        )
        if process.status != RunStatus.OK:
            raise DcpParserError(f"LATCH {variant} probe ended {process.status}")
        try:
            raw = output.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise DcpParserError(f"LATCH {variant} probe made no output") from exc
        parsed = (
            parse_wpath_report(raw) if variant == "wpath"
            else parse_wpath2_report(raw)
        )
        return parsed, {
            "source_relative_path": f"paper_analysis/scripts/{variant}.tcl",
            "source_tcl_sha256": expected,
            "copied_script": str(copied),
            "output_path": str(output),
            "output_sha256": _sha256_file(output),
            "runner": process.model_dump(mode="json"),
        }

    @staticmethod
    def _stage_input(source: Path, run_dir: Path, expected_sha256: str) -> Path:
        """Copy the shared DCP into a private path, then verify the exact bytes."""

        input_dir = run_dir / "input"
        input_dir.mkdir()
        target = input_dir / "input.dcp"
        temporary = input_dir / f".input.{uuid.uuid4().hex}.tmp"
        try:
            shutil.copyfile(source, temporary)
            if _sha256_file(temporary) != expected_sha256:
                raise DcpParserError("DCP changed while being copied")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def extract(
        self,
        dcp_path: str | Path,
        *,
        run_dir: Path,
        evidence: Evidence | None = None,
        design: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Return one traceable record for a DCP; the probe never modifies its input."""

        if evidence is None and self.meter is None:
            raise DcpParserError("full telemetry needs independent meter Evidence")
        self._verify_sources()
        # A supplied Evidence record is a software path; its real-tool origin
        # remains UNTESTED here unless the caller provides a completed meter run.
        evidence_origin = "supplied" if evidence is not None else "fresh_meter"
        source = Path(dcp_path).expanduser().resolve(strict=True)
        root = Path(run_dir).expanduser().resolve()
        if not root.is_relative_to(PROJECT_ROOT / "data"):
            raise DcpParserError("telemetry artifacts must stay inside this project data folder")
        _safe_tcl_path(root)
        root.mkdir(parents=True, exist_ok=True)
        before = _sha256_file(source)
        private = self._stage_input(source, root, before)
        probe_dir = root / "path_probe"
        probe_dir.mkdir()
        _safe_tcl_path(private)
        _safe_tcl_path(probe_dir)
        rendered = self.probe_template.format(dcp=private, out=probe_dir)
        script = probe_dir / "probe.tcl"
        with script.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        correlation_id = run_id or f"dcp-probe-{before[:12]}-{uuid.uuid4().hex[:10]}"
        process = self.runner.run(
            self._launch(probe_dir, script, correlation_id),
            run_id=correlation_id,
            run_dir=probe_dir,
        )
        if process.status != RunStatus.OK:
            raise DcpParserError(f"LATCH probe {correlation_id} ended {process.status}")
        report_path = probe_dir / "probe.rpt"
        try:
            report_text = report_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise DcpParserError(f"LATCH probe made no report: {report_path}") from exc
        if not report_text.strip():
            raise DcpParserError(f"LATCH probe report is empty: {report_path}")
        version = re.search(r"Tool Version\s*:\s*Vivado v\.([^\s|]+)", report_text)
        if version is None or version.group(1) != self.tool_version:
            raise DcpParserError("probe Vivado report version differs from configured tool")
        parsed = parse_probe_report(report_text)
        if parsed["timing"]["wns_ns"] is None or not parsed["utilization_summary"]:
            raise DcpParserError("probe lacks core timing or utilization facts")
        wpath, wpath_provenance = self._run_wpath(
            variant="wpath", private=private, root=root, correlation_id=correlation_id
        )
        wpath2, wpath2_provenance = self._run_wpath(
            variant="wpath2", private=private, root=root, correlation_id=correlation_id
        )
        # Both unchanged LATCH scripts inspect the same private checkpoint.
        for key in (
            "slack_ns", "requirement_ns", "data_path_delay_ns",
            "logic_delay_ns", "net_delay_ns", "logic_levels", "clock_skew_ns",
            "startpoint_pin", "endpoint_pin", "path_cells",
        ):
            if wpath[key] != wpath2[key]:
                raise DcpParserError(f"LATCH worst-path probes disagree on {key}")
        for key in ("x_min", "x_max", "y_min", "y_max"):
            if wpath["tile_span"] and wpath2["tile_span"]:
                if wpath["tile_span"][key] != wpath2["tile_span"][key]:
                    raise DcpParserError(f"LATCH worst-path probes disagree on {key}")
        state = DesignState(
            id=before,
            design=design or source.stem,
            stage="unknown",
            path=str(private),
            fake=False,
        )
        if evidence is None:
            assert self.meter is not None
            evidence = self.meter.measure(state, run_dir=root)
        if evidence.state_id != before or evidence.fake:
            raise DcpParserError("meter Evidence does not describe this real DCP")
        if self.meter is not None and evidence.meter_version != self.meter.version:
            raise DcpParserError("supplied Evidence uses a different meter version")
        required = (
            MetricName.WNS_NS,
            MetricName.TNS_NS,
            MetricName.PACKED_LUTS,
            MetricName.NETLIST_LUTS,
            MetricName.FFS,
            MetricName.DYNAMIC_POWER_W,
        )
        if any(evidence.metric(name) is None for name in required):
            raise DcpParserError("meter Evidence lacks timing, area, or power")
        # All three timing methods inspect one exact checkpoint.
        _require_consistent_timing(parsed, wpath, evidence)
        # wpath2's direct resource census must agree with the independent
        # meter and the separately parsed LATCH utilization report.
        resource_checks = (
            ("luts", MetricName.PACKED_LUTS, "CLB LUTs"),
            ("ffs", MetricName.FFS, "CLB Registers"),
        )
        for key, metric_name, row_name in resource_checks:
            observed = wpath2["utilization_counts"].get(key)
            row = parsed["utilization_summary"].get(row_name)
            if observed is not None and (
                row is None
                or observed != evidence.metric(metric_name).value
                or observed != row["used"]
            ):
                raise DcpParserError(f"LATCH {key} census differs from probe or meter")
        for key, row_name in (
            ("brams", "Block RAM Tile"), ("dsps", "DSPs"), ("urams", "URAM")
        ):
            observed = wpath2["utilization_counts"].get(key)
            row = parsed["utilization_summary"].get(row_name)
            if observed is not None and (row is None or observed != row["used"]):
                raise DcpParserError(f"LATCH {key} census differs between probes")
        after = _sha256_file(source)
        private_after = _sha256_file(private)
        if before != after or before != private_after:
            raise DcpParserError("input DCP changed during telemetry extraction")
        payload = {
            "schema_version": "dcp-telemetry-v1",
            "fake": False,
            "state_sha256": before,
            "probe": parsed,
            "wpath": wpath,
            "wpath2": wpath2,
            "metrics": {
                name.value: value.model_dump(mode="json")
                for name, value in evidence.metrics.items()
            },
            "validity": {
                "routed": evidence.routed,
                "route_errors": evidence.route_errors,
                "drc_errors": evidence.drc_errors,
                "constraint_fingerprint": evidence.constraint_fingerprint,
                "clock_digest": evidence.clock_digest,
                "v_props": evidence.v_props,
            },
            "provenance": {
                "source_dcp": str(source),
                "input_copy": str(private),
                "input_sha256_before": before,
                "input_sha256_after": after,
                "private_input_sha256_after": private_after,
                "latch_commit": LATCH_COMMIT,
                "extractor_sha256": self.extractor_sha256,
                "env_script_sha256": self.env_script_sha256,
                "evidence_origin": evidence_origin,
                "latch_report_parser_sha256": LATCH_REPORT_PARSER_SHA256,
                "latch_util_parser_sha256": LATCH_UTIL_PARSER_SHA256,
                "latch_timing_parser_sha256": LATCH_TIMING_PARSER_SHA256,
                "probe_template_sha256": PROBE_TEMPLATE_SHA256,
                "rendered_probe_sha256": _sha256_file(script),
                "probe_report_sha256": _sha256_file(report_path),
                "runner": process.model_dump(mode="json"),
                "wpath": wpath_provenance,
                "wpath2": wpath2_provenance,
                "evidence_id": evidence.id,
                "meter_version": evidence.meter_version,
                "meter_report_path": evidence.raw_report_path,
                "vivado_version": self.tool_version,
                "contract_run_id": correlation_id,
            },
        }
        self._verify_sources()
        _atomic_json(root / "dcp_telemetry.json", payload)
        return payload


def extract_dcp_telemetry(
    dcp_path: str | Path,
    *,
    run_dir: Path | None = None,
    runner: Runner | None = None,
    meter: Meter | None = None,
    evidence: Evidence | None = None,
    env_script: Path | None = None,
    tool_version: str = "2025.1",
    timeout_s: float = 1800,
    design: str | None = None,
) -> dict[str, Any]:
    """Convenient entry point using the project's shared slots and copied environment."""

    from orch.engine.meter import VivadoMeter
    from orch.engine.runner import Runner as ProcessRunner

    location = run_dir or PROJECT_ROOT / "data" / "dcp_telemetry" / uuid.uuid4().hex
    shared_runner = runner or ProcessRunner(PROJECT_ROOT / "data" / "slots")
    script = env_script or PROJECT_ROOT / "recipes" / "support" / "env.sh"
    measured_by = meter
    if measured_by is None and evidence is None:
        measured_by = VivadoMeter(
            runner=shared_runner,
            env_script=script,
            tool_version=tool_version,
        )
    extractor = DcpTelemetryExtractor(
        runner=shared_runner,
        env_script=script,
        tool_version=tool_version,
        meter=measured_by,
        timeout_s=timeout_s,
    )
    return extractor.extract(dcp_path, run_dir=location, evidence=evidence, design=design)
