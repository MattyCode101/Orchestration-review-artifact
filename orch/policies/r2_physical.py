"""Round-2 physical description of saved states (Wednesday-night note, section 3, principle 4).

The start state is described in full; every other state only by what moved meaningfully against the
start. Fields come from the telemetry the existing extraction already produces for a state (record
schema dcp-telemetry-v1: worst setup path, utilisation probe, fan-out, vectorless power); no field
needs a new Tcl script. They are grouped timing, utilisation and power, and each group can be shown
alone. This module is pure: it reads a record that is already a dict and writes text.

What a "meaningful" change is (declared before any run, not tuned): a field must move by at least its
absolute step and, where a relative step is given, by at least that share of its start value.
Utilisation percentages of the device are context for the start only (the device is large, so they
barely move); changes are shown in counts. Power uses the category against the design's recorded
power noise band when there is one, otherwise a 5% move.

Wanted but not available without new Tcl (listed in notes/feature_ablation.md): congestion by
direction, power by component, control sets, clock-region occupancy.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from orch.policies.llm_view import categorise

GROUPS = ("timing", "utilisation", "power")
GROUP_PHRASE = {"timing": "worst-path timing", "utilisation": "utilisation and fan-out", "power": "dynamic power"}
POWER_REL_WITHOUT_BAND = 0.05


@dataclass(frozen=True)
class FieldSpec:
    name: str
    group: str
    label: str
    kind: str                 # int | ns3 | ns1 | pct | w3 | str
    rel: float = 0.0          # minimum relative move against the start value (0: no relative test)
    abs: float = 0.0          # minimum absolute move
    compare: bool = True      # False: shown for the start only
    pct_field: str | None = None    # start only: append this percentage of the device


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("failing_endpoints", "timing", "failing endpoints", "int", rel=0.10, abs=10),
    FieldSpec("tns_ns", "timing", "total negative slack", "ns1", rel=0.10, abs=1.0),
    FieldSpec("logic_levels", "timing", "worst-path logic levels", "int", abs=1),
    FieldSpec("path_delay_ns", "timing", "worst-path delay", "ns3", rel=0.10, abs=0.05),
    FieldSpec("route_share_pct", "timing", "worst-path routing share", "pct", abs=5),
    FieldSpec("path_type", "timing", "worst-path type", "str"),
    FieldSpec("tile_span", "timing", "worst-path span in tiles", "int", rel=0.20, abs=3),
    FieldSpec("clb_luts", "utilisation", "CLB LUTs", "int", compare=False, pct_field="clb_luts_pct"),
    FieldSpec("clb_regs", "utilisation", "CLB registers", "int", rel=0.05, abs=50, pct_field="clb_regs_pct"),
    FieldSpec("bram_tiles", "utilisation", "BRAM tiles", "int", abs=1),
    FieldSpec("dsps", "utilisation", "DSPs", "int", abs=1),
    FieldSpec("urams", "utilisation", "URAMs", "int", abs=1),
    FieldSpec("small_lut_share_pct", "utilisation", "share of small LUTs (LUT1 to LUT3)", "pct", abs=5),
    FieldSpec("srl_luts", "utilisation", "shift-register LUTs", "int", rel=0.10, abs=10),
    FieldSpec("max_fanout", "utilisation", "maximum net fan-out", "int", rel=0.20, abs=50),
    FieldSpec("high_fanout_nets", "utilisation", "high-fan-out nets", "int", rel=0.20, abs=2),
    FieldSpec("dynamic_power_w", "power", "dynamic power", "w3"),
)
BY_GROUP = {g: tuple(f for f in FIELDS if f.group == g) for g in GROUPS}


def _num(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _path(record: Mapping[str, Any], *keys: str) -> Any:
    x: Any = record
    for key in keys:
        x = x.get(key) if isinstance(x, Mapping) else None
    return x


def facts_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """The description fields of one telemetry record (power is added from the view's metrics)."""

    delay, net = _num(_path(record, "wpath", "data_path_delay_ns")), _num(_path(record, "wpath", "net_delay_ns"))
    tile = _path(record, "wpath", "tile_span") or {}
    dx, dy = _num(tile.get("dx")), _num(tile.get("dy"))
    if dx is None or dy is None:
        corners = [_num(tile.get(k)) for k in ("x_max", "x_min", "y_max", "y_min")]
        dx, dy = (None, None) if None in corners else (corners[0] - corners[1], corners[2] - corners[3])
    lut_counts = _path(record, "probe", "lut_primitive_counts") or {}
    sizes = [_num(lut_counts.get(f"LUT{i}")) for i in range(1, 7)]
    small_share = None
    if None not in sizes and sum(sizes) > 0:
        small_share = 100.0 * sum(sizes[:3]) / sum(sizes)
    util = _path(record, "probe", "utilization_summary") or {}

    def used(name: str) -> float | None:
        return _num(_path(util, name, "used"))

    def pct(name: str) -> float | None:
        return _num(_path(util, name, "utilization_pct"))

    return {
        "failing_endpoints": _num(_path(record, "probe", "timing", "failing_endpoints")),
        "tns_ns": _num(_path(record, "metrics", "tns_ns", "value")),
        "logic_levels": _num(_path(record, "wpath", "logic_levels")),
        "path_delay_ns": delay,
        "route_share_pct": 100.0 * net / delay if (delay and net is not None and delay > 0) else None,
        "path_type": _path(record, "probe", "critical_path_features", "dominant_path_type"),
        "tile_span": None if dx is None or dy is None else dx + dy,
        "clb_luts": used("CLB LUTs"), "clb_luts_pct": pct("CLB LUTs"),
        "clb_regs": used("CLB Registers"), "clb_regs_pct": pct("CLB Registers"),
        "bram_tiles": used("Block RAM Tile"), "dsps": used("DSPs"), "urams": used("URAM"),
        "small_lut_share_pct": small_share,
        "srl_luts": _num(_path(record, "probe", "lut_as_shift_register")),
        "max_fanout": _num(_path(record, "probe", "fanout", "max_fanout")),
        "high_fanout_nets": _num(_path(record, "probe", "fanout", "num_high_fanout_nets")),
    }


def fmt(spec: FieldSpec, value: Any) -> str:
    if spec.kind == "str":
        return str(value)
    v = float(value)
    return {
        "int": f"{v:.0f}", "ns3": f"{v:.3f} ns", "ns1": f"{v:.1f} ns", "pct": f"{v:.0f}%", "w3": f"{v:.3f} W",
    }[spec.kind]


def meaningful(spec: FieldSpec, old: Any, new: Any) -> bool:
    if not spec.compare or old is None or new is None:
        return False
    if spec.kind == "str":
        return old != new
    delta = abs(float(new) - float(old))
    if delta <= 0.0 or delta < spec.abs - 1e-12:
        return False
    if spec.rel > 0.0:
        base = abs(float(old))
        return base == 0.0 or delta / base >= spec.rel - 1e-12
    return True


def start_text(facts: Mapping[str, Any], group: str) -> str:
    parts = []
    for spec in BY_GROUP[group]:
        value = facts.get(spec.name)
        if value is None:
            continue
        text = f"{spec.label} {fmt(spec, value)}"
        if spec.pct_field and facts.get(spec.pct_field) is not None:
            text += f" ({float(facts[spec.pct_field]):.1f}% of the device)"
        parts.append(text)
    return "; ".join(parts)


def change_text(start: Mapping[str, Any], facts: Mapping[str, Any], group: str, power_band: float | None) -> str:
    parts = []
    for spec in BY_GROUP[group]:
        old, new = start.get(spec.name), facts.get(spec.name)
        if spec.name == "dynamic_power_w":
            if old is None or new is None:
                continue
            if power_band:
                category = categorise(float(new) - float(old), power_band, higher_is_better=False)
                if category == "within_noise":
                    continue
                parts.append(f"{spec.label} {fmt(spec, old)} -> {fmt(spec, new)} ({category})")
            elif old and abs(float(new) - float(old)) / abs(float(old)) >= POWER_REL_WITHOUT_BAND:
                parts.append(f"{spec.label} {fmt(spec, old)} -> {fmt(spec, new)}")
            continue
        if meaningful(spec, old, new):
            parts.append(f"{spec.label} {fmt(spec, old)} -> {fmt(spec, new)}")
    return "; ".join(parts)


def render_physical_section(
    entries: Sequence[tuple[str, bool, Mapping[str, Any]]], groups: Sequence[str], *, power_band: float | None = None,
) -> str:
    """entries: (alias, is_start, facts) in alias order; the text of the section without outer blank lines."""

    shown = [g for g in GROUPS if g in groups]
    start = next(facts for _, is_start, facts in entries if is_start)
    lines = [
        "Physical description of the saved states",
        "Measured facts about each saved state: the start in full, every other state only where it differs "
        "meaningfully from the start. They describe what a state looks like, not what a worker will do with "
        "it. Fields shown: " + ", ".join(GROUP_PHRASE[g] for g in shown) + ".",
    ]
    for alias, is_start, facts in entries:
        lines.append(f"{alias} (start)" if is_start else alias)
        if is_start:
            lines += [f"  {g}: {body}" for g in shown if (body := start_text(facts, g))]
            continue
        body_lines = [f"  {g}: {body}" for g in shown if (body := change_text(start, facts, g, power_band))]
        lines += body_lines or ["  no meaningful change from the start in these fields"]
    return "\n".join(lines)
