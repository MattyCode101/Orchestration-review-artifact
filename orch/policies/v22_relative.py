"""Template v2.2, switch R: relative context (new file).

Each metric is also stated against an anchor the model can judge:
  WNS         as a share of the clock requirement of the worst path at the start
              (stored: record["wpath"]["requirement_ns"], the setup requirement of the start's worst path)
  packed LUTs as a share of the device (stored: record["probe"]["utilization_summary"]["CLB LUTs"]["available"]) and
              as a share of the start (the view's own metrics)
  dynamic power as a share of the start (the view's own metrics)

Only the start's telemetry record is read, so the section costs no extraction for the other saved states. When a record
has no requirement the WNS anchor is left out and the intro says so (the generic wording, no per-design text).
Pure functions; the record is a dict already.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from orch.contract.models import MetricName
from orch.contract.views import PolicyView
from orch.policies.llm_view import METRIC_NAMES
from orch.policies.r2_view import value_text
from orch.policies.v2_view import unique_states


@dataclass(frozen=True)
class Anchors:
    period_ns: float | None        # clock requirement of the start's worst path
    device_luts: float | None      # CLB LUTs available on the device


def _num(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def anchors_from_record(record: Mapping[str, Any]) -> Anchors:
    period = _num((record.get("wpath") or {}).get("requirement_ns"))
    util = ((record.get("probe") or {}).get("utilization_summary") or {}).get("CLB LUTs") or {}
    available = _num(util.get("available"))
    return Anchors(period if period and period > 0 else None, available if available and available > 0 else None)


def _share(value: float, reference: float) -> str:
    return f"{100.0 * value / reference:.1f}%"


def render_relative(view: PolicyView, aliases: Mapping[str, str], anchors: Anchors, *, header: str,
                    intro_all: str, intro_no_period: str) -> str:
    """The section text without the blank lines the prompt puts around it."""

    start = view.start
    intro = intro_all.format(period=f"{anchors.period_ns:.3f}") if anchors.period_ns else intro_no_period
    lines = [header, intro]
    for state in unique_states(view):
        parts = []
        wns = state.metrics.get(MetricName.WNS_NS)
        if wns is not None:
            text = f"{METRIC_NAMES[MetricName.WNS_NS]} {value_text(MetricName.WNS_NS, wns)}"
            if anchors.period_ns:
                text += f" ({_share(wns, anchors.period_ns)} of the clock requirement)"
            parts.append(text)
        luts, start_luts = state.metrics.get(MetricName.PACKED_LUTS), start.metrics.get(MetricName.PACKED_LUTS)
        if luts is not None:
            shares = []
            if anchors.device_luts:
                shares.append(f"{_share(luts, anchors.device_luts)} of the device")
            if start_luts:
                shares.append(f"{_share(luts, start_luts)} of the start")
            parts.append(f"{METRIC_NAMES[MetricName.PACKED_LUTS]} {value_text(MetricName.PACKED_LUTS, luts)}"
                         + (f" ({', '.join(shares)})" if shares else ""))
        power, start_power = state.metrics.get(MetricName.DYNAMIC_POWER_W), start.metrics.get(MetricName.DYNAMIC_POWER_W)
        if power is not None:
            text = f"{METRIC_NAMES[MetricName.DYNAMIC_POWER_W]} {value_text(MetricName.DYNAMIC_POWER_W, power)}"
            if start_power:
                text += f" ({_share(power, start_power)} of the start)"
            parts.append(text)
        lines.append(f"{aliases[state.state_id]}: " + "; ".join(parts))
    return "\n".join(lines)
