"""Template v2 physical description (prompt-improvement changes 16 to 18; build-review defect 8).

orch/policies/r2_physical.py stays byte-identical (the round-2 record). Package P3 renders instead:

  change 17  every saved state other than the start is described against the start AND against its parent,
             so a repair choice can see what the last call changed;
  change 18  a compact set of stable, design-wide fields (failing endpoints, total negative slack, registers,
             shift-register LUTs, block RAM, DSPs, fan-out, dynamic power). The two worst-path fields kept
             (path delay and logic levels) are shown for the start and otherwise only on a larger change, because
             the worst path itself changes between runs (defect 8: levels and tile span used to appear for about
             half of all states). Tile span, routing share and path type are dropped.
  change 16  the purpose sentence comes from the P3 package file, not from this module.

Without P3, the round-2 section is used (r2_physical.render_physical_section), with the v2 category words
when the P1 vocabulary is on (`render_r2_section`). Pure functions; the facts come from
r2_physical.facts_from_record plus the state's dynamic power.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from orch.policies import r2_physical as r2
from orch.policies.v2_vocab import categorise

HEADER = "Physical description of the saved states"


@dataclass(frozen=True)
class FieldV2:
    name: str
    group: str
    label: str
    kind: str                 # int | ns3 | ns1 | w3
    rel: float = 0.0          # minimum relative move against the reference value
    abs: float = 0.0          # minimum absolute move
    start_only: bool = False
    pct_field: str | None = None
    worst_path: bool = False


FIELDS_V2: tuple[FieldV2, ...] = (
    FieldV2("failing_endpoints", "timing", "failing endpoints", "int", rel=0.10, abs=10),
    FieldV2("tns_ns", "timing", "total negative slack", "ns1", rel=0.10, abs=1.0),
    FieldV2("path_delay_ns", "timing", "worst-path delay", "ns3", rel=0.20, abs=0.10, worst_path=True),
    FieldV2("logic_levels", "timing", "worst-path logic levels", "int", rel=0.25, abs=2, worst_path=True),
    FieldV2("clb_luts", "utilisation", "CLB LUTs", "int", start_only=True, pct_field="clb_luts_pct"),
    FieldV2("clb_regs", "utilisation", "CLB registers", "int", rel=0.05, abs=50, pct_field="clb_regs_pct"),
    FieldV2("srl_luts", "utilisation", "shift-register LUTs", "int", rel=0.10, abs=10),
    FieldV2("bram_tiles", "utilisation", "BRAM tiles", "int", abs=1),
    FieldV2("dsps", "utilisation", "DSPs", "int", abs=1),
    FieldV2("urams", "utilisation", "URAMs", "int", abs=1),
    FieldV2("max_fanout", "utilisation", "maximum net fan-out", "int", rel=0.20, abs=50),
    FieldV2("high_fanout_nets", "utilisation", "high-fan-out nets", "int", rel=0.20, abs=2),
    FieldV2("dynamic_power_w", "power", "dynamic power", "w3"),
)
GROUPS = ("timing", "utilisation", "power")


def fmt(spec: FieldV2, value: Any) -> str:
    v = float(value)
    return {"int": f"{v:.0f}", "ns3": f"{v:.3f} ns", "ns1": f"{v:.1f} ns", "w3": f"{v:.3f} W"}[spec.kind]


def meaningful(spec: FieldV2, old: Any, new: Any) -> bool:
    if spec.start_only or old is None or new is None:
        return False
    delta = abs(float(new) - float(old))
    if delta <= 0.0 or delta < spec.abs - 1e-12:
        return False
    if spec.rel > 0.0:
        base = abs(float(old))
        return base == 0.0 or delta / base >= spec.rel - 1e-12
    return True


def start_parts(facts: Mapping[str, Any], group: str) -> list[str]:
    parts = []
    for spec in FIELDS_V2:
        if spec.group != group:
            continue
        value = facts.get(spec.name)
        if value is None:
            continue
        if spec.name in ("bram_tiles", "dsps", "urams") and float(value) == 0.0:
            continue
        text = f"{spec.label} {fmt(spec, value)}"
        if spec.pct_field and facts.get(spec.pct_field) is not None:
            text += f" ({float(facts[spec.pct_field]):.1f}% of the device)"
        parts.append(text)
    return parts


def change_parts(ref: Mapping[str, Any], facts: Mapping[str, Any], *, power_band: float | None, categories: bool,
                 v2_vocab: bool) -> list[str]:
    parts = []
    for spec in FIELDS_V2:
        old, new = ref.get(spec.name), facts.get(spec.name)
        if spec.name == "dynamic_power_w":
            if old is None or new is None:
                continue
            if power_band:
                category = categorise(float(new) - float(old), power_band, higher_is_better=False, v2=v2_vocab)
                if category in ("within noise", "within_noise"):
                    continue
                suffix = f" ({category})" if categories else ""
                parts.append(f"{spec.label} {fmt(spec, old)} -> {fmt(spec, new)}{suffix}")
            elif old and abs(float(new) - float(old)) / abs(float(old)) >= r2.POWER_REL_WITHOUT_BAND:
                parts.append(f"{spec.label} {fmt(spec, old)} -> {fmt(spec, new)}")
            continue
        if meaningful(spec, old, new):
            parts.append(f"{spec.label} {fmt(spec, old)} -> {fmt(spec, new)}")
    return parts


def render_p3_section(
    entries: Sequence[tuple[str, str | None, Mapping[str, Any]]], *, intro: str, power_band: float | None,
    categories: bool = True, v2_vocab: bool = True,
) -> str:
    """entries: (alias, parent alias or None for the start, facts) in alias order."""

    by_alias = {alias: facts for alias, _, facts in entries}
    start = next(facts for _, parent, facts in entries if parent is None)
    lines = [HEADER, intro]
    for alias, parent, facts in entries:
        if parent is None:
            lines.append(f"{alias} (start)")
            lines += [f"  {g}: " + "; ".join(p) for g in GROUPS if (p := start_parts(facts, g))]
            continue
        lines.append(f"{alias} (parent {parent})")
        vs_start = change_parts(start, facts, power_band=power_band, categories=categories, v2_vocab=v2_vocab)
        lines.append("  vs start: " + ("; ".join(vs_start) or "no meaningful change"))
        if parent == "S0" or parent not in by_alias:
            lines.append("  vs parent: same as vs start" if parent == "S0" else "  vs parent: parent not described")
            continue
        vs_parent = change_parts(by_alias[parent], facts, power_band=power_band, categories=categories,
                                 v2_vocab=v2_vocab)
        lines.append("  vs parent: " + ("; ".join(vs_parent) or "no meaningful change"))
    return "\n".join(lines)


def render_r2_section(entries: Sequence[tuple[str, bool, Mapping[str, Any]]], groups: Sequence[str], *,
                      power_band: float | None, v2_vocab: bool, categories: bool = True) -> str:
    """The round-2 section; byte-identical to r2_physical when the round-1 vocabulary is in use."""

    if not v2_vocab and categories:
        return r2.render_physical_section(entries, groups, power_band=power_band)
    text = r2.render_physical_section(entries, groups, power_band=None)
    if not power_band:
        return text
    # only the power clause depends on the vocabulary: re-render it with the v2 words (or without a category)
    start = next(facts for _, is_start, facts in entries if is_start)
    out = r2.render_physical_section(entries, groups, power_band=power_band)
    for alias, is_start, facts in entries:
        old, new = start.get("dynamic_power_w"), facts.get("dynamic_power_w")
        if is_start or old is None or new is None:
            continue
        r1_cat = categorise(float(new) - float(old), power_band, higher_is_better=False, v2=False)
        if r1_cat == "within_noise":
            continue
        v2_cat = categorise(float(new) - float(old), power_band, higher_is_better=False, v2=True)
        r1_text = f"dynamic power {float(old):.3f} W -> {float(new):.3f} W ({r1_cat})"
        new_text = f"dynamic power {float(old):.3f} W -> {float(new):.3f} W" + (f" ({v2_cat})" if categories else "")
        out = out.replace(r1_text, new_text)
    return out
