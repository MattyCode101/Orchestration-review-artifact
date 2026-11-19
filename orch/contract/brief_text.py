"""Frozen brief loader and shared natural-language LLM rendering.

The source YAML is byte-pinned. T0, A0, T and A were frozen on 2026-11-05
(YAML SHA-256 9087784c...). P0 (minimise dynamic power; WNS >= start and packed
LUTs <= start) was added on 2026-11-06 as a dated addition with a new YAML
hash; the four earlier briefs are additionally pinned by contract fingerprint,
so an edit to any of them is refused. The rendered text is identical for the
LLM orchestrator and the general-agent harness; no policy-specific prompt text
is introduced here. The existing contract uses packed_luts for the YAML's
packed_clb_luts, the independently metered packed CLB LUT count.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from orch.contract.models import Brief, BriefTerm, MetricName, Objective

PROJECT = Path(__file__).resolve().parents[2]
BRIEFS_PATH = PROJECT / "configs/briefs.yaml"
FROZEN_BRIEFS_SHA256_2026_11_05 = (
    "9087784cf612d00b73b5535b98469df16dba7425df530b892e47bccdf2de4894"
)
FROZEN_BRIEFS_SHA256 = "9d4ae92f3dd4ccaff322b23a700196cf85aab1e5c4e83cfaa916fae9df39d878"
FROZEN_BRIEF_IDS = ("T0", "A0", "T", "A", "P0")
FOUR_BRIEF_FINGERPRINTS = {
    "T0": "8ab82885509af6ff2cc9c499852960af43a3186df744c9c2b75d5cce31ec61d7",
    "A0": "257f95b1b5aec8cfb5112c6f66daf96d6f9b9b682c18decb53d24d0f47e16705",
    "T": "b8207134d3e9fbb8181306ef708dd59fc499df5303d60a2c833c3add5f538a8c",
    "A": "2af97a18418d48bf65db59ab0384d37025ab95740c03b329de26d90c194e15ac",
}

METRICS = {
    "wns_ns": MetricName.WNS_NS,
    "packed_clb_luts": MetricName.PACKED_LUTS,
    "dynamic_power_w": MetricName.DYNAMIC_POWER_W,
}
DIRECTIONS = {"maximise": "max", "minimise": "min"}
LIMITS = {
    "<= start": ("<=", 1.0, 0.0),
    ">= start": (">=", 1.0, 0.0),
    "<= 1.01 * start": ("<=", 1.01, 0.0),
    ">= start - 0.100": (">=", 1.0, -0.1),
}
LIMIT_TEXT = {
    "<= start": "must not exceed its B2 value",
    ">= start": "must not fall below its B2 value",
    "<= 1.01 * start": "must not exceed 1.01 times its B2 value",
    ">= start - 0.100": "must not fall more than 0.100 ns below its B2 value",
}
METRIC_TEXT = {
    "wns_ns": "worst negative slack (WNS, ns)",
    "packed_clb_luts": "packed CLB LUT count",
    "dynamic_power_w": "dynamic power (W, Vivado vectorless estimate)",
}

TEMPLATE = (
    "Brief {brief_id} ({role}).\n"
    "Start from the B2 design checkpoint. Budget: at most {runs} recipe runs.\n"
    "Objective: {direction} {objective_metric}.\n"
    "Hard limit: {limit_metric} {limit_text}; compare against the B2 start measurement.\n"
    "Final pick: {final_pick}.\n"
    "Validity: fully routed; zero route and DRC errors; matching B2 constraint "
    "fingerprint and clock digest; no leftover DONT_TOUCH or fixed LOC/BEL; "
    "AMD simulation PASS.\n"
    "Reporting: timing gain as Fmax percent and area gain as packed CLB LUT-cut "
    "percent; also TNS, netlist LUT cells, and FFs.\n"
    "Noise: flag only; never use it to accept or reject.\n"
)
FROZEN_TEMPLATE_SHA256 = "e7bab972808d532aefc7a215da94a9da2d8bb6b37800a182cf7c06d66f959eba"
TEMPLATE_SHA256 = hashlib.sha256(TEMPLATE.encode("utf-8")).hexdigest()
if TEMPLATE_SHA256 != FROZEN_TEMPLATE_SHA256:
    raise ValueError("brief text template differs from the 2026-11-05 freeze")

# Dated 2026-11-06 variant for briefs with several hard limits (P0). The
# single-limit TEMPLATE above is untouched.
TEMPLATE_MULTI_LIMIT = (
    "Brief {brief_id} ({role}).\n"
    "Start from the B2 design checkpoint. Budget: at most {runs} recipe runs.\n"
    "Objective: {direction} {objective_metric}.\n"
    "Hard limits: {limits_text}; compare against the B2 start measurement.\n"
    "Final pick: {final_pick}.\n"
    "Validity: fully routed; zero route and DRC errors; matching B2 constraint "
    "fingerprint and clock digest; no leftover DONT_TOUCH or fixed LOC/BEL; "
    "AMD simulation PASS.\n"
    "Reporting: timing gain as Fmax percent, area gain as packed CLB LUT-cut "
    "percent and power gain as dynamic-power percent; also TNS, netlist LUT "
    "cells, and FFs.\n"
    "Noise: flag only; never use it to accept or reject.\n"
)


def load_frozen_spec(path: Path = BRIEFS_PATH) -> dict:
    """Load only the exact frozen YAML bytes, refusing an edited or alternate file."""

    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != FROZEN_BRIEFS_SHA256:
        raise ValueError("brief YAML hash differs from the 2026-11-06 freeze")
    spec = yaml.safe_load(raw)
    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise ValueError("invalid frozen brief schema")
    if tuple(spec.get("briefs", {})) != FROZEN_BRIEF_IDS:
        raise ValueError("frozen brief IDs or order differ")
    return spec


def _limits(row: dict) -> list[dict]:
    if ("limit" in row) == ("limits" in row):
        raise ValueError("a frozen brief needs exactly one of limit or limits")
    limits = [row["limit"]] if "limit" in row else list(row["limits"])
    if not limits:
        raise ValueError("a frozen brief needs at least one limit")
    return limits


def load_frozen_brief(brief_id: str, path: Path = BRIEFS_PATH) -> Brief:
    """Map the frozen presentation schema to the existing start-relative gate."""

    spec = load_frozen_spec(path)
    if brief_id not in spec["briefs"]:
        raise ValueError(f"brief is not in the frozen set: {brief_id}")
    row = spec["briefs"][brief_id]
    objective = row["objective"]
    limits = _limits(row)
    terms = []
    for limit in limits:
        comparison, factor, offset = LIMITS[limit["rule"]]
        term_id = (
            f"{brief_id.lower()}_limit" if len(limits) == 1
            else f"{brief_id.lower()}_limit_{METRICS[limit['metric']].value}"
        )
        terms.append(BriefTerm(
            id=term_id, metric=METRICS[limit["metric"]],
            comparison=comparison, factor=factor, offset=offset,
        ))
    brief = Brief(
        id=brief_id,
        objective=Objective(
            metric=METRICS[objective["metric"]],
            direction=DIRECTIONS[objective["direction"]],
        ),
        terms=tuple(terms),
        budget_runs=spec["common"]["budget"]["recipe_runs"],
        require_routed=True,
        max_drc_errors=0,
        require_constraint_match=True,
        require_clock_match=True,
        equivalence_policy="final_only",
    )
    expected = FOUR_BRIEF_FINGERPRINTS.get(brief_id)
    if expected is not None and brief.fingerprint() != expected:
        raise ValueError(f"frozen brief {brief_id} differs from its 2026-11-05 contract")
    return brief


def render_brief_text(brief: Brief | str) -> str:
    """Render the same pinned instructions for either LLM-facing agent.

    A supplied contract object must match the frozen mapping exactly, so a
    caller cannot quietly render altered limits under a familiar brief ID.
    """

    brief_id = brief.id if isinstance(brief, Brief) else brief
    expected = load_frozen_brief(brief_id)
    if isinstance(brief, Brief) and brief.fingerprint() != expected.fingerprint():
        raise ValueError("brief contract differs from the frozen specification")
    spec = load_frozen_spec()
    row = spec["briefs"][brief_id]
    limits = _limits(row)
    common = dict(
        brief_id=brief_id,
        role=row["role"],
        runs=spec["common"]["budget"]["recipe_runs"],
        direction=row["objective"]["direction"].capitalize(),
        objective_metric=METRIC_TEXT[row["objective"]["metric"]],
        final_pick=spec["common"]["final_pick"],
    )
    if len(limits) == 1:
        return TEMPLATE.format(
            limit_metric=METRIC_TEXT[limits[0]["metric"]],
            limit_text=LIMIT_TEXT[limits[0]["rule"]], **common,
        )
    limits_text = "; ".join(
        f"{METRIC_TEXT[limit['metric']]} {LIMIT_TEXT[limit['rule']]}" for limit in limits
    )
    return TEMPLATE_MULTI_LIMIT.format(limits_text=limits_text, **common)
