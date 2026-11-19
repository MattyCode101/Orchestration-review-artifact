"""Pre-registered, metric-only admission screen for held-out test designs.

The caller must independently establish physical validity, unchanged timing
constraints and AMD PASS before passing one-step metrics to these functions.
No candidate ranking or outcome summary is produced here.
"""
from __future__ import annotations

import math
from decimal import Decimal
from collections.abc import Mapping, Sequence

METRICS = ("wns_ns", "tns_ns", "packed_luts", "netlist_luts")


def noise_bands(
    baseline: Mapping[str, float],
    controls: Mapping[str, Mapping[str, float]],
    variants: Sequence[str],
) -> dict[str, float]:
    """Maximum absolute B2-to-control difference for each recorded metric."""

    if set(controls) != set(variants) or len(variants) != 5:
        raise ValueError("all five named noise controls are required")
    if any(name not in baseline or not math.isfinite(baseline[name]) for name in METRICS):
        raise ValueError("baseline lacks a finite noise metric")
    for variant in variants:
        if any(
            name not in controls[variant] or not math.isfinite(controls[variant][name])
            for name in METRICS
        ):
            raise ValueError(f"noise control lacks a finite metric: {variant}")
    return {
        name: float(max(
            abs(Decimal(str(controls[variant][name])) - Decimal(str(baseline[name])))
            for variant in variants
        ))
        for name in METRICS
    }


def screen(
    *,
    candidate_order: Sequence[str],
    selected_recipe_ids: Sequence[str],
    baselines: Mapping[str, Mapping[str, float]],
    bands: Mapping[str, Mapping[str, float]],
    one_steps: Mapping[str, Mapping[str, Mapping[str, float]]],
    places: int = 5,
    ineligible_designs: Sequence[str] = (),
) -> dict[str, bool]:
    """Admit in declared order if any selected recipe beats either strict band."""

    if len(candidate_order) != 7 or len(set(candidate_order)) != 7:
        raise ValueError("screen requires seven distinct ordered candidates")
    if len(selected_recipe_ids) != 10 or len(set(selected_recipe_ids)) != 10:
        raise ValueError("screen requires exactly ten selected recipes")
    if places != 5:
        raise ValueError("room has exactly five test places")
    if set(baselines) != set(candidate_order) or set(bands) != set(candidate_order):
        raise ValueError("baseline or noise bands differ from candidate order")
    if set(one_steps) != set(candidate_order):
        raise ValueError("one-step results differ from candidate order")
    ineligible = set(ineligible_designs)
    if not ineligible.issubset(candidate_order):
        raise ValueError("ineligible design is outside the frozen candidate order")
    admitted: dict[str, bool] = {}
    used = 0
    for design in candidate_order:
        if design in ineligible:
            admitted[design] = False
            continue
        baseline = baselines[design]
        band = bands[design]
        rows = one_steps[design]
        if set(rows) != set(selected_recipe_ids):
            raise ValueError(f"design lacks one of ten checked one-step outputs: {design}")
        for name in ("wns_ns", "packed_luts"):
            values = [baseline.get(name), band.get(name)]
            values += [rows[recipe].get(name) for recipe in selected_recipe_ids]
            if any(value is None or not math.isfinite(value) for value in values):
                raise ValueError(f"non-finite screen metric: {design} {name}")
            if band[name] < 0:
                raise ValueError("noise band cannot be negative")
        eligible = any(
            (
                Decimal(str(rows[recipe]["wns_ns"])) - Decimal(str(baseline["wns_ns"]))
                > Decimal(str(band["wns_ns"]))
                or Decimal(str(baseline["packed_luts"])) - Decimal(str(rows[recipe]["packed_luts"]))
                > Decimal(str(band["packed_luts"]))
            )
            for recipe in selected_recipe_ids
        )
        admitted[design] = bool(eligible and used < places)
        used += int(admitted[design])
    return admitted
