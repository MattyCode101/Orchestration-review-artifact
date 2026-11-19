"""Template v2 vocabulary (prompt-improvement change 8; build-review defect 2).

Change categories are measured in noise bands b of the design, in the metric's preferred direction:

    within noise          under 1 band
    small gain / loss     1 to 3 bands
    large gain / loss     3 to 30 bands
    very large gain/loss  over 30 bands      (new tier: round 1 saturated at "large" above 3 bands)

Edges follow orch/policies/llm_view.categorise: exactly 1 band is small, exactly 3 bands is still small, and
exactly 30 bands is still large, with the same 1e-9 relative tolerance.

Limit statuses (margin = distance inside the limit, negative when broken; b = the limit metric's band):

    at limit (equal to the start)    margin exactly zero on a start-relative limit (S0 on T0, A0 and P0)
    at limit                         margin exactly zero on any other limit
    within limit by under one band   0 < margin <= b
    within limit with margin         margin > b
    broken by under one band         0 < -margin <= b
    broken by more                   -margin > b

The round-1 vocabulary (orch/policies/llm_view) is unchanged and still used by every round-1 and round-2 prompt.
"""
from __future__ import annotations

from orch.policies.llm_view import CATEGORIES as R1_CATEGORIES
from orch.policies.llm_view import categorise as r1_categorise
from orch.policies.llm_view import limit_status as r1_limit_status

_REL = 1e-9
_INSIDE_TOLERANCE = 1e-9
VERY_LARGE_FROM = 30.0

CATEGORIES_V2 = (
    "very large gain", "large gain", "small gain", "within noise", "small loss", "large loss", "very large loss",
)
LIMIT_STATUSES_V2 = (
    "at limit (equal to the start)", "at limit", "within limit by under one band", "within limit with margin",
    "broken by under one band", "broken by more",
)
NOTHING_SELECTABLE = "nothing selectable"
NO_OPTIONS = "no options left"
PLACEMENT_R1 = {True: "placement preserved", False: "re-placing allowed"}
PLACEMENT_V2 = {True: "placement must not change", False: "placement may change"}


def signed_bands(delta: float, band: float, *, higher_is_better: bool) -> float:
    """The change in noise bands, positive when it moves the metric in its preferred direction."""

    if band <= 0.0:
        raise ValueError("a noise band must be positive")
    better = delta if higher_is_better else -delta
    return better / band


def categorise_v2(delta: float, band: float, *, higher_is_better: bool) -> str:
    better = delta if higher_is_better else -delta
    size = abs(better)
    if size == 0.0 or size < band * (1.0 - _REL):
        return "within noise"
    if size <= 3.0 * band * (1.0 + _REL):
        tier = "small"
    elif size <= VERY_LARGE_FROM * band * (1.0 + _REL):
        tier = "large"
    else:
        tier = "very large"
    return f"{tier} {'gain' if better > 0 else 'loss'}"


def categorise(delta: float, band: float, *, higher_is_better: bool, v2: bool) -> str:
    """One entry point for both vocabularies."""

    if v2:
        return categorise_v2(delta, band, higher_is_better=higher_is_better)
    return r1_categorise(delta, band, higher_is_better=higher_is_better)


def limit_status_v2(margin: float, band: float, *, start_relative: bool) -> str:
    if abs(margin) <= _INSIDE_TOLERANCE:
        return "at limit (equal to the start)" if start_relative else "at limit"
    if margin > 0.0:
        return "within limit with margin" if margin > band * (1.0 + _REL) else "within limit by under one band"
    return "broken by more" if -margin > band * (1.0 + _REL) else "broken by under one band"


def limit_status(margin: float, band: float, *, start_relative: bool, v2: bool) -> str:
    if v2:
        return limit_status_v2(margin, band, start_relative=start_relative)
    return r1_limit_status(margin, band)


def is_broken(status: str) -> bool:
    return status.startswith("broken")


def direction_of(category: str) -> str:
    """gain, loss or none: the vocabulary-free direction of a category (both vocabularies)."""

    if category in ("within noise", "within_noise"):
        return "none"
    if category.endswith("gain"):
        return "gain"
    if category.endswith("loss"):
        return "loss"
    return category


assert set(R1_CATEGORIES) == {"large_gain", "small_gain", "within_noise", "small_loss", "large_loss"}
