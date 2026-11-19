"""Template v2 (Thu 2026-11-12): shared setup for the experience-call generator and the decision bank.

Stored outcomes only: no tool job is ever started. The run graph is the power study's union of engine roots
(scripts/power_mediation_2026-11-10/pm_common.py: the sealed v4 training root first, the overnight power roots,
the power study's own root), extended by the power-readiness and bandit roots (training designs), read-only.
AMD rule: the power study's lenient mode (a state with no AMD record is accepted and listed; a recorded failure
is refused), because only about 30% of stored outputs carry an AMD record and the brief checks equivalence on the
final pick only. Worker planners: the real FINAL timing and area workers (pm_common.build_workers).

Noise bands: configs/noise.yaml; logicnets_jscl, which has no entry there, takes the band of
configs/noise_development.yaml (the run-3 Family N range the power study and the bandit study used).
"""
from __future__ import annotations
import os

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(os.environ.get("ORCH_PROJECT_ROOT") or Path(__file__).resolve().parents[2])
for extra in (PROJECT, PROJECT / "scripts", PROJECT / "scripts/power_mediation_2026-11-10",
              PROJECT / "scripts/r2_build_review_2026-11-12"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import pm_common as pm  # noqa: E402  (pinned exact-search setup of the power study; imported, never edited)

from orch.config import load_noise  # noqa: E402
from orch.contract.models import MetricName  # noqa: E402

WORK = PROJECT / "data/template_v2_2026-11-12"
TRAINING = ("vexriscv_re-place", "logicnets_jscl", "rosetta_spam-filter", "rosetta_3d-rendering", "finn_radioml")
BRIEFS = ("T0", "A0", "T", "A")
EXTRA_ROOTS = {
    "power_readiness": PROJECT / "data/power_readiness_2026-11-11/engine",
    "bandit": PROJECT / "data/bandit_2026-11-11/engine",
}


def jread(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def roots() -> dict[str, Path]:
    out = pm.union_roots(include_new=True)
    for name, root in EXTRA_ROOTS.items():
        if (root / "cache").is_dir() and (root / "runs").is_dir():
            out[name] = root
    return out


def context() -> dict[str, Any]:
    ctx = pm.build_context(include_new=True, roots=roots(), amd_lenient=True)
    ctx["roots"] = {k: str(v) for k, v in ctx["index"].roots.items()}
    return ctx


_DEV_NOISE = None


def bands_for(ctx: dict[str, Any], design: str) -> dict[MetricName, float]:
    """Absolute bands for the design's start (the logicnets band from the development noise file)."""

    global _DEV_NOISE
    _, evidence = ctx["starts"][design]
    bands = ctx["noise"].for_start(design, evidence)
    if MetricName.WNS_NS not in bands or MetricName.PACKED_LUTS not in bands:
        if _DEV_NOISE is None:
            _DEV_NOISE = load_noise(PROJECT / "configs/noise_development.yaml")
        dev = _DEV_NOISE.for_start(design, evidence)
        for metric, value in dev.items():
            bands.setdefault(metric, value)
    return bands


def metric_values(evidence) -> dict[str, float]:
    return {m.value: evidence.metric(m).value for m in (MetricName.WNS_NS, MetricName.PACKED_LUTS,
                                                       MetricName.DYNAMIC_POWER_W)}
