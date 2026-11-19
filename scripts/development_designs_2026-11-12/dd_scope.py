"""Development scope for the campaign harness (scripts/campaign_harness.py, imported, not edited).

The harness builds one Context per scope (designs, hash-pinned manifest, overlay root, noise function, telemetry provider).
Training and test scopes exist there; this module adds the development scope. The only additions are in this task's own file:
  * ChargingLedger: a new AMD validation is added to the job-second cap (the harness charges recipe runs and extractions only);
  * dev_noise: bands from configs/noise_development.yaml (same schema as configs/noise.yaml).
Stored outcomes of the benchmark screen replay from the manifest (roots bx_<design>); new outputs go to data/.../engine only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dd_common as dd  # noqa: E402
import campaign_harness as h  # noqa: E402
from orch.config import load_noise  # noqa: E402
from orch.contract.models import DesignState, Evidence, MetricName  # noqa: E402
from orch.engine.manifest_cache import CacheManifest  # noqa: E402
from orch.engine.read_through_ledger import ReadThroughLedger  # noqa: E402

MANIFEST_DIR = dd.WORK / "manifest"
MANIFEST = MANIFEST_DIR / "manifest.json"
MANIFEST_SHA = MANIFEST_DIR / "manifest.sha256"
PLAN_SHA = (dd.PROJECT / "notes/development_designs_plan.sha256").read_text(encoding="utf-8").split()[0]
CAPS = {"new_real_runs": 600, "job_hours": 150, "tokens": 0, "usd": 0.0}
EXTRACTION_JOBS = 150
SPLIT = yaml.safe_load((dd.PROJECT / "configs/development/split.yaml").read_text(encoding="utf-8"))
DESIGNS = tuple(SPLIT["designs"])
_NOISE = load_noise(dd.PROJECT / SPLIT["noise_file"])


def dev_noise(design: str, start: Evidence) -> dict[MetricName, float]:
    return _NOISE.for_start(design, start)


class ChargingLedger(ReadThroughLedger):
    """ReadThroughLedger that adds the wall time of a NEW validation to the job-second cap."""

    caps = None

    def verify(self, design: str, output_id: str) -> dict:
        known = self.record_location(design, output_id) is not None
        record = super().verify(design, output_id)
        if not known and self.caps is not None:
            self.caps.add_job_seconds(sum(float(a["wall_s"]) for a in record.get("attempts", [])))
        return record


def dev_scope(provider_for=None, work: Path | None = None) -> h.Scope:
    expected = MANIFEST_SHA.read_text(encoding="utf-8").strip()
    manifest = CacheManifest.load(MANIFEST, expected_hash=expected)
    starts = h._starts_from_manifest(manifest)
    if set(starts) != set(DESIGNS):
        raise SystemExit("development manifest does not hold exactly the registered starts")
    for d in DESIGNS:
        if d in dd.FORBIDDEN:
            raise SystemExit(f"refused: {d}")
    return h.Scope("development", DESIGNS, manifest, work or dd.WORK, starts, dev_noise, provider_for)


def build(mode: str, with_provider: bool = True, work: Path | None = None) -> h.Context:
    """mode 'replay': no tool backend (any recipe run is an error). mode 'run': real runs against the caps.
    work: the overlay's work root (default: the task's own data folder; the smoke test uses a throwaway one)."""

    root = work or dd.WORK
    caps = h.CapLedger(root, CAPS, PLAN_SHA) if mode == "run" else None
    provider_for = h.live_provider_factory(root, CAPS, PLAN_SHA, EXTRACTION_JOBS) if with_provider else None
    scope = dev_scope(provider_for, root)
    h.ReadThroughLedger = ChargingLedger
    ctx = h.build_context(scope, mode=mode, caps=caps)
    if isinstance(ctx.ledger, ChargingLedger):
        ctx.ledger.caps = caps
    return ctx


def load_state(ctx: h.Context, state_id: str) -> DesignState:
    """A stored or overlay state record, checked against its location."""

    path = ctx.store.state_path(state_id)
    state = DesignState.model_validate_json((path.parent / "state.json").read_bytes())
    if state.id != state_id or str(state.path) != str(path):
        raise SystemExit(f"state record differs from its location: {state_id}")
    return state
