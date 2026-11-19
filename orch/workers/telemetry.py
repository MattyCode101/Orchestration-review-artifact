"""Training-only input telemetry for fitted episode decisions.

The provider reads a saved utilization report when a committed training run
contains one. Other fields come from the pinned v3 record. It opens no DCP and
starts no tool. Every source is bound to the input state's SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from orch.analysis.fit_workers import (
    PROJECT,
    TRAINING_ROOT,
    TRAINING_V3_PARSER_SHA256,
    TRAINING_V3_SOURCE,
    extract_features,
    load_feature_spec,
)
from orch.contract.models import DesignState, Evidence
from orch.engine.dcp_parser import (
    _parse_lut_as_shift_register,
    _parse_lut_primitive_counts,
    _parse_utilization_report,
    parse_pblock_utilization_text,
)
from orch.engine.store import StateStore

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_RUN = re.compile(r"run_[0-9a-f]{32}\Z")
PILOT_TRAINING_ROOT = PROJECT / "data/pilots/thursday_night_train_2026-11-05"
PILOT_TRAINING_ROOT_V2 = PROJECT / "data/pilots/thursday_night_train_v2_2026-11-05"
PILOT_TRAINING_ROOT_V3 = PROJECT / "data/pilots/thursday_night_train_v3_2026-11-05"
PILOT_TRAINING_ROOT_V4 = PROJECT / "data/pilots/thursday_overnight_train_v4_2026-11-05"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SavedTelemetryProvider:
    """Use only named five-design training stores and pinned v3 telemetry.

    This strict root guard prevents a planning run from reading held-out
    outcome data. A future production provider requires a separate review.
    """

    def __init__(self, data_root: Path | str = TRAINING_ROOT) -> None:
        self.root = Path(data_root).expanduser().resolve(strict=True)
        permitted = {
            TRAINING_ROOT.resolve(strict=True), PILOT_TRAINING_ROOT.resolve(),
            PILOT_TRAINING_ROOT_V2.resolve(), PILOT_TRAINING_ROOT_V3.resolve(),
            PILOT_TRAINING_ROOT_V4.resolve(),
        }
        if self.root not in permitted:
            raise ValueError("telemetry provider refuses a non-training data root")
        self.v3 = TRAINING_V3_SOURCE.resolve(strict=True)
        self.spec = load_feature_spec()
        self.store = StateStore(self.root)
        self._identity = {
            "kind": "saved-training-telemetry-v1",
            "data_root": str(self.root),
            "v3_root": str(self.v3),
            "v3_plan_sha256": _sha(self.v3 / "plan.json"),
            "provider_sha256": _sha(Path(__file__)),
        }

    def identity(self) -> dict[str, str]:
        """Bind resume to the exact named data source and parser code."""

        return dict(self._identity)

    def _v3_record(
        self, state: DesignState, evidence: Evidence,
    ) -> tuple[dict[str, Any] | None, str | None]:
        if not _HASH.fullmatch(state.id):
            raise ValueError("input state is not a SHA-256 DCP identity")
        path = self.v3 / "states" / state.id / "dcp_telemetry.json"
        if not path.is_file():
            return None, None
        if not path.resolve(strict=True).is_relative_to(self.v3):
            raise ValueError("telemetry record escapes the pinned v3 source")
        raw = path.read_bytes()
        record = json.loads(raw)
        if (
            not isinstance(record, dict)
            or record.get("state_sha256") != state.id
            or record.get("fake") is not False
            or record.get("provenance", {}).get("extractor_sha256")
            != TRAINING_V3_PARSER_SHA256
        ):
            raise ValueError("v3 telemetry identity or parser version differs")
        validity = record.get("validity", {})
        for key in ("constraint_fingerprint", "clock_digest", "routed", "route_errors", "drc_errors"):
            if validity.get(key) != getattr(evidence, key):
                raise ValueError(f"v3 telemetry differs from meter {key}")
        for key in ("wns_ns", "packed_luts"):
            expected = evidence.metric(key)
            observed = record.get("metrics", {}).get(key, {}).get("value")
            tolerance = 0.001 if key == "wns_ns" else 0.0
            if (
                expected is None or type(observed) not in (int, float)
                or not math.isfinite(observed)
                or abs(expected.value - observed) > tolerance
            ):
                raise ValueError(f"v3 telemetry differs from meter {key}")
        return record, hashlib.sha256(raw).hexdigest()

    def _report(self, state: DesignState, evidence: Evidence, run_id: str | None) -> Path | None:
        candidates: list[Path] = []
        if run_id is not None:
            if not _RUN.fullmatch(run_id):
                raise ValueError("producer run ID is invalid")
            outcome = self.store.load_outcome(run_id)
            if outcome.state is None or outcome.state.id != state.id:
                raise ValueError("producer outcome differs from input state")
            ids = [run_id]
            if outcome.run.original_run_id is not None:
                ids.append(outcome.run.original_run_id)
            for item in ids:
                if not _RUN.fullmatch(item):
                    raise ValueError("original run ID is invalid")
                candidates.append(self.root / "runs" / item / "reports" / "final_utilization.rpt")
        if evidence.raw_report_path:
            candidates.append(Path(evidence.raw_report_path) / "final_utilization.rpt")
        for path in candidates:
            if not path.is_file():
                continue
            resolved = path.resolve(strict=True)
            if resolved.is_relative_to(self.root):
                return resolved
        return None

    def __call__(
        self, state: DesignState, evidence: Evidence,
        producer_recipe_id: str | None, run_id: str | None,
    ) -> dict[str, object]:
        """Return FittedWorker context and compact source provenance."""

        if state.fake or evidence.fake or evidence.state_id != state.id:
            raise ValueError("saved real telemetry needs matching real meter Evidence")
        canonical = self.store.state_path(state.id)
        if Path(state.path).resolve(strict=True) != canonical:
            raise ValueError("telemetry input path differs from its hash-checked state")
        if state.parent_id is None:
            if producer_recipe_id is not None or run_id is not None:
                raise ValueError("B2 start cannot have a producer")
        elif not producer_recipe_id or not run_id:
            raise ValueError("derived state needs its producing recipe and run")
        else:
            if not _RUN.fullmatch(run_id):
                raise ValueError("producer run ID is invalid")
            produced = self.store.load_outcome(run_id)
            if (
                produced.state != state
                or produced.evidence != evidence
                or produced.run.recipe_id != producer_recipe_id
                or produced.run.input_sha256_before != state.parent_id
            ):
                raise ValueError("producer identity differs from persisted run outcome")
        record, record_sha = self._v3_record(state, evidence)
        if record is None:
            if state.parent_id is None:
                raise FileNotFoundError("training B2 start lacks its pinned v3 telemetry")
            # D3 produced 28 new states after the v3 plan. A run report can
            # still supply area rows; missing path fields fall back to beta.
            record = {
                "state_sha256": state.id,
                "fake": False,
                "metrics": evidence.model_dump(mode="json")["metrics"],
                "probe": {},
                "wpath": {},
            }
        source: dict[str, object] = {
            "v3_record_sha256": record_sha,
            "utilization_report_sha256": None,
            "utilization_report": None,
            "state_sha256": state.id,
        }
        report = self._report(state, evidence, run_id)
        if report is not None:
            raw = report.read_bytes()
            text = raw.decode("utf-8", errors="replace")
            utilization = _parse_utilization_report(text)["utilization_summary"]
            packed = utilization.get("CLB LUTs", {}).get("used")
            measured = evidence.metric("packed_luts")
            if packed is None or measured is None or float(packed) != measured.value:
                raise ValueError("saved utilization report disagrees with meter packed LUTs")
            probe = dict(record.get("probe", {}))
            additions = {
                "utilization_summary": utilization,
                "pblock_sizing_resources": parse_pblock_utilization_text(text),
                "lut_as_shift_register": _parse_lut_as_shift_register(text),
                "lut_primitive_counts": _parse_lut_primitive_counts(text),
            }
            for key, value in additions.items():
                if value is not None and value != {}:
                    probe[key] = value
            merged = {**record, "probe": probe}
            old = extract_features(record, self.spec)
            new = extract_features(merged, self.spec)
            for recipe_id, value in new.items():
                prior = old[recipe_id]
                if prior is not None and value is not None and not math.isclose(
                    prior, value, rel_tol=1e-9, abs_tol=1e-9
                ):
                    raise ValueError(f"saved utilization conflicts with v3 feature {recipe_id}")
            record = merged
            source["utilization_report_sha256"] = hashlib.sha256(raw).hexdigest()
            source["utilization_report"] = str(report.relative_to(self.root))
        return {
            "telemetry": record,
            "produced_by_recipe_id": producer_recipe_id,
            "telemetry_source": source,
        }

