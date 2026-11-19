"""Labelled software-only backend seeded from the copied opportunity-study CSV.

A fake checkpoint is JSON, never a hardware DCP. Direct B2 recipe outcomes use
one eligible table row; missing observations use an explicitly synthetic neutral
effect. Later steps compose B2 deltas with a deterministic synthetic interaction.
All metrics, validity checks, costs, and output files are fake test data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from orch.contract.models import (
    Claims,
    DesignState,
    Evidence,
    MetricName,
    MetricValue,
    RunOutcome,
    RunRecord,
    RunStatus,
)
from orch.engine.store import StateStore, _atomic_bytes, _json_bytes
from orch.interfaces import RecipeSpec


SEED_SHA256 = "b68d42284464584edc8a363f46c58fb4880b9906413146b4f85b64dd5a903277"
CHECKPOINT_SCHEMA = "fake-checkpoint-v1"
FAKE_MODEL_VERSION = "fake-backend-v1"
# The gate compares metric methods exactly. Detailed direct/synthetic provenance
# is in Claims and the sidecar named by Evidence.raw_report_path.
METRIC_METHOD = "fake:CSV_B2_plus_synthetic_composition_v1"
METRIC_BASIS = "fake_routed_design"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(raw: str, *, field: str, row: int) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field} in seed row {row}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field} in seed row {row}")
    return value


def _luts(raw: str, *, field: str, row: int) -> int:
    value = _number(raw, field=field, row=row)
    if value < 1 or not value.is_integer():
        raise ValueError(f"invalid LUT count in seed row {row}: {value}")
    return int(value)


@dataclass(frozen=True)
class _Baseline:
    design: str
    line: int
    run_id: str
    wns_ns: float
    packed_luts: int
    dynamic_power_w: float
    noise_wns_ns: float
    noise_luts_pct: float
    noise_power_pct: float

    def metrics(self) -> dict[str, float]:
        return {
            MetricName.WNS_NS.value: self.wns_ns,
            MetricName.PACKED_LUTS.value: float(self.packed_luts),
            MetricName.DYNAMIC_POWER_W.value: self.dynamic_power_w,
        }


@dataclass(frozen=True)
class _DirectEffect:
    design: str
    seed_label: str
    line: int
    run: str
    run_id: str
    source_csv: str
    final_wns_ns: float
    final_packed_luts: int
    final_dynamic_power_w: float

    def metrics(self) -> dict[str, float]:
        return {
            MetricName.WNS_NS.value: self.final_wns_ns,
            MetricName.PACKED_LUTS.value: float(self.final_packed_luts),
            MetricName.DYNAMIC_POWER_W.value: self.final_dynamic_power_w,
        }


class FakeBackend:
    """Backend protocol implementation for end-to-end software tests only.

    The constructor requires a copied seed table. With neutral_synthetic,
    unobserved design/recipe pairs return the unchanged input metrics and
    cite that assumption in every output; this lets all eight B2 designs
    exercise the software loop without inventing a hardware observation.
    """

    def __init__(
        self,
        *,
        store: StateStore,
        seed_csv: Path | None = None,
        interaction_fraction: float = 0.10,
        missing_policy: Literal["neutral_synthetic", "error"] = "neutral_synthetic",
    ) -> None:
        if not 0 <= interaction_fraction <= 1:
            raise ValueError("interaction_fraction must be in [0, 1]")
        if missing_policy not in ("neutral_synthetic", "error"):
            raise ValueError("invalid missing_policy")
        project = Path(__file__).resolve().parents[2]
        self.store = store
        self.seed_csv = (seed_csv or project / "data/fake_seed/combined_results.csv").resolve()
        self.seed_sha256 = _sha256_file(self.seed_csv)
        if seed_csv is None and self.seed_sha256 != SEED_SHA256:
            raise ValueError("copied fake seed table differs from its recorded SHA-256")
        self.interaction_fraction = interaction_fraction
        self.missing_policy = missing_policy
        self._baselines, self._direct = self._load_seed()
        version_payload = {
            "model": FAKE_MODEL_VERSION,
            "source_sha256": _sha256_file(Path(__file__)),
            "seed_sha256": self.seed_sha256,
            "interaction_fraction": self.interaction_fraction,
            "missing_policy": self.missing_policy,
        }
        digest = hashlib.sha256(_json_bytes(version_payload)).hexdigest()
        self.meter_version = f"{FAKE_MODEL_VERSION}:{digest}"

    def _load_seed(self) -> tuple[dict[str, _Baseline], dict[tuple[str, str], _DirectEffect]]:
        baselines: dict[str, _Baseline] = {}
        candidate_rows: list[tuple[int, dict[str, str]]] = []
        with self.seed_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "design", "run", "run_id", "recipe_or_worker", "row_role",
                "figure_eligible", "input_state", "root_state", "validity_rule",
                "final_wns_ns", "final_clb_luts", "final_dynamic_w",
                "b2_wns_ns", "b2_clb_luts", "b2_dynamic_w",
                "noise_wns_ns", "noise_clb_luts_pct", "noise_dynamic_pct",
                "source_csv",
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("fake seed table is missing required columns")
            for line, row in enumerate(reader, start=2):
                if row["row_role"] == "B2 reference":
                    design = row["design"]
                    if design in baselines:
                        raise ValueError(f"duplicate B2 reference for {design}")
                    baseline = _Baseline(
                        design=design,
                        line=line,
                        run_id=row["run_id"],
                        wns_ns=_number(row["final_wns_ns"], field="final_wns_ns", row=line),
                        packed_luts=_luts(row["final_clb_luts"], field="final_clb_luts", row=line),
                        dynamic_power_w=_number(row["final_dynamic_w"], field="final_dynamic_w", row=line),
                        noise_wns_ns=_number(row["noise_wns_ns"], field="noise_wns_ns", row=line),
                        noise_luts_pct=_number(row["noise_clb_luts_pct"], field="noise_clb_luts_pct", row=line),
                        noise_power_pct=_number(row["noise_dynamic_pct"], field="noise_dynamic_pct", row=line),
                    )
                    if (
                        baseline.wns_ns != _number(row["b2_wns_ns"], field="b2_wns_ns", row=line)
                        or baseline.packed_luts != _luts(row["b2_clb_luts"], field="b2_clb_luts", row=line)
                        or baseline.dynamic_power_w != _number(row["b2_dynamic_w"], field="b2_dynamic_w", row=line)
                    ):
                        raise ValueError(f"B2 reference disagrees with B2 columns for {design}")
                    baselines[design] = baseline
                if (
                    row["row_role"] == "nominal candidate"
                    and row["figure_eligible"] == "True"
                    and row["input_state"] == "B2"
                    and row["root_state"] == "B2"
                ):
                    candidate_rows.append((line, row))
        direct: dict[tuple[str, str], _DirectEffect] = {}
        for line, row in candidate_rows:
            design = row["design"]
            if design not in baselines:
                raise ValueError(f"direct seed row lacks B2 reference: {design}")
            if row["validity_rule"] not in (
                "run3 collect3.py: valid3", "run4 collect4.py: valid"
            ):
                raise ValueError(f"unrecognized direct seed validity in row {line}")
            baseline = baselines[design]
            if (
                baseline.wns_ns != _number(row["b2_wns_ns"], field="b2_wns_ns", row=line)
                or baseline.packed_luts != _luts(row["b2_clb_luts"], field="b2_clb_luts", row=line)
                or baseline.dynamic_power_w != _number(row["b2_dynamic_w"], field="b2_dynamic_w", row=line)
            ):
                raise ValueError(f"direct seed row {line} uses a different start")
            effect = _DirectEffect(
                design=design,
                seed_label=row["recipe_or_worker"],
                line=line,
                run=row["run"],
                run_id=row["run_id"],
                source_csv=row["source_csv"],
                final_wns_ns=_number(row["final_wns_ns"], field="final_wns_ns", row=line),
                final_packed_luts=_luts(row["final_clb_luts"], field="final_clb_luts", row=line),
                final_dynamic_power_w=_number(row["final_dynamic_w"], field="final_dynamic_w", row=line),
            )
            key = (design, effect.seed_label)
            previous = direct.get(key)
            if previous is not None:
                # Run 4 repeats some run-3 rows. Reject conflicting duplicates
                # instead of silently changing a fake recipe's effect.
                if previous.metrics() != effect.metrics():
                    raise ValueError(f"conflicting direct observations for {key}")
                continue
            direct[key] = effect
        if not baselines:
            raise ValueError("fake seed table has no B2 references")
        return baselines, direct

    @property
    def designs(self) -> tuple[str, ...]:
        """Names with a copied B2 reference; no benchmark file is opened."""

        return tuple(sorted(self._baselines))

    def available_direct(self, design: str, recipe: RecipeSpec) -> bool:
        """Whether the CSV has a valid nominal B2 row for this exact pair."""

        return (design, self._seed_label(recipe)) in self._direct

    @staticmethod
    def _seed_label(recipe: RecipeSpec) -> str:
        return f"LATCH {recipe.flow}" if recipe.kind == "latch" else recipe.id

    def _payload(self, state: DesignState) -> dict[str, object]:
        if not state.fake:
            raise ValueError("fake backend rejects a real checkpoint")
        canonical = self.store.state_path(state.id)
        if Path(state.path).resolve() != canonical:
            raise ValueError("fake state path is not canonical")
        try:
            payload = json.loads(canonical.read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("fake state is not a JSON fake checkpoint") from exc
        if (
            payload.get("schema") != CHECKPOINT_SCHEMA
            or payload.get("fake") is not True
            or payload.get("seed_sha256") != self.seed_sha256
            or payload.get("design") != state.design
            or state.design not in self._baselines
        ):
            raise ValueError("fake state metadata differs from this seed/backend")
        sequence = payload.get("sequence")
        steps = payload.get("steps")
        metrics = payload.get("metrics")
        if (
            not isinstance(sequence, list)
            or not isinstance(steps, list)
            or len(sequence) != len(steps)
            or not isinstance(metrics, dict)
            or any(not isinstance(x, str) for x in sequence)
            or any(not isinstance(x, dict) or x.get("recipe_id") != sequence[i]
                   for i, x in enumerate(steps))
        ):
            raise ValueError("malformed fake checkpoint")
        for name in (
            MetricName.WNS_NS.value,
            MetricName.PACKED_LUTS.value,
            MetricName.DYNAMIC_POWER_W.value,
        ):
            value = metrics.get(name)
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"fake checkpoint lacks finite {name}")
        return payload

    def create_start(self, design: str) -> DesignState:
        """Create a distinct, content-addressed JSON start for one seed design."""

        baseline = self._baselines[design]
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "fake": True,
            "design": design,
            "seed_sha256": self.seed_sha256,
            "reference_line": baseline.line,
            "reference_run_id": baseline.run_id,
            "metrics": baseline.metrics(),
            "sequence": [],
            "steps": [],
        }
        content = _json_bytes(payload)
        input_dir = self.store.root / "fake_inputs"
        name = hashlib.sha256(design.encode("utf-8")).hexdigest() + ".json"
        source = input_dir / name
        if source.exists():
            if source.read_bytes() != content:
                raise ValueError(f"fake input file changed: {source}")
        else:
            _atomic_bytes(source, content)
        return self.store.import_state(source, design=design, fake=True)

    def _interaction(
        self, baseline: _Baseline, previous_recipe: str, recipe_id: str
    ) -> dict[str, float]:
        """Small deterministic interaction bounded by a fraction of the seed noise bands."""

        material = (
            f"{self.seed_sha256}|{baseline.design}|{previous_recipe}|{recipe_id}"
        ).encode("utf-8")
        digest = hashlib.sha256(material).digest()
        signed = [
            2 * int.from_bytes(digest[i:i + 4], "big") / (2**32 - 1) - 1
            for i in (0, 4, 8)
        ]
        return {
            MetricName.WNS_NS.value: round(
                signed[0] * self.interaction_fraction * baseline.noise_wns_ns, 6
            ),
            MetricName.PACKED_LUTS.value: float(round(
                signed[1] * self.interaction_fraction
                * baseline.noise_luts_pct / 100 * baseline.packed_luts
            )),
            MetricName.DYNAMIC_POWER_W.value: round(
                signed[2] * self.interaction_fraction
                * baseline.noise_power_pct / 100 * baseline.dynamic_power_w, 6
            ),
        }

    def _evidence(
        self, state: DesignState, payload: dict[str, object], *, run_dir: Path
    ) -> Evidence:
        """Read fake JSON and store its precise source beside the fake run."""

        run_dir = Path(run_dir).resolve()
        if not run_dir.is_relative_to(self.store.root):
            raise ValueError("fake evidence directory escapes the store")
        run_dir.mkdir(parents=True, exist_ok=True)
        source_path = run_dir / "fake_evidence_source.json"
        source = {
            "fake": True,
            "hardware_measurement": False,
            "state_id": state.id,
            "seed_csv": str(self.seed_csv),
            "seed_sha256": self.seed_sha256,
            "baseline_reference_line": payload["reference_line"],
            "baseline_reference_run_id": payload["reference_run_id"],
            "sequence": payload["sequence"],
            "steps": payload["steps"],
            "validity_method": "simulated routed/zero-DRC/unchanged-constraints for software tests",
            "metric_method": METRIC_METHOD,
        }
        _atomic_bytes(source_path, _json_bytes(source))
        raw_metrics = payload["metrics"]
        assert isinstance(raw_metrics, dict)
        metrics = {
            MetricName.WNS_NS: MetricValue(
                value=float(raw_metrics[MetricName.WNS_NS.value]),
                unit="ns", method=METRIC_METHOD, basis=METRIC_BASIS,
            ),
            MetricName.PACKED_LUTS: MetricValue(
                value=float(raw_metrics[MetricName.PACKED_LUTS.value]),
                unit="LUT", method=METRIC_METHOD, basis=METRIC_BASIS,
            ),
            MetricName.DYNAMIC_POWER_W: MetricValue(
                value=float(raw_metrics[MetricName.DYNAMIC_POWER_W.value]),
                unit="W", method=METRIC_METHOD, basis=METRIC_BASIS,
            ),
        }
        design = state.design
        fingerprint = hashlib.sha256(
            f"fake-constraints|{self.seed_sha256}|{design}".encode("utf-8")
        ).hexdigest()
        clocks = hashlib.sha256(
            f"fake-clocks|{self.seed_sha256}|{design}".encode("utf-8")
        ).hexdigest()
        evidence_data = {
            "state_id": state.id,
            "metrics": metrics,
            "routed": True,
            "route_errors": 0,
            "drc_errors": 0,
            "constraint_fingerprint": fingerprint,
            "clock_digest": clocks,
            "meter_version": self.meter_version,
            "raw_report_path": str(source_path),
            "fake": True,
        }
        encoded = _json_bytes({
            key: (
                {name.value: value.model_dump(mode="json") for name, value in val.items()}
                if key == "metrics" else val
            )
            for key, val in evidence_data.items()
        })
        evidence_id = hashlib.sha256(encoded).hexdigest()
        return Evidence(id=evidence_id, **evidence_data)

    def measure_start(self, state: DesignState, *, run_dir: Path) -> Evidence:
        """Return fake evidence for a JSON start without a recipe run."""

        payload = self._payload(state)
        return self._evidence(state, payload, run_dir=run_dir)

    def execute(
        self,
        state: DesignState,
        recipe: RecipeSpec,
        *,
        run_id: str,
        run_dir: Path,
    ) -> RunOutcome:
        """Simulate one recipe with explicit direct or synthetic provenance."""

        payload = self._payload(state)
        run_dir = Path(run_dir).resolve()
        if not run_dir.is_relative_to(self.store.root / "runs"):
            raise ValueError("fake run directory escapes the store's runs")
        baseline = self._baselines[state.design]
        seed_label = self._seed_label(recipe)
        effect = self._direct.get((state.design, seed_label))
        if effect is None and self.missing_policy == "error":
            raise KeyError(f"no direct B2 seed row for {state.design}/{recipe.id}")
        sequence = list(payload["sequence"])
        steps = list(payload["steps"])
        parent_metrics = dict(payload["metrics"])
        if effect is None:
            delta = {name: 0.0 for name in baseline.metrics()}
            component_kind = "synthetic_neutral_missing"
        else:
            delta = {
                name: effect.metrics()[name] - baseline.metrics()[name]
                for name in baseline.metrics()
            }
            component_kind = "direct_B2_seed_row"
        interaction = (
            self._interaction(baseline, sequence[-1], recipe.id)
            if sequence else {name: 0.0 for name in baseline.metrics()}
        )
        if not sequence and effect is not None:
            next_metrics = effect.metrics()
            result_kind = "direct_B2_seed_row"
        else:
            next_metrics = {
                MetricName.WNS_NS.value: round(
                    parent_metrics[MetricName.WNS_NS.value]
                    + delta[MetricName.WNS_NS.value]
                    + interaction[MetricName.WNS_NS.value], 6
                ),
                MetricName.PACKED_LUTS.value: float(max(1, round(
                    parent_metrics[MetricName.PACKED_LUTS.value]
                    + delta[MetricName.PACKED_LUTS.value]
                    + interaction[MetricName.PACKED_LUTS.value]
                ))),
                MetricName.DYNAMIC_POWER_W.value: max(0.0, round(
                    parent_metrics[MetricName.DYNAMIC_POWER_W.value]
                    + delta[MetricName.DYNAMIC_POWER_W.value]
                    + interaction[MetricName.DYNAMIC_POWER_W.value], 6
                )),
            }
            result_kind = (
                "synthetic_neutral_missing" if not sequence
                else "synthetic_composition"
            )
        step_source = {
            "recipe_id": recipe.id,
            "seed_label": seed_label,
            "component_kind": component_kind,
            "result_kind": result_kind,
            "seed_row_line": effect.line if effect is not None else None,
            "seed_run": effect.run if effect is not None else None,
            "seed_run_id": effect.run_id if effect is not None else None,
            "seed_source_csv": effect.source_csv if effect is not None else None,
            "delta_from_B2": delta,
            "interaction_from_previous_step": interaction,
            "missing_observation_assumption": (
                "unchanged metrics; no hardware observation exists"
                if effect is None else None
            ),
        }
        next_payload = {
            **payload,
            "metrics": next_metrics,
            "sequence": [*sequence, recipe.id],
            "steps": [*steps, step_source],
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        output = run_dir / "fake_output.json"
        _atomic_bytes(output, _json_bytes(next_payload))
        output_state = self.store.import_state(
            output,
            design=state.design,
            parent_id=state.id,
            made_by_run_id=run_id,
            fake=True,
        )
        evidence = self._evidence(output_state, next_payload, run_dir=run_dir)
        claims_values: dict[str, str | int | float | bool] = {
            "fake": True,
            "effect_kind": result_kind,
            "component_kind": component_kind,
            "seed_table_sha256": self.seed_sha256,
            "seed_label": seed_label,
            "wall_s_method": "synthetic fixed 1.0 s",
            "interaction_fraction": self.interaction_fraction,
        }
        if effect is not None:
            claims_values["seed_row_line"] = effect.line
            claims_values["seed_run_id"] = effect.run_id
            claims_values["seed_source_csv"] = effect.source_csv
        else:
            claims_values["missing_observation_assumption"] = (
                "neutral effect; no direct B2 row"
            )
        claims = Claims(
            id=f"{run_id}:claims",
            run_id=run_id,
            values=claims_values,
            fake=True,
        )
        run = RunRecord(
            id=run_id,
            recipe_id=recipe.id,
            recipe_version=recipe.version,
            status=RunStatus.OK,
            exit_code=0,
            wall_s=1.0,
            input_sha256_before=state.id,
            input_sha256_after=state.id,
            command=("fake_backend", recipe.id),
            tool_versions={
                "fake_model": FAKE_MODEL_VERSION,
                "seed_sha256": self.seed_sha256,
            },
            fake=True,
        )
        return RunOutcome(run=run, state=output_state, evidence=evidence, claims=claims)
