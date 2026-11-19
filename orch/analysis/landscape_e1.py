"""Read-only E1 landscape analysis of the five training designs.

The loader accepts only the pinned worker_fitting_train data root. It reads the
checksummed D1/D2 journal prefix and independent meter Evidence; it never opens
DCPs or held-out data; it validates saved AMD reports and invokes no tool.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import yaml

from orch.engine.journal import Journal
from scripts.index_worker_fitting_dcps import _verification

PROJECT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = (PROJECT / "data/sweeps/worker_fitting_train").resolve()
TRAIN_DESIGNS = (
    "vexriscv_re-place", "rosetta_spam-filter", "rosetta_3d-rendering",
    "logicnets_jscl", "finn_radioml",
)
REPORT = PROJECT / "notes/e1_training_landscape.md"
SCATTER_DIR = PROJECT / "notes/e1_scatter"
RUN_ID = re.compile(r"run_[0-9a-f]{32}\Z")
STATE_ID = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class Metrics:
    """The two meter values used by E1, with physical provenance."""

    wns_ns: float
    packed_luts: float
    state_id: str
    meter_version: str
    constraint_fingerprint: str
    clock_digest: str
    wns_method: tuple[str, str, str]
    lut_method: tuple[str, str, str]


@dataclass(frozen=True)
class Edge:
    """One latest committed D1/D2 sequence from the immutable prefix."""

    design: str
    sequence: tuple[str, ...]
    input_id: str | None
    output_id: str | None
    run_id: str | None
    run_status: str | None
    physically_valid: bool
    metrics: Metrics | None
    amd_status: str = "unknown"

    @property
    def eligible(self) -> bool:
        """A provisional candidate excludes known AMD FAILs."""
        return (
            self.physically_valid and self.run_status == "ok"
            and self.metrics is not None and self.amd_status != "fail"
        )


@dataclass(frozen=True)
class Scenario:
    """A start-relative timing or area brief variant."""

    label: str
    worker: str
    lut_cap_pct: float | None = None
    wns_floor_ns: float | None = None

    def valid(self, candidate: Metrics, start: Metrics) -> bool:
        """Apply the same exact comparisons as the contract gate."""
        if self.worker == "timing":
            assert self.lut_cap_pct is not None
            return candidate.packed_luts <= start.packed_luts * (
                1 + self.lut_cap_pct / 100
            )
        assert self.wns_floor_ns is not None
        return candidate.wns_ns >= start.wns_ns - self.wns_floor_ns

    def value(self, candidate: Metrics) -> float:
        """Return an increasing objective score for ranking."""
        return candidate.wns_ns if self.worker == "timing" else -candidate.packed_luts


SCENARIOS = (
    Scenario("T cap +0.5%", "timing", lut_cap_pct=0.5),
    Scenario("T cap +1% (default)", "timing", lut_cap_pct=1),
    Scenario("T cap +2%", "timing", lut_cap_pct=2),
    Scenario("T cap +5%", "timing", lut_cap_pct=5),
    Scenario("A floor -0.05 ns", "area", wns_floor_ns=0.05),
    Scenario("A floor -0.1 ns (default)", "area", wns_floor_ns=0.1),
    Scenario("A floor -0.2 ns", "area", wns_floor_ns=0.2),
)


@dataclass(frozen=True)
class Snapshot:
    """A fixed, checked D1/D2 prefix and its meter records."""

    starts: dict[str, Metrics]
    edges: dict[tuple[str, tuple[str, ...]], Edge]
    recipes: dict[str, tuple[str, ...]]
    noise: dict[str, dict[str, float]]
    journal_prefix_sha256: str
    journal_edge_count: int
    source_sha256: dict[str, str]

    def worker_of(self, recipe_id: str) -> str:
        """Find a recipe's signed worker set."""
        for worker, ids in self.recipes.items():
            if recipe_id in ids:
                return worker
        raise ValueError(f"recipe is not in a signed worker set: {recipe_id}")


@dataclass(frozen=True)
class Choice:
    """A best feasible state and the path interpretation."""

    edge: Edge | None
    metrics: Metrics
    sequence: tuple[str, ...]
    cross_worker: bool
    cross_worker_needed: bool
    intermediate_violates: bool


def require_training_root(data_root: Path | str) -> Path:
    """Refuse every data root except the single training allow-list entry."""
    actual = Path(data_root).expanduser().resolve(strict=True)
    if actual != TRAIN_ROOT:
        raise ValueError("data root is not the training allow-list root")
    return actual


def _inside(root: Path, path: Path) -> Path:
    """Reject a journal-derived path or symlink outside the training root."""
    actual = path.resolve(strict=True)
    if not actual.is_relative_to(root):
        raise ValueError("training record escapes the allow-list root")
    return actual


def _read_json(root: Path, path: Path) -> dict:
    value = json.loads(_inside(root, path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON record is not an object: {path}")
    return value


def _checked_run(root: Path, run_id: str) -> dict[str, dict]:
    """Read only a complete run whose artifact bytes match its marker."""
    directory = root / "runs" / run_id
    marker = _read_json(root, directory / "complete.json")
    files = marker.get("files")
    if marker.get("run_id") != run_id or not isinstance(files, dict):
        raise ValueError("run completion marker differs from journal")
    if "runinfo.json" not in files:
        raise ValueError("run completion marker lacks runinfo")
    allowed = {"runinfo.json", "output_state.json", "evidence.json", "claims.json"}
    result = {}
    for name, digest in files.items():
        if name not in allowed or not isinstance(digest, str) or not STATE_ID.fullmatch(digest):
            raise ValueError("run completion marker has an invalid artifact")
        raw = _inside(root, directory / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"run artifact checksum mismatch: {name}")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"run artifact is not an object: {name}")
        result[name] = value
    return result


def _numeric(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} is not a finite meter number")
    return float(value)


def _method(row: dict, label: str) -> tuple[str, str, str]:
    fields = (row.get("unit"), row.get("method"), row.get("basis"))
    if not all(isinstance(item, str) and item for item in fields):
        raise ValueError(f"{label} lacks meter method provenance")
    return fields


def _metrics(evidence: dict) -> Metrics:
    values = evidence.get("metrics")
    if not isinstance(values, dict):
        raise ValueError("evidence lacks metrics")
    wns = values.get("wns_ns")
    luts = values.get("packed_luts")
    if not isinstance(wns, dict) or not isinstance(luts, dict):
        raise ValueError("evidence lacks WNS or packed LUTs")
    state_id = evidence.get("state_id")
    if not isinstance(state_id, str) or not STATE_ID.fullmatch(state_id):
        raise ValueError("evidence state ID is invalid")
    if evidence.get("fake") is not False:
        raise ValueError("E1 refuses fake evidence")
    if evidence.get("routed") is not True or evidence.get("route_errors") != 0:
        raise ValueError("E1 evidence is not fully routed")
    if evidence.get("drc_errors") != 0:
        raise ValueError("E1 evidence has DRC errors")
    for name in ("meter_version", "constraint_fingerprint", "clock_digest"):
        if not isinstance(evidence.get(name), str) or not evidence[name]:
            raise ValueError(f"evidence lacks {name}")
    return Metrics(
        wns_ns=_numeric(wns.get("value"), "WNS"),
        packed_luts=_numeric(luts.get("value"), "packed LUTs"),
        state_id=state_id,
        meter_version=evidence["meter_version"],
        constraint_fingerprint=evidence["constraint_fingerprint"],
        clock_digest=evidence["clock_digest"],
        wns_method=_method(wns, "WNS"),
        lut_method=_method(luts, "packed LUTs"),
    )


def _same_basis(start: Metrics, output: Metrics) -> None:
    """A grid comparison must use the same meter and constraints as B2."""
    if (
        output.meter_version != start.meter_version
        or output.wns_method != start.wns_method
        or output.lut_method != start.lut_method
        or output.constraint_fingerprint != start.constraint_fingerprint
        or output.clock_digest != start.clock_digest
    ):
        raise ValueError("output Evidence basis differs from its B2 start")


def _journal_prefix(root: Path, *, final: bool = False) -> tuple[list[dict], str]:
    """Read a locked, checksummed journal through D2 or completed D3."""
    path = _inside(root, root / "episodes/worker_fitting_train/journal.jsonl")
    lock = _inside(root, path.parent / "journal.lock")
    with lock.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            raw = path.read_bytes()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    events, _ = Journal._parse(raw)
    stop = next(
        (index for index, event in enumerate(events)
         if event.get("type") == "phase_complete"
         and event.get("phase") == ("D3" if final else "D2")),
        None,
    )
    if stop is None:
        raise ValueError(
            ("D3" if final else "D2") + " has not committed a complete journal checkpoint"
        )
    lines = raw.splitlines(keepends=True)
    prefix = b"".join(lines[: stop + 1])
    return events[: stop + 1], hashlib.sha256(prefix).hexdigest()


def _start_evidence(root: Path, state_id: str, evidence_id: str) -> Metrics:
    if not STATE_ID.fullmatch(state_id):
        raise ValueError("B2 state ID is invalid")
    directory = _inside(root, root / "measurements" / state_id)
    matches = []
    for path in directory.glob("*/evidence.json"):
        record = _read_json(root, path)
        if record.get("id") == evidence_id:
            matches.append(_metrics(record))
    if len(matches) != 1 or matches[0].state_id != state_id:
        raise ValueError("unique measured B2 Evidence was not found")
    return matches[0]


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signed_sets(configuration: dict) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """Pin the exact signed recipe IDs and default brief terms."""
    config_dir = PROJECT / "configs"
    sets_path = config_dir / "recipe_sets.yaml"
    brief_t = config_dir / "briefs/timing.yaml"
    brief_a = config_dir / "briefs/area.yaml"
    signed = yaml.safe_load(sets_path.read_text(encoding="utf-8"))
    if signed.get("signed_off") is not True:
        raise ValueError("recipe sets are not signed")
    ids = {
        worker: tuple(signed["worker_sets"][worker])
        for worker in ("timing", "area")
    }
    journal_sets = configuration.get("recipe_sets", {})
    for worker, expected in ids.items():
        recorded = tuple(row["id"] for row in journal_sets.get(worker, ()))
        if expected != recorded or len(expected) != 5:
            raise ValueError(f"{worker} recipes differ from the D1/D2 journal")
    briefs = {
        "timing": yaml.safe_load(brief_t.read_text(encoding="utf-8")),
        "area": yaml.safe_load(brief_a.read_text(encoding="utf-8")),
    }
    if briefs["timing"].get("terms") != [{
        "id": "packed_lut_cap", "metric": "packed_luts",
        "comparison": "<=", "factor": 1.01, "offset": 0.0
    }] or briefs["area"].get("terms") != [{
        "id": "wns_floor", "metric": "wns_ns",
        "comparison": ">=", "factor": 1.0, "offset": -0.1
    }]:
        raise ValueError("default brief terms differ from the E1 grid")
    return ids, {
        "recipe_sets.yaml": _source_hash(sets_path),
        "briefs/timing.yaml": _source_hash(brief_t),
        "briefs/area.yaml": _source_hash(brief_a),
    }


def load_training(
    data_root: Path | str = TRAIN_ROOT, *, final: bool = False,
) -> Snapshot:
    """Load checked training D1/D2, or all completed AMD-PASS D1-D3."""
    root = require_training_root(data_root)
    events, prefix_hash = (
        _journal_prefix(root, final=True) if final else _journal_prefix(root)
    )
    if not events or events[0].get("type") != "configuration":
        raise ValueError("training journal lacks its configuration")
    config = events[0]["configuration"]
    if config.get("fake") is not False or set(config.get("designs", {})) != set(TRAIN_DESIGNS):
        raise ValueError("journal is not the five-design real training campaign")
    recipes, hashes = _signed_sets(config)
    hashes["verification/plan.json"] = _source_hash(
        _inside(root, root / "verification/plan.json")
    )
    all_ids = set(recipes["timing"] + recipes["area"])
    starts = {}
    noise = {}
    for design in TRAIN_DESIGNS:
        row = config["designs"][design]
        starts[design] = _start_evidence(
            root, row["start_state_id"], row["start_evidence_id"]
        )
        if starts[design].meter_version != config.get("meter_version"):
            raise ValueError("B2 meter version differs from journal")
        noise[design] = {
            key: _numeric(value, f"{design} {key} noise")
            for key, value in row.get("noise", {}).items()
        }
        if any(value < 0 for value in noise[design].values()):
            raise ValueError("noise band is negative")
    edges = {}
    edge_events = [event["edge"] for event in events
                   if event.get("type") == "edge"]
    for row in edge_events:
        design = row.get("design")
        sequence = tuple(row.get("sequence", ()))
        if (
            design not in TRAIN_DESIGNS
            or not 1 <= len(sequence) <= (4 if final else 2)
            or any(recipe_id not in all_ids for recipe_id in sequence)
            or row.get("fake") is not False
        ):
            raise ValueError("training journal contains an out-of-scope edge")
        key = (design, sequence)
        input_id = row.get("input_state_id")
        output_id = row.get("output_state_id")
        run_id = row.get("run_id")
        if input_id is not None and (
            not isinstance(input_id, str) or not STATE_ID.fullmatch(input_id)
        ):
            raise ValueError("edge input ID is invalid")
        if output_id is not None and (
            not isinstance(output_id, str) or not STATE_ID.fullmatch(output_id)
        ):
            raise ValueError("edge output ID is invalid")
        if run_id is not None and (
            not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id)
        ):
            raise ValueError("edge run ID is invalid")
        metrics = None
        if run_id is not None:
            artifacts = _checked_run(root, run_id)
            run = artifacts["runinfo.json"]
            if (
                run.get("id") != run_id
                or run.get("recipe_id") != sequence[-1]
                or run.get("input_sha256_before") != input_id
                or run.get("status") != row.get("run_status")
                or run.get("fake") is not False
            ):
                raise ValueError("journal edge differs from its RunRecord")
            if output_id is not None and row.get("physically_valid"):
                if artifacts.get("output_state.json", {}).get("id") != output_id:
                    raise ValueError("run output state differs from journal")
                if "evidence.json" not in artifacts:
                    raise ValueError("physically valid run lacks checked Evidence")
                metrics = _metrics(artifacts["evidence.json"])
                if metrics.state_id != output_id:
                    raise ValueError("edge output differs from meter Evidence")
                _same_basis(starts[design], metrics)
        if row.get("physically_valid") and (
            row.get("run_status") != "ok" or metrics is None
        ):
            raise ValueError("physically valid edge lacks an OK measured output")
        edges[key] = Edge(
            design=design, sequence=sequence, input_id=input_id,
            output_id=output_id, run_id=run_id,
            run_status=row.get("run_status"),
            physically_valid=bool(row.get("physically_valid")), metrics=metrics,
        )
    for (design, sequence), edge in edges.items():
        parent_id = (
            starts[design].state_id if len(sequence) == 1
            else edges[(design, sequence[:-1])].output_id
        )
        if edge.input_id is not None and edge.input_id != parent_id:
            raise ValueError("edge input differs from its parent output")
    # The index is advisory in provisional mode: known FAIL is excluded,
    # while missing/unknown verification remains explicitly provisional.
    index_path = root / "by_design/index.csv"
    index = {}
    if index_path.is_file():
        with _inside(root, index_path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["design"] not in TRAIN_DESIGNS:
                    raise ValueError("training index contains an unlisted design")
                if row["run_id"]:
                    index[(row["design"], row["run_id"])] = row["equivalence_status"].lower()
    checked_states = {}
    for key, edge in list(edges.items()):
        if edge.run_id is None or edge.output_id is None:
            continue
        state_key = (edge.design, edge.output_id)
        if state_key not in checked_states:
            status, _ = _verification(
                edge.design, edge.output_id, starts[edge.design].state_id
            )
            checked_states[state_key] = (
                "unknown" if status == "UNVERIFIED" else status.lower()
            )
        status = checked_states[state_key]
        if final and edge.physically_valid and status != "pass":
            raise ValueError("FINAL landscape requires AMD PASS for every saved output")
        indexed = index.get((edge.design, edge.run_id), "unknown")
        if indexed not in ("", "unknown", "unverified", status):
            raise ValueError("training index differs from canonical AMD record")
        edges[key] = replace(edge, amd_status=status)
    # Recheck the two default brief flags stored in the journal.
    for row in edge_events:
        edge = edges[(row["design"], tuple(row["sequence"]))]
        if not edge.eligible:
            continue
        start = starts[edge.design]
        assert edge.metrics is not None
        expected = {
            "brief_t": SCENARIOS[1].valid(edge.metrics, start),
            "brief_a": SCENARIOS[5].valid(edge.metrics, start),
        }
        if any(row.get("brief_valid", {}).get(name) != flag
               for name, flag in expected.items()):
            raise ValueError("default brief differs from committed gate verdict")
    if final and any(
        not edge.eligible or edge.amd_status != "pass" for edge in edges.values()
    ):
        raise ValueError("FINAL landscape has a non-PASS or unmeasured edge")
    return Snapshot(
        starts=starts, edges=edges, recipes=recipes, noise=noise,
        journal_prefix_sha256=prefix_hash, journal_edge_count=len(edge_events),
        source_sha256=hashes,
    )


def _points(snapshot: Snapshot, design: str) -> list[tuple[tuple[str, ...], Metrics, Edge | None]]:
    """Include B2 so within-two-step best never worsens a start."""
    points = [((), snapshot.starts[design], None)]
    points.extend(
        (sequence, edge.metrics, edge)
        for (name, sequence), edge in snapshot.edges.items()
        if name == design and edge.eligible and edge.metrics is not None
    )
    return points


def _cross(snapshot: Snapshot, sequence: tuple[str, ...]) -> bool:
    return len({snapshot.worker_of(recipe_id) for recipe_id in sequence}) > 1


def _repeat_unavailable(sequence: tuple[str, ...]) -> bool:
    """The frozen worker will not reapply a recipe to its own output state."""
    return any(left == right for left, right in zip(sequence, sequence[1:]))


def best_choice(
    snapshot: Snapshot, design: str, scenario: Scenario, *,
    allow_repeat: bool = True, max_depth: int | None = None,
) -> Choice:
    """Rank feasible states by brief objective, then shorter path and ID."""
    if design not in snapshot.starts:
        raise ValueError("design is outside the training snapshot")
    start = snapshot.starts[design]
    feasible = [
        (sequence, metrics, edge)
        for sequence, metrics, edge in _points(snapshot, design)
        if scenario.valid(metrics, start)
        and (max_depth is None or len(sequence) <= max_depth)
        and (allow_repeat or not _repeat_unavailable(sequence))
    ]
    if not feasible:
        raise ValueError("B2 is missing from its own feasible set")
    sequence, metrics, edge = min(
        feasible,
        key=lambda row: (
            -scenario.value(row[1]), len(row[0]), row[0], row[1].state_id
        ),
    )
    cross = _cross(snapshot, sequence)
    score = scenario.value(metrics)
    needed = cross and not any(
        not _cross(snapshot, path) and scenario.value(candidate) == score
        for path, candidate, _ in feasible
    )
    intermediate_violates = False
    if len(sequence) >= 2:
        for depth in range(1, len(sequence)):
            parent = snapshot.edges[(design, sequence[:depth])]
            if parent.metrics is None or not parent.eligible:
                raise ValueError("selected path lacks a valid intermediate")
            intermediate_violates |= not scenario.valid(parent.metrics, start)
    return Choice(
        edge=edge, metrics=metrics, sequence=sequence,
        cross_worker=cross, cross_worker_needed=needed,
        intermediate_violates=intermediate_violates,
    )


def noise_flags(snapshot: Snapshot, design: str, metrics: Metrics) -> tuple[bool | None, bool | None]:
    """Return start-relative timing and area noise flags; None means unmeasured."""
    start = snapshot.starts[design]
    bands = snapshot.noise[design]
    wns = bands.get("wns_ns")
    luts = bands.get("packed_luts")
    return (
        abs(metrics.wns_ns - start.wns_ns) <= wns if wns is not None else None,
        abs(metrics.packed_luts - start.packed_luts) <= luts if luts is not None else None,
    )


def fixed_baselines(
    snapshot: Snapshot, design: str, timing_beta_best: str, area_beta_best: str
) -> list[tuple[str, tuple[str, ...], Edge]]:
    """Return every fixed single recipe plus the two beta-best orders."""
    if timing_beta_best not in snapshot.recipes["timing"]:
        raise ValueError("timing beta-best recipe is not signed")
    if area_beta_best not in snapshot.recipes["area"]:
        raise ValueError("area beta-best recipe is not signed")
    paths = [
        (recipe_id, (recipe_id,))
        for recipe_id in snapshot.recipes["timing"] + snapshot.recipes["area"]
    ]
    paths.extend((
        ("A then B", (timing_beta_best, area_beta_best)),
        ("B then A", (area_beta_best, timing_beta_best)),
    ))
    result = []
    for label, sequence in paths:
        key = (design, sequence)
        if key not in snapshot.edges:
            raise ValueError(f"fixed baseline was not in D1/D2: {key}")
        result.append((label, sequence, snapshot.edges[key]))
    return result


def scatter_rows(snapshot: Snapshot, design: str) -> list[dict[str, object]]:
    """One row per exact checkpoint, with a shortest representative path."""
    representative = {}
    for sequence, metrics, edge in _points(snapshot, design):
        old = representative.get(metrics.state_id)
        if old is not None:
            old_path, old_metrics, _ = old
            if (
                old_metrics.wns_ns != metrics.wns_ns
                or old_metrics.packed_luts != metrics.packed_luts
            ):
                raise ValueError("one DCP hash has conflicting meter values")
            if (len(old_path), old_path) <= (len(sequence), sequence):
                continue
        representative[metrics.state_id] = (sequence, metrics, edge)
    rows = []
    start = snapshot.starts[design]
    for state_id, (sequence, metrics, edge) in sorted(representative.items()):
        noise_wns, noise_luts = noise_flags(snapshot, design, metrics)
        rows.append({
            "design": design, "state_id": state_id, "sequence": " > ".join(sequence) or "B2",
            "depth": len(sequence), "worker_path": " > ".join(
                snapshot.worker_of(recipe) for recipe in sequence
            ) or "B2", "wns_ns": metrics.wns_ns,
            "packed_luts": metrics.packed_luts,
            "wns_delta_ns": metrics.wns_ns - start.wns_ns,
            "packed_lut_delta": metrics.packed_luts - start.packed_luts,
            "brief_t_valid": SCENARIOS[1].valid(metrics, start),
            "brief_a_valid": SCENARIOS[5].valid(metrics, start),
            "within_wns_noise": noise_wns, "within_lut_noise": noise_luts,
            "amd_status": edge.amd_status if edge else "reference",
            "representative_path_policy_available": not _repeat_unavailable(sequence),
        })
    return rows


def _write_scatter_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_scatter(
    path: Path, rows: list[dict[str, object]], start: Metrics,
    *, label: str = "PROVISIONAL", max_depth: int = 2,
) -> bool:
    """Write an optional raster scatter using an already installed matplotlib."""
    import importlib.util

    if importlib.util.find_spec("matplotlib") is None:
        return False
    os.environ.setdefault(
        "MPLCONFIGDIR", str(PROJECT / "data/scratch_landscape/mplconfig")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 5))
    for depth in range(max_depth + 1):
        part = [row for row in rows if row["depth"] == depth]
        if part:
            axis.scatter(
                [row["packed_luts"] for row in part],
                [row["wns_ns"] for row in part],
                label=f"depth {depth}", s=20 if depth else 80, alpha=0.75,
            )
    axis.axvline(start.packed_luts * 1.01, color="grey", linestyle="--", linewidth=1)
    axis.axhline(start.wns_ns - 0.1, color="grey", linestyle=":", linewidth=1)
    axis.set_xlabel("Packed CLB LUTs (meter)")
    axis.set_ylabel("WNS (ns, meter)")
    axis.set_title(f"{label} training D1/D{max_depth} landscape")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def _flag(value: bool | None) -> str:
    return "yes" if value is True else "no" if value is False else "unknown"


def _path(sequence: tuple[str, ...]) -> str:
    return " > ".join(sequence) if sequence else "B2"


def _null_provenance(
    snapshot: Snapshot, timing_beta_best: str, area_beta_best: str
) -> str:
    """Verify fixed pair IDs against one frozen provisional null snapshot."""
    records = {}
    hashes = {}
    for worker, label, selected in (
        ("timing", "A", timing_beta_best),
        ("area", "B", area_beta_best),
    ):
        path = PROJECT / "configs/workers" / f"{label}_null_PROVISIONAL.json"
        raw = path.read_bytes()
        row = json.loads(raw)
        betas = row.get("beta_pct_by_recipe")
        if (
            row.get("schema_version") != "worker-null-v1-provisional"
            or row.get("status") != "PROVISIONAL"
            or row.get("worker") != worker
            or row.get("selected_recipe_id") != selected
            or tuple(row.get("training_designs", ())) != TRAIN_DESIGNS
            or not isinstance(betas, dict)
            or set(betas) != set(snapshot.recipes[worker])
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in betas.values())
        ):
            raise ValueError(f"{label} provisional null record is inconsistent")
        if min(betas, key=lambda key: (-betas[key], key)) != selected:
            raise ValueError(f"{label} fixed baseline is not beta-best")
        records[worker] = row
        hashes[label] = hashlib.sha256(raw).hexdigest()
    if (
        records["timing"].get("journal_snapshot_sha256")
        != records["area"].get("journal_snapshot_sha256")
        or records["timing"].get("feature_spec_sha256")
        != records["area"].get("feature_spec_sha256")
        or records["timing"].get("feature_spec_sha256")
        != _source_hash(PROJECT / "configs/worker_features.yaml")
        or records["timing"].get("created_at_utc")
        != records["area"].get("created_at_utc")
    ):
        raise ValueError("fixed baseline null records use different frozen inputs")
    return (
        f"These choices use the PROVISIONAL null snapshot at "
        f"{records['timing']['created_at_utc']} (journal SHA-256 "
        f"{records['timing']['journal_snapshot_sha256']}; "
        f"A JSON SHA-256 {hashes['A']}; B JSON SHA-256 {hashes['B']}). "
        f"That ranking includes available D3 edges; E1 outcomes below use "
        f"only the fixed D1/D2 prefix."
    )


def render_report(
    snapshot: Snapshot, timing_beta_best: str, area_beta_best: str,
    scatter_files: dict[str, tuple[str, str | None]],
    null_provenance: str,
) -> str:
    """Render all brief/grid winners and every fixed baseline from Evidence."""
    default_rows = []
    grid_rows = []
    baseline_sections = []
    cross_count = violation_count = 0
    unavailable_default = unavailable_grid = 0
    for design in TRAIN_DESIGNS:
        start = snapshot.starts[design]
        for scenario in SCENARIOS:
            choice = best_choice(snapshot, design, scenario)
            repeated = _repeat_unavailable(choice.sequence)
            unavailable_grid += repeated
            alternative = (
                best_choice(snapshot, design, scenario, allow_repeat=False)
                if repeated else None
            )
            policy_choice = alternative or choice
            alternative_label = (
                f"{_path(alternative.sequence)} "
                f"[{alternative.metrics.state_id[:12]}, "
                f"{alternative.metrics.wns_ns:.3f} ns, "
                f"{alternative.metrics.packed_luts:.0f} LUT]"
                if alternative else "same winner"
            )
            noise_wns, noise_luts = noise_flags(snapshot, design, choice.metrics)
            if "(default)" in scenario.label:
                unavailable_default += repeated
                cross_count += choice.cross_worker_needed
                violation_count += choice.intermediate_violates
                default_rows.append(
                    f"| {design} | {scenario.worker.upper()} | "
                    f"{choice.metrics.state_id[:12]} | {_path(choice.sequence)} | "
                    f"{choice.metrics.wns_ns:.3f} | {choice.metrics.packed_luts:.0f} | "
                    f"{_flag(choice.cross_worker_needed)} | "
                    f"{_flag(policy_choice.cross_worker_needed)} | "
                    f"{_flag(choice.intermediate_violates)} | "
                    f"{_flag(not repeated)} | {alternative_label} | "
                    f"{_flag(noise_wns)}/{_flag(noise_luts)} |"
                )
            grid_rows.append(
                f"| {design} | {scenario.label} | {choice.metrics.state_id[:12]} | "
                f"{_path(choice.sequence)} | {choice.metrics.wns_ns:.3f} | "
                f"{choice.metrics.packed_luts:.0f} | "
                f"{_flag(choice.cross_worker_needed)} | "
                f"{_flag(policy_choice.cross_worker_needed)} | "
                f"{_flag(choice.intermediate_violates)} | "
                f"{_flag(not repeated)} | {alternative_label} |"
            )
        baseline_sections.extend((
            f"### {design}",
            "",
            "| Baseline | Sequence | WNS ns | Packed LUTs | T valid | A valid | WNS/LUT within noise | AMD |",
            "|---|---|---:|---:|---|---|---|---|",
        ))
        for label, sequence, edge in fixed_baselines(
            snapshot, design, timing_beta_best, area_beta_best
        ):
            if edge.eligible:
                assert edge.metrics is not None
                metrics = edge.metrics
                flags = noise_flags(snapshot, design, metrics)
                wns = f"{metrics.wns_ns:.3f}"
                luts = f"{metrics.packed_luts:.0f}"
                t_valid = _flag(SCENARIOS[1].valid(metrics, start))
                a_valid = _flag(SCENARIOS[5].valid(metrics, start))
                noise = f"{_flag(flags[0])}/{_flag(flags[1])}"
            else:
                wns = luts = "—"
                t_valid = a_valid = noise = "no/unknown"
            baseline_sections.append(
                f"| {label} | {_path(sequence)} | {wns} | {luts} | "
                f"{t_valid} | {a_valid} | {noise} | {edge.amd_status} |"
            )
        baseline_sections.append("")
    amd_counts = {
        status: sum(edge.amd_status == status for edge in snapshot.edges.values())
        for status in ("pass", "fail", "unknown")
    }
    sources = "; ".join(
        f"{name} SHA-256 {digest}"
        for name, digest in sorted(snapshot.source_sha256.items())
    )
    scatter_links = [
        f"- {design}: [CSV](e1_scatter/{files[0]})"
        + (f", [PNG](e1_scatter/{files[1]})" if files[1] else "")
        for design, files in scatter_files.items()
    ]
    lines = [
        "# PROVISIONAL E1 training landscape",
        "",
        "## Summary",
        "",
        f"**PROVISIONAL, computed from recorded Evidence.** The fixed D1/D2 journal "
        f"prefix contains {snapshot.journal_edge_count} attempted edges on five training "
        f"designs. Under the default briefs, {cross_count}/10 best choices require a "
        f"cross-worker sequence and {violation_count}/10 traverse a brief-violating "
        f"intermediate. {unavailable_default}/10 default and "
        f"{unavailable_grid}/35 grid winners repeat their own recipe, so the "
        f"frozen worker cannot choose those paths. The tables show the best "
        f"no-repeat alternative when this happens.",
        "",
        f"**Interpretation.** Brief T maximizes WNS under a start-relative packed-LUT "
        f"cap; Brief A minimizes packed LUTs above a start-relative WNS floor. "
        f"The +1% LUT cap and -0.1 ns floor are the signed default briefs. B2 is "
        f"eligible as a zero-step result. A cross-worker path is needed only when "
        f"no equally good feasible within-worker or single-step result exists. "
        f"An intermediate violation is judged against the same start-relative "
        f"scenario as its final state. The exhaustive D2 data includes "
        f"same-recipe repeats; policy availability marks the frozen no-repeat "
        f"rule. Cross-worker necessity is shown for both the exhaustive "
        f"set and its no-repeat policy subset. Noise flags mean absolute delta from B2 "
        f"does not exceed the campaign's recorded band; unknown means no band.",
        "",
        "## Default brief winners",
        "",
        "| Design | Brief | State SHA prefix | Sequence | WNS ns | Packed LUTs | Cross needed exhaustive | Cross needed policy | Intermediate violation | Policy available | Best no-repeat alternative | WNS/LUT within noise |",
        "|---|---|---|---|---:|---:|---|---|---|---|---|---|",
        *default_rows,
        "",
        "## Constraint grid winners",
        "",
        "| Design | Scenario | State SHA prefix | Sequence | WNS ns | Packed LUTs | Cross needed exhaustive | Cross needed policy | Intermediate violation | Policy available | Best no-repeat alternative |",
        "|---|---|---|---|---:|---:|---|---|---|---|---|",
        *grid_rows,
        "",
        "## Fixed baselines",
        "",
        f"Beta-best timing: {timing_beta_best}; beta-best area: {area_beta_best}. "
        f"{null_provenance} Every design has all ten single-recipe baselines "
        f"plus these two fixed orders.",
        "",
        *baseline_sections,
        "## Timing versus area scatter",
        "",
        *scatter_links,
        "",
        "## Provenance and limits",
        "",
        f"- **Verified input:** D1/D2 journal prefix SHA-256 "
        f"{snapshot.journal_prefix_sha256}; generated "
        f"{datetime.now(timezone.utc).isoformat()}.",
        f"- **Verified configuration:** {sources}.",
        f"- **AMD canonical record check:** {amd_counts['pass']} PASS, {amd_counts['fail']} FAIL, "
        f"{amd_counts['unknown']} unknown among the {len(snapshot.edges)} latest "
        f"D1/D2 sequence records, cross-checked against the readable index. "
        f"Known FAIL is excluded; unknown remains provisional.",
        "- **Limit:** This is a training-only descriptive analysis, not an estimate "
        "of policy performance on held-out designs. The current journal may continue "
        "through D3/D4; this report deliberately freezes at D2 completion.",
        "",
    ]
    return "\n".join(lines)


def _output_inside(path: Path, allowed: Path) -> Path:
    actual = path.expanduser().resolve(strict=False)
    if not actual.is_relative_to(allowed.resolve(strict=False)):
        raise ValueError("output path is outside the allowed notes directory")
    return actual


def run(
    *, data_root: Path | str = TRAIN_ROOT,
    timing_beta_best: str, area_beta_best: str,
    report: Path | str = REPORT, scatter_dir: Path | str = SCATTER_DIR,
) -> Path:
    """Produce the provisional note and per-design scatter artifacts."""
    snapshot = load_training(data_root)
    report_path = _output_inside(Path(report), PROJECT / "notes")
    scatter_path = _output_inside(Path(scatter_dir), PROJECT / "notes/e1_scatter")
    scatter_path.mkdir(parents=True, exist_ok=True)
    (PROJECT / "data/scratch_landscape").mkdir(parents=True, exist_ok=True)
    outputs = {}
    for design in TRAIN_DESIGNS:
        rows = scatter_rows(snapshot, design)
        csv_name = f"{design}.csv"
        png_name = f"{design}.png"
        _write_scatter_csv(scatter_path / csv_name, rows)
        plotted = _plot_scatter(
            scatter_path / png_name, rows, snapshot.starts[design]
        )
        outputs[design] = (csv_name, png_name if plotted else None)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    null_provenance = _null_provenance(
        snapshot, timing_beta_best, area_beta_best
    )
    report_path.write_text(
        render_report(
            snapshot, timing_beta_best, area_beta_best, outputs,
            null_provenance,
        ),
        encoding="utf-8",
    )
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=TRAIN_ROOT)
    parser.add_argument("--timing-beta-best")
    parser.add_argument("--area-beta-best")
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--scatter-dir", type=Path, default=SCATTER_DIR)
    parser.add_argument("--final", action="store_true",
                        help="require completed AMD-PASS D1-D3 and render FINAL baselines")
    args = parser.parse_args(argv)
    if args.final:
        from orch.analysis.landscape_e1_final import (
            REPORT_FINAL, SCATTER_FINAL, run_final,
        )

        print(run_final(
            data_root=args.data_root,
            report=REPORT_FINAL if args.report == REPORT else args.report,
            scatter_dir=SCATTER_FINAL if args.scatter_dir == SCATTER_DIR else args.scatter_dir,
        ))
        return 0
    if not args.timing_beta_best or not args.area_beta_best:
        parser.error("provisional E1 needs both beta-best recipe IDs")
    print(run(
        data_root=args.data_root, timing_beta_best=args.timing_beta_best,
        area_beta_best=args.area_beta_best, report=args.report,
        scatter_dir=args.scatter_dir,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

