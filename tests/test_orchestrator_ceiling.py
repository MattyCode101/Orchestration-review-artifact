"""Synthetic checks for offline mandate simulation and cache pruning."""
from __future__ import annotations

from dataclasses import dataclass

from orch.analysis.orchestrator_ceiling import Simulator, beam_search, simulate_policy
from orch.contract.brief_text import load_frozen_brief
from orch.contract.models import (
    DesignState, Evidence, Mandate, MetricName, MetricValue, Objective,
    RunOutcome, RunRecord, RunStatus, WorkerCard, WorkerSettings,
)
from orch.policies.repair_loop import RepairLoopPolicy
from orch.workers.base import GenericWorker


@dataclass(frozen=True)
class Recipe:
    id: str
    worker: str
    version: str = "synthetic-v1"
    replaces: bool = False


class Registry:
    def __init__(self, *, area_replaces: bool = False) -> None:
        self.rows = {
            "ta": Recipe("ta", "timing"),
            "tb": Recipe("tb", "timing"),
            "aa": Recipe("aa", "area", replaces=area_replaces),
        }

    def get(self, recipe_id: str) -> Recipe:
        return self.rows[recipe_id]


class Oracle:
    def __init__(self, rows: dict[tuple[str, str], RunOutcome]) -> None:
        self.rows = rows
        self.lookups: list[tuple[str, str]] = []

    def get(self, base: DesignState, recipe_id: str) -> RunOutcome | None:
        key = base.id, recipe_id
        self.lookups.append(key)
        return self.rows.get(key)


def state(letter: str, *, parent: str | None = None) -> DesignState:
    digest = letter * 64
    return DesignState(
        id=digest, design="synthetic_train", stage="routed",
        parent_id=parent, path=f"/synthetic/{digest}.dcp", fake=False,
    )


def evidence(state_id: str, wns: float, luts: float) -> Evidence:
    return Evidence(
        id=f"e_{state_id[:12]}", state_id=state_id,
        metrics={
            MetricName.WNS_NS: MetricValue(
                value=wns, unit="ns", method="synthetic", basis="routed"
            ),
            MetricName.PACKED_LUTS: MetricValue(
                value=luts, unit="count", method="synthetic", basis="routed"
            ),
        },
        routed=True, route_errors=0, drc_errors=0,
        constraint_fingerprint="synthetic-xdc", clock_digest="synthetic-clock",
        v_props={
            "n_DONT_TOUCH": 0, "n_IS_LOC_FIXED": 0, "n_IS_BEL_FIXED": 0,
        },
        meter_version="synthetic-meter", fake=False,
    )


def outcome(base: DesignState, recipe: str, child: DesignState, *, wns: float, luts: float) -> RunOutcome:
    return RunOutcome(
        run=RunRecord(
            id=f"run_{base.id[:6]}_{recipe}", recipe_id=recipe,
            recipe_version="synthetic-v1", status=RunStatus.OK,
            input_sha256_before=base.id, input_sha256_after=base.id,
            fake=False,
        ),
        state=child, evidence=evidence(child.id, wns, luts),
    )


def setup(*, area_replaces: bool = False, missing_tb: bool = False) -> tuple[Simulator, Oracle, DesignState]:
    b2 = state("a")
    ta = state("b", parent=b2.id)
    tb = state("c", parent=b2.id)
    repaired = state("d", parent=ta.id)
    registry = Registry(area_replaces=area_replaces)
    rows = {
        (b2.id, "ta"): outcome(b2, "ta", ta, wns=-0.7, luts=110),
        (ta.id, "aa"): outcome(ta, "aa", repaired, wns=-0.5, luts=99),
    }
    if not missing_tb:
        rows[(b2.id, "tb")] = outcome(b2, "tb", tb, wns=-0.8, luts=95)
    oracle = Oracle(rows)
    workers = {
        "timing": GenericWorker(
            WorkerCard(
                id="timing", version="test", recipe_set_id="timing",
                objective=Objective(metric=MetricName.WNS_NS, direction="max"),
            ),
            ("ta", "tb"),
        ),
        "area": GenericWorker(
            WorkerCard(
                id="area", version="test", recipe_set_id="area",
                objective=Objective(metric=MetricName.PACKED_LUTS, direction="min"),
            ),
            ("aa",),
        ),
    }
    brief = load_frozen_brief("T0").model_copy(update={"budget_runs": 3})
    sim = Simulator(
        brief=brief, start_state=b2, start_evidence=evidence(b2.id, -1.0, 100),
        workers=workers, registry=registry, oracle=oracle, provider=None,
    )
    return sim, oracle, b2


def mandate(sim: Simulator, worker: str, base: str, *, tries: int, preserve: bool = False) -> Mandate:
    return Mandate(
        id=f"synthetic_{worker}_{base[:8]}_{tries}_{preserve}",
        worker_id=worker, base_state_id=base,
        settings=WorkerSettings(preserve_placement=preserve, max_runs=tries),
        brief_hash=sim.brief.fingerprint(),
    )


def test_cache_miss_prunes_the_entire_mandate() -> None:
    sim, oracle, b2 = setup(missing_tb=True)
    original = sim.initial()
    successor, reason = sim.apply(original, mandate(sim, "timing", b2.id, tries=2))
    assert successor is None and reason == "uncached"
    assert len(original.visits) == 1 and not original.calls
    assert oracle.lookups == [(b2.id, "ta"), (b2.id, "tb")]


def test_uncached_policy_result_is_censored() -> None:
    sim, _, b2 = setup(missing_tb=True)
    result = simulate_policy(sim, RepairLoopPolicy(), episode_id="uncached_repair")
    assert result.censored and result.reason == "uncached"
    assert result.used == 0 and result.final.state.id == b2.id


def test_unselected_valid_attempt_is_still_final_pick_and_history_is_per_base() -> None:
    sim, _, b2 = setup()
    first, reason = sim.apply(sim.initial(), mandate(sim, "timing", b2.id, tries=2))
    assert reason == "ok" and first is not None
    assert first.current.state.id == "b" * 64
    assert first.final(sim.brief).state.id == "c" * 64
    assert len(first.visits) == 3
    again, reason = sim.apply(first, mandate(sim, "timing", b2.id, tries=1))
    assert again is None and reason == "no_recipe"
    repaired, reason = sim.apply(
        first, mandate(sim, "area", first.current.state.id, tries=1, preserve=True)
    )
    assert reason == "ok" and repaired is not None
    assert repaired.final(sim.brief).state.id == "d" * 64


def test_preserve_placement_filters_a_replacing_recipe() -> None:
    sim, _, b2 = setup(area_replaces=True)
    first, reason = sim.apply(sim.initial(), mandate(sim, "timing", b2.id, tries=2))
    assert reason == "ok" and first is not None
    result, reason = sim.apply(
        first, mandate(sim, "area", first.current.state.id, tries=1, preserve=True)
    )
    assert result is None and reason == "no_recipe"


def test_beam_and_frozen_repair_policy_find_cross_worker_repair() -> None:
    sim, _, _ = setup()
    search = beam_search(sim, width=32)
    assert search.best.final(sim.brief).state.id == "d" * 64
    assert search.stats.cache_pruned > 0
    assert not search.exact  # A missing cache branch makes this a lower bound.
    repair = simulate_policy(sim, RepairLoopPolicy(), episode_id="synthetic_repair")
    assert repair.final.state.id == "d" * 64
    assert repair.used == 3 and not repair.censored
    assert tuple(call.worker for call in repair.calls) == ("timing", "area")


def test_beam_discard_marks_reachable_result_as_lower_bound() -> None:
    sim, _, _ = setup()
    search = beam_search(sim, width=1)
    assert search.stats.beam_discarded > 0
    assert not search.exact



def test_analysis_requires_both_approval_audits(tmp_path, monkeypatch) -> None:
    import pytest
    import orch.analysis.orchestrator_ceiling as ceiling

    monkeypatch.setattr(ceiling, "EQUIVALENCE_AUDIT", tmp_path / "missing_eq.json")
    monkeypatch.setattr(ceiling, "PROMOTION_AUDIT", tmp_path / "missing_promotion.json")
    with pytest.raises(FileNotFoundError, match="must both exist"):
        ceiling.load_approvals()


def test_cache_oracle_refuses_a_non_training_root(tmp_path) -> None:
    import pytest
    from orch.analysis.orchestrator_ceiling import PromotionSnapshotOracle

    with pytest.raises(ValueError, match="non-training pilot root"):
        PromotionSnapshotOracle(tmp_path, Registry(), "synthetic-meter", set())


def test_preflight_requires_all_six_meter_metrics(monkeypatch) -> None:
    import pytest
    import orch.analysis.orchestrator_ceiling as ceiling

    b2 = state("a")
    start = evidence(b2.id, -1.0, 100)
    row = ceiling.Known("synthetic_train", b2.id, (), start, "B2")
    monkeypatch.setattr(
        ceiling, "load_approvals",
        lambda: ("synthetic-meter", set(), "eq-hash", "promotion-hash", 94),
    )
    monkeypatch.setattr(
        ceiling, "load_known",
        lambda **_kwargs: ({"synthetic_train": (b2, start)}, [row], {}),
    )
    with pytest.raises(ValueError, match="six meter metrics"):
        ceiling.preflight()


def test_duplicate_dcp_hash_with_different_evidence_is_rejected() -> None:
    import pytest
    import orch.analysis.orchestrator_ceiling as ceiling

    b2 = state("a")
    first = ceiling.Known("synthetic_train", b2.id, (), evidence(b2.id, -1.0, 100), "B2")
    second = ceiling.Known("synthetic_train", b2.id, (), evidence(b2.id, -0.9, 100), "D1")
    with pytest.raises(ValueError, match="conflicting Evidence"):
        ceiling._check_duplicate_evidence((first, second))


def test_promotion_snapshot_ignores_a_changed_live_pointer(tmp_path, monkeypatch) -> None:
    import hashlib
    import json
    import pytest
    import orch.analysis.orchestrator_ceiling as ceiling
    from orch.engine.cache import RunCache

    root = tmp_path / "v4"
    root.mkdir()
    pilot_plan = root / "plan.json"
    pilot_plan.write_text("{}", encoding="utf-8")
    eq = tmp_path / "equivalence.json"
    eq.write_text("{}", encoding="utf-8")
    promotion = tmp_path / "promotion"
    promotion.mkdir()
    plan_path, audit_path = promotion / "plan.json", promotion / "audit.json"
    b2 = state("a")
    child = state("b", parent=b2.id)
    run_id = "run_" + "1" * 32
    key = RunCache.key(b2.id, "ta", "synthetic-v1")
    pinned = RunOutcome(
        run=RunRecord(
            id=run_id, recipe_id="ta", recipe_version="synthetic-v1",
            status=RunStatus.OK, input_sha256_before=b2.id,
            input_sha256_after=b2.id, fake=False,
        ),
        state=child, evidence=evidence(child.id, -0.7, 100),
    )
    complete = root / "runs" / run_id / "complete.json"
    complete.parent.mkdir(parents=True)
    complete.write_text('{"complete":true}', encoding="utf-8")
    omitted = [f"{index:064x}" for index in range(1, 95)]
    plan = {
        "schema_version": "training-cache-promotion-plan-v1",
        "audit_sha256": hashlib.sha256(eq.read_bytes()).hexdigest(),
        "pilot_plan_sha256": hashlib.sha256(pilot_plan.read_bytes()).hexdigest(),
        "destination": str(root), "tool_jobs_launched": 0,
        "ambiguous_keys_excluded": 94, "omitted_ambiguous_keys": omitted,
        "rows": [{
            "key": key, "alias_run_id": run_id, "design": "synthetic_train",
            "input_id": b2.id, "output_id": child.id,
            "recipe_id": "ta", "recipe_version": "synthetic-v1",
            "new_evidence_id": pinned.evidence.id,
        }],
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    audit = {
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "promoted_pointers": 1, "ambiguous_keys_excluded": 94,
        "tool_jobs_launched": 0,
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    class Store:
        def __init__(self, _root) -> None:
            pass

        def load_outcome(self, requested: str) -> RunOutcome:
            assert requested == run_id
            return pinned

    monkeypatch.setattr(ceiling, "PILOT_ROOT", root)
    monkeypatch.setattr(ceiling, "PROMOTION_PLAN", plan_path)
    monkeypatch.setattr(ceiling, "PROMOTION_AUDIT", audit_path)
    monkeypatch.setattr(ceiling, "EQUIVALENCE_AUDIT", eq)
    monkeypatch.setattr(ceiling, "TRAINING_DESIGNS", ("synthetic_train",))
    monkeypatch.setattr(ceiling, "StateStore", Store)
    oracle = ceiling.PromotionSnapshotOracle(
        root, Registry(), "synthetic-meter", {(b2.design, child.id)},
        approved_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    )
    live = root / "cache" / f"{key}.json"
    live.parent.mkdir()
    live.write_text(json.dumps({"key": key, "run_id": "run_" + "2" * 32}), encoding="utf-8")
    assert oracle.get(b2, "ta") == pinned
    oracle.require_stable_snapshot()
    complete.write_text('{"complete":false}', encoding="utf-8")
    with pytest.raises(ValueError, match="completion changed"):
        oracle.require_stable_snapshot()


def test_pre_pilot_source_scope_rejects_live_v4_journals(tmp_path, monkeypatch) -> None:
    import pytest
    import orch.analysis.orchestrator_ceiling as ceiling

    prior, live = tmp_path / "v3", tmp_path / "v4"
    prior_journal = prior / "episodes/pilot_old/journal.jsonl"
    live_journal = live / "episodes/pilot_new/journal.jsonl"
    for path in (prior_journal, live_journal):
        path.parent.mkdir(parents=True)
        path.write_text("", encoding="utf-8")
    monkeypatch.setattr(ceiling, "PRIOR_PILOT_ROOT", prior)
    monkeypatch.setattr(ceiling, "PILOT_ROOT", live)
    assert ceiling._pre_pilot_episode_journals(prior) == [prior_journal]
    with pytest.raises(ValueError, match="refuses live v4 journals"):
        ceiling._pre_pilot_episode_journals(live)


def test_selected_parent_uses_frozen_source_run_when_hash_has_multiple_depths() -> None:
    """One real selected parent is D1 even though the campaign is called D3."""
    import pytest
    import orch.analysis.orchestrator_ceiling as ceiling

    @dataclass(frozen=True)
    class Edge:
        run_id: str
        output_id: str
        eligible: bool = True
        amd_status: str = "pass"

    class Store:
        def __init__(self, rows: dict[str, RunOutcome]) -> None:
            self.rows = rows

        def load_outcome(self, run_id: str) -> RunOutcome:
            return self.rows[run_id]

    b2 = state("a")
    same = state("b", parent=b2.id)
    d1 = outcome(b2, "LATCH.auto_pblock", same, wns=-0.7, luts=100)
    d2 = outcome(same, "LATCH.phys_opt", same, wns=-0.7, luts=100)
    design = "synthetic_train"
    first = ("LATCH.auto_pblock",)
    second = (*first, "LATCH.phys_opt")
    known = {
        (design, same.id): [
            ceiling.Known(design, same.id, first, d1.evidence, "D1"),
            ceiling.Known(design, same.id, second, d2.evidence, "D2"),
        ]
    }
    edges = {
        (design, first): Edge(d1.run.id, same.id),
        (design, second): Edge(d2.run.id, same.id),
    }
    store = Store({d1.run.id: d1, d2.run.id: d2})
    plan = {"parent_hashes": [same.id]}
    selected = {
        "by_design": {
            design: {"selected_states": [{
                "hash": same.id, "dcp_sha256": same.id,
                "source_depth": 1, "source_sequence": "LATCH.auto_pblock",
                "source_run_id": d1.run.id,
            }]}
        }
    }
    paths = ceiling._selected_parent_sequences(selected, design, plan, edges, known, store)
    assert ceiling._selected_prefix(design, same.id, paths, known) == first

    record = selected["by_design"][design]["selected_states"][0]
    record.update({
        "source_depth": 2,
        "source_sequence": "LATCH.auto_pblock > LATCH.phys_opt",
        "source_run_id": d2.run.id,
    })
    paths = ceiling._selected_parent_sequences(selected, design, plan, edges, known, store)
    assert ceiling._selected_prefix(design, same.id, paths, known) == second
    record["source_run_id"] = d1.run.id
    with pytest.raises(ValueError, match="exact AMD-PASS fitting edge"):
        ceiling._selected_parent_sequences(selected, design, plan, edges, known, store)


def test_v4_plan_pins_registry_and_final_models(tmp_path) -> None:
    import hashlib
    import pytest
    import orch.analysis.orchestrator_ceiling as ceiling

    files = {
        "registry": tmp_path / "recipes/registry.yaml",
        "A_fit": tmp_path / "configs/workers/A_fit.json",
        "B_fit": tmp_path / "configs/workers/B_fit.json",
    }
    for name, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    plan = {
        "source_hashes": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in files.items()
        }
    }
    ceiling._require_v4_plan_pins(plan, tmp_path)
    files["registry"].write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="registry hash differs"):
        ceiling._require_v4_plan_pins(plan, tmp_path)


def test_copied_latest_amd_record_checks_canonical_and_raw_report(tmp_path) -> None:
    """Pilot latest records are copies, so their paths differ from canonical."""
    import hashlib
    import json
    import pytest
    import orch.analysis.orchestrator_ceiling as ceiling

    root = tmp_path
    design, start_id, output_id, fingerprint = (
        "synthetic_train", "a" * 64, "b" * 64, "f" * 64,
    )
    verification = root / "verification"
    verification.mkdir()
    plan = {"fingerprint": fingerprint, "vectors": 4, "precheck_vectors": 2}
    (verification / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    raw = {
        "overall_passed": True, "phase1_passed": True,
        "phase2_passed": True, "infrastructure_failure": False,
        "num_vectors": 4, "precheck_vectors": 2,
        "simulation_report": {
            "precheck_report": {
                "passed": True, "cycles_simulated": 2,
                "mismatch_count": 0, "protocol_mismatch_count": 0,
            },
            "full_report": {
                "passed": True, "cycles_simulated": 4,
                "mismatch_count": 0, "protocol_mismatch_count": 0,
            },
        },
    }
    raw_path = verification / "runs/raw.json"
    raw_path.parent.mkdir()
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    record = {
        "design": design, "reference_sha256": start_id,
        "output_sha256": output_id, "validator_fingerprint": fingerprint,
        "result": "pass", "raw_report_path": str(raw_path),
        "raw_report_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }
    canonical = ceiling._canonical_amd_path(root, design, output_id)
    canonical.parent.mkdir(parents=True)
    canonical.write_text(json.dumps(record), encoding="utf-8")
    latest = verification / "latest" / design / f"{output_id}.json"
    latest.parent.mkdir(parents=True)
    latest.write_bytes(canonical.read_bytes())
    assert latest.resolve() != canonical.resolve()
    ceiling._checked_amd_record(root, design, start_id, output_id, canonical)
    latest.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="pointer differs"):
        ceiling._checked_amd_record(root, design, start_id, output_id, canonical)


def test_noop_revisited_hash_signature_keeps_missing_identity() -> None:
    """A no-op recipe can visit the B2 hash with a real producer/run ID."""
    import orch.analysis.orchestrator_ceiling as ceiling

    sim, oracle, b2 = setup(missing_tb=True)
    same_hash = state("a", parent=b2.id)
    oracle.rows[(b2.id, "ta")] = outcome(
        b2, "ta", same_hash, wns=-1.0, luts=100
    )
    successor, reason = sim.apply(
        sim.initial(), mandate(sim, "timing", b2.id, tries=1)
    )
    assert reason == "ok" and successor is not None
    assert successor.visits[0].state.id == successor.visits[1].state.id
    visited = ceiling._signature(successor)[-1]
    assert visited == (
        (b2.id, (0, ""), (0, ""), 0),
        (b2.id, (1, "ta"), (1, successor.visits[1].run_id), 1),
    )
    assert beam_search(sim, width=32).stats.explored >= 2
