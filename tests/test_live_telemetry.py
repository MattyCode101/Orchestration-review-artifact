"""Live input-state telemetry for fitted workers on any design: caching, fail-closed refusals, job cap, cost."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from orch.analysis.fit_workers import extract_features, load_feature_spec
from orch.contract.models import DesignState, Evidence
from orch.workers.live_telemetry import (
    ExtractionBudget, ExtractionCapReached, LiveTelemetryProvider, TelemetryUnavailableError, total_extraction_cost,
)

ROOT = Path(__file__).resolve().parents[1]
STATE_A, STATE_B, STATE_C = "a" * 64, "b" * 64, "c" * 64
B_FEATURE_RECIPES = ("A_R_EA", "A_R_RSA_AREM", "A_R_RSA_SRLM")


def _state(state_id: str, tmp_path: Path, *, parent: str | None = None, fake: bool = False) -> DesignState:
    dcp = tmp_path / f"{state_id[:4]}.dcp"
    dcp.write_bytes(b"dcp")
    return DesignState(id=state_id, design="some_design", path=str(dcp), parent_id=parent, fake=fake)


def _evidence(state_id: str, *, fake: bool = False) -> Evidence:
    return Evidence.model_construct(id="e" + state_id[:4], state_id=state_id, metrics={}, fake=fake,
                                    meter_version="meter", routed=True, route_errors=0, drc_errors=0)


def _payload(state_id: str, *, drop: str | None = None, luts: int = 1000, srl: int = 10) -> dict:
    probe = {
        "utilization_summary": {"CLB LUTs": {"used": 900}, "CLB Registers": {"used": 800},
                                "Block RAM Tile": {"used": 0}, "DSPs": {"used": 0}, "URAM": {"used": 0}},
        "pblock_sizing_resources": {"luts": luts},
        "lut_as_shift_register": srl,
        "lut_primitive_counts": {f"LUT{i}": 10 * i for i in range(1, 7)},
        "timing": {"wns_ns": -0.1},
    }
    if drop:
        probe.pop(drop)
    return {
        "schema_version": "dcp-telemetry-v1", "fake": False, "state_sha256": state_id, "probe": probe,
        "wpath": {}, "wpath2": {}, "metrics": {"packed_luts": {"value": 900}, "wns_ns": {"value": -0.1}},
        "validity": {"v_props": {}},
        "provenance": {
            "extractor_sha256": "e" * 64, "contract_run_id": "run",
            "runner": {"wall_s": 50.0, "peak_memory_bytes": 3_000_000_000},
            "wpath": {"runner": {"wall_s": 40.0, "peak_memory_bytes": 2_000_000_000}},
            "wpath2": {"runner": {"wall_s": 45.0, "peak_memory_bytes": 2_500_000_000}},
        },
    }


class FakeExtractor:
    extractor_sha256 = "e" * 64
    env_script_sha256 = "f" * 64
    tool_version = "2025.1"

    def __init__(self, *, fail: Exception | None = None, drop: str | None = None, delay: float = 0.0) -> None:
        self.calls: list[str] = []
        self.fail, self.drop, self.delay = fail, drop, delay
        self.lock = threading.Lock()

    def extract(self, dcp_path, *, run_dir, evidence=None, design=None, run_id=None):
        with self.lock:
            self.calls.append(evidence.state_id)
        time.sleep(self.delay)
        if self.fail is not None:
            raise self.fail
        payload = _payload(evidence.state_id, drop=self.drop)
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "dcp_telemetry.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload


def _provider(tmp_path: Path, extractor=None, *, cap: int = 40, **kwargs) -> tuple[LiveTelemetryProvider, FakeExtractor]:
    extractor = extractor or FakeExtractor()
    budget = ExtractionBudget(tmp_path / "jobs.json", cap)
    provider = LiveTelemetryProvider(tmp_path / "cache", extractor=extractor, budget=budget,
                                     log_path=tmp_path / "log.jsonl", **kwargs)
    return provider, extractor


def test_the_default_required_features_are_the_three_that_worker_b_ranks_on(tmp_path) -> None:
    provider, _ = _provider(tmp_path)
    assert tuple(sorted(provider.required_recipes)) == B_FEATURE_RECIPES


def test_the_first_call_extracts_once_and_later_calls_use_the_cache_by_state_hash(tmp_path) -> None:
    provider, extractor = _provider(tmp_path)
    state = _state(STATE_A, tmp_path)
    first = provider(state, _evidence(STATE_A), None, None)
    second = provider(state, _evidence(STATE_A), None, None)
    assert extractor.calls == [STATE_A] and first == second


def test_the_cache_survives_a_new_provider_instance(tmp_path) -> None:
    first, extractor = _provider(tmp_path)
    state = _state(STATE_A, tmp_path)
    context = first(state, _evidence(STATE_A), None, None)
    again, other = _provider(tmp_path)
    assert again(state, _evidence(STATE_A), None, None) == context and other.calls == []


def test_the_context_has_the_shape_the_fitted_worker_reads(tmp_path) -> None:
    provider, _ = _provider(tmp_path)
    context = provider(_state(STATE_A, tmp_path), _evidence(STATE_A), None, None)
    assert context["telemetry"]["state_sha256"] == STATE_A and context["produced_by_recipe_id"] is None
    values = extract_features(context["telemetry"], load_feature_spec())
    assert all(values[recipe] is not None for recipe in B_FEATURE_RECIPES)
    assert context["telemetry_source"]["state_sha256"] == STATE_A


def test_the_source_record_is_the_same_on_a_miss_and_on_a_hit_so_a_journal_can_resume(tmp_path) -> None:
    provider, _ = _provider(tmp_path)
    state = _state(STATE_A, tmp_path)
    miss = provider(state, _evidence(STATE_A), None, None)["telemetry_source"]
    hit = provider(state, _evidence(STATE_A), None, None)["telemetry_source"]
    assert miss == hit and "wall_s" not in json.dumps(miss) and "cache" not in json.dumps(miss)


def test_a_failed_extraction_refuses_logs_the_reason_and_is_not_cached(tmp_path) -> None:
    provider, extractor = _provider(tmp_path, FakeExtractor(fail=RuntimeError("vivado exited 1")))
    state = _state(STATE_A, tmp_path)
    with pytest.raises(TelemetryUnavailableError, match="vivado exited 1"):
        provider(state, _evidence(STATE_A), None, None)
    events = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["event"] == "refused" and events[-1]["state_sha256"] == STATE_A
    assert "vivado exited 1" in events[-1]["reason"]
    with pytest.raises(TelemetryUnavailableError):
        provider(state, _evidence(STATE_A), None, None)
    assert extractor.calls == [STATE_A, STATE_A]            # nothing was cached from the failure


@pytest.mark.parametrize("dropped", ["lut_as_shift_register", "lut_primitive_counts", "pblock_sizing_resources"])
def test_a_record_without_a_feature_worker_b_ranks_on_is_refused_not_scored_from_the_mean(tmp_path, dropped) -> None:
    provider, _ = _provider(tmp_path, FakeExtractor(drop=dropped))
    with pytest.raises(TelemetryUnavailableError, match="A_R_"):
        provider(_state(STATE_A, tmp_path), _evidence(STATE_A), None, None)
    with pytest.raises(TelemetryUnavailableError, match="A_R_"):     # still refused from the stored record
        provider(_state(STATE_A, tmp_path), _evidence(STATE_A), None, None)


def test_a_derived_state_needs_its_producer_and_a_start_must_not_have_one(tmp_path) -> None:
    provider, _ = _provider(tmp_path)
    derived = _state(STATE_B, tmp_path, parent=STATE_A)
    with pytest.raises(TelemetryUnavailableError, match="producer"):
        provider(derived, _evidence(STATE_B), None, None)
    context = provider(derived, _evidence(STATE_B), "A_R_EA", "run_" + "0" * 32)
    assert context["produced_by_recipe_id"] == "A_R_EA"
    with pytest.raises(TelemetryUnavailableError, match="producer"):
        provider(_state(STATE_A, tmp_path), _evidence(STATE_A), "A_R_EA", "run_" + "0" * 32)


def test_a_fake_state_or_evidence_for_another_state_is_refused_before_any_extraction(tmp_path) -> None:
    provider, extractor = _provider(tmp_path)
    with pytest.raises(TelemetryUnavailableError, match="real"):
        provider(_state(STATE_A, tmp_path, fake=True), _evidence(STATE_A), None, None)
    with pytest.raises(TelemetryUnavailableError, match="Evidence"):
        provider(_state(STATE_A, tmp_path), _evidence(STATE_B), None, None)
    with pytest.raises(TelemetryUnavailableError, match="real"):
        provider(_state(STATE_A, tmp_path), _evidence(STATE_A, fake=True), None, None)
    assert extractor.calls == []


def test_the_job_cap_counts_every_extraction_including_failures_and_never_blocks_a_cached_state(tmp_path) -> None:
    provider, extractor = _provider(tmp_path, cap=2)
    a, b, c = (_state(s, tmp_path) for s in (STATE_A, STATE_B, STATE_C))
    provider(a, _evidence(STATE_A), None, None)
    provider(b, _evidence(STATE_B), None, None)
    with pytest.raises(ExtractionCapReached):
        provider(c, _evidence(STATE_C), None, None)
    assert provider(a, _evidence(STATE_A), None, None)           # cached: served
    assert extractor.calls == [STATE_A, STATE_B]
    assert provider.budget.count() == 2


def test_two_threads_asking_for_one_state_extract_it_once(tmp_path) -> None:
    provider, extractor = _provider(tmp_path, FakeExtractor(delay=0.3))
    state = _state(STATE_A, tmp_path)
    results: list[dict] = []
    threads = [threading.Thread(target=lambda: results.append(provider(state, _evidence(STATE_A), None, None)))
               for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert extractor.calls == [STATE_A] and results[0] == results[1]


def test_extraction_time_and_peak_memory_are_recorded_apart_from_the_recipe_cost(tmp_path) -> None:
    provider, _ = _provider(tmp_path)
    provider(_state(STATE_A, tmp_path), _evidence(STATE_A), None, None)
    cost = provider.cost_of(STATE_A)
    assert cost["vivado_launches"] == 3 and cost["runner_wall_s"] == [50.0, 40.0, 45.0]
    assert cost["wall_s_total"] == pytest.approx(135.0) and cost["peak_memory_bytes"] == 3_000_000_000
    assert cost["elapsed_s"] >= 0.0
    assert provider.cost_of(STATE_B) is None
    summary = total_extraction_cost(tmp_path / "cache", [STATE_A, STATE_B])
    assert summary["states_with_cost"] == 1 and summary["wall_s_total"] == pytest.approx(135.0)


def test_identity_names_the_extractor_and_does_not_change_between_calls(tmp_path) -> None:
    provider, _ = _provider(tmp_path)
    before = provider.identity()
    provider(_state(STATE_A, tmp_path), _evidence(STATE_A), None, None)
    assert provider.identity() == before
    assert before["kind"] == "live-dcp-telemetry-v1" and before["extractor_sha256"] == "e" * 64
    assert before["tool_version"] == "2025.1"


def test_a_stale_cached_record_from_another_extractor_is_refused_not_trusted(tmp_path) -> None:
    provider, _ = _provider(tmp_path)
    state = _state(STATE_A, tmp_path)
    provider(state, _evidence(STATE_A), None, None)
    other = FakeExtractor()
    other.extractor_sha256 = "9" * 64
    newer, _ = _provider(tmp_path, other)
    with pytest.raises(TelemetryUnavailableError, match="extractor"):
        newer(state, _evidence(STATE_A), None, None)


def test_the_producer_is_checked_against_the_store_when_one_is_given(tmp_path) -> None:
    state = _state(STATE_B, tmp_path, parent=STATE_A)
    evidence = _evidence(STATE_B)
    outcome = SimpleNamespace(state=state, evidence=evidence,
                              run=SimpleNamespace(recipe_id="A_R_EA", input_sha256_before=STATE_A, original_run_id=None))
    store = SimpleNamespace(load_outcome=lambda run_id: outcome)
    provider, _ = _provider(tmp_path, store=store)
    assert provider(state, evidence, "A_R_EA", "run_" + "1" * 32)
    with pytest.raises(TelemetryUnavailableError, match="producer"):
        provider(state, evidence, "A_R_RSA_MED", "run_" + "1" * 32)
