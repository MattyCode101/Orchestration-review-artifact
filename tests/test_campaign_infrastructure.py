"""Decision (d): an extraction failure or a persistent model-API failure is an infrastructure failure.

On the fake backend, with the real policies and the real harness cell runner. No tool, no network. Each failure
(1) ends the episode without a result row, (2) adds nothing to its journal, so no STOP, no schema-repair turn,
no fallback and no charged run, (3) is re-queued once at the end of its tier, and (4) resumes from the journal.
A second failure is dropped, never substituted. An exhausted extraction cap is a cap, not a failure.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import campaign_harness as h  # noqa: E402
import campaign_run as run  # noqa: E402
from orch.config import load_recipe_sets, load_worker_card  # noqa: E402
from orch.engine.apply import StepEngine  # noqa: E402
from orch.engine.cache import RunCache  # noqa: E402
from orch.engine.journal import Journal  # noqa: E402
from orch.engine.store import StateStore  # noqa: E402
from orch.policies.llm_client import LLMCallError, LLMUnavailableError  # noqa: E402
from orch.recipes.registry import load_registry  # noqa: E402
from orch.workers.base import GenericWorker  # noqa: E402
from orch.workers.fitted import FittedWorker, ScoredPlan  # noqa: E402
from orch.workers.live_telemetry import ExtractionCapReached, TelemetryUnavailableError  # noqa: E402
from scripts import test_campaign as tc  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]


class FlakyTransport:
    """The scripted model; call number `fail_on` (1-based, among attempts) raises the given error instead of replying."""

    def __init__(self, error: Exception | None = None, fail_on: set[int] | None = None) -> None:
        self.inner, self.error, self.fail_on, self.attempts = tc.ScriptedTransport(), error, fail_on or set(), 0

    def complete(self, prompt, params, schema):
        self.attempts += 1
        if self.error is not None and self.attempts in self.fail_on:
            raise self.error
        return self.inner.complete(prompt, params, schema)


class ScriptedFitted(FittedWorker):
    """A real FittedWorker whose scoring is replaced by the generic plan: only the provider call is under test."""

    def explain_plan(self, state, features, mandate, history):
        chosen = GenericWorker.plan(self, state, None, mandate, history)
        return ScoredPlan(selected_ids=tuple(chosen), ranked=(), excluded_history=(), excluded_replacing=(),
                          excluded_producer=(), incomplete_measurement=False, model_label="FINAL",
                          model_sha256=self.model_sha256)


class FlakyProvider:
    """Like the live provider: features are cached by state, so only the first call for a state can fail
    (an extraction); `fail_on_state` is the 1-based index of the distinct state whose first call fails once."""

    def __init__(self, error: Exception | None = None, fail_on_state: int | None = None) -> None:
        self.error, self.fail_on_state, self.seen, self.failed = error, fail_on_state, [], False

    def identity(self):
        return {"kind": "fake-provider-v1"}

    def __call__(self, state, evidence, producer_recipe_id, run_id):
        if state.id not in self.seen:
            if self.error is not None and len(self.seen) + 1 == self.fail_on_state and not self.failed:
                self.failed = True
                raise self.error
            self.seen.append(state.id)
        return {"telemetry": {}, "produced_by_recipe_id": producer_recipe_id, "telemetry_source": {"kind": "fake"}}


@dataclass
class World:
    ctx: SimpleNamespace
    store: StateStore
    backend: object
    work: Path
    results: Path
    captured: list


def world(tmp_path: Path, monkeypatch, *, provider=None, transport=None, fitted=False) -> World:
    store = StateStore(tmp_path / "engine")
    backend = tc.DryRunBackend(store)
    start = backend.create_start(tmp_path)
    engine = StepEngine(store=store, cache=RunCache(tmp_path / "engine"), backend=backend,
                        meter_version=backend.meter_version, cache_enabled=False)
    registry = load_registry(PROJECT)
    caps = h.CapLedger(tmp_path, {"new_real_runs": 50, "job_hours": 10.0, "tokens": 10**7, "usd": 10.0}, "plan-x")
    scope = SimpleNamespace(work_root=tmp_path, name="train", noise=lambda design, evidence: {})
    ctx = SimpleNamespace(scope=scope, caps=caps, notes={"fake_transport": transport}, engine=engine, starts={start.design: start},
                          registry=registry, provider=provider, backend=backend, ledger=None, store=store)
    if fitted:
        sets = load_recipe_sets(PROJECT / "configs/recipe_sets.yaml")
        workers = {wid: ScriptedFitted(load_worker_card(PROJECT / "configs/workers" / f"{wid}.yaml"),
                                       sets.worker_sets[wid], registry) for wid in ("timing", "area")}
    else:
        workers = tc.fake_workers("T0", general=False)
    monkeypatch.setattr(h, "make_workers", lambda c, cell: workers)
    captured: list = []

    def row(c, cell, result, policy, run_no):
        captured.append(result)
        return {"episode_id": cell.episode_id, "recipe_runs": result.recipe_runs}

    monkeypatch.setattr(h, "result_row", row)
    results = tmp_path / "results"
    results.mkdir()
    return World(ctx, store, backend, tmp_path, results, captured)


def _events(w: World, episode: str) -> list[dict]:
    return Journal(w.store.root).read(episode)


def _types(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


def _no_orchestration_decision(events: list[dict]) -> None:
    assert not {"final", "stop", "validation"} & set(_types(events))


def _policy_log(w: World) -> list[dict]:
    path = w.work / "telemetry.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.is_file() else []


# ------------------------------------------------------------------ the model API
@pytest.mark.parametrize("error", [LLMUnavailableError("the service stayed unavailable"), LLMCallError("empty bridge output")])
def test_a_persistent_api_failure_adds_nothing_to_the_journal_and_resumes_from_it(tmp_path, monkeypatch, error) -> None:
    transport = FlakyTransport(error, fail_on={3})
    w = world(tmp_path, monkeypatch, transport=transport)
    cell = h.Cell("dry_run_fake", "T0", "F1-R1", "m1", 1, "infra_api")
    status, info = h.run_cell(w.ctx, cell, w.results)
    assert status == "infrastructure_failure" and info["kind"] == "api" and type(error).__name__ in info["reason"]
    assert not (w.results / "infra_api.json").exists() and not w.captured
    before = _events(w, "infra_api")
    _no_orchestration_decision(before)
    steps_before = _types(before).count("step")
    assert steps_before >= 1
    log = _policy_log(w)
    assert log and not any(r.get("fallback_triggered") for r in log)                  # no fallback, no STOP
    assert all(r.get("repair_attempts", 0) == 0 for r in log)                          # no schema-repair turn
    assert w.ctx.caps.used()["tokens"] == 2 * 460                                      # the failed call was never charged
    assert w.backend.calls == steps_before                                             # no recipe run beyond the journal
    status, row = h.run_cell(w.ctx, cell, w.results)                                   # the re-queue: same episode id
    assert status == "done"
    after = _events(w, "infra_api")
    assert after[:len(before)] == before                                               # resumed, not restarted
    assert "final" in _types(after)
    assert w.captured[0].recipe_runs == _types(after).count("step")                   # only journalled runs are charged


def test_a_resumed_episode_does_not_ask_the_model_again_about_journalled_decisions(tmp_path, monkeypatch) -> None:
    transport = FlakyTransport(LLMUnavailableError("busy"), fail_on={3})
    w = world(tmp_path, monkeypatch, transport=transport)
    cell = h.Cell("dry_run_fake", "T0", "F1-R1", "m1", 1, "infra_cache")
    assert h.run_cell(w.ctx, cell, w.results)[0] == "infrastructure_failure"
    assert transport.inner.calls == 2
    assert h.run_cell(w.ctx, cell, w.results)[0] == "done"
    lines = [json.loads(x) for x in (w.work / "llm_calls.jsonl").read_text().splitlines()]
    assert transport.inner.calls == 4 and len(lines) == 4        # two decisions before the outage, two after; none repeated
    assert len({x["key"] for x in lines}) == 4


# ------------------------------------------------------------------ worker B extraction
def test_a_failed_extraction_adds_nothing_to_the_journal_and_resumes_from_it(tmp_path, monkeypatch) -> None:
    provider = FlakyProvider(TelemetryUnavailableError("telemetry refused for abc: extraction failed: exit 1"), fail_on_state=2)
    w = world(tmp_path, monkeypatch, provider=provider, fitted=True)
    cell = h.Cell("dry_run_fake", "T0", "v2", "x", 1, "infra_extraction")
    monkeypatch.setattr(h, "policy_for_approach", lambda approach, brief: h.FixedAlternationPolicy(first="B"))
    status, info = h.run_cell(w.ctx, cell, w.results)
    assert status == "infrastructure_failure" and info["kind"] == "extraction"
    assert not w.captured and not (w.results / "infra_extraction.json").exists()
    before = _events(w, "infra_extraction")
    _no_orchestration_decision(before)
    steps = _types(before).count("step")
    assert steps >= 1 and w.backend.calls == steps
    status, _ = h.run_cell(w.ctx, cell, w.results)
    assert status == "done"
    after = _events(w, "infra_extraction")
    assert after[:len(before)] == before and "final" in _types(after)
    assert w.captured[0].recipe_runs == _types(after).count("step")


def test_an_exhausted_extraction_cap_is_a_cap_not_a_failure(tmp_path, monkeypatch) -> None:
    provider = FlakyProvider(ExtractionCapReached("500 extraction jobs are already used"), fail_on_state=1)
    w = world(tmp_path, monkeypatch, provider=provider, fitted=True)
    cell = h.Cell("dry_run_fake", "T0", "v2", "x", 1, "infra_cap")
    monkeypatch.setattr(h, "policy_for_approach", lambda approach, brief: h.FixedAlternationPolicy(first="B"))
    status, info = h.run_cell(w.ctx, cell, w.results)
    assert status == "capped" and "extraction" in info["reason"]


def test_other_errors_are_not_infrastructure_failures(tmp_path, monkeypatch) -> None:
    provider = FlakyProvider(KeyError("a bug"), fail_on_state=1)
    w = world(tmp_path, monkeypatch, provider=provider, fitted=True)
    cell = h.Cell("dry_run_fake", "T0", "v2", "x", 1, "infra_bug")
    monkeypatch.setattr(h, "policy_for_approach", lambda approach, brief: h.FixedAlternationPolicy(first="B"))
    with pytest.raises(KeyError):
        h.run_cell(w.ctx, cell, w.results)


def test_the_cause_chain_is_followed() -> None:
    try:
        try:
            raise TelemetryUnavailableError("x")
        except TelemetryUnavailableError as inner:
            raise ValueError("wrapped") from inner
    except ValueError as exc:
        assert h.infrastructure_kind(exc) == "extraction"
    assert h.infrastructure_kind(ExtractionCapReached("cap")) is None
    assert h.infrastructure_kind(RuntimeError("other")) is None


# ------------------------------------------------------------------ the queue
def _cell(i: int, tier: int) -> h.Cell:
    return h.Cell("d", "T0", "a", "x", tier, f"cell{i}")


def test_one_requeue_after_the_tier_then_dropped_never_substituted(tmp_path) -> None:
    cells = [_cell(1, 1), _cell(2, 1), _cell(3, 2)]
    case = h.Cell("d", "T0", "a", "x", 1, "cell1")
    first = run.record_infra(tmp_path, case, {"kind": "api", "reason": "busy"}, 1)
    assert first["attempt"] == 1 and first["dropped"] is False
    fails = run.infra_failures(tmp_path)
    pending = run.pending_cells(cells, set(), set(), fails)
    assert [c.episode_id for c in pending] == ["cell1", "cell2", "cell3"]
    now = fails["cell1"][0]["at_epoch"] + run.REQUEUE_DELAY_S + 1
    assert [c.episode_id for c in run.eligible_now(pending, fails, now)] == ["cell2", "cell3"]       # waits for tier 1
    pending = run.pending_cells(cells, {"cell2"}, set(), fails)                                      # tier 1 is done
    ready = run.eligible_now(pending, fails, now)
    assert [c.episode_id for c in ready] == ["cell1", "cell3"]
    assert [c.episode_id for c in run.eligible_now(pending, fails, now - 100)] == ["cell3"]           # and after a pause
    second = run.record_infra(tmp_path, case, {"kind": "api", "reason": "busy"}, 1)
    assert second["attempt"] == 2 and second["dropped"] is True
    pending = run.pending_cells(cells, {"cell2"}, set(), run.infra_failures(tmp_path))
    assert [c.episode_id for c in pending] == ["cell3"]                                              # dropped, no substitute


def test_the_requeue_of_a_lower_tier_cell_does_not_wait_for_a_higher_tier(tmp_path) -> None:
    cells = [_cell(1, 1), _cell(2, 2)]
    run.record_infra(tmp_path, cells[0], {"kind": "extraction", "reason": "x"}, 1)
    fails = run.infra_failures(tmp_path)
    now = fails["cell1"][0]["at_epoch"] + run.REQUEUE_DELAY_S + 1
    pending = run.pending_cells(cells, set(), set(), fails)
    assert [c.episode_id for c in run.eligible_now(pending, fails, now)] == ["cell1", "cell2"]


def test_infrastructure_failures_are_counted_across_runs(tmp_path) -> None:
    case = _cell(1, 1)
    run.record_infra(tmp_path, case, {"kind": "api", "reason": "a"}, 1)
    again = run.record_infra(tmp_path, case, {"kind": "extraction", "reason": "b"}, 2)      # a later run: still the second
    assert again["dropped"] is True
