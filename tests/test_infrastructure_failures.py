"""Infrastructure failures end a call without becoming a decision (Sunday freeze-day check, item 1).

Two failures are covered, both on the fake backend with no network, no key and no Vivado:

  * a persistent model-API failure: the bridge's own retry code (tools/llm_bridge/call.py, waits of
    2, 4 and 8 s) runs against an injected SDK client that keeps failing, and the real BridgeTransport,
    CachedLLM, LLMPolicy and episode loop sit on top of it;
  * a refused worker B telemetry extraction at the worker boundary (TelemetryUnavailableError raised
    by the telemetry provider while the episode plans a fitted worker's call).

For each, the call must raise its own exception type; leave the episode journal exactly as it was
before the failing call; never become a STOP decision, a schema-repair turn, a fallback, a logged
decision or a charged recipe run; cache nothing; and let the same episode resume from its journal and
finish exactly as an uninterrupted run does, without paying for any decision twice.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import llm_support as ls
from orch.config import load_recipe_sets, load_worker_card
from orch.contract.brief_text import load_frozen_brief
from orch.contract.models import MetricName
from orch.engine.apply import StepEngine
from orch.engine.cache import RunCache
from orch.engine.journal import Journal
from orch.engine.store import StateStore
from orch.episode import run_episode
from orch.fake.backend import FakeBackend
from orch.policies import llm_client
from orch.policies.llm import DecisionRejected
from orch.policies.llm_client import BridgeTransport, LLMCallError, LLMUnavailableError, TelemetrySink
from orch.policies.llm_schemas import DecisionFormatError
from orch.recipes.registry import load_registry
from orch.workers.fitted import FittedWorker
from orch.workers.live_telemetry import ExtractionCapReached, TelemetryUnavailableError
from test_fitted_episode import FakeInputTelemetry

PROJECT = Path(__file__).resolve().parents[1]
BRIDGE = PROJECT / "tools/llm_bridge/call.py"
NOISE = {MetricName.WNS_NS: 0.05, MetricName.PACKED_LUTS: 5.0}
decision = ls.decision
EPISODE = "infra_unit"


def _load_bridge():
    spec = importlib.util.spec_from_file_location("llm_bridge_call_infrastructure", BRIDGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bridge = _load_bridge()


# ---------------------------------------------------------------- the model side
class SDKError(Exception):
    def __init__(self, code: int, message: str = "service unavailable") -> None:
        super().__init__(message)
        self.code, self.message = code, message


class ReadTimeout(Exception):  # the class name the HTTP library uses
    pass


class ConnectError(Exception):  # likewise
    pass


def _response(text: str):
    usage = SimpleNamespace(
        prompt_token_count=11, candidates_token_count=7, thoughts_token_count=None,
        total_token_count=18, cached_content_token_count=None,
    )
    return SimpleNamespace(
        text=text, usage_metadata=usage,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
    )


class FakeSDK:
    """Stands in for genai.Client. The numbered SDK calls in `failing` raise `error`; the others return the next text."""

    def __init__(self, texts, failing=(), error: Exception | None = None) -> None:
        self.texts, self.failing, self.error = list(texts), set(failing), error
        self.calls, self.prompts, self.models = 0, [], self

    def generate_content(self, *, model, contents, config):
        self.calls += 1
        self.prompts.append(contents)
        if self.calls in self.failing:
            raise self.error
        return _response(self.texts.pop(0))


def _bridge_in_process(monkeypatch, sdk: FakeSDK) -> list[float]:
    """Run tools/llm_bridge/call.py's call_live in this process in place of the subprocess; returns the waits."""

    waits: list[float] = []

    def run(command, *, input, capture_output, text, timeout):  # noqa: A002 - mirrors subprocess.run
        result = bridge.call_live(json.loads(input), client=sdk, sleep=waits.append)
        return subprocess.CompletedProcess(command, 0, json.dumps(result) + "\n", "")

    monkeypatch.setattr(llm_client, "subprocess", SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired))
    return waits


def _bridge_policy(tmp_path: Path):
    return ls.make_policy(tmp_path, transport=BridgeTransport("python3", "tools/llm_bridge/call.py"))


def _scenario() -> list[str]:
    return [
        decision("timing", "S0", preserve=False, attempts=2, reason="largest goal move first"),
        decision("area", "S1", preserve=True, attempts=1, strategy="REPAIR", reason="repair the broken limit from the violating state"),
        decision("timing", "S0", preserve=True, attempts=1, strategy="BACKTRACK", reason="try the start state again with placement kept"),
        decision(action="STOP", reason="nothing untried is likely to help"),
    ]


def _episode(policy, parts):
    engine, registry, workers, _, start = parts
    return run_episode(
        engine=engine, registry=registry, workers=workers, policy=policy, brief=load_frozen_brief("T0"),
        start_state=start, episode_id=EPISODE, noise=NOISE,
    )


def _kinds(engine) -> list[str]:
    return [event["type"] for event in Journal(engine.store.root).read(EPISODE)]


def _cached_replies(tmp_path: Path) -> int:
    return len(list((tmp_path / "llm_cache").rglob("*.json")))


def _spy_on_decisions(policy, engine) -> list[list[dict]]:
    """Snapshot the journal at the start of every decision, so a failing decision can be compared with it."""

    snapshots: list[list[dict]] = []
    real = policy.decide

    def spying(view):
        snapshots.append(Journal(engine.store.root).read(EPISODE))
        return real(view)

    policy.decide = spying
    return snapshots


UNAVAILABLE = [
    pytest.param(SDKError(429, "rate limited"), id="rate-limited-429"),
    pytest.param(SDKError(503), id="unavailable-503"),
    pytest.param(SDKError(500, "internal"), id="server-error-500"),
    pytest.param(ReadTimeout("read timed out"), id="timeout"),
    pytest.param(ConnectError("connection refused"), id="network"),
]


@pytest.mark.parametrize("error", UNAVAILABLE)
def test_a_persistent_api_failure_after_the_bridge_retries_changes_nothing_and_the_episode_resumes(tmp_path, monkeypatch, error) -> None:
    twin_dir = tmp_path / "twin"
    twin_dir.mkdir()
    twin_parts = ls.episode_setup(twin_dir)
    twin_policy, _, _ = ls.make_policy(twin_dir, *_scenario())
    twin = _episode(twin_policy, twin_parts)

    work = tmp_path / "work"
    work.mkdir()
    parts = ls.episode_setup(work)
    engine, backend = parts[0], parts[3]
    sdk = FakeSDK(_scenario(), failing={2, 3, 4, 5}, error=error)   # the second decision: the first call and its three retries
    waits = _bridge_in_process(monkeypatch, sdk)
    policy, _, sink = _bridge_policy(work)
    snapshots = _spy_on_decisions(policy, engine)

    with pytest.raises(LLMUnavailableError):
        _episode(policy, parts)

    assert sdk.calls == 5 and waits == [2, 4, 8]                    # one good call, then 1 + 3 attempts at the retry schedule
    assert len(set(sdk.prompts[1:5])) == 1                          # every attempt sent the same base prompt: no repair turn
    assert Journal(engine.store.root).read(EPISODE) == snapshots[-1]  # the failing decision wrote nothing
    assert _kinds(engine) == ["start", "mandate", "step", "step", "selection"]
    assert backend.calls == ["t1", "t2"]                            # only the first decision's two recipe runs were charged
    assert not {"stop", "final", "validation"} & set(_kinds(engine))
    records = TelemetrySink.read(sink.path)
    assert len(records) == 1 and policy.totals["decisions"] == 1    # the failed decision was never logged as a decision
    assert policy.totals["fallbacks"] == 0 and not records[0]["fallback_triggered"] and records[0]["repair_attempts"] == 0
    assert _cached_replies(work) == 1                               # the failure was not cached

    resumed_policy, _, _ = _bridge_policy(work)                     # a new process: new policy, same journal, cache and log
    resumed = _episode(resumed_policy, parts)

    assert resumed.resumed is True
    assert resumed.final_state.id == twin.final_state.id and resumed.final_verdict.brief_valid
    assert [s.output_state_id for s in resumed.steps] == [s.output_state_id for s in twin.steps]
    assert resumed.recipe_runs == twin.recipe_runs == 4 and resumed.charged_wall_s == twin.charged_wall_s
    assert resumed.stop_reason == twin.stop_reason == "policy_stop"
    assert backend.calls == ["t1", "t2", "a1", "t3"]                # no recipe run was repeated, skipped or added
    assert sdk.prompts[5] == sdk.prompts[1]                         # the retried decision asked the identical question
    assert _kinds(engine).count("mandate") == 3 and _kinds(engine).count("stop") == 1
    records = TelemetrySink.read(sink.path)
    assert [r["strategy"] for r in records] == ["PUSH", "REPAIR", "BACKTRACK", "STOP"]
    assert [r["fallback_triggered"] for r in records] == [False] * 4 and [r["repair_attempts"] for r in records] == [0] * 4
    assert sum(r["tokens"]["total"] for r in records) == 4 * 18    # four paid calls; the failed attempts cost nothing
    assert _cached_replies(work) == 4


def test_a_persistent_failure_on_a_repair_turn_is_neither_a_fallback_nor_a_logged_decision(tmp_path, monkeypatch) -> None:
    texts = [decision("ghost", "S0"), decision("timing", "S0", attempts=1), decision(action="STOP", reason="done")]
    parts = ls.episode_setup(tmp_path)
    engine, backend = parts[0], parts[3]
    sdk = FakeSDK(texts, failing={2, 3, 4, 5}, error=SDKError(503))   # the first reply is rejected; its repair turn cannot be answered
    waits = _bridge_in_process(monkeypatch, sdk)
    policy, _, sink = _bridge_policy(tmp_path)

    with pytest.raises(LLMUnavailableError):
        _episode(policy, parts)

    assert sdk.calls == 5 and waits == [2, 4, 8]
    assert _kinds(engine) == ["start"] and backend.calls == []
    assert TelemetrySink.read(sink.path) == [] and policy.totals["decisions"] == 0 and policy.totals["fallbacks"] == 0
    assert _cached_replies(tmp_path) == 1                           # only the rejected reply, which was a real answer

    resumed_policy, _, _ = _bridge_policy(tmp_path)
    resumed = _episode(resumed_policy, parts)

    first = TelemetrySink.read(sink.path)[0]
    assert first["repair_attempts"] == 1 and first["fallback_triggered"] is False and first["strategy"] == "PUSH"
    assert [c["cache_hit"] for c in first["calls"]] == [True, False]   # the rejected reply came from the cache, the repair was paid
    assert "ghost" in first["calls"][0]["error"]
    assert resumed.recipe_runs == 1 and backend.calls == ["t1"] and resumed.stop_reason == "policy_stop"


@pytest.mark.parametrize("error", [SDKError(401, "api key rejected"), SDKError(400, "bad request")], ids=["auth", "bad-request"])
def test_a_non_retryable_api_failure_is_raised_at_once_and_is_not_a_decision_either(tmp_path, monkeypatch, error) -> None:
    parts = ls.episode_setup(tmp_path)
    engine, backend = parts[0], parts[3]
    sdk = FakeSDK(_scenario(), failing={1}, error=error)
    waits = _bridge_in_process(monkeypatch, sdk)
    policy, _, sink = _bridge_policy(tmp_path)

    with pytest.raises(LLMCallError):
        _episode(policy, parts)

    assert sdk.calls == 1 and waits == []                           # no retry for a request the service will always refuse
    assert _kinds(engine) == ["start"] and backend.calls == []
    assert TelemetrySink.read(sink.path) == [] and policy.totals["fallbacks"] == 0 and _cached_replies(tmp_path) == 0


def test_the_infrastructure_exceptions_cannot_be_caught_by_the_repair_loop_or_a_value_error_handler() -> None:
    for cls in (LLMUnavailableError, LLMCallError, TelemetryUnavailableError, ExtractionCapReached):
        assert not issubclass(cls, (ValueError, DecisionFormatError, DecisionRejected))
    assert issubclass(ExtractionCapReached, TelemetryUnavailableError)


# ---------------------------------------------------------------- the worker boundary
def _fitted_parts(tmp_path: Path):
    store = StateStore(tmp_path / "data")
    backend = FakeBackend(store=store)
    engine = StepEngine(
        store=store, cache=RunCache(store.root), backend=backend,
        meter_version=backend.meter_version, cache_enabled=True,
    )
    registry = load_registry(PROJECT)
    sets = load_recipe_sets(PROJECT / "configs/recipe_sets.yaml")
    sets.require_signed_off()
    workers = {
        worker_id: FittedWorker(load_worker_card(PROJECT / "configs/workers" / f"{worker_id}.yaml"), sets.worker_sets[worker_id], registry)
        for worker_id in ("timing", "area")
    }
    return engine, registry, workers, backend, backend.create_start("vexriscv_re-place")


def _count_recipe_runs(backend) -> list[str]:
    ran: list[str] = []
    original = backend.execute

    def counting(state, recipe, *args, **kwargs):
        ran.append(recipe.id)
        return original(state, recipe, *args, **kwargs)

    backend.execute = counting
    return ran


class RefusingTelemetry(FakeInputTelemetry):
    """The synthetic input telemetry, except that the numbered calls are refused as the live provider refuses a failed extraction."""

    def __init__(self, refused=()) -> None:
        super().__init__()
        self.refused, self.calls = set(refused), 0

    def __call__(self, state, evidence, producer_recipe_id, run_id):
        self.calls += 1
        if self.calls in self.refused:
            raise TelemetryUnavailableError(f"telemetry refused for {state.id[:12]}: the extraction failed")
        return super().__call__(state, evidence, producer_recipe_id, run_id)


def _fitted_episode(policy, parts, provider):
    engine, registry, workers, _, start = parts
    return run_episode(
        engine=engine, registry=registry, workers=workers, policy=policy, brief=load_frozen_brief("A0"),
        start_state=start, episode_id=EPISODE, noise=NOISE, telemetry_provider=provider,
    )


def _worker_b_texts() -> tuple[str, str]:
    return decision("area", "S0", attempts=1, reason="area is the goal"), decision(action="STOP", reason="done")


def _uninterrupted_fitted_twin(tmp_path: Path):
    twin_dir = tmp_path / "twin"
    twin_dir.mkdir()
    parts = _fitted_parts(twin_dir)
    ran = _count_recipe_runs(parts[3])
    policy, _, _ = ls.make_policy(twin_dir, *_worker_b_texts())
    return _fitted_episode(policy, parts, RefusingTelemetry()), ran


@pytest.mark.parametrize(
    "refused_call, journalled, logged",
    [(1, ["start"], ["PUSH", "PUSH", "STOP"]), (2, ["start", "mandate"], ["PUSH", "STOP"])],
    ids=["before-the-mandate-is-journalled", "while-the-journalled-mandate-is-re-planned"],
)
def test_a_refused_worker_b_extraction_changes_nothing_and_the_episode_resumes(tmp_path, refused_call, journalled, logged) -> None:
    twin, twin_ran = _uninterrupted_fitted_twin(tmp_path)

    work = tmp_path / "work"
    work.mkdir()
    parts = _fitted_parts(work)
    engine, backend = parts[0], parts[3]
    ran = _count_recipe_runs(backend)
    policy, transport, sink = ls.make_policy(work, *_worker_b_texts())
    provider = RefusingTelemetry(refused={refused_call})            # the live provider's refusal: the extraction failed

    with pytest.raises(TelemetryUnavailableError, match="extraction failed"):
        _fitted_episode(policy, parts, provider)

    assert _kinds(engine) == journalled and ran == []               # no step, no selection, no stop, nothing charged
    assert len(transport.prompts) == 1                              # one model call, for the one decision that was made
    records = TelemetrySink.read(sink.path)
    assert len(records) == 1 and policy.totals["fallbacks"] == 0 and records[0]["repair_attempts"] == 0 and records[0]["strategy"] == "PUSH"
    assert _cached_replies(work) == 1

    resumed_policy, resumed_transport, _ = ls.make_policy(work, _worker_b_texts()[1])   # only the closing decision needs the model again
    resumed = _fitted_episode(resumed_policy, parts, RefusingTelemetry())

    assert resumed.resumed is True and len(resumed_transport.prompts) == 1
    assert ran == twin_ran and len(ran) == 1                       # the recipe ran once, exactly as in the uninterrupted episode
    assert [s.output_state_id for s in resumed.steps] == [s.output_state_id for s in twin.steps]
    assert resumed.final_state.id == twin.final_state.id and resumed.recipe_runs == twin.recipe_runs == 1
    assert resumed.charged_wall_s == twin.charged_wall_s and resumed.stop_reason == twin.stop_reason == "policy_stop"
    events = Journal(engine.store.root).read(EPISODE)
    plans = [e["worker_plan"] for e in events if e["type"] == "mandate"]
    assert len(plans) == 1 and plans[0]["telemetry_source"]["kind"] == "fake-test"   # planned from the provider's record, never from the recipe means
    records = TelemetrySink.read(sink.path)
    assert [r["strategy"] for r in records] == logged and not any(r["fallback_triggered"] for r in records)
    # a decision made before the refusal and not yet journalled is asked again on resume: the cache answers it
    # for nothing, and the log holds it twice (the second record is a cache hit with no fresh tokens)
    assert all(r["tokens"]["fresh_total"] == 0 and r["calls"][0]["cache_hit"] for r in records[1:-1])
    assert sum(r["tokens"]["fresh_total"] for r in records) == 2 * 120        # two paid calls in all (120 tokens each): the decision and the closing STOP
