"""Builders shared by the LLM policy tests: states, briefs, cards and policy views."""
from __future__ import annotations

import json

from orch.contract.models import (
    Brief,
    BriefTerm,
    MetricName,
    Objective,
    VerdictStatus,
    WorkerCard,
    WorkerSettings,
)
from orch.contract.views import CallSummary, PolicyView, StateSummary, StepSummary

DESIGN_SECRET = "vexriscv_re-place"


def sid(n: int) -> str:
    return f"{n:064x}"


def metrics(wns: float, luts: float, *, ffs: float = 500.0, netlist: float = 2000.0,
            power: float = 1.5) -> dict[MetricName, float]:
    return {
        MetricName.WNS_NS: wns, MetricName.PACKED_LUTS: luts, MetricName.FFS: ffs,
        MetricName.NETLIST_LUTS: netlist, MetricName.DYNAMIC_POWER_W: power,
    }


def state(n: int, wns: float, luts: float, *, brief_valid: bool = True, **extra: float) -> StateSummary:
    return StateSummary(
        state_id=sid(n), metrics=metrics(wns, luts, **extra), physically_valid=True,
        brief_valid=brief_valid,
    )


def brief_t0(budget: int = 6) -> Brief:
    """Maximise WNS; packed LUTs must not exceed the start value."""

    return Brief(
        id="T0", objective=Objective(metric=MetricName.WNS_NS, direction="max"),
        terms=(BriefTerm(id="t0_limit", metric=MetricName.PACKED_LUTS, comparison="<="),),
        budget_runs=budget,
    )


def brief_a(budget: int = 6) -> Brief:
    """Minimise packed LUTs; WNS may fall at most 0.1 ns below the start value."""

    return Brief(
        id="A", objective=Objective(metric=MetricName.PACKED_LUTS, direction="min"),
        terms=(BriefTerm(id="a_limit", metric=MetricName.WNS_NS, comparison=">=", offset=-0.1),),
        budget_runs=budget,
    )


def cards() -> tuple[WorkerCard, ...]:
    return (
        WorkerCard(
            id="timing", version="v", objective=Objective(metric=MetricName.WNS_NS, direction="max"),
            recipe_set_id="timing", may_replace=True,
        ),
        WorkerCard(
            id="area", version="v", objective=Objective(metric=MetricName.PACKED_LUTS, direction="min"),
            recipe_set_id="area", may_replace=True,
        ),
    )


def call(n: int, worker: str, base: int, selected: int | None, *, preserve: bool = False,
         attempts: int = 2, runs: int | None = None, reason: str | None = None,
         strategy: str = "PUSH", recipe: str | None = None) -> CallSummary:
    payload = {"s": strategy, "r": reason}
    if recipe is not None:
        payload["x"] = recipe
    ref = json.dumps(payload) if (reason is not None or recipe is not None) else ""
    return CallSummary(
        mandate_id=f"mandate_{n}", worker_id=worker, base_state_id=sid(base),
        settings=WorkerSettings(preserve_placement=preserve, max_runs=attempts),
        selected_state_id=None if selected is None else sid(selected),
        selected_verdict_status=None if selected is None else VerdictStatus.ACCEPTABLE,
        recipe_runs=attempts if runs is None else runs, context_ref=ref,
    )


def step(worker: str, base: int, out: int | None, *, preserve: bool = False, attempts: int = 2) -> StepSummary:
    return StepSummary(
        worker_id=worker, settings=WorkerSettings(preserve_placement=preserve, max_runs=attempts),
        base_state_id=sid(base), output_state_id=None if out is None else sid(out),
        verdict_status=VerdictStatus.ACCEPTABLE,
    )


def view(*, visited: list[StateSummary], calls: tuple[CallSummary, ...] = (),
         steps: tuple[StepSummary, ...] = (), budget_left: int = 6, brief: Brief | None = None,
         noise: dict[MetricName, float] | None = None, current: StateSummary | None = None,
         best: StateSummary | None = None, episode_id: str | None = None) -> PolicyView:
    brief = brief or brief_t0()
    noise = noise if noise is not None else {MetricName.WNS_NS: 0.1, MetricName.PACKED_LUTS: 10.0}
    return PolicyView(
        brief=brief, budget_left=budget_left, start=visited[0], current=current or visited[0],
        best_valid=best or visited[0], visited=tuple(visited), steps=steps, calls=calls,
        worker_cards=cards(), noise_bands=noise,
        episode_id=episode_id or f"{DESIGN_SECRET}_T0_secret_run",
    )


# ---------------------------------------------------------------- LLM and episode fakes
import hashlib  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

from orch.contract.models import (  # noqa: E402
    DesignState, Evidence, MetricValue, RunOutcome, RunRecord, RunStatus,
)
from orch.engine.apply import StepEngine  # noqa: E402
from orch.engine.cache import RunCache  # noqa: E402
from orch.engine.store import StateStore  # noqa: E402
from orch.interfaces import RecipeSpec  # noqa: E402
from orch.policies.llm_client import (  # noqa: E402
    CachedLLM, LLMParams, LLMReply, ReplayCache, TelemetrySink,
)
from orch.policies.llm_prompt import load_template  # noqa: E402
from orch.workers.base import GenericWorker  # noqa: E402

TEMPLATE = Path(__file__).resolve().parents[1] / "configs/prompts/orchestrator_v1.txt"
PROFILE_TEXT = {
    "timing": "  when placement is preserved: goal within_noise usually.",
    "area": "  when placement is preserved: goal small_gain often.",
}
PROFILE_SHA = {"timing": "1" * 64, "area": "2" * 64}


def decision(worker: str = "timing", base: str = "S0", *, preserve: bool = False, attempts: int = 1,
             respect: bool = False, strategy: str = "PUSH", reason: str = "push the goal",
             action: str = "INVOKE_WORKER") -> str:
    mandate = None if action == "STOP" else {
        "worker_id": worker, "base_state": base, "preserve_placement": preserve,
        "attempts": attempts, "respect_brief_limits": respect,
    }
    return json.dumps({
        "state_diagnosis": "diagnosis", "strategy": "STOP" if action == "STOP" else strategy,
        "conflict_check": "no gains to undo", "rationale_summary": reason, "action": action,
        "mandate": mandate,
    })


class ScriptedTransport:
    """Returns the next scripted reply text for each model call and records every prompt."""

    def __init__(self, *texts: str, tokens: tuple[int, int] = (100, 20)) -> None:
        self.queue = list(texts)
        self.prompts: list[str] = []
        self.tokens = tokens

    def complete(self, prompt, params, schema):
        self.prompts.append(prompt)
        if not self.queue:
            raise AssertionError("unexpected extra model call")
        text = self.queue.pop(0)
        return LLMReply(
            text=text, prompt_tokens=self.tokens[0], completion_tokens=self.tokens[1], thoughts_tokens=0,
            total_tokens=sum(self.tokens), latency_ms=5.0, finish_reason="STOP", model="scripted", attempts=1,
        )


def make_policy(tmp_path: Path, *texts: str, mode: str = "fuzzy", profiles_on: bool = True,
                lessons: str | None = None, cache_mode: str = "record", name: str | None = None,
                transport: ScriptedTransport | None = None, cache_dir: str = "llm_cache",
                telemetry: str = "telemetry.jsonl", max_repairs: int = 2):
    from orch.policies.llm import LLMPolicy, LLMPolicyConfig

    transport = transport if transport is not None else ScriptedTransport(*texts)
    llm = CachedLLM(
        None if cache_mode == "replay" else transport,
        ReplayCache(tmp_path / cache_dir, mode=cache_mode), LLMParams(model="scripted-model"),
    )
    config = LLMPolicyConfig(
        name=name or ("F1" if profiles_on else "F0"), mode=mode, profiles_on=profiles_on,
        template=load_template(TEMPLATE), profile_texts=dict(PROFILE_TEXT),
        profile_sha256=dict(PROFILE_SHA), lessons=lessons, max_repairs=max_repairs,
    )
    sink = TelemetrySink(tmp_path / telemetry)
    return LLMPolicy(config, llm, sink), transport, sink


@dataclass
class StubRegistry:
    entries: dict[str, RecipeSpec]

    def get(self, recipe_id: str) -> RecipeSpec:
        return self.entries[recipe_id]


class MiniBackend:
    """Deterministic test backend with three recipes per worker; never runs a tool."""

    meter_version = "mini-backend-v1"
    effects = {
        "t1": (0.20, 15.0), "t2": (0.05, 0.0), "t3": (-0.20, 0.0),
        "a1": (-0.01, -30.0), "a2": (-0.20, -60.0), "a3": (0.0, 0.0),
    }

    def __init__(self, store: StateStore, *, fail_on_call: int | None = None) -> None:
        self.store, self.fail_on_call = store, fail_on_call
        self.metrics: dict[str, tuple[float, float]] = {}
        self.calls: list[str] = []

    def create_start(self, folder: Path) -> DesignState:
        source = folder / "start.dcp"
        source.write_bytes(b"llm-policy-test-start")
        state = self.store.import_state(source, design="unit-test", fake=True)
        self.metrics[state.id] = (-1.0, 100.0)
        return state

    def _evidence(self, state: DesignState) -> Evidence:
        wns, luts = self.metrics[state.id]
        digest = hashlib.sha256(f"{state.id}:{wns}:{luts}".encode()).hexdigest()
        values = {
            MetricName.WNS_NS: wns, MetricName.PACKED_LUTS: luts, MetricName.FFS: 500.0,
            MetricName.NETLIST_LUTS: 2 * luts,
        }
        return Evidence(
            id=digest, state_id=state.id,
            metrics={m: MetricValue(value=v, unit="u", method="mini", basis="routed") for m, v in values.items()},
            routed=True, route_errors=0, drc_errors=0, constraint_fingerprint="xdc", clock_digest="clk",
            meter_version=self.meter_version, fake=True,
        )

    def measure_start(self, state, *, run_dir):
        return self._evidence(state)

    def execute(self, state, recipe, *, run_id, run_dir):
        self.calls.append(recipe.id)
        if self.fail_on_call == len(self.calls):
            self.fail_on_call = None
            raise RuntimeError("simulated crash between worker attempts")
        wns, luts = self.metrics[state.id]
        d_wns, d_luts = self.effects[recipe.id]
        output = run_dir / "out.dcp"
        output.write_bytes(f"{state.id}:{recipe.id}".encode())
        successor = self.store.import_state(
            output, design=state.design, parent_id=state.id, made_by_run_id=run_id, fake=True,
        )
        self.metrics[successor.id] = (wns + d_wns, luts + d_luts)
        return RunOutcome(
            run=RunRecord(
                id=run_id, recipe_id=recipe.id, recipe_version=recipe.version, status=RunStatus.OK,
                exit_code=0, wall_s=1.0, input_sha256_before=state.id, input_sha256_after=state.id, fake=True,
            ),
            state=successor, evidence=self._evidence(successor),
        )


def _recipe(recipe_id: str, worker: str) -> RecipeSpec:
    return RecipeSpec(
        id=recipe_id, worker=worker, kind="tcl", script=f"unit/{recipe_id}", replaces=False, cause="unit",
        source="unit", source_sha256="unit", timeout_s=10, version="unit-v1",
    )


def episode_setup(tmp_path: Path, *, fail_on_call: int | None = None):
    store = StateStore(tmp_path / "data")
    backend = MiniBackend(store, fail_on_call=fail_on_call)
    start = backend.create_start(tmp_path)
    engine = StepEngine(
        store=store, cache=RunCache(store.root), backend=backend,
        meter_version=backend.meter_version, cache_enabled=True,
    )
    registry = StubRegistry({
        **{r: _recipe(r, "timing") for r in ("t1", "t2", "t3")},
        **{r: _recipe(r, "area") for r in ("a1", "a2", "a3")},
    })
    workers = {c.id: GenericWorker(c, ids) for c, ids in zip(cards(), (("t1", "t2", "t3"), ("a1", "a2", "a3")))}
    return engine, registry, workers, backend, start
