"""Command-line entry points for the shared engine and labelled fake experiments.

Real measure/run-recipe/sweep paths are UNTESTED until the VexRiscv smoke
finishes. Fake commands are software checks; their results are never hardware
measurements. Production worker recipe sets stay unsigned until the authors select.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orch.analysis.report import Comparison, build_share_of_best_table, render_share_of_best_markdown
from orch.baselines.exhaustive import SweepResult, run_exhaustive
from orch.config import load_brief, load_noise, load_recipe_sets, load_worker_card
from orch.contract.models import DesignState, Evidence
from orch.engine.apply import StepEngine
from orch.engine.cache import RunCache
from orch.engine.store import StateStore, _atomic_bytes, _json_bytes, _safe_name
from orch.episode import EpisodeResult, FinalValidation, run_episode
from orch.fake.backend import FakeBackend
from orch.policies.random_policy import RandomPolicy
from orch.policies.trivial import TrivialPolicy
from orch.recipes.registry import Registry, load_registry
from orch.runtime import PROJECT_ROOT, SMOKE_STATUS, RealRuntime, make_real_runtime
from orch.workers.base import GenericWorker
from orch.workers.fitted import FittedWorker
from orch.workers.telemetry import SavedTelemetryProvider


FAKE_TEST_RECIPES = (
    "LATCH.phys_opt",
    "LATCH.retime",
    "A_R_EA",
    "A_E_DUPL",
)


@dataclass(frozen=True)
class _Context:
    """The engine and registry selected for one CLI call."""

    engine: StepEngine
    store: StateStore
    registry: Registry
    fake_backend: FakeBackend | None
    real_runtime: RealRuntime | None = None


def _data_root(args: argparse.Namespace) -> Path:
    """Keep every new artifact in the one authorized Development project."""

    default = PROJECT_ROOT / "data" / (
        "cli_fake" if args.backend == "fake" else "real"
    )
    root = (args.data_root or default).expanduser().resolve()
    if not root.is_relative_to(PROJECT_ROOT / "data"):
        raise ValueError("data root must stay inside the Development project")
    return root


def _context(args: argparse.Namespace) -> _Context:
    root = _data_root(args)
    if args.backend == "fake":
        store = StateStore(root)
        cache = RunCache(root)
        backend = FakeBackend(store=store)
        engine = StepEngine(
            store=store,
            cache=cache,
            backend=backend,
            meter_version=backend.meter_version,
            cache_enabled=True,
        )
        return _Context(engine, store, load_registry(PROJECT_ROOT), backend, None)
    runtime = make_real_runtime(
        root,
        cache_enabled=bool(getattr(args, "cache", False)),
        max_slots=int(getattr(args, "max_slots", 4)),
    )
    return _Context(runtime.engine, runtime.store, runtime.registry, None, runtime)


def _start(args: argparse.Namespace, context: _Context) -> DesignState:
    if context.fake_backend is not None:
        return context.fake_backend.create_start(args.design)
    if not args.input:
        raise ValueError("real backend requires --input")
    return context.store.import_state(Path(args.input), design=args.design)


def _brief(choice: str):
    names = {
        "timing": PROJECT_ROOT / "configs" / "briefs" / "timing.yaml",
        "area": PROJECT_ROOT / "configs" / "briefs" / "area.yaml",
    }
    return load_brief(names[choice])


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _measure(args: argparse.Namespace) -> int:
    context = _context(args)
    state = _start(args, context)
    evidence = context.engine.measure_start(state)
    _print_json({
        "fake": state.fake,
        "state_id": state.id,
        "evidence": evidence.model_dump(mode="json"),
    })
    return 0


def _run_recipe(args: argparse.Namespace) -> int:
    context = _context(args)
    state = _start(args, context)
    if args.start_input and context.fake_backend is not None:
        raise ValueError("--start-input applies only to a real checkpoint")
    if args.start_input:
        start_state = context.store.import_state(
            Path(args.start_input), design=args.design
        )
    else:
        start_state = state
    start = context.engine.measure_start(start_state)
    parent = start if start_state.id == state.id else context.engine.measure_start(state)
    recipe = context.registry.get(args.recipe)
    applied = context.engine.apply(
        state,
        recipe,
        brief=_brief(args.brief),
        start=start,
        parent=parent,
    )
    outcome = applied.outcome
    _print_json({
        "fake": state.fake,
        "run_id": outcome.run.id,
        "recipe_id": recipe.id,
        "run_status": outcome.run.status.value,
        "cache_hit": outcome.run.cache_hit,
        "charged_wall_s": outcome.run.wall_s,
        "output_state_id": outcome.state.id if outcome.state else None,
        "evidence": outcome.evidence.model_dump(mode="json") if outcome.evidence else None,
        "verdict": applied.verdict.model_dump(mode="json"),
    })
    return 0 if applied.verdict.physically_valid else 1


def _selected_recipes(args: argparse.Namespace, context: _Context):
    """Keep the fake integration set clearly distinct from signed choices."""

    if args.all_candidates and args.recipes:
        raise ValueError("choose either --all-candidates or --recipes")
    if args.all_candidates:
        by_worker = {"control": 0, "timing": 1, "area": 2, "power": 3}
        return tuple(sorted(
            context.registry.all(),
            key=lambda recipe: (by_worker.get(recipe.worker, 4), recipe.id),
        )), "all_registered_candidates"
    if args.recipes:
        ids = tuple(part.strip() for part in args.recipes.split(",") if part.strip())
        if len(ids) != len(set(ids)):
            raise ValueError("recipe list contains duplicates")
        return tuple(context.registry.get(recipe_id) for recipe_id in ids), "explicit"
    if context.fake_backend is None:
        raise ValueError("real sweep needs --recipes or --all-candidates")
    return tuple(
        context.registry.get(recipe_id) for recipe_id in FAKE_TEST_RECIPES
    ), "FAKE_TEST_ONLY_unsigned"


def _sweep(args: argparse.Namespace) -> int:
    if args.backend == "real" and args.depth == 2:
        raise ValueError("real depth-two sweep is excluded from tonight's scope")
    context = _context(args)
    start_state = _start(args, context)
    recipes, set_label = _selected_recipes(args, context)
    briefs = (_brief("timing"), _brief("area"))
    result = run_exhaustive(
        engine=context.engine,
        start_state=start_state,
        recipes=recipes,
        briefs=briefs,
        sweep_id=args.id,
        max_depth=args.depth,
    )
    folder = context.store.root / "campaigns" / _safe_name(
        "sweep-" + args.id, what="sweep id"
    )
    result_file = folder / "result.json"
    _atomic_bytes(result_file, _json_bytes(result.model_dump(mode="json")))
    rows = build_share_of_best_table(result)
    _print_json({
        "fake": result.fake,
        "design": result.design,
        "sweep_id": result.sweep_id,
        "depth": result.max_depth,
        "recipe_set": set_label,
        "selected_set_signed_off": False,
        "recipes": [recipe.id for recipe in recipes],
        "entries": len(result.entries),
        "charged_recipe_runs": result.total_recipe_runs,
        "actual_backend_runs": result.actual_backend_runs,
        "charged_wall_s": result.total_charged_wall_s,
        "journal_path": result.journal_path,
        "result_file": str(result_file),
        "share_of_best": [row.model_dump(mode="json") for row in rows],
    })
    return 0


def _episode_workers(context: _Context) -> tuple[dict[str, GenericWorker], str]:
    """Use an explicit fake test set or the authors' signed production recipe set."""

    if context.fake_backend is not None:
        sets = {
            "timing": ("LATCH.phys_opt", "LATCH.retime"),
            "area": ("A_R_EA", "A_E_DUPL"),
        }
        label = "FAKE_TEST_ONLY_unsigned"
    else:
        chosen = load_recipe_sets(PROJECT_ROOT / "configs" / "recipe_sets.yaml")
        chosen.require_signed_off()
        sets = chosen.worker_sets
        label = "Matt_signed"
    workers: dict[str, GenericWorker] = {}
    for worker_id in ("timing", "area", "power"):
        recipe_ids = tuple(sets.get(worker_id, ()))
        if not recipe_ids:
            continue
        card = load_worker_card(
            PROJECT_ROOT / "configs" / "workers" / f"{worker_id}.yaml"
        )
        workers[worker_id] = (
            FittedWorker(card, recipe_ids, context.registry)
            if context.real_runtime is not None and worker_id in ("timing", "area")
            else GenericWorker(card, recipe_ids)
        )
    if not workers:
        raise ValueError("no recipes selected for any worker")
    return workers, label


def _episode(args: argparse.Namespace) -> int:
    """Run or resume one policy episode with its immutable fixed brief."""

    context = _context(args)
    workers, set_label = _episode_workers(context)
    telemetry_provider = (
        SavedTelemetryProvider(context.store.root)
        if any(isinstance(worker, FittedWorker) for worker in workers.values())
        else None
    )
    start_state = _start(args, context)
    brief = _brief(args.brief)
    # The episode measures its start again through the engine. The fake
    # backend reuses its cached Evidence; real repetitions are still measured.
    first_evidence = context.engine.measure_start(start_state)
    noise = load_noise(PROJECT_ROOT / "configs" / "noise.yaml").for_start(
        args.design, first_evidence
    )
    policy = (
        TrivialPolicy()
        if args.policy == "trivial"
        else RandomPolicy(seed=args.seed)
    )
    validator = None
    if context.real_runtime is not None:
        runtime = context.real_runtime

        def validator(golden: DesignState, revised: DesignState) -> FinalValidation:
            """Run the copied AMD validator only on ranked final candidates."""

            run_id = "validator_" + uuid.uuid4().hex
            run_dir = context.store.root / "validators" / run_id
            process, check = runtime.validator.validate(
                runner=runtime.runner,
                golden_path=context.store.state_path(golden.id),
                revised_path=context.store.state_path(revised.id),
                reference_state_id=golden.id,
                run_id=run_id,
                run_dir=run_dir,
                vectors=brief.equivalence_vectors,
            )
            return FinalValidation(
                check=check,
                run_id=process.run_id,
                artifact_dir=str(run_dir),
                note="weak 200-vector final-only check; UNTESTED real validator path",
            )

    result = run_episode(
        engine=context.engine,
        registry=context.registry,
        workers=workers,
        policy=policy,
        brief=brief,
        start_state=start_state,
        episode_id=args.id,
        noise=noise,
        validator=validator,
        telemetry_provider=telemetry_provider,
    )
    result_file = context.store.root / "episodes" / _safe_name(
        args.id, what="episode id"
    ) / "result.json"
    _atomic_bytes(result_file, _json_bytes(result.model_dump(mode="json")))
    _print_json({
        "fake": result.fake,
        "episode_id": result.episode_id,
        "design": start_state.design,
        "brief_id": result.brief_id,
        "policy_id": result.policy_id,
        "recipe_set": set_label,
        "selected_set_signed_off": set_label == "Matt_signed",
        "worker_models": {
            worker_id: {
                "label": worker.model_label,
                "sha256": worker.model_sha256,
            }
            for worker_id, worker in workers.items()
            if isinstance(worker, FittedWorker)
        },
        "steps": len(result.steps),
        "charged_recipe_runs": result.recipe_runs,
        "charged_wall_s": result.charged_wall_s,
        "stop_reason": result.stop_reason,
        "final_state_id": result.final_state.id if result.final_state else None,
        "final_metrics": (
            {name.value: value.value for name, value in result.final_evidence.metrics.items()}
            if result.final_evidence else None
        ),
        "validation_status": result.validation_status,
        "resumed": result.resumed,
        "journal_path": result.journal_path,
        "result_file": str(result_file),
    })
    return 0


def _report(args: argparse.Namespace) -> int:
    root = _data_root(args)
    result_file = root / "campaigns" / _safe_name(
        "sweep-" + args.id, what="sweep id"
    ) / "result.json"
    sweep = SweepResult.model_validate_json(result_file.read_bytes())
    if sweep.fake != (args.backend == "fake"):
        raise ValueError("report backend label differs from saved sweep")
    comparisons = []
    for episode_id in args.episode_id:
        episode_file = root / "episodes" / _safe_name(
            episode_id, what="episode id"
        ) / "result.json"
        episode = EpisodeResult.model_validate_json(episode_file.read_bytes())
        if episode.start_state_id != sweep.start_state_id:
            raise ValueError("episode and sweep use different start checkpoints")
        if episode.fake != sweep.fake:
            raise ValueError("episode and sweep mix fake and real records")
        comparisons.append(Comparison(
            approach=episode.policy_id,
            brief_id=episode.brief_id,
            evidence=episode.final_evidence,
            verdict=episode.final_verdict,
            run_count=episode.recipe_runs,
            wall_s=episode.charged_wall_s,
            fake=episode.fake,
        ))
    rows = build_share_of_best_table(sweep, comparisons)
    print(render_share_of_best_markdown(rows), end="")
    return 0


def _recipes(args: argparse.Namespace) -> int:
    registry = load_registry(PROJECT_ROOT)
    recipes = registry.all()
    if args.worker:
        recipes = tuple(recipe for recipe in recipes if recipe.worker == args.worker)
    _print_json({
        "candidate_count": len(recipes),
        "selected_set_signed_off": False,
        "recipes": [{
            "id": recipe.id,
            "worker": recipe.worker,
            "kind": recipe.kind,
            "replaces": recipe.replaces,
            "version": recipe.version,
            "source_sha256": recipe.source_sha256,
        } for recipe in recipes],
    })
    return 0


def _status(args: argparse.Namespace) -> int:
    status: dict[str, object] = {"smoke_status_path": str(SMOKE_STATUS)}
    if SMOKE_STATUS.is_file():
        raw = json.loads(SMOKE_STATUS.read_text(encoding="utf-8"))
        status["smoke"] = {
            "phase": raw.get("phase"),
            "baseline_match": raw.get("baseline_match"),
            "deterministic": raw.get("deterministic"),
            "completed_recipe_runs": len(raw.get("runs", ())),
            "tagged_survivor_pids": raw.get("tags", {}).get("tagged_survivor_pids"),
            "error": raw.get("error"),
            "updated_at_utc": raw.get("updated_at_utc"),
        }
    else:
        status["smoke"] = {"phase": "not_started"}
    root = _data_root(args)
    status["data_root"] = str(root)
    status["saved_sweeps"] = [
        path.parent.name for path in sorted((root / "campaigns").glob("sweep-*/result.json"))
    ]
    status["saved_episodes"] = [
        path.parent.name for path in sorted((root / "episodes").glob("*/result.json"))
    ]
    _print_json(status)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Keep defaults safe for read-only or explicitly labelled fake work."""

    parser = argparse.ArgumentParser(prog="orch", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    recipes = sub.add_parser("recipes", help="list registered candidates")
    recipes.add_argument("--worker", choices=("control", "timing", "area", "power"))
    recipes.set_defaults(func=_recipes)

    for name, handler in (
        ("measure", _measure),
        ("run-recipe", _run_recipe),
        ("sweep", _sweep),
        ("episode", _episode),
        ("report", _report),
        ("status", _status),
    ):
        command = sub.add_parser(name)
        command.add_argument("--backend", choices=("fake", "real"), default="fake")
        command.add_argument("--data-root", type=Path)
        if name in ("measure", "run-recipe", "sweep", "episode"):
            command.add_argument("--design", required=True)
            command.add_argument("--input", type=Path)
            command.add_argument("--cache", action="store_true")
            command.add_argument("--max-slots", type=int, default=4)
        if name == "run-recipe":
            command.add_argument("--recipe", required=True)
            command.add_argument("--brief", choices=("timing", "area"), required=True)
            command.add_argument("--start-input", type=Path)
        if name == "sweep":
            command.add_argument("--id", required=True)
            command.add_argument("--depth", type=int, choices=(1, 2), default=1)
            command.add_argument("--recipes", help="comma-separated explicit recipe IDs")
            command.add_argument("--all-candidates", action="store_true")
        if name == "episode":
            command.add_argument("--id", required=True)
            command.add_argument("--brief", choices=("timing", "area"), required=True)
            command.add_argument("--policy", choices=("trivial", "random"), required=True)
            command.add_argument("--seed", type=int, default=0)
        if name == "report":
            command.add_argument("--id", required=True)
            command.add_argument("--episode-id", action="append", default=[])
        command.set_defaults(func=handler)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Return an exit code without printing raw tool logs or secrets."""

    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"orch: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
