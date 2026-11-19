"""Frozen option-B constraint-repair policy over the data-only PolicyView.

The policy reconstructs its cycle from completed call summaries and its
journaled context_ref. It has no runner, recipe or tool access.

Four pre-declared knobs (decision of 2026-11-06) vary the repair step:
repair_attempts (1 or 2), repair_preserve_placement (true or false),
gentle_step (on or off) and after_failed_repair (rollback or continue). The
defaults are the frozen v1 behaviour, and the default policy keeps version
1.0-frozen, so v1 mandates, contexts and journals are unchanged.

With several brief limits the repair goes to the worker whose objective is the
violated limit's metric (the first violated limit in brief order); the
primary is always the worker whose objective is the brief's.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from orch.contract.models import (
    BriefTerm, Mandate, MetricName, VerdictStatus, WorkerCard, WorkerSettings,
)
from orch.contract.views import CallSummary, PolicyView, StateSummary

_BASE_VERSION = "1.0-frozen"
_AFTER_FAILED_REPAIR = ("rollback", "continue")


@dataclass(frozen=True)
class _Next:
    phase: str
    cycle: int
    failed_before: int
    cycle_start_best_id: str
    repair_base_id: str | None = None
    repair_worker_id: str | None = None
    bold_base_id: str | None = None


def _card_for_metric(
    view: PolicyView, metric: MetricName, primary: WorkerCard | None = None,
) -> WorkerCard:
    matches = [card for card in view.worker_cards if card.objective.metric == metric]
    if len(matches) != 1 or (primary is not None and matches[0].id == primary.id):
        raise ValueError("brief needs distinct primary and repair worker cards")
    return matches[0]


def _cards(view: PolicyView) -> WorkerCard:
    """Find the primary objective worker and check every limit has its own worker."""

    if not view.brief.terms:
        raise ValueError("repair loop requires at least one brief limit")
    primary = [
        card for card in view.worker_cards
        if card.objective.metric == view.brief.objective.metric
    ]
    if len(primary) != 1:
        raise ValueError("brief needs distinct primary and repair worker cards")
    for term in view.brief.terms:
        _card_for_metric(view, term.metric, primary[0])
    return primary[0]


def _violated_term(view: PolicyView, state: StateSummary) -> BriefTerm | None:
    """First brief limit the state fails, using the gate's exact comparison."""

    for term in view.brief.terms:
        start_value = view.start.metrics.get(term.metric)
        value = state.metrics.get(term.metric)
        if start_value is None or value is None:
            continue
        limit = term.factor * start_value + term.offset
        within = value <= limit if term.comparison == "<=" else value >= limit
        if not within:
            return term
    return None


def _repair_card(view: PolicyView, primary: WorkerCard, state: StateSummary) -> WorkerCard:
    term = _violated_term(view, state) or view.brief.terms[0]
    return _card_for_metric(view, term.metric, primary)


def _state(view: PolicyView, state_id: str) -> StateSummary:
    """Resolve a saved policy-safe state without accessing an engine store."""

    for state in reversed((view.start, *view.visited)):
        if state.state_id == state_id:
            return state
    raise ValueError("policy call cites a state absent from the view")


def _improves(view: PolicyView, candidate: StateSummary | None, reference: StateSummary) -> bool:
    if candidate is None:
        return False
    metric: MetricName = view.brief.objective.metric
    first = candidate.metrics.get(metric)
    previous = reference.metrics.get(metric)
    if first is None or previous is None:
        return False
    return first > previous if view.brief.objective.direction == "max" else first < previous


def _context(call: CallSummary, policy: "RepairLoopPolicy") -> dict:
    try:
        value = json.loads(call.context_ref)
    except (ValueError, TypeError) as exc:
        raise ValueError("repair-loop call lacks its journaled cycle context") from exc
    if (
        not isinstance(value, dict)
        or value.get("policy") != policy.id
        or value.get("version") != policy.version
        or value.get("phase") not in ("bold", "repair", "gentle")
        or type(value.get("cycle")) is not int
        or type(value.get("failed_before")) is not int
        or value["cycle"] < 0
        or not 0 <= value["failed_before"] < 2
        or not isinstance(value.get("cycle_start_best_id"), str)
    ):
        raise ValueError("repair-loop call has invalid cycle context")
    return value


def _after_last(view: PolicyView, policy: "RepairLoopPolicy", primary: WorkerCard) -> _Next:
    """Derive the next phase from the last completed worker call."""

    best_now = (view.best_valid or view.start).state_id
    if not view.calls:
        return _Next("bold", 0, 0, best_now)
    call = view.calls[-1]
    previous = _context(call, policy)
    cycle = previous["cycle"]
    failed = previous["failed_before"]
    before_id = previous["cycle_start_best_id"]
    before = _state(view, before_id)
    selected = _state(view, call.selected_state_id) if call.selected_state_id else None
    returned_good = bool(
        selected and selected.brief_valid and _improves(view, selected, before)
    )
    visited_good = _improves(view, view.best_valid, before)
    new_cycle = _Next("bold", cycle + 1, 0, best_now)

    def failed_cycle() -> _Next:
        if policy.gentle_step:
            return _Next("gentle", cycle, failed, before_id)
        return _Next("bold", cycle + 1, failed + 1, best_now)

    if previous["phase"] == "bold":
        if returned_good:
            return new_cycle
        if (
            selected is not None and selected.physically_valid
            and call.selected_verdict_status == VerdictStatus.BRIEF_VIOLATION
            and _improves(view, selected, before)
        ):
            return _Next(
                "repair", cycle, failed, before_id, selected.state_id,
                _repair_card(view, primary, selected).id,
            )
        if visited_good:
            return new_cycle
        return failed_cycle()

    if previous["phase"] == "repair":
        if returned_good or visited_good:
            return new_cycle
        if (
            policy.after_failed_repair == "continue"
            and selected is not None and selected.physically_valid
            and _improves(view, selected, before)
        ):
            return _Next("bold", cycle + 1, failed, before_id, bold_base_id=selected.state_id)
        return failed_cycle()

    if returned_good or visited_good:
        return new_cycle
    return _Next("bold", cycle + 1, failed + 1, best_now)


class RepairLoopPolicy:
    """Fixed 2-run bold, repair and gentle cycle; repair knobs default to v1.

    A duplicate call is skipped. Two consecutive cycles with no objective
    improvement of a brief-valid state stop the episode.
    """

    id = "repair_loop"
    version = _BASE_VERSION

    def __init__(
        self, *, repair_attempts: int = 1, repair_preserve_placement: bool = True,
        gentle_step: bool = True, after_failed_repair: str = "rollback",
    ) -> None:
        if repair_attempts not in (1, 2) or type(repair_attempts) is not int:
            raise ValueError("repair_attempts must be 1 or 2")
        if type(repair_preserve_placement) is not bool or type(gentle_step) is not bool:
            raise ValueError("repair_preserve_placement and gentle_step must be booleans")
        if after_failed_repair not in _AFTER_FAILED_REPAIR:
            raise ValueError("after_failed_repair must be rollback or continue")
        self.repair_attempts = repair_attempts
        self.repair_preserve_placement = repair_preserve_placement
        self.gentle_step = gentle_step
        self.after_failed_repair = after_failed_repair
        self.variant = (
            repair_attempts, repair_preserve_placement, gentle_step, after_failed_repair,
        )
        if self.variant == (1, True, True, "rollback"):
            self.version = _BASE_VERSION
        else:
            self.version = (
                f"{_BASE_VERSION}+ra{repair_attempts}-rp{int(repair_preserve_placement)}"
                f"-g{int(gentle_step)}-{after_failed_repair}"
            )

    def decide(self, view: PolicyView) -> Mandate | None:
        if view.budget_left <= 0:
            return None
        primary = _cards(view)
        step = _after_last(view, self, primary)
        seen = {
            (
                call.worker_id, call.base_state_id,
                call.settings.preserve_placement,
                call.settings.resolved_max_runs(
                    next(card for card in view.worker_cards if card.id == call.worker_id)
                ),
            )
            for call in view.calls
        }
        best_id = (view.best_valid or view.start).state_id
        while step.failed_before < 2:
            if step.phase == "bold":
                card, base_id, preserve, attempts = (
                    primary, step.bold_base_id or best_id, False, 2,
                )
            elif step.phase == "repair":
                if step.repair_base_id is None or step.repair_worker_id is None:
                    raise ValueError("repair phase lacks its invalid output or worker")
                card = next(c for c in view.worker_cards if c.id == step.repair_worker_id)
                base_id = step.repair_base_id
                preserve = self.repair_preserve_placement
                attempts = self.repair_attempts
            elif step.phase == "gentle":
                card, base_id, preserve, attempts = primary, best_id, True, 1
            else:
                raise ValueError("unknown repair-loop phase")
            key = (card.id, base_id, preserve, attempts)
            if key in seen:
                if step.phase in ("bold", "repair"):
                    if self.gentle_step:
                        step = _Next(
                            "gentle", step.cycle, step.failed_before,
                            step.cycle_start_best_id,
                        )
                    else:
                        step = _Next("bold", step.cycle + 1, step.failed_before + 1, best_id)
                else:
                    failures = step.failed_before + 1
                    step = _Next("bold", step.cycle + 1, failures, best_id)
                continue
            if view.budget_left < attempts:
                return None
            context = json.dumps({
                "policy": self.id,
                "version": self.version,
                "phase": step.phase,
                "cycle": step.cycle,
                "failed_before": step.failed_before,
                "cycle_start_best_id": step.cycle_start_best_id,
            }, sort_keys=True, separators=(",", ":"))
            identity = json.dumps([
                view.episode_id, self.version, len(view.calls), key,
            ], separators=(",", ":"))
            mandate_id = "mandate_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            return Mandate(
                id=mandate_id,
                worker_id=card.id,
                base_state_id=base_id,
                settings=WorkerSettings(
                    preserve_placement=preserve, max_runs=attempts,
                ),
                guards=(),
                brief_hash=view.brief.fingerprint(),
                context_ref=context,
            )
        return None
