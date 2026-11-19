"""Pure, start-relative judgment of independently measured evidence.

The gate never chooses a successor state. A physically valid state may violate
a brief and still be useful to the policy as the base of a later recovery step.
See the final architecture, section 6, especially examples 1 and 2.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

from orch.contract.models import (
    Brief,
    CheckResult,
    Evidence,
    Guard,
    MetricName,
    MetricValue,
    RunStatus,
    Verdict,
    VerdictStatus,
)


GATE_VERSION = "0.1.0"


def _checked(
    passed: bool | None,
    *,
    observed: float | str | int | bool | None,
    threshold: float | str | int | bool | None,
    evidence_id: str,
    reason: str | None = None,
) -> CheckResult:
    """Turn a three-valued comparison into a cited contract check."""

    status = "unknown" if passed is None else "pass" if passed else "fail"
    return CheckResult(
        status=status,
        observed=observed,
        threshold=threshold,
        evidence_id=evidence_id,
        reason=reason,
    )


def _same_measurement(
    reference: Evidence, candidate: Evidence, metric: MetricName
) -> tuple[MetricValue | None, MetricValue | None, str | None]:
    """Refuse arithmetic across meter versions, methods, bases or units."""

    first = reference.metric(metric)
    second = candidate.metric(metric)
    if first is None or second is None:
        return first, second, f"missing {metric.value}"
    if reference.meter_version != candidate.meter_version:
        return first, second, "meter versions differ"
    if (first.unit, first.method, first.basis) != (
        second.unit,
        second.method,
        second.basis,
    ):
        return first, second, "metric unit, method or basis differs"
    return first, second, None


def _physical_checks(
    brief: Brief, start: Evidence, candidate: Evidence
) -> dict[str, CheckResult]:
    """Implement run-3 valid3's routed/DRC/fingerprint terms.

    The meter normalizes XDC and clock text using collect3.py:20-40. Property
    counts are diagnostic and do not enter this predicate (collect3.py:93-95).
    """

    checks: dict[str, CheckResult] = {}
    if start.fake != candidate.fake:
        checks["evidence_origin"] = _checked(
            False,
            observed="fake" if candidate.fake else "real",
            threshold="fake" if start.fake else "real",
            evidence_id=candidate.id,
            reason="cannot combine fake and real evidence",
        )
    if start.meter_version != candidate.meter_version:
        checks["meter_version"] = _checked(
            False,
            observed=candidate.meter_version,
            threshold=start.meter_version,
            evidence_id=candidate.id,
            reason="remeasure the start with the current meter",
        )
    if brief.require_routed:
        if candidate.routed is None or candidate.route_errors is None:
            routed_ok: bool | None = None
        else:
            routed_ok = candidate.routed and candidate.route_errors == 0
        checks["routed"] = _checked(
            routed_ok,
            observed=(
                None
                if candidate.routed is None or candidate.route_errors is None
                else f"routed={candidate.routed}; route_errors={candidate.route_errors}"
            ),
            threshold="routed=True; route_errors=0",
            evidence_id=candidate.id,
        )
    if candidate.drc_errors is None:
        drc_ok: bool | None = None
    else:
        drc_ok = candidate.drc_errors <= brief.max_drc_errors
    checks["drc_errors"] = _checked(
        drc_ok,
        observed=candidate.drc_errors,
        threshold=brief.max_drc_errors,
        evidence_id=candidate.id,
    )
    if brief.require_constraint_match:
        xdc_ok = (
            None
            if not start.constraint_fingerprint or not candidate.constraint_fingerprint
            else candidate.constraint_fingerprint == start.constraint_fingerprint
        )
        checks["timing_constraints"] = _checked(
            xdc_ok,
            observed=candidate.constraint_fingerprint,
            threshold=start.constraint_fingerprint,
            evidence_id=candidate.id,
        )
    if brief.require_clock_match:
        clock_ok = (
            None
            if not start.clock_digest or not candidate.clock_digest
            else candidate.clock_digest == start.clock_digest
        )
        checks["clocks"] = _checked(
            clock_ok,
            observed=candidate.clock_digest,
            threshold=start.clock_digest,
            evidence_id=candidate.id,
        )
    return checks


def _term_checks(
    brief: Brief, start: Evidence, candidate: Evidence
) -> dict[str, CheckResult]:
    """Judge each hard term against the episode start, never the parent."""

    results: dict[str, CheckResult] = {}
    for term in brief.terms:
        ref, got, issue = _same_measurement(start, candidate, term.metric)
        if issue is not None:
            results[term.id] = _checked(
                None,
                observed=got.value if got is not None else None,
                threshold=None,
                evidence_id=candidate.id,
                reason=issue,
            )
            continue
        assert ref is not None and got is not None
        limit = term.threshold(start)
        passed = (
            got.value <= limit
            if term.comparison == "<="
            else got.value >= limit
        )
        results[term.id] = _checked(
            passed,
            observed=got.value,
            threshold=limit,
            evidence_id=candidate.id,
        )
    return results


def _guard_checks(
    start: Evidence, candidate: Evidence, guards: Sequence[Guard]
) -> dict[str, CheckResult]:
    """Absolute mandate guards add information; they do not change brief validity."""

    results: dict[str, CheckResult] = {}
    for guard in guards:
        _, got, issue = _same_measurement(start, candidate, guard.metric)
        if issue is not None:
            results[guard.id] = _checked(
                None,
                observed=got.value if got is not None else None,
                threshold=guard.threshold,
                evidence_id=candidate.id,
                reason=issue,
            )
            continue
        assert got is not None
        passed = (
            got.value <= guard.threshold
            if guard.comparison == "<="
            else got.value >= guard.threshold
        )
        results[guard.id] = _checked(
            passed,
            observed=got.value,
            threshold=guard.threshold,
            evidence_id=candidate.id,
        )
    return results


def _deltas(
    reference: Evidence, candidate: Evidence
) -> dict[MetricName, float]:
    """Report only comparisons with the same measured meaning."""

    result: dict[MetricName, float] = {}
    for metric in MetricName:
        first, second, issue = _same_measurement(reference, candidate, metric)
        if issue is None:
            assert first is not None and second is not None
            delta = second.value - first.value
            if not math.isfinite(delta):
                raise ValueError(f"metric delta is not finite: {metric.value}")
            result[metric] = delta
    return result


def _with_id(
    verdict: Verdict,
    *,
    brief: Brief,
    start: Evidence,
    parent: Evidence,
    run_status: RunStatus,
) -> Verdict:
    """Bind a deterministic verdict id to its full judgment context."""

    payload = {
        "verdict": verdict.model_dump(mode="json", exclude={"id"}),
        "brief_hash": brief.fingerprint(),
        "start_evidence_id": start.id,
        "parent_evidence_id": parent.id,
        "run_status": run_status.value,
        "gate_version": GATE_VERSION,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return verdict.model_copy(
        update={"id": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}
    )

def judge(
    brief: Brief,
    start: Evidence,
    parent: Evidence,
    candidate: Evidence | None,
    guards: Sequence[Guard] = (),
    noise: Mapping[MetricName, float] | None = None,
    run_status: RunStatus = RunStatus.OK,
) -> Verdict:
    """Judge the measured state without keeping, rejecting or rolling it back.

    Noise flags describe start-relative changes; they never change a pass/fail.
    A run failure has no admissible evidence. An incomplete or incomparable
    measurement is INVALID rather than an accidental pass.
    """

    run_status = RunStatus(run_status)
    if candidate is None or run_status not in (RunStatus.OK, RunStatus.INVALID):
        return _with_id(
            Verdict(
                id="pending",
                evidence_id=candidate.id if candidate is not None else None,
                status=VerdictStatus.RUN_FAILED,
                physically_valid=False,
                brief_valid=False,
                guard_pass=False,
                fake=candidate.fake if candidate is not None else start.fake,
            ),
            brief=brief,
            start=start,
            parent=parent,
            run_status=run_status,
        )

    validity = _physical_checks(brief, start, candidate)
    terms = _term_checks(brief, start, candidate)
    guard_results = _guard_checks(start, candidate, guards)

    # Keep physical validity separate from missing objective evidence.
    physically_valid = (
        all(check.status == "pass" for check in validity.values())
        and run_status == RunStatus.OK
    )

    # An objective without comparable evidence cannot be ranked in final pick.
    _, _, objective_issue = _same_measurement(start, candidate, brief.objective.metric)
    if objective_issue is not None:
        validity["objective_metric"] = _checked(
            None,
            observed=None,
            threshold=brief.objective.metric.value,
            evidence_id=candidate.id,
            reason=objective_issue,
        )
    if run_status == RunStatus.INVALID:
        validity["run_status"] = _checked(
            False,
            observed=run_status.value,
            threshold=RunStatus.OK.value,
            evidence_id=candidate.id,
        )

    terms_pass = all(check.status == "pass" for check in terms.values())
    guard_pass = all(check.status == "pass" for check in guard_results.values())
    brief_valid = physically_valid and terms_pass and objective_issue is None

    hard_unknown = objective_issue is not None or any(
        check.status == "unknown" for check in terms.values()
    )
    guard_unknown = any(
        check.status == "unknown" for check in guard_results.values()
    )
    if not physically_valid or hard_unknown:
        status = VerdictStatus.INVALID
    elif not terms_pass:
        status = VerdictStatus.BRIEF_VIOLATION
    elif guard_unknown:
        status = VerdictStatus.INVALID
    elif not guard_pass:
        status = VerdictStatus.GUARD_VIOLATION
    else:
        status = VerdictStatus.ACCEPTABLE

    against_start = _deltas(start, candidate)
    against_parent = _deltas(parent, candidate)
    flags: dict[MetricName, bool] = {}
    for raw_metric, band in (noise or {}).items():
        metric = MetricName(raw_metric)
        if not math.isfinite(band) or band < 0:
            raise ValueError("noise band must be finite and nonnegative")
        if metric in against_start:
            flags[metric] = abs(against_start[metric]) <= band

    return _with_id(
        Verdict(
            id="pending",
            evidence_id=candidate.id,
            status=status,
            physically_valid=physically_valid,
            brief_valid=brief_valid,
            guard_pass=guard_pass,
            validity=validity,
            terms=terms,
            guards=guard_results,
            delta_from_start=against_start,
            delta_from_parent=against_parent,
            within_noise=flags,
            fake=candidate.fake,
        ),
        brief=brief,
        start=start,
        parent=parent,
        run_status=run_status,
    )

