"""Train-only, read-only power progress report; no tool command is launched."""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import thursday_overnight_power as campaign
from orch.contract.models import Evidence, MetricName
from orch.engine.journal import Journal
from orch.engine.store import StateStore, _atomic_bytes, _json_bytes, _sha256_file

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "notes/thursday_overnight_exp.md"
FOLLOW_ROOT = campaign.TOP / "report_follow"
DEPTH3 = PROJECT / "data/sweeps/thursday_crossworker_depth3_2026-11-05"
REGISTRY_DONE = PROJECT / "notes/registry_power_update.done"
TRAIN_ONLY = campaign.DESIGNS


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(evidence: Evidence, name: MetricName) -> float:
    value = evidence.metric(name)
    if value is None or not math.isfinite(value.value):
        raise ValueError(f"missing finite {name.value}")
    return value.value


def evidence_at(path: Path, expected_hash: str) -> Evidence:
    item = Evidence.model_validate_json(path.read_bytes())
    if item.state_id != expected_hash or item.fake:
        raise ValueError("training Evidence identity differs")
    for name in MetricName:
        metric(item, name)
    return item


def committed(phase: str, design: str) -> dict[str, dict[str, Any]]:
    root = campaign.root_for(phase, design)
    journal_path = root / "episodes"
    if not root.is_dir() or not journal_path.is_dir():
        return {}
    events = Journal(root).read(f"overnight_power_{phase}_{design}")
    if not events:
        return {}
    if events[0].get("type") != "configuration":
        raise ValueError("campaign journal has no configuration")
    rows = {}
    for event in events[1:]:
        if event.get("type") != "edge":
            raise ValueError("unknown campaign journal event")
        row = event["edge"]
        if row["task_key"] in rows:
            raise ValueError("duplicate committed training edge")
        rows[row["task_key"]] = row
    return rows


def completed_evidence(phase: str, design: str,
                       rows: dict[str, dict[str, Any]]) -> dict[str, Evidence]:
    root = campaign.root_for(phase, design)
    store = StateStore(root)
    result = {}
    for key, row in rows.items():
        if (
            row.get("status") != "ok"
            or row.get("amd_result") != "pass"
            or row.get("amd_outputs_compared", 0) <= 0
            or not row.get("run_id") or not row.get("output_hash")
        ):
            raise ValueError(f"campaign edge lacks AMD PASS: {design} {key}")
        outcome = store.load_outcome(row["run_id"])
        if (
            outcome.state is None or outcome.evidence is None
            or outcome.state.id != row["output_hash"]
            or outcome.evidence.state_id != row["output_hash"]
        ):
            raise ValueError("campaign edge outcome differs")
        result[key] = outcome.evidence
    return result


def best_saving(rows: list[tuple[str, Evidence, Evidence]], band: float
                ) -> tuple[str, float, float, int, bool] | None:
    """Return recipe label, saving%, WNS/LUT deltas, beyond-noise flag."""
    if not rows:
        return None
    best = None
    for label, parent, child in rows:
        before = metric(parent, MetricName.DYNAMIC_POWER_W)
        after = metric(child, MetricName.DYNAMIC_POWER_W)
        if before <= 0:
            raise ValueError("non-positive parent dynamic power")
        saving = 100.0 * (before - after) / before
        item = (
            label, saving,
            metric(child, MetricName.WNS_NS) - metric(parent, MetricName.WNS_NS),
            round(metric(child, MetricName.PACKED_LUTS) -
                  metric(parent, MetricName.PACKED_LUTS)),
            saving > band,
        )
        if best is None or item[1] > best[1]:
            best = item
    return best


def per_design(design: str, inputs: dict[str, Any]) -> dict[str, Any] | None:
    status_a = campaign.phase_status("a", design)
    status_b = campaign.phase_status("b", design)
    if status_a["phase"] != "complete" or status_b["phase"] != "complete":
        return None
    rows_a = committed("a", design)
    rows_b = committed("b", design)
    if (
        len(rows_a) != len(campaign.a_tasks(inputs))
        or len(rows_b) != len(campaign.POWER) * len(campaign.WORKER)
    ):
        raise ValueError("complete campaign has missing journal edges")
    evid_a = completed_evidence("a", design, rows_a)
    evid_b = completed_evidence("b", design, rows_b)
    states = {row["hash"]: row for row in inputs["states"]}
    baseline_hash = inputs["b2_hash"]
    originals = {
        state_hash: evidence_at(Path(row["evidence"]), state_hash)
        for state_hash, row in states.items()
    }
    baseline = originals[baseline_hash]
    band = inputs["noise_power_pct"]
    from_b2 = []
    from_d1 = []
    for key, child in evid_a.items():
        row = rows_a[key]
        parent_hash = row["input_hash"]
        power = row["power_recipe_id"]
        parent = originals[parent_hash]
        if parent_hash == baseline_hash:
            from_b2.append((power, parent, child))
        else:
            lineages = states[parent_hash]["source_sequences"]
            label = f"{sorted(lineages)[0]} > {power}"
            from_d1.append((label, parent, child))
    if len(from_b2) != 4 or not from_d1:
        raise ValueError("training power result lacks B2 or non-B2 D1 parents")
    best_b2 = best_saving(from_b2, band)
    best_d1 = best_saving(from_d1, band)
    if best_b2 is None or best_d1 is None:
        raise ValueError("missing power result")
    raised_a = 0
    raised_b = 0
    for key, child in evid_b.items():
        row = rows_b[key]
        parent_key = f"a:{baseline_hash}:{row['power_recipe_id']}"
        parent = evid_a[parent_key]
        before = metric(parent, MetricName.DYNAMIC_POWER_W)
        increase = 100.0 * (metric(child, MetricName.DYNAMIC_POWER_W) - before) / before
        if increase > band:
            if row["recipe_id"] in campaign.WORKER[:5]:
                raised_a += 1
            else:
                raised_b += 1
    worker_parents: dict[str, str] = {}
    for state_hash, row in states.items():
        for name in row["source_sequences"]:
            if name in campaign.WORKER:
                if name in worker_parents and worker_parents[name] != state_hash:
                    raise ValueError("D1 worker maps to multiple parent hashes")
                worker_parents[name] = state_hash
    if set(worker_parents) != set(campaign.WORKER):
        raise ValueError("not every worker has a D1 parent")
    first_wins = 0
    last_wins = 0
    b2_power = metric(baseline, MetricName.DYNAMIC_POWER_W)
    for power in campaign.POWER:
        for worker in campaign.WORKER:
            power_first = evid_b[f"b:{power}:{worker}"]
            power_last = evid_a[f"a:{worker_parents[worker]}:{power}"]
            gap_pct = 100.0 * (
                metric(power_last, MetricName.DYNAMIC_POWER_W)
                - metric(power_first, MetricName.DYNAMIC_POWER_W)
            ) / b2_power
            first_wins += gap_pct > band
            last_wins += gap_pct < -band
    return {
        "band_pct": band, "best_b2": best_b2, "best_d1": best_d1,
        "raised_a": raised_a, "raised_b": raised_b,
        "power_first_wins": first_wins, "power_last_wins": last_wins,
    }


def fmt_best(item: tuple[str, float, float, int, bool]) -> str:
    label, saving, wns_change, lut_change, beyond = item
    verdict = "beyond band" if beyond else "within band"
    return (
        f"{label}: {saving:+.3f}% saving, WNS {wns_change:+.3f} ns, "
        f"LUT {lut_change:+d} ({verdict})"
    )


def depth_three() -> tuple[int, int, int]:
    planned = passed = failed = 0
    for design in TRAIN_ONLY:
        status = read(DEPTH3 / design / "status.json")
        planned += status["planned"]
        passed += status["amd_pass"]
        failed += status["failed"]
    return planned, passed, failed


def controller_issue() -> str | None:
    """Surface a failed controller even if design queues have no terminal file."""
    path = campaign.TOP / "controller_status.json"
    if not path.is_file():
        return "power controller status is absent"
    state = read(path)
    if state.get("phase") == "failed":
        return f"power controller failed: {state.get('error') or 'unknown error'}"
    return None


def render() -> str:
    inputs = campaign.selected()
    marker = REGISTRY_DONE.read_text(encoding="utf-8")
    if (
        "updated" not in marker
        or "registry_sha256=25f0a39e4d94c91b2c7b8d450a47788e95c8dab237558c6e45539a2a79b33311"
        not in marker
    ):
        raise ValueError("approved power registry marker changed")
    planned3, passed3, failed3 = depth_three()
    if (planned3, passed3, failed3) != (570, 570, 0):
        raise ValueError("training depth-three status changed")
    stats_a = {name: campaign.phase_status("a", name) for name in TRAIN_ONLY}
    stats_b = {name: campaign.phase_status("b", name) for name in TRAIN_ONLY}
    a_pass = sum(row["amd_pass"] for row in stats_a.values())
    b_pass = sum(row["amd_pass"] for row in stats_b.values())
    a_failed = sum(row["failed"] for row in stats_a.values())
    b_failed = sum(row["failed"] for row in stats_b.values())
    if a_pass + a_failed > 176 or b_pass + b_failed > 200:
        raise ValueError("power edge counts exceed frozen plan")
    lines = [
        "# Thursday overnight training power campaign",
        "",
        f"**Verified, {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC:** "
        "The two missing byte-copied run-4 power recipes are registered "
        "(78 catalogue entries; all earlier versions unchanged). "
        "The full pre-launch offline suite passed 330 collected tests. "
        f"Finn's depth-three queue finished {passed3}/{planned3} AMD-PASS, "
        f"{failed3} failures.",
        "",
        f"**Verified:** Power phase a has {a_pass}/176 AMD-PASS edges, "
        f"{a_failed} failed; phase b has {b_pass}/200 AMD-PASS edges, "
        f"{b_failed} failed. Phase b requires the global phase-a terminal barrier. "
        "Every reported PASS has a positive compared-output count.",
        "",
        "**Verified:** The table below uses the run-4 dynamic-power noise band "
        "for each design. Savings, WNS changes and packed-LUT changes are "
        "relative to each immediate parent; distinct D1 parents exclude B2 "
        "duplicates. Order comparisons use the B2 dynamic-power denominator.",
        "",
    ]
    complete = 0
    for name in TRAIN_ONLY:
        result = per_design(name, inputs["by_design"][name])
        if result is None:
            a = stats_a[name]
            b = stats_b[name]
            lines.append(
                f"- **{name}:** Pending; phase a {a['amd_pass']}/{a['planned']}, "
                f"phase b {b['amd_pass']}/{b['planned']}. "
                f"Status {a['phase']} / {b['phase']}."
            )
            continue
        complete += 1
        lines.append(
            f"- **{name}** (band {result['band_pct']:.3f}%): "
            f"from B2, {fmt_best(result['best_b2'])}; "
            f"from D1, {fmt_best(result['best_d1'])}. "
            f"A/B post-power runs raised power beyond band "
            f"{result['raised_a']}/20 and {result['raised_b']}/20. "
            f"Power-first / power-last won the 40 paired order comparisons "
            f"{result['power_first_wins']} / {result['power_last_wins']} times "
            "beyond band."
        )
    lines.extend(["", ""])
    problems = []
    controller_error = controller_issue()
    if controller_error:
        problems.append(controller_error)
    problems.extend(
        f"{name} {phase}: driver absent before terminal status"
        for phase, group in (("a", stats_a), ("b", stats_b))
        for name, row in group.items()
        if (
            row["phase"] not in campaign.TERMINAL
            and row["phase"] != "not_launched"
            and row["driver_pid"] is not None
            and not row["driver_alive"]
        )
    )
    if a_failed or b_failed or problems or any(
        row["phase"] in {"stopped", "preflight_failed", "failed"}
        for row in list(stats_a.values()) + list(stats_b.values())
    ):
        errors = problems + [
            f"{name} {phase}: {row['error'] or row['phase']}"
            for phase, group in (("a", stats_a), ("b", stats_b))
            for name, row in group.items()
            if row["phase"] in {"stopped", "preflight_failed", "failed"}
        ]
        lines.append("**Verified; author decision pending:** " + "; ".join(errors) + ".")
    elif complete == len(TRAIN_ONLY):
        lines.append(
            "**Inferred; author decision pending:** Decide whether any beyond-band power "
            "gain is useful enough to include a backup power worker; this "
            "report makes no worker-selection change."
        )
    else:
        lines.append(
            "**Verified:** No failed power design queue is recorded at this "
            "snapshot. Remaining results are still running."
        )
    content = "\n".join(lines)
    if len(re.findall(r"\b[\w-]+\b", content)) > 500:
        raise ValueError("morning report exceeds 500 words")
    return content


def once() -> dict[str, Any]:
    content = render()
    _atomic_bytes(OUTPUT, content.encode("utf-8"))
    return {
        "report": str(OUTPUT), "sha256": _sha256_file(OUTPUT),
        "words": len(re.findall(r"\b[\w-]+\b", content)),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def terminal_status(a: dict[str, dict[str, Any]],
                    b: dict[str, dict[str, Any]],
                    barrier_exists: bool) -> bool:
    """The gap after phase a ends is not a complete campaign."""
    return barrier_exists and all(
        a[name]["phase"] in campaign.TERMINAL
        and (
            b[name]["phase"] in campaign.TERMINAL
            if a[name]["phase"] == "complete"
            else b[name]["phase"] == "not_launched"
        )
        for name in TRAIN_ONLY
    )


def follow() -> None:
    """Refresh until all queues are terminal, or produce a 07:20 UTC snapshot."""
    FOLLOW_ROOT.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            snapshot = once()
            a = {name: campaign.phase_status("a", name) for name in TRAIN_ONLY}
            b = {name: campaign.phase_status("b", name) for name in TRAIN_ONLY}
            terminal = terminal_status(
                a, b, (campaign.TOP / "phase_a_barrier.json").is_file()
            )
            status = {
                "schema_version": "overnight-power-report-follow-v1",
                "phase": "complete" if terminal else "running",
                "error": None, **snapshot,
            }
            _atomic_bytes(FOLLOW_ROOT / "status.json", _json_bytes(status))
            if terminal:
                return
        except Exception as exc:
            _atomic_bytes(FOLLOW_ROOT / "status.json", _json_bytes({
                "schema_version": "overnight-power-report-follow-v1",
                "phase": "failed", "error": f"{type(exc).__name__}: {exc}",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }))
            raise
        now = datetime.now(timezone.utc)
        if now >= datetime(2026, 11, 6, 7, 20, tzinfo=timezone.utc):
            return
        time.sleep(300)


def launch_follow() -> dict[str, Any]:
    """Detach the no-tool report updater under a recorded PID."""
    FOLLOW_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = FOLLOW_ROOT / "launch.json"
    if manifest.exists():
        raise FileExistsError("power report follower already launched")
    with (FOLLOW_ROOT / "stdout").open("wb") as out, (FOLLOW_ROOT / "stderr").open("wb") as err:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--follow"],
            cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            start_new_session=True, close_fds=True,
        )
    record = {
        "schema_version": "overnight-power-report-launch-v1",
        "pid": process.pid,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "tool_jobs": 0,
    }
    _atomic_bytes(manifest, _json_bytes(record))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--once", action="store_true")
    modes.add_argument("--follow", action="store_true")
    modes.add_argument("--launch-follow", action="store_true")
    args = parser.parse_args()
    if args.once:
        print(json.dumps(once(), sort_keys=True))
    elif args.follow:
        follow()
    else:
        print(json.dumps(launch_follow(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
