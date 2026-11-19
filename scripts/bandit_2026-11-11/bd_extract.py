"""Bandit (O1): extract the training call record from the sealed v4 journals. Read-only on v4.

One JSON line per episode: its start state, noise bands as recorded, and every worker call with the
absolute metrics of each attempt's output (start value + the verdict's delta from start). Nothing here
touches a recipe name: the call record is worker, base, placement scope, attempts and measured outcomes.
General-agent episodes are left out of the record: their mandates name a recipe directly.

usage: bd_extract.py [--out DIR]
"""
from __future__ import annotations
import os

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(os.environ.get("ORCH_PROJECT_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(PROJECT))
from orch.engine.store import _json_bytes  # noqa: E402

V4 = PROJECT / "data/pilots/thursday_overnight_train_v4_2026-11-05"
OUT = PROJECT / "data/bandit_2026-11-11/calls"
EXCLUDED_POLICIES = {"general_agent"}


def read_events(path: Path) -> list[dict]:
    """Checksummed events, exactly as orch.engine.journal.Journal does, without opening any lock file."""

    events = []
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break                                     # torn tail: ignored, as the engine ignores it
        wrapper = json.loads(line)
        event = wrapper["event"]
        if wrapper.get("sha256") != hashlib.sha256(_json_bytes(event)).hexdigest():
            raise ValueError(f"checksum mismatch in {path}")
        events.append(event)
    return events


def episode_record(episode_id: str, events: list[dict]) -> dict | None:
    start = events[0]
    if start.get("type") != "start":
        raise ValueError(f"{episode_id}: no start event")
    env = start["envelope"]
    if env["policy"]["id"] in EXCLUDED_POLICIES or env.get("fake"):
        return None
    start_metrics = {name: float(m["value"]) for name, m in start["start_evidence"]["metrics"].items()}
    record = {
        "episode_id": episode_id,
        "source": episode_id.split("_")[0],
        "design": env["start_state"]["design"],
        "start_state_id": env["start_state"]["id"],
        "policy_id": env["policy"]["id"],
        "policy_version": env["policy"]["version"],
        "brief": env["brief"]["id"],
        "budget_runs": int(env["brief"]["budget_runs"]),
        "noise": {k: float(v) for k, v in env["noise"].items()},
        "start_metrics": start_metrics,
        "calls": [],
        "stop_reason": None,
    }
    calls: dict[int, dict] = {}
    run_to_out: dict[str, str | None] = {}
    for event in events[1:]:
        kind = event["type"]
        if kind == "mandate":
            mandate = event["mandate"]
            calls[event["index"]] = {
                "i": event["index"],
                "worker": mandate["worker_id"],
                "base_id": mandate["base_state_id"],
                "preserve": bool(mandate["settings"]["preserve_placement"]),
                "max_runs": mandate["settings"].get("max_runs"),
                "budget_left_before": int(event["budget_left_before"]),
                "steps": [],
                "selected_out_id": None,
                "recipe_runs": 0,
                "charged_wall_s": 0.0,
            }
        elif kind == "step":
            verdict, step = event["verdict"], event["step"]
            physically_valid = bool(verdict["physically_valid"])
            metrics = None
            if physically_valid:
                delta = verdict["delta_from_start"]
                metrics = {name: start_metrics[name] + float(delta[name]) for name in start_metrics if name in delta}
            run_to_out[step["run_id"]] = step["output_state_id"]
            calls[event["decision_index"]]["steps"].append({
                "run_id": step["run_id"], "out_id": step["output_state_id"], "physically_valid": physically_valid,
                "status": verdict["status"], "metrics": metrics,
            })
        elif kind == "selection":
            call = calls[event["index"]]
            run_id = event["selected_run_id"]
            call["selected_out_id"] = run_to_out[run_id] if run_id is not None else None
            call["recipe_runs"] = int(event["recipe_runs"])
            call["charged_wall_s"] = float(event["charged_wall_s"])
            call["completed"] = True
        elif kind == "stop":
            record["stop_reason"] = event["reason"]
    record["calls"] = [calls[i] for i in sorted(calls) if calls[i].get("completed")]
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], Counter()
    for folder in sorted((V4 / "episodes").iterdir()):
        journal = folder / "journal.jsonl"
        if not journal.is_file():
            skipped["no_journal"] += 1
            continue
        record = episode_record(folder.name, read_events(journal))
        if record is None:
            skipped["general_agent_or_fake"] += 1
            continue
        if not record["calls"]:
            skipped["no_completed_call"] += 1
            continue
        rows.append(record)
    path = out / "episodes_v4.jsonl"
    text = "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows) + "\n"
    path.write_text(text, encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    by_source = Counter(r["source"] for r in rows)
    calls = [(r, c) for r in rows for c in r["calls"]]
    arm = Counter((c["worker"], "keep" if c["preserve"] else "free", c["max_runs"]) for _, c in calls)
    summary = {
        "schema": "bandit-call-record-v1", "source_root": str(V4), "episodes": len(rows), "calls": len(calls),
        "skipped": dict(skipped), "episodes_by_source": dict(by_source),
        "episodes_by_design": dict(Counter(r["design"] for r in rows)),
        "episodes_by_brief": dict(Counter(r["brief"] for r in rows)),
        "calls_by_worker": dict(Counter(c["worker"] for _, c in calls)),
        "calls_by_worker_scope_attempts": {"|".join(map(str, k)): v for k, v in sorted(arm.items(), key=str)},
        "calls_without_selected_state": sum(c["selected_out_id"] is None for _, c in calls),
        "calls_by_source_worker": dict(Counter(f"{r['source']}|{c['worker']}" for r, c in calls)),
        "record_sha256": sha,
    }
    (out / "episodes_v4.summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
