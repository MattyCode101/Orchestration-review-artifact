"""Replay equivalence on the training designs: the campaign harness, no model calls, no tool runs.

Runs F1-R1, F1-R0, E0-R1 and GA-R1 on the training designs (5 x 4 briefs, the two P0 episodes, the fresh repeats) from the development
the development run's recorded model replies and compares each episode's journal (decision path, final state,
charged cost) with the journal the development run stored. Writes the report under
data/campaign_replay_2026-11-07/. Training designs only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT))

import campaign_harness as h  # noqa: E402
from orch.engine.journal import Journal  # noqa: E402
from orch.engine.store import _atomic_bytes, _json_bytes, _sha256_file  # noqa: E402

WORK = PROJECT / "data/campaign_replay_2026-11-07"
MANIFEST = PROJECT / "data/campaign_prep_2026-11-07/train_cache_manifest.json"
# Every recorded flash-lite training episode of a campaign cell. GA-R1 is the new general-agent template (ga3);
# the first template's records (ga1) and the dropped F2 and the F0 / E1 / E0-R0 configurations are not campaign cells.
# uw holds the two worker C (P0) episodes; var1 and var2 are fresh repeats of F1-R1 and E0-R1.
STAGES = (("grid1", "F1", "F1-R0"), ("grid2", "F1", "F1-R1"), ("grid2", "E0", "E0-R1"), ("ga3", "GA", "GA-R1"),
          ("uw", "F1", "F1-R1"), ("uw", "E0", "E0-R1"),
          ("var1", "F1", "F1-R1"), ("var1", "E0", "E0-R1"), ("var2", "F1", "F1-R1"), ("var2", "E0", "E0-R1"))


def cells() -> list[h.Cell]:
    out = []
    for stage, config, approach in STAGES:
        folder = PROJECT / f"data/pilots/saturday_llm_{stage}_2026-11-06"
        work = json.loads((folder / "work.json").read_text())
        for c in work["cells"]:
            if c["config"] != config:
                continue
            episode = f"llm_{stage}_{config}_{c['design']}_{c['brief']}_{c['tag']}"
            if not (folder / "results" / f"{episode}.json").is_file():
                raise SystemExit(f"no stored result for {episode}")
            out.append(h.Cell(c["design"], c["brief"], approach, "r0", 0, episode, cache="replay"))
    return out


def shared_root(stage: str) -> Path:
    """Where the development run's stage kept its model replies (stages share one store, not always the same)."""

    plan = json.loads((PROJECT / f"data/pilots/saturday_llm_{stage}_2026-11-06/plan.json").read_text())
    return Path(plan["shared_root"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--work", default="attempt2")
    parser.add_argument("--shard", default="0/1")
    args = parser.parse_args()
    index, count = (int(x) for x in args.shard.split("/"))
    work = WORK / args.work
    scope = h.train_scope(work, MANIFEST)
    ctx = h.build_context(scope, mode="replay")
    rows = cells()[index::count]
    if args.limit:
        rows = rows[: args.limit]
    out = work / "reports"
    (work / "results").mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=True)
    dev_journals = Journal(h.V4)
    report = []
    for cell in rows:
        began = time.monotonic()
        entry = {"episode_id": cell.episode_id, "approach": cell.approach, "design": cell.design, "brief": cell.brief}
        try:
            ctx.notes["replay_cache_root"] = str(shared_root(cell.episode_id.split("_")[1]))
            status, row = h.run_cell(ctx, cell, work / "results")
            if status != "done":
                entry["status"] = status
            else:
                mine = h.normalised_events(Journal(ctx.store.root).read(cell.episode_id))
                theirs = h.normalised_events(dev_journals.read(cell.episode_id))
                diff = h.first_difference(theirs, mine)
                saved = json.loads((PROJECT / f"data/pilots/saturday_llm_{cell.episode_id.split('_')[1]}_2026-11-06/results/{cell.episode_id}.json").read_text())
                result_diff = {
                    k: [saved[k], row[k]] for k in ("final_state_id", "final_metrics", "runs_used", "stop_reason")
                    if saved[k] != row[k]
                }
                entry.update(status="match" if diff is None and not result_diff else "MISMATCH",
                             journal_events=len(mine), first_journal_difference=diff, result_differences=result_diff)
        except Exception as exc:  # reported, never hidden
            entry.update(status="ERROR", error=f"{type(exc).__name__}: {str(exc)[:300]}", trace=traceback.format_exc()[-1800:])
        entry["seconds"] = round(time.monotonic() - began, 1)
        report.append(entry)
        print(json.dumps({k: entry[k] for k in ("episode_id", "status", "seconds")}), flush=True)
    summary = {
        "schema_version": "campaign-replay-equivalence-v1", "shard": args.shard, "episodes": len(report),
        "match": sum(e["status"] == "match" for e in report), "other": [e for e in report if e["status"] != "match"],
        "tool_runs": h.NoToolBackend.calls, "manifest_sha256": scope.manifest.hash,
        "harness_sha256": _sha256_file(Path(h.__file__)), "all": report,
    }
    _atomic_bytes(out / f"replay_equivalence_{index}of{count}.json", _json_bytes(summary))
    print(json.dumps({k: summary[k] for k in ("episodes", "match", "tool_runs")}))
    return 0 if summary["match"] == len(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
