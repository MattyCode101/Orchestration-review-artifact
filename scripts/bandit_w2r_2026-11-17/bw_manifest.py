"""Learned Bandit W2R: read-only union manifest of every stored TEST outcome (Tue 17 Nov 2026), built with the campaign's generic builder (imported, not edited).

Roots = the roots of the post-freeze manifest (tr_manifest.roots('llm7') plus the engines of queues 5 and 7) plus the post-freeze queue's own TEST engine, last, so its
AMD records win a duplicated key. Nothing is written in a source root. Output: data/test_r2_2026-11-13/bandit_w2r_2026-11-17/manifest_test/ (written once).
usage: bw_manifest.py [--write]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
BASE = PROJECT / "data/test_r2_2026-11-13"
OUT = BASE / "bandit_w2r_2026-11-17"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "scripts/test_r2_2026-11-13"))
from orch.engine.store import _atomic_bytes, _json_bytes  # noqa: E402
import tr_manifest as tm  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    roots = tm.roots("llm7")
    roots["llm5_queue"] = BASE / "sealed/llm5/test/engine"
    roots["llm7_queue"] = BASE / "sealed/llm7/test/engine"
    roots["post_freeze_queue"] = BASE / "post_freeze_2026-11-16/test/engine"
    missing = [k for k, v in roots.items() if not (v / "cache").is_dir()]
    roots = {k: v for k, v in roots.items() if (v / "cache").is_dir()}
    ev = {"test_base": roots["test_base"], "campaign": roots["campaign"], "exhaustive_d3": roots["exhaustive_d3"]}
    body, report = tm.cm.build_generic(roots, tm.METER, tm.B2, ev)
    by_root: dict[str, int] = {}
    for e in body["entries"].values():
        r = e["root"].split(":")[0]
        by_root[r] = by_root.get(r, 0) + 1
    summary = {k: v for k, v in report.items() if k != "problems"}
    summary.update({"roots": {k: str(v) for k, v in roots.items()}, "roots_without_cache_folder": missing, "problems_count": len(report["problems"]),
                    "problems_first": report["problems"][:5], "entries_by_root": by_root})
    print(json.dumps(summary, indent=1, sort_keys=True, default=str))
    if args.write:
        folder = OUT / "manifest_test"
        if report["problems"]:
            print("refused: the manifest has problems")
            return 1
        if (folder / "manifest.json").exists():
            print("refused: manifest exists (written once)")
            return 2
        folder.mkdir(parents=True, exist_ok=True)
        _atomic_bytes(folder / "manifest.json", _json_bytes(body))
        _atomic_bytes(folder / "manifest.sha256", (report["manifest_sha256"] + "\n").encode())
        _atomic_bytes(folder / "report.json", _json_bytes(report))
        print("written", folder)
    return 0 if not report["problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
