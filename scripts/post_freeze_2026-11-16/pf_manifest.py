"""Post-freeze read-only union manifests (Mon 16 Nov 2026), built with the campaign's generic builder (imported, not edited).

  test   the roots of queue 7 (scripts/test_r2_2026-11-13/tr_manifest.roots('llm7')) plus the engines of queues 5 and 7, last, so their AMD
         records win a duplicated key
  train  the D6-v2 training roots (reversed, the v4 root wins) plus the power-review training engine, last
  dev    the D6-v2 development roots plus the power-review development engine, last
Nothing is written in a source root. One kind per process (the two campaign_manifest modules cannot share one interpreter).
Output: data/test_r2_2026-11-13/post_freeze_2026-11-16/manifest_<kind>/ (manifest.json, manifest.sha256, report.json), written once.
usage: pf_manifest.py test|train|dev [--write]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
BASE = PROJECT / "data/test_r2_2026-11-13"
OUT = BASE / "post_freeze_2026-11-16"
PR = BASE / "power_review"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
from orch.engine.store import _atomic_bytes, _json_bytes  # noqa: E402


def build(kind: str):
    if kind == "test":
        sys.path.insert(0, str(PROJECT / "scripts/test_r2_2026-11-13"))
        import tr_manifest as tm  # noqa: E402

        roots = tm.roots("llm7")
        roots["llm5_queue"] = BASE / "sealed/llm5/test/engine"
        roots["llm7_queue"] = BASE / "sealed/llm7/test/engine"
        missing = [k for k, v in roots.items() if not (v / "cache").is_dir()]
        roots = {k: v for k, v in roots.items() if (v / "cache").is_dir()}
        ev = {"test_base": roots["test_base"], "campaign": roots["campaign"], "exhaustive_d3": roots["exhaustive_d3"]}
        body, report = tm.cm.build_generic(roots, tm.METER, tm.B2, ev)
        return roots, missing, body, report
    sys.path.insert(0, str(PROJECT / "scripts/d6v2_2026-11-12"))
    import d6_manifests as dm  # noqa: E402

    meter, train_starts = dm.cm.train_inputs()
    if kind == "train":
        roots = dm.train_roots()
        roots["power_review_train"] = PR / "train/engine"
        starts, ev = train_starts, {"v4": dm.cm.V4}
    else:
        roots = dm.dev_roots()
        roots["power_review_dev"] = PR / "dev/engine"
        starts = dm.dev_starts()
        ev = {f"bx_{d}": dm.dc.BX / d for d in dm.dc.DEVELOPMENT}
    missing = [k for k, v in roots.items() if not (v / "cache").is_dir()]
    body, report = dm.cm.build_generic(roots, meter, starts, ev)
    return roots, missing, body, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kind", choices=("test", "train", "dev"))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    roots, missing, body, report = build(args.kind)
    summary = {k: v for k, v in report.items() if k != "problems"}
    summary.update({"kind": args.kind, "roots": {k: str(v) for k, v in roots.items()}, "roots_without_cache_folder": missing,
                    "problems_count": len(report["problems"]), "problems_first": report["problems"][:5]})
    by_root: dict[str, int] = {}
    for e in body["entries"].values():
        r = e["root"].split(":")[0]
        by_root[r] = by_root.get(r, 0) + 1
    summary["entries_by_root"] = by_root
    print(json.dumps(summary, indent=1, sort_keys=True, default=str))
    if args.write:
        folder = OUT / f"manifest_{args.kind}"
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
