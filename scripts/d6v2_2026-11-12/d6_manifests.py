"""D6-v2: build the two read-only union manifests with the campaign's generic builder (imported, not edited).

  training     the template-v2 bank's union of training roots (scripts/template_v2_2026-11-12/t2_common.roots: the
               sealed v4 training root, the power roots, the power-readiness and bandit roots); the builder keeps the
               last root of a duplicated key, so the order is reversed and the v4 root wins.
  development  the benchmark screen's per-design roots plus the development-designs task's overlay (its exhaustive
               two-step and baseline runs); the screen's own run wins a duplicated key.
Nothing is written in a source root. Usage: d6_manifests.py [--write]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d6_common as dc  # noqa: E402

sys.path.insert(0, str(dc.PROJECT / "scripts/template_v2_2026-11-12"))
import campaign_manifest as cm  # noqa: E402
import yaml  # noqa: E402
from orch.engine.store import _atomic_bytes, _json_bytes  # noqa: E402


def train_roots() -> dict[str, Path]:
    import t2_common as C  # noqa: E402

    return dict(reversed(list(C.roots().items())))


def dev_roots() -> dict[str, Path]:
    roots: dict[str, Path] = {}
    if (dc.DEV_ENGINE_D3 / "cache").is_dir():
        roots["dd"] = dc.DEV_ENGINE_D3
    for design in dc.DEVELOPMENT:
        roots[f"bx_{design}"] = dc.BX / dc.guard_design(design)
    return roots


def dev_starts() -> dict[str, str]:
    split = yaml.safe_load((dc.PROJECT / "configs/development/split.yaml").read_text(encoding="utf-8"))
    starts = {d: split["designs"][d]["b2_state_id"] for d in dc.DEVELOPMENT}
    if set(split["designs"]) != set(dc.DEVELOPMENT):
        raise SystemExit("configs/development/split.yaml names other development designs")
    return starts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    meter, train_starts = cm.train_inputs()
    out = {}
    for name, roots, starts, ev_roots, folder in (
        ("train", train_roots(), train_starts, {"v4": cm.V4}, dc.MANIFEST_TRAIN),
        ("dev", dev_roots(), dev_starts(), {f"bx_{d}": dc.BX / d for d in dc.DEVELOPMENT}, dc.MANIFEST_DEV),
    ):
        for design in starts:
            dc.guard_design(design)
        body, report = cm.build_generic(roots, meter, starts, ev_roots)
        summary = {k: v for k, v in report.items() if k != "problems"}
        summary["roots"] = {k: str(v) for k, v in roots.items()}
        summary["problems_count"] = len(report["problems"])
        summary["problems_first"] = report["problems"][:5]
        out[name] = summary
        if args.write:
            if report["problems"]:
                raise SystemExit(f"{name} manifest has problems: {report['problems'][:5]}")
            folder.mkdir(parents=True, exist_ok=True)
            _atomic_bytes(folder / "manifest.json", _json_bytes(body))
            _atomic_bytes(folder / "manifest.sha256", (report["manifest_sha256"] + "\n").encode())
            _atomic_bytes(folder / "report.json", _json_bytes(report))
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
