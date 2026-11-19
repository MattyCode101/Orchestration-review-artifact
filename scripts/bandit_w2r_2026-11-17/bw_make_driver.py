"""Learned Bandit W2R driver (Tue 17 Nov 2026): makes tr_common.py, tr_episode.py, tr_policies.py in scripts/bandit_w2r_2026-11-17/ from COPIES of the
post-freeze driver (scripts/post_freeze_2026-11-16, itself a patched copy of the pinned scripts/test_r2_2026-11-13). The originals are only read. Every change
is a substitution that must match exactly once, so a change in an original refuses the build instead of producing a different driver.

  tr_common.py    queue slot 'bw' only; run folder data/test_r2_2026-11-13/bandit_w2r_2026-11-17/; plan note notes/bandit_w2r_postfreeze_plan.md; caps 20 new tool runs
                  and 8 job-hours (decision of Tue 17 Nov); TEST designs only (no TRAINING or VALIDATION design is opened, no t2_or1200); the post-freeze telemetry cache is a read-only seed
  tr_policies.py  the row 'bandit' returns the W2R post-freeze entry point (orch.policies.bandit_w2r_postfreeze.bandit_w2r) instead of the frozen one, which refuses W2R
  tr_episode.py   unchanged copy
usage: python bw_make_driver.py   (project root, project venv; refuses to overwrite)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SRC = PROJECT / "scripts/post_freeze_2026-11-16"
DST = PROJECT / "scripts/bandit_w2r_2026-11-17"
LOG: list[str] = []


def sub(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"refused: {what}: expected exactly one match, found {text.count(old)}")
    LOG.append(what)
    return text.replace(old, new)


def line_sub(text: str, prefix: str, new_line: str, what: str) -> str:
    lines = text.split("\n")
    hits = [i for i, ln in enumerate(lines) if ln.startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"refused: {what}: expected exactly one line starting {prefix!r}, found {len(hits)}")
    lines[hits[0]] = new_line
    LOG.append(what)
    return "\n".join(lines)


def common(t: str) -> str:
    t = sub(t, 'POST-FREEZE COPY (Mon 16 Nov 2026) of scripts/test_r2_2026-11-13/tr_common.py: queue slot "pf" only, changes listed in pf_make_driver.py.',
            'LEARNED-BANDIT W2R COPY (Tue 17 Nov 2026) of scripts/post_freeze_2026-11-16/tr_common.py: queue slot "bw" only, changes listed in bw_make_driver.py.', "docstring")
    t = sub(t, 'if QUEUE != "pf":\n    raise SystemExit("refused: the post-freeze driver runs only with TR_QUEUE=pf (the earlier queues belong to scripts/test_r2_2026-11-13)")\n'
               '_QROOT = "post_freeze_2026-11-16"\nLLM_QUEUES = ("pf",)\n',
            'if QUEUE != "bw":\n    raise SystemExit("refused: the Learned-Bandit W2R driver runs only with TR_QUEUE=bw (the earlier queues belong to scripts/test_r2_2026-11-13, post_freeze_2026-11-16 and fill_2026-11-16)")\n'
            '_QROOT = "bandit_w2r_2026-11-17"\nLLM_QUEUES = ("bw",)\n', "queue slot and folder")
    t = sub(t, 'PLAN_NOTE = PROJECT / "notes/test_r2_post_freeze_plan.md"', 'PLAN_NOTE = PROJECT / "notes/bandit_w2r_postfreeze_plan.md"', "plan note")
    t = line_sub(t, "QUEUE_CAPS = ", 'QUEUE_CAPS = {"bw": {"new_real_runs": 20, "job_hours": 8, "tokens": 1000, "usd": 0.01}}      # decision of Tue 17 Nov: 20 new tool runs, 8 job-hours; no model call (tokens and US$ are placeholders that are never used)', "caps")
    t = sub(t, '        allowed, what = TEST + TRAINING + DEVELOPMENT, "TEST, TRAINING or VALIDATION (post-freeze rows)"\n',
            '        allowed, what = TEST, "TEST (Learned Bandit W2R: no TRAINING or VALIDATION design is opened)"\n', "guard_design: TEST only")
    t = sub(t, '+ [BASE / f"sealed/{q}/telemetry_cache" for q in ("llm1", "llm3", "llm4", "llm5", "llm6", "llm7")])',
            '+ [BASE / f"sealed/{q}/telemetry_cache" for q in ("llm1", "llm3", "llm4", "llm5", "llm6", "llm7")]\n                       + [BASE / "post_freeze_2026-11-16/telemetry_cache"])', "post-freeze telemetry cache as a read-only seed")
    return t


def policies(t: str) -> str:
    return sub(t, '        from orch.policies.bandit_frozen import bandit\n\n        return bandit(task["design"], brief, 0)\n',
               '        from orch.policies.bandit_w2r_postfreeze import bandit_w2r      # post-freeze: the frozen policy class with the W2R prior fitted on 17 Nov\n\n'
               '        return bandit_w2r(task["design"], brief, 0)\n', "bandit row = the W2R post-freeze entry point")


def main() -> int:
    DST.mkdir(exist_ok=True)
    out = {"tr_common.py": common((SRC / "tr_common.py").read_text(encoding="utf-8")),
           "tr_policies.py": policies((SRC / "tr_policies.py").read_text(encoding="utf-8")),
           "tr_episode.py": (SRC / "tr_episode.py").read_text(encoding="utf-8")}
    for name, body in out.items():
        target = DST / name
        if target.exists():
            print("exists, left as it is:", name)
            continue
        target.write_text(body, encoding="utf-8")
        print("wrote", name, hashlib.sha256(body.encode("utf-8")).hexdigest()[:12])
    print("changes:", "; ".join(LOG))
    print("originals read (sha256 first 12):", {n: hashlib.sha256((SRC / n).read_bytes()).hexdigest()[:12] for n in ("tr_common.py", "tr_policies.py", "tr_episode.py")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
