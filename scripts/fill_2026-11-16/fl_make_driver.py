"""Fill-run driver (Mon 16 Nov 2026): makes scripts/fill_2026-11-16/ from COPIES of scripts/post_freeze_2026-11-16/ (itself a patched copy of the pinned
driver scripts/test_r2_2026-11-13/). The originals are only read. Every change is a substitution that must match exactly once, so a change in an original
refuses the build instead of silently producing a different driver.

Changes against the post-freeze copy:
  tr_common.py  queue slot 'fl' only; run folder data/test_r2_2026-11-13/fill_2026-11-16/; plan note notes/test_r2_fill_plan.md; caps 350 new tool runs, 90
                job-hours, US$5 (decision of Mon 16 Nov, X4 tier 1); only TRAINING and VALIDATION designs may be opened (no TEST design, no t2_or1200);
                the post-freeze telemetry cache is a read-only seed.
  tr_queue.py   label; admission closes 20:00 UK (= UTC in November); status hint.
  tr_ctl.py     label.
  tr_plan.py    kind 'fill': GA-R1 (direct LLM tool control) on the TRAINING and VALIDATION designs, V9 repeats 2 and 3 on the power briefs.
  tr_episode.py, tr_policies.py, call_gateway_qwen.py: unchanged copies.
usage: python fl_make_driver.py   (run from the project root with the project venv; refuses to overwrite)
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SRC = PROJECT / "scripts/post_freeze_2026-11-16"
DST = PROJECT / "scripts/fill_2026-11-16"
LOG: list[str] = []


def sub(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"refused: {what}: expected exactly one match, found {text.count(old)}")
    LOG.append(what)
    return text.replace(old, new)


def common(t: str) -> str:
    t = sub(t, '"""POST-FREEZE COPY (Mon 16 Nov 2026) of scripts/test_r2_2026-11-13/tr_common.py: queue slot "pf" only, changes listed in pf_make_driver.py.',
            '"""FILL-RUN COPY (Mon 16 Nov 2026) of scripts/post_freeze_2026-11-16/tr_common.py (a copy of scripts/test_r2_2026-11-13/tr_common.py): queue slot "fl" only,\n'
            'changes listed in fl_make_driver.py (and, for the earlier copy, pf_make_driver.py).', "docstring")
    t = sub(t, 'if QUEUE != "pf":\n    raise SystemExit("refused: the post-freeze driver runs only with TR_QUEUE=pf (the earlier queues belong to scripts/test_r2_2026-11-13)")\n'
               '_QROOT = "post_freeze_2026-11-16"\nLLM_QUEUES = ("pf",)\n',
            'if QUEUE != "fl":\n    raise SystemExit("refused: the fill-run driver runs only with TR_QUEUE=fl (the earlier queues belong to scripts/test_r2_2026-11-13 and scripts/post_freeze_2026-11-16)")\n'
            '_QROOT = "fill_2026-11-16"\nLLM_QUEUES = ("fl",)\n', "queue slot and folder")
    t = sub(t, 'PLAN_NOTE = PROJECT / "notes/test_r2_post_freeze_plan.md"', 'PLAN_NOTE = PROJECT / "notes/test_r2_fill_plan.md"', "plan note")
    t = sub(t, 'QUEUE_CAPS = {"pf": {"new_real_runs": 80, "job_hours": 40, "tokens": 20_000_000, "usd": 8.0}}      # decision of Mon 16 Nov: 80 runs, 40 job-hours, US$8 for Gemini and ChatGPT together',
            'QUEUE_CAPS = {"fl": {"new_real_runs": 350, "job_hours": 90, "tokens": 40_000_000, "usd": 5.0}}      # decision of Mon 16 Nov (X4 tier 1): 350 new tool runs, 90 job-hours, US$5', "caps")
    t = sub(t, '        allowed, what = TEST + TRAINING + DEVELOPMENT, "TEST, TRAINING or VALIDATION (post-freeze rows)"\n',
            '        allowed, what = TRAINING + DEVELOPMENT, "TRAINING or VALIDATION (fill runs: no TEST design is opened)"\n', "guard_design: no TEST design")
    t = sub(t, '        if QUEUE != "pf":\n            raise SystemExit("refused: the TRAINING scope is opened only by the post-freeze queue")',
            '        if QUEUE != "fl":\n            raise SystemExit("refused: the TRAINING scope is opened only by the fill-run queue")', "train scope")
    t = sub(t, 'BASE / "power_review/telemetry_cache"]\n', 'BASE / "power_review/telemetry_cache", BASE / "post_freeze_2026-11-16/telemetry_cache"]\n', "post-freeze telemetry cache as a read-only seed")
    return t


def queue(t: str) -> str:
    t = sub(t, 'LABEL = "post-freeze rows"', 'LABEL = "fill runs"', "queue label")
    t = sub(t, "CUTOFF_UTC = datetime(2026, 11, 16, 12, 15, tzinfo=timezone.utc)      # admission closes 12:15 UK (= UTC in November), decision of Mon 16 Nov",
            "CUTOFF_UTC = datetime(2026, 11, 16, 20, 0, tzinfo=timezone.utc)      # admission closes 20:00 UK (= UTC in November), decision of Mon 16 Nov (X4 tier 1)", "cut-off")
    t = sub(t, "# no new episode after 12:15 UK; those in flight finish", "# no new episode after 20:00 UK; those in flight finish", "cut-off comment")
    t = sub(t, "TR_QUEUE=pf .venv/bin/python scripts/post_freeze_2026-11-16/tr_ctl.py status", "TR_QUEUE=fl .venv/bin/python scripts/fill_2026-11-16/tr_ctl.py status", "status hint")
    return t


def ctl(t: str) -> str:
    return sub(t, "'POST-FREEZE'", "'FILL RUNS'", "tr_ctl label")


FILL_ROWS = '''

def fill_rows() -> list[dict]:
    """Rows of the fill plan (notes/test_r2_fill_plan.md, Mon 16 Nov 2026), TRAINING and VALIDATION designs only, Gemini flash-lite at temperature 0.3:
    b = GA-R1 (direct LLM tool control, frozen template) on T0, A0, W2R, L2R and P0, repeats r1 to r3 (40 per repeat);
    c = V9 repeats r2 and r3 on W2R, L2R and P0 (24 per repeat; repeat 1 is the post-freeze run of this morning).
    Order = the queue order: every repeat-1 episode first, then every repeat 2, then every repeat 3, so that a partial finish still covers every cell."""

    out: list[dict] = []
    designs = tuple(tc.DEVELOPMENT) + tuple(tc.TRAINING)
    for tag, tier in (("r1", 1), ("r2", 2), ("r3", 3)):
        for brief in ("T0", "A0") + tc.THREE_WORKER:
            for design in designs:
                out.append(make(design, brief, "GA-R1", tag, None, tier=tier))
        if tag != "r1":
            for brief in tc.THREE_WORKER:
                for design in designs:
                    out.append(make(design, brief, "V9", tag, None, tier=tier))
    return out
'''


def plan(t: str) -> str:
    t = sub(t, 'choices=("baselines", "llm", "dry", "power", "post"))', 'choices=("baselines", "llm", "dry", "power", "post", "fill"))', "plan kinds")
    t = sub(t, '    elif args.kind == "post":\n        new = post_rows(args.groups)\n',
            '    elif args.kind == "post":\n        raise SystemExit("refused: the post-freeze rows belong to scripts/post_freeze_2026-11-16")\n'
            '    elif args.kind == "fill":\n        new = fill_rows()\n', "fill branch")
    t = sub(t, '(lambda t: t["tier"]) if args.kind == "post" else', '(lambda t: t["tier"]) if args.kind == "fill" else', "order of the fill rows (stable by tier)")
    t = sub(t, "\n\ndef main() -> int:\n", FILL_ROWS + "\n\ndef main() -> int:\n", "fill_rows")
    return t


def main() -> int:
    DST.mkdir(exist_ok=True)
    out = {
        "tr_common.py": common((SRC / "tr_common.py").read_text(encoding="utf-8")),
        "tr_queue.py": queue((SRC / "tr_queue.py").read_text(encoding="utf-8")),
        "tr_ctl.py": ctl((SRC / "tr_ctl.py").read_text(encoding="utf-8")),
        "tr_plan.py": plan((SRC / "tr_plan.py").read_text(encoding="utf-8")),
        "tr_episode.py": (SRC / "tr_episode.py").read_text(encoding="utf-8"),
        "tr_policies.py": (SRC / "tr_policies.py").read_text(encoding="utf-8"),
        "call_gateway_qwen.py": (SRC / "call_gateway_qwen.py").read_text(encoding="utf-8"),
    }
    for name, body in out.items():
        target = DST / name
        if target.exists():
            print("exists, left as it is:", name)
            continue
        target.write_text(body, encoding="utf-8")
        print("wrote", name, hashlib.sha256(body.encode("utf-8")).hexdigest()[:12])
    print("changes:", "; ".join(LOG))
    print("originals read (sha256 first 12):", {n: hashlib.sha256((SRC / n).read_bytes()).hexdigest()[:12]
                                                for n in ("tr_common.py", "tr_queue.py", "tr_ctl.py", "tr_plan.py", "tr_episode.py", "tr_policies.py", "call_gateway_qwen.py")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
