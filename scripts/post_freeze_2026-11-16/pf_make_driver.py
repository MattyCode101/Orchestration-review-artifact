"""Post-freeze driver (Mon 16 Nov 2026): makes scripts/post_freeze_2026-11-16/ from COPIES of the pinned driver files of
scripts/test_r2_2026-11-13/. The originals are only read. Every change is a substitution that must match exactly once (or a slice between two
markers), so a change in an original refuses the build instead of silently producing a different driver.

Files copied with changes: tr_common.py (queue slot 'pf' only; TEST + TRAINING + VALIDATION designs; caps 80 runs / 40 job-hours / US$8; stall
override 15 minutes; 24 tool jobs; Qwen token ledger; Qwen bridge copy; manifests and prices in the post-freeze folder), tr_queue.py (label,
12:15 UK admission cut-off, 15-minute wording), tr_ctl.py (label), tr_plan.py (kind 'post': rows a to d), tr_episode.py and tr_policies.py
(unchanged copies), call_gateway_qwen.py (tools/llm_bridge/call_gateway.py plus enable_thinking false for Qwen models).
usage: python pf_make_driver.py   (run from the project root with the project venv)
"""
from __future__ import annotations
import os

import hashlib
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SRC = PROJECT / "scripts/test_r2_2026-11-13"
DST = PROJECT / "scripts/post_freeze_2026-11-16"
BRIDGE = PROJECT / "tools/llm_bridge/call_gateway.py"
LOG: list[str] = []


def sub(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"refused: {what}: expected exactly one match, found {text.count(old)}")
    LOG.append(what)
    return text.replace(old, new)


def between(text: str, start: str, end_line_prefix: str, new: str, what: str, *, after: str | None = None) -> str:
    """Replace text from `start` to the end of the first line (at or after `start`) that begins with `end_line_prefix`."""

    if text.count(start) != 1:
        raise SystemExit(f"refused: {what}: start marker found {text.count(start)} times")
    i = text.index(start)
    j = text.index(end_line_prefix, text.index(after, i) if after else i)
    k = text.index("\n", j) + 1
    LOG.append(what)
    return text[:i] + new + text[k:]


def common(t: str) -> str:
    t = sub(t, '"""Test-run driver (Fri 13 Nov 2026): shared constants, scopes, noise, load rule, model and policy plug-ins.',
            '"""POST-FREEZE COPY (Mon 16 Nov 2026) of scripts/test_r2_2026-11-13/tr_common.py: queue slot "pf" only, changes listed in pf_make_driver.py.\n'
            'Test-run driver (Fri 13 Nov 2026): shared constants, scopes, noise, load rule, model and policy plug-ins.', "tr_common docstring")
    t = between(t, 'QUEUE = os.environ.get("TR_QUEUE", "")', 'ROOT = BASE / (',
                'QUEUE = os.environ.get("TR_QUEUE", "")\n'
                'if QUEUE != "pf":\n'
                '    raise SystemExit("refused: the post-freeze driver runs only with TR_QUEUE=pf (the earlier queues belong to scripts/test_r2_2026-11-13)")\n'
                '_QROOT = "post_freeze_2026-11-16"\n'
                'LLM_QUEUES = ("pf",)\n'
                'BASE = PROJECT / "data/test_r2_2026-11-13"\n'
                'ROOT = BASE / (f"{_QROOT}/dry_validation" if DRY else f"{_QROOT}/smoke" if SMOKE else _QROOT)\n',
                "queue slot, folders", after='ROOT = BASE / (')
    t = sub(t, 'MANIFEST_DEV = PROJECT / "data/d6v2_2026-11-12/manifest_dev"',
            'MANIFEST_DEV = PROJECT / "data/d6v2_2026-11-12/manifest_dev" if DRY else BASE / _QROOT / "manifest_dev"', "dev manifest")
    t = sub(t, 'MANIFEST_TRAIN = PROJECT / "data/d6v2_2026-11-12/manifest_train"', 'MANIFEST_TRAIN = BASE / _QROOT / "manifest_train"', "train manifest")
    t = sub(t, 'PRICES = BASE / "prices.json"', 'PRICES = BASE / _QROOT / "prices.json"', "prices file")
    t = between(t, 'PLAN_NOTE = PROJECT / {', '                       "llm6": "notes/test_r2_queue6_plan.md"}.get(QUEUE, "notes/test_r2_driver_plan.md")',
                'PLAN_NOTE = PROJECT / "notes/test_r2_post_freeze_plan.md"\n', "plan note")
    t = between(t, 'QUEUE_CAPS = {"": {', 'CAPS = (',
                'QUEUE_CAPS = {"pf": {"new_real_runs": 80, "job_hours": 40, "tokens": 20_000_000, "usd": 8.0}}      # decision of Mon 16 Nov: 80 runs, 40 job-hours, US$8 for Gemini and ChatGPT together\n'
                'QWEN_CAPS = {"new_real_runs": 1_000_000, "job_hours": 1_000_000, "tokens": 1_500_000, "usd": 1_000_000.0}      # Qwen\'s own token ledger: only its 1.5 million tokens bind\n'
                'QWEN_LEDGER_ROOT = BASE / _QROOT / "qwen_tokens"\n'
                'CAPS = ({"new_real_runs": 0, "job_hours": 1, "tokens": 2_000_000, "usd": 0.5} if REPLAY else QUEUE_CAPS[QUEUE])\n', "caps")
    t = sub(t, "MAX_TOOL_JOBS = 30", "MAX_TOOL_JOBS = 24", "24 tool jobs")
    t = sub(t, "STALL_OVERRIDE_MIN = 30", "STALL_OVERRIDE_MIN = 15", "15-minute stall override")
    i = t.index("_FROM_QUEUES = {")
    j = t.index("\n)\n", t.index("READ_ONLY_TELEMETRY = (", i)) + 3
    t = (t[:i] + 'READ_ONLY_TELEMETRY = (_TEST_TELEMETRY + _DEV_TELEMETRY + [BASE / "sealed/telemetry_cache", BASE / "power_review/telemetry_cache"]\n'
         '                       + [BASE / f"sealed/{q}/telemetry_cache" for q in ("llm1", "llm3", "llm4", "llm5", "llm6", "llm7")])\n' + t[j:])
    LOG.append("read-only telemetry seeds")
    t = sub(t, '    if DRY:\n        allowed, what = DEVELOPMENT, "VALIDATION (dry run)"\n    elif QUEUE == "power":\n'
               '        allowed, what = TRAINING + DEVELOPMENT, "TRAINING or VALIDATION (power review)"\n    else:\n        allowed, what = TEST, "TEST"\n',
            '    if DRY:\n        allowed, what = DEVELOPMENT, "VALIDATION (smoke test)"\n    else:\n'
            '        allowed, what = TEST + TRAINING + DEVELOPMENT, "TEST, TRAINING or VALIDATION (post-freeze rows)"\n', "guard_design")
    t = sub(t, 'path = NOISE_TEST if design in TEST else NOISE_TRAINPOWER if QUEUE == "power" else NOISE_BASE',
            'path = NOISE_TEST if design in TEST else NOISE_TRAINPOWER', "noise file of TRAINING and VALIDATION designs")
    t = sub(t, '        if QUEUE != "power":\n            raise SystemExit("refused: the TRAINING scope is opened only by the power-review queue")',
            '        if QUEUE != "pf":\n            raise SystemExit("refused: the TRAINING scope is opened only by the post-freeze queue")', "train scope")
    t = sub(t, 'record = self._v3(state_id) if QUEUE == "power" else None', 'record = self._v3(state_id)', "pinned v3 extraction for any state that has one")
    t = sub(t, '        return {"name": model[4:], "script": PROJECT / "tools/llm_bridge/call_gateway.py", "price": prices[model], "label": model}',
            '        script = Path(__file__).resolve().parent / "call_gateway_qwen.py" if model.startswith("gateway:Qwen/") else PROJECT / "tools/llm_bridge/call_gateway.py"\n'
            '        return {"name": model[4:], "script": script, "price": prices[model], "label": model}', "Qwen bridge copy")
    t = sub(t, 'def make_llm(task: dict, caps):\n',
            'def llm_caps(task: dict, caps):\n'
            '    """Qwen calls are counted in their own token ledger (1.5 million tokens, smoke test included); every other model in the shared ledger."""\n\n'
            '    import campaign_harness as h\n\n'
            '    if str(task.get("model") or "").startswith("gateway:Qwen/"):\n'
            '        return h.CapLedger(QWEN_LEDGER_ROOT, QWEN_CAPS, plan_sha())\n'
            '    return caps\n\n\n'
            'def make_llm(task: dict, caps):\n', "Qwen token ledger helper")
    t = sub(t, 'return h.LedgeredLLM(CachedLLM(transport, cache, params), caps, LLM_CALLS, task["episode_id"], spec["price"])',
            'return h.LedgeredLLM(CachedLLM(transport, cache, params), llm_caps(task, caps), LLM_CALLS, task["episode_id"], spec["price"])', "ledger of the LLM")
    return t


def queue(t: str) -> str:
    t = sub(t, "from datetime import datetime\n", "from datetime import datetime, timezone\n", "tr_queue import")
    t = sub(t, "TERMINAL = (", 'CUTOFF_UTC = datetime(2026, 11, 16, 12, 15, tzinfo=timezone.utc)      # admission closes 12:15 UK (= UTC in November), decision of Mon 16 Nov\nTERMINAL = (', "admission cut-off constant")
    t = between(t, 'LABEL = {"": "baselines"', '         "llm7": "ChatGPT block, queue 5"}[tc.QUEUE]', 'LABEL = "post-freeze rows"\n', "queue label")
    t = sub(t, "                if tc.STOPFILE.exists():\n                    self.stop_requested = True\n",
            "                if tc.STOPFILE.exists():\n                    self.stop_requested = True\n"
            "                if datetime.now(timezone.utc) >= CUTOFF_UTC:\n                    self.stop_requested = True      # no new episode after 12:15 UK; those in flight finish\n", "cut-off in the loop")
    t = sub(t, "admission stalled for more than 30 minutes, running up to 8 jobs regardless", "admission stalled for more than 15 minutes, running up to 8 jobs regardless", "stall wording")
    t = sub(t, "Counts and completion only (results are sealed). Status: .venv/bin/python scripts/test_r2_2026-11-13/tr_ctl.py status",
            "Counts and completion only. Status: TR_QUEUE=pf .venv/bin/python scripts/post_freeze_2026-11-16/tr_ctl.py status", "status hint")
    return t


def ctl(t: str) -> str:
    return sub(t, "{'DRY RUN' if tc.DRY else 'SEALED'}", "{'DRY RUN' if tc.DRY else 'POST-FREEZE'}", "tr_ctl label")


PLAN_POST = '''

QWEN = "gateway:Qwen/Qwen3.5-397B-A17B-FP8"
CHATGPT = "gateway:gpt-5.6-terra"


def post_rows(groups: list[str]) -> list[dict]:
    """Rows of the post-freeze plan (notes/test_r2_post_freeze_plan.md), interleaved over the requested groups so that they run in parallel:
    a = V9 template with Qwen on the 5 TEST designs x 7 briefs (35); b = V3, V6, V11 (Gemini) on the TEST power briefs (45); c = V9 (Gemini) on the
    power briefs of the 5 TRAINING and 3 VALIDATION designs (24); d (tier 2) = V9 with gpt-5.6-terra, repeats r2 and r3, TEST, 7 briefs (70)."""

    rows: dict[str, list[dict]] = {"a": [], "b": [], "c": [], "d": []}
    for design in DESIGN_ORDER:
        for brief in tc.BRIEFS:
            rows["a"].append(make(design, brief, "V9", "r1", QWEN, tier=1))
    for design in DESIGN_ORDER:
        for brief in tc.THREE_WORKER:
            for row in ("V3", "V6", "V11"):
                rows["b"].append(make(design, brief, row, "r1", None, tier=1))
    for design in POWER_DESIGNS:
        for brief in tc.THREE_WORKER:
            rows["c"].append(make(design, brief, "V9", "r1", None, tier=1))
    for design in DESIGN_ORDER:
        for brief in tc.BRIEFS:
            for rep in ("r2", "r3"):
                rows["d"].append(make(design, brief, "V9", rep, CHATGPT, tier=2))
    picked = [rows[g] for g in groups]
    out: list[dict] = []
    i = 0
    while any(i < len(r) for r in picked):
        for r in picked:
            if i < len(r):
                out.append(r[i])
        i += 1
    return out
'''


def plan(t: str) -> str:
    t = sub(t, 'ap.add_argument("kind", choices=("baselines", "llm", "dry", "power"))',
            'ap.add_argument("kind", choices=("baselines", "llm", "dry", "power", "post"))\n    ap.add_argument("--groups", nargs="*", default=["a", "b", "c"], help="kind post: row groups a b c d")', "plan kinds")
    t = sub(t, "    else:\n        new.append(make(args.design, args.brief, args.row, args.repeats[0], args.model, args.budget, args.tier))",
            '    elif args.kind == "post":\n        new = post_rows(args.groups)\n    else:\n        new.append(make(args.design, args.brief, args.row, args.repeats[0], args.model, args.budget, args.tier))', "post branch")
    t = sub(t, 'new_sorted = sorted(add, key=lambda t: (t["tier"], t["brief"] in tc.THREE_WORKER, t["design"]))',
            'new_sorted = sorted(add, key=(lambda t: t["tier"]) if args.kind == "post" else (lambda t: (t["tier"], t["brief"] in tc.THREE_WORKER, t["design"])))', "order of the post rows")
    t = sub(t, "\n\ndef main() -> int:\n", PLAN_POST + "\n\ndef main() -> int:\n", "post_rows")
    return t


def bridge(t: str) -> str:
    t = sub(t, '"""One-shot JSON caller for the gateway', '"""POST-FREEZE COPY (Mon 16 Nov 2026) of tools/llm_bridge/call_gateway.py: a request for a Qwen model also sends chat_template_kwargs\n'
            'enable_thinking false (Qwen3.5 thinks by default: a 60-token prompt used all 800 completion tokens on reasoning and returned no text). Nothing else changes.\n\n'
            'One-shot JSON caller for the gateway', "bridge docstring")
    t = sub(t, '    if req.get("seed") is not None:\n        payload["seed"] = req["seed"]\n',
            '    if req.get("seed") is not None:\n        payload["seed"] = req["seed"]\n'
            '    if "qwen" in str(req["model"]).lower():\n        payload["chat_template_kwargs"] = {"enable_thinking": False}\n', "thinking off for Qwen")
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
        "call_gateway_qwen.py": bridge(BRIDGE.read_text(encoding="utf-8")),
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
                                                for n in ("tr_common.py", "tr_queue.py", "tr_ctl.py", "tr_plan.py", "tr_episode.py", "tr_policies.py")})
    shutil.copymode(SRC / "tr_common.py", DST / "tr_common.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
