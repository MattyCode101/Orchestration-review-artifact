"""Offline evaluator on the decision bank: render a point under a template variant, call the model, parse, score.

No tool job. Model calls go through the project's bridge (gemini-3.1-flash-lite) behind a replay cache (record
mode: a rerun of the same prompt, parameters and schema costs nothing) and a spend ledger with a hard stop.

Per reply it records: parse/validity, the chosen action and mandate, the hindsight value of that mandate and the
regret in noise bands (best value minus chosen value; an unusable reply counts as the fallback STOP, as in the
policy), agreement with the hindsight-best value, placement kept, physical facts cited and whether each cited
number matches the physical description of a saved state in the prompt, the forecast against the stored outcome
(P4), tokens. The quiz is a separate call per point (same prompt up to "Reply format", then three questions).

    PYTHONPATH=.:scripts .venv/bin/python -B scripts/template_v2_2026-11-12/t2_eval.py --tag main \
        --variants R1,P1 --samples 3 --temperature 0.3 [--points b000,b001] [--quiz] [--threads 16]
"""
from __future__ import annotations
import os

import argparse
import hashlib
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT = Path(os.environ.get("ORCH_PROJECT_ROOT") or Path(__file__).resolve().parents[2])
for extra in (PROJECT, PROJECT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from orch.contract.models import MetricName  # noqa: E402
from orch.contract.views import PolicyView  # noqa: E402
from orch.policies.llm_client import (  # noqa: E402
    BridgeTransport, CachedLLM, LLMCallError, LLMParams, LLMUnavailableError, RateLimiter, ReplayCache,
)
from orch.policies.llm_ledger import LedgeredTransport, Price, SpendCapReached, SpendLedger  # noqa: E402
from orch.policies.llm_view import METRIC_NAMES  # noqa: E402
from orch.policies.r2_experience import build_table, load_calls  # noqa: E402
from orch.policies.r2_physical import facts_from_record  # noqa: E402
from orch.policies.v2_experience import load_calls_v2  # noqa: E402
from orch.policies.v2_prompt import Situation, Variant, load_assets, render  # noqa: E402
from orch.policies.v2_schemas import ReplyError, parse_reply, reply_text_fields  # noqa: E402
from orch.policies.v2_vocab import direction_of  # noqa: E402

WORK = PROJECT / "data/template_v2_2026-11-12"
BANK = WORK / "bank/bank.jsonl"
MODEL = "gemini-3.1-flash-lite"
PRICE = Price(0.25, 1.50)          # US$ per million tokens (input, output); matches the recorded spend lines
CAP_USD = 20.0
HEADROOM_USD = 0.6

VARIANTS: dict[str, Variant] = {
    "R1": Variant("R1"),
    "P1": Variant("P1", ("P1",)),
    "P12": Variant("P12", ("P1", "P2")),
    "P123": Variant("P123", ("P1", "P2", "P3")),
    "P1234": Variant("P1234", ("P1", "P2", "P3", "P4")),
    "noP1": Variant("noP1", ("P2", "P3", "P4")),
    "noP2": Variant("noP2", ("P1", "P3", "P4")),
    "noP3": Variant("noP3", ("P1", "P2", "P4")),
    "W2": Variant("W2", ("P1", "P2", "P3", "P4"), wording="W2"),
    "W3": Variant("W3", ("P1", "P2", "P3", "P4"), wording="W3"),
    "exact": Variant("exact", ("P1", "P2", "P3", "P4"), exact_only=True),
    "P1234r": Variant("P1234r", ("P1", "P2", "P3", "P4r")),       # P4 revision 1 (the stop rule's one revision)
}

# physical-description fields a reply may cite: (field, regex)
CITE = [
    ("failing_endpoints", r"failing[ -]end ?points?"),
    ("tns_ns", r"\bTNS\b|total negative slack"),
    ("logic_levels", r"logic[ -]levels?"),
    ("path_delay_ns", r"(?:worst[- ]path |data[- ]path |critical[- ]path )delay|path delay"),
    ("route_share_pct", r"routing share|route share"),
    ("clb_regs", r"CLB registers|register count"),
    ("srl_luts", r"shift[- ]register LUTs?|\bSRL"),
    ("bram_tiles", r"\bBRAM"),
    ("dsps", r"\bDSPs?\b"),
    ("max_fanout", r"fan-?out"),
    ("tile_span", r"tile span|span in tiles"),
]
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def load_bank(path: Path = BANK) -> dict[str, dict[str, Any]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)
            out[p["point_id"]] = p
    return out


class Bank:
    def __init__(self, path: Path = BANK) -> None:
        self.points = load_bank(path)
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assets = load_assets(PROJECT)
        self.calls_ab, self.calls_ab_sha = load_calls(PROJECT / "configs/experience/calls_ab.json")
        self.calls_v2, self.calls_v2_sha = load_calls_v2(PROJECT / "configs/experience/calls_v2.json")
        self._tables: dict[str, Any] = {}

    def situation(self, point: dict[str, Any]) -> Situation:
        design = point["design"]
        if design not in self._tables:
            self._tables[design] = build_table(self.calls_ab, self.calls_ab_sha, exclude_design=design)
        view = PolicyView.model_validate(point["view"])
        records = point["describe"]
        return Situation(view=view, design=design, describe=lambda sid: records[sid],
                         r2_experience=self._tables[design], calls_v2=self.calls_v2)


# ------------------------------------------------------------------ scoring
def physical_values(point: dict[str, Any]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for sid, record in point["describe"].items():
        for name, value in facts_from_record(record).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.setdefault(name, []).append(float(value))
    return values


def _matches(field: str, number: float, known: list[float]) -> bool:
    tol = {"tns_ns": 0.051, "path_delay_ns": 0.0051, "route_share_pct": 1.01}.get(field, 0.51)
    return any(abs(number - v) <= tol or abs(abs(number) - abs(v)) <= tol for v in known)


NUMBER_SEP = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")


def _numbers(segment: str) -> list[tuple[int, float]]:
    return [(m.start(), float(m.group(0).replace(",", ""))) for m in NUMBER_SEP.finditer(segment)]


def citations(text: str, point: dict[str, Any]) -> list[dict[str, Any]]:
    """Physical-field mentions; the number checked is the one nearest the field name, before or after it
    (thousands separators allowed). `correct`: it equals the field's value for some saved state (the declared
    rule); `correct_or_difference` also accepts a difference between two saved states' values (post hoc)."""

    known = physical_values(point)
    found = []
    for field, pattern in CITE:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            before = text[max(0, m.start() - 25): m.start()]
            after = text[m.end(): m.end() + 40]
            cands = [(len(before) - pos, val) for pos, val in _numbers(before)]
            cands += [(pos, val) for pos, val in _numbers(after)]
            entry: dict[str, Any] = {"field": field, "text": text[max(0, m.start() - 25): m.end() + 40]}
            if cands and field in known:
                value = min(cands, key=lambda c: c[0])[1]
                values = known[field]
                diffs = [abs(a - b) for a in values for b in values if a != b]
                entry.update(number=value, correct=_matches(field, value, values),
                             correct_or_difference=_matches(field, value, values) or _matches(field, abs(value), diffs))
            found.append(entry)
    return found


def forecast_score(parsed: dict[str, Any], row: dict[str, Any], variant: Variant, limit_names: list[str],
                   goal_name: str) -> dict[str, Any] | None:
    forecast = parsed.get("forecast")
    if not variant.p4 or parsed["action"] != "INVOKE_WORKER" or not isinstance(forecast, dict):
        return None
    truth = row["forecast_truth"]["respect_true" if parsed["mandate"]["respect"] else "respect_false"]
    if truth["outcome"] in ("not_stored", "no_options"):
        return {"scored": False, "reason": truth["outcome"]}
    out: dict[str, Any] = {"scored": True, "items": {}}
    for name in (goal_name, *limit_names):
        key = "goal" if name == goal_name else name
        said = forecast.get(key)
        if truth["outcome"] == "nothing_selectable":
            want, want_dir = "nothing selectable", "nothing selectable"
        else:
            change = truth["changes"].get(name, {})
            if "v2" not in change:
                continue
            want_dir = {"gain": "gain", "loss": "loss", "none": "no change"}[change["direction"]]
            want = want_dir if variant.exact_only else (change["v2"] if variant.p1 else change["r1"])
        said_dir = None if said is None else (said if said in ("gain", "loss", "no change", "nothing selectable")
                                              else {"none": "no change"}.get(direction_of(said), direction_of(said)))
        out["items"][key] = {"said": said, "want": want, "exact": said == want, "direction": said_dir == want_dir}
    return out


def score(point: dict[str, Any], variant: Variant, parsed: dict[str, Any] | None, error: str | None,
          aliases: dict[str, str]) -> dict[str, Any]:
    h = point["hindsight"]
    band = point["goal_band"]
    view = point["view"]
    goal_name = METRIC_NAMES[MetricName(view["brief"]["objective"]["metric"])]
    limit_names = [METRIC_NAMES[MetricName(t["metric"])] for t in view["brief"]["terms"]]
    out: dict[str, Any] = {"parsed": parsed is not None, "error": error}
    value, exact, row = h["stop_value"], True, None
    if parsed is not None and parsed["action"] == "INVOKE_WORKER":
        m = parsed["mandate"]
        if m["attempts"] > point["budget_left"]:
            out["error"], parsed = f"attempts {m['attempts']} exceed the remaining budget", None
        else:
            row = next((r for r in h["mandates"] if r["worker"] == m["worker"] and r["base"] == m["base"]
                        and r["preserve"] == m["preserve"] and r["attempts"] == m["attempts"]), None)
            if row is None:
                out["error"], parsed = "the mandate is not a legal one at this point", None
            else:
                value, exact = row["value"], row["exact"]
                out.update(mandate=m, tried_combination=row["tried_combination"], status=row["status"])
    out["valid"] = parsed is not None
    out["action"] = None if parsed is None else parsed["action"]
    out["fallback_stop"] = parsed is None
    out["value"], out["value_exact"] = value, exact
    out["best_value"], out["best_exact"] = h["best_value"], h["best_is_exact"]
    out["regret_bands"] = None if value is None or not band else (h["best_value"] - value) / band
    out["headroom_bands"] = None if not band else (h["best_value"] - h["stop_value"]) / band
    out["optimal"] = None if value is None else abs(h["best_value"] - value) <= 1e-9 * max(1.0, abs(h["best_value"]))
    out["keep"] = None if parsed is None or parsed["action"] != "INVOKE_WORKER" else parsed["mandate"]["preserve"]
    if row is not None:
        same = [r for r in h["mandates"] if r["worker"] == row["worker"] and r["base"] == row["base"]
                and r["value"] is not None]
        keep_best = max((r["value"] for r in same if r["preserve"]), default=None)
        free_best = max((r["value"] for r in same if not r["preserve"]), default=None)
        if keep_best is not None and free_best is not None and band and abs(keep_best - free_best) >= band:
            out["placement_better"] = "keep" if keep_best > free_best else "free"
    if parsed is not None:
        text = reply_text_fields(parsed)
        cites = citations(text, point)
        out["citations"] = cites
        fs = forecast_score(parsed, row, variant, limit_names, goal_name) if row is not None else None
        out["forecast"] = fs
        if variant.p4:
            out["evidence_items"] = len(parsed.get("evidence") or [])
            out["options_items"] = len(parsed.get("options") or [])
    return out


# ------------------------------------------------------------------ quiz
AMOUNTS = ["under 1 band", "1 to 3 bands", "over 3 bands"]
QUIZ_TEXT = (
    "Comprehension check. Answer only these three questions about the situation above; do not choose a mandate.\n"
    "- best_state: which saved state is currently the best on the goal among the states that meet every limit of "
    "the brief? Give its alias.\n"
    "- broken_limits: list every limit that a saved state breaks, with the state's alias and by how much "
    "(under 1 noise band, 1 to 3 bands, or over 3 bands). Leave out states that break no limit.\n"
    "- limit_owners: for each limit of the brief, which worker's own objective is that limit's metric?\n"
    "Reply with one JSON object in the structured format you are given.\n"
)


def quiz_prompt_and_schema(prompt: str, aliases: list[str], limits: list[str], workers: list[str]):
    head = prompt.split("Reply format\n")[0].rstrip("\n") + "\n\n"
    schema = {
        "type": "object",
        "properties": {
            "best_state": {"type": "string", "enum": aliases},
            "broken_limits": {"type": "array", "items": {"type": "object", "properties": {
                "state": {"type": "string", "enum": aliases}, "limit": {"type": "string", "enum": limits},
                "amount": {"type": "string", "enum": AMOUNTS}},
                "required": ["state", "limit", "amount"], "propertyOrdering": ["state", "limit", "amount"]}},
            "limit_owners": {"type": "array", "items": {"type": "object", "properties": {
                "limit": {"type": "string", "enum": limits}, "worker": {"type": "string", "enum": workers}},
                "required": ["limit", "worker"], "propertyOrdering": ["limit", "worker"]}},
        },
        "required": ["best_state", "broken_limits", "limit_owners"],
        "propertyOrdering": ["best_state", "broken_limits", "limit_owners"],
    }
    return head + QUIZ_TEXT, schema


def quiz_score(text: str, truth: dict[str, Any]) -> dict[str, Any]:
    try:
        data = json.loads(text.strip().strip("`").removeprefix("json"))
    except json.JSONDecodeError:
        return {"parsed": False}
    q1 = data.get("best_state") == truth["best_state"]
    said: dict[str, set] = {}
    amounts: dict[tuple[str, str], str] = {}
    for item in data.get("broken_limits") or []:
        if isinstance(item, dict):
            said.setdefault(item.get("state"), set()).add(item.get("limit"))
            amounts[(item.get("state"), item.get("limit"))] = item.get("amount")
    sets_ok = []
    amount_ok = []
    for alias, broken in truth["broken"].items():
        want = {b["metric"] for b in broken}
        sets_ok.append(said.get(alias, set()) == want)
        for b in broken:
            amount_ok.append(amounts.get((alias, b["metric"])) == b["amount"])
    owners = {i.get("limit"): i.get("worker") for i in data.get("limit_owners") or [] if isinstance(i, dict)}
    owner_ok = [owners.get(limit) == worker for limit, worker in truth["owners"].items()]
    parts = [float(q1), sum(sets_ok) / len(sets_ok), sum(owner_ok) / len(owner_ok)]
    if amount_ok:
        parts.append(sum(amount_ok) / len(amount_ok))
    return {"parsed": True, "best_state": q1, "broken_sets": sum(sets_ok) / len(sets_ok),
            "amounts": (sum(amount_ok) / len(amount_ok)) if amount_ok else None,
            "owners": sum(owner_ok) / len(owner_ok), "score": sum(parts) / len(parts)}


# ------------------------------------------------------------------ driver
def make_llm(temperature: float, seed: int, source: str, ledger: SpendLedger, limiter: RateLimiter) -> CachedLLM:
    bridge = BridgeTransport(PROJECT / "tools/llm_bridge/.venv/bin/python", PROJECT / "tools/llm_bridge/call.py",
                             rate_limiter=limiter)
    params = LLMParams(model=MODEL, temperature=temperature, max_output_tokens=2048, seed=seed,
                       thinking_budget=None, timeout_ms=60000)
    return CachedLLM(LedgeredTransport(bridge, ledger, source=source),
                     ReplayCache(WORK / "llm_cache", mode="record"), params)


def done_keys(path: Path) -> set[tuple]:
    if not path.is_file():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out.add((r["variant"], r["point"], r["sample"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--points", default="")
    ap.add_argument("--quiz", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--rate", type=int, default=400)
    args = ap.parse_args()
    bank = Bank()
    names = [v for v in args.variants.split(",") if v]
    point_ids = [p for p in args.points.split(",") if p] or sorted(bank.points)
    out_dir = WORK / "eval" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / ("quiz.jsonl" if args.quiz else "results.jsonl")
    done = done_keys(results)
    ledger = SpendLedger(WORK / "spend.jsonl", cap_usd=CAP_USD, price=PRICE, headroom_usd=HEADROOM_USD)
    limiter = RateLimiter(WORK / "ratelimit", args.rate)
    (out_dir / "run_info.json").write_text(json.dumps({
        "bank_sha256": bank.sha256, "calls_v2_sha256": bank.calls_v2_sha, "calls_ab_sha256": bank.calls_ab_sha,
        "templates": {k: t.sha256 for k, t in bank.assets.templates.items()}, "model": MODEL,
        "temperature": args.temperature, "samples": args.samples, "quiz": args.quiz,
    }, indent=1, sort_keys=True), encoding="utf-8")
    samples = [0] if args.quiz else list(range(args.samples))
    jobs = [(v, p, s) for v in names for p in point_ids for s in samples if (v, p, s) not in done]
    lock = threading.Lock()
    llms: dict[tuple, CachedLLM] = {}

    def llm_for(variant: str, seed: int) -> CachedLLM:
        key = (variant, seed)
        with lock:
            if key not in llms:
                llms[key] = make_llm(args.temperature, seed, f"t2_{args.tag}_{variant}{'_quiz' if args.quiz else ''}",
                                     ledger, limiter)
            return llms[key]

    def one(job):
        vname, pid, sample = job
        point, variant = bank.points[pid], VARIANTS[vname]
        sit = bank.situation(point)
        prompt, schema, meta = render(sit, variant, bank.assets)
        aliases = list(meta["aliases"])
        workers = [c.id for c in sit.view.worker_cards]
        limits = [METRIC_NAMES[t.metric] for t in sit.view.brief.terms]
        row: dict[str, Any] = {"variant": vname, "point": pid, "sample": sample, "prompt_sha256": meta["prompt_sha256"],
                               "prompt_chars": len(prompt), "template_sha256": meta["template_sha256"]}
        if args.quiz:
            prompt, schema = quiz_prompt_and_schema(prompt, aliases, limits, workers)
        try:
            result = llm_for(vname, sample).complete(prompt, schema)
        except SpendCapReached as exc:
            return {**row, "infra": f"spend cap: {exc}"}
        except (LLMUnavailableError, LLMCallError) as exc:
            return {**row, "infra": f"{type(exc).__name__}: {str(exc)[:200]}"}
        reply = result.reply
        row.update(prompt_tokens=reply.prompt_tokens, completion_tokens=reply.completion_tokens,
                   thoughts_tokens=reply.thoughts_tokens, total_tokens=reply.total_tokens,
                   cache_hit=result.cache_hit, finish_reason=reply.finish_reason, reply=reply.text[:6000])
        if args.quiz:
            row["quiz"] = quiz_score(reply.text, point["quiz"])
            return row
        parsed, error = None, None
        try:
            parsed = parse_reply(reply.text, variant.schema_flags(), workers=workers, aliases=aliases,
                                 limit_names=limits)
        except ReplyError as exc:
            error = str(exc)
        row.update(score(point, variant, parsed, error, meta["aliases"]))
        if parsed is not None:
            row["parsed_reply"] = parsed
        return row

    written = infra = 0
    with ThreadPoolExecutor(max_workers=args.threads) as pool, open(results, "a", encoding="utf-8") as handle:
        futures = [pool.submit(one, job) for job in jobs]
        for fut in as_completed(futures):
            row = fut.result()
            if "infra" in row:
                infra += 1
                print("INFRA", row["variant"], row["point"], row["sample"], row["infra"], flush=True)
                continue
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            written += 1
            if written % 100 == 0:
                print(f"{written}/{len(jobs)} written; spend US${ledger.total_usd():.2f}", flush=True)
    print(json.dumps({"written": written, "infra_failures": infra, "jobs": len(jobs),
                      "spend": ledger.summary()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
