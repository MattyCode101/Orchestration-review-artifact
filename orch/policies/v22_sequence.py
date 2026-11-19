"""Template v2.2, switch S: sequence experience, two-call patterns on other designs (new file).

Source: the stored call records of the five training designs, data/bandit_2026-11-11/calls/episodes_v4.jsonl (979 episode
journals of the v4 training root: repair-loop sweeps, pilots, fixed orders, random seeds, LLM runs, P0). Per episode, per pair
of consecutive completed calls (i, i+1): a state X that call i produced (physically valid) and that call i+1 used as its base.

  pattern "break"  X gained at least one noise band on the goal against the start and the engine verdict is BRIEF_VIOLATION
                   (X breaks a limit). The second call is a "repair" call when its worker's objective is the metric of the
                   first limit X breaks (brief order), otherwise "other". Outcomes, from the outputs of call i+1:
                   restored = an output with verdict ACCEPTABLE (meets every limit) and a goal gain of at least one band;
                   final = the best goal gain (in bands, against the start) among the ACCEPTABLE outputs (none when no output
                   met every limit).
  pattern "clean"  X gained at least one band on the goal and met every limit (ACCEPTABLE). Outcome: further = an ACCEPTABLE
                   output of call i+1 that beats X's goal gain by at least one band.

Bands are the design bands of the power study (data/bandit_2026-11-11/design_bands.json), because configs/noise.yaml has no
logicnets entry. Lines are pooled per brief over the designs other than the evaluated one (leave-one-design-out) and shown
when they rest on at least MIN_DESIGNS designs; the words are those of the single-call experience lines
(orch/policies/v2_experience.frequency_word). Pure aggregation and text; the table is built once by
scripts/v22_2026-11-13/v22_build_sequences.py into configs/experience/sequences_v22.json.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from orch.policies.v2_experience import frequency_word
from orch.policies.v2_vocab import CATEGORIES_V2, categorise_v2

SCHEMA = "v22-sequence-records-v1"
MIN_DESIGNS = 3
MENTION = 0.15
WORKER_METRIC = {"timing": "wns_ns", "area": "packed_luts", "power": "dynamic_power_w"}
BRIEF_IDS = ("T0", "A0", "T", "A", "P0")
NO_RESULT = "no result that met every limit"


def _gain_bands(direction: str, band: float, start_value: float, value: float) -> float:
    sign = 1.0 if direction == "max" else -1.0
    return sign * (value - start_value) / band


def build_records(episodes: Sequence[Mapping[str, Any]], bands: Mapping[str, Mapping[str, float]],
                  briefs: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Sequence records of every episode; `briefs` maps brief id -> orch.contract.models.Brief."""

    out: list[dict[str, Any]] = []
    for ep in episodes:
        brief = briefs.get(ep["brief"])
        band_row = bands.get(ep["design"])
        if brief is None or band_row is None:
            continue
        goal, direction = brief.objective.metric.value, brief.objective.direction
        band = float(band_row.get(goal) or 0.0)
        start = ep["start_metrics"]
        if band <= 0.0 or goal not in start:
            continue

        def gain(metrics: Mapping[str, float]) -> float | None:
            return None if goal not in metrics else _gain_bands(direction, band, start[goal], metrics[goal])

        def first_broken(metrics: Mapping[str, float]) -> str | None:
            for term in brief.terms:
                m = term.metric.value
                if m not in metrics or m not in start:
                    continue
                limit = term.factor * start[m] + term.offset
                if not (metrics[m] <= limit if term.comparison == "<=" else metrics[m] >= limit):
                    return m
            return None

        calls = [c for c in ep["calls"] if c.get("completed")]
        for i in range(len(calls) - 1):
            first, second = calls[i], calls[i + 1]
            for step in first["steps"]:
                if not step.get("physically_valid") or step.get("out_id") != second["base_id"]:
                    continue
                g = gain(step["metrics"])
                if g is None or g < 1.0:
                    continue
                broke = step["status"] == "BRIEF_VIOLATION"
                ok_gains = [gain(s["metrics"]) for s in second["steps"]
                            if s.get("physically_valid") and s["status"] == "ACCEPTABLE"]
                ok_gains = [x for x in ok_gains if x is not None]
                rec: dict[str, Any] = {
                    "design": ep["design"], "brief": ep["brief"], "episode": ep["episode_id"],
                    "pattern": "break" if broke else "clean", "first_gain_bands": round(g, 4),
                    "final_bands": round(max(ok_gains), 4) if ok_gains else None,
                }
                if broke:
                    metric = first_broken(step["metrics"])
                    rec["second"] = "repair" if metric is not None and WORKER_METRIC.get(second["worker"]) == metric else "other"
                    rec["restored"] = any(x >= 1.0 for x in ok_gains)
                else:
                    rec["second"] = "any"
                    rec["further"] = any(x >= g + 1.0 for x in ok_gains)
                out.append(rec)
    return out


def table_payload(records: Sequence[Mapping[str, Any]], *, source: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": SCHEMA, "source": dict(source), "records": list(records)}


def load_records(path: Path | str) -> tuple[list[dict[str, Any]], str]:
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("schema_version") != SCHEMA:
        raise ValueError(f"unknown sequence table schema {data.get('schema_version')!r}")
    return list(data["records"]), hashlib.sha256(raw).hexdigest()


def _final_clause(finals: Sequence[float | None]) -> str:
    n = len(finals)
    shares: dict[str, int] = defaultdict(int)
    for value in finals:
        shares[NO_RESULT if value is None else categorise_v2(value, 1.0, higher_is_better=True)] += 1
    order = (*CATEGORIES_V2, NO_RESULT)
    ranked = sorted(((count / n, name) for name, count in shares.items() if count / n >= MENTION - 1e-9),
                    key=lambda t: (-t[0], order.index(t[1])))
    return ", ".join(f"{name} {frequency_word(share)}" for share, name in ranked)


def _group(records: Sequence[Mapping[str, Any]], **where: Any) -> list[Mapping[str, Any]]:
    return [r for r in records if all(r.get(k) == v for k, v in where.items())]


def render_sequences(records: Sequence[Mapping[str, Any]], *, brief: str, exclude_design: str | None, header: str,
                     intro: str, empty: str, min_designs: int = MIN_DESIGNS) -> tuple[str, dict[str, Any]]:
    """Section text (without the blank lines the prompt puts around it) and what it rests on."""

    kept = [r for r in records if r["brief"] == brief and (exclude_design is None or r["design"] != exclude_design)]
    lines = [header, intro]
    shown: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []

    def add(key: str, group: list[Mapping[str, Any]], text) -> None:
        n, k = len(group), len({r["design"] for r in group})
        row = {"line": key, "sequences": n, "designs": k}
        if n == 0:
            return
        if k < min_designs:
            omitted.append(row)
            return
        shown.append(row)
        lines.append(text(n, k, group))

    add("break_repair", _group(kept, pattern="break", second="repair"),
        lambda n, k, g: (f"  After a call that gained on the goal but broke a limit ({n} sequences from {k} designs), "
                         "a repair call from that state (by the worker of the broken limit): restored every limit with a goal "
                         f"gain {frequency_word(sum(bool(r['restored']) for r in g) / n)}; goal result that met every limit "
                         f"after the pair: {_final_clause([r['final_bands'] for r in g])}."))
    add("break_other", _group(kept, pattern="break", second="other"),
        lambda n, k, g: (f"  After a call that gained on the goal but broke a limit ({n} sequences from {k} designs), "
                         "any other second call from that state: restored every limit with a goal gain "
                         f"{frequency_word(sum(bool(r['restored']) for r in g) / n)}; goal result that met every limit after "
                         f"the pair: {_final_clause([r['final_bands'] for r in g])}."))
    add("clean_any", _group(kept, pattern="clean"),
        lambda n, k, g: (f"  After a call that gained on the goal and met every limit ({n} sequences from {k} designs), "
                         "a second call from that state: gained at least one further noise band on the goal "
                         f"{frequency_word(sum(bool(r['further']) for r in g) / n)}; "
                         f"no result that met every limit {frequency_word(sum(r['final_bands'] is None for r in g) / n)}."))
    if not shown:
        lines.append(empty)
    text = "\n".join(lines)
    return text, {"shown": shown, "omitted": omitted, "records_in_scope": len(kept),
                  "designs_in_scope": sorted({r["design"] for r in kept}), "excluded_design": exclude_design,
                  "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
