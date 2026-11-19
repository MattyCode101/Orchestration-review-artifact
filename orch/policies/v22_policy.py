"""V22Policy: template v2.1 (V21Policy) with the six switches of template v2.2 (new file; nothing frozen is edited).

The decision loop is V21Policy.decide copied with one change: the prompt comes from orch.policies.v22_prompt.render_v22
instead of orch.policies.v2_prompt.render. With the default switches the prompts, mandates and telemetry fields equal
V21Policy's (tests/test_v22_policy.py). The harness note of the round-2 loop (the stop note) is inserted after the
state list of whichever form H uses. With C on, the telemetry record also holds the candidates offered and which one the
mandate equals (exactly, with other settings, a different call, or a stop).
"""
from __future__ import annotations

import json
from typing import Any

from orch.contract.models import Guard, Mandate, WorkerSettings
from orch.contract.views import PolicyView
from orch.policies.llm_prompt import repair_prompt
from orch.policies.llm_schemas import MAX_RATIONALE_CHARS
from orch.policies.llm_view import METRIC_NAMES
from orch.policies.round2_harness import V21Policy, _sha, insert_note, truncate_reply
from orch.policies.v2_prompt import Situation
from orch.policies.v2_schemas import ReplyError, parse_reply
from orch.policies.v22_candidates import which_candidate
from orch.policies.v22_prompt import V21_VARIANT, V22Context, render_v22, view_text_for
from orch.policies.v22_switches import Switches

VERSION_TAG = "v22a"


class V22Policy(V21Policy):
    id = "llm_orchestrator_v22"

    def __init__(self, *args: Any, switches: Switches, context: V22Context, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.variant.packages != V21_VARIANT.packages:
            raise ValueError("V22Policy is built on the template v2.1 variant")
        self.switches, self.context = switches, context
        self.version = f"{self.version}-{VERSION_TAG}-{switches.key}"

    def view_text(self, view: PolicyView) -> str:
        return view_text_for(view, self.switches)

    def decide(self, view: PolicyView) -> Mandate | None:
        if view.budget_left <= 0:
            return None
        note = self.harness_note
        sit = Situation(view=view, design=self.design, describe=self._describe, r2_experience=self.r2_experience,
                        calls_v2=self.calls_v2)
        rendered, schema, meta = render_v22(sit, self.variant, self.assets, self.switches, self.context)
        base_prompt = insert_note(rendered, self.view_text(view), note)
        aliases = meta["aliases"]
        workers = [c.id for c in view.worker_cards]
        limits = [METRIC_NAMES[t.metric] for t in view.brief.terms]
        flags = self.variant.schema_flags()
        base_sha = _sha(base_prompt)
        prompt, calls, parsed, last_error = base_prompt, [], None, None
        for _turn in range(1 + max(0, self.config.max_repairs)):
            result = self.llm.complete(prompt, schema)
            error = None
            try:
                parsed = parse_reply(result.reply.text, flags, workers=workers, aliases=list(aliases), limit_names=limits)
                self._check(parsed, view, aliases)
            except ReplyError as exc:
                error, parsed = str(exc), None
            calls.append(self._call_record(prompt, result, error))
            if error is None:
                break
            last_error = error
            prompt = repair_prompt(base_prompt, result.reply.text, error)
        cut: list[str] = []
        if parsed is not None:
            parsed, cut = truncate_reply(parsed)
        mandate = None
        if parsed is not None and parsed["action"] == "INVOKE_WORKER":
            m = parsed["mandate"]
            guards: tuple[Guard, ...] = ()
            if m["respect"]:
                guards = tuple(Guard(id=f"respect_{t.id}", metric=t.metric, comparison=t.comparison,
                                     threshold=t.factor * view.start.metrics[t.metric] + t.offset)
                               for t in view.brief.terms if t.metric in view.start.metrics)
            reason = str(parsed.get("rationale", "")).strip()[:MAX_RATIONALE_CHARS]
            identity = _sha(f"{view.episode_id}|{len(view.calls)}|{base_sha}|{json.dumps(m, sort_keys=True)}")
            mandate = Mandate(
                id=f"mandate_{identity[:24]}", worker_id=m["worker"], base_state_id=aliases[m["base"]],
                settings=WorkerSettings(preserve_placement=m["preserve"], max_runs=m["attempts"]), guards=guards,
                brief_hash=view.brief.fingerprint(),
                context_ref=json.dumps({"s": parsed.get("strategy") or "-", "r": reason, "p": base_sha[:16]},
                                       sort_keys=True, separators=(",", ":")),
            )
        sink, self.telemetry = self.telemetry, None          # write the record once, with the v2.2 fields added
        try:
            self._log_v21(view, meta, base_sha, _sha(rendered), calls, parsed, last_error, note, cut, mandate)
        finally:
            self.telemetry = sink
        record = dict(self.last_record or {})
        record["switches"] = self.switches.as_json()
        if "candidate_objects" in meta:
            cands = meta["candidate_objects"]
            record["candidates"] = [c.as_record() for c in cands]
            record["candidate_pick"] = which_candidate(cands, None if parsed is None else parsed["mandate"], aliases)
        if "relative" in meta:
            record["relative"] = meta["relative"]
        if "sequence" in meta:
            record["sequence"] = {k: meta["sequence"][k] for k in ("shown", "omitted", "records_in_scope", "sha256")}
        if "situation" in meta:
            record["situation"] = {k: meta["situation"][k] for k in
                                   ("blocks", "repair_states", "push_from", "shown", "omitted", "rows_in_scope", "sha256")
                                   if k in meta["situation"]}
        self.last_record = record
        if self.telemetry is not None:
            self.telemetry.append(record)
        return mandate
