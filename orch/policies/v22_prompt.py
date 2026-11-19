"""Template v2.2: the prompt renderer with the six switches (new file; orch/policies/v2_prompt.py is untouched).

`render_v22` is the v2.1 path of orch.policies.v2_prompt.render (packages P1, P2, P3, P4r) with the switches applied:

  E off   no experience section; the template sentences that point to experience lines and the two P4r field
          descriptions that cite them are cut (exact-once edits, configs/prompts/template_v22/switch_texts.yaml)
  P off   no physical description; the intro sentence loses its physical clause; the reply has no physical_facts field
  H       compact state list (orch/policies/v22_view.render_view_compact) instead of render_view_v2
  R on    "Relative context" section before the physical description (orch/policies/v22_relative)
  S on    sequence experience after the experience section (orch/policies/v22_sequence; no grid variant uses it)
  S2 on   situation experience S' after the experience section: lines for the situation of the current decision
          (orch/policies/v22_situation, table configs/experience/situations_v22.json)
  C on    "Candidate next calls" last before the reply format (orch/policies/v22_candidates; no grid variant uses it)
  U on    the Push consideration of the template becomes a neutral line (an exact-once template edit, switch_texts.yaml)

With the defaults the output equals orch.policies.v2_prompt.render for template v2.1 byte for byte (tested). The schema is
the v2.1 schema (P off drops physical_facts, because that field exists only when P3 is on).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from orch.contract.models import MetricName
from orch.policies.llm_schemas import (
    MAX_ATTEMPTS, MAX_CONFLICT_WORDS, MAX_DIAGNOSIS_WORDS, MAX_RATIONALE_CHARS, STRATEGIES,
)
from orch.policies.r2_prompt import section_block
from orch.policies.r2_view import goal_and_limits
from orch.policies.v2_experience import MIN_DESIGNS, SCOPES, line_stats, render_experience_v2, render_line
from orch.policies.v2_experience import experience_calls_for
from orch.policies.v2_physical import render_p3_section
from orch.policies.v2_prompt import AssemblyError, Assets, Situation, Variant, _parents, _physical_entries
from orch.policies.v2_schemas import SchemaFlags, reply_fields_text, reply_schema
from orch.policies.v2_text import START_ALIAS, render_brief_text_v2, render_cards_v2, render_vocabulary
from orch.policies.v2_view import ViewFlags, render_view_v2
from orch.policies.v22_candidates import Candidate, generate_candidates, render_candidates, shuffled
from orch.policies.v22_relative import anchors_from_record, render_relative
from orch.policies.v22_sequence import render_sequences
from orch.policies.v22_situation import render_situation
from orch.policies.v22_switches import Switches, reply_texts_for, template_text_for
from orch.policies.v22_view import render_view_compact

V21_VARIANT = Variant("template_v2.1", ("P1", "P2", "P3", "P4r"), wording="W1")
_NUMBER_WORDS = {1: "one", 2: "two", 3: "three"}


@dataclass
class V22Context:
    """What the new switches need beyond a v2.1 situation."""

    switch_texts: Mapping[str, Any]
    sequences: Sequence[Mapping[str, Any]] = field(default_factory=list)
    candidate_seed: int = 0
    situations: Sequence[Mapping[str, Any]] = field(default_factory=list)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def view_text_for(view, sw: Switches) -> str:
    return (render_view_v2(view, ViewFlags(labels=True, vocab=True, placement=True, exact_only=False)).text
            if sw.H == "full" else render_view_compact(view).text)


def render_v22(sit: Situation, variant: Variant, assets: Assets, sw: Switches,
               ctx: V22Context) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if variant.packages != V21_VARIANT.packages or variant.wording != V21_VARIANT.wording or variant.exact_only:
        raise ValueError("template v2.2 is built on the v2.1 variant (P1, P2, P3, P4r; wording W1)")
    view = sit.view
    workers = [card.id for card in view.worker_cards]
    max_attempts = min(MAX_ATTEMPTS, view.budget_left)
    _goal_name, limit_names = goal_and_limits(view)
    meta: dict[str, Any] = {"variant": variant.name, "combo": variant.combo, "switches": sw.as_json()}

    built = (render_view_v2(view, ViewFlags(labels=True, vocab=True, placement=True, exact_only=False))
             if sw.H == "full" else render_view_compact(view))
    view_text, alias_to_state = built.text, built.alias_to_state
    state_to_alias = {s: a for a, s in alias_to_state.items()}
    power_band = view.noise_bands.get(MetricName.DYNAMIC_POWER_W)

    physical = ""
    if sw.P:
        entries = _physical_entries(sit, alias_to_state)
        parents = _parents(view, state_to_alias)
        p3_entries = [(alias, parents.get(alias_to_state[alias]), facts) for alias, _, facts in entries]
        physical = render_p3_section(p3_entries, intro=assets.texts["P3"]["physical_intro"], power_band=power_band,
                                     categories=True, v2_vocab=True)

    goal_metric = view.brief.objective.metric.value
    limit_metrics = [t.metric.value for t in view.brief.terms]
    experience = ""
    if sw.E:
        t2 = assets.texts["P2"]
        experience, exp_meta = render_experience_v2(
            sit.calls_v2, brief=view.brief.id, goal=goal_metric, limits=limit_metrics, workers=workers,
            exclude_design=sit.design, header=t2["experience_header"], intro=t2["experience_intro"],
            empty_text=t2["experience_empty"], form="words", v2=True, placement_v2=True,
        )
        meta["experience"] = exp_meta

    texts = ctx.switch_texts
    relative = ""
    if sw.R:
        anchors = anchors_from_record(sit.describe(view.start.state_id))
        rt = texts["relative"]
        relative = render_relative(view, state_to_alias, anchors, header=rt["header"], intro_all=rt["intro_all"],
                                   intro_no_period=rt["intro_no_period"])
        meta["relative"] = {"period_ns": anchors.period_ns, "device_luts": anchors.device_luts}

    sequence = ""
    if sw.S:
        st = texts["sequence"]
        sequence, seq_meta = render_sequences(ctx.sequences, brief=view.brief.id, exclude_design=sit.design,
                                              header=st["header"], intro=st["intro"], empty=st["empty"])
        meta["sequence"] = seq_meta

    situation = ""
    if sw.S2:
        situation, sit_meta = render_situation(ctx.situations, view, design=sit.design, texts=texts["situation"],
                                               workers=workers)
        meta["situation"] = sit_meta

    candidates_text = ""
    if sw.C:
        decision_index = len(view.calls)
        cands = shuffled(generate_candidates(view), seed=ctx.candidate_seed, episode_id=view.episode_id,
                         decision_index=decision_index)
        ct = texts["candidates"]
        kept = experience_calls_for(sit.calls_v2, brief=view.brief.id, exclude_design=sit.design)

        def line_for(worker: str, preserve: bool) -> str | None:
            if not sw.E:
                return None
            stats = line_stats(kept, worker=worker, scope=SCOPES[0] if preserve else SCOPES[1], goal=goal_metric,
                               limits=limit_metrics, v2=True)
            if stats.calls == 0 or stats.designs < MIN_DESIGNS:
                return None
            return render_line(stats, goal=goal_metric, limits=limit_metrics, form="words", v2=True,
                               placement_v2=True).strip()

        candidates_text = render_candidates(cands, view, header=ct["header"], intro=ct["intro"], no_line=ct["no_line"],
                                            line_for=line_for if sw.E else None)
        meta["candidates"] = [c.as_record() for c in cands]
        meta["candidate_objects"] = cands

    template_text = template_text_for(assets.templates[variant.combo].text, sw, texts)
    flags_s = SchemaFlags(p2=True, p3=sw.P, p4=True, vocab_v2=True, exact_only=False, p4r=True)
    texts4 = reply_texts_for(assets.texts["P4r"], sw, texts)
    descriptions = assets.wordings[variant.wording]
    cards = render_cards_v2(view.worker_cards, descriptions=descriptions, placement_v2=True)
    schema = reply_schema(flags_s, workers=workers, aliases=list(alias_to_state), max_attempts=max_attempts,
                          limit_names=limit_names, texts=texts4)
    values = {
        "BRIEF": render_brief_text_v2(view.brief).rstrip(), "WORKER_CARDS": cards.rstrip(), "LESSONS_SECTION": "",
        "VIEW": view_text.rstrip("\n"),
        "PHYSICAL_SECTION": section_block(relative) + section_block(physical),
        "EXPERIENCE_SECTION": (section_block(experience) + section_block(situation) + section_block(sequence)
                               + section_block(candidates_text)),
        "SCHEMA": json.dumps(schema, indent=1, sort_keys=False),
        "STRATEGIES": ", ".join(STRATEGIES), "MAX_DIAGNOSIS_WORDS": str(MAX_DIAGNOSIS_WORDS),
        "MAX_CONFLICT_WORDS": str(MAX_CONFLICT_WORDS), "MAX_RATIONALE_CHARS": str(MAX_RATIONALE_CHARS),
        "MAX_ATTEMPTS": _NUMBER_WORDS.get(max_attempts, str(max_attempts)),
        "ATTEMPTS_RULE": ("Give exactly one attempt." if max_attempts <= 1 else
                          f"Give attempts between one and {_NUMBER_WORDS.get(max_attempts, str(max_attempts))}."),
        "VOCABULARY": render_vocabulary(exact_only=False), "START_ALIAS": START_ALIAS,
        "REPLY_FIELDS": reply_fields_text(schema, flags=flags_s),
    }
    prompt = template_text
    for name, value in values.items():
        prompt = prompt.replace("{{" + name + "}}", value)
    if "{{" in prompt:
        raise AssemblyError("unfilled placeholder in the rendered prompt")
    meta.update(aliases=dict(alias_to_state), template_sha256=_sha(template_text), prompt_sha256=_sha(prompt))
    return prompt, schema, meta


__all__ = ["Candidate", "V21_VARIANT", "V22Context", "render_v22", "view_text_for"]
