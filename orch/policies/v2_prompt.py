"""Template v2: package assembly and prompt rendering.

Packages (configs/prompts/template_v2/p1_clarity.yaml ... p4_reply.yaml) are exact text edits applied in order
P1, P2, P3, P4 to the round-1 wording (configs/prompts/orchestrator2_v1.txt), plus switches for the parts code
renders. `assemble` builds any package combination; scripts/template_v2_2026-11-12/t2_build_templates.py writes
each combination the evaluation uses as its own template file (configs/prompts/template_v2/assembled/).

`render` turns one decision situation into (prompt, schema, meta) for a variant. With no package and no variant
switch it reproduces the round-2 L3 prompt byte for byte (round-1 wording, history, physical description and
leave-one-design-out experience), which is the base of the cumulative ladder.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from orch.contract.brief_text import render_brief_text
from orch.contract.models import MetricName
from orch.contract.views import PolicyView
from orch.policies.llm_prompt import Template, render_cards, render_prompt
from orch.policies.llm_schemas import (
    MAX_ATTEMPTS, MAX_CONFLICT_WORDS, MAX_DIAGNOSIS_WORDS, MAX_RATIONALE_CHARS, STRATEGIES,
)
from orch.policies.llm_view import METRIC_NAMES
from orch.policies.r2_experience import ExperienceTable, render_experience_section
from orch.policies.r2_physical import facts_from_record
from orch.policies.r2_prompt import render_prompt_r2, section_block
from orch.policies.r2_view import Features, alias_order, build_r2_view, goal_and_limits, state_lookup
from orch.policies.v2_experience import CallV2, render_experience_v2
from orch.policies.v2_physical import render_p3_section, render_r2_section
from orch.policies.v2_schemas import SchemaFlags, reply_fields_text, reply_schema
from orch.policies.v2_text import START_ALIAS, render_brief_text_v2, render_cards_v2, render_vocabulary
from orch.policies.v2_view import ViewFlags, render_view_v2

PACKAGE_FILES = {"P1": "p1_clarity.yaml", "P2": "p2_neutral.yaml", "P3": "p3_physical.yaml", "P4": "p4_reply.yaml",
                 "P4r": "p4_reply_rev1.yaml"}
PACKAGE_ORDER = ("P1", "P2", "P3", "P4", "P4r")      # P4r is revision 1 of P4; the two are never combined
GROUPS_ALL = ("timing", "utilisation", "power")
_NUMBER_WORDS = {1: "one", 2: "two", 3: "three"}


class AssemblyError(ValueError):
    """A package edit does not apply exactly once."""


def load_package(folder: Path, name: str) -> dict[str, Any]:
    data = yaml.safe_load((folder / PACKAGE_FILES[name]).read_text(encoding="utf-8"))
    if data.get("package") != name:
        raise AssemblyError(f"{PACKAGE_FILES[name]} declares package {data.get('package')!r}")
    return data


def apply_edits(text: str, edits: Sequence[Mapping[str, Any]], *, package: str) -> str:
    for i, edit in enumerate(edits):
        new = edit["new"]
        if "old" in edit:
            count = text.count(edit["old"])
            if count != 1:
                raise AssemblyError(f"{package} edit {i}: old text occurs {count} times")
            text = text.replace(edit["old"], new)
        elif "after" in edit:
            marker = edit["after"]
            if text.count(marker) != 1:
                raise AssemblyError(f"{package} edit {i}: marker occurs {text.count(marker)} times")
            text = text[: text.index(marker) + len(marker)] + new
        elif "between" in edit:
            start, end = edit["between"]
            if text.count(start) != 1 or text.count(end) != 1 or text.index(start) > text.index(end):
                raise AssemblyError(f"{package} edit {i}: markers are not unique and ordered")
            text = text[: text.index(start)] + new + text[text.index(end):]
        else:
            raise AssemblyError(f"{package} edit {i}: unknown edit type")
    return text


def combo_name(packages: Sequence[str]) -> str:
    return "base" if not packages else "p" + "".join(p[1:] for p in sorted(packages))


def assemble(base_text: str, packages: Sequence[str], folder: Path) -> str:
    if "P4" in packages and "P4r" in packages:
        raise AssemblyError("P4 and its revision P4r cannot be combined")
    text = base_text
    for name in PACKAGE_ORDER:
        if name in packages:
            text = apply_edits(text, load_package(folder, name)["edits"], package=name)
    return text


@dataclass(frozen=True)
class Variant:
    """One template variant of the evaluation."""

    name: str
    packages: tuple[str, ...] = ()
    wording: str = "W1"           # worker description wording (P1 only)
    exact_only: bool = False      # noise categories hidden: exact values only

    @property
    def p1(self) -> bool:
        return "P1" in self.packages

    @property
    def p2(self) -> bool:
        return "P2" in self.packages

    @property
    def p3(self) -> bool:
        return "P3" in self.packages

    @property
    def p4(self) -> bool:
        return "P4" in self.packages or "P4r" in self.packages

    @property
    def p4r(self) -> bool:
        return "P4r" in self.packages

    @property
    def combo(self) -> str:
        return combo_name(self.packages)

    def schema_flags(self) -> SchemaFlags:
        return SchemaFlags(p2=self.p2, p3=self.p3, p4=self.p4, vocab_v2=self.p1, exact_only=self.exact_only,
                           p4r=self.p4r)


@dataclass
class Situation:
    """Everything a template needs about one decision point (built from a decision-bank snapshot)."""

    view: PolicyView
    design: str
    describe: Callable[[str], Mapping[str, Any]]
    r2_experience: ExperienceTable | None           # the round-2 table, evaluated design left out
    calls_v2: Sequence[CallV2] = field(default_factory=list)


@dataclass
class Assets:
    folder: Path
    templates: dict[str, Template]
    texts: dict[str, dict[str, str]]
    wordings: dict[str, dict[str, str]]


def load_assets(project: Path) -> Assets:
    from orch.policies.llm_prompt import load_template

    folder = project / "configs/prompts/template_v2"
    templates = {"base": load_template(project / "configs/prompts/orchestrator2_v1.txt")}
    for path in sorted((folder / "assembled").glob("orchestrator3_*.txt")):
        templates[path.stem.split("_", 1)[1]] = load_template(path)
    texts = {name: dict(load_package(folder, name).get("texts") or {}) for name in PACKAGE_ORDER}
    wordings = yaml.safe_load((folder / "worker_descriptions.yaml").read_text(encoding="utf-8"))
    return Assets(folder, templates, texts, wordings)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _physical_entries(sit: Situation, aliases: Mapping[str, str]) -> list[tuple[str, bool, dict[str, Any]]]:
    lookup = state_lookup(sit.view)
    out = []
    for alias in alias_order(list(aliases)):
        state_id = aliases[alias]
        facts = facts_from_record(sit.describe(state_id))
        facts["dynamic_power_w"] = lookup[state_id].metrics.get(MetricName.DYNAMIC_POWER_W)
        out.append((alias, alias == "S0", facts))
    return out


def _parents(view: PolicyView, state_to_alias: Mapping[str, str]) -> dict[str, str | None]:
    parents: dict[str, str | None] = {view.start.state_id: None}
    for step in view.steps:
        out_id = step.output_state_id
        if out_id is None or out_id == step.base_state_id or out_id in parents or out_id not in state_to_alias:
            continue
        parents[out_id] = state_to_alias.get(step.base_state_id, "unknown")
    return parents


def render(sit: Situation, variant: Variant, assets: Assets) -> tuple[str, dict[str, Any], dict[str, Any]]:
    view = sit.view
    workers = [card.id for card in view.worker_cards]
    max_attempts = min(MAX_ATTEMPTS, view.budget_left)
    goal_name, limit_names = goal_and_limits(view)
    meta: dict[str, Any] = {"variant": variant.name, "combo": variant.combo}

    if not variant.packages and not variant.exact_only:
        # the base of the ladder: the round-2 L3 prompt, built exactly as R2Policy builds it
        features = Features(history=True, physical=GROUPS_ALL, experience="fuzzy", experience_source="lodo")
        built = build_r2_view(view, features)
        physical = render_r2_section(_physical_entries(sit, built.alias_to_state), GROUPS_ALL,
                                     power_band=view.noise_bands.get(MetricName.DYNAMIC_POWER_W), v2_vocab=False)
        assert sit.r2_experience is not None
        experience = render_experience_section(sit.r2_experience, goal=goal_name, limits=limit_names,
                                               workers=workers, form="fuzzy")
        schema = reply_schema(SchemaFlags(), workers=workers, aliases=list(built.alias_to_state),
                              max_attempts=max_attempts, limit_names=limit_names)
        prompt = render_prompt_r2(
            assets.templates["base"], brief_text=render_brief_text(view.brief),
            cards_text=render_cards(view.worker_cards, None), view_text=built.text, schema=schema,
            max_attempts=max_attempts, physical_text=physical, experience_text=experience,
        )
        meta.update(aliases=dict(built.alias_to_state), template_sha256=assets.templates["base"].sha256,
                    prompt_sha256=_sha(prompt))
        return prompt, schema, meta

    template = assets.templates[variant.combo]
    flags = ViewFlags(labels=variant.p1, vocab=variant.p1, placement=variant.p2, exact_only=variant.exact_only)
    if variant.p1 or variant.p2 or variant.exact_only:
        built_v2 = render_view_v2(view, flags)
        view_text, alias_to_state = built_v2.text, built_v2.alias_to_state
    else:
        built = build_r2_view(view, Features(history=True))
        view_text, alias_to_state = built.text, dict(built.alias_to_state)
    state_to_alias = {s: a for a, s in alias_to_state.items()}
    entries = _physical_entries(sit, alias_to_state)
    power_band = view.noise_bands.get(MetricName.DYNAMIC_POWER_W)
    if variant.p3:
        parents = _parents(view, state_to_alias)
        p3_entries = [(alias, parents.get(alias_to_state[alias]), facts) for alias, _, facts in entries]
        physical = render_p3_section(p3_entries, intro=assets.texts["P3"]["physical_intro"], power_band=power_band,
                                     categories=not variant.exact_only, v2_vocab=variant.p1)
    else:
        physical = render_r2_section(entries, GROUPS_ALL, power_band=power_band, v2_vocab=variant.p1,
                                     categories=not variant.exact_only)
    if variant.p2:
        goal_metric = view.brief.objective.metric.value
        limit_metrics = [t.metric.value for t in view.brief.terms]
        t2 = assets.texts["P2"]
        experience, exp_meta = render_experience_v2(
            sit.calls_v2, brief=view.brief.id, goal=goal_metric, limits=limit_metrics, workers=workers,
            exclude_design=sit.design, header=t2["experience_header"], intro=t2["experience_intro"],
            empty_text=t2["experience_empty"], form="exact" if variant.exact_only else "words", v2=variant.p1,
            placement_v2=True,
        )
        meta["experience"] = exp_meta
    else:
        assert sit.r2_experience is not None
        experience = render_experience_section(sit.r2_experience, goal=goal_name, limits=limit_names,
                                               workers=workers, form="fuzzy")
    descriptions = assets.wordings[variant.wording] if variant.p1 else None
    cards = render_cards_v2(view.worker_cards, descriptions=descriptions, placement_v2=variant.p2)
    brief_text = render_brief_text_v2(view.brief) if variant.p1 else render_brief_text(view.brief)
    flags_s = variant.schema_flags()
    texts4 = assets.texts["P4r" if variant.p4r else "P4"]
    schema = reply_schema(flags_s, workers=workers, aliases=list(alias_to_state), max_attempts=max_attempts,
                          limit_names=limit_names, texts=texts4)
    values = {
        "BRIEF": brief_text.rstrip(), "WORKER_CARDS": cards.rstrip(), "LESSONS_SECTION": "",
        "VIEW": view_text.rstrip("\n"), "PHYSICAL_SECTION": section_block(physical),
        "EXPERIENCE_SECTION": section_block(experience), "SCHEMA": json.dumps(schema, indent=1, sort_keys=False),
        "STRATEGIES": ", ".join(STRATEGIES), "MAX_DIAGNOSIS_WORDS": str(MAX_DIAGNOSIS_WORDS),
        "MAX_CONFLICT_WORDS": str(MAX_CONFLICT_WORDS), "MAX_RATIONALE_CHARS": str(MAX_RATIONALE_CHARS),
        "MAX_ATTEMPTS": _NUMBER_WORDS.get(max_attempts, str(max_attempts)),
        "ATTEMPTS_RULE": ("Give exactly one attempt." if max_attempts <= 1 else
                          f"Give attempts between one and {_NUMBER_WORDS.get(max_attempts, str(max_attempts))}."),
        "VOCABULARY": render_vocabulary(exact_only=variant.exact_only), "START_ALIAS": START_ALIAS,
        "REPLY_FIELDS": reply_fields_text(schema, flags=flags_s) if variant.p4 else "",
    }
    prompt = template.text
    for name, value in values.items():
        prompt = prompt.replace("{{" + name + "}}", value)
    if "{{" in prompt:
        raise AssemblyError("unfilled placeholder in the rendered prompt")
    meta.update(aliases=dict(alias_to_state), template_sha256=template.sha256, prompt_sha256=_sha(prompt))
    return prompt, schema, meta


__all__ = ["Assets", "AssemblyError", "Situation", "Variant", "apply_edits", "assemble", "combo_name",
           "load_assets", "load_package", "render", "render_prompt"]
