"""Round-2 harness rules on the LLM policies: the stop note in the view and free-text truncation in code."""
from __future__ import annotations

import json
from pathlib import Path

import llm_support as ls
from orch.contract.brief_text import load_frozen_brief
from orch.contract.models import MetricName
from orch.episode_r2 import STOP_NOTE
from orch.policies.llm import LLMPolicy, LLMPolicyConfig
from orch.policies.llm_client import CachedLLM, LLMParams, ReplayCache, TelemetrySink
from orch.policies.llm_prompt import load_template
from orch.policies.r2_experience import Call, build_table
from orch.policies.round2_harness import TEXT_LIMITS, F1Policy, V21Policy, insert_note, truncate_reply
from orch.policies.v2_experience import CallV2
from orch.policies.v2_policy import V2Policy
from orch.policies.v2_prompt import Variant, load_assets

ROOT = Path(__file__).resolve().parents[1]
V21 = Variant("template_v2.1", ("P1", "P2", "P3", "P4r"))


def v21_reply(*, reasoning="short", rationale="because", evidence=None, options=None, physical="none",
              action="INVOKE_WORKER"):
    mandate = None if action == "STOP" else {"worker_id": "timing", "base_state": "S0", "placement": "may change",
                                             "attempts": 1, "respect_brief_limits": True}
    forecast = None if action == "STOP" else {"goal": "small gain", "packed_luts": "within noise"}
    return json.dumps({
        "reasoning": reasoning, "evidence": evidence if evidence is not None else [],
        "physical_facts": physical,
        "options": options if options is not None else [{"option": "a", "expected_effect": "b"}] * 2,
        "rationale": rationale, "action": action, "mandate": mandate, "forecast": forecast,
    })


def make_v2(tmp_path, cls, *texts):
    transport = ls.ScriptedTransport(*texts)
    llm = CachedLLM(transport, ReplayCache(tmp_path / "cache", mode="record"), LLMParams(model="scripted"))
    assets = load_assets(ROOT)
    config = LLMPolicyConfig(name="v21", mode="fuzzy", profiles_on=False, template=assets.templates["base"])
    table = build_table([Call(d, w, s, True, {"wns": "within_noise"}) for d in ("a", "b") for w in ("timing", "area")
                         for s in ("preserve_placement_true", "preserve_placement_false")], "x")
    calls_v2 = [CallV2(d, "T0", "timing", "preserve_placement_true", "selected", {"wns_ns": 0.0, "packed_luts": 10.0},
                       {"wns_ns": 0.2, "packed_luts": 10.0}, {"wns_ns": 0.1, "packed_luts": 1.0}) for d in "abcd"]
    sink = TelemetrySink(tmp_path / "t.jsonl")
    policy = cls(config, llm, sink, variant=V21, assets=assets, design="z", describe=lambda sid: {},
                 r2_experience=table, calls_v2=calls_v2)
    return policy, transport


def make_f1(tmp_path, cls, *texts):
    transport = ls.ScriptedTransport(*texts)
    llm = CachedLLM(transport, ReplayCache(tmp_path / "cache", mode="record"), LLMParams(model="scripted"))
    config = LLMPolicyConfig(name="F1", mode="fuzzy", profiles_on=False,
                             template=load_template(ROOT / "configs/prompts/orchestrator_v2.txt"))
    return cls(config, llm, TelemetrySink(tmp_path / "t.jsonl")), transport


def view(budget_left=6):
    visited = [ls.state(10, -0.3, 1000), ls.state(11, -0.2, 1001, brief_valid=False)]
    return ls.view(visited=visited, calls=(), steps=(), budget_left=budget_left, brief=load_frozen_brief("T0"),
                   noise={MetricName.WNS_NS: 0.1, MetricName.PACKED_LUTS: 10.0})


def test_v21_without_note_sends_the_v2_prompt_and_the_same_mandate(tmp_path: Path) -> None:
    a, ta = make_v2(tmp_path / "a", V2Policy, v21_reply())
    b, tb = make_v2(tmp_path / "b", V21Policy, v21_reply())
    ma, mb = a.decide(view()), b.decide(view())
    assert ta.prompts == tb.prompts
    assert ma == mb and ma is not None
    assert b.version == a.version + "-r2h1"
    assert b.last_record["harness_note"] is None and b.last_record["truncated_fields"] == []


def test_v21_note_is_one_line_at_the_end_of_the_view(tmp_path: Path) -> None:
    plain, tp = make_v2(tmp_path / "a", V21Policy, v21_reply())
    noted, tn = make_v2(tmp_path / "b", V21Policy, v21_reply())
    plain.decide(view())
    noted.harness_note = STOP_NOTE
    noted.decide(view())
    text = noted.view_text(view()).rstrip("\n")
    assert tn.prompts[0].count(STOP_NOTE) == 1
    assert ("Current situation\n" + text + "\n" + STOP_NOTE + "\n") in tn.prompts[0]
    assert tn.prompts[0].replace(STOP_NOTE + "\n", "", 1) == tp.prompts[0]
    assert noted.last_record["harness_note"] == STOP_NOTE


def test_insert_note_refuses_a_prompt_without_the_view_block() -> None:
    import pytest

    with pytest.raises(ValueError):
        insert_note("no view here", "view", STOP_NOTE)
    assert insert_note("x", "view", None) == "x"


def test_v21_truncates_every_free_text_field(tmp_path: Path) -> None:
    long_words = " ".join(f"w{i}" for i in range(300))
    evidence = [{"source": "state", "label": "L" * 200, "fact": long_words} for _ in range(10)]
    options = [{"option": long_words, "expected_effect": long_words} for _ in range(10)]
    reply = v21_reply(reasoning=long_words, rationale="r" * 900, evidence=evidence, options=options, physical=long_words)
    policy, _ = make_v2(tmp_path, V21Policy, reply)
    mandate = policy.decide(view())
    record = policy.last_record
    assert len(record["reasoning"].split()) == TEXT_LIMITS["reasoning_words"]
    assert len(record["physical_facts"].split()) == TEXT_LIMITS["physical_facts_words"]
    assert len(record["rationale"]) == TEXT_LIMITS["rationale_chars"]
    assert len(record["evidence"]) == TEXT_LIMITS["evidence_items"]
    assert all(len(e["label"]) <= TEXT_LIMITS["evidence_label_chars"] for e in record["evidence"])
    assert all(len(e["fact"].split()) <= TEXT_LIMITS["evidence_fact_words"] for e in record["evidence"])
    assert len(record["options"]) == TEXT_LIMITS["options_items"]
    assert all(len(o["option"].split()) <= TEXT_LIMITS["option_words"] for o in record["options"])
    assert set(record["truncated_fields"]) == {"reasoning", "physical_facts", "rationale", "evidence", "options"}
    assert mandate is not None and len(json.loads(mandate.context_ref)["r"]) <= 200
    stored = [json.loads(line) for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert stored[-1]["reasoning"] == record["reasoning"]


def test_truncate_reply_keeps_fields_inside_the_limits() -> None:
    parsed = {"action": "STOP", "mandate": None, "reasoning": "a b c", "rationale": "ok", "physical_facts": "x",
              "evidence": [{"source": "brief", "label": "B", "fact": "f"}], "options": [{"option": "o", "expected_effect": "e"}]}
    out, cut = truncate_reply(parsed)
    assert out == parsed and cut == []


def test_f1_without_note_sends_the_round1_prompt(tmp_path: Path) -> None:
    reply = ls.decision("timing", "S0", preserve=True, attempts=1)
    a, ta = make_f1(tmp_path / "a", LLMPolicy, reply)
    b, tb = make_f1(tmp_path / "b", F1Policy, reply)
    assert a.decide(view()) == b.decide(view())
    assert ta.prompts == tb.prompts
    assert b.last_record["harness_note"] is None and b.version.endswith("-r2h1")


def test_f1_note_is_the_last_line_of_its_view(tmp_path: Path) -> None:
    reply = ls.decision("timing", "S0", preserve=True, attempts=1)
    plain, tp = make_f1(tmp_path / "a", F1Policy, reply)
    noted, tn = make_f1(tmp_path / "b", F1Policy, reply)
    plain.decide(view())
    noted.harness_note = STOP_NOTE
    noted.decide(view())
    assert tn.prompts[0].count(STOP_NOTE) == 1
    assert tn.prompts[0].replace("\n" + STOP_NOTE, "", 1) == tp.prompts[0]
    i = tn.prompts[0].index(STOP_NOTE)
    assert tn.prompts[0].rfind("Current situation", 0, i) > tn.prompts[0].rfind("Workers", 0, i)
    assert noted.last_record["harness_note"] == STOP_NOTE
    lines = [json.loads(x) for x in (tmp_path / "b" / "t.jsonl").read_text().splitlines()]
    assert len(lines) == 1 and lines[0]["harness_note"] == STOP_NOTE
