"""Template v2.2: the renderer and the policy (new tests). Synthetic situations; the real-bank identity check skips when the
bank file is absent."""
from __future__ import annotations

import json
from pathlib import Path

import llm_support as ls
import pytest
from orch.contract.models import MetricName
from orch.episode_r2 import STOP_NOTE
from orch.policies.llm import LLMPolicyConfig
from orch.policies.llm_client import CachedLLM, LLMParams, ReplayCache, TelemetrySink
from orch.policies.round2_harness import V21Policy
from orch.policies.v2_experience import SCOPES, CallV2
from orch.policies.v2_prompt import Situation, load_assets, render
from orch.policies.v22_policy import V22Policy
from orch.policies.v22_prompt import V21_VARIANT, V22Context, render_v22
from orch.policies.v22_switches import VARIANTS, Switches, load_switch_texts

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data/template_v2_2026-11-12/bank/bank.jsonl"
ASSETS = load_assets(ROOT)
TEXTS = load_switch_texts(ROOT)
RECORD = {"wpath": {"requirement_ns": 1.5}, "probe": {"utilization_summary": {"CLB LUTs": {"available": 100000}}}}
SEQUENCES = [{"design": d, "brief": "T0", "pattern": "break", "second": "repair", "restored": True, "final_bands": 4.0,
              "first_gain_bands": 2.0, "episode": "x"} for d in ("a", "b", "c", "d")]
CALLS_V2 = [CallV2(d, "T0", w, s, "selected", {"wns_ns": 0.0, "packed_luts": 10.0}, {"wns_ns": 0.2, "packed_luts": 10.0},
                   {"wns_ns": 0.1, "packed_luts": 1.0}) for d in "abcd" for w in ("timing", "area") for s in SCOPES]


def view(*, broken: bool = False, calls: bool = False):
    visited = [ls.state(10, -0.30, 1000)]
    if broken:
        visited.append(ls.state(11, -0.10, 1030, brief_valid=False))
    kwargs = {}
    if calls:
        kwargs = {"calls": (ls.call(1, "timing", 10, 11, attempts=2, runs=2, reason="push"),),
                  "steps": (ls.step("timing", 10, 11),), "budget_left": 4}
    return ls.view(visited=visited, **kwargs)


def situation(v, design: str = "z") -> Situation:
    return Situation(view=v, design=design, describe=lambda sid: RECORD, r2_experience=None, calls_v2=CALLS_V2)


def _sit_row(design, worker, preserve, situation, **kw):
    base = {"design": design, "group": "TRAINING", "brief": "T0", "source": "v4", "worker": worker, "preserve": preserve,
            "base": "b", "n_runs": 2, "situation": situation, "depth_b": None, "outcome": "selected", "d_goal_b": 2.0,
            "d_limit_b": 0.0, "restored": None, "goal": "wns_ns", "limit": "packed_luts"}
    base.update(kw)
    return base


SIT_ROWS = [_sit_row(d, w, p, s, **({"depth_b": 3.0, "restored": w == "area" and not p} if s == "break" else {}))
            for d in "abcd" for w in ("timing", "area") for p in (True, False) for s in ("start", "gain", "break")]
CTX = V22Context(TEXTS, SEQUENCES, candidate_seed=0, situations=SIT_ROWS)
# The switches of revision 1 (candidate options C and two-call sequences S) are kept in the code but no grid variant uses
# them; these combinations keep their tests alive.
C_ONLY = Switches(C=True)
CSR = Switches(C=True, S=True, R=True)
COMPACT_CSR = Switches(H="compact", C=True, S=True, R=True, P=False)


def test_default_switches_equal_the_frozen_v21_render() -> None:
    for v in (view(), view(broken=True, calls=True)):
        frozen, schema, meta = render(situation(v), V21_VARIANT, ASSETS)
        mine, schema2, meta2 = render_v22(situation(v), V21_VARIANT, ASSETS, Switches(), CTX)
        assert mine == frozen and schema2 == schema
        assert meta2["template_sha256"] == meta["template_sha256"] and meta2["aliases"] == meta["aliases"]


@pytest.mark.skipif(not BANK.exists(), reason="the template-v2 decision bank is not on this machine")
def test_default_switches_equal_the_frozen_render_on_the_real_bank_points() -> None:
    import sys

    sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/template_v2_2026-11-12")]
    import t2_eval as E

    bank = E.Bank()
    for pid in ("b101", "b105", "b119"):
        sit = bank.situation(bank.points[pid])
        assert render_v22(sit, V21_VARIANT, bank.assets, Switches(), CTX)[0] == render(sit, V21_VARIANT, bank.assets)[0]


def test_each_switch_leaves_its_trace() -> None:
    marks = {"E": "Past experience with the workers on other designs", "P": "Physical description of the saved states",
             "R": "Relative context", "S": "Past experience with two calls in a row", "C": "Candidate next calls",
             "S2": "Past experience in the situation of this decision"}
    neutral = "- Any worker can move the goal: check what each worker and setting did to the goal in this situation."
    push = "- Push: the worker whose objective matches the goal usually moves the goal most"
    combos = {**VARIANTS, "C": C_ONLY, "CSR": CSR, "compact CSR": COMPACT_CSR}
    for name, sw in combos.items():
        prompt, schema, _meta = render_v22(situation(view(broken=True, calls=True)), V21_VARIANT, ASSETS, sw, CTX)
        assert "{{" not in prompt
        for flag, mark in marks.items():
            if flag == "E":
                assert (mark in prompt) == sw.E, (name, flag)
            else:
                assert (mark in prompt) == getattr(sw, flag), (name, flag)
        assert (neutral in prompt) == sw.U and (push in prompt) == (not sw.U), name
        assert ("trajectory log:" in prompt) == (sw.H == "full"), name
        assert ("physical_facts" in schema["properties"]) == sw.P, name
        assert ("physical_facts" in prompt) == sw.P, name
        assert ("experience" in prompt.lower()) == (sw.experience_shown or sw.C), name


def test_section_order_is_state_list_relative_physical_experience_situation_sequence_candidates_reply() -> None:
    prompt, *_ = render_v22(situation(view(broken=True)), V21_VARIANT, ASSETS, Switches(C=True, S=True, S2=True, R=True), CTX)
    order = ["Current situation", "Relative context", "Physical description of the saved states",
             "Past experience with the workers on other designs", "Past experience in the situation of this decision",
             "Past experience with two calls in a row", "Candidate next calls", "Reply format"]
    positions = [prompt.index(m) for m in order]
    assert positions == sorted(positions)
    assert "(-20.0% of the clock requirement)" in prompt                      # the anchors come from the start's record


def test_the_situation_section_follows_the_decision_and_names_no_design() -> None:
    opening, *_ = render_v22(situation(view()), V21_VARIANT, ASSETS, VARIANTS["V7"], CTX)
    assert "Situation: no call has been made yet." in opening and "Situation: a saved state gained" not in opening
    broken, _, meta = render_v22(situation(view(broken=True, calls=True)), V21_VARIANT, ASSETS, VARIANTS["V7"], CTX)
    assert "S1 (packed_luts broken by 3.0 noise bands)" in broken and "Limit broken by over two bands in the base state:" in broken
    assert "restored the limit and kept a goal gain of at least one band" in broken
    assert meta["situation"]["blocks"] == ["repair"] and meta["situation"]["excluded_design"] == "z"
    for text in (opening, broken):
        assert "d_alpha" not in text and " design a " not in text
    none, *_ = render_v22(situation(view()), V21_VARIANT, ASSETS, VARIANTS["V6"],
                          V22Context(TEXTS, [], situations=[r for r in SIT_ROWS if r["design"] in "ab"]))
    assert none.count(": no evidence for this situation") == 4                    # two designs: no line rests on three


def test_candidate_section_names_no_rule_and_is_deterministic() -> None:
    a, _, meta = render_v22(situation(view(broken=True)), V21_VARIANT, ASSETS, C_ONLY, CTX)
    b, *_ = render_v22(situation(view(broken=True)), V21_VARIANT, ASSETS, C_ONLY, CTX)
    assert a == b and [c["kind"] for c in sorted(meta["candidates"], key=lambda c: c["label"])]
    block = a[a.index("Candidate next calls"):a.index("Reply format")]
    assert "Option A:" in block and "Option C:" in block
    for word in ("push", "repair", "gentle", "recommend", "best"):
        assert word not in block.lower().replace("best state", "")
    assert "Experience for timing, placement may change (4 calls from 4 designs)" in block
    other = CTX.__class__(TEXTS, SEQUENCES, candidate_seed=7, situations=SIT_ROWS)
    c, *_ = render_v22(situation(view(broken=True)), V21_VARIANT, ASSETS, C_ONLY, other)
    def unlabelled(text: str) -> list[str]:
        return sorted(line.split(": ", 1)[1] if line.startswith("Option ") else line for line in text.splitlines())

    assert unlabelled(c) == unlabelled(a)                                      # same content, order and labels may differ


def test_candidates_without_experience_lines_when_e_is_off() -> None:
    prompt, *_ = render_v22(situation(view()), V21_VARIANT, ASSETS, Switches(E=False, C=True), CTX)
    assert "Candidate next calls" in prompt and "Experience for" not in prompt


def test_the_renderer_refuses_another_variant() -> None:
    from orch.policies.v2_prompt import Variant

    with pytest.raises(ValueError):
        render_v22(situation(view()), Variant("template_v2", ("P1", "P2", "P3")), ASSETS, Switches(), CTX)


# ------------------------------------------------------------------------------------------------ policy
def v21_reply(worker="timing", base="S0", placement="may change", attempts=2, action="INVOKE_WORKER"):
    mandate = None if action == "STOP" else {"worker_id": worker, "base_state": base, "placement": placement,
                                             "attempts": attempts, "respect_brief_limits": False}
    return json.dumps({"reasoning": "r", "evidence": [], "physical_facts": "none",
                       "options": [{"option": "a", "expected_effect": "b"}] * 2, "rationale": "because", "action": action,
                       "mandate": mandate, "forecast": None if action == "STOP" else {"goal": "small gain", "packed_luts": "within noise"}})


def make(tmp_path: Path, cls, reply: str, **extra):
    transport = ls.ScriptedTransport(reply)
    llm = CachedLLM(transport, ReplayCache(tmp_path / "cache", mode="record"), LLMParams(model="scripted"))
    config = LLMPolicyConfig(name="v22", mode="fuzzy", profiles_on=False, template=ASSETS.templates["base"])
    sink = TelemetrySink(tmp_path / "t.jsonl")
    policy = cls(config, llm, sink, variant=V21_VARIANT, assets=ASSETS, design="z", describe=lambda sid: RECORD,
                 r2_experience=None, calls_v2=CALLS_V2, **extra)
    return policy, transport


def test_v22_with_default_switches_sends_the_v21_prompt_and_the_same_mandate(tmp_path: Path) -> None:
    a, ta = make(tmp_path / "a", V21Policy, v21_reply())
    b, tb = make(tmp_path / "b", V22Policy, v21_reply(), switches=Switches(), context=CTX)
    ma, mb = a.decide(view()), b.decide(view())
    assert ta.prompts == tb.prompts and ma == mb and ma is not None
    assert b.version.startswith(a.version) and b.version.endswith("-v22a-EPHrcstu")
    assert b.last_record["switches"] == Switches().as_json() and "candidates" not in b.last_record


def test_v22_records_the_candidates_offered_and_the_pick(tmp_path: Path) -> None:
    policy, transport = make(tmp_path, V22Policy, v21_reply("area", "S1", "may change", 1), switches=CSR, context=CTX)
    mandate = policy.decide(view(broken=True))
    assert mandate is not None and mandate.worker_id == "area"
    record = json.loads((tmp_path / "t.jsonl").read_text().splitlines()[-1])
    assert len(record["candidates"]) == 3 and record["switches"]["C"] is True
    assert record["candidate_pick"]["kind"] == "repair" and record["candidate_pick"]["exact"] is True
    assert record["relative"]["period_ns"] == 1.5 and record["sequence"]["shown"]
    assert len((tmp_path / "t.jsonl").read_text().splitlines()) == 1             # the record is written once


def test_v22_records_the_situation_lines_it_showed(tmp_path: Path) -> None:
    policy, transport = make(tmp_path, V22Policy, v21_reply("area", "S1", "may change", 1), switches=VARIANTS["V7"], context=CTX)
    mandate = policy.decide(view(broken=True, calls=True))                  # a broken state exists only after a call
    assert mandate is not None and mandate.worker_id == "area"
    record = json.loads((tmp_path / "t.jsonl").read_text().splitlines()[-1])
    assert record["switches"]["S2"] is True and record["switches"]["U"] is True and "candidates" not in record
    assert record["situation"]["blocks"] == ["repair"] and record["situation"]["repair_states"][0][0] == "S1"
    assert record["situation"]["shown"] and len(record["situation"]["sha256"]) == 64
    assert "Any worker can move the goal" in transport.prompts[0] and "- Push:" not in transport.prompts[0]
    assert policy.version.endswith("-v22a-EPHRcsTU")
    assert len((tmp_path / "t.jsonl").read_text().splitlines()) == 1


def test_v22_stop_note_goes_after_the_compact_state_list(tmp_path: Path) -> None:
    policy, transport = make(tmp_path, V22Policy, v21_reply(), switches=VARIANTS["V8"], context=CTX)
    policy.harness_note = STOP_NOTE
    policy.decide(view(broken=True))
    prompt = transport.prompts[0]
    assert prompt.count(STOP_NOTE) == 1
    anchor = "Current situation\n" + policy.view_text(view(broken=True)).rstrip("\n") + "\n" + STOP_NOTE + "\n"
    assert anchor in prompt and "trajectory log" not in prompt
    stop, _ = make(tmp_path / "s", V22Policy, v21_reply(action="STOP"), switches=C_ONLY, context=CTX)
    assert stop.decide(view()) is None
    assert json.loads((tmp_path / "s" / "t.jsonl").read_text().splitlines()[-1])["candidate_pick"]["pick"] == "stop"
