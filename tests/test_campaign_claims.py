"""The pre-registered claims C1 to C4 are decided mechanically: ties, strictness, no-room cells, missing cells."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import campaign_analysis as ca  # noqa: E402

DESIGNS = ("d1", "d2")
START = {"wns_ns": -1.0, "packed_luts": 1000.0, "dynamic_power_w": 2.0}
BAND = {"wns_ns": 0.01, "packed_luts": 5.0, "dynamic_power_w": 0.02}
BEST = {"T0": -0.0, "T": 0.0, "A0": 500.0, "A": 500.0, "P0": 1.0}           # best known per brief (all designs alike)
METRIC = {"T0": "wns_ns", "T": "wns_ns", "A0": "packed_luts", "A": "packed_luts", "P0": "dynamic_power_w"}
TEMPLATE, PROFILE = "t" * 64, "p" * 64


def frame(no_room: tuple[str, str] | None = None) -> ca.Frame:
    best = {(d, b): BEST[b] for d in DESIGNS for b in ca.ALL_BRIEFS}
    if no_room:
        best[no_room] = START[METRIC[no_room[1]]] + (0.001 if METRIC[no_room[1]] == "wns_ns" else 0.0)    # inside the band
    return ca.Frame(DESIGNS, {d: dict(START) for d in DESIGNS}, {d: dict(BAND) for d in DESIGNS}, best, room_rule=True)


def value_for(brief: str, share: float) -> float:
    start, best = START[METRIC[brief]], BEST[brief]
    return start + share * (best - start)


def rows(spec: dict[str, dict[str, float]], *, llm: bool = False, p0_template: str = TEMPLATE, tags: dict | None = None):
    """spec: approach -> brief -> share (the same for both designs)."""

    out = defaultdict(lambda: defaultdict(list))
    for approach, briefs in spec.items():
        for brief, share in briefs.items():
            for d in DESIGNS:
                row = {"design": d, "brief": brief, "approach": approach, "tag": "m1" if approach in ca.LLM else "x",
                       "episode_id": f"{d}_{brief}_{approach}", "final_state_id": f"{approach}{brief}{d}",
                       "final_metrics": {METRIC[brief]: value_for(brief, share)}}
                if approach == "F1-R1":
                    template = p0_template if brief == "P0" else TEMPLATE
                    row["tag"], row["llm"] = "m1", {"template_sha256": template, "profile_sha256": {"power": PROFILE},
                                                    "workers_seen": ["area", "power", "timing"] if brief == "P0" else ["area", "timing"]}
                out[approach][(d, brief)].append(row)
    return out


FOUR = ("T0", "A0", "T", "A")
BASE = {"single_best_non_llm": "v2", "per_brief_best_non_llm": {b: "v2" for b in FOUR}}


def uniform(share: float, briefs=FOUR) -> dict[str, float]:
    return {b: share for b in briefs}


def verdicts(spec, fr=None, **kw):
    result = ca.claims(fr or frame(), rows(spec, **kw), BASE, manifest_profiles={"power": PROFILE})
    return {k.split()[0]: v["verdict"] for k, v in result.items() if isinstance(v, dict) and "verdict" in v}, result


def test_all_four_claims_supported_when_f1_leads_everywhere() -> None:
    v, _ = verdicts({"F1-R1": uniform(0.8, ca.ALL_BRIEFS), "v2": uniform(0.6, ca.ALL_BRIEFS), "GA-R1": uniform(0.4),
                     "v1": uniform(0.5)})
    assert v == {"C1": "supported", "C2": "supported", "C3": "supported", "C4": "supported"}


def test_c1_an_exact_tie_on_the_worst_brief_is_supported_and_a_higher_other_is_not() -> None:
    tie = {"F1-R1": {"T0": 0.5, "A0": 0.9, "T": 0.9, "A": 0.9}, "v2": {"T0": 0.9, "A0": 0.5, "T": 0.9, "A": 0.9}}
    assert verdicts(tie)[0]["C1"] == "supported"
    lower = {"F1-R1": {"T0": 0.49, "A0": 0.9, "T": 0.9, "A": 0.9}, "v2": {"T0": 0.9, "A0": 0.5, "T": 0.9, "A": 0.9}}
    assert verdicts(lower)[0]["C1"] == "not supported"


def test_c1_counts_every_ablation_and_the_playbook() -> None:
    assert verdicts({"F1-R1": uniform(0.6), "E0-R1": uniform(0.7), "v2": uniform(0.1)})[0]["C1"] == "not supported"
    spec = {"F1-R1": uniform(0.5), "v2": uniform(0.9, ("A0", "A")) | uniform(0.1, ("T0", "T")),
            "v1": uniform(0.9, ("T0", "T")) | uniform(0.1, ("A0", "A"))}
    assert ca.claims(frame(), rows(spec), BASE, manifest_profiles={"power": PROFILE})["C1 robustness"]["verdict"] == "supported"
    playbook = {"single_best_non_llm": "v2", "per_brief_best_non_llm": {"T0": "v1", "A0": "v2", "T": "v1", "A": "v2"}}
    result = ca.claims(frame(), rows(spec), playbook, manifest_profiles={"power": PROFILE})
    assert result["C1 robustness"]["verdict"] == "not supported"           # no single approach beats F1-R1, the playbook does


def test_c1_is_not_supported_if_f1_lacks_a_brief() -> None:
    assert verdicts({"F1-R1": uniform(0.9, ("T0", "A0", "T")), "v2": uniform(0.1)})[0]["C1"] == "not supported"


def test_c1_random_is_a_mean_over_its_seeds() -> None:
    spec = {"F1-R1": uniform(0.5), "v2": uniform(0.1)}
    r = rows(spec)
    for seed, share in enumerate((0.9, 0.9, 0.1)):
        for brief in FOUR:
            for d in DESIGNS:
                r[f"random_{seed}"][(d, brief)].append({"design": d, "brief": brief, "approach": f"random_{seed}", "tag": "x",
                                                         "episode_id": f"{d}{brief}{seed}", "final_state_id": "s",
                                                         "final_metrics": {METRIC[brief]: value_for(brief, share)}})
    for brief in FOUR:
        for d in DESIGNS:
            r["random"][(d, brief)] = [x for s in range(3) for x in r[f"random_{s}"][(d, brief)]]
    table = {t["approach"]: t for t in ca.robustness(frame(), r, ["random"])}
    assert abs(table["random"]["worst_brief"] - (0.9 + 0.9 + 0.1) / 3) < 1e-9


def test_c2_equal_means_are_supported_lower_is_not_and_the_sign_test_does_not_decide() -> None:
    equal = {"F1-R1": uniform(0.5), "v2": uniform(0.5)}
    v, result = verdicts(equal)
    assert v["C2"] == "supported"
    assert result["C2 versus rule-based"]["ties"] == 4 and result["C2 versus rule-based"]["wins"] == 0
    assert verdicts({"F1-R1": uniform(0.49), "v2": uniform(0.5)})[0]["C2"] == "not supported"
    one_win = {"F1-R1": {"T0": 0.9, "A0": 0.5, "T": 0.5, "A": 0.5}, "v2": uniform(0.5)}
    assert verdicts(one_win)[0]["C2"] == "supported"                                     # p of one win is 1.0, verdict unchanged


def test_c2_leaves_a_no_room_cell_out_of_the_means() -> None:
    fr = frame(no_room=("d1", "T0"))
    spec = {"F1-R1": {"T0": 0.0, "A0": 0.6, "T": 0.6, "A": 0.6}, "v2": uniform(0.5)}
    _, result = verdicts(spec, fr)
    c2 = result["C2 versus rule-based"]
    assert c2["no_room_cells"] == 1 and c2["cells_compared"] == 3                        # d1 T0 out; d2 T0, d1 A0, d2 A0 in


def test_c2_a_missing_cell_is_dropped_from_the_comparison_only() -> None:
    r = rows({"F1-R1": uniform(0.6), "v2": uniform(0.5)})
    del r["F1-R1"][("d1", "A0")]
    result = ca.claims(frame(), r, BASE, manifest_profiles={"power": PROFILE})
    c2 = result["C2 versus rule-based"]
    assert c2["missing_cells"] == 1 and c2["cells_compared"] == 3 and c2["verdict"] == "supported"


def test_c3_conditions() -> None:
    full = {"F1-R1": uniform(0.7, ca.ALL_BRIEFS), "v2": uniform(0.5, ca.ALL_BRIEFS)}
    assert verdicts(full)[0]["C3"] == "supported"
    assert verdicts(full, p0_template="x" * 64)[0]["C3"] == "not supported"              # another template on P0
    loss = {"F1-R1": {**uniform(0.7, ca.ALL_BRIEFS), "P0": 0.1}, "v2": uniform(0.5, ca.ALL_BRIEFS)}
    assert verdicts(loss)[0]["C3"] == "not supported"                                    # a loss beyond the power band
    inside = {"F1-R1": {**uniform(0.7, ca.ALL_BRIEFS), "P0": 0.49}, "v2": uniform(0.5, ca.ALL_BRIEFS)}   # 0.01 share = 0.01 W < band
    assert verdicts(inside)[0]["C3"] == "supported"
    assert verdicts({"F1-R1": uniform(0.7), "v2": uniform(0.5)})[0]["C3"] == "not supported"             # no P0 rows


def test_c3_needs_the_power_profile_hash_and_all_three_workers() -> None:
    spec = {"F1-R1": uniform(0.7, ca.ALL_BRIEFS), "v2": uniform(0.5, ca.ALL_BRIEFS)}
    r = rows(spec)
    for items in r["F1-R1"].values():
        for row in items:
            if row["brief"] == "P0":
                row["llm"]["workers_seen"] = ["area", "timing"]
    assert ca.claims(frame(), r, BASE, manifest_profiles={"power": PROFILE})["C3 unseen worker"]["verdict"] == "not supported"
    r = rows(spec)
    assert ca.claims(frame(), r, BASE, manifest_profiles={"power": "other"})["C3 unseen worker"]["verdict"] == "not supported"


def test_c4_needs_strictly_above() -> None:
    assert verdicts({"F1-R1": uniform(0.5), "GA-R1": uniform(0.5), "v2": uniform(0.1)})[0]["C4"] == "not supported"
    assert verdicts({"F1-R1": uniform(0.51), "GA-R1": uniform(0.5), "v2": uniform(0.1)})[0]["C4"] == "supported"
    assert verdicts({"F1-R1": uniform(0.9), "v2": uniform(0.1)})[0]["C4"] == "not supported"            # no GA-R1 rows


def test_a_fresh_repeat_never_enters_a_mean() -> None:
    r = rows({"F1-R1": uniform(0.5), "v2": uniform(0.5)})
    for key, items in list(r["F1-R1"].items()):
        repeat = dict(items[0], tag="m2", final_metrics={next(iter(items[0]["final_metrics"])): 0.0})
        items.append(repeat)
    main = ca.main_run_rows(r)
    assert all(len(v) == 1 and v[0]["tag"] == "m1" for v in main["F1-R1"].values())
