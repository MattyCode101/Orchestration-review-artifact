"""Pure-function checks for the Friday P0 pilot driver."""
from __future__ import annotations

from scripts import friday_p0 as p0


def test_schedule_is_every_policy_on_every_training_design_once() -> None:
    rows = p0.rows()
    assert len(rows) == len(p0.POLICIES) * 5 == 55 == len(set(rows))
    assert {brief for _, brief, _ in rows} == {"P0"}
    assert len(p0.POLICIES) == 11
    # policy-major: the first five cells are one policy over all designs
    assert [row[2] for row in rows[:5]] == ["repair_loop"] * 5
    assert {row[0] for row in rows[:5]} == set(p0.DESIGNS)


def test_policy_keys_map_to_distinct_frozen_policies() -> None:
    versions = {key: p0.policy_for(key).version for key in p0.POLICIES}
    ids = {key: p0.policy_for(key).id for key in p0.POLICIES}
    assert versions["repair_loop"] == "1.0-frozen"
    assert ids["fixed_CAB"] == "fixed_CAB" and ids["single_specialist"] == "single_specialist"
    assert len({(ids[k], versions[k]) for k in p0.POLICIES}) == 11
    for order in ("CAB", "CBA", "ACB", "ABC", "BCA", "BAC"):
        assert p0.policy_for(f"fixed_{order}").order == order
    for seed in (0, 1, 2):
        assert p0.policy_for(f"random_seed_{seed}").seed == seed


def test_episode_ids_are_unique_and_distinct_from_the_sweep() -> None:
    ids = {p0.episode_id(*row) for row in p0.rows()}
    assert len(ids) == 55 and all(i.startswith("p0_") for i in ids)


def test_p0_pins_cover_every_file_that_defines_the_p0_policies() -> None:
    pinned = set(p0.SOURCE_FILES.values())
    for needed in (
        "scripts/friday_p0.py", "orch/policies/repair_loop.py",
        "orch/policies/relevant_workers.py", "orch/policies/training_baselines.py",
        "orch/contract/brief_text.py", "configs/briefs.yaml",
        "configs/recipe_sets.yaml", "configs/power_recipe_set.yaml",
        "configs/workers/power.yaml",
    ):
        assert needed in pinned
