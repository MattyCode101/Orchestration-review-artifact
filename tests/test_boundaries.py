"""Architectural boundaries that must survive later policy implementations."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from orch.contract.models import (
    Brief,
    Evidence,
    Guard,
    Mandate,
    MetricName,
    MetricValue,
)
from orch.config import load_brief


ROOT = Path(__file__).resolve().parents[1]


def start_evidence() -> Evidence:
    return Evidence(
        id="start-evidence",
        state_id="a" * 64,
        metrics={
            MetricName.WNS_NS: MetricValue(
                value=-0.785, unit="ns", method="test-meter", basis="routed"
            ),
            MetricName.PACKED_LUTS: MetricValue(
                value=1895, unit="count", method="test-meter", basis="routed"
            ),
        },
        routed=True,
        route_errors=0,
        drc_errors=0,
        constraint_fingerprint="xdc",
        clock_digest="clock",
        meter_version="test-meter",
    )


def test_policy_modules_cannot_import_execution_or_recipe_code() -> None:
    """Deciders only receive the serializable view and produce mandates."""

    forbidden = (
        "orch.engine.runner",
        "orch.recipes.adapters",
        "orch.recipes.tcl",
    )
    for path in (ROOT / "orch" / "policies").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
                if node.level and (node.module or "").startswith(("engine", "recipes")):
                    pytest.fail(f"{path} imports execution code relatively")
        for name in imports:
            assert not name.startswith(forbidden), f"{path} imports {name}"
        assert "recipes/tcl/" not in source, f"{path} names a recipe script"


@pytest.mark.parametrize(
    ("brief_file", "guard"),
    [
        (
            "timing.yaml",
            Guard(
                id="loose_lut",
                metric=MetricName.PACKED_LUTS,
                comparison="<=",
                threshold=2000,
            ),
        ),
        (
            "area.yaml",
            Guard(
                id="loose_wns",
                metric=MetricName.WNS_NS,
                comparison=">=",
                threshold=-0.90,
            ),
        ),
    ],
)
def test_mandate_rejects_guard_looser_than_brief(
    brief_file: str, guard: Guard
) -> None:
    brief = load_brief(ROOT / "configs" / "briefs" / brief_file)
    mandate = Mandate(
        id="m",
        worker_id="area",
        base_state_id="a" * 64,
        guards=(guard,),
        brief_hash=brief.fingerprint(),
    )
    with pytest.raises(ValueError, match="loosens brief"):
        mandate.validate_against(brief, start_evidence())


def test_mandate_accepts_additional_strict_guard() -> None:
    brief = load_brief(ROOT / "configs" / "briefs" / "area.yaml")
    mandate = Mandate(
        id="m",
        worker_id="area",
        base_state_id="a" * 64,
        guards=(
            Guard(
                id="strict_wns",
                metric=MetricName.WNS_NS,
                comparison=">=",
                threshold=-0.80,
            ),
        ),
        brief_hash=brief.fingerprint(),
    )
    mandate.validate_against(brief, start_evidence())


def test_mandate_cannot_name_recipes_or_change_brief() -> None:
    brief = load_brief(ROOT / "configs" / "briefs" / "timing.yaml")
    with pytest.raises(ValidationError):
        Mandate.model_validate(
            {
                "id": "m",
                "worker_id": "timing",
                "base_state_id": "a" * 64,
                "brief_hash": brief.fingerprint(),
                "allowed_recipes": ["A.phys_opt"],
            }
        )
    stale = Mandate(
        id="m", worker_id="timing", base_state_id="a" * 64, brief_hash="stale"
    )
    with pytest.raises(ValueError, match="brief hash"):
        stale.validate_against(brief, start_evidence())


def test_contract_objects_export_json_schema() -> None:
    """The provisional records can be validated and archived as JSON."""

    from orch.contract import models

    for name in (
        "Brief",
        "WorkerCard",
        "DesignState",
        "Mandate",
        "RunRecord",
        "Claims",
        "Evidence",
        "Verdict",
        "Step",
    ):
        cls = getattr(models, name)
        schema = cls.model_json_schema()
        assert schema["type"] == "object"
        assert "schema_version" in schema["properties"]

