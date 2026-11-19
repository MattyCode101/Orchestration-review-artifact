"""The stage driver's Saturday-night additions: own shared directory, run cap, spend ledger, thinking level."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
ENV_NAMES = ("SATURDAY_LLM_SHARED", "SATURDAY_LLM_RUN_CAP", "SATURDAY_LLM_USD_CAP", "SATURDAY_LLM_PRICE_IN",
             "SATURDAY_LLM_PRICE_OUT", "SATURDAY_LLM_USD_HEADROOM", "SATURDAY_LLM_CONFIG")


def _driver(monkeypatch, tmp_path, **env):
    """Reload the driver under an environment; point its shared directory into tmp_path afterwards."""

    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    module = importlib.reload(importlib.import_module("scripts.saturday_llm"))
    module.derived_shared_name = module.SHARED.name
    module.SHARED = tmp_path / "shared"
    return module


@pytest.fixture(autouse=True)
def _restore_driver_after_the_test(monkeypatch):
    yield
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    importlib.reload(importlib.import_module("scripts.saturday_llm"))


def test_the_defaults_are_the_saturday_values(monkeypatch, tmp_path) -> None:
    driver = _driver(monkeypatch, tmp_path)
    assert driver.derived_shared_name == "saturday_llm_shared" and driver.CAP == 400
    assert driver.spend_settings() is None and driver.spend_ledger() is None


def test_the_environment_moves_the_shared_directory_and_the_run_cap(monkeypatch, tmp_path) -> None:
    driver = _driver(monkeypatch, tmp_path, SATURDAY_LLM_SHARED="saturday_llm_argon_shared", SATURDAY_LLM_RUN_CAP="200")
    assert driver.derived_shared_name == "saturday_llm_argon_shared" and driver.CAP == 200


def test_a_spend_cap_needs_both_prices(monkeypatch, tmp_path) -> None:
    driver = _driver(monkeypatch, tmp_path, SATURDAY_LLM_USD_CAP="35")
    with pytest.raises(driver.pilot.PilotSafetyStop, match="PRICE"):
        driver.spend_settings()


def test_the_spend_settings_carry_the_cap_the_prices_and_the_headroom(monkeypatch, tmp_path) -> None:
    driver = _driver(monkeypatch, tmp_path, SATURDAY_LLM_USD_CAP="35", SATURDAY_LLM_PRICE_IN="2.0",
                     SATURDAY_LLM_PRICE_OUT="10.0")
    assert driver.spend_settings() == {
        "cap_usd": 35.0, "headroom_usd": 1.5, "price_in_per_million": 2.0, "price_out_per_million": 10.0,
    }
    assert driver.spend_ledger().stop_usd == pytest.approx(33.5)


def test_with_a_spend_cap_the_transport_is_wrapped_and_the_parameters_follow_the_config(monkeypatch, tmp_path) -> None:
    config = yaml.safe_load((ROOT / "configs/policies/llm.yaml").read_text(encoding="utf-8"))
    config.update({"model": "gemini-4-argon", "temperature": None, "seed": None, "thinking_level": "medium",
                   "max_output_tokens": 16384})
    path = tmp_path / "argon.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    driver = _driver(monkeypatch, tmp_path, SATURDAY_LLM_CONFIG=str(path), SATURDAY_LLM_USD_CAP="35",
                     SATURDAY_LLM_PRICE_IN="2.0", SATURDAY_LLM_PRICE_OUT="10.0")
    driver.set_stage("unit")
    llm = driver.make_llm(driver.Cell("F1", "d", "T0", "r0", "record", None, "orchestrator", "ab"))
    assert type(llm.transport).__name__ == "LedgeredTransport"
    params = llm.params
    assert (params.model, params.temperature, params.seed, params.thinking_level) == ("gemini-4-argon", None, None, "medium")
    assert params.max_output_tokens == 16384


def test_without_a_spend_cap_the_transport_is_the_plain_bridge(monkeypatch, tmp_path) -> None:
    driver = _driver(monkeypatch, tmp_path)
    driver.set_stage("unit")
    llm = driver.make_llm(driver.Cell("F1", "d", "T0", "r0", "record", None, "orchestrator", "ab"))
    assert type(llm.transport).__name__ == "BridgeTransport" and llm.params.thinking_level is None
    assert llm.params.temperature == 0.0 and llm.params.seed == 0


def test_the_new_modules_are_in_the_stage_pin_list() -> None:
    text = (ROOT / "scripts/saturday_llm.py").read_text(encoding="utf-8")
    assert '"orch/engine/start_properties.py"' in text and '"orch/policies/llm_ledger.py"' in text


def test_a_dropped_configuration_cannot_be_planned(monkeypatch, tmp_path) -> None:
    """F2 (the lessons block) is a negative training result and is dropped (decision of 2026-11-07)."""

    driver = _driver(monkeypatch, tmp_path)
    assert driver.settings()["configurations"]["F2"].get("dropped") is True
    for name in ("F1", "F0", "E0", "E1"):
        assert not driver.settings()["configurations"][name].get("dropped")
    kwargs = dict(tag="r0", cache="record", fold_mode="own", kind="orchestrator", workers="ab",
                  briefs=["T0"], designs=["vexriscv_re-place"])
    with pytest.raises(ValueError, match="dropped"):
        driver.make_work("unit", ["F2"], **kwargs)
    assert len(driver.make_work("unit", ["F1"], **{**kwargs, "fold_mode": "none"})) == 1
