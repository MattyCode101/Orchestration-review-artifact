"""Registry and adapter conformance without launching Vivado or LATCH."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from orch.config import load_recipe_sets
from orch.contract.models import DesignState, RunStatus
from orch.interfaces import ProcessResult
from orch.recipes.adapters.features import FeaturesAdapter
from orch.recipes.adapters.latch import LatchAdapter
from orch.recipes.adapters.tcl import TclAdapter
from orch.recipes.adapters.validator import ValidatorAdapter
from orch.recipes.registry import load_registry, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def process(run_dir: Path, status: RunStatus = RunStatus.OK) -> ProcessResult:
    return ProcessResult(
        run_id="test-run",
        status=status,
        exit_code=0 if status == RunStatus.OK else 1,
        wall_s=1,
        process_group_kills=0,
        tag_sweep_kills=0,
        tag="test-tag",
        stdout_path=str(run_dir / "stdout.txt"),
        stderr_path=str(run_dir / "stderr.txt"),
    )


def test_all_candidates_and_copied_tcl_are_content_verified() -> None:
    registry = load_registry(ROOT)
    assert len(registry.all()) == 78
    assert len(registry.for_worker("area")) == 64
    assert len(registry.for_worker("timing")) == 8
    assert len(registry.for_worker("power")) == 5
    assert len(registry.for_worker("control")) == 1
    for recipe in registry.all():
        assert len(recipe.version) == 64
        if recipe.kind == "tcl":
            assert recipe.script is not None
            assert sha256_file(ROOT / recipe.script) == recipe.source_sha256
    assert not registry.get("A_E_DUPL").replaces
    assert registry.get("LATCH.auto_pblock").replaces
    assert not registry.get("LATCH.phys_opt").replaces


def test_versions_change_with_tool_adapter_and_timeout(tmp_path: Path) -> None:
    original = load_registry(ROOT)
    changed_tool = load_registry(
        ROOT, tool_versions={"vivado": "test-only-different-tool-version"}
    )
    changed_adapter = load_registry(
        ROOT, adapter_versions={"tcl": "test-only-adapter-revision"}
    )
    assert original.get("C0_reroute").version != changed_tool.get("C0_reroute").version
    assert original.get("C0_reroute").version != changed_adapter.get("C0_reroute").version
    assert original.get("LATCH.phys_opt").version != changed_tool.get("LATCH.phys_opt").version

    # Copy Tcl byte-for-byte; mutate only test YAML to check timeout invalidation.
    import shutil

    shutil.copytree(ROOT / "recipes", tmp_path / "recipes")
    raw = yaml.safe_load((tmp_path / "recipes/registry.yaml").read_text())
    next(r for r in raw["recipes"] if r["id"] == "C0_reroute")["timeout_s"] += 1
    (tmp_path / "recipes/registry.yaml").write_text(yaml.safe_dump(raw))
    altered = load_registry(tmp_path)
    assert original.get("C0_reroute").version != altered.get("C0_reroute").version


def test_recipe_sets_match_matts_train_only_directive() -> None:
    selected = load_recipe_sets(ROOT / "configs/recipe_sets.yaml")
    selected.require_signed_off()
    assert selected.worker_sets == {
        "timing": (
            "LATCH.phys_opt", "LATCH.retime", "LATCH.auto_pblock",
            "LATCH.hardblock_relocate", "LATCH.memory_path_opt",
        ),
        "area": (
            "A_R_EA", "A_R_RSA_SRLM", "A_R_RSA_AREM",
            "A_R_RSA_MED", "A_E_DUPL_Q1000",
        ),
        "power": (),
    }
    registry = load_registry(ROOT)
    for worker, ids in selected.worker_sets.items():
        for recipe_id in ids:
            assert registry.get(recipe_id).worker == worker


def test_tcl_adapter_reads_native_status_not_only_exit_code(tmp_path: Path) -> None:
    recipe = load_registry(ROOT).get("C0_reroute")
    work = tmp_path / "tcl_run"
    input_dcp = tmp_path / "source.dcp"
    input_dcp.write_bytes(b"dummy checkpoint")
    adapter = TclAdapter(ROOT)
    spec = adapter.build_launch(
        recipe, input_path=input_dcp, run_dir=work, run_id="test-run"
    )
    assert spec.argv[:2] == ("bash", "-c")
    subprocess.run(["bash", "-n", "-c", spec.argv[2]], check=True)
    assert "vivado -mode batch -nojournal -log vivado.log" in spec.argv[2]
    assert sha256_file(work / f"{recipe.id}.tcl") == recipe.source_sha256
    assert spec.input_copy_path == str(work / "input/input.dcp")
    assert (work / "input/input.dcp").read_bytes() == b"dummy checkpoint"
    assert adapter.classify(recipe, run_dir=work, process=process(work)) == RunStatus.TOOL_ERROR
    (work / "status.txt").write_text("FAILED_AT incr_read\n")
    (work / "out.dcp").write_bytes(b"partial")
    assert adapter.classify(recipe, run_dir=work, process=process(work)) == RunStatus.TOOL_ERROR
    (work / "status.txt").write_text("OK\n")
    (work / "out.dcp").unlink()
    assert adapter.classify(recipe, run_dir=work, process=process(work)) == RunStatus.NO_OUTPUT
    (work / "out.dcp").write_bytes(b"new checkpoint")
    assert adapter.classify(recipe, run_dir=work, process=process(work)) == RunStatus.OK
    assert adapter.parse_claims(recipe, run_dir=work, run_id="test-run").values["native_status"] == "OK"


def test_latch_adapter_uses_neutral_input_and_selected_output(tmp_path: Path) -> None:
    recipe = load_registry(ROOT).get("LATCH.phys_opt")
    fake_latch = tmp_path / "fake_latch"
    fake_latch.mkdir()
    (fake_latch / "dcp_optimizer.py").write_text("# test placeholder\n")
    fake_python = tmp_path / "venv_latch/bin/python"
    fake_python.parent.mkdir(parents=True)
    fake_python.symlink_to(Path(sys.executable))
    fake_java = tmp_path / "fake_jre/bin"
    fake_java.mkdir(parents=True)
    (fake_java / "java").write_text("")
    source = tmp_path / "original_design_name_B2.dcp"
    source.write_bytes(b"dummy checkpoint")
    adapter = LatchAdapter(
        ROOT,
        latch_root=fake_latch,
        latch_python=fake_python,
        java_home=tmp_path / "fake_jre",
        strict_pinning=False,
        stability_interval_s=0,
        publication_timeout_s=0,
    )
    work = tmp_path / "latch_run"
    spec = adapter.build_launch(
        recipe, input_path=source, run_dir=work, run_id="test-run"
    )
    subprocess.run(["bash", "-n", "-c", spec.argv[2]], check=True)
    # Resolving this symlink would invoke system Python instead of the venv.
    assert str(fake_python) in spec.argv[2]
    assert str(fake_python.resolve()) not in spec.argv[2]
    assert (work / "input/design.dcp").read_bytes() == b"dummy checkpoint"
    assert "original_design_name" not in spec.argv[2]
    assert "--no-dotenv --pipeline fixed --plan phys_opt --budget-runs 1" in spec.argv[2]
    assert spec.output_hint == str(work / "out.dcp")
    assert spec.script_sha256 == sha256_file(fake_latch / "dcp_optimizer.py")
    assert spec.input_copy_path == str(work / "input/design.dcp")
    transition = work / "latch_run/transition_000_phys_opt"
    transition.mkdir(parents=True)
    (transition / "output.dcp").write_bytes(b"rejected transition")
    assert adapter.classify(recipe, run_dir=work, process=process(work)) == RunStatus.NO_OUTPUT
    (work / "out.dcp").write_bytes(b"selected output")
    assert adapter.find_output(recipe, run_dir=work) == work / "out.dcp"
    (work / "stdout.txt").write_text("Final Fmax: 433.65 MHz (WNS: -0.736 ns)\n")
    claims = adapter.parse_claims(recipe, run_dir=work, run_id="test-run")
    assert claims.values["latch_reported_wns_ns"] == -0.736


def test_validator_adapter_copies_private_sources_and_reads_weak_report(
    tmp_path: Path,
) -> None:
    golden = tmp_path / "golden.dcp"
    revised = tmp_path / "revised.dcp"
    golden.write_bytes(b"golden")
    revised.write_bytes(b"revised")
    adapter = ValidatorAdapter(ROOT)
    work = tmp_path / "validation"
    spec = adapter.build_launch(
        golden_path=golden, revised_path=revised, run_dir=work, vectors=200
    )
    subprocess.run(["bash", "-n", "-c", spec.argv[2]], check=True)
    assert "--vectors 200" in spec.argv[2]
    assert (work / "validator/validate_dcps.py").is_file()
    assert (work / "validator/RapidWright").is_symlink()
    assert (work / "input/golden.dcp").read_bytes() == b"golden"
    temp = work / "validator/dcp_validation_test"
    temp.mkdir()
    (temp / "validation_report.json").write_text(
        json.dumps({"overall_passed": True, "infrastructure_failure": False})
    )
    check = adapter.result(
        run_dir=work,
        process=process(work),
        reference_state_id="a" * 64,
        vectors=200,
    )
    assert check.result == "pass"
    assert check.strength == "weak"
    assert (work / "validator_work/validation_report.json").is_file()
    assert not temp.exists()


def test_features_placeholder_refuses_unsourced_queries(tmp_path: Path) -> None:
    state = DesignState(
        id="a" * 64, design="test", path=str(tmp_path / "input.dcp")
    )
    with pytest.raises(NotImplementedError, match="sourced Tcl"):
        FeaturesAdapter().query(state, run_dir=tmp_path)



def test_support_script_hash_changes_both_recipe_kind_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from orch.recipes import registry as registry_module

    baseline = load_registry(ROOT)
    original_hash = registry_module.sha256_file

    def changed_hash(path: Path) -> str:
        if path.name == "lib_run3.tcl":
            return "0" * 64
        return original_hash(path)

    monkeypatch.setattr(registry_module, "sha256_file", changed_hash)
    changed = load_registry(ROOT)
    assert baseline.get("C0_reroute").version != changed.get("C0_reroute").version
    assert baseline.get("LATCH.phys_opt").version != changed.get("LATCH.phys_opt").version


def test_registry_rejects_wrong_script_hash_without_editing_tcl(
    tmp_path: Path,
) -> None:
    import shutil

    shutil.copytree(ROOT / "recipes", tmp_path / "recipes")
    path = tmp_path / "recipes/registry.yaml"
    raw = yaml.safe_load(path.read_text())
    next(r for r in raw["recipes"] if r["id"] == "C0_reroute")[
        "source_sha256"
    ] = "0" * 64
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="script hash mismatch"):
        load_registry(tmp_path)



def test_registry_rejects_wrong_latch_launcher_hash(tmp_path: Path) -> None:
    import shutil

    shutil.copytree(ROOT / "recipes", tmp_path / "recipes")
    path = tmp_path / "recipes/registry.yaml"
    raw = yaml.safe_load(path.read_text())
    next(r for r in raw["recipes"] if r["id"] == "LATCH.phys_opt")[
        "source_sha256"
    ] = "0" * 64
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="LATCH launcher source hash mismatch"):
        load_registry(tmp_path)



def test_validator_preserves_virtualenv_interpreter_symlink(tmp_path: Path) -> None:
    """The venv launcher path must survive even when it points to system Python."""

    launcher = tmp_path / "python"
    launcher.symlink_to(sys.executable)
    adapter = ValidatorAdapter(ROOT, python=launcher)
    assert adapter.python == launcher.absolute()
    assert adapter.python != launcher.resolve()
