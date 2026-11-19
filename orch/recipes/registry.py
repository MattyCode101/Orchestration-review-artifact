"""Data-driven candidate registry with computed, content-based recipe versions.

UNTESTED REAL TOOL PATH: loading and hashing is unit tested; executing a
registered recipe through the new adapters awaits the real smoke.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from orch.interfaces import RecipeSpec


ADAPTER_VERSIONS = {
    "tcl": "tcl-adapter-v0.1.0",
    "latch": "latch-adapter-v0.1.0",
}
REGISTRY_SCHEMA = "registry-v0-provisional"


def sha256_file(path: Path) -> str:
    """Hash bytes without normalizing line endings or reading Tcl as text."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(payload: dict[str, object]) -> str:
    """Stable hash used in the state/recipe/version run-cache key."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Registry:
    """Validated candidates; selected worker subsets live in recipe_sets.yaml."""

    root: Path
    recipes: Mapping[str, RecipeSpec]
    catalogue_sha256: str
    tool_versions: Mapping[str, str]

    def get(self, recipe_id: str) -> RecipeSpec:
        return self.recipes[recipe_id]

    def for_worker(self, worker_id: str) -> tuple[RecipeSpec, ...]:
        return tuple(r for r in self.recipes.values() if r.worker == worker_id)

    def all(self) -> tuple[RecipeSpec, ...]:
        return tuple(self.recipes.values())


def load_registry(
    root: Path | None = None,
    *,
    tool_versions: Mapping[str, str] | None = None,
    adapter_versions: Mapping[str, str] | None = None,
) -> Registry:
    """Load YAML and compute versions from the bytes and run environment.

    A version includes the copied recipe script or LATCH flow, adapter version,
    observed tool version, sourced env.sh, Tcl helpers and timeout. The extra
    inputs prevent a changed helper or deadline from reusing a stale cache
    outcome. Source: final architecture section 4.1; R3 C0_reroute.tcl:6-7.
    """

    project = (root or Path(__file__).resolve().parents[2]).resolve()
    path = project / "recipes/registry.yaml"
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or raw.get("schema_version") != REGISTRY_SCHEMA:
        raise ValueError("unsupported recipe registry schema")
    entries = raw.get("recipes")
    if not isinstance(entries, list):
        raise ValueError("registry recipes must be a list")
    versions = dict(raw.get("tool_versions") or {})
    versions.update(tool_versions or {})
    adapters = dict(ADAPTER_VERSIONS)
    adapters.update(adapter_versions or {})
    support = project / "recipes/support"
    env_hash = sha256_file(support / "env.sh")
    tcl_helpers = {
        "lib_measure": sha256_file(support / "lib_measure.tcl"),
        "lib_run3": sha256_file(support / "lib_run3.tcl"),
        "measure2": sha256_file(support / "measure2.tcl"),
        "diag_lut": sha256_file(support / "diag_lut.tcl"),
    }
    recipe_root = (project / "recipes/tcl").resolve()
    latch_launcher_sha = sha256_file(support / "latch_job.sh")
    resolved: dict[str, RecipeSpec] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("recipe entry must be an object")
        recipe_id = str(entry["id"])
        if recipe_id in resolved:
            raise ValueError(f"duplicate recipe id {recipe_id}")
        kind = entry.get("kind")
        if kind not in ADAPTER_VERSIONS:
            raise ValueError(f"unsupported recipe kind {kind}")
        if kind == "tcl":
            rel = Path(str(entry["script"]))
            script = (project / rel).resolve()
            if not script.is_relative_to(recipe_root):
                raise ValueError(f"recipe script escapes recipes/tcl: {recipe_id}")
            actual_sha = sha256_file(script)
            if actual_sha != entry["source_sha256"]:
                raise ValueError(f"script hash mismatch for {recipe_id}")
            executable_source = {"script_sha256": actual_sha}
            tool = versions["vivado"]
            java = versions["jre_tcl"]
            helpers: dict[str, str] = tcl_helpers
        else:
            if not entry.get("flow"):
                raise ValueError(f"LATCH recipe lacks flow: {recipe_id}")
            if entry["source_sha256"] != latch_launcher_sha:
                raise ValueError(f"LATCH launcher source hash mismatch for {recipe_id}")
            executable_source = {
                "flow": entry["flow"],
                "launcher_source_sha256": latch_launcher_sha,
            }
            tool = versions["latch"] + "; " + versions["vivado"]
            java = versions["jre_latch"]
            helpers = tcl_helpers
        payload: dict[str, object] = {
            "id": recipe_id,
            "kind": kind,
            "source": executable_source,
            "adapter_version": adapters[kind],
            "adapter_source_sha256": sha256_file(
                Path(__file__).resolve().parent / "adapters" / f"{kind}.py"
            ),
            "tool_version": tool,
            "java_version": java,
            "env_sha256": env_hash,
            "helper_sha256": helpers,
            "timeout_s": int(entry["timeout_s"]),
        }
        resolved[recipe_id] = RecipeSpec.model_validate(
            {**entry, "version": _version(payload)}
        )
    return Registry(
        root=project,
        recipes=resolved,
        catalogue_sha256=str(raw["source_catalogue_sha256"]),
        tool_versions=versions,
    )

