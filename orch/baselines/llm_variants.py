"""Named, hashed LLM cells for the test campaign: F1-R1, F1-R0, E0-R1 and GA-R1.

A variant pairs a prompt wording (R0 or R1) with a configuration of the model file. Its hash covers
everything that defines the policy except the model: the cell definition, the template file and its
SHA-256, the configuration's mode, profiles and lessons settings and, when profiles are on, the profile
manifest. The model is one value of the model file and is frozen with that file. F2 is dropped and
cannot be selected. build_policy builds the policy exactly as the Saturday stage driver did.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from orch.baselines.general_agent import build_general_agent
from orch.policies.llm import LLMPolicy, LLMPolicyConfig
from orch.policies.llm_client import CachedLLM, TelemetrySink
from orch.policies.llm_prompt import load_template

VARIANTS_FILE = "configs/policies/llm_variants.yaml"


class VariantDropped(KeyError):
    """The named configuration was dropped and cannot be selected."""


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str                    # orchestrator | general
    configuration: str | None    # F1, E0 ... for an orchestrator, None for the general agent
    revision: str
    role: str
    template_path: str
    template_sha256: str
    mode: str
    profiles_on: bool
    lessons: bool
    profile_manifest_sha256: str | None
    sha256: str

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "configuration": self.configuration, "revision": self.revision,
            "template": {"path": self.template_path, "sha256": self.template_sha256},
            "mode": self.mode, "profiles_on": self.profiles_on, "lessons": self.lessons,
            "profile_manifest_sha256": self.profile_manifest_sha256,
        }


class VariantSet:
    def __init__(self, variants: Mapping[str, Variant], dropped: Mapping[str, str], file_sha256: str) -> None:
        self._variants, self._dropped = dict(variants), dict(dropped)
        self.file_sha256 = file_sha256
        self.sha256 = hashlib.sha256(_canonical(
            {"file": file_sha256, "variants": {n: v.sha256 for n, v in sorted(self._variants.items())}}
        ).encode("utf-8")).hexdigest()

    def names(self) -> list[str]:
        return list(self._variants)

    def get(self, name: str) -> Variant:
        head = name.split("-")[0]
        if name in self._dropped or head in self._dropped:
            raise VariantDropped(f"{name} is dropped: {self._dropped.get(name) or self._dropped.get(head)}")
        return self._variants[name]


def load_variants(project: Path | str, variants_file: str = VARIANTS_FILE) -> VariantSet:
    project = Path(project)
    path = project / variants_file
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "llm-variants-v1":
        raise ValueError("unknown variants file schema")
    model_cfg = yaml.safe_load((project / data["model_file"]).read_text(encoding="utf-8"))
    configurations = model_cfg["configurations"]
    manifest = project / model_cfg["profiles_dir"] / "manifest.json"
    resolved: dict[str, Variant] = {}
    for name, spec in data["variants"].items():
        revision = data["revisions"][spec["revision"]]
        if spec["kind"] == "orchestrator":
            conf = configurations[spec["configuration"]]
            if conf.get("dropped"):
                raise ValueError(f"variant {name} names the dropped configuration {spec['configuration']}")
            template_path, mode = revision["orchestrator"], conf["mode"]
            profiles_on, lessons, configuration = bool(conf["profiles"]), bool(conf["lessons"]), spec["configuration"]
        elif spec["kind"] == "general":
            template_path, mode, profiles_on, lessons, configuration = revision["general_agent"], "fuzzy", True, False, None
        else:
            raise ValueError(f"variant {name}: unknown kind {spec['kind']!r}")
        profile_manifest = _sha_file(manifest) if profiles_on else None
        template_sha = _sha_file(project / template_path)
        body = {
            "name": name, "kind": spec["kind"], "configuration": configuration, "revision": spec["revision"],
            "template": {"path": template_path, "sha256": template_sha}, "mode": mode,
            "profiles_on": profiles_on, "lessons": lessons, "profile_manifest_sha256": profile_manifest,
        }
        resolved[name] = Variant(
            name=name, kind=spec["kind"], configuration=configuration, revision=spec["revision"],
            role=str(spec.get("role", "")), template_path=template_path, template_sha256=template_sha, mode=mode,
            profiles_on=profiles_on, lessons=lessons, profile_manifest_sha256=profile_manifest,
            sha256=hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest(),
        )
    return VariantSet(resolved, data.get("dropped", {}), _sha_file(path))


def build_policy(variant: Variant, project: Path | str, model_cfg: Mapping[str, Any], llm: CachedLLM,
                 sink: TelemetrySink, worker_ids: Sequence[str]):
    """The policy of one episode, built as the Saturday stage driver built it."""

    project = Path(project)
    if variant.kind == "general":
        cfg = {**model_cfg, "templates": {**model_cfg["templates"], "general_agent": variant.template_path}}
        return build_general_agent(project, cfg, llm, sink, list(worker_ids))
    texts: dict[str, str] = {}
    hashes: dict[str, str] = {}
    if variant.profiles_on:
        profiles = project / model_cfg["profiles_dir"]
        manifest = json.loads((profiles / "manifest.json").read_text(encoding="utf-8"))
        for worker in worker_ids:
            texts[worker] = yaml.safe_load((profiles / f"{worker}.yaml").read_text(encoding="utf-8"))["text"]
            hashes[worker] = manifest["files"][f"{worker}.yaml"]
    return LLMPolicy(
        LLMPolicyConfig(
            name=variant.configuration, mode=variant.mode, profiles_on=variant.profiles_on,
            template=load_template(project / variant.template_path), profile_texts=texts, profile_sha256=hashes,
            lessons=None, max_repairs=model_cfg["max_repair_turns"],
        ), llm, sink,
    )
