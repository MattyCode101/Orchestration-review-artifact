"""Decision policy boundary: a view goes in, a mandate or stop comes out."""

from __future__ import annotations

from typing import Protocol

from orch.contract.models import Mandate
from orch.contract.views import PolicyView


class Policy(Protocol):
    id: str
    version: str

    def decide(self, view: PolicyView) -> Mandate | None:
        """Select a worker, base state and limits without naming a recipe."""

