"""Reserved boundary for worker B's design-feature queries.

UNTESTED PLACEHOLDER. No Tcl query is written here because the required
queries have not been verified from run-3/run-4 source and a live tool run.
"""

from __future__ import annotations

from pathlib import Path

from orch.contract.models import DesignState


class FeaturesAdapter:
    """Explicitly unavailable until feature queries are sourced and exercised."""

    kind = "features"
    version = "features-placeholder-v0"

    def query(self, state: DesignState, *, run_dir: Path) -> dict[str, object]:
        raise NotImplementedError(
            "worker-B design features require sourced Tcl and a live verification"
        )

