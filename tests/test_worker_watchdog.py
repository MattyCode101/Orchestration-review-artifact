"""Safe watchdog cleanup is scoped to a campaign root and exact job tag."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

from orch.engine.jobtag import TAG_KEY, sweep_tag
from scripts import watch_worker_fitting_train as watchdog


def test_watchdog_cleans_only_dummy_campaign_tag(tmp_path: Path, monkeypatch) -> None:
    """A crashed-driver cleanup terminates its own group and leaves no tag."""

    root = tmp_path / "dummy_campaign"
    root.mkdir()
    monkeypatch.setattr(watchdog, "ROOT", root)
    monkeypatch.setattr(watchdog, "STATUS", root / "status.json")
    monkeypatch.setattr(watchdog, "AUDIT", root / "watchdog_cleanup_audit.json")
    tag = uuid.uuid4().hex
    env = os.environ.copy()
    env[TAG_KEY] = tag
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=root, env=env, start_new_session=True,
    )
    try:
        tags, groups = watchdog._targets()
        assert tag in tags
        assert child.pid in groups
        audit = watchdog._cleanup()
        child.wait(timeout=5)
        assert audit["exact_tags"] == [tag]
        assert audit["survivors"] == {}
        assert json.loads(watchdog.STATUS.read_text())["phase"] == "failed"
    finally:
        if child.poll() is None:
            sweep_tag(tag, grace_s=0.1)
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)

