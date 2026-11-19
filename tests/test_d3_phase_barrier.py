"""The campaign phase callback must wait on the existing progress lock."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from orch.engine.store import _file_lock


_CHILD = r"""
import os
from pathlib import Path
from scripts import launch_worker_fitting_train as launch

launch.DATA_ROOT = Path(os.environ['D3_BARRIER_TEST_DATA'])
launch.PROGRESS_NOTE = Path(os.environ['D3_BARRIER_TEST_NOTE'])
launch._phase_callback({
    'phase': 'D3',
    'at_utc': '2026-11-04T18:00:00+00:00',
    'summary': {
        'planned_sequences': 1,
        'attempted_edges': 1,
        'physical_valid_edges': 1,
        'cache_hits': 0,
        'new_backend_edges': 1,
        'statuses': {'ok': 1},
    },
})
"""


def test_d3_callback_blocks_before_d4_when_progress_lock_is_held(tmp_path: Path) -> None:
    """The D3 report is durable while its callback waits for the lock."""
    data_root = tmp_path / 'data'
    progress_note = tmp_path / 'notes' / 'progress.md'
    progress_note.parent.mkdir(parents=True)
    progress_note.write_text('# Progress\n', encoding='utf-8')
    report = data_root / 'phase_reports' / 'D3.json'
    env = os.environ.copy()
    env['D3_BARRIER_TEST_DATA'] = str(data_root)
    env['D3_BARRIER_TEST_NOTE'] = str(progress_note)
    process = None
    try:
        with _file_lock(progress_note.with_suffix('.lock')):
            process = subprocess.Popen(
                [sys.executable, '-c', _CHILD],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + 10
            while not report.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert report.exists(), 'D3 report was not written before the progress lock'
            assert process.poll() is None, 'callback did not block on the held lock'
            assert json.loads(report.read_text(encoding='utf-8'))['phase'] == 'D3'
            assert progress_note.read_text(encoding='utf-8') == '# Progress\n'
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, (stdout, stderr)
        assert 'D3 complete' in progress_note.read_text(encoding='utf-8')
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)