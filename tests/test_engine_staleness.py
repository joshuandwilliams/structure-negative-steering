"""Is the vendored engine still in sync with the upstream pipeline?

This test needs the receptor-resurfacing-pipeline clone. Where that clone is
absent (CI, the HPC, a fresh laptop) the test SKIPS rather than fails — it is a
convenience guard for the machine where you actually develop, not a hard gate.

It shells out to scripts/sync_from_pipeline.py --check, which recomputes the
import closure from the live pipeline and compares it (files + hashes + commit)
to engine/_UPSTREAM.json. A non-zero exit means the engine is behind upstream;
the captured output names exactly which files drifted.

Override the pipeline location with the NEGSTEER_PIPELINE env var.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNC_TOOL = REPO_ROOT / "scripts" / "sync_from_pipeline.py"
DEFAULT_PIPELINE = Path.home() / "Documents" / "GitHub" / "receptor-resurfacing-pipeline"


def _pipeline_path() -> Path:
    return Path(os.environ.get("NEGSTEER_PIPELINE", str(DEFAULT_PIPELINE))).expanduser()


@pytest.mark.local_unit
def test_engine_in_sync_with_pipeline() -> None:
    pipeline = _pipeline_path()
    if not pipeline.is_dir():
        pytest.skip(
            f"pipeline clone not found at {pipeline} "
            f"(set NEGSTEER_PIPELINE to run this drift check)"
        )
    result = subprocess.run(
        [sys.executable, str(SYNC_TOOL), "--check", "--pipeline", str(pipeline)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "Vendored engine is behind the pipeline. Run "
        "scripts/sync_from_pipeline.py to refresh, then commit.\n\n"
        + result.stdout
        + result.stderr
    )
