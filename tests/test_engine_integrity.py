"""Integrity of the vendored negative-steering engine.

These tests run *anywhere* (laptop, HPC, CI) — they need only the files under
``engine/`` and the recorded manifest, never the upstream pipeline clone.

They guard that nobody has hand-edited a vendored file: each file's hash must
still match what scripts/sync_from_pipeline.py recorded in _UPSTREAM.json.
Import-closure (did we vendor everything the engine imports?) is verified
against the live pipeline by tests/test_engine_staleness.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = REPO_ROOT / "engine"
MANIFEST_PATH = ENGINE_DIR / "_UPSTREAM.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST_PATH.exists(), (
        "engine/_UPSTREAM.json is missing — run scripts/sync_from_pipeline.py"
    )
    return json.loads(MANIFEST_PATH.read_text())


@pytest.mark.local_unit
def test_manifest_lists_files(manifest: dict) -> None:
    assert manifest.get("synced_files"), "manifest records no synced files"


@pytest.mark.local_unit
def test_vendored_files_match_manifest(manifest: dict) -> None:
    """Every vendored file is present and unmodified since the last sync."""
    mismatches = []
    for rel, expected in sorted(manifest["synced_files"].items()):
        path = ENGINE_DIR / rel
        if not path.exists():
            mismatches.append(f"missing:  {rel}")
        elif _sha256(path) != expected:
            mismatches.append(f"modified: {rel}")
    assert not mismatches, (
        "Vendored engine files differ from _UPSTREAM.json. Do NOT hand-edit "
        "engine/ — fix it upstream in the pipeline and re-run "
        "scripts/sync_from_pipeline.py.\n  " + "\n  ".join(mismatches)
    )


@pytest.mark.local_unit
def test_no_stray_files_under_engine(manifest: dict) -> None:
    """engine/ contains only the manifest plus exactly the vendored files."""
    tracked = set(manifest["synced_files"]) | {"_UPSTREAM.json"}
    actual = {
        str(p.relative_to(ENGINE_DIR))
        for p in ENGINE_DIR.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    stray = sorted(actual - tracked)
    assert not stray, (
        "Unexpected files under engine/ (not vendored by sync_from_pipeline.py):"
        "\n  " + "\n  ".join(stray)
    )
