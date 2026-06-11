#!/usr/bin/env python3
"""Vendor the negative-steering engine from receptor-resurfacing-pipeline.

The pipeline is the single source of truth for the negative-steering engine.
This script computes the *import closure* of the negative-steering entry points
inside the pipeline's ``bin/`` (so a newly added sibling import is picked up
automatically), copies those Python files plus the Nextflow module and the
shell orchestrator into ``engine/``, and records the upstream git commit and a
per-file sha256 in ``engine/_UPSTREAM.json``.

Nothing in the pipeline is ever modified — this is strictly a pull, pipeline
-> standalone.

Usage:
  scripts/sync_from_pipeline.py                 # vendor / refresh engine/
  scripts/sync_from_pipeline.py --check         # report drift vs upstream; no writes
  scripts/sync_from_pipeline.py --pipeline PATH # override pipeline location

Exit codes:
  0  in sync (or sync written successfully)
  1  --check found the vendored engine is behind / has drifted from upstream
  2  usage / environment error (pipeline or a seed file not found)
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# --- Configuration -------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = REPO_ROOT / "engine"
MANIFEST_PATH = ENGINE_DIR / "_UPSTREAM.json"

DEFAULT_PIPELINE = Path.home() / "Documents" / "GitHub" / "receptor-resurfacing-pipeline"

# Python entry points the Nextflow module / orchestrator invoke directly.
# Their sibling-import closure within bin/ is computed automatically, so this
# list only needs the scripts that are *called*, not every helper they import.
PY_SEEDS: List[str] = [
    "boltz2_negative_steering.py",
    "boltz2_iterate_steering.py",
    "negative_steering_run.py",
    "extract_passing.py",
    "derive_design_region.py",
    "derive_true_interface.py",
    "cross_summary_v2.py",
    "negsteer_plots.py",
    "negsteer_within_sequence_plots.py",
]

# Non-Python files copied verbatim. (relative-dir, filename) under the pipeline
# root; mirrored at the same relative path under engine/.
VERBATIM_SEEDS: List[Tuple[str, str]] = [
    ("bin", "negative_steering_run_one.sh"),
    ("modules", "negative_steering.nf"),
]


# --- Helpers -------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def git_commit(pipeline: Path) -> Tuple[str, bool]:
    """Return (commit_sha, dirty) for the pipeline working tree."""
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(pipeline), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(pipeline), "status", "--porcelain"],
            text=True,
        )
        return sha, bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", False


def git_remote(pipeline: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(pipeline), "remote", "get-url", "origin"],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def sibling_imports(py_path: Path) -> Set[str]:
    """Module names imported by ``py_path`` (top-level names only).

    Walks the full AST so imports nested inside functions / conditionals are
    captured too. Returns bare top-level names (e.g. ``contig_utils``), which
    the caller resolves against the pipeline bin/ directory.
    """
    names: Set[str] = set()
    tree = ast.parse(py_path.read_text(), filename=str(py_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # Only absolute, sibling-style imports resolve against bin/.
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def compute_closure(bin_dir: Path) -> List[str]:
    """Return the sorted list of bin/*.py filenames in the engine closure."""
    seen: Set[str] = set()
    queue: List[str] = list(PY_SEEDS)
    while queue:
        fname = queue.pop()
        if fname in seen:
            continue
        path = bin_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"seed/closure file missing in pipeline bin/: {fname}")
        seen.add(fname)
        for mod in sibling_imports(path):
            candidate = f"{mod}.py"
            if (bin_dir / candidate).exists() and candidate not in seen:
                queue.append(candidate)
    return sorted(seen)


def build_plan(pipeline: Path) -> Dict[str, str]:
    """Map engine-relative path -> source absolute path hash, for the full set.

    Returns ``{rel_path: sha256}`` where rel_path is relative to engine/
    (e.g. ``bin/contig_utils.py``, ``modules/negative_steering.nf``).
    """
    bin_dir = pipeline / "bin"
    if not bin_dir.is_dir():
        raise FileNotFoundError(f"pipeline bin/ not found at {bin_dir}")

    plan: Dict[str, Path] = {}
    for fname in compute_closure(bin_dir):
        plan[f"bin/{fname}"] = bin_dir / fname
    for subdir, fname in VERBATIM_SEEDS:
        src = pipeline / subdir / fname
        if not src.exists():
            raise FileNotFoundError(f"verbatim seed missing in pipeline: {subdir}/{fname}")
        plan[f"{subdir}/{fname}"] = src
    return plan


# --- Commands ------------------------------------------------------------

def do_sync(pipeline: Path) -> int:
    plan = build_plan(pipeline)
    sha, dirty = git_commit(pipeline)

    synced: Dict[str, str] = {}
    for rel, src in sorted(plan.items()):
        dest = ENGINE_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        synced[rel] = sha256_file(dest)

    manifest = {
        "_comment": "Auto-generated by scripts/sync_from_pipeline.py. Do not edit by hand. "
                    "Files under engine/ are VENDORED from the pipeline; edit them there, not here.",
        "upstream_remote": git_remote(pipeline),
        "upstream_path": str(pipeline),
        "upstream_commit": sha,
        "upstream_dirty": dirty,
        "synced_files": synced,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"Vendored {len(synced)} files into {ENGINE_DIR.relative_to(REPO_ROOT)}/ "
          f"from {pipeline} @ {sha[:12]}{' (dirty)' if dirty else ''}")
    for rel in sorted(synced):
        print(f"  + {rel}")
    if dirty:
        print("WARNING: pipeline working tree is dirty; vendored copy may not match a clean commit.")
    return 0


def do_check(pipeline: Path) -> int:
    if not MANIFEST_PATH.exists():
        print("No engine/_UPSTREAM.json — engine has never been synced. Run without --check.")
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text())
    recorded: Dict[str, str] = manifest.get("synced_files", {})

    plan = build_plan(pipeline)
    current = {rel: sha256_file(src) for rel, src in plan.items()}

    added = sorted(set(current) - set(recorded))
    removed = sorted(set(recorded) - set(current))
    changed = sorted(r for r in set(current) & set(recorded) if current[r] != recorded[r])

    cur_sha, dirty = git_commit(pipeline)
    rec_sha = manifest.get("upstream_commit", "unknown")

    if not (added or removed or changed):
        print(f"engine/ is in sync with {pipeline} "
              f"(upstream @ {cur_sha[:12]}{' dirty' if dirty else ''}).")
        return 0

    print(f"engine/ is BEHIND upstream {pipeline}")
    print(f"  recorded commit: {rec_sha[:12]}   current commit: {cur_sha[:12]}"
          f"{' (dirty)' if dirty else ''}")
    for rel in changed:
        print(f"  ~ changed: {rel}")
    for rel in added:
        print(f"  + new:     {rel}")
    for rel in removed:
        print(f"  - gone:    {rel}")
    print("\nRun  scripts/sync_from_pipeline.py  to refresh the vendored engine.")
    return 1


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE,
                    help=f"path to the receptor-resurfacing-pipeline clone "
                         f"(default: {DEFAULT_PIPELINE})")
    ap.add_argument("--check", action="store_true",
                    help="report whether the vendored engine is behind upstream; write nothing")
    args = ap.parse_args(argv)

    pipeline = args.pipeline.expanduser().resolve()
    if not pipeline.is_dir():
        print(f"ERROR: pipeline clone not found at {pipeline}", file=sys.stderr)
        return 2

    try:
        return do_check(pipeline) if args.check else do_sync(pipeline)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
