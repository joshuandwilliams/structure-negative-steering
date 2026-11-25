#!/usr/bin/env python3
"""Typed CLI entry point for writing the cross-sequence summary.

To the Nextflow process this is a drop-in replacement for
cross_summary.py.  It takes the same flags, --passing-summary-dir,
--published-runs-dir, --scored-metadata and --output, and writes a
byte-identical CSV.

The work goes through ``DesignCohort.emit_cross_summary_from_dirs`` in
design_cohort.py, which hands the column-level CSV construction to
``cross_summary.aggregate``.  Later refactors can therefore work
through the type hierarchy without changing the Nextflow process or
this CLI.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))  # pragma: no cover - import order

from design_cohort import DesignCohort  # noqa: E402

_SEQ_NAME_RE = re.compile(r"^[A-Za-z0-9_.+\-]+$")


def _parse_passing_summary_arg(
    arg: str, seq_name_mode: str
) -> Tuple[str, Path]:
    """Parse one --passing-summary argument into (seq_name, path).

    NAME=PATH gives the name directly.  Otherwise it comes from the
    path's parent or grandparent directory, as `seq_name_mode` selects.
    """
    if "=" in arg:
        name, path_str = arg.split("=", 1)
        return name, Path(path_str)
    path = Path(arg)
    if seq_name_mode == "grandparent":
        return path.parent.parent.name, path
    return path.parent.name, path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--passing-summary",
        action="append", default=[], metavar="[NAME=]PATH",
        help="Path to a per-sequence passing_summary.csv.",
    )
    parser.add_argument(
        "--passing-summary-dir",
        action="append", default=[], metavar="DIR",
        help="Directory with one subdir per MPNN sequence.",
    )
    parser.add_argument(
        "--seq-name-from-parent-dir",
        dest="seq_name_mode", action="store_const", const="parent",
        default="parent",
    )
    parser.add_argument(
        "--seq-name-from-grandparent-dir",
        dest="seq_name_mode", action="store_const", const="grandparent",
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
    )
    parser.add_argument(
        "--strict", action="store_true",
    )
    parser.add_argument(
        "--published-runs-dir", type=Path, default=None,
    )
    parser.add_argument(
        "--scored-metadata", type=Path, default=None,
    )
    args = parser.parse_args()

    if args.published_runs_dir is not None:
        args.published_runs_dir = args.published_runs_dir.resolve()

    explicit_pairs: List[Tuple[str, Path]] = []
    for raw in args.passing_summary:
        # _parse_passing_summary_arg only does a str.split and a
        # Path(), neither of which raises ValueError, so this handler
        # cannot fire today.  It stays because parsers gain validation.
        try:
            explicit_pairs.append(
                _parse_passing_summary_arg(raw, args.seq_name_mode)
            )
        except ValueError as e:  # pragma: no cover - no input reaches it
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    if not explicit_pairs and not args.passing_summary_dir:
        print(
            "ERROR: supply --passing-summary and/or --passing-summary-dir.",
            file=sys.stderr,
        )
        return 2

    return DesignCohort.emit_cross_summary_from_dirs(
        runs_dirs=args.passing_summary_dir,
        output_path=args.output,
        published_runs_dir=args.published_runs_dir,
        scored_metadata_path=args.scored_metadata,
        strict=args.strict,
        explicit_pairs=explicit_pairs if explicit_pairs else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
