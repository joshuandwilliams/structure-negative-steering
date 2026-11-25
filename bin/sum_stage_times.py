#!/usr/bin/env python3
"""Sum the per-stage seconds a run recorded into run_one_runtime_sec.txt.

The single SLURM job that ran every stage in sequence wrote its own
elapsed time. The stages are separate Nextflow tasks now, so each appends its
seconds to <run>/.stage_times and this sums them, the same quantity the serial
job measured, which the compute-cost analysis reads.

Wall-clock across the run differs, because tasks overlap. Summing keeps
the value comparable with the committed benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def total_seconds(stage_times: Path) -> int:
    """Add up the seconds in a .stage_times file.

    Each line is a stage name and its seconds, whitespace-separated.
    Other lines are skipped.  Missing files give 0.
    """
    if not stage_times.is_file():
        return 0
    total = 0
    for line in stage_times.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            total += int(parts[1])
    return total


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-times", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    args.output.write_text(f"{total_seconds(args.stage_times)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
