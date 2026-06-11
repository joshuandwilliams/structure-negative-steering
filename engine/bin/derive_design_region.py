#!/usr/bin/env python3
"""
derive_design_region.py
-----------------------
Extract the ProteinMPNN-designed residue positions for a given
RFDiffusion design from `rfdiffusion_metrics.json`, in the 1-based
positional-index coordinate system used by
`boltz2_iterate_steering.py compute-final-metrics
--design-region-positions-file`.

Background
==========
`rfdiffusion_metrics.json` stores per-design state with the following
relevant fields (one entry inside each element of the top-level
`designs` array):

  per_design_design_residues
      List of synthetic residue numbers (not real PDB resnums — these
      are offset from the max fixed resnum by 1000 to avoid collision
      with fixed residues).  These identify the de novo residues
      RFDiffusion built for this design.

  receptor_position_order
      The full list of receptor residue numbers in atom-walk order,
      i.e. the order they appear in the prediction PDB.  This is the
      same convention the steering pipeline uses to derive
      `true_interface_idx`.

What this script does
=====================
1. Loads rfdiffusion_metrics.json.
2. Finds the design entry matching the supplied name (accepting
   "design_3", "design_3.pdb", or the bare integer "3").
3. Intersects per_design_design_residues with receptor_position_order:
   each synthetic resnum in the design-residue list is looked up in
   receptor_position_order to get its 0-based positional index.
4. Converts 0-based indices to 1-based (compute-final-metrics's
   convention) and writes them to the output file.

Output format
=============
A text file consumable by compute-final-metrics's
--design-region-positions-file flag.  Format:

  # derived from <path to rfdiffusion_metrics.json>
  # design: design_3
  # source field: per_design_design_residues (N residues)
  # coordinate system: 1-based positional indices on the receptor
  #                    chain, matching prediction PDB sequence order
  <pos1>,<pos2>,<pos3>,...

Usage
=====
  python3 derive_design_region.py \\
      --metrics-json /path/to/rfdiffusion_metrics.json \\
      --design design_3 \\
      --output design_3_design_region.txt

  # Optionally cross-check against the true_interface_idx that
  # plan.json already has for this design (catches coordinate-
  # system mismatches):
  python3 derive_design_region.py \\
      --metrics-json ... --design design_3 --output ... \\
      --cross-check-plan /path/to/cycle_0/plan.json

On success, exits 0 and prints a summary to stderr.  On any error
(missing file, design not found, empty result, cross-check mismatch)
exits non-zero with a descriptive message.
"""

import argparse
import json
import sys
from pathlib import Path


def normalise_design_id(raw):
    """Accept 'design_3', 'design_3.pdb', or '3'; return all three
    forms so we can match against whatever key rfdiffusion_metrics.json
    happens to use."""
    raw = str(raw).strip()
    # Strip .pdb if present
    if raw.endswith(".pdb"):
        stem = raw[:-4]
    else:
        stem = raw
    # If it's a bare integer, prepend "design_"
    if stem.isdigit():
        stem = f"design_{stem}"
    return {
        "bare": stem,                 # "design_3"
        "with_pdb": f"{stem}.pdb",    # "design_3.pdb"
    }


def find_design_entry(metrics, design_id):
    """Locate the design record inside metrics['designs'] matching
    `design_id`.  Returns the entry dict, or raises KeyError with a
    listing of available designs if not found."""
    designs = metrics.get("designs")
    if not isinstance(designs, list):
        raise ValueError(
            "rfdiffusion_metrics.json has no top-level 'designs' array "
            "(got type %s)" % type(designs).__name__
        )
    ids = normalise_design_id(design_id)
    # The JSON typically stores the design identifier under the key
    # "design".  Accept both "design_3" and "design_3.pdb" forms.
    for entry in designs:
        key = entry.get("design") or entry.get("name") or ""
        key = str(key).strip()
        if key in (ids["bare"], ids["with_pdb"]):
            return entry
        # Also accept "design_3" matching a stored "design_3.pdb"
        # and vice versa via stem comparison.
        if key.endswith(".pdb") and key[:-4] == ids["bare"]:
            return entry
    # Not found — build a helpful error message listing what IS there.
    available = sorted(
        str(e.get("design") or e.get("name") or "?") for e in designs
    )
    preview = ", ".join(available[:10])
    if len(available) > 10:
        preview += f", ... ({len(available)} total)"
    raise KeyError(
        f"No design entry matching {design_id!r} "
        f"(tried {ids['bare']!r} and {ids['with_pdb']!r}). "
        f"Available designs: {preview}"
    )


def derive_positional_indices(entry):
    """Return the list of 0-based positional indices corresponding
    to the design region of `entry`, by intersecting
    per_design_design_residues with receptor_position_order.

    This is the exact transform from the user's notes on
    rfdiffusion_metrics.json:

        design_region_indices = [
            i for i, rn in enumerate(entry["receptor_position_order"])
            if rn in set(entry["per_design_design_residues"])
        ]

    Returns indices (0-based) sorted ascending.  Raises ValueError
    if either field is missing or empty.
    """
    design_residues = entry.get("per_design_design_residues")
    position_order = entry.get("receptor_position_order")
    if design_residues is None:
        raise ValueError(
            "design entry has no 'per_design_design_residues' field"
        )
    if position_order is None:
        raise ValueError(
            "design entry has no 'receptor_position_order' field"
        )
    if not design_residues:
        raise ValueError(
            "'per_design_design_residues' is empty — design region "
            "has no residues to extract"
        )
    if not position_order:
        raise ValueError(
            "'receptor_position_order' is empty — cannot derive "
            "positional indices"
        )

    design_set = set(design_residues)
    indices = [
        i for i, rn in enumerate(position_order) if rn in design_set
    ]

    # Sanity: every synthetic resnum in design_residues should be
    # present in position_order.  If any are missing, the two lists
    # are describing different atom walks and the derivation is
    # suspect — fail loudly.
    found_resnums = {position_order[i] for i in indices}
    missing = design_set - found_resnums
    if missing:
        raise ValueError(
            f"{len(missing)} synthetic resnum(s) from "
            f"per_design_design_residues were not found in "
            f"receptor_position_order: {sorted(missing)[:20]}"
            f"{'...' if len(missing) > 20 else ''}. The two fields "
            f"may describe different atom walks; derivation aborted."
        )

    return sorted(indices)


def cross_check_with_plan(plan_path, design_indices_0based):
    """Cross-check the derived design region against the
    `true_interface_idx` field in `plan.json`.  Reports whether every
    residue in true_interface_idx is inside the derived design region
    (which it MUST be on a receptor-resurfacing design — the true
    interface is a subset of the design region by construction).

    Prints a warning if the true interface is not a subset, which
    would indicate either a coordinate-system mismatch (bug in this
    script) or a genuinely non-receptor-resurfacing design where the
    true interface extends outside the de novo block (legal but
    unusual — worth surfacing to the user).

    Returns True if the check passes, False otherwise.  Does not
    raise — cross-check is advisory, not gating.
    """
    plan_path = Path(plan_path)
    if not plan_path.exists():
        print(f"  cross-check: plan.json not found at {plan_path}, "
              f"skipping", file=sys.stderr)
        return True
    try:
        plan = json.loads(plan_path.read_text())
    except Exception as e:
        print(f"  cross-check: failed to read plan.json: {e}",
              file=sys.stderr)
        return True

    true_idx = plan.get("true_interface_idx")
    if not true_idx:
        print("  cross-check: plan.json has no "
              "true_interface_idx field; skipping", file=sys.stderr)
        return True

    design_set = set(design_indices_0based)
    true_set = set(int(i) for i in true_idx)
    outside = sorted(true_set - design_set)
    inside = sorted(true_set & design_set)
    print(f"  cross-check: plan.json true_interface_idx has "
          f"{len(true_set)} residues (0-based)", file=sys.stderr)
    print(f"               {len(inside)} inside derived design region, "
          f"{len(outside)} outside", file=sys.stderr)
    if outside:
        print(f"  WARNING: true interface has residues OUTSIDE the "
              f"derived design region: {outside}", file=sys.stderr)
        print("           This may indicate a coordinate-system "
              "mismatch between rfdiffusion_metrics.json and "
              "plan.json, or a design whose native binding site "
              "extends outside the de novo block.", file=sys.stderr)
        return False
    print("               OK: true interface is a subset of the "
          "derived design region.", file=sys.stderr)
    return True


def format_output(indices_1based, metrics_path, design_id, ranges):
    """Build the output file contents.  `ranges=True` collapses
    contiguous runs into ranges ("30-78"), False writes a plain
    comma list."""
    header_lines = [
        f"# derived from {metrics_path}",
        f"# design: {design_id}",
        f"# source field: per_design_design_residues "
        f"({len(indices_1based)} residues)",
        "# coordinate system: 1-based positional indices on the "
        "receptor chain,",
        "#                    matching prediction PDB sequence order",
    ]
    if not ranges:
        body = ",".join(str(p) for p in indices_1based)
    else:
        # Collapse contiguous runs: [30,31,32,35,36,37] -> "30-32,35-37"
        out_parts = []
        if indices_1based:
            start = indices_1based[0]
            prev = start
            for p in indices_1based[1:]:
                if p == prev + 1:
                    prev = p
                    continue
                if start == prev:
                    out_parts.append(str(start))
                else:
                    out_parts.append(f"{start}-{prev}")
                start = p
                prev = p
            if start == prev:
                out_parts.append(str(start))
            else:
                out_parts.append(f"{start}-{prev}")
        body = ",".join(out_parts)
    return "\n".join(header_lines) + "\n" + body + "\n"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--metrics-json", type=Path, required=True,
                        help="Path to rfdiffusion_metrics.json")
    parser.add_argument("--design", required=True,
                        help="Design identifier, e.g. 'design_3', "
                             "'design_3.pdb', or '3'")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output text file path.  Existing files "
                             "are overwritten.")
    parser.add_argument("--no-ranges", action="store_true",
                        help="Emit a plain comma-separated list "
                             "instead of collapsed ranges.  Default "
                             "is to collapse contiguous runs into "
                             "'start-end' ranges for readability.")
    parser.add_argument("--cross-check-plan", type=Path, default=None,
                        help="Optional path to cycle_0/plan.json.  "
                             "When supplied, verifies that the "
                             "derived design region is a superset "
                             "of plan.json's true_interface_idx.  "
                             "Advisory only: a mismatch prints a "
                             "warning but does not change the output.")
    args = parser.parse_args()

    if not args.metrics_json.exists():
        print(f"ERROR: metrics JSON not found: {args.metrics_json}",
              file=sys.stderr)
        return 2
    try:
        metrics = json.loads(args.metrics_json.read_text())
    except Exception as e:
        print(f"ERROR: failed to parse {args.metrics_json}: {e}",
              file=sys.stderr)
        return 2

    try:
        entry = find_design_entry(metrics, args.design)
    except (KeyError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    try:
        indices_0based = derive_positional_indices(entry)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if not indices_0based:
        print("ERROR: derivation produced an empty index list",
              file=sys.stderr)
        return 2

    # Print a summary
    indices_1based = [i + 1 for i in indices_0based]
    n = len(indices_1based)
    lo, hi = indices_1based[0], indices_1based[-1]
    span = hi - lo + 1
    contiguous = (n == span)
    print(f"  design: {args.design}", file=sys.stderr)
    print(f"  per_design_design_residues: "
          f"{len(entry.get('per_design_design_residues') or [])} residues",
          file=sys.stderr)
    print(f"  receptor_position_order length: "
          f"{len(entry.get('receptor_position_order') or [])}",
          file=sys.stderr)
    print(f"  derived design region: {n} positions "
          f"(1-based range {lo}..{hi}, "
          f"{'contiguous' if contiguous else 'with gaps'})",
          file=sys.stderr)

    # Cross-check if asked
    if args.cross_check_plan is not None:
        cross_check_with_plan(
            args.cross_check_plan, indices_0based
        )

    # Write output
    text = format_output(
        indices_1based,
        args.metrics_json,
        args.design,
        ranges=(not args.no_ranges),
    )
    args.output.write_text(text)
    print(f"  wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
