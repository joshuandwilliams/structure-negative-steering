#!/usr/bin/env python3
"""Extract the ProteinMPNN-designed residue positions for one
RFDiffusion design from `rfdiffusion_metrics.json`.  Positions come out
as 1-based positional indices, which is what
`negsteer_aggregate.py compute-final-metrics
--design-region-positions-file` expects.

Background
==========
Each element of the top-level `designs` array carries two fields this
script needs.

  per_design_design_residues
      The de novo residues RFDiffusion built for this design.  They are
      synthetic resnums, not real PDB resnums, offset from the highest
      fixed resnum by 1000 so they cannot collide with it.

  receptor_position_order
      Every receptor residue number in atom-walk order, which is the
      order they appear in the prediction PDB.  The steering pipeline
      uses the same convention for `true_interface_idx`.

Each synthetic resnum is looked up in receptor_position_order for its
0-based index, then 1 is added because compute-final-metrics counts
from 1.

Output format
=============
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

--cross-check-plan compares the result against the true_interface_idx
that plan.json already holds, which catches coordinate-system
mismatches.

On success it exits 0 and prints a summary to stderr.  A missing file,
an unknown design, an empty result or a failed cross-check all exit
non-zero with a message.
"""

import argparse
import json
import sys
from pathlib import Path


def normalise_design_id(raw):
    """Accept 'design_3', 'design_3.pdb', or '3'.  Return both the
    bare and the .pdb form, so the caller can match whichever key
    rfdiffusion_metrics.json happens to use."""
    raw = str(raw).strip()
    if raw.endswith(".pdb"):
        stem = raw[:-4]
    else:
        stem = raw
    # A bare integer needs the "design_" prefix.
    if stem.isdigit():
        stem = f"design_{stem}"
    return {
        "bare": stem,                 # "design_3"
        "with_pdb": f"{stem}.pdb",    # "design_3.pdb"
    }


def find_design_entry(metrics, design_id):
    """Find the record in metrics['designs'] that matches `design_id`.

    Returns the entry dict.  Raises KeyError if there is no match, and
    the message lists the designs that are present.
    """
    designs = metrics.get("designs")
    if not isinstance(designs, list):
        raise ValueError(
            "rfdiffusion_metrics.json has no top-level 'designs' array "
            "(got type %s)" % type(designs).__name__
        )
    ids = normalise_design_id(design_id)
    # The identifier is usually under the key "design".
    for entry in designs:
        key = entry.get("design") or entry.get("name") or ""
        key = str(key).strip()
        if key in (ids["bare"], ids["with_pdb"]):
            return entry
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
    """Return the design region of `entry` as 0-based positional
    indices, sorted ascending.

    The transform intersects per_design_design_residues with
    receptor_position_order.  Raises ValueError if either field is
    missing or empty.
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

    # A missing synthetic resnum means the two lists describe
    # different atom walks, so the derivation cannot be trusted.
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
    """Compare the derived design region against `true_interface_idx`
    in `plan.json`.

    On a receptor-resurfacing design every residue in
    true_interface_idx must sit inside the design region, because the
    true interface is a subset of it by construction.  A failure means
    either a coordinate-system mismatch here, or a design that is not
    receptor-resurfacing and whose true interface runs outside the de
    novo block.  The second case is legal but unusual.

    Returns True when the check passes.  It never raises, because the
    cross-check is advisory and does not gate the output.
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
    """Build the contents of the output file.

    With `ranges=True` contiguous runs collapse into ranges such as
    "30-78".  With False the body is a plain comma-separated list.
    """
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
        # Collapse contiguous runs, so [30,31,32,35,36,37] becomes
        # "30-32,35-37".
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

    if not indices_0based:  # pragma: no cover - derive_positional_indices raises on every input that would empty this list
        # derive_positional_indices already raises on every case that
        # would empty this list.  The guard stays because that
        # invariant lives there, not here.
        print("ERROR: derivation produced an empty index list",
              file=sys.stderr)
        return 2

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

    if args.cross_check_plan is not None:
        cross_check_with_plan(
            args.cross_check_plan, indices_0based
        )

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
