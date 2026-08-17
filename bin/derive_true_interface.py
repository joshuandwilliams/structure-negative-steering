#!/usr/bin/env python3
"""
derive_true_interface.py
------------------------
Extract the receptor residues that contact the effector in an
RFDiffusion design, in the 0-based positional-index coordinate system
used by ``boltz2_negative_steering.py``'s
``--true-interface-indices-file`` flag.

Why this script exists
======================
``boltz2_negative_steering.py plan`` normally finds the "true
interface" by running its own heavy-atom contact detection on the
ground-truth PDB.  That does not work for RFDiffusion designs: the
designed receptor chain is Cα-only, so heavy-atom contacts cannot be
computed.  For those inputs, the contact set has to be pre-computed
and passed in via ``--true-interface-indices-file``.

``RFDIFFUSION_FILTER`` already did the contact detection (using Cα-Cα
distances with a tunable cutoff) and wrote every design's contact
residues into ``rfdiffusion_metrics.json`` under
``receptor_contact_residues``, alongside the
``receptor_position_order`` list that ``derive_design_region.py``
uses as its index key.  This script intersects the two in exactly the
same way — producing a text file consumable by
``boltz2_negative_steering.py`` — so the two protected sets stay in a
consistent coordinate system.

Output format
=============
One integer per line (0-based positional index on the receptor chain,
matching prediction PDB sequence order).  The
``boltz2_negative_steering.py`` parser also accepts comma- or
whitespace-separated values and ``#`` line comments, but one-per-line
is the most diffable.  A provenance header is prepended as
``#``-prefixed comment lines.

Usage
=====
    python3 derive_true_interface.py \\
        --metrics-json /path/to/rfdiffusion_metrics.json \\
        --design design_3 \\
        --output design_3_true_interface.txt

Cross-check
===========
Optionally pass ``--cross-check-design-region <file>`` to verify that
the derived interface is a subset of the previously-computed
design-region file (should hold for receptor-resurfacing designs
where the true binding site is engineered inside the de novo region).
Advisory only — prints a warning on mismatch, does not change the
output.
"""

import argparse
import json
import sys
from pathlib import Path


def normalise_design_id(raw):
    """Accept 'design_3', 'design_3.pdb', or '3'; return both
    canonical forms so we can match against whatever key
    rfdiffusion_metrics.json happens to use."""
    raw = str(raw).strip()
    if raw.endswith(".pdb"):
        stem = raw[:-4]
    else:
        stem = raw
    if stem.isdigit():
        stem = f"design_{stem}"
    return {"bare": stem, "with_pdb": f"{stem}.pdb"}


def find_design_entry(metrics, design_id):
    designs = metrics.get("designs")
    if not isinstance(designs, list):
        raise ValueError(
            "rfdiffusion_metrics.json has no top-level 'designs' array "
            "(got type %s)" % type(designs).__name__
        )
    ids = normalise_design_id(design_id)
    for entry in designs:
        key = entry.get("design") or entry.get("name") or ""
        key = str(key).strip()
        if key in (ids["bare"], ids["with_pdb"]):
            return entry
        if key.endswith(".pdb") and key[:-4] == ids["bare"]:
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
    """Intersect ``receptor_contact_residues`` with
    ``receptor_position_order``; return 0-based positional indices
    sorted ascending.

    This mirrors the transform in ``derive_design_region.py`` exactly,
    the only difference being the source field
    (``receptor_contact_residues`` here,
    ``per_design_design_residues`` there).
    """
    contact_residues = entry.get("receptor_contact_residues")
    position_order = entry.get("receptor_position_order")
    if contact_residues is None:
        raise ValueError(
            "design entry has no 'receptor_contact_residues' field"
        )
    if position_order is None:
        raise ValueError(
            "design entry has no 'receptor_position_order' field"
        )
    if not contact_residues:
        raise ValueError(
            "'receptor_contact_residues' is empty — the design has no "
            "Cα contacts with the effector.  Either the contact cutoff "
            "was too strict or the design is not a binder."
        )
    if not position_order:
        raise ValueError(
            "'receptor_position_order' is empty — cannot derive "
            "positional indices"
        )

    contact_set = set(contact_residues)
    indices = [
        i for i, rn in enumerate(position_order) if rn in contact_set
    ]

    # Sanity: every resnum in receptor_contact_residues should be
    # present in receptor_position_order.  If any are missing, the
    # two lists describe different atom walks and the derivation is
    # suspect — fail loudly.
    found_resnums = {position_order[i] for i in indices}
    missing = contact_set - found_resnums
    if missing:
        raise ValueError(
            f"{len(missing)} resnum(s) from receptor_contact_residues "
            f"were not found in receptor_position_order: "
            f"{sorted(missing)[:20]}"
            f"{'...' if len(missing) > 20 else ''}. "
            f"The two fields may describe different atom walks; "
            f"derivation aborted."
        )

    return sorted(indices)


def cross_check_design_region(design_region_path, interface_indices_0based):
    """Verify that the derived true interface is a subset of the
    design region file (RFDiffusion + MPNN resurfacing pattern:
    binding site is inside the de novo block).

    Reads any line not starting with ``#`` and treats ``,``-separated
    integers and ``start-end`` ranges as 1-based positions (the
    convention ``derive_design_region.py`` writes).  Advisory only:
    prints a warning on mismatch, does not raise.
    """
    path = Path(design_region_path)
    if not path.exists():
        print(f"  cross-check: design-region file not found at "
              f"{path}, skipping", file=sys.stderr)
        return True
    try:
        text = path.read_text()
    except Exception as e:
        print(f"  cross-check: failed to read {path}: {e}",
              file=sys.stderr)
        return True

    region_1based = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for token in line.replace(",", " ").split():
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                try:
                    lo, hi = token.split("-", 1)
                    lo = int(lo)
                    hi = int(hi)
                except ValueError:
                    continue
                for p in range(lo, hi + 1):
                    region_1based.add(p)
            else:
                try:
                    region_1based.add(int(token))
                except ValueError:
                    continue

    if not region_1based:
        print(f"  cross-check: design-region file {path} contained no "
              f"parseable positions; skipping", file=sys.stderr)
        return True

    # Interface indices here are 0-based; design-region file is
    # 1-based.  Bring interface into 1-based for comparison.
    interface_1based = {i + 1 for i in interface_indices_0based}
    outside = sorted(interface_1based - region_1based)
    inside = sorted(interface_1based & region_1based)
    print(f"  cross-check: design region has {len(region_1based)} "
          f"positions (1-based)", file=sys.stderr)
    print(f"               true interface {len(interface_1based)} "
          f"positions; {len(inside)} inside design region, "
          f"{len(outside)} outside", file=sys.stderr)
    if outside:
        print(f"  WARNING: true interface has residues OUTSIDE the "
              f"design region: {outside}", file=sys.stderr)
        print("           Expected a subset relationship for "
              "receptor-resurfacing designs.  Either the contact "
              "cutoff is set such that contacts extend into fixed "
              "residues, or the inputs are mismatched.",
              file=sys.stderr)
        return False
    print("               OK: true interface is a subset of the "
          "design region.", file=sys.stderr)
    return True


def format_output(indices_0based, metrics_path, design_id, n_contact_resnums):
    header_lines = [
        f"# derived from {metrics_path}",
        f"# design: {design_id}",
        f"# source field: receptor_contact_residues "
        f"({n_contact_resnums} residues)",
        "# coordinate system: 0-based positional indices on the "
        "receptor chain,",
        "#                    matching prediction PDB sequence order",
    ]
    body = "\n".join(str(i) for i in indices_0based)
    return "\n".join(header_lines) + "\n" + body + "\n"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--metrics-json", type=Path, required=True,
                        help="Path to rfdiffusion_metrics.json "
                             "(produced by RFDIFFUSION_FILTER)")
    parser.add_argument("--design", required=True,
                        help="Design identifier, e.g. 'design_3', "
                             "'design_3.pdb', or '3'")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output text file path.  Existing files "
                             "are overwritten.")
    parser.add_argument("--cross-check-design-region", type=Path, default=None,
                        help="Optional path to the design-region file "
                             "produced by derive_design_region.py.  "
                             "Verifies that the derived true interface "
                             "is a subset of the design region.  "
                             "Advisory only.")
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

    n = len(indices_0based)
    lo, hi = indices_0based[0], indices_0based[-1]
    print(f"  design: {args.design}", file=sys.stderr)
    print(f"  receptor_contact_residues: "
          f"{len(entry.get('receptor_contact_residues') or [])} residues",
          file=sys.stderr)
    print(f"  receptor_position_order length: "
          f"{len(entry.get('receptor_position_order') or [])}",
          file=sys.stderr)
    print(f"  derived true interface: {n} positions "
          f"(0-based range {lo}..{hi})", file=sys.stderr)

    if args.cross_check_design_region is not None:
        cross_check_design_region(
            args.cross_check_design_region, indices_0based
        )

    n_contact_resnums = len(entry.get("receptor_contact_residues") or [])
    text = format_output(
        indices_0based, args.metrics_json, args.design,
        n_contact_resnums,
    )
    args.output.write_text(text)
    print(f"  wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
