#!/usr/bin/env python3
"""Extract the receptor residues that contact the effector in an
RFDiffusion design.  Positions come out as 0-based positional indices,
which is what ``boltz2_negative_steering.py``'s
``--true-interface-indices-file`` flag expects.

``plan`` in ``boltz2_negative_steering.py`` normally finds the "true
interface" itself with heavy-atom contact detection on the ground-truth
PDB.  An RFDiffusion design's receptor chain is Cα-only, so the contact
set must be computed in advance and passed in through that flag.

The upstream ``RFDIFFUSION_FILTER`` step, run outside this repo, has
already done the detection with Cα-Cα distances and a tunable cutoff.
It writes them into ``rfdiffusion_metrics.json`` under
``receptor_contact_residues``, next to the ``receptor_position_order``
list ``derive_design_region.py`` uses as its index key.  Intersecting
those two fields the same way keeps the design region and the true
interface in matching coordinate systems.

The output is one integer per line, each a 0-based positional index on
the receptor chain in prediction PDB sequence order.  Beware the
difference from ``derive_design_region.py``, which writes 1-based
positions.  The parser also accepts comma- or whitespace-separated
values and ``#`` line comments, and one per line gives the clearest
diffs.  A ``#`` provenance header goes at the top.

    python3 derive_true_interface.py \\
        --metrics-json /path/to/rfdiffusion_metrics.json \\
        --design design_3 \\
        --output design_3_true_interface.txt

``--cross-check-design-region <file>`` checks that the derived interface
is a subset of the design-region file, which should hold for
receptor-resurfacing designs.  It is advisory, so a mismatch only
warns.
"""

import argparse
import json
import sys
from pathlib import Path


def normalise_design_id(raw):
    """Accept 'design_3', 'design_3.pdb', or '3'.  Return both the
    bare and .pdb forms, so the caller matches whichever key
    rfdiffusion_metrics.json uses."""
    raw = str(raw).strip()
    if raw.endswith(".pdb"):
        stem = raw[:-4]
    else:
        stem = raw
    if stem.isdigit():
        stem = f"design_{stem}"
    return {"bare": stem, "with_pdb": f"{stem}.pdb"}


def find_design_entry(metrics, design_id):
    """Find the record in metrics['designs'] that matches `design_id`.

    Raises KeyError if nothing matches, with the present designs listed
    in the message.
    """
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
    ``receptor_position_order``.  Return 0-based positional indices,
    sorted ascending.

    The same transform as ``derive_design_region.py``, which reads
    ``per_design_design_residues`` instead.
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

    # A missing resnum means the two lists describe different atom
    # walks, so the derivation cannot be trusted.
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
    """Check that the derived true interface is a subset of the
    design-region file.  In RFDiffusion plus MPNN resurfacing the
    binding site sits inside the de novo block, so the subset relation
    should hold.

    It reads every non-``#`` line, accepting comma-separated integers
    and ``start-end`` ranges as 1-based positions, because that is what
    ``derive_design_region.py`` writes.

    The check is advisory, printing a warning on a mismatch rather than
    raising.
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

    # The interface is 0-based and the design-region file 1-based.
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
    """Build the contents of the output file.  A ``#`` provenance
    header comes first, then one 0-based index per line."""
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

    if not indices_0based:  # pragma: no cover - derive_positional_indices raises on every input that would empty this list
        # derive_positional_indices already raises on every case that
        # would empty this list.  The guard stays because that
        # invariant lives there, not here.
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
