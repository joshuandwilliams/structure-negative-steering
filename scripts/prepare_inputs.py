#!/usr/bin/env python3
"""Prepare standalone negative-steering inputs from a solved complex.

`negative_steering_run_one.sh` needs more than a bare structure: it wants the
two chain sequences plus a true-interface and a design-region index file. This
script derives all of them from a single complex PDB (e.g. a reference from
structure-prediction-benchmarking/data/complexes_for_benchmarking), so you can
point the engine at, say, 6G10.pdb and run.

Outputs (written into --outdir), in the exact formats the engine expects:
  receptor.fasta       receptor chain sequence
  effector.fasta       effector chain sequence
  true_interface.txt   0-based positional indices on the receptor chain — the
                       interface to steer AWAY from
  design_region.txt    1-based positional indices on the receptor chain — the
                       residues negative steering is allowed to mutate

Interface residues come from one of two modes:
  --derive                 heavy-atom contacts within --contact-cutoff of the
                           effector (the default; "the interface in the complex")
  --interface-file FILE    read positional indices from FILE (0-based, one per
                           line or comma-separated; '#' comments ignored)

The contact detection and sequence extraction reuse the *vendored engine*
helpers (find_contact_residues_heavy, get_chain_sequence), so the indices and
sequences line up exactly with what the engine consumes at run time.

Chains default to receptor=A, effector=B (the complexes_for_benchmarking
convention); override with --receptor-chain / --effector-chain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_BIN = REPO_ROOT / "engine" / "bin"


def _load_engine_helpers():
    """Import the vendored engine helpers (need gemmi or biopython + numpy)."""
    sys.path.insert(0, str(ENGINE_BIN))
    try:
        from boltz2_negative_steering import get_chain_sequence  # noqa: E402
        from compute_metrics import find_contact_residues_heavy  # noqa: E402
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise SystemExit(
            f"ERROR: could not import engine helpers ({exc}).\n"
            f"prepare_inputs needs numpy plus gemmi (or biopython) for sequence "
            f"extraction. Install the experiments extras:\n"
            f"    pip install -e '.[experiments]'"
        )
    return get_chain_sequence, find_contact_residues_heavy


def _parse_index_file(path: Path) -> List[int]:
    """Read 0-based positional indices: one-per-line or comma-separated."""
    out: List[int] = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for tok in line.replace(",", " ").split():
            out.append(int(tok))
    return sorted(set(out))


def _write_fasta(path: Path, header: str, seq: str) -> None:
    path.write_text(f">{header}\n{seq}\n")


def _write_true_interface(path: Path, idx0: List[int], *, complex_pdb: Path,
                          receptor_chain: str, effector_chain: str,
                          source: str) -> None:
    """0-based, one per line — matches derive_true_interface.py's format."""
    with path.open("w") as f:
        f.write(f"# derived from complex {complex_pdb}\n")
        f.write(f"# interface source: {source}\n")
        f.write(f"# receptor chain: {receptor_chain}\n")
        f.write(f"# effector chain: {effector_chain}\n")
        f.write(f"# n contacts: {len(idx0)}\n")
        f.write("# coordinate system: 0-based positional indices on the receptor "
                "chain, PDB walk order\n")
        for p in idx0:
            f.write(f"{p}\n")


def _write_design_region(path: Path, idx1: List[int], *, complex_pdb: Path,
                         receptor_chain: str, source: str) -> None:
    """1-based comma-separated — matches derive_design_region.py's format."""
    with path.open("w") as f:
        f.write(f"# derived from complex {complex_pdb}\n")
        f.write(f"# design region source: {source}\n")
        f.write(f"# receptor chain: {receptor_chain}\n")
        f.write(f"# design region size: {len(idx1)} residues\n")
        f.write("# coordinate system: 1-based positional indices on the receptor "
                "chain, PDB walk order\n")
        f.write((",".join(str(p) for p in idx1) if idx1 else "") + "\n")


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--complex", required=True, type=Path,
                    help="Solved complex PDB (the comparison basis / ground truth).")
    ap.add_argument("--outdir", required=True, type=Path,
                    help="Directory to write the prepared inputs into.")
    ap.add_argument("--receptor-chain", default="A", help="Receptor chain ID (default A).")
    ap.add_argument("--effector-chain", default="B", help="Effector chain ID (default B).")

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--derive", action="store_true", default=True,
                      help="Derive the interface from heavy-atom contacts (default).")
    mode.add_argument("--interface-file", type=Path, default=None,
                      help="Instead of deriving, read 0-based interface indices from this file.")
    ap.add_argument("--contact-cutoff", type=float, default=5.0,
                    help="Heavy-atom contact cutoff in Angstroms for --derive (default 5.0).")

    ap.add_argument("--design-region", choices=["none", "interface", "all"], default="interface",
                    help="Receptor residues the engine PROTECTS from steering mutations: "
                         "'none' (protect only the true interface — right for a native "
                         "complex), 'interface', or 'all' (whole chain). Overridden by "
                         "--design-region-file.")
    ap.add_argument("--design-region-file", type=Path, default=None,
                    help="Explicit 1-based design-region indices file (overrides --design-region).")
    args = ap.parse_args(argv)

    if not args.complex.is_file():
        print(f"ERROR: complex PDB not found: {args.complex}", file=sys.stderr)
        return 2

    get_chain_sequence, find_contact_residues_heavy = _load_engine_helpers()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # --- Sequences ---------------------------------------------------------
    rec_seq = get_chain_sequence(args.complex, args.receptor_chain)
    eff_seq = get_chain_sequence(args.complex, args.effector_chain)
    if not rec_seq:
        print(f"ERROR: receptor chain '{args.receptor_chain}' empty in {args.complex}",
              file=sys.stderr)
        return 2
    if not eff_seq:
        print(f"ERROR: effector chain '{args.effector_chain}' empty in {args.complex}",
              file=sys.stderr)
        return 2
    stem = args.complex.stem
    _write_fasta(args.outdir / "receptor.fasta", f"{stem}_receptor", rec_seq)
    _write_fasta(args.outdir / "effector.fasta", f"{stem}_effector", eff_seq)
    print(f"Receptor chain {args.receptor_chain}: {len(rec_seq)} residues")
    print(f"Effector chain {args.effector_chain}: {len(eff_seq)} residues")

    # --- Interface (0-based) ----------------------------------------------
    if args.interface_file is not None:
        if not args.interface_file.is_file():
            print(f"ERROR: --interface-file not found: {args.interface_file}", file=sys.stderr)
            return 2
        iface0 = _parse_index_file(args.interface_file)
        iface_source = f"provided file {args.interface_file}"
        bad = [i for i in iface0 if i < 0 or i >= len(rec_seq)]
        if bad:
            print(f"ERROR: provided interface indices out of range 0..{len(rec_seq) - 1}: "
                  f"{bad[:10]}{'...' if len(bad) > 10 else ''}", file=sys.stderr)
            return 2
    else:
        contacts = find_contact_residues_heavy(
            str(args.complex), args.receptor_chain, args.effector_chain,
            args.contact_cutoff, expected_rec_seq=rec_seq,
        )
        iface0 = sorted(i for i, _ in contacts)
        iface_source = f"heavy-atom contacts <= {args.contact_cutoff} A"
    print(f"True interface: {len(iface0)} receptor residues ({iface_source})")
    _write_true_interface(
        args.outdir / "true_interface.txt", iface0,
        complex_pdb=args.complex, receptor_chain=args.receptor_chain,
        effector_chain=args.effector_chain, source=iface_source,
    )

    # --- Design region (1-based) ------------------------------------------
    if args.design_region_file is not None:
        if not args.design_region_file.is_file():
            print(f"ERROR: --design-region-file not found: {args.design_region_file}",
                  file=sys.stderr)
            return 2
        design1 = _parse_index_file(args.design_region_file)
        dr_source = f"provided file {args.design_region_file}"
    elif args.design_region == "all":
        design1 = list(range(1, len(rec_seq) + 1))
        dr_source = "whole receptor chain"
    elif args.design_region == "none":
        design1 = []
        dr_source = "none (protect only the true interface)"
    else:  # interface
        design1 = [i + 1 for i in iface0]
        dr_source = "interface residues (1-based)"
    print(f"Design region: {len(design1)} receptor residues ({dr_source})")
    _write_design_region(
        args.outdir / "design_region.txt", design1,
        complex_pdb=args.complex, receptor_chain=args.receptor_chain, source=dr_source,
    )

    print(f"\nPrepared inputs in {args.outdir}/")
    for name in ("receptor.fasta", "effector.fasta", "true_interface.txt", "design_region.txt"):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
