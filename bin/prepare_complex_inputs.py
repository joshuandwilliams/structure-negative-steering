#!/usr/bin/env python3
"""Prepare standalone negative-steering inputs from a solved complex.

`negative_steering_run_one.sh` needs the two chain sequences plus a
true-interface and a design-region index file. This derives all of them from
one complex PDB, such as a reference from
structure-prediction-benchmarking/data/complexes_for_benchmarking.

Outputs (written into --outdir), in the formats the engine expects:
  receptor.fasta       receptor chain sequence
  effector.fasta       effector chain sequence
  true_interface.txt   0-based receptor indices, the interface to steer AWAY
                       from
  design_region.txt    1-based receptor indices, the residues negative
                       steering may mutate
  interface_ablation.txt  under --ablate only, the native and ablated residue
                       at every interface position

Under --ablate the sequence in receptor.fasta is no longer the one in the
complex. The engine takes it as an equal-length override and still scores
against the untouched complex, which is what makes it a control.

Interface residues are either derived from heavy-atom contacts within
--contact-cutoff of the effector, the default, or read from --interface-file.

Contact detection and sequence extraction call the engine's own
find_contact_residues_heavy and get_chain_sequence, so the indices and
sequences match what the engine reads at run time.

Chains default to receptor=A and effector=B, the
complexes_for_benchmarking convention.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_BIN = REPO_ROOT / "bin"


def _load_engine_helpers():
    """Import the engine helpers, which need numpy plus gemmi or
    biopython."""
    sys.path.insert(0, str(ENGINE_BIN))
    try:
        from boltz2_negative_steering import get_chain_sequence  # noqa: E402
        from compute_metrics import find_contact_residues_heavy  # noqa: E402
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise SystemExit(
            f"ERROR: could not import engine helpers ({exc}).\n"
            f"prepare_complex_inputs needs numpy plus gemmi (or biopython) for sequence "
            f"extraction. Install the runtime extra:\n"
            f"    pip install -e '.[runtime]'"
        )
    return get_chain_sequence, find_contact_residues_heavy


def _parse_index_file(path: Path) -> List[int]:
    """Read 0-based positional indices, one per line or comma-separated,
    sorted and deduplicated."""
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


def ablate_alanine(seq: str, idx0: List[int]) -> str:
    """Set every selected position to alanine, leaving the rest alone."""
    chars = list(seq)
    for i in idx0:
        chars[i] = "A"
    return "".join(chars)


def ablate_scramble(seq: str, idx0: List[int], seed: int) -> str:
    """Seeded permutation of the residues at the selected positions.

    Length and amino-acid composition are preserved, and positions outside
    idx0 are kept verbatim. Unless the selected residues are all identical the
    permutation is redrawn until it differs from the input, so the ablation
    cannot silently return the native sequence.
    """
    if not idx0:
        return seq
    pool = [seq[i] for i in idx0]
    rng = random.Random(seed)
    distinct = len(set(pool)) > 1
    for _ in range(64):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        if not distinct or shuffled != pool:
            break
    chars = list(seq)
    for i, residue in zip(idx0, shuffled):
        chars[i] = residue
    return "".join(chars)


def default_ablation_seed(complex_pdb: Path) -> int:
    """A per-target seed from the file name, so a re-run reproduces the same
    permutation."""
    digest = hashlib.sha256(complex_pdb.stem.encode()).hexdigest()
    return int(digest[:8], 16)


def _write_ablation_record(path: Path, *, mode: str, seed: int, idx0: List[int],
                           before: str, after: str, complex_pdb: Path) -> None:
    """Record what the ablation changed, so the run carries its own
    provenance."""
    moved = [i for i in idx0 if before[i] != after[i]]
    with path.open("w") as f:
        f.write(f"# receptor interface ablation applied by {Path(__file__).name}\n")
        f.write(f"# complex: {complex_pdb}\n")
        f.write(f"# mode: {mode}\n")
        f.write(f"# seed: {seed if mode == 'scrambled' else 'n/a'}\n")
        f.write(f"# positions ablated: {len(idx0)}\n")
        f.write(f"# positions changed: {len(moved)}\n")
        f.write("# coordinate system: 1-based positional indices on the receptor "
                "chain, PDB walk order\n")
        f.write("# pos\tnative\tablated\n")
        for i in idx0:
            f.write(f"{i + 1}\t{before[i]}\t{after[i]}\n")


def _write_true_interface(path: Path, idx0: List[int], *, complex_pdb: Path,
                          receptor_chain: str, effector_chain: str,
                          source: str) -> None:
    """Write 0-based indices, one per line.

    The format the design pipeline's true-interface files use, which is
    what --true-interface-indices-file parses.

    Do not fold this and _write_design_region into a generic header writer.
    It was tried, and a byte comparison failed on all 16 cases for one
    character: the first header line is `# derived from complex <path>` with
    no colon, and the other five are `# key: value`. A generic
    `# {key}: {value}` writer adds that colon, which rewrites every committed
    derived file while looking like a pure refactor."""
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
    """Write 1-based indices, comma-separated.

    The format --design-region-indices-file parses.  It counts from 1,
    unlike the true-interface file above."""
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

    ap.add_argument("--ablate", choices=["none", "alanine", "scrambled"], default="none",
                    help="Destroy the receptor's true interface before the run. 'alanine' "
                         "sets every interface position to A, 'scrambled' permutes the "
                         "residues already there at fixed composition. The ground truth "
                         "PDB is untouched, so restraints and RMSD still reference the "
                         "solved pose (default none).")
    ap.add_argument("--ablate-seed", type=int, default=None,
                    help="Seed for --ablate scrambled. Defaults to a hash of the complex "
                         "stem, so a re-run reproduces the same permutation.")
    args = ap.parse_args(argv)

    if not args.complex.is_file():
        print(f"ERROR: complex PDB not found: {args.complex}", file=sys.stderr)
        return 2

    get_chain_sequence, find_contact_residues_heavy = _load_engine_helpers()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # --- Sequences ---------------------------------------------------------
    rec_seq = get_chain_sequence(args.complex, args.receptor_chain)
    eff_seq = get_chain_sequence(args.complex, args.effector_chain)
    # The engine parser raises ValueError on an absent chain, so neither
    # guard below can fire. They stay in case a future parser returns ""
    # instead of raising.
    if not rec_seq:  # pragma: no cover - engine raises first
        print(f"ERROR: receptor chain '{args.receptor_chain}' empty in {args.complex}",
              file=sys.stderr)
        return 2
    if not eff_seq:  # pragma: no cover - engine raises first
        print(f"ERROR: effector chain '{args.effector_chain}' empty in {args.complex}",
              file=sys.stderr)
        return 2
    stem = args.complex.stem
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

    # --- Interface ablation (control) --------------------------------------
    # The receptor FASTA is written here, not with the effector above,
    # because the ablation needs the interface indices first.
    ablation_record = args.outdir / "interface_ablation.txt"
    if args.ablate == "none":
        if ablation_record.exists():
            ablation_record.unlink()
        _write_fasta(args.outdir / "receptor.fasta", f"{stem}_receptor", rec_seq)
    else:
        if not iface0:
            print("ERROR: --ablate needs a non-empty true interface.", file=sys.stderr)
            return 2
        seed = (args.ablate_seed if args.ablate_seed is not None
                else default_ablation_seed(args.complex))
        if args.ablate == "alanine":
            ablated = ablate_alanine(rec_seq, iface0)
        else:
            ablated = ablate_scramble(rec_seq, iface0, seed)
        changed = sum(1 for i in iface0 if rec_seq[i] != ablated[i])
        print(f"Interface ablation: {args.ablate}, {len(iface0)} positions, "
              f"{changed} changed"
              + (f", seed {seed}" if args.ablate == "scrambled" else ""))
        if changed == 0:
            print("ERROR: ablation changed nothing, so the control is not a control.",
                  file=sys.stderr)
            return 2
        _write_fasta(args.outdir / "receptor.fasta",
                     f"{stem}_receptor_ablate_{args.ablate}", ablated)
        _write_ablation_record(
            ablation_record, mode=args.ablate, seed=seed, idx0=iface0,
            before=rec_seq, after=ablated, complex_pdb=args.complex,
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
