#!/usr/bin/env python3
"""
compute_metrics.py  (minimal boltz2-only build)
-----------------------------------------------
Compute confidence metrics from Boltz-2 structure prediction outputs.
A cut-down build of the full benchmark-pipeline compute_metrics.py,
holding only what `negsteer_aggregate.py compute-final-metrics` needs.

The structural-RMSD code is gone, meaning Kabsch fitting, sequence
alignment, reference comparison and matchmaker-style iterative pruning.
The steering pipeline already writes authoritative per-design RMSDs to
`all_results_multicycle.csv` with the correct truth-to-prediction chain
mapping, so recomputing them here would add nothing.

CSV_FIELDS lists the columns.  af_rank_score is 0.8 * iptm + 0.2 * ptm,
AlphaFold-Multimer's ranking score.  An earlier version wrote it out
mislabelled as 'actifptm', so it keeps its correct name here.

Usage:
    python compute_metrics.py \\
        --model boltz2 \\
        --prediction-dir <dir containing one or more *.pdb files> \\
        --chain-lengths <rec_len> <eff_len> \\
        --output-csv metrics.csv

--prediction-dir must point at Boltz's own nested output directory, the
one holding the confidence, pae and plddt sidecar files.  A copied
`prediction.pdb` has none beside it, so pointing this flag at that
copy's parent yields ptm=0, iptm=0 and pae=0, with only pLDDT surviving
through plddt_from_pdb's B-factor-column fallback.
"""

import argparse
import csv
import glob
import json
import os
import sys
import traceback

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# ipSAE (Dunbrack 2025)
# ═══════════════════════════════════════════════════════════════════════════

def _ipsae_d0(n):
    """TM-score d0 formula used by Dunbrack 2025's ipSAE.

    Raw formula: d0 = 1.24 * (n - 15)^(1/3) - 1.8.  This returns
    max(d0, 0.5), the TM-align convention.  The raw formula dips under
    0.5 for n roughly between 15 and 28, where without the clamp the
    denominator explodes and the pTM kernel sends the score to 0 for any
    PAE above about 1 Å.

    `n` is the interacting set size for one reference residue, how many
    residues of the opposite chain fell below the PAE cutoff for it,
    not the chain length.  That separates ipSAE from a global pTM-style
    metric and makes it asymmetric, because the two directions count
    different partners and get different d0 values.
    """
    if n < 16:
        return 0.5
    d0 = 1.24 * ((n - 15) ** (1.0 / 3.0)) - 1.8
    return max(d0, 0.5)


def compute_ipsae_one_direction(pae_ij, cutoff=10.0):
    """ipSAE for direction i->j (Dunbrack 2025).

    For each reference residue i, n_below counts the partner residues
    j with PAE_ij < cutoff, the interacting set size _ipsae_d0 turns
    into d0.  The per-residue score is

        s_i = mean over j with PAE<cutoff of  1 / (1 + (PAE_ij/d0)^2)

    and the ipSAE is the MAXIMUM s_i.  Returns
    (score, best_residue_idx), or (0.0, -1) when no reference residue
    has a partner below the cutoff.
    """
    n_i, n_j = pae_ij.shape
    scores = []
    for i in range(n_i):
        row = pae_ij[i, :]
        below = row[row < cutoff]
        n_below = len(below)
        if n_below == 0:
            scores.append(0.0)
            continue
        d0 = _ipsae_d0(n_below)
        if d0 <= 0:  # pragma: no cover - _ipsae_d0 returns max(d0, 0.5), never <= 0
            scores.append(0.0)
            continue
        s = 1.0 / (1.0 + (below / d0) ** 2)
        scores.append(float(s.mean()))
    if not scores:
        return 0.0, -1
    return max(scores), int(np.argmax(scores))


def _pae_to_ptm_score(pae_ij, n_interface):
    """Per-residue pTM kernel for a PAE slice.

    Returns one score per reference residue, each the mean of
    1 / (1 + (PAE/d0)^2) over the partner columns given.

    `n_interface` is the size fed to the d0 formula, and that choice
    distinguishes the metrics.  Vanilla pTM uses the total chain length,
    ipTM (AlphaFold-Multimer) the inter-chain partner length, actifPTM
    (Varga 2025) the count of residues with an inter-chain PAE pair
    below the cutoff.
    """
    if pae_ij.size == 0 or n_interface < 1:
        return np.zeros(pae_ij.shape[0], dtype=float)
    d0 = _ipsae_d0(n_interface)
    if d0 <= 0:  # pragma: no cover - _ipsae_d0 returns max(d0, 0.5), never <= 0
        return np.zeros(pae_ij.shape[0], dtype=float)
    kernel = 1.0 / (1.0 + (pae_ij / d0) ** 2)
    if kernel.ndim == 1:
        return kernel
    return kernel.mean(axis=1)


def compute_actifptm(pae_matrix, chain_lengths, cutoff=10.0):
    """actifPTM (Varga, Ovchinnikov, Schueler-Furman 2025).

    actifPTM is ipTM restricted to the "interacting residue set", the
    residues with at least one inter-chain PAE entry below the cutoff.
    The pTM kernel's d0 comes from the SIZE OF THAT SET, not the full
    chain length.  For a two-chain complex A, B:

      1. n_int_A counts A's residues with at least one PAE[A->B] entry
         below `cutoff`, n_int_B the same through PAE[B->A].
      2. For each interacting residue i in A,
            s_i = mean over interacting j in B of
                  1 / (1 + (PAE_ij / d0(n_int_B))^2)
         and score B's interacting residues the same way.
      3. actifPTM is max(max s_i, max s_j).

    Returns a float on [0, 1].  An empty interacting set gives 0.0.
    """
    if pae_matrix is None or len(chain_lengths) < 2:
        return 0.0

    len_a = chain_lengths[0]
    len_b = sum(chain_lengths[1:])
    total = pae_matrix.shape[0]
    if total != len_a + len_b:
        if total > 0 and len_a < total:
            len_b = total - len_a
        else:
            return 0.0

    pae_ab = pae_matrix[:len_a, len_a:len_a + len_b]
    pae_ba = pae_matrix[len_a:len_a + len_b, :len_a]

    # A residue qualifies if it has at least one inter-chain neighbour
    # with PAE below the cutoff.
    int_a_mask = (pae_ab < cutoff).any(axis=1)
    int_b_mask = (pae_ba < cutoff).any(axis=1)
    n_int_a = int(int_a_mask.sum())
    n_int_b = int(int_b_mask.sum())
    if n_int_a == 0 or n_int_b == 0:
        return 0.0

    pae_ab_sub = pae_ab[int_a_mask][:, int_b_mask]
    pae_ba_sub = pae_ba[int_b_mask][:, int_a_mask]

    scores_ab = _pae_to_ptm_score(pae_ab_sub, n_int_b)
    scores_ba = _pae_to_ptm_score(pae_ba_sub, n_int_a)

    best_ab = float(scores_ab.max()) if scores_ab.size else 0.0
    best_ba = float(scores_ba.max()) if scores_ba.size else 0.0

    return round(max(best_ab, best_ba), 4)


def compute_ipsae(pae_matrix, chain_lengths, cutoff=10.0):
    """Compute ipSAE in both directions from the full PAE matrix.

    Returns ipsae_ab, ipsae_ba and their minimum.  The two differ
    because each one's d0 comes from its own partner count.
    """
    if pae_matrix is None or len(chain_lengths) < 2:
        return {"ipsae_ab": 0.0, "ipsae_ba": 0.0, "ipsae_min": 0.0}

    len_a = chain_lengths[0]
    len_b = sum(chain_lengths[1:])
    total = pae_matrix.shape[0]

    if total != len_a + len_b:
        print(f"  WARNING: pAE matrix size {total} != expected {len_a + len_b}")
        if total > 0 and len_a < total:
            len_b = total - len_a
        else:
            return {"ipsae_ab": 0.0, "ipsae_ba": 0.0, "ipsae_min": 0.0}

    pae_ab = pae_matrix[:len_a, len_a:len_a + len_b]
    pae_ba = pae_matrix[len_a:len_a + len_b, :len_a]

    ipsae_ab, _ = compute_ipsae_one_direction(pae_ab, cutoff)
    ipsae_ba, _ = compute_ipsae_one_direction(pae_ba, cutoff)

    return {
        "ipsae_ab": round(ipsae_ab, 4),
        "ipsae_ba": round(ipsae_ba, 4),
        "ipsae_min": round(min(ipsae_ab, ipsae_ba), 4),
    }


def compute_ipae(pae_matrix, chain_lengths):
    """Mean interchain PAE (both off-diagonal blocks)."""
    if pae_matrix is None or len(chain_lengths) < 2:
        return 0.0
    len_a = chain_lengths[0]
    total = pae_matrix.shape[0]
    if total != sum(chain_lengths):
        return 0.0
    block_ab = pae_matrix[:len_a, len_a:]
    block_ba = pae_matrix[len_a:, :len_a]
    all_interchain = np.concatenate([block_ab.flatten(), block_ba.flatten()])
    return round(float(all_interchain.mean()), 2)


def compute_pae_pass_frac(pae_matrix, chain_lengths, cutoff):
    """Fraction of inter-chain PAE values strictly below the ipSAE
    cutoff, so how much of the interface ipSAE covers.  A value of 0.05
    means only 5% of pairs contributed, so ipSAE is summarising a very
    thin slice.
    """
    if pae_matrix is None or len(chain_lengths) < 2:
        return 0.0
    len_a = chain_lengths[0]
    total = pae_matrix.shape[0]
    if total != sum(chain_lengths):
        return 0.0
    block_ab = pae_matrix[:len_a, len_a:]
    block_ba = pae_matrix[len_a:, :len_a]
    all_interchain = np.concatenate([block_ab.flatten(), block_ba.flatten()])
    if all_interchain.size == 0:
        return 0.0
    n_pass = int(np.sum(all_interchain < cutoff))
    return round(n_pass / float(all_interchain.size), 4)


def _pae_derived_metrics(pae_matrix, chain_lengths, cutoff=10.0):
    """Compute all PAE-derived metrics in one call.

    ipSAE runs at two PAE cutoffs in one pass.  `cutoff` defaults to
    10 Å, the Dunbrack 2025 default, giving ipsae_ab, ipsae_ba and
    ipsae_min.  The second is a fixed 15 Å, more permissive for
    borderline poses, giving the _15 variants.  Both run here because
    the matrix is already in memory, where the earlier implementation
    reloaded it from disk.
    """
    pae_mean = round(float(pae_matrix.mean()), 2) if pae_matrix is not None else 0.0
    ipae = compute_ipae(pae_matrix, chain_lengths)
    ipsae = compute_ipsae(pae_matrix, chain_lengths, cutoff=cutoff)
    ipsae_15 = compute_ipsae(pae_matrix, chain_lengths, cutoff=15.0)
    pae_pass_frac = compute_pae_pass_frac(pae_matrix, chain_lengths, cutoff)
    return {
        "pae_mean": pae_mean,
        "ipae": ipae,
        "pae_pass_frac": pae_pass_frac,
        **ipsae,
        "ipsae_ab_15": ipsae_15["ipsae_ab"],
        "ipsae_ba_15": ipsae_15["ipsae_ba"],
        "ipsae_min_15": ipsae_15["ipsae_min"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Interface contact analysis + mutation-reliance check
# ═══════════════════════════════════════════════════════════════════════════

def _parse_heavy_atoms_by_chain(pdb_path):
    """Parse ATOM records from a PDB and return a dict:
        { chain_id: [ (resseq, x, y, z), ... ] }
    Hydrogens and HETATM are excluded, residue numbers come from the PDB
    as-is, insertion codes are dropped, and altlocs other than '' or 'A'
    are skipped.
    """
    by_chain = {}
    try:
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                # Fixed-column PDB parsing.  The spec numbers columns
                # from 1, Python slices from 0 and exclude the end.
                altloc = line[16:17]
                if altloc not in (" ", "", "A"):
                    continue
                element = line[76:78].strip()
                if element == "H":
                    continue
                # With a blank element column, infer from the atom
                # name, where a leading H means hydrogen.
                if not element:
                    atom_name = line[12:16].strip()
                    if atom_name.startswith("H") or (
                        len(atom_name) >= 2 and atom_name[0].isdigit() and atom_name[1] == "H"
                    ):
                        continue
                chain_id = line[21:22].strip() or " "
                try:
                    resseq = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    continue
                by_chain.setdefault(chain_id, []).append((resseq, x, y, z))
    except Exception as e:
        print(f"  WARNING: failed to parse heavy atoms from {pdb_path}: {e}")
        return {}
    return by_chain


def compute_effector_interface_residues(
    ground_truth_pdb,
    true_interface_residues_1b,
    receptor_chain,
    effector_chain,
    cutoff_angstroms=8.0,
):
    """Find the effector's "interface residues" in the ground truth.

    A residue qualifies when its Cα sits within `cutoff_angstroms` of
    the Cα of any receptor residue in `true_interface_residues_1b`.
    Later predictions restrict contact detection to these residues'
    heavy atoms, so an off-interface contact cannot affect the
    contamination check.  The cutoff defaults to 8.0 Å, the RFDiffusion
    filter's Cα-distance convention for `true_interface_idx`.

    The filter works at residue level, not atom level, because the
    RFDiffusion ground truth holds Cα atoms only.  An atom-level mask
    from it would hold only Cα atoms, which almost never form
    heavy-to-heavy contacts at 5 Å against a full-side-chain prediction.

    Returns a set of 1-based POSITIONAL indices into the effector chain,
    in prediction-PDB order.  The ground-truth PDB may instead use a
    synthetic resseq scheme, because RFDiffusion offsets residue numbers
    by 1000 or more above the highest fixed one, while Boltz always
    numbers chains positionally from 1.  A ground-truth-resseq filter
    would therefore match nothing.  An empty set on failure means "no
    filter", and callers fall back to unrestricted atoms.
    """
    if not ground_truth_pdb:
        return set()
    true_set_1b = set(int(i) for i in (true_interface_residues_1b or []))
    if not true_set_1b:
        return set()

    # Cα positions per chain in file order, which is sequence order.
    # Effector residues carry their 1-based rank in that order, the
    # coordinate system the prediction uses.
    rec_ca = []       # list of (x, y, z), indexed 0-based positional
    eff_ca = []       # list of (pos_1b, x, y, z)
    seen_rec = set()
    seen_eff = set()
    try:
        with open(ground_truth_pdb) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                altloc = line[16]
                if altloc not in (" ", "A"):
                    continue
                if line[12:16].strip() != "CA":
                    continue
                chain_id = line[21]
                try:
                    resseq = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    continue
                if chain_id == receptor_chain and resseq not in seen_rec:
                    seen_rec.add(resseq)
                    rec_ca.append((x, y, z))
                elif chain_id == effector_chain and resseq not in seen_eff:
                    seen_eff.add(resseq)
                    pos_1b = len(eff_ca) + 1
                    eff_ca.append((pos_1b, x, y, z))
    except Exception as e:
        print(f"  WARNING: failed to parse ground truth {ground_truth_pdb}: {e}")
        return set()

    if not rec_ca or not eff_ca:
        return set()

    # true_interface_residues_1b holds 1-based positional indices into
    # the receptor chain, in file order.
    iface_ca = []
    for pos_1b in sorted(true_set_1b):
        idx = pos_1b - 1
        if 0 <= idx < len(rec_ca):
            iface_ca.append(rec_ca[idx])
    if not iface_ca:
        return set()
    iface_arr = np.array(iface_ca, dtype=np.float64)

    # An effector residue joins when its Cα is within cutoff of any
    # true-interface receptor Cα.  Its 1-based positional index goes
    # in, NOT its ground-truth resseq.
    cutoff2 = float(cutoff_angstroms) ** 2
    interface_residues = set()
    for pos_1b, x, y, z in eff_ca:
        diffs = iface_arr - np.array([x, y, z], dtype=np.float64)
        d2 = (diffs * diffs).sum(axis=1)
        if (d2 < cutoff2).any():
            interface_residues.add(pos_1b)

    return interface_residues


def compute_interface_contacts(
    pdb_path,
    receptor_chain,
    effector_chain,
    cutoff=5.0,
    effector_atom_filter=None,
):
    """Return the set of 1-based receptor residue numbers whose heavy
    atoms come within `cutoff` Å of any effector heavy atom.

    Chain IDs are used as-is, so they must match the letters in the
    file.  Boltz-2 predictions normally use "A" and "B", from
    plan.json's pred_receptor_chain and pred_effector_chain.

    `effector_atom_filter` names the effector atoms counting as
    interface atoms, so an off-interface brush no longer registers.  The
    preferred form is a set of effector resseq integers from
    compute_effector_interface_residues, which explains why filtering
    happens at residue level.  The older (resseq, atom_name) pair form
    is still accepted for plan.json files already written that way,
    collapsed internally so {(R, "CA"), (R, "CB")} becomes {R}.  None
    means every effector heavy atom counts.

    Returns an empty set if either chain is missing from the PDB.
    """
    by_chain = _parse_heavy_atoms_by_chain(pdb_path)
    if receptor_chain not in by_chain or effector_chain not in by_chain:
        return set()

    rec_atoms = by_chain[receptor_chain]
    eff_atoms = by_chain[effector_chain]
    if not rec_atoms or not eff_atoms:  # pragma: no cover - _parse_heavy_atoms_by_chain creates a chain key only by appending an atom, and the membership test above covers an absent chain
        return set()

    # Accepts a set of ints, or (resseq, atom_name) tuples collapsed by
    # keeping only resseq.
    filter_residues = None
    if effector_atom_filter is not None:
        try:
            filter_list = list(effector_atom_filter)
        except TypeError:
            filter_list = []
        filter_residues = set()
        for item in filter_list:
            if isinstance(item, int):
                filter_residues.add(item)
            elif isinstance(item, (tuple, list)) and len(item) >= 1:
                try:
                    filter_residues.add(int(item[0]))
                except (TypeError, ValueError):
                    pass
        if not filter_residues:
            filter_residues = None   # an empty filter means no filter

    if filter_residues is not None:
        eff_atoms = [
            (r, x, y, z) for (r, x, y, z) in eff_atoms
            if r in filter_residues
        ]

    if not eff_atoms:
        return set()

    rec_xyz = np.array([(x, y, z) for (_r, x, y, z) in rec_atoms], dtype=np.float64)
    rec_res = np.array([r for (r, _x, _y, _z) in rec_atoms], dtype=np.int32)
    eff_xyz = np.array([(x, y, z) for (_r, x, y, z) in eff_atoms], dtype=np.float64)

    cutoff2 = float(cutoff) ** 2

    # Pairwise squared distances, shape (n_rec, n_eff).  An 80-residue
    # receptor and a 100-residue effector give roughly 50k atom pairs,
    # small enough that no chunking is needed.
    diff = rec_xyz[:, None, :] - eff_xyz[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    atom_in_contact = (d2 < cutoff2).any(axis=1)
    # A receptor residue with at least one contacting atom counts.
    contact_residues = set(int(r) for r in rec_res[atom_in_contact])
    return contact_residues


def find_contact_residues_heavy(
    pdb_path,
    receptor_chain,
    effector_chain,
    cutoff=5.0,
    expected_rec_seq=None,
):
    """Per-residue heavy-atom contact analysis on the receptor chain.

    Returns (positional_index_0b, closest_distance_angstroms) tuples for
    the receptor residues whose closest heavy-atom distance to the
    effector is under the cutoff.

    Two differences from compute_interface_contacts:
      1. POSITIONAL indices, 0-based in PDB walk order, not seqid
         residue numbers.  The design-region and true-interface
         files are both in positional coordinates, so a caller
         comparing against them needs these.
      2. The closest distance per contacting residue, used by
         downstream weighting, not a plain contact flag.

    ``boltz2_negative_steering.find_contact_residues_heavy`` is a second
    implementation with the same signature, also insertion-code aware,
    and the two agree exactly on a well-formed structure: the same
    contact set and the same distances over all 18 complexes in
    ``data/reference_complexes/``.

    It sorts by distance rather than by index and returns full-precision
    distances, and it RAISES on a malformed structure where this one
    degrades.  This copy is the lenient one on purpose, because it runs
    at setup rather than mid-run.  They are therefore not
    interchangeable.  Its own docstring has the full account.

    When `expected_rec_seq` is supplied, a residue count unequal to its
    length raises ValueError.  That catches PDB-versus-FASTA
    disagreements which would otherwise misalign positional indices
    downstream.  Hydrogens and non-ATOM records are excluded, and a
    missing chain gives an empty list.
    """
    # _parse_heavy_atoms_by_chain keys on resseq alone, but this keeps
    # the (resseq, icode) tuple so residues sharing a number with
    # different insertion codes get distinct indices.  Matches
    # read_residue_heavy_atoms in boltz2_negative_steering.py.
    rec_residue_order = []   # list of (resseq, icode) in PDB walk order
    rec_residue_atoms = {}   # (resseq, icode) -> list of (x, y, z)
    eff_atoms_xyz = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            altloc = line[16:17]
            if altloc not in (" ", "", "A"):
                continue
            element = line[76:78].strip() if len(line) >= 78 else ""
            if element == "H":
                continue
            if not element:
                atom_name = line[12:16].strip()
                if atom_name.startswith("H") or (
                    len(atom_name) >= 2
                    and atom_name[0].isdigit()
                    and atom_name[1] == "H"
                ):
                    continue
            chain_id = line[21:22].strip() or " "
            try:
                resseq = int(line[22:26])
                icode = line[26:27]
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue
            if chain_id == receptor_chain:
                key = (resseq, icode)
                if key not in rec_residue_atoms:
                    rec_residue_order.append(key)
                    rec_residue_atoms[key] = []
                rec_residue_atoms[key].append((x, y, z))
            elif chain_id == effector_chain:
                eff_atoms_xyz.append((x, y, z))

    if not rec_residue_order or not eff_atoms_xyz:
        return []

    if expected_rec_seq is not None and len(rec_residue_order) != len(expected_rec_seq):
        raise ValueError(
            f"receptor residue count from {pdb_path} chain "
            f"{receptor_chain!r} ({len(rec_residue_order)}) does not "
            f"match expected_rec_seq length ({len(expected_rec_seq)})."
        )

    eff_xyz = np.array(eff_atoms_xyz, dtype=np.float64)
    out = []
    cutoff_f = float(cutoff)
    for pos_idx, key in enumerate(rec_residue_order):
        rec_xyz = np.array(rec_residue_atoms[key], dtype=np.float64)
        diff = rec_xyz[:, None, :] - eff_xyz[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        min_d = float(np.sqrt(d2.min()))
        if min_d < cutoff_f:
            out.append((pos_idx, round(min_d, 3)))
    return out


def parse_mutated_positions(spec):
    """Parse a `--mutated-positions` CLI argument.

    Accepts a plain comma-separated list, "5,26,44,46", or a ChimeraX
    spec with chain, "/A:5,26,44,46".

    Returns a sorted list of ints, or [] for empty input.  An invalid
    token is skipped with a warning, which beats a hard failure
    part-way through a run.
    """
    if not spec:
        return []
    s = spec.strip()
    if s.startswith("/"):
        colon = s.find(":")
        if colon != -1:
            s = s[colon + 1:]
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            print(f"  WARNING: ignoring non-integer mutated position {tok!r}")
    return sorted(set(out))


def classify_mutation_reliance(
    contact_residues,
    mutated_positions,
):
    """Compare contact residue numbers against the mutated positions.

    This is the AUTHORITATIVE contamination check for the
    negative-steering pipeline.  Returns
    (n_on_mut, mutated_contact_positions), where n_on_mut > 0 means a
    steering mutation contacts the effector in the predicted pose.  The
    wet-lab construct will not carry that residue, so the metrics are
    invalid unless the reversion pass recovers it.
    `mutated_contact_positions` is exactly what that pass must revert.
    """
    if not mutated_positions:
        return (0, [])
    mut_set = set(mutated_positions)
    overlap = sorted(contact_residues & mut_set)
    return (len(overlap), overlap)


# ═══════════════════════════════════════════════════════════════════════════
# pLDDT from PDB B-factor column
# ═══════════════════════════════════════════════════════════════════════════

def plddt_from_pdb(pdb_path):
    """Average pLDDT from the CA B-factor column, the fallback when no
    plddt sidecar .npz is available."""
    bfactors = []
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    try:
                        bfactors.append(float(line[60:66]))
                    except (ValueError, IndexError):
                        pass
    except Exception:
        return 0.0
    if not bfactors:
        return 0.0
    return round(float(np.mean(bfactors)), 2)


# ═══════════════════════════════════════════════════════════════════════════
# Entry finalisation
# ═══════════════════════════════════════════════════════════════════════════

def _finalize_entry(entry, pae_matrix, chain_lengths, pae_cutoff=10.0):
    """Apply defaults, then compute the PAE-derived metrics, actifPTM
    and af_rank_score.  The last two are separate columns, defined in
    compute_actifptm and the module docstring."""
    entry.setdefault("ptm", 0.0)
    entry.setdefault("iptm", 0.0)
    entry.setdefault("avg_plddt", 0.0)
    entry.update(_pae_derived_metrics(pae_matrix, chain_lengths, cutoff=pae_cutoff))
    entry["actifptm"] = compute_actifptm(pae_matrix, chain_lengths, cutoff=pae_cutoff)
    entry["af_rank_score"] = round(0.8 * entry["iptm"] + 0.2 * entry["ptm"], 4)
    return entry


# ═══════════════════════════════════════════════════════════════════════════
# Native-PDB-dependent metrics, moved here from
# compute_interface_metrics.py so they are computed per prediction,
# beside the PAE-derived metrics, instead of in a separate later pass.
# ═══════════════════════════════════════════════════════════════════════════

# Gaussian weight parameters for the weighted Jaccard, read from
# PipelineInternalThresholds with a literal fallback.  They stay fixed
# across cohorts so numbers stay comparable.  A cohort needing
# different parameters gets a NEW column, not a retune of these.
try:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from pipeline_thresholds import PipelineInternalThresholds as _PIT  # noqa: E402
    _T = _PIT.default()
    _WJ_MU = _T.weighted_jaccard_mu
    _WJ_TWO_SIGMA_SQ = _T.weighted_jaccard_two_sigma_sq
except Exception:
    _WJ_MU = 4.0
    _WJ_TWO_SIGMA_SQ = 2.25


def _gaussian_contact_weight(distance_angstroms):
    """Per-pair Gaussian weight: exp(-(d - 4)^2 / 2.25)."""
    delta = distance_angstroms - _WJ_MU
    return float(np.exp(-(delta * delta) / _WJ_TWO_SIGMA_SQ))


def _collect_chain_cb_positions(structure, chain_id):
    """
    Return [(positional_index_0b, cb_xyz, residue_name)] for every
    residue in the named chain, using Cβ where present and Cα for
    glycine and Cα-only models.  Positional rather than seqid.num
    indexing keeps the (i, j) keys comparable between model and ground
    truth when their numbering schemes differ.
    """
    out = []
    pos_idx = 0
    for model in structure:
        for chain in model:
            if chain.name != chain_id:
                continue
            for res in chain:
                cb_atom = None
                ca_atom = None
                for atom in res:
                    if atom.name == "CB":
                        cb_atom = atom
                    elif atom.name == "CA":
                        ca_atom = atom
                anchor = cb_atom if cb_atom is not None else ca_atom
                if anchor is None:
                    pos_idx += 1
                    continue
                xyz = np.array(
                    [anchor.pos.x, anchor.pos.y, anchor.pos.z],
                    dtype=np.float64,
                )
                out.append((pos_idx, xyz, res.name))
                pos_idx += 1
            break  # first matching chain only
        break  # first model only
    return out


def _build_weighted_pair_map(structure, receptor_chain, effector_chain,
                              pair_cutoff):
    """
    {(receptor_pos_0b, effector_pos_0b): weight} for every Cβ-Cβ pair
    within pair_cutoff Å.  A pair beyond the cutoff carries a weight of
    about 0, so dropping it costs nothing.
    """
    rec = _collect_chain_cb_positions(structure, receptor_chain)
    eff = _collect_chain_cb_positions(structure, effector_chain)
    if not rec or not eff:
        return {}

    rec_xyz = np.array([r[1] for r in rec])
    eff_xyz = np.array([e[1] for e in eff])
    diffs = rec_xyz[:, None, :] - eff_xyz[None, :, :]
    dists = np.sqrt((diffs * diffs).sum(axis=2))

    pair_map = {}
    keep_i, keep_j = np.where(dists <= float(pair_cutoff))
    for ii, jj in zip(keep_i.tolist(), keep_j.tolist()):
        d = float(dists[ii, jj])
        pair_map[(rec[ii][0], eff[jj][0])] = _gaussian_contact_weight(d)
    return pair_map


def compute_weighted_jaccard(model_pdb, native_pdb, receptor_chain,
                              effector_chain, pair_cutoff=8.0):
    """
    Smooth distance-weighted Jaccard contact overlap, model vs native.

        weighted_jaccard
            = sum_pairs min(w_model(d), w_native(d))
            / sum_pairs max(w_model(d), w_native(d))

    where w(d) = exp(-(d - 4)^2 / 2.25).

    Returns (weighted_jaccard_float, error_str_or_None).  Two empty pair
    maps mean neither structure has an interface, giving (NaN, None).
    Exactly one empty gives 0.0, and so does a zero denominator.
    """
    try:
        import gemmi
    except ImportError as e:
        return None, f"gemmi_import_failed:{e}"
    try:
        m_st = gemmi.read_structure(str(model_pdb))
        n_st = gemmi.read_structure(str(native_pdb))
    except Exception as e:  # noqa: BLE001
        return None, f"structure_load_failed:{type(e).__name__}"

    try:
        model_pairs = _build_weighted_pair_map(
            m_st, receptor_chain, effector_chain, pair_cutoff,
        )
        native_pairs = _build_weighted_pair_map(
            n_st, receptor_chain, effector_chain, pair_cutoff,
        )
    except Exception as e:  # noqa: BLE001
        return None, f"pair_map_failed:{type(e).__name__}"

    if not model_pairs and not native_pairs:
        return float("nan"), None
    if not model_pairs or not native_pairs:
        return 0.0, None

    union = set(model_pairs) | set(native_pairs)
    num = 0.0
    den = 0.0
    for k in union:
        wm = model_pairs.get(k, 0.0)
        wn = native_pairs.get(k, 0.0)
        num += min(wm, wn)
        den += max(wm, wn)
    if den == 0.0:
        return 0.0, None
    return round(num / den, 4), None


def compute_intact_core(model_pdb, native_pdb, plddt_threshold,
                         intact_threshold, receptor_chain, effector_chain):
    """
    Core-only intact filter.

    Trims the receptor by per-residue pLDDT, Kabsch-aligns the trimmed
    cores of model and native, then computes the whole-complex Cα RMSD
    over the residues they share.  Returns
    (intact_flag, rmsd_angstroms, error_str_or_None), intact_flag 1 when
    that RMSD is below intact_threshold.
    """
    try:
        import gemmi
    except ImportError as e:
        return None, None, f"gemmi_import_failed:{e}"

    try:
        model = gemmi.read_structure(str(model_pdb))
        native = gemmi.read_structure(str(native_pdb))
    except Exception as e:  # noqa: BLE001
        return None, None, f"structure_load_failed:{type(e).__name__}"

    def chain_ca_records(st, chain_id):
        for m in st:
            for ch in m:
                if ch.name != chain_id:
                    continue
                for res in ch:
                    for atom in res:
                        if atom.name == "CA":
                            yield (res.seqid.num, np.array([
                                atom.pos.x, atom.pos.y, atom.pos.z,
                            ]), atom.b_iso)
                            break

    try:
        model_recs = {num: (xyz, bf) for (num, xyz, bf)
                      in chain_ca_records(model, receptor_chain)}
        native_recs = {num: (xyz, bf) for (num, xyz, bf)
                       in chain_ca_records(native, receptor_chain)}
    except Exception as e:  # noqa: BLE001  # pragma: no cover - chain_ca_records only walks gemmi objects the read_structure above already loaded, so nothing in it raises
        return None, None, f"chain_parse_failed:{type(e).__name__}"

    common = sorted(set(model_recs) & set(native_recs))
    kept = [r for r in common
            if model_recs[r][1] is not None
            and model_recs[r][1] >= plddt_threshold]
    if len(kept) < 3:
        return None, None, f"too_few_core_residues:{len(kept)}"

    M = np.array([model_recs[r][0] for r in kept])
    N = np.array([native_recs[r][0] for r in kept])
    M_c = M - M.mean(axis=0)
    N_c = N - N.mean(axis=0)
    H = M_c.T @ N_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    # Whole-complex Cα set for RMSD.
    model_full = {(receptor_chain, num): xyz for (num, xyz, _bf)
                  in chain_ca_records(model, receptor_chain)}
    model_full.update({(effector_chain, num): xyz for (num, xyz, _bf)
                       in chain_ca_records(model, effector_chain)})
    native_full = {(receptor_chain, num): xyz for (num, xyz, _bf)
                   in chain_ca_records(native, receptor_chain)}
    native_full.update({(effector_chain, num): xyz for (num, xyz, _bf)
                        in chain_ca_records(native, effector_chain)})

    shared = sorted(set(model_full) & set(native_full))
    if len(shared) < 3:  # pragma: no cover - kept is a subset of shared and len(kept) >= 3 was asserted above
        return None, None, f"too_few_shared_residues:{len(shared)}"

    M_all = np.array([model_full[k] for k in shared])
    N_all = np.array([native_full[k] for k in shared])
    M_aligned = (M_all - M.mean(axis=0)) @ R.T + N.mean(axis=0)
    diff = M_aligned - N_all
    rmsd = float(np.sqrt((diff * diff).sum(axis=1).mean()))
    return (1 if rmsd < intact_threshold else 0), round(rmsd, 3), None


def compute_interface_plddt(model_pdb, receptor_chain, effector_chain,
                             contact_cutoff=5.0):
    """
    Mean Cα pLDDT over the interface residues, moved here from
    run_biophysical_metrics.py so it runs per prediction.

    Returns (interface_plddt_0_to_1, error_str_or_None).  Boltz writes
    pLDDT on 0-100 in the PDB B-factor column while downstream
    thresholds expect 0-1, so input on 0-100 is normalised.
    """
    try:
        import gemmi
    except ImportError as e:
        return None, f"gemmi_import_failed:{e}"

    try:
        st = gemmi.read_structure(str(model_pdb))
    except Exception as e:  # noqa: BLE001
        return None, f"structure_load_failed:{type(e).__name__}"

    rec_atoms = []
    eff_atoms = []
    for m in st:
        for ch in m:
            dest = (rec_atoms if ch.name == receptor_chain
                    else eff_atoms if ch.name == effector_chain
                    else None)
            if dest is None:
                continue
            for res in ch:
                for atom in res:
                    if atom.element.name == "H":
                        continue
                    dest.append((res.seqid.num,
                                 np.array([atom.pos.x, atom.pos.y, atom.pos.z])))
        break

    if not rec_atoms or not eff_atoms:
        return None, "chain_not_found"

    rec_coords = np.array([xyz for _, xyz in rec_atoms])
    eff_coords = np.array([xyz for _, xyz in eff_atoms])
    rec_resnums = np.array([n for n, _ in rec_atoms])
    eff_resnums = np.array([n for n, _ in eff_atoms])

    interface = set()
    chunk = 2048
    for i in range(0, len(rec_coords), chunk):
        r_block = rec_coords[i:i + chunk]
        d = np.linalg.norm(
            r_block[:, None, :] - eff_coords[None, :, :], axis=-1
        )
        hits_r, hits_e = np.where(d < contact_cutoff)
        for ri, ei in zip(hits_r, hits_e):
            interface.add((receptor_chain, int(rec_resnums[i + ri])))
            interface.add((effector_chain, int(eff_resnums[ei])))

    if not interface:
        return None, "no_interface_residues"

    plddts = []
    for m in st:
        for ch in m:
            for res in ch:
                if (ch.name, res.seqid.num) not in interface:
                    continue
                for atom in res:
                    if atom.name == "CA":
                        plddts.append(atom.b_iso)
                        break
        break

    if not plddts:
        return None, "no_interface_cas"

    mean_plddt = float(np.mean(plddts))
    if mean_plddt > 1.5:
        mean_plddt /= 100.0
    return round(mean_plddt, 4), None


# ═══════════════════════════════════════════════════════════════════════════
# Path helpers
# ═══════════════════════════════════════════════════════════════════════════

def _sidecar_index(pred_dir):
    """Map every filename under pred_dir to its paths, shallowest first.

    parse_boltz2 looks up three sidecars per PDB: confidence_<name>.json,
    plddt_<name>.npz and pae_<name>.npz.  Each lookup used to be three
    globs, one of which walked the whole of pred_dir recursively, so the
    tree was walked three times per PDB.  At the benchmark's 25 models,
    5 seeds x 5 diffusion samples, that is 75 walks of one directory.
    Walking once and indexing by filename gives the same answer.
    """
    index = {}
    for dirpath, _dirnames, filenames in os.walk(pred_dir):
        for nm in filenames:
            index.setdefault(nm, []).append(os.path.join(dirpath, nm))
    for paths in index.values():
        paths.sort(key=lambda p: (p.count(os.sep), p))
    return index


def _sidecar(index, pdb_dir, name, kind, ext):
    """The best path for one sidecar, or None.

    `kind` is both the filename prefix and the sibling directory name,
    which Boltz keeps the same: confidence_<name>.json sits either beside
    the PDB or in a sibling `confidence/`.

    Preference order is the one the three chained globs had, because the
    caller takes the first hit: beside the PDB, then the sibling category
    directory, then the shallowest path anywhere in the tree.  normpath on
    both sides is what makes the sibling case match, since
    `a/pdb/../confidence/x` and `a/confidence/x` are the same file under
    different strings.
    """
    filename = f"{kind}_{name}.{ext}"
    candidates = index.get(filename, ())
    if not candidates:
        return None
    here = os.path.join(pdb_dir, filename)
    sibling = os.path.normpath(os.path.join(pdb_dir, "..", kind, filename))
    for preferred in (here, sibling):
        for c in candidates:
            if os.path.normpath(c) == preferred:
                return c
    return candidates[0]


def _is_reference(path):
    """True for a copied reference PDB, which the parser skips."""
    return os.path.basename(path) == "reference.pdb"


# ═══════════════════════════════════════════════════════════════════════════
# Boltz-2 parser
# ═══════════════════════════════════════════════════════════════════════════

def parse_boltz2(
    pred_dir,
    chain_lengths,
    pae_cutoff=10.0,
    receptor_chain="A",
    effector_chain="B",
    mutated_positions=None,
    contact_cutoff=5.0,
    effector_atom_filter=None,
    native_pdb=None,
    intact_plddt_threshold=50.0,
    intact_rmsd_threshold=5.0,
    weighted_jaccard_pair_cutoff=8.0,
):
    """
    Parse Boltz-2 outputs.

    `pred_dir` must be the directory holding the sidecar files, not a
    parent directory holding a copied prediction.pdb.  The expected
    layout is:
        output/predictions/<job>/pdb/<n>_model_<M>.pdb
        output/predictions/<job>/pae/<n>_model_<M>.npz
        output/predictions/<job>/confidence/<n>_model_<M>.json
        output/predictions/<job>/plddt/<n>_model_<M>.npz
    Some Boltz versions write flatter layouts, and the globs below
    cover those too.

    Supplying `mutated_positions`, the design's cumulative steering
    mutations as 1-based receptor residue numbers, turns on the
    interface contact analysis and the mutation-reliance flag.  None or
    empty leaves that column blank, which suits cold-start rows.

    `effector_atom_filter` restricts contact detection to the effector's
    interface atoms.  See compute_effector_interface_residues and
    compute_interface_contacts.
    """
    if mutated_positions is None:
        mutated_positions = []
    results = []

    pdb_files = sorted(glob.glob(os.path.join(pred_dir, "**", "*.pdb"), recursive=True))
    pdb_files = [p for p in pdb_files if not _is_reference(p)]

    if not pdb_files:
        print(f"  WARNING: No PDB files found in {pred_dir}")
        return results

    sidecars = _sidecar_index(pred_dir)

    for pdb_path in pdb_files:
        name = os.path.basename(pdb_path).replace(".pdb", "")
        pdb_dir = os.path.dirname(pdb_path)
        entry = {"pdb_path": pdb_path, "model_name": name}

        # --- Confidence JSON ---
        conf_file = _sidecar(sidecars, pdb_dir, name, "confidence", "json")
        if conf_file:
            try:
                with open(conf_file) as f:
                    conf = json.load(f)
                entry["ptm"] = round(float(conf.get("ptm", 0.0)), 4)
                entry["iptm"] = round(float(
                    conf.get("iptm", conf.get("protein_iptm", 0.0))
                ), 4)
                # complex_plddt is the whole-complex pLDDT on a 0-1
                # scale.  Its own column makes the Boltz-native signal
                # available without re-averaging the per-residue .npz.
                if "complex_plddt" in conf:
                    entry["complex_plddt"] = round(
                        float(conf["complex_plddt"]), 4
                    )
            except Exception as e:
                print(f"  WARNING: conf parse failed for {name}: {e}")

        # --- pLDDT ---
        plddt_file = _sidecar(sidecars, pdb_dir, name, "plddt", "npz")
        if plddt_file:
            try:
                data = np.load(plddt_file)
                arr = data[list(data.keys())[0]]
                if arr.ndim > 1:
                    arr = arr.flatten()
                mean_plddt = float(arr.mean())
                # Boltz-2 writes the per-residue .npz on a 0-1 scale,
                # while the benchmark pipeline and plddt_from_pdb's
                # B-factor fallback use 0-100, so rescaling here puts
                # every output row on 0-100.  The guard tests max, not
                # the mean, so a file already on 0-100 is not doubled.
                if float(arr.max()) <= 1.5:
                    mean_plddt *= 100.0
                entry["avg_plddt"] = round(mean_plddt, 2)
            except Exception as e:
                print(f"  WARNING: plddt parse failed for {name}: {e}")
        if "avg_plddt" not in entry:
            entry["avg_plddt"] = plddt_from_pdb(pdb_path)

        # --- PAE matrix ---
        pae_file = _sidecar(sidecars, pdb_dir, name, "pae", "npz")
        pae_matrix = None
        if pae_file:
            try:
                data = np.load(pae_file)
                for key in data.keys():
                    arr = np.array(data[key])
                    if arr.ndim == 2:
                        pae_matrix = arr
                        break
                    elif arr.ndim == 3:
                        pae_matrix = arr[0]
                        break
            except Exception as e:
                print(f"  WARNING: pae parse failed for {name}: {e}")

        _finalize_entry(entry, pae_matrix, chain_lengths, pae_cutoff=pae_cutoff)

        # --- Interface contacts + mutation reliance ---
        # A failure here, from missing chain letters or a parse error,
        # falls back to an empty set and a blank flag, so the row still
        # gets written.
        try:
            contacts = compute_interface_contacts(
                pdb_path,
                receptor_chain=receptor_chain,
                effector_chain=effector_chain,
                cutoff=contact_cutoff,
                effector_atom_filter=effector_atom_filter,
            )
        except Exception as e:
            print(f"  WARNING: contact analysis failed for {name}: {e}")
            contacts = set()

        entry["n_contact_residues"] = len(contacts)
        entry["contact_residues"] = ",".join(str(r) for r in sorted(contacts))
        n_on_mut, mutated_contact_positions = classify_mutation_reliance(
            contacts, mutated_positions
        )
        entry["n_contacts_on_mutated_positions"] = n_on_mut
        entry["mutated_contact_positions"] = ",".join(
            str(p) for p in mutated_contact_positions
        )
        entry["contact_cutoff_used"] = contact_cutoff

        # --- interface_plddt (needs the model PDB only) ---
        ifp_val, ifp_err = compute_interface_plddt(
            pdb_path,
            receptor_chain=receptor_chain,
            effector_chain=effector_chain,
            contact_cutoff=contact_cutoff,
        )
        entry["interface_plddt"] = (
            f"{ifp_val:.4f}" if ifp_val is not None else ""
        )

        # --- intact_core + weighted_jaccard (need native_pdb) ---
        # A None native_pdb leaves both columns blank.  Computing them
        # in this per-prediction loop sends the values down the same
        # aggregator, passing_summary and cross_summary chain as the
        # PAE metrics.
        if native_pdb is not None:
            ic_flag, _ic_rmsd, _ic_err = compute_intact_core(
                pdb_path, native_pdb,
                plddt_threshold=intact_plddt_threshold,
                intact_threshold=intact_rmsd_threshold,
                receptor_chain=receptor_chain,
                effector_chain=effector_chain,
            )
            entry["intact_core"] = "" if ic_flag is None else str(ic_flag)

            wj_val, _wj_err = compute_weighted_jaccard(
                pdb_path, native_pdb,
                receptor_chain=receptor_chain,
                effector_chain=effector_chain,
                pair_cutoff=weighted_jaccard_pair_cutoff,
            )
            if wj_val is None:
                entry["weighted_jaccard"] = ""
            elif isinstance(wj_val, float) and np.isnan(wj_val):
                entry["weighted_jaccard"] = ""
            else:
                entry["weighted_jaccard"] = f"{wj_val:.4f}"
        else:
            entry["intact_core"] = ""
            entry["weighted_jaccard"] = ""

        results.append(entry)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Parser registry + CSV schema
# ═══════════════════════════════════════════════════════════════════════════

PARSERS = {
    "boltz2": parse_boltz2,
    # boltz1 uses the same output layout as boltz2.
    "boltz1": parse_boltz2,
}

CSV_FIELDS = [
    "model",
    "model_name",
    "pdb_path",
    "avg_plddt",
    "complex_plddt",
    "ptm",
    "iptm",
    "pae_mean",
    "ipae",
    "pae_pass_frac",
    "ipsae_ab",
    "ipsae_ba",
    "ipsae_min",
    "ipsae_ab_15",
    "ipsae_ba_15",
    "ipsae_min_15",
    "actifptm",
    "af_rank_score",
    # Interface contacts and mutation reliance, filled in for every
    # parsed PDB.  mutated_contact_positions is blank when
    # --mutated-positions was not supplied.  See
    # classify_mutation_reliance for the contamination rule.
    "n_contact_residues",
    "contact_residues",
    "n_contacts_on_mutated_positions",
    "mutated_contact_positions",
    "contact_cutoff_used",
    # intact_core and weighted_jaccard need --native-pdb and stay blank
    # without it.  interface_plddt needs the model PDB only.
    "interface_plddt",
    "intact_core",
    "weighted_jaccard",
]


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compute confidence metrics from Boltz-2 structure "
                    "prediction outputs (minimal build)",
    )
    parser.add_argument("--model", required=True, choices=list(PARSERS.keys()))
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--chain-lengths", nargs="+", type=int, required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--pae-cutoff", type=float, default=10.0,
                        help="PAE cutoff (Å) for ipSAE.  Residue pairs "
                             "with inter-chain PAE strictly less than "
                             "this contribute to ipSAE; pairs at or "
                             "above are excluded.  Default 10 follows "
                             "Dunbrack 2025; 15 is also reasonable for "
                             "permissive scoring.  pae_pass_frac in the "
                             "output reports what fraction of pairs "
                             "passed at the cutoff used.")
    # --receptor-chain and --effector-chain were once accepted and
    # ignored.  The interface contact analysis now uses them, and the
    # defaults match plan.json's pred_receptor_chain and
    # pred_effector_chain.
    parser.add_argument("--receptor-chain", default="A",
                        help="Receptor chain ID in the prediction PDB "
                             "(used by the interface contact analysis). "
                             "Default: A.")
    parser.add_argument("--effector-chain", default="B",
                        help="Effector chain ID in the prediction PDB "
                             "(used by the interface contact analysis). "
                             "Default: B.")
    parser.add_argument("--mutated-positions", default="",
                        help="Comma-separated 1-based residue numbers "
                             "on the receptor chain that were altered "
                             "by the steering pipeline, e.g. "
                             "'5,26,44,46'.  Also accepts the ChimeraX "
                             "form '/A:5,26,44,46'.  When supplied, the "
                             "output CSV's mutated_contact_positions "
                             "column lists the positions in this set that "
                             "contact the effector, and "
                             "n_contacts_on_mutated_positions counts "
                             "them.  Any such position means the pose "
                             "depends on a residue the wet-lab construct "
                             "will not carry.  Blank or unset leaves the "
                             "position list empty, which is right for a "
                             "cold start because it has no mutations.")
    parser.add_argument("--contact-cutoff", type=float, default=5.0,
                        help="Heavy-atom distance cutoff (Å) for the "
                             "interface contact definition.  A receptor "
                             "residue is counted as a contact if any "
                             "of its heavy atoms is strictly closer "
                             "than this to any effector heavy atom.  "
                             "Default: 5.0.")
    parser.add_argument("--effector-atom-filter-json", default="",
                        help="Path to a JSON file containing the "
                             "effector interface-atom set, as a list "
                             "of [resseq, atom_name] pairs.  When "
                             "supplied, contact detection is "
                             "restricted to these effector atoms only, "
                             "so a brush against a non-binding face does "
                             "not register as a contact.  The steering "
                             "pipeline writes the set to plan.json under "
                             "'effector_interface_residues', or under the "
                             "older 'effector_interface_atoms' key.  Point "
                             "this flag at plan.json and it reads whichever "
                             "key is there.  Left unset, every effector "
                             "atom counts.")
    parser.add_argument("--native-pdb", default="",
                        help="Path to the ground-truth (native) PDB for "
                             "this design.  When supplied, the per-PDB "
                             "loop also computes intact_core (Kabsch-"
                             "aligned receptor-core RMSD vs native) and "
                             "weighted_jaccard (Gaussian-weighted Cβ-Cβ "
                             "contact overlap vs native).  Blank / unset "
                             "→ both columns are emitted blank.")
    parser.add_argument("--intact-plddt-threshold", type=float, default=50.0,
                        help="Per-residue pLDDT cutoff for the receptor "
                             "core trim used by intact_core.  Default 50.")
    parser.add_argument("--intact-rmsd-threshold", type=float, default=5.0,
                        help="RMSD cutoff (Å) for the intact_core flag. "
                             "Whole-complex Cα RMSD < this on the "
                             "Kabsch-aligned receptor core → intact_core=1, "
                             "else 0.  Default 5.0.")
    parser.add_argument("--weighted-jaccard-pair-cutoff", type=float, default=8.0,
                        help="Cβ-Cβ distance (Å) beyond which residue "
                             "pairs are dropped from the weighted_jaccard "
                             "computation.  Pairs at d ≫ 4 carry weight "
                             "≈ 0 anyway, so this just keeps the union "
                             "denominator from being inflated with zero-"
                             "summands.  Default 8.0.")
    args = parser.parse_args()

    print(f"=== Computing metrics for {args.model} ===")
    print(f"Prediction dir: {args.prediction_dir}")
    print(f"Chain lengths:  {args.chain_lengths}")

    if not os.path.exists(args.prediction_dir):
        print(f"ERROR: Prediction directory not found: {args.prediction_dir}")
        sys.exit(1)

    parse_fn = PARSERS[args.model]

    mutated_positions = parse_mutated_positions(args.mutated_positions)
    if mutated_positions:
        print(f"Mutated positions: {mutated_positions} (on chain {args.receptor_chain})")

    print(f"Contact cutoff:  {args.contact_cutoff} Å heavy-atom")

    # The JSON top level may be a list of ints, a list of
    # [resseq, atom_name] pairs, or a plan.json dict read from
    # 'effector_interface_residues' or the older
    # 'effector_interface_atoms'.
    effector_atom_filter = None
    if args.effector_atom_filter_json:
        try:
            with open(args.effector_atom_filter_json) as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                pairs = (raw.get("effector_interface_residues")
                         or raw.get("effector_interface_atoms")
                         or [])
            elif isinstance(raw, list):
                pairs = raw
            else:
                pairs = []
            flt = set()
            for p in pairs:
                if isinstance(p, int):
                    flt.add(p)
                elif isinstance(p, (list, tuple)) and len(p) >= 1:
                    try:
                        flt.add(int(p[0]))
                    except (TypeError, ValueError):
                        pass
            effector_atom_filter = flt if flt else None
            if effector_atom_filter:
                print(f"Effector interface residue filter: "
                      f"{len(effector_atom_filter)} residues "
                      f"(contact detection restricted to the interface)")
            else:
                print("Effector filter: loaded but empty — "
                      "contamination check runs on all effector atoms "
                      "(legacy behaviour)")
        except Exception as e:
            print(f"WARNING: could not load effector filter "
                  f"from {args.effector_atom_filter_json}: {e}")
            print("  Contamination check will run on all effector atoms.")
            effector_atom_filter = None

    # None if blank or missing from disk.
    native_pdb_arg = args.native_pdb.strip() if args.native_pdb else ""
    native_pdb_path = None
    if native_pdb_arg:
        if os.path.exists(native_pdb_arg):
            native_pdb_path = native_pdb_arg
            print(f"Native PDB:      {native_pdb_arg} "
                  f"(intact_core + weighted_jaccard ON)")
        else:
            print(f"WARNING: --native-pdb path does not exist: "
                  f"{native_pdb_arg} — intact_core + weighted_jaccard "
                  f"will be blank")

    try:
        results = parse_fn(args.prediction_dir, args.chain_lengths,
                           pae_cutoff=args.pae_cutoff,
                           receptor_chain=args.receptor_chain,
                           effector_chain=args.effector_chain,
                           mutated_positions=mutated_positions,
                           contact_cutoff=args.contact_cutoff,
                           effector_atom_filter=effector_atom_filter,
                           native_pdb=native_pdb_path,
                           intact_plddt_threshold=args.intact_plddt_threshold,
                           intact_rmsd_threshold=args.intact_rmsd_threshold,
                           weighted_jaccard_pair_cutoff=args.weighted_jaccard_pair_cutoff)
    except Exception as e:
        print(f"ERROR: Parser failed for {args.model}: {e}")
        traceback.print_exc()
        results = []

    if not results:
        print(f"WARNING: No predictions found for {args.model}")
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w") as f:
            f.write(",".join(CSV_FIELDS) + "\n")
        return

    for r in results:
        r["model"] = args.model

    # Rank by actifptm, best first.
    results.sort(key=lambda x: x.get("actifptm", 0.0), reverse=True)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nWrote {len(results)} entries to {args.output_csv}")

    best = results[0]
    print(f"\nBest model ({args.model}): {best.get('model_name', 'unknown')}")
    for k in ("avg_plddt", "complex_plddt", "ptm", "iptm",
              "pae_mean", "ipae", "pae_pass_frac",
              "ipsae_ab", "ipsae_ba", "ipsae_min",
              "ipsae_ab_15", "ipsae_ba_15", "ipsae_min_15",
              "actifptm", "af_rank_score",
              "interface_plddt", "intact_core", "weighted_jaccard"):
        print(f"  {k:14s} = {best.get(k, 'N/A')}")


if __name__ == "__main__":
    main()
