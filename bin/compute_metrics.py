#!/usr/bin/env python3
"""
compute_metrics.py  (minimal boltz2-only build)
-----------------------------------------------
Compute confidence metrics from Boltz-2 structure prediction outputs.

This is a cut-down build of the full benchmark-pipeline compute_metrics.py,
containing only what `negsteer_aggregate.py compute-final-metrics`
needs: the Boltz-2 parser and the confidence / PAE-derived metrics.

All structural-RMSD code (Kabsch fitting, sequence alignment, reference
comparison, matchmaker-style iterative pruning) has been removed — the
steering pipeline already writes authoritative per-design RMSDs to
`all_results_multicycle.csv` using the correct truth-vs-pred chain
mapping, so there is no need to recompute them here.

Metrics computed:
    - avg_plddt:     Average per-residue pLDDT (0-100 scale)
    - ptm:           Predicted TM-score (global) — from Boltz confidence.json
    - iptm:          Interface predicted TM-score — from Boltz confidence.json
    - pae_mean:      Mean predicted aligned error (full matrix)
    - ipae:          Interface PAE (mean of both inter-chain PAE blocks)
    - ipsae_ab:      ipSAE chain A->B (Dunbrack 2025, TM-score d0)
    - ipsae_ba:      ipSAE chain B->A
    - ipsae_min:     min(ipsae_ab, ipsae_ba)
    - actifptm:      actifPTM (Varga, Ovchinnikov, Schueler-Furman 2025) —
                     pTM kernel restricted to residues with ≥1 inter-chain
                     PAE below cutoff; d0 from interacting set size
    - af_rank_score: 0.8 * iptm + 0.2 * ptm (AF-Multimer ranking score).
                     Prior to the v8 patch this was being written out
                     mislabelled as 'actifptm'; kept under its correct
                     name for continuity.

Usage:
    python compute_metrics.py \\
        --model boltz2 \\
        --prediction-dir <dir containing one or more *.pdb files> \\
        --chain-lengths <rec_len> <eff_len> \\
        --output-csv metrics.csv
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

    Raw formula: d0 = 1.24 * (n - 15)^(1/3) - 1.8
    Clamped: we return max(d0, 0.5) — matching the TM-align
    convention that d0 is never less than 0.5 Å.  The raw formula
    dips below 0.5 for n in roughly the 15-28 range, and without
    clamping the pTM kernel blows up for any PAE ≥ ~1 Å in that
    range (score → 0 from a denominator explosion).

    `n` here is the number of inter-chain residues of the opposite
    chain that fell below the PAE cutoff for this reference residue
    (i.e. the "interacting set size" for this residue), NOT the
    total chain length.  Using the interacting set size is what
    distinguishes ipSAE from a global pTM-style metric.
    """
    if n < 16:
        return 0.5
    d0 = 1.24 * ((n - 15) ** (1.0 / 3.0)) - 1.8
    return max(d0, 0.5)


def compute_ipsae_one_direction(pae_ij, cutoff=10.0):
    """ipSAE for direction i->j (Dunbrack 2025).

    For each residue i in the reference chain, take its PAE row
    against every residue j in the partner chain.  Let n_below be
    the count of j's with PAE_ij < cutoff (the interacting set size
    for i).  Compute d0 from n_below via _ipsae_d0, then the
    per-residue score is:

        s_i = mean over j-with-PAE<cutoff of  1 / (1 + (PAE_ij/d0)^2)

    The ipSAE is the MAXIMUM s_i over all reference-chain residues.
    Returns (score, best_residue_idx).

    If no residue in the reference chain has any partner-chain
    neighbour below the PAE cutoff, the score is 0.0 and the
    best_residue_idx is -1.
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
        if d0 <= 0:
            scores.append(0.0)
            continue
        s = 1.0 / (1.0 + (below / d0) ** 2)
        scores.append(float(s.mean()))
    if not scores:
        return 0.0, -1
    return max(scores), int(np.argmax(scores))


def _pae_to_ptm_score(pae_ij, n_interface):
    """Per-residue pTM-style score given a PAE row-slice and the
    effective N used in the TM-score d0 formula.

    Returns an array of per-row scores (one per reference residue),
    each being the mean of 1 / (1 + (PAE/d0)^2) over the partner
    columns provided.  The score is the pTM kernel.

    `n_interface` is the size used in the d0 formula.  In vanilla
    pTM this is the total chain length; in ipTM (AF-Multimer) it
    is the inter-chain partner length; in actifPTM (Varga 2025) it
    is the count of interacting residues (those with ≥1 inter-chain
    PAE pair below the PAE cutoff).
    """
    if pae_ij.size == 0 or n_interface < 1:
        return np.zeros(pae_ij.shape[0], dtype=float)
    d0 = _ipsae_d0(n_interface)
    if d0 <= 0:
        return np.zeros(pae_ij.shape[0], dtype=float)
    kernel = 1.0 / (1.0 + (pae_ij / d0) ** 2)
    if kernel.ndim == 1:
        return kernel
    return kernel.mean(axis=1)


def compute_actifptm(pae_matrix, chain_lengths, cutoff=10.0):
    """actifPTM (Varga, Ovchinnikov, Schueler-Furman 2025).

    actifPTM is ipTM restricted to the "interacting residue set" —
    residues that have at least one inter-chain PAE entry below
    the PAE cutoff.  The d0 in the pTM kernel is computed from the
    SIZE OF THE INTERACTING SET, not the full chain length.

    Procedure (for a two-chain complex A, B):

      1. Identify interacting residues in chain A as those with
         ≥1 entry in the PAE[A->B] block below `cutoff`.  Let
         n_int_A be the count.
      2. Symmetrically, identify interacting residues in chain B
         via PAE[B->A], let n_int_B be the count.
      3. Form the restricted PAE sub-block PAE[interacting_A,
         interacting_B] (and the transpose for B->A).
      4. For each interacting residue i in A, score:
            s_i = mean over interacting j in B of
                  1 / (1 + (PAE_ij / d0(n_int_B))^2)
         and similarly s_j for interacting residues in B.
      5. actifPTM_A->B = max over interacting i in A of s_i.
         actifPTM_B->A = max over interacting j in B of s_j.
         actifPTM = max(actifPTM_A->B, actifPTM_B->A).

    When no residue has any inter-chain PAE below cutoff, the
    interacting set is empty and actifPTM is 0.0 by convention.

    Returns a float on [0, 1].
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

    # Interacting residue masks: residues with ≥1 inter-chain neighbour
    # whose PAE is below the cutoff.
    int_a_mask = (pae_ab < cutoff).any(axis=1)
    int_b_mask = (pae_ba < cutoff).any(axis=1)
    n_int_a = int(int_a_mask.sum())
    n_int_b = int(int_b_mask.sum())
    if n_int_a == 0 or n_int_b == 0:
        return 0.0

    # Restrict the PAE blocks to the interacting sets.
    # actifPTM_A->B: for each interacting i in A, mean score over
    # interacting j in B, with d0 computed from n_int_B (the size
    # of the partner interacting set, matching Varga's definition).
    pae_ab_sub = pae_ab[int_a_mask][:, int_b_mask]
    pae_ba_sub = pae_ba[int_b_mask][:, int_a_mask]

    scores_ab = _pae_to_ptm_score(pae_ab_sub, n_int_b)
    scores_ba = _pae_to_ptm_score(pae_ba_sub, n_int_a)

    best_ab = float(scores_ab.max()) if scores_ab.size else 0.0
    best_ba = float(scores_ba.max()) if scores_ba.size else 0.0

    return round(max(best_ab, best_ba), 4)


def compute_ipsae(pae_matrix, chain_lengths, cutoff=10.0):
    """Compute ipSAE from full pAE matrix and chain length list."""
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
    cutoff.  Diagnoses how much of the interface ipSAE is actually
    being computed over: a value of 0.50 means half of all inter-chain
    residue pairs passed the threshold and contributed to the score,
    a value of 0.05 means only 5% did and ipSAE is summarising a
    very thin slice of the interface.
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

    iPSAE is computed at TWO PAE cutoffs in the same pass:
    - cutoff (default 10 Å): the canonical Dunbrack 2025 default,
      surfaced as ipsae_ab / ipsae_ba / ipsae_min.
    - 15 Å: a more permissive cutoff useful for borderline poses,
      surfaced as ipsae_ab_15 / ipsae_ba_15 / ipsae_min_15.
    Both are essentially free here — the PAE matrix is already loaded
    and compute_ipsae's cost is dominated by matrix masking.  Computing
    both upfront avoids the whole second-pass-from-disk problem the
    earlier orthogonal-metrics implementation hit.
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
    Hydrogens (element H) are excluded.  Only ATOM records are read
    (HETATM is skipped).  Residue number is 1-based as it appears in
    the PDB (we do NOT renumber).  Insertion codes and altlocs other
    than '' / 'A' are ignored to keep residue identity stable.
    """
    by_chain = {}
    try:
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                # PDB fixed-column parsing (columns 1-based in the spec;
                # slicing is 0-based and end-exclusive).
                altloc = line[16:17]
                if altloc not in (" ", "", "A"):
                    continue
                element = line[76:78].strip()
                if element == "H":
                    continue
                # Fallback: if element column is blank, infer from atom
                # name; names starting with H are hydrogens.
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
    """Identify the set of effector residue numbers that constitute
    the "interface effector residues" — residues whose Cα is within
    `cutoff_angstroms` of any receptor residue in
    `true_interface_residues_1b` (Cα-based) in the ground truth.

    This defines the subset of effector RESIDUES that are considered
    to be on the functional binding surface.  Contact detection in
    downstream predictions is then restricted to heavy atoms of these
    residues, so that off-interface contacts (e.g. a floppy effector
    loop brushing against a non-binding receptor face) do not factor
    into the contamination check.

    Residue-level filtering (rather than atom-level) is required
    because the RFDiffusion ground truth is Cα-only: an atom-level
    mask built from a Cα-only ground truth captures only Cα atoms,
    which almost never produces heavy-to-heavy contacts at 5 Å in a
    predicted structure with full side chains.  A residue-level
    mask passes all heavy atoms of the interface residues through
    from the prediction, preserving sensitivity.

    Cutoff defaults to 8.0 Å, matching the RFDiffusion filter's
    Cα-distance convention for `true_interface_idx`.

    Returns a set of 1-based POSITIONAL indices into the effector
    chain (matching the order residues appear in the prediction PDB,
    which renumbers chains starting from 1).  This coordinate system
    is critical: the ground-truth PDB may use a synthetic resseq
    scheme (RFDiffusion offsets by 1000+ from the max fixed resnum),
    but Boltz predictions always number chains 1-based positional.
    A filter in GT-resseq space would match nothing in the prediction.

    Returns empty set on any failure — callers should treat empty as
    "no filter" and fall back to unrestricted atoms.
    """
    if not ground_truth_pdb:
        return set()
    true_set_1b = set(int(i) for i in (true_interface_residues_1b or []))
    if not true_set_1b:
        return set()

    # Collect Cα positions per chain, in file order (= sequence order).
    # For the effector chain, also remember each residue's 1-based
    # positional index (i.e. its rank in file order, starting at 1)
    # — that's the coordinate system the prediction uses.
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

    # Receptor interface Cα positions (true_interface_residues_1b are
    # 1-based positional indices into the receptor chain in file order).
    iface_ca = []
    for pos_1b in sorted(true_set_1b):
        idx = pos_1b - 1
        if 0 <= idx < len(rec_ca):
            iface_ca.append(rec_ca[idx])
    if not iface_ca:
        return set()
    iface_arr = np.array(iface_ca, dtype=np.float64)

    # Any effector residue whose Cα sits within cutoff of any true-
    # interface receptor Cα is part of the interface residue set.
    # Return its 1-based positional index (NOT its ground-truth resseq).
    cutoff2 = float(cutoff_angstroms) ** 2
    interface_residues = set()
    for pos_1b, x, y, z in eff_ca:
        diffs = iface_arr - np.array([x, y, z], dtype=np.float64)
        d2 = (diffs * diffs).sum(axis=1)
        if (d2 < cutoff2).any():
            interface_residues.add(pos_1b)

    return interface_residues


# Legacy name kept as a thin wrapper so other modules that still
# import compute_effector_interface_atom_mask keep working during
# the transition.  Callers should migrate to the residue-level
# function above.
def compute_effector_interface_atom_mask(
    ground_truth_pdb,
    true_interface_residues_1b,
    receptor_chain,
    effector_chain,
    cutoff_angstroms=8.0,
):
    """Deprecated alias returning a dict keyed by (resseq, atom_name)
    for backward compatibility.  The underlying logic now only uses
    residue-level filtering, so this wrapper expands the residue set
    to a "match everything with this resseq" set by using a sentinel
    atom_name of None.  Prefer compute_effector_interface_residues.
    """
    residues = compute_effector_interface_residues(
        ground_truth_pdb, true_interface_residues_1b,
        receptor_chain, effector_chain, cutoff_angstroms,
    )
    return {(r, None): (0.0, 0.0, 0.0) for r in residues}


def compute_interface_contacts(
    pdb_path,
    receptor_chain,
    effector_chain,
    cutoff=5.0,
    effector_atom_filter=None,
):
    """Return the set of 1-based receptor residue numbers whose heavy
    atoms come within `cutoff` Å of any effector heavy atom.

    Uses the PDB chain IDs as-is: receptor_chain / effector_chain must
    match the letters present in the file (for Boltz2 predictions this
    is typically "A" / "B", set by plan.json's pred_*_chain).

    If `effector_atom_filter` is supplied, it must be a set of
    (effector_resseq_1b, atom_name) pairs identifying the subset of
    effector atoms that count as "interface atoms".  Contacts between
    receptor residues and effector atoms OUTSIDE this set are
    ignored.  This is the v8 Level 1 contamination-check fix:
    contacts with off-interface effector atoms (e.g. a floppy bit
    of the effector brushing against a non-binding receptor face)
    no longer contribute to contact detection.

    As of the residue-level-filter fix, the preferred filter form
    is a set of effector resseq integers — use
    compute_effector_interface_residues to produce it.  The legacy
    atom-pair form is still accepted (for backward compatibility
    with existing plan.json files) and is collapsed internally to
    its residue set: an atom-pair filter of {(R, "CA"), (R, "CB")}
    becomes a residue filter of {R}.  Residue-level filtering is
    required because RFDiffusion ground truths are Cα-only, and
    an atom-level filter built from a Cα-only PDB captures only
    Cα atoms — which almost never register as heavy-to-heavy
    contacts in predictions with full side chains.

    When `effector_atom_filter` is None, every heavy atom of the
    effector is used (legacy behaviour).

    Returns empty set if either chain is missing from the PDB.
    """
    by_chain = _parse_heavy_atoms_by_chain(pdb_path)
    if receptor_chain not in by_chain or effector_chain not in by_chain:
        return set()

    rec_atoms = by_chain[receptor_chain]
    eff_atoms = by_chain[effector_chain]
    if not rec_atoms or not eff_atoms:
        return set()

    # Apply the effector filter at the residue level.  Accept two
    # input shapes:
    #  - set of ints: a residue-set filter (preferred)
    #  - set/iter of (resseq, atom_name) tuples: legacy atom filter;
    #    we collapse to the residue set by taking only resseq.
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
            filter_residues = None   # empty filter = no filter

    if filter_residues is not None:
        eff_atoms = [
            (r, x, y, z) for (r, x, y, z) in eff_atoms
            if r in filter_residues
        ]

    if not eff_atoms:
        return set()

    # Vectorise with numpy.  Build two (N, 3) arrays plus the receptor
    # residue-number array, compute pairwise squared distances, threshold,
    # and reduce across the effector axis per receptor atom, then per
    # receptor residue.
    rec_xyz = np.array([(x, y, z) for (_r, x, y, z) in rec_atoms], dtype=np.float64)
    rec_res = np.array([r for (r, _x, _y, _z) in rec_atoms], dtype=np.int32)
    eff_xyz = np.array([(x, y, z) for (_r, x, y, z) in eff_atoms], dtype=np.float64)

    cutoff2 = float(cutoff) ** 2

    # Pairwise squared distances: (n_rec, n_eff).  For a receptor with
    # ~80 residues and an effector ~100 residues this is ~50k atom
    # pairs, trivially small; no chunking needed.
    diff = rec_xyz[:, None, :] - eff_xyz[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    # Per receptor atom, is it within cutoff of ANY effector atom?
    atom_in_contact = (d2 < cutoff2).any(axis=1)
    # Any receptor residue with at least one contacting atom is a
    # contact residue.
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

    See also ``boltz2_negative_steering.find_contact_residues_heavy``
    — a second implementation with the same signature that returns
    identical positional indices (both icode-aware as of Phase 4) but
    additionally sorts by distance and uses richer per-residue atom
    bookkeeping.  Future refactor: collapse both behind a shared
    pdb_atom_io helper.

    Walks receptor residues in PDB order, computes the closest heavy-
    atom distance to any effector heavy atom, and returns a list of
    (positional_index_0b, closest_distance_angstroms) tuples for the
    receptor residues whose closest distance is < cutoff.

    Differs from compute_interface_contacts in TWO ways:
      1. Returns POSITIONAL indices (0-based, in PDB walk order) rather
         than seqid residue numbers.  This is what
         derive_input_design_region.py needs because its design region
         and true interface are both expressed in positional coords.
      2. Carries the closest distance per contacting residue, not just
         a binary contact flag.  Useful for downstream weighting.

    The expected_rec_seq argument is an optional sanity check.  If
    supplied, the walked-residue count must match its length; mismatch
    raises ValueError.  This catches PDB-vs-FASTA inconsistencies that
    would silently misalign positional indices with downstream consumers.

    Hydrogens are excluded.  Only ATOM records are read.  Returns an
    empty list if either chain is missing.
    """
    # Inline icode-aware parser.  Unlike _parse_heavy_atoms_by_chain
    # (which keys by resseq only), this preserves the (resseq, icode)
    # tuple so that residues with the same number but different
    # insertion codes — common in wwPDB structures — get distinct
    # positional indices.  Matches read_residue_heavy_atoms in
    # bin/boltz2_negative_steering.py.
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

    Accepts two forms for robustness:
      - Plain comma-separated list:  "5,26,44,46"
      - ChimeraX spec with chain:    "/A:5,26,44,46"

    Returns a sorted list of ints, or [] for empty input.
    Invalid tokens are skipped with a warning (we prefer a partial
    list + warning to a hard failure mid-run).
    """
    if not spec:
        return []
    s = spec.strip()
    # Strip a leading "/<chain>:" if present
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


def parse_position_list(spec):
    """Parse a position-list CLI argument with range support.

    Accepts (all forms in a single call, comma-separated):
      - Plain ints:                  "5,26,44,46"
      - Ranges (inclusive):          "30-78"
      - Mixed:                       "5,30-78,92"
      - ChimeraX spec with chain:    "/A:5,30-78,92"
      - Whitespace inside tokens:    "5, 30 - 78, 92"

    This is a strict superset of parse_mutated_positions; it is kept
    separate so that --mutated-positions retains its existing strict
    comma-list semantics (mutations are small sets, ranges would be
    misleading there) and so --design-region-positions can use the
    range form, which is far more ergonomic for design regions that
    often span 40+ contiguous residues.

    Returns a sorted list of unique ints, or [] for empty input.
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
        if "-" in tok:
            parts = tok.split("-", 1)
            try:
                a = int(parts[0].strip())
                b = int(parts[1].strip())
            except ValueError:
                print(f"  WARNING: ignoring malformed range {tok!r}")
                continue
            if a > b:
                a, b = b, a
            out.extend(range(a, b + 1))
        else:
            try:
                out.append(int(tok))
            except ValueError:
                print(f"  WARNING: ignoring non-integer position {tok!r}")
    return sorted(set(out))


def load_positions_file(path):
    """Load a position list from a text file.  Format: one or more
    integers per line, whitespace- or comma-separated; '#' starts a
    line comment.  Range syntax ("30-78") is supported the same way
    parse_position_list handles it.
    """
    if not path:
        return []
    tokens = []
    with open(path) as f:
        for line in f:
            # Strip inline comments
            if "#" in line:
                line = line.split("#", 1)[0]
            line = line.strip()
            if not line:
                continue
            tokens.append(line)
    joined = ",".join(tokens)
    return parse_position_list(joined)


def classify_mutation_reliance(
    contact_residues,
    mutated_positions,
):
    """Compare a set of contact residue numbers against a list of
    mutated positions.  This is the AUTHORITATIVE contamination check
    for the negative-steering pipeline.

    Returns (n_on_mut, mutated_contact_positions).

    n_on_mut > 0 means this design is contaminated: at least one
    steering mutation is contacting the effector in the predicted pose.
    The wet-lab construct will not carry that residue, so the pose's
    metrics are invalid unless the reversion pass recovers it.

    `mutated_contact_positions` is the sorted list of such positions
    — these are exactly the positions the reversion pass needs to
    revert.
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
    """Extract average pLDDT from B-factor column of CA atoms."""
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
    """Apply defaults, compute PAE-derived metrics, compute actifPTM.

    actifPTM is the Varga, Ovchinnikov & Schueler-Furman 2025
    interface-restricted ipTM: the pTM kernel evaluated over the set
    of residues that have at least one inter-chain PAE below the
    cutoff, with d0 computed from the interacting set size (not
    the full chain length).  This is the metric the steering
    pipeline now surfaces under the 'actifptm' column.

    A separate 'af_rank_score' column holds the legacy
    0.8 * iptm + 0.2 * ptm combination, which is AlphaFold-Multimer's
    default model-ranking score.  Prior to this patch, that formula
    was being mislabelled as 'actifptm'.
    """
    entry.setdefault("ptm", 0.0)
    entry.setdefault("iptm", 0.0)
    entry.setdefault("avg_plddt", 0.0)
    entry.update(_pae_derived_metrics(pae_matrix, chain_lengths, cutoff=pae_cutoff))
    entry["actifptm"] = compute_actifptm(pae_matrix, chain_lengths, cutoff=pae_cutoff)
    entry["af_rank_score"] = round(0.8 * entry["iptm"] + 0.2 * entry["ptm"], 4)
    return entry


# ═══════════════════════════════════════════════════════════════════════════
# Native-PDB-dependent metrics (P0-29 / P0-38 — moved here from
# compute_interface_metrics.py so they get computed per-prediction
# alongside the PAE-derived metrics, rather than as a post-hoc pass.
# ═══════════════════════════════════════════════════════════════════════════

# Gaussian weight parameters for weighted Jaccard (P0-38).  Sourced from
# PipelineInternalThresholds (Phase 4 §2.13) with literal fallback.
# Values are kept stable across cohorts — see the original docstring in
# compute_interface_metrics.py.  If a future cohort needs different
# parameters, emit a NEW column rather than re-tuning these.
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
    residue in the named chain.  Cβ when present, Cα fallback for
    glycine and Cα-only models.  Positional indexing (NOT seqid.num)
    keeps the (i, j) keys comparable across model and ground-truth
    PDBs even when their numbering schemes differ.
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
    {(receptor_pos_0b, effector_pos_0b): weight} over all pairs within
    pair_cutoff Å Cβ–Cβ.  Pairs beyond cutoff contribute weight ≈ 0
    anyway so dropping them costs nothing.
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
    Smooth distance-weighted Jaccard contact overlap between model and
    native (P0-38).

        weighted_jaccard
            = sum_pairs min(w_model(d), w_native(d))
            / sum_pairs max(w_model(d), w_native(d))

    where w(d) = exp(-(d - 4)^2 / 2.25).

    Returns (weighted_jaccard_float, error_str_or_None).  Returns
    (NaN, None) when both pair maps are empty (no interface to
    compare); 0.0 when exactly one is empty.
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
    Core-only intact filter.  Trim the receptor by per-residue pLDDT,
    Kabsch-align the trimmed receptor cores between model and native,
    then compute whole-complex Cα RMSD on the trimmed set.  Emit
    intact_flag = 1 if RMSD < intact_threshold else 0.

    Returns (intact_flag, rmsd_angstroms, error_str_or_None).
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
    except Exception as e:  # noqa: BLE001
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
    if len(shared) < 3:
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
    Mean Cα pLDDT over interface residues — same metric that
    run_biophysical_metrics.py used to compute, but moved here so it
    runs per-prediction alongside the other PAE-and-pLDDT metrics.

    Returns (interface_plddt_0_to_1, error_str_or_None).
    Normalises 0-100 input to 0-1 (Boltz pLDDT is on 0-100 in PDB
    B-factor; downstream thresholds expect 0-1).
    """
    try:
        import gemmi
    except ImportError as e:
        return None, f"gemmi_import_failed:{e}"

    try:
        st = gemmi.read_structure(str(model_pdb))
    except Exception as e:  # noqa: BLE001
        return None, f"structure_load_failed:{type(e).__name__}"

    # Collect heavy atoms per chain
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

    # Mean Cα pLDDT over interface residues.
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

def _dedup_paths(paths):
    """Deduplicate file paths preserving order."""
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _is_reference(path):
    """Check if file is a copied reference PDB that should be skipped."""
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

    Output structure:
        output/predictions/<job>/pdb/<n>_model_<M>.pdb
        output/predictions/<job>/pae/<n>_model_<M>.npz
        output/predictions/<job>/confidence/<n>_model_<M>.json
        output/predictions/<job>/plddt/<n>_model_<M>.npz

    Or flatter layouts depending on version.

    When `mutated_positions` is supplied (a list of 1-based receptor
    residue numbers corresponding to the cumulative steering mutations
    for this design), an interface contact analysis is run and the
    mutation-reliance flag is computed.  When it is None or empty the
    flag column comes out blank — appropriate for cold-start rows or
    any call site that hasn't wired the mutation list through.

    If `effector_atom_filter` is supplied (a set of (resseq, atom_name)
    tuples identifying the interface-atom subset of the effector),
    heavy-atom contact detection is restricted to those atoms only.
    This is the v8 Level 1 contamination-check fix — see
    compute_effector_interface_atom_mask.
    """
    if mutated_positions is None:
        mutated_positions = []
    results = []

    pdb_files = sorted(glob.glob(os.path.join(pred_dir, "**", "*.pdb"), recursive=True))
    pdb_files = [p for p in pdb_files if not _is_reference(p)]

    if not pdb_files:
        print(f"  WARNING: No PDB files found in {pred_dir}")
        return results

    for pdb_path in pdb_files:
        name = os.path.basename(pdb_path).replace(".pdb", "")
        pdb_dir = os.path.dirname(pdb_path)
        entry = {"pdb_path": pdb_path, "model_name": name}

        # --- Confidence JSON ---
        conf_files = _dedup_paths(
            glob.glob(os.path.join(pdb_dir, f"confidence_{name}.json")) +
            glob.glob(os.path.join(pdb_dir, "..", "confidence", f"confidence_{name}.json")) +
            glob.glob(os.path.join(pred_dir, "**", f"confidence_{name}.json"), recursive=True)
        )
        if conf_files:
            try:
                with open(conf_files[0]) as f:
                    conf = json.load(f)
                entry["ptm"] = round(float(conf.get("ptm", 0.0)), 4)
                entry["iptm"] = round(float(
                    conf.get("iptm", conf.get("protein_iptm", 0.0))
                ), 4)
                # Boltz-2 also writes complex_plddt (whole-complex
                # pLDDT) directly in the confidence JSON, on a 0-1
                # scale.  We surface it as a separate column so you
                # get a direct Boltz-native confidence signal without
                # having to re-average the per-residue .npz.
                if "complex_plddt" in conf:
                    entry["complex_plddt"] = round(
                        float(conf["complex_plddt"]), 4
                    )
            except Exception as e:
                print(f"  WARNING: conf parse failed for {name}: {e}")

        # --- pLDDT ---
        plddt_files = _dedup_paths(
            glob.glob(os.path.join(pdb_dir, f"plddt_{name}.npz")) +
            glob.glob(os.path.join(pdb_dir, "..", "plddt", f"plddt_{name}.npz")) +
            glob.glob(os.path.join(pred_dir, "**", f"plddt_{name}.npz"), recursive=True)
        )
        if plddt_files:
            try:
                data = np.load(plddt_files[0])
                arr = data[list(data.keys())[0]]
                if arr.ndim > 1:
                    arr = arr.flatten()
                mean_plddt = float(arr.mean())
                # Boltz-2 writes the per-residue .npz on a 0-1 scale,
                # whereas the full benchmark pipeline (and the PDB
                # B-factor-column fallback in plddt_from_pdb) use the
                # conventional 0-100 scale.  Normalise here so every
                # row in the output CSV is on 0-100 regardless of
                # which path it came from.  Guard on max<=1.5 rather
                # than mean so we don't double-multiply a file that
                # happens to already be on 0-100.
                if float(arr.max()) <= 1.5:
                    mean_plddt *= 100.0
                entry["avg_plddt"] = round(mean_plddt, 2)
            except Exception as e:
                print(f"  WARNING: plddt parse failed for {name}: {e}")
        if "avg_plddt" not in entry:
            entry["avg_plddt"] = plddt_from_pdb(pdb_path)

        # --- PAE matrix ---
        pae_files = _dedup_paths(
            glob.glob(os.path.join(pdb_dir, f"pae_{name}.npz")) +
            glob.glob(os.path.join(pdb_dir, "..", "pae", f"pae_{name}.npz")) +
            glob.glob(os.path.join(pred_dir, "**", f"pae_{name}.npz"), recursive=True)
        )
        pae_matrix = None
        if pae_files:
            try:
                data = np.load(pae_files[0])
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
        # Compute heavy-atom contacts between receptor and effector
        # chains in the PDB we actually loaded (metric_pdb's equivalent
        # — here, pdb_path itself).  If the contact analysis fails for
        # any reason (chain letters missing, parse error) we fall back
        # to empty set + blank flag so the row still gets written.
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

        # --- interface_plddt (always computable from model PDB alone) ---
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
        # Both metrics require a ground-truth structure for comparison.
        # When native_pdb is None (e.g. a benchmark run that doesn't
        # have one), the columns are emitted blank.  When native_pdb
        # IS supplied, we compute both inside the per-prediction loop
        # so the values flow through the same aggregator → passing_summary
        # → cross_summary chain as the PAE-derived metrics.
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
    # boltz1 uses the same output layout as boltz2
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
    # Interface contacts / mutation reliance.  Populated for every
    # parsed PDB regardless of whether --mutated-positions was
    # supplied; mutated_contact_positions
    # are blank when no mutations were passed.
    #
    # This IS the authoritative contamination check for the negative-
    # steering pipeline: any mutated residue that contacts the effector
    # in the predicted pose invalidates the metrics, because the wet-
    # lab construct will not carry the steered residue at that position
    # (regardless of whether the position is "protected" or not).
    "n_contact_residues",
    "contact_residues",
    "n_contacts_on_mutated_positions",
    "mutated_contact_positions",
    "contact_cutoff_used",
    # interface_plddt: mean Cα pLDDT over interface residues.  Always
    # computable from the model PDB alone.  intact_core + weighted_
    # jaccard require --native-pdb; left blank when not supplied.
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
    # receptor-chain / effector-chain were previously accepted for
    # compatibility only.  They are now used by the interface contact
    # analysis to identify which chain IDs in the prediction PDB to
    # treat as receptor vs effector.  Defaults match plan.json's
    # pred_*_chain defaults.
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
                             "output CSV's mutated_contact_positions column is "
                             "set to 'clean' if no contact residue "
                             "overlaps this set, or 'contaminated' if "
                             "at least one does.  Blank / unset → flag "
                             "left empty (appropriate for cold-start "
                             "rows).")
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
                             "restricted to these effector atoms only "
                             "(v8 Level 1 contamination-check fix). "
                             "The steering pipeline writes this list "
                             "to plan.json under 'effector_interface_atoms' "
                             "— point this flag at plan.json directly "
                             "and it will read from that key.  Blank / "
                             "unset → no filter (legacy behaviour).")
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

    # Load the effector interface filter if one was supplied.
    # Accepts either a JSON file whose top-level is:
    #   - a list of ints (residue set — preferred, post-v8 residue-fix format)
    #   - a list of [resseq, atom_name] pairs (legacy atom filter, collapsed
    #     to residue set internally by compute_interface_contacts)
    # OR a JSON file that is a plan.json dict with any of these keys:
    #   - 'effector_interface_residues' (preferred — the v8 residue set)
    #   - 'effector_interface_atoms' (legacy atom list — collapsed internally)
    # so the caller can point this flag at plan.json directly.
    effector_atom_filter = None
    if args.effector_atom_filter_json:
        try:
            with open(args.effector_atom_filter_json) as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                # Prefer the residue list; fall back to atom list
                pairs = (raw.get("effector_interface_residues")
                         or raw.get("effector_interface_atoms")
                         or [])
            elif isinstance(raw, list):
                pairs = raw
            else:
                pairs = []
            # Build the filter set: ints or (resseq, atom_name) pairs.
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
                      f"(v8 Level 1 contamination fix on)")
            else:
                print("Effector filter: loaded but empty — "
                      "contamination check runs on all effector atoms "
                      "(legacy behaviour)")
        except Exception as e:
            print(f"WARNING: could not load effector filter "
                  f"from {args.effector_atom_filter_json}: {e}")
            print("  Contamination check will run on all effector atoms.")
            effector_atom_filter = None

    # Resolve native_pdb to a usable Path (or None if blank/missing).
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

    # Rank by actifptm, best first
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
