#!/usr/bin/env python3
"""
Trial negative-design steering for Boltz binding-mode prediction.

Three subcommands, so the workflow can be split across SLURM jobs:

    plan         Initial prediction, receptor-aligned effector RMSD vs
                 the ground truth, then N steered receptor FASTAs,
                 per-design YAMLs and plan.json.  One GPU job.
    predict-one  One steered design by index (0 .. N-1).  One array
                 task, writing a results JSON.
    collect      Rank the per-design JSONs by receptor-aligned effector
                 RMSD into steered_results.csv and summary.txt.  One
                 CPU job depending on the array.

The Kabsch fit and the sequence-aligned RMSD are inlined below, the
latter over ``Bio.Align.PairwiseAligner``.  The only sibling this
module imports is ``compute_metrics``, lazily, inside ``cmd_plan``.

Negative-steering rationale
---------------------------
If Boltz places the effector on the wrong surface, every receptor Cα
within --interface-cutoff Å of an effector Cα counts as *spurious*
interface.  Each is mutated away from the wild type, and ten such
variants are re-predicted.  The hypothesis is that knocking out the
wrong interface forces Boltz to find the correct site, which should
matter most when that site is absent from the PDB training set.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Three-letter to one-letter amino acid map.  It covers the 20 standard
# residues plus the modified codes in PDB and mmCIF files, meaning
# selenomethionine MSE, selenocysteine SEC, pyrrolysine PYL,
# hydroxyproline HYP and phospho-Thr/Ser/Tyr TPO/SEP/PTR.  Use
# `.get(code, "X")` so an unknown code becomes "X".
#
# It lived in contig_utils.py, which was RFDiffusion contig parsing with this
# dict attached.  The parsing went with the rest of the design-campaign code,
# and this is the only thing that was ever imported from it.
THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
    "MSE": "M", "HYP": "P", "TPO": "T", "SEP": "S", "PTR": "Y",
}

# cmd_collect redefines both as locals.  Keep the copies in sync.
RECEPTOR_INTACT_CUTOFF = 5.0   # Å Cα RMSD vs ground truth
EFFECTOR_INTACT_CUTOFF = 5.0   # Å Cα RMSD vs ground truth

# Kabsch plus sequence-aligned RMSD, inlined rather than imported from
# the pipeline's bin/ so this script can be dropped anywhere on the
# cluster.  Behaviour matches the whole-fit `effector_rmsd` in
# boltz2_verify_binding.
from collections import defaultdict  # noqa: E402

# BioPython does the sequence-aligned residue pairing.  The Boltz1 and
# Boltz2 containers include it.
try:
    from Bio.Align import PairwiseAligner, substitution_matrices
except ImportError as _e:
    sys.stderr.write(
        "ERROR: BioPython is required for sequence-aligned residue pairing.\n"
        f"  ImportError: {_e}\n"
        "  Run this script inside the benchmark singularity image\n"
        "  (the submit script does this automatically).\n"
    )
    sys.exit(1)

def _read_ca_seq_by_chain(pdb_path):
    """Return {chain_id: (coords (N,3), sequence_str)}.

    First altloc per (chain, resnum, icode) wins, in file order, so
    seq_index matches the rest of the pipeline.  Mirrors
    boltz2_verify_binding.
    """
    coords_by_chain = defaultdict(list)
    seq_by_chain = defaultdict(list)
    seen = set()
    try:
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                if line[12:16].strip() != "CA":
                    continue
                chain = line[21]
                resnum = line[22:26]
                icode = line[26]
                key = (chain, resnum, icode)
                if key in seen:
                    continue
                seen.add(key)
                resname = line[17:20].strip()
                one = THREE_TO_ONE.get(resname, "X")
                xyz = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
                coords_by_chain[chain].append(xyz)
                seq_by_chain[chain].append(one)
    except Exception as e:
        print(f"  WARNING: Could not read {pdb_path}: {e}")
        return {}
    return {
        ch: (np.array(coords_by_chain[ch], dtype=np.float64),
             "".join(seq_by_chain[ch]))
        for ch in coords_by_chain
    }


_PAIRWISE_ALIGNER = None


def _get_pairwise_aligner():
    """Global Needleman-Wunsch with ChimeraX matchmaker defaults:
    BLOSUM62, gap-open -10, gap-extend -0.5."""
    global _PAIRWISE_ALIGNER
    if _PAIRWISE_ALIGNER is None:
        a = PairwiseAligner()
        a.mode = "global"
        a.substitution_matrix = substitution_matrices.load("BLOSUM62")
        a.open_gap_score = -10.0
        a.extend_gap_score = -0.5
        _PAIRWISE_ALIGNER = a
    return _PAIRWISE_ALIGNER


def _seqalign_pair_indices(seq_pred, seq_ref):
    """Return (pred_idx, ref_idx), parallel lists of the positions
    that align without a gap.  Empty if either input is empty or
    alignment fails."""
    if not seq_pred or not seq_ref:
        return [], []
    try:
        aligner = _get_pairwise_aligner()
        alns = aligner.align(seq_pred, seq_ref)
        if len(alns) == 0:
            return [], []
        aln = alns[0]
    except Exception as e:
        print(f"  WARNING: Sequence alignment failed: {e}")
        return [], []
    pred_blocks, ref_blocks = aln.aligned
    pred_idx, ref_idx = [], []
    for (p_start, p_end), (r_start, r_end) in zip(pred_blocks, ref_blocks):
        block_len = min(p_end - p_start, r_end - r_start)
        for k in range(block_len):
            pred_idx.append(int(p_start + k))
            ref_idx.append(int(r_start + k))
    return pred_idx, ref_idx


def _kabsch_align(P, Q):
    """Kabsch superposition of P onto Q.  Returns (rmsd, R, t, n)
    with P_aligned = (R @ P.T).T + t."""
    n = min(len(P), len(Q))
    if n == 0:
        return float("nan"), np.eye(3), np.zeros(3), 0
    if len(P) != len(Q):
        P, Q = P[:n], Q[:n]
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    p = P - cP
    q = Q - cQ
    H = p.T @ q
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cQ - R @ cP
    P_aligned = (R @ P.T).T + t
    rmsd = float(np.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1))))
    return rmsd, R, t, n


def compute_binding_rmsds(pred_pdb, design_pdb,
                          rec_chain="A", eff_chain="B",
                          des_rec_chain=None, des_eff_chain=None):
    """Whole-fit receptor-aligned effector RMSD, the gating metric for
    this experiment.

    Receptor and effector chains are paired independently by global
    Needleman-Wunsch alignment with matchmaker defaults.  The receptor
    pairs drive a Kabsch fit, that transform is applied rigidly to the
    aligned effector pairs, and the reported number is the
    receptor-aligned effector Cα RMSD.

    Returns receptor_aligned_effector_rmsd, independent_receptor_rmsd
    and independent_effector_rmsd (float, or None on failure), the
    whole-fit values from boltz2_verify_binding under clearer names.
    """
    if des_rec_chain is None:
        des_rec_chain = rec_chain
    if des_eff_chain is None:
        des_eff_chain = eff_chain

    result = {
        "receptor_aligned_effector_rmsd": None,
        "independent_receptor_rmsd": None,
        "independent_effector_rmsd": None,
    }

    if not pred_pdb or not os.path.exists(pred_pdb):
        print(f"  ERROR: Predicted PDB not found: {pred_pdb}")
        return result
    if not design_pdb or not os.path.exists(design_pdb):
        print(f"  ERROR: Design PDB not found: {design_pdb}")
        return result

    pred_chains = _read_ca_seq_by_chain(pred_pdb)
    des_chains = _read_ca_seq_by_chain(design_pdb)

    for label, ch, table in [
        ("pred receptor", rec_chain, pred_chains),
        ("pred effector", eff_chain, pred_chains),
        ("design receptor", des_rec_chain, des_chains),
        ("design effector", des_eff_chain, des_chains),
    ]:
        if ch not in table:
            print(f"  ERROR: {label} chain '{ch}' not found "
                  f"(available: {list(table.keys())})")
            return result

    pred_rec_coords, pred_rec_seq = pred_chains[rec_chain]
    pred_eff_coords, pred_eff_seq = pred_chains[eff_chain]
    des_rec_coords,  des_rec_seq  = des_chains[des_rec_chain]
    des_eff_coords,  des_eff_seq  = des_chains[des_eff_chain]

    rec_pred_idx, rec_des_idx = _seqalign_pair_indices(pred_rec_seq, des_rec_seq)
    eff_pred_idx, eff_des_idx = _seqalign_pair_indices(pred_eff_seq, des_eff_seq)
    if not rec_pred_idx:
        print("  ERROR: Receptor sequence alignment produced zero pairs")
        return result
    if not eff_pred_idx:
        print("  ERROR: Effector sequence alignment produced zero pairs")
        return result

    pred_rec = pred_rec_coords[rec_pred_idx]
    des_rec  = des_rec_coords[rec_des_idx]
    pred_eff = pred_eff_coords[eff_pred_idx]
    des_eff  = des_eff_coords[eff_des_idx]

    rmsd_rec, R, t, _ = _kabsch_align(pred_rec, des_rec)
    if rmsd_rec == rmsd_rec:  # not NaN
        result["independent_receptor_rmsd"] = round(rmsd_rec, 3)
        Pe_transformed = (R @ pred_eff.T).T + t
        rmsd_eff = float(
            np.sqrt(np.mean(np.sum((Pe_transformed - des_eff) ** 2, axis=1)))
        )
        result["receptor_aligned_effector_rmsd"] = round(rmsd_eff, 3)

    # Fitting the effector to itself, with no receptor involved,
    # catches Boltz deforming its fold.  Read alongside
    # independent_receptor_rmsd to tell "the proteins are wrong" apart
    # from "they are fine but wrongly oriented".
    rmsd_eff_ind, _, _, _ = _kabsch_align(pred_eff, des_eff)
    if rmsd_eff_ind == rmsd_eff_ind:
        result["independent_effector_rmsd"] = round(rmsd_eff_ind, 3)

    return result


# ── Sequence extraction from PDB (from extract_sequences.py) ───────


def extract_sequences_gemmi(path):
    """Preferred reader, since gemmi handles PDB and mmCIF.

    Known limit on mmCIF input. When one auth_asym_id spans two polymer
    label_asym_id subchains, gemmi merges them into one chain but
    get_polymer() returns only the first subchain, so the sequence comes back
    short with no warning. Every index downstream is positional, so a short
    receptor shifts the protected set and the candidate pool. No caller hits
    it today, because the only mmCIF this pipeline writes is the single-chain
    effector template, and the PDB path is unaffected. test_engine_guards.py
    pins the current result.
    """
    import gemmi
    st = gemmi.read_structure(path)
    st.setup_entities()

    chains = []
    seen = set()
    for model in st:
        for chain in model:
            if chain.name in seen:
                continue
            seq = []
            for res in chain.get_polymer():
                one = THREE_TO_ONE.get(res.name, "X")
                seq.append(one)
            if seq:
                seen.add(chain.name)
                chains.append({
                    "id": chain.name,
                    "sequence": "".join(seq),
                    "length": len(seq),
                })
        break  # first model only
    return chains


def extract_sequences_biopython(path):
    """Fallback: extract sequences using BioPython."""
    from Bio.PDB import MMCIFParser, PDBParser

    if path.endswith(".cif") or path.endswith(".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure("s", path)
    chains = []
    seen = set()
    for model in structure:
        for chain in model:
            if chain.id in seen:
                continue
            seq = []
            for res in chain:
                if res.id[0] != " ":
                    continue  # skip hetero
                one = THREE_TO_ONE.get(res.resname.strip(), "X")
                seq.append(one)
            if seq:
                seen.add(chain.id)
                chains.append({
                    "id": chain.id,
                    "sequence": "".join(seq),
                    "length": len(seq),
                })
        break
    return chains


def extract_sequences(path) -> List[Dict[str, any]]:
    """Chain sequences from a PDB, via gemmi, else BioPython."""
    try:
        chains = extract_sequences_gemmi(path)
        if chains:
            return chains
    except ImportError:
        pass
    except Exception as e:
        print(f"WARNING: gemmi failed ({e}), trying BioPython...", file=sys.stderr)

    try:
        chains = extract_sequences_biopython(path)
        if chains:
            return chains
    except ImportError:
        pass
    except Exception as e:
        print(f"WARNING: BioPython failed ({e})", file=sys.stderr)

    raise RuntimeError("Neither gemmi nor BioPython could parse the file.")


def get_chain_sequence(pdb_path: Path, chain_id: str) -> str:
    """One chain's sequence from a PDB file."""
    chains = extract_sequences(str(pdb_path))
    chain_map = {c["id"]: c["sequence"] for c in chains}

    if chain_id not in chain_map:
        available = list(chain_map.keys())
        raise ValueError(
            f"Chain {chain_id} not found in {pdb_path}. "
            f"Available chains: {available}"
        )

    return chain_map[chain_id]


# ── Negative-steering modes ────────────────────────────────────────
# Four sub-experiments to compare.
#
# strong  — Bulky aromatics, strong charges and proline.  Aggressive,
#           and it routinely deforms the receptor.  The original
#           method's baseline.
# mild    — Charge and H-bond disruption only (D/E/K/R), in a
#           wild-type volume envelope, so it breaks the interface
#           network without misfolding the receptor.
# conservative — Volume-preserving chemistry flips, one mutant per
#           wild-type residue (table below).  Deterministic, so with
#           pool == max_mutations it gives N identical designs.
# alanine — Classical alanine scanning, every selected residue to A,
#           or to S when already A.  Finds the load-bearing contacts
#           without disturbing the fold.
STEERING_SETS = {
    "strong": list("WYFRKEDP"),
    "mild":   list("DEKR"),
}

# One volume-matched chemistry flip per standard amino acid, from
# classic Dayhoff substitution patterns and alanine-scanning practice.
# Every entry aims for the smallest geometric change that flips the
# biochemistry.
CONSERVATIVE_SUBSTITUTIONS = {
    "A": "S",  # tiny hydrophobic → tiny polar
    "V": "T",  # branched β hydrophobic → branched β polar
    "L": "N",  # mid-size hydrophobic → mid-size polar
    "I": "N",  # branched hydrophobic → mid-size polar
    "M": "Q",  # long hydrophobic → long polar
    "F": "H",  # aromatic → aromatic+polar/charged
    "W": "Q",  # bulky aromatic → bulky polar
    "Y": "F",  # aromatic+OH → aromatic-OH (loses H-bond donor)
    "K": "E",  # +ve charge → -ve charge, similar length
    "R": "E",  # +ve charge → -ve charge
    "D": "N",  # -ve charge → polar (same volume)
    "E": "Q",  # -ve charge → polar (same volume)
    "N": "D",  # polar → -ve charge
    "Q": "E",  # polar → -ve charge
    "H": "L",  # weak charge / aromatic → hydrophobic
    "S": "A",  # tiny polar → tiny hydrophobic
    "T": "V",  # branched polar → branched hydrophobic
    "C": "S",  # similar volume, no thiol
    "G": "A",  # smallest possible addition
    "P": "A",  # break the kink
}

# Boltz prediction chain IDs are always A/B, whatever the ground truth
# uses, so the YAML stays independent of the truth chain IDs and we
# never rely on Boltz to preserve arbitrary labels.  The translation
# happens only in the compute_binding_rmsds call, which cross-maps via
# des_rec_chain / des_eff_chain.
PRED_REC_CHAIN = "A"
PRED_EFF_CHAIN = "B"


# ── I/O helpers ────────────────────────────────────────────────────
def write_single_seq_a3m(out_path: Path, header: str, seq: str) -> None:
    """Single-sequence A3M.  Boltz expects the file to exist whenever
    the YAML gives an `msa:` field."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f">{header}\n{seq}\n")


def extract_effector_template_cif(
    ground_truth_pdb: Path,
    truth_eff_chain: str,
    pred_eff_chain: str,
    out_path: Path,
) -> Path:
    """Extract the effector chain from the ground-truth complex as a
    single-chain mmCIF, relabelled to the prediction chain ID.

    Boltz predicts the effector poorly from sequence alone.  Only the
    receptor is ever mutated here, so the template pins the effector's
    fold without biasing which surface Boltz picks for the receptor.

    mmCIF because Boltz1/2 and AF3 both expect mmCIF templates.  Boltz
    matches chain label_asym_id against the YAML's template_id field,
    and relabelling to pred_eff_chain makes that unambiguous whatever
    the ground truth calls its effector.

    AF3 needs three fields for residue numbering and template
    indexing, and gemmi will not populate them from a PDB input:

      - _entity_poly.pdbx_seq_one_letter_code  (the chain sequence)
      - _entity_poly_seq                       (one row per residue)
      - _atom_site.label_seq_id                (integer per atom)

    setup_entities() identifies the chain as a polypeptide(L) polymer,
    but from PDB it leaves Entity.full_sequence and Residue.label_seq
    unset, because PDB format carries neither.  The CIF writer then
    emits '.' and '?' placeholders and AF3 fails with
        ValueError: invalid literal for int() with base 10: '.'

    So the code prunes every chain but the effector, preserving gemmi's
    entity bookkeeping, renames the survivor, calls setup_entities(),
    sets those two fields by hand and validates all three in the
    written CIF.
    """
    import gemmi
    st = gemmi.read_structure(str(ground_truth_pdb))

    chain_names = {c.name for model in st for c in model}
    if truth_eff_chain not in chain_names:
        raise ValueError(
            f"Effector chain {truth_eff_chain!r} not found in "
            f"{ground_truth_pdb}; chains present: {sorted(chain_names)}"
        )

    for model in st:
        for cname in [c.name for c in model if c.name != truth_eff_chain]:
            model.remove_chain(cname)

    # Many PDBs give crystallographic waters the protein's chain ID,
    # for example AF3-output complexes where chain B holds effector
    # plus waters.  AF3 tolerates label_seq_id='.' on HETATM rows, but
    # mixing polymer and non-polymer rows in one entity confuses its
    # template featuriser.
    st.remove_waters()

    if pred_eff_chain != truth_eff_chain:
        for model in st:
            for chain in model:
                if chain.name == truth_eff_chain:
                    chain.name = pred_eff_chain

    st.name = ground_truth_pdb.stem + "_effector_template"

    st.setup_entities()

    for model in st:
        for chain in model:
            polymer = chain.get_polymer()
            entity = st.get_entity_of(polymer)
            if entity is None:
                raise RuntimeError(
                    f"No Entity associated with polymer of chain "
                    f"{chain.name!r} after setup_entities() — "
                    f"input PDB may be malformed"
                )
            entity.full_sequence = [r.name for r in polymer]
            for i, r in enumerate(polymer, start=1):
                r.label_seq = i

    # AF3's templates.py:get_polymer_features raises
    # `ValueError: The structure must have a release date.` without
    # one.  It reads the minimum of all values under
    # `_pdbx_audit_revision_history.revision_date` (see
    # alphafold3/structure/mmcif.py:get_release_date).  PDB-derived
    # mmCIFs always carry it, coordinate-only PDBs do not.
    #
    # init_mmcif_loop puts the loop_ inside the CIF document, where it
    # has to be.  Raw text appended after the document is written lands
    # outside the parser's view, because gemmi's writer closes each
    # block.  The date value does not matter, because AF3 is not
    # date-filtering here (one template, always used).
    st.info["_pdbx_database_status.recvd_initial_deposition_date"] = "2024-01-01"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    groups = gemmi.MmcifOutputGroups(True)
    doc = st.make_mmcif_document(groups)
    block = doc[0]
    loop = block.init_mmcif_loop("_pdbx_audit_revision_history.", [
        "ordinal",
        "data_content_type",
        "major_revision",
        "minor_revision",
        "revision_date",
    ])
    loop.add_row(["1", "'Structure model'", "1", "0", "2024-01-01"])
    doc.write_file(str(out_path))

    # Validating here costs under 100 ms instead of 30 minutes into an
    # AF3 GPU job.
    text = out_path.read_text()

    if "_entity_poly_seq.entity_id" not in text:
        raise RuntimeError(  # pragma: no cover - Entity.full_sequence was set above, so only a gemmi writer regression drops this block
            f"_entity_poly_seq block missing from generated CIF "
            f"{out_path} — AF3's template parser will reject this. "
            f"This indicates Entity.full_sequence was not populated."
        )

    if "_pdbx_audit_revision_history.revision_date" not in text:
        raise RuntimeError(  # pragma: no cover - the loop was added to this document above, so only a gemmi writer regression drops it
            f"_pdbx_audit_revision_history.revision_date missing from "
            f"generated CIF {out_path} — AF3 will fail with "
            f"`ValueError: The structure must have a release date.`"
        )

    # AF3 looks the field up through its mmCIF parser, not by searching
    # raw text, so re-parse with gemmi and confirm it is findable under
    # its full key.  This catches a field outside the data block the
    # parser reads.
    try:
        check_doc = gemmi.cif.read(str(out_path))
        check_block = check_doc[0]
        check_vals = list(check_block.find_values(
            "_pdbx_audit_revision_history.revision_date"
        ))
    except Exception as e:  # pragma: no cover - gemmi wrote this file four lines above, so only a gemmi regression makes it unparseable
        raise RuntimeError(
            f"Generated CIF {out_path} fails to re-parse cleanly: {e}"
        )
    if not check_vals:
        raise RuntimeError(  # pragma: no cover - the text check above passed, so only a gemmi writer/reader disagreement hides the key
            f"_pdbx_audit_revision_history.revision_date is in the "
            f"text of {out_path} but the mmCIF parser does not "
            f"discover it under that key — AF3 will fail with "
            f"`ValueError: The structure must have a release date.`"
        )

    lines = text.splitlines()
    one_letter = None
    for i, line in enumerate(lines):
        if "_entity_poly.pdbx_seq_one_letter_code" in line:
            for j in range(i + 1, min(i + 5, len(lines))):
                ln = lines[j].strip()
                if ln and not ln.startswith("_") and not ln.startswith("loop_"):
                    one_letter = ln.split()[-1]
                    break
            break
    if not one_letter or one_letter in ("?", "."):
        raise RuntimeError(  # pragma: no cover - Entity.full_sequence was set above, so the writer always emits the sequence here
            f"_entity_poly.pdbx_seq_one_letter_code is empty in "
            f"{out_path}: {one_letter!r} — AF3 requires this field. "
            f"This indicates Entity.full_sequence was not populated."
        )

    for line in lines:
        if line.startswith(("ATOM ", "HETATM ")):
            cols = line.split()
            # _atom_site loop column order in this writer:
            # group_PDB id type_symbol label_atom_id label_alt_id
            # label_comp_id label_asym_id label_entity_id label_seq_id
            # → label_seq_id is column index 8 (0-based)
            if len(cols) >= 9 and cols[8] == ".":
                raise RuntimeError(
                    f"_atom_site.label_seq_id is '.' in {out_path}: "
                    f"{line[:80]}... — AF3 will fail with "
                    f"`ValueError: invalid literal for int() with base 10: '.'`. "
                    f"This indicates Residue.label_seq was not populated."
                )

    return out_path


def write_boltz_yaml(
    out_dir: Path,
    rec_seq: str,
    eff_seq: str,
    rec_chain: str,
    eff_chain: str,
    effector_template_cif: Path = None,
    template_threshold: float = 1.0,
    constraints_block: Optional[List[str]] = None,
) -> Path:
    """Minimal Boltz2 YAML: two protein chains, single-seq A3Ms.

    With `effector_template_cif`, the effector chain gets a structural
    template from that CIF.  It is never applied to the receptor.
    `chain_id` names the chain in this prediction and `template_id` the
    chain inside the CIF, both as bare strings rather than lists.

    `force: true` adds a steering potential during diffusion, and then
    `threshold` is the flat-bottom upper bound in angstroms on how far
    each templated CB may sit from the rigidly aligned template
    position.  Without `force` the threshold is infinite and the
    template only enters as a feature.  Boltz raises if `force` is set
    without a `threshold`.

    `templates:` and `constraints:` are both top-level keys, siblings
    of `sequences:`.  The constraints block is pre-rendered by
    build_benchmark_style_constraints_block.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    rec_a3m = out_dir / "msa" / f"chain_{rec_chain}.a3m"
    eff_a3m = out_dir / "msa" / f"chain_{eff_chain}.a3m"
    write_single_seq_a3m(rec_a3m, f"receptor_{rec_chain}", rec_seq)
    write_single_seq_a3m(eff_a3m, f"effector_{eff_chain}", eff_seq)

    yaml_lines = [
        "version: 1",
        "sequences:",
        "  - protein:",
        f"      id: {rec_chain}",
        f"      sequence: {rec_seq}",
        f"      msa: {rec_a3m.resolve()}",
        "  - protein:",
        f"      id: {eff_chain}",
        f"      sequence: {eff_seq}",
        f"      msa: {eff_a3m.resolve()}",
    ]

    if effector_template_cif is not None:
        # Top-level key, sibling of "sequences:". Boltz reads templates
        # only from the schema root, at boltz 2.2.1
        # data/parse/schema.py:1603 (`schema.get("templates", [])`). A
        # chain entry's own keys are never inspected past the entity
        # type, so a templates block nested under "  - protein:" is
        # dropped without an error. This code emitted it there until
        # 2026-11-25, and every prediction before that ran untemplated.
        yaml_lines.extend([
            "templates:",
            f"  - cif: {effector_template_cif.resolve()}",
            f"    chain_id: {eff_chain}",
            f"    template_id: {eff_chain}",
            "    force: true",
            f"    threshold: {template_threshold}",
        ])

    if constraints_block:
        # Top-level key, sibling of "sequences:", as the benchmarking
        # repo appends constraints.yaml to input.yaml.
        yaml_lines.append("constraints:")
        yaml_lines.extend(constraints_block)

    yaml_lines.append("")
    yaml_path = out_dir / "input.yaml"
    yaml_path.write_text("\n".join(yaml_lines))
    return yaml_path


# ── Boltz invocation ───────────────────────────────────────────────
def run_boltz(
    yaml_path: Path,
    out_dir: Path,
    container: str,
    recycling_steps: int,
    diffusion_samples: int,
    seed: int,
    no_kernels: bool,
) -> Path:
    """Run Boltz on a single YAML and return the predicted PDB path.

    No --model flag, so Boltz uses the container's default, Boltz2 in
    the benchmark image.  --use_potentials is on so the effector
    template's `force: true` is enforced.

    Inside a Singularity container, detected via SINGULARITY_NAME or
    SINGULARITY_CONTAINER, boltz is called directly, because nested
    singularity exec calls are not supported.  Otherwise the call is
    wrapped in singularity exec --nv.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    in_singularity = bool(
        os.environ.get("SINGULARITY_NAME")
        or os.environ.get("SINGULARITY_CONTAINER")
        or os.environ.get("APPTAINER_NAME")
        or os.environ.get("APPTAINER_CONTAINER")
    )

    boltz_args = [
        "boltz", "predict",
        str(yaml_path),
        "--out_dir", str(out_dir),
        "--recycling_steps", str(recycling_steps),
        "--diffusion_samples", str(diffusion_samples),
        "--seed", str(seed),
        "--num_workers", "0",
        "--output_format", "pdb",
        "--write_full_pae",
        "--use_potentials",
        "--override",
    ]
    if no_kernels:
        boltz_args.append("--no_kernels")

    if in_singularity:
        cmd = boltz_args
    else:
        cmd = [
            "singularity", "exec", "--nv",
            "--bind", f"{out_dir.resolve()}:{out_dir.resolve()}",
            "--bind", f"{yaml_path.parent.resolve()}:{yaml_path.parent.resolve()}",
            container,
        ] + boltz_args

    print(f"  $ {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        raise RuntimeError(
            f"Boltz prediction failed for {yaml_path} (exit {res.returncode})"
        )

    # Boltz's output layout has changed across versions, so glob
    # out_dir recursively rather than tracking layouts.  We created it
    # empty, so anything in it came from Boltz.
    all_pdbs = sorted(out_dir.rglob("*.pdb"))
    # Some Boltz versions cache reference structures under msa/.
    pdbs = [p for p in all_pdbs if "msa" not in p.parts]
    if not pdbs:
        sys.stderr.write("─── Boltz stdout ───\n")
        sys.stderr.write(res.stdout)
        sys.stderr.write("─── Boltz stderr ───\n")
        sys.stderr.write(res.stderr)
        sys.stderr.write(f"─── Tree of {out_dir} ───\n")
        for p in sorted(out_dir.rglob("*")):
            sys.stderr.write(f"  {p}\n")
        raise RuntimeError(
            f"No PDB output found anywhere under {out_dir} after Boltz "
            f"reported success."
        )
    pdbs.sort(key=lambda p: (
        "model_0" not in p.name and "rank_0" not in p.name,
        p.name,
    ))
    return pdbs[0]


# ── Interface residue identification ───────────────────────────────
@dataclass
class CAEntry:
    chain: str
    resnum: int
    resname: str
    one: str
    xyz: np.ndarray
    seq_index: int   # 0-based position within its chain


def read_ca_atoms(pdb_path: Path) -> List[CAEntry]:
    """Single-pass Cα reader.  The first altloc per residue wins, and
    seq_index is per-chain file order, matching what Boltz emits.

    The dedup key includes the insertion code (PDB column 26), because
    residues 100, 100A and 100B are distinct and each needs its own
    seq_index.  wwPDB structures use insertion codes routinely, in
    antibody loops and kinase activation segments.

    ``rfdiffusion_filter.read_ca_atoms``, in another repository, shares
    the name but returns plain coordinate dicts and takes a chain=
    filter.  This one handles altlocs and insertion codes.
    """
    entries: List[CAEntry] = []
    seen = set()
    chain_counters: dict = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            name = line[12:16].strip()
            if name != "CA":
                continue
            alt = line[16]
            if alt != " " and alt not in " A":
                continue
            chain = line[21]
            resnum = int(line[22:26])
            icode = line[26]
            resname = line[17:20].strip()
            key = (chain, resnum, icode)
            if key in seen:
                continue
            seen.add(key)
            if chain not in chain_counters:
                chain_counters[chain] = 0
            seq_idx = chain_counters[chain]
            chain_counters[chain] += 1
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            entries.append(CAEntry(
                chain=chain, resnum=resnum, resname=resname,
                one=THREE_TO_ONE.get(resname, "X"),
                xyz=np.array([x, y, z]),
                seq_index=seq_idx,
            ))
    return entries


def _read_ca_seq_from_chain(pdb_path: Path, chain_id: str) -> str:
    """Read the receptor's Cα one-letter sequence in file order.  A
    safety check that ground truth, wild-type prediction and steered
    prediction agree residue for residue."""
    return "".join(a.one for a in read_ca_atoms(pdb_path) if a.chain == chain_id)


@dataclass
class ResidueAtoms:
    """One residue's heavy atoms, plus the metadata to map back into
    the per-chain seq_index space everything else uses."""
    chain: str
    seq_index: int            # 0-based position within its chain
    resnum: int               # PDB residue number, for readable output
    resname: str
    one: str
    atoms: Dict[str, np.ndarray]   # atom name -> xyz


def read_residue_heavy_atoms(pdb_path: Path) -> List[ResidueAtoms]:
    """Read every residue with all its heavy (non-hydrogen) atoms.

    Per-chain file order matches read_ca_atoms, so seq_index values
    are interchangeable between the two readers.  That is what lets
    find_contact_residues_heavy return seq_indices lining up with the
    protected-set indices from the Cα reader.  Keeps only altloc blank
    or "A".
    """
    by_key: Dict[Tuple[str, int, str], ResidueAtoms] = {}
    chain_counters: Dict[str, int] = {}
    order: List[Tuple[str, int, str]] = []   # preserves first-seen order

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            name = line[12:16].strip()
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            # Drop hydrogens, by element column or atom-name prefix.
            if element == "H":
                continue
            if not element and (name.startswith("H") or
                                (len(name) >= 2 and name[0].isdigit() and name[1] == "H")):
                continue

            alt = line[16]
            if alt != " " and alt not in " A":
                continue

            chain = line[21]
            resnum = int(line[22:26])
            icode = line[26]
            resname = line[17:20].strip()
            key = (chain, resnum, icode)

            xyz = np.array([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ])

            if key not in by_key:
                if chain not in chain_counters:
                    chain_counters[chain] = 0
                seq_idx = chain_counters[chain]
                chain_counters[chain] += 1
                by_key[key] = ResidueAtoms(
                    chain=chain,
                    seq_index=seq_idx,
                    resnum=resnum,
                    resname=resname,
                    one=THREE_TO_ONE.get(resname, "X"),
                    atoms={},
                )
                order.append(key)
            by_key[key].atoms[name] = xyz

    return [by_key[k] for k in order]


def find_contact_residues_heavy(
    pdb_path: Path,
    rec_chain: str,
    eff_chain: str,
    cutoff: float,
    expected_rec_seq: str = None,
) -> List[Tuple[int, float]]:
    """Closest-heavy-atom contact detection.

    Returns (receptor_seq_index, min_distance) for every receptor
    residue whose closest heavy atom is within `cutoff` Å of any
    effector heavy atom, sorted ascending, so callers can take the top
    N strongest contacts.

    Cα-Cα distance miscounts long-side-chain pairs.  A Trp-Trp π-stack
    can have its Cαs 11 Å apart while the rings touch, so the metric
    is the minimum over heavy atoms.  4.5 Å is the literature standard
    for "in contact".

    `expected_rec_seq` is checked against the receptor Cα sequence in
    the file, so drift between ground truth, wild-type prediction and
    steered prediction is caught at source rather than silently giving
    wrong indices.

    ``compute_metrics.find_contact_residues_heavy``, used only by
    ``bin/prepare_complex_inputs.py``, is a second implementation
    with the same signature and the same icode-aware bucketing.  On a
    well-formed structure the two agree exactly: over all 18 benchmark
    complexes, in negative-steering-analyses, they return the same contact set
    and the same distances to within 0.001 A.  They differ in two ways
    that matter.

    This copy sorts by distance, so a caller can take the N strongest
    contacts, and returns full-precision distances.  That copy sorts by
    index and rounds to 3 dp.

    More importantly, this copy RAISES on a malformed structure where
    that one degrades.  A missing receptor chain, or a residue number
    that will not parse, is a ValueError here and an empty list or a
    skipped atom there.  Both are deliberate and separately tested, at
    test_engine_branches.py:1015 and
    test_compute_metrics_branches.py:548.  The reason is where each one
    runs: this is the engine mid-run, where bad inputs mean stopping
    before more GPU time is spent, and that is setup, where one bad
    line should not lose the whole preparation.

    So the two are not interchangeable and collapsing them would need a
    strictness flag.  Earlier versions of this docstring called for the
    merge without noticing the failure modes differ.
    """
    residues = read_residue_heavy_atoms(pdb_path)
    rec_residues = [r for r in residues if r.chain == rec_chain]
    eff_residues = [r for r in residues if r.chain == eff_chain]

    if not rec_residues:
        raise ValueError(f"No residues found for receptor chain {rec_chain} in {pdb_path}")
    if not eff_residues:
        raise ValueError(f"No residues found for effector chain {eff_chain} in {pdb_path}")

    if expected_rec_seq is not None:
        actual = "".join(r.one for r in rec_residues)
        if actual != expected_rec_seq:
            raise ValueError(
                f"Receptor sequence mismatch in {pdb_path}:\n"
                f"  expected ({len(expected_rec_seq)}): {expected_rec_seq}\n"
                f"  actual   ({len(actual)}): {actual}\n"
                "This breaks the index alignment between wrong-interface "
                "detection and protected-residue exclusion. Aborting."
            )

    eff_xyz = np.vstack([
        np.stack(list(r.atoms.values())) for r in eff_residues
    ])

    contacts: List[Tuple[int, float]] = []
    for r in rec_residues:
        if not r.atoms:
            continue
        rec_xyz = np.stack(list(r.atoms.values()))   # (n_atoms, 3)
        # (n_rec, 1, 3) - (1, n_eff, 3) -> (n_rec, n_eff, 3)
        diffs = rec_xyz[:, None, :] - eff_xyz[None, :, :]
        d2 = np.sum(diffs * diffs, axis=2)
        min_d = float(np.sqrt(d2.min()))
        if min_d <= cutoff:
            contacts.append((r.seq_index, min_d))

    contacts.sort(key=lambda t: t[1])
    return contacts


# ── Benchmark-style Boltz-2 constraints (pocket + dense contact) ───
# Ported from structure-prediction-benchmarking's BOLTZ2_CONSTRAINED
# process (modules/boltz2.nf and bin/extract_constraints_boltz2.py).
# Cα-Cα pocket and contact restraints, derived once from the
# ground-truth complex, that steer every prediction toward the true
# binding pose.  The benchmark's own defaults, left fixed rather than
# made knobs, so "benchmark-style steering" means one thing in both
# repositories.
_BOLTZ_CONSTRAINT_CONTACT_CUTOFF = 10.0
_BOLTZ_CONSTRAINT_CONTACT_MAX = 50
_BOLTZ_CONSTRAINT_CONTACT_TOLERANCE = 0.0
_BOLTZ_CONSTRAINT_POCKET_CUTOFF = 8.0
_BOLTZ_CONSTRAINT_POCKET_MAX_DISTANCE = 8.0


def _ca_token_map(atoms: List[CAEntry]) -> Dict[int, np.ndarray]:
    """{1-based Boltz token index: Cα xyz} for one chain's CAEntry list.

    Boltz numbers a chain's tokens 1..N in the same per-chain file
    order read_ca_atoms uses, so the token index is seq_index + 1.
    """
    return {a.seq_index + 1: a.xyz for a in atoms}


def pocket_residues_ca(
    rec_ca: Dict[int, np.ndarray], eff_ca: Dict[int, np.ndarray], cutoff: float
) -> List[int]:
    """Receptor token indices with any Cα within `cutoff` Å of an
    effector Cα.  Sorted ascending."""
    eff_xyz = np.stack(list(eff_ca.values()))
    selected = []
    for tok, xyz in rec_ca.items():
        d = np.sqrt(((eff_xyz - xyz) ** 2).sum(axis=1))
        if float(d.min()) <= cutoff:
            selected.append(tok)
    return sorted(selected)


def contact_pairs_ca(
    rec_ca: Dict[int, np.ndarray], eff_ca: Dict[int, np.ndarray],
    cutoff: float, max_pairs: int,
) -> List[Tuple[int, int, float]]:
    """Closest inter-chain Cα-Cα token pairs within `cutoff` Å, sorted
    by distance ascending and truncated to `max_pairs`."""
    pairs = []
    for r_tok, r_xyz in rec_ca.items():
        for e_tok, e_xyz in eff_ca.items():
            d = float(np.linalg.norm(r_xyz - e_xyz))
            if d <= cutoff:
                pairs.append((r_tok, e_tok, d))
    pairs.sort(key=lambda t: t[2])
    return pairs[:max_pairs]


def _format_boltz_pocket_block(
    rec_chain: str, eff_chain: str, residues: List[int], max_distance: float
) -> str:
    contacts_str = ", ".join(f"[{rec_chain}, {r}]" for r in residues)
    return "\n".join([
        "  - pocket:",
        f"      binder: {eff_chain}",
        f"      contacts: [{contacts_str}]",
        f"      max_distance: {max_distance}",
        "      force: true",
    ])


def _format_boltz_contact_block(
    rec_chain: str, eff_chain: str, rec_tok: int, eff_tok: int, max_distance: float
) -> str:
    return "\n".join([
        "  - contact:",
        f"      token1: [{rec_chain}, {rec_tok}]",
        f"      token2: [{eff_chain}, {eff_tok}]",
        f"      max_distance: {max_distance}",
        "      force: true",
    ])


def build_benchmark_style_constraints_block(
    ground_truth: Path,
    truth_rec_chain: str,
    truth_eff_chain: str,
    pred_rec_chain: str,
    pred_eff_chain: str,
) -> List[str]:
    """Pocket and contact constraint lines for write_boltz_yaml's
    `constraints_block`, derived once from the ground-truth complex.
    Same geometry and thresholds as the benchmarking repository's
    BOLTZ2_CONSTRAINED process.

    Which residues and pairs are chosen comes from the ground truth's
    own chain IDs.  The emitted labels use the PREDICTION chain IDs,
    because that is what the constraints must reference inside the
    Boltz input.yaml they are appended to.
    """
    atoms = read_ca_atoms(ground_truth)
    rec_atoms = [a for a in atoms if a.chain == truth_rec_chain]
    eff_atoms = [a for a in atoms if a.chain == truth_eff_chain]
    if not rec_atoms:
        raise ValueError(f"No Cα atoms for receptor chain {truth_rec_chain} in {ground_truth}")
    if not eff_atoms:
        raise ValueError(f"No Cα atoms for effector chain {truth_eff_chain} in {ground_truth}")

    rec_ca = _ca_token_map(rec_atoms)
    eff_ca = _ca_token_map(eff_atoms)

    pocket = pocket_residues_ca(rec_ca, eff_ca, _BOLTZ_CONSTRAINT_POCKET_CUTOFF)
    contacts = contact_pairs_ca(
        rec_ca, eff_ca, _BOLTZ_CONSTRAINT_CONTACT_CUTOFF, _BOLTZ_CONSTRAINT_CONTACT_MAX
    )
    if not pocket and not contacts:
        raise ValueError(
            f"No benchmark-style constraints generated from {ground_truth} "
            f"(chains {truth_rec_chain}/{truth_eff_chain}) — check chain IDs."
        )

    lines = []
    if pocket:
        lines.append(_format_boltz_pocket_block(
            pred_rec_chain, pred_eff_chain, pocket, _BOLTZ_CONSTRAINT_POCKET_MAX_DISTANCE,
        ))
    for r_tok, e_tok, d in contacts:
        max_d = round(d + _BOLTZ_CONSTRAINT_CONTACT_TOLERANCE, 1)
        lines.append(_format_boltz_contact_block(
            pred_rec_chain, pred_eff_chain, r_tok, e_tok, max_d,
        ))

    print(f"  Benchmark-style Boltz constraints: {len(pocket)} pocket residue(s), "
          f"{len(contacts)} contact pair(s) (from {ground_truth.name} "
          f"{truth_rec_chain}/{truth_eff_chain})")
    return lines


def _load_true_interface_indices(
    indices_arg: "Optional[str]",
    indices_file: "Optional[Path]",
    receptor_length: int,
) -> "Optional[List[int]]":
    """Parse pre-computed true-interface seq_idx values from either a
    comma-separated CLI string or a file.  Returns a sorted list of
    unique 0-based indices, or None if neither source was given.

    The two sources are mutually exclusive.  In the file, '#' starts a
    comment and the tokens may be separated by commas, whitespace or
    newlines interchangeably.

    Every index is validated against `receptor_length` and raises at
    plan time, the last stage that knows that length authoritatively.
    One off-by-one would silently poison the protected set for the rest
    of the run.
    """
    if indices_arg is None and indices_file is None:
        return None
    if indices_arg is not None and indices_file is not None:
        raise ValueError(
            "--true-interface-indices and --true-interface-indices-file "
            "are mutually exclusive; supply one or the other, not both."
        )

    raw: List[str] = []
    if indices_arg is not None:
        raw.append(indices_arg)
    else:
        if not indices_file.exists():
            raise FileNotFoundError(
                f"--true-interface-indices-file not found: {indices_file}"
            )
        with open(indices_file) as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                raw.append(stripped)

    tokens: List[str] = []
    for chunk in raw:
        for part in chunk.replace(",", " ").split():
            if part:
                tokens.append(part)

    try:
        parsed = [int(t) for t in tokens]
    except ValueError as e:
        raise ValueError(
            f"Could not parse true-interface indices as integers: {e}. "
            f"Tokens: {tokens!r}"
        )

    if not parsed:
        raise ValueError(
            "True-interface indices were supplied but parsed to an empty "
            "list. Check the input."
        )

    out_of_range = [i for i in parsed if i < 0 or i >= receptor_length]
    if out_of_range:
        raise ValueError(
            f"True-interface indices out of range [0, {receptor_length}): "
            f"{out_of_range}. These must be 0-based seq_idx values into "
            f"the receptor chain in file order."
        )

    return sorted(set(parsed))


def _load_design_region_indices(
    indices_file: "Optional[Path]",
    receptor_length: int,
) -> List[int]:
    """Parse the RFDiffusion design region from --design-region-indices-file.
    Returns a sorted list of 0-based indices, or [] without a file.

    Three input forms are interchangeable within one file:
      - comma, whitespace or newline separated 1-based positions
      - range syntax like "33-46,74-79"
      - ChimeraX prefix like "/A:5,26,44,46" (the "/A:" is stripped)
    '#' starts a comment.

    Positions are validated against `receptor_length`, so a bad file
    cannot silently poison the protected set downstream.
    """
    if indices_file is None:
        return []
    if not indices_file.exists():
        raise FileNotFoundError(
            f"--design-region-indices-file not found: {indices_file}"
        )

    tokens: List[str] = []
    with open(indices_file) as f:
        for line in f:
            if "#" in line:
                line = line.split("#", 1)[0]
            line = line.strip()
            if not line:
                continue
            tokens.append(line)
    joined = ",".join(tokens)
    if joined.startswith("/"):
        colon = joined.find(":")
        if colon != -1:
            joined = joined[colon + 1:]

    parsed_1based: List[int] = []
    for tok in joined.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a_s, _, b_s = tok.partition("-")
            try:
                a, b = int(a_s.strip()), int(b_s.strip())
                if a > b:
                    a, b = b, a
                parsed_1based.extend(range(a, b + 1))
            except ValueError:
                print(f"  WARN: ignoring malformed range {tok!r}",
                      file=sys.stderr)
        else:
            try:
                parsed_1based.append(int(tok))
            except ValueError:
                print(f"  WARN: ignoring non-integer position {tok!r}",
                      file=sys.stderr)

    for p in parsed_1based:
        if p < 1 or p > receptor_length:
            raise ValueError(
                f"Design region position {p} out of range "
                f"[1, {receptor_length}]"
            )
    return sorted(set(p - 1 for p in parsed_1based))


def _second_shell_fallback(
    pred_rec_residues: List[ResidueAtoms],
    pred_eff_residues: List[ResidueAtoms],
    predicted_wrong_idx: List[int],
    true_set: set,
    surface_flags_cache: Dict[int, bool],
    radius: float = 6.0,
    pool_size: int = 6,
    min_pool_size: int = 3,
) -> List[int]:
    """Build a second-shell candidate pool when the naive
    "wrong interface minus protected set" pool is empty.

    `true_set` is a historical name. The only caller passes the full
    protected set, which is the true interface union the design region,
    so no design-region residue can leak into the pool.

    Candidates are the receptor residues in neither the wrong interface
    nor that excluded set, whose Cα is within `radius` of an effector
    heavy atom (default 6 Å, close enough that a bulky side-chain
    mutation can plausibly reach it) and which pass the
    surface-exposure test.  The nearest `pool_size` are kept.  Fewer
    than `min_pool_size` survivors (default 3) returns an empty list,
    because a clean skip_steering beats a pool padded with residues
    too far away to affect the effector.

    Ranking is on effector distance, not on distance to the
    wrong-interface residues as an earlier version did.  That could
    pick residues near the wrong interface in receptor space but
    nowhere near the effector, where no side chain could perturb
    binding.  Where the wrong and true interfaces overlap heavily, as
    in 7B1I, this gives only 2-4 residues, so the cold-start
    prediction is what gets scored.
    """
    rec_by_idx = {r.seq_index: r for r in pred_rec_residues}
    wrong_set = set(predicted_wrong_idx)

    effector_coords: List[np.ndarray] = []
    for r in pred_eff_residues:
        for _, coord in r.atoms.items():
            effector_coords.append(coord)
    if not effector_coords:
        return []
    effector_stack = np.stack(effector_coords, axis=0)  # (N, 3)

    min_dist_to_effector: Dict[int, float] = {}
    for r in pred_rec_residues:
        if r.seq_index in wrong_set:
            continue   # skip the wrong interface itself
        if r.seq_index in true_set:
            continue   # skip the protected true site
        if "CA" not in r.atoms:
            continue
        ca = r.atoms["CA"]
        diffs = effector_stack - ca[None, :]
        dists = np.linalg.norm(diffs, axis=1)
        best = float(dists.min())
        if best <= radius:
            min_dist_to_effector[r.seq_index] = best

    candidates: List[Tuple[float, int]] = []
    for seq_idx, dist in min_dist_to_effector.items():
        if seq_idx in surface_flags_cache:
            if not surface_flags_cache[seq_idx]:
                continue
        else:
            r = rec_by_idx.get(seq_idx)
            if r is None:
                continue
            flag = is_surface_exposed(r, pred_rec_residues)
            surface_flags_cache[seq_idx] = flag
            if not flag:
                continue
        candidates.append((dist, seq_idx))

    candidates.sort()
    selected = sorted(seq_idx for _, seq_idx in candidates[:pool_size])
    if len(selected) < min_pool_size:
        return []
    return selected


def _write_initial_only_csv(
    workdir: Path,
    plan: Dict,
    initial_ra_eff: float,
    initial_ind_rec: float,
    initial_ind_eff: float,
) -> None:
    """Write a `steered_results.csv` holding only the cold-start
    baseline row.

    Called at every skip_steering return site, so negsteer_aggregate's
    aggregate and compute-final-metrics subcommands always find at
    least one row per experiment workdir.  An empty file would give
    zero rows in the comparison table with no error.

    The interface-analysis fields need `true_interface_idx` and
    `initial_wrong_interface_idx` in the plan.  cmd_plan persists both
    BEFORE the empty-pool skip, so the 7B1I-style case, where analysis
    runs but the pool comes out empty, keeps its metrics.  The
    `--skip-steering` branch returns before any analysis, so there they
    stay blank.
    """
    initial_pdb = workdir / "initial_prediction.pdb"
    initial_intact = 1 if (
        not np.isnan(initial_ind_rec) and not np.isnan(initial_ind_eff)
        and initial_ind_rec <= RECEPTOR_INTACT_CUTOFF
        and initial_ind_eff <= EFFECTOR_INTACT_CUTOFF
    ) else 0

    true_idx = plan.get("true_interface_idx") or []

    # Take initial_wrong_interface_idx from the plan when it is there,
    # otherwise detect it from the initial prediction PDB.  The plan
    # carries it only when cmd_plan reached wrong-interface detection,
    # so early exits such as --skip-steering never populate it, and
    # without this fallback the jaccard columns stayed blank even with
    # both PDBs available.  That happened to d0_s2.
    initial_wrong_idx = plan.get("initial_wrong_interface_idx")
    if initial_wrong_idx is None and true_idx and initial_pdb.exists():
        try:
            pred_rec_chain = plan.get("pred_receptor_chain") or "A"
            pred_eff_chain = plan.get("pred_effector_chain") or "B"
            rec_seq = plan.get("wild_type_receptor_seq") or ""
            contact_cutoff = float(plan.get("contact_cutoff") or 5.0)
            wrong_contacts = find_contact_residues_heavy(
                initial_pdb,
                pred_rec_chain, pred_eff_chain,
                contact_cutoff,
                expected_rec_seq=rec_seq if rec_seq else None,
            )
            initial_wrong_idx = sorted(i for i, _ in wrong_contacts)
        except Exception as e:
            print(f"  WARN: per-row interface detection failed for "
                  f"{initial_pdb.name}: {e}")
            initial_wrong_idx = []
    elif initial_wrong_idx is None:
        initial_wrong_idx = []

    # A known truth interface is the prerequisite for every jaccard
    # below.  An empty initial_wrong_idx gives jaccard 0, not
    # undefined, since the helper returns 0 / |true|.  The old gate,
    # `if true_idx and initial_wrong_idx`, read that as "could not
    # compute" and wrote blanks, surfacing as a missing true_jaccard in
    # cross_summary's rep_true_jaccard_median, in the composite score
    # and in the plots.
    if true_idx:
        n_iface = len(initial_wrong_idx)
        n_shared_true = len(set(initial_wrong_idx) & set(true_idx))
        initial_true_jaccard = jaccard(initial_wrong_idx, true_idx)
        n_iface_str: object = n_iface
        n_shared_true_str: object = n_shared_true
        n_iface_val: object = n_iface
        true_j: object = initial_true_jaccard
        # The cold-start prediction against itself is trivially 1.0
        # with any contacts, and NaN with none.
        wrong_j_val: object = (1.0 if initial_wrong_idx
                                else float("nan"))
    else:
        n_iface_str = ""
        n_shared_true_str = ""
        n_iface_val = ""
        true_j = ""
        wrong_j_val = ""

    initial_row = {
        "rank": 0,
        "design": "initial",
        "total_mutations": 0,
        "receptor_aligned_effector_rmsd":
            initial_ra_eff if not np.isnan(initial_ra_eff) else "",
        "delta_receptor_aligned_effector_rmsd": 0.0,
        "independent_receptor_rmsd":
            initial_ind_rec if not np.isnan(initial_ind_rec) else "",
        "independent_effector_rmsd":
            initial_ind_eff if not np.isnan(initial_ind_eff) else "",
        "receptor_intact": initial_intact,
        "wrong_jaccard": wrong_j_val,
        "n_shared_wrong": n_iface_str,
        "true_jaccard": true_j,
        "n_shared_true": n_shared_true_str,
        "n_design_interface_residues": n_iface_val,
        "status": "initial",
        "pdb": str(initial_pdb.resolve()) if initial_pdb.exists() else "",
    }

    csv_path = workdir / "steered_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "rank", "design", "total_mutations",
            "receptor_aligned_effector_rmsd",
            "delta_receptor_aligned_effector_rmsd",
            "independent_receptor_rmsd",
            "independent_effector_rmsd",
            "receptor_intact",
            "wrong_jaccard", "n_shared_wrong",
            "true_jaccard",  "n_shared_true",
            "n_design_interface_residues",
            "status", "pdb",
        ])
        w.writeheader()
        w.writerow(initial_row)
    print(f"Wrote {csv_path} (initial row only, skip_steering)")


def _write_initial_multiseed_csv(
    workdir: Path,
    plan: Dict,
    cold_start_seeds: List[Dict],
) -> None:
    """Write steered_results.csv with one row per cold-start seed.

    Called when cmd_plan skips steering because every cold-start seed
    was already clean, so the wild type is a confirmed binder with
    nothing to steer away from.  sequence_group is 0 on every row, so
    the aggregator collapses the seeds into one multi-seed group with
    N = num_seeds, and seed_index is seed_offset as for steered
    designs.  The 0 mutations and 0 mutated contacts are true by
    construction, so negsteer_aggregate._per_seed_verdict_breakdown
    calls these rows clean_steered.  No reversion runs, so
    _classify_outcome returns ("no_reversion", ...) and cross_summary
    puts the row in tier A.

    The interface metrics are computed PER SEED, by
    find_contact_residues_heavy on each seed's own prediction PDB,
    under the same true_interface_idx rule as _write_initial_only_csv.
    The earlier code took them once from
    plan["initial_wrong_interface_idx"] for every seed, which was
    wrong twice over.  Different seeds give different interfaces, and
    an empty plan value blanked every seed even where the predictions
    had detectable ones.  That is what made d0_s2 lose its
    true_jaccard despite three valid PDBs.  An unreadable PDB blanks
    that one seed only.
    """
    true_idx = plan.get("true_interface_idx") or []
    pred_rec_chain = plan.get("pred_receptor_chain") or "A"
    pred_eff_chain = plan.get("pred_effector_chain") or "B"
    rec_seq = plan.get("wild_type_receptor_seq") or ""
    contact_cutoff = float(plan.get("contact_cutoff") or 5.0)

    rows = []
    for s in cold_start_seeds:
        ra = s["ra_eff"]
        rec = s["ind_rec"]
        eff = s["ind_eff"]
        intact = 1 if (
            not np.isnan(rec) and not np.isnan(eff)
            and rec <= RECEPTOR_INTACT_CUTOFF
            and eff <= EFFECTOR_INTACT_CUTOFF
        ) else 0
        seed_offset = s["seed_offset"]
        seed_pdb_path = s.get("pdb_path", "")

        # Interface fields stay blank when a seed's PDB is missing or
        # the truth interface is unknown.
        n_iface_val: object = ""
        n_shared_true_str: object = ""
        true_j: object = ""
        wrong_j_val: object = ""
        if true_idx and seed_pdb_path and Path(seed_pdb_path).exists():
            try:
                seed_wrong_contacts = find_contact_residues_heavy(
                    Path(seed_pdb_path),
                    pred_rec_chain, pred_eff_chain,
                    contact_cutoff,
                    expected_rec_seq=rec_seq if rec_seq else None,
                )
                seed_wrong_idx = sorted(i for i, _ in seed_wrong_contacts)
                n_iface_val = len(seed_wrong_idx)
                n_shared_true_str = len(set(seed_wrong_idx) & set(true_idx))
                true_j = jaccard(seed_wrong_idx, true_idx)
                # Trivially 1.0 with any contacts, NaN with none.
                wrong_j_val = (1.0 if seed_wrong_idx
                               else float("nan"))
            except Exception as e:
                # Soft failure.  This seed keeps its RMSD and intact
                # fields, since one malformed PDB should not abort the
                # whole cohort row.
                print(f"    WARN: per-seed interface detection failed "
                      f"for seed {seed_offset} "
                      f"({Path(seed_pdb_path).name}): {e}")

        rows.append({
            "rank": seed_offset,
            "design": ("initial" if seed_offset == 0
                       else f"initial_s{seed_offset}"),
            "sequence_group": 0,
            "seed_index": seed_offset,
            "total_mutations": 0,
            "receptor_aligned_effector_rmsd":
                ra if not np.isnan(ra) else "",
            "delta_receptor_aligned_effector_rmsd": 0.0,
            "independent_receptor_rmsd":
                rec if not np.isnan(rec) else "",
            "independent_effector_rmsd":
                eff if not np.isnan(eff) else "",
            "receptor_intact": intact,
            "wrong_jaccard": wrong_j_val,
            "n_shared_wrong": n_iface_val,
            "true_jaccard": true_j,
            "n_shared_true": n_shared_true_str,
            "n_design_interface_residues": n_iface_val,
            "status": "initial",
            "pdb": s.get("pdb_path", ""),
        })

    csv_path = workdir / "steered_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "rank", "design", "sequence_group", "seed_index",
            "total_mutations",
            "receptor_aligned_effector_rmsd",
            "delta_receptor_aligned_effector_rmsd",
            "independent_receptor_rmsd",
            "independent_effector_rmsd",
            "receptor_intact",
            "wrong_jaccard", "n_shared_wrong",
            "true_jaccard",  "n_shared_true",
            "n_design_interface_residues",
            "status", "pdb",
        ])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Wrote {csv_path} ({len(rows)} cold-start seed rows, "
          f"skip_steering=all_seeds_clean)")


def is_surface_exposed(
    residue: ResidueAtoms,
    all_rec_residues: List[ResidueAtoms],
    neighbour_radius: float = 10.0,
) -> bool:
    """Cβ-vector test for surface exposure.

    True iff the Cα→Cβ direction points away from the centroid of all
    OTHER receptor Cαs within `neighbour_radius` Å, meaning the side
    chain leaves the local core.  Glycine has no Cβ and counts as not
    surface-exposed.  We do not want to mutate glycine anyway, since it
    is structural.

    A cheap heuristic, not a real solvent-accessible surface area.  It
    gets confused in deep concave pockets, where "outward" can mean
    "into another receptor surface", so it is applied to mutation
    candidates only, never to the protected-set definition.
    """
    if "CA" not in residue.atoms or "CB" not in residue.atoms:
        return False

    ca = residue.atoms["CA"]
    cb = residue.atoms["CB"]
    v_side = cb - ca
    n_side = np.linalg.norm(v_side)
    if n_side < 1e-6:
        return False
    v_side = v_side / n_side

    neighbour_cas = []
    for other in all_rec_residues:
        if other.seq_index == residue.seq_index:
            continue
        if "CA" not in other.atoms:
            continue
        d = np.linalg.norm(other.atoms["CA"] - ca)
        if d <= neighbour_radius:
            neighbour_cas.append(other.atoms["CA"])

    if not neighbour_cas:
        # An isolated residue counts as surface-exposed by default.
        return True

    centroid = np.mean(np.stack(neighbour_cas), axis=0)
    v_core = centroid - ca
    n_core = np.linalg.norm(v_core)
    if n_core < 1e-6:
        return True
    v_core = v_core / n_core

    return float(np.dot(v_side, v_core)) < 0.0


def jaccard(a: List[int], b: List[int]) -> float:
    """Jaccard overlap between two index sets.  1.0 is an identical
    interface and 0.0 shares no receptor residues.  Two empty sets give
    NaN, since there is no interface to compare."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return float("nan")
    return len(sa & sb) / len(sa | sb)


# ── Receptor-aligned effector RMSD + receptor RMSD ─────────────────
def binding_rmsds(
    pred_pdb: Path,
    truth_pdb: Path,
    pred_rec_chain: str,
    pred_eff_chain: str,
    truth_rec_chain: str,
    truth_eff_chain: str,
) -> Tuple[float, float, float]:
    """Return
    (receptor_aligned_effector_rmsd, independent_receptor_rmsd, independent_effector_rmsd).

    - receptor_aligned_effector_rmsd is the effector Cα RMSD after the
      receptor's Kabsch transform is applied rigidly.  This is the
      gating metric, the one that answers "is the binding mode right".
      5.0 Å is the cutoff.
    - independent_receptor_rmsd is the receptor's own fold quality,
      catching collapse caused by heavy mutation.
    - independent_effector_rmsd is the effector's own fold quality,
      with no receptor involved, catching Boltz deforming it.

    Boltz emits A/B while the reference might use A/C, so callers pass
    both chain-ID pairs explicitly.
    """
    result = compute_binding_rmsds(
        str(pred_pdb), str(truth_pdb),
        rec_chain=pred_rec_chain, eff_chain=pred_eff_chain,
        des_rec_chain=truth_rec_chain, des_eff_chain=truth_eff_chain,
    )
    eff = result.get("receptor_aligned_effector_rmsd")
    rec = result.get("independent_receptor_rmsd")
    eff_ind = result.get("independent_effector_rmsd")
    return (
        float(eff)     if eff     is not None else float("nan"),
        float(rec)     if rec     is not None else float("nan"),
        float(eff_ind) if eff_ind is not None else float("nan"),
    )


# ── Negative steering (mutation design) ────────────────────────────
def pick_mutant_residue(
    wt: str,
    mode: str,
    rng: random.Random,
) -> str:
    """Choose the mutant amino acid for a wild-type residue under a
    steering mode.

    - strong and mild pick at random from the matching STEERING_SETS
      pool, never the wild type.
    - conservative looks up CONSERVATIVE_SUBSTITUTIONS, falling back
      to alanine if that returns the wild type, which no entry should.
    - alanine returns A, except for a wild-type A, which returns S,
      the closest innocuous alternative.
    """
    if mode in ("strong", "mild"):
        pool = STEERING_SETS[mode]
        options = [r for r in pool if r != wt]
        if not options:  # pragma: no cover - no pool is a single residue
            # A backstop that cannot fire with the current pools. "mild"
            # is DEKR and "strong" is WYFRKEDP, so removing one wild type
            # always leaves at least three choices. It would only fire if
            # a pool were narrowed to the wild type alone.
            options = [r for r in STEERING_SETS["strong"] if r != wt]
        return rng.choice(options)

    if mode == "conservative":
        mut = CONSERVATIVE_SUBSTITUTIONS.get(wt, "A")
        if mut == wt:
            mut = "A"  # pragma: no cover - no CONSERVATIVE_SUBSTITUTIONS entry maps a residue to itself
        return mut

    if mode == "alanine":
        return "S" if wt == "A" else "A"

    raise ValueError(f"Unknown steering mode: {mode}")


def make_steered_sequence(
    original_seq: str,
    candidate_pool: List[int],
    max_mutations: int,
    mode: str,
    rng: random.Random,
) -> Tuple[str, List[Tuple[int, str, str]]]:
    """Build one steered receptor sequence.

    cmd_plan has already filtered `candidate_pool` to surface-exposed
    positions outside the protected set.  We sample
    min(max_mutations, len(pool)) of them WITHOUT replacement, then
    mutate each according to `mode`.

    With pool == max_mutations every design hits the same positions and
    only the residue identities differ, and only in the random modes.
    With pool > max_mutations the positions vary too, which is what
    makes single-site and few-site sampling experiments possible.

    Returns (mutated_sequence, [(1-based_pos, wt, mut), ...]).
    """
    if not candidate_pool:
        return original_seq, []

    n = min(max_mutations, len(candidate_pool))
    chosen = rng.sample(candidate_pool, n)
    chosen.sort()  # consistent output ordering

    seq_list = list(original_seq)
    mutations = []
    for i in chosen:
        wt = seq_list[i]
        mut = pick_mutant_residue(wt, mode, rng)
        if mut == wt:
            continue   # Defensive. Should not happen. Never self-mutate.  # pragma: no cover - every mode returns a residue other than wt, and no CONSERVATIVE_SUBSTITUTIONS entry maps a residue to itself
        seq_list[i] = mut
        mutations.append((i + 1, wt, mut))  # 1-based for output
    return "".join(seq_list), mutations


# ── Subcommand: plan ───────────────────────────────────────────────
def cmd_plan(args: argparse.Namespace) -> int:
    workdir: Path = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    print("[plan] Negative steering for Boltz", flush=True)
    print(f"  ground truth: {args.ground_truth}")
    print(f"  truth receptor chain: {args.receptor_chain}")
    print(f"  truth effector chain: {args.effector_chain}")
    print(f"  prediction chains:    {PRED_REC_CHAIN} (rec) / {PRED_EFF_CHAIN} (eff)")

    try:
        rec_seq_truth = get_chain_sequence(args.ground_truth, args.receptor_chain)
        eff_seq_truth = get_chain_sequence(args.ground_truth, args.effector_chain)
        print(f"  truth receptor seq: {len(rec_seq_truth)} residues")
        print(f"  truth effector seq: {len(eff_seq_truth)} residues")
    except Exception as e:
        print(f"ERROR: Failed to extract sequences from {args.ground_truth}: {e}")
        return 1

    # Optional sequence overrides feed Boltz a different sequence while
    # the RMSD and contact machinery stays anchored to the ground
    # truth.  The override must match the truth length, because every
    # downstream index is positional, covering the true-site set, the
    # candidate pool and the surface flags.  RMSD still works, because
    # compute_binding_rmsds aligns with Needleman-Wunsch.
    def _load_seq_override(path, ref_seq, label):
        if path is None:
            return ref_seq
        if not path.exists():
            print(f"ERROR: --{label.lower()}-fasta path does not exist: {path}")
            raise SystemExit(1)
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        seq = "".join(line for line in lines if not line.startswith(">"))
        if not seq:
            print(f"ERROR: --{label.lower()}-fasta is empty: {path}")
            raise SystemExit(1)
        if len(seq) != len(ref_seq):
            print(
                f"ERROR: {label} override length {len(seq)} != "
                f"ground-truth length {len(ref_seq)}.\n"
                f"  override: {seq}\n"
                f"  truth:    {ref_seq}\n"
                f"  Index-based contact and RMSD machinery requires "
                f"equal length (point mutations only, no indels)."
            )
            raise SystemExit(1)
        diffs = [(i + 1, r, s) for i, (r, s) in enumerate(zip(ref_seq, seq)) if r != s]
        diff_str = ", ".join(f"{r}{i}{s}" for i, r, s in diffs) or "(none)"
        print(f"  {label} sequence OVERRIDDEN from {path}")
        print(f"    {len(diffs)} mutation(s) vs ground truth: {diff_str}")
        return seq

    rec_seq = _load_seq_override(args.receptor_fasta, rec_seq_truth, "Receptor")
    eff_seq = _load_seq_override(args.effector_fasta, eff_seq_truth, "Effector")

    if len(rec_seq) < 20 or len(eff_seq) < 20:
        print("  WARNING: short sequence(s) — double-check chain IDs.")

    # The ground truth's effector chain templates the effector in
    # EVERY prediction here, initial and steered alike.  Only the
    # receptor sequence ever changes.  See
    # extract_effector_template_cif.
    eff_template_path = workdir / "effector_template.cif"
    if args.effector_template:
        try:
            extract_effector_template_cif(
                args.ground_truth,
                args.effector_chain,
                PRED_EFF_CHAIN,
                eff_template_path,
            )
            print(f"  Wrote effector template: {eff_template_path}")
        except Exception as e:
            print(f"ERROR: Effector template extraction failed: {e}")
            return 1
    else:
        eff_template_path = None
        print("  Effector template DISABLED (--no-effector-template)")

    # Benchmark-style constraints (--boltz-constraints), computed once
    # from the ground truth and reused verbatim for EVERY prediction in
    # this run, the initial one, the extra cold-start seeds and every
    # steered design.
    constraints_block: Optional[List[str]] = None
    if args.boltz_constraints:
        try:
            constraints_block = build_benchmark_style_constraints_block(
                args.ground_truth, args.receptor_chain, args.effector_chain,
                PRED_REC_CHAIN, PRED_EFF_CHAIN,
            )
        except Exception as e:
            print(f"ERROR: Benchmark-style Boltz constraints failed: {e}")
            return 1

    init_dir = workdir / "initial"
    yaml_path = write_boltz_yaml(
        init_dir / "input",
        rec_seq, eff_seq,
        PRED_REC_CHAIN, PRED_EFF_CHAIN,
        effector_template_cif=eff_template_path,
        constraints_block=constraints_block,
    )
    print(f"  Wrote wild-type YAML: {yaml_path}")

    print("  Running initial Boltz...")
    try:
        pred_pdb = run_boltz(
            yaml_path, init_dir,
            args.boltz_container,
            args.recycling_steps, args.diffusion_samples,
            args.seed, args.no_kernels,
        )
        initial_ra_eff, initial_ind_rec, initial_ind_eff = binding_rmsds(
            pred_pdb, args.ground_truth,
            PRED_REC_CHAIN, PRED_EFF_CHAIN,
            args.receptor_chain, args.effector_chain,
        )
        shutil.copy2(pred_pdb, workdir / "initial_prediction.pdb")
        print(f"  Initial receptor_aligned_effector_rmsd: {initial_ra_eff:.3f} Å")
        print(f"  Initial independent_receptor_rmsd:      {initial_ind_rec:.3f} Å")
        print(f"  Initial independent_effector_rmsd:      {initial_ind_eff:.3f} Å")
    except Exception as e:
        print(f"ERROR: Initial prediction failed: {e}")
        return 1

    # ── Multi-seed cold-start ────────────────────────────────────────
    # The prediction above is seed 0.  With --num-seeds above 1, run
    # num_seeds-1 further cold-start predictions of the SAME wild-type
    # sequence.  Without them, a sequence whose initial prediction hits
    # the right pose aggregates as a singleton, and extract_passing.py
    # filters singletons out.
    #
    # The extra seeds also decide whether steering runs at all, by the
    # policy below.  --skip-steering is excluded, because that path
    # asks for one diagnostic prediction only.
    cold_start_seeds = [{
        "seed_offset": 0,
        "boltz_seed": args.seed,
        "ra_eff": initial_ra_eff,
        "ind_rec": initial_ind_rec,
        "ind_eff": initial_ind_eff,
        "pdb_path": str((workdir / "initial_prediction.pdb").resolve()),
    }]
    if args.num_seeds > 1 and not args.skip_steering:
        print(f"  Running {args.num_seeds - 1} additional cold-start "
              f"seed(s) for multi-seed confirmation...")
        for extra_idx in range(1, args.num_seeds):
            extra_dir = workdir / f"initial_s{extra_idx}"
            extra_dir.mkdir(exist_ok=True)
            extra_yaml = write_boltz_yaml(
                extra_dir / "input",
                rec_seq, eff_seq,
                PRED_REC_CHAIN, PRED_EFF_CHAIN,
                effector_template_cif=eff_template_path,
                constraints_block=constraints_block,
            )
            # boltz_seed = args.seed + extra_idx.  This matches the
            # offset convention used downstream for steered designs,
            # which use args.seed + design_idx + 1.
            extra_boltz_seed = args.seed + extra_idx
            try:
                extra_pdb = run_boltz(
                    extra_yaml, extra_dir,
                    args.boltz_container,
                    args.recycling_steps, args.diffusion_samples,
                    extra_boltz_seed, args.no_kernels,
                )
                ra2, rec2, eff2 = binding_rmsds(
                    extra_pdb, args.ground_truth,
                    PRED_REC_CHAIN, PRED_EFF_CHAIN,
                    args.receptor_chain, args.effector_chain,
                )
                # A sibling of initial_prediction.pdb, so downstream
                # tools find each cold-start seed PDB by a predictable
                # name.
                kept_pdb = workdir / f"initial_prediction_s{extra_idx}.pdb"
                shutil.copy2(extra_pdb, kept_pdb)
                cold_start_seeds.append({
                    "seed_offset": extra_idx,
                    "boltz_seed": extra_boltz_seed,
                    "ra_eff": ra2,
                    "ind_rec": rec2,
                    "ind_eff": eff2,
                    "pdb_path": str(kept_pdb.resolve()),
                })
                print(f"    seed {extra_idx}: ra_eff={ra2:.3f} Å  "
                      f"ind_rec={rec2:.3f} Å  ind_eff={eff2:.3f} Å")
            except Exception as e:
                # Tolerate a per-seed failure rather than abort.  The
                # NaN entry keeps the aggregator seeing N seeds.
                print(f"    seed {extra_idx}: FAILED ({e})")
                cold_start_seeds.append({
                    "seed_offset": extra_idx,
                    "boltz_seed": extra_boltz_seed,
                    "ra_eff": float("nan"),
                    "ind_rec": float("nan"),
                    "ind_eff": float("nan"),
                    "pdb_path": "",
                })

    # Clean means receptor-aligned effector RMSD at or below the
    # threshold AND both chains intact, the per-seed clean_steered rule
    # in negsteer_aggregate._per_seed_verdict_breakdown.
    n_clean = 0
    for s in cold_start_seeds:
        ra = s["ra_eff"]
        rec = s["ind_rec"]
        eff = s["ind_eff"]
        if (not np.isnan(ra) and not np.isnan(rec) and not np.isnan(eff)
                and ra <= args.rmsd_threshold
                and rec <= RECEPTOR_INTACT_CUTOFF
                and eff <= EFFECTOR_INTACT_CUTOFF):
            n_clean += 1
    n_total_cold = len(cold_start_seeds)
    # All-clean policy.  Steering is skipped only when EVERY cold-start
    # seed passes, confirming the wild type as a cold-start binder.
    # One failing seed runs the full pipeline, and the cold-start
    # results are then screening artefacts, not candidates.  This is
    # stricter than a majority rule.
    cold_start_all_clean = (n_clean == n_total_cold and n_total_cold > 0)

    # ── TRUE binding-site detection, before every early exit ─────────
    # Run up-front so plan["true_interface_idx"] is populated even when
    # --skip-steering or cold_start_all_clean returns early.  Otherwise
    # the cold-start CSV writers have no truth set for per-seed jaccard
    # and the metrics come out blank, as happened to d0_s2 when
    # detection sat after the early exits.
    true_interface_source = "heavy_atom_contact"
    precomputed_true_idx = _load_true_interface_indices(
        args.true_interface_indices,
        args.true_interface_indices_file,
        len(rec_seq_truth),
    )
    if precomputed_true_idx is not None:
        true_interface_source = "pre_computed"
        print(f"  Using pre-computed TRUE interface "
              f"({len(precomputed_true_idx)} residues) — "
              f"heavy-atom contact detection on ground truth skipped")
        true_contacts = [(i, float("nan")) for i in precomputed_true_idx]
    else:
        print(f"  Identifying TRUE binding site on ground truth "
              f"(heavy-atom contact ≤ {args.contact_cutoff} Å)...")
        try:
            true_contacts = find_contact_residues_heavy(
                args.ground_truth,
                args.receptor_chain, args.effector_chain,
                args.contact_cutoff,
                expected_rec_seq=rec_seq_truth,
            )
        except Exception as e:
            print(f"ERROR: True-interface detection failed: {e}")
            return 1
    true_idx = sorted(i for i, _ in true_contacts)
    print(f"  {len(true_idx)} receptor residues at the TRUE interface")

    # Store BOTH the A/B prediction chain IDs and the user's
    # ground-truth ones, so the array stage can call the RMSD
    # function.
    plan = {
        "ground_truth": str(args.ground_truth.resolve()),
        "truth_receptor_chain": args.receptor_chain,
        "truth_effector_chain": args.effector_chain,
        "pred_receptor_chain": PRED_REC_CHAIN,
        "pred_effector_chain": PRED_EFF_CHAIN,
        "wild_type_receptor_seq": rec_seq,
        "initial_receptor_aligned_effector_rmsd": initial_ra_eff,
        "initial_independent_receptor_rmsd": initial_ind_rec,
        "initial_independent_effector_rmsd": initial_ind_eff,
        "rmsd_threshold": args.rmsd_threshold,
        "contact_cutoff": args.contact_cutoff,
        "max_mutations": args.max_mutations,
        "candidate_pool_size": args.candidate_pool_size,
        "mode": args.mode,
        "effector_template_cif":
            str(eff_template_path.resolve()) if eff_template_path else None,
        # Reused by reversion.py rather than recomputed.
        "boltz_constraints_block": constraints_block,
        "n_designs": args.n_designs,
        "num_seeds": args.num_seeds,
        "boltz_container": args.boltz_container,
        "recycling_steps": args.recycling_steps,
        "diffusion_samples": args.diffusion_samples,
        "seed": args.seed,
        "no_kernels": args.no_kernels,
        "receptor_fasta_override":
            str(args.receptor_fasta.resolve()) if args.receptor_fasta else None,
        "effector_fasta_override":
            str(args.effector_fasta.resolve()) if args.effector_fasta else None,
        "designs": [],
        "true_interface_idx": true_idx,
        "true_interface_source": true_interface_source,
        # One entry per initial prediction.  When cold_start_all_clean
        # these become steered_results.csv, this sequence's final
        # candidate set.  Otherwise they are screening-only.
        "cold_start_seeds": cold_start_seeds,
        "cold_start_n_clean": n_clean,
        "cold_start_all_clean": cold_start_all_clean,
    }

    # --skip-steering early return, for sequence-walk diagnostics that
    # want only the cold-start prediction and its three RMSDs.
    if args.skip_steering:
        print("  --skip-steering set — not generating steered designs.")
        plan["skip_steering"] = True
        (workdir / "plan.json").write_text(json.dumps(plan, indent=2))
        _write_initial_only_csv(
            workdir, plan,
            initial_ra_eff=initial_ra_eff,
            initial_ind_rec=initial_ind_rec,
            initial_ind_eff=initial_ind_eff,
        )
        return 0

    # Apply the all-clean policy decided above.
    if cold_start_all_clean:
        print(f"  Cold-start all-clean ({n_clean}/{n_total_cold} seeds "
              f"with ra_eff ≤ {args.rmsd_threshold} Å and intact) — "
              f"no steering needed.")
        plan["skip_steering"] = True
        (workdir / "plan.json").write_text(json.dumps(plan, indent=2))
        _write_initial_multiseed_csv(
            workdir, plan,
            cold_start_seeds=cold_start_seeds,
        )
        return 0
    else:
        print(f"  Cold-start: {n_clean}/{n_total_cold} seeds clean — "
              f"running negative steering "
              f"(cold-start rows will NOT be in the candidate set).")

    # ── The TRUE binding site on the ground truth ───────────────────
    # true_idx, true_contacts and true_interface_source were computed
    # before the early exits.  They hold the receptor residues in
    # heavy-atom contact with the effector at or below contact_cutoff
    # Å, and MUST NOT be mutated, since the point of the experiment is
    # to push Boltz toward that surface.
    #
    # Index alignment: gemmi.get_polymer's receptor sequence went
    # verbatim into the Boltz YAML, so the truth and prediction Cα
    # sequences must match in file order.  find_contact_residues_heavy
    # checks that through expected_rec_seq.

    # ── Identify the WRONG (predicted) binding site ─────────────────
    print("  Identifying WRONG (predicted) interface residues...")
    try:
        wrong_contacts = find_contact_residues_heavy(
            pred_pdb,
            PRED_REC_CHAIN, PRED_EFF_CHAIN,
            args.contact_cutoff,
            expected_rec_seq=rec_seq,
        )
    except Exception as e:
        print(f"ERROR: Wrong-interface detection failed: {e}")
        return 1
    predicted_wrong_idx = sorted(i for i, _ in wrong_contacts)
    print(f"  {len(predicted_wrong_idx)} receptor residues at the wrong interface")

    # ── Surface-pointing test on every wrong-interface residue ──────
    # Read from the prediction PDB, so the Cβ direction reflects what
    # Boltz built.  The ground truth's geometry would mislead, since
    # the goal is to disrupt the WRONG complex.
    pred_residues = read_residue_heavy_atoms(pred_pdb)
    pred_rec_residues = [r for r in pred_residues if r.chain == PRED_REC_CHAIN]
    pred_eff_residues = [r for r in pred_residues if r.chain == PRED_EFF_CHAIN]
    pred_rec_by_idx = {r.seq_index: r for r in pred_rec_residues}
    surface_flags: Dict[int, bool] = {}
    for i in predicted_wrong_idx:
        r = pred_rec_by_idx.get(i)
        if r is None:  # pragma: no cover - predicted_wrong_idx and pred_rec_by_idx both come from read_residue_heavy_atoms(pred_pdb), so every index is in the map
            surface_flags[i] = False
            continue
        surface_flags[i] = is_surface_exposed(r, pred_rec_residues)

    # ── Build the candidate pool ────────────────────────────────────
    # Start from the wrong-interface residues, nearest contact first.
    # Drop any in the protected set or failing the surface test, then
    # take the top --candidate-pool-size.  See make_steered_sequence
    # for what the pool-to-max_mutations ratio controls.

    # ── Load the RFDiffusion design region ──────────────────────────
    # Parsed before candidate filtering, so the design_region_union
    # protected set can use it.  An earlier version parsed it after the
    # pool was built, too late to matter.
    drif = getattr(args, "design_region_indices_file", None)
    drif_path = Path(drif) if drif is not None else None
    try:
        design_region_idx = _load_design_region_indices(
            drif_path, len(rec_seq)
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if design_region_idx:
        print(f"  design region: {len(design_region_idx)} positions "
              f"(0-based) loaded from {drif_path.name}")
    plan["design_region_idx"] = design_region_idx

    # ── Build the protected set ─────────────────────────────────────
    # The receptor positions excluded from the candidate pool.  A
    # steering mutation here lands either on the true interface, where
    # the effector is supposed to bind, or inside the RFDiffusion
    # design region, the likeliest place for the final scored
    # prediction to make contact.  Either way it contaminates the
    # rescue by construction, so the rescued pose can no longer be
    # attributed to the design.
    #
    # --protected-set-source picks between the true interface alone and
    # its union with the design region, the default.  Neither expands
    # by Cα radius, because that would gut the pool on small receptors.
    protected_source = getattr(args, "protected_set_source",
                               "design_region_union")
    true_set = set(true_idx)
    if protected_source == "design_region_union":
        if design_region_idx:
            protected_set = true_set | set(design_region_idx)
            print(f"  Protected set: true_interface ∪ design_region "
                  f"= {len(protected_set)} positions "
                  f"({len(true_set)} true + "
                  f"{len(set(design_region_idx) - true_set)} "
                  f"design-region-only)")
        else:
            protected_set = set(true_set)
            print(f"  Protected set: design_region_union requested but "
                  f"no design region supplied — falling back to "
                  f"true_interface only ({len(protected_set)} positions)")
    else:
        protected_set = set(true_set)
        print(f"  Protected set: true_interface only "
              f"({len(protected_set)} positions)")
    plan["protected_set_source"] = protected_source
    plan["protected_set_idx"] = sorted(protected_set)

    candidates = []  # (dist, seq_idx)
    for seq_idx, dist in wrong_contacts:
        if seq_idx in protected_set:
            continue
        if not surface_flags.get(seq_idx, False):
            continue
        candidates.append((dist, seq_idx))
    candidates.sort()  # by distance ascending

    # Protection is strict.  A pool smaller than --max-mutations caps
    # the per-design count rather than loosening protection.

    # ── Overlap detection ───────────────────────────────────────────
    # How much of the predicted wrong interface overlaps the true
    # interface, as a fraction of the wrong interface.  Above 50% the
    # complex is in the pathological "almost already correct" regime,
    # where the filter removes most candidates as protected and the
    # survivors near-duplicate the second-shell fallback.
    #
    # interfaces_overlap is defined against true_set, not the protected
    # set, so the threshold does not shift with --protected-set-source.
    # n_overlap_protected is reported separately when the protected set
    # reaches past the true interface.
    n_wrong = len(predicted_wrong_idx)
    n_overlap_true = sum(1 for i in predicted_wrong_idx if i in true_set)
    n_overlap_protected = sum(
        1 for i in predicted_wrong_idx if i in protected_set
    )
    n_buried = sum(1 for i in predicted_wrong_idx if not surface_flags.get(i, False))
    interfaces_overlap = (n_overlap_true / n_wrong) if n_wrong else 0.0
    print(f"  Filtering: {n_wrong} wrong-interface residues")
    print(f"    -{n_overlap_true:>3d} overlap the true interface")
    if protected_set != true_set:
        print(f"    -{n_overlap_protected:>3d} overlap the full protected set "
              f"(true ∪ design region)")
    print(f"    -{n_buried:>3d} fail the surface test (buried / Gly)")
    print(f"    ={len(candidates):>3d} candidates remain")
    print(f"  Wrong/true interface overlap: {interfaces_overlap:.1%}")

    # Halve max_mutations in that regime.  An aggressive count buys
    # little when the cold-start is already close, and the second-shell
    # mutations it would spend itself on are less targeted than direct
    # contacts.  The halving rounds UP, with a floor of 1, so we never
    # ask for zero mutations.
    max_mutations_effective = args.max_mutations
    max_mutations_reduced_from: "Optional[int]" = None
    max_mutations_reduced_to: "Optional[int]" = None
    max_mutations_reduced_reason: "Optional[str]" = None
    if interfaces_overlap > 0.5:
        max_mutations_effective = max(1, (args.max_mutations + 1) // 2)
        if max_mutations_effective < args.max_mutations:
            max_mutations_reduced_from = args.max_mutations
            max_mutations_reduced_to = max_mutations_effective
            max_mutations_reduced_reason = "interfaces_overlap>50%"
            print(f"  Interfaces overlap >50%: reducing max_mutations "
                  f"from {args.max_mutations} to {max_mutations_effective} "
                  f"(gentler nudge for near-correct cold-start)")

    # Cap at the candidate pool size, because no design can sample
    # more unique positions than the pool holds.  The cap replaces an
    # earlier true_interface_fallback, which loosened protection and so
    # contaminated the design region.  Capping keeps protection strict,
    # and the second-shell fallback still covers an empty pool.
    if len(candidates) < max_mutations_effective:
        prev = max_mutations_effective
        max_mutations_effective = max(1, len(candidates))
        if max_mutations_effective < prev:
            # Overwrite reduced_from/to/reason only when the pool cap
            # is stricter than the overlap halving, so the audit trail
            # stays continuous.
            if (max_mutations_reduced_from is None
                    or max_mutations_effective < max_mutations_reduced_to):
                max_mutations_reduced_from = args.max_mutations
                max_mutations_reduced_to = max_mutations_effective
                max_mutations_reduced_reason = (
                    f"pool_size<{prev} (strict protection preserved)"
                )
            print(f"  Candidate pool ({len(candidates)}) < requested "
                  f"max_mutations ({prev}): capping to "
                  f"{max_mutations_effective}.  Strict protection "
                  f"preserved — design region remains unmutated.")

    # --candidate-pool-size is a floor, not a ceiling.  Every design
    # has to draw max_mutations_effective distinct positions, which a
    # smaller pool could not supply, so the pool is raised to that
    # count whenever the requested size falls below it.
    pool_size = max(args.candidate_pool_size, max_mutations_effective)
    candidate_pool = sorted(seq_idx for _, seq_idx in candidates[:pool_size])

    # ── Second-shell fallback ────────────────────────────────────────
    # An empty pool alongside a non-empty wrong interface means the
    # protected-set exclusion removed every direct candidate.  That
    # happens when the wrong interface overlaps the true one, as in
    # 7B1I, or sits entirely inside the design region under
    # design_region_union.  See _second_shell_fallback for the geometry
    # and the 3-candidate minimum.
    second_shell_used = False
    second_shell_pool_size = 0
    if (not candidate_pool) and predicted_wrong_idx:
        print()
        print("  =====================================================")
        print("  WARNING: naive candidate pool is empty.")
        print(f"  Wrong interface overlap with true interface: "
              f"{interfaces_overlap:.0%}")
        if protected_set != true_set:
            n_wrong_in_protected = sum(
                1 for i in predicted_wrong_idx if i in protected_set
            )
            print(f"  Wrong-interface residues in protected set: "
                  f"{n_wrong_in_protected}/{len(predicted_wrong_idx)}")
        print("  Every direct candidate was protected.  Falling back")
        print("  to second-shell residues (Cα within 6 Å of the")
        print("  predicted effector, excluding the wrong interface")
        print("  and the protected set).")
        if max_mutations_reduced_from is not None:
            print(f"  max_mutations reduced: "
                  f"{max_mutations_reduced_from} -> {max_mutations_reduced_to}")
        print("  Requires >= 3 candidates to proceed; otherwise we")
        print("  skip_steering and score the cold-start prediction as-is.")
        print("  =====================================================")
        candidate_pool = _second_shell_fallback(
            pred_rec_residues=pred_rec_residues,
            pred_eff_residues=pred_eff_residues,
            predicted_wrong_idx=predicted_wrong_idx,
            true_set=protected_set,  # historical name, see the docstring
            surface_flags_cache=surface_flags,
            radius=6.0,
            pool_size=pool_size,
            min_pool_size=3,
        )
        second_shell_pool_size = len(candidate_pool)
        if candidate_pool:
            second_shell_used = True
            print(f"  Second-shell pool: {len(candidate_pool)} residues")
        else:
            print("  Second-shell fallback produced < 3 candidates. "
                  "Skipping steering.")

    print(f"  Candidate pool: top {len(candidate_pool)} "
          f"(--candidate-pool-size {args.candidate_pool_size})"
          f"{' [second-shell fallback]' if second_shell_used else ''}")
    print(f"  Per-design mutations: {max_mutations_effective} sampled from the pool")
    print(f"  Steering mode: {args.mode}")

    # The design loop below reads args.max_mutations, so pushing the
    # effective value back is what makes a halving or a cap take
    # effect.
    args.max_mutations = max_mutations_effective

    if (args.mode == "conservative"
            and len(candidate_pool) <= args.max_mutations
            and args.n_designs > 1):
        print("  WARNING: mode=conservative with pool == max_mutations "
              "will produce N IDENTICAL designs.  Increase "
              "--candidate-pool-size for diversity.")

    # predict-one needs these for Jaccard overlap against the original
    # wrong interface, and post-hoc analysis to see what was selected.
    # This runs BEFORE the empty-pool skip, so _write_initial_only_csv
    # still sees the analysis.
    plan["true_interface_idx"] = true_idx
    plan["true_interface_source"] = true_interface_source
    plan["initial_wrong_interface_idx"] = predicted_wrong_idx
    plan["candidate_pool"] = candidate_pool

    # design_region_idx was written onto `plan` above.  No re-parsing.

    # ── Effector interface residue mask ─────────────────────────────
    # An effector residue counts as interface when its Cα is within
    # 8 Å of any receptor residue in true_interface_idx, matching the
    # RFDiffusion filter's Cα-distance convention.  The contamination
    # check then restricts contact detection to these residues' heavy
    # atoms, so a floppy effector loop brushing a non-binding receptor
    # face no longer counts.
    #
    # The filter has to be residue-level, because the RFDiffusion
    # ground truth is Cα-only.  An atom-level mask from it would hold
    # Cα atoms alone, and a 5 Å heavy-to-heavy contact between two Cα
    # atoms almost never registers in a prediction with side chains.
    #
    # An empty 'effector_interface_residues' in plan.json means no
    # filter, so contact detection is unrestricted.
    try:
        from compute_metrics import compute_effector_interface_residues
        true_idx_1b = [i + 1 for i in true_idx]  # 0-based -> 1-based
        eff_iface_residues = compute_effector_interface_residues(
            args.ground_truth,
            true_idx_1b,
            args.receptor_chain,
            args.effector_chain,
            cutoff_angstroms=8.0,
        )
        eff_iface_residues_list = sorted(int(r) for r in eff_iface_residues)
        plan["effector_interface_residues"] = eff_iface_residues_list
        plan["effector_interface_atom_cutoff_angstroms"] = 8.0
        # Back-compatibility alias for code paths that look for the
        # atom key.  compute_interface_contacts ignores atom_name when
        # it collapses the pairs to a residue set.
        plan["effector_interface_atoms"] = [
            [r, "any"] for r in eff_iface_residues_list
        ]
        print(f"  effector interface residue mask: "
              f"{len(eff_iface_residues_list)} residues (Cα cutoff 8.0 Å)")
    except Exception as e:
        print(f"  WARNING: could not compute effector interface residue mask: {e}")
        print("  Contamination checks will fall back to unfiltered "
              "contacts (legacy behaviour).")
        plan["effector_interface_residues"] = []
        plan["effector_interface_atoms"] = []
        plan["effector_interface_atom_cutoff_angstroms"] = None

    plan["n_protected"] = n_overlap_true
    plan["interfaces_overlap"] = round(interfaces_overlap, 4)
    plan["second_shell_fallback_used"] = second_shell_used
    plan["second_shell_pool_size"] = second_shell_pool_size
    if max_mutations_reduced_from is not None:
        plan["max_mutations_reduced_from"] = max_mutations_reduced_from
        plan["max_mutations_reduced_to"] = max_mutations_reduced_to
        plan["max_mutations_reduced_reason"] = max_mutations_reduced_reason
        # max_mutations_requested is the audit trail of what was asked
        # for.  max_mutations is overwritten with the effective value,
        # so downstream readers see the count actually used.
        plan["max_mutations_requested"] = max_mutations_reduced_from
        plan["max_mutations"] = max_mutations_effective
    plan["max_mutations_effective"] = max_mutations_effective

    # ── Pathological case: switch to sampling mode ──────────────────
    # An empty naive pool plus a second-shell fallback under 3
    # candidates means there is no physical room to steer by mutation.
    # Everything touching the effector is also on the true site, and
    # everything within mutation range is protected or buried.  7B1I is
    # the canonical example.
    #
    # Sampling mode replaces mutation with repetition.  The same
    # wild-type sequence is re-predicted N times under different seeds.
    # Boltz's diffusion sampling is stochastic, so 50 re-predictions
    # give 50 slightly different poses, some possibly closer to the
    # true site.  This pays off when the cold-start is already close.
    # 7B1I's 5.469 Å sits just above the 5.0 Å production threshold.
    sampling_mode = (
        (not candidate_pool)
        and bool(predicted_wrong_idx)
        and interfaces_overlap > 0.5
        and not second_shell_used
    )

    if not candidate_pool and not sampling_mode:
        if not predicted_wrong_idx:
            skip_reason = "no interface residues found"
            print(f"  No interface residues found within {args.contact_cutoff} Å.")
        else:
            skip_reason = "no mutatable residues after filtering"
            print("  No mutatable residues left after exclusion + surface "
                  "filter + second-shell fallback.  Aborting.")
        plan["skip_steering"] = True
        plan["skip_reason"] = skip_reason
        (workdir / "plan.json").write_text(json.dumps(plan, indent=2))
        # Single-row CSV here.  This path fires when steering found no
        # valid candidates, so the cold-start was not all-clean, or the
        # function would have returned earlier.  Its cold-start rows
        # must NOT be promoted, and the single row lands as a
        # "singleton" that extract_passing.py filters out.
        _write_initial_only_csv(
            workdir, plan,
            initial_ra_eff=initial_ra_eff,
            initial_ind_rec=initial_ind_rec,
            initial_ind_eff=initial_ind_eff,
        )
        return 1

    if sampling_mode:
        print()
        print("  =====================================================")
        print("  SAMPLING MODE: wrong/true interfaces overlap at "
              f"{interfaces_overlap:.0%}")
        print("  and the second-shell fallback produced < 3 candidates.")
        print("  No room to steer by mutation, but the cold-start")
        print(f"  ra_eff ({initial_ra_eff:.2f} Å) may be within reach of")
        print("  Boltz's sampling variance.  Running the wild-type")
        print(f"  sequence {args.n_designs} times with different seeds")
        print("  so Boltz's stochastic diffusion can try different poses.")
        print("  =====================================================")
        plan["sampling_mode"] = True
        plan["sampling_mode_reason"] = (
            f"interfaces overlap at {interfaces_overlap:.0%}; "
            f"second-shell fallback produced < 3 candidates; "
            f"re-sampling cold-start with {args.n_designs} seeds"
        )
        # skip_steering is deliberately not set, because predict-one
        # DOES need to run.  The empty candidate_pool is how the design
        # loop below knows to skip make_steered_sequence.

    # Both interface log files are vacuous in sampling mode, where the
    # pool is empty and no protected residue is distinct from the wrong
    # interface.
    if not sampling_mode:
        interface_log = workdir / "wrong_interface_residues.txt"
        with open(interface_log, "w") as f:
            f.write("# 1-based receptor residue, wt one-letter, "
                    "min_heavy_dist_to_eff (Å), in_true_site, "
                    "in_protected_set, surface_exposed, in_pool\n")
            # Sorted by distance ascending, so the table reads
            # top-down with the strongest contacts first.
            pool_set = set(candidate_pool)
            for seq_idx, dist in wrong_contacts:
                in_true = 1 if seq_idx in true_set else 0
                in_protected = 1 if seq_idx in protected_set else 0
                surface = 1 if surface_flags.get(seq_idx, False) else 0
                in_pool = 1 if seq_idx in pool_set else 0
                f.write(f"{seq_idx + 1}\t{rec_seq[seq_idx]}\t"
                        f"{dist:.2f}\t{in_true}\t{in_protected}\t"
                        f"{surface}\t{in_pool}\n")
        print(f"  Wrote {interface_log}")

        true_log = workdir / "true_interface_residues.txt"
        with open(true_log, "w") as f:
            f.write("# 1-based receptor residue, wt one-letter, "
                    "min_heavy_dist_to_eff (Å) — protected from mutation\n")
            for seq_idx, dist in sorted(true_contacts, key=lambda t: t[1]):
                f.write(f"{seq_idx + 1}\t{rec_seq[seq_idx]}\t{dist:.2f}\n")
        print(f"  Wrote {true_log}")

    rng = random.Random(args.seed)
    steered_root = workdir / "steered"
    steered_root.mkdir(exist_ok=True)

    num_seeds = args.num_seeds
    n_designs = args.n_designs

    # ── Generate unique sequences, deduplicated ─────────────────────
    # Each make_steered_sequence call is a random draw, so two draws
    # can collide.  Rather than spend GPU slots on duplicates, retry
    # and stop at n_designs uniques.  A combinatorial space smaller
    # than n_designs warns and returns as many as exist.
    MAX_RETRIES = n_designs * 10  # generous budget to find uniques

    unique_sequences: List[Tuple[str, List[Tuple[int, str, str]]]] = []
    seen_seqs: set = set()

    if sampling_mode:
        # One unique sequence, replicated n_designs × num_seeds times
        # under different seeds.
        unique_sequences.append((rec_seq, []))
        seen_seqs.add(rec_seq)
        if n_designs > 1:
            print(f"  Sampling mode: 1 unique sequence × "
                  f"{n_designs * num_seeds} total predictions "
                  f"({n_designs} design slots × {num_seeds} seed(s))")
    else:
        attempts = 0
        while len(unique_sequences) < n_designs and attempts < MAX_RETRIES:
            attempts += 1
            seq, muts = make_steered_sequence(
                rec_seq, candidate_pool, args.max_mutations, args.mode, rng,
            )
            if seq not in seen_seqs:
                seen_seqs.add(seq)
                unique_sequences.append((seq, muts))

        if len(unique_sequences) < n_designs:
            print(f"  WARNING: only {len(unique_sequences)} unique sequences "
                  f"found after {attempts} attempts (requested {n_designs}). "
                  f"The candidate pool ({len(candidate_pool)} positions, "
                  f"max {args.max_mutations} mutations, mode={args.mode}) "
                  f"may have limited combinatorial diversity.")
        else:
            collisions = attempts - len(unique_sequences)
            if collisions > 0:
                print(f"  Generated {len(unique_sequences)} unique sequences "
                      f"({collisions} collision(s) deduplicated)")

    n_unique = len(unique_sequences)
    total_predictions = n_unique * num_seeds
    if sampling_mode:
        total_predictions = n_designs * num_seeds

    plan["n_unique_sequences"] = n_unique
    plan["total_predictions"] = total_predictions

    print(f"  {n_unique} unique sequence(s) × {num_seeds} seed(s) = "
          f"{total_predictions} total prediction(s)")

    # ── Expand each unique sequence into num_seeds design entries ───
    design_idx = 0
    for seq_group, (mutated_seq, mutations) in enumerate(unique_sequences):
        # In sampling mode the one sequence is replicated n_designs
        # times, so n_designs keeps meaning "how many design slots".
        n_slots = n_designs if sampling_mode else 1

        for slot in range(n_slots):
            for seed_idx in range(num_seeds):
                if sampling_mode:
                    name = f"design_{slot:02d}_s{seed_idx}"
                elif num_seeds == 1:
                    name = f"design_{seq_group:02d}"
                else:
                    name = f"design_{seq_group:02d}_s{seed_idx}"

                d = steered_root / name
                d.mkdir(exist_ok=True)

                (d / "mutations.tsv").write_text(
                    "pos1\twt\tmut\n" + "".join(
                        f"{p}\t{w}\t{m}\n" for p, w, m in mutations
                    )
                )
                (d / "receptor.fasta").write_text(
                    f">receptor_{name}\n{mutated_seq}\n"
                )
                write_boltz_yaml(
                    d / "input",
                    mutated_seq, eff_seq,
                    PRED_REC_CHAIN, PRED_EFF_CHAIN,
                    effector_template_cif=eff_template_path,
                    constraints_block=constraints_block,
                )

                boltz_seed = args.seed + design_idx + 1

                plan["designs"].append({
                    "index": design_idx,
                    "name": name,
                    "dir": str(d.resolve()),
                    "yaml": str((d / "input" / "input.yaml").resolve()),
                    "total_mutations": sum(
                        1 for a, b in zip(rec_seq, mutated_seq) if a != b
                    ),
                    "mutated_positions": sorted(
                        p - 1 for p, _, _ in mutations
                    ),
                    "sequence_group": seq_group,
                    "seed_index": seed_idx,
                    "boltz_seed": boltz_seed,
                })
                design_idx += 1

    (workdir / "plan.json").write_text(json.dumps(plan, indent=2))
    print(f"\nWrote plan.json with {len(plan['designs'])} designs.")
    print("Ready for the array stage.")
    return 0


# ── Subcommand: predict-one ────────────────────────────────────────
def cmd_predict_one(args: argparse.Namespace) -> int:
    workdir: Path = args.workdir
    plan = json.loads((workdir / "plan.json").read_text())

    if plan.get("skip_steering"):
        print("plan.json says skip_steering=true; nothing to do.")
        return 0

    designs = plan["designs"]
    idx = args.index
    if idx < 0 or idx >= len(designs):
        print(f"index {idx} out of range (have {len(designs)} designs)",
              file=sys.stderr)
        return 2
    design = designs[idx]
    name = design["name"]
    d = Path(design["dir"])
    yaml_path = Path(design["yaml"])

    # ── Resume ──────────────────────────────────────────────────────
    # Large standalone-benchmark complexes can outrun the SLURM
    # walltime.  On resubmit the plan stage re-derives the SAME
    # designs, because its RNG is seeded, so one already on disk can be
    # reused.  Guard against drift by confirming its receptor sequence
    # matches this design's.
    _existing_pred = d / "prediction.pdb"
    if (d / "result.json").exists() and _existing_pred.exists():
        try:
            _want = (d / "receptor.fasta").read_text().splitlines()[-1].strip()
            _have = get_chain_sequence(_existing_pred, plan["pred_receptor_chain"])
            if _want and _have == _want:
                print(f"[predict-one] {name} (index {idx}) — already predicted, "
                      f"skipping (resume)", flush=True)
                return 0
        except Exception:
            pass

    print(f"[predict-one] {name} (index {idx})", flush=True)

    try:
        pred_pdb = run_boltz(
            yaml_path, d,
            plan["boltz_container"],
            plan["recycling_steps"], plan["diffusion_samples"],
            # The fallback formula keeps plan.json files written
            # before the num_seeds feature working.
            design.get("boltz_seed", int(plan["seed"]) + idx + 1),
            plan["no_kernels"],
        )
        ra_eff, ind_rec, ind_eff = binding_rmsds(
            pred_pdb, Path(plan["ground_truth"]),
            plan["pred_receptor_chain"], plan["pred_effector_chain"],
            plan["truth_receptor_chain"], plan["truth_effector_chain"],
        )
        shutil.copy2(pred_pdb, d / "prediction.pdb")

        # Same-site check against the wild-type wrong interface and
        # the true binding site, both from plan.json, under the plan
        # stage's contact definition, so the Jaccards are directly
        # comparable.  expected_rec_seq is the mutated sequence, since
        # this receptor differs from the wild type.
        try:
            mutated_seq = (d / "receptor.fasta").read_text().splitlines()[-1].strip()
            steered_contacts = find_contact_residues_heavy(
                d / "prediction.pdb",
                plan["pred_receptor_chain"], plan["pred_effector_chain"],
                plan["contact_cutoff"],
                expected_rec_seq=mutated_seq,
            )
            steered_iface = sorted(i for i, _ in steered_contacts)
            initial_wrong = plan.get("initial_wrong_interface_idx", [])
            true_site = plan.get("true_interface_idx", [])
            wrong_jaccard = jaccard(steered_iface, initial_wrong)
            true_jaccard = jaccard(steered_iface, true_site)
            n_shared_wrong = len(set(steered_iface) & set(initial_wrong))
            n_shared_true = len(set(steered_iface) & set(true_site))
        except Exception as e:
            print(f"  WARNING: same-site check failed: {e}", file=sys.stderr)
            steered_iface = []
            wrong_jaccard = float("nan")
            true_jaccard = float("nan")
            n_shared_wrong = 0
            n_shared_true = 0

        result = {
            "index": idx,
            "design": name,
            "total_mutations": design["total_mutations"],
            "receptor_aligned_effector_rmsd": ra_eff,
            "independent_receptor_rmsd": ind_rec,
            "independent_effector_rmsd": ind_eff,
            "wrong_jaccard": wrong_jaccard,
            "true_jaccard": true_jaccard,
            "n_shared_wrong": n_shared_wrong,
            "n_shared_true": n_shared_true,
            "n_design_interface_residues": len(steered_iface),
            "steered_interface_idx": steered_iface,
            "pdb": str((d / "prediction.pdb").resolve()),
            "status": "ok",
        }
    except Exception as e:
        print(f"  FAILED: {e}", file=sys.stderr)
        result = {
            "index": idx,
            "design": name,
            "total_mutations": design["total_mutations"],
            "receptor_aligned_effector_rmsd": float("nan"),
            "independent_receptor_rmsd": float("nan"),
            "independent_effector_rmsd": float("nan"),
            "wrong_jaccard": float("nan"),
            "true_jaccard": float("nan"),
            "n_shared_wrong": 0,
            "n_shared_true": 0,
            "n_design_interface_residues": 0,
            "steered_interface_idx": [],
            "pdb": "",
            "status": f"failed: {e}",
        }

    # NaN is not valid JSON, so serialise it to null on the way out.
    out = {k: (None if isinstance(v, float) and np.isnan(v) else v)
           for k, v in result.items()}
    (d / "result.json").write_text(json.dumps(out, indent=2))
    print(f"  receptor_aligned_effector_rmsd = {result['receptor_aligned_effector_rmsd']:.3f} Å")
    print(f"  independent_receptor_rmsd      = {result['independent_receptor_rmsd']:.3f} Å"
          f"  {'⚠ receptor deformed' if result['independent_receptor_rmsd'] > 5.0 else ''}")
    print(f"  independent_effector_rmsd      = {result['independent_effector_rmsd']:.3f} Å"
          f"  {'⚠ effector deformed' if result['independent_effector_rmsd'] > 5.0 else ''}")
    print(f"  wrong_jaccard                  = {result['wrong_jaccard']:.3f} "
          f"({result['n_shared_wrong']} shared with initial wrong site)")
    print(f"  true_jaccard                   = {result['true_jaccard']:.3f} "
          f"({result['n_shared_true']} shared with true site)")
    return 0 if result["status"] == "ok" else 1


# ── Subcommand: collect ────────────────────────────────────────────
def cmd_collect(args: argparse.Namespace) -> int:
    workdir: Path = args.workdir
    plan_path = workdir / "plan.json"

    if not plan_path.exists():
        # The plan stage crashed before writing plan.json, or never
        # ran.  collect runs on `afterany`, so it fires even when the
        # upstream chain was cancelled.  Write a marker rather than
        # raising.
        msg = (
            f"ERROR: {plan_path} does not exist.\n"
            f"  The plan stage failed or never ran.  Check:\n"
            f"  {workdir}/logs/plan_*.err\n"
        )
        sys.stderr.write(msg)
        (workdir / "summary.txt").write_text(msg)
        return 1

    plan = json.loads(plan_path.read_text())

    if plan.get("skip_steering"):
        print("plan.json says skip_steering=true; nothing to collect.")
        return 0

    rows: List[dict] = []
    for design in plan["designs"]:
        d = Path(design["dir"])
        rj = d / "result.json"
        seq_group = design.get("sequence_group", 0)
        seed_index = design.get("seed_index", 0)
        if not rj.exists():
            rows.append({
                "design": design["name"],
                "total_mutations": design["total_mutations"],
                "receptor_aligned_effector_rmsd": float("nan"),
                "independent_receptor_rmsd": float("nan"),
                "independent_effector_rmsd": float("nan"),
                "wrong_jaccard": float("nan"),
                "true_jaccard": float("nan"),
                "n_shared_wrong": 0,
                "n_shared_true": 0,
                "n_design_interface_residues": 0,
                "pdb": "",
                "status": "missing",
                "sequence_group": seq_group,
                "seed_index": seed_index,
            })
            continue
        r = json.loads(rj.read_text())
        ra_eff  = r.get("receptor_aligned_effector_rmsd")
        ind_rec = r.get("independent_receptor_rmsd")
        ind_eff = r.get("independent_effector_rmsd")
        wj = r.get("wrong_jaccard")
        tj = r.get("true_jaccard")
        rows.append({
            "design": r["design"],
            "total_mutations": r["total_mutations"],
            "receptor_aligned_effector_rmsd":
                float("nan") if ra_eff  is None else float(ra_eff),
            "independent_receptor_rmsd":
                float("nan") if ind_rec is None else float(ind_rec),
            "independent_effector_rmsd":
                float("nan") if ind_eff is None else float(ind_eff),
            "wrong_jaccard": float("nan") if wj is None else float(wj),
            "true_jaccard": float("nan") if tj is None else float(tj),
            "n_shared_wrong": int(r.get("n_shared_wrong") or 0),
            "n_shared_true": int(r.get("n_shared_true") or 0),
            "n_design_interface_residues":
                int(r.get("n_design_interface_residues") or 0),
            "pdb": r.get("pdb", ""),
            "status": r.get("status", ""),
            "sequence_group": seq_group,
            "seed_index": seed_index,
        })

    # A trustworthy design has both proteins folded.  Without this
    # filter the best row by receptor-aligned effector RMSD could have
    # a wrecked receptor, making the metric meaningless.  The
    # thresholds allow jitter and loop movement while flagging real
    # deformation.
    RECEPTOR_INTACT_CUTOFF = 5.0   # Å Cα RMSD vs ground truth
    EFFECTOR_INTACT_CUTOFF = 5.0   # Å Cα RMSD vs ground truth

    def is_intact(row):
        rec = row["independent_receptor_rmsd"]
        eff = row["independent_effector_rmsd"]
        if np.isnan(rec) or np.isnan(eff):
            return False
        return rec <= RECEPTOR_INTACT_CUTOFF and eff <= EFFECTOR_INTACT_CUTOFF

    for r in rows:
        r["receptor_intact"] = 1 if is_intact(r) else 0

    # A positive delta means the binding mode moved closer to the
    # truth than the wild-type prediction.
    initial_ra_eff = float(plan["initial_receptor_aligned_effector_rmsd"])
    initial_ind_rec = float(plan.get("initial_independent_receptor_rmsd", float("nan")))
    initial_ind_eff = float(plan.get("initial_independent_effector_rmsd", float("nan")))
    for r in rows:
        if np.isnan(r["receptor_aligned_effector_rmsd"]):
            r["delta_receptor_aligned_effector_rmsd"] = float("nan")
        else:
            r["delta_receptor_aligned_effector_rmsd"] = (
                initial_ra_eff - r["receptor_aligned_effector_rmsd"]
            )

    # Intact designs first, then deformed ones, both ascending by
    # receptor_aligned_effector_rmsd.  In the deformed block that
    # number means nothing biologically, and is ranked the same way
    # only for visual consistency.  `receptor_intact == 0` marks the
    # discard.
    def _sort_key(r, key):
        v = r[key]
        return (np.isnan(v), v)

    intact_rows   = [r for r in rows if r["receptor_intact"]]
    deformed_rows = [r for r in rows if not r["receptor_intact"]]
    intact_rows.sort(  key=lambda r: _sort_key(r, "receptor_aligned_effector_rmsd"))
    deformed_rows.sort(key=lambda r: _sort_key(r, "receptor_aligned_effector_rmsd"))
    rows = intact_rows + deformed_rows

    # The wild-type prediction is the baseline, at rank 0, so it sits
    # above the steered designs without joining their numbering.
    # wrong_jaccard is 1.0 by definition, since this row IS the wrong
    # site, and true_jaccard is that site's overlap with the true
    # site.
    initial_wrong_idx = plan.get("initial_wrong_interface_idx", [])
    true_idx = plan.get("true_interface_idx", [])
    initial_n_iface = len(initial_wrong_idx)
    initial_n_shared_true = len(set(initial_wrong_idx) & set(true_idx))
    initial_true_jaccard = jaccard(initial_wrong_idx, true_idx)
    initial_intact = 1 if (
        not np.isnan(initial_ind_rec) and not np.isnan(initial_ind_eff)
        and initial_ind_rec <= RECEPTOR_INTACT_CUTOFF
        and initial_ind_eff <= EFFECTOR_INTACT_CUTOFF
    ) else 0
    initial_pdb = workdir / "initial_prediction.pdb"
    initial_row = {
        "design": "initial",
        "total_mutations": 0,
        "receptor_aligned_effector_rmsd": initial_ra_eff,
        "delta_receptor_aligned_effector_rmsd": 0.0,
        "independent_receptor_rmsd": initial_ind_rec,
        "independent_effector_rmsd": initial_ind_eff,
        "receptor_intact": initial_intact,
        "wrong_jaccard": 1.0,
        "n_shared_wrong": initial_n_iface,
        "true_jaccard": initial_true_jaccard,
        "n_shared_true": initial_n_shared_true,
        "n_design_interface_residues": initial_n_iface,
        "status": "initial",
        "pdb": str(initial_pdb.resolve()) if initial_pdb.exists() else "",
    }

    csv_path = workdir / "steered_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "rank", "design", "sequence_group", "seed_index",
            "total_mutations",
            "receptor_aligned_effector_rmsd",
            "delta_receptor_aligned_effector_rmsd",
            "independent_receptor_rmsd",
            "independent_effector_rmsd",
            "receptor_intact",
            "wrong_jaccard", "n_shared_wrong",
            "true_jaccard",  "n_shared_true",
            "n_design_interface_residues",
            "status", "pdb",
        ])
        w.writeheader()
        # Wild-type prediction first, at rank 0.
        initial_row["sequence_group"] = ""
        initial_row["seed_index"] = ""
        w.writerow({"rank": 0, **initial_row})
        # Then the steered designs, ranked 1..N (intact first).
        for rank, r in enumerate(rows, start=1):
            w.writerow({"rank": rank, **r})
    print(f"Wrote {csv_path}")

    # ── Multi-seed aggregate summary ────────────────────────────────
    num_seeds = plan.get("num_seeds", 1)
    if num_seeds > 1:
        from collections import defaultdict
        groups: Dict[int, List[dict]] = defaultdict(list)
        for r in rows:
            sg = r.get("sequence_group")
            if sg is not None and sg != "":
                groups[int(sg)].append(r)

        agg_rows = []
        for sg in sorted(groups.keys()):
            members = groups[sg]
            n_seeds_ok = len(members)
            # The first member supplies the fields that do not vary.
            first = members[0]
            agg = {
                "sequence_group": sg,
                "n_seeds": n_seeds_ok,
                "total_mutations": first["total_mutations"],
                "rep_design": first["design"],
            }
            # Aggregate numeric metrics
            for key in ("receptor_aligned_effector_rmsd",
                        "independent_receptor_rmsd",
                        "independent_effector_rmsd"):
                vals = [m[key] for m in members
                        if not np.isnan(m[key])]
                if vals:
                    agg[f"{key}_mean"] = round(np.mean(vals), 4)
                    agg[f"{key}_std"] = round(np.std(vals, ddof=1), 4) if len(vals) > 1 else 0.0
                    agg[f"{key}_min"] = round(min(vals), 4)
                    agg[f"{key}_max"] = round(max(vals), 4)
                else:
                    for suffix in ("_mean", "_std", "_min", "_max"):
                        agg[f"{key}{suffix}"] = ""

            # Count intact across seeds
            n_intact_seeds = sum(1 for m in members if m["receptor_intact"])
            agg["n_intact_seeds"] = n_intact_seeds
            agg["any_seed_intact"] = 1 if n_intact_seeds > 0 else 0
            agg_rows.append(agg)

        agg_csv_path = workdir / "steered_results_aggregate.csv"
        if agg_rows:
            agg_fields = list(agg_rows[0].keys())
            with open(agg_csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=agg_fields)
                w.writeheader()
                for ar in agg_rows:
                    w.writerow(ar)
            print(f"Wrote {agg_csv_path} ({len(agg_rows)} sequence groups)")

    def format_best(label, row):
        """Return one block of "best <label>" lines for the summary."""
        if row is None or np.isnan(row["receptor_aligned_effector_rmsd"]):
            return f"{label}: no successful designs\n"
        delta = row["delta_receptor_aligned_effector_rmsd"]
        rec = row["independent_receptor_rmsd"]
        eff = row["independent_effector_rmsd"]
        rec_warn = "  ⚠ receptor deformed" if rec > 5.0 else ""
        eff_warn = "  ⚠ effector deformed" if eff > 5.0 else ""
        n_if = row["n_design_interface_residues"]
        return (
            f"{label}: {row['receptor_aligned_effector_rmsd']:.3f} Å "
            f"({row['design']})\n"
            f"  delta_receptor_aligned_effector_rmsd: {delta:+.3f} Å\n"
            f"  independent_receptor_rmsd:            {rec:.3f} Å{rec_warn}\n"
            f"  independent_effector_rmsd:            {eff:.3f} Å{eff_warn}\n"
            f"  wrong_jaccard:                        {row['wrong_jaccard']:.3f} "
            f"({row['n_shared_wrong']} of {n_if} shared with initial wrong site)\n"
            f"  true_jaccard:                         {row['true_jaccard']:.3f} "
            f"({row['n_shared_true']} of {n_if} shared with true site)\n"
        )

    # Best by receptor_aligned_effector_rmsd with no filter, so any
    # fold qualifies.  After the intact-first sort rows[0] is the top
    # intact design, so the overall best is picked separately.
    non_nan = [r for r in rows
               if not np.isnan(r["receptor_aligned_effector_rmsd"])]
    best_overall = (
        min(non_nan, key=lambda r: r["receptor_aligned_effector_rmsd"])
        if non_nan else None
    )

    # The biologically meaningful best, because the binding-mode RMSD
    # only reflects docking when both partners are folded.
    best_intact = intact_rows[0] if intact_rows else None
    n_intact = len(intact_rows)

    n_protected = plan.get("n_protected", 0)
    n_true = len(plan.get("true_interface_idx", []))
    n_initial_wrong = len(plan.get("initial_wrong_interface_idx", []))
    n_pool = len(plan.get("candidate_pool", []))
    mode = plan.get("mode", "?")
    max_mut = plan.get("max_mutations", "?")
    pool_size = plan.get("candidate_pool_size", "?")
    eff_template = "yes" if plan.get("effector_template_cif") else "no"

    summary = (
        f"Boltz negative steering — summary\n"
        f"=================================\n"
        f"Ground truth:           {plan['ground_truth']}\n"
        f"Receptor chain (truth): {plan['truth_receptor_chain']}\n"
        f"Effector chain (truth): {plan['truth_effector_chain']}\n"
        f"Contact cutoff:         {plan['contact_cutoff']} Å (heavy-atom)\n"
        f"Effector template used: {eff_template}\n"
        f"Steering mode:          {mode}\n"
        f"Max mutations / design: {max_mut}\n"
        f"Candidate pool size:    {pool_size}\n"
        f"Seeds per sequence:     {num_seeds}\n"
        f"Unique sequences:       {plan.get('n_unique_sequences', len(plan['designs']))}\n"
        f"Designs predicted:      {len(plan['designs'])}\n"
        f"Designs intact:         {n_intact} of {len(rows)}  "
        f"(independent_receptor_rmsd ≤ {RECEPTOR_INTACT_CUTOFF} Å AND "
        f"independent_effector_rmsd ≤ {EFFECTOR_INTACT_CUTOFF} Å)\n\n"
        f"True binding site:      {n_true} residue(s)\n"
        f"Initial wrong site:     {n_initial_wrong} residue(s)\n"
        f"Protected (overlap):    {n_protected} residue(s)\n"
        f"Candidate pool:         {n_pool} residue(s)\n\n"
        f"Initial receptor_aligned_effector_rmsd: {initial_ra_eff:.3f} Å\n"
        f"Initial independent_receptor_rmsd:      {initial_ind_rec:.3f} Å\n"
        f"Initial independent_effector_rmsd:      {initial_ind_eff:.3f} Å\n\n"
        f"{format_best('Best overall (any fold)', best_overall)}\n"
        f"{format_best('Best with intact proteins', best_intact)}"
    )
    (workdir / "summary.txt").write_text(summary)
    print()
    print(summary)
    return 0


# ── CLI ────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("plan", help="initial prediction + write designs")
    pp.add_argument("--ground-truth", type=Path, required=True,
                    help="Ground truth PDB (sequences will be extracted from this)")
    pp.add_argument("--receptor", dest="receptor_chain", required=True,
                    help="Chain ID for receptor in ground truth PDB")
    pp.add_argument("--effector", dest="effector_chain", required=True,
                    help="Chain ID for effector in ground truth PDB")
    pp.add_argument("--workdir", type=Path, required=True)
    pp.add_argument("--boltz-container", required=True)
    pp.add_argument("--rmsd-threshold", type=float, default=5.0,
                    help="Skip steering if initial effector_rmsd ≤ this (Å)")
    pp.add_argument("--contact-cutoff", type=float, default=4.5,
                    help="Heavy-atom contact cutoff for interface "
                         "detection (Å, default 4.5)")
    pp.add_argument("--mode",
                    choices=["strong", "mild", "conservative", "alanine"],
                    default="strong",
                    help="Steering mode (default: strong). "
                         "strong = bulky aromatics + charges + proline; "
                         "mild = D/E/K/R only; "
                         "conservative = volume-matched chemistry flip; "
                         "alanine = classical alanine scanning.")
    pp.add_argument("--max-mutations", type=int, default=6,
                    help="Number of receptor residues to mutate per "
                         "design (default 6). Sampled without replacement "
                         "from the candidate pool.")
    pp.add_argument("--candidate-pool-size", type=int, default=6,
                    help="Size of the pool of candidate residues to sample "
                         "from (default 6, i.e. == max-mutations so all "
                         "designs hit the same positions). Set larger than "
                         "max-mutations to enable single-site / few-site "
                         "sampling experiments — e.g. --max-mutations 1 "
                         "--candidate-pool-size 6 --n-designs 6 gives "
                         "classical scanning across the top 6 contacts.")
    pp.add_argument("--protected-set-source",
                    choices=["true_interface", "design_region_union"],
                    default="design_region_union",
                    help="Which positions are excluded from the steering "
                         "candidate pool.  'true_interface' excludes only "
                         "the residues that contact the effector in the "
                         "ground truth.  'design_region_union', the "
                         "default, also excludes the RFDiffusion design "
                         "region.  A steering mutation inside that region "
                         "contaminates the rescue by construction, because "
                         "the construct carries the designed sequence "
                         "rather than the steered one.  The union keeps "
                         "the candidate pool larger than a Ca-radius "
                         "expansion around the interface would.  It needs "
                         "--design-region-indices-file and falls back to "
                         "true_interface without one.")
    pp.add_argument("--effector-template",
                    dest="effector_template", action="store_true", default=False,
                    help="Use the ground-truth effector chain as a "
                         "structural template for Boltz.  OFF by default.  "
                         "The effector is fixed throughout the experiment, "
                         "so pinning its fold doesn't bias which surface "
                         "Boltz picks for the receptor.  It does need the "
                         "ground-truth complex, so a novel target cannot "
                         "use it.  Every committed result was produced "
                         "without it, and until 2026-11-25 this flag was "
                         "on by default but the template never reached "
                         "Boltz, so turning it on now is a change of "
                         "behaviour rather than a return to an old one.")
    pp.add_argument("--no-effector-template",
                    dest="effector_template", action="store_false",
                    help="Force the effector template off.  Redundant now "
                         "that off is the default, and kept so a caller "
                         "that passes it keeps working.")
    pp.add_argument("--boltz-constraints",
                    dest="boltz_constraints", action="store_true", default=False,
                    help="Add the same pocket + dense-contact constraints used "
                         "by structure-prediction-benchmarking's BOLTZ2_CONSTRAINED "
                         "(Cα pocket ≤8Å + up to 50 contact pairs ≤10Å, derived "
                         "once from the ground-truth complex) to EVERY Boltz-2 "
                         "prediction in this run.  That is the initial "
                         "prediction, the extra cold-start seeds, every "
                         "steered design and the reversion pass.  Off by "
                         "default.")
    pp.add_argument("--n-designs", type=int, default=10)
    pp.add_argument("--num-seeds", type=int, default=1,
                    help="Number of independent Boltz seeds per unique "
                         "sequence (default 1).  Total predictions = "
                         "n_unique_sequences × num_seeds.  Higher values "
                         "give confidence intervals on metrics at the "
                         "cost of GPU time.  --num-seeds 1 --n-designs "
                         "100 maximises sequence-space exploration; "
                         "--num-seeds 5 --n-designs 20 gives 5-fold "
                         "confidence on 20 unique sequences for the "
                         "same GPU budget.")
    pp.add_argument("--recycling-steps", type=int, default=3)
    pp.add_argument("--diffusion-samples", type=int, default=1)
    pp.add_argument("--seed", type=int, default=0)
    pp.add_argument("--no-kernels", action="store_true")

    # These separate the sequence Boltz folds from the reference used
    # for contact filtering and RMSD ranking.  The use case is an
    # A-to-B sequence walk, where Boltz gets a sequence mutated toward
    # another complex but the ranking still runs against the original
    # ground truth.  The length must match, because that machinery is
    # index-based.
    pp.add_argument("--receptor-fasta", type=Path, default=None,
                    help="Override the receptor sequence Boltz folds. "
                         "Ground truth still anchors the contact "
                         "filter, true-site protection, and RMSD "
                         "ranking — only the input sequence changes. "
                         "Length must match the ground-truth receptor "
                         "chain (point mutations only, no indels).")
    pp.add_argument("--effector-fasta", type=Path, default=None,
                    help="Override the effector sequence Boltz folds. "
                         "Same semantics as --receptor-fasta. Length "
                         "must match the ground-truth effector chain.")

    # These bypass heavy-atom contact detection on the ground truth.
    # Use them when the ground truth is Cα-only, such as an RFDiffusion
    # design.  The indices are 0-based seq_idx values into the receptor
    # chain in file order, as find_contact_residues_heavy would have
    # returned.
    pp.add_argument("--true-interface-indices", type=str, default=None,
                    help="Comma-separated list of 0-based receptor "
                         "seq_idx values that make up the TRUE (protected) "
                         "interface. Bypasses heavy-atom contact detection "
                         "on the ground truth. Use this when the ground "
                         "truth is Cα-only (e.g. an RFDiffusion design) "
                         "and the contact list has been precomputed "
                         "elsewhere. Mutually exclusive with "
                         "--true-interface-indices-file.")
    pp.add_argument("--true-interface-indices-file", type=Path, default=None,
                    help="Path to a file containing 0-based receptor "
                         "seq_idx values for the TRUE interface, one per "
                         "line or comma/whitespace separated. Same semantics "
                         "as --true-interface-indices. Lines starting "
                         "with '#' are ignored.")

    # The receptor positions RFDiffusion built de novo and ProteinMPNN
    # will design.  The reversion pass protects
    # design_region ∪ true_interface, because the wet-lab construct
    # will not carry them as-is, so a steering mutation landing there
    # has to be reverted.  The design-region file is 1-based
    # coordinates, converted to 0-based to match true_interface_idx.
    pp.add_argument("--design-region-indices-file", type=Path, default=None,
                    help="Path to a file containing 1-BASED receptor "
                         "positions that constitute the ProteinMPNN-"
                         "designed region for this design.  Produced "
                         "upstream from rfdiffusion_metrics.json by the "
                         "design pipeline.  Accepts comma-"
                         "separated integers and 'start-end' ranges; "
                         "'#' starts a line comment.  When supplied, "
                         "plan.json gains a 'design_region_idx' field "
                         "(0-based, matching true_interface_idx) that "
                         "the reversion pass reads at end-of-cycle.  "
                         "Optional: without it, the reversion pass "
                         "falls back to protecting only the true "
                         "interface.")

    # Diagnostic mode.  Run the initial prediction, compute the three
    # RMSDs, write plan.json and stop.  A cheap way to characterise a
    # variant's cold-start pose.  Same as the all-clean early return,
    # but unconditional.
    pp.add_argument("--skip-steering", action="store_true",
                    help="Run only the initial prediction; do not "
                         "generate or schedule any steered designs. "
                         "plan.json is still written so that collect "
                         "produces a one-line summary.")
    pp.set_defaults(func=cmd_plan)

    pr = sub.add_parser("predict-one", help="predict one steered design")
    pr.add_argument("--workdir", type=Path, required=True)
    pr.add_argument("--index", type=int, required=True,
                    help="0-based design index (= SLURM_ARRAY_TASK_ID)")
    pr.set_defaults(func=cmd_predict_one)

    pc = sub.add_parser("collect", help="rank + write final CSV")
    pc.add_argument("--workdir", type=Path, required=True)
    pc.set_defaults(func=cmd_collect)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
