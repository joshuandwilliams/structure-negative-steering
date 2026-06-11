#!/usr/bin/env python3
"""
boltz2_iterate_steering.py
==========================
Multi-cycle iterative negative steering on top of boltz2_negative_steering.

Each pathway is a chain of single intact-and-novel designs through cycles:
  cycle_0/design_05 -> cycle_1/design_03 -> cycle_2/design_07 -> ...

A cycle-N+1 design "passes" if (a) the receptor stays folded and (b) its
effector_rmsd to every upstream pathway prediction (parent, grandparent,
..., wild-type) is >= --novelty-cutoff.  Up to --max-passing survivors
per parent are kept, selected by max-min distance to upstream poses
(furthest-from-anything-seen).

Subcommands
-----------
  iterate-plan
      Build cycle N+1 plan for one (parent_pathway, parent_design)
      pair.  Reads the parent's prediction.pdb, runs contact detection
      against the new wrong interface, accumulates mutations from
      every cycle in the parent chain, generates --n-designs candidate
      sequences, writes plan.json.

  iterate-collect
      Read every cycle-N+1 candidate result.json for one pathway, run
      the novelty filter against all upstream predictions, run the
      max-min selection, write passing.json.  If cycle index < max
      and there are survivors, self-resubmit cycle N+2 jobs via
      sbatch (one chain per survivor).

  aggregate
      Walk the whole multi-cycle tree under a workdir and write
      pathways.json (the whole tree), cycle_statistics.csv (bell-curve
      data), and all_results_multicycle.csv (every prediction ever
      made, ranked).

  compute-final-metrics
      Opt-in post-aggregation step.  For every row in
      all_results_multicycle.csv that passes a final ranking filter
      (default: receptor_aligned_effector_rmsd < 3 Å AND
      receptor_intact == 1), shell out to compute_metrics.py to
      compute confidence metrics (pTM, ipTM, ipSAE, actifPTM, pLDDT,
      PAE) and write a joined all_results_multicycle_with_metrics.csv.
      Structural RMSDs are NOT re-computed here — the steering
      pipeline already writes them to all_results_multicycle.csv
      (receptor_aligned_effector_rmsd, independent_receptor_rmsd,
      independent_effector_rmsd) using the correct truth-vs-pred
      chain mapping.  Kept separate from `aggregate` so the fast
      stdlib-only pass stays fast; this one needs numpy and should
      run in the boltz container.

The single-cycle script (boltz2_negative_steering.py) is responsible
for cycle 0 — this orchestrator only handles cycles 1+.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# numpy is only needed by the in-container phases (compute-distances,
# kickoff-distances).  The phase-2 entry points (iterate-collect,
# kickoff, aggregate) run outside the singularity container where
# numpy is not installed, so we defer the numpy and bns imports to
# inside the functions that actually need them.
#
# Anything that needs to test for NaN in phase 2 uses math.isnan
# from stdlib, which works on plain Python floats.

SCRIPT_DIR = Path(__file__).resolve().parent

# Make sibling bin/ modules importable at module load time so the
# reversion-confidence constant below can be imported unconditionally.
# reversion.py is pure-stdlib at module level (its header guarantees
# this), so this insert does NOT pull numpy / Bio.Align into Phase-2
# entry points.  boltz2_negative_steering import stays lazy because it
# DOES transitively import numpy and Bio.Align.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Single source of truth for the reverted-confidence column set —
# defined in reversion.py next to the per_seed_records dict it must
# stay in sync with.  See that file for the field-by-field rationale.
from reversion import _REVERTED_CONFIDENCE_FIELDS  # noqa: E402

# Receptor-fold "intact" threshold — sourced from PipelineInternalThresholds
# (Phase 4 §2.13) with literal fallback so this module stays importable in
# environments without the thresholds catalogue (Phase 2 entry points).
try:
    from pipeline_thresholds import PipelineInternalThresholds as _PIT  # noqa: E402
    _INTACT_THRESHOLD = _PIT.default().intact_threshold
except Exception:
    _INTACT_THRESHOLD = 5.0

# Lazy import of boltz2_negative_steering — it transitively imports
# numpy and Bio.Align which are only present inside the singularity
# container.  Phase-2 entry points (iterate-collect, kickoff,
# aggregate) must NOT trigger this import.
_BNS_MODULE = None

def _lazy_bns():
    global _BNS_MODULE
    if _BNS_MODULE is None:
        sys.path.insert(0, str(SCRIPT_DIR))
        import boltz2_negative_steering as bns
        _BNS_MODULE = bns
    return _BNS_MODULE


# ───────────────────────────────────────────────────────────────────────
# Pathway labels and tree navigation
# ───────────────────────────────────────────────────────────────────────
# A pathway label is a dotted concatenation of (cycle, design_index)
# pairs, e.g. "c0d05.c1d03.c2d07" meaning "cycle 0 design 05 -> cycle 1
# design 03 -> cycle 2 design 07".  The label is unique across the
# whole experiment and is used both as a directory name and as the
# index into pathways.json.

def make_pathway_label(parent_label: Optional[str], cycle: int, design_idx: int) -> str:
    """Append a (cycle, design) pair to a parent pathway label."""
    seg = f"c{cycle}d{design_idx:02d}"
    if parent_label is None or parent_label == "":
        return seg
    return f"{parent_label}.{seg}"


def parse_pathway_label(label: str) -> List[Tuple[int, int]]:
    """Inverse of make_pathway_label.  Returns list of (cycle, design)."""
    out = []
    for seg in label.split("."):
        # seg is "cNdMM"
        if not seg.startswith("c") or "d" not in seg:
            raise ValueError(f"Bad pathway segment: {seg}")
        cyc, _, dsg = seg[1:].partition("d")
        out.append((int(cyc), int(dsg)))
    return out


def pathway_workdir(experiment_root: Path, label: str) -> Path:
    """Map a *parent* pathway label to its directory under
    experiment_root.  Cycle 0 lives at <experiment_root>/cycle_0/
    (no pathway prefix because there's only one parent — the wild
    type).  Higher cycles live at <experiment_root>/cycle_N/pathway_<label>/.

    NOTE: the input must be a *parent* label (the label that names
    the directory).  Passing a *leaf* label here for cycle>0 returns
    a path to a directory that does not exist — leaf labels are
    design entries inside their parent's directory, not directories
    of their own.  read_upstream_chain handles this distinction
    inline; this helper is kept for callers that already have the
    parent label in hand.
    """
    segs = parse_pathway_label(label)
    last_cycle = segs[-1][0]
    if last_cycle == 0:
        return experiment_root / "cycle_0"
    return experiment_root / f"cycle_{last_cycle}" / f"pathway_{label}"


def parent_pathway_label(label: str) -> Optional[str]:
    """Drop the last segment.  Returns None for cycle-0 leaves."""
    segs = label.split(".")
    if len(segs) <= 1:
        return None
    return ".".join(segs[:-1])


# ───────────────────────────────────────────────────────────────────────
# Reading the upstream chain
# ───────────────────────────────────────────────────────────────────────
def read_upstream_chain(experiment_root: Path, leaf_label: str) -> List[Dict]:
    """Walk from a leaf pathway up to (but not including) the wild-type
    initial prediction.  Returns a list of dicts in oldest-first order:

        [
          {label, cycle, design_idx, prediction_pdb, mutated_seq, mutated_positions, plan_json},
          ...
        ]

    'mutated_positions' here is the SET of receptor positions that were
    mutated to reach this prediction (cumulative across cycles).
    The wild-type prediction is added separately by the caller — it's
    not part of the chain because it has no design index.

    Layout note: a cycle-N pathway directory is named after its
    PARENT pathway label, e.g. cycle_1/pathway_c0d05/ holds the
    cycle-1 children of cycle-0 design 05.  The design at index 03
    inside that dir is the leaf c0d05.c1d03 (a design entry, NOT a
    directory).  So when looking up a cycle>0 entry we use the LEAF's
    parent label as the directory name, not the leaf label itself.
    """
    chain = []
    label = leaf_label
    while label is not None:
        segs = parse_pathway_label(label)
        cycle, design_idx = segs[-1]
        wd = experiment_root / "cycle_0" if cycle == 0 else (
            experiment_root / f"cycle_{cycle}"
            / f"pathway_{parent_pathway_label(label)}"
        )
        plan = json.loads((wd / "plan.json").read_text())

        if cycle == 0:
            # Cycle 0 — design lives in steered/<name>/
            design_meta = plan["designs"][design_idx]
            d_dir = Path(design_meta["dir"])
            pred_pdb = d_dir / "prediction.pdb"
            mut_path = d_dir / "mutations.tsv"
            mutated_positions = parse_mutations_tsv(mut_path)
            mutated_seq = (d_dir / "receptor.fasta").read_text().splitlines()[-1].strip()
        else:
            # Cycles 1+ — same layout but inside a pathway dir
            design_meta = plan["designs"][design_idx]
            d_dir = Path(design_meta["dir"])
            pred_pdb = d_dir / "prediction.pdb"
            mut_path = d_dir / "mutations.tsv"
            mutated_positions = parse_mutations_tsv(mut_path)
            mutated_seq = (d_dir / "receptor.fasta").read_text().splitlines()[-1].strip()

        chain.append({
            "label": label,
            "cycle": cycle,
            "design_idx": design_idx,
            "prediction_pdb": str(pred_pdb),
            "mutated_seq": mutated_seq,
            "mutated_positions": mutated_positions,
            "plan_json": str(wd / "plan.json"),
        })
        label = parent_pathway_label(label)

    chain.reverse()  # oldest first
    return chain


def parse_mutations_tsv(tsv_path: Path) -> List[int]:
    """Read a mutations.tsv file and return the 0-based positions
    that were mutated.  Format: header 'pos1\\twt\\tmut' then one
    line per mutation with 1-based pos1."""
    if not tsv_path.exists():
        return []
    out = []
    with open(tsv_path) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if not parts or not parts[0]:
                continue
            out.append(int(parts[0]) - 1)
    return out


def read_mutations_tsv_full(tsv_path: Path) -> List[Tuple[int, str, str]]:
    """Read a mutations.tsv file and return full (pos1, wt, mut)
    tuples.  Preserves 1-based positions (the TSV's native format)
    and uppercase one-letter codes."""
    if not tsv_path.exists():
        return []
    out = []
    with open(tsv_path) as f:
        next(f)  # header: pos1\twt\tmut
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3 or not parts[0]:
                continue
            try:
                pos = int(parts[0])
            except ValueError:
                continue
            wt = parts[1].strip().upper()
            mut = parts[2].strip().upper()
            if not wt or not mut:
                continue
            out.append((pos, wt, mut))
    return out


def format_mutation_strings(
    mutations: List[Tuple[int, str, str]],
    chain_id: str = "A",
) -> Tuple[str, str]:
    """Build the two portable mutation-description strings from a
    list of (pos1, wt, mut) tuples.

    Returns (chimerax_spec, aa_changes):
      chimerax_spec:  "/A:5,12,37"  — copy-paste ready; user prepends
                      their own "#N" model number.  Positions are sorted
                      ascending and deduplicated.
      aa_changes:     "L5A V12K G37P" — space-separated wt+pos+mut
                      tokens, also sorted by position.

    Empty mutation list → ("", "").
    """
    if not mutations:
        return ("", "")
    # Sort by position and deduplicate: if the same position appears
    # twice (cumulative history across cycles where an ancestor touched
    # a position and a descendant touched it again), keep the first
    # occurrence in the chain order — which is the earliest edit, not
    # the most recent one.  Dedup by position only for the chimerax
    # spec; the aa-changes string keeps every entry because re-edits
    # are scientifically meaningful (even though the pipeline's
    # burnt-positions logic usually prevents them).
    seen = set()
    unique_positions = []
    for pos, _wt, _mut in mutations:
        if pos in seen:
            continue
        seen.add(pos)
        unique_positions.append(pos)
    unique_positions.sort()
    chimerax_spec = f"/{chain_id}:" + ",".join(str(p) for p in unique_positions)

    aa_changes_sorted = sorted(mutations, key=lambda t: t[0])
    aa_changes = " ".join(f"{wt}{pos}{mut}" for pos, wt, mut in aa_changes_sorted)

    return (chimerax_spec, aa_changes)


def read_cumulative_mutations(
    experiment_root: Path,
    leaf_label: str,
) -> List[Tuple[int, str, str]]:
    """Walk from a leaf pathway up to cycle 0 and collect every
    (pos1, wt, mut) tuple from every ancestor's mutations.tsv.
    Returns them in oldest-first order (cycle 0 edits first).
    Used by cmd_aggregate to build the mutations_chimerax and
    mutations_aa columns for cycle-1+ rows, where the set of edits
    that produced the design is the union of its ancestors' edits.

    Note: this is a thinner variant of read_upstream_chain that
    bypasses the per-entry sequence/pdb loading and just pulls
    mutations.tsv from each ancestor's design directory.  Kept
    separate so it can be called without triggering the heavier
    chain-walking machinery.
    """
    out: List[Tuple[int, str, str]] = []
    label = leaf_label
    # Walk up from leaf to root.  We collect in leaf-first order,
    # then reverse.
    rev_accum: List[List[Tuple[int, str, str]]] = []
    while label is not None:
        segs = parse_pathway_label(label)
        cycle, design_idx = segs[-1]
        if cycle == 0:
            wd = experiment_root / "cycle_0"
            # Flat-layout fallback
            if not (wd / "plan.json").exists() and (experiment_root / "plan.json").exists():
                wd = experiment_root
        else:
            wd = (
                experiment_root / f"cycle_{cycle}"
                / f"pathway_{parent_pathway_label(label)}"
            )
        plan_path = wd / "plan.json"
        if plan_path.exists():
            try:
                plan = json.loads(plan_path.read_text())
                design_meta = plan["designs"][design_idx]
                d_dir = Path(design_meta["dir"])
                mut_path = d_dir / "mutations.tsv"
                rev_accum.append(read_mutations_tsv_full(mut_path))
            except Exception:
                rev_accum.append([])
        else:
            rev_accum.append([])
        label = parent_pathway_label(label)
    # Reverse so cycle-0 mutations come first
    for lst in reversed(rev_accum):
        out.extend(lst)
    return out


def accumulated_mutated_positions(chain: List[Dict]) -> set:
    """Union of mutated positions across the whole chain (cumulative
    across cycles).  Used to mark positions as 'burnt' so we don't
    re-mutate them in the next cycle."""
    out = set()
    for entry in chain:
        out.update(entry["mutated_positions"])
    return out


# ───────────────────────────────────────────────────────────────────────
# Novelty + maximin selection
# ───────────────────────────────────────────────────────────────────────
def compute_pose_distance(candidate_pdb: Path, reference_pdb: Path,
                          pred_rec_chain: str, pred_eff_chain: str) -> float:
    """receptor_aligned_effector_rmsd between two predicted complexes
    that share both protein sequences and chain naming.  Aligns the
    receptors via Kabsch then measures the effector displacement —
    same machinery as binding_rmsds, just both arguments are
    predictions instead of one being a ground truth.

    Container-only — calls into boltz2_negative_steering which needs
    numpy and Bio.Align.
    """
    bns = _lazy_bns()
    result = bns.compute_binding_rmsds(
        str(candidate_pdb), str(reference_pdb),
        rec_chain=pred_rec_chain, eff_chain=pred_eff_chain,
        des_rec_chain=pred_rec_chain, des_eff_chain=pred_eff_chain,
    )
    v = result.get("receptor_aligned_effector_rmsd")
    return float(v) if v is not None else float("nan")


def maximin_select(candidates: List[Dict], k: int) -> List[Dict]:
    """Select up to k candidates from `candidates` by max-min distance
    to the upstream pose set.  Each candidate dict must have a key
    'min_dist_to_upstream' (float).  Picks the k with the largest
    such value, ties broken by lower ra_eff vs ground truth.

    For the tie-break, prefers `_active_ra_eff_for_selection` if
    present (set by finalize on `pose_holds` rows so the selector
    sees the reverted ra_eff), falling back to
    `receptor_aligned_effector_rmsd_vs_truth` (the steered value).
    Steered base columns are never overwritten — the dedicated
    `_active_*` keys carry the post-reversion view used for selection.
    """
    def _ra_eff(c: Dict) -> float:
        v = c.get("_active_ra_eff_for_selection")
        if v is None:
            v = c.get("receptor_aligned_effector_rmsd_vs_truth", float("inf"))
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("inf")
    sorted_c = sorted(
        candidates,
        key=lambda c: (-c["min_dist_to_upstream"], _ra_eff(c)),
    )
    return sorted_c[:k]


# ───────────────────────────────────────────────────────────────────────
# Statistics tracking
# ───────────────────────────────────────────────────────────────────────
def append_cycle_statistics(experiment_root: Path,
                            cycle: int,
                            pathway_label: str,
                            n_generated: int,
                            n_intact: int,
                            n_novel: int,
                            n_selected: int,
                            min_dists: List[float],
                            ra_eff_vs_truth: List[float]):
    """Append one row per (cycle, pathway) to cycle_statistics.csv.
    The file is created on first write.

    Stats include the full list of min_dists and ra_eff_vs_truth as
    semicolon-joined strings so downstream analysis can fit
    distributions / draw bell curves without re-walking the tree.
    """
    stats_path = experiment_root / "cycle_statistics.csv"
    is_new = not stats_path.exists()

    with open(stats_path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow([
                "cycle", "pathway", "n_generated", "n_intact",
                "n_novel", "n_selected",
                "min_dist_to_upstream_values",
                "receptor_aligned_effector_rmsd_vs_truth_values",
            ])
        w.writerow([
            cycle, pathway_label, n_generated, n_intact,
            n_novel, n_selected,
            ";".join(f"{v:.3f}" for v in min_dists),
            ";".join(f"{v:.3f}" for v in ra_eff_vs_truth),
        ])


# ───────────────────────────────────────────────────────────────────────
# Pathway resubmission
# ───────────────────────────────────────────────────────────────────────
def resubmit_pathway_chains(
    *,
    submit_script: Path,
    selected: List[Dict],
    parent_label: Optional[str],
    cycle: int,
    experiment_root: Path,
    max_cycles: int,
    max_passing: int,
    novelty_cutoff: float,
    n_designs: Optional[int],
    announce: str,
    failure_prefix: str,
) -> None:
    """sbatch one cycle-(cycle+1) follow-up chain per selected design.

    Shared scaffolding for cmd_iterate_collect, cmd_iterate_collect_finalize,
    cmd_kickoff, and cmd_kickoff_finalize.  Prints ``announce``, then for
    each candidate builds the standard sbatch command and runs it.  A
    subprocess failure logs "{failure_prefix} {label}: {e}" to stderr but
    does NOT abort the rest of the batch — losing one resubmission is
    preferable to losing the whole pathway.

    The caller is responsible for verifying ``submit_script.exists()`` and
    handling the missing-script case (the warning wording differs between
    iterate and kickoff call sites, which is why that check stays at the
    caller).
    """
    print(announce)
    for sel in selected:
        new_label = make_pathway_label(parent_label, cycle, sel["design_idx"])
        cmd = [
            str(submit_script),
            "--experiment-root", str(experiment_root),
            "--parent-pathway", new_label,
            "--cycle", str(cycle + 1),
            "--max-cycles", str(max_cycles),
            "--max-passing", str(max_passing),
            "--novelty-cutoff", str(novelty_cutoff),
        ]
        if n_designs is not None:
            cmd.extend(["--n-designs", str(n_designs)])
        print(f"    -> {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"    {failure_prefix} {new_label}: {e}", file=sys.stderr)


# ───────────────────────────────────────────────────────────────────────
# Subcommand: iterate-plan
# ───────────────────────────────────────────────────────────────────────
def cmd_iterate_plan(args: argparse.Namespace) -> int:
    """Build the plan.json for one cycle-N+1 pathway.

    Required args:
        --experiment-root      : top-level experiment dir
        --parent-pathway       : the parent's pathway label
                                 (e.g. "c0d05" for a cycle-1 plan,
                                 "c0d05.c1d03" for cycle-2)
        --cycle                : the NEW cycle index (parent's cycle + 1)
        --workdir              : the new pathway's workdir (will be created)

    Mode / max-mutations / pool / contact-cutoff / etc. are NOT
    arguments — they're inherited from the parent's plan.json so
    every cycle in a pathway uses the same parameters.

    Container-only — uses heavy-atom contact detection, surface
    filter, and the make_steered_sequence machinery from
    boltz2_negative_steering.
    """
    bns = _lazy_bns()
    print(f"[iterate-plan] cycle {args.cycle} pathway {args.parent_pathway}", flush=True)

    experiment_root = args.experiment_root.resolve()
    parent_label = args.parent_pathway
    new_cycle = args.cycle
    new_workdir = args.workdir.resolve()
    new_workdir.mkdir(parents=True, exist_ok=True)

    # Walk up the chain to read every upstream prediction + mutations.
    chain = read_upstream_chain(experiment_root, parent_label)
    if not chain:
        print(f"ERROR: empty upstream chain for {parent_label}", file=sys.stderr)
        return 1
    parent_entry = chain[-1]
    parent_pdb = Path(parent_entry["prediction_pdb"])
    parent_seq = parent_entry["mutated_seq"]
    burnt_positions = accumulated_mutated_positions(chain)

    print(f"  upstream chain length: {len(chain)} predictions")
    print(f"  parent prediction:     {parent_pdb}")
    print(f"  burnt positions:       {sorted(burnt_positions)}")

    # Read cycle 0's plan to inherit the parameters and the protected
    # set + ground truth pointer.  Cycle 0 lives at experiment_root/cycle_0/.
    cycle0_plan = json.loads((experiment_root / "cycle_0" / "plan.json").read_text())
    mode             = cycle0_plan["mode"]
    max_mutations    = int(cycle0_plan["max_mutations"])
    pool_size        = int(cycle0_plan["candidate_pool_size"])
    contact_cutoff   = float(cycle0_plan["contact_cutoff"])
    n_designs        = int(args.n_designs) if args.n_designs is not None else int(cycle0_plan["n_designs"])
    boltz_container  = cycle0_plan["boltz_container"]
    recycling_steps  = int(cycle0_plan["recycling_steps"])
    diffusion_samples = int(cycle0_plan["diffusion_samples"])
    seed_base        = int(cycle0_plan["seed"])
    no_kernels       = bool(cycle0_plan["no_kernels"])
    num_seeds        = int(cycle0_plan.get("num_seeds", 1))
    pred_rec_chain   = cycle0_plan["pred_receptor_chain"]
    pred_eff_chain   = cycle0_plan["pred_effector_chain"]
    truth_rec_chain  = cycle0_plan["truth_receptor_chain"]
    truth_eff_chain  = cycle0_plan["truth_effector_chain"]
    ground_truth     = Path(cycle0_plan["ground_truth"])
    # Wild-type receptor sequence is persisted in cycle_0/plan.json
    # by boltz2_negative_steering.py; every cycle computes
    # total_mutations as the Hamming distance from this sequence to
    # the new design's mutated receptor.  Fall back to the cycle-0
    # parent's sequence (via chain walking) if the field is missing,
    # for back-compat with older runs.
    wt_rec_seq = cycle0_plan.get("wild_type_receptor_seq")
    if not wt_rec_seq:
        # Older runs: reconstruct WT from ground truth.  Requires
        # numpy/gemmi, which is fine here — cmd_iterate_plan runs in
        # the container.
        wt_rec_seq = bns.get_chain_sequence(ground_truth, truth_rec_chain)
        print(f"  NOTE: cycle_0/plan.json has no wild_type_receptor_seq; "
              f"reconstructed from ground truth ({len(wt_rec_seq)} aa)")
    eff_template_cif = (
        Path(cycle0_plan["effector_template_cif"])
        if cycle0_plan.get("effector_template_cif") else None
    )

    # Run heavy-atom contact detection on the parent prediction to
    # find the NEW wrong interface.  expected_rec_seq is the parent's
    # mutated sequence (because the receptor in parent.pdb has the
    # accumulated mutations from cycles 0..N).
    print(f"  Identifying NEW wrong interface (heavy-atom ≤ {contact_cutoff} Å)...")
    try:
        wrong_contacts = bns.find_contact_residues_heavy(
            parent_pdb,
            pred_rec_chain, pred_eff_chain,
            contact_cutoff,
            expected_rec_seq=parent_seq,
        )
    except Exception as e:
        print(f"ERROR: contact detection on parent failed: {e}", file=sys.stderr)
        return 1
    predicted_wrong_idx = sorted(i for i, _ in wrong_contacts)
    print(f"  {len(predicted_wrong_idx)} residues at the new wrong interface")

    # Surface filter, computed on the parent prediction's geometry
    pred_residues = bns.read_residue_heavy_atoms(parent_pdb)
    pred_rec_residues = [r for r in pred_residues if r.chain == pred_rec_chain]
    pred_rec_by_idx = {r.seq_index: r for r in pred_rec_residues}
    surface_flags: Dict[int, bool] = {}
    for i in predicted_wrong_idx:
        r = pred_rec_by_idx.get(i)
        if r is None:
            surface_flags[i] = False
            continue
        surface_flags[i] = bns.is_surface_exposed(r, pred_rec_residues)

    # Build the candidate pool: top-N closest contacts that are
    # surface-exposed, NOT protected, AND not already burnt by an
    # earlier cycle's mutation.
    #
    # P0.1: read the resolved protected set from cycle_0/plan.json
    # (true interface ∪ design region by default).  Falls back to
    # true_interface_idx for back-compat with cycle_0/plan.json files
    # written before P0.1 — those plans don't have protected_set_idx.
    true_set = set(cycle0_plan.get("true_interface_idx", []))
    protected_set = set(cycle0_plan.get("protected_set_idx", []))
    if not protected_set:
        protected_set = set(true_set)
    candidates = []
    for seq_idx, dist in wrong_contacts:
        if seq_idx in protected_set:
            continue
        if not surface_flags.get(seq_idx, False):
            continue
        if seq_idx in burnt_positions:
            continue
        candidates.append((dist, seq_idx))
    candidates.sort()
    pool_size_eff = max(pool_size, max_mutations)
    candidate_pool = sorted(seq_idx for _, seq_idx in candidates[:pool_size_eff])

    n_overlap_true = sum(1 for i in predicted_wrong_idx if i in true_set)
    n_overlap_protected = sum(
        1 for i in predicted_wrong_idx if i in protected_set
    )
    n_buried = sum(1 for i in predicted_wrong_idx if not surface_flags.get(i, False))
    n_burnt_in_wrong = sum(1 for i in predicted_wrong_idx if i in burnt_positions)
    print(f"  Filtering: {len(predicted_wrong_idx)} new wrong-interface residues")
    print(f"    -{n_overlap_true:>3d} overlap the true interface")
    if protected_set != true_set:
        print(f"    -{n_overlap_protected:>3d} overlap the full protected set "
              f"(true ∪ design region)")
    print(f"    -{n_buried:>3d} fail the surface test")
    print(f"    -{n_burnt_in_wrong:>3d} already mutated in earlier cycles")
    print(f"    ={len(candidates):>3d} candidates remain")
    print(f"  Candidate pool: top {len(candidate_pool)}")

    # Build the upstream-pose list NOW so it can be persisted into
    # plan.json — iterate-collect needs it for the novelty check, and
    # we'd rather pin it at plan time than re-walk the tree later.
    upstream_pdbs = [c["prediction_pdb"] for c in chain]
    # Also include the wild-type initial prediction (cycle 0's wild
    # type baseline) — that's the very first reference pose.
    wild_type_pdb = experiment_root / "cycle_0" / "initial_prediction.pdb"
    upstream_pdbs.insert(0, str(wild_type_pdb))

    plan = {
        "kind": "iterate",
        "cycle": new_cycle,
        "pathway": parent_label + f".c{new_cycle}d??",   # placeholder; not used
        "parent_pathway": parent_label,
        "parent_prediction": str(parent_pdb),
        "experiment_root": str(experiment_root),
        "ground_truth": str(ground_truth),
        "truth_receptor_chain": truth_rec_chain,
        "truth_effector_chain": truth_eff_chain,
        "pred_receptor_chain": pred_rec_chain,
        "pred_effector_chain": pred_eff_chain,
        "contact_cutoff": contact_cutoff,
        "max_mutations": max_mutations,
        "candidate_pool_size": pool_size,
        "mode": mode,
        "effector_template_cif": str(eff_template_cif) if eff_template_cif else None,
        "n_designs": n_designs,
        "num_seeds": num_seeds,
        "boltz_container": boltz_container,
        "recycling_steps": recycling_steps,
        "diffusion_samples": diffusion_samples,
        "seed": seed_base + 1000 * new_cycle,   # offset per cycle for diversity
        "no_kernels": no_kernels,
        "burnt_positions": sorted(burnt_positions),
        "candidate_pool": candidate_pool,
        "upstream_prediction_pdbs": upstream_pdbs,
        "true_interface_idx": cycle0_plan.get("true_interface_idx", []),
        # P0.1: forward the protected set so any downstream consumer
        # that reads cycle-N plan.json (e.g. analysis scripts) sees
        # the same exclusion set that gated candidate selection.
        "protected_set_idx": sorted(protected_set),
        "protected_set_source": cycle0_plan.get(
            "protected_set_source", "true_interface"),
        "initial_wrong_interface_idx": predicted_wrong_idx,
        "designs": [],
    }

    if not candidate_pool:
        print("  No mutatable candidates remain — pathway exhausted at this depth.")
        plan["skip_steering"] = True
        plan["exhausted"] = True
        (new_workdir / "plan.json").write_text(json.dumps(plan, indent=2))
        return 0

    # Generate per-design YAMLs.  We start from the parent's mutated
    # sequence (NOT the wild-type), and the new mutations are applied
    # on top of it.
    import random
    rng = random.Random(plan["seed"])

    # Effector sequence: honour the cycle-0 override if present, so
    # multi-cycle runs that started from a sequence-walk waypoint
    # don't silently revert the effector to the truth sequence in
    # cycle 1+.  The receptor side propagates by itself because each
    # cycle reads parent_seq from the parent's receptor.fasta, which
    # already carries the override forward.
    eff_override_path = cycle0_plan.get("effector_fasta_override")
    if eff_override_path:
        try:
            lines = [
                l.strip()
                for l in Path(eff_override_path).read_text().splitlines()
                if l.strip()
            ]
            eff_seq = "".join(l for l in lines if not l.startswith(">"))
            print(f"  effector seq: OVERRIDE from {eff_override_path} "
                  f"({len(eff_seq)} residues)")
        except Exception as e:
            print(f"ERROR: failed to load effector override "
                  f"{eff_override_path}: {e}", file=sys.stderr)
            return 1
    else:
        eff_seq = bns.get_chain_sequence(ground_truth, truth_eff_chain)

    steered_root = new_workdir / "steered"
    steered_root.mkdir(exist_ok=True)

    # ── Generate unique sequences with dedup ────────────────────────
    MAX_RETRIES = n_designs * 10
    unique_sequences: List[Tuple[str, List[Tuple[int, str, str]]]] = []
    seen_seqs: set = set()
    attempts = 0

    while len(unique_sequences) < n_designs and attempts < MAX_RETRIES:
        attempts += 1
        seq, muts = bns.make_steered_sequence(
            parent_seq, candidate_pool, max_mutations, mode, rng,
        )
        if seq not in seen_seqs:
            seen_seqs.add(seq)
            unique_sequences.append((seq, muts))

    if len(unique_sequences) < n_designs:
        print(f"  WARNING: only {len(unique_sequences)} unique sequences "
              f"found after {attempts} attempts (requested {n_designs}).")
    else:
        collisions = attempts - len(unique_sequences)
        if collisions > 0:
            print(f"  Generated {len(unique_sequences)} unique sequences "
                  f"({collisions} collision(s) deduplicated)")

    n_unique = len(unique_sequences)
    total_predictions = n_unique * num_seeds
    plan["n_unique_sequences"] = n_unique
    plan["total_predictions"] = total_predictions
    print(f"  {n_unique} unique sequence(s) × {num_seeds} seed(s) = "
          f"{total_predictions} total prediction(s)")

    # ── Expand each unique sequence into num_seeds design entries ───
    design_idx = 0
    for seq_group, (mutated_seq, mutations) in enumerate(unique_sequences):
        for seed_idx in range(num_seeds):
            if num_seeds == 1:
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
            bns.write_boltz_yaml(
                d / "input",
                mutated_seq, eff_seq,
                pred_rec_chain, pred_eff_chain,
                effector_template_cif=eff_template_cif,
            )

            boltz_seed = plan["seed"] + design_idx + 1

            plan["designs"].append({
                "index": design_idx,
                "name": name,
                "dir": str(d.resolve()),
                "yaml": str((d / "input" / "input.yaml").resolve()),
                "total_mutations": sum(
                    1 for a, b in zip(wt_rec_seq, mutated_seq) if a != b
                ),
                "mutated_positions_this_cycle": sorted(
                    p - 1 for p, _, _ in mutations
                ),
                "sequence_group": seq_group,
                "seed_index": seed_idx,
                "boltz_seed": boltz_seed,
            })
            design_idx += 1

    (new_workdir / "plan.json").write_text(json.dumps(plan, indent=2))
    print(f"\nWrote plan.json with {len(plan['designs'])} designs.")
    return 0


# ───────────────────────────────────────────────────────────────────────
# Subcommand: compute-distances (Phase 1, in container)
# ───────────────────────────────────────────────────────────────────────
def cmd_compute_distances(args: argparse.Namespace) -> int:
    """Phase 1 of cycle-N+1 collection.

    Reads every candidate's result.json, computes the receptor-aligned
    effector RMSD vs every upstream prediction in the pathway, and
    writes the result to <workdir>/distances.json.

    This phase NEEDS the container because compute_pose_distance calls
    into boltz2_negative_steering's RMSD machinery (numpy + Bio.Align +
    gemmi), none of which are available on the login node.

    Phase 2 (cmd_iterate_collect) reads distances.json and runs the
    filter/maximin/sbatch logic in pure stdlib outside the container.
    """
    workdir = args.workdir.resolve()
    plan_path = workdir / "plan.json"
    if not plan_path.exists():
        msg = f"ERROR: {plan_path} does not exist\n"
        sys.stderr.write(msg)
        # Phase 2 will see no distances.json and bail safely.
        return 1

    plan = json.loads(plan_path.read_text())
    cycle = int(plan["cycle"])
    parent_label = plan["parent_pathway"]
    pred_rec_chain = plan["pred_receptor_chain"]
    pred_eff_chain = plan["pred_effector_chain"]

    if plan.get("exhausted") or plan.get("skip_steering"):
        print("[compute-distances] pathway exhausted, writing empty distances.json")
        (workdir / "distances.json").write_text(json.dumps({
            "cycle": cycle,
            "parent_pathway": parent_label,
            "exhausted": True,
            "n_generated": 0,
            "candidates": [],
        }, indent=2))
        return 0

    upstream_pdbs = [Path(p) for p in plan["upstream_prediction_pdbs"]]

    candidates: List[Dict] = []
    n_generated = len(plan["designs"])
    for design in plan["designs"]:
        d = Path(design["dir"])
        rj = d / "result.json"
        if not rj.exists():
            print(f"  WARN: missing result.json for {design['name']}", file=sys.stderr)
            continue
        r = json.loads(rj.read_text())
        if r.get("status") != "ok":
            print(f"  {design['name']}: status={r.get('status')}, skipping")
            continue

        ra_eff_truth = r.get("receptor_aligned_effector_rmsd")
        ind_rec      = r.get("independent_receptor_rmsd")
        ind_eff      = r.get("independent_effector_rmsd")
        if ra_eff_truth is None or ind_rec is None or ind_eff is None:
            continue

        candidate_pdb = d / "prediction.pdb"
        # Compute distance to EVERY upstream prediction.
        dists = []
        for up in upstream_pdbs:
            try:
                dist = compute_pose_distance(
                    candidate_pdb, up, pred_rec_chain, pred_eff_chain,
                )
            except Exception as e:
                print(f"  WARN: pose distance failed for {design['name']} vs {up}: {e}",
                      file=sys.stderr)
                dist = None
            # Use None instead of NaN so the JSON is portable and the
            # phase-2 code doesn't need to handle NaN sentinels.
            dists.append(None if (dist is None or _is_nan(dist)) else float(dist))

        finite_dists = [x for x in dists if x is not None]
        min_dist = min(finite_dists) if finite_dists else None

        candidates.append({
            "design": r["design"],
            "design_idx": design["index"],
            # P0 audit (multi-seed): carry sequence_group + seed_index
            # from the plan into the candidate dict so per-sequence
            # aggregation downstream can group on them.
            "sequence_group": design.get("sequence_group", ""),
            "seed_index": design.get("seed_index", ""),
            "pdb": str(candidate_pdb),
            # Top-level fields used by phase 2's filters.  Kept here so
            # phase 2 doesn't need to dig into the nested result dict.
            "receptor_aligned_effector_rmsd_vs_truth": float(ra_eff_truth),
            "independent_receptor_rmsd": float(ind_rec),
            "independent_effector_rmsd": float(ind_eff),
            "dists_to_upstream": dists,
            "min_dist_to_upstream": min_dist,
            # Full result.json content carried through verbatim so the
            # aggregator can surface every per-design metric (Jaccards,
            # mutation count, status, n_design_interface_residues, etc.)
            # in all_results_multicycle.csv without losing information
            # at the phase-1/phase-2 boundary.
            "result": r,
        })

    distances_doc = {
        "cycle": cycle,
        "parent_pathway": parent_label,
        "n_generated": n_generated,
        "candidates": candidates,
        "upstream_pdbs": [str(p) for p in upstream_pdbs],
    }
    (workdir / "distances.json").write_text(json.dumps(distances_doc, indent=2))
    print(f"[compute-distances] wrote {len(candidates)} candidate(s) to distances.json")
    return 0


def _is_nan(x):
    """Tiny helper that handles either numpy NaN or Python NaN
    without forcing a numpy import on the caller."""
    try:
        return math.isnan(x)
    except TypeError:
        return False


# ───────────────────────────────────────────────────────────────────────
# Reverted-confidence forwarding
# ───────────────────────────────────────────────────────────────────────
# `_REVERTED_CONFIDENCE_FIELDS` (imported at module top from reversion)
# is the canonical en-bloc forward set.  Structural RMSDs and
# contact-flag fields are NOT in it — they are overlaid inline in
# the finalize / aggregate paths because they also rename to canonical
# columns on pose_holds rows.  See reversion.py for the rationale and
# the lockstep requirement with per_seed_records.


def _copy_reverted_confidence(dst: Dict, rev: Dict) -> None:
    """Copy every reverted confidence metric from `rev` (a single
    entry from reversion_results.json) into the candidate dict `dst`
    that will be written into passing.json.  Missing keys are
    skipped so the column stays blank on rows that don't have
    that metric (e.g. chai predictions that don't emit actifptm)."""
    for k in _REVERTED_CONFIDENCE_FIELDS:
        v = rev.get(k)
        if v is not None:
            dst[k] = v


# ───────────────────────────────────────────────────────────────────────
# Subcommand: iterate-collect (Phase 2, outside container)
# ───────────────────────────────────────────────────────────────────────
def cmd_iterate_collect(args: argparse.Namespace) -> int:
    """Phase 2: read distances.json, apply intact + novelty filters,
    maximin-select up to --max-passing survivors, write passing.json,
    append cycle stats, and self-resubmit cycle N+2 jobs via sbatch
    if there are survivors and we haven't hit max-cycles.

    Pure stdlib — runs OUTSIDE the singularity container so sbatch
    is on PATH.  Does not import numpy or boltz2_negative_steering.
    """
    workdir = args.workdir.resolve()
    plan_path = workdir / "plan.json"
    distances_path = workdir / "distances.json"

    if not plan_path.exists():
        msg = f"ERROR: {plan_path} does not exist\n"
        sys.stderr.write(msg)
        (workdir / "passing.json").write_text(json.dumps({"error": msg}))
        return 1
    if not distances_path.exists():
        msg = (f"ERROR: {distances_path} does not exist — phase 1 "
               f"(compute-distances) probably failed.  Check the in-container "
               f"step in the same SLURM job's log.\n")
        sys.stderr.write(msg)
        (workdir / "passing.json").write_text(json.dumps({"error": msg}))
        return 1

    plan = json.loads(plan_path.read_text())
    distances = json.loads(distances_path.read_text())

    experiment_root = Path(plan["experiment_root"])
    cycle = int(plan["cycle"])
    parent_label = plan["parent_pathway"]
    novelty_cutoff = float(args.novelty_cutoff)
    max_passing = int(args.max_passing)
    max_cycles = int(args.max_cycles)

    if plan.get("exhausted") or plan.get("skip_steering") or distances.get("exhausted"):
        print("[iterate-collect] pathway exhausted, nothing to do")
        (workdir / "passing.json").write_text(json.dumps({
            "cycle": cycle,
            "parent_pathway": parent_label,
            "exhausted": True,
            "n_generated": 0,
            "n_intact": 0,
            "n_novel": 0,
            "n_selected": 0,
            "selected": [],
        }, indent=2))
        append_cycle_statistics(
            experiment_root, cycle, parent_label,
            n_generated=0, n_intact=0, n_novel=0, n_selected=0,
            min_dists=[], ra_eff_vs_truth=[],
        )
        return 0

    n_generated = int(distances.get("n_generated", 0))

    # Apply intact + novelty filters using the precomputed distances
    candidates = []
    for c in distances.get("candidates", []):
        intact = (
            c["independent_receptor_rmsd"] <= _INTACT_THRESHOLD
            and c["independent_effector_rmsd"] <= _INTACT_THRESHOLD
        )
        c2 = dict(c)
        c2["receptor_intact"] = intact
        candidates.append(c2)

    intact_candidates = [c for c in candidates if c["receptor_intact"]]
    novel_candidates = [
        c for c in intact_candidates
        if c["min_dist_to_upstream"] is not None
        and c["min_dist_to_upstream"] >= novelty_cutoff
    ]
    selected = maximin_select(novel_candidates, max_passing)

    n_intact = len(intact_candidates)
    n_novel = len(novel_candidates)
    n_selected = len(selected)
    print(f"[iterate-collect] cycle {cycle} pathway {parent_label}: "
          f"{n_generated} generated -> {n_intact} intact -> "
          f"{n_novel} novel -> {n_selected} selected")

    # Persist passing.json (same shape as before so aggregate keeps working)
    passing = {
        "cycle": cycle,
        "parent_pathway": parent_label,
        "novelty_cutoff": novelty_cutoff,
        "max_passing": max_passing,
        "n_generated": n_generated,
        "n_intact": n_intact,
        "n_novel": n_novel,
        "n_selected": n_selected,
        "all_candidates": candidates,
        "selected": selected,
    }
    (workdir / "passing.json").write_text(json.dumps(passing, indent=2, default=str))

    append_cycle_statistics(
        experiment_root, cycle, parent_label,
        n_generated=n_generated, n_intact=n_intact,
        n_novel=n_novel, n_selected=n_selected,
        min_dists=[c["min_dist_to_upstream"] for c in selected],
        ra_eff_vs_truth=[c["receptor_aligned_effector_rmsd_vs_truth"] for c in selected],
    )

    # Self-resubmit if cycle < max_cycles and there are survivors.
    # We're OUTSIDE the container here, so sbatch is on PATH and we
    # can call the submit script directly.
    if n_selected == 0:
        print(f"  no survivors — pathway terminates at cycle {cycle}")
        return 0
    if cycle >= max_cycles:
        print(f"  reached max-cycles ({max_cycles}) — not resubmitting")
        return 0

    submit_script = SCRIPT_DIR / "submit_boltz2_iterate_steering.sh"
    if not submit_script.exists():
        print(f"  WARN: cannot find submit script at {submit_script}; "
              f"survivors will not be iterated further", file=sys.stderr)
        return 0

    resubmit_pathway_chains(
        submit_script=submit_script,
        selected=selected,
        parent_label=parent_label,
        cycle=cycle,
        experiment_root=experiment_root,
        max_cycles=max_cycles,
        max_passing=max_passing,
        novelty_cutoff=novelty_cutoff,
        n_designs=args.n_designs,
        announce=f"  resubmitting {n_selected} cycle-{cycle + 1} chain(s)",
        failure_prefix="FAILED to resubmit",
    )
    return 0


# ───────────────────────────────────────────────────────────────────────
# Subcommand: iterate-collect-prefilter  (host, phase 2a)
# ───────────────────────────────────────────────────────────────────────
#
# The intact filter + novelty/maximin/resubmit logic of the original
# cmd_iterate_collect is split in two so a reversion-validation chain
# can run in between:
#
#   compute-distances  →  iterate-collect-prefilter   (emits prefilter.json)
#                    →  build-contaminated          (emits contaminated.json)
#                    →  plan-reversions             (emits reversion_plan.json)
#                    →  predict-reversion-one × N
#                    →  harvest-reversions          (emits reversion_results.json)
#                    →  iterate-collect-finalize    (emits passing.json, resubmits)
#
# Prefilter does only the intact filter; it does NOT run novelty or
# maximin and does NOT resubmit.  It writes prefilter.json with the
# list of intact candidates, which downstream phases read in turn.
# It also writes an exhausted-pathway short-circuit into
# prefilter.json when the pathway is done, so downstream phases can
# skip their work cleanly.
def cmd_iterate_collect_prefilter(args: argparse.Namespace) -> int:
    workdir = args.workdir.resolve()
    plan_path = workdir / "plan.json"
    distances_path = workdir / "distances.json"

    if not plan_path.exists():
        msg = f"ERROR: {plan_path} does not exist\n"
        sys.stderr.write(msg)
        (workdir / "prefilter.json").write_text(json.dumps({"error": msg}))
        return 1
    if not distances_path.exists():
        msg = (f"ERROR: {distances_path} does not exist — phase 1 "
               f"(compute-distances) probably failed.\n")
        sys.stderr.write(msg)
        (workdir / "prefilter.json").write_text(json.dumps({"error": msg}))
        return 1

    plan = json.loads(plan_path.read_text())
    distances = json.loads(distances_path.read_text())

    cycle = int(plan["cycle"])
    parent_label = plan["parent_pathway"]

    if plan.get("exhausted") or plan.get("skip_steering") or distances.get("exhausted"):
        print("[iterate-collect-prefilter] pathway exhausted, short-circuiting")
        (workdir / "prefilter.json").write_text(json.dumps({
            "cycle": cycle,
            "parent_pathway": parent_label,
            "exhausted": True,
            "n_generated": 0,
            "n_intact": 0,
            "intact_candidates": [],
        }, indent=2))
        return 0

    n_generated = int(distances.get("n_generated", 0))

    intact_candidates: List[Dict] = []
    for c in distances.get("candidates", []):
        intact = (
            c["independent_receptor_rmsd"] <= _INTACT_THRESHOLD
            and c["independent_effector_rmsd"] <= _INTACT_THRESHOLD
        )
        c2 = dict(c)
        c2["receptor_intact"] = intact
        if intact:
            intact_candidates.append(c2)

    print(f"[iterate-collect-prefilter] cycle {cycle} pathway {parent_label}: "
          f"{n_generated} generated -> {len(intact_candidates)} intact")

    (workdir / "prefilter.json").write_text(json.dumps({
        "cycle": cycle,
        "parent_pathway": parent_label,
        "n_generated": n_generated,
        "n_intact": len(intact_candidates),
        "intact_candidates": intact_candidates,
    }, indent=2, default=str))
    return 0


# ───────────────────────────────────────────────────────────────────────
# Subcommand: build-contaminated  (container, phase 2b)
# ───────────────────────────────────────────────────────────────────────
#
# Container-phase CPU work: for every intact candidate in prefilter.json,
# shell out to compute_metrics.py to run the contact analysis.  A design
# is flagged contaminated if any of its steering mutations contacts the
# effector in the steered prediction; those contacting positions are
# what the reversion pass will revert.  Write contaminated.json listing
# every flagged design, in the schema plan-reversions consumes.
def cmd_build_contaminated(args: argparse.Namespace) -> int:
    workdir = args.workdir.resolve()
    plan_path = workdir / "plan.json"
    prefilter_path = workdir / "prefilter.json"

    if not plan_path.exists():
        print(f"ERROR: {plan_path} not found", file=sys.stderr)
        return 2
    if not prefilter_path.exists():
        print(f"ERROR: {prefilter_path} not found — run "
              f"iterate-collect-prefilter first", file=sys.stderr)
        return 2

    plan = json.loads(plan_path.read_text())
    prefilter = json.loads(prefilter_path.read_text())

    # Short-circuit: exhausted pathway → nothing to do
    if prefilter.get("exhausted"):
        print("[build-contaminated] pathway exhausted, nothing to do")
        (workdir / "contaminated.json").write_text(json.dumps({
            "cycle": prefilter.get("cycle"),
            "parent_pathway": prefilter.get("parent_pathway"),
            "n_checked": 0,
            "n_contaminated": 0,
            "contaminated": [],
        }, indent=2))
        return 0

    # Resolve experiment_root + cycle_0 plan in a way that works for
    # both cycle 0 (workdir = <root>/cycle_0/, plan IS cycle0_plan)
    # and cycle 1+ (workdir = <root>/cycle_N/pathway_<label>/, cycle_0
    # plan must be loaded from <root>/cycle_0/plan.json).
    #
    # The cycle-N+1 plan.json (written by cmd_iterate_plan) carries an
    # "experiment_root" field explicitly.  The cycle-0 plan.json (written
    # by boltz2_negative_steering.cmd_plan) does NOT — so we detect the
    # cycle-0 case and derive experiment_root from the workdir's parent.
    plan_cycle = int(plan.get("cycle", 0))
    if "experiment_root" in plan:
        experiment_root = Path(plan["experiment_root"])
    else:
        # Cycle 0: workdir = <root>/cycle_0/ (nested layout) or <root>/
        # (flat layout).  We assume nested here because that's what the
        # kickoff chain uses; the flat-layout single-cycle submit script
        # never invokes build-contaminated.
        experiment_root = workdir.parent

    if plan_cycle == 0:
        # The current plan IS cycle 0 — no need to re-read from disk.
        cycle0_plan = plan
        cycle0_plan_path = plan_path
    else:
        cycle0_plan_path = experiment_root / "cycle_0" / "plan.json"
        if not cycle0_plan_path.exists():
            print(f"ERROR: {cycle0_plan_path} not found — cannot resolve "
                  f"chain info", file=sys.stderr)
            return 2
        cycle0_plan = json.loads(cycle0_plan_path.read_text())

    pred_rec_chain = cycle0_plan.get("pred_receptor_chain", "A")
    pred_eff_chain = cycle0_plan.get("pred_effector_chain", "B")

    # Resolve compute_metrics.py (sibling of this script unless overridden)
    compute_metrics_script = args.compute_metrics_script
    if compute_metrics_script is None:
        compute_metrics_script = SCRIPT_DIR / "compute_metrics.py"
    compute_metrics_script = Path(compute_metrics_script).resolve()
    if not compute_metrics_script.exists():
        print(f"ERROR: compute_metrics.py not found at "
              f"{compute_metrics_script}", file=sys.stderr)
        return 2

    scratch_dir = workdir / "contamination_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Per-seed evaluations: populated below, one entry per
    # passing_candidate (whether or not it's contaminated).  After the
    # loop we group these by design and apply the CL-3 majority rule
    # (see CL-3 in notes/phase4_architecture_spec.md): reversion runs
    # only when n_contaminated >= ceil(n_correctly_placed / 2).
    per_seed_evaluations: List[Dict] = []
    contaminated_entries: List[Dict] = []
    n_checked = 0
    intact_candidates = prefilter.get("intact_candidates", [])

    # Only check candidates whose steered prediction passes the ra_eff
    # threshold.  The reversion pass validates whether a GOOD steered
    # pose survives without the contaminating mutations.  Running it on
    # a design whose steered pose is already terrible (ra_eff >> cutoff)
    # doesn't answer that question — it's just asking whether Boltz
    # happens to find the right pose on a fresh random seed.
    rmsd_threshold = float(plan.get("rmsd_threshold", 5.0))
    ra_eff_key = "receptor_aligned_effector_rmsd_vs_truth"
    passing_candidates = []
    n_skipped_ra_eff = 0
    for cand in intact_candidates:
        try:
            ra_eff = float(cand.get(ra_eff_key, float("inf")))
        except (ValueError, TypeError):
            ra_eff = float("inf")
        if ra_eff < rmsd_threshold:
            passing_candidates.append(cand)
        else:
            n_skipped_ra_eff += 1

    print(f"[build-contaminated] {len(intact_candidates)} intact candidates, "
          f"{len(passing_candidates)} pass ra_eff < {rmsd_threshold} Å"
          + (f" ({n_skipped_ra_eff} skipped)" if n_skipped_ra_eff else ""))

    for cand in passing_candidates:
        n_checked += 1
        label_suffix = cand.get("design")
        design_idx = cand.get("design_idx")
        pdb_str = cand.get("pdb")
        if not pdb_str or not label_suffix:
            print(f"  WARN: skipping candidate with missing fields "
                  f"({cand.get('design')})", file=sys.stderr)
            continue
        pdb = Path(pdb_str)
        if not pdb.exists():
            print(f"  WARN: pdb not found: {pdb}", file=sys.stderr)
            continue
        design_workdir = pdb.parent

        # Build label: cycle 0 labels are "design_NN", cycle-N+1 are
        # "parent.cNdMM".  We can reconstruct from plan + design_idx.
        # Note: prefilter["cycle"] is the CHILD cycle (the cycle that
        # just finished predicting), so pass it directly to
        # make_pathway_label — do NOT subtract 1.  This matches the
        # convention used by cmd_iterate_collect, cmd_iterate_plan's
        # resubmit, and cmd_aggregate.
        cycle = int(prefilter.get("cycle", 0))
        parent_label = prefilter.get("parent_pathway")
        is_cycle_zero = (parent_label is None and cycle == 0)
        if is_cycle_zero:
            label = label_suffix  # "design_NN"
        else:
            label = make_pathway_label(parent_label, cycle, design_idx)

        # Walk the cumulative mutation history for this design so we
        # can pass --mutated-positions to compute_metrics.py and (for
        # contaminated outcomes) stash the full [(pos1, wt, mut)] list
        # in contaminated.json for the reversion pass.
        #
        # Cycle-0 designs use a flat "design_NN" label which
        # parse_pathway_label cannot parse, so read_cumulative_mutations
        # would fail.  For cycle 0 the design's own mutations.tsv is
        # the full history — no ancestors to walk — so just read it
        # directly from the design workdir.
        cumulative: List[Tuple[int, str, str]] = []
        if is_cycle_zero:
            mut_path = design_workdir / "mutations.tsv"
            if mut_path.exists():
                try:
                    cumulative = read_mutations_tsv_full(mut_path)
                except Exception as e:
                    print(f"  WARN {label}: read_mutations_tsv_full failed: {e}",
                          file=sys.stderr)
        else:
            try:
                cumulative = read_cumulative_mutations(experiment_root, label)
            except Exception as e:
                print(f"  WARN {label}: read_cumulative_mutations failed: {e}",
                      file=sys.stderr)

        mutated_positions_1 = sorted(set(pos for pos, _, _ in cumulative))
        mutated_cli = ",".join(str(p) for p in mutated_positions_1)

        # Count CAs per chain for --chain-lengths
        from collections import Counter
        ca_counts: Counter = Counter()
        with open(pdb) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    ca_counts[line[21:22].strip() or " "] += 1
        rec_len = ca_counts.get(pred_rec_chain, 0)
        eff_len = ca_counts.get(pred_eff_chain, 0)
        if rec_len == 0 or eff_len == 0:
            ordered = ca_counts.most_common()
            if len(ordered) >= 2:
                rec_len = rec_len or ordered[0][1]
                eff_len = eff_len or ordered[1][1]
        if rec_len == 0 or eff_len == 0:
            print(f"  WARN {label}: could not determine chain lengths",
                  file=sys.stderr)
            continue

        # compute_metrics.py expects sidecar files (confidence/pae/plddt)
        # alongside a nested model PDB, not next to `prediction.pdb`
        # which is a copy.  Reuse _locate_boltz_sidecar_pdb so we point
        # the parser at the directory that actually has the sidecars.
        sidecar = _locate_boltz_sidecar_pdb(pdb)
        pred_dir = sidecar.parent if sidecar is not None else pdb.parent

        per_row_csv = scratch_dir / f"{label.replace('.', '_')}.csv"

        cmd = [
            sys.executable, str(compute_metrics_script),
            "--model", "boltz2",
            "--prediction-dir", str(pred_dir),
            "--chain-lengths", str(rec_len), str(eff_len),
            "--output-csv", str(per_row_csv),
            "--pae-cutoff", str(args.pae_cutoff),
            "--receptor-chain", pred_rec_chain,
            "--effector-chain", pred_eff_chain,
            "--contact-cutoff", str(args.contact_cutoff),
            # v8 Level 1: restrict contact detection to the effector
            # interface-atom set, read from cycle_0/plan.json.  An
            # empty list in plan.json (fallback case) leaves contact
            # detection unfiltered — compute_metrics.py handles that
            # internally.
            "--effector-atom-filter-json", str(cycle0_plan_path),
        ]
        if mutated_cli:
            cmd.extend(["--mutated-positions", mutated_cli])

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=600)
        except subprocess.TimeoutExpired:
            print(f"  WARN {label}: compute_metrics.py timed out",
                  file=sys.stderr)
            continue
        if proc.returncode != 0:
            print(f"  WARN {label}: compute_metrics.py rc={proc.returncode}: "
                  f"{(proc.stderr or '').strip()[:200]}",
                  file=sys.stderr)
            continue
        if not per_row_csv.exists():
            print(f"  WARN {label}: compute_metrics.py produced no CSV",
                  file=sys.stderr)
            continue

        with open(per_row_csv) as f:
            cm_rows = list(csv.DictReader(f))
        if not cm_rows:
            print(f"  WARN {label}: compute_metrics.py CSV was empty",
                  file=sys.stderr)
            continue
        cm = cm_rows[0]

        # Parse contact residues: 1-based positions that form contacts
        # with the effector in the steered prediction.
        contact_residues_str = cm.get("contact_residues", "")
        steered_contact_residues = sorted(set(
            int(p) for p in contact_residues_str.split(",")
            if p.strip()
        ))

        # CONTAMINATION RULE
        # ==================
        # ANY mutated position that forms a contact with the effector
        # in the steered prediction is contamination.  The wet-lab
        # construct carries the original (pre-steering) sequence — it
        # will NOT carry the steered residue at any position, whether
        # that position is inside the protected set or not.  So a
        # mutation at a non-protected position contacting the effector
        # is just as invalidating as one inside the protected set:
        # in both cases the rescue pose depends on a residue state
        # that won't exist when the construct is expressed.
        #
        # positions_to_revert = mutated_positions ∩ contact_residues
        # No protected-set filter.
        mutated_positions_set = set(mutated_positions_1)
        contact_set = set(steered_contact_residues)
        positions_to_revert = sorted(mutated_positions_set & contact_set)

        # CL-3 (Phase 4): collect every per-seed evaluation here, even
        # the uncontaminated ones, so we can compute
        # n_correctly_placed and n_contaminated per design after the
        # loop and apply the majority gating rule.
        per_seed_evaluations.append({
            "design_slot": cand.get("design"),  # cycle-0 slot name,
                                                # e.g. "design_03" — the
                                                # CL-3 grouping key
            "is_contaminated": bool(positions_to_revert),
            "entry": {
                "label": label,
                "design_idx": design_idx,
                "seed_index": cand.get("seed_index", 0),
                "design_workdir": str(design_workdir),
                "cumulative_mutations": [
                    [p, w, m] for p, w, m in cumulative
                ],
                "positions_to_revert": positions_to_revert,
                "steered_ra_eff_vs_truth": cand.get(
                    "receptor_aligned_effector_rmsd_vs_truth"
                ),
                "steered_contact_residues": steered_contact_residues,
            },
        })

    # ── CL-3 majority gating ────────────────────────────────────────
    # Group per-seed evaluations by design_slot, then for each design:
    #   n_correctly_placed = total seeds of this design that reached
    #     this point (already filtered to intact + ra_eff < threshold
    #     upstream).
    #   n_contaminated     = subset that have positions_to_revert.
    # Reversion runs IFF n_correctly_placed > 0 AND n_contaminated >=
    # ceil(n_correctly_placed / 2).  See notes/phase4_architecture_
    # spec.md §CL-3 for the rationale.  Replaces the pre-Phase-4 per-
    # (design, seed) gate that triggered reversion on any single
    # contaminated seed and was driving Tier-B-shaped results into
    # unnecessary reversion attempts.
    from collections import defaultdict
    by_design: Dict[str, List[Dict]] = defaultdict(list)
    for ev in per_seed_evaluations:
        by_design[ev["design_slot"]].append(ev)
    n_designs_evaluated = len(by_design)
    n_designs_triggering = 0
    for design_slot, evals in by_design.items():
        n_correctly_placed = len(evals)
        n_contaminated = sum(1 for e in evals if e["is_contaminated"])
        if n_correctly_placed == 0:
            continue
        majority_threshold = math.ceil(n_correctly_placed / 2)
        if n_contaminated < majority_threshold:
            # Below-majority contamination — leave the steered result
            # alone.  Contaminated seeds count as failures in n_pass
            # at the aggregate level (they're not pass-equivalent), but
            # we don't try to revert them.
            continue
        n_designs_triggering += 1
        # Emit only the genuinely-contaminated entries — these are the
        # seeds whose reverted sequence the reversion stage will
        # predict (after dedup-by-sequence in write_reversion_plan).
        contaminated_entries.extend(
            e["entry"] for e in evals if e["is_contaminated"]
        )

    print(f"[build-contaminated] CL-3 gating: {n_designs_triggering} / "
          f"{n_designs_evaluated} designs trigger reversion "
          f"({len(contaminated_entries)} contaminated entries queued)")

    (workdir / "contaminated.json").write_text(json.dumps({
        "cycle": prefilter.get("cycle"),
        "parent_pathway": prefilter.get("parent_pathway"),
        "n_checked": n_checked,
        "n_contaminated": len(contaminated_entries),
        "contaminated": contaminated_entries,
    }, indent=2, default=str))

    # Tidy up scratch on clean runs; leave it if anything failed for
    # post-mortem.
    if n_checked == len(intact_candidates):
        try:
            shutil.rmtree(scratch_dir)
        except Exception:
            pass

    return 0


# ───────────────────────────────────────────────────────────────────────
# Subcommand: iterate-collect-finalize  (host, phase 2c)
# ───────────────────────────────────────────────────────────────────────
#
# Reads prefilter.json + (optionally) reversion_results.json, routes
# reversion verdicts, re-applies the structural filter to reverted
# designs that claim pose_holds, merges pose_holds designs back into
# the intact list with their sequence overrides, then runs the
# existing novelty/maximin/resubmit logic.
#
# For pose_holds designs: we DESTRUCTIVELY overwrite the steered
# design's receptor.fasta with the reverted sequence, so that cycle
# N+1's read_upstream_chain picks up the reverted sequence without
# any new override logic.  We also write an effective_mutations.tsv
# alongside mutations.tsv for aggregators that want to surface the
# post-reversion mutation set, and a reversion_applied.json marker
# so downstream code can tell which designs were reverted.
def cmd_iterate_collect_finalize(args: argparse.Namespace) -> int:
    workdir = args.workdir.resolve()
    plan_path = workdir / "plan.json"
    prefilter_path = workdir / "prefilter.json"

    if not plan_path.exists():
        print(f"ERROR: {plan_path} does not exist", file=sys.stderr)
        return 1
    if not prefilter_path.exists():
        print(f"ERROR: {prefilter_path} does not exist — run "
              f"iterate-collect-prefilter first", file=sys.stderr)
        return 1

    plan = json.loads(plan_path.read_text())
    prefilter = json.loads(prefilter_path.read_text())

    experiment_root = Path(plan["experiment_root"])
    cycle = int(plan["cycle"])
    parent_label = plan["parent_pathway"]
    novelty_cutoff = float(args.novelty_cutoff)
    max_passing = int(args.max_passing)
    max_cycles = int(args.max_cycles)

    if prefilter.get("exhausted"):
        print("[iterate-collect-finalize] pathway exhausted, nothing to do")
        (workdir / "passing.json").write_text(json.dumps({
            "cycle": cycle,
            "parent_pathway": parent_label,
            "exhausted": True,
            "n_generated": 0,
            "n_intact": 0,
            "n_novel": 0,
            "n_selected": 0,
            "selected": [],
        }, indent=2))
        append_cycle_statistics(
            experiment_root, cycle, parent_label,
            n_generated=0, n_intact=0, n_novel=0, n_selected=0,
            min_dists=[], ra_eff_vs_truth=[],
        )
        return 0

    n_generated = int(prefilter.get("n_generated", 0))
    intact_candidates: List[Dict] = list(prefilter.get("intact_candidates", []))

    # Load reversion results if any contaminated designs were processed.
    # Missing file is fine — it means build-contaminated found nothing
    # to revert (no mutations contacting the effector).
    reversion_results: Dict[str, Dict] = {}
    reversion_results_path = workdir / "reversion_results.json"
    if reversion_results_path.exists():
        reversion_results = json.loads(reversion_results_path.read_text())

    # Bug 1 guard: load contaminated.json so we can detect designs that
    # were flagged contaminated by build-contaminated but never received
    # a verdict (e.g. plan-reversions crashed, or staged zero entries
    # despite non-empty contaminated.json).  Such designs MUST be dropped
    # — silently keeping them would let unvalidated contaminated designs
    # pass through to the wet lab.
    contaminated_labels: set = set()
    contaminated_path = workdir / "contaminated.json"
    if contaminated_path.exists():
        cdoc = json.loads(contaminated_path.read_text())
        contaminated_labels = {
            e.get("label") for e in cdoc.get("contaminated", [])
            if e.get("label")
        }
    missing_verdict_labels: set = (
        contaminated_labels - set(reversion_results.keys())
    )
    if missing_verdict_labels:
        print(f"[iterate-collect-finalize] WARN: "
              f"{len(missing_verdict_labels)} contaminated design(s) have "
              f"no verdict in reversion_results.json — will be dropped: "
              f"{sorted(missing_verdict_labels)}", file=sys.stderr)

    # Apply verdicts.  For each contaminated candidate, either:
    #   - pose_holds: destructively overwrite receptor.fasta with
    #     the reverted sequence, update the candidate's metrics to
    #     the reverted ones, mark it reverted, keep it in the list
    #   - pose_collapses / new_contamination: drop it from the list
    #   - missing verdict (Bug 1 guard): drop it
    #
    # The mapping between candidate and reversion result is by label.
    # We reconstruct the label the same way build-contaminated did.
    pose_holds_count = 0
    pose_collapses_count = 0
    new_contamination_count = 0
    dropped_labels: List[str] = []
    reverted_labels: List[str] = []
    # Designs that got dropped by verdict routing — surfaced into
    # passing.json's all_candidates with a reversion_verdict marker
    # so the aggregator CSV can render them.  They do NOT flow into
    # novelty/maximin (those run on `kept` only).
    dropped_rows: List[Dict] = []

    def _label_for(c: Dict) -> str:
        di = c.get("design_idx")
        if parent_label is None and cycle == 0:
            return c.get("design", f"design_{di:02d}" if di is not None else "?")
        return make_pathway_label(parent_label, cycle, di if di is not None else 0)

    kept: List[Dict] = []
    missing_verdict_count = 0
    for c in intact_candidates:
        label = _label_for(c)
        rev = reversion_results.get(label)
        if rev is None:
            # Bug 1 guard: if this label was flagged contaminated but
            # has no verdict, drop it instead of silently keeping it.
            if label in contaminated_labels:
                missing_verdict_count += 1
                dropped_labels.append(label)
                print(f"  [missing_verdict] {label}: dropped — flagged "
                      f"contaminated but no entry in reversion_results.json")
                c_drop = dict(c)
                c_drop["reversion_verdict"] = "missing_verdict"
                c_drop["reversion_dropped"] = 1
                # Steered metrics already in base columns of c_drop;
                # no overlay required (and no reverted prediction exists).
                dropped_rows.append(c_drop)
                continue
            # Not contaminated → keep as is
            kept.append(c)
            continue

        verdict = rev.get("verdict", "")
        if verdict == "pose_holds":
            pose_holds_count += 1
            reverted_labels.append(label)
            # Destructively rewrite receptor.fasta with the reverted
            # sequence so cycle N+1 reads the reverted one.
            design_workdir = Path(c["pdb"]).parent
            reverted_fasta = Path(rev.get("reverted_receptor_fasta", ""))
            if not reverted_fasta.exists():
                # The harvest phase records reverted_prediction_pdb,
                # not reverted_receptor_fasta; the FASTA lives next to
                # it in the reversion staging dir.  Derive it.
                rev_pdb = rev.get("reverted_prediction_pdb")
                if rev_pdb:
                    reverted_fasta = (Path(rev_pdb).parent.parent.parent
                                      / "receptor.fasta")
                    # Boltz's nested output is
                    # rev_dir/boltz_results_*/predictions/*/*_model_0.pdb
                    # so .parent.parent.parent gets us up to rev_dir.
                    # We explicitly re-resolve via rev_dir if possible.
                # Best source: reversion_plan.json has rev_dir by label
                rp_path = workdir / "reversion_plan.json"
                if rp_path.exists():
                    rp = json.loads(rp_path.read_text())
                    for entry in rp.get("entries", []):
                        if entry.get("label") == label:
                            reverted_fasta = Path(entry["reverted_receptor_fasta"])
                            break
            if reverted_fasta.exists():
                try:
                    reverted_seq = reverted_fasta.read_text().splitlines()
                    reverted_seq = "".join(
                        l.strip() for l in reverted_seq
                        if l.strip() and not l.startswith(">")
                    )
                    (design_workdir / "receptor.fasta").write_text(
                        f">receptor_reverted\n{reverted_seq}\n"
                    )
                    # Write a marker so later code can tell this
                    # design was reverted
                    (design_workdir / "reversion_applied.json").write_text(
                        json.dumps({
                            "label": label,
                            "verdict": verdict,
                            "reverted_sequence": reverted_seq,
                            "reversion_result": rev,
                        }, indent=2, default=str)
                    )
                    print(f"  [pose_holds] {label}: rewrote "
                          f"{design_workdir/'receptor.fasta'}")
                except Exception as e:
                    print(f"  WARN {label}: failed to rewrite "
                          f"receptor.fasta: {e}", file=sys.stderr)
            else:
                print(f"  WARN {label}: reverted FASTA not found at "
                      f"{reverted_fasta}", file=sys.stderr)

            # Build the kept candidate dict.  Critical invariant: the
            # base columns (receptor_aligned_effector_rmsd_vs_truth,
            # independent_receptor_rmsd, independent_effector_rmsd)
            # MUST hold the steered values and are never overwritten
            # by reversion.  Reverted values live in dedicated
            # reverted_* keys.  For novelty/maximin to operate on the
            # post-reversion pose, we set _active_ra_eff_for_selection
            # which maximin_select reads in preference to the steered
            # base column.
            c2 = dict(c)
            ra = rev.get("reverted_ra_eff_vs_truth")
            if ra is not None:
                c2["reverted_ra_eff_vs_truth"] = float(ra)
                c2["_active_ra_eff_for_selection"] = float(ra)
            ind_rec = rev.get("reverted_independent_receptor_rmsd")
            if ind_rec is not None:
                c2["reverted_independent_receptor_rmsd"] = float(ind_rec)
            ind_eff = rev.get("reverted_independent_effector_rmsd")
            if ind_eff is not None:
                c2["reverted_independent_effector_rmsd"] = float(ind_eff)
            _copy_reverted_confidence(c2, rev)
            c2["reversion_verdict"] = verdict
            c2["reverted_sequence_used_as_parent"] = 1
            # Re-apply the structural filter on the REVERTED metrics
            # only.  Steered intact-ness was already checked upstream
            # (these are intact_candidates by definition).
            reverted_intact = (
                (ind_rec is not None and float(ind_rec) <= 5.0)
                and (ind_eff is not None and float(ind_eff) <= 5.0)
            )
            if not reverted_intact:
                print(f"  [pose_holds→drop] {label}: reverted fails "
                      f"intact filter, dropping anyway")
                dropped_labels.append(label)
                c2["reversion_verdict"] = "pose_holds_not_intact"
                c2["reversion_dropped"] = 1
                dropped_rows.append(c2)
                continue
            c2["reversion_dropped"] = 0
            kept.append(c2)
        elif verdict == "pose_collapses":
            pose_collapses_count += 1
            dropped_labels.append(label)
            print(f"  [pose_collapses] {label}: {rev.get('reason','')}")
            c_drop = dict(c)
            c_drop["reversion_verdict"] = verdict
            c_drop["reversion_dropped"] = 1
            # Steered metrics already in base columns of c_drop.
            # Overlay reverted metrics so the CSV shows what the
            # reverted prediction looked like.
            if rev.get("reverted_ra_eff_vs_truth") is not None:
                c_drop["reverted_ra_eff_vs_truth"] = float(rev["reverted_ra_eff_vs_truth"])
            if rev.get("reverted_independent_receptor_rmsd") is not None:
                c_drop["reverted_independent_receptor_rmsd"] = float(rev["reverted_independent_receptor_rmsd"])
            if rev.get("reverted_independent_effector_rmsd") is not None:
                c_drop["reverted_independent_effector_rmsd"] = float(rev["reverted_independent_effector_rmsd"])
            _copy_reverted_confidence(c_drop, rev)
            dropped_rows.append(c_drop)
        elif verdict == "new_contamination":
            new_contamination_count += 1
            dropped_labels.append(label)
            print(f"  [new_contamination] {label}: {rev.get('reason','')}")
            c_drop = dict(c)
            c_drop["reversion_verdict"] = verdict
            c_drop["reversion_dropped"] = 1
            if rev.get("reverted_ra_eff_vs_truth") is not None:
                c_drop["reverted_ra_eff_vs_truth"] = float(rev["reverted_ra_eff_vs_truth"])
            if rev.get("reverted_independent_receptor_rmsd") is not None:
                c_drop["reverted_independent_receptor_rmsd"] = float(rev["reverted_independent_receptor_rmsd"])
            if rev.get("reverted_independent_effector_rmsd") is not None:
                c_drop["reverted_independent_effector_rmsd"] = float(rev["reverted_independent_effector_rmsd"])
            _copy_reverted_confidence(c_drop, rev)
            dropped_rows.append(c_drop)
        else:
            # Unknown verdict → conservative, drop
            dropped_labels.append(label)
            print(f"  WARN {label}: unknown verdict {verdict!r}, dropping")
            c_drop = dict(c)
            c_drop["reversion_verdict"] = verdict or "unknown"
            c_drop["reversion_dropped"] = 1
            dropped_rows.append(c_drop)

    intact_candidates = kept

    novel_candidates = [
        c for c in intact_candidates
        if c.get("min_dist_to_upstream") is not None
        and c["min_dist_to_upstream"] >= novelty_cutoff
    ]
    selected = maximin_select(novel_candidates, max_passing)

    n_intact = len(intact_candidates)
    n_novel = len(novel_candidates)
    n_selected = len(selected)
    print(f"[iterate-collect-finalize] cycle {cycle} pathway {parent_label}: "
          f"{n_generated} generated -> {n_intact} intact (after verdicts) -> "
          f"{n_novel} novel -> {n_selected} selected")
    print(f"  verdicts: pose_holds={pose_holds_count} "
          f"pose_collapses={pose_collapses_count} "
          f"new_contamination={new_contamination_count}")

    passing = {
        "cycle": cycle,
        "parent_pathway": parent_label,
        "novelty_cutoff": novelty_cutoff,
        "max_passing": max_passing,
        "n_generated": n_generated,
        "n_intact": n_intact,
        "n_novel": n_novel,
        "n_selected": n_selected,
        "pose_holds_count": pose_holds_count,
        "pose_collapses_count": pose_collapses_count,
        "new_contamination_count": new_contamination_count,
        "reverted_labels": reverted_labels,
        "dropped_labels": dropped_labels,
        # all_candidates merges kept (survivors that flowed into
        # novelty/maximin) with dropped_rows (verdict drops).  Dropped
        # rows carry reversion_dropped=1 so the aggregator can render
        # them without treating them as next-cycle survivors.
        "all_candidates": intact_candidates + dropped_rows,
        "selected": selected,
    }
    (workdir / "passing.json").write_text(json.dumps(passing, indent=2, default=str))

    append_cycle_statistics(
        experiment_root, cycle, parent_label,
        n_generated=n_generated, n_intact=n_intact,
        n_novel=n_novel, n_selected=n_selected,
        min_dists=[c["min_dist_to_upstream"] for c in selected],
        ra_eff_vs_truth=[c["receptor_aligned_effector_rmsd_vs_truth"] for c in selected],
    )

    if n_selected == 0:
        print(f"  no survivors — pathway terminates at cycle {cycle}")
        return 0
    if cycle >= max_cycles:
        print(f"  reached max-cycles ({max_cycles}) — not resubmitting")
        return 0

    submit_script = SCRIPT_DIR / "submit_boltz2_iterate_steering.sh"
    if not submit_script.exists():
        print(f"  WARN: cannot find submit script at {submit_script}; "
              f"survivors will not be iterated further", file=sys.stderr)
        return 0

    resubmit_pathway_chains(
        submit_script=submit_script,
        selected=selected,
        parent_label=parent_label,
        cycle=cycle,
        experiment_root=experiment_root,
        max_cycles=max_cycles,
        max_passing=max_passing,
        novelty_cutoff=novelty_cutoff,
        n_designs=args.n_designs,
        announce=f"  resubmitting {n_selected} cycle-{cycle + 1} chain(s)",
        failure_prefix="FAILED to resubmit",
    )
    return 0


# ───────────────────────────────────────────────────────────────────────
# End of iterate-collect split
# ───────────────────────────────────────────────────────────────────────


def cmd_kickoff_distances(args: argparse.Namespace) -> int:
    """Phase 1 of kickoff — compute pose distances for every cycle-0
    design vs the wild-type prediction.  Writes cycle_0/kickoff_distances.json.

    Container-only — uses compute_pose_distance which needs numpy and
    Bio.Align via boltz2_negative_steering.

    Phase 2 (cmd_kickoff) reads kickoff_distances.json and runs the
    filter/maximin/sbatch logic in pure stdlib outside the container.
    """
    experiment_root = args.experiment_root.resolve()
    cycle0_dir = experiment_root / "cycle_0"
    if not cycle0_dir.is_dir():
        print(f"ERROR: no cycle_0 under {experiment_root}", file=sys.stderr)
        return 2

    plan_path = cycle0_dir / "plan.json"
    if not plan_path.exists():
        print(f"ERROR: no plan.json at {plan_path}", file=sys.stderr)
        return 2
    plan = json.loads(plan_path.read_text())

    if plan.get("skip_steering"):
        print("[kickoff-distances] cycle 0 skipped steering — writing empty distances")
        (cycle0_dir / "kickoff_distances.json").write_text(json.dumps({
            "skip_steering": True,
            "n_generated": 0,
            "candidates": [],
        }, indent=2))
        return 0

    pred_rec_chain = plan["pred_receptor_chain"]
    pred_eff_chain = plan["pred_effector_chain"]
    wild_type_pdb = cycle0_dir / "initial_prediction.pdb"
    if not wild_type_pdb.exists():
        print(f"ERROR: missing {wild_type_pdb}", file=sys.stderr)
        return 2

    candidates: List[Dict] = []
    for design in plan["designs"]:
        d = Path(design["dir"])
        rj = d / "result.json"
        if not rj.exists():
            continue
        r = json.loads(rj.read_text())
        if r.get("status") != "ok":
            continue
        ra_eff_truth = r.get("receptor_aligned_effector_rmsd")
        ind_rec      = r.get("independent_receptor_rmsd")
        ind_eff      = r.get("independent_effector_rmsd")
        if ra_eff_truth is None or ind_rec is None or ind_eff is None:
            continue

        candidate_pdb = d / "prediction.pdb"
        try:
            dist = compute_pose_distance(
                candidate_pdb, wild_type_pdb,
                pred_rec_chain, pred_eff_chain,
            )
        except Exception as e:
            print(f"  WARN: pose distance failed for {design['name']}: {e}",
                  file=sys.stderr)
            dist = None
        # Use None instead of NaN for JSON portability
        dist_val = None if (dist is None or _is_nan(dist)) else float(dist)

        candidates.append({
            "design": r["design"],
            "design_idx": design["index"],
            # P0 audit (multi-seed): carry sequence_group + seed_index
            # from the plan into the candidate dict so per-sequence
            # aggregation downstream can group on them.
            "sequence_group": design.get("sequence_group", ""),
            "seed_index": design.get("seed_index", ""),
            "pdb": str(candidate_pdb),
            "receptor_aligned_effector_rmsd_vs_truth": float(ra_eff_truth),
            "independent_receptor_rmsd": float(ind_rec),
            "independent_effector_rmsd": float(ind_eff),
            "dists_to_upstream": [dist_val],
            "min_dist_to_upstream": dist_val,
            # Carry the full result.json through so the aggregator
            # can surface every per-design metric in cycle_0/passing.json
            # alongside the kickoff distance/selection info.
            "result": r,
        })

    doc = {
        "n_generated": len(plan["designs"]),
        "candidates": candidates,
        "wild_type_pdb": str(wild_type_pdb),
    }
    (cycle0_dir / "kickoff_distances.json").write_text(json.dumps(doc, indent=2))
    print(f"[kickoff-distances] wrote {len(candidates)} candidate(s) to kickoff_distances.json")
    return 0


def cmd_kickoff(args: argparse.Namespace) -> int:
    """Phase 2 of kickoff — read kickoff_distances.json, apply intact +
    novelty filters, maximin-select up to --max-passing survivors,
    write cycle_0/passing.json, and submit one cycle-1 chain per
    survivor via sbatch.

    Pure stdlib — runs OUTSIDE the container so sbatch is on PATH.
    Does not import numpy or boltz2_negative_steering.
    """
    experiment_root = args.experiment_root.resolve()
    cycle0_dir = experiment_root / "cycle_0"
    if not cycle0_dir.is_dir():
        print(f"ERROR: no cycle_0 under {experiment_root}", file=sys.stderr)
        return 2

    distances_path = cycle0_dir / "kickoff_distances.json"
    if not distances_path.exists():
        msg = (f"ERROR: {distances_path} does not exist — phase 1 "
               f"(kickoff-distances) probably failed.  Check the in-container "
               f"step in the same SLURM job's log.\n")
        sys.stderr.write(msg)
        return 1

    distances = json.loads(distances_path.read_text())
    n_generated = int(distances.get("n_generated", 0))

    if distances.get("skip_steering"):
        print("cycle 0 skipped steering — nothing to iterate.")
        return 0

    novelty_cutoff = float(args.novelty_cutoff)
    max_passing = int(args.max_passing)
    max_cycles = int(args.max_cycles)

    # Apply intact + novelty filters using the precomputed distances.
    candidates = []
    for c in distances.get("candidates", []):
        intact = (
            c["independent_receptor_rmsd"] <= _INTACT_THRESHOLD
            and c["independent_effector_rmsd"] <= _INTACT_THRESHOLD
        )
        c2 = dict(c)
        c2["receptor_intact"] = intact
        candidates.append(c2)

    intact_candidates = [c for c in candidates if c["receptor_intact"]]
    novel = [
        c for c in intact_candidates
        if c["min_dist_to_upstream"] is not None
        and c["min_dist_to_upstream"] >= novelty_cutoff
    ]
    n_intact = len(intact_candidates)
    n_novel = len(novel)
    selected = maximin_select(novel, max_passing)
    n_selected = len(selected)

    print(f"[kickoff] cycle 0: {n_generated} designs -> "
          f"{n_intact} intact -> {n_novel} novel (≥{novelty_cutoff} Å) "
          f"-> {n_selected} selected for cycle 1")

    (cycle0_dir / "passing.json").write_text(json.dumps({
        "cycle": 0,
        "parent_pathway": None,
        "novelty_cutoff": novelty_cutoff,
        "max_passing": max_passing,
        "n_generated": n_generated,
        "n_intact": n_intact,
        "n_novel": n_novel,
        "n_selected": n_selected,
        "all_candidates": candidates,
        "selected": selected,
    }, indent=2, default=str))

    append_cycle_statistics(
        experiment_root, 0, "cycle_0",
        n_generated=n_generated,
        n_intact=n_intact, n_novel=n_novel, n_selected=n_selected,
        min_dists=[c["min_dist_to_upstream"] for c in selected],
        ra_eff_vs_truth=[c["receptor_aligned_effector_rmsd_vs_truth"] for c in selected],
    )

    if n_selected == 0:
        print("  no qualifying designs at cycle 0 — nothing to iterate.")
        return 0
    if max_cycles < 1:
        print(f"  max-cycles={max_cycles} — not iterating.")
        return 0

    submit_script = SCRIPT_DIR / "submit_boltz2_iterate_steering.sh"
    if not submit_script.exists():
        print(f"  WARN: cannot find {submit_script}; not iterating",
              file=sys.stderr)
        return 0

    resubmit_pathway_chains(
        submit_script=submit_script,
        selected=selected,
        parent_label=None,
        cycle=0,
        experiment_root=experiment_root,
        max_cycles=max_cycles,
        max_passing=max_passing,
        novelty_cutoff=novelty_cutoff,
        n_designs=args.n_designs,
        announce=f"  submitting {n_selected} cycle-1 chain(s)",
        failure_prefix="FAILED to submit cycle 1 for",
    )
    return 0


# ───────────────────────────────────────────────────────────────────────
# Subcommand: kickoff-prefilter  (host, cycle-0 phase 2a)
# ───────────────────────────────────────────────────────────────────────
#
# Cycle-0 equivalent of iterate-collect-prefilter.  Reads
# kickoff_distances.json, applies the intact filter, writes
# cycle_0/prefilter.json.  Does not run novelty/maximin/resubmit —
# those happen in kickoff-finalize after the reversion chain runs.
def cmd_kickoff_prefilter(args: argparse.Namespace) -> int:
    experiment_root = args.experiment_root.resolve()
    cycle0_dir = experiment_root / "cycle_0"
    if not cycle0_dir.is_dir():
        print(f"ERROR: no cycle_0 under {experiment_root}", file=sys.stderr)
        return 2

    distances_path = cycle0_dir / "kickoff_distances.json"
    if not distances_path.exists():
        msg = (f"ERROR: {distances_path} does not exist — phase 1 "
               f"(kickoff-distances) probably failed.\n")
        sys.stderr.write(msg)
        (cycle0_dir / "prefilter.json").write_text(json.dumps({"error": msg}))
        return 1

    distances = json.loads(distances_path.read_text())
    n_generated = int(distances.get("n_generated", 0))

    if distances.get("skip_steering"):
        print("[kickoff-prefilter] cycle 0 skipped steering — "
              "writing empty prefilter")
        (cycle0_dir / "prefilter.json").write_text(json.dumps({
            "cycle": 0,
            "parent_pathway": None,
            "skip_steering": True,
            "exhausted": True,
            "n_generated": 0,
            "n_intact": 0,
            "intact_candidates": [],
        }, indent=2))
        return 0

    intact_candidates: List[Dict] = []
    for c in distances.get("candidates", []):
        intact = (
            c["independent_receptor_rmsd"] <= _INTACT_THRESHOLD
            and c["independent_effector_rmsd"] <= _INTACT_THRESHOLD
        )
        c2 = dict(c)
        c2["receptor_intact"] = intact
        if intact:
            intact_candidates.append(c2)

    print(f"[kickoff-prefilter] cycle 0: {n_generated} generated -> "
          f"{len(intact_candidates)} intact")

    (cycle0_dir / "prefilter.json").write_text(json.dumps({
        "cycle": 0,
        "parent_pathway": None,
        "n_generated": n_generated,
        "n_intact": len(intact_candidates),
        "intact_candidates": intact_candidates,
    }, indent=2, default=str))
    return 0


# ───────────────────────────────────────────────────────────────────────
# Subcommand: kickoff-finalize  (host, cycle-0 phase 2c)
# ───────────────────────────────────────────────────────────────────────
#
# Cycle-0 equivalent of iterate-collect-finalize.  Reads
# cycle_0/prefilter.json + (optionally) cycle_0/reversion_results.json,
# routes verdicts, re-applies structural filter on reverted designs,
# merges pose_holds back into the intact list with their reverted
# sequence rewritten in place, runs novelty/maximin, writes
# cycle_0/passing.json, and submits cycle-1 chains.
def cmd_kickoff_finalize(args: argparse.Namespace) -> int:
    experiment_root = args.experiment_root.resolve()
    cycle0_dir = experiment_root / "cycle_0"
    if not cycle0_dir.is_dir():
        print(f"ERROR: no cycle_0 under {experiment_root}", file=sys.stderr)
        return 2

    prefilter_path = cycle0_dir / "prefilter.json"
    if not prefilter_path.exists():
        print(f"ERROR: {prefilter_path} not found — run "
              f"kickoff-prefilter first", file=sys.stderr)
        return 1
    prefilter = json.loads(prefilter_path.read_text())

    novelty_cutoff = float(args.novelty_cutoff)
    max_passing = int(args.max_passing)
    max_cycles = int(args.max_cycles)

    if prefilter.get("exhausted") or prefilter.get("skip_steering"):
        print("cycle 0 exhausted — nothing to iterate.")
        return 0

    n_generated = int(prefilter.get("n_generated", 0))
    intact_candidates: List[Dict] = list(prefilter.get("intact_candidates", []))

    # Load reversion results if any contaminated designs were processed
    reversion_results: Dict[str, Dict] = {}
    reversion_results_path = cycle0_dir / "reversion_results.json"
    if reversion_results_path.exists():
        reversion_results = json.loads(reversion_results_path.read_text())

    # Bug 1 guard: load contaminated.json so we can detect designs that
    # were flagged contaminated by build-contaminated but never received
    # a verdict.  Such designs MUST be dropped — silently keeping them
    # would let unvalidated contaminated designs pass through.
    contaminated_labels: set = set()
    contaminated_path = cycle0_dir / "contaminated.json"
    if contaminated_path.exists():
        cdoc = json.loads(contaminated_path.read_text())
        contaminated_labels = {
            e.get("label") for e in cdoc.get("contaminated", [])
            if e.get("label")
        }
    missing_verdict_labels: set = (
        contaminated_labels - set(reversion_results.keys())
    )
    if missing_verdict_labels:
        print(f"[kickoff-finalize] WARN: "
              f"{len(missing_verdict_labels)} contaminated design(s) have "
              f"no verdict in reversion_results.json — will be dropped: "
              f"{sorted(missing_verdict_labels)}", file=sys.stderr)

    # Route verdicts.  Cycle-0 labels are just the design name (e.g.
    # "design_40") since there's no parent pathway.
    pose_holds_count = 0
    pose_collapses_count = 0
    new_contamination_count = 0
    dropped_labels: List[str] = []
    reverted_labels: List[str] = []
    # Designs that got dropped by verdict routing — surfaced into
    # passing.json's all_candidates with a reversion_verdict marker
    # so the aggregator CSV can render them.  They do NOT flow into
    # novelty/maximin.
    dropped_rows: List[Dict] = []

    def _label_for(c: Dict) -> str:
        return c.get("design", "?")

    kept: List[Dict] = []
    missing_verdict_count = 0
    for c in intact_candidates:
        label = _label_for(c)
        rev = reversion_results.get(label)
        if rev is None:
            # Bug 1 guard: if this label was flagged contaminated but
            # has no verdict, drop it instead of silently keeping it.
            if label in contaminated_labels:
                missing_verdict_count += 1
                dropped_labels.append(label)
                print(f"  [missing_verdict] {label}: dropped — flagged "
                      f"contaminated but no entry in reversion_results.json")
                c_drop = dict(c)
                c_drop["reversion_verdict"] = "missing_verdict"
                c_drop["reversion_dropped"] = 1
                # Steered metrics already in base columns of c_drop;
                # no overlay required (and no reverted prediction exists).
                dropped_rows.append(c_drop)
                continue
            kept.append(c)
            continue

        verdict = rev.get("verdict", "")
        if verdict == "pose_holds":
            pose_holds_count += 1
            reverted_labels.append(label)
            design_workdir = Path(c["pdb"]).parent
            # Locate reverted FASTA from reversion_plan.json
            reverted_fasta = None
            rp_path = cycle0_dir / "reversion_plan.json"
            if rp_path.exists():
                rp = json.loads(rp_path.read_text())
                for entry in rp.get("entries", []):
                    if entry.get("label") == label:
                        reverted_fasta = Path(entry["reverted_receptor_fasta"])
                        break
            if reverted_fasta and reverted_fasta.exists():
                try:
                    seq_lines = reverted_fasta.read_text().splitlines()
                    reverted_seq = "".join(
                        l.strip() for l in seq_lines
                        if l.strip() and not l.startswith(">")
                    )
                    (design_workdir / "receptor.fasta").write_text(
                        f">receptor_reverted\n{reverted_seq}\n"
                    )
                    (design_workdir / "reversion_applied.json").write_text(
                        json.dumps({
                            "label": label,
                            "verdict": verdict,
                            "reverted_sequence": reverted_seq,
                            "reversion_result": rev,
                        }, indent=2, default=str)
                    )
                    print(f"  [pose_holds] {label}: rewrote "
                          f"{design_workdir/'receptor.fasta'}")
                except Exception as e:
                    print(f"  WARN {label}: failed to rewrite "
                          f"receptor.fasta: {e}", file=sys.stderr)
            else:
                print(f"  WARN {label}: reverted FASTA not found",
                      file=sys.stderr)

            # Build the kept candidate dict.  Critical invariant:
            # the base columns hold the steered values; reverted
            # values live only in dedicated reverted_* keys.  For
            # selection, _active_ra_eff_for_selection carries the
            # post-reversion ra_eff so maximin operates on the
            # reverted pose.
            c2 = dict(c)
            ra = rev.get("reverted_ra_eff_vs_truth")
            if ra is not None:
                c2["reverted_ra_eff_vs_truth"] = float(ra)
                c2["_active_ra_eff_for_selection"] = float(ra)
            ind_rec = rev.get("reverted_independent_receptor_rmsd")
            if ind_rec is not None:
                c2["reverted_independent_receptor_rmsd"] = float(ind_rec)
            ind_eff = rev.get("reverted_independent_effector_rmsd")
            if ind_eff is not None:
                c2["reverted_independent_effector_rmsd"] = float(ind_eff)
            _copy_reverted_confidence(c2, rev)
            c2["reversion_verdict"] = verdict
            c2["reverted_sequence_used_as_parent"] = 1
            reverted_intact = (
                (ind_rec is not None and float(ind_rec) <= 5.0)
                and (ind_eff is not None and float(ind_eff) <= 5.0)
            )
            if not reverted_intact:
                print(f"  [pose_holds→drop] {label}: reverted fails "
                      f"intact filter, dropping anyway")
                dropped_labels.append(label)
                c2["reversion_verdict"] = "pose_holds_not_intact"
                c2["reversion_dropped"] = 1
                dropped_rows.append(c2)
                continue
            c2["reversion_dropped"] = 0
            kept.append(c2)
        elif verdict == "pose_collapses":
            pose_collapses_count += 1
            dropped_labels.append(label)
            print(f"  [pose_collapses] {label}: {rev.get('reason','')}")
            c_drop = dict(c)
            c_drop["reversion_verdict"] = verdict
            c_drop["reversion_dropped"] = 1
            # Steered metrics already in base columns of c_drop.
            # Overlay reverted metrics so the CSV shows what the
            # reverted prediction looked like.
            if rev.get("reverted_ra_eff_vs_truth") is not None:
                c_drop["reverted_ra_eff_vs_truth"] = float(rev["reverted_ra_eff_vs_truth"])
            if rev.get("reverted_independent_receptor_rmsd") is not None:
                c_drop["reverted_independent_receptor_rmsd"] = float(rev["reverted_independent_receptor_rmsd"])
            if rev.get("reverted_independent_effector_rmsd") is not None:
                c_drop["reverted_independent_effector_rmsd"] = float(rev["reverted_independent_effector_rmsd"])
            _copy_reverted_confidence(c_drop, rev)
            dropped_rows.append(c_drop)
        elif verdict == "new_contamination":
            new_contamination_count += 1
            dropped_labels.append(label)
            print(f"  [new_contamination] {label}: {rev.get('reason','')}")
            c_drop = dict(c)
            c_drop["reversion_verdict"] = verdict
            c_drop["reversion_dropped"] = 1
            if rev.get("reverted_ra_eff_vs_truth") is not None:
                c_drop["reverted_ra_eff_vs_truth"] = float(rev["reverted_ra_eff_vs_truth"])
            if rev.get("reverted_independent_receptor_rmsd") is not None:
                c_drop["reverted_independent_receptor_rmsd"] = float(rev["reverted_independent_receptor_rmsd"])
            if rev.get("reverted_independent_effector_rmsd") is not None:
                c_drop["reverted_independent_effector_rmsd"] = float(rev["reverted_independent_effector_rmsd"])
            _copy_reverted_confidence(c_drop, rev)
            dropped_rows.append(c_drop)
        else:
            dropped_labels.append(label)
            print(f"  WARN {label}: unknown verdict {verdict!r}, dropping")
            c_drop = dict(c)
            c_drop["reversion_verdict"] = verdict or "unknown"
            c_drop["reversion_dropped"] = 1
            dropped_rows.append(c_drop)

    intact_candidates = kept

    novel = [
        c for c in intact_candidates
        if c.get("min_dist_to_upstream") is not None
        and c["min_dist_to_upstream"] >= novelty_cutoff
    ]
    n_intact = len(intact_candidates)
    n_novel = len(novel)
    selected = maximin_select(novel, max_passing)
    n_selected = len(selected)

    print(f"[kickoff-finalize] cycle 0: {n_generated} designs -> "
          f"{n_intact} intact (after verdicts) -> {n_novel} novel -> "
          f"{n_selected} selected for cycle 1")
    print(f"  verdicts: pose_holds={pose_holds_count} "
          f"pose_collapses={pose_collapses_count} "
          f"new_contamination={new_contamination_count}")

    (cycle0_dir / "passing.json").write_text(json.dumps({
        "cycle": 0,
        "parent_pathway": None,
        "novelty_cutoff": novelty_cutoff,
        "max_passing": max_passing,
        "n_generated": n_generated,
        "n_intact": n_intact,
        "n_novel": n_novel,
        "n_selected": n_selected,
        "pose_holds_count": pose_holds_count,
        "pose_collapses_count": pose_collapses_count,
        "new_contamination_count": new_contamination_count,
        "reverted_labels": reverted_labels,
        "dropped_labels": dropped_labels,
        "all_candidates": intact_candidates + dropped_rows,
        "selected": selected,
    }, indent=2, default=str))

    append_cycle_statistics(
        experiment_root, 0, "cycle_0",
        n_generated=n_generated,
        n_intact=n_intact, n_novel=n_novel, n_selected=n_selected,
        min_dists=[c["min_dist_to_upstream"] for c in selected],
        ra_eff_vs_truth=[c["receptor_aligned_effector_rmsd_vs_truth"] for c in selected],
    )

    if n_selected == 0:
        print("  no qualifying designs at cycle 0 — nothing to iterate.")
        return 0
    if max_cycles < 1:
        print(f"  max-cycles={max_cycles} — not iterating.")
        return 0

    submit_script = SCRIPT_DIR / "submit_boltz2_iterate_steering.sh"
    if not submit_script.exists():
        print(f"  WARN: cannot find {submit_script}; not iterating",
              file=sys.stderr)
        return 0

    resubmit_pathway_chains(
        submit_script=submit_script,
        selected=selected,
        parent_label=None,
        cycle=0,
        experiment_root=experiment_root,
        max_cycles=max_cycles,
        max_passing=max_passing,
        novelty_cutoff=novelty_cutoff,
        n_designs=args.n_designs,
        announce=f"  submitting {n_selected} cycle-1 chain(s)",
        failure_prefix="FAILED to submit cycle 1 for",
    )
    return 0


# ───────────────────────────────────────────────────────────────────────
# Subcommand: plan-reversions
# ───────────────────────────────────────────────────────────────────────
#
# Runs AFTER structural filtering inside a cycle's collect phase.
# Takes the set of structurally-passing designs, identifies which
# ones were flagged `contaminated` by build-contaminated (i.e. have
# any mutated residue contacting the effector in the steered pose),
# and stages reverted Boltz YAMLs for each one under
# <workdir>/reversions/<safe_label>/.  Produces reversion_plan.json.
#
# Inputs (all file-based so this subcommand can be tested standalone):
#   <workdir>/plan.json                — for chain IDs, effector fasta,
#                                        boltz container + hyperparams,
#                                        ground truth path
#   <workdir>/contaminated.json        — written by the collect phase;
#                                        list of contaminated candidates,
#                                        each with design_workdir +
#                                        positions_to_revert +
#                                        cumulative_mutations + label
#
# Outputs:
#   <workdir>/reversion_plan.json      — manifest listing every staged
#                                        reversion, consumed by
#                                        predict-reversion-one and
#                                        harvest-reversions
#
# This must run INSIDE the singularity container because it imports
# boltz2_negative_steering.write_boltz_yaml (which lives in that
# module's module-level scope and cannot be lazy-imported cleanly
# without triggering the numpy+Bio imports).
def cmd_plan_reversions(args: argparse.Namespace) -> int:
    workdir: Path = args.workdir.resolve()
    plan_path = workdir / "plan.json"
    contaminated_path = workdir / "contaminated.json"

    if not plan_path.exists():
        print(f"ERROR: {plan_path} not found — plan-reversions must "
              f"run after iterate-plan/compute-distances", file=sys.stderr)
        return 2
    if not contaminated_path.exists():
        # Permissive: build-contaminated may have skipped if there
        # were no intact candidates.  Write an empty reversion_plan.json
        # so downstream stages no-op cleanly.
        print("[plan-reversions] no contaminated.json — writing empty "
              "reversion_plan.json (no reversions to stage)")
        plan_for_chains = json.loads(plan_path.read_text())
        (workdir / "reversion_plan.json").write_text(json.dumps({
            "workdir": str(workdir),
            "n_contaminated": 0,
            "n_staged": 0,
            "pred_receptor_chain": plan_for_chains.get("pred_receptor_chain", "A"),
            "pred_effector_chain": plan_for_chains.get("pred_effector_chain", "B"),
            "entries": [],
        }, indent=2))
        return 0

    plan = json.loads(plan_path.read_text())
    contaminated_doc = json.loads(contaminated_path.read_text())
    contaminated_list = contaminated_doc.get("contaminated", [])

    if not contaminated_list:
        print("[plan-reversions] no contaminated designs — "
              "writing empty reversion_plan.json")
        (workdir / "reversion_plan.json").write_text(json.dumps({
            "workdir": str(workdir),
            "n_contaminated": 0,
            "n_staged": 0,
            "pred_receptor_chain": plan.get("pred_receptor_chain", "A"),
            "pred_effector_chain": plan.get("pred_effector_chain", "B"),
            "entries": [],
        }, indent=2))
        return 0

    # Load effector sequence the same way cmd_plan does: from the
    # ground-truth PDB's effector chain, via the steering module's
    # get_chain_sequence.  (Or from an effector-fasta override if
    # plan.json recorded one.)
    script_dir = SCRIPT_DIR
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from boltz2_negative_steering import get_chain_sequence  # type: ignore
    import reversion  # type: ignore

    ground_truth = Path(plan["ground_truth"])
    truth_eff_chain = plan["truth_effector_chain"]
    eff_override = plan.get("effector_fasta_override")
    if eff_override:
        # Single-record FASTA read, same shape as reversion's helper
        text = Path(eff_override).read_text()
        seq_lines: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq_lines.append(line)
        effector_seq = "".join(seq_lines)
    else:
        effector_seq = get_chain_sequence(ground_truth, truth_eff_chain)

    plan_meta = {
        "effector_seq": effector_seq,
        "effector_template_cif": plan.get("effector_template_cif"),
        "pred_receptor_chain": plan.get("pred_receptor_chain", "A"),
        "pred_effector_chain": plan.get("pred_effector_chain", "B"),
        "base_seed": int(plan.get("seed", 0)),
        "num_seeds": int(plan.get("num_seeds", 1)),
    }

    n = len(contaminated_list)
    print(f"[plan-reversions] staging {n} reverted Boltz YAMLs")
    manifest_path = reversion.write_reversion_plan(
        workdir, contaminated_list, plan_meta
    )
    manifest = json.loads(manifest_path.read_text())
    print(f"  wrote {manifest_path} ({manifest['n_staged']}/{n} staged)")
    if manifest["n_staged"] < n:
        print(f"  WARN: {n - manifest['n_staged']} contaminated design(s) "
              f"could not be staged — see stderr above for reasons",
              file=sys.stderr)
    return 0


# ───────────────────────────────────────────────────────────────────────
# Subcommand: predict-reversion-one
# ───────────────────────────────────────────────────────────────────────
#
# GPU array task: run Boltz on one staged reversion YAML by array index.
# Modelled on boltz2_negative_steering.cmd_predict_one.
def cmd_predict_reversion_one(args: argparse.Namespace) -> int:
    workdir: Path = args.workdir.resolve()
    plan_path = workdir / "plan.json"
    manifest_path = workdir / "reversion_plan.json"

    if not plan_path.exists():
        print(f"ERROR: {plan_path} not found", file=sys.stderr)
        return 2
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found — run plan-reversions first",
              file=sys.stderr)
        return 2

    plan = json.loads(plan_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("entries", [])

    idx = args.index
    if idx < 0 or idx >= len(entries):
        # Out-of-range is not an error on array tasks — it just means
        # this array slot has no work.  Exit 0 so the array completes
        # cleanly even if the shell script over-provisioned tasks.
        print(f"[predict-reversion-one] index {idx} out of range "
              f"(have {len(entries)} reversions) — nothing to do")
        return 0

    entry = entries[idx]
    label = entry["label"]
    rev_dir = Path(entry["rev_dir"])
    yaml_path = Path(entry["yaml"])

    print(f"[predict-reversion-one] {label} (index {idx})", flush=True)

    script_dir = SCRIPT_DIR
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from boltz2_negative_steering import run_boltz  # type: ignore

    try:
        pred_pdb = run_boltz(
            yaml_path, rev_dir,
            plan["boltz_container"],
            plan["recycling_steps"], plan["diffusion_samples"],
            # Use pre-computed seed if available (new manifest schema);
            # fall back to old formula for backward compatibility.
            entry.get("boltz_seed",
                       int(plan["seed"]) + 100000 + idx + 1),
            plan["no_kernels"],
        )
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        # Still write a stub result so harvest can report this cleanly
        (rev_dir / "predict_error.txt").write_text(str(e))
        return 1

    # Copy the chosen PDB out to a stable name, same as predict-one does
    shutil.copy2(pred_pdb, rev_dir / "prediction.pdb")
    print(f"  wrote {rev_dir / 'prediction.pdb'}")
    return 0


# ───────────────────────────────────────────────────────────────────────
# Subcommand: harvest-reversions
# ───────────────────────────────────────────────────────────────────────
#
# In-container harvest: for every staged reversion, compute reverted
# structural metrics (ra_eff vs ground truth AND vs steered parent),
# contact analysis, contamination check on the reverted prediction,
# and the pose_holds / pose_collapses / new_contamination verdict.
# Writes reversion_results.json.
def cmd_harvest_reversions(args: argparse.Namespace) -> int:
    workdir: Path = args.workdir.resolve()
    plan_path = workdir / "plan.json"
    manifest_path = workdir / "reversion_plan.json"

    if not plan_path.exists():
        print(f"ERROR: {plan_path} not found", file=sys.stderr)
        return 2
    if not manifest_path.exists():
        # Permissive: if plan-reversions never ran (e.g. because
        # build-contaminated found nothing or crashed upstream),
        # write an empty results doc so kickoff-finalize /
        # iterate-collect-finalize can still run without a reversion
        # phase.  No verdicts will be applied; every intact design
        # flows through as-is.
        print("[harvest-reversions] no reversion_plan.json — writing "
              "empty reversion_results.json")
        (workdir / "reversion_results.json").write_text("{}\n")
        return 0

    plan = json.loads(plan_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    # Resolve compute_metrics.py path (same logic as cmd_compute_final_metrics)
    compute_metrics_script = args.compute_metrics_script
    if compute_metrics_script is None:
        compute_metrics_script = SCRIPT_DIR / "compute_metrics.py"
    compute_metrics_script = Path(compute_metrics_script).resolve()
    if not compute_metrics_script.exists():
        print(f"ERROR: compute_metrics.py not found at {compute_metrics_script}",
              file=sys.stderr)
        return 2

    script_dir = SCRIPT_DIR
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import reversion  # type: ignore

    print(f"[harvest-reversions] harvesting "
          f"{len(manifest.get('entries', []))} reverted prediction(s)")
    results = reversion.harvest_reversion_results(
        workdir=workdir,
        ground_truth_pdb=Path(plan["ground_truth"]),
        truth_receptor_chain=plan["truth_receptor_chain"],
        truth_effector_chain=plan["truth_effector_chain"],
        compute_metrics_script=compute_metrics_script,
        pae_cutoff=float(args.pae_cutoff),
        contact_cutoff=float(args.contact_cutoff),
    )

    # Apply verdict classification per label.  We need the steered
    # row to classify, which lives in contaminated.json.  Verdict
    # threshold defaults match compute-final-metrics' RFDiffusion
    # tier: ra_eff < 5.0 and intact.
    contaminated_path = workdir / "contaminated.json"
    steered_by_label: Dict[str, Dict] = {}
    if contaminated_path.exists():
        cdoc = json.loads(contaminated_path.read_text())
        for c in cdoc.get("contaminated", []):
            steered_by_label[c["label"]] = {
                "receptor_aligned_effector_rmsd": c.get("steered_ra_eff_vs_truth"),
                "receptor_intact": 1,  # had to be intact to make it here
            }

    # Build the contamination-gating set: union of the RFDiffusion
    # design region and the true interface, 1-based to match the
    # coordinate system of reverted_mutated_contact_positions.  This
    # mirrors the pre-reversion contamination policy: only flag
    # mutated contacts at positions inside these regions as
    # new_contamination, since positions outside these regions
    # were already accepted as valid steering contacts on the
    # steered pose and are not reverted.
    gating_1b: Set[int] = set()
    dr = plan.get("design_region_idx") or []
    ti = plan.get("true_interface_idx") or []
    for i in dr:
        gating_1b.add(int(i) + 1)
    for i in ti:
        gating_1b.add(int(i) + 1)
    if gating_1b:
        print(f"[harvest-reversions] contamination gating set: "
              f"{len(gating_1b)} positions (design region ∪ true interface)")
    else:
        print("[harvest-reversions] WARN: no design_region_idx or "
              "true_interface_idx in plan.json — falling back to "
              "legacy ungated contamination check")

    verdicts: Dict[str, Dict] = {}
    for label, rev in results.items():
        steered_row = steered_by_label.get(label, {})
        verdict, reason = reversion.classify_reversion_verdict(
            steered_row, rev,
            structural_ra_eff_cutoff=float(args.structural_ra_eff_cutoff),
            structural_intact_required=True,
            contamination_gating_positions=gating_1b if gating_1b else None,
        )
        verdicts[label] = {
            "verdict": verdict,
            "reason": reason,
            **rev,
        }
        print(f"  {label:30s} verdict={verdict:18s} {reason}")

    out_path = workdir / "reversion_results.json"
    out_path.write_text(json.dumps(verdicts, indent=2, default=str))
    print(f"wrote {out_path}")
    return 0


# ───────────────────────────────────────────────────────────────────────
# CSV schema helpers — translate internal row dicts to the clean
# all_results_multicycle.csv schema at the write boundary.
# ───────────────────────────────────────────────────────────────────────
#
# Internal row dicts (built by cmd_aggregate) carry the historical
# field names because that's what the rest of the pipeline produces
# and reads.  At write time we translate to a clean schema with
# consistent steered_/reverted_ prefixes, drop redundant columns,
# add the derived columns (passes_filter flags, reverted mutation
# strings, unified ranks), and order columns into three logical
# blocks:
#
#   A. Identification          (cycle, pathway, design, status, pdb)
#   B. Steering details        (mutations, RMSDs, jaccards, filter,
#                               then steered_<cm> metrics)
#   C. Reversion details       (verdict, mutations, RMSDs, filter,
#                               then reverted metrics)
#   End. Selection + ranks     (min_dist_to_upstream, selected, ranks)

# Map of (internal key) -> (new CSV column name).  Only keys that need
# renaming appear here; others pass through unchanged.
_AGG_RENAME = {
    # Steering RMSDs
    "receptor_aligned_effector_rmsd":  "steered_ra_eff_vs_truth",
    "independent_receptor_rmsd":       "steered_independent_receptor_rmsd",
    "independent_effector_rmsd":       "steered_independent_effector_rmsd",
    "receptor_intact":                 "steered_receptor_intact",
    # Steering metadata + interface
    "total_mutations":                 "steered_total_mutations",
    "mutations_chimerax":              "steered_mutations_chimerax",
    "mutations_aa":                    "steered_mutations_aa",
    "wrong_jaccard":                   "steered_wrong_jaccard",
    "n_shared_wrong":                  "steered_n_shared_wrong",
    "true_jaccard":                    "steered_true_jaccard",
    "n_shared_true":                   "steered_n_shared_true",
    "n_design_interface_residues":     "steered_n_design_interface_residues",
}

# Internal keys that must be DROPPED at write time (they were
# duplicates or are now superseded).
_AGG_DROP = {
    "delta_receptor_aligned_effector_rmsd",  # trivially derivable
    "rank",                                  # legacy cycle-0 row index
    "rank_by_truth",                         # superseded by rank_by_ra_eff
    # The old "steered_*" fields the code used to write under those
    # exact names (when the base columns had been overwritten).  After
    # the finalize refactor, base columns are no longer overwritten,
    # so these duplicates are unnecessary.  Drop if present (defensive).
    "steered_ra_eff_vs_truth",
    "steered_independent_receptor_rmsd",
    "steered_independent_effector_rmsd",
    # Internal selection-only key, never user-facing
    "_active_ra_eff_for_selection",
}


def _translate_aggregate_row(r: Dict) -> Dict:
    """Apply _AGG_RENAME and _AGG_DROP to produce a row dict ready
    for CSV writing.  Also adds derived columns:
      - steered_passes_filter (computed from steered_receptor_intact
        and steered_ra_eff_vs_truth — populated by the writer if the
        downstream compute-final-metrics step recorded filter inputs)
      - reverted_passes_filter (likewise from reverted_receptor_intact)
      - reverted_receptor_intact (derived from reverted RMSDs)

    Pure dict transformation; does not touch other rows.
    """
    out: Dict = {}
    # First, drop the keys that should never appear in output
    for k, v in r.items():
        if k in _AGG_DROP:
            continue
        out_key = _AGG_RENAME.get(k, k)
        out[out_key] = v

    # Derive reverted_receptor_intact from reverted RMSDs (if present)
    rev_rec = out.get("reverted_independent_receptor_rmsd", "")
    rev_eff = out.get("reverted_independent_effector_rmsd", "")
    if rev_rec != "" and rev_eff != "":
        try:
            out["reverted_receptor_intact"] = (
                1 if (float(rev_rec) <= 5.0 and float(rev_eff) <= 5.0) else 0
            )
        except (ValueError, TypeError):
            out["reverted_receptor_intact"] = ""
    else:
        out["reverted_receptor_intact"] = ""

    return out


def _populate_reverted_mutations(rows: List[Dict],
                                 experiment_root: Path) -> None:
    """For pose_holds rows, compute the post-reversion mutation set
    (cumulative mutations minus reverted positions) and write
    reverted_total_mutations / reverted_mutations_chimerax /
    reverted_mutations_aa.  All other rows leave these blank.

    Reads cycle_N/(pathway_<parent>/)?reversion_results.json keyed
    by pathway label to get positions_to_revert.  Reads cycle_0/
    plan.json once for pred_receptor_chain.
    """
    # Resolve pred_receptor_chain and contact_cutoff from cycle_0 plan
    pred_rec_chain = "A"
    cycle0_contact_cutoff = ""
    cycle0_plan_path = experiment_root / "cycle_0" / "plan.json"
    if cycle0_plan_path.exists():
        try:
            cp = json.loads(cycle0_plan_path.read_text())
            pred_rec_chain = cp.get("pred_receptor_chain", "A")
            cycle0_contact_cutoff = str(cp.get("contact_cutoff", ""))
        except Exception:
            pass

    # Cache reversion_results.json reads by workdir
    rev_cache: Dict[Path, Dict] = {}

    def _load_rev_results(cycle: int, parent_pathway: str) -> Dict:
        if cycle == 0:
            wd = experiment_root / "cycle_0"
        else:
            wd = (experiment_root / f"cycle_{cycle}"
                  / f"pathway_{parent_pathway}")
        if wd in rev_cache:
            return rev_cache[wd]
        rr_path = wd / "reversion_results.json"
        data: Dict = {}
        if rr_path.exists():
            try:
                data = json.loads(rr_path.read_text())
            except Exception:
                data = {}
        rev_cache[wd] = data
        return data

    # Cache contaminated.json reads by workdir
    contam_cache: Dict[Path, Dict] = {}

    def _load_contaminated_doc(cycle: int, parent_pathway: str) -> Dict:
        if cycle == 0:
            wd = experiment_root / "cycle_0"
        else:
            wd = (experiment_root / f"cycle_{cycle}"
                  / f"pathway_{parent_pathway}")
        if wd in contam_cache:
            return contam_cache[wd]
        path = wd / "contaminated.json"
        data: Dict = {}
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except Exception:
                data = {}
        contam_cache[wd] = data
        return data

    # Cache reversion_plan.json reads by workdir (for positions_to_revert)
    rev_plan_cache: Dict[Path, Dict] = {}

    def _load_rev_plan(cycle: int, parent_pathway: str) -> Dict:
        if cycle == 0:
            wd = experiment_root / "cycle_0"
        else:
            wd = (experiment_root / f"cycle_{cycle}"
                  / f"pathway_{parent_pathway}")
        if wd in rev_plan_cache:
            return rev_plan_cache[wd]
        rp_path = wd / "reversion_plan.json"
        data: Dict = {}
        if rp_path.exists():
            try:
                data = json.loads(rp_path.read_text())
            except Exception:
                data = {}
        rev_plan_cache[wd] = data
        return data

    for r in rows:
        # Default blank
        r.setdefault("reverted_total_mutations", "")
        r.setdefault("reverted_mutations_chimerax", "")
        r.setdefault("reverted_mutations_aa", "")
        r.setdefault("steered_mutated_contact_positions", "")
        r.setdefault("reverted_mutated_contact_positions", "")

        verdict = r.get("reversion_verdict") or ""
        cycle = int(r.get("cycle", 0) or 0)
        parent_pathway = r.get("parent_pathway", "") or ""

        # Derive the per-design label used by contaminated.json and
        # reversion_results.json.  Cycle-0 rows have pathway="cycle_0"
        # (a constant, not a per-design identifier); the per-design
        # identifier is in the "design" column (e.g. "design_04").
        # Cycle-1+ rows have pathway = the leaf label (e.g. "c0d05.c1d03").
        if cycle == 0:
            leaf_label = r.get("design", "")
        else:
            leaf_label = r.get("pathway", "")

        # For ANY row that went through reversion (pose_holds or any
        # dropped verdict), surface the set of mutated positions that
        # were contacting the effector in the steered prediction —
        # the contamination that triggered the reversion.
        if verdict != "":
            cdoc = _load_contaminated_doc(cycle, parent_pathway)
            for e in cdoc.get("contaminated", []):
                if e.get("label") == leaf_label:
                    pos = e.get("positions_to_revert") or []
                    r["steered_mutated_contact_positions"] = ",".join(
                        str(p) for p in sorted(pos)
                    )
                    break

        # For ANY reverted row (pose_holds, pose_collapses,
        # new_contamination), populate the reverted contact data
        # from reversion_results.json.  The per-seed aggregator
        # downstream needs the REAL contamination set per seed to
        # compute the per-position majority correctly.  Blanking
        # these columns for non-pose_holds rows (the pre-audit
        # behaviour) caused the aggregator to see empty contact sets
        # for seeds whose harvest actually flagged contamination,
        # inflating the pose_holds count and misclassifying the
        # outcome as pose_holds when some seeds actually
        # said new_contamination.
        if verdict != "":
            rev_data = _load_rev_results(cycle, parent_pathway)
            rev_entry = rev_data.get(leaf_label) or {}
            rmcp = rev_entry.get("reverted_mutated_contact_positions") or []
            r["reverted_mutated_contact_positions"] = ",".join(
                str(p) for p in sorted(rmcp)
            )
            r["reverted_n_contact_residues"] = rev_entry.get(
                "reverted_n_contact_residues", "")
            r["reverted_contact_residues"] = rev_entry.get(
                "reverted_contact_residues", "")
            r["reverted_contact_cutoff_used"] = cycle0_contact_cutoff

        if verdict != "pose_holds":
            continue

        # pose_holds: compute reverted_mutations_* from remaining mutations.
        # (reverted contact data was already populated above for all
        # reverted rows.)
        rev_data = _load_rev_results(cycle, parent_pathway)
        rev_entry = rev_data.get(leaf_label) or {}

        # Read positions_to_revert from contaminated.json (the
        # authoritative source — reversion_results.json does not
        # carry this field).  Fall back to reversion_plan.json
        # entries if contaminated.json lookup fails.
        positions_to_revert: set = set()
        cdoc = _load_contaminated_doc(cycle, parent_pathway)
        for e in cdoc.get("contaminated", []):
            if e.get("label") == leaf_label:
                positions_to_revert = set(
                    int(p) for p in (e.get("positions_to_revert") or [])
                )
                break
        if not positions_to_revert:
            # Fallback: reversion_plan.json entries also carry the list
            rp = _load_rev_plan(cycle, parent_pathway)
            for e in rp.get("entries", []):
                if e.get("label") == leaf_label:
                    positions_to_revert = set(
                        int(p) for p in (e.get("positions_to_revert") or [])
                    )
                    break

        # Cumulative mutation history.  For cycle-0 rows the leaf_label
        # is "design_NN" which parse_pathway_label cannot parse, so
        # read_cumulative_mutations would crash.  Read mutations.tsv
        # directly from the design's workdir instead.
        cum_muts: List[Tuple[int, str, str]] = []
        if cycle == 0:
            pdb_col = r.get("pdb", "")
            if pdb_col:
                mut_tsv = Path(pdb_col).parent / "mutations.tsv"
                cum_muts = read_mutations_tsv_full(mut_tsv)
        else:
            try:
                cum_muts = read_cumulative_mutations(
                    experiment_root, leaf_label)
            except Exception:
                cum_muts = []

        # Filter out reverted positions; for positions touched twice
        # (rare) keep the most recent edit
        latest_by_pos: Dict[int, Tuple[int, str, str]] = {}
        for pos, wt, mut in cum_muts:
            latest_by_pos[pos] = (pos, wt, mut)
        remaining = [
            t for pos, t in latest_by_pos.items()
            if pos not in positions_to_revert
        ]
        remaining.sort(key=lambda t: t[0])
        cx, aa = format_mutation_strings(remaining, chain_id=pred_rec_chain)
        r["reverted_total_mutations"] = len(remaining)
        r["reverted_mutations_chimerax"] = cx
        r["reverted_mutations_aa"] = aa


def _row_is_clean_steered(r: Dict) -> bool:
    """Row represents a clean steered design that survived everything:
    not contaminated AND steered prediction passes the structural
    filter (intact AND ra_eff < 5 Å).

    P0.4 (changed from earlier behaviour): the eligibility predicate
    used to gate on `steered_ipsae_min` being populated, as a proxy
    for "compute-final-metrics ran on this row, which it only does
    for ra_eff-passing rows".  That proxy breaks under
    `compute-final-metrics --populate-all`, which populates ipSAE
    for every non-dropped row including wrong-interface ones, and
    would otherwise let those rows into the composite ranking.  The
    proxy is now replaced with a direct ra_eff check against a
    hardcoded 5 Å threshold matching the steering pipeline's default
    (rmsd_threshold).  Rows missing ra_eff entirely are still
    excluded.
    """
    if r.get("design") == "initial":
        return False
    verdict = (r.get("reversion_verdict") or "").strip()
    if verdict != "":
        return False
    try:
        if int(r.get("steered_receptor_intact", 0) or 0) != 1:
            return False
    except (ValueError, TypeError):
        return False
    # Direct ra_eff check (replaces the legacy ipSAE-presence proxy).
    ra_str = r.get("steered_ra_eff_vs_truth", "")
    if ra_str in ("", None):
        return False
    try:
        ra = float(ra_str)
    except (ValueError, TypeError):
        return False
    if math.isnan(ra) or math.isinf(ra):
        return False
    if ra >= 5.0:
        return False
    return True


def _row_is_pose_holds(r: Dict) -> bool:
    """Row represents a pose_holds design whose reverted prediction
    passes the structural filter (intact AND ra_eff < 5 Å).

    Mirrors the ra_eff check in _row_is_clean_steered so the two
    eligibility predicates are consistent: a pose_holds row whose
    reverted prediction has terrible ra_eff (e.g. 25 Å) should not
    be ranking-eligible, just as a clean steered row with terrible
    steered ra_eff is excluded.  This was previously only enforced
    implicitly by the reversion harvest's intact filter, which does
    not cover ra_eff."""
    if r.get("reversion_verdict") != "pose_holds":
        return False
    try:
        if int(r.get("reverted_receptor_intact", 0) or 0) != 1:
            return False
    except (ValueError, TypeError):
        return False
    ra_str = r.get("reverted_ra_eff_vs_truth", "")
    if ra_str in ("", None):
        return False
    try:
        ra = float(ra_str)
    except (ValueError, TypeError):
        return False
    if math.isnan(ra) or math.isinf(ra):
        return False
    if ra >= 5.0:
        return False
    return True


def _ranking_metric(r: Dict, name: str) -> Optional[float]:
    """Return the canonical value of `name` for ranking purposes.
    `name` is one of 'ra_eff_vs_truth', 'ipsae_min'.

    For clean steered rows, reads the steered_ value.
    For pose_holds rows, reads the reverted_ value.
    Returns None if the row is ineligible or the value is missing.
    """
    if _row_is_clean_steered(r):
        prefix = "steered_"
    elif _row_is_pose_holds(r):
        prefix = "reverted_"
    else:
        return None
    v = r.get(prefix + name, "")
    if v in ("", None):
        return None
    try:
        v = float(v)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (ValueError, TypeError):
        return None


def _ranking_composite(r: Dict) -> Optional[float]:
    """Composite ranker score (P0.3): true_jaccard − 0.05 · ra_eff_vs_truth.
    Higher is better.

    Eligibility matches _ranking_metric.  ra_eff is sourced from
    steered_ or reverted_ per the row type; true_jaccard is always
    sourced from steered_true_jaccard because the jaccard overlap is
    computed once on the steered prediction and not recomputed after
    reversion (the interface identity is carried by the designed
    sequence, not the steering mutations).

    Returns None if either component is missing / unparseable or the
    row is not ranking-eligible.
    """
    ra = _ranking_metric(r, "ra_eff_vs_truth")
    if ra is None:
        return None
    v = r.get("steered_true_jaccard", "")
    if v in ("", None):
        return None
    try:
        tj = float(v)
        if math.isnan(tj) or math.isinf(tj):
            return None
    except (ValueError, TypeError):
        return None
    return tj - 0.05 * ra


def _compute_unified_ranks(rows: List[Dict]) -> None:
    """Add rank_by_ra_eff and rank_by_composite_score columns.  Both
    ranks are 1-based; rank 1 is best.  Eligible rows: clean steered
    (no contamination, passes filter) OR pose_holds (passes reverted
    filter).  Ineligible rows get blank.

    P0.3: rank_by_composite_score is the primary ranker.  Score is
    true_jaccard − 0.05 · ra_eff_vs_truth; higher is better.
    P0.4: rank_by_ipsae_min is dropped entirely.
    """
    # Initialise blanks
    for r in rows:
        r.setdefault("rank_by_ra_eff", "")
        r.setdefault("rank_by_composite_score", "")

    # rank_by_ra_eff: ascending (lower is better)
    eligible = []
    for r in rows:
        v = _ranking_metric(r, "ra_eff_vs_truth")
        if v is not None:
            eligible.append((v, id(r), r))
    eligible.sort(key=lambda t: (t[0], t[1]))
    for rank_idx, (_, _, r) in enumerate(eligible, start=1):
        r["rank_by_ra_eff"] = rank_idx

    # rank_by_composite_score: descending (higher is better).  Rows
    # missing either component (ra_eff or steered_true_jaccard) get a
    # blank rank — typically the ra_eff-only rows on which
    # compute-final-metrics never populated interface metrics.
    eligible = []
    for r in rows:
        v = _ranking_composite(r)
        if v is not None:
            eligible.append((v, id(r), r))
    # Sort descending (higher composite = better = lower rank)
    eligible.sort(key=lambda t: (-t[0], t[1]))
    for rank_idx, (_, _, r) in enumerate(eligible, start=1):
        r["rank_by_composite_score"] = rank_idx


# ───────────────────────────────────────────────────────────────────────
# Per-sequence aggregation across seeds
# ───────────────────────────────────────────────────────────────────────
#
# The pipeline operates on UNIQUE SEQUENCES, but each unique sequence is
# predicted with `num_seeds` independent Boltz seeds.  These per-seed
# predictions need to be collapsed into one summary record per unique
# sequence before contamination, ranking, or reversion decisions are
# made.  This block contains the helpers that do that.
#
# Aggregation rules (per user spec, P0 audit):
#   - continuous metrics → median across seeds, with MAD as spread
#   - 0/1 categoricals (e.g. receptor_intact) → majority vote, ≥ ceil(N/2)
#   - position-set categoricals (mutated_contact_positions) → per-position
#     majority: a position is in the aggregate iff ≥ ceil(N/2) seeds
#     report it as a mutated contact in their per-seed prediction
#
# Seed counts are also recorded so a downstream consumer can distinguish
# between e.g. a 3/3 pose_holds (consistent rescue) and a 2/3 pose_holds
# (one seed dissented) when picking between otherwise-similar designs.

def _agg_continuous(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    """Aggregate a list of per-seed values into (median, MAD, n_used).

    None / NaN / Inf entries are dropped.  MAD is the median of absolute
    deviations from the median; pairs naturally with median for small N
    (SD on N=3 is dominated by single outliers and gives wildly variable
    estimates run-to-run, MAD is robust).

    Returns (None, None, 0) if no usable values.
    """
    clean: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(f) or math.isinf(f):
            continue
        clean.append(f)
    if not clean:
        return (None, None, 0)
    med = statistics.median(clean)
    if len(clean) == 1:
        # MAD undefined for N=1; report 0.0 to mean "no spread visible"
        return (med, 0.0, 1)
    mad = statistics.median(abs(v - med) for v in clean)
    return (med, mad, len(clean))


def _agg_majority_binary(values: List[Optional[int]]) -> Tuple[Optional[int], int, int]:
    """Aggregate a list of per-seed 0/1 values into
    (majority_value, n_positive, n_used).

    Majority threshold is ≥ ceil(N/2).  None / unparseable entries are
    dropped from the count (tolerates partial failures).  Returns
    (None, 0, 0) if no usable values.
    """
    clean: List[int] = []
    for v in values:
        if v is None:
            continue
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if i not in (0, 1):
            continue
        clean.append(i)
    if not clean:
        return (None, 0, 0)
    n_used = len(clean)
    n_pos = sum(clean)
    threshold = (n_used + 1) // 2  # ceil(N/2)
    return (1 if n_pos >= threshold else 0, n_pos, n_used)


def _agg_majority_positions(
    per_seed_position_sets: List[List[int]],
) -> Tuple[List[int], Dict[int, int], int]:
    """Aggregate per-seed position sets into a per-position majority vote.

    A position is in the aggregate set iff ≥ ceil(N/2) seeds report it.
    Returns (aggregate_positions, position_to_seed_count, n_seeds_used).

    `position_to_seed_count` is exposed so the aggregator can record
    near-miss positions (e.g. a position that 1 of 3 seeds reported)
    as diagnostic context, not just the majority result.

    Empty input → ([], {}, 0).
    """
    if not per_seed_position_sets:
        return ([], {}, 0)
    n_seeds = len(per_seed_position_sets)
    threshold = (n_seeds + 1) // 2
    counts: Dict[int, int] = {}
    for ps in per_seed_position_sets:
        seen_in_this_seed = set()
        for p in ps:
            try:
                pi = int(p)
            except (TypeError, ValueError):
                continue
            if pi in seen_in_this_seed:
                continue
            seen_in_this_seed.add(pi)
            counts[pi] = counts.get(pi, 0) + 1
    aggregate = sorted(p for p, c in counts.items() if c >= threshold)
    return (aggregate, counts, n_seeds)


def _parse_position_csv(s: str) -> List[int]:
    """Parse a comma-separated string of 1-based positions into a list
    of ints.  Tolerates blanks and non-integer tokens.  Used to convert
    contact_residues / mutated_contact_positions CSV strings back into
    sets for aggregation."""
    if not s:
        return []
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            continue
    return out


# Continuous metrics that should be aggregated by median+MAD.  Mirrors
# the Block-B/C continuous columns from _aggregate_csv_fieldnames.
_AGG_CONTINUOUS_METRICS = (
    "ra_eff_vs_truth",
    "independent_receptor_rmsd",
    "independent_effector_rmsd",
    "wrong_jaccard",
    "true_jaccard",
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
    # 15Å iPSAE — same machinery, more permissive PAE cutoff.
    "ipsae_ab_15",
    "ipsae_ba_15",
    "ipsae_min_15",
    "actifptm",
    # Per-prediction native-PDB-dependent metrics (P0-29 / P0-38),
    # moved from compute_interface_metrics.py / run_biophysical_metrics.py
    # so they aggregate alongside the Boltz-derived metrics.
    "interface_plddt",
    "weighted_jaccard",
)

# 0/1 categorical metrics aggregated by majority.
_AGG_BINARY_METRICS = (
    "receptor_intact",
    # intact_core: per-prediction Kabsch-aligned receptor-core RMSD
    # < threshold flag.  Binary 0/1 → majority across seeds.
    "intact_core",
)

# Integer count metrics.  These get median + MAD too — they're
# discrete but ordered, and median across N=3 still makes sense.
_AGG_INTEGER_METRICS = (
    "total_mutations",
    "n_shared_wrong",
    "n_shared_true",
    "n_design_interface_residues",
    "n_contact_residues",
    "n_contacts_on_mutated_positions",
)

# Position-set columns (CSV strings of 1-based positions) aggregated
# by per-position majority.
_AGG_POSITION_SET_METRICS = (
    "mutated_contact_positions",
    "contact_residues",
)


def _aggregate_per_sequence(
    per_seed_rows: List[Dict],
    prefix: str,
) -> Dict:
    """Aggregate a list of per-seed rows (all sharing one unique
    sequence) into a single summary dict.

    `prefix` is "steered_" or "reverted_".  All metric column names
    in the per-seed rows are expected to start with this prefix; the
    aggregated output uses `<prefix><metric>_median`, `_mad`, `_n_used`
    for continuous; `<prefix><metric>_majority`, `_n_positive`,
    `_n_used` for binary; `<prefix><metric>_majority` (CSV string),
    `<prefix><metric>_per_seed_counts` (JSON map), `_n_used` for
    position sets.

    Identification fields (`design`, `pdb`, `cycle`, `pathway`) are
    pulled from the FIRST row.  Rows in the same sequence_group should
    share these by construction (same unique sequence → same design
    name minus the `_sN` suffix).  The aggregated `design` field is
    the bare name without the `_sN` suffix.

    Returns the aggregated dict.  Caller is responsible for merging
    aggregated steered + aggregated reverted records into one row.
    """
    if not per_seed_rows:
        return {}

    out: Dict[str, any] = {}

    # Continuous metrics → median + MAD
    for metric in _AGG_CONTINUOUS_METRICS + _AGG_INTEGER_METRICS:
        col = prefix + metric
        vals = [r.get(col) for r in per_seed_rows]
        med, mad, n_used = _agg_continuous(vals)
        out[f"{prefix}{metric}_median"] = "" if med is None else round(med, 4)
        out[f"{prefix}{metric}_mad"] = "" if mad is None else round(mad, 4)
        out[f"{prefix}{metric}_n_used"] = n_used

    # Binary metrics → majority vote
    for metric in _AGG_BINARY_METRICS:
        col = prefix + metric
        vals = [r.get(col) for r in per_seed_rows]
        majority, n_pos, n_used = _agg_majority_binary(vals)
        out[f"{prefix}{metric}_majority"] = "" if majority is None else majority
        out[f"{prefix}{metric}_n_positive"] = n_pos
        out[f"{prefix}{metric}_n_used"] = n_used

    # Position-set metrics → per-position majority
    for metric in _AGG_POSITION_SET_METRICS:
        col = prefix + metric
        position_sets = [_parse_position_csv(r.get(col, "")) for r in per_seed_rows]
        agg_positions, counts, n_used = _agg_majority_positions(position_sets)
        out[f"{prefix}{metric}_majority"] = ",".join(
            str(p) for p in agg_positions
        )
        # Per-position counts as a compact JSON map for diagnostics.
        # Empty → empty string (not "{}") so downstream readers can use
        # a simple truthiness test.
        out[f"{prefix}{metric}_per_seed_counts"] = (
            json.dumps({str(p): c for p, c in sorted(counts.items())})
            if counts else ""
        )
        out[f"{prefix}{metric}_n_used"] = n_used

    return out


def _classify_outcome(
    agg: Dict,
    verdict_counts: Dict[str, int],
    structural_ra_eff_cutoff: float = 5.0,
    contamination_gating_positions: Optional[Set[int]] = None,
) -> Tuple[str, str]:
    """Re-derive the reversion outcome from the AGGREGATED reverted
    metrics.  Mirrors classify_reversion_verdict but operates on the
    median / majority columns produced by _aggregate_per_sequence.

    Returns (outcome, reason).  Outcome is one of pose_holds,
    pose_collapses, new_contamination, no_reversion.

    no_reversion is returned when the aggregated record carries no
    reverted metrics at all (i.e. no seed in this group needed
    reversion: either cold-start skip_steering, or steering ran but
    no seed had contamination on mutated positions).

    The aggregated pose_holds outcome requires ALL of:
      1. The aggregated structural filter passes (intact_majority = 1
         AND median ra_eff < cutoff AND no position-majority contamination)
      2. At least ONE individual per-seed verdict says pose_holds
         (verdict_counts["pose_holds"] >= 1)

    Rationale for the any-seed gate (v8, replaces the earlier ceil(N/2)
    majority gate from Bug E): with `num_seeds = 3` and a per-seed
    pose_holds rate around 10-15% on this scaffold (measured from v7),
    majority-of-3 is architecturally unreachable even when several
    designs individually have one clean-pass seed.  The aggregated
    median/position-majority rules already exclude the "every seed
    failed differently" artefact that Bug E was originally aimed at:
    if no seed individually passes, pose_holds count == 0 and this
    gate short-circuits.  If exactly one seed passes and the other
    two fail in different ways, the aggregated position-majority
    check on `reverted_mutated_contact_positions_majority` only
    flags contamination when ≥ ceil(N/2) seeds contaminate at a
    shared position — so sporadic per-seed contamination at unrelated
    positions does NOT trigger it, which is what Bug E was worried
    about.  That concern is now handled by also requiring the
    per-seed failure distribution not to be a majority-failure:
    we additionally require pose_holds + no_data
    > each of (pose_collapses, new_contamination).
    In words: among seeds that produced a verdict, the "pass or no-op"
    count must exceed either of the failure counts.

    `verdict_counts` is the {verdict: count} dict produced by
    _per_seed_verdict_breakdown.  Keys: pose_holds, pose_collapses,
    new_contamination, no_data, clean_steered.
    """
    n_used = int(agg.get("reverted_ra_eff_vs_truth_n_used", 0) or 0)
    if n_used == 0:
        return ("no_reversion", "no reverted prediction available")

    nph = int(verdict_counts.get("pose_holds", 0))
    npc = int(verdict_counts.get("pose_collapses", 0))
    nnc = int(verdict_counts.get("new_contamination", 0))
    nnd = int(verdict_counts.get("no_data", 0))
    ncs = int(verdict_counts.get("clean_steered", 0))
    n_total_verdicts = nph + npc + nnc + nnd + ncs
    # Pose-passing-equivalent count: explicit per-seed pose_holds
    # PLUS clean_steered (steered prediction was structurally fine
    # AND had no contamination, so no reversion was needed — the
    # steered structure is the final verdict for that seed).
    nph_eff = nph + ncs

    # Structural filter on the aggregated reverted metrics.
    intact_majority = agg.get("reverted_receptor_intact_majority", "")
    if intact_majority != 1:
        return ("pose_collapses",
                f"reverted not intact by majority "
                f"({agg.get('reverted_receptor_intact_n_positive', 0)}/"
                f"{agg.get('reverted_receptor_intact_n_used', 0)} seeds intact)")

    ra_med = agg.get("reverted_ra_eff_vs_truth_median", "")
    if ra_med == "" or ra_med is None:
        return ("pose_collapses", "reverted_ra_eff_vs_truth median unavailable")
    try:
        ra_med_f = float(ra_med)
    except (TypeError, ValueError):
        return ("pose_collapses", "reverted_ra_eff_vs_truth median unparseable")
    if ra_med_f >= structural_ra_eff_cutoff:
        return ("pose_collapses",
                f"reverted ra_eff median {ra_med_f:.2f} ≥ "
                f"cutoff {structural_ra_eff_cutoff:.2f}")

    # Contamination check on the aggregated reverted contact set.
    # A position only appears in `reverted_mutated_contact_positions_majority`
    # if ≥ ceil(N/2) seeds individually contacted at that position, so
    # sporadic per-seed contamination at unrelated positions does not
    # trigger this — which is the Bug E "every seed failed differently"
    # artefact we still need to guard against.
    #
    # notes11 gating: Only positions inside the design region ∪ true
    # interface count as contamination, matching the pre-reversion
    # policy.  Positions outside the gated set (e.g. off-interface
    # "reaching bit" contacts) are acceptable by construction and
    # should not downgrade the aggregated verdict.
    contam_majority = agg.get(
        "reverted_mutated_contact_positions_majority", "")
    if contam_majority:
        # Parse the majority list and filter to gated positions only.
        all_positions = [int(p) for p in str(contam_majority).split(",")
                         if p.strip().lstrip("-").isdigit()]
        if contamination_gating_positions is not None:
            gated_positions = [p for p in all_positions
                               if p in contamination_gating_positions]
            ungated_positions = [p for p in all_positions
                                 if p not in contamination_gating_positions]
        else:
            gated_positions = all_positions
            ungated_positions = []

        if gated_positions:
            reason = (f"reverted pose has mutated contact(s) by majority "
                      f"at gated position(s) "
                      f"(design region ∪ true interface): {gated_positions}")
            if ungated_positions:
                reason += (f"; also touching ungated positions "
                           f"{ungated_positions} (acceptable)")
            return ("new_contamination", reason)
        # else: all majority contacts are at ungated positions — fall
        # through to the pose_holds path

    # Any-seed gate: at least one seed must individually pass —
    # either as an explicit pose_holds verdict OR as clean_steered
    # (no reversion needed because steered prediction had zero
    # contamination on mutated positions).  This is the weaker
    # gate that replaces the Bug E majority-of-N gate; see
    # docstring for rationale.
    if nph_eff < 1:
        # Zero seeds passed individually.  Choose downgraded verdict
        # based on plurality of failure modes; ties favour
        # pose_collapses as the more conservative call.
        if nnc > npc:
            downgraded = "new_contamination"
        else:
            downgraded = "pose_collapses"
        return (downgraded,
                f"aggregated structural/position rule said pose_holds but "
                f"0/{n_total_verdicts} seeds individually agree "
                f"(pose_collapses={npc}, new_contamination={nnc}, "
                f"no_data={nnd}, clean_steered={ncs}); "
                f"downgrading to {downgraded}")

    # Bug-E residual guard: if the minority-of-failures is still the
    # majority of seeds in either direction, that's a weak signal —
    # we need the number of seeds that didn't individually fail
    # (pose_holds + clean_steered + no_data) to be at least as large
    # as each failure count.  This prevents "1 passes, 2 fail the
    # same way (but at different positions so no position reaches
    # majority)" from being called pose_holds.
    non_failing = nph_eff + nnd
    if non_failing < max(npc, nnc):
        if nnc > npc:
            downgraded = "new_contamination"
        else:
            downgraded = "pose_collapses"
        return (downgraded,
                f"{nph_eff}/{n_total_verdicts} seed(s) individually pass "
                f"({nph} pose_holds + {ncs} clean_steered) "
                f"but failure modes dominate (pose_collapses={npc}, "
                f"new_contamination={nnc}, no_data={nnd}); "
                f"downgrading to {downgraded}")

    return ("pose_holds",
            f"reverted pose holds "
            f"({agg.get('reverted_receptor_intact_n_positive', 0)}/"
            f"{agg.get('reverted_receptor_intact_n_used', 0)} seeds intact, "
            f"median ra_eff {ra_med_f:.2f} Å, "
            f"{nph_eff}/{n_total_verdicts} seeds individually pass "
            f"[{nph} pose_holds + {ncs} clean_steered])")


def _per_seed_verdict_breakdown(
    per_seed_reverted_rows: List[Dict],
    structural_ra_eff_cutoff: float = 5.0,
) -> Dict[str, int]:
    """Run the SINGLE-SEED verdict logic on each reverted per-seed row
    and return {verdict: count}.  Counts seeds in five buckets:
    pose_holds, pose_collapses, new_contamination, no_data,
    clean_steered.  Feeds into n_pass (= pose_holds + clean_steered)
    and into the aggregate outcome classifier.

    The authoritative per-seed verdict is the `reversion_verdict`
    column, which was set by `classify_reversion_verdict` in
    cmd_harvest_reversions on the fresh reversion_results.json data
    (before the aggregate CSV blanked contact columns for non-pose_holds
    rows in _populate_reverted_mutations).  We prefer that column when
    present and fall back to re-deriving the verdict from the raw
    reverted_* columns only when it isn't.

    The `clean_steered` bucket (added later) covers the case where
    the steered prediction had ZERO contamination on the mutated
    positions, so the reversion pass was correctly skipped — there
    was nothing to revert.  These rows have a blank reversion_verdict
    AND blank reverted_* columns AND
    steered_n_contacts_on_mutated_positions == 0.  Without this
    bucket they were misclassified as no_data, which incorrectly
    demoted otherwise-strong designs in the per-sequence triage
    (notes12 / chat 'rank-1 issue').  clean_steered is treated as
    pose-passing-equivalent in _classify_outcome and in
    the cross-sequence tier classifier.

    Without the verdict-column preference for the explicit-verdict
    cases, rows whose reverted_mutated_contact_positions column was
    blanked for display would be misclassified here as pose_holds,
    inflating the pose_holds count.
    """
    counts = {
        "pose_holds": 0,
        "pose_collapses": 0,
        "new_contamination": 0,
        "no_data": 0,
        "clean_steered": 0,
    }
    for r in per_seed_reverted_rows:
        # Prefer the authoritative verdict column when present.  Map
        # any pose_holds-but-dropped states (e.g. pose_holds_not_intact,
        # missing_verdict) to pose_collapses for the count.
        authoritative = (r.get("reversion_verdict") or "").strip()
        if authoritative in ("pose_holds", "pose_collapses",
                             "new_contamination"):
            counts[authoritative] += 1
            continue
        if authoritative in ("pose_holds_not_intact", "missing_verdict",
                             "unknown"):
            counts["pose_collapses"] += 1
            continue
        if authoritative != "":
            # Unknown non-empty verdict string — surface as no_data so
            # it's visible but doesn't silently get bucketed.
            counts["no_data"] += 1
            continue

        # Verdict column is blank.  Two sub-cases:
        #
        # (i)  Steered prediction had ZERO contamination on mutated
        #      positions, so reversion was skipped intentionally.
        #      The steered structure is the final result for this
        #      seed — count it as clean_steered (pass-equivalent).
        #
        # (ii) Reversion was attempted but the row is missing the
        #      reverted_* fields needed to classify it.  Re-derive
        #      from raw reverted_* columns; fall back to no_data
        #      if anything is missing or unparseable.
        steered_n_contacts_str = r.get(
            "steered_n_contacts_on_mutated_positions", "")
        try:
            steered_n_contacts = int(steered_n_contacts_str)
        except (TypeError, ValueError):
            steered_n_contacts = None

        # Heuristic for (i): steered row is intact, has zero
        # contamination, and the reverted_* fields are entirely
        # blank (no attempt was made).  Require the steered row
        # itself to also be structurally OK (intact + ra_eff in
        # range) — otherwise a "no contamination" steered miss
        # would silently count as a pass.
        if steered_n_contacts == 0:
            try:
                steered_intact = int(r.get("steered_receptor_intact", ""))
            except (TypeError, ValueError):
                steered_intact = None
            try:
                steered_ra = float(r.get("steered_ra_eff_vs_truth", ""))
            except (TypeError, ValueError):
                steered_ra = None
            reverted_attempted = any(
                (r.get(k) or "") != ""
                for k in ("reverted_ra_eff_vs_truth",
                          "reverted_receptor_intact",
                          "reverted_total_mutations")
            )
            if (steered_intact == 1
                    and steered_ra is not None
                    and not math.isnan(steered_ra)
                    and not math.isinf(steered_ra)
                    and steered_ra < structural_ra_eff_cutoff
                    and not reverted_attempted):
                counts["clean_steered"] += 1
                continue
            # Otherwise fall through to the (ii) re-derivation path.

        # Fall back to re-deriving from raw columns for rows that
        # never went through reversion (verdict column blank).
        intact = r.get("reverted_receptor_intact", "")
        try:
            intact_i = int(intact)
        except (TypeError, ValueError):
            counts["no_data"] += 1
            continue
        if intact_i != 1:
            counts["pose_collapses"] += 1
            continue
        try:
            ra = float(r.get("reverted_ra_eff_vs_truth", ""))
        except (TypeError, ValueError):
            counts["no_data"] += 1
            continue
        if math.isnan(ra) or math.isinf(ra):
            counts["no_data"] += 1
            continue
        if ra >= structural_ra_eff_cutoff:
            counts["pose_collapses"] += 1
            continue
        contam_str = r.get("reverted_mutated_contact_positions", "") or ""
        contam = _parse_position_csv(contam_str)
        if contam:
            counts["new_contamination"] += 1
            continue
        counts["pose_holds"] += 1
    return counts


def _aggregate_csv_fieldnames(rows: List[Dict]) -> List[str]:
    """Return the column order for all_results_multicycle.csv (and the
    joined all_results_multicycle_with_metrics.csv).  Three logical
    blocks (A: identification, B: steering, C: reversion) plus a tail
    of selection / ranks.  Any row keys not in this list get appended
    (sorted) at the very end so unexpected fields aren't silently
    dropped.

    The Block B `steered_*` confidence columns are populated by
    compute-final-metrics; they're listed here so that whether or not
    compute-final-metrics has run, the column order in the produced
    CSV is the same and consumers can rely on it.
    """
    preferred = [
        # ── A. Identification ────────────────────────────────────
        "cycle", "pathway", "design",
        # P0 audit (multi-seed): per-seed grouping metadata.  Two
        # rows with the same sequence_group differ only by Boltz
        # seed; per-sequence aggregation collapses them.
        "sequence_group", "seed_index",
        "status", "pdb",
        # ── B. Steering details ──────────────────────────────────
        # Metadata
        "steered_total_mutations",
        "steered_mutations_chimerax",
        "steered_mutations_aa",
        # Structural RMSDs vs truth (the steered prediction's values,
        # NEVER overwritten by reversion)
        "steered_ra_eff_vs_truth",
        "steered_independent_receptor_rmsd",
        "steered_independent_effector_rmsd",
        "steered_receptor_intact",
        # Interface-similarity
        "steered_wrong_jaccard",
        "steered_n_shared_wrong",
        "steered_true_jaccard",
        "steered_n_shared_true",
        "steered_n_design_interface_residues",
        # Confidence metrics on the steered prediction (populated by
        # compute-final-metrics for rows that pass its filter)
        "steered_avg_plddt",
        "steered_complex_plddt",
        "steered_ptm",
        "steered_iptm",
        "steered_pae_mean",
        "steered_ipae",
        "steered_pae_pass_frac",
        "steered_pae_cutoff_used",
        "steered_ipsae_ab",
        "steered_ipsae_ba",
        "steered_ipsae_min",
        "steered_actifptm",
        # Contact analysis + contamination check.  Any mutated residue
        # that contacts the effector invalidates the metrics, so the
        # count (n_contacts_on_mutated_positions) and the list
        # (mutated_contact_positions) together identify contamination.
        "steered_n_contact_residues",
        "steered_contact_residues",
        "steered_n_contacts_on_mutated_positions",
        "steered_mutated_contact_positions",
        "steered_contact_cutoff_used",
        # Confidence flag + error column
        "steered_confidence_flag",
        "steered_error",
        # ── C. Reversion details ─────────────────────────────────
        "reversion_verdict",
        "reversion_dropped",
        "reverted_sequence_used_as_parent",
        # Reverted metadata
        "reverted_total_mutations",
        "reverted_mutations_chimerax",
        "reverted_mutations_aa",
        # Reverted structural RMSDs vs truth
        "reverted_ra_eff_vs_truth",
        "reverted_independent_receptor_rmsd",
        "reverted_independent_effector_rmsd",
        "reverted_receptor_intact",
        # Reverted contamination check: any remaining mutation
        # contacting the effector in the reverted prediction.
        # Should be empty for pose_holds by definition of the verdict.
        "reverted_mutated_contact_positions",
        # Reverted contact analysis (from harvest-reversions)
        "reverted_n_contact_residues",
        "reverted_contact_residues",
        "reverted_contact_cutoff_used",
        # Reverted confidence suite (populated for pose_holds rows
        # by harvest-reversions; blank otherwise)
        "reverted_avg_plddt",
        "reverted_complex_plddt",
        "reverted_ptm",
        "reverted_iptm",
        "reverted_pae_mean",
        "reverted_ipae",
        "reverted_pae_pass_frac",
        "reverted_ipsae_ab",
        "reverted_ipsae_ba",
        "reverted_ipsae_min",
        "reverted_actifptm",
        # ── End. Selection / pathway-internal / ranks ────────────
        "min_dist_to_upstream",
        "selected_for_next_cycle",
        "rank_by_ra_eff",
        "rank_by_composite_score",
    ]
    all_keys = {k for r in rows for k in r.keys()}
    extra = sorted(k for k in all_keys
                   if k not in preferred and k != "parent_pathway")
    # parent_pathway: silently dropped (derivable from pathway)
    return [k for k in preferred if k in all_keys] + extra


# ───────────────────────────────────────────────────────────────────────
# Subcommand: aggregate
# ───────────────────────────────────────────────────────────────────────
def cmd_aggregate(args: argparse.Namespace) -> int:
    """Walk the entire experiment tree, build pathways.json + a flat
    all_results_multicycle.csv, and report the bell-curve summary."""
    experiment_root = args.experiment_root.resolve()
    if not experiment_root.is_dir():
        print(f"ERROR: {experiment_root} not a directory", file=sys.stderr)
        return 2

    # Discover every cycle dir and every pathway under it.
    # The submit script uses two different layouts depending on
    # --n-cycles:
    #   --n-cycles 1: cycle-0 files (plan.json, steered_results.csv,
    #       steered/, initial_prediction.pdb) live directly in the
    #       experiment root.  There is no cycle_0/ subdirectory.
    #   --n-cycles 2+: cycle-0 files live under experiment_root/cycle_0/
    #       and additional cycle_N/ subdirectories are created as the
    #       chain runs.
    # Detect which layout we're in by checking for the canonical
    # cycle-0 artefact (plan.json) in each location.
    pathways: List[Dict] = []
    nested_cycle_0 = experiment_root / "cycle_0"
    if (nested_cycle_0 / "plan.json").exists():
        cycle_0_dir = nested_cycle_0
        print(f"  cycle_0 layout: nested ({cycle_0_dir})")
    elif (experiment_root / "plan.json").exists():
        cycle_0_dir = experiment_root
        print(f"  cycle_0 layout: flat ({cycle_0_dir})")
    else:
        print(f"ERROR: no cycle_0 plan.json under {experiment_root} "
              f"(checked {nested_cycle_0}/plan.json and "
              f"{experiment_root}/plan.json)", file=sys.stderr)
        return 2

    # Cycle 0 — special, the single-cycle script writes here
    plan0 = json.loads((cycle_0_dir / "plan.json").read_text())
    pathways.append({
        "label": "cycle_0",
        "cycle": 0,
        "workdir": str(cycle_0_dir),
        "n_designs": len(plan0.get("designs", [])),
    })

    # Cycles 1+ — pathway_<label> subdirs under cycle_N/
    for cycle_dir in sorted(experiment_root.glob("cycle_*")):
        if not cycle_dir.is_dir():
            continue   # skip e.g. cycle_statistics.csv which the glob also matches
        if cycle_dir.name == "cycle_0":
            continue
        cycle_num = int(cycle_dir.name.split("_")[1])
        for pdir in sorted(cycle_dir.glob("pathway_*")):
            label = pdir.name[len("pathway_"):]
            pj = pdir / "plan.json"
            if not pj.exists():
                continue
            pl = json.loads(pj.read_text())
            pathways.append({
                "label": label,
                "cycle": cycle_num,
                "workdir": str(pdir),
                "parent_pathway": pl.get("parent_pathway"),
                "n_designs": len(pl.get("designs", [])),
                "exhausted": pl.get("exhausted", False),
            })

    (experiment_root / "pathways.json").write_text(
        json.dumps(pathways, indent=2)
    )

    # Look up the wild-type baseline ra_eff so we can compute a
    # consistent delta_receptor_aligned_effector_rmsd column for
    # cycle 1+ rows (cycle 0 already has it from steered_results.csv,
    # using the same baseline).
    cycle0_plan = json.loads((cycle_0_dir / "plan.json").read_text())
    wt_ra_eff = cycle0_plan.get("initial_receptor_aligned_effector_rmsd")
    # Prediction receptor chain ID, used to build the
    # mutations_chimerax column.  Default "A" if the plan doesn't
    # record it (old runs).
    pred_rec_chain_for_spec = cycle0_plan.get("pred_receptor_chain", "A")

    # Cycle 0's kickoff (if it ran) writes pose distances + selection
    # info into cycle_0/passing.json.  Index by design index so we
    # can merge it onto the steered_results.csv rows.
    cycle0_passing_path = cycle_0_dir / "passing.json"
    cycle0_extra_by_design: Dict[str, Dict] = {}
    if cycle0_passing_path.exists():
        cp = json.loads(cycle0_passing_path.read_text())
        sel_idxs_c0 = {s["design_idx"] for s in cp.get("selected", [])}
        for c in cp.get("all_candidates", []):
            cycle0_extra_by_design[c["design"]] = {
                "min_dist_to_upstream": c.get("min_dist_to_upstream"),
                "selected_for_next_cycle": 1 if c["design_idx"] in sel_idxs_c0 else 0,
                # Reversion fields — blank for non-reverted designs.
                # Note: steered_* base columns are preserved in the
                # row's existing receptor_aligned_effector_rmsd /
                # independent_*_rmsd fields and will be renamed by
                # _translate_aggregate_row at write time.  Reverted
                # values live only in dedicated reverted_* keys.
                "reversion_verdict": c.get("reversion_verdict", ""),
                "reversion_dropped": c.get("reversion_dropped", ""),
                "reverted_sequence_used_as_parent": c.get(
                    "reverted_sequence_used_as_parent", ""
                ),
                "reverted_ra_eff_vs_truth": c.get("reverted_ra_eff_vs_truth", ""),
                "reverted_independent_receptor_rmsd": c.get(
                    "reverted_independent_receptor_rmsd", ""
                ),
                "reverted_independent_effector_rmsd": c.get(
                    "reverted_independent_effector_rmsd", ""
                ),                **{k: c.get(k, "") for k in _REVERTED_CONFIDENCE_FIELDS},
            }

    # Flat CSV: every prediction from every cycle, ranked by ra_eff vs truth
    rows = []
    for p in pathways:
        wd = Path(p["workdir"])
        # Cycle 0 has its own steered_results.csv from the single-cycle script
        if p["label"] == "cycle_0":
            csv_path = wd / "steered_results.csv"
            if not csv_path.exists():
                # Skip_steering case with a pre-patch driver: no CSV
                # was written, but the cold-start prediction + plan.json
                # still exist.  Reconstruct a single initial row from
                # them so the comparison table gets SOMETHING for this
                # complex.  This path fires for existing 7B1I-style
                # workdirs that were aggregated before the patch.
                initial_pdb = wd / "initial_prediction.pdb"
                if (cycle_0_dir / "plan.json").exists() and initial_pdb.exists():
                    print("  cycle_0: no steered_results.csv — "
                          "reconstructing initial row from plan.json + "
                          "initial_prediction.pdb")
                    plan_data = cycle0_plan
                    initial_ra = plan_data.get(
                        "initial_receptor_aligned_effector_rmsd", "")
                    initial_ind_rec = plan_data.get(
                        "initial_independent_receptor_rmsd", "")
                    initial_ind_eff = plan_data.get(
                        "initial_independent_effector_rmsd", "")
                    try:
                        intact_flag = (
                            1 if (float(initial_ind_rec) <= 5.0
                                  and float(initial_ind_eff) <= 5.0)
                            else 0
                        )
                    except (TypeError, ValueError):
                        intact_flag = ""
                    # Interface fields: only present if interface
                    # analysis ran (not the case for all skip types)
                    initial_wrong = plan_data.get("initial_wrong_interface_idx") or []
                    true_set = plan_data.get("true_interface_idx") or []
                    # P0 jaccard-empty-wrong fix: only the truth set is
                    # required.  Empty initial_wrong gives jaccard = 0,
                    # not undefined.  See same fix in
                    # boltz2_negative_steering.py _write_initial_only_csv.
                    if true_set:
                        n_iface = len(initial_wrong)
                        n_shared_true = len(set(initial_wrong) & set(true_set))
                        true_j = n_shared_true / max(
                            len(set(initial_wrong) | set(true_set)), 1)
                        wrong_j = 1.0 if initial_wrong else float("nan")
                    else:
                        n_iface = ""
                        n_shared_true = ""
                        true_j = ""
                        wrong_j = ""
                    reconstructed = {
                        "rank": "0",
                        "design": "initial",
                        "total_mutations": "0",
                        "receptor_aligned_effector_rmsd": str(initial_ra),
                        "delta_receptor_aligned_effector_rmsd": "0.0",
                        "independent_receptor_rmsd": str(initial_ind_rec),
                        "independent_effector_rmsd": str(initial_ind_eff),
                        "receptor_intact": str(intact_flag),
                        "wrong_jaccard": str(wrong_j),
                        "n_shared_wrong": str(n_iface),
                        "true_jaccard": str(true_j),
                        "n_shared_true": str(n_shared_true),
                        "n_design_interface_residues": str(n_iface),
                        "status": "initial",
                        "pdb": str(initial_pdb.resolve()),
                        "pathway": "cycle_0",
                        "parent_pathway": "",
                        "cycle": 0,
                        # P0 audit: initial baseline is a single
                        # prediction (no steering, no multi-seed),
                        # so it gets its own pseudo-group with
                        # seed_index 0 — keeps grouping logic uniform.
                        "sequence_group": "initial",
                        "seed_index": 0,
                        "min_dist_to_upstream": "",
                        "selected_for_next_cycle": "",
                        "mutations_chimerax": "",
                        "mutations_aa": "",
                    }
                    rows.append(reconstructed)
                else:
                    print("  cycle_0: no steered_results.csv AND no "
                          "initial_prediction.pdb — skipping")
                continue
            with open(csv_path) as f:
                for r in csv.DictReader(f):
                    r["pathway"] = "cycle_0"
                    r["parent_pathway"] = ""
                    r["cycle"] = 0
                    # Merge in min_dist + selected info from kickoff,
                    # if available.  The 'initial' baseline row has
                    # no design entry so it stays blank.
                    extra = cycle0_extra_by_design.get(r.get("design", ""), {})
                    r["min_dist_to_upstream"] = extra.get("min_dist_to_upstream", "")
                    r["selected_for_next_cycle"] = extra.get("selected_for_next_cycle", "")
                    # Reversion fields — blank for non-reverted designs
                    for k in (
                        "reversion_verdict",
                        "reversion_dropped",
                        "reverted_sequence_used_as_parent",
                        "reverted_ra_eff_vs_truth",
                        "reverted_independent_receptor_rmsd",
                        "reverted_independent_effector_rmsd",                        *_REVERTED_CONFIDENCE_FIELDS,
                    ):
                        r[k] = extra.get(k, "")
                    # Mutation strings: parse mutations.tsv from the
                    # design dir (derived from the pdb path).  The
                    # 'initial' baseline row(s) have no mutations so
                    # both columns stay empty.  Multi-seed cold-start
                    # produces design names "initial", "initial_s1",
                    # "initial_s2", ... — all are baseline rows with
                    # zero mutations and no mutations.tsv (notes12).
                    design_name = r.get("design", "")
                    if design_name == "initial" or design_name.startswith("initial_s"):
                        r["mutations_chimerax"] = ""
                        r["mutations_aa"] = ""
                    else:
                        pdb_col = r.get("pdb", "")
                        if pdb_col:
                            mut_tsv = Path(pdb_col).parent / "mutations.tsv"
                            muts = read_mutations_tsv_full(mut_tsv)
                        else:
                            muts = []
                        cx, aa = format_mutation_strings(
                            muts, chain_id=pred_rec_chain_for_spec,
                        )
                        r["mutations_chimerax"] = cx
                        r["mutations_aa"] = aa
                    rows.append(r)
            continue

        # Cycles 1+: read passing.json (which contains all candidates with metrics)
        passing_path = wd / "passing.json"
        if not passing_path.exists():
            continue
        passing = json.loads(passing_path.read_text())
        sel_idxs = {s["design_idx"] for s in passing.get("selected", [])}
        # p["label"] is the PARENT pathway (e.g. "c0d05"); each candidate
        # in this directory becomes a LEAF pathway by appending its
        # (cycle, design_idx) to the parent.
        parent_label = p["label"]
        for c in passing.get("all_candidates", []):
            leaf_label = make_pathway_label(parent_label, p["cycle"], c["design_idx"])
            ra_eff = c["receptor_aligned_effector_rmsd_vs_truth"]
            # delta vs the wild-type baseline (positive = improvement,
            # i.e. effector pulled closer to the true site)
            delta = (wt_ra_eff - ra_eff) if wt_ra_eff is not None else ""

            # The full result.json content from compute-distances —
            # contains every per-design metric we want to surface.
            # Older runs (pre-result-passthrough fix) won't have this
            # embedded; fall back to reading result.json directly from
            # the design directory so the aggregator is self-healing.
            result_full = c.get("result")
            if not result_full:
                rj_path = Path(c["pdb"]).parent / "result.json"
                if rj_path.exists():
                    try:
                        result_full = json.loads(rj_path.read_text())
                    except Exception:
                        result_full = {}
                else:
                    result_full = {}

            row = {
                "pathway": leaf_label,
                "parent_pathway": parent_label,
                "cycle": p["cycle"],
                "design": c["design"],
                # P0 audit (multi-seed): expose sequence_group +
                # seed_index in the per-seed CSV so the per-sequence
                # aggregator can group on them without parsing the
                # _sN suffix in the design name.
                "sequence_group": c.get("sequence_group", ""),
                "seed_index": c.get("seed_index", ""),
                # Prefer total_mutations; fall back to legacy
                # n_mutations key for old runs (where it meant
                # per-cycle mutation count, not cumulative).
                "total_mutations": result_full.get(
                    "total_mutations",
                    result_full.get("n_mutations", ""),
                ),
                "receptor_aligned_effector_rmsd": ra_eff,
                "delta_receptor_aligned_effector_rmsd": delta,
                "independent_receptor_rmsd": c["independent_receptor_rmsd"],
                "independent_effector_rmsd": c["independent_effector_rmsd"],
                "receptor_intact": 1 if c["receptor_intact"] else 0,
                "wrong_jaccard": result_full.get("wrong_jaccard", ""),
                "n_shared_wrong": result_full.get("n_shared_wrong", ""),
                "true_jaccard": result_full.get("true_jaccard", ""),
                "n_shared_true": result_full.get("n_shared_true", ""),
                "n_design_interface_residues": result_full.get(
                    "n_design_interface_residues", ""),
                "min_dist_to_upstream": c["min_dist_to_upstream"],
                "selected_for_next_cycle": 1 if c["design_idx"] in sel_idxs else 0,
                "status": result_full.get("status", ""),
                "pdb": c["pdb"],
                # Reversion columns — populated when iterate-collect-
                # finalize ran a reversion pass over this design.  Rows
                # that were never reverted (non-contaminated) leave
                # these blank.  Rows that were dropped
                # (pose_collapses, new_contamination) carry
                # reversion_dropped=1 and DO NOT flow to next cycle.
                # The base columns above hold the steered values
                # untouched; reverted values live ONLY in the
                # reverted_* keys.
                "reversion_verdict": c.get("reversion_verdict", ""),
                "reversion_dropped": c.get("reversion_dropped", ""),
                "reverted_sequence_used_as_parent": c.get(
                    "reverted_sequence_used_as_parent", ""
                ),
                "reverted_ra_eff_vs_truth": c.get("reverted_ra_eff_vs_truth", ""),
                "reverted_independent_receptor_rmsd": c.get(
                    "reverted_independent_receptor_rmsd", ""
                ),
                "reverted_independent_effector_rmsd": c.get(
                    "reverted_independent_effector_rmsd", ""
                ),                **{k: c.get(k, "") for k in _REVERTED_CONFIDENCE_FIELDS},
            }
            # Cumulative mutations walked up the ancestry chain.
            cum_muts = read_cumulative_mutations(experiment_root, leaf_label)
            cx, aa = format_mutation_strings(
                cum_muts, chain_id=pred_rec_chain_for_spec,
            )
            row["mutations_chimerax"] = cx
            row["mutations_aa"] = aa
            rows.append(row)

    out_csv = experiment_root / "all_results_multicycle.csv"
    if rows:
        # ── Translate internal dict keys to the new clean CSV schema ──
        # Internal keys preserve historical names so upstream code keeps
        # working.  Here at the write boundary we rename to a clean
        # schema with consistent steered_/reverted_ prefixes, drop
        # redundant columns, and add new derived columns.
        rows = [_translate_aggregate_row(r) for r in rows]

        # ── Compute reverted_mutations_chimerax / _aa / _total for ──
        # pose_holds rows: cumulative mutation set minus the positions
        # that were reverted.  All other rows leave these blank.
        # The reversion data lives in cycle_N/(pathway_<label>/)? 
        # reversion_results.json keyed by pathway label.
        _populate_reverted_mutations(rows, experiment_root)

        # ── Compute unified ranks: for each ranking-eligible row, ──
        # use the canonical metric source (steered or reverted) per
        # the eligibility rules.  Both ranks are 1-based; rank 1 is
        # best.  Blank for ineligible rows.
        _compute_unified_ranks(rows)

        # ── Sort rows: cycle ascending, then initial baseline first ──
        # within cycle 0, then by rank_by_composite_score ascending
        # (blank ranks at the bottom).  P0.4: this sort key was
        # rank_by_ipsae_min; since that column is dropped, we key on
        # the new primary ranker instead.
        def _sort_key(r):
            try:
                cyc = int(r.get("cycle", 0) or 0)
            except (ValueError, TypeError):
                cyc = 0
            is_initial = 0 if r.get("design") == "initial" else 1
            rk = r.get("rank_by_composite_score", "")
            try:
                rk_num = int(rk) if rk != "" else 10**9
            except (ValueError, TypeError):
                rk_num = 10**9
            return (cyc, is_initial, rk_num)
        rows.sort(key=_sort_key)

        # ── Write with the new column order ──
        fieldnames = _aggregate_csv_fieldnames(rows)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"Wrote {out_csv} ({len(rows)} rows)")
    print(f"Wrote {experiment_root / 'pathways.json'} ({len(pathways)} pathways)")

    stats_path = experiment_root / "cycle_statistics.csv"
    if stats_path.exists():
        print(f"Bell-curve data: {stats_path}")

    return 0


# ───────────────────────────────────────────────────────────────────────
# compute-final-metrics: run compute_metrics.py on every design in
# all_results_multicycle.csv that passes a final ranking filter.
# ───────────────────────────────────────────────────────────────────────
def _count_ca_per_chain(pdb_path: Path) -> "Dict[str, int]":
    """Count Cα atoms per chain ID from a PDB file.  Stdlib only —
    keeps this subcommand runnable outside the numpy/gemmi container."""
    counts: Dict[str, int] = {}
    try:
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                if line[12:16].strip() != "CA":
                    continue
                chain_id = line[21:22].strip() or " "
                counts[chain_id] = counts.get(chain_id, 0) + 1
    except Exception as e:
        print(f"  WARNING: failed to count CA atoms in {pdb_path}: {e}",
              file=sys.stderr)
    return counts


def _float_or_none(x) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _locate_boltz_sidecar_pdb(prediction_pdb: Path) -> "Optional[Path]":
    """Find the original boltz-output PDB whose sidecar files
    (confidence_*.json, pae_*.npz, plddt_*.npz) live next to it.

    The steering pipeline copies Boltz's chosen PDB out to
    `<design>/prediction.pdb`, but the confidence / PAE / pLDDT
    sidecar files stay nested under Boltz's own output layout
    (e.g. `<design>/predictions/<stem>/pdb/<stem>_model_0.pdb`
    with `confidence_<stem>_model_0.json` alongside it).  The
    sidecar files are keyed off the ORIGINAL stem, not the string
    "prediction", so compute_metrics.py's parser only finds them
    if we point it at the nested directory.

    For the cold-start prediction (`initial_prediction.pdb` in the
    experiment root), the Boltz output lives in a sibling `initial/`
    directory — not in a design subdirectory — so we handle that
    case specially to avoid globbing the entire experiment tree and
    picking up unrelated design PDBs.

    Strategy: recursively glob the starting directory for PDBs,
    ignore the flat `prediction.pdb` copy itself, and among the
    rest prefer one that has a sibling `confidence_<stem>.json`
    (or that sits next to a `confidence/` sibling directory
    containing one).  Fall back to any nested PDB.  Return None
    if nothing else is found — callers should then fall back to
    the prediction.pdb's own parent.
    """
    # Special case: the cold-start baseline written by
    # boltz2_negative_steering.py as `<experiment_root>/initial_prediction.pdb`.
    # Its sidecar files live in `<experiment_root>/initial/...`,
    # not alongside prediction.pdb.
    #
    # Multi-seed cold-start (notes12) writes additional PDBs as
    # `<experiment_root>/initial_prediction_s<N>.pdb` (N=1, 2, ...)
    # with sidecar files under `<experiment_root>/initial_s<N>/`.
    # Route each to its own seed dir to avoid globbing siblings.
    if prediction_pdb.name == "initial_prediction.pdb":
        experiment_root = prediction_pdb.parent
        initial_dir = experiment_root / "initial"
        if initial_dir.is_dir():
            design_dir = initial_dir
        else:
            design_dir = experiment_root
    elif (prediction_pdb.name.startswith("initial_prediction_s")
          and prediction_pdb.name.endswith(".pdb")):
        experiment_root = prediction_pdb.parent
        # Strip "initial_prediction_" prefix and ".pdb" suffix to get "sN"
        suffix = prediction_pdb.name[len("initial_prediction_"):-len(".pdb")]
        seed_dir = experiment_root / f"initial_{suffix}"
        if seed_dir.is_dir():
            design_dir = seed_dir
        else:
            design_dir = experiment_root
    else:
        design_dir = prediction_pdb.parent
    candidates: List[Path] = []
    for p in design_dir.rglob("*.pdb"):
        if p.resolve() == prediction_pdb.resolve():
            continue
        if "msa" in p.parts:
            continue  # defensive: skip cached reference structures
        candidates.append(p)
    if not candidates:
        return None

    def _has_confidence(pdb: Path) -> bool:
        stem = pdb.stem
        d = pdb.parent
        # Same-dir sidecar
        if (d / f"confidence_{stem}.json").exists():
            return True
        # Sibling `confidence/` directory (Boltz's layered layout)
        if (d.parent / "confidence" / f"confidence_{stem}.json").exists():
            return True
        return False

    with_conf = [p for p in candidates if _has_confidence(p)]
    if with_conf:
        # Prefer model_0 / rank_0 if present
        with_conf.sort(key=lambda p: (
            "model_0" not in p.name and "rank_0" not in p.name,
            str(p),
        ))
        return with_conf[0]

    candidates.sort(key=lambda p: (
        "model_0" not in p.name and "rank_0" not in p.name,
        str(p),
    ))
    return candidates[0]


# ───────────────────────────────────────────────────────────────────────
# Subcommand: aggregate-per-sequence
# ───────────────────────────────────────────────────────────────────────
def cmd_aggregate_per_sequence(args: argparse.Namespace) -> int:
    """Collapse per-seed rows into one aggregated row per unique sequence.

    Reads the per-seed CSV (all_results_multicycle_with_metrics.csv if
    compute-final-metrics has run, else all_results_multicycle.csv),
    groups rows by `sequence_group`, and writes:

      raw_per_seed_results.csv         — copy of the input, untouched
                                         (renamed for clarity)
      aggregated_results.csv           — one row per unique sequence,
                                         median + MAD on continuous,
                                         majority on categorical, plus
                                         the verdict re-derived from the
                                         aggregated reverted record and
                                         per-seed verdict counts

    Aggregation rules (per P0 audit decisions):
      - continuous metrics      → median + MAD across seeds
      - 0/1 categoricals        → majority vote, ≥ ceil(N/2)
      - position-set categoricals → per-position majority
      - canonical seed (for `pdb`):
          * odd N  → seed at the median ra_eff
          * even N → seed with the WORSE (higher) ra_eff among the
                     two middle seeds, plus a warning column
          * N=1    → that seed, no warning

    Reversion-side aggregation:
      - Reverted metrics in the per-seed CSV are populated only on
        rows that flowed through the reversion pass.  The aggregator
        deduplicates reverted records by sequence_hash via the
        `reversion_plan.json` manifest, aggregates per unique reverted
        sequence, then maps the aggregate back to every steered
        sequence whose contaminated state pointed at that reverted
        hash.  This matches the user's stage-4 spec: one reverted
        result mapped onto several steering results.
    """
    experiment_root: Path = args.experiment_root.resolve()
    if not experiment_root.is_dir():
        print(f"ERROR: {experiment_root} not a directory", file=sys.stderr)
        return 2

    # Prefer the with_metrics CSV (if compute-final-metrics has run)
    # because we want the aggregated record to reflect confidence
    # metrics too.  Fall back to the unmetriced CSV with a warning.
    metrics_csv = experiment_root / "all_results_multicycle_with_metrics.csv"
    plain_csv = experiment_root / "all_results_multicycle.csv"
    if metrics_csv.exists():
        per_seed_csv = metrics_csv
    elif plain_csv.exists():
        per_seed_csv = plain_csv
        print(f"  NOTE: {metrics_csv.name} not found; aggregating "
              f"{plain_csv.name} (no confidence metrics will be "
              f"available in the aggregate)")
    else:
        print(f"ERROR: neither {metrics_csv.name} nor {plain_csv.name} "
              f"found under {experiment_root}.  Run `aggregate` first.",
              file=sys.stderr)
        return 2

    with open(per_seed_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  read {len(rows)} per-seed rows from {per_seed_csv.name}")

    # ── Load contamination gating set from cycle_0/plan.json ──────────
    # See classify_reversion_verdict (reversion.py) and the notes11
    # policy: only positions inside the design region ∪ true interface
    # count as contamination.  cycle_0/plan.json stores the indices
    # 0-based; convert to 1-based to match the coordinate system of
    # reverted_mutated_contact_positions_majority.  If plan.json can't
    # be located (unusual layouts / very old runs), the gate is
    # disabled and legacy behaviour prevails.
    gating_1b: Set[int] = set()
    # cold-start trigger threshold (from plan.json).  Used by the
    # ranking-eligibility gate below so that cold-start aggregates
    # whose ra_eff is in (5.0, rmsd_threshold] still get a rank
    # (matches the cold-start all-clean trigger in
    # boltz2_negative_steering.py).  Defaults to 5.0 (legacy
    # behaviour) when no plan can be loaded.
    cold_start_rmsd_threshold: float = 5.0
    for candidate in (experiment_root / "cycle_0" / "plan.json",
                      experiment_root / "plan.json"):
        if candidate.exists():
            try:
                pl = json.loads(candidate.read_text())
                for i in (pl.get("design_region_idx") or []):
                    gating_1b.add(int(i) + 1)
                for i in (pl.get("true_interface_idx") or []):
                    gating_1b.add(int(i) + 1)
                try:
                    cold_start_rmsd_threshold = float(
                        pl.get("rmsd_threshold", cold_start_rmsd_threshold))
                except (TypeError, ValueError):
                    pass
                if gating_1b:
                    print(f"  contamination gating set: {len(gating_1b)} "
                          f"positions from {candidate.relative_to(experiment_root)} "
                          f"(design region ∪ true interface)")
                break
            except (ValueError, OSError) as e:
                print(f"  WARN: could not parse {candidate}: {e}")
    if not gating_1b:
        print("  WARN: no plan.json gating set found — aggregated verdict "
              "contamination check will use legacy (ungated) behaviour")

    # ── Copy per-seed input to raw_per_seed_results.csv ───────────────
    # Provides a stable filename downstream that means "every seed's
    # raw outputs", regardless of whether compute-final-metrics ran.
    raw_csv = experiment_root / "raw_per_seed_results.csv"
    shutil.copy2(per_seed_csv, raw_csv)
    print(f"  wrote {raw_csv.name} (copy of {per_seed_csv.name})")

    # ── Group rows by (cycle, pathway, sequence_group) ────────────────
    # cycle and pathway scope the sequence_group: two designs in
    # different cycles or pathways with the same sequence_group integer
    # are NOT the same unique sequence.  Empty sequence_group means a
    # row that pre-dates the multi-seed plumbing (or the initial
    # baseline); each such row is its own group of size 1.
    from collections import defaultdict
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    singletons: List[Dict] = []
    for r in rows:
        sg = r.get("sequence_group", "")
        if sg in ("", None):
            # Initial-baseline rows or pre-multi-seed rows: treat each
            # as its own singleton group with no aggregation.
            singletons.append(r)
            continue
        key = (str(r.get("cycle", "0")), str(r.get("pathway", "")), str(sg))
        groups[key].append(r)

    n_groups = len(groups)
    n_singletons = len(singletons)
    n_seeds_total = sum(len(v) for v in groups.values())
    print(f"  {n_groups} unique sequence group(s) × ~{n_seeds_total // max(n_groups, 1)} "
          f"seeds + {n_singletons} singleton(s)")

    # ── Aggregate each group ──────────────────────────────────────────
    aggregated_rows: List[Dict] = []

    def _canonical_seed(group: List[Dict]) -> Tuple[Dict, str]:
        """Pick the canonical seed for diagnostics.  Returns (row, warning).

        Sort group by steered_ra_eff_vs_truth ascending (NaN/missing →
        infinity, so they sort to the end).  Pick:
          - odd N: index (N-1)//2 (the middle)
          - even N: index N//2  (the upper-middle, i.e. the WORSE of
            the two middle seeds — conservative)
          - N=1: that seed
        Returns the canonical row and a warning string (empty if no
        warning needed).
        """
        def _ra(r):
            v = r.get("steered_ra_eff_vs_truth", "")
            try:
                f = float(v)
                if math.isnan(f) or math.isinf(f):
                    return float("inf")
                return f
            except (TypeError, ValueError):
                return float("inf")
        sorted_group = sorted(group, key=_ra)
        n = len(sorted_group)
        if n == 1:
            return (sorted_group[0], "")
        if n % 2 == 1:
            return (sorted_group[(n - 1) // 2], "")
        # Even N: pick the upper-middle (worse) seed and warn.
        return (sorted_group[n // 2],
                f"even_n={n}_chose_worse_of_two_middle_seeds")

    for key, group in sorted(groups.items()):
        cycle_str, pathway_str, sg_str = key

        canonical_row, canonical_warning = _canonical_seed(group)

        # Bare design name (strip _sN suffix if present).  Falls back to
        # the canonical seed's design field if no suffix is present.
        design_name = canonical_row.get("design", "")
        if "_s" in design_name:
            base = design_name.rsplit("_s", 1)[0]
            # Validate the suffix is actually a numeric seed index
            suf = design_name.rsplit("_s", 1)[1]
            if suf.isdigit():
                design_name = base

        agg = {
            "cycle": cycle_str,
            "pathway": pathway_str,
            "sequence_group": sg_str,
            "design": design_name,
            "n_seeds": len(group),
            "canonical_seed_index": canonical_row.get("seed_index", ""),
            "canonical_pdb": canonical_row.get("pdb", ""),
            "canonical_seed_choice_warning": canonical_warning,
        }

        # Steered aggregation
        agg.update(_aggregate_per_sequence(group, prefix="steered_"))

        # Reverted aggregation — only run if any group member has
        # reversion_verdict populated (i.e. the steering pipeline
        # flagged this sequence as contaminated and reversion ran).
        any_reverted = any(
            (r.get("reversion_verdict") or "").strip() != ""
            for r in group
        )
        # Per-seed verdict breakdown is always computed: it drives both
        # the outcome classifier (when reversion ran) and n_pass (which
        # is path-agnostic — pose_holds + clean_steered regardless of
        # whether reversion was triggered).
        verdict_counts = _per_seed_verdict_breakdown(group)

        if any_reverted:
            agg.update(_aggregate_per_sequence(group, prefix="reverted_"))
            outcome, reason = _classify_outcome(
                agg, verdict_counts,
                contamination_gating_positions=gating_1b if gating_1b else None,
            )
            agg["outcome"] = outcome
            agg["outcome_reason"] = reason
        else:
            # No seed needed reversion (cold-start skip_steering, OR
            # steering ran but no seed had contamination on mutated
            # positions).  Outcome is no_reversion; n_pass is still
            # derived from per-seed verdicts below — a wrong-placement
            # no-contamination group will have all seeds = no_data and
            # n_pass = 0 (NOT n_seeds), correctly landing in tier none.
            agg["outcome"] = "no_reversion"
            agg["outcome_reason"] = ""

        # n_pass: how many seeds passed structural+contamination filters.
        # Counted as pose_holds (reversion succeeded) + clean_steered
        # (reversion correctly skipped because steered already passed).
        agg["n_pass"] = (verdict_counts["pose_holds"]
                        + verdict_counts["clean_steered"])

        aggregated_rows.append(agg)

    # Singletons (typically the cycle-0 initial baseline) — pass
    # through with n_seeds=1 and no aggregation suffixes on the
    # value columns; they go in as-is so the aggregated CSV still
    # carries them.  We DO populate the canonical_pdb field so
    # consumers don't have to special-case singletons.
    for r in singletons:
        passthrough = dict(r)
        passthrough["n_seeds"] = 1
        passthrough["canonical_seed_index"] = r.get("seed_index", "")
        passthrough["canonical_pdb"] = r.get("pdb", "")
        passthrough["canonical_seed_choice_warning"] = ""
        passthrough["outcome"] = "singleton"
        passthrough["outcome_reason"] = (
            "no sequence_group — singleton row, no aggregation applied"
        )
        passthrough["n_pass"] = ""
        aggregated_rows.append(passthrough)

    # ── Recompute composite ranks on the aggregated rows ──────────────
    # The ranker now operates on the median-aggregated rows.  We use
    # the existing _ranking_composite logic but on the aggregated
    # column names (e.g. steered_ra_eff_vs_truth_median in place of
    # steered_ra_eff_vs_truth).  Wrap in an adapter:
    def _rank_continuous(r: Dict, name_median: str) -> Optional[float]:
        v = r.get(name_median, "")
        if v in ("", None):
            return None
        try:
            f = float(v)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        except (ValueError, TypeError):
            return None

    def _agg_row_is_clean_steered(r: Dict) -> bool:
        # Cold-start all-clean aggregates (notes12) have
        # design == "initial" AND outcome == "no_reversion".
        # They MUST be ranking-eligible — they're the final output
        # for cold-start binders.  We admit them via a slightly looser
        # ra_eff cap (matching the cold-start trigger threshold from
        # plan.json, default 5.0 for legacy compatibility).  Steered
        # aggregates use the legacy 5.0 cap unchanged.
        is_cold_start = (
            r.get("design") == "initial"
            and (r.get("outcome") or "").strip() == "no_reversion"
        )
        if r.get("design") == "initial" and not is_cold_start:
            # design=="initial" without no_reversion outcome was the
            # legacy single-row baseline used by extract_passing.py
            # to sanity-check; preserve that exclusion to avoid
            # promoting reconstructed initial rows from old workdirs.
            return False
        outcome = (r.get("outcome") or "").strip()
        if outcome not in ("", "no_reversion"):
            return False
        if r.get("steered_receptor_intact_majority") != 1:
            return False
        ra = _rank_continuous(r, "steered_ra_eff_vs_truth_median")
        if ra is None:
            return False
        cap = cold_start_rmsd_threshold if is_cold_start else 5.0
        if ra >= cap:
            return False
        return True

    def _agg_row_is_pose_holds(r: Dict) -> bool:
        if r.get("outcome") != "pose_holds":
            return False
        if r.get("reverted_receptor_intact_majority") != 1:
            return False
        ra = _rank_continuous(r, "reverted_ra_eff_vs_truth_median")
        if ra is None or ra >= 5.0:
            return False
        return True

    def _agg_composite(r: Dict) -> Optional[float]:
        if _agg_row_is_clean_steered(r):
            ra = _rank_continuous(r, "steered_ra_eff_vs_truth_median")
        elif _agg_row_is_pose_holds(r):
            ra = _rank_continuous(r, "reverted_ra_eff_vs_truth_median")
        else:
            return None
        if ra is None:
            return None
        tj = _rank_continuous(r, "steered_true_jaccard_median")
        if tj is None:
            return None
        return tj - 0.05 * ra

    def _agg_ra_eff(r: Dict) -> Optional[float]:
        if _agg_row_is_clean_steered(r):
            return _rank_continuous(r, "steered_ra_eff_vs_truth_median")
        if _agg_row_is_pose_holds(r):
            return _rank_continuous(r, "reverted_ra_eff_vs_truth_median")
        return None

    # Initialise blanks
    for r in aggregated_rows:
        r.setdefault("rank_by_ra_eff", "")
        r.setdefault("rank_by_composite_score", "")

    # rank_by_ra_eff: ascending (lower is better)
    eligible = []
    for r in aggregated_rows:
        v = _agg_ra_eff(r)
        if v is not None:
            eligible.append((v, id(r), r))
    eligible.sort(key=lambda t: (t[0], t[1]))
    for rank_idx, (_, _, r) in enumerate(eligible, start=1):
        r["rank_by_ra_eff"] = rank_idx

    # rank_by_composite_score: descending (higher is better)
    eligible = []
    for r in aggregated_rows:
        v = _agg_composite(r)
        if v is not None:
            eligible.append((v, id(r), r))
    eligible.sort(key=lambda t: (-t[0], t[1]))
    for rank_idx, (_, _, r) in enumerate(eligible, start=1):
        r["rank_by_composite_score"] = rank_idx

    # ── Write aggregated_results.csv ──────────────────────────────────
    # Build the column list deterministically so the CSV layout is
    # stable across runs.  Block order: identification, steered,
    # reverted, verdict + ranks.  Any unexpected keys (e.g. from
    # singleton passthrough) get appended in sorted order at the end.
    preferred_cols = [
        # Identification
        "cycle", "pathway", "sequence_group", "design",
        "n_seeds", "canonical_seed_index", "canonical_pdb",
        "canonical_seed_choice_warning",
        # Steered aggregates — generated by _aggregate_per_sequence
        # in this order:
    ]
    for metric in (_AGG_INTEGER_METRICS + _AGG_CONTINUOUS_METRICS):
        preferred_cols.extend([
            f"steered_{metric}_median",
            f"steered_{metric}_mad",
            f"steered_{metric}_n_used",
        ])
    for metric in _AGG_BINARY_METRICS:
        preferred_cols.extend([
            f"steered_{metric}_majority",
            f"steered_{metric}_n_positive",
            f"steered_{metric}_n_used",
        ])
    for metric in _AGG_POSITION_SET_METRICS:
        preferred_cols.extend([
            f"steered_{metric}_majority",
            f"steered_{metric}_per_seed_counts",
            f"steered_{metric}_n_used",
        ])
    # Reverted aggregates (same shape, reverted_ prefix)
    for metric in (_AGG_INTEGER_METRICS + _AGG_CONTINUOUS_METRICS):
        preferred_cols.extend([
            f"reverted_{metric}_median",
            f"reverted_{metric}_mad",
            f"reverted_{metric}_n_used",
        ])
    for metric in _AGG_BINARY_METRICS:
        preferred_cols.extend([
            f"reverted_{metric}_majority",
            f"reverted_{metric}_n_positive",
            f"reverted_{metric}_n_used",
        ])
    for metric in _AGG_POSITION_SET_METRICS:
        preferred_cols.extend([
            f"reverted_{metric}_majority",
            f"reverted_{metric}_per_seed_counts",
            f"reverted_{metric}_n_used",
        ])
    # Outcome + ranks
    preferred_cols.extend([
        "outcome",
        "outcome_reason",
        "n_pass",
        "rank_by_ra_eff",
        "rank_by_composite_score",
    ])
    all_keys = {k for r in aggregated_rows for k in r.keys()}
    extras = sorted(k for k in all_keys if k not in preferred_cols)
    fieldnames = [k for k in preferred_cols if k in all_keys] + extras

    out_csv = experiment_root / "aggregated_results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in aggregated_rows:
            w.writerow(r)
    print(f"  wrote {out_csv.name} ({len(aggregated_rows)} rows)")

    # Quick outcome tally for the log
    from collections import Counter
    outcomes = Counter(
        r.get("outcome", "") for r in aggregated_rows
    )
    print("  Outcome tally:")
    for v, c in sorted(outcomes.items()):
        print(f"    {v:<22s} {c:>4d}")

    # Even-N warnings tally
    n_warned = sum(
        1 for r in aggregated_rows
        if r.get("canonical_seed_choice_warning")
    )
    if n_warned:
        print(f"  Even-N canonical-seed warnings: {n_warned} row(s) "
              f"(degraded — some seeds failed; see "
              f"canonical_seed_choice_warning column)")

    return 0


def _passes_final_metrics_filter(row: Dict, args: argparse.Namespace,
                                  metric_col: str, threshold: float,
                                  skip_steering_cycle0: bool) -> bool:
    """Apply the cmd_compute_final_metrics passing-row filter to one row."""
    # P0.2 + P0-audit: --populate-all truly means "populate every
    # row" — including rows dropped by reversion verdict routing.
    # Per-seed aggregation requires every seed's steered_*
    # confidence metrics so the median across seeds is meaningful;
    # skipping reversion_dropped rows would leave the dropped
    # seeds as holes in the per-seed CSV and bias the aggregate.
    if args.populate_all:
        return True
    # Default (no --populate-all): skip rows dropped by reversion
    # verdict routing — confidence metrics on a discarded
    # prediction are wasted work.  pose_holds rows are NOT
    # dropped (reversion_dropped == 0); we still compute steered_*
    # cm metrics on them so the CSV shows what the steered
    # prediction looked like before reversion.
    try:
        if int(row.get("reversion_dropped", "0") or "0") == 1:
            return False
    except (TypeError, ValueError):
        pass
    # Skip_steering exemption: include the initial row(s) if their
    # parent workdir had skip_steering=true.  Applies only to
    # cycle-0 initial baselines (cycle-1+ rows never have
    # design=="initial").  Multi-seed cold-start (notes12) writes
    # additional rows with design names "initial_s1", "initial_s2",
    # ... — all are baseline rows that need confidence metrics.
    _design = row.get("design", "")
    if (skip_steering_cycle0
            and (_design == "initial" or _design.startswith("initial_s"))
            and str(row.get("cycle", "0")) == "0"):
        if args.require_intact:
            try:
                if int(row.get("steered_receptor_intact", "0") or "0") != 1:
                    return False
            except (TypeError, ValueError):
                return False
        return True
    # Require steered_receptor_intact == 1 unless explicitly disabled.
    if args.require_intact:
        try:
            if int(row.get("steered_receptor_intact", "0") or "0") != 1:
                return False
        except (TypeError, ValueError):
            return False
    v = _float_or_none(row.get(metric_col))
    if v is None:
        return False
    return v < threshold


def _resume_key_for_row(row: Dict) -> str:
    """Stable per-design resume key (pdb path is unique per design)."""
    # Resume key is the absolute pdb path because the `pathway` field
    # is shared across all cycle-0 designs (they all get the literal
    # string "cycle_0" from cmd_aggregate, which would collide).  The
    # pdb path is unique per design.  Falls back to pathway then to
    # an empty string for safety.
    return row.get("pdb") or row.get("pathway") or ""


def _load_existing_metrics_rows(output_csv: Path,
                                 prefixed_metric_keys: List[str]) -> Dict[str, Dict]:
    """Load existing-output rows keyed by resume key, keeping only those with at least one populated metric."""
    existing_by_key: Dict[str, Dict] = {}
    if not output_csv.exists():
        return existing_by_key
    with open(output_csv) as f:
        er = csv.DictReader(f)
        for row in er:
            key = _resume_key_for_row(row)
            if not key:
                continue
            # Only treat as "done" if at least one metric field is non-empty.
            has_any = any(row.get(k, "") not in ("", None)
                          for k in prefixed_metric_keys)
            if has_any:
                existing_by_key[key] = row
    return existing_by_key


def _classify_steered_confidence_flag(
    row: Dict,
    flag_pass_frac_min: float,
    flag_iptm_min: float,
    flag_complex_plddt_min: float,
    flag_ipae_max: float,
) -> str:
    """Classify steered confidence flag (ok, single trigger, or multiple); '' for rows without metrics."""
    ipsae = _float_or_none(row.get("steered_ipsae_min"))
    if ipsae is None:
        # No metrics — nothing to classify.
        return ""
    triggered = []
    pf = _float_or_none(row.get("steered_pae_pass_frac"))
    if pf is not None and pf < flag_pass_frac_min:
        triggered.append("low_pass_frac")
    ip = _float_or_none(row.get("steered_iptm"))
    if ip is not None and ip < flag_iptm_min:
        triggered.append("low_iptm")
    cp = _float_or_none(row.get("steered_complex_plddt"))
    if cp is not None and cp < flag_complex_plddt_min:
        triggered.append("low_plddt")
    ipae = _float_or_none(row.get("steered_ipae"))
    if ipae is not None and ipae > flag_ipae_max:
        triggered.append("high_ipae")
    if not triggered:
        return "ok"
    if len(triggered) == 1:
        return triggered[0]
    return "multiple"


def _write_metrics_csv(output_csv: Path, joined_rows: List[Dict]) -> None:
    """Write joined rows to output_csv using canonical Block A/B/C/End column order."""
    fieldnames = _aggregate_csv_fieldnames(joined_rows)
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r2 in joined_rows:
            w.writerow(r2)


def _process_passing_row(
    row: Dict,
    args: argparse.Namespace,
    prefixed_metric_keys: List[str],
    metric_keys_raw: List[str],
    metric_key_prefix: str,
    existing_by_key: Dict[str, Dict],
    scratch_dir: Path,
    compute_metrics_script: Path,
    cycle_0_plan: Dict,
    cycle_0_plan_path: Path,
    pred_rec_chain: str,
    pred_eff_chain: str,
    model_name: str,
    metric_col: str,
    passing_progress: int,
    n_passing_total: int,
) -> Tuple[Dict, str]:
    """Process one passing row: returns (out_row, status) where status is 'resumed', 'computed', or 'failed'."""
    pathway_key = _resume_key_for_row(row)
    # pathway_label kept for human-readable progress messages
    pathway_label = row.get("pathway") or row.get("pdb") or "?"
    pdb_path_str = row.get("pdb", "")
    pdb_path = Path(pdb_path_str) if pdb_path_str else None

    # Resume.  Take the CURRENT base row (which may have fresh
    # columns like mutations_aa, rank_by_truth, or new derived
    # columns that a previous run didn't have) and overlay just
    # the steered_* / reverted_* metric values from the cached row onto it.  This
    # prevents the resume path from stomping fresh base columns
    # with stale data from an older output CSV.
    if args.skip_existing and pathway_key in existing_by_key:
        cached = existing_by_key[pathway_key]
        merged = dict(row)  # start from fresh base row
        for k in prefixed_metric_keys:
            v = cached.get(k, "")
            if v not in ("", None):
                merged[k] = v
        # Also carry over steered_error if it was set (usually blank
        # on success, but preserve failure state for diagnostics).
        if cached.get("steered_error"):
            merged["steered_error"] = cached["steered_error"]
        return merged, "resumed"

    out_row = dict(row)
    for k in prefixed_metric_keys:
        out_row.setdefault(k, "")
    out_row.setdefault("steered_error", "")

    if pdb_path is None or not pdb_path.exists():
        msg = f"pdb not found: {pdb_path_str}"
        print(f"  [{passing_progress}/{n_passing_total}] {pathway_label}: SKIP ({msg})")
        out_row["steered_error"] = msg
        return out_row, "failed"

    # compute_metrics.py's Boltz-2 parser looks for
    # confidence_<stem>.json, pae_<stem>.npz, plddt_<stem>.npz
    # alongside each PDB.  The steering pipeline copies the chosen
    # Boltz output to `prediction.pdb` but leaves those sidecar
    # files nested under Boltz's native output layout (e.g.
    # `<design>/predictions/<stem>/pdb/<stem>_model_0.pdb` with
    # `confidence_<stem>_model_0.json` next to it).  If we pointed
    # --prediction-dir at `prediction.pdb`'s parent we'd get
    # metrics with ptm=0, iptm=0, pae=0 (only pLDDT would work,
    # via the B-factor-column fallback).  Instead, locate the
    # nested sidecar PDB and point the parser at its directory.
    sidecar_pdb = _locate_boltz_sidecar_pdb(pdb_path)
    if sidecar_pdb is not None:
        pred_dir = sidecar_pdb.parent
        lookup_stem = sidecar_pdb.stem
        metric_pdb = sidecar_pdb
    else:
        pred_dir = pdb_path.parent
        lookup_stem = pdb_path.stem
        metric_pdb = pdb_path

    # Determine chain lengths from the PDB we'll actually parse,
    # so CA counts match whatever compute_metrics.py ingests.
    ca_counts = _count_ca_per_chain(metric_pdb)
    rec_len = ca_counts.get(pred_rec_chain, 0)
    eff_len = ca_counts.get(pred_eff_chain, 0)
    if rec_len == 0 or eff_len == 0:
        # Fallback: use the two largest chains
        ordered = sorted(ca_counts.items(), key=lambda kv: -kv[1])
        if len(ordered) >= 2:
            rec_len = rec_len or ordered[0][1]
            eff_len = eff_len or ordered[1][1]
    if rec_len == 0 or eff_len == 0:
        msg = (f"could not determine chain lengths from {metric_pdb.name} "
               f"(CA counts: {ca_counts})")
        print(f"  [{passing_progress}/{n_passing_total}] {pathway_label}: SKIP ({msg})")
        out_row["steered_error"] = msg
        return out_row, "failed"

    # One output CSV per row, lives in the scratch dir.  Use a
    # short, filesystem-safe key derived from cycle + design name
    # rather than the full pdb path.
    cyc = row.get("cycle", "0")
    des = row.get("design", "unknown")
    safe_key = f"cycle{cyc}_{des}".replace("/", "_").replace(" ", "_")
    per_row_csv = scratch_dir / f"{safe_key}.csv"

    # Look up this row's cumulative mutation set.  After
    # cmd_aggregate's _translate_aggregate_row pass, the column
    # is named `steered_mutations_chimerax` (renamed from the
    # legacy `mutations_chimerax`).  We read the post-rename name
    # with a fallback to the legacy name for any path that
    # bypasses _translate_aggregate_row.  Format is
    # "/A:5,26,44,46" (chain letter + 1-based positions).
    # compute_metrics.py's parse_mutated_positions accepts this
    # form directly, so we forward the string as-is.  Blank string
    # (cold start, no mutations) is fine: compute_metrics.py will
    # emit the contact columns with blank mutated_contact_positions.
    #
    # Bug history: this previously read only `mutations_chimerax`,
    # which after the _AGG_RENAME refactor is always blank after
    # aggregate runs.  Consequence: compute_metrics.py ran without
    # --mutated-positions on every row, blanking the
    # steered_mutated_contact_positions and
    # steered_n_contacts_on_mutated_positions columns for every
    # design — INCLUDING the ones that HAD been correctly populated
    # by _populate_reverted_mutations in cmd_aggregate (which
    # compute-final-metrics runs after).
    mutations_cx = (
        row.get("steered_mutations_chimerax", "")
        or row.get("mutations_chimerax", "")
        or ""
    )

    cmd = [
        args.python_executable, str(compute_metrics_script),
        "--model", model_name,
        "--prediction-dir", str(pred_dir),
        "--chain-lengths", str(rec_len), str(eff_len),
        "--output-csv", str(per_row_csv),
        "--pae-cutoff", str(args.pae_cutoff),
        "--receptor-chain", pred_rec_chain,
        "--effector-chain", pred_eff_chain,
        "--contact-cutoff", str(args.contact_cutoff),
        # v8 Level 1: effector atom filter from cycle_0 plan.json.
        # compute_metrics.py reads the 'effector_interface_atoms'
        # key directly; no materialised filter file needed.
        "--effector-atom-filter-json", str(cycle_0_plan_path),
    ]
    # Per-prediction native-PDB-dependent metrics (intact_core +
    # weighted_jaccard).  cycle_0/plan.json carries the absolute
    # path to the ground-truth PDB under "ground_truth" — pass it
    # through so compute_metrics.py can populate those columns
    # alongside the PAE-derived ones.  When the cycle_0 plan
    # doesn't carry a ground_truth (defensive), we just omit the
    # flag and the metrics columns come out blank.
    _native_pdb = cycle_0_plan.get("ground_truth", "")
    if _native_pdb:
        cmd.extend(["--native-pdb", str(_native_pdb)])
    if mutations_cx:
        cmd.extend(["--mutated-positions", mutations_cx])
    # No --reference-pdb: we only want confidence metrics here.
    # Structural RMSDs come from all_results_multicycle.csv (base
    # columns), which already uses the correct truth-vs-pred
    # chain mapping.

    print(f"  [{passing_progress}/{n_passing_total}] {pathway_label}  "
          f"{metric_col}={row.get(metric_col)}  pdb={metric_pdb.name}")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=args.subprocess_timeout,
        )
    except subprocess.TimeoutExpired:
        msg = f"compute_metrics.py timed out after {args.subprocess_timeout}s"
        print(f"    ERROR: {msg}")
        out_row["steered_error"] = msg
        return out_row, "failed"
    except Exception as e:
        msg = f"subprocess failed: {e}"
        print(f"    ERROR: {msg}")
        out_row["steered_error"] = msg
        return out_row, "failed"

    if proc.returncode != 0:
        msg = (f"compute_metrics.py exited with {proc.returncode}: "
               f"{(proc.stderr or '').strip()[:300]}")
        print(f"    ERROR: {msg}")
        if args.verbose and proc.stdout:
            print("    --- stdout ---")
            print(proc.stdout)
        out_row["steered_error"] = msg
        return out_row, "failed"

    if not per_row_csv.exists():
        msg = f"compute_metrics.py produced no CSV at {per_row_csv}"
        print(f"    ERROR: {msg}")
        out_row["steered_error"] = msg
        return out_row, "failed"

    # Read back the per-row CSV.  If multiple models were picked
    # up (shouldn't happen with one-design-per-dir, but defend
    # against it), select the row whose model_name matches the
    # pdb stem we asked about.
    try:
        with open(per_row_csv) as f:
            cm_rows = list(csv.DictReader(f))
    except Exception as e:
        msg = f"could not read {per_row_csv}: {e}"
        print(f"    ERROR: {msg}")
        out_row["steered_error"] = msg
        return out_row, "failed"

    if not cm_rows:
        msg = "compute_metrics.py CSV was empty"
        print(f"    ERROR: {msg}")
        out_row["steered_error"] = msg
        return out_row, "failed"

    chosen = None
    for cr in cm_rows:
        if cr.get("model_name") == lookup_stem:
            chosen = cr
            break
    if chosen is None:
        chosen = cm_rows[0]  # fall back to first (and usually only)

    for k in metric_keys_raw:
        out_row[metric_key_prefix + k] = chosen.get(k, "")
    return out_row, "computed"


def cmd_compute_final_metrics(args: argparse.Namespace) -> int:
    """For every row in all_results_multicycle.csv that passes the
    final ranking filter, shell out to compute_metrics.py to compute
    confidence / interface / structural-RMSD metrics, and write a
    single joined CSV alongside all_results_multicycle.csv.

    This is an opt-in post-aggregation step.  The fast `aggregate`
    pass stays stdlib-only; the heavy Kabsch/numpy work only runs
    when explicitly requested.
    """
    experiment_root: Path = args.experiment_root.resolve()
    if not experiment_root.is_dir():
        print(f"ERROR: {experiment_root} not a directory", file=sys.stderr)
        return 2

    all_results_csv = experiment_root / "all_results_multicycle.csv"
    if not all_results_csv.exists():
        print(f"ERROR: {all_results_csv} not found — run `aggregate` first.",
              file=sys.stderr)
        return 2

    # ── Resolve compute_metrics.py ─────────────────────────────────────
    compute_metrics_script = args.compute_metrics_script
    if compute_metrics_script is None:
        # Default: sibling of this script, i.e. the benchmark pipeline root.
        compute_metrics_script = Path(__file__).resolve().parent / "compute_metrics.py"
    compute_metrics_script = Path(compute_metrics_script).resolve()
    if not compute_metrics_script.exists():
        print(f"ERROR: compute_metrics.py not found at {compute_metrics_script}.\n"
              f"       Pass --compute-metrics-script to point at it explicitly.",
              file=sys.stderr)
        return 2

    # ── Read prediction chain IDs from cycle_0/plan.json ───────────────
    # We only need the prediction chain IDs (to count Cα atoms per
    # chain when building --chain-lengths for compute_metrics.py).
    # The reference PDB and truth chain IDs are NOT needed: structural
    # RMSDs are already in all_results_multicycle.csv from the
    # steering pipeline, and compute_metrics.py is only being used
    # here for confidence metrics (pTM / ipTM / ipSAE / actifPTM /
    # pLDDT / PAE), none of which depend on a reference.
    # Detect cycle-0 layout (nested under cycle_0/ for multi-cycle runs,
    # flat in the experiment root for --n-cycles 1 runs).  Same logic
    # as cmd_aggregate.
    nested_cycle_0 = experiment_root / "cycle_0"
    if (nested_cycle_0 / "plan.json").exists():
        cycle_0_plan_path = nested_cycle_0 / "plan.json"
    elif (experiment_root / "plan.json").exists():
        cycle_0_plan_path = experiment_root / "plan.json"
    else:
        print(f"ERROR: no cycle_0 plan.json under {experiment_root} "
              f"(checked {nested_cycle_0}/plan.json and "
              f"{experiment_root}/plan.json)", file=sys.stderr)
        return 2
    cycle_0_plan = json.loads(cycle_0_plan_path.read_text())

    pred_rec_chain = cycle_0_plan.get("pred_receptor_chain", "A")
    pred_eff_chain = cycle_0_plan.get("pred_effector_chain", "B")

    # ── Read all_results_multicycle.csv and apply the filter ──────────
    with open(all_results_csv) as f:
        reader = csv.DictReader(f)
        base_rows = list(reader)
        base_fieldnames = list(reader.fieldnames or [])

    metric_col = args.metric_column
    threshold = args.rmsd_threshold

    # Detect skip_steering at cycle 0 — if the cycle-0 plan has
    # skip_steering=true (or no `designs` array at all), the
    # filter criterion doesn't apply to the initial row: there
    # are no steered designs to filter against, and the initial
    # row IS the only row.  We still want confidence metrics on
    # it so the comparison table has something to rank.
    _skip_steering_cycle0 = bool(cycle_0_plan.get("skip_steering"))
    if _skip_steering_cycle0:
        print("  cycle_0 plan has skip_steering=true — will include "
              "the initial row in metrics regardless of ra_eff filter")

    passing_rows = [
        r for r in base_rows
        if _passes_final_metrics_filter(
            r, args, metric_col, threshold, _skip_steering_cycle0,
        )
    ]
    print(f"Read {len(base_rows)} rows from {all_results_csv.name}")
    if args.populate_all:
        print("  Filter: --populate-all set — all rows except "
              "reversion_dropped==1")
    else:
        print(f"  Filter: {metric_col} < {threshold} Å"
              + (" AND steered_receptor_intact == 1" if args.require_intact else "")
              + " AND reversion_dropped != 1")
    print(f"  Passing: {len(passing_rows)}")

    # ── Resume support: load any existing output and skip finished rows
    output_csv: Path = args.output_csv
    if output_csv is None:
        output_csv = experiment_root / "all_results_multicycle_with_metrics.csv"
    output_csv = Path(output_csv).resolve()

    # The per-design metrics we harvest from compute_metrics.py output.
    # We NEVER add these as new columns if they collide with existing
    # base columns — we prefix them with `steered_` so the join is lossless
    # AND the prefix correctly indicates that these confidence metrics
    # were computed on the STEERED prediction.  For pose_holds rows the
    # steered_* metrics describe the pre-reversion pose; corresponding
    # reverted_* metrics are populated by harvest-reversions.
    #
    # We deliberately DO NOT harvest the structural RMSD columns
    # (rmsd_receptor, rmsd_effector_*, n_receptor_ca, n_effector_ca)
    # from compute_metrics.py:
    #   1. `all_results_multicycle.csv` already carries the
    #      authoritative per-design RMSDs from the steering pipeline
    #      (receptor_aligned_effector_rmsd, independent_receptor_rmsd,
    #      independent_effector_rmsd) in its base columns, and those
    #      use the correct truth-vs-pred chain mapping.
    #   2. compute_metrics.py uses a single --receptor-chain /
    #      --effector-chain pair for BOTH the prediction and the
    #      reference, so on complexes where truth chains != pred
    #      chains (7QPX/7QZD — truth B/C vs pred A/B) its structural
    #      RMSDs would be wrong.
    # Skipping them also means we don't need --reference-pdb at all,
    # which in turn means no Kabsch fit per design — much faster.
    METRIC_KEYS_RAW = [
        "avg_plddt", "complex_plddt",
        "ptm", "iptm", "pae_mean", "ipae", "pae_pass_frac",
        "ipsae_ab", "ipsae_ba", "ipsae_min",
        # 15Å iPSAE — same Dunbrack iPSAE machinery as the 10Å trio
        # above, evaluated at a more permissive PAE cutoff.  Computed
        # in the same compute_metrics.py call so no extra subprocess
        # work — the PAE matrix is already loaded.
        "ipsae_ab_15", "ipsae_ba_15", "ipsae_min_15",
        "actifptm", "af_rank_score",
        # Per-prediction native-PDB-dependent metrics.  These were
        # previously computed post-hoc by compute_interface_metrics.py
        # (and run_biophysical_metrics.py for interface_plddt) on
        # survivors only — moved here so every steered prediction is
        # scored on the same metric set as it lands, without a
        # separate sidecar-finding pass.  intact_core and
        # weighted_jaccard are populated when --native-pdb was passed
        # to compute_metrics.py (the cycle_0 ground truth for this
        # design); blank otherwise.  interface_plddt only needs the
        # model PDB and is always populated.
        "interface_plddt",
        "intact_core",
        "weighted_jaccard",
        # Interface contacts / contamination check.  See
        # compute_metrics.py:compute_interface_contacts /
        # classify_mutation_reliance for definitions.  n_contact_residues
        # and contact_residues are populated for every row; the
        # contamination columns (n_contacts_on_mutated_positions,
        # mutated_contact_positions) are
        # populated only when the mutation list was supplied — i.e.
        # never for cold-start / initial rows.
        #
        # The authoritative contamination check: any mutated residue
        # that contacts the effector invalidates the metrics.  The
        # list of such positions is in `mutated_contact_positions`.
        "n_contact_residues", "contact_residues",
        "n_contacts_on_mutated_positions",
        "mutated_contact_positions",
        "contact_cutoff_used",
    ]
    METRIC_KEY_PREFIX = "steered_"
    prefixed_metric_keys = [METRIC_KEY_PREFIX + k for k in METRIC_KEYS_RAW]

    # Load existing output CSV for resume (helper handles the file-exists
    # gate and the "row has at least one populated metric" predicate).
    existing_by_key: Dict[str, Dict] = {}
    if args.skip_existing:
        existing_by_key = _load_existing_metrics_rows(
            output_csv, prefixed_metric_keys,
        )
        if existing_by_key:
            print(f"  Resume: {len(existing_by_key)} rows already have metrics "
                  f"in {output_csv.name}")

    # ── Set up a scratch dir for per-row compute_metrics CSVs ──────────
    scratch_dir = experiment_root / "final_metrics_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # ── Build the augmented fieldnames up front so the output CSV is
    # consistent across rows / partial runs.
    augmented_fieldnames = list(base_fieldnames)
    for k in prefixed_metric_keys:
        if k not in augmented_fieldnames:
            augmented_fieldnames.append(k)
    # Derived columns added by the post-processing pass below.
    for k in ("steered_pae_cutoff_used",
              "steered_confidence_flag",
              "steered_error"):
        if k not in augmented_fieldnames:
            augmented_fieldnames.append(k)

    # ── Iterate over EVERY row in all_results_multicycle.csv ──────────
    # Rows that don't pass the filter are still preserved in the
    # output (with empty steered_* / reverted_* fields) so the joined CSV is a true
    # superset of all_results_multicycle.csv.  We only shell out to
    # compute_metrics.py for rows that DO pass the filter.
    model_name = args.model
    joined_rows: List[Dict] = []
    n_computed = 0
    n_skipped_resume = 0
    n_skipped_filter = 0
    n_failed = 0
    passing_set = {id(r) for r in passing_rows}
    n_passing_total = len(passing_rows)
    passing_progress = 0  # 1-based counter for the printed progress

    for row in base_rows:
        # Non-passing rows: preserve as-is with empty steered_* / reverted_* fields.
        if id(row) not in passing_set:
            out_row = dict(row)
            for k in prefixed_metric_keys:
                out_row.setdefault(k, "")
            out_row.setdefault("steered_error", "")
            joined_rows.append(out_row)
            n_skipped_filter += 1
            continue

        passing_progress += 1

        out_row, status = _process_passing_row(
            row, args,
            prefixed_metric_keys, METRIC_KEYS_RAW, METRIC_KEY_PREFIX,
            existing_by_key, scratch_dir,
            compute_metrics_script, cycle_0_plan, cycle_0_plan_path,
            pred_rec_chain, pred_eff_chain,
            model_name, metric_col,
            passing_progress, n_passing_total,
        )
        joined_rows.append(out_row)
        if status == "resumed":
            n_skipped_resume += 1
        elif status == "computed":
            n_computed += 1
            # Write output CSV after every successful row so resume works
            # cleanly even if the job is killed mid-loop.  Use the same
            # canonical Block A/B/C/End layout as the final writer so the
            # column order is consistent whether the job completes or is
            # killed.
            _write_metrics_csv(output_csv, joined_rows)
        else:  # "failed"
            n_failed += 1

    # ── Post-processing pass: derive steered_pae_cutoff_used,
    # steered_confidence_flag, and recompute the unified ranks ─────────
    # We compute these AFTER the harvest loop because they need a
    # global view of all populated rows (for the rank) and they need
    # to apply the threshold flags consistently across both freshly-
    # computed rows and resumed rows from a previous run.
    pae_cutoff_used = float(args.pae_cutoff)
    flag_pass_frac_min     = float(args.flag_pass_frac_min)
    flag_iptm_min          = float(args.flag_iptm_min)
    flag_complex_plddt_min = float(args.flag_complex_plddt_min)
    flag_ipae_max          = float(args.flag_ipae_max)

    for r in joined_rows:
        r["steered_pae_cutoff_used"] = pae_cutoff_used
        r["steered_confidence_flag"] = _classify_steered_confidence_flag(
            r, flag_pass_frac_min, flag_iptm_min,
            flag_complex_plddt_min, flag_ipae_max,
        )

    # Recompute the unified ranks now that compute-final-metrics has
    # populated the steered_* confidence columns.  rank_by_ra_eff and
    # rank_by_composite_score were already populated by aggregate
    # (both use only steering-pipeline outputs that are available
    # pre-compute-final-metrics) and will be re-derived here too —
    # values are deterministic, so the second computation produces
    # identical output.
    _compute_unified_ranks(joined_rows)

    # Final write (covers the case where every row was resume-skipped
    # or failed and the loop never hit the incremental write path).
    # Reorder columns into the canonical Block A / B / C / End layout
    # so the CSV the user actually reads has steered_* metrics inserted
    # into Block B alongside the steering RMSDs, not appended at the
    # end as legacy steered_* / reverted_* columns used to be.
    _write_metrics_csv(output_csv, joined_rows)

    print()
    print(f"Wrote {output_csv} ({len(joined_rows)} rows)")
    print(f"  computed now    : {n_computed}")
    print(f"  resumed         : {n_skipped_resume}")
    print(f"  failed          : {n_failed}")
    print(f"  filtered out    : {n_skipped_filter} (preserved with empty steered_* / reverted_*)")
    if n_failed == 0:
        # Tidy up scratch only on clean runs so failed rows remain
        # inspectable.
        try:
            shutil.rmtree(scratch_dir)
        except Exception:
            pass
    else:
        print(f"  scratch dir kept for inspection: {scratch_dir}")

    return 0 if n_failed == 0 else 1


# ───────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("iterate-plan",
                        help="Build cycle-N+1 plan for one pathway")
    pp.add_argument("--experiment-root", type=Path, required=True)
    pp.add_argument("--parent-pathway", required=True,
                    help="Parent pathway label, e.g. c0d05")
    pp.add_argument("--cycle", type=int, required=True,
                    help="The new cycle index (parent's cycle + 1)")
    pp.add_argument("--workdir", type=Path, required=True)
    pp.add_argument("--n-designs", type=int, default=None,
                    help="Override n-designs for this cycle (default: "
                         "inherit from cycle 0)")
    pp.set_defaults(func=cmd_iterate_plan)

    pcd = sub.add_parser("compute-distances",
                         help="Phase 1 (in container): compute pose "
                              "distances for every candidate in a cycle-N+1 "
                              "pathway, write distances.json")
    pcd.add_argument("--workdir", type=Path, required=True)
    pcd.set_defaults(func=cmd_compute_distances)

    pc = sub.add_parser("iterate-collect",
                        help="Phase 2 (outside container): filter + "
                             "maximin + maybe self-resubmit via sbatch")
    pc.add_argument("--workdir", type=Path, required=True)
    pc.add_argument("--max-cycles", type=int, required=True)
    pc.add_argument("--max-passing", type=int, default=5,
                    help="Max survivors per parent per cycle (default: 5)")
    pc.add_argument("--novelty-cutoff", type=float, default=10.0,
                    help="Min effector_rmsd to every upstream pose "
                         "for a candidate to be considered novel (default: 10 Å)")
    pc.add_argument("--n-designs", type=int, default=None,
                    help="Override n-designs for the next cycle (default: "
                         "inherit from cycle 0)")
    pc.set_defaults(func=cmd_iterate_collect)

    # ─── iterate-collect-prefilter (host, cycle-1+ phase 2a) ────────
    pcp = sub.add_parser(
        "iterate-collect-prefilter",
        help="Host, phase 2a: apply intact filter to distances.json, "
             "write prefilter.json.  Does not run novelty/maximin/"
             "resubmit — those happen in iterate-collect-finalize "
             "after the reversion chain runs.",
    )
    pcp.add_argument("--workdir", type=Path, required=True)
    pcp.set_defaults(func=cmd_iterate_collect_prefilter)

    # ─── build-contaminated (container, phase 2b) ───────────────────
    pbc = sub.add_parser(
        "build-contaminated",
        help="Container, phase 2b: read prefilter.json, run contact "
             "analysis on every intact candidate via compute_metrics.py, "
             "flag any design where a mutated residue contacts the "
             "effector, write contaminated.json",
    )
    pbc.add_argument("--workdir", type=Path, required=True)
    pbc.add_argument("--compute-metrics-script", type=Path, default=None,
                     help="Path to compute_metrics.py (default: sibling "
                          "of this script)")
    pbc.add_argument("--pae-cutoff", type=float, default=10.0,
                     help="PAE cutoff (Å) for ipSAE in compute_metrics.py "
                          "(default: 10.0)")
    pbc.add_argument("--contact-cutoff", type=float, default=5.0,
                     help="Heavy-atom contact cutoff (Å) for the contact "
                          "definition (default: 5.0)")
    pbc.set_defaults(func=cmd_build_contaminated)

    # ─── iterate-collect-finalize (host, cycle-1+ phase 2c) ─────────
    pcf = sub.add_parser(
        "iterate-collect-finalize",
        help="Host, phase 2c: read prefilter.json + (optional) "
             "reversion_results.json, route verdicts, re-apply "
             "structural filter on reverted designs, run novelty/"
             "maximin, write passing.json, resubmit next cycle",
    )
    pcf.add_argument("--workdir", type=Path, required=True)
    pcf.add_argument("--max-cycles", type=int, required=True)
    pcf.add_argument("--max-passing", type=int, default=5)
    pcf.add_argument("--novelty-cutoff", type=float, default=10.0)
    pcf.add_argument("--n-designs", type=int, default=None)
    pcf.set_defaults(func=cmd_iterate_collect_finalize)

    # ─── kickoff-prefilter (host, cycle-0 phase 2a) ─────────────────
    pkp = sub.add_parser(
        "kickoff-prefilter",
        help="Host, cycle-0 phase 2a: apply intact filter to "
             "kickoff_distances.json, write cycle_0/prefilter.json",
    )
    pkp.add_argument("--experiment-root", type=Path, required=True)
    pkp.set_defaults(func=cmd_kickoff_prefilter)

    # ─── kickoff-finalize (host, cycle-0 phase 2c) ──────────────────
    pkf = sub.add_parser(
        "kickoff-finalize",
        help="Host, cycle-0 phase 2c: read cycle_0/prefilter.json + "
             "(optional) cycle_0/reversion_results.json, route "
             "verdicts, run novelty/maximin, write cycle_0/passing.json, "
             "submit cycle-1 chains",
    )
    pkf.add_argument("--experiment-root", type=Path, required=True)
    pkf.add_argument("--max-cycles", type=int, required=True)
    pkf.add_argument("--max-passing", type=int, default=5)
    pkf.add_argument("--novelty-cutoff", type=float, default=10.0)
    pkf.add_argument("--n-designs", type=int, default=None)
    pkf.set_defaults(func=cmd_kickoff_finalize)

    # ─── plan-reversions ────────────────────────────────────────────
    ppr = sub.add_parser(
        "plan-reversions",
        help="Stage reverted Boltz YAMLs for contaminated designs "
             "identified by the structural-filter phase",
    )
    ppr.add_argument("--workdir", type=Path, required=True,
                     help="Collect-phase workdir containing plan.json "
                          "and contaminated.json")
    ppr.set_defaults(func=cmd_plan_reversions)

    # ─── predict-reversion-one ──────────────────────────────────────
    pprd = sub.add_parser(
        "predict-reversion-one",
        help="GPU array task: run Boltz on one staged reversion by "
             "array index",
    )
    pprd.add_argument("--workdir", type=Path, required=True,
                      help="Collect-phase workdir containing "
                           "reversion_plan.json")
    pprd.add_argument("--index", type=int, required=True,
                      help="0-based array index into reversion_plan.json's "
                           "entries list")
    pprd.set_defaults(func=cmd_predict_reversion_one)

    # ─── harvest-reversions ─────────────────────────────────────────
    phr = sub.add_parser(
        "harvest-reversions",
        help="Compute reverted metrics + pose_holds/pose_collapses/"
             "new_contamination verdicts for every staged reversion",
    )
    phr.add_argument("--workdir", type=Path, required=True,
                     help="Collect-phase workdir containing "
                          "reversion_plan.json and plan.json")
    phr.add_argument("--structural-ra-eff-cutoff", type=float, default=5.0,
                     help="Reverted ra_eff cutoff for the pose_holds "
                          "verdict.  A reverted pose is classified as "
                          "pose_collapses if reverted_ra_eff_vs_truth "
                          "is >= this value.  Default 5.0 Å.")
    phr.add_argument("--pae-cutoff", type=float, default=10.0,
                     help="PAE cutoff (Å) for ipSAE on reverted "
                          "predictions.  Forwarded to compute_metrics.py.  "
                          "Default 10.0.")
    phr.add_argument("--contact-cutoff", type=float, default=5.0,
                     help="Heavy-atom contact cutoff (Å) for the "
                          "interface contact definition on reverted "
                          "predictions.  Default 5.0.")
    phr.add_argument("--compute-metrics-script", type=Path, default=None,
                     help="Path to compute_metrics.py (default: sibling "
                          "of this script)")
    phr.set_defaults(func=cmd_harvest_reversions)

    pa = sub.add_parser("aggregate",
                        help="Walk the tree and produce summary CSVs")
    pa.add_argument("--experiment-root", type=Path, required=True)
    pa.set_defaults(func=cmd_aggregate)

    paps = sub.add_parser(
        "aggregate-per-sequence",
        help="Collapse per-seed rows into one aggregated row per "
             "unique sequence (median + MAD on continuous metrics, "
             "majority vote on categoricals).  Reads "
             "all_results_multicycle_with_metrics.csv (or the "
             "unmetriced fallback) and writes raw_per_seed_results.csv "
             "+ aggregated_results.csv.  Run AFTER aggregate "
             "(and ideally after compute-final-metrics so confidence "
             "metrics are part of the aggregate).",
    )
    paps.add_argument("--experiment-root", type=Path, required=True)
    paps.set_defaults(func=cmd_aggregate_per_sequence)

    pcfm = sub.add_parser(
        "compute-final-metrics",
        help="Run compute_metrics.py on every design in "
             "all_results_multicycle.csv that passes a final "
             "ranking filter (opt-in post-aggregation step)",
    )
    pcfm.add_argument("--experiment-root", type=Path, required=True,
                      help="Experiment root containing all_results_multicycle.csv "
                           "and cycle_0/plan.json")
    pcfm.add_argument("--rmsd-threshold", type=float, default=5.0,
                      help="Only compute metrics for designs whose "
                           "--metric-column value is strictly less than "
                           "this (default: 5.0 Å)")
    pcfm.add_argument("--metric-column", default="steered_ra_eff_vs_truth",
                      help="Column in all_results_multicycle.csv used for "
                           "the final filter (default: "
                           "steered_ra_eff_vs_truth)")
    pcfm.add_argument("--require-intact", dest="require_intact",
                      action="store_true", default=True,
                      help="Also require steered_receptor_intact == 1 (default: on)")
    pcfm.add_argument("--no-require-intact", dest="require_intact",
                      action="store_false",
                      help="Disable the steered_receptor_intact == 1 filter")
    pcfm.add_argument("--model", default="boltz2",
                      help="Model name passed to compute_metrics.py "
                           "(default: boltz2)")
    pcfm.add_argument("--compute-metrics-script", type=Path, default=None,
                      help="Path to compute_metrics.py "
                           "(default: sibling of this script)")
    pcfm.add_argument("--python-executable", default="python",
                      help="Python interpreter used to run compute_metrics.py "
                           "(default: 'python' — the interpreter on $PATH, "
                           "which inside the boltz container is the one "
                           "with numpy installed).  Use 'python3' or an "
                           "absolute path if your environment differs.")
    pcfm.add_argument("--pae-cutoff", type=float, default=10.0,
                      help="PAE cutoff (Å) for ipSAE.  Forwarded to "
                           "compute_metrics.py.  Default 10 Å follows "
                           "Dunbrack 2025; 15 is also reasonable for "
                           "permissive scoring.  Inspect steered_pae_pass_frac "
                           "in the output to see what fraction of "
                           "interface pairs survived this cutoff for "
                           "each design.")
    pcfm.add_argument("--contact-cutoff", type=float, default=5.0,
                      help="Heavy-atom distance cutoff (Å) for the "
                           "interface contact definition used by the "
                           "mutation-reliance check.  A receptor "
                           "residue is counted as a contact if any of "
                           "its heavy atoms is strictly closer than "
                           "this to any effector heavy atom.  "
                           "steered_n_contacts_on_mutated_positions is then set: "
                           "'clean' (contact set disjoint from the "
                           "row's steering mutations) or 'contaminated' "
                           "(at least one contact residue is a "
                           "steering mutation).  Default 5.0.")
    pcfm.add_argument("--flag-pass-frac-min", type=float, default=0.10,
                      help="Confidence flag: rows with steered_pae_pass_frac "
                           "below this get flagged as 'low_pass_frac' "
                           "(ipSAE summarising too few residue pairs to "
                           "be statistically reliable).  Default 0.10.")
    pcfm.add_argument("--flag-iptm-min", type=float, default=0.30,
                      help="Confidence flag: rows with steered_iptm below "
                           "this get flagged as 'low_iptm' (Boltz not "
                           "confident the interface is real).  "
                           "Default 0.30 — informally used as the AF-"
                           "Multimer 'no interaction' boundary.")
    pcfm.add_argument("--flag-complex-plddt-min", type=float, default=0.70,
                      help="Confidence flag: rows with steered_complex_plddt "
                           "below this get flagged as 'low_plddt' "
                           "(overall structure suspect).  Default 0.70.")
    pcfm.add_argument("--flag-ipae-max", type=float, default=15.0,
                      help="Confidence flag: rows with steered_ipae ABOVE "
                           "this (Å) get flagged as 'high_ipae' (Boltz "
                           "uncertain about inter-chain placement).  "
                           "Default 15.0.")
    pcfm.add_argument("--populate-all", dest="populate_all",
                      action="store_true", default=False,
                      help="P0.2: populate confidence / interface metrics "
                           "on EVERY non-dropped row, not just rows that "
                           "pass the ra_eff + intact filter.  Useful for "
                           "cross-task AUROC analysis where filtered-away "
                           "rows are needed as the wrong-interface side "
                           "of the ROC curve.  reversion_dropped==1 rows "
                           "are still skipped because their metrics are "
                           "meaningless.  Default: off (preserves legacy "
                           "behaviour where only filter-passing rows get "
                           "metrics).")
    pcfm.add_argument("--output-csv", type=Path, default=None,
                      help="Output CSV path "
                           "(default: <experiment-root>/all_results_multicycle_with_metrics.csv)")
    pcfm.add_argument("--skip-existing", dest="skip_existing",
                      action="store_true", default=True,
                      help="Resume: skip rows that already have metrics "
                           "in the output CSV (default: on)")
    pcfm.add_argument("--no-skip-existing", dest="skip_existing",
                      action="store_false",
                      help="Force recomputation of every passing row")
    pcfm.add_argument("--subprocess-timeout", type=int, default=1800,
                      help="Per-design compute_metrics.py timeout in "
                           "seconds (default: 1800)")
    pcfm.add_argument("--verbose", action="store_true",
                      help="Print compute_metrics.py stdout on failure")
    pcfm.set_defaults(func=cmd_compute_final_metrics)

    pkd = sub.add_parser("kickoff-distances",
                         help="Phase 1 (in container) of kickoff: compute "
                              "pose distances for every cycle-0 design vs "
                              "the wild-type prediction")
    pkd.add_argument("--experiment-root", type=Path, required=True)
    pkd.set_defaults(func=cmd_kickoff_distances)

    pk = sub.add_parser("kickoff",
                        help="Phase 2 (outside container) of kickoff: "
                             "filter + maximin + submit cycle 1 chains")
    pk.add_argument("--experiment-root", type=Path, required=True)
    pk.add_argument("--max-cycles", type=int, required=True)
    pk.add_argument("--max-passing", type=int, default=5)
    pk.add_argument("--novelty-cutoff", type=float, default=10.0)
    pk.add_argument("--n-designs", type=int, default=None)
    pk.set_defaults(func=cmd_kickoff)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
