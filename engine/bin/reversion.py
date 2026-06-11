#!/usr/bin/env python3
"""
reversion.py
------------
Helpers for the post-steering reversion validation pass.

Context
=======
After the steering pipeline produces a rescued pose on some steered
sequence, any mutated residue that is contacting the effector in that
pose invalidates the prediction's metrics: the wet-lab construct will
be expressed from the original (pre-steering) sequence and will not
carry the steered residue at that position.  These "contaminating
mutations" are surfaced by compute_metrics.py:classify_mutation_reliance
as `mutated_contact_positions`.

A rescued pose that depends on such mutated contacts is not a trust-
worthy prediction for wet-lab triage.

The reversion pass answers the question:
    "If we reverted the contaminating mutations back to their
    wild-type identity (leaving every other steering mutation in
    place), would Boltz still place the effector in the same pose?"

This file contains the three pure-logic pieces that build and harvest
reverted predictions:

    build_reverted_sequence(steered_seq, cumulative_mutations, positions_to_revert)
        Build a reverted receptor sequence by taking the steered
        sequence and, for each position listed in positions_to_revert,
        substituting back the wild-type residue from the cumulative
        mutation history.  Pure string op.

    write_reversion_plan(workdir, contaminated, plan_meta)
        For each contaminated design, write a Boltz YAML with the
        reverted sequence under workdir/reversions/design_XX/.
        Write a reversion_plan.json manifest listing every staged
        reversion and the metadata needed to harvest its results
        later.  Pure stdlib, runs on the host.

    harvest_reversion_results(workdir, plan_meta, compute_metrics_script)
        After Boltz has run on every staged YAML, read each prediction
        back, compute reverted structural + contact + confidence
        metrics, and return a dict mapping design label to the full
        per-design reverted metric record.  Uses numpy + Bio.Align via
        the steering pipeline's compute_binding_rmsds.  Runs inside
        the singularity container.

Coordinate conventions
======================
- Receptor sequence positions are 1-based everywhere the user sees
  them (mutations.tsv, mutations_chimerax, contact_residues), and
  0-based inside the sequence strings themselves.  This module does
  the 1↔0 conversion at the sequence-building boundary and keeps
  everything else in 1-based.
- The "cumulative mutation history" is a list of (pos1, wt, mut)
  tuples in chronological order (cycle-0 first), as produced by
  boltz2_iterate_steering.read_cumulative_mutations.  If the same
  position was touched twice across cycles the earliest wt is the
  true wild-type.

This module does NOT import numpy or Bio at module level, so it can
be imported safely from host-side code that needs build_reverted_sequence
or write_reversion_plan.  harvest_reversion_results imports numpy-
dependent helpers inside the function body.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ───────────────────────────────────────────────────────────────────────
# Reverted-confidence forwarding — single source of truth
# ───────────────────────────────────────────────────────────────────────
# Confidence-suite columns that boltz2_iterate_steering.py forwards en
# bloc from a reverted prediction's per_seed_records entry into the
# downstream candidate dict / aggregated CSVs.  The structural RMSD
# subset (reverted_ra_eff*, reverted_independent_*), the contact-flag
# fields (reverted_n_contact_residues, reverted_contact_residues,
# reverted_mutated_contact_positions), reverted_prediction_pdb,
# reverted_intact, and reverted_ra_eff_vs_steered are deliberately NOT
# in this list — they are overlaid inline because they either rename to
# canonical columns on pose_holds rows or are diagnostic-only.
#
# This tuple MUST stay in lockstep with the keys assigned into
# `per_seed_records[(label, seed_index)]` inside
# harvest_reversion_results below.  Six historical bugs have come from
# the two drifting apart; defining the list here once and importing it
# in boltz2_iterate_steering.py removes the dual-source surface.
_REVERTED_CONFIDENCE_FIELDS = (
    "reverted_ipsae_min",
    "reverted_ipsae_ab",
    "reverted_ipsae_ba",
    # 15Å iPSAE variants — added so reverted predictions carry the
    # same metric set as steered ones.  harvest_reversion_results
    # harvests these from compute_metrics.py.
    "reverted_ipsae_min_15",
    "reverted_ipsae_ab_15",
    "reverted_ipsae_ba_15",
    "reverted_iptm",
    "reverted_ptm",
    "reverted_actifptm",
    "reverted_af_rank_score",
    "reverted_complex_plddt",
    "reverted_avg_plddt",
    "reverted_ipae",
    "reverted_pae_mean",
    "reverted_pae_pass_frac",
    # Native-PDB-dependent metrics (require --native-pdb to
    # compute_metrics.py — harvest_reversion_results passes it).
    # Without propagation here the values would be harvested but
    # dropped before the aggregator.
    "reverted_interface_plddt",
    "reverted_intact_core",
    "reverted_weighted_jaccard",
    # Structural-Jaccard fields — harvest_reversion_results computes
    # these per seed by re-running find_contact_residues_heavy on each
    # reverted PDB and comparing to the cycle_0 plan's
    # true_interface_idx and initial_wrong_interface_idx.  Mirrors
    # what boltz2_negative_steering._write_initial_multiseed_csv does
    # on the steered side.  Without these the aggregator's
    # reverted_true_jaccard_median / reverted_wrong_jaccard_median /
    # reverted_n_shared_*_median /
    # reverted_n_design_interface_residues_median columns are blank,
    # which in turn means cross_summary's rep_*_median
    # values can't fall back to the reverted side for pose_holds rows
    # and the dispersion plots drop those seeds because tj is None.
    "reverted_true_jaccard",
    "reverted_wrong_jaccard",
    "reverted_n_shared_true",
    "reverted_n_shared_wrong",
    "reverted_n_design_interface_residues",
    "reverted_n_contacts_on_mutated_positions",
)


# ───────────────────────────────────────────────────────────────────────
# Pure sequence-level reversion
# ───────────────────────────────────────────────────────────────────────

def build_reverted_sequence(
    steered_seq: str,
    cumulative_mutations: List[Tuple[int, str, str]],
    positions_to_revert: List[int],
) -> Tuple[str, List[Tuple[int, str, str]]]:
    """Build a reverted receptor sequence.

    Arguments
    ---------
    steered_seq
        The full steered receptor sequence as it was fed into Boltz
        for the rescued pose (i.e. the sequence AFTER all steering
        mutations across all cycles have been applied).  Read this
        from `<design_workdir>/receptor.fasta` — the steering
        pipeline writes it there at plan time.

    cumulative_mutations
        Full chronological mutation history for the design's pathway,
        as `(pos1, wt, mut)` tuples.  Obtained from
        boltz2_iterate_steering.read_cumulative_mutations(experiment_root,
        leaf_label).  Cycle-0 mutations come first.  If the same
        position was edited twice across cycles, the earliest wt is
        the true wild-type and is used for the reversion.

    positions_to_revert
        List of 1-based positions to revert.  Typically this is the
        `positions_to_revert` field from a contaminated.json entry —
        the set of mutated residues that are contacting the effector
        in the steered prediction (the contaminating mutations).

    Returns
    -------
    (reverted_seq, applied_reversions)
        reverted_seq is the steered sequence with the specified
        positions substituted back to their true wild-type.
        applied_reversions is the list of (pos1, mut_at_steer, wt)
        tuples that were actually applied — useful for logging and
        for populating the reversion manifest.  Positions in
        positions_to_revert that could not be found in
        cumulative_mutations (should not happen if the inputs are
        consistent, but belt-and-braces) are skipped and reported
        via a warning to stderr.

    Errors
    ------
    Raises ValueError if steered_seq is empty, if a position in
    positions_to_revert is out of range for the sequence length, or
    if the steered sequence's current residue at a reversion position
    does not match the "mut" in cumulative_mutations for that
    position (which would indicate the inputs are describing different
    designs).
    """
    if not steered_seq:
        raise ValueError("steered_seq is empty")
    n = len(steered_seq)

    # Deduplicate positions_to_revert and validate range
    unique_positions = sorted(set(int(p) for p in positions_to_revert))
    for pos1 in unique_positions:
        if pos1 < 1 or pos1 > n:
            raise ValueError(
                f"position {pos1} is out of range for a "
                f"{n}-residue receptor sequence"
            )

    # Build a pos1 -> wt lookup from the cumulative mutation history.
    # If the same position appears twice (rare — happens when an
    # ancestor edited a position and a descendant re-edited it),
    # keep the FIRST occurrence because that's the true wild-type.
    # Also build a pos1 -> most-recent-mut lookup for the sanity
    # check against the steered sequence.
    pos_to_wt: Dict[int, str] = {}
    pos_to_latest_mut: Dict[int, str] = {}
    for pos1, wt, mut in cumulative_mutations:
        if pos1 not in pos_to_wt:
            pos_to_wt[pos1] = wt
        pos_to_latest_mut[pos1] = mut  # overwrite — keep latest

    seq_list = list(steered_seq)
    applied: List[Tuple[int, str, str]] = []
    missing: List[int] = []
    for pos1 in unique_positions:
        if pos1 not in pos_to_wt:
            missing.append(pos1)
            continue
        wt = pos_to_wt[pos1]
        latest_mut = pos_to_latest_mut[pos1]
        current = seq_list[pos1 - 1]
        if current != latest_mut:
            raise ValueError(
                f"steered sequence mismatch at position {pos1}: "
                f"cumulative mutation history says most-recent "
                f"edit is {latest_mut!r} but steered_seq has "
                f"{current!r}.  The inputs may be describing "
                f"different designs, or the steered FASTA may "
                f"be stale."
            )
        seq_list[pos1 - 1] = wt
        applied.append((pos1, latest_mut, wt))

    if missing:
        sys.stderr.write(
            f"  WARNING: {len(missing)} position(s) in positions_to_revert "
            f"had no entry in cumulative_mutations and were skipped: "
            f"{missing}\n"
        )

    return "".join(seq_list), applied


# ───────────────────────────────────────────────────────────────────────
# Host-side reversion plan staging
# ───────────────────────────────────────────────────────────────────────

def _read_receptor_fasta(fasta_path: Path) -> str:
    """Single-record FASTA reader.  Returns the sequence as a plain
    string (newlines stripped, no header).  Raises if the file has
    zero or multiple records."""
    text = fasta_path.read_text()
    seq_lines: List[str] = []
    n_headers = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            n_headers += 1
            if n_headers > 1:
                raise ValueError(
                    f"{fasta_path} has multiple FASTA records — expected one"
                )
            continue
        seq_lines.append(line)
    if n_headers == 0:
        raise ValueError(f"{fasta_path} has no FASTA header")
    if not seq_lines:
        raise ValueError(f"{fasta_path} has header but no sequence")
    return "".join(seq_lines)


def _read_ca_chain(pdb_path: Path, chain: str):
    """Yield one residue 3-letter code per CA atom on `chain` in
    `pdb_path`, in file order.  Stdlib only; used by the per-seed
    reverted-jaccard length-guard so we can sanity-check
    receptor-chain length without importing numpy or boltz code.
    Empty stream if the chain isn't present."""
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            if (line[21:22].strip() or " ") != chain:
                continue
            yield line[17:20].strip()


def write_reversion_plan(
    workdir: Path,
    contaminated: List[Dict],
    plan_meta: Dict,
) -> Path:
    """Stage reverted Boltz YAMLs for every contaminated design and
    write a reversion_plan.json manifest.

    Deduplication: if multiple contaminated designs revert to the same
    receptor sequence, only one Boltz YAML is staged.  The manifest
    records which designs share each unique reverted sequence so that
    harvest-reversions can copy results to all of them.

    Multi-seed: when plan_meta["num_seeds"] > 1, each unique reverted
    sequence gets num_seeds prediction slots (each with a distinct Boltz
    seed), just like the steered prediction pass.

    Arguments
    ---------
    workdir
        The collect-phase workdir for this cycle.  Reversion artefacts
        go under `workdir/reversions/`.

    contaminated
        List of dicts describing the contaminated designs.  Each MUST
        carry: label, design_idx, design_workdir, cumulative_mutations,
        positions_to_revert, steered_ra_eff_vs_truth,
        steered_contact_residues.

    plan_meta
        Dict with per-cycle info: effector_seq, effector_template_cif,
        pred_receptor_chain, pred_effector_chain, base_seed, num_seeds.

    Returns
    -------
    Path to the written `reversion_plan.json`.
    """
    import hashlib

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from boltz2_negative_steering import write_boltz_yaml  # type: ignore

    reversions_root = workdir / "reversions"
    reversions_root.mkdir(parents=True, exist_ok=True)

    effector_seq = plan_meta["effector_seq"]
    effector_template_cif = plan_meta.get("effector_template_cif")
    if effector_template_cif is not None:
        effector_template_cif = Path(effector_template_cif)
    pred_receptor_chain = plan_meta.get("pred_receptor_chain", "A")
    pred_effector_chain = plan_meta.get("pred_effector_chain", "B")
    base_seed = int(plan_meta.get("base_seed", 0))
    num_seeds = int(plan_meta.get("num_seeds", 1))

    # ── Phase 1: build all reverted sequences ──────────────────────
    # Collect (reverted_seq, metadata) for each contaminated design,
    # then group by sequence.
    per_design: List[Dict] = []
    for c in contaminated:
        label = c["label"]
        design_workdir = Path(c["design_workdir"])
        steered_fasta = design_workdir / "receptor.fasta"
        if not steered_fasta.exists():
            sys.stderr.write(
                f"  SKIP {label}: no receptor.fasta at {steered_fasta}\n"
            )
            continue

        try:
            steered_seq = _read_receptor_fasta(steered_fasta)
        except ValueError as e:
            sys.stderr.write(f"  SKIP {label}: {e}\n")
            continue

        try:
            reverted_seq, applied = build_reverted_sequence(
                steered_seq,
                c["cumulative_mutations"],
                c["positions_to_revert"],
            )
        except ValueError as e:
            sys.stderr.write(
                f"  SKIP {label}: reversion build failed: {e}\n"
            )
            continue

        per_design.append({
            "label": label,
            "design_idx": c.get("design_idx"),
            # P0 audit: forwarded from the contaminated entry; needed
            # to align this steered design with the matching reverted
            # seed in the harvest (Q1=b, Q2=α).
            "seed_index": int(c.get("seed_index", 0) or 0),
            "design_workdir": str(design_workdir),
            "steered_fasta": str(steered_fasta),
            "reverted_seq": reverted_seq,
            "applied": applied,
            "contaminated_entry": c,
        })

    # ── Phase 2: group by unique reverted sequence ─────────────────
    from collections import OrderedDict
    seq_groups: OrderedDict[str, List[Dict]] = OrderedDict()
    for d in per_design:
        seq = d["reverted_seq"]
        seq_groups.setdefault(seq, []).append(d)

    n_unique = len(seq_groups)
    n_total = len(per_design)
    n_deduped = n_total - n_unique
    if n_deduped > 0:
        print(f"  [plan-reversions] {n_total} contaminated → "
              f"{n_unique} unique reverted sequence(s) "
              f"({n_deduped} duplicate(s) deduplicated)")
    else:
        print(f"  [plan-reversions] {n_total} contaminated → "
              f"{n_unique} unique reverted sequence(s) (no duplicates)")

    # ── Phase 3: stage one YAML per unique sequence × num_seeds ────
    manifest_entries: List[Dict] = []
    prediction_idx = 0  # global index across all unique seqs × seeds

    for seq_hash_idx, (reverted_seq, group_members) in enumerate(
        seq_groups.items()
    ):
        seq_hash = hashlib.sha256(
            reverted_seq.encode("ascii")
        ).hexdigest()[:12]
        # Use first member's label for the directory name
        canonical_label = group_members[0]["label"]
        safe_canonical = canonical_label.replace(".", "_").replace("/", "_")
        # P0 audit (multi-seed): all_label_seeds replaces the legacy
        # all_labels list-of-strings.  Each entry pairs a steered
        # label with the seed_index it had on the steered side, so
        # the harvest can align it with the matching reverted seed.
        all_label_seeds = [
            {"label": d["label"], "seed_index": d["seed_index"]}
            for d in group_members
        ]
        # Legacy field kept for any consumer reading the manifest who
        # only needs the label list.
        all_labels = [d["label"] for d in group_members]

        for seed_idx in range(num_seeds):
            if num_seeds == 1:
                rev_dirname = f"rev_{safe_canonical}"
            else:
                rev_dirname = f"rev_{safe_canonical}_s{seed_idx}"

            rev_dir = reversions_root / rev_dirname
            rev_dir.mkdir(parents=True, exist_ok=True)

            # Write receptor FASTA
            (rev_dir / "receptor.fasta").write_text(
                f">receptor_reverted_{rev_dirname}\n{reverted_seq}\n"
            )

            # Write Boltz YAML
            yaml_path = write_boltz_yaml(
                out_dir=rev_dir,
                rec_seq=reverted_seq,
                eff_seq=effector_seq,
                rec_chain=pred_receptor_chain,
                eff_chain=pred_effector_chain,
                effector_template_cif=effector_template_cif,
            )

            # Metadata for the canonical member
            first = group_members[0]
            c = first["contaminated_entry"]
            metadata = {
                "label": canonical_label,
                "all_labels": all_labels,
                "design_idx": first["design_idx"],
                "design_workdir": first["design_workdir"],
                "steered_receptor_fasta": first["steered_fasta"],
                "steered_prediction_pdb": str(
                    Path(first["design_workdir"]) / "prediction.pdb"),
                "reverted_receptor_fasta": str(
                    rev_dir / "receptor.fasta"),
                "positions_to_revert": sorted(set(
                    int(p) for p in c["positions_to_revert"]
                )),
                "applied_reversions": [
                    {"pos1": p, "steered": s, "wild_type": w}
                    for p, s, w in first["applied"]
                ],
                "steered_ra_eff_vs_truth": c.get(
                    "steered_ra_eff_vs_truth"),
                "steered_contact_residues": sorted(
                    c.get("steered_contact_residues", [])),
                "cumulative_mutations": [
                    {"pos1": p, "wt": wt, "mut": mut}
                    for p, wt, mut in c["cumulative_mutations"]
                ],
                "sequence_hash": seq_hash,
                "seed_index": seed_idx,
            }
            (rev_dir / "reversion_metadata.json").write_text(
                json.dumps(metadata, indent=2)
            )

            # Boltz seed: offset from base to avoid collision with
            # steered predictions.  Use 100000+ range as before, plus
            # the prediction_idx for uniqueness across seeds.
            boltz_seed = base_seed + 100000 + prediction_idx + 1

            manifest_entries.append({
                "label": canonical_label,
                "all_labels": all_labels,
                # P0 audit: per-seed-index pairing info — the harvest
                # uses this to assign each reverted-seed result to the
                # right steered per-seed label.
                "all_label_seeds": all_label_seeds,
                "safe_name": rev_dirname,
                "rev_dir": str(rev_dir),
                "yaml": str(yaml_path),
                "reverted_receptor_fasta": str(
                    rev_dir / "receptor.fasta"),
                "design_workdir": first["design_workdir"],
                "positions_to_revert": sorted(set(
                    int(p) for p in c["positions_to_revert"]
                )),
                "n_reversions_applied": len(first["applied"]),
                "sequence_hash": seq_hash,
                "sequence_group_idx": seq_hash_idx,
                "seed_index": seed_idx,
                "boltz_seed": boltz_seed,
            })
            prediction_idx += 1

    manifest = {
        "workdir": str(workdir),
        "n_contaminated": len(contaminated),
        "n_unique_sequences": n_unique,
        "n_staged": len(manifest_entries),
        "num_seeds": num_seeds,
        "pred_receptor_chain": pred_receptor_chain,
        "pred_effector_chain": pred_effector_chain,
        "entries": manifest_entries,
    }
    manifest_path = workdir / "reversion_plan.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


# ───────────────────────────────────────────────────────────────────────
# In-container harvest
# ───────────────────────────────────────────────────────────────────────

def _resolve_cycle0_plan_path(workdir: Path) -> Optional[Path]:
    """Locate cycle_0/plan.json from a reversion workdir, or None."""
    if (workdir / "plan.json").exists():
        try:
            _wd_plan = json.loads((workdir / "plan.json").read_text())
            if int(_wd_plan.get("cycle", 0)) == 0:
                return workdir / "plan.json"
        except Exception:
            pass
    for p in [workdir.parent, workdir.parent.parent]:
        candidate = p / "cycle_0" / "plan.json"
        if candidate.exists():
            return candidate
    return None


def _load_cycle0_plan_context(
    cycle0_plan_path: Optional[Path],
    default_contact_cutoff: float,
) -> Dict:
    """Read cycle_0 plan.json and return per-seed-jaccard context fields."""
    ctx = {
        "true_idx": [],
        "wrong_idx": [],
        "rec_seq": "",
        "pred_rec": "A",
        "pred_eff": "B",
        "contact_cutoff": float(default_contact_cutoff),
    }
    if cycle0_plan_path is None:
        return ctx
    try:
        _plan_dict = json.loads(cycle0_plan_path.read_text())
        ctx["true_idx"] = list(_plan_dict.get("true_interface_idx") or [])
        ctx["wrong_idx"] = list(
            _plan_dict.get("initial_wrong_interface_idx") or []
        )
        ctx["rec_seq"] = (
            _plan_dict.get("wild_type_receptor_seq") or ""
        )
        ctx["pred_rec"] = (
            _plan_dict.get("pred_receptor_chain") or "A"
        )
        ctx["pred_eff"] = (
            _plan_dict.get("pred_effector_chain") or "B"
        )
        try:
            ctx["contact_cutoff"] = float(
                _plan_dict.get("contact_cutoff") or default_contact_cutoff
            )
        except (TypeError, ValueError):
            pass
    except Exception:
        pass
    return ctx


def _locate_reverted_prediction_pdb(rev_dir: Path) -> Optional[Path]:
    """Locate the reverted prediction PDB inside a reversion subdir."""
    candidates = [
        p for p in rev_dir.rglob("*.pdb")
        if "msa" not in p.parts and p.name != "reference.pdb"
    ]
    candidates.sort(key=lambda p: (
        p.name != "prediction.pdb",
        "model_0" not in p.name and "rank_0" not in p.name,
        str(p),
    ))
    if not candidates:
        return None
    return candidates[0]


def _count_pdb_chain_ca_lengths(
    reverted_pdb: Path, pred_rec: str, pred_eff: str
) -> Tuple[int, int]:
    """Return (rec_len, eff_len) CA counts; falls back to two largest chains."""
    from collections import Counter
    ca_counts: Counter = Counter()
    with open(reverted_pdb) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                ca_counts[line[21:22].strip() or " "] += 1
    rec_len = ca_counts.get(pred_rec, 0)
    eff_len = ca_counts.get(pred_eff, 0)
    if rec_len == 0 or eff_len == 0:
        ordered = ca_counts.most_common()
        if len(ordered) >= 2:
            rec_len = rec_len or ordered[0][1]
            eff_len = eff_len or ordered[1][1]
    return rec_len, eff_len


def _build_remaining_mutations_str(
    rev_dir: Path, reverted_positions: Set[int]
) -> str:
    """Comma-joined remaining-mutation positions for the reverted design."""
    cumulative = [
        (m["pos1"], m["wt"], m["mut"])
        for m in json.loads(
            (rev_dir / "reversion_metadata.json").read_text()
        ).get("cumulative_mutations", [])
    ]
    remaining_muts = [
        pos1 for pos1, _wt, _mut in cumulative
        if pos1 not in reverted_positions
    ]
    return ",".join(str(p) for p in sorted(set(remaining_muts)))


def _run_compute_metrics_for_reverted(cmd, per_row_csv: Path):
    """Run compute_metrics.py for a reverted prediction; return cm row dict or error string."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
    except subprocess.TimeoutExpired:
        return "compute_metrics.py timed out"
    if proc.returncode != 0:
        return (f"compute_metrics.py failed rc={proc.returncode}: "
                f"{(proc.stderr or '').strip()[:300]}")
    if not per_row_csv.exists():
        return "compute_metrics.py produced no CSV"
    try:
        with open(per_row_csv) as f:
            cm_rows = list(csv.DictReader(f))
    except Exception as e:
        return f"could not read {per_row_csv}: {e}"
    if not cm_rows:
        return "compute_metrics.py CSV was empty"
    return cm_rows[0]


def _compute_reverted_seed_jaccards(
    reverted_pdb: Path,
    cycle0_true_idx: List[int],
    cycle0_wrong_idx: List[int],
    cycle0_rec_seq: str,
    cycle0_pred_rec: str,
    cycle0_pred_eff: str,
    cycle0_contact_cutoff: float,
    label: str,
    seed_index: int,
    find_contact_residues_heavy,
    jaccard,
) -> Tuple[object, object, object, object, object]:
    """Per-seed structural Jaccard against cycle_0 true/wrong interface indices."""
    rev_true_jaccard: object = ""
    rev_wrong_jaccard: object = ""
    rev_n_shared_true: object = ""
    rev_n_shared_wrong: object = ""
    rev_n_design_iface: object = ""
    if cycle0_true_idx and reverted_pdb.exists():
        try:
            # IMPORTANT: do NOT pass expected_rec_seq=cycle0_rec_seq
            # here.  cycle0's wild_type_receptor_seq is the
            # pre-steering MPNN sequence; the reverted prediction
            # carries surviving steering mutations and therefore
            # WILL differ from it at every position the reversion
            # pass left in place.  Passing the wild-type as the
            # expected sequence makes the guard inside
            # find_contact_residues_heavy raise ValueError on every
            # pose_holds row (verified empirically against
            # design_7_seq_3 sg=0: 5 surviving mutations →
            # ValueError → silently swallowed → blank jaccards).
            #
            # The guard exists to catch index-misalignment between
            # the per-seed contact set and cycle0_true_idx.  AA-level
            # match isn't the right invariant for that here; LENGTH
            # match is.  We enforce length below explicitly so the
            # safety property is retained without breaking on
            # expected mutations.
            seed_contacts = find_contact_residues_heavy(
                reverted_pdb,
                cycle0_pred_rec, cycle0_pred_eff,
                cycle0_contact_cutoff,
                expected_rec_seq=None,
            )
            # Length-only safety net: index N in seed_contact_idx
            # must refer to the same position as index N in
            # cycle0_true_idx.  If chain length differs (wrong PDB
            # picked up, chain layout changed) the indices aren't
            # comparable and the jaccard would be garbage.
            if cycle0_rec_seq:
                actual_len = sum(
                    1 for _ in _read_ca_chain(
                        reverted_pdb, cycle0_pred_rec
                    )
                )
                if actual_len != len(cycle0_rec_seq):
                    raise ValueError(
                        f"Receptor chain length mismatch in "
                        f"{reverted_pdb}: actual={actual_len} "
                        f"expected={len(cycle0_rec_seq)} — "
                        f"index alignment with cycle0_true_idx "
                        f"would be invalid"
                    )
            seed_contact_idx = sorted(i for i, _d in seed_contacts)
            rev_n_design_iface = len(seed_contact_idx)
            rev_n_shared_true = len(
                set(seed_contact_idx) & set(cycle0_true_idx)
            )
            rev_true_jaccard = jaccard(seed_contact_idx, cycle0_true_idx)
            if cycle0_wrong_idx:
                rev_n_shared_wrong = len(
                    set(seed_contact_idx) & set(cycle0_wrong_idx)
                )
                rev_wrong_jaccard = jaccard(
                    seed_contact_idx, cycle0_wrong_idx
                )
        except Exception as e:
            # Soft failure: leave all jaccard fields blank, but
            # log to stderr so the next failure doesn't disappear
            # silently the way the original ValueError did.
            sys.stderr.write(
                f"  WARN reverted-jaccard {label} s{seed_index}: "
                f"{type(e).__name__}: {e}\n"
            )
    return (rev_true_jaccard, rev_wrong_jaccard, rev_n_shared_true,
            rev_n_shared_wrong, rev_n_design_iface)


def _pair_per_seed_records_to_steered_labels(
    per_seed_records: Dict[Tuple[str, int], Dict],
    manifest: Dict,
) -> Dict[str, Dict]:
    """Pair per-(canonical_label, seed) reverted records to each steered label."""
    results: Dict[str, Dict] = {}
    for entry in manifest.get("entries", []):
        rev_seed_idx = int(entry.get("seed_index", 0) or 0)
        canonical = entry["label"]
        # Prefer the new all_label_seeds field; fall back to all_labels
        # for back-compat with manifests written before this fix.
        all_label_seeds = entry.get("all_label_seeds")
        if all_label_seeds is None:
            all_label_seeds = [
                {"label": L, "seed_index": rev_seed_idx}
                for L in entry.get("all_labels", [canonical])
            ]
        for member in all_label_seeds:
            steered_label = member["label"]
            steered_seed_idx = int(member.get("seed_index", 0) or 0)
            # Pair only when the reverted seed_index matches this
            # steered seed_index.
            if steered_seed_idx != rev_seed_idx:
                continue
            rec = per_seed_records.get((canonical, rev_seed_idx))
            if rec is None:
                rec = {
                    "error": (
                        f"no reverted record for canonical={canonical} "
                        f"at seed_index={rev_seed_idx} (harvest failed "
                        f"for that seed)"
                    ),
                }
            results[steered_label] = dict(rec)
    return results


def _persist_reversion_results_to_disk(
    workdir: Path,
    results: Dict[str, Dict],
    per_seed_records: Dict[Tuple[str, int], Dict],
) -> None:
    """Write reversion_results.json and reversion_results_per_seed.json."""
    results_path = workdir / "reversion_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    per_seed_path = workdir / "reversion_results_per_seed.json"
    per_seed_jsonable: Dict[str, Dict[str, Dict]] = {}
    for (lbl, sidx), rec in per_seed_records.items():
        per_seed_jsonable.setdefault(lbl, {})[str(sidx)] = rec
    per_seed_path.write_text(
        json.dumps(per_seed_jsonable, indent=2, default=str)
    )


def harvest_reversion_results(
    workdir: Path,
    ground_truth_pdb: Path,
    truth_receptor_chain: str,
    truth_effector_chain: str,
    compute_metrics_script: Path,
    pae_cutoff: float = 10.0,
    contact_cutoff: float = 5.0,
) -> Dict[str, Dict]:
    """Read every staged reverted prediction, compute metrics, and
    return a per-label result dict.

    This function runs INSIDE the singularity container because it
    needs numpy + Bio.Align for Kabsch/sequence alignment via
    boltz2_negative_steering.compute_binding_rmsds.  It also shells
    out to compute_metrics.py for the confidence / contact analysis,
    mirroring what cmd_compute_final_metrics does per row.

    Arguments
    ---------
    workdir                    Collect workdir containing reversion_plan.json.
    ground_truth_pdb           RFDiffusion design PDB for design_3-style
                               comparison of the reverted pose.
    truth_receptor_chain       Receptor chain letter in ground_truth_pdb.
    truth_effector_chain       Effector chain letter in ground_truth_pdb.
    compute_metrics_script     Path to compute_metrics.py (for confidence
                               metrics on reverted predictions).
    pae_cutoff                 Forwarded to compute_metrics.py (ipSAE).
    contact_cutoff             Heavy-atom contact cutoff for the contact
                               analysis on reverted predictions.

    Returns
    -------
    Dict keyed by pathway label, each value a dict with:
        reverted_ra_eff_vs_truth         float
        reverted_ra_eff_vs_steered       float
        reverted_independent_receptor_rmsd   float
        reverted_independent_effector_rmsd   float
        reverted_intact                   bool
        reverted_n_contact_residues       int
        reverted_contact_residues         List[int]
        reverted_mutated_contact_positions  List[int]
        reverted_prediction_pdb           str  (path)
        plus every entry in `_REVERTED_CONFIDENCE_FIELDS` (the
        confidence-suite forwarded en bloc to downstream CSVs).
    If a reverted prediction failed or is missing, the value is a
    dict with {"error": "..."} and no metric fields.
    """
    # Lazy numpy imports — only need them in the harvest phase
    import numpy as np  # noqa: F401

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from boltz2_negative_steering import (  # type: ignore
        binding_rmsds,
        find_contact_residues_heavy,
        jaccard,
    )

    manifest_path = workdir / "reversion_plan.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found — run plan-reversions first"
        )
    manifest = json.loads(manifest_path.read_text())

    # v8 Level 1: resolve the cycle_0 plan.json path so we can pass
    # it as the effector-atom-filter source to compute_metrics.py.
    # The reversion workdir layout mirrors the steering workdir:
    #   - Cycle 0 (flat layout):     workdir == <root>/  OR workdir == <root>/cycle_0/
    #                                 plan.json lives in workdir itself.
    #   - Cycle 1+ (nested layout):  workdir == <root>/cycle_N/pathway_<label>/
    #                                 plan.json for this pathway lives in workdir;
    #                                 cycle_0/plan.json lives at <root>/cycle_0/plan.json.
    # For the atom filter we specifically want cycle_0/plan.json because
    # the ground truth + true_interface that define the interface atoms
    # are only set there.
    cycle0_plan_path = _resolve_cycle0_plan_path(workdir)
    # If still not found, proceed without filter — contact detection
    # falls back to legacy unfiltered behaviour on reverted predictions.

    # Load the cycle_0 plan (when locatable) to extract true_interface_idx
    # and initial_wrong_interface_idx — these drive per-seed
    # structural-Jaccard computation against each reverted prediction.
    # Steered predictions already get this treatment in
    # boltz2_negative_steering._write_initial_multiseed_csv /
    # _write_initial_only_csv; reverted predictions need the same so
    # the aggregator can compute reverted_true_jaccard_median /
    # reverted_wrong_jaccard_median etc.  When the plan can't be loaded
    # (legacy runs, parsing error) the per-seed jaccard fields stay
    # blank and the aggregator emits "" for them.
    _cycle0_ctx = _load_cycle0_plan_context(cycle0_plan_path, contact_cutoff)
    cycle0_true_idx: List[int] = _cycle0_ctx["true_idx"]
    cycle0_wrong_idx: List[int] = _cycle0_ctx["wrong_idx"]
    cycle0_rec_seq: str = _cycle0_ctx["rec_seq"]
    cycle0_pred_rec: str = _cycle0_ctx["pred_rec"]
    cycle0_pred_eff: str = _cycle0_ctx["pred_eff"]
    cycle0_contact_cutoff: float = _cycle0_ctx["contact_cutoff"]

    # P0 audit (multi-seed): build per-seed records first, then pair
    # them with steered labels by matching seed_index.  Legacy code
    # wrote results[label] in a loop with the same label, so for
    # num_seeds > 1 only the LAST seed's record survived and the
    # other seeds' GPU work was discarded.  See the audit notes.
    #
    # per_seed_records[(canonical_label, seed_index)] = result_dict
    per_seed_records: Dict[Tuple[str, int], Dict] = {}

    # results is the legacy public return: {steered_label: result_dict}
    # Built at the end by walking all_label_seeds and looking up the
    # matching reverted seed_index per steered label.
    results: Dict[str, Dict] = {}

    for entry in manifest.get("entries", []):
        label = entry["label"]
        seed_index = int(entry.get("seed_index", 0) or 0)
        rev_dir = Path(entry["rev_dir"])
        design_workdir = Path(entry["design_workdir"])
        steered_pdb = design_workdir / "prediction.pdb"

        # Locate the reverted prediction PDB.  Boltz writes to
        # `<rev_dir>/predictions/<stem>/pdb/<stem>_model_0.pdb`
        # or a similar nested path; to stay independent of Boltz's
        # exact layout we glob for any .pdb inside rev_dir that isn't
        # a reference.
        reverted_pdb = _locate_reverted_prediction_pdb(rev_dir)
        if reverted_pdb is None:
            per_seed_records[(label, seed_index)] = {
                "error": f"no reverted prediction PDB found under {rev_dir}"
            }
            continue

        pred_rec = manifest.get("pred_receptor_chain", "A")
        pred_eff = manifest.get("pred_effector_chain", "B")

        # --- Structural metrics vs ground truth ---
        try:
            ra_eff_vs_truth, ind_rec, ind_eff = binding_rmsds(
                pred_pdb=reverted_pdb,
                truth_pdb=ground_truth_pdb,
                pred_rec_chain=pred_rec,
                pred_eff_chain=pred_eff,
                truth_rec_chain=truth_receptor_chain,
                truth_eff_chain=truth_effector_chain,
            )
        except Exception as e:
            per_seed_records[(label, seed_index)] = {
                "error": f"binding_rmsds vs truth failed: {e}"
            }
            continue

        # --- Structural metrics vs steered parent ---
        # Same function, just with the steered prediction as the
        # "truth" reference.  This measures how far the reverted
        # pose has drifted from its own steered parent.
        try:
            ra_eff_vs_steered, _, _ = binding_rmsds(
                pred_pdb=reverted_pdb,
                truth_pdb=steered_pdb,
                pred_rec_chain=pred_rec,
                pred_eff_chain=pred_eff,
                truth_rec_chain=pred_rec,
                truth_eff_chain=pred_eff,
            )
        except Exception as e:
            ra_eff_vs_steered = float("nan")
            sys.stderr.write(
                f"  WARN {label}: binding_rmsds vs steered failed: "
                f"{e} — continuing with ra_eff_vs_steered=nan\n"
            )

        intact = bool(ind_rec <= 5.0 and ind_eff <= 5.0)

        # --- Confidence / contact metrics via compute_metrics.py ---
        per_row_csv = rev_dir / "metrics.csv"

        # Count CAs for chain-lengths arg
        rec_len, eff_len = _count_pdb_chain_ca_lengths(
            reverted_pdb, pred_rec, pred_eff
        )

        # mutations_chimerax for the REVERTED design = the original
        # steered mutations MINUS the reverted positions, because
        # those positions now carry wild-type again.
        reverted_positions = set(entry.get("positions_to_revert") or [])
        remaining_muts_str = _build_remaining_mutations_str(
            rev_dir, reverted_positions
        )

        cmd = [
            sys.executable, str(compute_metrics_script),
            "--model", "boltz2",
            "--prediction-dir", str(rev_dir),
            "--chain-lengths", str(rec_len), str(eff_len),
            "--output-csv", str(per_row_csv),
            "--pae-cutoff", str(pae_cutoff),
            "--receptor-chain", pred_rec,
            "--effector-chain", pred_eff,
            "--contact-cutoff", str(contact_cutoff),
            # Pass the ground-truth PDB so compute_metrics.py also
            # populates intact_core, weighted_jaccard, and
            # interface_plddt for the reverted prediction.  Without
            # this, those columns are emitted blank.  The steered
            # pipeline already passes --native-pdb (cycle_0 ground
            # truth) for steered predictions, so this brings reverted
            # predictions in line.
            "--native-pdb", str(ground_truth_pdb),
        ]
        # v8 Level 1: restrict reverted-pose contact detection to the
        # interface effector atoms defined in cycle_0/plan.json.  If
        # the plan couldn't be located (legacy runs, unusual layouts),
        # the filter is omitted and contact detection falls back to
        # unfiltered behaviour.
        if cycle0_plan_path is not None:
            cmd.extend(["--effector-atom-filter-json", str(cycle0_plan_path)])
        if remaining_muts_str:
            cmd.extend(["--mutated-positions", remaining_muts_str])

        cm_or_err = _run_compute_metrics_for_reverted(cmd, per_row_csv)
        if isinstance(cm_or_err, str):
            per_seed_records[(label, seed_index)] = {"error": cm_or_err}
            continue
        cm = cm_or_err

        def _fl(k: str) -> Optional[float]:
            v = cm.get(k, "")
            if v in ("", None):
                return None
            try:
                return float(v)
            except ValueError:
                return None

        def _int(k: str) -> Optional[int]:
            v = cm.get(k, "")
            if v in ("", None):
                return None
            try:
                return int(v)
            except ValueError:
                return None

        # CONTAMINATION RULE on the reverted prediction
        # ==============================================
        # Any remaining mutation (i.e. any position in `remaining_muts`
        # — cumulative mutations minus the positions this pass reverted)
        # that contacts the effector in the reverted prediction is
        # new contamination: the rescue pose is held in place by a
        # residue the construct will not carry.
        #
        # This is the SAME rule as build-contaminated uses on the
        # steered prediction.  The protected-set-based fields are kept
        # for diagnostics only; they do NOT drive the verdict.
        reverted_contact_str = cm.get("contact_residues", "") or ""
        reverted_contact_set = set(
            int(p) for p in reverted_contact_str.split(",")
            if p.strip().isdigit()
        )
        remaining_muts_set = set(
            int(p) for p in (remaining_muts_str or "").split(",")
            if p.strip().isdigit()
        )
        mutated_contact_positions = sorted(
            remaining_muts_set & reverted_contact_set
        )

        # Per-seed structural Jaccard against the ground-truth interface.
        # Mirrors what _write_initial_multiseed_csv does on the steered
        # side: re-run find_contact_residues_heavy on THIS seed's PDB to
        # get 0-based positional contact indices, then jaccard against
        # the cycle_0 plan's true_interface_idx and
        # initial_wrong_interface_idx.  When the cycle_0 plan couldn't
        # be loaded (defaults: empty lists), all jaccard fields stay
        # blank — DictWriter writes "" rather than dropping the row.
        (rev_true_jaccard, rev_wrong_jaccard, rev_n_shared_true,
         rev_n_shared_wrong, rev_n_design_iface) = (
            _compute_reverted_seed_jaccards(
                reverted_pdb,
                cycle0_true_idx, cycle0_wrong_idx, cycle0_rec_seq,
                cycle0_pred_rec, cycle0_pred_eff, cycle0_contact_cutoff,
                label, seed_index,
                find_contact_residues_heavy, jaccard,
            )
        )

        per_seed_records[(label, seed_index)] = {
            "reverted_prediction_pdb": str(reverted_pdb),
            "reverted_ra_eff_vs_truth": float(ra_eff_vs_truth),
            "reverted_ra_eff_vs_steered": (
                None if ra_eff_vs_steered != ra_eff_vs_steered  # NaN
                else float(ra_eff_vs_steered)
            ),
            "reverted_independent_receptor_rmsd": float(ind_rec),
            "reverted_independent_effector_rmsd": float(ind_eff),
            "reverted_intact": intact,
            "reverted_n_contact_residues": _int("n_contact_residues"),
            "reverted_contact_residues": cm.get("contact_residues", ""),
            # The authoritative contamination signal for the reverted
            # prediction — drives classify_reversion_verdict.
            "reverted_mutated_contact_positions": mutated_contact_positions,
            "reverted_n_contacts_on_mutated_positions": len(
                mutated_contact_positions
            ),
            # Per-seed structural Jaccard fields — computed against
            # cycle_0 plan's true / initial-wrong interface indices.
            "reverted_true_jaccard": rev_true_jaccard,
            "reverted_wrong_jaccard": rev_wrong_jaccard,
            "reverted_n_shared_true": rev_n_shared_true,
            "reverted_n_shared_wrong": rev_n_shared_wrong,
            "reverted_n_design_interface_residues": rev_n_design_iface,
            "reverted_ipsae_min": _fl("ipsae_min"),
            "reverted_ipsae_ab": _fl("ipsae_ab"),
            "reverted_ipsae_ba": _fl("ipsae_ba"),
            # 15Å iPSAE variants — same Dunbrack iPSAE machinery as the
            # 10Å trio above, evaluated at a more permissive PAE
            # cutoff.  Always emitted by compute_metrics.py.
            "reverted_ipsae_min_15": _fl("ipsae_min_15"),
            "reverted_ipsae_ab_15": _fl("ipsae_ab_15"),
            "reverted_ipsae_ba_15": _fl("ipsae_ba_15"),
            "reverted_iptm": _fl("iptm"),
            "reverted_ptm": _fl("ptm"),
            "reverted_actifptm": _fl("actifptm"),
            "reverted_af_rank_score": _fl("af_rank_score"),
            "reverted_complex_plddt": _fl("complex_plddt"),
            "reverted_avg_plddt": _fl("avg_plddt"),
            "reverted_ipae": _fl("ipae"),
            "reverted_pae_mean": _fl("pae_mean"),
            "reverted_pae_pass_frac": _fl("pae_pass_frac"),
            # Native-PDB-dependent metrics (require --native-pdb to
            # compute_metrics.py — passed in the cmd above).  Mirror
            # the steered side so plots can compare interface_plddt /
            # intact_core / weighted_jaccard between stages.
            "reverted_interface_plddt": _fl("interface_plddt"),
            "reverted_intact_core":     _int("intact_core"),
            "reverted_weighted_jaccard": _fl("weighted_jaccard"),
        }

    # ── Pair per-seed records back to steered labels (P0 audit) ──────
    # The legacy block here picked the canonical label's record (which
    # was already collapsed to one seed by the loop overwrite bug) and
    # copied it verbatim to every shared label.  We now have one
    # record PER (canonical_label, seed_index), and we pair each
    # steered label with the matching reverted seed_index.
    #
    # Each manifest entry covers (unique_reverted_seq, reverted_seed_idx).
    # Its all_label_seeds field lists the steered labels that share
    # this reverted sequence, each with its own steered seed_index.
    # A steered label gets the reverted record at the matching
    # seed_index — so design_05_s0 ↔ rev_seed_0, design_05_s1 ↔
    # rev_seed_1, etc.  See the audit conversation Q2=α.
    #
    # Steered labels whose seed_index has no corresponding reverted
    # record (would only happen if the harvest failed for that
    # particular reverted seed) get a stub error record so the
    # downstream reader doesn't see a blank.
    results = _pair_per_seed_records_to_steered_labels(
        per_seed_records, manifest
    )

    # Persist to disk alongside the manifest for traceability.
    # Two files: the legacy {label: result} for back-compat with the
    # finalize readers, and a richer {label: {seed_index: result}}
    # for any post-hoc analysis that needs the per-seed structure.
    _persist_reversion_results_to_disk(workdir, results, per_seed_records)
    return results


# ───────────────────────────────────────────────────────────────────────
# Verdict classification
# ───────────────────────────────────────────────────────────────────────

def classify_reversion_verdict(
    steered: Dict,
    reverted: Dict,
    structural_ra_eff_cutoff: float,
    structural_intact_required: bool = True,
    contamination_gating_positions: Optional[Set[int]] = None,
) -> Tuple[str, str]:
    """Classify the outcome of the reversion pass for one design.

    Structural filter is repeated on the reverted prediction:
        reverted must be intact AND reverted_ra_eff_vs_truth < cutoff

    Contamination rule on the reverted prediction:

      A mutated residue contacting the effector in the reverted
      prediction counts as new_contamination *only if that position
      falls inside `contamination_gating_positions`* — i.e. inside
      the union of the RFDiffusion design region and the true
      interface.

      The gating mirrors the asymmetric policy used at the pre-
      reversion (build-contaminated) stage: mutations at positions
      outside the gated set are already accepted as valid steering
      contacts on the steered pose (they are not reverted, and they
      persist into the reverted sequence on purpose), so flagging
      those same positions as "new contamination" on the reverted
      pose would be inconsistent with the pre-reversion policy.
      Only contacts at design-region-or-true-interface positions
      represent genuinely new contamination — i.e. the reverted
      pose has moved relative to the steered pose and a mutated
      residue has landed in the region we actually care about.

      If `contamination_gating_positions` is None (legacy behaviour),
      the gate is disabled and any mutated contact fires
      new_contamination, matching the pre-notes11 semantics.

    Verdict resolution:
      - reverted fails the structural filter  → 'pose_collapses'
      - reverted has mutated contact residues
        inside the gating set                 → 'new_contamination'
      - otherwise                             → 'pose_holds'
    """
    if "error" in reverted:
        return ("pose_collapses",
                f"reverted prediction failed: {reverted['error']}")

    reverted_intact = bool(reverted.get("reverted_intact", False))
    reverted_ra_eff = reverted.get("reverted_ra_eff_vs_truth")

    if structural_intact_required and not reverted_intact:
        return ("pose_collapses",
                f"reverted not intact (rec_rmsd="
                f"{reverted.get('reverted_independent_receptor_rmsd')}, "
                f"eff_rmsd="
                f"{reverted.get('reverted_independent_effector_rmsd')})")

    if reverted_ra_eff is None:
        return ("pose_collapses",
                "reverted_ra_eff_vs_truth unavailable")
    if reverted_ra_eff >= structural_ra_eff_cutoff:
        return ("pose_collapses",
                f"reverted ra_eff {reverted_ra_eff:.2f} ≥ "
                f"cutoff {structural_ra_eff_cutoff:.2f}")

    mutated_contacts = reverted.get("reverted_mutated_contact_positions") or []
    if contamination_gating_positions is not None:
        # Only flag positions inside the gated set as contamination.
        # Positions outside the gate are acceptable by construction
        # (they passed the pre-reversion filter on the steered pose,
        # so they are acceptable on the reverted pose too).
        gated_contacts = [p for p in mutated_contacts
                          if p in contamination_gating_positions]
        ungated_contacts = [p for p in mutated_contacts
                            if p not in contamination_gating_positions]
        if gated_contacts:
            reason = (f"reverted pose has {len(gated_contacts)} mutated "
                      f"residue(s) contacting the effector at gated "
                      f"position(s) (design region ∪ true interface): "
                      f"{gated_contacts}")
            if ungated_contacts:
                reason += (f"; also touching ungated positions "
                           f"{ungated_contacts} (acceptable — matches "
                           f"pre-reversion policy)")
            return ("new_contamination", reason)
        if ungated_contacts:
            # No gated contamination, but ungated contacts exist.
            # Record them in the reason for transparency but still
            # return pose_holds.
            return ("pose_holds",
                    f"reverted pose holds structurally and cleanly; "
                    f"ungated mutated contacts at {ungated_contacts} "
                    f"accepted under gated-contamination policy")
        return ("pose_holds", "reverted pose holds structurally and cleanly")

    # Legacy path: no gating set supplied, flag any mutated contact
    if mutated_contacts:
        return ("new_contamination",
                f"reverted pose still has {len(mutated_contacts)} mutated "
                f"residue(s) contacting the effector: {mutated_contacts}")

    return ("pose_holds", "reverted pose holds structurally and cleanly")
