#!/usr/bin/env python3
"""
reversion.py
------------
Library and CLI for the post-steering reversion validation pass.

A mutated residue touching the effector in a rescued pose invalidates
that prediction's metrics, because the wet-lab construct is expressed
from the pre-steering sequence and never carries it.  Such residues are
the "contaminating mutations", reported by
compute_metrics.py:classify_mutation_reliance as
`mutated_contact_positions`.

The reversion pass asks one question.  If we put the contaminating
mutations back to wild-type and leave every other steering mutation in
place, does Boltz still place the effector in the same pose?

Four CLI stages answer it, each carrying its own contract below.
build-contaminated flags the contaminated designs, plan-reversions
stages their reverted Boltz YAMLs, predict-reversion-one runs Boltz on
one of them, and harvest-reversions reads the results back and
classifies a verdict.

Receptor positions are 1-based everywhere but inside sequence strings.
Nothing here imports numpy or Bio at module level, so host-side code can
import build_reverted_sequence or write_reversion_plan safely.
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
# negsteer_aggregate.py imports this tuple and forwards these columns as
# one block into the downstream candidate dict and the aggregated CSVs.
#
# Left out on purpose and overlaid inline instead: reverted_ra_eff*,
# reverted_independent_*, reverted_n_contact_residues,
# reverted_contact_residues, reverted_mutated_contact_positions,
# reverted_prediction_pdb and reverted_intact.  Each either renames to a
# canonical column on pose_holds rows or is diagnostic only.
#
# Keep it in step with the keys assigned into
# per_seed_records[(label, seed_index)] in harvest_reversion_results
# below.  Six bugs came from the two drifting apart.
_REVERTED_CONFIDENCE_FIELDS = (
    "reverted_ipsae_min",
    "reverted_ipsae_ab",
    "reverted_ipsae_ba",
    # 15 Å iPSAE variants, the same Dunbrack calculation at a more
    # permissive PAE cutoff, so reverted predictions carry the same
    # metric set as steered ones.
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
    # These three need --native-pdb passed to compute_metrics.py, which
    # harvest_reversion_results does.  Without forwarding them here the
    # values would be harvested and then dropped before the aggregator.
    "reverted_interface_plddt",
    "reverted_intact_core",
    "reverted_weighted_jaccard",
    # Structural-Jaccard fields, computed per seed against the cycle_0
    # plan's true_interface_idx and initial_wrong_interface_idx.  Drop
    # them and the aggregator's reverted_*_jaccard_median,
    # reverted_n_shared_*_median and
    # reverted_n_design_interface_residues_median columns go blank,
    # cross_summary's rep_*_median then cannot fall back to the reverted
    # side on pose_holds rows, and the dispersion plots drop those seeds
    # because tj is None.
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

    `steered_seq` is the receptor as Boltz saw it for the rescued pose,
    read from the `<design_workdir>/receptor.fasta` the steering
    pipeline writes at plan time.  `cumulative_mutations` is the
    design's chronological `(pos1, wt, mut)` history, and where a
    position was edited twice the earliest wt is the true wild-type.
    `positions_to_revert` holds 1-based positions, normally a
    contaminated.json entry's `positions_to_revert` field.

    Returns (reverted_seq, applied_reversions), the latter the
    (pos1, mut_at_steer, wt) tuples actually applied, for logging and
    the manifest.  A requested position missing from
    cumulative_mutations is skipped with a stderr warning, which
    consistent inputs never trigger.

    Raises ValueError on an empty steered_seq, an out-of-range position,
    or a steered residue that disagrees with the "mut" in
    cumulative_mutations.  The last case means the two inputs describe
    different designs.
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

    # Keep the FIRST wt per position, which is the true wild-type when
    # an ancestor and a descendant both edited it.  pos_to_latest_mut
    # holds the most recent residue, for the check against steered_seq.
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
    """Yield one residue 3-letter code per CA atom on `chain`, in file
    order.  Stdlib only, so the reverted-jaccard length guard can check
    receptor-chain length without numpy or boltz code.  Yields nothing
    if the chain is absent."""
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

    Designs reverting to the same receptor sequence share one staged
    YAML, and the manifest records which designs share it so
    harvest-reversions can copy the result to all of them.  When
    plan_meta["num_seeds"] > 1, each unique reverted sequence gets
    num_seeds prediction slots with their own Boltz seeds, matching the
    steered prediction pass.

    `workdir` is this cycle's collect-phase workdir, and artefacts go
    under `workdir/reversions/`.  Each `contaminated` dict must carry
    label, design_idx, design_workdir, cumulative_mutations,
    positions_to_revert, steered_ra_eff_vs_truth and
    steered_contact_residues.  `plan_meta` must carry effector_seq,
    effector_template_cif, pred_receptor_chain, pred_effector_chain,
    base_seed, num_seeds and boltz_constraints_block.

    Returns the path to the written `reversion_plan.json`.
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
    boltz_constraints_block = plan_meta.get("boltz_constraints_block")

    # ── Phase 1: build all reverted sequences ──────────────────────
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
            # The harvest pairs each steered design with the reverted
            # prediction of the same seed_index, so the index travels
            # with the design into the manifest.
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
        # all_label_seeds replaces an older plain label list that lost
        # the seed information.  Each entry pairs a steered label with
        # its steered-side seed_index for the harvest to match on.
        all_label_seeds = [
            {"label": d["label"], "seed_index": d["seed_index"]}
            for d in group_members
        ]
        # Older field, kept for any manifest reader that only needs the
        # label list.
        all_labels = [d["label"] for d in group_members]

        for seed_idx in range(num_seeds):
            if num_seeds == 1:
                rev_dirname = f"rev_{safe_canonical}"
            else:
                rev_dirname = f"rev_{safe_canonical}_s{seed_idx}"

            rev_dir = reversions_root / rev_dirname
            rev_dir.mkdir(parents=True, exist_ok=True)

            (rev_dir / "receptor.fasta").write_text(
                f">receptor_reverted_{rev_dirname}\n{reverted_seq}\n"
            )

            yaml_path = write_boltz_yaml(
                out_dir=rev_dir,
                rec_seq=reverted_seq,
                eff_seq=effector_seq,
                rec_chain=pred_receptor_chain,
                eff_chain=pred_effector_chain,
                effector_template_cif=effector_template_cif,
                constraints_block=boltz_constraints_block,
            )

            # Metadata for the canonical member.
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

            # Offset the Boltz seed by 100000 so it cannot collide with
            # a steered prediction's seed.  Adding prediction_idx keeps
            # each staged seed distinct.
            boltz_seed = base_seed + 100000 + prediction_idx + 1

            manifest_entries.append({
                "label": canonical_label,
                "all_labels": all_labels,
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
    """Return the (receptor, effector) CA counts. A prediction that does not use
    the named chains falls back to the two largest."""
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
    """Run compute_metrics.py on a reverted prediction. Returns its row as a
    dict, or an error string naming what went wrong."""
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
            # Do NOT pass expected_rec_seq=cycle0_rec_seq here.
            # cycle_0's wild_type_receptor_seq is the pre-steering MPNN
            # sequence, while the reverted prediction keeps the steering
            # mutations the pass left in place, so the two differ there.
            # Passing it made the guard inside
            # find_contact_residues_heavy raise ValueError on every
            # pose_holds row.  Verified against design_7_seq_3 sg=0,
            # which has 5 surviving mutations.  The ValueError was
            # swallowed below and the jaccards came out blank.
            #
            # The guard catches index misalignment between the per-seed
            # contact set and cycle0_true_idx.  Amino-acid identity is
            # the wrong invariant for that and LENGTH is the right one,
            # which the check below enforces instead.
            seed_contacts = find_contact_residues_heavy(
                reverted_pdb,
                cycle0_pred_rec, cycle0_pred_eff,
                cycle0_contact_cutoff,
                expected_rec_seq=None,
            )
            # Index N in seed_contact_idx must mean the same position as
            # index N in cycle0_true_idx.  A different chain length
            # means the wrong PDB or a changed chain layout, so the
            # jaccard would be meaningless.
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
            # Leave every jaccard field blank, but log to stderr so the
            # next failure does not vanish the way the original
            # ValueError did.
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
        # Fall back to all_labels so manifests written before per-seed
        # pairing existed still load.
        all_label_seeds = entry.get("all_label_seeds")
        if all_label_seeds is None:
            all_label_seeds = [
                {"label": L, "seed_index": rev_seed_idx}
                for L in entry.get("all_labels", [canonical])
            ]
        for member in all_label_seeds:
            steered_label = member["label"]
            steered_seed_idx = int(member.get("seed_index", 0) or 0)
            # Pair only when the two seed indices are equal.
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


def _compute_metrics_cmd(script, prediction_dir, *, rec_len, eff_len,
                         output_csv, pae_cutoff, receptor_chain,
                         effector_chain, contact_cutoff,
                         native_pdb=None, effector_atom_filter=None,
                         mutated_positions=None):
    """The compute_metrics.py command line both reversion stages build.

    The ten shared flags are here.  The three optional ones are not shared:

      native_pdb              harvest only.  Without it intact_core,
                              weighted_jaccard and interface_plddt come out
                              blank.  The steered side passes the cycle_0
                              ground truth too, so the two stay comparable.
      effector_atom_filter    restricts contact detection to the effector
                              interface-atom set in cycle_0/plan.json.  An
                              empty list there leaves detection unfiltered,
                              which compute_metrics.py handles.
      mutated_positions       build-contaminated only, for the
                              contamination check.

    Keyword-only after the first two, because ten positional arguments in a
    row would be worse than the duplication this replaces.
    """
    cmd = [
        sys.executable, str(script),
        "--model", "boltz2",
        "--prediction-dir", str(prediction_dir),
        "--chain-lengths", str(rec_len), str(eff_len),
        "--output-csv", str(output_csv),
        "--pae-cutoff", str(pae_cutoff),
        "--receptor-chain", receptor_chain,
        "--effector-chain", effector_chain,
        "--contact-cutoff", str(contact_cutoff),
    ]
    if native_pdb is not None:
        cmd += ["--native-pdb", str(native_pdb)]
    if effector_atom_filter is not None:
        cmd += ["--effector-atom-filter-json", str(effector_atom_filter)]
    if mutated_positions:
        cmd += ["--mutated-positions", mutated_positions]
    return cmd


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
    return {pathway_label: result_dict}.

    This runs INSIDE the singularity container.  It needs numpy and
    Bio.Align for the Kabsch and sequence alignment inside
    boltz2_negative_steering.compute_binding_rmsds, and it shells out to
    compute_metrics.py for the confidence and contact analysis, as
    cmd_compute_final_metrics does per row.

    `ground_truth_pdb` is the RFDiffusion design PDB the reverted pose
    is compared against, with its own chain letters.

    Each result dict carries reverted_ra_eff_vs_truth and
    reverted_ra_eff_vs_steered (ra_eff against the ground truth and
    against the steered parent), both reverted_independent_* RMSDs,
    reverted_intact, reverted_prediction_pdb, the three
    reverted_*contact* fields, and every entry in
    `_REVERTED_CONFIDENCE_FIELDS`.  A failed or missing prediction gives
    {"error": "..."} with no metric fields.
    """
    # Lazy numpy import.  Only the harvest phase needs it.
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

    # cycle_0/plan.json is the effector-atom-filter source for
    # compute_metrics.py.  That filter restricts contact detection to
    # the effector's interface atoms, so an off-interface brush against
    # a non-binding face does not register.  Only the cycle_0 plan will
    # do, because the ground truth and true_interface defining those
    # atoms are set there and nowhere else.  Not found means no filter.
    #
    # plan.json sits in workdir itself for a flat cycle-0 run.  A nested
    # run leaves cycle_0/plan.json at <root>/cycle_0/plan.json.
    cycle0_plan_path = _resolve_cycle0_plan_path(workdir)

    # The same plan supplies true_interface_idx and
    # initial_wrong_interface_idx for the per-seed structural Jaccard.
    # Without them the aggregator writes "" for
    # reverted_true_jaccard_median and the rest.
    _cycle0_ctx = _load_cycle0_plan_context(cycle0_plan_path, contact_cutoff)
    cycle0_true_idx: List[int] = _cycle0_ctx["true_idx"]
    cycle0_wrong_idx: List[int] = _cycle0_ctx["wrong_idx"]
    cycle0_rec_seq: str = _cycle0_ctx["rec_seq"]
    cycle0_pred_rec: str = _cycle0_ctx["pred_rec"]
    cycle0_pred_eff: str = _cycle0_ctx["pred_eff"]
    cycle0_contact_cutoff: float = _cycle0_ctx["contact_cutoff"]

    # per_seed_records[(canonical_label, seed_index)] = result_dict.
    # The older code wrote results[label] in a loop under the same
    # label, so with num_seeds > 1 only the LAST seed's record survived
    # and the other seeds' GPU work was thrown away.
    per_seed_records: Dict[Tuple[str, int], Dict] = {}

    # The public return value, {steered_label: result_dict}, filled in
    # at the end from all_label_seeds.
    results: Dict[str, Dict] = {}

    for entry in manifest.get("entries", []):
        label = entry["label"]
        seed_index = int(entry.get("seed_index", 0) or 0)
        rev_dir = Path(entry["rev_dir"])
        design_workdir = Path(entry["design_workdir"])
        steered_pdb = design_workdir / "prediction.pdb"

        # Boltz writes the prediction to a nested path such as
        # `<rev_dir>/predictions/<stem>/pdb/<stem>_model_0.pdb`, so the
        # helper globs rather than assuming that layout.
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
        # The steered prediction stands in as the truth reference, so
        # this measures drift from the parent.
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

        rec_len, eff_len = _count_pdb_chain_ca_lengths(
            reverted_pdb, pred_rec, pred_eff
        )

        # The reverted design's mutation list is the steered one minus
        # the reverted positions, which carry wild-type again.
        reverted_positions = set(entry.get("positions_to_revert") or [])
        remaining_muts_str = _build_remaining_mutations_str(
            rev_dir, reverted_positions
        )

        cmd = _compute_metrics_cmd(
            compute_metrics_script, rev_dir,
            rec_len=rec_len, eff_len=eff_len, output_csv=per_row_csv,
            pae_cutoff=pae_cutoff, receptor_chain=pred_rec,
            effector_chain=pred_eff, contact_cutoff=contact_cutoff,
            native_pdb=ground_truth_pdb,
            effector_atom_filter=cycle0_plan_path)
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
        # The same rule build-contaminated applies to the steered
        # prediction.  Any surviving mutation contacting the effector is
        # new contamination, because the rescue pose is then held by a
        # residue the construct will not carry.  The protected-set
        # fields are diagnostic only.
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

        # Mirrors boltz2_negative_steering's
        # _write_initial_multiseed_csv and _write_initial_only_csv on
        # the steered side.  With no cycle_0 plan both index lists are
        # empty and every jaccard field stays blank.
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
            # prediction.  classify_reversion_verdict reads it.
            "reverted_mutated_contact_positions": mutated_contact_positions,
            "reverted_n_contacts_on_mutated_positions": len(
                mutated_contact_positions
            ),
            "reverted_true_jaccard": rev_true_jaccard,
            "reverted_wrong_jaccard": rev_wrong_jaccard,
            "reverted_n_shared_true": rev_n_shared_true,
            "reverted_n_shared_wrong": rev_n_shared_wrong,
            "reverted_n_design_interface_residues": rev_n_design_iface,
            "reverted_ipsae_min": _fl("ipsae_min"),
            "reverted_ipsae_ab": _fl("ipsae_ab"),
            "reverted_ipsae_ba": _fl("ipsae_ba"),
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
            # These three need the --native-pdb passed in the cmd above.
            "reverted_interface_plddt": _fl("interface_plddt"),
            "reverted_intact_core":     _int("intact_core"),
            "reverted_weighted_jaccard": _fl("weighted_jaccard"),
        }

    # ── Pair per-seed records back to steered labels ─────────────────
    # Each manifest entry covers one (unique reverted sequence, reverted
    # seed index) pair, and its all_label_seeds lists the steered labels
    # sharing that sequence with their own steered seed indices.  A
    # steered label takes the reverted record whose seed index equals
    # its own, so design_05_s0 pairs with reverted seed 0 and
    # design_05_s1 with reverted seed 1.  The older block instead copied
    # the canonical label's single record to every shared label, and
    # that record had already been collapsed to one seed by the
    # overwrite bug noted above.
    results = _pair_per_seed_records_to_steered_labels(
        per_seed_records, manifest
    )

    # Both sit next to the manifest.  The finalize readers expect
    # {label: result}, and {label: {seed_index: result}} keeps the
    # per-seed structure for post-hoc analysis.
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

    The structural filter is repeated on the reverted prediction, which
    must be intact and have reverted_ra_eff_vs_truth < cutoff.

    A mutated residue contacting the effector counts as
    new_contamination only when its position lies inside
    `contamination_gating_positions`, the union of the RFDiffusion
    design region and the true interface.  Positions outside that set
    were accepted as valid steering contacts at the pre-reversion
    (build-contaminated) stage and deliberately left in the reverted
    sequence, so flagging them here would contradict that.  A gated
    contact instead means the reverted pose has moved and a mutated
    residue has landed where it matters.  Passing None turns the gate
    off, the behaviour before the gated set existed.

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
            # Still pose_holds, but name them in the reason.
            return ("pose_holds",
                    f"reverted pose holds structurally and cleanly; "
                    f"ungated mutated contacts at {ungated_contacts} "
                    f"accepted under gated-contamination policy")
        return ("pose_holds", "reverted pose holds structurally and cleanly")

    # No gating set supplied, so flag any mutated contact.
    if mutated_contacts:
        return ("new_contamination",
                f"reverted pose still has {len(mutated_contacts)} mutated "
                f"residue(s) contacting the effector: {mutated_contacts}")

    return ("pose_holds", "reverted pose holds structurally and cleanly")


# ======================================================================
# Command-line stages
#
# Moved here from boltz2_iterate_steering.py, which no longer exists.
# They belong beside the library above rather than in a module named
# after a loop that was never run.
# ======================================================================

import argparse  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pathway_labels import (  # noqa: E402
    _locate_boltz_sidecar_pdb,
    read_mutations_tsv_full,
)


# ───────────────────────────────────────────────────────────────────────
# Subcommand: build-contaminated  (container, phase 2b)
# ───────────────────────────────────────────────────────────────────────
#
# Container-phase CPU work.  For every intact candidate in
# prefilter.json it shells out to compute_metrics.py for the contact
# analysis.  A design is contaminated when a steering mutation contacts
# the effector in the steered prediction, and those positions are the
# ones the reversion pass reverts.  Writes contaminated.json in the
# schema plan-reversions consumes.
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

    # An exhausted pathway has nothing to check.
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

    # The workdir is cycle 0's, so the plan it holds IS the cycle-0
    # plan.  Multi-cycle runs ran from <root>/cycle_N/pathway_<label>/
    # and loaded <root>/cycle_0/plan.json separately, keyed off an
    # "experiment_root" field only the multi-cycle planner wrote.  That
    # planner and its reader are archived in
    # archive/boltz2_iterate_steering_multicycle.py and
    # archive/cycle1plus_aggregation.py.
    cycle0_plan = plan
    cycle0_plan_path = plan_path

    pred_rec_chain = cycle0_plan.get("pred_receptor_chain", "A")
    pred_eff_chain = cycle0_plan.get("pred_effector_chain", "B")

    # compute_metrics.py sits beside this script unless overridden.
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

    # One entry per passing candidate, contaminated or not.  The
    # majority gate after the loop groups them by design.
    per_seed_evaluations: List[Dict] = []
    contaminated_entries: List[Dict] = []
    n_checked = 0
    intact_candidates = prefilter.get("intact_candidates", [])

    # Only candidates whose steered prediction passes the ra_eff
    # threshold.  The reversion pass asks whether a GOOD steered pose
    # survives without the contaminating mutations.  On an already
    # far-off pose it would instead ask whether Boltz happens to find
    # the right pose on a fresh random seed.
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

        # The label is the flat cycle-0 "design_NN".  Multi-cycle runs
        # built a dotted lineage label, "parent.cNdMM", in the archived
        # archive/cycle1plus_aggregation.py.
        label = label_suffix

        # The cumulative mutation history supplies --mutated-positions
        # for compute_metrics.py, and for contaminated designs the full
        # [(pos1, wt, mut)] list goes into contaminated.json.  A cycle-0
        # design has no ancestors, so its own mutations.tsv is the whole
        # history.  Multi-cycle runs walked the lineage instead.
        cumulative: List[Tuple[int, str, str]] = []
        mut_path = design_workdir / "mutations.tsv"
        if mut_path.exists():
            try:
                cumulative = read_mutations_tsv_full(mut_path)
            except Exception as e:
                print(f"  WARN {label}: read_mutations_tsv_full failed: {e}",
                      file=sys.stderr)

        mutated_positions_1 = sorted(set(pos for pos, _, _ in cumulative))
        mutated_cli = ",".join(str(p) for p in mutated_positions_1)

        # CA counts per chain for --chain-lengths.
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

        # compute_metrics.py looks for the confidence, pae and plddt
        # sidecar files beside the nested model PDB.  `prediction.pdb`
        # is only a copy and has no sidecars next to it, so pointing
        # the parser at its directory would lose every confidence
        # metric.  _locate_boltz_sidecar_pdb finds the real directory.
        sidecar = _locate_boltz_sidecar_pdb(pdb)
        pred_dir = sidecar.parent if sidecar is not None else pdb.parent

        per_row_csv = scratch_dir / f"{label.replace('.', '_')}.csv"

        cmd = _compute_metrics_cmd(
            compute_metrics_script, pred_dir,
            rec_len=rec_len, eff_len=eff_len, output_csv=per_row_csv,
            pae_cutoff=args.pae_cutoff, receptor_chain=pred_rec_chain,
            effector_chain=pred_eff_chain, contact_cutoff=args.contact_cutoff,
            effector_atom_filter=cycle0_plan_path,
            mutated_positions=mutated_cli)

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

        # 1-based positions contacting the effector in the steered
        # prediction.
        contact_residues_str = cm.get("contact_residues", "")
        steered_contact_residues = sorted(set(
            int(p) for p in contact_residues_str.split(",")
            if p.strip()
        ))

        # CONTAMINATION RULE
        # ==================
        # ANY mutated position contacting the effector in the steered
        # prediction is contamination.  The wet-lab construct carries
        # the original pre-steering sequence, so it carries no steered
        # residue at all, protected position or not.  Hence
        # positions_to_revert = mutated_positions ∩ contact_residues,
        # with no protected-set filter.
        mutated_positions_set = set(mutated_positions_1)
        contact_set = set(steered_contact_residues)
        positions_to_revert = sorted(mutated_positions_set & contact_set)

        # Clean seeds are recorded too, because the majority gate below
        # needs n_correctly_placed as well as n_contaminated.
        per_seed_evaluations.append({
            "design_slot": cand.get("design"),  # cycle-0 slot name,
                                                # e.g. "design_03".
                                                # The majority gate
                                                # groups on this.
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

    # ── Majority gating ─────────────────────────────────────────────
    # Per design_slot, n_correctly_placed is the seeds that got this far
    # (filtered upstream to intact and ra_eff < threshold) and
    # n_contaminated is those with positions_to_revert.  Reversion runs
    # only when n_contaminated >= ceil(n_correctly_placed / 2), so at
    # least half of a design's correctly-placed seeds must be
    # contaminated.  The earlier gate fired per (design, seed) on any
    # single contaminated seed, which sent one-seed artefacts into a
    # pass costing a full prediction each.
    from collections import defaultdict
    by_design: Dict[str, List[Dict]] = defaultdict(list)
    for ev in per_seed_evaluations:
        by_design[ev["design_slot"]].append(ev)
    n_designs_evaluated = len(by_design)
    n_designs_triggering = 0
    for design_slot, evals in by_design.items():
        n_correctly_placed = len(evals)
        n_contaminated = sum(1 for e in evals if e["is_contaminated"])
        if n_correctly_placed == 0:  # pragma: no cover - defaultdict key
            # Every by_design key is created by an append, so a group
            # always has at least one evaluation.
            continue
        majority_threshold = math.ceil(n_correctly_placed / 2)
        if n_contaminated < majority_threshold:
            # Below-majority contamination leaves the steered result
            # alone.  Those seeds still count as failures in n_pass,
            # they just are not reverted.
            continue
        n_designs_triggering += 1
        # Only the genuinely contaminated entries, whose reverted
        # sequences the reversion stage predicts after
        # write_reversion_plan deduplicates them.
        contaminated_entries.extend(
            e["entry"] for e in evals if e["is_contaminated"]
        )

    print(f"[build-contaminated] majority gate: {n_designs_triggering} / "
          f"{n_designs_evaluated} designs trigger reversion "
          f"({len(contaminated_entries)} contaminated entries queued)")

    (workdir / "contaminated.json").write_text(json.dumps({
        "cycle": prefilter.get("cycle"),
        "parent_pathway": prefilter.get("parent_pathway"),
        "n_checked": n_checked,
        "n_contaminated": len(contaminated_entries),
        "contaminated": contaminated_entries,
    }, indent=2, default=str))

    # Delete the scratch directory on a clean run and keep it for a
    # post-mortem otherwise.  Every candidate completing the loop
    # appends exactly one per-seed evaluation, so a shortfall means one
    # was dropped by a missing PDB, undeterminable chain lengths or a
    # compute_metrics.py failure, and its per-row CSV is the only record
    # of what that call saw.  n_checked counts candidates entered, so it
    # equals the passing count whether or not any failed.
    if len(per_seed_evaluations) == len(passing_candidates):
        try:
            shutil.rmtree(scratch_dir)
        except Exception:
            pass

    return 0


# ───────────────────────────────────────────────────────────────────────
# Subcommand: plan-reversions
# ───────────────────────────────────────────────────────────────────────
#
# Runs after structural filtering, inside a cycle's collect phase.  It
# stages a reverted Boltz YAML under <workdir>/reversions/<safe_label>/
# for each design build-contaminated flagged.
#
# Every input is a file, so this subcommand can be tested standalone.
# It reads <workdir>/plan.json for chain IDs, the effector fasta, the
# boltz container and hyperparameters and the ground-truth path, and
# <workdir>/contaminated.json for the candidates.  It writes
# <workdir>/reversion_plan.json, which predict-reversion-one and
# harvest-reversions read.
#
# This must run INSIDE the singularity container, because it imports
# boltz2_negative_steering.write_boltz_yaml, which sits at that module's
# top level and cannot be lazy-imported without pulling in numpy and
# Bio.
def cmd_plan_reversions(args: argparse.Namespace) -> int:
    workdir: Path = args.workdir.resolve()
    plan_path = workdir / "plan.json"
    contaminated_path = workdir / "contaminated.json"

    if not plan_path.exists():
        print(f"ERROR: {plan_path} not found — plan-reversions must "
              f"run after iterate-plan/compute-distances", file=sys.stderr)
        return 2
    if not contaminated_path.exists():
        # build-contaminated skips when there are no intact candidates.
        # Write an empty reversion_plan.json so the downstream stages
        # do nothing instead of failing.
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

    # The effector sequence comes from the ground-truth PDB's effector
    # chain via get_chain_sequence, as cmd_plan does.  An
    # effector-fasta override in plan.json takes precedence.
    script_dir = SCRIPT_DIR
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import reversion  # type: ignore
    from boltz2_negative_steering import get_chain_sequence  # type: ignore

    ground_truth = Path(plan["ground_truth"])
    truth_eff_chain = plan["truth_effector_chain"]
    eff_override = plan.get("effector_fasta_override")
    if eff_override:
        # Single-record FASTA read, same shape as _read_receptor_fasta.
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
        "boltz_constraints_block": plan.get("boltz_constraints_block"),
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
# GPU array task.  Runs Boltz on one staged reversion YAML, chosen by
# array index.  Modelled on boltz2_negative_steering.cmd_predict_one.
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
        # An out-of-range index means this slot has no work, not an
        # error.  Exit 0 so the array still completes cleanly when the
        # shell script over-provisioned it.
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
            # Older manifests have no boltz_seed field, so fall
            # back to the original formula.
            entry.get("boltz_seed",
                       int(plan["seed"]) + 100000 + idx + 1),
            plan["no_kernels"],
        )
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        # Write a stub result so the harvest can report this cleanly.
        (rev_dir / "predict_error.txt").write_text(str(e))
        return 1

    # Copy the chosen PDB to a stable name, as predict-one does.
    shutil.copy2(pred_pdb, rev_dir / "prediction.pdb")
    print(f"  wrote {rev_dir / 'prediction.pdb'}")
    return 0


# ───────────────────────────────────────────────────────────────────────
# Subcommand: harvest-reversions
# ───────────────────────────────────────────────────────────────────────
#
# In-container harvest.  Per staged reversion it computes the reverted
# structural metrics, the contact analysis, the contamination check and
# the pose_holds, pose_collapses or new_contamination verdict.  Writes
# reversion_results.json.
def cmd_harvest_reversions(args: argparse.Namespace) -> int:
    workdir: Path = args.workdir.resolve()
    plan_path = workdir / "plan.json"
    manifest_path = workdir / "reversion_plan.json"

    if not plan_path.exists():
        print(f"ERROR: {plan_path} not found", file=sys.stderr)
        return 2
    if not manifest_path.exists():
        # plan-reversions may never have run, because
        # build-contaminated found nothing or crashed upstream.  An
        # empty results document lets the finalize stages still run,
        # with no verdicts applied and every intact design unchanged.
        print("[harvest-reversions] no reversion_plan.json — writing "
              "empty reversion_results.json")
        (workdir / "reversion_results.json").write_text("{}\n")
        return 0

    plan = json.loads(plan_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    # Resolved the same way cmd_compute_final_metrics does.
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

    # Classification needs the steered row, which lives in
    # contaminated.json.  The default thresholds match
    # compute-final-metrics' RFDiffusion tier, ra_eff < 5.0 and intact.
    contaminated_path = workdir / "contaminated.json"
    steered_by_label: Dict[str, Dict] = {}
    if contaminated_path.exists():
        cdoc = json.loads(contaminated_path.read_text())
        for c in cdoc.get("contaminated", []):
            steered_by_label[c["label"]] = {
                "receptor_aligned_effector_rmsd": c.get("steered_ra_eff_vs_truth"),
                "receptor_intact": 1,  # it had to be intact to get here
            }

    # The contamination-gating set is the union of the RFDiffusion design
    # region and the true interface, 1-based to match
    # reverted_mutated_contact_positions.  classify_reversion_verdict
    # explains the policy.
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



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

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

    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
