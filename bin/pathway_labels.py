"""Pathway labels and the cumulative-mutation history they index.

Split out of boltz2_iterate_steering.py, which no longer exists. These six
helpers are the only code the aggregation and reversion stages share, so they
live here rather than being duplicated into both.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


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


def parent_pathway_label(label: str) -> Optional[str]:
    """Drop the last segment.  Returns None for cycle-0 leaves."""
    segs = label.split(".")
    if len(segs) <= 1:
        return None
    return ".".join(segs[:-1])


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
