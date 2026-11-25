"""The two readers the aggregation and reversion stages share.

Split out of boltz2_iterate_steering.py, now gone.

read_mutations_tsv_full reads one design's mutation record. A single-cycle
design has no ancestors, so its mutations.tsv is its whole history.

_locate_boltz_sidecar_pdb finds the nested model PDB that Boltz's confidence
JSON, PAE and pLDDT arrays sit beside. prediction.pdb is a copy with none of
them next to it, so a metrics call on its parent returns ptm=0, iptm=0 and
pae=0, with only pLDDT surviving through the B-factor fallback.

The dotted pathway-label scheme that used to live here indexed a design's
lineage across cycles. Cycles were a second round of steering on the first
round's survivors and no longer exist. The scheme is in
archive/cycle1plus_aggregation.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def read_mutations_tsv_full(tsv_path: Path) -> List[Tuple[int, str, str]]:
    """Read a mutations.tsv file into (pos1, wt, mut) tuples, keeping
    the TSV's native 1-based positions and uppercase one-letter
    codes."""
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


def _locate_boltz_sidecar_pdb(prediction_pdb: Path) -> "Optional[Path]":
    """Find the original boltz-output PDB whose sidecar files
    (confidence_*.json, pae_*.npz, plddt_*.npz) live next to it.

    The steering pipeline copies Boltz's chosen PDB to
    `<design>/prediction.pdb`, while the sidecars stay nested under
    Boltz's own layout, such as
    `<design>/predictions/<stem>/pdb/<stem>_model_0.pdb` with
    `confidence_<stem>_model_0.json` alongside.  They are keyed off the
    ORIGINAL stem, not "prediction", so compute_metrics.py's parser
    only finds them in the nested directory.

    The cold-start `initial_prediction.pdb` in the experiment root is a
    special case, its Boltz output living in a sibling `initial/`
    directory, so globbing the whole tree would pick up unrelated
    design PDBs.

    It globs recursively, ignores the flat `prediction.pdb` copy, and
    prefers one with a sibling `confidence_<stem>.json` or a
    `confidence/` sibling directory holding one.  It falls back to any
    nested PDB, returning None when there is none, where callers should
    fall back to prediction.pdb's own parent.
    """
    # The cold-start baseline that boltz2_negative_steering.py writes
    # as `<experiment_root>/initial_prediction.pdb` keeps its sidecars
    # in `<experiment_root>/initial/...`.  A cold start of more than
    # one seed writes `initial_prediction_s<N>.pdb` for N = 1, 2 and so
    # on, with sidecars under `initial_s<N>/`.  Route each PDB to its
    # own seed directory, so the glob skips its siblings.
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
        # Strip the prefix and suffix to get "sN".
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
            continue  # skip cached reference structures
        candidates.append(p)
    if not candidates:
        return None

    def _has_confidence(pdb: Path) -> bool:
        stem = pdb.stem
        d = pdb.parent
        # Same-dir sidecar
        if (d / f"confidence_{stem}.json").exists():
            return True
        # Sibling `confidence/` dir, Boltz's layered layout
        if (d.parent / "confidence" / f"confidence_{stem}.json").exists():
            return True
        return False

    with_conf = [p for p in candidates if _has_confidence(p)]
    if with_conf:
        # Prefer model_0 / rank_0
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
