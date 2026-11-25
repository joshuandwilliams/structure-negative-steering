"""The two readers the aggregation and reversion stages share.

Split out of boltz2_iterate_steering.py, which no longer exists.

read_mutations_tsv_full reads one design's mutation record. A single-cycle
design has no ancestors, so its own mutations.tsv is its whole history.

_locate_boltz_sidecar_pdb finds the nested model PDB that Boltz's confidence
JSON, PAE and pLDDT arrays sit beside. prediction.pdb is a copy with none of
them next to it, so pointing a metrics call at its parent returns ptm=0,
iptm=0 and pae=0 with only pLDDT surviving through the B-factor fallback.

The dotted pathway-label scheme that used to live here indexed a design's
lineage across cycles. Cycles were a second round of steering on the first
round's survivors and no longer exist; the scheme is in
archive/cycle1plus_aggregation.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))  # pragma: no cover - import order


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
    # A cold start that runs more than one seed writes an extra PDB
    # per seed, as `<experiment_root>/initial_prediction_s<N>.pdb` for
    # N = 1, 2 and so on.  Each one's sidecar files go under
    # `<experiment_root>/initial_s<N>/`.  Route each PDB to its own
    # seed directory, so the glob does not pick up its siblings.
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
