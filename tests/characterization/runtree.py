"""Build a synthetic negative-steering run tree with known values.

The aggregation stages read a finished run off disk. Testing them against a
real run makes the test depend on a 15 GB tree that lives on the cluster, and
asserting on it only proves the aggregator agrees with whatever it produced
last time. These builders construct the tree instead, so each test states the
answer before running the code.

The layouts here are taken from a real run, `experiments/dose-sweep/n50/6G10/
run/`, not from the aggregator's docstrings. Two cycle-0 layouts exist: a
nested one with a `cycle_0/` subdirectory, and a flat one where the same files
sit in the experiment root. Both are buildable, because the aggregator still
detects and supports each.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# The columns steered_results.csv carries, in the order the engine writes
# them. The aggregator renames these to steered_* on the way out.
STEERED_COLUMNS = [
    "rank", "design", "total_mutations", "mutations_chimerax", "mutations_aa",
    "receptor_aligned_effector_rmsd", "delta_receptor_aligned_effector_rmsd",
    "independent_receptor_rmsd", "independent_effector_rmsd",
    "receptor_intact", "wrong_jaccard", "n_shared_wrong", "true_jaccard",
    "n_shared_true", "n_design_interface_residues", "status", "pdb",
]


def steered_row(design: str, *, rank: int = 1, ra_eff: float = 2.0,
                ind_rec: float = 1.0, ind_eff: float = 1.0,
                status: str = "ok", mutations: int = 3,
                pdb: str = "", **extra) -> Dict[str, str]:
    """One row of steered_results.csv, with every column present."""
    row = {
        "rank": str(rank),
        "design": design,
        "total_mutations": str(mutations),
        "mutations_chimerax": "/A:5,7",
        "mutations_aa": "A5K,G7W",
        "receptor_aligned_effector_rmsd": str(ra_eff),
        "delta_receptor_aligned_effector_rmsd": "0.0",
        "independent_receptor_rmsd": str(ind_rec),
        "independent_effector_rmsd": str(ind_eff),
        "receptor_intact": "1" if (ind_rec <= 5.0 and ind_eff <= 5.0) else "0",
        "wrong_jaccard": "0.1",
        "n_shared_wrong": "1",
        "true_jaccard": "0.8",
        "n_shared_true": "8",
        "n_design_interface_residues": "10",
        "status": status,
        "pdb": pdb,
    }
    row.update({k: str(v) for k, v in extra.items()})
    return row


def plan(*, designs: Iterable[Dict] = (), initial_ra_eff: float = 23.9,
         initial_ind_rec: float = 1.0, initial_ind_eff: float = 1.0,
         skip_steering: bool = False, rec_chain: str = "A",
         eff_chain: str = "B", **extra) -> Dict:
    """A cycle-0 plan.json carrying the fields the aggregator reads."""
    doc = {
        "pred_receptor_chain": rec_chain,
        "pred_effector_chain": eff_chain,
        "truth_receptor_chain": rec_chain,
        "truth_effector_chain": eff_chain,
        "initial_receptor_aligned_effector_rmsd": initial_ra_eff,
        "initial_independent_receptor_rmsd": initial_ind_rec,
        "initial_independent_effector_rmsd": initial_ind_eff,
        "designs": list(designs),
        "true_interface_idx": [10, 11, 12],
        "initial_wrong_interface_idx": [40, 41],
        "num_seeds": 1,
    }
    if skip_steering:
        doc["skip_steering"] = True
    doc.update(extra)
    return doc


def build(root: Path, *, nested: bool = True,
          steered_rows: Optional[List[Dict[str, str]]] = None,
          plan_doc: Optional[Dict] = None,
          passing: Optional[Dict] = None,
          write_steered_csv: bool = True,
          design_names: Iterable[str] = ()) -> Path:
    """Write a cycle-0 run tree under `root` and return the cycle-0 directory.

    `nested` picks the cycle_0/ layout; False puts the same files in the root,
    which is what a single-cycle run produces.
    """
    cycle0 = root / "cycle_0" if nested else root
    cycle0.mkdir(parents=True, exist_ok=True)

    designs = []
    for i, name in enumerate(design_names):
        d = cycle0 / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "prediction.pdb").write_text("ATOM\n")
        designs.append({"dir": str(d), "index": i, "name": name})

    doc = plan_doc if plan_doc is not None else plan(designs=designs)
    if plan_doc is not None and designs and not doc.get("designs"):
        doc["designs"] = designs
    (cycle0 / "plan.json").write_text(json.dumps(doc, indent=2))
    (cycle0 / "initial_prediction.pdb").write_text("ATOM\n")

    if write_steered_csv:
        rows = steered_rows if steered_rows is not None else []
        with open(cycle0 / "steered_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=STEERED_COLUMNS,
                               extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    if passing is not None:
        (cycle0 / "passing.json").write_text(json.dumps(passing, indent=2))

    return cycle0


def passing_doc(candidates: Iterable[Dict], selected_idx: Iterable[int] = ()) -> Dict:
    """A cycle_0/passing.json the aggregator can merge onto the rows."""
    return {
        "cycle": 0,
        "all_candidates": list(candidates),
        "selected": [{"design_idx": i} for i in selected_idx],
    }


def candidate(design: str, design_idx: int, **extra) -> Dict:
    c = {"design": design, "design_idx": design_idx,
         "min_dist_to_upstream": 4.0}
    c.update(extra)
    return c
