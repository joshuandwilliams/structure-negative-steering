def _sidecar_rank(pdb: Path) -> tuple:
    """Sort key preferring Boltz's first-ranked model."""
    return ("model_0" not in pdb.name and "rank_0" not in pdb.name, str(pdb))


def _has_confidence(pdb: Path) -> bool:
    """True when a confidence JSON sits beside the PDB, in either layout."""
    return ((pdb.parent / f"confidence_{pdb.stem}.json").exists()
            or (pdb.parent.parent / "confidence"
                / f"confidence_{pdb.stem}.json").exists())


def _sidecar_search_dir(prediction_pdb: Path) -> Path:
    """Where to look for the nested model PDB.

    A design's sidecars are under the design directory. The cold-start
    baselines are not: boltz2_negative_steering.py writes
    `<root>/initial_prediction.pdb` with sidecars in `<root>/initial/`, and
    `<root>/initial_prediction_s<N>.pdb` with sidecars in `<root>/initial_s<N>/`.
    Routing each to its own seed directory is what stops the glob picking up
    a sibling seed's PDBs.
    """
    name = prediction_pdb.name
    if not name.startswith("initial_prediction") or not name.endswith(".pdb"):
        return prediction_pdb.parent
    root = prediction_pdb.parent
    suffix = name[len("initial_prediction"):-len(".pdb")]   # "" or "_s1"
    seed_dir = root / f"initial{suffix}"
    return seed_dir if seed_dir.is_dir() else root


def _locate_boltz_sidecar_pdb(prediction_pdb: Path) -> "Optional[Path]":
    """Find the Boltz-output PDB whose sidecar files sit next to it.

    The pipeline copies Boltz's chosen PDB to `<design>/prediction.pdb` while
    the sidecars stay under Boltz's own layout, such as
    `<design>/predictions/<stem>/pdb/<stem>_model_0.pdb` with
    `confidence_<stem>_model_0.json` alongside. They are keyed off the
    ORIGINAL stem, not "prediction", so compute_metrics.py's parser only finds
    them in the nested directory.

    Returns the best nested PDB, preferring one that has a confidence JSON
    beside it, then Boltz's first-ranked model. None when there is no nested
    PDB, where callers fall back to prediction.pdb's own parent.
    """
    search_dir = _sidecar_search_dir(prediction_pdb)
    # resolve() is syscalls, and the target does not change per candidate.
    target = prediction_pdb.resolve()
    candidates = [p for p in search_dir.rglob("*.pdb")
                  if p.resolve() != target and "msa" not in p.parts]
    if not candidates:
        return None
    with_conf = [p for p in candidates if _has_confidence(p)]
    return min(with_conf or candidates, key=_sidecar_rank)
