def _sidecar_index(pred_dir):
    """Map every filename under pred_dir to its paths, nearest first.

    parse_boltz2 looks up three sidecars per PDB: confidence_<name>.json,
    plddt_<name>.npz and pae_<name>.npz. Each lookup used to be three globs,
    one of which walked the whole of pred_dir recursively, so the tree was
    walked three times per PDB. With 25 models that is 75 walks of the same
    directory. Walking once and indexing by filename gives the same answer.

    Order matters, because the caller takes the first hit. The old order was
    beside the PDB, then a sibling category directory, then anywhere in the
    tree. Sorting each filename's paths by depth reproduces that: a sidecar
    beside the PDB is shallower than one in a sibling directory, which is
    shallower than one further down.
    """
    index = {}
    for dirpath, _dirnames, filenames in os.walk(pred_dir):
        for nm in filenames:
            index.setdefault(nm, []).append(os.path.join(dirpath, nm))
    for paths in index.values():
        paths.sort(key=lambda p: (p.count(os.sep), p))
    return index


def _sidecar(index, pdb_dir, name, kind, ext):
    """The best path for one sidecar, or None.

    `kind` is the filename prefix and the sibling directory name, which Boltz
    keeps the same: confidence_<name>.json lives either beside the PDB or in
    a sibling `confidence/`.
    """
    filename = f"{kind}_{name}.{ext}"
    candidates = index.get(filename, ())
    if not candidates:
        return None
    # Beside the PDB wins, then the sibling category directory, then depth.
    here = os.path.join(pdb_dir, filename)
    sibling = os.path.normpath(
        os.path.join(pdb_dir, "..", kind, filename))
    for preferred in (here, sibling):
        for c in candidates:
            if os.path.normpath(c) == preferred:
                return c
    return candidates[0]
