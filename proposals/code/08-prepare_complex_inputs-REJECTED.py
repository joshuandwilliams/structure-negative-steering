def _write_index_file(path: Path, values, *, header: dict, body: str) -> None:
    """Write a "# key: value" header then one body block.

    Both index files the engine reads have this shape. They differ in their
    header keys, in their coordinate base, and in whether the values go one
    per line or comma-separated, so the caller supplies all three.
    """
    with path.open("w") as f:
        for key, value in header.items():
            f.write(f"# {key}: {value}\n")
        f.write(body)


def _write_true_interface(path: Path, idx0: List[int], *, complex_pdb: Path,
                          receptor_chain: str, effector_chain: str,
                          source: str) -> None:
    """Write 0-based indices, one per line.

    The format the design pipeline's true-interface files use, which is
    what --true-interface-indices-file parses.
    """
    _write_index_file(path, idx0, header={
        "derived from complex": complex_pdb,
        "interface source": source,
        "receptor chain": receptor_chain,
        "effector chain": effector_chain,
        "n contacts": len(idx0),
        "coordinate system": ("0-based positional indices on the receptor "
                              "chain, PDB walk order"),
    }, body="".join(f"{p}\n" for p in idx0))


def _write_design_region(path: Path, idx1: List[int], *, complex_pdb: Path,
                         receptor_chain: str, source: str) -> None:
    """Write 1-based indices, comma-separated.

    The format --design-region-indices-file parses. It counts from 1,
    unlike the true-interface file above.
    """
    _write_index_file(path, idx1, header={
        "derived from complex": complex_pdb,
        "design region source": source,
        "receptor chain": receptor_chain,
        "design region size": f"{len(idx1)} residues",
        "coordinate system": ("1-based positional indices on the receptor "
                              "chain, PDB walk order"),
    }, body=",".join(str(p) for p in idx1) + "\n")
