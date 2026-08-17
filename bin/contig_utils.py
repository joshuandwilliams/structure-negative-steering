#!/usr/bin/env python3
"""
contig_utils.py
---------------
Shared contig parsing and resolution utilities for RFDiffusion scripts.

Provides functions to:
  - Read chain residue ranges from PDB files
  - Parse contig strings into typed segment descriptors
  - Resolve bare chain refs and remap residue numbers to match PDB
  - Identify design vs fixed regions
  - Compute expected chain lengths
"""

import sys

# Three-letter to one-letter amino acid map.  Covers the 20 standard
# residues plus common non-standard / modified codes encountered in PDB
# and mmCIF files (selenomethionine MSE, selenocysteine SEC, pyrrolysine
# PYL, hydroxyproline HYP, phospho-Thr/Ser/Tyr TPO/SEP/PTR).  Unknown
# residue codes should map to "X" by callers using `.get(code, "X")`.
THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
    "MSE": "M", "HYP": "P", "TPO": "T", "SEP": "S", "PTR": "Y",
}


def get_chain_residue_range(pdb_path, chain_id):
    """Return (min_resnum, max_resnum) for a chain, or (None, None) if absent."""
    resnums = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[21] == chain_id:
                try:
                    resnums.add(int(line[22:26].strip()))
                except ValueError:
                    pass
    return (min(resnums), max(resnums)) if resnums else (None, None)


def get_chain_residues_sorted(pdb_path, chain_id):
    """Return sorted list of unique residue numbers for a chain."""
    resnums = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[21] == chain_id:
                try:
                    resnums.add(int(line[22:26].strip()))
                except ValueError:
                    pass
    return sorted(resnums)


def parse_block_segments(block, rec_chain=None, pdb_path=None):
    """
    Parse a single contig block (e.g. "A1-390/20-40/A421-438") into a list
    of typed segment descriptors:
        ("fixed", chain, start, end)
        ("denovo", "min-max")
        ("break",)
        ("passthrough", raw_string)

    Thin adapter around :class:`contig_spec.ContigChain` (Phase 4).  All
    RFDiffusion contig-grammar features — including the chain-break
    marker (`0`) and bare-chain-letter passthrough — round-trip through
    the typed form.  Optional ``pdb_path`` triggers post-parse
    resolution of passthrough segments to fixed residue ranges (via
    ``get_chain_residue_range``).

    Returns ``(descs, block_chain)`` to preserve the legacy CLI contract
    used by ``resolve_contigs`` and ``remap_segments_to_pdb`` downstream.
    """
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from contig_spec import (  # noqa: E402
        BreakSegment,
        ContigSpec,
        DeNovoSegment,
        FixedSegment,
        PassthroughSegment,
    )

    # ContigSpec.from_string parses multi-chain space-separated blocks.
    # parse_block_segments operates on a single block, so wrap-and-pick.
    try:
        spec = ContigSpec.from_string(block)
    except ValueError as e:
        raise ValueError(f"contig block {block!r} parse failed: {e}")
    if not spec.chains:
        return [], None
    chain = spec.chains[0]
    block_chain = chain.chain_id

    descs = []
    # Bare-chain blocks (e.g. `B` alone) parse into ContigChain with
    # segments=(); legacy parse_block_segments emitted a single
    # passthrough descriptor for these.  Preserve that behaviour.
    if not chain.segments:
        if pdb_path:
            lo, hi = get_chain_residue_range(pdb_path, block_chain)
            if lo is not None:
                return [("fixed", block_chain, lo, hi)], block_chain
        return [("passthrough", block_chain)], block_chain

    for seg in chain.segments:
        if isinstance(seg, FixedSegment):
            descs.append(("fixed", block_chain, seg.start, seg.end))
        elif isinstance(seg, DeNovoSegment):
            descs.append(("denovo", f"{seg.min_len}-{seg.max_len}"))
        elif isinstance(seg, BreakSegment):
            descs.append(("break",))
        elif isinstance(seg, PassthroughSegment):
            if pdb_path:
                lo, hi = get_chain_residue_range(pdb_path, seg.chain)
                if lo is not None:
                    descs.append(("fixed", seg.chain, lo, hi))
                    continue
            descs.append(("passthrough", seg.chain))
    return descs, block_chain


def _resolve_denovo_lengths(seg_descs, pdb_total):
    """
    For variable-length de novo segments, resolve actual lengths using the
    total PDB residue count. Returns {seg_index: resolved_length}.
    """
    total_fixed = sum(d[3] - d[2] + 1 for d in seg_descs if d[0] == "fixed")
    denovo_known = 0
    variable_indices = []

    for i, desc in enumerate(seg_descs):
        if desc[0] == "denovo":
            parts = desc[1].split("-")
            n_min, n_max = int(parts[0]), int(parts[1])
            if n_min == n_max:
                denovo_known += n_min
            else:
                variable_indices.append(i)

    remaining = pdb_total - total_fixed - denovo_known
    resolved = {}

    if len(variable_indices) == 1:
        resolved[variable_indices[0]] = remaining
    elif len(variable_indices) > 1:
        for idx in variable_indices:
            parts = seg_descs[idx][1].split("-")
            resolved[idx] = int(parts[0])  # use minimum

    return resolved


def remap_segments_to_pdb(seg_descs, block_chain, pdb_path):
    """
    Check if fixed-segment residue references exist in the PDB.
    If not, remap them by walking the PDB residues sequentially.
    Returns the (possibly remapped) segment descriptors.

    Fails loudly if the block references a chain that has no atoms in
    the PDB at all — silently passing through such a contig would let
    RFDiffusion fail later with a confusing "(<chain>, 1) is not in pdb
    file!" assertion error.
    """
    if block_chain is None:
        return seg_descs

    pdb_res_sorted = get_chain_residues_sorted(pdb_path, block_chain)
    pdb_set = set(pdb_res_sorted)
    pdb_total = len(pdb_res_sorted)

    if pdb_total == 0:
        # If this block has fixed segments referencing this chain, the
        # contig is broken — RFDiffusion will fail downstream with a
        # confusing assertion.  Surface a clear error here instead.
        has_fixed_segs = any(d[0] == "fixed" for d in seg_descs)
        if has_fixed_segs:
            referenced = ",".join(
                f"{d[1]}{d[2]}-{d[3]}" if d[2] != d[3] else f"{d[1]}{d[2]}"
                for d in seg_descs if d[0] == "fixed"
            )
            sys.exit(
                f"ERROR: contig references chain '{block_chain}' "
                f"({referenced}) but the input PDB has no atoms for "
                f"chain '{block_chain}'.  Check that the chain letters "
                f"in your contig string match the chains in the PDB."
            )
        return seg_descs

    # Check if remapping is needed
    needs_remap = any(
        r not in pdb_set
        for d in seg_descs if d[0] == "fixed"
        for r in range(d[2], d[3] + 1)
    )

    if not needs_remap:
        return seg_descs

    resolved_denovo = _resolve_denovo_lengths(seg_descs, pdb_total)

    pos = 0
    remapped = []
    for i, desc in enumerate(seg_descs):
        if desc[0] == "fixed":
            length = desc[3] - desc[2] + 1
            new_start = pdb_res_sorted[pos]
            new_end = pdb_res_sorted[pos + length - 1]
            remapped.append(("fixed", desc[1], new_start, new_end))
            pos += length
        elif desc[0] == "denovo":
            parts = desc[1].split("-")
            n_min, n_max = int(parts[0]), int(parts[1])
            dn_len = n_min if n_min == n_max else resolved_denovo.get(i, n_min)
            remapped.append(desc)
            pos += dn_len
        else:
            remapped.append(desc)

    print(f"Remapped fixed segments to match PDB numbering "
          f"(chain {block_chain}: {pdb_total} residues)", file=sys.stderr)
    return remapped


def segments_to_string(seg_descs):
    """Convert typed segment descriptors back to a contig string fragment."""
    parts = []
    for desc in seg_descs:
        if desc[0] == "break":
            parts.append("0")
        elif desc[0] == "fixed":
            parts.append(f"{desc[1]}{desc[2]}-{desc[3]}")
        elif desc[0] == "denovo":
            parts.append(desc[1])
        elif desc[0] == "passthrough":
            parts.append(desc[1])
    return "/".join(parts)


def resolve_contigs(raw_contigs, pdb_path):
    """
    Convert user-friendly contig notation into RFDiffusion-compatible format.

    Handles: bare chain letters, single-residue shorthands, bare de novo
    numbers, and remaps fixed-segment residues to match actual PDB numbering.

    Delegates per-block parsing to :func:`parse_block_segments`, which is
    a thin adapter over :class:`contig_spec.ContigSpec`.  The only
    non-delegated branch is the bare-chain shortcut — it goes through a
    direct PDB lookup so that a chain not present in the PDB surfaces a
    visible WARNING (the ContigSpec/passthrough path would silently emit
    the bare letter back instead).
    """
    blocks = raw_contigs.replace(",", " ").replace(":", " ").split()
    out_blocks = []

    for block in blocks:
        # Bare chain letter (e.g. "B") — direct PDB lookup so the
        # missing-chain case surfaces a warning rather than passing
        # through silently as a PassthroughSegment.
        if len(block) == 1 and block.isalpha():
            lo, hi = get_chain_residue_range(pdb_path, block)
            if lo is not None:
                out_blocks.append(f"{block}{lo}-{hi}")
            else:
                print(f"WARNING: Chain '{block}' not found in PDB", file=sys.stderr)
                out_blocks.append(block)
            continue

        seg_descs, block_chain = parse_block_segments(block, pdb_path=pdb_path)
        seg_descs = remap_segments_to_pdb(seg_descs, block_chain, pdb_path)
        out_blocks.append(segments_to_string(seg_descs))

    return " ".join(out_blocks)


def get_expected_chain_lengths(contigs, rec_chain, eff_chain):
    """
    Parse a resolved contig string and return expected (rec_len, eff_len).
    De novo segments use the midpoint of min-max as expected length.
    """
    blocks = contigs.split()
    rec_len = eff_len = 0

    for block in blocks:
        block_len = 0
        block_chain = None

        for seg in block.split("/"):
            seg = seg.strip()
            if not seg or seg == "0":
                continue
            if seg[0].isalpha():
                block_chain = seg[0].upper()
                rest = seg[1:]
                if "-" in rest:
                    parts = rest.split("-")
                    block_len += int(parts[1]) - int(parts[0]) + 1
                else:
                    block_len += 1
            elif seg[0].isdigit():
                if "-" in seg:
                    parts = seg.split("-")
                    block_len += (int(parts[0]) + int(parts[1])) // 2
                else:
                    block_len += int(seg)

        if block_chain == rec_chain.upper():
            rec_len = block_len
        elif block_chain == eff_chain.upper():
            eff_len = block_len

    return rec_len, eff_len


def parse_design_region(contigs, rec_chain):
    """
    Identify de novo (design) and fixed residue sets from the contig string.
    De novo residues are those in gaps between consecutive fixed ranges.
    Returns (design_residues, fixed_residues) as sets of native residue
    numbers on the receptor chain.

    Thin adapter around :class:`contig_spec.ContigSpec`: parses the
    contig once, looks up the receptor chain (case-insensitive), and
    derives the design region as the gaps between consecutive fixed
    segments in native numbering.  Returns empty sets if the contig
    can't be parsed or the receptor chain isn't present.
    """
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from contig_spec import ContigSpec  # noqa: E402

    try:
        spec = ContigSpec.from_string(contigs)
    except ValueError:
        return set(), set()

    rec_upper = rec_chain.upper()
    chain = next(
        (c for c in spec.chains if c.chain_id.upper() == rec_upper),
        None,
    )
    if chain is None:
        return set(), set()

    fixed_segments = chain.fixed_segments
    if not fixed_segments:
        return set(), set()

    fixed_residues: set = set()
    fixed_ranges = []
    for seg in fixed_segments:
        fixed_residues.update(range(seg.start, seg.end + 1))
        fixed_ranges.append((seg.start, seg.end))

    design_residues: set = set()
    sorted_ranges = sorted(fixed_ranges)
    for a, b in zip(sorted_ranges, sorted_ranges[1:]):
        gap_start = a[1] + 1
        gap_end = b[0] - 1
        if gap_end >= gap_start:
            design_residues.update(range(gap_start, gap_end + 1))

    return design_residues, fixed_residues
