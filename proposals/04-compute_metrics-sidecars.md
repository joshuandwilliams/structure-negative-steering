# 04 — `bin/compute_metrics.py`, the sidecar lookup

**Proposal: accept.** This is the one change where speed is the point. **4.2×
faster** on `parse_boltz2` at the benchmark's own model count, and **25 lines
longer**.

Corrected after applying it. This file first claimed 96× and 24 lines shorter.
96× was the lookup measured in isolation; end to end the lookup is a fraction
of `parse_boltz2`, which also reads PDBs, loads npz arrays and computes
metrics. And the two helpers with their docstrings cost more than the three
glob chains and `_dedup_paths` saved, so the file goes 1,485 → 1,510 lines. I
estimated the line count instead of measuring it. The speed is still the
reason to take it; "shorter as well" was wrong.

## The problem

`parse_boltz2` walks `pred_dir` once to find the model PDBs, then for each PDB
looks up three sidecars: `confidence_<name>.json`, `plddt_<name>.npz` and
`pae_<name>.npz`. Each lookup is three globs chained together, and the third
one is recursive:

```python
conf_files = _dedup_paths(
    glob.glob(os.path.join(pdb_dir, f"confidence_{name}.json")) +
    glob.glob(os.path.join(pdb_dir, "..", "confidence", f"confidence_{name}.json")) +
    glob.glob(os.path.join(pred_dir, "**", f"confidence_{name}.json"), recursive=True)
)
```

`pred_dir` does not change across the loop. So that third glob walks the whole
output tree, and it happens three times per PDB. At 25 models — 5 seeds × 5
diffusion samples, the benchmark's own setting from
`experiments/benchmarking/*/config.yml` — that is **75 recursive walks of the
same directory**.

## The change

Walk once before the loop, index by filename, look up by key.

```python
def _sidecar_index(pred_dir):
    """Map every filename under pred_dir to its paths, nearest first."""
    index = {}
    for dirpath, _dirnames, filenames in os.walk(pred_dir):
        for nm in filenames:
            index.setdefault(nm, []).append(os.path.join(dirpath, nm))
    for paths in index.values():
        paths.sort(key=lambda p: (p.count(os.sep), p))
    return index


def _sidecar(index, pdb_dir, name, kind, ext):
    """The best path for one sidecar, or None."""
    filename = f"{kind}_{name}.{ext}"
    candidates = index.get(filename, ())
    if not candidates:
        return None
    here = os.path.join(pdb_dir, filename)
    sibling = os.path.normpath(os.path.join(pdb_dir, "..", kind, filename))
    for preferred in (here, sibling):
        for c in candidates:
            if os.path.normpath(c) == preferred:
                return c
    return candidates[0]
```

Then each of the three lookups becomes one line:

```python
conf_file  = _sidecar(index, pdb_dir, name, "confidence", "json")
plddt_file = _sidecar(index, pdb_dir, name, "plddt", "npz")
pae_file   = _sidecar(index, pdb_dir, name, "pae", "npz")
```

The three `kind` values are also the sibling directory names, which is Boltz's
own convention: `confidence_x.json` lives either beside the PDB or in a
sibling `confidence/`. That is what lets one helper serve all three and
removes the only use of `_dedup_paths`.

**Preference order is preserved exactly**, which is the thing that had to be
got right because the caller takes element zero. The old order was: beside the
PDB, then the sibling category directory, then anywhere in the tree. The new
code checks those two explicit locations first and falls back to the
shallowest remaining path.

## Speed, measured

Two measurements, and the difference between them is why the headline changed.

The **lookup in isolation**, three sidecars across all PDBs:

| Models | Files in tree | Current | Proposed | Speedup |
|---:|---:|---:|---:|---:|
| 5 | 60 | 12.8 ms | 0.8 ms | 16× |
| 25 | 300 | 78.3 ms | 0.8 ms | 96× |
| 125 | 1,300 | 559.4 ms | 2.3 ms | 244× |

**`parse_boltz2` end to end**, which is what a run actually spends:

| Models | Files in tree | Current | Proposed | Speedup |
|---:|---:|---:|---:|---:|
| 5 | 30 | 65.6 ms | 5.1 ms | 12.9× |
| 25 | 150 | 67.5 ms | 16.1 ms | **4.2×** |
| 125 | 750 | 385.4 ms | 91.3 ms | 4.2× |

The second table is the honest one. The first measures only the part that
changed.

The honest context: `negsteer_aggregate.py` runs `compute_metrics.py` as a
subprocess per passing row, so a benchmark target makes tens to low hundreds
of these calls. At 78 ms each that is a few seconds per target against a run
of 19.5 to 263 minutes. **So this does not shorten a run noticeably.** It is
worth taking because the code is shorter as well, not because it saves time.

Where it would matter is a dose-sweep arm at n=50 with many passing rows, and
on a shared filesystem where each `stat` is a network round trip rather than a
local one. I have not measured that, and I will not claim it.

## Verification

- Both Boltz output trees in `tests/characterization/fixtures`: 3 lookups per
  PDB, identical paths.
- 300 synthetic trees with the sidecar randomly present beside the PDB, in a
  sibling `confidence/`, deeper in the tree, or in several of those at once:
  906 lookups total, zero differences.

## Line count

**25 lines in**, 1,485 → 1,510. The three 5-line glob chains become three
1-line calls and `_dedup_paths` goes with its docstring, 25 lines out. The two
new helpers cost 50 lines including their docstrings. This is the one proposal
in the folder that makes the file longer.

## Is every line necessary

`os.path.normpath` on both sides of the comparison is. Without it
`a/pdb/../confidence/x.json` and `a/confidence/x.json` are different strings
for the same file, and the sibling-directory preference would never match.
That is exactly what the old `glob` did implicitly by resolving the pattern.
