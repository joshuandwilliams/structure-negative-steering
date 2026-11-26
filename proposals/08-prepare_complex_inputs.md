# 08 — `bin/prepare_complex_inputs.py`, 329 lines

**Proposal: reject the merge.** My own byte-comparison killed it, which is the
reason this file is worth reading.

## What looked mergeable

`_write_true_interface` and `_write_design_region` are 16 and 14 lines of the
same shape: open a file, write six or five `#` header lines, write the values.

```python
with path.open("w") as f:
    f.write(f"# derived from complex {complex_pdb}\n")
    f.write(f"# interface source: {source}\n")
    f.write(f"# receptor chain: {receptor_chain}\n")
    f.write(f"# effector chain: {effector_chain}\n")
    f.write(f"# n contacts: {len(idx0)}\n")
    f.write("# coordinate system: 0-based positional indices on the receptor "
            "chain, PDB walk order\n")
    for p in idx0:
        f.write(f"{p}\n")
```

So I wrote the obvious helper: take a `dict` of header keys and a body string,
emit `# {key}: {value}` per entry.

## Why it is rejected

The byte-comparison failed on all 16 cases, and the diff is one character:

```
original:  # derived from complex /c/x.pdb
generic:   # derived from complex: /c/x.pdb
```

**The first header line has no colon and the other five do.** A generic
`# {key}: {value}` helper cannot emit that without a special case for the
first key, and a helper with a special case for one of six keys is longer and
worse than the two explicit writers.

The colon is cosmetic. `#` lines are skipped by the reader, at
`boltz2_negative_steering.py:1210`. So I could change it and nothing would
break. I am not proposing that either: it would alter the first line of every
index file a future run writes, for no gain beyond letting a helper be
slightly shorter, and index files are the inputs the engine's positional
indices are read from. Changing their format to tidy a writer is the wrong
trade.

## What I would take instead

Add the missing colon, as a one-character consistency fix, and leave the two
writers alone. That is a cosmetic change to a committed output format, so it
is your call rather than mine, and it is not in a diff here.

## The one real defect in this file

`_load_engine_helpers` has an unguarded `sys.path.insert`:

```python
def _load_engine_helpers():
    sys.path.insert(0, str(ENGINE_BIN))
    try:
        from boltz2_negative_steering import get_chain_sequence
```

Measured: 200 calls add 201 entries to `sys.path`. It is called once, at line
217, so in practice it adds one entry per process and costs nothing. It is the
same latent bug as the one in 06 that *was* costing something, so the guard is
worth adding for consistency:

```python
if str(ENGINE_BIN) not in sys.path:
    sys.path.insert(0, str(ENGINE_BIN))
```

Two lines for one, and it matches what every other module in `bin/` does —
11 of the 14 insert sites already have that guard.

## Speed

Nothing measured beyond the `sys.path` count. The file reads one PDB and
writes four small text files, once per target.
