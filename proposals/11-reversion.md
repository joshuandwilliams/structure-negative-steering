# 11 — `bin/reversion.py`, 1,826 lines

**Proposal: accept the argv builder. 18 lines out, and the flag ordering stops
being a thing you have to keep in your head.**

## The repetition

`reversion.py` builds the same `compute_metrics.py` command line twice, at
line 884 in `harvest_reversion_results` and line 1298 in
`cmd_build_contaminated`. Ten flags are identical, in the same order:

```python
cmd = [
    sys.executable, str(compute_metrics_script),
    "--model", "boltz2",
    "--prediction-dir", str(rev_dir),
    "--chain-lengths", str(rec_len), str(eff_len),
    "--output-csv", str(per_row_csv),
    "--pae-cutoff", str(pae_cutoff),
    "--receptor-chain", pred_rec,
    "--effector-chain", pred_eff,
    "--contact-cutoff", str(contact_cutoff),
    ...
]
```

They differ only in which of three optional flags they add:
`--native-pdb` at the harvest site, `--effector-atom-filter-json` at both but
conditionally at one, `--mutated-positions` at the build-contaminated site.

## The change

One builder with the ten shared flags and three keyword arguments for the
optional ones. The important part is that each optional flag keeps its own
comment, because each exists for a different reason:

```python
def _compute_metrics_cmd(script, prediction_dir, *, rec_len, eff_len,
                         output_csv, pae_cutoff, receptor_chain,
                         effector_chain, contact_cutoff,
                         native_pdb=None, effector_atom_filter=None,
                         mutated_positions=None):
    """The compute_metrics.py command line both reversion stages build.

    The ten shared flags are here. The three optional ones are not shared:

      native_pdb              harvest only. Without it intact_core,
                              weighted_jaccard and interface_plddt come out
                              blank. The steered side passes the cycle_0
                              ground truth too, so the two stay comparable.
      effector_atom_filter    restricts contact detection to the effector
                              interface-atom set in cycle_0/plan.json. An
                              empty list there leaves detection unfiltered,
                              which compute_metrics.py handles.
      mutated_positions       build-contaminated only, for the contamination
                              check.
    """
```

Each call site becomes one call:

```python
cmd = _compute_metrics_cmd(
    compute_metrics_script, rev_dir, rec_len=rec_len, eff_len=eff_len,
    output_csv=per_row_csv, pae_cutoff=pae_cutoff,
    receptor_chain=pred_rec, effector_chain=pred_eff,
    contact_cutoff=contact_cutoff, native_pdb=ground_truth_pdb,
    effector_atom_filter=cycle0_plan_path)
```

Keyword-only after the first two arguments, which is the point: ten positional
arguments in a row would be worse than the duplication.

## Verification

Both call sites reconstructed from the committed source and compared
element-by-element against the builder's output: **identical argument lists**,
including order.

That is the whole check, and it is sufficient here: the function's only output
is a list of strings handed to `subprocess.run`, so list equality is behaviour
equality.

## What I rejected

**The third copy, in `negsteer_aggregate.py:2073`.** Earlier analysis flagged
three copies of this argv builder across two files. Factoring all three means
putting the builder in a module both files import, and the only candidates are
`compute_metrics.py` itself — which would make the caller import the thing it
is about to run as a subprocess — or a new module holding one function.
Neither is better than one duplicated block across a file boundary. The two
copies **inside** `reversion.py` are worth merging; the third is not.

**`cmd_build_contaminated`, 299 lines.** The largest function in the file. I
read it for a column group or a repeated block and found neither: it is the
CL-3 majority gate, the per-seed contamination check and the scratch-directory
bookkeeping, interleaved because the gate's result decides whether the rest
runs. Splitting it is a real refactor with real risk to the gate that decides
which designs get reverted, and 299 lines of straight-line logic is not what
is wrong with this repo.

## Speed

No measurable change. The builder returns a list that is immediately handed to
`subprocess.run`, which spawns a Python interpreter — tens of milliseconds
against the microsecond the list construction costs. Called once per candidate.

## Is every line necessary

The three optional flags are, and the comments on them more so. The
`--native-pdb` comment records that without it three metric columns come out
blank, which is the kind of thing that is invisible until a figure has a hole
in it. Moving them into the docstring keeps them attached to the flag rather
than to one of the two call sites.
