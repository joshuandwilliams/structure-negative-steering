# 12 — `bin/boltz2_negative_steering.py`, 3,339 lines

**Proposal: change nothing. This is the one I would leave alone.**

It is last in the order because it is the biggest file, has the biggest
function, and is the one where a mistake is most expensive. It is also the one
where I found the least worth changing, and that is the finding rather than a
failure to look.

## `cmd_plan`, 872 lines

| | Count |
|:---|---:|
| lines | 872 |
| statements | 357 |
| `print` calls | 92 |
| comment-only lines | 195 |
| blank lines | 56 |

872 lines for 357 statements is 2.44 lines per statement, the highest of any
large function in `bin/`. 195 comment lines is 22% of the function.

## Why the 92 prints are not bloat

I went in expecting to fold them. They are the run's log, and on a cluster
they are the only record of what the stage decided. Reading them in order:

```
  Wrote effector template: …
  Wrote wild-type YAML: …
  Initial receptor_aligned_effector_rmsd: 35.981 Å
  Initial independent_receptor_rmsd:       1.204 Å
  Initial independent_effector_rmsd:       2.117 Å
  Running 2 additional cold-start predictions…
    seed 1: ra_eff=34.772 Å …
  Using pre-computed TRUE interface …
  73 receptor residues at the TRUE interface
  Cold-start: 2/3 seeds clean — …
  61 receptor residues at the wrong interface
  design region: 24 positions …
```

Each line is a different decision with a different number. There is no group
of three or more that share a shape, so there is nothing to put in a loop.
The three `Initial *_rmsd` lines are the closest, and folding them into a loop
over a tuple of labels would save two lines and lose the aligned columns.

## What I considered and rejected

**Splitting `cmd_plan`.** It runs the cold start, decides whether steering is
needed at all, derives the true interface, derives the wrong interface,
computes the protected set, picks the mutations and stages the designs. Those
are separable on paper. In the code they are not independent: the cold-start
result decides whether the rest runs, the true interface feeds the protected
set, and the wrong interface is computed against the cold-start pose. A split
would need the intermediate state passed between five functions, which is
either a long parameter list or a context object.

The argument against is sharper than the argument for. This function decides
which residues get mutated, which is the method. Every committed number
depends on it, and I cannot verify a restructure the way I verified the others
— there is no fixture that exercises a full `cmd_plan` without a GPU, because
the function's core is a Boltz call. I would be changing the most important
function in the repo on the strength of the test suite alone.

**The 195 comment lines.** I read them looking for restatement. They are not
restatement: they record why a threshold is what it is, which past bug a guard
exists for, and what the coordinate system of each index list is. Examples of
the kind of thing that is in there and should stay:

- which prefix `skip_steering` writes and why it is `all_seeds_clean` rather
  than a majority rule
- that the 4.5 Å naive pool and the 6 Å CA fallback leave a band where
  neither applies
- that `pick_mutant_residue`'s pool is never a single residue, which is why
  its backstop carries a pragma

A prose pass already ran over this file: 7,497 words down to 5,206, a 31% cut.
What is left is load-bearing.

## The one thing I would do, and it is not a refactor

`extract_effector_template_cif` is 182 lines and its docstring is 35 of them.
That docstring is the best thing in the file: it explains why gemmi leaves
`Entity.full_sequence` unset when reading a PDB, what AF3 does with
`label_seq_id='.'`, and the exact `ValueError` that follows. I would not
touch it.

## Speed

Nothing to propose. The function's wall-clock is one to several `boltz
predict` calls, each minutes. From `analysis/03-compute-cost/`, 20.6 seconds
per prediction is the median and 28.7 the worst. Python that runs for
microseconds between those calls is not measurable.

I did check the one thing that could have mattered: `run_boltz` calls
`out_dir.rglob('*')` inside a loop at line 795. It is bounded by the number of
output files Boltz wrote for one prediction, and it runs once per prediction,
after a call that took minutes. Not worth touching.

## Is every line necessary

For 3,339 lines I cannot claim to have justified each one individually, and I
will not pretend to. What I did check:

- every function is reached by a workflow stage except
  `_read_ca_seq_from_chain`, 5 lines, covered in the orphans PR and kept
  because it is a per-chain variant of a live reader
- the 41 string literals of eight or more elements are the column-name lists,
  which belong to the output schema rather than to this file's logic
- the 125 `print` calls are the stage log, with no group of three sharing a
  shape

That is the honest extent of it.
