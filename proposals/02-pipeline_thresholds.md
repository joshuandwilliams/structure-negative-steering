# 02 — `bin/pipeline_thresholds.py`, 158 lines

**Proposal: change nothing.**

This file is 44 statements in 158 lines, a ratio of 3.6, the highest in
`bin/`. That ratio is normally the signal of padding. Here it is not: the file
is one frozen dataclass with 25 fields and three methods, and the lines that
are not statements are the comments explaining what each threshold gates.

## Is every line necessary

I asked the sharper version of that question: **is every threshold used?**
Searching every tracked `.py` outside this module and `tests/` for each field
name, 12 of the 25 are read somewhere in this repo and **13 are not**:

```
orthogonal_sc_min                   orthogonal_bsa_min
orthogonal_ddg_max                  orthogonal_af3_ra_max
orthogonal_plddt_min_informational  composite_ra_eff_weight
interface_plddt_trim_threshold      controls_warning_ipsae_max
controls_warning_ra_eff_min         max_poly_x
min_pct_identity                    max_pct_identity
sc_threshold
```

That looks like 13 removable fields. It is not, and this is why the proposal
is to change nothing. Every one of the 13 is read by
`receptor-resurfacing-pipeline`, which ships its own copy of this module:

| Field | Files using it there |
|:---|---:|
| `sc_threshold` | 46 |
| `max_poly_x`, `min_pct_identity`, `max_pct_identity` | 31 each |
| `interface_plddt_trim_threshold` | 31 |
| `controls_warning_ipsae_max`, `controls_warning_ra_eff_min` | 30 each |
| `orthogonal_af3_ra_max` | 5 |
| `orthogonal_sc_min`, `orthogonal_bsa_min`, `orthogonal_ddg_max` | 3 each |
| `composite_ra_eff_weight` | 2 |
| `orthogonal_plddt_min_informational` | 1 |

The module's own docstring says what it is for: "one catalogue holding every
research threshold that used to be hard-coded across `bin/`", and "Changing a
threshold here is a change to the research methodology". Its value is that
both repos read one catalogue. Deleting the fields this repo happens not to
read would fork it, and the next time the two are compared the diff would look
like a methodology change rather than a deletion.

So the honest answer to "is every line necessary for the tool" is **no, 13
fields are not needed by this tool** — and they should stay anyway, because
the thing they serve is the shared catalogue rather than this repo's engine.

## What I would do instead

One comment, at the top of the class, saying that. Right now a reader who
greps for `sc_threshold` and finds nothing has no way to learn why it exists.
That is three lines, not a refactor, and I have not written it as a diff
because it is the kind of sentence you may want to word yourself.

## Speed

Not applicable. The dataclass is constructed once per process via
`PipelineInternalThresholds.default()`.
