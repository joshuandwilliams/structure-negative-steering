# 01 — `bin/sum_stage_times.py`, 47 lines

**Proposal: change nothing.**

This is the whole file minus its docstring: 24 statements, two functions.

```python
def total_seconds(stage_times: Path) -> int:
    if not stage_times.is_file():
        return 0
    total = 0
    for line in stage_times.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            total += int(parts[1])
    return total


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-times", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    args.output.write_text(f"{total_seconds(args.stage_times)}\n")
    return 0
```

## Is every line necessary

Yes. Taking them in turn:

- `is_file()` guard: a missing `.stage_times` has to give 0 rather than raise,
  because POSTPROCESS runs it unconditionally and a stub run has no stage
  times. Removing it turns an expected case into a stage failure.
- `len(parts) == 2`: the file is appended to concurrently by tasks on
  different nodes over a shared filesystem, so a torn line is possible. The
  module docstring records that one `predict_one` line was lost that way on
  the first cluster run. A line that is not exactly two fields is skipped
  rather than guessed at.
- `lstrip("-").isdigit()`: skips a non-numeric second field without raising.
  Same reason.
- `total += int(parts[1])` inside the loop rather than `sum(...)` over a
  comprehension: a comprehension would need the same two guards in a filter
  and would read worse, not better.

## Could it be shorter

Marginally, and it would be worse. The comprehension version:

```python
return sum(int(p[1]) for p in (l.split() for l in text.splitlines())
           if len(p) == 2 and p[1].lstrip("-").isdigit())
```

That is one line instead of six and I would not take it. The guard conditions
are the interesting part of this function, and burying them in a nested
generator makes the one thing worth reading hard to find.

## Speed

Irrelevant and not worth measuring. The file it reads has one line per stage,
so ten lines on a real run. This runs once per target.
