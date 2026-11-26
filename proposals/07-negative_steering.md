# 07 — `bin/negative_steering.py`, 296 lines

**Proposal: change nothing.** I went in expecting to fold the four
validation guards and concluded against it. The reasoning is below,
because the reason not to is the useful part.

## What I expected to find, and did not

296 lines with 20 `print` calls looked like a reporting-heavy file. It is not.
The prints are four `ERROR:` guards, one JSON emission, four `--dry-run`
lines, and a six-line run summary. Each says something different. There is no
repetition to fold.

## The one change worth taking: the four validation guards

```python
if not cfg_path.is_file():
    print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
    return 2
...
missing = [k for k in REQUIRED if not cfg.get(k)]
if missing:
    print(f"ERROR: config missing required keys: {', '.join(missing)}", file=sys.stderr)
    return 2
...
if not complex_pdb.is_file():
    print(f"ERROR: reference_pdb not found: {complex_pdb}", file=sys.stderr)
    return 2
if not ORCHESTRATOR.is_file():
    print(f"ERROR: engine orchestrator not found: {ORCHESTRATOR}", file=sys.stderr)
    return 2
```

Three of the four are the same shape: check a path, print, return 2. One line
each with a helper:

```python
def _require_file(path, what):
    """None when the path is a file, else the error message to print."""
    return None if path.is_file() else f"ERROR: {what} not found: {path}"
```

That saves 6 lines of 296 and I am **not proposing it**, for one reason: the
four guards fire in a required order. `cfg_path` has to be checked before
`_load_yaml` reads it, `missing` before `cfg["reference_pdb"]` is indexed, and
`complex_pdb` after the config is parsed. A list of checks run in a loop would
either lose that ordering or need the loop interleaved with the parsing, which
is longer than what is there now.

The straight-line version makes the order obvious. That is worth six lines.

## What I would change

Nothing. This file is 135 statements in 296 lines and the gap is docstrings
and the `--dry-run` help text, both of which earn their space.

## Is every line necessary

One thing is close to unnecessary and I checked it. `_load_engine_helpers`
in `prepare_complex_inputs.py` (see 08) has an unguarded `sys.path.insert`
that adds an entry per call. In this file the equivalent is at module level
and runs once, so the same pattern is harmless here.

## Speed

Measured nothing, because there is nothing to measure. The file parses one
YAML file of about 60 keys and writes one shell script. It runs twice per
target, once for `--emit-json` in READ_CONFIG and once in PREPARE.
