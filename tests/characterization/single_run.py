"""Where the single-run test tree is, and what its config says to expect.

Two modules assert on the tree that tests/test_run.slurm.sh produces:
test_engine_single_run.py on one run, and
test_single_run_cli_matches_orchestrator.py on an orchestrator run against a
--via-cli one. Both need the same two things, the tree's location and the
config's own values, so both read them from here.

This module exists because of what happened when they did not. Every expected
value used to be written into the assertions by hand. Commit 8429e44 moved the
config from 6Q76 at one design, one seed and one diffusion sample to 6G10 at
3 x 3 x 5, and the literals stayed, so five assertions failed against a
perfectly good run. The CLI comparison carried the same stale literal in
`6Q76_single_run` and a worse version of the same fault: it read an
environment variable, NEGSTEER_SINGLE_RUN_BASE, that nothing in the repository
sets, while the runner exports NEGSTEER_SINGLE_RUN_DIR. On a relocated tree it
therefore looked at the repository default, found nothing and skipped, which
reads as "nothing to compare" rather than as a broken test.

Nobody saw either for the same reason the tier exists. Running it needs a
finished GPU run.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# NEGSTEER_SINGLE_RUN_DIR is the one name, and tests/test_run.slurm.sh exports
# it when it runs the assertions against a tree somewhere other than the
# checkout.
SINGLE_RUN = Path(os.environ.get("NEGSTEER_SINGLE_RUN_DIR",
                                 REPO_ROOT / "tests" / "test_run"))


def read_csv(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def config() -> dict[str, str]:
    """The run's own config.yml, as plain key to string value.

    Hand-parsed rather than via PyYAML: these modules are imported on the
    laptop tier too, where yaml is in the `runtime` extra and nothing else here
    needs it. Only scalar `key: value` lines matter.
    """
    cfg: dict[str, str] = {}
    path = SINGLE_RUN / "config.yml"
    if not path.is_file():
        return cfg
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, _, v = line.partition(":")
        cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg


CFG = config()
N_DESIGNS = int(CFG.get("negsteer_n_designs", 0) or 0)
NUM_SEEDS = int(CFG.get("negsteer_num_seeds", 0) or 0)
DIFFUSION = int(CFG.get("negsteer_diffusion_samples", 0) or 0)
RUN_NAME = CFG.get("name", "")
