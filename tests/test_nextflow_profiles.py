"""Profile structure, because getting it wrong sent Boltz to a node with no GPU.

Nextflow 25.10.x REPLACES a `withLabel:` block when a profile defines the same
selector. 26.04 merges. The cluster runs 25.10.4, so

    hpc { process { withLabel: gpu { executor = 'slurm' } } }

collapsed the gpu label to that one line, dropping queue, cpus, memory, time and
clusterOptions. Nextflow submitted a bare sbatch, SLURM placed the job on
whatever partition it liked with one CPU and no GPU, and Boltz died with "No
supported gpu backend found!".

These are text-level checks, so they run anywhere with no Nextflow installed.
CI resolves the config with the cluster's own Nextflow and asserts the queue and
the --gres survive, which is the authoritative version of this.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "nextflow.config"

# The cluster profile. Directives for it live in the top-level `process` block.
CLUSTER_PROFILE = "hpc"


def _profile_bodies() -> dict[str, str]:
    """Each profile's body, by brace matching from `profiles {`."""
    text = CONFIG.read_text()
    start = text.index("profiles {")
    depth, i = 0, start + len("profiles")
    while True:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    body = text[start + len("profiles {"): i]

    out: dict[str, str] = {}
    for m in re.finditer(r"^\s{4}(\w+)\s*\{", body, flags=re.M):
        name = m.group(1)
        d, j = 0, m.end() - 1
        while True:
            if body[j] == "{":
                d += 1
            elif body[j] == "}":
                d -= 1
                if d == 0:
                    break
            j += 1
        out[name] = body[m.end(): j]
    return out


@pytest.mark.logic
def test_there_is_no_profile_named_standard() -> None:
    """Nextflow applies `standard` whenever -profile is omitted. Having one
    meant a bare `nextflow run` on the cluster silently took laptop settings."""
    assert "standard" not in _profile_bodies()


@pytest.mark.logic
def test_the_cluster_profile_redefines_no_label_selector() -> None:
    """It must not touch a `withLabel:` block at all. On 25.10 that replaces
    the block above rather than adding to it, and the resources vanish."""
    body = _profile_bodies()[CLUSTER_PROFILE]
    assert "withLabel" not in body, (
        f"`{CLUSTER_PROFILE}` redefines a withLabel selector. On the cluster's "
        "Nextflow that DROPS queue, cpus, memory, time and clusterOptions from "
        "the block above. Set a different key, or repeat every directive."
    )


@pytest.mark.logic
def test_the_cluster_directives_live_in_the_process_block() -> None:
    """Where the hpc profile leaves them, so they must actually be there.

    The partition names are not hard-coded, because they differ per site. What
    must survive is the directive itself, wired to the param that carries the
    value. A gpu label that lost `clusterOptions` gets a bare sbatch, lands on
    whatever partition SLURM picks with no card, and Boltz dies with "No
    supported gpu backend found!"."""
    text = CONFIG.read_text()
    process_block = text[text.index("\nprocess {"): text.index("profiles {")]
    for needed in ("queue          = params.cpu_queue".replace("  ", " "),
                   "queue         = params.gpu_queue".replace("  ", " "),
                   "clusterOptions = params.gpu_cluster_options"):
        flat = " ".join(process_block.split())
        assert " ".join(needed.split()) in flat, f"{needed} missing"


@pytest.mark.logic
def test_every_site_param_has_a_default_that_works_off_this_cluster() -> None:
    """A clone with no site.conf must resolve. A null queue makes Nextflow
    omit `-p` entirely and take the cluster's default partition, which is the
    only neutral answer: every site names its partitions differently.

    Nextflow 25.10.4 SlurmExecutor.groovy:98 is what makes null safe, with
    `if( task.config.queue )` guarding the directive."""
    text = CONFIG.read_text()
    for param in ("gpu_queue", "cpu_queue"):
        line = next(ln for ln in text.splitlines()
                    if ln.strip().startswith(f"{param} "))
        assert "System.getenv(" in line, f"{param} must read the environment"
        assert "?: null" in line, f"{param} must fall back to null, not a name"
    assert "NEGSTEER_GPU_CLUSTER_OPTIONS') ?: '--gres=gpu:1'" in text, (
        "the gres default must survive, because asking for no GPU is never right")


@pytest.mark.logic
def test_the_laptop_profile_is_complete_in_itself() -> None:
    """It does get replaced, so every directive it needs must be written out
    rather than inherited from the block above."""
    body = _profile_bodies()["local"]
    for label in ("tiny", "cpu", "gpu"):
        chunk = body.split(f"withLabel: {label}")[1].split("}")[0]
        assert "executor = 'local'" in chunk, f"{label} must pin the local executor"
        assert "cpus" in chunk and "memory" in chunk, f"{label} must set its own size"


@pytest.mark.logic
def test_gpu_concurrency_is_bounded_on_the_label_not_the_executor() -> None:
    """One gpu task holds one card, so maxForks on that label IS the share of
    the partition this repo takes. executor.queueSize cannot do it, because it is
    shared with the cpu-labelled stages and a 90-target sweep has many of those.
    Bounding only queueSize lets collect and postprocess tasks take slots that
    Boltz should be using."""
    text = CONFIG.read_text()
    process_block = text[text.index("\nprocess {"): text.index("profiles {")]
    gpu = process_block.split("withLabel: gpu")[1]
    assert "maxForks" in gpu, "the gpu label must bound its own concurrency"


@pytest.mark.logic
def test_queue_size_does_not_throttle_the_gpu_bound() -> None:
    """queueSize counts every SLURM task in flight. If it were at or below the
    GPU bound, the cpu stages would silently steal GPU concurrency and the
    maxForks value would stop meaning what it says."""
    text = CONFIG.read_text()
    forks = int(re.search(r"max_gpu_jobs.*\?: '(\d+)'", text).group(1))
    queue = int(re.search(r"^\s*queueSize\s*=\s*(\d+)", text, flags=re.M).group(1))
    assert queue > forks, (
        f"queueSize {queue} must exceed max_gpu_jobs {forks}, or the cpu stages "
        "compete with Boltz for the same slots.")


@pytest.mark.logic
def test_the_gpu_bound_defaults_low_and_reads_the_site() -> None:
    """The committed default is a share nobody has to think about before their
    first run. The real card count is site-specific, so it comes from
    NEGSTEER_MAX_GPU_JOBS. Asking for more cards than exist cannot help, and
    a default sized for one particular partition would monopolise a smaller
    one."""
    text = CONFIG.read_text()
    line = next(ln for ln in text.splitlines()
                if ln.strip().startswith("max_gpu_jobs"))
    assert "NEGSTEER_MAX_GPU_JOBS" in line, "the bound must read the site setting"
    default = int(re.search(r"\?: '(\d+)'", line).group(1))
    assert 0 < default <= 8, (
        f"the committed default {default} should be modest; a site raises it")
