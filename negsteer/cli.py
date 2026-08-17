"""The `negsteer` command line interface.

One command, one run directory. This is the supported way to call negative
steering from another pipeline, in the same way RFDiffusion or ProteinMPNN
would be called: a containerised binary with explicit inputs and a documented
output contract.

    singularity exec --nv negsteer.img negsteer run \
        --name design_00 \
        --receptor-fasta receptor.fasta \
        --effector-fasta effector.fasta \
        --reference-pdb  complex.pdb \
        --design-region  design_region.txt \
        --true-interface true_interface.txt \
        --outdir         design_00

It runs the same nine stages, in the same order, with the same arguments as
bin/negative_steering_run_one.sh, which produced the committed benchmark. The
difference is the interface, not the science.

On nesting: the stages call `boltz` directly when already inside a container,
which the engine detects from SINGULARITY_NAME. That is why this can run
in-container while the old shell orchestrator could not, and why
--boltz-container is only needed when running on a bare host.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import (
    OUTPUT_CONTRACT_VERSION,
    OUTPUT_FILES,
    RESULT_MANIFEST,
    __version__,
)


def _find_bin() -> Path:
    """Locate the engine scripts.

    bin/ is a directory of standalone entry points invoked by path, not an
    importable package, so it has to be found rather than imported. Three
    places, most specific first:

      NEGSTEER_BIN            an explicit override, for an unusual layout
      <package>/../bin        a source checkout, where negsteer/ sits beside bin/
      /opt/...  /bin          the container, where pip has put the package in
                              site-packages and bin/ was staged separately

    Raising here beats failing at the first stage with "no such file", which
    reads as a broken engine rather than a broken installation.
    """
    # An override that is set but wrong is an error, not a hint. Falling
    # through would run a different engine than the caller asked for, and
    # a typo would look like it worked.
    override = os.environ.get("NEGSTEER_BIN")
    if override:
        path = Path(override)
        if not (path / "boltz2_negative_steering.py").is_file():
            raise FileNotFoundError(
                f"NEGSTEER_BIN is set to {path}, which holds no "
                "boltz2_negative_steering.py")
        return path

    candidates = [
        Path(__file__).resolve().parent.parent / "bin",
        Path("/opt/structure-negative-steering/bin"),
    ]
    for c in candidates:
        if (c / "boltz2_negative_steering.py").is_file():
            return c
    raise FileNotFoundError(
        "cannot find the engine scripts. Looked in:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nSet NEGSTEER_BIN to the directory holding "
          "boltz2_negative_steering.py.")


BIN = _find_bin()

ENGINE = BIN / "boltz2_negative_steering.py"
ITERATE = BIN / "boltz2_iterate_steering.py"
EXTRACT = BIN / "extract_passing.py"
CROSS_SUMMARY = BIN / "cross_summary_cli.py"

# predict-one and predict-reversion-one return 1 for a per-design failure and
# 0 otherwise. Anything else is fatal, which is what the shell orchestrator
# did and what stops one bad design taking the cohort with it.
PER_DESIGN_FAILURE = 1


class StageError(RuntimeError):
    """A stage exited non-zero in a way that is not a per-design failure."""


def _in_container() -> bool:
    return bool(
        os.environ.get("SINGULARITY_NAME")
        or os.environ.get("SINGULARITY_CONTAINER")
        or os.environ.get("APPTAINER_NAME")
        or os.environ.get("APPTAINER_CONTAINER")
    )


def _running_image() -> str | None:
    """Path of the image we are running inside, if we are.

    The runtime exports it, and it is the honest answer to "which image should
    the engine use", because it is the one already loaded.
    """
    return (os.environ.get("SINGULARITY_CONTAINER")
            or os.environ.get("APPTAINER_CONTAINER"))


def _resolve_container(explicit: str | None) -> str | None:
    """The image the plan stage is told about.

    plan requires --boltz-container whether or not it will use it. Inside a
    container it does not: it detects the runtime and calls boltz directly,
    nested singularity being unsupported. But argparse still demands the flag,
    so omitting it fails before the engine gets to decide anything.

    So pass the running image. It is accurate rather than a placeholder, and it
    keeps the flag meaningful if the engine ever does shell out again.
    """
    return explicit or _running_image()


def _run(argv: list[str], *, tolerate: int | None = None, dry: bool = False) -> int:
    printable = " ".join(str(a) for a in argv)
    print(f"  $ {printable}", flush=True)
    if dry:
        return 0
    rc = subprocess.run([str(a) for a in argv]).returncode
    if rc != 0 and rc != tolerate:
        raise StageError(f"exited {rc}: {printable}")
    return rc


def _stage(script: Path, *args) -> list[str]:
    return [sys.executable, str(script), *[str(a) for a in args]]


def _count(path: Path, key: str) -> int:
    """Read a fan-out width from a plan file, tolerating its absence."""
    if not path.is_file():
        return 0
    try:
        return len(json.loads(path.read_text()).get(key, []))
    except (json.JSONDecodeError, OSError):
        return 0


def _plan_args(a: argparse.Namespace) -> list[str]:
    """Knobs forwarded to the plan stage.

    Only flags the caller actually set are forwarded, so an unset knob keeps
    the engine's own default rather than this CLI silently restating it.
    """
    out: list[str] = []
    for flag, value in (
        ("--n-designs", a.n_designs),
        ("--num-seeds", a.num_seeds),
        ("--diffusion-samples", a.diffusion_samples),
        ("--recycling-steps", a.recycling_steps),
        ("--max-mutations", a.max_mutations),
        ("--candidate-pool-size", a.candidate_pool_size),
        ("--protected-set-source", a.protected_set_source),
        ("--contact-cutoff", a.contact_cutoff),
        ("--seed", a.seed),
    ):
        if value is not None:
            out += [flag, str(value)]
    if a.boltz_constraints:
        out.append("--boltz-constraints")
    if a.no_effector_template:
        out.append("--no-effector-template")
    if a.plan_extra_args:
        out += a.plan_extra_args.split()
    return out


def cmd_run(a: argparse.Namespace) -> int:
    outdir = Path(a.outdir).resolve()
    cycle0 = outdir / "cycle_0"
    outdir.mkdir(parents=True, exist_ok=True)

    image = _resolve_container(a.boltz_container)
    if not image:
        sys.stderr.write(
            "ERROR: no Boltz-2 image. Not running inside a container, and no\n"
            "       --boltz-container given. Pass one, or invoke this whole command\n"
            "       with singularity exec --nv <image> negsteer run ...\n")
        return 2

    # Always passed. plan requires it even when it will not use it.
    container = ["--boltz-container", image]
    started = time.time()

    print(f"negsteer {__version__} — {a.name}", flush=True)

    print("\n--- 1/9 plan", flush=True)
    rc = _run(_stage(
        ENGINE, "plan",
        "--ground-truth", a.reference_pdb,
        "--receptor", a.receptor_chain,
        "--effector", a.effector_chain,
        "--workdir", cycle0,
        "--receptor-fasta", a.receptor_fasta,
        "--effector-fasta", a.effector_fasta,
        "--true-interface-indices-file", a.true_interface,
        "--design-region-indices-file", a.design_region,
        *container, *_plan_args(a),
    ), tolerate=PER_DESIGN_FAILURE, dry=a.dry_run)

    plan = cycle0 / "plan.json"
    if not a.dry_run:
        if not plan.is_file():
            raise StageError(f"plan stage produced no {plan}")
        # skip_steering is a legitimate soft exit: the cold start already
        # passed, so there is nothing to steer away from.
        skipped = bool(json.loads(plan.read_text()).get("skip_steering"))
        if rc != 0 and not skipped:
            raise StageError(f"plan stage exited {rc}")
    else:
        skipped = False

    n_designs = 0 if skipped else _count(plan, "designs")
    print(f"\n--- 2/9 predict-one x {n_designs}", flush=True)
    for i in range(n_designs):
        _run(_stage(ENGINE, "predict-one", "--workdir", cycle0, "--index", i),
             tolerate=PER_DESIGN_FAILURE, dry=a.dry_run)

    print("\n--- 3/9 collect", flush=True)
    _run(_stage(ENGINE, "collect", "--workdir", cycle0), dry=a.dry_run)

    print("\n--- 4/9 reversion prep", flush=True)
    for sub, where in (("kickoff-distances", ("--experiment-root", outdir)),
                       ("kickoff-prefilter", ("--experiment-root", outdir)),
                       ("build-contaminated", ("--workdir", cycle0)),
                       ("plan-reversions", ("--workdir", cycle0))):
        _run(_stage(ITERATE, sub, *where), dry=a.dry_run)

    n_reversions = _count(cycle0 / "reversion_plan.json", "entries")
    print(f"\n--- 5/9 predict-reversion-one x {n_reversions}", flush=True)
    for i in range(n_reversions):
        _run(_stage(ITERATE, "predict-reversion-one", "--workdir", cycle0, "--index", i),
             tolerate=PER_DESIGN_FAILURE, dry=a.dry_run)

    print("\n--- 6/9 harvest + finalize", flush=True)
    _run(_stage(ITERATE, "harvest-reversions", "--workdir", cycle0), dry=a.dry_run)
    # --max-cycles 0 finalises this cycle without spawning another. This tool
    # is single-cycle by construction and has no path to submit a follow-up.
    _run(_stage(ITERATE, "kickoff-finalize",
                "--experiment-root", outdir,
                "--max-cycles", 0,
                "--max-passing", a.max_passing,
                "--novelty-cutoff", a.novelty_cutoff), dry=a.dry_run)

    print("\n--- 7/9 aggregate + metrics", flush=True)
    _run(_stage(ITERATE, "aggregate", "--experiment-root", outdir), dry=a.dry_run)
    _run(_stage(ITERATE, "compute-final-metrics",
                "--experiment-root", outdir,
                "--rmsd-threshold", a.rmsd_threshold,
                "--metric-column", a.metric_column,
                "--contact-cutoff", a.postprocess_contact_cutoff,
                *(["--populate-all"] if a.populate_all else [])), dry=a.dry_run)
    _run(_stage(ITERATE, "aggregate-per-sequence", "--experiment-root", outdir),
         dry=a.dry_run)

    # Stamped before extract_passing so it measures the Boltz and metrics work,
    # which is the comparable quantity. extract_passing reads it back into
    # passing_summary.csv, so it has to exist by now.
    elapsed = int(time.time() - started)
    if not a.dry_run:
        (outdir / OUTPUT_FILES["runtime_seconds"]).write_text(f"{elapsed}\n")

    print("\n--- 8/9 extract passing", flush=True)
    _run(_stage(EXTRACT, "--input", outdir / OUTPUT_FILES["aggregated_results"]),
         dry=a.dry_run)

    # Non-fatal. The engine results are already written, and a tiering hiccup
    # must not fail a run that produced them.
    print("\n--- 9/9 cross summary", flush=True)
    tiered = True
    try:
        _run(_stage(CROSS_SUMMARY,
                    "--passing-summary", f"{a.name}={outdir / OUTPUT_FILES['passing_summary']}",
                    "--output", outdir / OUTPUT_FILES["cross_sequence_summary"]),
             dry=a.dry_run)
    except StageError as exc:
        tiered = False
        sys.stderr.write(f"WARNING: tiering failed, results preserved. {exc}\n")

    manifest = {
        "contract_version": OUTPUT_CONTRACT_VERSION,
        "negsteer_version": __version__,
        "name": a.name,
        "status": "skipped_steering" if skipped else "ok",
        "skip_steering": skipped,
        "n_designs_planned": n_designs,
        "n_reversions_planned": n_reversions,
        "tiering_succeeded": tiered,
        "runtime_seconds": elapsed,
        "outputs": {
            key: (name if (outdir / name).exists() else None)
            for key, name in OUTPUT_FILES.items()
        },
    }
    if not a.dry_run:
        (outdir / RESULT_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nnegsteer done — {a.name} in {elapsed // 60}m {elapsed % 60}s")
    print(f"  {outdir}/{RESULT_MANIFEST}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="negsteer",
        description="Negative steering on a receptor/effector pair.")
    p.add_argument("--version", action="version",
                   version=f"negsteer {__version__} "
                           f"(output contract v{OUTPUT_CONTRACT_VERSION})")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run negative steering, write a run directory")

    req = r.add_argument_group("inputs")
    req.add_argument("--name", required=True,
                     help="identifier for this run; keys the tiering output")
    req.add_argument("--receptor-fasta", required=True)
    req.add_argument("--effector-fasta", required=True)
    req.add_argument("--reference-pdb", required=True,
                     help="solved or designed complex, used as the comparison basis")
    req.add_argument("--design-region", required=True,
                     help="1-based residue list that must not be mutated")
    req.add_argument("--true-interface", required=True,
                     help="0-based residue indices of the interface to protect")
    req.add_argument("--outdir", required=True)

    ch = r.add_argument_group("chains")
    ch.add_argument("--receptor-chain", default="A")
    ch.add_argument("--effector-chain", default="B")

    st = r.add_argument_group("steering knobs, unset means the engine default")
    st.add_argument("--n-designs", type=int)
    st.add_argument("--num-seeds", type=int)
    st.add_argument("--diffusion-samples", type=int)
    st.add_argument("--recycling-steps", type=int)
    st.add_argument("--max-mutations", type=int)
    st.add_argument("--candidate-pool-size", type=int)
    st.add_argument("--protected-set-source")
    st.add_argument("--contact-cutoff", type=float)
    st.add_argument("--seed", type=int)
    st.add_argument("--boltz-constraints", action="store_true",
                    help="add reference-derived restraints; needs a known interface")
    st.add_argument("--no-effector-template", action="store_true")
    st.add_argument("--plan-extra-args", default="",
                    help="escape hatch, forwarded to the plan stage verbatim")

    pp = r.add_argument_group("post-processing")
    pp.add_argument("--rmsd-threshold", type=float, default=5.0)
    pp.add_argument("--metric-column", default="receptor_aligned_effector_rmsd")
    pp.add_argument("--postprocess-contact-cutoff", type=float, default=5.0)
    pp.add_argument("--populate-all", action="store_true")
    pp.add_argument("--max-passing", type=int, default=5)
    pp.add_argument("--novelty-cutoff", type=float, default=10.0)

    rt = r.add_argument_group("runtime")
    rt.add_argument("--boltz-container",
                    help="only needed when not already inside a container")
    rt.add_argument("--dry-run", action="store_true",
                    help="print the stage commands without running them")
    r.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except StageError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
