#!/usr/bin/env python3
"""Run negative steering from a YAML config. The Python half of the runner.

This runs INSIDE the Boltz container, which ships yaml, gemmi, biopython and
numpy. The airgapped HPC has nothing on the host, so
scripts/negative_steering.slurm.sh invokes it via `singularity exec`.

It parses and validates the config, resolving every path relative to the
config file's own directory, then prepares the engine inputs with
bin/prepare_complex_inputs.py into <cfg_dir>/inputs/.

It writes, but never runs, <cfg_dir>/run/launch.sh, which calls the engine
orchestrator with every argument resolved. Launching is the host's job, since
the orchestrator does its own `singularity exec --nv` and singularity cannot
nest. launch.sh records what was run.

Its 20 print calls look like a reporting-heavy file and are not. They are four
`ERROR:` guards, one JSON emission, four --dry-run lines and a six-line run
summary, each saying something the others do not, so there is nothing to fold.

Usage, normally via negative_steering.slurm.sh, but runnable on a Mac with the
[runtime] extra:

  negative_steering.py CONFIG      # prepare + write launch.sh
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPARE = REPO_ROOT / "bin" / "prepare_complex_inputs.py"
ENGINE_BIN = REPO_ROOT / "bin"
ORCHESTRATOR = ENGINE_BIN / "negative_steering_run_one.sh"

REQUIRED = ("reference_pdb", "receptor_chain", "effector_chain")

# Written by prepare_complex_inputs.py, consumed by the orchestrator.
INPUT_FILES = ("receptor.fasta", "effector.fasta", "true_interface.txt",
               "design_region.txt")


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        raise SystemExit(
            "ERROR: PyYAML is required to read the config. It ships in the Boltz "
            "container, so run this via scripts/negative_steering.slurm.sh. "
            "On a Mac: pip install -e '.[test]'."
        )
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"ERROR: {path} did not parse to a mapping.")
    return data


def _resolve(cfg_dir: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (cfg_dir / p)


def _build_prepare_cmd(cfg: Dict[str, Any], cfg_dir: Path, inputs_dir: Path) -> List[str]:
    """Build the prepare_complex_inputs.py command for this config."""
    cmd = [
        sys.executable, str(PREPARE),
        "--complex", str(_resolve(cfg_dir, cfg["reference_pdb"])),
        "--outdir", str(inputs_dir),
        "--receptor-chain", str(cfg["receptor_chain"]),
        "--effector-chain", str(cfg["effector_chain"]),
    ]
    mode = str(cfg.get("interface_mode", "derive")).lower()
    if mode == "provide":
        rf = cfg.get("interface_residues_file")
        if not rf:
            raise SystemExit("ERROR: interface_mode: provide needs interface_residues_file.")
        cmd += ["--interface-file", str(_resolve(cfg_dir, rf))]
    else:
        cmd += ["--derive", "--contact-cutoff", str(cfg.get("interface_contact_cutoff", 5.0))]

    dr = cfg.get("design_region", "none")
    if isinstance(dr, str) and dr.lower() in ("none", "all", "interface"):
        cmd += ["--design-region", dr.lower()]
    else:
        cmd += ["--design-region-file", str(_resolve(cfg_dir, str(dr)))]

    ablate = str(cfg.get("interface_ablation", "none")).lower()
    if ablate not in ("none", "alanine", "scrambled"):
        raise SystemExit(
            f"ERROR: interface_ablation must be none, alanine or scrambled (got: {ablate})")
    if ablate != "none":
        cmd += ["--ablate", ablate]
        if cfg.get("ablation_seed") is not None:
            cmd += ["--ablate-seed", str(cfg["ablation_seed"])]
    return cmd


def _build_plan_extra_args(cfg: Dict[str, Any]) -> str:
    """Build the --plan-extra-args string for the orchestrator.

    Every default lives here, so a config file names only the keys it changes.
    """
    parts = [
        f"--mode {cfg.get('negsteer_mode', 'strong')}",
        f"--max-mutations {cfg.get('negsteer_max_mutations', 6)}",
        f"--candidate-pool-size {cfg.get('negsteer_candidate_pool_size', 6)}",
        f"--protected-set-source {cfg.get('negsteer_protected_set_source', 'design_region_union')}",
        f"--n-designs {cfg.get('negsteer_n_designs', 10)}",
        f"--num-seeds {cfg.get('negsteer_num_seeds', 1)}",
        f"--diffusion-samples {cfg.get('negsteer_diffusion_samples', 1)}",
        f"--recycling-steps {cfg.get('negsteer_recycling_steps', 3)}",
        f"--seed {cfg.get('negsteer_seed', 0)}",
        f"--rmsd-threshold {cfg.get('negsteer_rmsd_threshold', 5.0)}",
        f"--contact-cutoff {cfg.get('negsteer_contact_cutoff', 5.0)}",
    ]
    # Opt-in. The engine's own default is off, so a config that omits the
    # key gets no template.
    if cfg.get("negsteer_effector_template", False):
        parts.append("--effector-template")
    if cfg.get("negsteer_no_kernels", False):
        parts.append("--no-kernels")
    if cfg.get("negsteer_boltz_constraints", False):
        parts.append("--boltz-constraints")
    return " ".join(parts)


def _build_run_cmd(cfg: Dict[str, Any], cfg_dir: Path, inputs_dir: Path,
                   run_dir: Path, boltz_container: str) -> List[str]:
    """Build the orchestrator command, every path resolved."""
    return [
        "bash", str(ORCHESTRATOR),
        "--seq-name", str(cfg.get("name", cfg_dir.name)),
        "--ground-truth", str(_resolve(cfg_dir, cfg["reference_pdb"])),
        "--receptor-chain", str(cfg["receptor_chain"]),
        "--effector-chain", str(cfg["effector_chain"]),
        "--receptor-fasta", str(inputs_dir / "receptor.fasta"),
        "--effector-fasta", str(inputs_dir / "effector.fasta"),
        "--true-interface-indices-file", str(inputs_dir / "true_interface.txt"),
        "--design-region-indices-file", str(inputs_dir / "design_region.txt"),
        "--workdir", str(run_dir),
        "--bin-dir", str(ENGINE_BIN),
        "--boltz-container", boltz_container,
        "--plan-extra-args", _build_plan_extra_args(cfg),
        "--postprocess-rmsd-threshold",
        str(cfg.get("negsteer_postprocess_rmsd_threshold", 5.0)),
        "--postprocess-metric-column",
        str(cfg.get("negsteer_postprocess_metric_column", "steered_ra_eff_vs_truth")),
        "--postprocess-contact-cutoff",
        str(cfg.get("negsteer_postprocess_contact_cutoff", 5.0)),
    ]


def _check_inputs_present(inputs_dir: Path) -> None:
    """Raise SystemExit if prepare missed any expected input file."""
    missing = [f for f in INPUT_FILES if not (inputs_dir / f).is_file()]
    if missing:
        raise SystemExit(
            f"ERROR: prepare did not produce {', '.join(missing)} in {inputs_dir}."
        )


def _write_launch_script(run_dir: Path, run_cmd: List[str]) -> Path:
    """Write run/launch.sh and return its path, without running it."""
    run_dir.mkdir(parents=True, exist_ok=True)
    launch = run_dir / "launch.sh"
    launch.write_text(
        "#!/bin/bash\n"
        "# Auto-generated by negative_steering.py. Run this ON THE HOST.\n"
        "# The orchestrator it calls does its own `singularity exec --nv`, and\n"
        "# singularity cannot be nested, so this cannot run inside a container.\n"
        "set -euo pipefail\n"
        "exec " + shlex.join(run_cmd) + "\n"
    )
    launch.chmod(0o755)
    return launch


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--prepare-only", action="store_true",
                    help="Prepare inputs only; do not write launch.sh.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the prepare/run commands without changing anything.")
    ap.add_argument("--boltz-container", default=None,
                    help="Container path baked into launch.sh (overrides config).")
    ap.add_argument("--emit-json", action="store_true",
                    help="Print the resolved config as JSON and exit. main.nf "
                         "uses this so the config defaults live here only.")
    args = ap.parse_args(argv)

    cfg_path = args.config.expanduser().resolve()
    if not cfg_path.is_file():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = _load_yaml(cfg_path)
    cfg_dir = cfg_path.parent

    missing = [k for k in REQUIRED if not cfg.get(k)]
    if missing:
        print(f"ERROR: config missing required keys: {', '.join(missing)}", file=sys.stderr)
        return 2

    complex_pdb = _resolve(cfg_dir, cfg["reference_pdb"])
    if not complex_pdb.is_file():
        print(f"ERROR: reference_pdb not found: {complex_pdb}", file=sys.stderr)
        return 2
    if not ORCHESTRATOR.is_file():
        print(f"ERROR: engine orchestrator not found: {ORCHESTRATOR}", file=sys.stderr)
        return 2

    inputs_dir = cfg_dir / "inputs"
    run_dir = cfg_dir / "run"
    prepare_cmd = _build_prepare_cmd(cfg, cfg_dir, inputs_dir)
    # --boltz-container, then the config, then site.conf's NEGSTEER_BOLTZ_IMG.
    # A config that leaves the key null takes the site setting, so one value
    # covers every such config.
    boltz_container = (args.boltz_container
                       or cfg.get("boltz_container")
                       or os.environ.get("NEGSTEER_BOLTZ_IMG"))

    if args.emit_json:
        # The Nextflow workflow reads this rather than parsing YAML, so the
        # config keys and defaults are defined in one place.
        print(json.dumps({
            "name": str(cfg.get("name", cfg_dir.name)),
            "config": str(cfg_path),
            "cfg_dir": str(cfg_dir),
            "inputs_dir": str(inputs_dir),
            "run_dir": str(run_dir),
            "container": str(boltz_container) if boltz_container else "",
            "plan_args": " ".join([
                "--ground-truth", str(complex_pdb),
                "--receptor", str(cfg["receptor_chain"]),
                "--effector", str(cfg["effector_chain"]),
                "--receptor-fasta", str(inputs_dir / "receptor.fasta"),
                "--effector-fasta", str(inputs_dir / "effector.fasta"),
                "--true-interface-indices-file", str(inputs_dir / "true_interface.txt"),
                "--design-region-indices-file", str(inputs_dir / "design_region.txt"),
                _build_plan_extra_args(cfg),
            ]),
            "postprocess_rmsd_threshold":
                str(cfg.get("negsteer_postprocess_rmsd_threshold", 5.0)),
            "postprocess_metric_column":
                str(cfg.get("negsteer_postprocess_metric_column",
                            "steered_ra_eff_vs_truth")),
            "postprocess_contact_cutoff":
                str(cfg.get("negsteer_postprocess_contact_cutoff", 5.0)),
        }))
        return 0

    if args.dry_run:
        print("# step 1 - prepare inputs (in container):")
        print(shlex.join(prepare_cmd))
        if boltz_container:
            print("\n# step 2 - launch.sh would contain (run on host):")
            print(shlex.join(
                _build_run_cmd(cfg, cfg_dir, inputs_dir, run_dir, str(boltz_container))))
        else:
            print("\n# (no boltz_container set, so the run step cannot be generated)")
        return 0

    print(f"=== prepare inputs -> {inputs_dir} ===")
    subprocess.run(prepare_cmd, check=True)
    _check_inputs_present(inputs_dir)

    if args.prepare_only:
        print("\nprepare-only: inputs ready, launch.sh not written.")
        return 0

    if not boltz_container:
        print("ERROR: no Boltz image. Pass --boltz-container, set "
              "boltz_container in the config, or set NEGSTEER_BOLTZ_IMG in "
              "site.conf. Use --prepare-only to skip the run step.",
              file=sys.stderr)
        return 2

    run_cmd = _build_run_cmd(cfg, cfg_dir, inputs_dir, run_dir, str(boltz_container))
    launch = _write_launch_script(run_dir, run_cmd)
    print(f"\n=== negative steering: {cfg.get('name', cfg_dir.name)} ===")
    print(f"  complex (ground truth): {complex_pdb} "
          f"(receptor={cfg['receptor_chain']}, effector={cfg['effector_chain']})")
    print(f"  inputs:                 {inputs_dir}")
    print(f"  workdir:                {run_dir}")
    print(f"  container:              {boltz_container}")
    print(f"\nWrote {launch}")
    print(f"LAUNCH_SCRIPT={launch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
