#!/usr/bin/env python3
"""Run negative steering from a YAML config — the Python brain.

This script is meant to run INSIDE a container (the Boltz image ships yaml +
gemmi + biopython + numpy).  On the airgapped HPC nothing is installed on the
host, so scripts/run_from_config.sh invokes this via `singularity exec`.

It does the two things that need Python:
  1. parse the config;
  2. prepare the engine inputs (scripts/prepare_inputs.py) into <cfg_dir>/inputs/.

It then WRITES (but never runs) <cfg_dir>/run/launch.sh — a host-runnable
script that invokes the engine.  Launching the engine is the host's job because
the orchestrator does its own `singularity exec --nv` and you cannot nest
singularity from inside a container.

All config paths are resolved relative to the config file's directory.

Usage (normally via run_from_config.sh, but also runnable directly on a Mac with
the [experiments] extras installed):
  run_from_config.py CONFIG                      # prepare + write launch.sh
  run_from_config.py CONFIG --prepare-only       # prepare only, no launch.sh
  run_from_config.py CONFIG --dry-run            # print commands, change nothing
  run_from_config.py CONFIG --boltz-container IMG # container baked into launch.sh
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPARE = REPO_ROOT / "scripts" / "prepare_inputs.py"
RUNNER = REPO_ROOT / "scripts" / "run_negative_steering.sh"

REQUIRED = ("reference_pdb", "receptor_chain", "effector_chain")


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        raise SystemExit(
            "ERROR: PyYAML is required to read the config. It ships in the Boltz "
            "container — run this via scripts/run_from_config.sh (singularity), "
            "or on a Mac: pip install -e '.[test]'."
        )
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"ERROR: {path} did not parse to a mapping.")
    return data


def _resolve(cfg_dir: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (cfg_dir / p)


def _build_prepare_cmd(cfg: Dict[str, Any], cfg_dir: Path, inputs_dir: Path) -> List[str]:
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
    return cmd


def _build_plan_extra_args(cfg: Dict[str, Any]) -> str:
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
    if not cfg.get("negsteer_effector_template", True):
        parts.append("--no-effector-template")
    if cfg.get("negsteer_no_kernels", False):
        parts.append("--no-kernels")
    return " ".join(parts)


def _build_run_cmd(cfg: Dict[str, Any], cfg_dir: Path, inputs_dir: Path,
                   run_dir: Path, boltz_container: str) -> List[str]:
    return [
        "bash", str(RUNNER),
        "--name", str(cfg.get("name", cfg_dir.name)),
        "--complex", str(_resolve(cfg_dir, cfg["reference_pdb"])),
        "--inputs-dir", str(inputs_dir),
        "--boltz-container", boltz_container,
        "--workdir", str(run_dir),
        "--receptor-chain", str(cfg["receptor_chain"]),
        "--effector-chain", str(cfg["effector_chain"]),
        "--plan-extra-args", _build_plan_extra_args(cfg),
        "--",
        "--postprocess-rmsd-threshold", str(cfg.get("negsteer_postprocess_rmsd_threshold", 5.0)),
        "--postprocess-metric-column", str(cfg.get("negsteer_postprocess_metric_column",
                                                    "steered_ra_eff_vs_truth")),
        "--postprocess-contact-cutoff", str(cfg.get("negsteer_postprocess_contact_cutoff", 5.0)),
    ]


def _write_launch_script(run_dir: Path, run_cmd: List[str]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    launch = run_dir / "launch.sh"
    launch.write_text(
        "#!/bin/bash\n"
        "# Auto-generated by run_from_config.py. Run this ON THE HOST.\n"
        "# The orchestrator it calls does its own `singularity exec --nv`.\n"
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

    inputs_dir = cfg_dir / "inputs"
    run_dir = cfg_dir / "run"
    prepare_cmd = _build_prepare_cmd(cfg, cfg_dir, inputs_dir)
    boltz_container = args.boltz_container or cfg.get("boltz_container")

    if args.dry_run:
        print("# step 1 — prepare inputs (in container):")
        print(shlex.join(prepare_cmd))
        if boltz_container:
            print("\n# step 2 — launch.sh would contain (run on host):")
            print(shlex.join(_build_run_cmd(cfg, cfg_dir, inputs_dir, run_dir, str(boltz_container))))
        else:
            print("\n# (no boltz_container set — run step cannot be generated)")
        return 0

    print(f"=== prepare inputs -> {inputs_dir} ===")
    subprocess.run(prepare_cmd, check=True)

    if args.prepare_only:
        print("\nprepare-only: inputs ready; launch.sh not written.")
        return 0

    if not boltz_container:
        print("ERROR: no boltz_container in config and --boltz-container not given; "
              "cannot write launch.sh (use --prepare-only to skip).", file=sys.stderr)
        return 2

    run_cmd = _build_run_cmd(cfg, cfg_dir, inputs_dir, run_dir, str(boltz_container))
    launch = _write_launch_script(run_dir, run_cmd)
    print(f"\nWrote {launch}")
    print(f"LAUNCH_SCRIPT={launch}")
    print(f"Launch the engine on the host with:  bash {launch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
