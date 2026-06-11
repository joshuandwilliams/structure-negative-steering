#!/bin/bash
#
# Standalone negative-steering runner.
#
# Thin convenience wrapper around engine/bin/negative_steering_run_one.sh: it
# fills in the engine bin-dir and the four prepared-input paths from a
# prepare_inputs.py output directory, then drives the full single-cycle
# negative-steering pipeline against one complex.
#
# REQUIRES A GPU + the Boltz-2 Singularity image (boltz2_negsteer.img) — this
# runs on the HPC, not your laptop.
#
# Usage:
#   ./scripts/run_negative_steering.sh \
#       --name 6G10_test \
#       --complex experiments/6G10/6G10.pdb \
#       --inputs-dir experiments/6G10/inputs \
#       --boltz-container /path/to/boltz2_negsteer.img \
#       [--workdir experiments/6G10/run] \
#       [--receptor-chain A] [--effector-chain B] \
#       [--plan-extra-args "--mode mild --n-designs 5 --num-seeds 1"] \
#       [-- ... any extra negative_steering_run_one.sh args ...]
#
# Prepare the --inputs-dir first with:
#   ./scripts/prepare_inputs.py --complex <pdb> --outdir <inputs-dir> [--derive | --interface-file ...]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_BIN="${REPO_ROOT}/engine/bin"
ORCHESTRATOR="${ENGINE_BIN}/negative_steering_run_one.sh"

NAME=""
COMPLEX=""
INPUTS_DIR=""
BOLTZ_CONTAINER=""
WORKDIR=""
RECEPTOR_CHAIN="A"
EFFECTOR_CHAIN="B"
PLAN_EXTRA_ARGS=""
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)             NAME="$2";             shift 2 ;;
        --complex)          COMPLEX="$2";          shift 2 ;;
        --inputs-dir)       INPUTS_DIR="$2";       shift 2 ;;
        --boltz-container)  BOLTZ_CONTAINER="$2";  shift 2 ;;
        --workdir)          WORKDIR="$2";          shift 2 ;;
        --receptor-chain)   RECEPTOR_CHAIN="$2";   shift 2 ;;
        --effector-chain)   EFFECTOR_CHAIN="$2";   shift 2 ;;
        --plan-extra-args)  PLAN_EXTRA_ARGS="$2";  shift 2 ;;
        --)                 shift; PASSTHROUGH=("$@"); break ;;
        -h|--help)          sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "run_negative_steering.sh: unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ── Required-arg + file checks ──────────────────────────────────────
missing=()
[[ -z "$NAME"            ]] && missing+=(--name)
[[ -z "$COMPLEX"         ]] && missing+=(--complex)
[[ -z "$INPUTS_DIR"      ]] && missing+=(--inputs-dir)
[[ -z "$BOLTZ_CONTAINER" ]] && missing+=(--boltz-container)
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "ERROR: missing required args: ${missing[*]}" >&2
    exit 2
fi

WORKDIR="${WORKDIR:-${REPO_ROOT}/experiments/${NAME}/run}"

REC_FASTA="${INPUTS_DIR}/receptor.fasta"
EFF_FASTA="${INPUTS_DIR}/effector.fasta"
TRUE_IFACE="${INPUTS_DIR}/true_interface.txt"
DESIGN_REGION="${INPUTS_DIR}/design_region.txt"

for f in "$ORCHESTRATOR" "$COMPLEX" "$REC_FASTA" "$EFF_FASTA" "$TRUE_IFACE" "$DESIGN_REGION"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required file not found: $f" >&2
        [[ "$f" == "$REC_FASTA" || "$f" == "$DESIGN_REGION" ]] && \
            echo "       (did you run scripts/prepare_inputs.py --outdir ${INPUTS_DIR} first?)" >&2
        exit 2
    fi
done

mkdir -p "$WORKDIR"

echo "=== negative steering: ${NAME} ==="
echo "  complex (ground truth): ${COMPLEX}  (receptor=${RECEPTOR_CHAIN}, effector=${EFFECTOR_CHAIN})"
echo "  inputs:                 ${INPUTS_DIR}"
echo "  workdir:                ${WORKDIR}"
echo "  container:              ${BOLTZ_CONTAINER}"
echo "  plan-extra-args:        ${PLAN_EXTRA_ARGS:-<none>}"
echo

bash "$ORCHESTRATOR" \
    --seq-name "$NAME" \
    --ground-truth "$COMPLEX" \
    --receptor-chain "$RECEPTOR_CHAIN" \
    --effector-chain "$EFFECTOR_CHAIN" \
    --receptor-fasta "$REC_FASTA" \
    --effector-fasta "$EFF_FASTA" \
    --true-interface-indices-file "$TRUE_IFACE" \
    --design-region-indices-file "$DESIGN_REGION" \
    --workdir "$WORKDIR" \
    --bin-dir "$ENGINE_BIN" \
    --boltz-container "$BOLTZ_CONTAINER" \
    --plan-extra-args "$PLAN_EXTRA_ARGS" \
    "${PASSTHROUGH[@]}"
orch_rc=$?
if [[ $orch_rc -ne 0 ]]; then
    echo "ERROR: negative_steering_run_one.sh exited ${orch_rc} — skipping tiering." >&2
    exit "$orch_rc"
fi

# ── Tiering: pick the representative steered design + assign cross_tier ──────
# Mirrors the pipeline's NEGSTEER_CROSS_SEQUENCE stage. cross_summary_v2.py reads
# the run's passing_summary.csv, picks ONE representative (tier-then-composite
# policy) and writes cross_sequence_summary.csv with cross_tier:
#   A: n_pass == n_seeds   B: 1 < n_pass < n_seeds   C: n_pass == 1   none: 0
# Runs in-container (CPU). NON-FATAL: the engine results are already written, so a
# tiering hiccup must not fail the run.
PASSING="${WORKDIR}/passing_summary.csv"
XSUMM="${WORKDIR}/cross_sequence_summary.csv"
if [[ -f "$PASSING" ]]; then
    echo
    echo "=== tiering: representative + cross_tier -> cross_sequence_summary.csv ==="
    if singularity exec --bind "$WORKDIR" --bind "$ENGINE_BIN" "$BOLTZ_CONTAINER" \
            python "$ENGINE_BIN/cross_summary_v2.py" \
                --passing-summary "${NAME}=${PASSING}" \
                --output "$XSUMM"; then
        echo "wrote $XSUMM"
    else
        echo "WARNING: tiering step failed; engine results preserved in $WORKDIR." >&2
    fi
else
    echo "WARNING: no passing_summary.csv in $WORKDIR — skipping tiering." >&2
fi
