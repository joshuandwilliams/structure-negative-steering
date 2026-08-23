#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# negative_steering_run_one.sh
# ─────────────────────────────────────────────────────────────────────
# Inline orchestrator for ONE MPNN sequence through the complete
# single-cycle negative-steering pipeline.  Called by the NextFlow
# NEGSTEER_RUN_ONE process after it has allocated a GPU node.
#
# This is the NextFlow-native equivalent of the six-stage SLURM chain
# in submit_boltz2_negative_steering.sh, collapsed into a single
# sequential shell flow that runs inside one allocated job.  Trading
# parallelism across designs (the old GPU array) for a dramatic cut in
# SLURM submissions — the old chain produced 3 + N*num_seeds + 3 jobs
# per MPNN sequence; this produces 1.  With N=20, num_seeds=3,
# mpnn_top_n=64, that's 64 GPU jobs total instead of ~4200.
#
# The GPU is loaded once, designs are predicted sequentially on the
# same node, reversions are run on the same node, and all
# post-processing (aggregate, compute-final-metrics,
# aggregate-per-sequence, extract_passing) runs inline as CPU work
# before the job ends.
#
# Exit code
# ---------
# 0 on success.  Non-zero only when an irrecoverable error (container
# missing, plan.json never produced, orchestrator phase failure)
# prevents the pipeline from finishing cleanly.  Per-design prediction
# failures are tolerated and reported by the collect / aggregate
# phases downstream — they show up as `status != ok` rows, NOT as
# non-zero exits from this script.
#
# Usage — see the argument-parsing block below.  All paths are
# absolutised before being passed to the container to avoid
# host/container CWD mismatches.
# ─────────────────────────────────────────────────────────────────────

set -uo pipefail

# ─── Argument defaults ──────────────────────────────────────────────
SEQ_NAME=""
GROUND_TRUTH=""
RECEPTOR_CHAIN=""
EFFECTOR_CHAIN=""
RECEPTOR_FASTA=""
EFFECTOR_FASTA=""
TRUE_INTERFACE_FILE=""
DESIGN_REGION_FILE=""
WORKDIR=""
BIN_DIR=""
BOLTZ_CONTAINER=""
# PLAN_EXTRA_ARGS: a single string of extra flags forwarded verbatim
# to `plan`.  Word-split at consumption time with the standard shell
# word-splitting rules.  Keeps the NextFlow side simple — users pass
# one string, we don't try to enforce any structure.
PLAN_EXTRA_ARGS=""
PP_RMSD_THRESHOLD="5.0"
PP_METRIC_COLUMN="steered_ra_eff_vs_truth"
PP_CONTACT_CUTOFF="5.0"
PP_POPULATE_ALL=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seq-name)                      SEQ_NAME="$2";               shift 2 ;;
        --ground-truth)                  GROUND_TRUTH="$2";           shift 2 ;;
        --receptor-chain)                RECEPTOR_CHAIN="$2";         shift 2 ;;
        --effector-chain)                EFFECTOR_CHAIN="$2";         shift 2 ;;
        --receptor-fasta)                RECEPTOR_FASTA="$2";         shift 2 ;;
        --effector-fasta)                EFFECTOR_FASTA="$2";         shift 2 ;;
        --true-interface-indices-file)   TRUE_INTERFACE_FILE="$2";    shift 2 ;;
        --design-region-indices-file)    DESIGN_REGION_FILE="$2";     shift 2 ;;
        --workdir)                       WORKDIR="$2";                shift 2 ;;
        --bin-dir)                       BIN_DIR="$2";                shift 2 ;;
        --boltz-container)               BOLTZ_CONTAINER="$2";        shift 2 ;;
        --plan-extra-args)               PLAN_EXTRA_ARGS="$2";        shift 2 ;;
        --postprocess-rmsd-threshold)    PP_RMSD_THRESHOLD="$2";      shift 2 ;;
        --postprocess-metric-column)     PP_METRIC_COLUMN="$2";       shift 2 ;;
        --postprocess-contact-cutoff)    PP_CONTACT_CUTOFF="$2";      shift 2 ;;
        --postprocess-populate-all)      PP_POPULATE_ALL=1;           shift 1 ;;
        --no-postprocess-populate-all)   PP_POPULATE_ALL=0;           shift 1 ;;
        *)
            echo "negative_steering_run_one.sh: unknown arg: $1" >&2
            exit 2 ;;
    esac
done

# ─── Required-arg check ─────────────────────────────────────────────
missing=()
[[ -z "$SEQ_NAME"            ]] && missing+=(--seq-name)
[[ -z "$GROUND_TRUTH"        ]] && missing+=(--ground-truth)
[[ -z "$RECEPTOR_CHAIN"      ]] && missing+=(--receptor-chain)
[[ -z "$EFFECTOR_CHAIN"      ]] && missing+=(--effector-chain)
[[ -z "$RECEPTOR_FASTA"      ]] && missing+=(--receptor-fasta)
[[ -z "$EFFECTOR_FASTA"      ]] && missing+=(--effector-fasta)
[[ -z "$TRUE_INTERFACE_FILE" ]] && missing+=(--true-interface-indices-file)
[[ -z "$DESIGN_REGION_FILE"  ]] && missing+=(--design-region-indices-file)
[[ -z "$WORKDIR"             ]] && missing+=(--workdir)
[[ -z "$BIN_DIR"             ]] && missing+=(--bin-dir)
[[ -z "$BOLTZ_CONTAINER"     ]] && missing+=(--boltz-container)
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "ERROR: missing required args: ${missing[*]}" >&2
    exit 2
fi

for f in "$GROUND_TRUTH" "$RECEPTOR_FASTA" "$EFFECTOR_FASTA" \
         "$TRUE_INTERFACE_FILE" "$DESIGN_REGION_FILE"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: input file not found: $f" >&2
        exit 2
    fi
done

PY_NS="${BIN_DIR}/boltz2_negative_steering.py"
PY_COLD="${BIN_DIR}/negsteer_coldstart.py"
PY_REV="${BIN_DIR}/reversion.py"
PY_AGG="${BIN_DIR}/negsteer_aggregate.py"
PY_EXTRACT="${BIN_DIR}/extract_passing.py"
for py in "$PY_NS" "$PY_COLD" "$PY_REV" "$PY_AGG" "$PY_EXTRACT"; do
    if [[ ! -f "$py" ]]; then
        echo "ERROR: required Python script not found: $py" >&2
        exit 2
    fi
done
if [[ ! -f "$BOLTZ_CONTAINER" ]]; then
    echo "ERROR: Boltz container not found: $BOLTZ_CONTAINER" >&2
    exit 2
fi

# ─── Absolutise paths ───────────────────────────────────────────────
GT_ABS="$(readlink -f "$GROUND_TRUTH")"
REC_FASTA_ABS="$(readlink -f "$RECEPTOR_FASTA")"
EFF_FASTA_ABS="$(readlink -f "$EFFECTOR_FASTA")"
TRUE_IFACE_ABS="$(readlink -f "$TRUE_INTERFACE_FILE")"
DESIGN_REGION_ABS="$(readlink -f "$DESIGN_REGION_FILE")"
BIN_DIR_ABS="$(readlink -f "$BIN_DIR")"

mkdir -p "$WORKDIR"
WORKDIR_ABS="$(cd "$WORKDIR" && pwd)"
CYCLE0_ABS="$WORKDIR_ABS/cycle_0"
mkdir -p "$CYCLE0_ABS" "$WORKDIR_ABS/logs"

# ─── Build bind-flag array for singularity ──────────────────────────
# Assemble a dedup'd list of parent dirs we need inside the container.
# We add --bind entries as pairs of array items so the final command
# can be built as an argv array (no eval, no quoting ambiguity).
declare -a BIND_DIRS=()
add_bind() {
    local d="$1" existing
    for existing in "${BIND_DIRS[@]:-}"; do
        [[ "$d" == "$existing" ]] && return
    done
    BIND_DIRS+=("$d")
}
add_bind "$BIN_DIR_ABS"
add_bind "$WORKDIR_ABS"
add_bind "$(dirname "$GT_ABS")"
add_bind "$(dirname "$REC_FASTA_ABS")"
add_bind "$(dirname "$EFF_FASTA_ABS")"
add_bind "$(dirname "$TRUE_IFACE_ABS")"
add_bind "$(dirname "$DESIGN_REGION_ABS")"

declare -a BIND_FLAGS=()
for d in "${BIND_DIRS[@]}"; do
    BIND_FLAGS+=("--bind" "${d}:${d}")
done

# Helpers that wrap `singularity exec [--nv] --bind ... container`
# so the call sites below look like a normal argv invocation.
sing_gpu() {
    singularity exec --nv "${BIND_FLAGS[@]}" "$BOLTZ_CONTAINER" "$@"
}
sing_cpu() {
    singularity exec       "${BIND_FLAGS[@]}" "$BOLTZ_CONTAINER" "$@"
}

# ─── Banner ─────────────────────────────────────────────────────────
echo "──────────────────────────────────────────────────────────────"
echo "negative_steering_run_one — $SEQ_NAME"
echo "──────────────────────────────────────────────────────────────"
echo "  ground truth:   $GT_ABS"
echo "  receptor chain: $RECEPTOR_CHAIN"
echo "  effector chain: $EFFECTOR_CHAIN"
echo "  receptor fasta: $REC_FASTA_ABS"
echo "  effector fasta: $EFF_FASTA_ABS"
echo "  true interface: $TRUE_IFACE_ABS"
echo "  design region:  $DESIGN_REGION_ABS"
echo "  workdir:        $WORKDIR_ABS"
echo "  cycle_0:        $CYCLE0_ABS"
echo "  bin dir:        $BIN_DIR_ABS"
echo "  container:      $BOLTZ_CONTAINER"
echo "  plan extra:     $PLAN_EXTRA_ARGS"
echo "──────────────────────────────────────────────────────────────"

# ─── Start the cumulative wall-clock timer ─────────────────────────
# Stamped into run_one_runtime_sec.txt at the end of this script so
# downstream aggregation (cmd_aggregate_per_sequence) can pick it up
# as `run_one_runtime_sec` in aggregated_results.csv.  Whole-script
# wall-clock covers every phase (plan + predict-one × N + collect +
# reversion + harvest + finalize + postprocess), which is what you
# want for sequence-level throughput monitoring.  SECONDS is a bash
# built-in that increments once per wall-clock second from this
# assignment onwards.
SECONDS=0

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

# ─── Word-split PLAN_EXTRA_ARGS into an array ───────────────────────
# Relies on IFS (default: whitespace) — callers must quote values
# internal to the string if they contain spaces, but for this pipeline
# the plan flags are all simple tokens so this is safe.
declare -a PLAN_EXTRA
if [[ -n "$PLAN_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    PLAN_EXTRA=( $PLAN_EXTRA_ARGS )
else
    PLAN_EXTRA=()
fi

# ═════════════════════════════════════════════════════════════════════
# Stage 1: plan (GPU)
# ═════════════════════════════════════════════════════════════════════
echo ""
echo "─── STAGE 1: plan ────────────────────────────────────────────"
sing_gpu python "$PY_NS" plan \
    --ground-truth   "$GT_ABS" \
    --receptor       "$RECEPTOR_CHAIN" \
    --effector       "$EFFECTOR_CHAIN" \
    --workdir        "$CYCLE0_ABS" \
    --boltz-container "$BOLTZ_CONTAINER" \
    --receptor-fasta "$REC_FASTA_ABS" \
    --effector-fasta "$EFF_FASTA_ABS" \
    --true-interface-indices-file "$TRUE_IFACE_ABS" \
    --design-region-indices-file  "$DESIGN_REGION_ABS" \
    "${PLAN_EXTRA[@]}"
PLAN_RC=$?
if [[ $PLAN_RC -ne 0 ]]; then
    # skip_steering is a valid soft exit: plan.json and steered_results.csv
    # are already written with the cold-start row. Let execution fall through
    # to the existing skip_steering handling below. Any other non-zero exit
    # is still a hard failure.
    if [[ -f "$CYCLE0_ABS/plan.json" ]] && \
       python3 -c "import json,sys; p=json.load(open('$CYCLE0_ABS/plan.json')); sys.exit(0 if p.get('skip_steering') else 1)"; then
        echo "  plan stage exited $PLAN_RC with skip_steering=true — continuing."
    else
        fail "plan stage exited $PLAN_RC"
    fi
fi
if [[ ! -f "$CYCLE0_ABS/plan.json" ]]; then
    fail "plan stage did not produce $CYCLE0_ABS/plan.json"
fi

# Read skip_steering flag and n_designs from plan.json (stdlib Python
# on host — no need for the container for a JSON peek).
SKIP_STEERING=$(python3 -c "
import json
p = json.load(open('$CYCLE0_ABS/plan.json'))
print('1' if p.get('skip_steering') else '0')
")
N_DESIGNS=$(python3 -c "
import json
p = json.load(open('$CYCLE0_ABS/plan.json'))
print(len(p.get('designs', [])))
")
echo "  plan.json: skip_steering=$SKIP_STEERING, n_designs=$N_DESIGNS"

# ═════════════════════════════════════════════════════════════════════
# Stage 2: predict-one, looped over 0..N_DESIGNS-1 (GPU)
# ═════════════════════════════════════════════════════════════════════
# Each invocation is independent and re-reads plan.json, so the loop
# is identical in behaviour to the SLURM array.  predict-one returns
# 1 for per-design failures (which we tolerate) and 2 for invalid
# index (which should never happen given we loop to N_DESIGNS).  Only
# exit code 2 or higher is fatal here.
echo ""
echo "─── STAGE 2: predict-one × $N_DESIGNS ────────────────────────"
if [[ "$SKIP_STEERING" == "1" ]]; then
    echo "  skip_steering=true; no predictions to run."
elif [[ "$N_DESIGNS" -eq 0 ]]; then
    echo "  n_designs=0; nothing to predict."
else
    N_PRED_OK=0
    N_PRED_FAIL=0
    for (( i=0; i<N_DESIGNS; i++ )); do
        echo ""
        echo "  ── predict-one $i / $((N_DESIGNS - 1)) ──"
        # errexit is deliberately NOT enabled for this script (`set -uo pipefail`
        # above), so a non-zero rc here already does not abort. Capturing rc is
        # all that is needed. This block used to be wrapped in `set +e` ... `set
        # -e`, and that trailing `set -e` switched errexit ON for everything
        # after the first loop iteration, which is not what the script runs
        # under anywhere else. The visible symptom was the `ls *.csv` at the
        # very end aborting an otherwise successful run with exit 1 whenever it
        # matched nothing.
        sing_gpu python "$PY_NS" predict-one \
            --workdir "$CYCLE0_ABS" \
            --index   "$i"
        rc=$?
        if [[ $rc -eq 0 ]]; then
            N_PRED_OK=$((N_PRED_OK + 1))
        elif [[ $rc -eq 1 ]]; then
            N_PRED_FAIL=$((N_PRED_FAIL + 1))
            echo "  (design $i: failed, continuing)"
        else
            fail "predict-one index=$i exited $rc (fatal)"
        fi
    done
    echo ""
    echo "  predict-one summary: $N_PRED_OK ok, $N_PRED_FAIL failed"
fi

# ═════════════════════════════════════════════════════════════════════
# Stage 3: collect (CPU, in-container)
# ═════════════════════════════════════════════════════════════════════
echo ""
echo "─── STAGE 3: collect ─────────────────────────────────────────"
sing_cpu python "$PY_NS" collect --workdir "$CYCLE0_ABS" \
    || fail "collect stage failed"

# ═════════════════════════════════════════════════════════════════════
# Stage 4: reversion preparation (CPU, in-container)
# ═════════════════════════════════════════════════════════════════════
echo ""
echo "─── STAGE 4: reversion prep ──────────────────────────────────"
sing_cpu python "$PY_COLD" kickoff-distances \
    --experiment-root "$WORKDIR_ABS" \
    || fail "kickoff-distances failed"

sing_cpu python "$PY_COLD" kickoff-prefilter \
    --experiment-root "$WORKDIR_ABS" \
    || fail "kickoff-prefilter failed"

sing_cpu python "$PY_REV" build-contaminated \
    --workdir "$CYCLE0_ABS" \
    || fail "build-contaminated failed"

sing_cpu python "$PY_REV" plan-reversions \
    --workdir "$CYCLE0_ABS" \
    || fail "plan-reversions failed"

if [[ -f "$CYCLE0_ABS/reversion_plan.json" ]]; then
    N_REVERSIONS=$(python3 -c "
import json
m = json.load(open('$CYCLE0_ABS/reversion_plan.json'))
print(len(m.get('entries', [])))
")
else
    N_REVERSIONS=0
fi
echo "  staged $N_REVERSIONS reversion(s)"

# ═════════════════════════════════════════════════════════════════════
# Stage 5: predict-reversion-one, looped (GPU)
# ═════════════════════════════════════════════════════════════════════
# predict-reversion-one returns 0 for out-of-range indices AND for
# successful predictions; returns 1 only for per-design failures.
echo ""
echo "─── STAGE 5: predict-reversion-one × $N_REVERSIONS ───────────"
if [[ "$N_REVERSIONS" -eq 0 ]]; then
    echo "  no reversions staged; skipping."
else
    N_REV_OK=0
    N_REV_FAIL=0
    for (( i=0; i<N_REVERSIONS; i++ )); do
        echo ""
        echo "  ── predict-reversion-one $i / $((N_REVERSIONS - 1)) ──"
        # As in stage 2: no errexit to suspend, and no errexit to restore.
        sing_gpu python "$PY_REV" predict-reversion-one \
            --workdir "$CYCLE0_ABS" \
            --index   "$i"
        rc=$?
        if [[ $rc -eq 0 ]]; then
            N_REV_OK=$((N_REV_OK + 1))
        elif [[ $rc -eq 1 ]]; then
            N_REV_FAIL=$((N_REV_FAIL + 1))
            echo "  (reversion $i: failed, continuing)"
        else
            fail "predict-reversion-one index=$i exited $rc (fatal)"
        fi
    done
    echo ""
    echo "  predict-reversion-one summary: $N_REV_OK ok, $N_REV_FAIL failed"
fi

# ═════════════════════════════════════════════════════════════════════
# Stage 6: harvest-reversions + kickoff-finalize (CPU, in-container)
# ═════════════════════════════════════════════════════════════════════
# kickoff-finalize with --max-cycles 0 finalises cycle 0 without
# spawning further chains (the iterate script short-circuits on
# max_cycles < 1 — see cmd_kickoff_finalize).
echo ""
echo "─── STAGE 6: harvest + finalize ──────────────────────────────"
sing_cpu python "$PY_REV" harvest-reversions \
    --workdir "$CYCLE0_ABS" \
    || fail "harvest-reversions failed"

sing_cpu python "$PY_COLD" kickoff-finalize \
    --experiment-root "$WORKDIR_ABS" \
    --max-cycles      0 \
    --max-passing     5 \
    --novelty-cutoff  10.0 \
    || fail "kickoff-finalize failed"

# ═════════════════════════════════════════════════════════════════════
# Stage 7: post-processing (CPU, in-container)
# ═════════════════════════════════════════════════════════════════════
# Mirrors submit_postprocess.sh's four phases.  Produces the per-
# sequence deliverables:
#   all_results_multicycle.csv
#   all_results_multicycle_with_metrics.csv
#   raw_per_seed_results.csv
#   aggregated_results.csv
#   passing_summary.csv
echo ""
echo "─── STAGE 7: post-processing ─────────────────────────────────"

declare -a POPULATE_FLAG=()
if [[ "$PP_POPULATE_ALL" -eq 1 ]]; then
    POPULATE_FLAG=(--populate-all)
fi

sing_cpu python "$PY_AGG" aggregate \
    --experiment-root "$WORKDIR_ABS" \
    || fail "aggregate failed"

sing_cpu python "$PY_AGG" compute-final-metrics \
    --experiment-root "$WORKDIR_ABS" \
    --rmsd-threshold  "$PP_RMSD_THRESHOLD" \
    --metric-column   "$PP_METRIC_COLUMN" \
    --contact-cutoff  "$PP_CONTACT_CUTOFF" \
    "${POPULATE_FLAG[@]}" \
    || fail "compute-final-metrics failed"

sing_cpu python "$PY_AGG" aggregate-per-sequence \
    --experiment-root "$WORKDIR_ABS" \
    || fail "aggregate-per-sequence failed"

# Stamp cumulative wall-clock so extract_passing can inject it into
# passing_summary.csv (and thence into cross_sequence_summary.csv).
# This excludes the extract_passing step itself, which is CPU-only
# and takes <1s — the signal we care about is the Boltz + metrics
# work, which is all done by this point.
_RUN_ONE_ELAPSED="$SECONDS"
printf '%s\n' "$_RUN_ONE_ELAPSED" > "$WORKDIR_ABS/run_one_runtime_sec.txt" \
    || echo "WARN: could not write run_one_runtime_sec.txt" >&2

sing_cpu python "$PY_EXTRACT" \
    --input "$WORKDIR_ABS/aggregated_results.csv" \
    || fail "extract_passing failed"

# ═════════════════════════════════════════════════════════════════════
# Done
# ═════════════════════════════════════════════════════════════════════
_MIN=$(( _RUN_ONE_ELAPSED / 60 ))
_SEC=$(( _RUN_ONE_ELAPSED % 60 ))

echo ""
echo "──────────────────────────────────────────────────────────────"
echo "negative_steering_run_one DONE — $SEQ_NAME"
echo "  workdir: $WORKDIR_ABS"
echo "  wall-clock: ${_MIN}m ${_SEC}s (${_RUN_ONE_ELAPSED}s)"
# A cosmetic listing. `|| true` so that matching nothing can never decide the
# exit code of the run, whatever errexit is doing.
ls -1 "$WORKDIR_ABS"/*.csv 2>/dev/null | sed 's/^/    /' || true
echo "──────────────────────────────────────────────────────────────"
exit 0
