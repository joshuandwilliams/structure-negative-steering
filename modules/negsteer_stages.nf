// Negative-steering stages, one process per stage of a single-cycle run.
//
// The engine writes into a persistent tree that later stages walk by absolute
// path from plan.json, so each process takes the run directory as a val and
// writes in place. Each process also copies the artefact it produced into its
// task directory and declares it as a path output. That is what Nextflow hashes
// for -resume; without it a resumed run repeats work it has already done.
//
// Every process writes its elapsed seconds to its OWN file under
// <run>/.stage_times.d/, and POSTPROCESS concatenates that directory into
// <run>/.stage_times and sums it into run_one_runtime_sec.txt.
//
// One file per stage rather than one shared file, because the two fan-outs
// append concurrently from different nodes over a shared filesystem. The first
// real cluster run lost one of nine predict_one lines that way, which silently
// understates the per-run cost the compute-cost analysis reads. Concatenating
// at the end is also idempotent, so a -resume that re-runs POSTPROCESS cannot
// double-count.

nextflow.enable.dsl = 2


// The one stage that cannot assume a container, because the container path is
// what it is reading. It resolves that path as plain text first, then runs the
// driver inside it like every other stage does. Running the driver on the host
// instead is what broke the first cluster run: the driver needs PyYAML, and the
// airgapped host has stdlib only.
//
// The host-python branch is not dead code. CI has PyYAML installed and no
// singularity, and READ_CONFIG has no stub, so that is the path it takes.
process NEGSTEER_READ_CONFIG {
    tag "${config.simpleName}"
    label 'tiny'

    input:
    path config

    output:
    stdout emit: json

    script:
    """
    CFG="\$(readlink -f ${config})"
    CONTAINER="\$(bash ${params.repo_root}/scripts/resolve_boltz_container.sh \\
        "\$CFG" "${params.boltz_container ?: ''}")"

    if [ -n "\$CONTAINER" ] && [ -f "\$CONTAINER" ]; then
        singularity exec --bind ${params.repo_root} --bind "\$(dirname "\$CFG")" \\
            "\$CONTAINER" \\
            python ${params.repo_root}/scripts/negative_steering.py "\$CFG" --emit-json
    elif [ "${params.allow_host_python}" = "true" ]; then
        # Opted into explicitly, and only ever by CI. A GitHub runner has
        # pip-installed PyYAML and no image to run anything inside. Enabling
        # this on the cluster would mean reading the config with whatever
        # Python the host happens to have, which is the thing the containers
        # exist to prevent.
        python3 ${params.repo_root}/scripts/negative_steering.py "\$CFG" --emit-json
    else
        echo "ERROR: no Boltz container for \$CFG." >&2
        echo "Set boltz_container: in the config, or pass --boltz_container IMG." >&2
        echo "Running the driver on the host needs --allow_host_python, which is for CI." >&2
        exit 1
    fi
    """
}


process NEGSTEER_PREPARE {
    tag "${name}"
    label 'cpu'

    input:
    tuple val(name), val(config), val(run_dir), val(container)

    output:
    tuple val(name), val(config), val(run_dir), val(container), emit: ready
    path "receptor.fasta"

    script:
    """
    mkdir -p ${run_dir}
    rm -rf ${run_dir}/.stage_times.d
    mkdir -p ${run_dir}/.stage_times.d

    _t0=\$SECONDS
    singularity exec --bind ${params.repo_root} ${container} \\
        python ${params.repo_root}/scripts/negative_steering.py \\
            ${config} --prepare-only
    printf 'prepare %s\\n' "\$((SECONDS - _t0))" > ${run_dir}/.stage_times.d/prepare

    cp ${run_dir}/../inputs/receptor.fasta .
    """

    stub:
    """
    mkdir -p ${run_dir}/.stage_times.d
    printf '>r\\nACDEF\\n' > receptor.fasta
    """
}


// plan.json is the authority for how many designs were staged, not the config:
// the plan can stage fewer than asked, or none when the cold start passes.
process NEGSTEER_PLAN {
    tag "${name}"
    label 'gpu'

    input:
    tuple val(name), val(run_dir), val(container), val(plan_args)

    output:
    tuple val(name), val(run_dir), val(container), stdout, emit: planned
    path "plan.json"

    script:
    """
    _t0=\$SECONDS
    singularity exec --nv --bind ${params.repo_root} ${container} \\
        python ${params.repo_root}/bin/boltz2_negative_steering.py plan \\
            --workdir ${run_dir}/cycle_0 \\
            --boltz-container ${container} \\
            ${plan_args} 1>&2
    printf 'plan %s\\n' "\$((SECONDS - _t0))" > ${run_dir}/.stage_times.d/plan

    cp ${run_dir}/cycle_0/plan.json .
    singularity exec --bind ${params.repo_root} --bind "\$PWD" ${container} \\
        python -c "
import json, sys
plan = json.load(open('plan.json'))
sys.stdout.write('0' if plan.get('skip_steering') else str(len(plan.get('designs', []))))
"
    """

    stub:
    """
    mkdir -p ${run_dir}/cycle_0
    echo '{"designs": [{}, {}]}' > plan.json
    printf '%s' "\${STUB_N_DESIGNS:-2}"
    """
}


// A failed design must not kill the others, which is what the shell
// orchestrator did. The count is asserted downstream rather than assumed.
process NEGSTEER_PREDICT_ONE {
    tag "${name}/design_${index}"
    label 'gpu'

    input:
    tuple val(name), val(run_dir), val(container), val(index)

    output:
    tuple val(name), val(run_dir), val(container), val(index), emit: predicted

    script:
    """
    _t0=\$SECONDS
    singularity exec --nv --bind ${params.repo_root} ${container} \\
        python ${params.repo_root}/bin/boltz2_negative_steering.py predict-one \\
            --workdir ${run_dir}/cycle_0 --index ${index}
    printf 'predict_one_${index} %s\\n' "\$((SECONDS - _t0))" > ${run_dir}/.stage_times.d/predict_one_${index}
    """

    stub:
    """
    printf 'predict_one_${index} 0\\n' > ${run_dir}/.stage_times.d/predict_one_${index}
    """
}


process NEGSTEER_COLLECT {
    tag "${name}"
    label 'cpu'

    input:
    tuple val(name), val(run_dir), val(container)

    output:
    tuple val(name), val(run_dir), val(container), emit: collected
    path "steered_results.csv"

    script:
    """
    _t0=\$SECONDS
    singularity exec --bind ${params.repo_root} ${container} \\
        python ${params.repo_root}/bin/boltz2_negative_steering.py collect \\
            --workdir ${run_dir}/cycle_0
    printf 'collect %s\\n' "\$((SECONDS - _t0))" > ${run_dir}/.stage_times.d/collect

    cp ${run_dir}/cycle_0/steered_results.csv .
    """

    stub:
    """
    echo "design,receptor_aligned_effector_rmsd" > steered_results.csv
    """
}


// One process, not four: each step reads what the previous wrote in the same
// directory, so splitting them costs four submissions and buys nothing.
process NEGSTEER_REVERSION_PREP {
    tag "${name}"
    label 'cpu'

    input:
    tuple val(name), val(run_dir), val(container)

    output:
    tuple val(name), val(run_dir), val(container), stdout, emit: staged
    path "reversion_plan.json"

    script:
    """
    COLD=${params.repo_root}/bin/negsteer_coldstart.py
    REV=${params.repo_root}/bin/reversion.py
    AGG=${params.repo_root}/bin/negsteer_aggregate.py
    SING="singularity exec --bind ${params.repo_root} --bind \$PWD ${container} python"

    _t0=\$SECONDS
    \$SING \$COLD kickoff-distances  --experiment-root ${run_dir} 1>&2
    \$SING \$COLD kickoff-prefilter  --experiment-root ${run_dir} 1>&2
    \$SING \$REV build-contaminated --workdir ${run_dir}/cycle_0 1>&2
    \$SING \$REV plan-reversions    --workdir ${run_dir}/cycle_0 1>&2
    printf 'reversion_prep %s\\n' "\$((SECONDS - _t0))" > ${run_dir}/.stage_times.d/reversion_prep

    if [ -f ${run_dir}/cycle_0/reversion_plan.json ]; then
        cp ${run_dir}/cycle_0/reversion_plan.json .
    else
        echo '{"entries": []}' > reversion_plan.json
    fi
    \$SING -c "
import json, sys
sys.stdout.write(str(len(json.load(open('reversion_plan.json')).get('entries', []))))
"
    """

    stub:
    """
    echo '{"entries": [{}]}' > reversion_plan.json
    printf '%s' "\${STUB_N_REVERSIONS:-1}"
    """
}


process NEGSTEER_PREDICT_REVERSION {
    tag "${name}/reversion_${index}"
    label 'gpu'

    input:
    tuple val(name), val(run_dir), val(container), val(index)

    output:
    tuple val(name), val(run_dir), val(container), val(index), emit: predicted

    script:
    """
    _t0=\$SECONDS
    singularity exec --nv --bind ${params.repo_root} ${container} \\
        python ${params.repo_root}/bin/reversion.py \\
            predict-reversion-one --workdir ${run_dir}/cycle_0 --index ${index}
    printf 'predict_reversion_${index} %s\\n' "\$((SECONDS - _t0))" > ${run_dir}/.stage_times.d/predict_reversion_${index}
    """

    stub:
    """
    printf 'predict_reversion_${index} 0\\n' > ${run_dir}/.stage_times.d/predict_reversion_${index}
    """
}


// --max-cycles 0 is what stops the engine submitting a follow-up cycle. It was
// always this repo's setting, and is now structural: there is no submit path.
process NEGSTEER_HARVEST {
    tag "${name}"
    label 'cpu'

    input:
    tuple val(name), val(run_dir), val(container)

    output:
    tuple val(name), val(run_dir), val(container), emit: harvested
    path "reversion_results.json"

    script:
    """
    COLD=${params.repo_root}/bin/negsteer_coldstart.py
    REV=${params.repo_root}/bin/reversion.py
    AGG=${params.repo_root}/bin/negsteer_aggregate.py
    SING="singularity exec --bind ${params.repo_root} ${container} python"

    _t0=\$SECONDS
    \$SING \$REV harvest-reversions --workdir ${run_dir}/cycle_0
    \$SING \$COLD kickoff-finalize \\
        --experiment-root ${run_dir} \\
        --max-cycles 0 \\
        --max-passing ${params.max_passing} \\
        --novelty-cutoff ${params.novelty_cutoff}
    printf 'harvest %s\\n' "\$((SECONDS - _t0))" > ${run_dir}/.stage_times.d/harvest

    cp ${run_dir}/cycle_0/reversion_results.json . 2>/dev/null \\
        || echo '[]' > reversion_results.json
    """

    stub:
    """
    echo '[]' > reversion_results.json
    """
}


// run_one_runtime_sec.txt is the per-run cost the compute-cost analysis reads.
// Under the old design it was one job's elapsed time; the stages are separate
// tasks now, so it is the sum of their recorded seconds. That is the same
// quantity the serial job measured, and stays comparable with the benchmark.
process NEGSTEER_POSTPROCESS {
    tag "${name}"
    label 'cpu'

    input:
    tuple val(name), val(run_dir), val(container), val(rmsd_threshold), val(metric_column), val(contact_cutoff)

    output:
    tuple val(name), val(run_dir), val(container), emit: postprocessed
    path "passing_summary.csv"
    path "run_one_runtime_sec.txt"

    script:
    def populate = params.postprocess_populate_all ? '--populate-all' : ''
    """
    COLD=${params.repo_root}/bin/negsteer_coldstart.py
    REV=${params.repo_root}/bin/reversion.py
    AGG=${params.repo_root}/bin/negsteer_aggregate.py
    SING="singularity exec --bind ${params.repo_root} ${container} python"

    _t0=\$SECONDS
    \$SING \$AGG aggregate --experiment-root ${run_dir}

    \$SING \$AGG compute-final-metrics \\
        --experiment-root ${run_dir} \\
        --rmsd-threshold  ${rmsd_threshold} \\
        --metric-column   ${metric_column} \\
        --contact-cutoff  ${contact_cutoff} \\
        ${populate}

    \$SING \$AGG aggregate-per-sequence --experiment-root ${run_dir}

    \$SING ${params.repo_root}/bin/extract_passing.py \\
        --input ${run_dir}/aggregated_results.csv
    printf 'postprocess %s\\n' "\$((SECONDS - _t0))" > ${run_dir}/.stage_times.d/postprocess

    # Assemble the record from the per-stage files. Written fresh rather than
    # appended, so re-running this stage under -resume cannot double-count.
    cat ${run_dir}/.stage_times.d/* > ${run_dir}/.stage_times

    \$SING ${params.repo_root}/bin/sum_stage_times.py \\
        --stage-times ${run_dir}/.stage_times \\
        --output ${run_dir}/run_one_runtime_sec.txt

    cp ${run_dir}/run_one_runtime_sec.txt .
    cp ${run_dir}/passing_summary.csv .
    """

    stub:
    """
    printf 'postprocess 1\\n' > ${run_dir}/.stage_times.d/postprocess
    cat ${run_dir}/.stage_times.d/* > ${run_dir}/.stage_times
    # Shell, not sum_stage_times.py. A stub run has no container, and the host
    # is not allowed to run the engine's Python. The real sum is asserted by
    # tests/characterization/, not by walking the DAG.
    awk '{ total += \$2 } END { print total + 0 }' ${run_dir}/.stage_times \\
        > ${run_dir}/run_one_runtime_sec.txt
    cp ${run_dir}/run_one_runtime_sec.txt .
    echo "mpnn_sequence" > passing_summary.csv
    """
}


// Non-fatal: the engine results are already written, and a tiering hiccup must
// not fail a run that produced them.
process NEGSTEER_CROSS_SUMMARY {
    tag "${name}"
    label 'cpu'
    errorStrategy 'ignore'

    input:
    tuple val(name), val(run_dir), val(container)

    output:
    tuple val(name), val(run_dir), emit: summarised
    path "cross_sequence_summary.csv"

    script:
    """
    singularity exec --bind ${params.repo_root} ${container} \\
        python ${params.repo_root}/bin/cross_summary_cli.py \\
            --passing-summary "${name}=${run_dir}/passing_summary.csv" \\
            --output ${run_dir}/cross_sequence_summary.csv

    cp ${run_dir}/cross_sequence_summary.csv .
    """

    stub:
    """
    echo "mpnn_sequence,cross_tier" > cross_sequence_summary.csv
    """
}
