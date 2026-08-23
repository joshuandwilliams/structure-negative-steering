#!/usr/bin/env nextflow
// Negative steering over one or more benchmark targets.
//
//   nextflow run main.nf --targets "experiments/benchmarking/**/config.yml"
//   nextflow run main.nf --targets tests/single_run_test/config.yml
//
// The two fan-outs, one Boltz call per steered design and one per reverted
// design, were bash loops inside a single SLURM job. They are channels now, so
// each prediction is a task with its own resources, retry and failure.
// The sbatch self-resubmission was removed with the multi-cycle harness: this
// repo always ran with --max-cycles 0, so it never fired here.
//
// Config is read by scripts/negative_steering.py --emit-json rather than parsed
// here, so key names and defaults live in one place.

nextflow.enable.dsl = 2

include {
    NEGSTEER_READ_CONFIG;
    NEGSTEER_PREPARE;
    NEGSTEER_PLAN;
    NEGSTEER_PREDICT_ONE;
    NEGSTEER_COLLECT;
    NEGSTEER_REVERSION_PREP;
    NEGSTEER_PREDICT_REVERSION;
    NEGSTEER_HARVEST;
    NEGSTEER_POSTPROCESS;
    NEGSTEER_CROSS_SUMMARY;
} from './modules/negsteer_stages.nf'


// ---- Parameters ------------------------------------------------------------
// Declared in nextflow.config's `params {}` block, NOT here. A `params.x = ...`
// in this script's body is not visible to modules/negsteer_stages.nf on the
// Nextflow the cluster runs, and renders as the string "null" inside a task
// script. See the comment above that block.


workflow {

    if( !params.targets )
        error "Pass --targets, for example --targets 'experiments/benchmarking/**/config.yml'"

    // One JSON blob per target, produced by the Python driver.
    cfg = NEGSTEER_READ_CONFIG( Channel.fromPath(params.targets, checkIfExists: true) )
            .json
            .map { txt -> new groovy.json.JsonSlurper().parseText(txt.trim()) }

    // ---- prepare, then plan ------------------------------------------------
    prepared = NEGSTEER_PREPARE(
        cfg.map { c -> tuple(c.name, c.config, c.run_dir, c.container) }
    )

    planned = NEGSTEER_PLAN(
        prepared.ready
            .join( cfg.map { c -> tuple(c.name, c.plan_args) } )
            .map { name, config, run_dir, container, plan_args ->
                   tuple(name, run_dir, container, plan_args) }
    )

    // ---- fan out one prediction per staged design --------------------------
    // The width comes from plan.json rather than from the config, because the
    // plan legitimately stages fewer designs than asked, or none at all when
    // the cold start already passes.
    design_tasks = planned.planned
        .flatMap { name, run_dir, container, n ->
            def count = (n.toString().trim() ?: '0') as Integer
            count > 0 ? (0..<count).collect { i -> tuple(name, run_dir, container, i) } : []
        }

    predicted = NEGSTEER_PREDICT_ONE( design_tasks )

    // groupTuple is the barrier. collect waits for every design of a target,
    // including the ones that failed and were ignored.
    all_designs_done = predicted.predicted
        .map { name, run_dir, container, i -> tuple(name, run_dir, container) }
        .groupTuple()
        .map { name, run_dirs, containers -> tuple(name, run_dirs[0], containers[0]) }

    // A target whose cold start already passed stages no designs, so it never
    // reaches collect through the fan-out. Mix that case back in.
    no_designs = planned.planned
        .filter { name, run_dir, container, n -> ((n.toString().trim() ?: '0') as Integer) == 0 }
        .map { name, run_dir, container, n -> tuple(name, run_dir, container) }

    collected = NEGSTEER_COLLECT( all_designs_done.mix(no_designs) )

    // ---- reversion: prep, fan out, harvest ---------------------------------
    staged = NEGSTEER_REVERSION_PREP( collected.collected )

    reversion_tasks = staged.staged
        .flatMap { name, run_dir, container, n ->
            def count = (n.toString().trim() ?: '0') as Integer
            count > 0 ? (0..<count).collect { i -> tuple(name, run_dir, container, i) } : []
        }

    reverted = NEGSTEER_PREDICT_REVERSION( reversion_tasks )

    all_reversions_done = reverted.predicted
        .map { name, run_dir, container, i -> tuple(name, run_dir, container) }
        .groupTuple()
        .map { name, run_dirs, containers -> tuple(name, run_dirs[0], containers[0]) }

    // Same shape as above: no contaminated design means no reversion to run,
    // which is a normal outcome rather than a stall.
    no_reversions = staged.staged
        .filter { name, run_dir, container, n -> ((n.toString().trim() ?: '0') as Integer) == 0 }
        .map { name, run_dir, container, n -> tuple(name, run_dir, container) }

    harvested = NEGSTEER_HARVEST( all_reversions_done.mix(no_reversions) )

    postprocessed = NEGSTEER_POSTPROCESS(
        harvested.harvested
            .join( cfg.map { c -> tuple(c.name,
                                        c.postprocess_rmsd_threshold,
                                        c.postprocess_metric_column,
                                        c.postprocess_contact_cutoff) } )
    )

    NEGSTEER_CROSS_SUMMARY( postprocessed.postprocessed )
}
