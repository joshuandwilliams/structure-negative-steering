// Drop-in replacement for receptor-resurfacing-pipeline's negative-steering
// module. Copy this into that repo's modules/ and delete the 455-line
// negative_steering.nf it replaces.
//
// The pipeline used to carry its own copy of the engine: bin/*.py plus a
// 458-line bash orchestrator, invoked with --bin-dir so the caller had to know
// this repo's internal layout. It now calls one command in one image, the same
// way it calls RFDiffusion.
//
// What the caller no longer has to know:
//   - where bin/ lives, or that bin/ exists
//   - which stages run, or in what order
//   - that the run is single-cycle, or what --max-cycles means
//   - which output files appear, or when one is legitimately absent
//
// Read negsteer_result.json rather than globbing the run directory. Its
// contract_version is what to pin; anything not named in its "outputs" block
// is an implementation detail.

process NEGATIVE_STEERING {
    tag "${name}"
    label 'gpu'

    publishDir "${params.outdir}/negative_steering/runs", mode: 'copy'

    input:
    tuple val(name), path(receptor_fasta), path(effector_fasta),
          path(reference_pdb), path(design_region), path(true_interface)

    output:
    tuple val(name), path("${name}"),                      emit: run_dir
    tuple val(name), path("${name}/negsteer_result.json"), emit: result
    path "${name}/passing_summary.csv", optional: true,    emit: passing

    script:
    // Knobs the pipeline sets. Anything left null keeps the engine default
    // rather than being restated here, so the two cannot drift apart.
    def knobs = [
        '--n-designs'        : params.negsteer_n_designs,
        '--num-seeds'        : params.negsteer_num_seeds,
        '--diffusion-samples': params.negsteer_diffusion_samples,
        '--recycling-steps'  : params.negsteer_recycling_steps,
        '--max-mutations'    : params.negsteer_max_mutations,
    ].findAll { _k, v -> v != null }
     .collect { k, v -> "${k} ${v}" }
     .join(' ')

    def constraints = params.negsteer_boltz_constraints ? '--boltz-constraints' : ''
    """
    negsteer run \\
        --name            ${name} \\
        --receptor-fasta  ${receptor_fasta} \\
        --effector-fasta  ${effector_fasta} \\
        --reference-pdb   ${reference_pdb} \\
        --design-region   ${design_region} \\
        --true-interface  ${true_interface} \\
        --receptor-chain  ${params.negsteer_receptor_chain ?: 'A'} \\
        --effector-chain  ${params.negsteer_effector_chain ?: 'B'} \\
        --outdir          ${name} \\
        ${knobs} ${constraints}
    """
}
