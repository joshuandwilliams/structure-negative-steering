/*
 * =============================================================================
 * Negative-steering module
 * =============================================================================
 * Replaces the old boltz2.nf module.  For each MPNN-designed sequence,
 * runs a full single-cycle negative-steering experiment driven by
 * boltz2_negative_steering.py + boltz2_iterate_steering.py:
 *
 *   plan  →  predict-one (×N)  →  collect  →
 *   kickoff-distances + kickoff-prefilter + build-contaminated +
 *   plan-reversions  →  predict-reversion-one (×M)  →
 *   harvest-reversions + kickoff-finalize  →
 *   aggregate  →  compute-final-metrics  →  aggregate-per-sequence  →
 *   extract_passing
 *
 * All stages for one MPNN sequence are collapsed into a SINGLE SLURM
 * GPU job (NEGSTEER_RUN_ONE) via bin/negative_steering_run_one.sh.
 * This avoids the submission explosion the old SLURM-array chain would
 * create when scaled to 64+ MPNN sequences, while keeping cross-
 * sequence parallelism (one NextFlow task per MPNN sequence, capped by
 * params.max_boltz2_parallel).
 *
 * Cross-sequence aggregation (NEGSTEER_CROSS_SEQUENCE) picks the
 * tier-then-composite representative from each per-sequence
 * passing_summary.csv and produces a single ranked triage table.
 *
 * Container notes (Singularity 3.8 + .img format on NBI HPC):
 *   - --nv required for GPU processes
 *   - --bind $PWD:$PWD is REQUIRED to make work dir writable
 *   - --bind applied to every input file's parent directory so the
 *     container sees identical paths to the host (negative-steering
 *     Python scripts resolve many paths absolutely via plan.json).
 */


/*
 * NEGSTEER_DERIVE_INDICES
 * ───────────────────────
 * Derive the two index files that negative-steering's plan stage
 * consumes via --true-interface-indices-file and
 * --design-region-indices-file.  Both files are produced from a
 * single rfdiffusion_metrics.json + design-name pair:
 *
 *   derive_design_region.py   → <design>_design_region.txt (1-based)
 *   derive_true_interface.py  → <design>_true_interface.txt (0-based)
 *
 * Called once per passing RFDiffusion design.  Cheap (stdlib + JSON).
 */
process NEGSTEER_DERIVE_INDICES {
    tag "${design_pdb.baseName}"
    label 'cpu'

    publishDir "${params.outdir}/negative_steering/indices", mode: 'copy'

    input:
    path design_pdb                 // e.g. design_3.pdb
    path rfdiffusion_metrics_json   // shared rfdiffusion_metrics.json
    path design_region_script
    path true_interface_script

    output:
    // Tuple carries the design stem alongside the two index files so
    // downstream joins on design name stay explicit.
    tuple val(design_pdb.baseName),
          path("${design_pdb.baseName}_design_region.txt"),
          path("${design_pdb.baseName}_true_interface.txt"),
          emit: indices

    script:
    def design_stem = design_pdb.baseName
    """
    set -euo pipefail

    # ── Design region (1-based) ──────────────────────────────────
    singularity exec --bind \${PWD}:\${PWD} ${params.boltz2_container} \\
        python ${design_region_script} \\
            --metrics-json ${rfdiffusion_metrics_json} \\
            --design ${design_stem} \\
            --output ${design_stem}_design_region.txt

    # ── True interface (0-based) + cross-check ───────────────────
    # Cross-check validates that the derived true interface is a
    # subset of the design region (advisory only — warnings surface
    # in the process log).
    singularity exec --bind \${PWD}:\${PWD} ${params.boltz2_container} \\
        python ${true_interface_script} \\
            --metrics-json ${rfdiffusion_metrics_json} \\
            --design ${design_stem} \\
            --output ${design_stem}_true_interface.txt \\
            --cross-check-design-region ${design_stem}_design_region.txt

    echo "Derived index files for ${design_stem}:"
    ls -la ${design_stem}_design_region.txt ${design_stem}_true_interface.txt
    """
}


/*
 * NEGSTEER_RUN_ONE
 * ────────────────
 * Run the complete single-cycle negative-steering pipeline for ONE
 * MPNN sequence on one GPU node.  The heavy lifting lives in
 * bin/negative_steering_run_one.sh which loops every phase inline
 * (plan → predict-one×N → collect → reversion prep → predict-
 * reversion-one×M → harvest + finalize → postprocess).
 *
 * Input tuple:
 *   seq_name       — e.g. "design_3_seq_1"
 *   design_stem    — the parent RFDiffusion design name, e.g. "design_3"
 *   mpnn_fasta     — path to design_X_seq_Y.fasta (has >receptor + >effector)
 *   design_pdb     — path to the parent RFDiffusion design PDB (design_X.pdb)
 *   design_region  — path to <design_X>_design_region.txt
 *   true_interface — path to <design_X>_true_interface.txt
 *
 * Output:
 *   per_sequence_workdir — the entire workdir (cycle_0/, all CSVs,
 *                          plan.json, etc.) for downstream aggregation
 *                          and for human inspection.
 *
 * Concurrency: capped at params.max_boltz2_parallel by withName config.
 */
process NEGSTEER_RUN_ONE {
    tag "${seq_name}"
    label 'gpu'

    publishDir "${params.outdir}/negative_steering/runs", mode: 'copy'

    input:
    tuple val(seq_name), val(design_stem),
          path(mpnn_fasta), path(design_pdb),
          path(design_region), path(true_interface)
    val  receptor_chain
    val  effector_chain
    val  plan_extra_args
    val  postprocess_rmsd_threshold
    val  postprocess_metric_column
    val  postprocess_contact_cutoff
    // Stage the orchestrator bash script as a path input so edits to
    // it bust the cache.  NOTE: bin/negative_steering_run_one.sh
    // invokes several bin/*.py sub-scripts via --bin-dir
    // ${projectDir}/bin; those indirect dependencies are NOT
    // content-hashed by this declaration.  Declaring them all
    // individually would invent a pattern the rest of the codebase
    // does not use, so it is left for a future targeted task.
    path orchestrator_script

    output:
    // Publish the entire per-sequence workdir.  passing_summary.csv
    // and aggregated_results.csv are both inside it; the cross-
    // sequence aggregator picks them up by glob.
    path "${seq_name}/", emit: per_sequence_workdir

    script:
    """
    set -euo pipefail

    # ─── Split the MPNN FASTA into receptor.fasta + effector.fasta ───
    # Negative steering's plan stage takes the two chains as separate
    # FASTA files via --receptor-fasta / --effector-fasta.  The MPNN
    # pipeline writes them combined as:
    #   >receptor
    #   SEQ
    #   >effector
    #   SEQ
    # Tiny stdlib-only Python split keeps us out of awk-edge-case territory.
    mkdir -p ${seq_name}/inputs

    python3 - <<'PYEOF'
from pathlib import Path
in_path = Path("${mpnn_fasta}")
out_dir = Path("${seq_name}/inputs")
out_dir.mkdir(parents=True, exist_ok=True)

current = None
chains = {"receptor": [], "effector": []}
with in_path.open() as f:
    for line in f:
        line = line.rstrip()
        if line.startswith(">"):
            header = line[1:].strip().lower()
            if header.startswith("receptor"):
                current = "receptor"
            elif header.startswith("effector"):
                current = "effector"
            else:
                current = None
        elif current is not None and line:
            chains[current].append(line.strip())

for chain in ("receptor", "effector"):
    joined = "".join(chains[chain])
    if not joined:
        raise SystemExit(f"ERROR: {chain} chain empty in {in_path}")
    (out_dir / f"{chain}.fasta").write_text(f">{chain}\\n{joined}\\n")

print(f"wrote {out_dir/'receptor.fasta'} ({len(chains['receptor'][0])} residues) "
      f"and {out_dir/'effector.fasta'}")
PYEOF

    # ─── Invoke the inline orchestrator ──────────────────────────────
    # bin/negative_steering_run_one.sh expects absolute paths; it
    # calls readlink -f internally but --bin-dir must resolve cleanly.
    bash ${orchestrator_script} \\
        --seq-name                     "${seq_name}" \\
        --ground-truth                 "${design_pdb}" \\
        --receptor-chain               "${receptor_chain}" \\
        --effector-chain               "${effector_chain}" \\
        --receptor-fasta               "${seq_name}/inputs/receptor.fasta" \\
        --effector-fasta               "${seq_name}/inputs/effector.fasta" \\
        --true-interface-indices-file  "${true_interface}" \\
        --design-region-indices-file   "${design_region}" \\
        --workdir                      "${seq_name}" \\
        --bin-dir                      "${projectDir}/bin" \\
        --boltz-container              "${params.boltz2_container}" \\
        --plan-extra-args              "${plan_extra_args}" \\
        --postprocess-rmsd-threshold   "${postprocess_rmsd_threshold}" \\
        --postprocess-metric-column    "${postprocess_metric_column}" \\
        --postprocess-contact-cutoff   "${postprocess_contact_cutoff}"

    # Sanity: the final deliverable must exist or publishDir will still
    # publish the directory but downstream aggregation will have
    # nothing to pick up.  Fail loudly here so the log shows the
    # broken invariant immediately.
    if [[ ! -f "${seq_name}/passing_summary.csv" ]]; then
        echo "ERROR: ${seq_name}/passing_summary.csv not produced" >&2
        ls -la ${seq_name}/ || true
        exit 1
    fi
    """
}


/*
 * NEGSTEER_CROSS_SEQUENCE
 * ───────────────────────
 * Aggregate every per-sequence passing_summary.csv into a single
 * ranked cross-sequence triage table using the tier-then-composite
 * policy from notes 11:
 *
 *   Tier A: n_pass == n_seeds (e.g. 3/3 pass-equivalent)
 *   Tier B: 1 < n_pass < n_seeds (e.g. 2/3)
 *   Tier C: n_pass == 1
 *
 * Within each tier, cross-rank by the composite score
 * (true_jaccard − 0.05 · ra_eff_vs_truth).  Sequences with an empty
 * passing_summary.csv appear in the output with cross_tier == "none"
 * and the cross-rank columns blank.
 */
process NEGSTEER_CROSS_SEQUENCE {
    tag "cross_sequence"
    label 'cpu'

    publishDir "${params.outdir}/negative_steering", mode: 'copy'

    input:
    // Full list of per-sequence workdirs (collected upstream).  Each
    // directory is expected to be named after the MPNN sequence; the
    // aggregator infers the sequence name from the directory name.
    path per_sequence_workdirs
    // Entire bin/ directory.  Staged into the work dir as ``bin/``; the
    // CLI invocation runs ``python bin/cross_summary_v2.py …`` so
    // Python's sibling-import resolution finds every transitively-
    // imported module (design_cohort, cross_sequence_summary,
    // extract_passing, …) without sys.path acrobatics.  Nextflow
    // content-hashes the whole directory, closing the indirect-import
    // cache-busting gap noted in notes/remediation_state.md.
    path bin_dir, name: 'bin'
    // Optional MPNN scored_metadata.csv (from MPNN_DESIGN_REGION_SCORE).
    // Surfaces corrected_receptor / designed_residues / native_residues
    // in cross_sequence_summary.csv.  Pass `file('NO_FILE')` (or any
    // missing path) to skip the join — the script tolerates a missing
    // file and emits blank columns.
    path scored_metadata

    output:
    path "cross_sequence_summary.csv", emit: cross_summary

    script:
    def metadata_arg = (scored_metadata.name == 'NO_FILE')
        ? ''
        : "--scored-metadata ${scored_metadata}"
    """
    set -euo pipefail

    # Stage a single root that contains one subdir per MPNN sequence.
    # Nextflow stages each input directory as a sibling; we symlink
    # them into an aggregator/ root so --passing-summary-dir gets a
    # clean tree to walk.
    mkdir -p aggregator
    for d in ${per_sequence_workdirs}; do
        if [[ -d "\$d" ]]; then
            ln -sf "\$(readlink -f \$d)" "aggregator/\$(basename \$d)"
        fi
    done

    echo "Aggregator root:"
    # `ls | head -N` under `set -euo pipefail` is a SIGPIPE trap: when
    # head closes its stdin after N lines, ls gets SIGPIPE and exits
    # 141, which pipefail propagates as the pipeline's exit status,
    # killing the process before the aggregator runs.  Trailing
    # `|| true` swallows that 141 without disabling pipefail for the
    # singularity invocation below.  Same pattern as
    # negsteer_af3_nomsa.nf:224.
    ls -la aggregator/ | head -40 || true

    singularity exec --bind \${PWD}:\${PWD} ${params.boltz2_container} \\
        python bin/cross_summary_v2.py \\
            --passing-summary-dir aggregator \\
            --published-runs-dir  ${params.outdir}/negative_steering/runs \\
            ${metadata_arg} \\
            --output cross_sequence_summary.csv

    echo "Cross-sequence summary head:"
    head -5 cross_sequence_summary.csv || true
    """
}


/*
 * NEGSTEER_PLOTS
 * ──────────────
 * Cohort-level diagnostic plots from cross_sequence_summary.csv.
 * Mirrors the test harness at tests/negative_steering/test_negsteer_plots.py
 * exactly — bin/negsteer_plots.py is a verbatim lift of that script
 * with only the test-path fallback removed (and matching docstring).
 *
 * Seven PNGs (see bin/negsteer_plots.py docstring for the full
 * catalogue):
 *   1. negsteer_tier_landscape.png
 *   2. negsteer_seed_outcomes_heatmap.png
 *   3. negsteer_seed_outcomes_bars.png
 *   4. negsteer_ra_eff_vs_jaccard.png
 *   5. negsteer_filter_cascade.png
 *   6. negsteer_controls_diagnostic.png
 *   7. negsteer_mutation_impact.png
 *
 * Inputs:
 *   - cross_summary: cross_sequence_summary.csv from
 *     NEGSTEER_CROSS_SEQUENCE.
 *   - per_sequence_workdirs: every per-sequence workdir (collected),
 *     symlinked under runs/ so cohort plots include tier-none
 *     sequences (they have aggregated_results.csv but are absent
 *     from cross_summary because they have zero passing rows).
 *   - input_design_region: input_design_region.txt from
 *     DERIVE_INPUT_INDICES; used by the mutation-impact plot to
 *     shade design-region positions on the input PDB.
 *
 * Container: rfdiff_container — plotting only needs matplotlib /
 * numpy / stdlib, mirroring MPNN_PLOTS / RFDIFFUSION_PLOTS.  Does
 * NOT need boltz2_container.  Confidence-cascade thresholds match
 * extract_passing.py defaults; override via params if needed.
 */
process NEGSTEER_PLOTS {
    tag "negsteer_plots"
    label 'cpu'

    publishDir "${params.outdir}/plots", mode: 'copy'

    input:
    path cross_summary
    path per_sequence_workdirs
    path input_design_region
    path plots_script

    output:
    path "negsteer_*.png", emit: plots

    script:
    """
    set -euo pipefail

    # Stage per-sequence workdirs under runs/ so the cohort plot can
    # discover tier-none sequences via their aggregated_results.csv
    # (they're absent from cross_summary).  Same staging hack as
    # NEGSTEER_CROSS_SEQUENCE.
    mkdir -p runs
    for d in ${per_sequence_workdirs}; do
        if [[ -d "\$d" ]]; then
            ln -sf "\$(readlink -f \$d)" "runs/\$(basename \$d)"
        fi
    done

    singularity exec \\
        --bind \${PWD}:\${PWD} \\
        --env MPLCONFIGDIR=/tmp \\
        ${params.rfdiff_container} \\
        python ${plots_script} \\
            --csv                  ${cross_summary} \\
            --runs-dir             runs \\
            --input-design-region  ${input_design_region} \\
            --outdir               .
    """
}


/*
 * NEGSTEER_WITHIN_SEQUENCE_PLOTS
 * ──────────────────────────────
 * Per-seed (within-sequence) dispersion plots.  Companion to
 * NEGSTEER_PLOTS — that one plots cohort-level summaries; this one
 * shows the spread of seeds within each sequence so cohort medians
 * can be sanity-checked against per-seed dispersion.
 *
 * Mirrors tests/negative_steering/test_negsteer_within_sequence_plots.py
 * exactly — bin/negsteer_within_sequence_plots.py is a verbatim
 * lift with only the production docstring header changed.
 *
 * Five PNGs:
 *   1. negsteer_per_seed_dispersion_overview.png
 *   2. negsteer_per_seed_dispersion_grid.png
 *   3. negsteer_per_design_stage_trajectories.png
 *   4. negsteer_weighted_vs_true_jaccard.png
 *   5. negsteer_composite_vs_confidence.png
 *
 * Inputs identical to NEGSTEER_PLOTS minus input_design_region —
 * the within-sequence plots are PDB-position-agnostic.
 */
process NEGSTEER_WITHIN_SEQUENCE_PLOTS {
    tag "negsteer_within_sequence_plots"
    label 'cpu'

    publishDir "${params.outdir}/plots", mode: 'copy'

    input:
    path cross_summary
    path per_sequence_workdirs
    path plots_script

    output:
    path "negsteer_*.png", emit: plots

    script:
    """
    set -euo pipefail

    # Same per-sequence staging hack as NEGSTEER_CROSS_SEQUENCE /
    # NEGSTEER_PLOTS.  The within-sequence plots walk runs/<name>/
    # for raw_per_seed_results.csv per sequence.
    mkdir -p runs
    for d in ${per_sequence_workdirs}; do
        if [[ -d "\$d" ]]; then
            ln -sf "\$(readlink -f \$d)" "runs/\$(basename \$d)"
        fi
    done

    singularity exec \\
        --bind \${PWD}:\${PWD} \\
        --env MPLCONFIGDIR=/tmp \\
        ${params.rfdiff_container} \\
        python ${plots_script} \\
            --runs-dir          runs \\
            --cross-summary-csv ${cross_summary} \\
            --outdir            .
    """
}
