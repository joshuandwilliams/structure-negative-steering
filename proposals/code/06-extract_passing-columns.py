# ── the output columns, in order, each with the rule that fills it ───────────
# One list, so a column cannot be in the CSV header and unfilled by
# extract_row, which csv.DictWriter does not catch in that direction.
#
#   AS_IS    r[name]
#   STEERED  r["steered_" + name], whatever the outcome
#   FINAL    r[final + name], where final is reverted_ when reversion ran
#   OR_STEER r[final + name], falling back to r["steered_" + name]
#   CONF     the same, with a _median suffix on a bare metric name
#   SEEDS    from raw_per_seed_results.csv, not from this row
#   LATER    main() fills it
AS_IS, STEERED, FINAL, OR_STEER, CONF, SEEDS, LATER = range(7)

COLUMNS = (
    ("rank_by_composite_score", AS_IS), ("rank_by_ra_eff", AS_IS),
    ("cycle", AS_IS), ("pathway", AS_IS), ("sequence_group", AS_IS),
    ("design", AS_IS), ("canonical_pdb", AS_IS),
    ("canonical_seed_index", AS_IS),
    ("canonical_seed_choice_warning", AS_IS), ("n_seeds", AS_IS),
    ("n_pass", AS_IS), ("outcome", AS_IS),

    # Every seed of one sequence has the same mutation set, so the median
    # just reports that single value.
    ("total_mutations_median", OR_STEER),

    # Mutations as a chain selection.  "steered" is every steering mutation
    # that went into Boltz.  "majority" is the positions still mutated in a
    # majority of seeds' reverted predictions, "all" the union across seeds,
    # the worst case for what the construct would retain.  None of these are
    # in the wet-lab construct: they describe the sequence Boltz scored.
    ("steered_mutations_chimerax", SEEDS), ("steered_mutations_aa", SEEDS),
    ("reverted_mutations_majority_chimerax", SEEDS),
    ("reverted_mutations_majority_aa", SEEDS),
    ("reverted_mutations_all_chimerax", SEEDS),
    ("reverted_mutations_all_aa", SEEDS),
    ("reverted_mutations_seeds_identical", SEEDS),

    # Structural RMSDs against ground truth, the primary triage metric.  The
    # spread columns (MAD, n_used) are in aggregated_results.csv.  No
    # fallback: a reverted row missing these is broken, not steered.
    ("ra_eff_vs_truth_median", FINAL),
    ("independent_receptor_rmsd_median", FINAL),
    ("independent_effector_rmsd_median", FINAL),

    # Interface similarity, always the steered side.  The designed interface
    # is a property of the sequence rather than the pose, and reverted
    # predictions do not recompute jaccard.
    ("wrong_jaccard_median", STEERED), ("true_jaccard_median", STEERED),
    ("n_shared_true_median", STEERED), ("n_shared_wrong_median", STEERED),

    # Contact analysis follows the structural metrics' prediction.
    ("n_contact_residues_median", OR_STEER),
    ("contact_residues_majority", OR_STEER),
    ("mutated_contact_positions_majority", OR_STEER),

    # Confidence, medians only; spread is in aggregated_results.csv.
    # ipsae_*_15 is the same calculation at a more permissive PAE cutoff.
    # interface_plddt and weighted_jaccard moved here from the post-hoc
    # orthogonal-metrics pass.
    ("iptm", CONF), ("ptm", CONF), ("actifptm", CONF),
    ("complex_plddt", CONF), ("avg_plddt", CONF), ("ipae", CONF),
    ("pae_mean", CONF), ("pae_pass_frac", CONF), ("ipsae_min", CONF),
    ("ipsae_ab_15", CONF), ("ipsae_ba_15", CONF), ("ipsae_min_15", CONF),
    ("interface_plddt", CONF), ("weighted_jaccard", CONF),

    # Binary per seed, so a majority rather than a median.
    ("intact_core_majority", OR_STEER),

    ("confidence_flag", LATER),

    # Cumulative wall-clock seconds for this sequence's
    # negative_steering_run_one.sh invocation, from run_one_runtime_sec.txt.
    # Repeats on every row, since one invocation covers one sequence.
    ("run_one_runtime_sec", LATER),
)

OUTPUT_FIELDS = [c + "_median" if rule is CONF else c for c, rule in COLUMNS]

_REVERSION_OUTCOMES = ("pose_holds", "pose_collapses", "new_contamination")
