def _compute_metrics_cmd(script, prediction_dir, *, rec_len, eff_len,
                         output_csv, pae_cutoff, receptor_chain,
                         effector_chain, contact_cutoff,
                         native_pdb=None, effector_atom_filter=None,
                         mutated_positions=None):
    """The compute_metrics.py command line both reversion stages build.

    The ten shared flags are here. The three optional ones are not shared:

      native_pdb              harvest only. Without it intact_core,
                              weighted_jaccard and interface_plddt come out
                              blank. The steered side passes the cycle_0
                              ground truth too, so the two stay comparable.
      effector_atom_filter    restricts contact detection to the effector
                              interface-atom set in cycle_0/plan.json. An
                              empty list there leaves detection unfiltered,
                              which compute_metrics.py handles.
      mutated_positions       build-contaminated only, for the contamination
                              check.
    """
    cmd = [
        sys.executable, str(script),
        "--model", "boltz2",
        "--prediction-dir", str(prediction_dir),
        "--chain-lengths", str(rec_len), str(eff_len),
        "--output-csv", str(output_csv),
        "--pae-cutoff", str(pae_cutoff),
        "--receptor-chain", receptor_chain,
        "--effector-chain", effector_chain,
        "--contact-cutoff", str(contact_cutoff),
    ]
    if native_pdb is not None:
        cmd += ["--native-pdb", str(native_pdb)]
    if effector_atom_filter is not None:
        cmd += ["--effector-atom-filter-json", str(effector_atom_filter)]
    if mutated_positions:
        cmd += ["--mutated-positions", mutated_positions]
    return cmd
