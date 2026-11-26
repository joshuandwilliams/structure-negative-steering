"""PipelineInternalThresholds — one catalogue holding every research
threshold that used to be hard-coded across `bin/`.  The defaults live
here, in this single Python module, and a caller can override them
programmatically but normally does not.

This file is 44 statements in 158 lines, the highest line-to-statement ratio
in bin/. That ratio usually means padding; here the extra lines are the
comments recording what each threshold gates, which is the only place that is
written down. Stripping them to shorten the file would delete the answer to
"what does this number do".

Changing a threshold here is a change to the research methodology.
That is deliberately separate from the user-facing pipeline parameters,
which are set per experiment in each `config.yml`.

Design of the type:
- Immutable.  Every override returns a new instance.
- One home.  The defaults are Python literals on the dataclass, so they
  are version-controlled next to the code that uses them.  A YAML file
  was rejected for that reason.
- Used by passing an instance into every method that does a threshold
  check, such as `StageResult.triggers_next_stage(thresholds)`.

Tier 0 type — no upstream dependencies on other typed modules.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Dict


@dataclass(frozen=True)
class PipelineInternalThresholds:
    """The catalogue of internal research-methodology thresholds.

    Every default repeats a value that was hard-coded somewhere under
    `bin/`, or used as a function's default argument, so adopting it
    changed no results.

    All fields are immutable.  Use `with_overrides({...})` for a
    modified copy.
    """

    # ── Structural thresholds (negsteer per-seed gating) ──────────────
    #
    # `steered_ra_eff` and `reverted_ra_eff` are the "correctly placed"
    # bar, per `cmd_build_contaminated` in reversion.py and
    # `_classify_outcome` in negsteer_aggregate.py.
    # `cold_start_rmsd_threshold` mirrors the pipeline parameter
    # `negsteer_rmsd_threshold`, set per experiment in its config.yml.
    # `intact_threshold` is the receptor-fold bar on
    # independent_receptor_rmsd.
    steered_ra_eff: float = 5.0
    reverted_ra_eff: float = 5.0
    cold_start_rmsd_threshold: float = 6.0
    intact_threshold: float = 5.0

    # ── Confidence thresholds (used by extract_passing.confidence_flag) ──
    #
    # These match the fallback literals in `_compute_confidence_flag`
    # in bin/extract_passing.py.  pae_pass_frac_min floors the fraction
    # of inter-chain residue pairs whose PAE is below the cutoff.
    iptm_min: float = 0.30
    complex_plddt_min: float = 0.70
    ipae_max: float = 15.0
    pae_pass_frac_min: float = 0.10

    # ── Orthogonal gate ──────────────────────────────────────────────
    #
    # The gate uses shape complementarity, buried surface area and ΔΔG
    # only, excluding interface_plddt as a model-confidence number
    # rather than a physical measurement.  The defaults came from the
    # deleted orthogonal_metrics_plots.py and
    # merge_orthogonal_metrics.py.  The ΔΔG default is the
    # Bennett et al. 2023 calibration reference.  af3_ra_max is
    # informational and not gated.
    orthogonal_sc_min: float = 0.55
    orthogonal_bsa_min: float = 600.0
    orthogonal_ddg_max: float = -30.0
    orthogonal_af3_ra_max: float = 5.0
    # Kept for the cohort-summary plot, not part of the
    # orthogonal_filters gate.
    orthogonal_plddt_min_informational: float = 0.75

    # ── Contact and interface geometry ────────────────────────────────
    #
    # contact_cutoff is the heavy-atom contact distance, used by
    # find_contact_residues_heavy and the negsteer contamination check.
    # The weighted_jaccard_* values match compute_metrics.py's fallback
    # literals _WJ_MU=4.0 and _WJ_TWO_SIGMA_SQ=2.25, so
    # sigma = sqrt(2.25 / 2), about 1.06.  The deleted
    # compute_interface_metrics.py held an identical copy.
    contact_cutoff: float = 4.5
    weighted_jaccard_mu: float = 4.0
    weighted_jaccard_two_sigma_sq: float = 2.25
    weighted_jaccard_pair_cutoff: float = 8.0

    # ── Composite score ───────────────────────────────────────────────
    #
    # true_jaccard - composite_ra_eff_weight * ra_eff.  The 0.05
    # default matches COMPOSITE_RA_EFF_WEIGHT in negsteer_aggregate.py
    # and the plot scripts.
    composite_ra_eff_weight: float = 0.05

    # ── Interface and control diagnostics ─────────────────────────────
    #
    # interface_plddt_trim_threshold drops an interface residue below
    # that per-residue pLDDT from the interface_plddt average.  The
    # controls_warning_* values flag a negative control that looks
    # suspiciously good.  All three defaults came from main.nf
    # parameters.
    interface_plddt_trim_threshold: float = 50.0
    controls_warning_ipsae_max: float = 0.5
    controls_warning_ra_eff_min: float = 5.0

    # ── Sequence QC ───────────────────────────────────────────────────
    #
    # Quality-control limits on MPNN sequences, from the main.nf
    # parameters max_poly_x, min_pct_identity and max_pct_identity.
    max_poly_x: int = 5
    min_pct_identity: float = 0.0
    max_pct_identity: float = 100.0

    # ── Sc filter (Rosetta pre-MPNN) ──────────────────────────────────
    #
    # Shape complementarity for the Rosetta filter before MPNN.  The
    # 0.62 default came from a main.nf parameter and follows
    # Overath et al., 2025.
    sc_threshold: float = 0.62

    # ── Reversion gating ─────────────────────────────────────────────
    #
    # StageResult.triggers_next_stage decides whether reversion runs,
    # using steered_ra_eff for "correctly placed" and contact_cutoff
    # for contamination.  It needs no field of its own, its only other
    # term being ceil(n_correctly_placed / 2).

    def as_dict(self) -> Dict[str, Any]:
        """Return every threshold value in one flat dict."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def with_overrides(
        self, overrides: Dict[str, Any]
    ) -> "PipelineInternalThresholds":
        """Return a new instance with the given overrides applied.

        An unknown override key raises ``TypeError``, which catches
        typos.
        """
        valid = {f.name for f in fields(self)}
        unknown = set(overrides) - valid
        if unknown:
            raise TypeError(
                "Unknown threshold override(s): "
                f"{sorted(unknown)}.  Valid: {sorted(valid)}."
            )
        return replace(self, **overrides)

    @classmethod
    def default(cls) -> "PipelineInternalThresholds":
        """Return an instance of every default.  Same as ``cls()``, but
        easier for a caller to find."""
        return cls()
