"""
pipeline_thresholds.py
----------------------
PipelineInternalThresholds — the catalogue of every hard-coded research
threshold currently scattered across `bin/`.  Defaults live here, in one
Python module.  The user CAN override (programmatically) but normally
doesn't — these are research-methodology defaults the user has chosen as
good values for their pipeline.

This module is the home of the upcoming Task 47 threshold audit.
Changing a threshold here is a research-methodology change with an
audit trail; changing a `PipelineParams` field is a user-contract change
with a different audit trail (see bin/pipeline_params.py).

Type design (Phase 4 spec §2.13):
- Immutable: every override returns a new instance.
- Single home: defaults are Python literals on the dataclass, version-
  controlled with the code (chosen over YAML so the methodology stays
  bundled with the code that uses it).
- Consumed by passing an instance through every method that does a
  threshold check (`StageResult.triggers_next_stage(thresholds)`,
  `OrthogonalMetrics.passes_orthogonal_filters(thresholds)`, etc.).

Tier 0 type — no upstream dependencies on other Phase 4 types.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Dict


@dataclass(frozen=True)
class PipelineInternalThresholds:
    """The catalogue of internal research-methodology thresholds.

    Every default is the value currently hard-coded somewhere in the
    `bin/` tree (or used as a default argument in a current function).
    The initial population mirrors what the codebase does today — Task
    47 will revise individual values once the audit decides them.

    All fields are immutable.  To produce a modified instance, use
    `with_overrides({...})` — never assign in place.
    """

    # ── Structural thresholds (negsteer per-seed gating) ──────────────
    #
    # `steered_ra_eff` and `reverted_ra_eff` set the "correctly placed"
    # bar for steered and reverted predictions respectively.  Today both
    # are 5.0 Å (see `cmd_build_contaminated` and `_classify_outcome`).
    # `cold_start_rmsd_threshold` corresponds to today's
    # `params.negsteer_rmsd_threshold` (default 6.0 — see main.nf:159).
    # `intact_threshold` is the receptor-fold-quality bar used by the
    # intact-check (independent_receptor_rmsd ≤ this).
    steered_ra_eff: float = 5.0
    reverted_ra_eff: float = 5.0
    cold_start_rmsd_threshold: float = 6.0
    intact_threshold: float = 5.0

    # ── Confidence thresholds (used by extract_passing.confidence_flag) ──
    #
    # Defaults from `bin/extract_passing.py:_compute_confidence_flag`
    # (lines 188-190).  pae_pass_frac_min is the floor on the fraction
    # of inter-chain residue pairs whose PAE is below the cutoff.
    iptm_min: float = 0.30
    complex_plddt_min: float = 0.70
    ipae_max: float = 15.0
    pae_pass_frac_min: float = 0.10

    # ── Orthogonal gate (Sc + BSA + ΔΔG only — interface_plddt is NOT
    # in the gate per Session 5 Q118 and commit 47cb9f2) ───────────────
    #
    # Defaults from bin/orthogonal_metrics_plots.py:87-93 and the
    # current merge_orthogonal_metrics.py.  ΔΔG default is the Bennett
    # et al. 2023 calibration reference.  af3_ra_max is informational
    # only (NOT gated) but kept here for the cohort-summary plot.
    orthogonal_sc_min: float = 0.55
    orthogonal_bsa_min: float = 600.0
    orthogonal_ddg_max: float = -30.0
    orthogonal_af3_ra_max: float = 5.0
    # Retained for cohort-summary plot use; explicitly NOT part of the
    # orthogonal_filters gate (see Phase 4 spec §2.11 boundary note and
    # notes/inventory/06_ubiquitous_language.md §passes_orthogonal_filters).
    orthogonal_plddt_min_informational: float = 0.75

    # ── Contact and interface geometry ────────────────────────────────
    #
    # contact_cutoff is the heavy-atom distance threshold for declaring
    # two residues "in contact" — used by find_contact_residues_heavy
    # and the negsteer contamination check.  weighted_jaccard_*
    # constants come from compute_metrics.py and compute_interface_
    # metrics.py (the two existing copies have identical values:
    # _WJ_MU=4.0 and _WJ_TWO_SIGMA_SQ=2.25; sigma = sqrt(2.25/2) ≈ 1.06).
    contact_cutoff: float = 4.5
    weighted_jaccard_mu: float = 4.0
    weighted_jaccard_two_sigma_sq: float = 2.25
    weighted_jaccard_pair_cutoff: float = 8.0

    # ── Composite score ───────────────────────────────────────────────
    #
    # composite = true_jaccard − composite_ra_eff_weight × ra_eff.
    # Default 0.05 from boltz2_iterate_steering.py and the plot scripts'
    # COMPOSITE_RA_EFF_WEIGHT constant.
    composite_ra_eff_weight: float = 0.05

    # ── Interface and control diagnostics ─────────────────────────────
    #
    # interface_plddt_trim_threshold: per-residue pLDDT below which an
    # interface residue's plddt is excluded from the interface_plddt
    # average.  Default 50.0 from main.nf:207.
    # controls_warning_*: thresholds that flag a negative control as
    # SUSPICIOUSLY GOOD (a sign the control isn't behaving as a control).
    # Defaults from main.nf:199-201.
    interface_plddt_trim_threshold: float = 50.0
    controls_warning_ipsae_max: float = 0.5
    controls_warning_ra_eff_min: float = 5.0

    # ── Sequence QC ───────────────────────────────────────────────────
    #
    # MPNN sequence quality control limits.  Defaults from main.nf
    # parameters max_poly_x / min_pct_identity / max_pct_identity.
    max_poly_x: int = 5
    min_pct_identity: float = 0.0
    max_pct_identity: float = 100.0

    # ── Sc filter (Rosetta pre-MPNN) ──────────────────────────────────
    #
    # Shape complementarity threshold for the Rosetta pre-MPNN filter
    # pass.  Default 0.62 from main.nf:93 (Overath et al., 2025).
    sc_threshold: float = 0.62

    # ── Reversion gating (CL-3 — new in Phase 4) ──────────────────────
    #
    # The CL-3 rule itself is implemented in StageResult.triggers_next_
    # stage (steered) and uses the structural thresholds above —
    # specifically steered_ra_eff for "correctly placed" and
    # contact_cutoff for contamination detection.  No new threshold
    # field is needed for the CL-3 rule beyond ceil(n_correctly_placed
    # / 2), which is an arithmetic identity, not a configurable value.

    def as_dict(self) -> Dict[str, Any]:
        """Flat snapshot of all threshold values."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def with_overrides(
        self, overrides: Dict[str, Any]
    ) -> "PipelineInternalThresholds":
        """Return a NEW instance with the given overrides applied.

        Original instance is unchanged (frozen dataclass).  Raises
        ``TypeError`` if any override key is not a known threshold —
        catches typos.
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
        """Return an instance with all defaults (the canonical research-
        methodology values currently in use).  Equivalent to ``cls()``
        but more discoverable from a caller."""
        return cls()
