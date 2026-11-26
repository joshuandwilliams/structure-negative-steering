def _stamp_ranks(scored, column: str) -> None:
    """Sort (value, rec) pairs tier-first and stamp 1-based ranks.

    Tier first in both rankings: a tier B record with a slightly higher raw
    composite still ranks below every tier A record, because tier is a
    reliability measure and the composite is not. `value` is already signed so
    that ascending order is "better first", so the caller negates a composite
    and leaves an ra_eff alone. id() breaks remaining ties as stably as
    Python's own sort would.
    """
    scored.sort(key=lambda pair: (_TIER_ORDER[pair[1]["cross_tier"]],
                                  pair[0], id(pair[1])))
    for rank_idx, (_value, rec) in enumerate(scored, start=1):
        rec[column] = rank_idx


def _assign_cross_ranks(records: List[Dict]) -> None:
    """Compute and stamp cross_rank_by_composite / cross_rank_by_ra_eff in place."""
    scored_ra = []
    scored_composite = []
    for rec in records:
        if rec.get("row_type", "steered") != "steered":
            continue
        rep_cols = {
            "ra_eff_vs_truth_median":
                rec.get("rep_ra_eff_vs_truth_median", ""),
            "true_jaccard_median":
                rec.get("rep_true_jaccard_median", ""),
        }
        # Requiring a placed tier keeps the "none"-tier stubs out.
        if rec["cross_tier"] in ("A", "B", "C"):
            comp = _composite_score(rep_cols)
            if comp is not None:
                rec["cross_composite_score"] = f"{comp:.6f}"
                scored_composite.append((-comp, rec))   # higher is better
            ra = _try_float(rep_cols["ra_eff_vs_truth_median"])
            if ra is not None:
                scored_ra.append((ra, rec))             # lower is better
    _stamp_ranks(scored_composite, "cross_rank_by_composite")
    _stamp_ranks(scored_ra, "cross_rank_by_ra_eff")
