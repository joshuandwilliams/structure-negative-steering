# Derived once at import, so extract_row branches per RULE rather than per
# column. Five tight comprehensions instead of a 47-iteration if/elif chain.
_AS_IS_COLS    = tuple(c for c, r in COLUMNS if r is AS_IS)
_STEERED_COLS  = tuple(c for c, r in COLUMNS if r is STEERED)
_FINAL_COLS    = tuple(c for c, r in COLUMNS if r is FINAL)
_OR_STEER_COLS = tuple(c for c, r in COLUMNS if r is OR_STEER)
_CONF_COLS     = tuple(c for c, r in COLUMNS if r is CONF)
_SEED_COLS     = tuple(c for c, r in COLUMNS if r is SEEDS)


def _positions_from_chimx(s):
    """Residue numbers out of a "/A:5,7,22" chain selection."""
    if not s or ":" not in s:
        return []
    return [int(p) for p in s.split(":", 1)[1].split(",")
            if p.strip().lstrip("-").isdigit()]


def _mutation_columns(per_seed_rows):
    """The seven SEEDS columns, or blanks without any seed rows.

    The steered set is identical across a sequence group's seeds, so the
    first seed carrying one supplies it. The reverted set can differ,
    because each seed reverts against its own steered prediction's contact
    list, so it is summarised across seeds.
    """
    if not per_seed_rows:
        return dict.fromkeys(_SEED_COLS, "")

    steered = next((ps for ps in per_seed_rows
                    if ps.get("steered_mutations_chimerax")), {})
    majority, union, identical = _summarize_mutations_across_seeds(
        per_seed_rows, "reverted_mutations_chimerax")

    def aa_for(chimx):
        return _muts_aa_for_positions(per_seed_rows, "reverted_mutations_aa",
                                      _positions_from_chimx(chimx))

    return {
        "steered_mutations_chimerax": steered.get(
            "steered_mutations_chimerax", ""),
        "steered_mutations_aa": steered.get("steered_mutations_aa", ""),
        "reverted_mutations_majority_chimerax": majority,
        "reverted_mutations_majority_aa": aa_for(majority),
        "reverted_mutations_all_chimerax": union,
        "reverted_mutations_all_aa": aa_for(union),
        "reverted_mutations_seeds_identical": identical,
    }


def extract_row(r, per_seed_rows=None):
    """Build a passing_summary row from one aggregated_results row.

    COLUMNS above names every column and the rule that fills it. The rule
    that matters is which stage's prediction a metric comes from. Reversion
    was attempted for pose_holds, pose_collapses and new_contamination, so
    the reverted prediction is the final design even when reversion failed,
    and steered_* would present the still-mutated pre-reversion prediction
    as the result. It was not attempted for no_reversion, where the steered
    prediction had no contamination, nor for a singleton, where only one
    seed had data.

    Past bug, fixed 2026-04-29. This used steered_* for everything except
    pose_holds, so pose_collapses and new_contamination rows reported their
    cold-start metrics and looked unrealistically good. It never showed in
    tier-A, B or C rows, since main() filters down to
    {pose_holds, no_reversion, ""}. It surfaced when cross_summary.py's
    tier-none fallback, added the same day, began reading
    aggregated_results.csv directly, which does contain them.

    `per_seed_rows` is the raw_per_seed_results rows for this sequence group.
    """
    final = ("reverted_" if (r.get("outcome") or "") in _REVERSION_OUTCOMES
             else "steered_")
    out = {}
    for c in _AS_IS_COLS:
        out[c] = r.get(c, "")
    for c in _STEERED_COLS:
        out[c] = r.get("steered_" + c, "")
    for c in _FINAL_COLS:
        out[c] = _get(r, final + c)
    for c in _OR_STEER_COLS:
        out[c] = _get(r, final + c, "steered_" + c)
    for c in _CONF_COLS:
        out[c + "_median"] = _get(r, final + c + "_median",
                                  "steered_" + c + "_median")
    out.update(_mutation_columns(per_seed_rows))
    out["confidence_flag"] = _compute_confidence_flag(out)
    return out
