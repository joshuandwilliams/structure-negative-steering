# Resolved once at import rather than on every call.  The old code did the
# sys.path insert and the import inside _compute_confidence_flag, so a run
# over N rows pushed N copies of bin/ onto sys.path and re-resolved
# __file__ N times.  500 calls added 500 duplicate entries, which
# tests/characterization/test_modules_repair_sys_path.py asserts must not
# happen for the modules it covers.
#
# The fallbacks are the values that were hard-coded across bin/ before
# pipeline_thresholds existed, kept so this script still runs with that
# module off the path.
try:
    from pipeline_thresholds import PipelineInternalThresholds
    _T = PipelineInternalThresholds.default()
    _DEFAULT_PASS_FRAC_MIN = _T.pae_pass_frac_min
    _DEFAULT_IPTM_MIN = _T.iptm_min
    _DEFAULT_COMPLEX_PLDDT_MIN = _T.complex_plddt_min
    _DEFAULT_IPAE_MAX = _T.ipae_max
except ImportError:  # pragma: no cover - the module sits beside this one
    _DEFAULT_PASS_FRAC_MIN = 0.10
    _DEFAULT_IPTM_MIN = 0.30
    _DEFAULT_COMPLEX_PLDDT_MIN = 0.70
    _DEFAULT_IPAE_MAX = 15.0

# (column, comparison, threshold, label).  "<" flags a value below the
# threshold, ">" above it.  The default used when the column is missing is
# the one that does NOT trigger, so an absent metric is not a failure.
_CONFIDENCE_CHECKS = (
    ("pae_pass_frac_median", "<", _DEFAULT_PASS_FRAC_MIN, "low_pass_frac", 1),
    ("iptm_median", "<", _DEFAULT_IPTM_MIN, "low_iptm", 1),
    ("complex_plddt_median", "<", _DEFAULT_COMPLEX_PLDDT_MIN, "low_plddt", 1),
    ("ipae_median", ">", _DEFAULT_IPAE_MAX, "high_ipae", 0),
)


def _compute_confidence_flag(out, pass_frac_min=None, iptm_min=None,
                             complex_plddt_min=None, ipae_max=None):
    """Compute confidence_flag from the median metrics in `out`.

    Working on medians means a sequence is flagged only when its typical
    prediction fails the filter, so a single rogue seed does not flag it.

    Thresholds come from PipelineInternalThresholds at import time.  A
    caller can override any of the four.  A metric that is absent or
    unparseable does not trigger: the row is reporting what it has, and a
    missing column is not evidence of low confidence.
    """
    overrides = (pass_frac_min, iptm_min, complex_plddt_min, ipae_max)
    triggers = []
    for (col, op, default, label, missing), override in zip(
            _CONFIDENCE_CHECKS, overrides):
        limit = default if override is None else override
        try:
            value = float(out.get(col, missing))
        except (ValueError, TypeError):
            continue
        if (value < limit) if op == "<" else (value > limit):
            triggers.append(label)
    if not triggers:
        return "ok"
    return triggers[0] if len(triggers) == 1 else "multiple"
