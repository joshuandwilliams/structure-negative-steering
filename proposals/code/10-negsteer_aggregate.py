# Inside _process_passing_row, replacing eight four-line failure returns.
def fail(msg, *, prefix=False):
    """Record the failure on the row and return the caller's result.

    One failure costs one design rather than the whole stage: the row comes
    back with steered_error set and the metric columns blank, which the
    aggregate CSV carries so the loss is visible rather than silent.

    `prefix` picks the progress-prefixed SKIP line the first call site used;
    the rest print a plain ERROR.
    """
    if prefix:
        print(f"  [{passing_progress}/{n_passing_total}] {pathway_label}: "
              f"SKIP ({msg})")
    else:
        print(f"    ERROR: {msg}")
    out_row["steered_error"] = msg
    return out_row, "failed"


# Then each of the eight becomes one line, for example:
#
#     if pdb_path is None or not pdb_path.exists():
#         return fail(f"pdb not found: {pdb_path_str}", prefix=True)
#
#     except subprocess.TimeoutExpired:
#         return fail(f"compute_metrics.py timed out after "
#                     f"{args.subprocess_timeout}s")
#
#     if not cm_rows:
#         return fail("compute_metrics.py CSV was empty")
#
# The returncode != 0 branch keeps its own code, because it also prints
# proc.stdout under --verbose.
