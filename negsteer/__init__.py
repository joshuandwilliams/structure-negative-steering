"""Negative steering as a callable tool.

Mutates the receptor surface residues a prediction wrongly places in contact
with the effector, so the model can no longer settle there. Residues at the
interface the caller intends to keep are protected, via --true-interface, so
what is removed is a competing site rather than the intended one. The method
is never told where to bind, only where not to break.

`negsteer run` takes a receptor sequence, an effector sequence, a reference
complex and two residue-index files, and writes a run directory. Callers are
expected to read negsteer_result.json rather than glob for files.
"""

__version__ = "1.0.0"

# The output contract, versioned separately from the package. A caller pins
# this, not __version__: the package may change without the shape of the run
# directory changing, and a caller only cares about the latter.
#
# 1  first published contract
OUTPUT_CONTRACT_VERSION = 1

# Written at the top level of --outdir. Anything not listed here is an
# implementation detail and may move without the contract version changing.
RESULT_MANIFEST = "negsteer_result.json"

# Named outputs, keyed by the name they carry in the manifest. A file can be
# absent when the run took a path that does not produce it, which is why the
# manifest reports presence rather than the caller assuming it.
OUTPUT_FILES = {
    "per_seed_results": "raw_per_seed_results.csv",
    "aggregated_results": "aggregated_results.csv",
    "passing_summary": "passing_summary.csv",
    "cross_sequence_summary": "cross_sequence_summary.csv",
    "runtime_seconds": "run_one_runtime_sec.txt",
}
