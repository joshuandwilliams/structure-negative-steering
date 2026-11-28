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

# Read from the installed distribution rather than written here, so there is
# one declaration. Hand-writing it here meant pyproject.toml said 0.0.0.9000
# while `negsteer --version` and every negsteer_result.json said 1.0.0.
#
# The fallback covers running from a source tree that was never pip-installed,
# which the test suite does. It repeats pyproject's value and is the only place
# that can drift, so it is kept identical on purpose.
try:  # pragma: no cover - the installed path is what CI exercises
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("structure-negative-steering")
except (ImportError, PackageNotFoundError):  # pragma: no cover
    __version__ = "0.0.0.9000"

# The output contract, versioned separately from the package. A caller pins
# this rather than __version__. The package may change without the shape of
# the run directory changing, and a caller only cares about the latter.
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
