#!/bin/bash
#
# Print the Boltz container for a config, using nothing but bash, grep and sed.
#
#   resolve_boltz_container.sh CONFIG [OVERRIDE]
#
# The airgapped HPC has no PyYAML on the host, so the container path cannot be
# read with a YAML parser. It has to come out of the file as plain text, because
# the container is what the parser would have to run inside. That is the same
# chicken-and-egg negative_steering.slurm.sh solves, and this is its extraction
# lifted out so the two cannot drift.
#
# Prints the resolved path, or nothing when there is none. Always exits 0. The
# caller decides whether an empty answer is fatal, because it is not always:
# a CI runner has PyYAML installed and no singularity at all.
#
# tests/characterization/test_container_resolution.py asserts this agrees with
# PyYAML on every committed config. A silent disagreement would send a run to a
# different container than the one the config names.

set -euo pipefail

CONFIG="${1:-}"
OVERRIDE="${2:-}"

if [ -n "${OVERRIDE}" ]; then
    printf '%s' "${OVERRIDE}"
    exit 0
fi

if [ -z "${CONFIG}" ] || [ ! -f "${CONFIG}" ]; then
    exit 0
fi

# `|| true` because no match is a normal answer, not a failure. Without it
# grep's exit 1 propagates through pipefail and kills the caller.
LINE="$(grep -E '^[[:space:]]*boltz_container:' "${CONFIG}" | head -1 || true)"

if [ -z "${LINE}" ]; then
    exit 0
fi

# Everything after the first colon, left-trimmed.
VALUE="${LINE#*:}"
VALUE="${VALUE#"${VALUE%%[![:space:]]*}"}"

case "${VALUE}" in
    \"*)
        # Quoted, so the value ends at the closing quote. A '#' inside quotes is
        # part of the path, not a comment, which is how PyYAML reads it too.
        VALUE="${VALUE#\"}"
        VALUE="${VALUE%%\"*}"
        ;;
    "'"*)
        VALUE="${VALUE#\'}"
        VALUE="${VALUE%%\'*}"
        ;;
    *)
        # Unquoted. A '#' preceded by whitespace starts a comment. These configs
        # do use trailing comments, so dropping this would hand back a path with
        # a comment glued to it.
        VALUE="${VALUE%%[[:space:]]#*}"
        # Right-trim. Only quoted values had their trailing space stripped above.
        VALUE="${VALUE%"${VALUE##*[![:space:]]}"}"
        ;;
esac

printf '%s' "${VALUE}"
