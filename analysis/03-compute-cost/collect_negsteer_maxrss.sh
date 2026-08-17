#!/bin/bash
# Recover peak memory for the negative-steering benchmark runs from Slurm
# accounting. Run this ON THE HPC, from anywhere.
#
#   bash collect_negsteer_maxrss.sh > negsteer_maxrss.csv
#
# Then copy negsteer_maxrss.csv back to
#   structure-negative-steering/analysis/03-compute-cost/
#
# Several targets were retried, so more than one job id can exist per target.
# Every id is emitted with its Slurm Elapsed; the staging script picks the one
# whose elapsed matches run_one_runtime_sec.txt, which is the run the analysis
# actually used.
set -uo pipefail

REPO="${1:-/hpc-home/jowillia/receptor_design/structure-negative-steering}"

echo "target,jobid,maxrss_gb,elapsed,state,runtime_sec_recorded"

for out in "$REPO"/experiments/benchmarking/*/*/negsteer_*.out; do
	[ -e "$out" ] || continue
	dir=$(dirname "$out")
	# The tree is <variant>/<target>/. Emit the constrained ones as
	# <target>_constrained so the CSV keeps one unambiguous key per run.
	target=$(basename "$dir")
	case "$(basename "$(dirname "$dir")")" in
		constrained) target="${target}_constrained" ;;
	esac
	jobid=$(basename "$out" .out)
	jobid=${jobid#negsteer_}

	recorded=""
	[ -f "$dir/run/run_one_runtime_sec.txt" ] &&
		recorded=$(tr -d '[:space:]' <"$dir/run/run_one_runtime_sec.txt")

	# MaxRSS is reported per job step, so take the largest across steps. Slurm
	# writes it with a unit suffix (K/M/G); normalise everything to GB.
	sacct -j "$jobid" --noheader -P -o JobID,MaxRSS,Elapsed,State |
		awk -v t="$target" -v j="$jobid" -v rec="$recorded" -F'|' '
		{
			rss = $2; gb = 0
			if (rss ~ /K$/) { sub(/K$/, "", rss); gb = rss / 1048576 }
			else if (rss ~ /M$/) { sub(/M$/, "", rss); gb = rss / 1024 }
			else if (rss ~ /G$/) { sub(/G$/, "", rss); gb = rss + 0 }
			if (gb > max) max = gb
			if ($1 == j) { el = $3; st = $4 }
		}
		END { printf "%s,%s,%.3f,%s,%s,%s\n", t, j, max, el, st, rec }'
done
