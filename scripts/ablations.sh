#!/usr/bin/env bash
# Secondary experiment: the primary planned profile against its ablations (reported separately from the
# headline baseline-vs-planned comparison). Extra arguments are passed to `cvorch bench`.
set -euo pipefail
cd "$(dirname "$0")/.."

stamp=$(date +%Y%m%d-%H%M%S)
root="results/ablations-$stamp"
profiles=(planned planned_no_repair planned_no_grounding planned_minimal planned_no_validator planned_strict_schema
          planned_auto_prereq baseline_step_guard)

dirs=()
for p in "${profiles[@]}"; do
  uv run cvorch bench --profile "$p" --out "$root/$p" "$@"
  dirs+=("$root/$p")
done
uv run cvorch compare "${dirs[@]}" --out "$root/comparison.md"
echo "comparison written to $root/comparison.md"
