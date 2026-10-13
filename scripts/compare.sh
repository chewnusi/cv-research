#!/usr/bin/env bash
# Run the same benchmark with both orchestration modes (only ORCH_MODE differs) and compare.
# Extra arguments are passed to `cvorch bench`, e.g.:  scripts/compare.sh --repeats 3
set -euo pipefail
cd "$(dirname "$0")/.."

stamp=$(date +%Y%m%d-%H%M%S)
root="results/compare-$stamp"

ORCH_MODE=baseline uv run cvorch bench --out "$root/baseline" "$@"
ORCH_MODE=planned  uv run cvorch bench --out "$root/planned"  "$@"
uv run cvorch compare "$root/baseline" "$root/planned" --out "$root/comparison.md"
echo "comparison written to $root/comparison.md"
