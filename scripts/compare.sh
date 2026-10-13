#!/usr/bin/env bash
# Primary experiment: run the same benchmark with the two primary profiles (baseline vs planned) and compare.
# Only the orchestration strategy differs; model, prompts, tool contracts, workspace and cases are shared.
# Extra arguments are passed to `cvorch bench`, e.g.:  scripts/compare.sh --repeats 3
set -euo pipefail
cd "$(dirname "$0")/.."

stamp=$(date +%Y%m%d-%H%M%S)
root="results/compare-$stamp"

uv run cvorch bench --profile baseline --out "$root/baseline" "$@"
uv run cvorch bench --profile planned  --out "$root/planned"  "$@"
uv run cvorch compare "$root/baseline" "$root/planned" --out "$root/comparison.md"
echo "comparison written to $root/comparison.md"
