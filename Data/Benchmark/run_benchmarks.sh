#!/usr/bin/env bash
#===============================================================================
# run_benchmarks.sh
#-------------------------------------------------------------------------------
# Execute reactor, IO conversion, and clustering benchmarks with optional limit.
#
# Usage:
#   ./run_benchmarks.sh [LIMIT]
#
#   LIMIT:
#     - positive integer : number of entries to process in each benchmark
#     - None             : process all entries (no limit)
#     - (omitted)        : defaults to 10
#
# Place this script in the "Data/Benchmark" directory.
#===============================================================================

set -euo pipefail

# Determine script directory (Data/Benchmark)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse limit argument
if [[ "${1:-}" == "None" || "${1:-}" == "none" ]]; then
    LIMIT_FLAG=""       # no limit flag → Python scripts run all
    DISPLAY_LIMIT="all"
elif [[ -n "${1:-}" ]]; then
    LIMIT_FLAG="$1"
    DISPLAY_LIMIT="$1"
else
    LIMIT_FLAG="10"
    DISPLAY_LIMIT="10"
fi

echo "Running reactor benchmark with limit=${DISPLAY_LIMIT}"
if [[ -n "$LIMIT_FLAG" ]]; then
    python "$SCRIPT_DIR/reactor/benchmark_reactor.py" --limit "$LIMIT_FLAG"
else
    python "$SCRIPT_DIR/reactor/benchmark_reactor.py"
fi

echo "Running IO conversion benchmark with limit=${DISPLAY_LIMIT}"
if [[ -n "$LIMIT_FLAG" ]]; then
    python "$SCRIPT_DIR/io/io_convert.py" --limit "$LIMIT_FLAG"
else
    python "$SCRIPT_DIR/io/io_convert.py"
fi

echo "Running clustering benchmark with limit=${DISPLAY_LIMIT}"
if [[ -n "$LIMIT_FLAG" ]]; then
    python "$SCRIPT_DIR/cluster/benchmarking_clustering.py" --limit "$LIMIT_FLAG"
else
    python "$SCRIPT_DIR/cluster/benchmarking_clustering.py"
fi

echo "All benchmarks completed."
