#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
REPETITIONS=5
LIMIT=""
CASE_TIMEOUT=60
GMAPACHE_ENV=aam
OUTPUT_DIR=""
FORCE=0

usage() {
    cat <<'EOF'
Run the hydrogen-extension comparison with one process and one thread.

Usage:
  Experiment/Lewis/hydrogen_expand/run_comparison.sh [OPTIONS]

Options:
  --repetitions N       Executions per method (default: 5)
  --limit N             Pilot source limit; omit for all 109 (104 eligible)
  --case-timeout SEC    Per-case timeout (default: 60)
  --gmapache-env NAME   Conda env containing GranMapache (default: aam)
  --output-dir DIR      Output directory (default: timestamp under Lewis/Runs)
  --force               Permit overwrite in an explicit output directory
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repetitions) REPETITIONS="${2:?missing repetitions}"; shift 2 ;;
        --limit) LIMIT="${2:?missing limit}"; shift 2 ;;
        --case-timeout) CASE_TIMEOUT="${2:?missing timeout}"; shift 2 ;;
        --gmapache-env) GMAPACHE_ENV="${2:?missing environment}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?missing output directory}"; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
    RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
    OUTPUT_DIR="${SCRIPT_DIR}/../Runs/${RUN_STAMP}-hydrogen-expand"
fi

ARGS=(
    --output-dir "${OUTPUT_DIR}"
    --repetitions "${REPETITIONS}"
    --case-timeout "${CASE_TIMEOUT}"
    --gmapache-env "${GMAPACHE_ENV}"
)
if [[ -n "${LIMIT}" ]]; then
    ARGS+=(--limit "${LIMIT}")
fi
if [[ "${FORCE}" -eq 1 ]]; then
    ARGS+=(--force)
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

cd "${REPO_ROOT}"
python "${SCRIPT_DIR}/benchmark.py" "${ARGS[@]}"
