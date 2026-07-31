#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PARTIALAAMS_URL=https://github.com/TieuLongPhan/PartialAAMs.git
PARTIALAAMS_COMMIT=008173ed7a943ab03f1b8a33bfe5c7aea84dace9

REPETITIONS=5
LIMIT=""
CASE_TIMEOUT=10
SYNKIT_ENV=synkit
EXTERNAL_ENV=aam
PARTIALAAMS=/tmp/PartialAAMs-008
OUTPUT_DIR=""
FORCE=0

usage() {
    cat <<'EOF'
Run the hydrogen-extension comparison with one process and one thread.

Usage:
  Experiment/Lewis/hydrogen_expand/run_comparison.sh [OPTIONS]

Options:
  --repetitions N       Executions per method (default: 5)
  --limit N             Pilot input limit; omit for all 109 records
  --case-timeout SEC    Per-case timeout (default: 10)
  --synkit-env NAME     Environment for HExtend (default: synkit)
  --external-env NAME   Environment for GM/RB1/RB2 (default: aam)
  --partialaams DIR     Pinned PartialAAMs checkout (cloned if absent)
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
        --synkit-env) SYNKIT_ENV="${2:?missing environment}"; shift 2 ;;
        --external-env) EXTERNAL_ENV="${2:?missing environment}"; shift 2 ;;
        --partialaams) PARTIALAAMS="${2:?missing checkout}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?missing output directory}"; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -e "${PARTIALAAMS}" ]]; then
    git clone "${PARTIALAAMS_URL}" "${PARTIALAAMS}"
    git -C "${PARTIALAAMS}" checkout --detach "${PARTIALAAMS_COMMIT}"
fi
if [[ ! -d "${PARTIALAAMS}/.git" ]]; then
    echo "PartialAAMs path is not a Git checkout: ${PARTIALAAMS}" >&2
    exit 1
fi
ACTUAL_COMMIT="$(git -C "${PARTIALAAMS}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${PARTIALAAMS_COMMIT}" ]]; then
    echo "PartialAAMs must be pinned at ${PARTIALAAMS_COMMIT}; found ${ACTUAL_COMMIT}" >&2
    exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
    OUTPUT_DIR="${SCRIPT_DIR}/../Runs/${RUN_STAMP}-hydrogen-expand"
fi

ARGS=(
    --output-dir "${OUTPUT_DIR}"
    --partialaams "${PARTIALAAMS}"
    --external-env "${EXTERNAL_ENV}"
    --repetitions "${REPETITIONS}"
    --case-timeout "${CASE_TIMEOUT}"
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
conda run --no-capture-output -n "${SYNKIT_ENV}" python \
    "${SCRIPT_DIR}/benchmark.py" "${ARGS[@]}"
