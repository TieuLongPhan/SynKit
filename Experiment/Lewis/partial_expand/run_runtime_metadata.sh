#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

usage() {
    cat <<'EOF'
Collect reaction-level partial-extension runtime metadata.

Usage:
  Experiment/Lewis/partial_expand/run_runtime_metadata.sh [OPTIONS]

Runs the current SynKit/LWG implementation and the historical RB1, RB2, and GM
implementations on the general corpus. Each method is executed five times by
default. No figure is generated and bulky candidate/case files are discarded.

Options:
  --repetitions N       Independent executions per method (default: 5)
  --limit N             Pilot input limit; omit for the full corpus
  --case-timeout SEC    Per-reaction timeout (default: 10)
  --progress-every N    Progress interval (default: 500; 0 disables)
  --synkit-env NAME     Current LWG conda environment (default: synkit)
  --external-env NAME   Set both historical environments (default: aam)
  --gm-env NAME         GM conda environment (overrides --external-env)
  --rb-env NAME         RB1/RB2 conda environment (overrides --external-env)
  --partialaams DIR     Historical PartialAAMs checkout (cloned if absent)
  --gmapache DIR        Optional GranMapache checkout for commit provenance
  --current-only        Run only current SynKit/LWG
  --historical-only     Run only GM/RB1/RB2; useful for resuming
  --output-dir DIR      Output root (default: timestamp under Experiment/Lewis/Runs)
  --force               Permit overwrite inside an explicit output directory
  -h, --help            Show this help
EOF
}

REPETITIONS=5
CASE_TIMEOUT=10
PROGRESS_EVERY=500
SYNKIT_ENV=synkit
EXTERNAL_ENV=aam
GM_ENV=""
RB_ENV=""
PARTIALAAMS=/tmp/PartialAAMs-008
PARTIALAAMS_URL=https://github.com/TieuLongPhan/PartialAAMs.git
PARTIALAAMS_COMMIT=008173ed7a943ab03f1b8a33bfe5c7aea84dace9
GMAPACHE=""
LIMIT=""
OUTPUT_DIR=""
FORCE=0
MODE=all

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repetitions)
            REPETITIONS="${2:?--repetitions requires an integer}"
            shift 2
            ;;
        --limit)
            LIMIT="${2:?--limit requires an integer}"
            shift 2
            ;;
        --case-timeout)
            CASE_TIMEOUT="${2:?--case-timeout requires seconds}"
            shift 2
            ;;
        --progress-every)
            PROGRESS_EVERY="${2:?--progress-every requires an integer}"
            shift 2
            ;;
        --synkit-env)
            SYNKIT_ENV="${2:?--synkit-env requires a name}"
            shift 2
            ;;
        --external-env)
            EXTERNAL_ENV="${2:?--external-env requires a name}"
            shift 2
            ;;
        --gm-env)
            GM_ENV="${2:?--gm-env requires a name}"
            shift 2
            ;;
        --rb-env)
            RB_ENV="${2:?--rb-env requires a name}"
            shift 2
            ;;
        --partialaams)
            PARTIALAAMS="${2:?--partialaams requires a path}"
            shift 2
            ;;
        --gmapache)
            GMAPACHE="${2:?--gmapache requires a path}"
            shift 2
            ;;
        --current-only)
            if [[ "${MODE}" == historical ]]; then
                echo "--current-only and --historical-only are mutually exclusive" >&2
                exit 2
            fi
            MODE=current
            shift
            ;;
        --historical-only)
            if [[ "${MODE}" == current ]]; then
                echo "--current-only and --historical-only are mutually exclusive" >&2
                exit 2
            fi
            MODE=historical
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="${2:?--output-dir requires a path}"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
    RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
    OUTPUT_DIR="${SCRIPT_DIR}/../Runs/${RUN_STAMP}-partial-expand-runtime"
fi
RUN_CURRENT=1
RUN_HISTORICAL=1
if [[ "${MODE}" == current ]]; then
    RUN_HISTORICAL=0
elif [[ "${MODE}" == historical ]]; then
    RUN_CURRENT=0
fi
if [[ "${RUN_HISTORICAL}" -eq 1 ]]; then
    if [[ ! -e "${PARTIALAAMS}" ]]; then
        echo "Cloning historical PartialAAMs ${PARTIALAAMS_COMMIT}"
        git clone "${PARTIALAAMS_URL}" "${PARTIALAAMS}"
        git -C "${PARTIALAAMS}" checkout --detach "${PARTIALAAMS_COMMIT}"
    fi
    if [[ ! -d "${PARTIALAAMS}/.git" ]]; then
        echo "PartialAAMs path is not a Git checkout: ${PARTIALAAMS}" >&2
        exit 1
    fi
    PARTIALAAMS_ACTUAL_COMMIT="$(git -C "${PARTIALAAMS}" rev-parse HEAD)"
    if [[ "${PARTIALAAMS_ACTUAL_COMMIT}" != "${PARTIALAAMS_COMMIT}" ]]; then
        echo "PartialAAMs must be pinned at ${PARTIALAAMS_COMMIT}; found ${PARTIALAAMS_ACTUAL_COMMIT}" >&2
        exit 1
    fi
fi
if [[ "${FORCE}" -ne 1 ]]; then
    if [[ "${RUN_CURRENT}" -eq 1 && -e "${OUTPUT_DIR}/current-lwg" ]]; then
        echo "Refusing existing current-LWG output: ${OUTPUT_DIR}/current-lwg" >&2
        exit 1
    fi
    if [[ "${RUN_HISTORICAL}" -eq 1 && -e "${OUTPUT_DIR}/historical" ]]; then
        echo "Refusing existing historical output: ${OUTPUT_DIR}/historical" >&2
        exit 1
    fi
fi
mkdir -p "${OUTPUT_DIR}"

LIMIT_ARGS=()
FORCE_ARGS=()
if [[ -n "${LIMIT}" ]]; then
    LIMIT_ARGS=(--limit "${LIMIT}")
fi
if [[ "${FORCE}" -eq 1 ]]; then
    FORCE_ARGS=(--force)
fi
GMAPACHE_ARGS=()
if [[ -n "${GMAPACHE}" ]]; then
    GMAPACHE_ARGS=(--gmapache "${GMAPACHE}")
fi
GM_ENV="${GM_ENV:-${EXTERNAL_ENV}}"
RB_ENV="${RB_ENV:-${EXTERNAL_ENV}}"

cd "${REPO_ROOT}"

if [[ "${RUN_CURRENT}" -eq 1 ]]; then
    echo "== current SynKit/LWG: ${REPETITIONS} executions =="
    conda run --no-capture-output -n "${SYNKIT_ENV}" python \
        "${SCRIPT_DIR}/repeat_synkit.py" \
        --suite general \
        --repetitions "${REPETITIONS}" \
        --case-timeout "${CASE_TIMEOUT}" \
        --progress-every "${PROGRESS_EVERY}" \
        --output-dir "${OUTPUT_DIR}/current-lwg" \
        --metadata-only \
        "${LIMIT_ARGS[@]}" "${FORCE_ARGS[@]}"
fi

if [[ "${RUN_HISTORICAL}" -eq 1 ]]; then
    echo "== historical RB1/RB2/GM: ${REPETITIONS} executions each =="
    conda run --no-capture-output -n "${SYNKIT_ENV}" python \
        "${SCRIPT_DIR}/repeat_external.py" \
        --gm-conda-env "${GM_ENV}" \
        --rb-conda-env "${RB_ENV}" \
        --repetitions "${REPETITIONS}" \
        --case-timeout "${CASE_TIMEOUT}" \
        --progress-every "${PROGRESS_EVERY}" \
        --partialaams "${PARTIALAAMS}" \
        "${GMAPACHE_ARGS[@]}" \
        --output-dir "${OUTPUT_DIR}/historical" \
        --metadata-only \
        "${LIMIT_ARGS[@]}" "${FORCE_ARGS[@]}"
fi

echo "Runtime metadata complete: ${OUTPUT_DIR}"
