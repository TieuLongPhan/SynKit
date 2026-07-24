#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

usage() {
    cat <<'EOF'
Run the Lewis benchmark experiments into a fresh results directory.

Usage:
  Experiment/Lewis/run_experiments.sh EXPERIMENT [OPTIONS]

Experiments:
  partial_expand  Run current SynKit partial-AAM expansion on general + radical
  rule_replay     Run tuple and typesGH forward/backward rule replay
  mech_path       Run corpus reconstruction audit + MechanismBench evidence
  all             Run all three experiments in the order above

Options:
  --limit N                 Pilot limit per corpus (omit for full runs)
  --repetitions N           Partial-expansion repetitions (default: 5)
  --evidence-repetitions N  MechanismBench timing repetitions (default: 3)
  --case-timeout SECONDS    Expansion/replay case timeout (default: 10)
  --progress-every N        Progress interval (default: 500; 0 disables)
  --conda-env NAME          Current SynKit conda environment (default: synkit)
  --output-root DIR         Results directory (default: timestamp under Runs)
  --external                Also rerun historical GM/RB1/RB2 partial expansion
  --external-conda-env NAME Historical comparison environment (default: aam)
  --force                   Permit overwrite inside an explicit output root
  -h, --help                Show this help

Examples:
  ./Experiment/Lewis/run_experiments.sh all --limit 10 --repetitions 1
  ./Experiment/Lewis/run_experiments.sh mech_path
  ./Experiment/Lewis/run_experiments.sh partial_expand --external

The original corpora are read-only inputs. Failure CSVs written by mech_path
contain only one-based source-row IDs.
EOF
}

if [[ $# -eq 0 ]]; then
    usage
    exit 2
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

EXPERIMENT="$1"
shift
case "${EXPERIMENT}" in
    partial_expand|rule_replay|mech_path|all) ;;
    *)
        echo "Unknown experiment: ${EXPERIMENT}" >&2
        usage >&2
        exit 2
        ;;
esac

LIMIT=""
REPETITIONS=5
EVIDENCE_REPETITIONS=3
CASE_TIMEOUT=10
PROGRESS_EVERY=500
CONDA_ENV=synkit
EXTERNAL_CONDA_ENV=aam
OUTPUT_ROOT=""
RUN_EXTERNAL=0
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit)
            LIMIT="${2:?--limit requires an integer}"
            shift 2
            ;;
        --repetitions)
            REPETITIONS="${2:?--repetitions requires an integer}"
            shift 2
            ;;
        --evidence-repetitions)
            EVIDENCE_REPETITIONS="${2:?--evidence-repetitions requires an integer}"
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
        --conda-env)
            CONDA_ENV="${2:?--conda-env requires a name}"
            shift 2
            ;;
        --external-conda-env)
            EXTERNAL_CONDA_ENV="${2:?--external-conda-env requires a name}"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="${2:?--output-root requires a directory}"
            shift 2
            ;;
        --external)
            RUN_EXTERNAL=1
            shift
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

if [[ -z "${OUTPUT_ROOT}" ]]; then
    RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
    OUTPUT_ROOT="${SCRIPT_DIR}/Runs/${RUN_STAMP}-${EXPERIMENT}"
fi
if [[ -e "${OUTPUT_ROOT}" && "${FORCE}" -ne 1 ]]; then
    echo "Refusing existing output root: ${OUTPUT_ROOT}; pass --force" >&2
    exit 1
fi
mkdir -p "${OUTPUT_ROOT}"

PYTHON=(conda run --no-capture-output -n "${CONDA_ENV}" python)
LIMIT_ARGS=()
FORCE_ARGS=()
if [[ -n "${LIMIT}" ]]; then
    LIMIT_ARGS=(--limit "${LIMIT}")
fi
if [[ "${FORCE}" -eq 1 ]]; then
    FORCE_ARGS=(--force)
fi

run_partial_expand() {
    local output="${OUTPUT_ROOT}/partial_expand"
    echo "== partial_expand: general =="
    "${PYTHON[@]}" "${SCRIPT_DIR}/partial_expand/repeat_synkit.py" \
        --suite general \
        --repetitions "${REPETITIONS}" \
        --case-timeout "${CASE_TIMEOUT}" \
        --progress-every "${PROGRESS_EVERY}" \
        --output-dir "${output}/synkit-general" \
        "${LIMIT_ARGS[@]}" "${FORCE_ARGS[@]}"

    echo "== partial_expand: radical =="
    "${PYTHON[@]}" "${SCRIPT_DIR}/partial_expand/repeat_synkit.py" \
        --suite radical \
        --repetitions "${REPETITIONS}" \
        --case-timeout "${CASE_TIMEOUT}" \
        --progress-every "${PROGRESS_EVERY}" \
        --output-dir "${output}/synkit-radical" \
        "${LIMIT_ARGS[@]}" "${FORCE_ARGS[@]}"

    if [[ "${RUN_EXTERNAL}" -eq 1 ]]; then
        echo "== partial_expand: historical GM/RB1/RB2 =="
        "${PYTHON[@]}" "${SCRIPT_DIR}/partial_expand/repeat_external.py" \
            --conda-env "${EXTERNAL_CONDA_ENV}" \
            --repetitions "${REPETITIONS}" \
            --case-timeout "${CASE_TIMEOUT}" \
            --progress-every "${PROGRESS_EVERY}" \
            --output-dir "${output}/external" \
            "${LIMIT_ARGS[@]}" "${FORCE_ARGS[@]}"
    fi
}

run_rule_replay() {
    echo "== rule_replay =="
    "${PYTHON[@]}" "${SCRIPT_DIR}/rule_replay/benchmark.py" \
        --case-timeout "${CASE_TIMEOUT}" \
        --progress-every "${PROGRESS_EVERY}" \
        --output-dir "${OUTPUT_ROOT}/rule_replay" \
        "${LIMIT_ARGS[@]}"
}

run_mech_path() {
    local output="${OUTPUT_ROOT}/mech_path"
    echo "== mech_path: corpus reconstruction audit =="
    "${PYTHON[@]}" "${SCRIPT_DIR}/mech_path/audit.py" \
        --progress-every "${PROGRESS_EVERY}" \
        --output-dir "${output}/reconstruction_audit" \
        "${LIMIT_ARGS[@]}" "${FORCE_ARGS[@]}"

    echo "== mech_path: reviewed MechanismBench evidence =="
    "${PYTHON[@]}" "${SCRIPT_DIR}/mech_path/evidence.py" \
        --repetitions "${EVIDENCE_REPETITIONS}" \
        --output "${output}/mechanismbench-evidence.json"
}

cd "${REPO_ROOT}"
case "${EXPERIMENT}" in
    partial_expand) run_partial_expand ;;
    rule_replay) run_rule_replay ;;
    mech_path) run_mech_path ;;
    all)
        run_partial_expand
        run_rule_replay
        run_mech_path
        ;;
esac

echo "Completed ${EXPERIMENT}"
echo "Results: ${OUTPUT_ROOT}"
