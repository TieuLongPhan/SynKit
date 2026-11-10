#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
cd "${ROOT_DIR}"

if (( $# )); then
    exec "${PYTHON_BIN}" -m pytest "$@"
fi
TEST_PATHS=(Test/)
for EXPERIMENT_TEST_PATH in Experiment/CRN/tests Experiment/atom_map/tests; do
    if [[ -d "${EXPERIMENT_TEST_PATH}" ]]; then
        TEST_PATHS+=("${EXPERIMENT_TEST_PATH}")
    fi
done
exec "${PYTHON_BIN}" -m pytest "${TEST_PATHS[@]}"
