#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
cd "${ROOT_DIR}"

if (( $# )); then
    exec "${PYTHON_BIN}" -m pytest "$@"
fi
exec "${PYTHON_BIN}" -m pytest Test/
