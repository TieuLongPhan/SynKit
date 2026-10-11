#!/usr/bin/env bash
# Build the Sphinx documentation into the GitHub Pages output directory.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SOURCE_DIR="$ROOT_DIR/doc"
BUILD_DIR="$ROOT_DIR/docs"
PYTHON_BIN="${PYTHON:-python}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/synkit-matplotlib-cache}"

mkdir -p "$MPLCONFIGDIR"
export MPLCONFIGDIR

if ! "${PYTHON_BIN}" -m sphinx --version >/dev/null 2>&1; then
    echo "sphinx-build is not installed. Install the documentation dependencies first:" >&2
    echo "  python -m pip install -r doc/requirements.txt" >&2
    exit 1
fi

exec "${PYTHON_BIN}" -m sphinx -E -W --keep-going -b html "$SOURCE_DIR" "$BUILD_DIR"
