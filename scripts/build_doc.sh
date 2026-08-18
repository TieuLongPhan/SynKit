#!/usr/bin/env bash
# Build the Sphinx documentation into the GitHub Pages output directory.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SOURCE_DIR="$ROOT_DIR/doc"
BUILD_DIR="$ROOT_DIR/docs"
MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/synkit-matplotlib-cache}"

mkdir -p "$MPLCONFIGDIR"
export MPLCONFIGDIR

if ! command -v sphinx-build >/dev/null 2>&1; then
    echo "sphinx-build is not installed. Install the documentation dependencies first:" >&2
    echo "  python -m pip install -r doc/requirements.txt" >&2
    exit 1
fi

exec sphinx-build -E -W --keep-going -b html "$SOURCE_DIR" "$BUILD_DIR"
