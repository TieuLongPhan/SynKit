#!/usr/bin/env bash
# Build the Sphinx documentation into the GitHub Pages output directory.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$ROOT_DIR/doc"
BUILD_DIR="$ROOT_DIR/docs"

if ! command -v sphinx-build >/dev/null 2>&1; then
    echo "sphinx-build is not installed. Install the documentation dependencies first:" >&2
    echo "  python -m pip install -r doc/requirements.txt" >&2
    exit 1
fi

exec sphinx-build -E -W --keep-going -b html "$SOURCE_DIR" "$BUILD_DIR"
