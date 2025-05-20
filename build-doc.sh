#!/bin/bash

set -e

echo "Checking and installing documentation dependencies..."

# Function to check and install a Python package if missing
install_if_missing() {
    PACKAGE=$1
    python -c "import $PACKAGE" 2>/dev/null || {
        echo "Installing $PACKAGE..."
        pip install "$PACKAGE"
    }
}

# Check and install main Sphinx packages
install_if_missing sphinx
install_if_missing sphinx_rtd_theme

# sphinxcontrib-bibtex can have a dash in its import, so:
pip show sphinxcontrib-bibtex >/dev/null 2>&1 || {
    echo "Installing sphinxcontrib-bibtex..."
    pip install sphinxcontrib-bibtex
}

echo "Building Sphinx documentation..."

# Build the docs (adjust source/build dirs as needed)
python3 -m sphinx -b html ./doc ./docs

echo "Documentation built in ./docs"