import os
import sys

# -- Path setup --------------------------------------------------------------
# Add project root to sys.path to import the package
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "synkit"
author = "Tieu-Long Phan"

# Dynamically detect version:
# 1) Try package metadata (requires pip install -e .)
# 2) Fallback to synkit.__version__ if available
# 3) Default to known release if neither works
from importlib.metadata import version as _get_version, PackageNotFoundError


try:
    release = _get_version("synkit")
except PackageNotFoundError:
    try:
        import synkit
        release = synkit.__version__
    except (ImportError, AttributeError):
        # Fallback default
        release = "0.0.9"
# Use only major.minor for short version
version = ".".join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.githubpages",
    "sphinxcontrib.bibtex",
    # "sphinx.ext.napoleon",  # un-comment if using Google/NumPy docstrings
]

bibtex_bibfiles = ["refs.bib"]
templates_path = ["_templates"]
exclude_patterns = []
autosectionlabel_prefix_document = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
