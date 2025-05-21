import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "synkit"
author = "Tieu-Long Phan"
release = "0.0.8"
version = release

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
