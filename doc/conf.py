"""Sphinx configuration for the SynKit documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# Make the source tree importable by autodoc.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, os.path.abspath(".."))

from synkit.version import __version__


# ---------------------------------------------------------------------
# Helpers: release/version resolution
# ---------------------------------------------------------------------
def _major_minor(release: str) -> str:
    """
    Derive major.minor from a release string.

    :param release: Release string, e.g. "1.2.3".
    :type release: str
    :return: Major.minor string, e.g. "1.2".
    :rtype: str
    """
    parts = release.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return release


# ---------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------
project = "synkit"
author = "SynKit contributors"
copyright = f"{datetime.now().year}, {author}"

release = __version__

version = _major_minor(release)

rst_prolog = (
    f".. |synkit_release| replace:: {release}\n"
    f".. |synkit_version| replace:: {version}\n"
)


# ---------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.extlinks",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx.ext.graphviz",
]

bibtex_bibfiles = ["refs.bib"]

templates_path = ["_templates"]
(HERE / "_templates").mkdir(parents=True, exist_ok=True)

# Retain the reaction-stereochemistry source while omitting it from the
# published documentation until that interface is ready for release.
exclude_patterns: list[str] = ["stereo_reactions.rst"]

autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2
autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

intersphinx_mapping = (
    {}
    if os.environ.get("SYNKIT_DOCS_OFFLINE")
    else {
        "python": ("https://docs.python.org/3", None),
        "networkx": ("https://networkx.org/documentation/stable/", None),
    }
)

extlinks = {
    "gh": ("https://github.com/TieuLongPhan/SynKit/%s", "%s"),
    "issue": ("https://github.com/TieuLongPhan/SynKit/issues/%s", "#%s"),
}


# ---------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------
html_title = "SynKit Documentation"
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.svg"

html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_context = {
    "github_user": "TieuLongPhan",
    "github_repo": "SynKit",
    "github_version": "main",
    "doc_path": "doc",
}


# ---------------------------------------------------------------------
# Theme selection (pydata with RTD fallback)
# ---------------------------------------------------------------------
try:
    import pydata_sphinx_theme  # noqa: F401

    html_theme = "pydata_sphinx_theme"

    html_theme_options = {
        "logo": {
            "image_light": "_static/logo.svg",
            "image_dark": "_static/logo-dark.svg",
            "alt_text": "SynKit",
        },
        "navbar_start": ["navbar-logo"],
        "navbar_center": [],
        "navbar_end": ["theme-switcher", "navbar-icon-links"],
        "icon_links": [
            {
                "name": "GitHub",
                "url": "https://github.com/TieuLongPhan/SynKit",
                "icon": "fa-brands fa-github",
            },
            {
                "name": "Releases",
                "url": "https://github.com/tieulongphan/synkit/releases",
                "icon": "fa-solid fa-tags",
            },
            {
                "name": "PyPI",
                "url": "https://pypi.org/project/synkit/",
                "icon": "fa-solid fa-box",
            },
            {
                "name": "Issues",
                "url": "https://github.com/TieuLongPhan/SynKit/issues",
                "icon": "fa-regular fa-circle-question",
            },
        ],
        "show_nav_level": 2,
        "navigation_depth": 4,
        "show_toc_level": 3,
        "search_bar_text": "Search the docs…",
        "use_edit_page_button": True,
        "secondary_sidebar_items": ["page-toc"],
        "footer_start": ["copyright"],
    }

    html_sidebars = {
        "**": [
            "sidebar-globaltoc.html",
        ]
    }


except Exception:
    html_theme = "sphinx_rtd_theme"
    html_theme_options = {
        "logo_only": False,
        "collapse_navigation": False,
        "display_version": True,
        "navigation_depth": 4,
    }
    html_sidebars = {
        "**": [
            "about.html",
            "navigation.html",
            "searchbox.html",
            "relations.html",
            "sourcelink.html",
            "localtoc.html",
        ]
    }


# ---------------------------------------------------------------------
# UX extension options
# ---------------------------------------------------------------------
togglebutton_hint = "Show"
togglebutton_hint_hide = "Hide"

copybutton_prompt_is_regexp = True
copybutton_prompt_text = r"^\$ |^>>> |^\.\.\. "

todo_include_todos = False

source_suffix = {
    ".rst": "restructuredtext",
}
