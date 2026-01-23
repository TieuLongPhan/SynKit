"""
Sphinx configuration for SynKit.

Modern, stable configuration:
- Uses pydata_sphinx_theme (modern UI + light/dark switcher)
- Falls back to sphinx_rtd_theme if pydata is not available
- Avoids fragile template overrides (DO NOT create _templates/sidebar-nav-bs.html)
- LEFT sidebar shows the toctree (sidebar-nav-bs)
- RIGHT sidebar shows the on-this-page outline (page-toc)

IMPORTANT
---------
Your docs folder appears to be `doc/` (not `docs/`), so html_context["doc_path"] is "doc".
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as _get_version
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
HERE = Path(__file__).resolve().parent  # e.g. <repo>/doc
REPO_ROOT = HERE.parent

# Ensure repo root is importable (preferred) and keep your original behaviour
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, os.path.abspath(".."))


# ---------------------------------------------------------------------
# Helpers: robust release/version resolution
# ---------------------------------------------------------------------
def _git_describe(repo_root: Path) -> Optional[str]:
    """
    Try to derive a version-like string from git tags.

    :param repo_root: Repository root containing the .git directory.
    :type repo_root: Path
    :return: A tag/describe string if git is available, otherwise None.
    :rtype: Optional[str]
    """
    cmds = [
        ["git", "describe", "--tags", "--abbrev=0"],
        ["git", "describe", "--tags", "--always"],
    ]
    for cmd in cmds:
        try:
            out = subprocess.check_output(
                cmd,
                cwd=str(repo_root),
                stderr=subprocess.DEVNULL,
            )
            s = out.decode("utf-8", errors="replace").strip()
            if s:
                return s
        except Exception:
            continue
    return None


def _normalize_release(s: str) -> str:
    """
    Normalize common tag strings into a PEP 440-ish release string.

    Examples:
      - v1.2.3 -> 1.2.3
      - 1.2.3-4-g<sha> -> 1.2.3
      - 1.2.3rc1 -> 1.2.3rc1

    :param s: Raw version string from package metadata or git.
    :type s: str
    :return: Normalized release string.
    :rtype: str
    """
    s = s.strip()
    if s.startswith("v"):
        s = s[1:]

    m = re.match(r"^(\d+\.\d+\.\d+(?:[a-zA-Z0-9\.]*)?)", s)
    if m:
        return m.group(1)
    return s


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

try:
    release = _get_version("synkit")
except PackageNotFoundError:
    git_rel = _git_describe(REPO_ROOT)
    release = _normalize_release(git_rel) if git_rel else "1.1.1"
else:
    release = _normalize_release(release)

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

exclude_patterns: list[str] = []

autosectionlabel_prefix_document = True
autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
}

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
        # Header
        "navbar_start": ["navbar-logo"],
        # IMPORTANT: keep navbar-nav so the theme can compute the active section
        # and the left sidebar "sidebar-nav-bs" can show your toctree subtree.
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
        # Navigation / TOC behavior
        "show_nav_level": 2,
        "navigation_depth": 4,
        "show_toc_level": 3,
        "search_bar_text": "Search the docs…",
        "use_edit_page_button": True,
        # RIGHT sidebar
        "secondary_sidebar_items": ["page-toc"],
        "footer_start": ["copyright"],
    }

    # LEFT sidebar: toctree navigation + search field
    # NOTE: Do NOT create doc/_templates/sidebar-nav-bs.html (would override theme).
    html_sidebars = {
        "**": [
            # "search-field.html",
            "sidebar-globaltoc.html",  # <-- our global tree
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
