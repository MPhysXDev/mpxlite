# Copyright (c) 2025 Thierry Valet
# SPDX-License-Identifier: MIT

"""Sphinx configuration for the mpxlite documentation."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# -- Project information -----------------------------------------------------

project = "mpxlite"
author = "Thierry Valet"
copyright = "2025, Thierry Valet"

try:
    release = _pkg_version("mpxlite")
except PackageNotFoundError:
    release = "0.0.0"

version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
]

# -- AutoAPI -----------------------------------------------------------------

autoapi_type = "python"
autoapi_dirs = ["../src/mpxlite"]
autoapi_root = "_autoapi"
autoapi_keep_files = False
autoapi_add_toctree_entry = False
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    # imported-members deliberately OFF: re-exports from the top-level
    # __init__.py (e.g. ``mpxlite.SolverLite`` <- ``mpxlite.base.SolverLite``)
    # would otherwise be documented twice and trip Sphinx's hard
    # duplicate-object check.
]
autoapi_python_class_content = "class"

# 100% RST documentation. ``myst_parser`` is kept in the extension list as a
# safety net but no .md sources are expected in the tree.
source_suffix = {
    ".rst": "restructuredtext",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"mpxlite {release}"
html_static_path = ["_static"]

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
