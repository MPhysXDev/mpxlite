# Copyright (c) 2025 Thierry Valet
# SPDX-License-Identifier: MIT

"""Smoke tests on the mpxlite package itself."""

from __future__ import annotations

import re

import mpxlite


def test_package_importable() -> None:
    """The package can be imported."""
    assert mpxlite is not None


def test_package_has_version() -> None:
    """``__version__`` is set and matches PEP 440 (rough check)."""
    version = mpxlite.__version__
    assert isinstance(version, str)
    assert re.match(r"^\d+\.\d+\.\d+", version), version


def test_top_level_exports() -> None:
    """The public API is exported at the package top level."""
    expected = {
        "GetDPError",
        "SolverLite",
        "SolverResult",
        "WRAPPER_PATH",
        "WorkflowData",
        "find_getdp",
        "parse_complex_scalar",
        "parse_complex_table",
        "parse_real_table",
    }
    assert set(mpxlite.__all__) == expected
    for name in expected:
        assert hasattr(mpxlite, name), f"{name} not exported"


def test_wrapper_path_is_bundled_and_executable() -> None:
    """The bundled launcher is reachable and has the executable bit set."""
    import os

    assert mpxlite.WRAPPER_PATH.is_file(), mpxlite.WRAPPER_PATH
    assert os.access(mpxlite.WRAPPER_PATH, os.X_OK)
