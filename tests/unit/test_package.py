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


def test_public_api_export_list_is_empty_for_now() -> None:
    """A1 scaffold exports nothing yet; later commits populate ``__all__``."""
    assert mpxlite.__all__ == []
