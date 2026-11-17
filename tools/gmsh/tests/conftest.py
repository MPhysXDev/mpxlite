# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the GNU General Public
# License (the file COPYING of the directory above) for more details.

"""Tests of the Gmsh toolkit of the jobs: ``python -m pytest tools/gmsh/tests``.

They import ``gmsh`` (the module of the engine: ``import gmsh`` in the Python
of an mpxlite installation), NumPy, :mod:`mpxlite` and the toolkit beside
this directory. Without Gmsh they are skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import gmsh  # noqa: F401
except ImportError:  # pragma: no cover - an environment without the engine
    collect_ignore_glob = ["test_*.py"]


@pytest.fixture
def model():
    """Gmsh initialised with an empty model, one thread, no terminal output."""
    import gmsh

    gmsh.initialize(readConfigFiles=False, interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.model.add("test")
    try:
        yield gmsh
    finally:
        gmsh.finalize()
