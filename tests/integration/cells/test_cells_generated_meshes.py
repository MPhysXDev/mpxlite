# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""Periodic facet order on meshes built with Gmsh (:mod:`mpxlite.cells.periodic`).

The meshes are written by the user's scripts ``box.py`` and ``skew_box.py``
of this directory, which import Gmsh: a rectangular box (clean with the
numbering of Gmsh), a hexagonal box (renumbered by
:func:`~mpxlite.cells.periodic.renumber_periodic`; without it, the original
defect), and a box deliberately renumbered to break the order.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from mpxlite.cells import check_periodic_facet_order, renumber_periodic
from mpxlite.cells.periodic import _renumber_by
from tests.integration.cells import skew_box
from tests.integration.cells.box import BoxParams, generate_mesh
from tests.integration.cells.skew_box import SkewBoxParams

L = 30.0e-3


def test_box_mesh_is_clean(tmp_path: Path) -> None:
    path = generate_mesh(BoxParams(7.5e-3, 7.5e-3, L / 6, L / 6, L / 6), tmp_path / "b.msh")
    report = check_periodic_facet_order(path)
    assert report.has_periodic and report.n_links == 4 and report.n_facets > 0
    assert report.ok, report.summary()


def _hex_params() -> SkewBoxParams:
    a = L / 4
    return SkewBoxParams(a, a / 2, a * math.sqrt(3) / 2, L / 6, L / 6, L / 10)


def test_skew_box_regression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """skew_box renumbers through the core utility: clean; without it: the original defect."""
    clean = check_periodic_facet_order(skew_box.generate_mesh(_hex_params(), tmp_path / "s.msh"))
    assert clean.has_periodic and clean.ok, clean.summary()
    monkeypatch.setattr(skew_box, "_renumber_periodic", lambda *a, **k: None)
    raw = check_periodic_facet_order(skew_box.generate_mesh(_hex_params(), tmp_path / "raw.msh"))
    assert raw.n_bad > 0 and raw.n_unmapped == 0
    p = _hex_params()
    fixed = renumber_periodic(raw.path, (p.t1_x, 0.0), (p.t2_x, p.t2_y), output=tmp_path / "f.msh")
    assert fixed.ok and fixed.n_facets == raw.n_facets
    assert (tmp_path / "f.msh").read_bytes() == (tmp_path / "s.msh").read_bytes()


def test_intentionally_reversed_box_is_detected(tmp_path: Path) -> None:
    """Number the -x face by ascending y and the +x face by descending y."""
    path = generate_mesh(BoxParams(7.5e-3, 7.5e-3, L / 6, L / 6, L / 6), tmp_path / "b.msh")
    report = _renumber_by(path, lambda x, y, z: (y if x < 0 else -y, z, x))
    assert report.n_bad > 0 and report.n_unmapped == 0
    assert renumber_periodic(path, (7.5e-3, 0.0), (0.0, 7.5e-3)).ok
