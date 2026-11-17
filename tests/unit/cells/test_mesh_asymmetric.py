# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""T0: asymmetric units of the plane groups (:mod:`mpxlite.cells.asymmetric`), no gmsh.

The 12 rectangular / square groups, generated from Cartesian generators
about a generic centre: the images of the unit tile the cell exactly once
(sampled points), the edges are mirrors or paired by the expected
operations, the cut segments lie on preimages of the window lines; the
unsupported cases are refused.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mpxlite.cells.asymmetric import AsymmetricUnitError, asymmetric_unit
from mpxlite.cells.detect import wallpaper_name
from mpxlite.cells.spacegroup import LatticeOp, lattice_group

RECT = ((1.0, 0.0), (0.0, 1.37))
SQUARE = ((1.0, 0.0), (0.0, 1.0))
CRECT = ((1.0, 0.0), (0.0, 1.6))  # conventional cell of cm / cmm
I2 = np.eye(2)
MX = np.diag([-1.0, 1.0])
MY = np.diag([1.0, -1.0])
R90 = np.array([[0.0, -1.0], [1.0, 0.0]])
C = np.array([0.23, 0.41])  # generic origin of the generators (lattice coordinates)

# name: (lattice, generators (g, Cartesian translation about the origin), mirror/paired edges)
GROUPS = {
    "p1": (RECT, [], "pppp"),
    "p2": (RECT, [(-I2, (0, 0))], "pppppp"),
    "pm": (RECT, [(MX, (0, 0))], "pmpm"),
    "pg": (RECT, [(MX, (0, 0.685))], "pppppp"),
    "cm": (CRECT, [(MX, (0, 0)), (I2, (0.5, 0.8))], "pmpm"),
    "pmm": (RECT, [(MX, (0, 0)), (MY, (0, 0))], "mmmm"),
    "pmg": (RECT, [(-I2, (0, 0)), (MX, (0.5, 0))], None),
    "pgg": (RECT, [(-I2, (0, 0)), (MX, (0.5, 0.685))], "pppppp"),
    "cmm": (CRECT, [(MX, (0, 0)), (MY, (0, 0)), (I2, (0.5, 0.8))], None),
    "p4": (SQUARE, [(R90, (0, 0))], "pppp"),
    "p4m": (SQUARE, [(R90, (0, 0)), (MX, (0, 0))], "mmm"),
    "p4g": (SQUARE, [(R90, (0, 0)), (MX, (0.5, 0.5))], None),
}


def _ops(lattice, gens, centre=C):
    t = np.asarray(lattice, dtype=float)
    c = centre @ t
    return t, lattice_group(
        LatticeOp.from_cartesian(f"g{k}", g, t, translation=c - g @ c + np.asarray(tr))
        for k, (g, tr) in enumerate(gens)
    )


def _coverage(unit, t, ops, n=3000, seed=0):
    pts = np.random.default_rng(seed).random((n, 2)) @ t
    cnt = np.zeros(n, dtype=int)
    for op in ops:
        g, tr = op.cartesian(t)
        for i in range(-3, 4):
            for j in range(-3, 4):
                cnt += unit.contains((pts - tr - np.array([i, j]) @ t) @ g, -1e-12)
    return cnt


@pytest.mark.parametrize("name", list(GROUPS))
@pytest.mark.parametrize("lo", [(0.0, 0.0), (0.13, -0.29)])
def test_unit_tiles_the_cell(name, lo):
    lattice, gens, kinds = GROUPS[name]
    t, ops = _ops(lattice, gens)
    assert wallpaper_name(ops, t) == name
    unit = asymmetric_unit(ops, t, centre=C @ t, window_lo=lo)
    assert unit.name == name
    assert unit.area == pytest.approx(abs(np.linalg.det(t)) / len(ops), rel=1e-12)
    assert np.all(_coverage(unit, t, ops) == 1)
    got = "".join(e.kind[0] for e in unit.edges)
    if kinds is not None:
        assert sorted(got) == sorted(kinds)
    # paired edges: A(partner) = edge, end points included
    for e in unit.pairs():
        p = unit.edges[e.partner]
        ends = {tuple(np.round(e.g @ x + e.t, 9)) for x in (p.a, p.b)}
        assert ends == {tuple(np.round(x, 9)) for x in (e.a, e.b)}
    # mirrors: fixed point-wise by a reflection of the group (det -1)
    for e in unit.edges:
        if e.kind == "mirror":
            assert any(op.det == -1 for op in ops if op.name == e.op)
    # the unit's centroid lies in the window
    cen = unit.vertices.mean(axis=0) @ np.linalg.inv(t) - np.asarray(lo)
    assert np.all((cen >= 0) & (cen < 1))


@pytest.mark.parametrize("name", ["p2", "pg", "pmm", "p4", "p4m", "cmm"])
def test_cuts_lie_on_preimages_of_the_window_lines(name):
    lattice, gens, _ = GROUPS[name]
    t, ops = _ops(lattice, gens)
    lo = np.array([0.13, -0.29])
    unit = asymmetric_unit(ops, t, centre=C @ t, window_lo=lo)
    tinv = np.linalg.inv(t)
    for seg in unit.cuts:
        u = seg @ tinv
        ok = False
        for op in ops[1:]:
            w = op.apply(u) - lo
            ok |= any(np.allclose(w[:, i], np.rint(w[0, i]), atol=1e-9) for i in (0, 1))
        assert ok
        assert np.all(unit.contains(seg, 1e-12))


def test_p4m_unit_about_the_checkerboard_centre():
    big_l = 4e-3
    t = np.array([[big_l, 0.0], [0.0, big_l]])
    from mpxlite.cells.symmetric import c4v_lattice_ops

    unit = asymmetric_unit(c4v_lattice_ops(big_l), t, centre=(big_l / 2, big_l / 2),
                           window_lo=(-0.25, -0.25))  # fmt: skip
    assert unit.name == "p4m" and unit.translations == ((0, 0),)
    assert all(e.kind == "mirror" for e in unit.edges)
    # one corner on the 4-fold axis at the centre (L/2, L/2)
    assert np.min(np.linalg.norm(unit.vertices - big_l / 2, axis=1)) < 1e-15
    assert len(unit.cuts) == 2  # x = L/4, y = L/4: images of the window faces


def test_refused():
    t = np.array([[1.0, 0.0], [0.3, 1.1]])
    rot2 = -np.eye(2)
    for gens in ([], [LatticeOp.from_cartesian("C2", rot2, t)]):  # p1, p2: any lattice
        unit = asymmetric_unit(lattice_group(gens), t)
        assert abs(unit.area - abs(np.linalg.det(t)) / (1 + len(gens))) < 1e-12
    hexa = np.array([[1.0, 0.0], [0.5, math.sqrt(3) / 2]])
    a = math.radians(60)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    with pytest.raises(AsymmetricUnitError, match="non-rectangular lattice"):
        asymmetric_unit(lattice_group([LatticeOp.from_cartesian("C6", rot, hexa)]), hexa)
    # pmm on a 2 x 1 supercell: 8 operations modulo the supercell
    sup = np.array([[2.0, 0.0], [0.0, 1.0]])
    gens = [LatticeOp.from_cartesian(n, g, sup) for n, g in (("mx", MX), ("my", MY))]
    gens.append(LatticeOp("t", np.eye(2), (0.5, 0.0)))
    with pytest.raises(AsymmetricUnitError, match="supercell"):
        asymmetric_unit(lattice_group(gens), sup, name="pmm")
