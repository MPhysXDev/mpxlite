# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""T0: lattice operations and the symmetry-preserving numbering (:mod:`mpxlite.cells.spacegroup`).

The meshes are the hand-built C4v wedge of :mod:`test_mesh_symmetric`
replicated by :func:`mpxlite.cells.symmetric.build_c4v_mesh` (no gmsh).
"""

from __future__ import annotations

import math
import shutil

import numpy as np
import pytest

from mpxlite.cells import UnsupportedMeshFormatError, check_periodic_facet_order
from mpxlite.cells.spacegroup import (
    LatticeOp,
    SymmetricNumberingError,
    check_symmetric_numbering,
    lattice_group,
    renumber_symmetric,
    symmetric_node_order,
)
from mpxlite.cells.symmetric import C4V_NAMES, build_c4v_mesh, c4v_lattice_ops
from tests.unit.cells.test_mesh_symmetric import PHYS, L, _wedge

SQUARE = ((L, 0.0), (0.0, L))


def test_lattice_op_round_trip_and_algebra():
    c = (L / 2, L / 2)
    c4 = LatticeOp.from_cartesian("C4", [[0, -1], [1, 0]], SQUARE, centre=c)
    g, t = c4.cartesian(SQUARE)
    assert np.allclose(g, [[0, -1], [1, 0]]) and np.allclose(g @ c + t, c)
    assert c4.det == 1
    ident = c4.compose(c4).compose(c4).compose(c4)
    assert np.array_equal(ident.matrix, np.eye(2)) and np.allclose(ident.shift % 1.0, 0.0)
    back = c4.compose(c4.inverse())
    assert np.array_equal(back.matrix, np.eye(2)) and np.allclose(back.shift, 0.0)
    u = np.array([[0.1, 0.2], [0.7, -0.3]])
    assert np.allclose(c4.inverse().apply(c4.apply(u)), u)
    with pytest.raises(ValueError, match="lattice"):
        LatticeOp.from_cartesian(
            "r30", [[math.cos(0.5), -math.sin(0.5)], [math.sin(0.5), math.cos(0.5)]], SQUARE
        )
    with pytest.raises(ValueError, match="unimodular"):
        LatticeOp("bad", [[2, 0], [0, 1]], [0, 0])


def test_lattice_group_orders():
    ops = {op.name: op for op in c4v_lattice_ops(L)}
    assert len(lattice_group([ops["C2"]])) == 2
    assert len(lattice_group([ops["C4"]])) == 4
    assert len(lattice_group([ops["C4"], ops["mx"]])) == 8
    assert [op.name for op in c4v_lattice_ops(L, names=("C2",))] == ["C2"]
    assert set(C4V_NAMES) == set(ops)
    # hexagonal lattice, C6 about the origin: not square-specific
    hexa = ((1.0, 0.0), (0.5, math.sqrt(3) / 2))
    a = math.pi / 3
    c6 = LatticeOp.from_cartesian(
        "C6", [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], hexa
    )
    assert len(lattice_group([c6])) == 6
    # a glide (mirror + half translation) closes modulo the lattice
    glide = LatticeOp("glide", [[-1, 0], [0, 1]], [0.0, 0.5])
    assert len(lattice_group([glide])) == 2


def test_order_is_orbit_constant():
    rng = np.random.default_rng(3)
    pts = np.column_stack([rng.uniform(-L / 4, 3 * L / 4, (40, 2)), rng.uniform(0, 1e-3, 40)])
    grp = lattice_group(c4v_lattice_ops(L, names=("C4",)))
    g, t = grp[1].cartesian(SQUARE)
    imgs = np.column_stack([pts[:, :2] @ g.T + t, pts[:, 2]])
    allp = np.vstack([pts, imgs])
    order = symmetric_node_order(allp, SQUARE, grp)
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    # a point and its image are adjacent (same orbit key): their ranks never
    # interleave with another orbit's
    pair = np.sort(np.column_stack([rank[:40], rank[40:]]), axis=1)
    assert np.all(pair[:, 1] - pair[:, 0] == 1)


def test_default_numbering_breaks_the_order_symmetric_one_keeps_it(tmp_path):
    coords, blocks = _wedge()
    plain = build_c4v_mesh(tmp_path / "plain.msh", coords, blocks, L, PHYS)
    ops = {op.name: op for op in c4v_lattice_ops(L)}
    rep = check_symmetric_numbering(plain, SQUARE, ops["C2"])
    assert rep.n_unmapped == 0 and rep.n_reordered > 0, rep.summary()
    sym = build_c4v_mesh(
        tmp_path / "sym.msh", coords, blocks, L, PHYS, numbering=c4v_lattice_ops(L)
    )
    assert check_periodic_facet_order(sym).ok
    for op in ops.values():
        rep = check_symmetric_numbering(sym, SQUARE, op)
        assert rep.ok and rep.n_elements > 0 and not rep.tag_map, rep.summary()
    # same mesh up to the numbering
    a, b = plain.read_text().splitlines(), sym.read_text().splitlines()
    assert len(a) == len(b)
    # renumbering the default mesh in place gives the same file
    again = tmp_path / "again.msh"
    shutil.copy(plain, again)
    report = renumber_symmetric(again, SQUARE, c4v_lattice_ops(L))
    assert report.ok
    assert all(check_symmetric_numbering(again, SQUARE, op).ok for op in ops.values())


def test_refuses_an_element_with_two_nodes_of_one_orbit(tmp_path):
    coords, blocks = _wedge()
    out = build_c4v_mesh(tmp_path / "m.msh", coords, blocks, L, PHYS)
    quarter = LatticeOp("t", np.eye(2), [0.25, 0.0])  # the wedge has edges of length L/4 along x
    with pytest.raises(SymmetricNumberingError, match="two nodes of one orbit"):
        renumber_symmetric(out, SQUARE, [quarter])


def test_check_flags_a_non_symmetric_operation(tmp_path):
    coords, blocks = _wedge()
    out = build_c4v_mesh(tmp_path / "m.msh", coords, blocks, L, PHYS, numbering=c4v_lattice_ops(L))
    off = LatticeOp.from_cartesian("C2'", [[-1, 0], [0, -1]], SQUARE, centre=(0.3 * L, 0.5 * L))
    rep = check_symmetric_numbering(out, SQUARE, off)
    assert not rep.ok and rep.n_unmapped > 0


def test_lattice_group_representatives_have_reduced_shifts():
    # mesh-m3: C4 about (L/2, L/2) has tau = (1, 0); its powers are not lattice-shifted
    c4 = LatticeOp.from_cartesian("C4", [[0, -1], [1, 0]], SQUARE, centre=(L / 2, L / 2))
    assert np.allclose(c4.shift, (1.0, 0.0))
    grp = lattice_group([c4])
    assert [op.key() for op in grp] == [
        op.key() for op in (grp[0], c4, c4.compose(c4), c4.compose(c4).compose(c4))
    ]
    for op in grp:
        assert np.all(op.shift >= -1e-9) and np.all(op.shift < 1.0 - 1e-9), op.shift
    full = lattice_group(c4v_lattice_ops(L))
    assert len(full) == 8 and all(np.all((op.shift >= -1e-9) & (op.shift < 1)) for op in full)


def test_renumber_symmetric_refuses_second_order_elements(tmp_path):
    # mesh-m2: the orbit-conflict check (and the gate) cover linear elements only
    coords, blocks = _wedge()
    out = build_c4v_mesh(tmp_path / "m.msh", coords, blocks, L, PHYS)
    lines = out.read_text(encoding="utf-8").split("\n")
    i = lines.index("$Elements")
    n = int(lines[i + 1])
    lines[i + 1] = str(n + 1)
    lines.insert(i + 2 + n, f"{n + 1} 11 2 11 11 " + " ".join(str(k) for k in range(1, 11)))
    out.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(UnsupportedMeshFormatError, match=r"element type\(s\) \[11\]"):
        renumber_symmetric(out, SQUARE, c4v_lattice_ops(L))


def test_check_symmetric_numbering_tolerates_a_rounding_boundary(tmp_path):
    # mesh-m4: a node moved within the 1e-9 tolerance across a rounding boundary of the key
    coords, blocks = _wedge()
    out = build_c4v_mesh(tmp_path / "m.msh", coords, blocks, L, PHYS, numbering=c4v_lattice_ops(L))
    lines = out.read_text(encoding="utf-8").split("\n")
    i = lines.index("$Nodes")
    s = 1e-9 * L
    k = i + 2 + int(lines[i + 1]) // 2
    num, x, y, z = lines[k].split()
    a = float(x) / s
    frac = a - round(a)
    lines[k] = f"{num} {(round(a) + (0.55 if frac >= 0 else -0.55)) * s!r} {y} {z}"
    out.write_text("\n".join(lines), encoding="utf-8")
    for op in c4v_lattice_ops(L):
        rep = check_symmetric_numbering(out, SQUARE, op)
        assert rep.ok, rep.summary()
