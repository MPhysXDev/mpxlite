# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the GNU General Public
# License (the file COPYING of the directory above) for more details.

"""The symmetric periodic mesh of a tagged cell (:mod:`mpxlite_gmsh.symmetric`).

Synthetic OpenCASCADE cells of the 12 rectangular / square groups ``p1``,
``p2``, ``pm``, ``pg``, ``cm``, ``pmm``, ``pmg``, ``pgg``, ``cmm``, ``p4``,
``p4m``, ``p4g`` (ITA settings; a generic motif orbit, tags ``patch_k``, and a
pad orbit on a symmetry element, tags ``pad_k``, both cut by the unit), the
window on the symmetry elements through the origin and off them, coarse,
through :func:`~mpxlite_gmsh.symmetric.build_symmetric_mesh`: every operation
maps the mesh onto itself with the node order preserved and moves the tags
as detected (checked on a generating set of the group, which suffices);
curved features (discs, a ring, a rounded patch) exchanged by ``C2`` are
detected as ``p2`` (not ``p1``), and a ``p1`` with a near symmetry is meshed as
``p1`` with a note; groups of dimension 0 or 1 are refused; the curve masters
of the paired faces are kept consistent across pairs.
"""

from __future__ import annotations

import math

import gmsh
import numpy as np
import pytest
from mpxlite_gmsh.symmetric import build_symmetric_mesh

from mpxlite.cells.autosym import MeshSizing, SymmetricMeshError
from mpxlite.cells.periodic import check_periodic_facet_order
from mpxlite.cells.spacegroup import LatticeOp, check_symmetric_numbering, lattice_group
from mpxlite.cells.symmetry_gate import require_mesh_symmetry

# ----- synthetic OCC cells (gmsh) ----------------------------------------------------------

LX, LY = 4e-3, 5.48e-3
I2 = np.eye(2)
MX = np.diag([-1.0, 1.0])
MY = np.diag([1.0, -1.0])
R90 = np.array([[0.0, -1.0], [1.0, 0.0]])
SYNTH = {
    "p1": (((LX, 0), (0, LY)), [], (0.5, 0.5)),
    "p2": (((LX, 0), (0, LY)), [(-I2, (0, 0))], (0.0, 0.0)),
    "pm": (((LX, 0), (0, LY)), [(MX, (0, 0))], (0.0, 0.3)),
    "pg": (((LX, 0), (0, LY)), [(MX, (0, LY / 2))], (0.0, 0.3)),
    "cm": (((LX, 0), (0, 1.6 * LX)), [(MX, (0, 0)), (I2, (LX / 2, 0.8 * LX))], (0.0, 0.3)),
    "pmm": (((LX, 0), (0, LY)), [(MX, (0, 0)), (MY, (0, 0))], (0.0, 0.0)),
    "pmg": (((LX, 0), (0, LY)), [(-I2, (0, 0)), (MX, (LX / 2, 0))], (0.25, 0.3)),
    "pgg": (((LX, 0), (0, LY)), [(-I2, (0, 0)), (MX, (LX / 2, LY / 2))], (0.0, 0.0)),
    "cmm": (((LX, 0), (0, 1.6 * LX)), [(MX, (0, 0)), (MY, (0, 0)), (I2, (LX / 2, 0.8 * LX))],
            (0.0, 0.0)),
    "p4": (((LX, 0), (0, LX)), [(R90, (0, 0))], (0.0, 0.0)),
    "p4m": (((LX, 0), (0, LX)), [(R90, (0, 0)), (MX, (0, 0))], (0.0, 0.0)),
    "p4g": (((LX, 0), (0, LX)), [(R90, (0, 0)), (MX, (LX / 2, LX / 2))], (0.0, 0.0)),
}  # fmt: skip
"""name: (lattice, generators (ITA settings), pad position on a symmetry element (lattice))."""
WINDOWS = {"on": (0.0, 0.0), "off": (0.13, -0.29)}
"""Window corners: on the symmetry elements through the origin, and off them."""
MOTIF = np.array([(0.0, 0.0), (1.0, 0.15), (0.8, 0.9), (0.1, 0.7)])


def _synthetic_cell(t, ops, pad_at, lo=(0.13, -0.29), zs=(-1.5e-3, 0.0, 2.5e-3)):
    """Slab + vacuum over the window ``[lo, lo + 1)``; motif and pad orbits on z = 0."""
    occ = gmsh.model.occ
    zp = zs[1]
    lx, ly = t[0][0], t[1][1]
    ox, oy = np.asarray(lo) @ t
    slab, vac, rect = _cell_boxes(t, lo, zs)
    tinv = np.linalg.inv(t)

    def poly(xy):
        p = [occ.addPoint(x, y, zp) for x, y in xy]
        lines = [occ.addLine(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]
        return occ.addPlaneSurface([occ.addCurveLoop(lines)])

    def spread(c):
        img = [op.apply(np.asarray(c)) for op in ops]
        return min((np.linalg.norm(((a - b) - np.rint(a - b)) @ t)
                    for i, a in enumerate(img) for b in img[:i]), default=9.0)  # fmt: skip

    scale = math.sqrt(lx * ly)
    cmot = max([(0.31, 0.23), (0.37, 0.13), (0.17, 0.09), (0.41, 0.29)], key=spread)
    base = np.array(cmot) @ t + 0.1 * scale * (MOTIF - MOTIF.mean(axis=0))
    pad = 0.06 * scale * np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)]) + np.asarray(pad_at) @ t
    shapes = []
    for cls, shape in (("patch", base), ("pad", pad)):
        seen: list[np.ndarray] = []
        for op in ops:
            g, tr = op.cartesian(t)
            img = shape @ g.T + tr
            c = img.mean(axis=0) @ tinv
            if any(np.max(np.abs((c - s) - np.rint(c - s))) < 1e-9 for s in seen):
                continue
            seen.append(c)
            img = img - np.floor(img.mean(axis=0) @ tinv - np.asarray(lo) + 1e-9) @ t
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    xy = img + np.array([i, j]) @ t
                    if (xy[:, 0].max() > ox and xy[:, 0].min() < ox + lx
                            and xy[:, 1].max() > oy and xy[:, 1].min() < oy + ly):  # fmt: skip
                        shapes.append((f"{cls}_{len(seen) - 1}", poly(xy)))
    _fragment_and_tag(t, lo, zs, slab, vac, rect, shapes)


def _cell_boxes(t, lo, zs):
    """Slab, vacuum and the z = 0 rectangle over the window ``[lo, lo + 1)``."""
    occ = gmsh.model.occ
    z0, zp, z1 = zs
    lx, ly = t[0][0], t[1][1]
    ox, oy = np.asarray(lo) @ t
    return (occ.addBox(ox, oy, z0, lx, ly, zp - z0), occ.addBox(ox, oy, zp, lx, ly, z1 - zp),
            occ.addRectangle(ox, oy, zp, lx, ly))  # fmt: skip


def _fragment_and_tag(t, lo, zs, slab, vac, rect, shapes):
    """Fragment the z = 0 ``shapes`` (name, surface) into the boxes; physical groups."""
    occ = gmsh.model.occ
    z0, _, z1 = zs
    lx, ly = t[0][0], t[1][1]
    ox, oy = np.asarray(lo) @ t
    _, omap = occ.fragment([(3, slab), (3, vac), (2, rect)], [(2, s) for _, s in shapes])
    occ.synchronize()
    inside = {s for _, s in omap[2]}
    groups: dict[str, set[int]] = {}
    drop = []
    for n, (name, _) in enumerate(shapes):
        for _, c in omap[3 + n]:
            (groups.setdefault(name, set()).add(c) if c in inside else drop.append((2, c)))
    occ.remove(drop, recursive=True)
    occ.synchronize()
    add = gmsh.model.addPhysicalGroup
    add(3, [s for _, s in omap[0]], 11, "slab")
    add(3, [s for _, s in omap[1]], 12, "vacuum")
    add(2, sorted(inside - set().union(*groups.values())), 2, "bare")
    for k, (name, ss) in enumerate(sorted(groups.items())):
        add(2, sorted(ss), 1000 + k, name)
    faces: dict[str, list[int]] = {}
    for _, s in gmsh.model.getEntities(2):
        pts = gmsh.model.getBoundary([(2, s)], combined=False, oriented=False, recursive=True)
        xyz = np.array([gmsh.model.getValue(0, abs(p), []) for _, p in pts])
        for name, axis, value in (("ground", 2, z0), ("top", 2, z1), ("periodic_xm", 0, ox),
                                  ("periodic_xp", 0, ox + lx), ("periodic_ym", 1, oy),
                                  ("periodic_yp", 1, oy + ly)):  # fmt: skip
            if np.all(np.abs(xyz[:, axis] - value) < 1e-12):
                faces.setdefault(name, []).append(s)
    for tag, name in zip((101, 103, 301, 302, 303, 304), ("ground", "top", "periodic_xm",
                         "periodic_xp", "periodic_ym", "periodic_yp"), strict=True):  # fmt: skip
        add(2, faces[name], tag, name)


def _generators(ops):
    """Indices of a generating set (products of order-preserving maps preserve the order)."""
    keys = {ops[0].key()}
    gens: list[int] = []
    for k, op in enumerate(ops[1:], start=1):
        if op.key() not in keys:
            gens.append(k)
            keys = {o.key() for o in lattice_group([ops[i] for i in gens])}
    return gens


def _check_group_invariance(rep, t):
    tags = {tg for _, tg in rep.physical.values()}
    sym = rep.symmetry
    for k in _generators(sym.ops):
        op, perm = sym.ops[k], sym.permutations[k]
        r = check_symmetric_numbering(rep.path, t, op)
        assert r.ok, r.summary()
        want = {a: b for a, b in perm.items() if a != b and a in tags}
        assert {a: b for _, a, b in r.tag_map} == want, (op.name, r.summary())
    assert check_periodic_facet_order(rep.path).ok
    # the mandatory gate accepts the mesh with the detected operations and tag images
    maps = {o.name: p for o, p in zip(sym.ops, sym.permutations, strict=True)}
    gate = require_mesh_symmetry(rep.path, t, sym.ops, tag_maps=maps, what=sym.name)
    assert gate.group_order == len(sym.ops)


@pytest.mark.parametrize("window", list(WINDOWS))
@pytest.mark.parametrize("name", list(SYNTH))
def test_synthetic_cells(name, window, tmp_path):

    lattice, gens, pad_at = SYNTH[name]
    t = np.asarray(lattice, dtype=float)
    ops = lattice_group(
        LatticeOp.from_cartesian(f"g{k}", g, t, translation=tr) for k, (g, tr) in enumerate(gens)
    )
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        _synthetic_cell(t, ops, pad_at, lo=WINDOWS[window])
        sizing = MeshSizing({"slab": 1.5e-3, "vacuum": 2.0e-3}, 0.9e-3, 1.0e-3)
        rep = build_symmetric_mesh(tmp_path / f"{name}.msh", t, sizing)
    finally:
        gmsh.finalize()
    assert rep.symmetry.name == name and rep.unit.name == name, rep.symmetry.summary()
    assert rep.symmetry.order == len(ops)
    # edges joining two nodes of one orbit occur where paired faces meet at a 4-fold axis
    assert (rep.n_split > 0) == (name in ("p4", "p4g"))
    assert (rep.n_paired > 0) == (name not in ("pmm", "p4m"))  # units bounded by mirrors only
    assert {"pad_0", "patch_0", "periodic_xm", "periodic_yp", "slab"} <= set(rep.physical)
    _check_group_invariance(rep, t)


CURVED_L = 4e-3
CURVED_AT = ((0.9e-3, 0.4e-3), (-0.9e-3, -0.4e-3))
"""Two feature sites exchanged by C2 about the centre of the window [-L/2, L/2)^2."""


def _curved_cell(shape: str, r: float) -> None:
    """Two equal curved features built at the two sites (gmsh discretises them as translates)."""
    occ = gmsh.model.occ
    t = np.array([[CURVED_L, 0.0], [0.0, CURVED_L]])
    lo, zs = (-0.5, -0.5), (-1.0e-3, 0.0, 2.0e-3)
    slab, vac, rect = _cell_boxes(t, lo, zs)
    shapes = []
    for k, (x, y) in enumerate(CURVED_AT):
        if shape == "disk":
            s = occ.addDisk(x, y, 0.0, r, r)
        elif shape == "ring":
            (_, s), = occ.cut([(2, occ.addDisk(x, y, 0.0, r, r))],
                              [(2, occ.addDisk(x, y, 0.0, 0.5 * r, 0.5 * r))])[0]  # fmt: skip
        else:  # rounded-corner patch
            s = occ.addRectangle(x - r, y - 0.6 * r, 0.0, 2 * r, 1.2 * r, roundedRadius=0.3 * r)
        shapes.append((f"feat_{k}", s))
    _fragment_and_tag(t, lo, zs, slab, vac, rect, shapes)


@pytest.mark.parametrize(
    ("shape", "r"),
    [("disk", 0.45e-3), ("disk", 0.55e-3), ("disk", 0.7e-3), ("ring", 0.6e-3), ("rounded", 0.5e-3)],
)
def test_curved_features_detected(shape, r, tmp_path):
    # mesh-1: on a coarse throw-away mesh two discs exchanged by C2 were detected as p1

    t = np.array([[CURVED_L, 0.0], [0.0, CURVED_L]])
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        _curved_cell(shape, r)
        sizing = MeshSizing({"slab": 1.5e-3, "vacuum": 2.0e-3}, 0.9e-3, 1.0e-3)
        rep = build_symmetric_mesh(tmp_path / "curved.msh", t, sizing)
    finally:
        gmsh.finalize()
    sym = rep.symmetry
    assert sym.name == "p2" and rep.unit.name == "p2", sym.summary()
    assert sym.tol < 1e-3 * CURVED_L
    f0, f1 = rep.physical["feat_0"][1], rep.physical["feat_1"][1]
    c2 = sym.permutations[1]
    assert c2[f0] == f1 and c2[f1] == f0
    _check_group_invariance(rep, t)


def test_trivial_group_with_near_symmetries_is_reported(tmp_path):
    # a disc 1 % larger than its C2 partner: p1 with a near C2 -> a p1 mesh and a note

    t = np.array([[CURVED_L, 0.0], [0.0, CURVED_L]])
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        lo, zs = (-0.5, -0.5), (-1.0e-3, 0.0, 2.0e-3)
        slab, vac, rect = _cell_boxes(t, lo, zs)
        (x0, y0), (x1, y1) = CURVED_AT
        shapes = [("feat_0", occ.addDisk(x0, y0, 0.0, 0.5e-3, 0.5e-3)),
                  ("feat_1", occ.addDisk(x1, y1, 0.0, 0.502e-3, 0.502e-3))]  # fmt: skip
        _fragment_and_tag(t, lo, zs, slab, vac, rect, shapes)
        sizing = MeshSizing({"slab": 1.5e-3, "vacuum": 2.0e-3}, 0.9e-3, 1.0e-3)
        rep = build_symmetric_mesh(tmp_path / "near.msh", t, sizing)
        assert rep.symmetry.name == "p1"
        assert any("plain periodic (p1)" in n for n in rep.notes)
    finally:
        gmsh.finalize()


def test_refuses_lower_dimensional_groups(tmp_path):

    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        box = gmsh.model.occ.addBox(0, 0, 0, 1e-3, 1e-3, 1e-3)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [box], 1, "vol")
        gmsh.model.addPhysicalGroup(1, [1], 2, "wire")
        with pytest.raises(SymmetricMeshError, match="dimension 1"):
            build_symmetric_mesh(tmp_path / "x.msh", ((1e-3, 0), (0, 1e-3)),
                                 MeshSizing({"vol": 5e-4}, 5e-4, 1e-3))  # fmt: skip
    finally:
        gmsh.finalize()


def test_pair_faces_curve_masters_across_pairs(monkeypatch):
    # mesh-m6: a curve given its master by two pairs of faces
    from types import SimpleNamespace

    from mpxlite_gmsh import symmetric as autosym

    def shift(dx, dy):
        a = np.eye(4)
        a[:2, 3] = dx, dy
        return a

    seg = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    at = {1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (1.0, -1.0, 0.0)}
    monkeypatch.setattr(autosym, "_curve_samples", lambda c, n: seg + np.array(at[c]))
    calls = []
    # the calls of _pair_faces go to a stand-in of Gmsh, which records setPeriodic
    mesh = SimpleNamespace(setPeriodic=lambda *a: calls.append(a))
    monkeypatch.setattr(autosym, "gmsh", SimpleNamespace(model=SimpleNamespace(mesh=mesh)))
    edges = [SimpleNamespace(partner=k, index=k + 1, op=f"t{k}") for k in (0, 2)]
    unit = SimpleNamespace(pairs=lambda: edges)

    def run(*plans):
        it = iter(plans)
        monkeypatch.setattr(autosym, "_pair_plan", lambda *a: next(it))
        calls.clear()
        return autosym._pair_faces(unit, {}, 1e-9)

    # the same master by two different maps: refused (it was overwritten silently)
    with pytest.raises(autosym.SymmetricMeshError, match="different maps"):
        run(([], shift(1.0, 0.0), [(2, 1)]), ([], shift(0.0, 1.0), [(2, 1)]))
    # the same map twice: consistent, nothing added
    assert run(([], shift(1.0, 0.0), [(2, 1)]), ([], shift(1.0, 0.0), [(2, 1)])) == 0
    assert calls == []
    # another root by the second pair: its root is linked to the expected one
    run(([], shift(1.0, 0.0), [(2, 1)]), ([], shift(0.0, 1.0), [(2, 3)]))
    ((dim, (sl,), (ma,), aff),) = calls
    assert (dim, sl, ma) == (1, 3, 1)
    assert np.allclose(np.reshape(aff, (4, 4)), shift(1.0, -1.0))
