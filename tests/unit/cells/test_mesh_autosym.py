# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""T0: generic replication helpers of the symmetric periodic mesh (:mod:`mpxlite.cells.symmetric`).

:func:`~mpxlite.cells.symmetric.merge_points` across bucket faces,
:func:`~mpxlite.cells.symmetric.replicate_group` against
:func:`~mpxlite.cells.symmetric.replicate_c4v` on the hand-built wedge of
``test_mesh_symmetric``, :func:`~mpxlite.cells.symmetric.split_orbit_edges`
on a tetrahedron across a 4-fold axis. The symmetric meshes of OpenCASCADE
cells are tested with the Gmsh job that builds them.
"""

from __future__ import annotations

import numpy as np
import pytest

from mpxlite.cells.spacegroup import LatticeOp, lattice_group
from mpxlite.cells.symmetric import (
    CUT,
    c4v_lattice_ops,
    merge_points,
    replicate_c4v,
    replicate_group,
    split_orbit_edges,
)
from tests.unit.cells.test_mesh_symmetric import L, _wedge

# ----- pure helpers ------------------------------------------------------------------------


def test_merge_points_across_bucket_faces():
    tol = 1e-9  # buckets of 1e-6
    base = np.array([[0.5e-6, 0.5e-6, 0.5e-6], [1.0, 2.0, 3.0], [7.5e-6, 0.5e-6, 0.5e-6]])
    face = [[3e-6 - 0.3 * tol, 0.5e-6, 0.5e-6], [3e-6 + 0.3 * tol, 0.5e-6, 0.5e-6]]
    pts = np.vstack([base, base + 0.4 * tol, face])
    uniq, inv = merge_points(pts, tol)
    assert len(uniq) == 4
    assert inv[0] == inv[3] and inv[1] == inv[4] and inv[2] == inv[5] and inv[6] == inv[7]
    with pytest.raises(RuntimeError, match="bucket size"):
        merge_points(np.array([[0.1e-6, 0, 0], [0.6e-6, 0, 0]]), tol)


def test_replicate_group_equals_replicate_c4v():
    coords, blocks = _wedge()
    vol = [b for b in blocks if b[0] != CUT]
    ref_nodes, ref = replicate_c4v(coords, vol, L)
    t = np.array([[L, 0.0], [0.0, L]])
    nodes, out = replicate_group(coords, vol, t, c4v_lattice_ops(L), window_lo=(-0.25, -0.25))
    assert len(nodes) == len(ref_nodes)
    key = lambda p: np.round(p / L, 9).tolist()  # noqa: E731
    assert sorted(map(tuple, key(nodes))) == sorted(map(tuple, key(ref_nodes)))
    for kind in ("slab", "vacuum", "patch", "bare", "ground", "top"):
        a = sorted(sorted(map(tuple, key(nodes[c]))) for k, _, cc in out if k == kind for c in cc)
        b = sorted(sorted(map(tuple, key(ref_nodes[c]))) for k, _, cc in ref if k == kind
                   for c in cc)  # fmt: skip
        assert a == b, kind


def test_split_orbit_edges_at_a_four_fold_axis():
    # unit [0, 1/2]^2 of p4 about the origin; a tetrahedron joins (h, 0, 0) and (0, h, 0)
    t = np.eye(2)
    ops = lattice_group([LatticeOp.from_cartesian("C4", [[0, -1], [1, 0]], t)])
    h = 0.1
    coords = np.array([[0, 0, 0], [h, 0, 0], [0, h, 0], [0.05, 0.05, 0.1]], dtype=float)
    blocks = [("vol", 4, np.array([[0, 1, 2, 3]])), ("face", 2, np.array([[0, 1, 2]]))]
    xyz, out, n = split_orbit_edges(coords, blocks, t, ops)
    assert n == 1 and len(xyz) == 5 and np.allclose(xyz[4], (h / 2, h / 2, 0))
    tets = next(c for k, _, c in out if k == "vol")
    assert len(tets) == 2 and not any({1, 2} <= set(r) for r in tets.tolist())
    vols = [np.linalg.det(xyz[r[1:]] - xyz[r[0]]) for r in tets]
    assert np.allclose(sum(vols), np.linalg.det(coords[1:] - coords[0]))
    assert len(next(c for k, _, c in out if k == "face")) == 2
    with pytest.raises(RuntimeError, match="boundary of the unit"):
        split_orbit_edges(coords, blocks, t, ops, interior=lambda p: np.zeros(len(p), bool))
