# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""T0: C4v wedge replication (:mod:`mpxlite.cells.symmetric`) on a hand-built wedge mesh.

The wedge ``0 <= v <= u <= L/2`` (``(u, v) = (x, y) - c``, ``c = (L/2, L/2)``)
is split in the plane into the triangles O-A-F, A-C-G, A-G-F, F-G-H (A-F
on ``u = L/4`` and F-G on ``v = L/4``: the images of the window faces of
``[-L/4, 3L/4]^2``), extruded over two layers ``z in [-a, 0, b]``, and every
prism is cut into 3 tetrahedra by the global-index rule (conforming
diagonals). No gmsh.
"""

from __future__ import annotations

import numpy as np
import pytest

from mpxlite.cells import check_mesh_topology, check_periodic_facet_order
from mpxlite.cells.symmetric import (
    C4V,
    CUT,
    PERIODIC_NAMES,
    build_c4v_mesh,
    c4v_images,
    replicate_c4v,
)
from mpxlite.cells.topology import MeshTopologyError

L = 4e-3
Z = (-1e-3, 0.0, 1.5e-3)
PHYS = {
    "patch": (2, 1),
    "bare": (2, 2),
    "slab": (3, 11),
    "vacuum": (3, 12),
    "ground": (2, 101),
    "top": (2, 103),
    "periodic_xm": (2, 301),
    "periodic_xp": (2, 302),
    "periodic_ym": (2, 303),
    "periodic_yp": (2, 304),
}


def _wedge():
    h = L / 2
    uv = {"O": (0, 0), "A": (h / 2, 0), "C": (h, 0), "F": (h / 2, h / 2), "G": (h, h / 2),
          "H": (h, h)}  # fmt: skip
    names = list(uv)
    coords = np.array([(h + uv[n][0], h + uv[n][1], z) for z in Z for n in names])
    nid = {(n, k): k * len(names) + i for k in range(3) for i, n in enumerate(names)}
    tris = (("O", "A", "F"), ("A", "C", "G"), ("A", "G", "F"), ("F", "G", "H"))
    tets = {"slab": [], "vacuum": []}
    for layer, kind in ((0, "slab"), (1, "vacuum")):
        for tri in tris:
            i, j, k = sorted(tri, key=names.index)
            b = {n: nid[(n, layer)] for n in (i, j, k)}
            t = {n: nid[(n, layer + 1)] for n in (i, j, k)}
            for tet in ((b[i], b[j], b[k], t[k]), (b[i], b[j], t[j], t[k]),
                        (b[i], t[i], t[j], t[k])):  # fmt: skip
                p = coords[list(tet)]
                if np.linalg.det(p[1:] - p[0]) < 0:
                    tet = (tet[1], tet[0], tet[2], tet[3])
                tets[kind].append(tet)
    blocks = [(k, 4, np.array(v)) for k, v in tets.items()]
    plane = [tuple(nid[(n, 1)] for n in tri) for tri in tris]
    blocks.append(("patch", 2, np.array(plane[:1])))
    blocks.append(("bare", 2, np.array(plane[1:])))
    blocks.append(("ground", 2, np.array([tuple(nid[(n, 0)] for n in tri) for tri in tris])))
    blocks.append(("top", 2, np.array([tuple(nid[(n, 2)] for n in tri) for tri in tris])))
    # cut faces: the vertical quads on A-F and F-G, split as the tets split them
    faces = []
    for p_, q_ in (("A", "F"), ("F", "G")):
        i, j = sorted((p_, q_), key=names.index)
        for layer in (0, 1):
            bi, bj = nid[(i, layer)], nid[(j, layer)]
            ti, tj = nid[(i, layer + 1)], nid[(j, layer + 1)]
            faces += [(bi, bj, tj), (bi, ti, tj)]
    blocks.append((CUT, 2, np.array(faces)))
    return coords, blocks


def test_group_is_c4v():
    mats = [np.array(g) for g in C4V]
    assert len({m.tobytes() for m in mats}) == 8
    for a in mats:
        for b in mats:
            assert any(np.array_equal(a @ b, c) for c in mats)
    assert sum(np.linalg.det(m) < 0 for m in mats) == 4


def test_replicated_mesh_is_conforming_periodic_and_invariant(tmp_path):
    coords, blocks = _wedge()
    out = build_c4v_mesh(tmp_path / "c4v.msh", coords, blocks, L, PHYS)
    topo = check_mesh_topology(out)
    assert topo.ok and topo.checked_faces, topo.summary()
    assert topo.n_volumes == 8 * 24
    assert check_periodic_facet_order(out).ok
    nodes, elems = replicate_c4v(coords, blocks, L)
    assert np.allclose(nodes[:, :2].min(axis=0), -L / 4, atol=1e-15)
    assert np.allclose(nodes[:, :2].max(axis=0), 3 * L / 4, atol=1e-15)
    kinds = {k for k, _, _ in elems}
    assert set(PERIODIC_NAMES) <= kinds and CUT not in kinds
    # every tetrahedron positively oriented
    for kind, etype, conn in elems:
        if etype == 4:
            p = nodes[conn]
            assert np.all(np.linalg.det(p[:, 1:] - p[:, :1]) > 0), kind
    # the node set is invariant under C4v about c modulo the lattice
    # (folded: the faces x, y = 3L/4 coincide with -L/4; the identity image is the reference)
    imgs = [{tuple(np.round(x / (1e-9 * L)).astype(np.int64)) for x in img}
            for img in c4v_images(nodes, L)]  # fmt: skip
    assert len(imgs[0]) < len(nodes)  # the Bloch-face nodes folded onto their masters
    assert all(img == imgs[0] for img in imgs[1:])
    # Bloch faces: slave = master + L
    for m, s, axis in (("periodic_xm", "periodic_xp", 0), ("periodic_ym", "periodic_yp", 1)):
        xm = np.unique(np.concatenate([c for k, _, c in elems if k == m]))
        xs = np.unique(np.concatenate([c for k, _, c in elems if k == s]))
        assert np.allclose(nodes[xm, axis], -L / 4) and np.allclose(nodes[xs, axis], 3 * L / 4)
        assert len(xm) == len(xs)


def test_relabel_splits_blocks(tmp_path):
    coords, blocks = _wedge()

    def relabel(kind, xyz):
        if kind != "patch":
            return None
        cen = xyz.mean(axis=1)
        return np.where(cen[:, 0] > L / 2, "patch_r", "patch_l")

    phys = {**PHYS, "patch_l": (2, 5), "patch_r": (2, 6)}
    _, elems = replicate_c4v(coords, blocks, L, relabel=relabel)
    names = [k for k, _, _ in elems]
    assert "patch" not in names and "patch_l" in names and "patch_r" in names
    n_patch = sum(len(c) for k, _, c in elems if k.startswith("patch_"))
    assert n_patch == 8
    assert check_mesh_topology(build_c4v_mesh(tmp_path / "r.msh", coords, blocks, L, phys,
                                              relabel=relabel)).ok  # fmt: skip


def test_refuses_a_detached_face(tmp_path):
    coords, blocks = _wedge()
    # a top triangle shifted off the volume: detached after replication
    bad = coords.copy()
    x0 = L / 2
    extra = np.array([[x0 + 1e-4, x0 + 0.5e-4, 0.7e-3], [x0 + 2e-4, x0 + 0.5e-4, 0.7e-3],
                      [x0 + 2e-4, x0 + 1e-4, 0.7e-3]])  # fmt: skip
    bad = np.vstack([bad, extra])
    n = len(coords)
    blocks = [*blocks, ("top", 2, np.array([[n, n + 1, n + 2]]))]
    with pytest.raises(MeshTopologyError):
        build_c4v_mesh(tmp_path / "bad.msh", bad, blocks, L, PHYS)


def test_empty_physical_groups_are_not_written(tmp_path):
    # mesh-m12: a group without elements is not listed in $PhysicalNames
    coords, blocks = _wedge()
    out = build_c4v_mesh(tmp_path / "m.msh", coords, blocks, L, {**PHYS, "unused": (2, 77)})
    lines = out.read_text(encoding="utf-8").split("\n")
    i = lines.index("$PhysicalNames")
    names = lines[i + 2 : i + 2 + int(lines[i + 1])]
    assert len(names) == len(PHYS) and not any('"unused"' in n for n in names)
