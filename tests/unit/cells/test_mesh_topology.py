# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""T0: mesh conformity check (orphan nodes, triangles detached from the volume).

A tiny two-tetrahedron mesh with its boundary triangles is conforming; the
broken variants (a triangle that is no tet face, as a Bloch face left
outside the window by a faulty replication; a node of no tet) are counted
by :func:`mpxlite.cells.check_mesh_topology`; a non-tetrahedral volume skips
the face check and the report says why.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mpxlite.cells import check_mesh_topology

# two tets (1 2 3 4) and (2 3 4 5) sharing the face (2 3 4)
NODES = {1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0), 4: (0, 0, 1), 5: (1, 1, 1),
         6: (2, 0, 0), 7: (2, 1, 0), 8: (2, 0, 1)}  # fmt: skip
TETS = ((1, 2, 3, 4), (2, 3, 4, 5))
FACES = ((1, 2, 3), (1, 2, 4), (2, 3, 5))  # boundary faces of the tets


def _msh(path: Path, *, tets=TETS, tris=FACES, nodes=NODES, v41: bool = False) -> Path:
    if v41:
        lines = [
            "$MeshFormat",
            "4.1 0 8",
            "$EndMeshFormat",
            "$Nodes",
            f"1 {len(nodes)} 1 {max(nodes)}",
        ]
        lines += [f"3 1 0 {len(nodes)}"] + [str(k) for k in nodes]
        lines += [" ".join(map(str, xyz)) for xyz in nodes.values()]
        lines += ["$EndNodes", "$Elements", f"2 {len(tets) + len(tris)} 1 {len(tets) + len(tris)}"]
        lines += [f"3 1 4 {len(tets)}"] + [
            f"{i + 1} " + " ".join(map(str, t)) for i, t in enumerate(tets)
        ]
        lines += [f"2 1 2 {len(tris)}"]
        lines += [f"{len(tets) + i + 1} " + " ".join(map(str, t)) for i, t in enumerate(tris)]
        lines += ["$EndElements"]
    else:
        lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(len(nodes))]
        lines += [f"{k} {x} {y} {z}" for k, (x, y, z) in nodes.items()]
        rows = [f"4 2 11 11 {' '.join(map(str, t))}" for t in tets]
        rows += [f"2 2 101 101 {' '.join(map(str, t))}" for t in tris]
        lines += ["$EndNodes", "$Elements", str(len(rows))]
        lines += [f"{i + 1} {r}" for i, r in enumerate(rows)] + ["$EndElements"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


GOOD_NODES = {k: v for k, v in NODES.items() if k <= 5}


@pytest.mark.parametrize("v41", [False, True])
def test_conforming_and_broken_meshes(tmp_path: Path, v41: bool) -> None:
    good = check_mesh_topology(_msh(tmp_path / "good.msh", nodes=GOOD_NODES, v41=v41))
    assert good.ok and good.checked_faces and good.n_volumes == 2 and good.n_triangles == 3
    # a detached triangle (nodes 6 7 8 belong to no tet) and its orphan nodes
    bad = check_mesh_topology(_msh(tmp_path / "bad.msh", tris=(*FACES, (6, 7, 8)), v41=v41))
    assert not bad.ok
    assert bad.n_detached_triangles == 1 and bad.detached_triangles == ((6, 7, 8),)
    assert bad.n_orphan_nodes == 3 and set(bad.orphan_nodes) == {6, 7, 8}
    # a triangle on tet nodes that is not a tet face (1 3 5)
    off = check_mesh_topology(
        _msh(tmp_path / "off.msh", nodes=GOOD_NODES, tris=(*FACES, (1, 3, 5)), v41=v41)
    )
    assert off.n_detached_triangles == 1 and off.n_orphan_nodes == 0
    # an orphan node alone
    orphan = check_mesh_topology(_msh(tmp_path / "orphan.msh", v41=v41))
    assert orphan.n_orphan_nodes == 3 and orphan.n_detached_triangles == 0
    assert "not a face" in bad.summary()


def test_surface_only_mesh_is_not_checked(tmp_path: Path) -> None:
    rep = check_mesh_topology(_msh(tmp_path / "s.msh", tets=(), nodes=GOOD_NODES))
    assert rep.ok and not rep.checked_faces and rep.faces_skipped == "no volume element"
    assert "check skipped: no volume element" in rep.summary()


def test_skipped_face_check_is_reported(tmp_path: Path) -> None:
    # mesh-m1: a prism volume skips the face check; the report says why
    path = _msh(tmp_path / "case.msh", nodes=GOOD_NODES)
    text = path.read_text(encoding="utf-8").replace(
        "$EndElements", "9 6 2 12 12 1 2 3 4 5 1\n$EndElements"
    )
    path.write_text(text.replace("$Elements\n5\n", "$Elements\n6\n"), encoding="utf-8")
    rep = check_mesh_topology(path)
    assert rep.n_volumes == 3 and not rep.checked_faces
    assert rep.faces_skipped == "non-tetrahedral volume elements (gmsh types [6])"
    assert "detached-face check skipped" in rep.summary()
    good = check_mesh_topology(_msh(tmp_path / "g.msh", nodes=GOOD_NODES))
    assert good.checked_faces and good.faces_skipped == "" and "check skipped" not in good.summary()
