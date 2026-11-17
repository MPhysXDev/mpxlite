# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""T0: the symmetry gate (:mod:`mpxlite.cells.symmetry_gate`).

The meshes are the hand-built C4v wedge of :mod:`test_mesh_symmetric`
replicated by :func:`mpxlite.cells.symmetric.build_c4v_mesh` (no gmsh), and
hand-written triangle fans. Positive: every C4v operation on the symmetric
numbering. Negative: a perturbed node, a renumbered mesh, the periodic
numbering, a wrong / missing tag permutation, an off-centre operation, an
element with two nodes of one orbit; the opt-outs of the call sites warn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mpxlite.cells.spacegroup import LatticeOp
from mpxlite.cells.symmetric import C4V_NAMES, build_c4v_mesh, c4v_lattice_ops
from mpxlite.cells.symmetry_gate import (
    MeshSymmetryError,
    SymmetryGateSkipped,
    check_mesh_symmetry,
    element_tag_map,
    require_mesh_symmetry,
)
from tests.unit.cells.test_mesh_symmetric import PHYS, L, _wedge

SQUARE = ((L, 0.0), (0.0, L))
PHYS_LR = {**PHYS, "patch_l": (2, 5), "patch_r": (2, 6)}


def _split_patch(kind, xyz):
    if kind != "patch":
        return None
    return np.where(xyz.mean(axis=1)[:, 0] > L / 2, "patch_r", "patch_l")


@pytest.fixture(scope="module")
def meshes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("gate")
    coords, blocks = _wedge()
    ops = c4v_lattice_ops(L)
    return {
        "sym": build_c4v_mesh(root / "sym.msh", coords, blocks, L, PHYS, numbering=ops),
        "plain": build_c4v_mesh(root / "plain.msh", coords, blocks, L, PHYS),
        "lr": build_c4v_mesh(
            root / "lr.msh", coords, blocks, L, PHYS_LR, relabel=_split_patch, numbering=ops
        ),
    }


def _edit_nodes(src: Path, dst: Path, fn) -> Path:
    lines = src.read_text().splitlines()
    i = lines.index("$Nodes")
    n = int(lines[i + 1])
    rows = [line.split() for line in lines[i + 2 : i + 2 + n]]
    fn(rows)
    lines[i + 2 : i + 2 + n] = [" ".join(r) for r in rows]
    dst.write_text("\n".join(lines) + "\n")
    return dst


def test_symmetric_mesh_passes_every_operation(meshes):
    rep = require_mesh_symmetry(meshes["sym"], SQUARE, c4v_lattice_ops(L))
    assert rep.ok and rep.group_order == 8 and rep.n_orbit_conflicts == 0
    assert len(rep.operations) == 8
    for o in rep.operations:
        assert o.n_elements > 0 and o.max_node_residual < 1e-15 * L, o.summary()
        assert not o.tag_map
    assert "ok" in rep.summary()


def test_periodic_numbering_fails_on_the_node_order(meshes):
    ops = {o.name: o for o in c4v_lattice_ops(L)}
    rep = check_mesh_symmetry(meshes["plain"], SQUARE, [ops["C2"]])
    (c2,) = rep.operations
    assert not rep.ok and c2.n_nodes_unmapped == 0 and c2.n_elements_unmapped == 0
    assert c2.n_reordered > 0 and "another node order" in c2.example
    with pytest.raises(MeshSymmetryError, match=r"C2.*another node order") as err:
        require_mesh_symmetry(meshes["plain"], SQUARE, [ops["C2"]], what="test")
    assert err.value.report is not None and not err.value.report.ok
    assert str(err.value).startswith("test:")


def test_perturbed_node_is_reported_with_its_mismatch(meshes, tmp_path):
    delta = 1e-6 * L

    def move(rows):
        for r in rows:  # a node (c + (L/4, 0), z = 0) on the mirror y = c only
            x, y, z = float(r[1]) - L / 2, float(r[2]) - L / 2, float(r[3])
            if abs(x - L / 4) < 1e-12 and abs(y) < 1e-12 and z == 0.0:
                r[1] = repr(float(r[1]) + delta)
                return
        raise AssertionError("no such node")  # pragma: no cover

    bad = _edit_nodes(meshes["sym"], tmp_path / "pert.msh", move)
    rep = check_mesh_symmetry(bad, SQUARE, c4v_lattice_ops(L))
    assert not rep.ok
    # E and the mirror through the node still hold; the other 6 do not
    assert {o.op for o in rep.failing} == set(C4V_NAMES) - {"E", "my"}
    for o in rep.failing:
        assert o.n_nodes_unmapped >= 1
        assert 0.5 * delta < o.worst_node_mismatch <= 2.01 * delta, o.summary()
    with pytest.raises(MeshSymmetryError, match="node images are not nodes"):
        require_mesh_symmetry(bad, SQUARE, c4v_lattice_ops(L))


def test_renumbered_mesh_fails(meshes, tmp_path):
    def swap(rows):  # exchange the coordinates of two nodes: same set, other numbers
        a, b = 3, len(rows) // 2
        rows[a][1:], rows[b][1:] = rows[b][1:], rows[a][1:]

    lines = meshes["sym"].read_text().splitlines()
    i = lines.index("$Nodes")
    na, nb = lines[i + 2 + 3].split()[0], lines[i + 2 + int(lines[i + 1]) // 2].split()[0]
    tmp = _edit_nodes(meshes["sym"], tmp_path / "s0.msh", swap)
    lines = tmp.read_text().splitlines()
    j = lines.index("$Elements")
    for k in range(j + 2, j + 2 + int(lines[j + 1])):
        t = lines[k].split()
        nt = int(t[2])
        t[3 + nt :] = [nb if v == na else na if v == nb else v for v in t[3 + nt :]]
        lines[k] = " ".join(t)
    out = tmp_path / "swap.msh"
    out.write_text("\n".join(lines) + "\n")
    rep = check_mesh_symmetry(out, SQUARE, c4v_lattice_ops(L))
    assert not rep.ok
    assert all(o.n_nodes_unmapped == 0 and o.n_elements_unmapped == 0 for o in rep.operations)
    assert sum(o.n_reordered for o in rep.operations) > 0


def test_tag_permutation_must_be_declared(meshes):
    ops = {o.name: o for o in c4v_lattice_ops(L)}
    lr = meshes["lr"]
    # mx exchanges the two patch halves: refused undeclared, accepted declared
    rep = check_mesh_symmetry(lr, SQUARE, [ops["mx"], ops["my"]])
    (mx, my) = rep.operations
    assert my.ok and not mx.ok and mx.n_reordered == 0
    assert {(d, a) for d, a, _, _ in mx.tag_errors} == {(2, 5), (2, 6)}
    ok = check_mesh_symmetry(lr, SQUARE, [ops["mx"]], tag_maps={"mx": {5: 6, 6: 5}})
    assert ok.ok and ok.operations[0].tag_map == ((2, 5, 6), (2, 6, 5))
    # a wrong permutation (declared but not the mesh's)
    wrong = check_mesh_symmetry(lr, SQUARE, [ops["my"]], tag_maps={"my": {5: 6, 6: 5}})
    assert not wrong.ok
    # C4 splits each half between both tags: no permutation is consistent
    c4 = check_mesh_symmetry(lr, SQUARE, [ops["C4"]], tag_maps={"C4": {5: 6, 6: 5}})
    assert not c4.ok and c4.operations[0].tag_errors
    with pytest.raises(ValueError, match="unknown operations"):
        check_mesh_symmetry(lr, SQUARE, [ops["mx"]], tag_maps={"C4": {}})


def test_element_tag_map():
    assert element_tag_map([[1000], [1001]], [1, 0]) == {1000: (1001,), 1001: (1000,)}
    assert element_tag_map([[7, 8], [9]], [0, 1]) == {7: (7, 8), 8: (7, 8), 9: (9,)}
    with pytest.raises(ValueError):
        element_tag_map([[1]], [0, 0])


def test_off_centre_operation_fails(meshes):
    off = LatticeOp.from_cartesian("C2'", [[-1, 0], [0, -1]], SQUARE, centre=(0.3 * L, 0.5 * L))
    rep = check_mesh_symmetry(meshes["sym"], SQUARE, [off])
    (o,) = rep.operations
    assert not rep.ok and o.n_nodes_unmapped > 0 and o.worst_node_mismatch > 1e-3 * L


def _fan(path: Path, lx: float, ly: float) -> Path:
    """Triangles of a cell ``[0, lx] x [0, ly]``: the diamond of the edge midpoints, fanned
    about the centre, and the four corner triangles (MSH 2.2, no ``$Periodic``)."""
    c = (lx / 2, ly / 2)
    pts = [c, (lx / 2, 0), (lx, ly / 2), (lx / 2, ly), (0, ly / 2), (0, 0), (lx, 0), (lx, ly),
           (0, ly)]  # fmt: skip
    tris = [(1, 2, 3), (1, 3, 4), (1, 4, 5), (1, 5, 2), (2, 7, 3), (3, 8, 4), (4, 9, 5),
            (5, 6, 2)]  # fmt: skip
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(len(pts))]
    lines += [f"{k + 1} {x!r} {y!r} 0.0" for k, (x, y) in enumerate(pts)]
    lines += ["$EndNodes", "$Elements", str(len(tris))]
    lines += [f"{k + 1} 2 2 7 1 {a} {b} {c_}" for k, (a, b, c_) in enumerate(tris)]
    lines += ["$EndElements", ""]
    path.write_text("\n".join(lines))
    return path


def test_two_nodes_of_one_orbit_are_refused(tmp_path):
    fan = _fan(tmp_path / "fan.msh", L, L)
    ops = {o.name: o for o in c4v_lattice_ops(L)}
    # invariant under C4 (nodes, elements), but every triangle holds m and C4 m
    rep = check_mesh_symmetry(fan, SQUARE, [ops["C4"]])
    (c4,) = rep.operations
    assert c4.n_nodes_unmapped == 0 and c4.n_elements_unmapped == 0
    assert rep.group_order == 4 and rep.n_orbit_conflicts == 8 and not rep.ok
    with pytest.raises(MeshSymmetryError, match="two nodes of one orbit"):
        require_mesh_symmetry(fan, SQUARE, [ops["C4"]])
    # C2 alone has no such element
    assert check_mesh_symmetry(fan, SQUARE, [ops["C2"]]).n_orbit_conflicts == 0


def test_rectangular_lattice_mirrors(tmp_path):
    lx, ly = L, 0.6 * L
    rect = ((lx, 0.0), (0.0, ly))
    fan = _fan(tmp_path / "rect.msh", lx, ly)
    c = (lx / 2, ly / 2)
    mx = LatticeOp.from_cartesian("mx", [[-1, 0], [0, 1]], rect, centre=c)
    my = LatticeOp.from_cartesian("my", [[1, 0], [0, -1]], rect, centre=c)
    rep = check_mesh_symmetry(fan, rect, [mx, my])
    assert rep.group_order == 4 and rep.n_orbit_conflicts == 0
    assert all(o.n_nodes_unmapped == 0 and o.n_elements_unmapped == 0 for o in rep.operations)
    # a glide (mirror y, half a period along x) maps the nodes onto nodes but
    # the diamond onto the corner triangles: not a symmetry of this cell
    glide = LatticeOp("g", [[1, 0], [0, -1]], [0.5, 1.0])
    (g,) = check_mesh_symmetry(fan, rect, [glide]).operations
    assert g.n_nodes_unmapped == 0 and g.n_elements_unmapped > 0 and not g.ok


def test_opt_out_warns():
    from mpxlite.cells.symmetry_gate import warn_gate_skipped

    with pytest.warns(SymmetryGateSkipped, match="WITHOUT checking"):
        warn_gate_skipped("here")
