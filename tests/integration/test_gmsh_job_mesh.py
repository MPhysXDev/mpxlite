# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""The job ``periodic_cell_mesh``, on cells made conformal by ``periodic_cell_geometry``.

Each test is a user script: it builds a cell with ``gmsh`` (millimetres or
metres), names its parts with physical groups, writes an XAO file, runs the
geometry job and then the mesh job with the parameters a plug-in derives
(bulk sizes given as numbers: a tenth of the wavelength at 8 GHz; the edges
of the named surfaces refined). Every mesh is checked: periodic facet order,
symmetric numbering of every operation, the symmetry gate.

Cells: a patch cell (and its finer-edge mesh), two patches with a strip in a
narrow gap (and the classes that let the mirror exchange the patches), the
feature sizes of the gap, a uniform slab, a cell with no symmetry, a cell
whose group the mesher does not support (cm), an anisotropy that restricts
the operations, an oblique cell, an open bottom (a layer below), a tall
layer, a vertical strip.
"""

from __future__ import annotations

import math
from pathlib import Path

import gmsh
import numpy as np
import pytest

import mpxlite
from mpxlite.cells import check_mesh_topology
from tests.integration.gmsh_cells import (
    AIR,
    FR4,
    MM,
    bulk,
    check_mesh,
    geometry_job,
    mesh_job,
    mesh_parameters,
    name,
    polygon,
    write_xao,
)

_PERIODIC = {"periodic_xm", "periodic_xp", "periodic_ym", "periodic_yp"}


def _conformal(tmp_path: Path, build, **parameters) -> Path:
    xao = write_xao(tmp_path / "user.xao", build)
    return geometry_job(tmp_path, xao, **parameters).outputs["geometry"]


def _patch_cell(period: float = 10 * MM, side: float = 9 * MM, thickness: float = 1.6 * MM):
    def build() -> None:
        occ = gmsh.model.occ
        margin = (period - side) / 2
        name(
            {"substrate": occ.addBox(0, 0, 0, period, period, thickness)},
            {"ground": occ.addRectangle(0, 0, 0, period, period),
             "patch": occ.addRectangle(margin, margin, thickness, side, side)},
        )  # fmt: skip

    return build


def test_patch_cell(tmp_path: Path) -> None:
    xao = _conformal(tmp_path, _patch_cell())
    params = mesh_parameters({"substrate": FR4}, ["ground", "patch"])
    mesh = mesh_job(tmp_path, xao, **params)
    check_mesh(mesh)
    rep = mesh.report
    assert rep["symmetry"]["name"] == "p4m"
    assert rep["layers"] == {"top": ["_top_face"], "above": ["_layer_above"]}
    assert set(rep["physical"]) >= _PERIODIC
    for part in ("substrate", "ground", "patch", "_top_face", "_layer_above"):
        assert part in rep["physical"]
    assert rep["mesh"]["nodes"] > 0 and rep["mesh"]["tetrahedra"] > rep["mesh"]["unit_tetrahedra"]
    # the finer edge mesh (factor 0.5 on the edge sizes)
    fine = mesh_job(
        tmp_path,
        xao,
        **mesh_parameters(
            {"substrate": FR4}, ["ground", "patch"], factor=0.5, output="cell_fine.msh"
        ),
    )
    check_mesh(fine)
    assert fine.outputs["mesh"].name == "cell_fine.msh"
    assert fine.report["mesh"]["tetrahedra"] > rep["mesh"]["tetrahedra"]
    assert fine.report["sizes"]["edge_min"] < rep["sizes"]["edge_min"]


def _two_patches() -> None:
    occ = gmsh.model.occ
    gap = 0.2 * MM
    x0 = 4 * MM - gap / 2
    name(
        {"substrate": occ.addBox(0, 0, 0, 8 * MM, 5 * MM, 1 * MM)},
        {"ground": occ.addRectangle(0, 0, 0, 8 * MM, 5 * MM),
         "left": occ.addRectangle(0.5 * MM, 1 * MM, 1 * MM, x0 - 0.5 * MM, 3 * MM),
         "right": occ.addRectangle(x0 + gap, 1 * MM, 1 * MM, x0 - 0.5 * MM, 3 * MM),
         "switch": occ.addRectangle(x0, 2 * MM, 1 * MM, gap, 1 * MM)},
    )  # fmt: skip


def test_strip_in_narrow_gap(tmp_path: Path) -> None:
    geo = geometry_job(
        tmp_path,
        write_xao(tmp_path / "user.xao", _two_patches),
        strips=[{"name": "switch", "ends": ["ground", "left", "right"]}],
    )
    shape = geo.report["strips"]["switch"]
    assert shape["direction"] == pytest.approx((1.0, 0.0, 0.0))
    assert shape["width"] == pytest.approx(1 * MM, rel=1e-6)
    assert shape["length"] == pytest.approx(0.2 * MM, rel=1e-6)
    sizes = {"substrate": bulk(3.0, loss_tangent=0.01)}
    refined = ["ground", "left", "right", "switch"]
    # the classes of a plug-in: each named patch its own class, the strips one class
    symmetry = {"classes": {"switch": "strip"}, "directions": {"switch": shape["direction"]}}
    mesh = mesh_job(tmp_path, geo.outputs["geometry"],
                    **mesh_parameters(sizes, refined, symmetry=symmetry))  # fmt: skip
    check_mesh(mesh)
    # the mirror x would exchange the two named patches: only the mirror y remains
    assert mesh.report["symmetry"]["name"] == "pm"
    assert "switch" in mesh.report["physical"]
    # the patches given one class: the mirror x exchanges them
    shared = {**symmetry, "classes": {"switch": "strip", "left": "patch", "right": "patch"}}
    both = mesh_job(tmp_path, geo.outputs["geometry"],
                    **mesh_parameters(sizes, refined, symmetry=shared, output="b.msh"))  # fmt: skip
    check_mesh(both)
    assert both.report["symmetry"]["name"] == "pmm"
    mx = [o for o in both.report["symmetry"]["operations"] if o["permutation"].get("left")
          == "right"]  # fmt: skip
    assert mx and all(o["direction_signs"].get("switch") in (-1, 1) for o in mx)


def test_feature_sizes_of_the_gap(tmp_path: Path) -> None:
    """The smallest feature is the gap: the smallest edge size is a sixth of it."""
    xao = _conformal(tmp_path, _two_patches)
    sizes = {"substrate": bulk(3.0, loss_tangent=0.01)}
    mesh = mesh_job(tmp_path, xao, **mesh_parameters(sizes, ["left", "right", "switch"]))
    rep = mesh.report["sizes"]
    # the switch width (1 mm) appears too, the gap (0.2 mm) is the smallest
    assert rep["edge_min"] == pytest.approx(0.2 * MM / 6, rel=1e-9)
    assert rep["refined_curves"] > 0 and rep["edge_max"] > rep["edge_min"]


def _slab() -> None:
    occ = gmsh.model.occ
    name(
        {"substrate": occ.addBox(0, 0, 0, 6 * MM, 6 * MM, 1 * MM)},
        {"ground": occ.addRectangle(0, 0, 0, 6 * MM, 6 * MM)},
    )


def test_uniform_slab(tmp_path: Path) -> None:
    mesh = mesh_job(tmp_path, _conformal(tmp_path, _slab),
                    **mesh_parameters({"substrate": FR4}, ["ground"]))  # fmt: skip
    check_mesh(mesh)
    assert mesh.report["symmetry"]["name"] == "p4m"
    assert any("continuous" in n for n in mesh.report["notes"])


def _irregular() -> None:
    occ = gmsh.model.occ
    name(
        {"substrate": occ.addBox(0, 0, 0, 10 * MM, 8 * MM, 1 * MM)},
        {"ground": occ.addRectangle(0, 0, 0, 10 * MM, 8 * MM),
         "patch": polygon([(2 * MM, 1.5 * MM), (8 * MM, 2 * MM), (4 * MM, 6.5 * MM)], 1 * MM)},
    )  # fmt: skip


def test_no_symmetry(tmp_path: Path) -> None:
    mesh = mesh_job(tmp_path, _conformal(tmp_path, _irregular),
                    **mesh_parameters({"substrate": FR4}, ["ground", "patch"]))  # fmt: skip
    check_mesh(mesh)
    assert mesh.report["symmetry"]["name"] == "p1"


def test_symmetry_mode_none(tmp_path: Path) -> None:
    """``symmetry.mode`` ``none``: plain periodicity on a symmetric cell."""
    params = mesh_parameters({"substrate": FR4}, ["ground", "patch"], symmetry={"mode": "none"})
    mesh = mesh_job(tmp_path, _conformal(tmp_path, _patch_cell()), **params)
    check_mesh(mesh)
    assert mesh.report["symmetry"]["name"] == "p1"
    assert mesh.report["detected"]["name"] == "p4m"


def _kite() -> None:
    occ = gmsh.model.occ
    name(
        {"substrate": occ.addBox(0, 0, 0, 8 * MM, 8 * MM, 1 * MM)},
        {"ground": occ.addRectangle(0, 0, 0, 8 * MM, 8 * MM),
         "patch": polygon([(2 * MM, 2 * MM), (5 * MM, 3 * MM), (5.5 * MM, 5.5 * MM),
                           (3 * MM, 5 * MM)], 1 * MM)},
    )  # fmt: skip


def test_unsupported_group_falls_back(tmp_path: Path) -> None:
    # a kite symmetric about the diagonal only: cm in a primitive cell -> p1
    mesh = mesh_job(tmp_path, _conformal(tmp_path, _kite),
                    **mesh_parameters({"substrate": FR4}, ["ground", "patch"]))  # fmt: skip
    check_mesh(mesh)
    rep = mesh.report
    assert rep["detected"]["name"] == "cm" and rep["symmetry"]["name"] == "p1"
    assert any("subgroup p1" in n for n in rep["notes"])


def test_anisotropy_restricts_the_group(tmp_path: Path) -> None:
    # permittivity (3, 4, 4): x and y differ, only the diagonal operations are allowed
    params = mesh_parameters(
        {"substrate": bulk(4.0)}, ["ground", "patch"], symmetry={"allowed": "diagonal"}
    )
    mesh = mesh_job(tmp_path, _conformal(tmp_path, _patch_cell()), **params)
    check_mesh(mesh)
    assert mesh.report["symmetry"]["name"] == "pmm"
    assert mesh.report["detected"]["name"] == "p4m"


def _oblique_disc() -> None:
    occ = gmsh.model.occ
    pu, pv, a = 8 * MM, 7 * MM, math.radians(70)
    corners = [(0, 0), (pu, 0), (pu + pv * math.cos(a), pv * math.sin(a)),
               (pv * math.cos(a), pv * math.sin(a))]  # fmt: skip
    ex = occ.extrude([(2, polygon(corners, 0.0))], 0, 0, 1 * MM)
    vol = next(t for d, t in ex if d == 3)
    centre = (np.array(corners[2]) / 2).tolist()
    name(
        {"substrate": vol},
        {"ground": polygon(corners, 0.0),
         "disc": occ.addDisk(centre[0], centre[1], 1 * MM, 2 * MM, 2 * MM)},
    )  # fmt: skip


def test_oblique_cell(tmp_path: Path) -> None:
    mesh = mesh_job(tmp_path, _conformal(tmp_path, _oblique_disc),
                    **mesh_parameters({"substrate": FR4}, ["ground", "disc"]))  # fmt: skip
    check_mesh(mesh)
    assert mesh.report["symmetry"]["name"] == "p2"
    assert mesh.report["cell"]["lattice"][1][0] == pytest.approx(7 * MM * math.cos(1.2217304764))


def _open_slab() -> None:
    occ = gmsh.model.occ
    name(
        {"substrate": occ.addBox(0, 0, 0, 6 * MM, 6 * MM, 1 * MM)},
        {"patch": occ.addRectangle(1 * MM, 1 * MM, 1 * MM, 4 * MM, 4 * MM)},
    )


def test_open_bottom(tmp_path: Path) -> None:
    """A layer below the cell.

    The plug-in's rule that a layer below is given if and only if the bottom
    is open (``if and only if``) is a rule of the plug-in, not of the job.
    """
    below = {"thickness": 3 * MM, "volume": "_layer_below", "face": "_bottom_face",
             "size": bulk(2.0, loss_tangent=0.01)}  # fmt: skip
    above = {"thickness": 4 * MM, "volume": "_layer_above", "face": "_top_face",
             "size": AIR}  # fmt: skip
    params = mesh_parameters({"substrate": FR4}, ["patch"], above=above, below=below)
    mesh = mesh_job(tmp_path, _conformal(tmp_path, _open_slab), **params)
    check_mesh(mesh)
    rep = mesh.report
    assert rep["layers"]["bottom"] == ["_bottom_face"]
    assert rep["layers"]["below"] == ["_layer_below"]
    assert "bare_bottom" not in rep["layers"]
    assert rep["symmetry"]["name"] == "p4m"
    assert rep["cell"]["z"] == pytest.approx([-3 * MM, 5 * MM])


def test_bare_bottom_named(tmp_path: Path) -> None:
    """Without a layer below, the bare faces of z = 0 get the name ``bare_bottom``."""
    mesh = mesh_job(tmp_path, _conformal(tmp_path, _open_slab),
                    **mesh_parameters({"substrate": FR4}, ["patch"]))  # fmt: skip
    assert mesh.report["layers"]["bare_bottom"] == ["_bottom"]
    assert "_bottom" in mesh.report["physical"]


# ----- with the geometry job in millimetres -----

_MM = {"scale": 1e-3, "unit": "mm", "output_unit": "m"}


def _example_patch_cell() -> None:
    kernel = gmsh.model.occ
    period, side, thickness = 10.0, 9.0, 1.6
    margin = (period - side) / 2
    name(
        {"substrate": kernel.addBox(0, 0, 0, period, period, thickness)},
        {"ground": kernel.addRectangle(0, 0, 0, period, period),
         "patch": kernel.addRectangle(margin, margin, thickness, side, side)},
    )  # fmt: skip


def test_example_patch_cell(tmp_path: Path) -> None:
    xao = _conformal(tmp_path, _example_patch_cell, **_MM)
    mesh = mesh_job(tmp_path, xao, **mesh_parameters({"substrate": FR4}, ["ground", "patch"]))
    check_mesh(mesh)
    assert mesh.report["symmetry"]["name"] == "p4m"
    assert {"substrate", "ground", "patch"} <= set(mesh.report["physical"])


def _longest_edge(path: Path, group: list[int]) -> float:
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        longest = 0.0
        for ent in gmsh.model.getEntitiesForPhysicalGroup(*group):
            _, _, conn = gmsh.model.mesh.getElements(2, int(ent))
            tri = np.asarray(conn[0], dtype=np.int64).reshape(-1, 3)
            nodes = [gmsh.model.mesh.getNode(int(n))[0] for n in tri.ravel()]
            xyz = np.array(nodes).reshape(-1, 3, 3)
            for i in range(3):
                e = np.linalg.norm(xyz[:, i] - xyz[:, (i + 1) % 3], axis=1)
                longest = max(longest, float(e.max()))
        return longest
    finally:
        gmsh.finalize()


def test_example_patch_cell_tall_layer(tmp_path: Path) -> None:
    # 34.4 mm of air on a 1.6 mm cell; the coarse detection mesh of this tall
    # prism once verified false shifts
    xao = _conformal(tmp_path, _example_patch_cell, **_MM)
    above = {"thickness": 0.0344, "volume": "_layer_above", "face": "_top_face",
             "size": AIR}  # fmt: skip
    params = mesh_parameters({"substrate": FR4}, ["ground", "patch"], above=above)
    mesh = mesh_job(tmp_path, xao, **params)
    check_mesh(mesh)
    assert mesh.report["symmetry"]["name"] == "p4m"
    # the top face is no coarser than lambda / 6 at 8 GHz in the air
    longest = _longest_edge(mesh.outputs["mesh"], mesh.report["physical"]["_top_face"])
    assert longest <= 10 * AIR / 6


def _via_cell(period: float, side: float, height: float, width: float):
    def build() -> None:
        occ = gmsh.model.occ
        margin = (period - side) / 2
        x0, y = (period - width) / 2, period / 2
        p = [occ.addPoint(x, y, z) for x, z in
             ((x0, 0.0), (x0 + width, 0.0), (x0 + width, height), (x0, height))]  # fmt: skip
        lines = [occ.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
        via = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        name(
            {"substrate": occ.addBox(0, 0, 0, period, period, height)},
            {"ground": occ.addRectangle(0, 0, 0, period, period),
             "patch": occ.addRectangle(margin, margin, height, side, side),
             "via": via},
        )  # fmt: skip

    return build


def test_vertical_strip(tmp_path: Path) -> None:
    xao = write_xao(tmp_path / "user.xao", _via_cell(10.0, 9.0, 1.6, 1.0))
    geo = geometry_job(tmp_path, xao, strips=[{"name": "via", "ends": ["ground", "patch"]}],
                       **_MM)  # fmt: skip
    shape = geo.report["strips"]["via"]
    assert shape["direction"] == [0.0, 0.0, 1.0]
    symmetry = {"classes": {"via": "strip"}, "directions": {"via": shape["direction"]}}
    params = mesh_parameters({"substrate": FR4}, ["ground", "patch", "via"], symmetry=symmetry)
    mesh = mesh_job(tmp_path, geo.outputs["geometry"], **params)
    check_mesh(mesh)
    assert check_mesh_topology(mesh.outputs["mesh"]).ok
    # the mirror across the via plane keeps the via; x -> -x reverses nothing
    assert mesh.report["symmetry"]["name"] == "pmm"


# ----- refusals -----


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"sizes": {"volumes": {}}}, "sizes.missing"),
        ({"sizes": {"volumes": {"substrate": FR4, "nothing": 1.0}}}, "sizes.unknown"),
        ({"sizes": {"volumes": {"substrate": -1.0}}}, "parameter.invalid"),
        ({"mesh": {"element_order": 2}}, "mesh.element_order"),
        ({"mesh": {"options": {"General.NumThreads": 8}}}, "parameter.invalid"),
        ({"symmetry": {"mode": "magic"}}, "parameter.invalid"),
        ({"mesh": {"output": "../x.msh"}}, "parameter.invalid"),
    ],
)
def test_invalid_parameters(tmp_path: Path, change: dict, code: str) -> None:
    xao = _conformal(tmp_path, _slab)
    params = mesh_parameters({"substrate": FR4}, ["ground"])
    for key, value in change.items():
        params[key] = {**params[key], **value}
    with pytest.raises(mpxlite.GmshJobError) as caught:
        mesh_job(tmp_path, xao, **params)
    assert (caught.value.kind, caught.value.code) == ("input", code), caught.value
