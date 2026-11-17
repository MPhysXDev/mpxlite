# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""The job ``periodic_cell_geometry``, driven by user scripts.

Each test builds a cell with the ``gmsh`` module, names its parts with
physical groups (the parts may overlap or touch: the job makes them
conform), writes an XAO file, and runs the job: accepted cells and their
report (a patch cell in millimetres, two patches bridged by a strip, a
vertical strip from the ground to a patch, oblique prisms, a cell read back
from its conformal file), and one cell per rule refusal, with its code.
"""

from __future__ import annotations

import math
from pathlib import Path

import gmsh
import numpy as np
import pytest

import mpxlite
from tests.integration.gmsh_cells import geometry_job, name, polygon, write_xao

_PATCH = {"period": 10.0, "patch_side": 9.0, "thickness": 1.6}
_MM = {"scale": 1e-3, "unit": "mm", "output_unit": "m"}


def _patch_cell(period: float, patch_side: float, thickness: float) -> None:
    """The patch cell of a getting-started script (millimetres)."""
    kernel = gmsh.model.occ
    margin = (period - patch_side) / 2
    substrate = kernel.addBox(0, 0, 0, period, period, thickness)
    ground = kernel.addRectangle(0, 0, 0, period, period)
    patch = kernel.addRectangle(margin, margin, thickness, patch_side, patch_side)
    name({"substrate": substrate}, {"ground": ground, "patch": patch})


def _run(tmp_path: Path, build, **parameters) -> mpxlite.GmshJobResult:
    xao = write_xao(tmp_path / "user.xao", build)
    return geometry_job(tmp_path, xao, **{**_MM, **parameters})


def _refused(tmp_path: Path, build, code: str, match: str, **parameters) -> mpxlite.GmshJobError:
    with pytest.raises(mpxlite.GmshJobError, match=match) as caught:
        _run(tmp_path, build, **parameters)
    assert caught.value.kind == "rule", caught.value
    assert caught.value.code == code, caught.value
    return caught.value


def _parts(report: dict, dim: int) -> dict[str, dict]:
    return {p["name"]: p for p in report["parts"] if p["dim"] == dim}


def _covered_by(face: dict, names: set[str]) -> float:
    """Fraction of the area of a face not covered by the surfaces ``names``."""
    bare = sum(p["area"] for p in face["pieces"] if not set(p["surfaces"]) & names)
    return bare / face["area"]


# --------------------------------------------------------------------------
# Geometry functions of the test cells (millimetres).


def _two_patches(gap_width: float = 2.0, element_length: float = 1.0) -> None:
    """Two patches on a grounded slab bridged by a strip."""
    occ = gmsh.model.occ
    slab = occ.addBox(0, 0, 0, 10, 10, 1.6)
    ground = occ.addRectangle(0, 0, 0, 10, 10)
    half = element_length / 2
    left = occ.addRectangle(1, 2, 1.6, 4 - half, 6)
    right = occ.addRectangle(5 + half, 2, 1.6, 4 - half, 6)
    element = occ.addRectangle(5 - half, 5 - gap_width / 2, 1.6, element_length, gap_width)
    name({"slab": slab}, {"ground": ground, "left": left, "right": right, "load": element})


def _vertical_element() -> None:
    """A strip standing from the ground to a patch."""
    occ = gmsh.model.occ
    slab = occ.addBox(0, 0, 0, 10, 10, 1.6)
    ground = occ.addRectangle(0, 0, 0, 10, 10)
    patch = occ.addRectangle(2, 2, 1.6, 6, 6)
    element = occ.addRectangle(4.5, 5, 0, 1, 1.6)
    occ.rotate([(2, element)], 4.5, 5, 0, 1, 0, 0, math.pi / 2)
    name({"slab": slab}, {"ground": ground, "patch": patch, "via": element})


def _oblique(pu: float = 10.0, pv: float = 8.0, angle: float = 60.0, height: float = 2.0):
    """A slab on a parallelogram footprint, with a central patch."""

    def build() -> None:
        occ = gmsh.model.occ
        a = math.radians(angle)
        corners = [
            (0, 0),
            (pu, 0),
            (pu + pv * math.cos(a), pv * math.sin(a)),
            (pv * math.cos(a), pv * math.sin(a)),
        ]
        base = polygon(corners, 0.0)
        out = occ.extrude([(2, base)], 0, 0, height)
        slab = next(t for d, t in out if d == 3)
        cx = (pu + pv * math.cos(a)) / 2
        cy = pv * math.sin(a) / 2
        patch = occ.addDisk(cx, cy, height, 2, 2)
        name({"slab": slab}, {"patch": patch})

    return build


# --------------------------------------------------------------------------
# Accepted cells.


def test_patch_cell_accepted_in_mm(tmp_path: Path) -> None:
    result = _run(tmp_path, lambda: _patch_cell(**_PATCH))
    rep = result.report
    cell = rep["cell"]
    assert cell["sides"][0] == pytest.approx(0.010, rel=1e-9)
    assert cell["sides"][1] == pytest.approx(0.010, rel=1e-9)
    assert cell["angle"] == pytest.approx(math.pi / 2, abs=1e-12)
    assert cell["height"] == pytest.approx(0.0016, rel=1e-9)
    assert set(_parts(rep, 3)) == {"substrate"}
    assert set(_parts(rep, 2)) == {"ground", "patch"}
    bottom, top = rep["faces"]["bottom"], rep["faces"]["top"]
    # the bottom face is the ground, exactly; the patch lies on the top face
    assert [p["surfaces"] for p in bottom["pieces"]] == [["ground"]]
    assert ["patch"] in [p["surfaces"] for p in top["pieces"]]
    assert _parts(rep, 2)["patch"]["on"]["top"] == pytest.approx(81e-6, rel=1e-9)
    assert len(rep["faces"]["sides"]) == 4
    (u, v) = cell["lattice"]
    assert u == pytest.approx((0.010, 0.0))
    assert v == pytest.approx((0.0, 0.010), abs=1e-15)
    # the geometry is written in metres
    assert _parts(rep, 3)["substrate"]["measure"] == pytest.approx(1.6e-7, rel=1e-6)
    assert result.outputs["geometry"].suffix == ".xao"
    assert rep["periodic"]["matched"] and len(rep["periodic"]["pairs"]) == 2


def test_two_patches_bridged_by_strip(tmp_path: Path) -> None:
    rep = _run(
        tmp_path, _two_patches, strips=[{"name": "load", "ends": ["ground", "left", "right"]}]
    ).report
    shape = rep["strips"]["load"]
    assert shape["direction"] == pytest.approx((1.0, 0.0, 0.0))
    assert shape["width"] == pytest.approx(2.0e-3, rel=1e-6)
    assert shape["length"] == pytest.approx(1.0e-3, rel=1e-6)
    assert shape["centre"] == pytest.approx((5e-3, 5e-3, 1.6e-3), rel=1e-6)
    assert shape["ends"] == ["left", "right"]
    assert shape["area"] == pytest.approx(2e-6, rel=1e-6)


def test_vertical_strip_from_ground_to_patch(tmp_path: Path) -> None:
    rep = _run(
        tmp_path, _vertical_element, strips=[{"name": "via", "ends": ["ground", "patch"]}]
    ).report
    shape = rep["strips"]["via"]
    assert shape["direction"] == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)
    assert shape["width"] == pytest.approx(1.0e-3, rel=1e-6)
    assert shape["length"] == pytest.approx(1.6e-3, rel=1e-6)
    assert shape["ends"] == ["ground", "patch"]


def test_oblique_prism_60_degrees(tmp_path: Path) -> None:
    rep = _run(tmp_path, _oblique()).report
    cell = rep["cell"]
    assert cell["sides"][0] == pytest.approx(0.010, rel=1e-9)
    assert cell["sides"][1] == pytest.approx(0.008, rel=1e-9)
    assert cell["angle"] == pytest.approx(math.pi / 3, rel=1e-9)
    assert cell["height"] == pytest.approx(0.002, rel=1e-9)
    assert not cell["rectangular"]
    assert len(rep["faces"]["bottom"]["pieces"]) == 1
    assert len(rep["faces"]["sides"]) == 4


def test_oblique_prism_120_degrees(tmp_path: Path) -> None:
    rep = _run(tmp_path, _oblique(angle=120.0)).report
    assert rep["cell"]["angle"] == pytest.approx(2 * math.pi / 3, rel=1e-9)


def test_xao_round_trip(tmp_path: Path) -> None:
    """The conformal file of the job, read again, gives the same cell."""
    reference = _run(tmp_path, lambda: _patch_cell(**_PATCH)).report
    again = geometry_job(
        tmp_path, _run(tmp_path, lambda: _patch_cell(**_PATCH)).outputs["geometry"]
    ).report
    for k in (0, 1):
        assert again["cell"]["sides"][k] == pytest.approx(reference["cell"]["sides"][k], rel=1e-12)
    assert again["cell"]["height"] == pytest.approx(reference["cell"]["height"], rel=1e-12)
    assert set(_parts(again, 3)) == set(_parts(reference, 3))
    assert set(_parts(again, 2)) == set(_parts(reference, 2))
    assert [p["surfaces"] for p in again["faces"]["bottom"]["pieces"]] == [["ground"]]


def test_read_geometry_on_open_model_in_metres(tmp_path: Path) -> None:
    """A cell in metres: the default scale 1."""

    def build() -> None:
        name({"air": gmsh.model.occ.addBox(0, 0, 0, 0.01, 0.02, 0.001)})

    xao = write_xao(tmp_path / "m.xao", build)
    rep = geometry_job(tmp_path, xao).report
    assert rep["cell"]["sides"] == pytest.approx((0.01, 0.02))
    assert rep["scale"] == 1.0


def test_scaled_geometry_keeps_planes_and_booleans(tmp_path: Path) -> None:
    """A geometry in mm, scaled to metres: planes stay planes, names stay.

    With a B-spline scaling, an intersection of the substrate with a box
    inside it was empty on a cell of this kind.
    """
    result = _run(tmp_path, lambda: _patch_cell(**_PATCH))
    assert result.report["cell"]["sides"][0] == pytest.approx(10e-3)
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(result.outputs["geometry"]))
        assert {gmsh.model.getType(2, s) for _, s in gmsh.model.getEntities(2)} == {"Plane"}
        names = {
            gmsh.model.getPhysicalName(d, t): list(gmsh.model.getEntitiesForPhysicalGroup(d, t))
            for d, t in gmsh.model.getPhysicalGroups()
        }
        assert sorted(names) == ["ground", "patch", "substrate"]
        occ = gmsh.model.occ
        copy = occ.copy([(3, int(names["substrate"][0]))])
        foot = np.array([[1, 1], [2, 1], [2, 2], [1, 2]]) * 1e-3
        base = polygon([tuple(p) for p in foot], -1e-3)
        box = next(t for d, t in occ.extrude([(2, base)], 0, 0, 1.1e-2) if d == 3)
        out, _ = occ.intersect(copy, [(3, box)])
        occ.synchronize()
        assert out and occ.getMass(*out[0]) == pytest.approx(1e-6 * 1.6e-3)
    finally:
        gmsh.finalize()


@pytest.mark.parametrize("ground", [True, False])
def test_real_geometry_bottom_cover(tmp_path: Path, ground: bool) -> None:
    """The pieces of the bottom face and the surfaces that cover them.

    The plug-in's rule on top of it (a closed bottom refuses a medium below;
    an open one is refused with the uncovered fraction) stays in the plug-in;
    the fraction it states (81 %) is the one read here.
    """

    def build() -> None:
        kernel = gmsh.model.occ
        period, side, thickness = 10.0, 9.0, 1.6
        margin = (period - side) / 2
        substrate = kernel.addBox(0, 0, 0, period, period, thickness)
        surfaces = {"patch": kernel.addRectangle(margin, margin, thickness, side, side)}
        if ground:
            surfaces["ground"] = kernel.addRectangle(0, 0, 0, period, period)
        else:  # a ground plane with a square hole of side patch_side
            plate = kernel.addRectangle(0, 0, 0, period, period)
            hole = kernel.addRectangle(margin, margin, 0, side, side)
            cut, _ = kernel.cut([(2, plate)], [(2, hole)])
            surfaces["ground"] = cut[0][1]
        name({"substrate": substrate}, surfaces)

    bottom = _run(tmp_path, build).report["faces"]["bottom"]
    bare = _covered_by(bottom, {"ground", "patch"})
    if ground:
        assert bare == 0.0
    else:
        assert f"{100.0 * bare:.3g}" == "81"


# --------------------------------------------------------------------------
# Refusals of the naming (the parts of the file).


def test_name_used_for_volume_and_surface(tmp_path: Path) -> None:
    def build() -> None:
        box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        face = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
        name({"x": box}, {"x": face})

    _refused(tmp_path, build, "names.twice", r"'x' is used twice")


def test_tag_given_two_names(tmp_path: Path) -> None:
    def build() -> None:
        box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        name({"a": box, "b": [box]})

    err = _refused(tmp_path, build, "names.shared_entity", r"tag 1 is given two names")
    assert err.data["names"] == ["a", "b"]


def test_unknown_tag(tmp_path: Path) -> None:
    """A group naming an entity that does not exist does not reach the file.

    The plug-in's naming function refused a tag that does not exist
    (``'ground' is given the tag 42, but the model``); with physical groups,
    Gmsh itself warns the script that writes the file and leaves the group
    out of it: the job sees no part 'ground', and a strip ending on it is
    refused.
    """

    def build() -> None:
        box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [box], name="a")
        gmsh.model.addPhysicalGroup(2, [42], name="ground")

    xao = write_xao(tmp_path / "u.xao", build)
    rep = geometry_job(tmp_path, xao).report
    assert {p["name"] for p in rep["parts"]} == {"a"}
    with pytest.raises(mpxlite.GmshJobError, match=r"\['ground'\].*not named surfaces") as e:
        geometry_job(tmp_path, xao, strips=[{"name": "a", "ends": ["ground"]}])
    assert (e.value.kind, e.value.code) == ("input", "strip.unknown_end")


def test_overlapping_surfaces(tmp_path: Path) -> None:
    def build() -> None:
        occ = gmsh.model.occ
        box = occ.addBox(0, 0, 0, 1, 1, 1)
        ground = occ.addRectangle(0, 0, 0, 1, 1)
        sheet = occ.addRectangle(0.2, 0.2, 0, 0.3, 0.3)
        name({"a": box}, {"ground": ground, "sheet": sheet})

    _refused(tmp_path, build, "names.overlap", r"surfaces 'ground' and 'sheet' overlap")


def test_overlapping_volumes(tmp_path: Path) -> None:
    def build() -> None:
        occ = gmsh.model.occ
        box = occ.addBox(0, 0, 0, 1, 1, 1)
        rod = occ.addBox(0.4, 0.4, 0, 0.2, 0.2, 1)
        name({"host": box, "rod": rod})

    err = _refused(tmp_path, build, "names.overlap", r"volumes 'host' and 'rod' overlap")
    assert err.data == {"kind": "volume", "names": ["host", "rod"]}


# --------------------------------------------------------------------------
# Refusals of the rules of a periodic prism cell.


def test_unnamed_volume(tmp_path: Path) -> None:
    def build() -> None:
        occ = gmsh.model.occ
        a = occ.addBox(0, 0, 0, 10, 10, 1)
        occ.addBox(0, 0, 1, 10, 10, 1)
        name({"a": a})

    _refused(tmp_path, build, "cell.unnamed_volume", r"centred at .* has no name")


def test_function_without_names(tmp_path: Path) -> None:
    """A file without any named part: no volume to name.

    The plug-in's message (``named no part``) is about its geometry function.
    """

    def build() -> None:
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)

    _refused(tmp_path, build, "cell.unnamed_volume", r"has no name")


def test_bottom_not_at_z_0(tmp_path: Path) -> None:
    def build() -> None:
        name({"a": gmsh.model.occ.addBox(0, 0, 1, 10, 10, 1)})

    err = _refused(
        tmp_path,
        build,
        "prism.bottom",
        r"must lie in the plane z = 0; the lowest point of the geometry is at z = 1 mm",
    )
    assert err.data["z_min"] == pytest.approx(1.0)


def test_corner_not_at_origin(tmp_path: Path) -> None:
    def build() -> None:
        name({"a": gmsh.model.occ.addBox(-5, -5, 0, 10, 10, 1)})

    _refused(
        tmp_path,
        build,
        "prism.corner",
        r"corner of the cell footprint must be at the origin.*\(-5, -5, 0\) mm",
    )


def test_first_side_not_along_x(tmp_path: Path) -> None:
    def build() -> None:
        box = gmsh.model.occ.addBox(0, 0, 0, 10, 10, 1)
        gmsh.model.occ.rotate([(3, box)], 0, 0, 0, 0, 0, 1, 0.2)
        name({"a": box})

    _refused(
        tmp_path,
        build,
        "prism.first_side",
        r"first side of the cell footprint must lie along \+x from the origin",
    )


def test_footprint_not_a_parallelogram(tmp_path: Path) -> None:
    def build() -> None:
        base = polygon([(0, 0), (10, 0), (8, 10), (0, 10)], 0)
        out = gmsh.model.occ.extrude([(2, base)], 0, 0, 1)
        name({"a": next(t for d, t in out if d == 3)})

    _refused(tmp_path, build, "prism.footprint", r"must be a parallelogram")


def test_geometry_leaving_the_prism(tmp_path: Path) -> None:
    def build() -> None:
        occ = gmsh.model.occ
        box = occ.addBox(0, 0, 0, 10, 10, 1)
        bump = occ.addBox(10, 4, 0, 1, 2, 1)
        out, _ = occ.fuse([(3, box)], [(3, bump)])
        name({"a": out[0][1]})

    _refused(
        tmp_path,
        build,
        "prism.outside",
        r"must be bounded by a prism.*the face centred at .* leaves this prism",
    )


def test_gap_between_volumes(tmp_path: Path) -> None:
    def build() -> None:
        occ = gmsh.model.occ
        a = occ.addBox(0, 0, 0, 10, 10, 1)
        b = occ.addBox(0, 0, 1.5, 10, 10, 0.5)
        name({"a": a, "b": b})

    err = _refused(
        tmp_path,
        build,
        "cell.not_filled",
        r"they leave a gap: their volumes add up to 150 mm\^3, the prism holds 200 mm\^3",
    )
    assert err.data["overlap"] is False


def _unfragmented(boxes: dict) -> None:
    for part, box in boxes.items():
        tag = gmsh.model.occ.addBox(*box)
        name({part: tag})


def test_overlap_in_a_file(tmp_path: Path) -> None:
    """Two named volumes that overlap: the conforming fragment refuses them.

    The plug-in read a file without fragmenting it and refused it on the
    filling (``they overlap: their volumes add up to 104 mm^3, the prism
    holds 100 mm^3``); the job always makes the parts conform first, and the
    overlap of two named volumes is refused by name.
    """
    build = lambda: _unfragmented({"host": (0, 0, 0, 10, 10, 1), "rod": (4, 4, 0, 2, 2, 1)})  # noqa: E731
    _refused(tmp_path, build, "names.overlap", r"volumes 'host' and 'rod' overlap")


def test_volumes_not_sharing_their_face(tmp_path: Path) -> None:
    """Two touching volumes of a file: the job makes them share their face.

    The plug-in refused a file of touching volumes that do not share their
    face (``do not share their common face``); the job conforms them, and the
    rule R5 (an outer face inside the prism) is shown on a gap instead.
    """
    build = lambda: _unfragmented({"a": (0, 0, 0, 10, 10, 1), "b": (0, 0, 1, 10, 10, 1)})  # noqa: E731
    rep = _run(tmp_path, build).report
    assert rep["cell"]["height"] == pytest.approx(2e-3)

    def hole() -> None:
        occ = gmsh.model.occ
        host = occ.addBox(0, 0, 0, 10, 10, 1)
        void = occ.addBox(4, 4, 0.25, 2, 2, 0.5)
        out, _ = occ.cut([(3, host)], [(3, void)])
        name({"host": out[0][1]})

    # an inner cavity: its faces bound the volume inside the prism (and it leaves a gap,
    # found first by the filling rule R4)
    _refused(tmp_path, hole, "cell.not_filled", r"leave a gap")


def test_file_not_xao(tmp_path: Path) -> None:
    """The plug-in's 'file with dimensions' refusal concerns its API; the format is the job's."""
    path = tmp_path / "x.step"
    path.write_text("not a geometry", encoding="utf-8")
    with pytest.raises(mpxlite.GmshJobError, match=r"must be a \.xao file") as caught:
        geometry_job(tmp_path, path)
    assert (caught.value.kind, caught.value.code) == ("input", "input.format")


def test_not_periodic_volume(tmp_path: Path) -> None:
    """A rod on the side face x = 0 only.

    Both the rod and the host have no image on x = 10 mm; the piece reported
    first follows the entity order of the conformal model. The plug-in's
    in-process model reported the rod (``'rod' at (0, 5, 0.5) mm``); the
    file-based model of the job reports the host piece below the rod.
    """

    def build() -> None:
        occ = gmsh.model.occ
        host = occ.addBox(0, 0, 0, 10, 10, 1)
        rod = occ.addBox(0, 4, 0, 2, 2, 1)
        out, _ = occ.cut([(3, host)], [(3, rod)], removeTool=False)
        name({"host": out[0][1], "rod": rod})

    err = _refused(
        tmp_path,
        build,
        "cell.not_periodic",
        r"must be periodic in \(x, y\).*'host' at \(0, 2, 0.5\) mm on the side face x = 0 "
        r"has no image on the opposite side face x = 10 mm",
    )
    assert err.data["names"] == ["host"]
    assert err.data["face"] == "x = 0" and err.data["opposite"] == "x = 10 mm"


def test_not_periodic_named_surface(tmp_path: Path) -> None:
    def build() -> None:
        occ = gmsh.model.occ
        box = occ.addBox(0, 0, 0, 10, 10, 1)
        strip = occ.addRectangle(2, 0, 1, 3, 6)
        name({"a": box}, {"strip": strip})

    _refused(
        tmp_path,
        build,
        "cell.not_periodic",
        r"'strip' at .* on the side face y = 0 has no image on the opposite side face y = 10",
    )


def test_not_periodic_names_differ(tmp_path: Path) -> None:
    def build() -> None:
        occ = gmsh.model.occ
        box = occ.addBox(0, 0, 0, 10, 10, 1)
        left = occ.addRectangle(2, 0, 1, 3, 5)
        right = occ.addRectangle(2, 5, 1, 3, 5)
        name({"a": box}, {"left": left, "right": right})

    _refused(
        tmp_path, build, "cell.not_periodic", r"'left' at .* on the side face y = 0 has no image"
    )


def test_periodic_strip_across_the_cell(tmp_path: Path) -> None:
    def build() -> None:
        occ = gmsh.model.occ
        box = occ.addBox(0, 0, 0, 10, 10, 1)
        strip = occ.addRectangle(2, 0, 1, 3, 10)
        name({"a": box}, {"strip": strip})

    assert set(_parts(_run(tmp_path, build).report, 2)) == {"strip"}


# --------------------------------------------------------------------------
# Refusals of the strip measurement.


def _element_cell(element, patches=((1, 2, 3.5, 6), (5.5, 2, 3.5, 6)), names=("left", "right")):
    def build() -> None:
        occ = gmsh.model.occ
        slab = occ.addBox(0, 0, 0, 10, 10, 1.6)
        surfaces: dict[str, list[int]] = {}
        for part, (x, y, dx, dy) in zip(names, patches, strict=True):
            surfaces.setdefault(part, []).append(occ.addRectangle(x, y, 1.6, dx, dy))
        surfaces["load"] = element(occ)
        name({"slab": slab}, surfaces)

    strips = [{"name": "load", "ends": sorted(set(names))}]
    return build, strips


def _strip_refused(tmp_path: Path, build, strips, code: str, match: str) -> mpxlite.GmshJobError:
    return _refused(tmp_path, build, code, match, strips=strips)


def test_element_free_sides_not_parallel(tmp_path: Path) -> None:
    def element(occ):
        p = [occ.addPoint(4.5, 4, 1.6), occ.addPoint(5.5, 3, 1.6),
             occ.addPoint(5.5, 7, 1.6), occ.addPoint(4.5, 6, 1.6)]  # fmt: skip
        lines = [occ.addLine(p[k], p[(k + 1) % 4]) for k in range(4)]
        return occ.addPlaneSurface([occ.addCurveLoop(lines)])

    err = _strip_refused(
        tmp_path,
        *_element_cell(element),
        "strip.free_sides",
        r"'load' must have two parallel straight free sides.*not parallel",
    )
    assert err.data["free_sides"] == 2


def test_element_curved_free_side(tmp_path: Path) -> None:
    def element(occ):
        p = [occ.addPoint(4.5, 4, 1.6), occ.addPoint(5.5, 4, 1.6),
             occ.addPoint(5.5, 6, 1.6), occ.addPoint(4.5, 6, 1.6)]  # fmt: skip
        centre = occ.addPoint(5, 4.5, 1.6)
        lines = [
            occ.addCircleArc(p[0], centre, p[1]),
            occ.addLine(p[1], p[2]),
            occ.addLine(p[2], p[3]),
            occ.addLine(p[3], p[0]),
        ]
        return occ.addPlaneSurface([occ.addCurveLoop(lines)])

    _strip_refused(tmp_path, *_element_cell(element), "strip.curved_side", r"free sides is curved")


def test_element_with_one_end_free(tmp_path: Path) -> None:
    err = _strip_refused(
        tmp_path,
        *_element_cell(lambda occ: occ.addRectangle(4.5, 4, 1.6, 0.7, 2)),
        "strip.ends",
        r"must have two ends, each on an end surface; found 1",
    )
    assert err.data["ends"] == 1


def test_strip_joins_a_surface_to_its_periodic_image(tmp_path: Path) -> None:
    build, strips = _element_cell(
        lambda occ: occ.addRectangle(3, 4, 1.6, 4, 2),
        patches=((0, 2, 3, 6), (7, 2, 3, 6)),
        names=("patch", "patch"),
    )
    shape = _run(tmp_path, build, strips=strips).report["strips"]["load"]
    assert shape["ends"] == ["patch", "patch"]
    assert shape["direction"] == pytest.approx((1.0, 0.0, 0.0))


def test_element_ends_on_one_piece_inside_the_cell(tmp_path: Path) -> None:
    err = _strip_refused(
        tmp_path,
        *_element_cell(
            lambda occ: occ.addRectangle(4.5, 4, 1.6, 1, 2),
            patches=((1, 2, 3.5, 6), (5.5, 2, 3.5, 6), (4.5, 7, 1, 1)),
            names=("left", "right", "bridge"),
        ),
        "strip.short_circuit",
        r"on one piece of the end surfaces \('left', 'right', connected inside the cell\)",
    )
    assert err.data["surfaces"] == ["left", "right"]


def test_element_between_two_pieces_of_one_name(tmp_path: Path) -> None:
    build, strips = _element_cell(
        lambda occ: occ.addRectangle(4.5, 4, 1.6, 1, 2), names=("patch", "patch")
    )
    assert _run(tmp_path, build, strips=strips).report["strips"]["load"]["ends"] == [
        "patch",
        "patch",
    ]


def test_element_touching_a_side_face(tmp_path: Path) -> None:
    err = _strip_refused(
        tmp_path,
        *_element_cell(
            lambda occ: occ.addRectangle(4.5, 0, 1.6, 1, 10),
            patches=((1, 0, 3.5, 10), (5.5, 0, 3.5, 10)),
        ),
        "strip.touches_side",
        r"'load' touches the side face y = 0 of the cell.*choose another cell of the same "
        r"periodic structure",
    )
    assert err.data["face"] == "y = 0"


def test_element_unknown_name(tmp_path: Path) -> None:
    err = _refused(
        tmp_path,
        lambda: _patch_cell(**_PATCH),
        "strip.unknown",
        r"'gap' is not a named surface",
        strips=[{"name": "gap", "ends": []}],
    )
    assert err.data["surfaces"] == ["ground", "patch"]


def test_unnamed_construction_surface_removed(tmp_path: Path) -> None:
    def build() -> None:
        occ = gmsh.model.occ
        box = occ.addBox(0, 0, 0, 10, 10, 1)
        occ.addRectangle(1, 1, 0.5, 1, 1)
        name({"a": box})

    result = _run(tmp_path, build)
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(result.outputs["geometry"]))
        assert len(gmsh.model.getEntities(2)) == 6
        assert not gmsh.model.mesh.getEmbedded(3, 1)
    finally:
        gmsh.finalize()


def test_invalid_parameters(tmp_path: Path) -> None:
    """Parameters of the job: kind ``input``, code ``parameter.invalid``."""
    xao = write_xao(tmp_path / "p.xao", lambda: _patch_cell(**_PATCH))
    for parameters in ({"scale": 0.0}, {"scale": "mm"}, {"strips": {"x": 1}},
                       {"strips": [{"name": "patch"}]}, {"unit": 3}):  # fmt: skip
        with pytest.raises(mpxlite.GmshJobError) as caught:
            geometry_job(tmp_path, xao, **parameters)
        assert (caught.value.kind, caught.value.code) == ("input", "parameter.invalid"), parameters
