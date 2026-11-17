# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the GNU General Public
# License (the file COPYING of the directory above) for more details.

"""The named geometry (:mod:`mpxlite_gmsh.named`) and the messages of Gmsh."""

from __future__ import annotations

import gmsh
import numpy as np
import pytest
from mpxlite_gmsh import session
from mpxlite_gmsh.named import NamedGeometry

from mpxlite.gmshjob import GmshJobError


def _overlapping_parts() -> None:
    """A substrate box, a cover box above it, a patch on their interface and a construction line."""
    occ = gmsh.model.occ
    sub = occ.addBox(0, 0, 0, 1, 1, 0.5)
    cover = occ.addBox(0, 0, 0.5, 1, 1, 0.5)
    patch = occ.addRectangle(0.2, 0.2, 0.5, 0.4, 0.4)
    occ.addLine(occ.addPoint(2, 2, 2), occ.addPoint(3, 3, 3))  # stray
    occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [sub], name="sub")
    gmsh.model.addPhysicalGroup(3, [cover], name="cover")
    gmsh.model.addPhysicalGroup(2, [patch], name="patch")


def test_conform_carries_the_names(model, tmp_path):
    _overlapping_parts()
    geo = NamedGeometry()
    geo.conform()
    groups = geo.groups()
    assert set(groups) == {"sub", "cover", "patch"}
    assert [groups[n][0] for n in ("sub", "cover", "patch")] == [3, 3, 2]
    assert geo.measure("patch") == pytest.approx(0.16)
    assert geo.measure("sub") == pytest.approx(0.5)
    # the patch is a piece of the shared face of the two boxes
    (face,) = geo.entities("patch")
    assert geo.volumes_of_face(face) == ["cover", "sub"]
    assert gmsh.model.getEntitiesInBoundingBox(1.9, 1.9, 1.9, 3.1, 3.1, 3.1) == []  # stray gone
    # written and read back: the names, not the numbers, are the identity
    path = geo.write(tmp_path / "c.xao")
    gmsh.clear()
    again = NamedGeometry.load(path)
    assert {n: (d, len(e)) for n, (d, e) in again.groups().items()} == {
        n: (d, len(e)) for n, (d, e) in groups.items()
    }
    assert again.measure("patch") == pytest.approx(0.16)


def test_conform_refuses_overlapping_volumes(model):
    occ = gmsh.model.occ
    a = occ.addBox(0, 0, 0, 1, 1, 1)
    b = occ.addBox(0.5, 0, 0, 1, 1, 1)
    occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [a], name="a")
    gmsh.model.addPhysicalGroup(3, [b], name="b")
    with pytest.raises(GmshJobError) as caught:
        NamedGeometry().conform()
    assert (caught.value.kind, caught.value.code) == ("rule", "names.overlap")
    assert caught.value.data == {"kind": "volume", "names": ["a", "b"]}


@pytest.mark.parametrize(
    ("groups", "code"),
    [
        ([(3, "")], "names.unnamed"),
        ([(3, "v"), (1, "wire")], "names.dimension"),
        ([(3, "v"), (2, "v")], "names.twice"),
    ],
)
def test_malformed_groups(model, groups, code):
    box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    for dim, name in groups:
        gmsh.model.addPhysicalGroup(dim, [box if dim == 3 else 1], name=name)
    with pytest.raises(GmshJobError) as caught:
        NamedGeometry.groups()
    assert (caught.value.kind, caught.value.code) == ("rule", code)


def test_shared_entity_refused(model):
    box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [box], name="a")
    gmsh.model.addPhysicalGroup(3, [box], name="b")
    with pytest.raises(GmshJobError) as caught:
        NamedGeometry().conform()
    assert caught.value.code == "names.shared_entity"


def test_scale_keeps_names_and_planes(model):
    _overlapping_parts()
    geo = NamedGeometry()
    geo.conform()
    geo.scale(1e-3)
    assert geo.measure("sub") == pytest.approx(0.5e-9)
    assert geo.measure("patch") == pytest.approx(0.16e-6)
    assert np.allclose(geo.centre("patch"), (0.4e-3, 0.4e-3, 0.5e-3))
    assert all(gmsh.model.getType(2, t) == "Plane" for _, t in gmsh.model.getEntities(2))


def test_fragment_with_a_slab_keeps_the_parts(model):
    _overlapping_parts()
    geo = NamedGeometry()
    geo.conform()
    before = {n: (d, geo.measure(n)) for n, (d, _) in geo.groups().items()}
    tags = {gmsh.model.getPhysicalName(d, t): t for d, t in gmsh.model.getPhysicalGroups()}
    top = gmsh.model.occ.addBox(0, 0, 1, 1, 1, 0.25)
    (pieces,) = geo.fragment_with([(3, top)], 1.0)
    assert [d for d, _ in pieces] == [3]
    after = {n: (d, geo.measure(n)) for n, (d, _) in geo.groups().items()}
    assert after.keys() == before.keys()
    for name, (dim, measure) in before.items():
        assert after[name][0] == dim and after[name][1] == pytest.approx(measure)
    assert {gmsh.model.getPhysicalName(d, t): t for d, t in gmsh.model.getPhysicalGroups()} == tags


def test_fragment_with_a_tool_that_splits_a_part(model):
    _overlapping_parts()
    geo = NamedGeometry()
    geo.conform()
    cutter = gmsh.model.occ.addBox(0.5, -1, -1, 3, 3, 3)
    with pytest.raises(GmshJobError) as caught:
        geo.fragment_with([(3, cutter)], 1.0)
    assert (caught.value.kind, caught.value.code) == ("gmsh", "names.lost")


def test_questions(model):
    _overlapping_parts()
    geo = NamedGeometry("mm")
    geo.conform()
    assert np.allclose(geo.bounding_box("sub"), [[0, 0, 0], [1, 1, 0.5]], atol=1e-6)
    assert len(geo.corner_points("patch")) == 4
    assert geo.extent() == pytest.approx(1.0)
    plane = geo.faces_on_plane((0, 0, 1), 0.5, 1e-9)
    assert set(geo.entities("patch")) <= set(plane) and len(plane) == 2
    assert geo.touching("sub", "cover") and geo.touching("patch", "sub")
    assert geo.point_in("sub", (0.5, 0.5, 0.25)) and not geo.point_in("sub", (0.5, 0.5, 0.75))
    assert geo.point_in("patch", (0.4, 0.4, 0.5)) and not geo.point_in("patch", (0.9, 0.9, 0.5))
    assert np.allclose(geo.closest_point("patch", (0.4, 0.4, 0.9)), (0.4, 0.4, 0.5))
    assert np.allclose(geo.closest_point("sub", (0.5, 0.5, 2.0)), (0.5, 0.5, 0.5))
    # the bottom faces of the two boxes, x = 0 and x = 1: matched by the translation (1, 0, 0)
    left = geo.faces_on_plane((1, 0, 0), 0.0, 1e-9)
    right = geo.faces_on_plane((1, 0, 0), 1.0, 1e-9)
    shift = np.eye(4)
    shift[0, 3] = 1.0
    pairs = geo.faces_match(left, right, shift, 1e-9)
    assert pairs is not None and len(pairs) == 2
    assert geo.faces_match(left, right, np.eye(4), 1e-9) is None


def test_messages_turn_errors_into_failures(model):
    messages = session.Messages()
    gmsh.logger.start()
    with pytest.raises(Exception, match="Unknown OpenCASCADE entity"):
        gmsh.model.occ.getMass(3, 99)
    with pytest.raises(GmshJobError) as caught:
        messages.check("a query")
    assert (caught.value.kind, caught.value.code) == ("gmsh", "gmsh.error")
    assert caught.value.data["step"] == "a query"
    assert messages.collected[0][0] == "error"
    assert messages.drain() == []  # the logger was emptied


def test_fatal_warnings():
    assert any(p.search("Could not recover edge 12") for p in session.FATAL_WARNINGS)
    assert not any(p.search("Mesh is periodic: 2 surfaces") for p in session.FATAL_WARNINGS)
