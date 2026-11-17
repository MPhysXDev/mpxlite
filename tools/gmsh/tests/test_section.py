# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the GNU General Public
# License (the file COPYING of the directory above) for more details.

"""Layered sections (:mod:`mpxlite_gmsh.section`): parameters, size law, model and mesh."""

from __future__ import annotations

import gmsh
import numpy as np
import pytest
from mpxlite_gmsh import section

from mpxlite.gmshjob import GmshJobError


def _params(**changes):
    params = {
        "columns": {
            "x": [0.0, 1.0, 3.0],
            "cells": [{"name": "a", "cells": 2}, {"name": "b", "cells": 4, "progression": 1.2}],
        },
        "core": {"y": [0.0, 0.5], "rows": 1},
        "boundaries": {"core_left": "left", "outer": "outer"},
        "above": {"name": "up", "height": 1.0, "mesh": "free"},
        "sizes": {
            "floor": [{"size": 0.5}, {"size": 0.1, "rate": 2.0, "centre": -1.0, "half_width": 0.5}],
            "base": [{"size": 0.1}],
            "rate": 0.25,
            "min": 0.1,
            "max": 0.5,
        },
    }
    params.update(changes)
    return params


def test_parse() -> None:
    s = section.parse(_params())
    assert s.x == (0.0, 1.0, 3.0) and s.x_cells == 6 and s.rows == 1
    assert s.columns[1] == section.Column("b", 4, 1.2)
    assert s.above == section.Layer("up", 1.0, False) and s.below is None
    assert s.group_names() == [(2, "a"), (2, "b"), (2, "up"), (1, "left"), (1, "outer")]
    assert s.structured_elements() == 12


def test_outer_without_layers_has_no_group() -> None:
    s = section.parse(_params(above=None, sizes=None))
    assert s.group_names() == [(2, "a"), (2, "b"), (1, "left")]


def test_size_law_expression() -> None:
    law = section.parse(_params()).sizes
    assert law.expression(0.0, 0.5) == (
        "Min(0.5, Max(Min(0.5, 0.1 + 2.0*Max(0, Fabs(x - (-1.0)) - 0.5)), "
        "0.1 + 0.25*Max(y - 0.5, 0.0 - y)))"
    )
    # the expression is the law: evaluate it at a few points
    f = law.expression(0.0, 0.5).replace("Min", "min").replace("Max", "max").replace("Fabs", "abs")
    for x, y in ((-1.0, 0.5), (0.2, 1.5), (3.0, -2.0)):
        floor = min(0.5, 0.1 + 2.0 * max(0, abs(x + 1.0) - 0.5))
        expected = min(0.5, max(floor, 0.1 + 0.25 * max(y - 0.5, -y)))
        assert eval(f, {"x": x, "y": y, "min": min, "max": max, "abs": abs}) == expected


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"core": {"y": [0.0, 0.5]}}, "parameter.missing"),
        ({"columns": {"x": [0.0, 1.0], "cells": []}}, "parameter.invalid"),
        ({"columns": {"x": [1.0, 0.0], "cells": [{"name": "a", "cells": 1}]}}, "section.order"),
        ({"sizes": None}, "sizes.missing"),
        ({"boundaries": {"core_left": "a"}}, "names.twice"),
        ({"boundaries": {"core_left": "s", "core_right": "s"}}, "names.twice"),
        ({"lines": [{"name": "m", "x": [2.0]}]}, "lines.off_column"),
        ({"tags": {"a": 1, "b": 1}}, "tags.twice"),
        ({"tags": {"zzz": 1}}, "tags.unknown"),
        ({"mesh": {"element_order": True}}, "mesh.element_order"),
    ],
)
def test_refusals(change, code) -> None:
    with pytest.raises(GmshJobError) as caught:
        section.parse(_params(**change))
    assert caught.value.code == code


def test_tags_of_a_line_and_a_surface_may_be_equal() -> None:
    s = section.parse(_params(lines=[{"name": "m", "x": [1.0]}], tags={"a": 1, "m": 1}))
    assert s.tags == {"a": 1, "m": 1}


def test_build_and_mesh(model) -> None:
    s = section.parse(_params(lines=[{"name": "m", "x": [1.0]}], tags={"b": 7}))
    groups = section.build(s)
    assert groups == {
        "a": (2, 8), "b": (2, 7), "up": (2, 9), "left": (1, 10), "outer": (1, 11), "m": (1, 12),
    }  # fmt: skip
    gmsh.model.mesh.generate(2)
    shortest, longest, count = section.edge_lengths(1e-3)
    # the core: 6 x 1 cells, two triangles each; the free layer above
    core = sum(len(t) for t in gmsh.model.mesh.getElements(2, 1)[1])
    assert core == 4
    assert count > 12
    first = 2 * (1.2 - 1) / (1.2**4 - 1)  # the first cell of the graded column
    assert 0 < shortest <= first * 1e-3 * (1 + 1e-12)
    assert longest == pytest.approx(np.hypot(first * 1.2**3, 0.5) * 1e-3)  # its last diagonal
    # the line m is the vertical curve at x = 1
    (curve,) = gmsh.model.getEntitiesForPhysicalGroup(1, 12)
    lo = gmsh.model.getBoundingBox(1, curve)
    assert lo[0] == pytest.approx(1.0) and lo[3] == pytest.approx(1.0)
    nodes = gmsh.model.mesh.getNodes(1, int(curve), includeBoundary=True)[1].reshape(-1, 3)
    assert np.allclose(nodes[:, 0], 1.0)
