# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""The job ``layered_section_mesh``: layered rectangular sections in two dimensions.

The job has no input file: each test gives the section as parameters, as a
plug-in does, runs the job and reads the written MSH 2.2 file back with a
small reader of its own (nodes, elements by type and physical tag, names).

Sections: a structured core alone (cells and rows, the four sides, a line
group, given and default tags); graded columns; a core between a
structured layer above and a free layer below (the size law); quadrangles;
second-order elements; the same parameters twice (the same file); the
element limit; refused parameters.
"""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import mpxlite

JOB = "layered_section_mesh"
_TRIANGLE, _QUADRANGLE, _LINE, _TRIANGLE6 = 2, 3, 1, 9


def _run(tmp_path: Path, params: dict[str, Any], run_id: str | None = None):
    return mpxlite.run_gmsh_job(JOB, parameters=params, runs_dir=tmp_path / "jobs", run_id=run_id)


def _read(path: Path) -> dict[str, Any]:
    """Nodes, elements ``(type, physical tag, nodes)`` and physical names of an MSH 2.2 file."""
    lines = path.read_text(encoding="ascii").splitlines()
    assert lines[1].split()[0] == "2.2"
    names, nodes, elements = {}, {}, []
    k = 0
    while k < len(lines):
        head = lines[k]
        if head == "$PhysicalNames":
            for line in lines[k + 2 : k + 2 + int(lines[k + 1])]:
                dim, tag, name = line.split(maxsplit=2)
                names[name.strip('"')] = (int(dim), int(tag))
        elif head == "$Nodes":
            for line in lines[k + 2 : k + 2 + int(lines[k + 1])]:
                tag, x, y, z = line.split()
                nodes[int(tag)] = (float(x), float(y), float(z))
        elif head == "$Elements":
            for line in lines[k + 2 : k + 2 + int(lines[k + 1])]:
                f = [int(v) for v in line.split()]
                elements.append((f[1], f[3], f[3 + f[2] :]))
        k += 1
    return {"names": names, "nodes": nodes, "elements": elements}


def _core(**changes: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "scale": 1e-3,
        "unit": "mm",
        "columns": {
            "x": [-2.0, 0.0, 1.0, 3.0],
            "cells": [
                {"name": "left", "cells": 4},
                {"name": "middle", "cells": 2},
                {"name": "right", "cells": 4},
            ],
        },
        "core": {"y": [0.0, 1.0], "rows": 2},
        "boundaries": {
            "core_top": "top",
            "core_bottom": "bottom",
            "core_left": "west",
            "core_right": "east",
        },
        "lines": [{"name": "cuts", "x": [0.0, 1.0]}],
        "tags": {"left": 1, "middle": 2, "right": 3, "top": 101},
    }
    params.update(changes)
    return params


def _layered(**changes: Any) -> dict[str, Any]:
    params = _core(
        above={"name": "cover", "height": 2.0, "mesh": "structured", "rows": 4},
        below={"name": "base", "height": 3.0, "mesh": "free"},
        sizes={
            "floor": [{"size": 0.5}, {"size": 0.25, "rate": 0.5, "half_width": 1.0}],
            "base": [{"size": 0.25}],
            "rate": 0.3,
            "min": 0.25,
            "max": 1.0,
        },
    )
    params["boundaries"] = {**params["boundaries"], "outer": "outer"}
    params.update(changes)
    return params


def test_structured_core(tmp_path: Path) -> None:
    result = _run(tmp_path, _core())
    rep = result.report
    msh = _read(result.outputs["mesh"])
    assert rep["unit"] == "mm" and rep["scale"] == 1e-3
    assert rep["section"]["x"] == pytest.approx([-2e-3, 3e-3])
    assert rep["section"]["y"] == pytest.approx([0.0, 1e-3])
    assert rep["section"]["x_cells"] == 10 and rep["section"]["rows"] == 2
    assert rep["mesh"]["nodes"] == 11 * 3 == len(msh["nodes"])
    assert rep["mesh"]["surface_elements"] == 2 * 10 * 2
    physical = rep["physical"]
    assert physical == {
        "left": [2, 1], "middle": [2, 2], "right": [2, 3],
        "top": [1, 101], "bottom": [1, 102], "west": [1, 103], "east": [1, 104],
        "cuts": [1, 105],
    }  # fmt: skip
    assert {n: list(v) for n, v in msh["names"].items()} == physical
    count = Counter((t, g) for t, g, _ in msh["elements"])
    assert count[(_TRIANGLE, 1)] == 16 and count[(_TRIANGLE, 2)] == 8
    assert count[(_LINE, 101)] == 10 and count[(_LINE, 103)] == 2 and count[(_LINE, 105)] == 4
    xs = sorted({round(x / 1e-3, 9) for x, _, _ in msh["nodes"].values()})
    assert xs == pytest.approx([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3])
    lines = {round(msh["nodes"][n][0] / 1e-3, 9)
             for t, g, ns in msh["elements"] if g == 105 for n in ns}  # fmt: skip
    assert lines == {0.0, 1.0}
    h = rep["mesh"]["edge_length"]
    assert h["min"] == pytest.approx(0.5e-3)
    assert h["max"] == pytest.approx(np.hypot(0.5, 0.5) * 1e-3)
    assert result.messages == ()


def test_graded_columns(tmp_path: Path) -> None:
    params = _core()
    params["columns"]["cells"][2]["progression"] = 1.5
    msh = _read(_run(tmp_path, params).outputs["mesh"])
    xs = np.array(sorted({x / 1e-3 for x, y, _ in msh["nodes"].values() if abs(y) < 1e-12}))
    right = np.diff(xs[xs >= 1.0 - 1e-9])
    assert len(right) == 4
    assert right[1:] / right[:-1] == pytest.approx([1.5, 1.5, 1.5])
    assert right.sum() == pytest.approx(2.0)


def test_layers_and_size_law(tmp_path: Path) -> None:
    result = _run(tmp_path, _layered())
    rep = result.report
    msh = _read(result.outputs["mesh"])
    assert rep["section"]["y"] == pytest.approx([-3e-3, 3e-3])
    assert rep["section"]["core"] == pytest.approx([0.0, 1e-3])
    physical = rep["physical"]
    # groups without a given tag: numbered after the largest given one, in the order of creation
    assert physical["cover"] == [2, 102] and physical["base"] == [2, 103]
    assert physical["bottom"] == [1, 104] and physical["outer"] == [1, 107]
    assert physical["cuts"] == [1, 108]
    count = Counter((t, g) for t, g, _ in msh["elements"])
    assert count[(_TRIANGLE, 102)] == 2 * 10 * 4  # structured: 10 cells across, 4 rows
    assert count[(_TRIANGLE, 103)] > 0
    assert count[(_LINE, 107)] > 4 + 10 + 4  # the structured sides and top, the free ones
    # the free layer: its nodes on the core face are those of the core columns
    face = {round(x / 1e-3, 9) for x, y, _ in msh["nodes"].values() if abs(y) < 1e-12}
    assert len(face) == 11
    # the size law: no edge of the free layer above the maximum, the smallest at the core
    h = rep["mesh"]["edge_length"]
    assert h["max"] <= 1.0e-3 * 1.5
    assert h["min"] >= 0.2e-3


def test_quadrangles_and_second_order(tmp_path: Path) -> None:
    quad = _run(tmp_path, _layered(mesh={"element_type": "quadrangles"}), "quad")
    msh = _read(quad.outputs["mesh"])
    count = Counter((t, g) for t, g, _ in msh["elements"])
    assert count[(_QUADRANGLE, 1)] == 8 and count[(_QUADRANGLE, 102)] == 40
    second = _run(tmp_path, _core(mesh={"element_order": 2}), "second")
    msh = _read(second.outputs["mesh"])
    types = {t for t, _, _ in msh["elements"]}
    assert _TRIANGLE6 in types and _TRIANGLE not in types
    assert len(msh["nodes"]) == 21 * 5


@pytest.mark.parametrize("diagonals", ["left", "right", "alternate_left", "alternate_right"])
def test_diagonals(tmp_path: Path, diagonals: str) -> None:
    result = _run(tmp_path, _core(mesh={"diagonals": diagonals}))
    assert result.report["mesh"]["surface_elements"] == 40


def test_same_parameters_same_file(tmp_path: Path) -> None:
    a = _run(tmp_path, _layered(), "a").outputs["mesh"].read_bytes()
    b = _run(tmp_path, _layered(), "b").outputs["mesh"].read_bytes()
    assert a == b


def test_element_limit(tmp_path: Path) -> None:
    with pytest.raises(mpxlite.GmshJobError) as before:
        _run(tmp_path, _core(mesh={"max_elements": 39}), "before")
    assert (before.value.kind, before.value.code) == ("gmsh", "mesh.too_many_elements")
    assert before.value.data == {"estimate": 40, "limit": 39, "when": "before"}
    with pytest.raises(mpxlite.GmshJobError) as after:
        _run(tmp_path, _layered(mesh={"max_elements": 130}), "after")
    assert after.value.data["when"] == "after" and after.value.data["count"] > 130


def _set(params: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    params = copy.deepcopy(params)
    node = params
    keys = path.split(".")
    for key in keys[:-1]:
        node = node[int(key)] if isinstance(node, list) else node.setdefault(key, {})
    if isinstance(node, list):
        node[int(keys[-1])] = value
    elif value is None:
        node.pop(keys[-1], None)
    else:
        node[keys[-1]] = value
    return params


@pytest.mark.parametrize(
    ("path", "value", "kind", "code"),
    [
        ("core.rows", None, "input", "parameter.missing"),
        ("core.rows", 0, "input", "parameter.invalid"),
        ("core.y", [1.0, 0.0], "rule", "section.order"),
        ("columns.x", [-2.0, 1.0, 0.0, 3.0], "rule", "section.order"),
        ("columns.x", [-2.0, 3.0], "input", "parameter.invalid"),
        ("columns.cells.0", {"name": "left", "cells": 1.5}, "input", "parameter.invalid"),
        ("columns.cells.0", {"name": "", "cells": 1}, "input", "parameter.invalid"),
        ("columns.cells.0", {"name": "l", "cells": 1, "progression": 0}, "input",
         "parameter.invalid"),
        ("lines", [{"name": "cuts", "x": [0.5]}], "rule", "lines.off_column"),
        ("lines", [{"name": "top", "x": [0.0]}], "rule", "names.twice"),
        ("tags", {"left": 1, "middle": 1}, "rule", "tags.twice"),
        ("tags", {"nowhere": 7}, "input", "tags.unknown"),
        ("boundaries", {"north": "n"}, "input", "parameter.invalid"),
        ("above", {"name": "left", "height": 1.0, "mesh": "structured", "rows": 1}, "rule",
         "names.twice"),
        ("above", {"name": "cover", "height": 1.0, "mesh": "structured"}, "input",
         "parameter.invalid"),
        ("above", {"name": "cover", "height": 1.0}, "input", "sizes.missing"),
        ("sizes", {"floor": [], "base": [{"size": 1}], "rate": 0, "min": 1, "max": 1}, "input",
         "parameter.invalid"),
        ("mesh", {"element_type": "hexagons"}, "input", "parameter.invalid"),
        ("mesh", {"diagonals": "up"}, "input", "parameter.invalid"),
        ("mesh", {"element_order": 3}, "input", "mesh.element_order"),
        ("mesh", {"output": "../x.msh"}, "input", "parameter.invalid"),
        ("scale", 0.0, "input", "parameter.invalid"),
    ],
)  # fmt: skip
def test_invalid_parameters(tmp_path: Path, path: str, value: Any, kind: str, code: str) -> None:
    with pytest.raises(mpxlite.GmshJobError) as caught:
        _run(tmp_path, _set(_core(), path, value))
    assert (caught.value.kind, caught.value.code) == (kind, code), caught.value


def test_size_law_without_free_layer_is_refused(tmp_path: Path) -> None:
    params = _layered(below=None)
    with pytest.raises(mpxlite.GmshJobError) as caught:
        _run(tmp_path, params)
    assert (caught.value.code, caught.value.data["parameter"]) == ("parameter.invalid", "sizes")
