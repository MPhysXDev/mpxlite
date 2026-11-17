# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""Helpers of the tests of the Gmsh jobs: user scripts that write cells, job runs, checks.

A test is a user script: it builds its geometry with the ``gmsh`` module,
names its parts with Gmsh's physical groups, writes an XAO file and hands it
to a job (:func:`mpxlite.run_gmsh_job`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import gmsh

import mpxlite
from mpxlite.cells import (
    check_periodic_facet_order,
    check_symmetric_numbering,
    require_mesh_symmetry,
)
from mpxlite.cells.jsonio import symmetry_from_json

MM = 1e-3
SPEED_OF_LIGHT = 299_792_458.0


def name(volumes: Mapping[str, Any], surfaces: Mapping[str, Any] | None = None) -> None:
    """Names the parts of the current model with physical groups (no fragment)."""
    gmsh.model.occ.synchronize()
    for dim, parts in ((3, volumes), (2, surfaces or {})):
        for part, tags in parts.items():
            tags = list(tags) if isinstance(tags, list | tuple) else [tags]
            gmsh.model.addPhysicalGroup(dim, tags, name=part)


def write_xao(path: Path, build: Callable[[], None]) -> Path:
    """Runs ``build`` in a new Gmsh model and writes the model to the XAO file ``path``."""
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("cell")
        build()
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def polygon(xy: Sequence[tuple[float, float]], z: float) -> int:
    """A plane surface bounded by the polygon ``xy`` in the plane ``z``."""
    occ = gmsh.model.occ
    p = [occ.addPoint(x, y, z) for x, y in xy]
    lines = [occ.addLine(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]
    return occ.addPlaneSurface([occ.addCurveLoop(lines)])


def geometry_job(tmp_path: Path, xao: Path, **parameters: Any) -> mpxlite.GmshJobResult:
    """The job ``periodic_cell_geometry`` on ``xao``."""
    return mpxlite.run_gmsh_job(
        "periodic_cell_geometry",
        inputs={"geometry": xao},
        parameters=parameters,
        runs_dir=tmp_path / "jobs",
        timeout=600,
    )


def mesh_job(tmp_path: Path, xao: Path, **parameters: Any) -> mpxlite.GmshJobResult:
    """The job ``periodic_cell_mesh`` on ``xao``."""
    return mpxlite.run_gmsh_job(
        "periodic_cell_mesh",
        inputs={"geometry": xao},
        parameters=parameters,
        runs_dir=tmp_path / "jobs",
        timeout=1800,
    )


def bulk(permittivity: float, frequency: float = 8e9, loss_tangent: float = 0.0) -> float:
    """A tenth of the wavelength in a medium (the numbers a plug-in would give)."""
    n = math.sqrt(permittivity * math.hypot(1.0, loss_tangent))
    return SPEED_OF_LIGHT / (frequency * n) / 10.0


AIR = bulk(1.0)
FR4 = bulk(4.4)
ABOVE = {"thickness": 5 * MM, "volume": "_layer_above", "face": "_top_face", "size": AIR}
EDGE_RULE = {
    "divisions": 6.0,
    "slope": 0.5,
    "min_fraction": 1e-3,
    "line_pieces": 8,
    "curve_pieces": 32,
    "extent_fraction": 0.5,
}


def mesh_parameters(
    volumes: Mapping[str, float],
    refined: Sequence[str],
    *,
    factor: float = 1.0,
    above: Mapping[str, Any] | None = None,
    below: Mapping[str, Any] | None = None,
    symmetry: Mapping[str, Any] | None = None,
    output: str = "cell.msh",
) -> dict[str, Any]:
    """The parameters of ``periodic_cell_mesh`` as a plug-in would derive them."""
    layers: dict[str, Any] = {"above": dict(above or ABOVE)}
    if below is not None:
        layers["below"] = dict(below)
    else:
        layers["bare_bottom"] = "_bottom"
    return {
        "sizes": {
            "volumes": dict(volumes),
            "refined_edges": [{"group": n, "factor": factor} for n in refined],
            "edge_rule": EDGE_RULE,
        },
        "layers": layers,
        "symmetry": {"mode": "detect", **dict(symmetry or {})},
        "mesh": {"output": output},
    }


def check_mesh(result: mpxlite.GmshJobResult) -> None:
    """Periodic facet order, symmetric numbering and the symmetry gate of a job's mesh."""
    path = result.outputs["mesh"]
    tags = {n: int(v[1]) for n, v in result.report["physical"].items()}
    sym = symmetry_from_json(result.report["symmetry"], tags)
    t = sym.lattice
    assert check_periodic_facet_order(path).ok
    for op in sym.ops[1:]:
        assert check_symmetric_numbering(path, t, op).ok, op.name
    maps = {o.name: p for o, p in zip(sym.ops, sym.permutations, strict=True)}
    require_mesh_symmetry(path, t, sym.ops, tag_maps=maps, what=sym.name)
