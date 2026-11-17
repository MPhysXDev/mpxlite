# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the GNU General Public
# License (the file COPYING of the directory above) for more details.

"""Job ``periodic_cell_mesh``: the symmetric periodic mesh of a named prism cell.

Input ``geometry``: the conformal named geometry of a periodic prism cell
(the output of the job ``periodic_cell_geometry``, XAO).

Steps: open the file and read the prism (:func:`~mpxlite_gmsh.prism.read_cell`);
add the layers above / below (:func:`~mpxlite_gmsh.layers.add_layers`);
derive the sizes from the controls given by name
(:func:`~mpxlite_gmsh.sizing.cell_sizing`); detect the symmetry, cut the
asymmetric unit, pair its faces, mesh it, replicate it and write the
periodic mesh (:func:`~mpxlite_gmsh.symmetric.build_symmetric_mesh`); check
the node order of the periodic facets
(:func:`mpxlite.cells.periodic.check_periodic_facet_order`).

Parameters (see the developer documentation for every field and default)::

    {"sizes": {"volumes": {"<volume>": h, ...},
               "refined_edges": [{"group": "<surface>", "factor": 1.0}, ...],
               "edge_rule": {"divisions": 6, "slope": 0.5, "min_fraction": 0.001,
                             "line_pieces": 8, "curve_pieces": 32,
                             "extent_fraction": 0.5},
               "min_size": null, "max_size": null},
     "layers": {"above": {"thickness": t, "volume": "<name>", "face": "<name>",
                          "size": h},
                "below": {...}, "bare_bottom": "<name>"},
     "symmetry": {"mode": "detect" | "none", "classes": {"<tag name>": label},
                  "directions": {"<tag name>": [x, y, z]},
                  "allowed": "all" | "diagonal" | [[[a, b], [c, d]], ...],
                  "tolerance": null, "numbering": true, "window_lo": [0, 0]},
     "mesh": {"element_order": 1, "options": {"Mesh.Algorithm": 6, ...},
              "max_elements": 5000000, "output": "cell.msh"},
     "unit": ""}

Output ``mesh``: the MSH 2.2 ASCII file with its ``$Periodic`` section. The
report: ``cell``, ``layers``, ``sizes``, ``symmetry``, ``detected``,
``asymmetric_unit``, ``mesh``, ``physical``, ``notes``, ``timings``.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Hashable, Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gmsh
import numpy as np
from mpxlite_gmsh.job import Job, check_number, run
from mpxlite_gmsh.layers import Layer, add_layers
from mpxlite_gmsh.named import NamedGeometry
from mpxlite_gmsh.prism import read_cell
from mpxlite_gmsh.sizing import EdgeRule, cell_sizing
from mpxlite_gmsh.symmetric import DEFAULT_MESH_OPTIONS, MeshLimits, build_symmetric_mesh

from mpxlite.cells.autosym import MeshSizing, SymmetricMeshError
from mpxlite.cells.jsonio import symmetry_to_json, unit_to_json
from mpxlite.cells.periodic import check_periodic_facet_order
from mpxlite.cells.topology import MeshTopologyError
from mpxlite.gmshjob import GmshJobError

DEFAULT_OUTPUT = "cell.msh"
DEFAULT_MAX_ELEMENTS = 5_000_000
_MESH_OPTIONS = frozenset(
    {
        "Mesh.Algorithm",
        "Mesh.Algorithm3D",
        "Mesh.Optimize",
        "Mesh.OptimizeNetgen",
        "Mesh.Smoothing",
        "Mesh.RandomFactor",
        "Mesh.RandomFactor3D",
        "Mesh.AngleToleranceFacetOverlap",
    }
)


def _mapping(job: Job, path: str) -> dict[str, Any]:
    value = job.parameter(path, {})
    if not isinstance(value, Mapping):
        raise job.fail_input(path, f"the parameter {path!r} must be an object")
    return dict(value)


def _layer(job: Job, path: str) -> tuple[Layer, float | None] | None:
    spec = job.parameter(path, None)
    if spec is None:
        return None
    if not isinstance(spec, Mapping) or not all(
        isinstance(spec.get(k), str) and spec.get(k) for k in ("volume", "face")
    ):
        raise job.fail_input(
            path, f"{path} must be {{'thickness': t, 'volume': <name>, 'face': <name>}}"
        )
    thickness = check_number(f"{path}.thickness", spec.get("thickness"), minimum=0, strict=True)
    size = spec.get("size")
    if size is not None:
        size = check_number(f"{path}.size", size, minimum=0, strict=True)
    return Layer(thickness, spec["volume"], spec["face"]), size


def _edge_rule(job: Job) -> EdgeRule:
    spec = _mapping(job, "sizes.edge_rule")
    base = EdgeRule()
    known = {f.name for f in EdgeRule.__dataclass_fields__.values()}
    unknown = sorted(set(spec) - known)
    if unknown:
        raise job.fail_input("sizes.edge_rule", f"unknown fields of sizes.edge_rule: {unknown}")
    values: dict[str, Any] = {}
    for name in known:
        if name not in spec:
            continue
        value = spec[name]
        path = f"sizes.edge_rule.{name}"
        if name == "extent_fraction" and value is None:
            values[name] = None
        elif name in ("line_pieces", "curve_pieces"):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise job.fail_input(path, f"{path} must be a positive integer; found {value!r}")
            values[name] = value
        else:
            values[name] = check_number(path, value, minimum=0, strict=True)
    return EdgeRule(**{**base.__dict__, **values})


def _sizes(job: Job, layers: list[tuple[Layer, float | None]]) -> tuple[dict, dict]:
    volumes = _mapping(job, "sizes.volumes")
    sizes = {n: check_number(f"sizes.volumes.{n}", h, minimum=0, strict=True)
             for n, h in volumes.items()}  # fmt: skip
    for layer, size in layers:
        if size is None:
            continue
        if layer.volume in sizes and sizes[layer.volume] != size:
            raise job.fail_input(
                "sizes.volumes", f"two sizes for the layer volume {layer.volume!r}"
            )
        sizes[layer.volume] = size
    refined: dict[str, float] = {}
    items = job.parameter("sizes.refined_edges", [])
    if not isinstance(items, list):
        raise job.fail_input("sizes.refined_edges", "sizes.refined_edges must be a list")
    for k, item in enumerate(items):
        path = f"sizes.refined_edges[{k}]"
        if not isinstance(item, Mapping) or not isinstance(item.get("group"), str):
            raise job.fail_input(path, f"{path} must be {{'group': <surface>, 'factor': f}}")
        factor = check_number(f"{path}.factor", item.get("factor", 1.0), minimum=0, strict=True)
        refined[item["group"]] = min(factor, refined.get(item["group"], math.inf))
    return sizes, refined


def _allowed(job: Job) -> Callable[[np.ndarray], bool] | None:
    mode = job.parameter("symmetry.mode", "detect")
    allowed = job.parameter("symmetry.allowed", "all")
    if mode not in ("detect", "none"):
        raise job.fail_input("symmetry.mode", f"symmetry.mode must be detect or none; {mode!r}")
    if mode == "none":
        return lambda g: bool(np.allclose(g, np.eye(2), atol=1e-9))
    if allowed == "all":
        return None
    if allowed == "diagonal":
        return lambda g: bool(abs(g[0, 1]) < 1e-9 and abs(g[1, 0]) < 1e-9)
    try:
        mats = [np.asarray(m, dtype=float).reshape(2, 2) for m in allowed]
    except (TypeError, ValueError):
        raise job.fail_input(
            "symmetry.allowed",
            "symmetry.allowed must be 'all', 'diagonal' or a list of 2 x 2 matrices",
        ) from None
    return lambda g: any(np.allclose(g, m, atol=1e-9) for m in mats)


def _label(value: Any) -> Hashable:
    return tuple(_label(v) for v in value) if isinstance(value, list) else value


def _by_tag(
    job: Job, tags: Mapping[str, tuple[int, int]]
) -> tuple[dict[int, Hashable] | None, dict[int, list[float]] | None]:
    classes_spec = job.parameter("symmetry.classes", None)
    classes = None
    if classes_spec is not None:
        if not isinstance(classes_spec, Mapping):
            raise job.fail_input("symmetry.classes", "symmetry.classes must be an object")
        classes = {
            t: (_label(classes_spec[n]) if n in classes_spec else ("own", n))
            for n, (_, t) in tags.items()
        }
    directions_spec = _mapping(job, "symmetry.directions")
    directions = {}
    for n, d in directions_spec.items():
        if n not in tags:
            continue
        vec = np.asarray(d, dtype=float).reshape(-1)
        if vec.shape != (3,) or not np.all(np.isfinite(vec)):
            raise job.fail_input(f"symmetry.directions.{n}", "a direction is [x, y, z]")
        directions[tags[n][1]] = vec.tolist()
    return classes, directions or None


def _limits(job: Job) -> MeshLimits:
    order = job.parameter("mesh.element_order", 1)
    if order != 1:
        raise GmshJobError(
            "input",
            "mesh.element_order",
            "the symmetric periodic mesh is made of linear simplices: mesh.element_order "
            f"must be 1; found {order!r}",
            {"element_order": order},
        )
    options = _mapping(job, "mesh.options")
    unknown = sorted(set(options) - _MESH_OPTIONS)
    if unknown:
        raise job.fail_input("mesh.options", f"mesh options not accepted: {unknown}")
    opts = {k: check_number(f"mesh.options.{k}", v) for k, v in options.items()}
    for key, gmsh_key in (("sizes.min_size", "Mesh.MeshSizeMin"),
                          ("sizes.max_size", "Mesh.MeshSizeMax")):  # fmt: skip
        value = job.parameter(key, None)
        if value is not None:
            opts[gmsh_key] = check_number(key, value, minimum=0, strict=True)
    max_elements = job.parameter("mesh.max_elements", DEFAULT_MAX_ELEMENTS)
    if isinstance(max_elements, bool) or not isinstance(max_elements, int) or max_elements < 1:
        raise job.fail_input("mesh.max_elements", "mesh.max_elements must be a positive integer")
    return MeshLimits(
        max_elements=max_elements,
        options={**DEFAULT_MESH_OPTIONS, **opts},
        check=job.messages.check,
    )


def _sizes_report(sizing: MeshSizing) -> dict[str, Any]:
    edges = [h for v in (sizing.curve_edges or {}).values() for h in np.ravel(v)]
    return {
        "bulk": dict(sizing.bulk),
        "h_min": sizing.h_min,
        "h_max": sizing.h_max,
        "slope": sizing.slope,
        "refined_curves": len(sizing.curve_edges or {}),
        "edge_min": min(edges) if edges else None,
        "edge_max": max(edges) if edges else None,
    }


def main(job: Job) -> None:
    """The job (see the module documentation)."""
    source = job.input("geometry", (".xao",))
    unit = job.parameter("unit", "")
    above = _layer(job, "layers.above")
    below = _layer(job, "layers.below")
    bare = job.parameter("layers.bare_bottom", None)
    sizes, refined = _sizes(job, [x for x in (above, below) if x is not None])
    rule = _edge_rule(job)
    admissible = _allowed(job)
    limits = _limits(job)
    numbering = job.parameter("symmetry.numbering", True)
    tol = job.parameter("symmetry.tolerance", None)
    if tol is not None:
        tol = check_number("symmetry.tolerance", tol, minimum=0, strict=True)
    window_lo = job.parameter("symmetry.window_lo", [0.0, 0.0])
    output = job.parameter("mesh.output", DEFAULT_OUTPUT)
    if not isinstance(output, str) or "/" in output or not output.endswith(".msh"):
        raise job.fail_input("mesh.output", "mesh.output must be a file name ending in .msh")

    geometry = NamedGeometry.load(source, unit)
    job.messages.check("the reading of the geometry")
    facts = read_cell(geometry)
    lattice = np.array(facts.prism.lattice, dtype=float)
    roles = add_layers(
        geometry,
        lattice,
        facts.prism.height,
        None if above is None else above[0],
        None if below is None else below[0],
        bare if below is None else None,
    )
    job.messages.check("the addition of the layers")
    tags = {gmsh.model.getPhysicalName(d, t): (d, t) for d, t in gmsh.model.getPhysicalGroups()}
    names = {t: n for n, (_, t) in tags.items()}
    sizing = cell_sizing(sizes, refined, lattice, rule)
    classes, directions = _by_tag(job, tags)
    path = job.output("mesh", output)
    try:
        result = build_symmetric_mesh(
            path,
            lattice,
            sizing,
            window_lo=window_lo,
            classes=classes,
            directions=directions,
            tol=tol,
            numbering=bool(numbering),
            admissible=admissible,
            limits=limits,
        )
    except (SymmetricMeshError, MeshTopologyError) as error:
        raise GmshJobError(
            "gmsh",
            "mesh.symmetric" if isinstance(error, SymmetricMeshError) else "mesh.topology",
            f"the symmetric periodic mesh of the cell failed: {error}",
            {"error": str(error)},
        ) from None
    order = check_periodic_facet_order(path)
    if not order.ok:
        raise GmshJobError(
            "gmsh",
            "mesh.periodic_order",
            f"the mesh fails the periodic facet order check: {order.summary()}",
            {"summary": order.summary()},
        )
    z_low = -below[0].thickness if below else 0.0
    names_all = {**names, **{tag: n for n, (_, tag) in result.physical.items()}}
    job.report.update(
        unit=unit,
        cell={
            "lattice": lattice.tolist(),
            "height": facts.prism.height,
            "z": [z_low, facts.prism.height + (above[0].thickness if above else 0.0)],
        },
        layers=roles,
        sizes=_sizes_report(sizing),
        symmetry=symmetry_to_json(result.symmetry, names_all),
        detected=None if result.detected is None else symmetry_to_json(result.detected, names_all),
        asymmetric_unit=unit_to_json(result.unit),
        mesh={
            "file": output,
            "format": "MSH 2.2 ASCII, $Periodic",
            "nodes": result.n_nodes,
            "tetrahedra": result.n_tetrahedra,
            "unit_tetrahedra": result.n_unit_tetrahedra,
            "split_edges": result.n_split,
            "paired_surfaces": result.n_paired,
            "periodic_facets": order.n_facets,
            "summary": result.summary(),
        },
        physical={n: list(v) for n, v in result.physical.items()},
        notes=list(result.notes),
        timings=dict(result.timings),
    )


if __name__ == "__main__":
    run(main)
