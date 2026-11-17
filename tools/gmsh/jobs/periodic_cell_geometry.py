# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the GNU General Public
# License (the file COPYING of the directory above) for more details.

"""Job ``periodic_cell_geometry``: a named periodic prism cell, made conformal and read.

Input ``geometry``: an XAO file whose parts are physical groups with names,
volumes and surfaces (they may overlap or touch: the job makes them
conform, every piece keeping the name of its part).

Steps: open the file; make the parts conform
(:meth:`~mpxlite_gmsh.named.NamedGeometry.conform`); check the rules R1 to R6
of a periodic prism cell in the unit of the file
(:func:`~mpxlite_gmsh.prism.accept_cell`); scale the model by ``scale``;
classify its outer faces; measure the bridging strips (rules L1 to L6,
:func:`~mpxlite_gmsh.strips.measure_strip`); write the conformal named
geometry (XAO, scaled).

Parameters (all optional)::

    {"scale": 0.001,          # output length = scale x file length (default 1)
     "unit": "mm",            # name of the unit of the file, for messages
     "output_unit": "m",      # name of the output unit, for messages and face
                              # labels (default: unit if scale is 1, else "")
     "tolerance": 1e-6,       # relative geometric tolerance (default 1e-6)
     "strips": [{"name": "s", "ends": ["a", "b", ...]}, ...]}

Output ``geometry``: ``cell.xao``. The report: ``cell``, ``parts``, ``faces``,
``periodic``, ``strips`` (lengths in output units; see the developer
documentation for the exact schema).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gmsh
import numpy as np
from mpxlite_gmsh import named
from mpxlite_gmsh.job import Job, run
from mpxlite_gmsh.named import NamedGeometry
from mpxlite_gmsh.prism import CellFacts, accept_cell, read_cell
from mpxlite_gmsh.strips import measure_strip

from mpxlite.gmshjob import GmshJobError

OUTPUT = "cell.xao"


def _strip_requests(job: Job) -> list[tuple[str, list[str]]]:
    """``(strip name, end surface names)`` of the parameter ``strips``."""
    value = job.parameter("strips", [])
    if not isinstance(value, list):
        raise job.fail_input("strips", "the parameter 'strips' must be a list")
    out = []
    for k, item in enumerate(value):
        ok = (
            isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and isinstance(item.get("ends"), list)
            and all(isinstance(n, str) for n in item["ends"])
        )
        if not ok:
            raise job.fail_input(
                f"strips[{k}]",
                f"strips[{k}] must be {{'name': <surface>, 'ends': [<surface>, ...]}}; "
                f"found {item!r}",
            )
        out.append((item["name"], list(item["ends"])))
    return out


def _piece(tag: int, surface_of: dict[int, list[str]], volume_of: dict[int, str]) -> dict:
    occ = gmsh.model.occ
    up, _ = gmsh.model.getAdjacencies(2, tag)
    return {
        "area": occ.getMass(2, tag),
        "centre": list(occ.getCenterOfMass(2, tag)),
        "surfaces": sorted(surface_of.get(tag, [])),
        "volumes": sorted({volume_of.get(int(v), "") for v in up}),
    }


def _faces(facts: CellFacts, matched: list[dict]) -> dict:
    surface_of: dict[int, list[str]] = {}
    for name, tags in facts.surfaces.items():
        for t in tags:
            surface_of.setdefault(t, []).append(name)
    volume_of = {t: n for n, tags in facts.volumes.items() for t in tags}

    def face(tags: tuple[int, ...]) -> dict:
        pieces = [_piece(t, surface_of, volume_of) for t in tags]
        return {"area": float(sum(p["area"] for p in pieces)), "pieces": pieces}

    opposite = {m["face"]: m["opposite"] for m in matched}
    opposite.update({m["opposite"]: m["face"] for m in matched})
    sides = []
    for label, normal, offset, shift in facts.prism.faces():
        sides.append(
            {
                "face": label,
                "normal": normal.tolist(),
                "offset": offset,
                "opposite": opposite.get(label),
                "translation": shift.tolist(),
                **face(facts.sides[label]),
            }
        )
    return {"bottom": face(facts.bottom), "top": face(facts.top), "sides": sides}


def _parts(facts: CellFacts) -> list[dict]:
    occ = gmsh.model.occ
    out = []
    for groups, dim in ((facts.volumes, 3), (facts.surfaces, 2)):
        for name, tags in groups.items():
            masses = np.array([occ.getMass(dim, t) for t in tags])
            centres = np.array([occ.getCenterOfMass(dim, t) for t in tags])
            boxes = np.array([np.reshape(gmsh.model.getBoundingBox(dim, t), (2, 3)) for t in tags])
            on = {
                "bottom": sum(occ.getMass(2, t) for t in tags if t in facts.bottom),
                "top": sum(occ.getMass(2, t) for t in tags if t in facts.top),
                "sides": sum(
                    occ.getMass(2, t) for t in tags for s in facts.sides.values() if t in s
                ),
            }
            out.append(
                {
                    "name": name,
                    "dim": dim,
                    "pieces": len(tags),
                    "measure": float(masses.sum()),
                    "centre": np.average(centres, axis=0, weights=masses).tolist(),
                    "bounding_box": [
                        boxes[:, 0].min(axis=0).tolist(),
                        boxes[:, 1].max(axis=0).tolist(),
                    ],
                    **({"on": on} if dim == 2 else {}),
                }
            )
    return out


def _cell(facts: CellFacts, scale: float) -> dict:
    p = facts.prism
    area = p.pu * p.pv * math.sin(p.alpha)
    return {
        "lattice": [list(v) for v in p.lattice],
        "sides": [p.pu, p.pv],
        "angle": p.alpha,
        "angle_degrees": math.degrees(p.alpha),
        "rectangular": p.rectangular,
        "height": p.height,
        "footprint_area": area,
        "volume": area * p.height,
        "scale": scale,
    }


def main(job: Job) -> None:
    """The job (see the module documentation)."""
    source = job.input("geometry", (".xao",))
    scale = job.number("scale", 1.0, minimum=0.0, strict=True)
    unit = job.parameter("unit", "")
    if not isinstance(unit, str):
        raise job.fail_input("unit", "the parameter 'unit' must be a string")
    output_unit = job.parameter("output_unit", unit if scale == 1.0 else "")
    if not isinstance(output_unit, str):
        raise job.fail_input("output_unit", "the parameter 'output_unit' must be a string")
    rtol = job.number("tolerance", named.RTOL, minimum=0.0, strict=True)
    named.RTOL = rtol
    strips = _strip_requests(job)

    geometry = NamedGeometry.load(source, unit)
    job.messages.check("the reading of the geometry")
    geometry.conform()
    job.messages.check("the conforming fragment of the parts")
    facts, matched = accept_cell(geometry)
    if scale != 1.0:
        geometry.scale(scale)
        job.messages.check("the scaling of the model")
    geometry.unit = output_unit
    facts = read_cell(geometry)  # the tags of a scaled model are new; labels in output units
    faces = facts.prism.faces()
    matched = [
        {"face": faces[k][0], "opposite": faces[k + 2][0], "pieces": m["pieces"]}
        for k, m in enumerate(matched)
    ]
    surfaces = dict(facts.surfaces)
    measured = {}
    for name, ends in strips:
        unknown = [n for n in ends if n not in surfaces]
        if unknown:
            raise GmshJobError(
                "input",
                "strip.unknown_end",
                f"the end surfaces {unknown} of the strip {name!r} are not named surfaces "
                f"of the geometry; named surfaces: {sorted(surfaces)}",
                {"strip": name, "unknown": unknown},
            )
        shape = measure_strip(facts.prism, surfaces, name, {n: surfaces[n] for n in ends})
        measured[name] = {
            "direction": list(shape.direction),
            "width": shape.width,
            "length": shape.length,
            "centre": list(shape.centre),
            "ends": list(shape.ends),
            "area": shape.area,
        }
    job.messages.check("the measurement of the strips")
    geometry.write(job.output("geometry", OUTPUT))
    job.messages.check("the writing of the geometry")
    job.report.update(
        unit=unit,
        output_unit=output_unit,
        scale=scale,
        tolerance={"relative": rtol, "length": facts.prism.tol},
        cell=_cell(facts, scale),
        parts=_parts(facts),
        faces=_faces(facts, matched),
        periodic={"matched": True, "pairs": matched},
        strips=measured,
    )


if __name__ == "__main__":
    run(main)
