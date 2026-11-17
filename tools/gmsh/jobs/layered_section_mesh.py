# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the GNU General Public
# License (the file COPYING of the directory above) for more details.

"""Job ``layered_section_mesh``: the mesh of a layered rectangular section in two dimensions.

No input file: the section is described by its parameters
(:mod:`mpxlite_gmsh.section`): a structured core band cut into columns, an
optional layer above and below it (structured, or free with a size law), the
names of the parts and their physical tags.

Steps: read and check the parameters (:func:`~mpxlite_gmsh.section.parse`);
refuse a section whose structured parts alone exceed the element limit;
build the model and mesh it (:func:`~mpxlite_gmsh.section.build`); count
the surface elements against the limit; write the mesh.

Parameters (lengths in model units; see the developer documentation for
every field and default)::

    {"scale": 1.0,     # output length = scale x model length
     "unit": "",       # name of the model unit, for messages
     "columns": {"x": [x0, x1, ..., xn],
                 "cells": [{"name": "<surface>", "cells": n, "progression": r}, ...]},
     "core": {"y": [y0, y1], "rows": m},
     "above": {"name": "<surface>", "height": h, "mesh": "structured" | "free",
               "rows": k},
     "below": {...},
     "sizes": {"floor": [{"size": a, "rate": b, "centre": c, "half_width": d}, ...],
               "base": [...], "rate": g, "min": h_min, "max": h_max},
     "boundaries": {"core_top": "<curve>", "core_bottom": ..., "core_left": ...,
                    "core_right": ..., "outer": ...},
     "lines": [{"name": "<curve>", "x": [x, ...]}],
     "tags": {"<name>": tag},
     "mesh": {"element_type": "triangles" | "quadrangles",
              "diagonals": "alternate_left", "element_order": 1,
              "max_elements": 5000000, "output": "section.msh"}}

Output ``mesh``: MSH 2.2 ASCII, in output units. The report: ``unit``,
``scale``, ``section``, ``mesh``, ``physical``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gmsh
from mpxlite_gmsh.job import Job, run
from mpxlite_gmsh.section import build, edge_lengths, parse

from mpxlite.gmshjob import GmshJobError

DEFAULT_OUTPUT = "section.msh"


def _too_many(count: int, limit: int, when: str) -> GmshJobError:
    key = "estimate" if when == "before" else "count"
    return GmshJobError(
        "gmsh",
        "mesh.too_many_elements",
        f"the section would hold {count} surface elements, above the limit of {limit}",
        {key: count, "limit": limit, "when": when},
    )


def main(job: Job) -> None:
    """The job (see the module documentation)."""
    unit = job.parameter("unit", "")
    output = job.parameter("mesh.output", DEFAULT_OUTPUT)
    if not isinstance(output, str) or "/" in output or not output.endswith(".msh"):
        raise job.fail_input("mesh.output", "mesh.output must be a file name ending in .msh")
    section = parse(job.parameters)
    estimate = section.structured_elements()
    if estimate > section.max_elements:
        raise _too_many(estimate, section.max_elements, "before")

    groups = build(section)
    job.messages.check("the construction of the section")
    gmsh.model.mesh.generate(2)
    job.messages.check("the mesh generation")
    shortest, longest, count = edge_lengths(section.scale)
    if count > section.max_elements:
        raise _too_many(count, section.max_elements, "after")
    path = job.output("mesh", output)
    gmsh.write(str(path))
    job.messages.check("the writing of the mesh")

    s = section.scale
    y0, y1 = section.y
    low = y0 - (section.below.height if section.below else 0.0)
    high = y1 + (section.above.height if section.above else 0.0)
    nodes = len(gmsh.model.mesh.getNodes()[0])
    job.report.update(
        unit=unit,
        scale=s,
        section={
            "x": [section.x[0] * s, section.x[-1] * s],
            "y": [low * s, high * s],
            "core": [y0 * s, y1 * s],
            "x_cells": section.x_cells,
            "rows": section.rows,
        },
        mesh={
            "file": output,
            "format": "MSH 2.2 ASCII",
            "nodes": nodes,
            "surface_elements": count,
            "edge_length": {"min": shortest, "max": longest},
        },
        physical={name: list(v) for name, v in groups.items()},
    )


if __name__ == "__main__":
    run(main)
