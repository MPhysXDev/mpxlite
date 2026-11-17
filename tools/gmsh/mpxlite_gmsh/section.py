# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the GNU General Public
# License (the file COPYING of the directory above) for more details.

"""Layered rectangular sections in two dimensions: a structured core with layers.

A section lies in the plane ``z = 0``. Its **core** is the band
``y0 <= y <= y1`` cut along x into **columns** at the abscissae
``x_0 < x_1 < ... < x_n``; every column is a transfinite surface with
``cells`` cells along x (lengths in a geometric progression of ratio
``progression`` from left to right) and the same number of rows through the
band. A **layer** may lie above the core (``y1`` to ``y1 + height``) and
one below it (``y0 - height`` to ``y0``); each spans the whole width and is
either *structured* (a transfinite surface: ``rows`` uniform rows, one
uniform cell per core column cell across) or *free* (unstructured, sized
by the size law of :class:`SizeLaw`).

Every part is named: the columns by their surface group (columns may share
a group), each layer by its own group, the four sides of the core, the outer
curves of the layers and vertical lines through the core at column
boundaries by curve groups. :func:`parse` reads and checks the parameters
(a :class:`~mpxlite.GmshJobError` on every refusal); :func:`build` makes the
model and its mesh controls in the running Gmsh; :func:`edge_lengths` gives
the shortest and longest edge of the surface elements.

The entities are created in a fixed order (the core points of ``y0`` then of
``y1``, the core curves, the core surfaces, the layer above, the layer below)
and the groups in a fixed order (columns, layers, sides, outer curves,
lines): the same parameters give the same file, byte for byte.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import gmsh
import numpy as np

from mpxlite.gmshjob import GmshJobError
from mpxlite_gmsh.job import check_number

__all__ = [
    "ARRANGEMENTS",
    "DEFAULT_MAX_ELEMENTS",
    "ELEMENT_TYPES",
    "SIDES",
    "Column",
    "Layer",
    "Line",
    "Ramp",
    "Section",
    "SizeLaw",
    "build",
    "edge_lengths",
    "parse",
]

ELEMENT_TYPES = ("triangles", "quadrangles")
"""Element types of the surfaces (``quadrangles``: recombined)."""

ARRANGEMENTS = {
    "left": "Left",
    "right": "Right",
    "alternate_left": "AlternateLeft",
    "alternate_right": "AlternateRight",
}
"""Diagonals of the structured triangles -> Gmsh's arrangement of a transfinite surface."""

SIDES = ("core_top", "core_bottom", "core_left", "core_right", "outer")
"""The curve groups of :attr:`Section.boundaries`, in the order they are created."""

DEFAULT_MAX_ELEMENTS = 5_000_000
"""Default limit of the surface elements of a section."""

_MAX_ORDER = 2


@dataclasses.dataclass(frozen=True)
class Column:
    """A column of the core.

    Attributes:
        name: Its surface group.
        cells: Cells along x (at least 1).
        progression: Ratio of consecutive cell lengths, from left to right.
    """

    name: str
    cells: int
    progression: float = 1.0


@dataclasses.dataclass(frozen=True)
class Layer:
    """A layer above or below the core.

    Attributes:
        name: Its surface group.
        height: Its height (positive).
        structured: Transfinite (``rows`` uniform rows) or free (sized by
            the size law).
        rows: Rows of a structured layer.
    """

    name: str
    height: float
    structured: bool
    rows: int = 0


@dataclasses.dataclass(frozen=True)
class Line:
    """A curve group of vertical lines through the core.

    Attributes:
        name: Its curve group.
        x: Abscissae of the lines, each a column boundary.
    """

    name: str
    x: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class Ramp:
    """``size + rate * max(0, |x - centre| - half_width)``: a size along x.

    A ramp without ``rate`` is the constant ``size``.
    """

    size: float
    rate: float | None = None
    centre: float = 0.0
    half_width: float = 0.0

    def expression(self) -> str:
        """The ramp as a formula of Gmsh's ``MathEval`` field."""
        if self.rate is None:
            return _num(self.size)
        return (
            f"{_num(self.size)} + {_num(self.rate)}"
            f"*Max(0, Fabs(x - {_num(self.centre)}) - {_num(self.half_width)})"
        )


@dataclasses.dataclass(frozen=True)
class SizeLaw:
    """The element size in the free layers.

    ``size(x, y) = min(max, max(floor(x), base(x) + rate * d(y)))``, with
    ``floor`` and ``base`` the smallest of their ramps and ``d(y)`` the
    distance from the core (``max(y - y1, y0 - y)``). The mesh sizes are
    bounded by ``minimum`` and ``maximum`` (``Mesh.MeshSizeMin`` /
    ``Mesh.MeshSizeMax``).
    """

    floor: tuple[Ramp, ...]
    base: tuple[Ramp, ...]
    rate: float
    minimum: float
    maximum: float

    def expression(self, y0: float, y1: float) -> str:
        """The law as a formula of Gmsh's ``MathEval`` field."""
        distance = f"Max(y - {_num(y1)}, {_num(y0)} - y)"
        growth = f"{_fold(self.base)} + {_num(self.rate)}*{distance}"
        return f"Min({_num(self.maximum)}, Max({_fold(self.floor)}, {growth}))"


@dataclasses.dataclass(frozen=True)
class Section:
    """A layered section (see the module documentation).

    Attributes:
        x: Column boundaries, strictly increasing.
        columns: The columns, left to right.
        y: ``(y0, y1)`` of the core.
        rows: Rows of the core.
        above: The layer above the core, or ``None``.
        below: The layer below the core, or ``None``.
        sizes: The size law of the free layers (required with one).
        boundaries: Side -> curve group (:data:`SIDES`; a side not given
            has no group).
        lines: Vertical line groups.
        tags: Group name -> physical tag (the others numbered after).
        element_type: One of :data:`ELEMENT_TYPES`.
        arrangement: Diagonals of the structured triangles (a key of
            :data:`ARRANGEMENTS`).
        element_order: 1 or 2.
        scale: Output length per model length.
        max_elements: Limit of the surface elements.
    """

    x: tuple[float, ...]
    columns: tuple[Column, ...]
    y: tuple[float, float]
    rows: int
    above: Layer | None = None
    below: Layer | None = None
    sizes: SizeLaw | None = None
    boundaries: Mapping[str, str] = dataclasses.field(default_factory=dict)
    lines: tuple[Line, ...] = ()
    tags: Mapping[str, int] = dataclasses.field(default_factory=dict)
    element_type: str = "triangles"
    arrangement: str = "alternate_left"
    element_order: int = 1
    scale: float = 1.0
    max_elements: int = DEFAULT_MAX_ELEMENTS

    @property
    def layers(self) -> tuple[Layer, ...]:
        """The layers, above first."""
        return tuple(layer for layer in (self.above, self.below) if layer is not None)

    @property
    def x_cells(self) -> int:
        """Cells across the section."""
        return sum(c.cells for c in self.columns)

    def structured_elements(self) -> int:
        """Surface elements of the structured parts (exact, before meshing)."""
        per_cell = 1 if self.element_type == "quadrangles" else 2
        rows = self.rows + sum(layer.rows for layer in self.layers if layer.structured)
        return per_cell * self.x_cells * rows

    def group_names(self) -> list[tuple[int, str]]:
        """``(dimension, name)`` of every group, in the order they are created."""
        names: list[tuple[int, str]] = []
        for column in self.columns:
            if (2, column.name) not in names:
                names.append((2, column.name))
        names += [(2, layer.name) for layer in self.layers]
        for side in SIDES:
            if side in self.boundaries and (side != "outer" or self.layers):
                names.append((1, self.boundaries[side]))
        names += [(1, line.name) for line in self.lines]
        return names


def _num(value: float) -> str:
    text = repr(float(value))
    return f"({text})" if text.startswith("-") else text


def _fold(ramps: Sequence[Ramp]) -> str:
    text = ramps[0].expression()
    for ramp in ramps[1:]:
        text = f"Min({text}, {ramp.expression()})"
    return text


# -- parameters --------------------------------------------------------------


def _invalid(path: str, message: str, **data: Any) -> GmshJobError:
    return GmshJobError("input", "parameter.invalid", message, {"parameter": path, **data})


def _get(params: Mapping[str, Any], path: str, default: Any = ..., kind: type | None = None) -> Any:
    node: Any = params
    for key in path.split("."):
        if not isinstance(node, Mapping) or node.get(key) is None:
            if default is ...:
                raise GmshJobError(
                    "input",
                    "parameter.missing",
                    f"the section needs the parameter {path!r}",
                    {"parameter": path},
                )
            return default
        node = node[key]
    if kind is not None and not isinstance(node, kind):
        raise _invalid(path, f"the parameter {path!r} must be a {kind.__name__}; found {node!r}")
    return node


def _name(path: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise _invalid(path, f"{path} must be a non-empty name; found {value!r}")
    return value


def _count(path: str, value: Any, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _invalid(path, f"{path} must be an integer of at least {minimum}; found {value!r}")
    return value


def _ramps(params: Mapping[str, Any], path: str) -> tuple[Ramp, ...]:
    items = _get(params, path, kind=list)
    if not items:
        raise _invalid(path, f"{path} must hold at least one ramp")
    ramps = []
    for k, item in enumerate(items):
        where = f"{path}[{k}]"
        if not isinstance(item, Mapping):
            raise _invalid(where, f"{where} must be {{'size': h, 'rate': r, ...}}")
        unknown = sorted(set(item) - {"size", "rate", "centre", "half_width"})
        if unknown:
            raise _invalid(where, f"unknown fields of {where}: {unknown}")
        size = check_number(f"{where}.size", item.get("size"), minimum=0, strict=True)
        rate = item.get("rate")
        if rate is not None:
            rate = check_number(f"{where}.rate", rate, minimum=0)
        centre = check_number(f"{where}.centre", item.get("centre", 0.0))
        half = check_number(f"{where}.half_width", item.get("half_width", 0.0), minimum=0)
        ramps.append(Ramp(size, rate, centre, half))
    return tuple(ramps)


def _size_law(params: Mapping[str, Any]) -> SizeLaw | None:
    if _get(params, "sizes", None) is None:
        return None
    _get(params, "sizes", kind=Mapping)
    minimum = check_number("sizes.min", _get(params, "sizes.min"), minimum=0, strict=True)
    maximum = check_number("sizes.max", _get(params, "sizes.max"), minimum=minimum)
    rate = check_number("sizes.rate", _get(params, "sizes.rate"), minimum=0)
    floor, base = _ramps(params, "sizes.floor"), _ramps(params, "sizes.base")
    return SizeLaw(floor, base, rate, minimum, maximum)


def _layer(params: Mapping[str, Any], path: str) -> Layer | None:
    spec = _get(params, path, None)
    if spec is None:
        return None
    if not isinstance(spec, Mapping):
        raise _invalid(path, f"{path} must be {{'name', 'height', 'mesh', 'rows'}}")
    name = _name(f"{path}.name", spec.get("name"))
    height = check_number(f"{path}.height", spec.get("height"), minimum=0, strict=True)
    mesh = spec.get("mesh", "free")
    if mesh not in ("structured", "free"):
        raise _invalid(
            f"{path}.mesh", f"{path}.mesh must be 'structured' or 'free'; found {mesh!r}"
        )
    if mesh == "structured":
        return Layer(name, height, True, _count(f"{path}.rows", spec.get("rows")))
    if spec.get("rows") is not None:
        raise _invalid(f"{path}.rows", f"{path}.rows is for a structured layer")
    return Layer(name, height, False)


def _columns(params: Mapping[str, Any]) -> tuple[tuple[float, ...], tuple[Column, ...]]:
    xs = _get(params, "columns.x", kind=list)
    items = _get(params, "columns.cells", kind=list)
    x = tuple(check_number(f"columns.x[{k}]", v) for k, v in enumerate(xs))
    if len(x) < 2 or len(items) != len(x) - 1:
        raise _invalid(
            "columns",
            "columns.x must hold n + 1 >= 2 abscissae and columns.cells n columns; "
            f"found {len(x)} and {len(items)}",
        )
    if any(b <= a for a, b in itertools.pairwise(x)):
        raise GmshJobError(
            "rule",
            "section.order",
            "the column boundaries columns.x must be strictly increasing",
            {"x": list(x)},
        )
    columns = []
    for k, item in enumerate(items):
        where = f"columns.cells[{k}]"
        if not isinstance(item, Mapping):
            raise _invalid(where, f"{where} must be {{'name', 'cells', 'progression'}}")
        unknown = sorted(set(item) - {"name", "cells", "progression"})
        if unknown:
            raise _invalid(where, f"unknown fields of {where}: {unknown}")
        columns.append(
            Column(
                _name(f"{where}.name", item.get("name")),
                _count(f"{where}.cells", item.get("cells")),
                check_number(
                    f"{where}.progression", item.get("progression", 1.0), minimum=0, strict=True
                ),
            )
        )
    return x, tuple(columns)


def _lines(params: Mapping[str, Any], x: Sequence[float]) -> tuple[Line, ...]:
    items = _get(params, "lines", [], kind=list)
    tol = 1e-9 * (x[-1] - x[0])
    lines = []
    for k, item in enumerate(items):
        where = f"lines[{k}]"
        if not isinstance(item, Mapping) or not isinstance(item.get("x"), list) or not item["x"]:
            raise _invalid(where, f"{where} must be {{'name': <curve group>, 'x': [x, ...]}}")
        name = _name(f"{where}.name", item.get("name"))
        at = tuple(check_number(f"{where}.x[{j}]", v) for j, v in enumerate(item["x"]))
        off = [v for v in at if min(abs(b - v) for b in x) > tol]
        if off:
            raise GmshJobError(
                "rule",
                "lines.off_column",
                f"the line group {name!r} has lines off the column boundaries: {off}",
                {"name": name, "x": off},
            )
        lines.append(Line(name, at))
    return tuple(lines)


def _check_names(section: Section) -> None:
    seen: dict[str, str] = {}
    kinds = [("column", c.name) for c in section.columns]
    kinds += [("layer", layer.name) for layer in section.layers]
    kinds += [(f"side {side}", name) for side, name in section.boundaries.items()]
    kinds += [("line", line.name) for line in section.lines]
    for kind, name in kinds:
        first = seen.setdefault(name, kind)
        if first != kind or (kind != "column" and kinds.count((kind, name)) > 1):
            raise GmshJobError(
                "rule",
                "names.twice",
                f"the name {name!r} is given to two groups ({first} and {kind})",
                {"name": name, "groups": [first, kind]},
            )
    names = {n for _, n in section.group_names()}
    unknown = sorted(set(section.tags) - names)
    if unknown:
        raise GmshJobError(
            "input",
            "tags.unknown",
            f"physical tags given for groups the section does not have: {unknown}",
            {"names": unknown},
        )
    dims = dict((n, d) for d, n in section.group_names())
    used: dict[tuple[int, int], str] = {}
    for name, tag in section.tags.items():
        other = used.setdefault((dims[name], tag), name)
        if other != name:
            raise GmshJobError(
                "rule",
                "tags.twice",
                f"the physical tag {tag} of dimension {dims[name]} is given to {other!r} "
                f"and {name!r}",
                {"tag": tag, "names": [other, name]},
            )


def parse(params: Mapping[str, Any]) -> Section:
    """The section of the parameters of the job ``layered_section_mesh``.

    Raises:
        GmshJobError: kind ``input`` (``parameter.missing``,
            ``parameter.invalid``, ``sizes.missing``, ``mesh.element_order``,
            ``tags.unknown``) or ``rule`` (``section.order``,
            ``lines.off_column``, ``names.twice``, ``tags.twice``).
    """
    scale = check_number("scale", _get(params, "scale", 1.0), minimum=0, strict=True)
    x, columns = _columns(params)
    y = _get(params, "core.y", kind=list)
    if len(y) != 2:
        raise _invalid("core.y", f"core.y must be [y0, y1]; found {y!r}")
    y0, y1 = (check_number(f"core.y[{k}]", v) for k, v in enumerate(y))
    if y1 <= y0:
        raise GmshJobError("rule", "section.order", "the core needs y0 < y1", {"y": [y0, y1]})
    rows = _count("core.rows", _get(params, "core.rows"))
    above, below = _layer(params, "above"), _layer(params, "below")
    sizes = _size_law(params)
    free = [layer.name for layer in (above, below) if layer is not None and not layer.structured]
    if free and sizes is None:
        raise GmshJobError(
            "input",
            "sizes.missing",
            f"the free layers {free} need the size law 'sizes'",
            {"names": free},
        )
    if sizes is not None and not free:
        raise _invalid("sizes", "the size law 'sizes' applies to free layers; there is none")
    boundaries = _get(params, "boundaries", {}, kind=Mapping)
    unknown = sorted(set(boundaries) - set(SIDES))
    if unknown:
        raise _invalid("boundaries", f"unknown sides {unknown}; the sides are {list(SIDES)}")
    sides = {s: _name(f"boundaries.{s}", boundaries[s]) for s in SIDES if s in boundaries}
    tags = _get(params, "tags", {}, kind=Mapping)
    tags = {str(n): _count(f"tags.{n}", t) for n, t in tags.items()}
    element_type = _get(params, "mesh.element_type", "triangles")
    if element_type not in ELEMENT_TYPES:
        raise _invalid("mesh.element_type", f"mesh.element_type must be one of {ELEMENT_TYPES}")
    arrangement = _get(params, "mesh.diagonals", "alternate_left")
    if arrangement not in ARRANGEMENTS:
        raise _invalid("mesh.diagonals", f"mesh.diagonals must be one of {sorted(ARRANGEMENTS)}")
    order = _get(params, "mesh.element_order", 1)
    if isinstance(order, bool) or not isinstance(order, int) or not 1 <= order <= _MAX_ORDER:
        raise GmshJobError(
            "input",
            "mesh.element_order",
            f"mesh.element_order must be 1 or 2; found {order!r}",
            {"element_order": order},
        )
    limit = _get(params, "mesh.max_elements", DEFAULT_MAX_ELEMENTS)
    section = Section(
        x=x,
        columns=columns,
        y=(y0, y1),
        rows=rows,
        above=above,
        below=below,
        sizes=sizes,
        boundaries=sides,
        lines=_lines(params, x),
        tags=tags,
        element_type=element_type,
        arrangement=arrangement,
        element_order=order,
        scale=scale,
        max_elements=_count("mesh.max_elements", limit),
    )
    _check_names(section)
    return section


# -- model and mesh ----------------------------------------------------------


def _options(section: Section) -> None:
    number = gmsh.option.setNumber
    quad = section.element_type == "quadrangles"
    number("Mesh.RandomSeed", 1)
    number("Mesh.Algorithm", 8 if quad else 6)  # Frontal-Delaunay for quads / Frontal-Delaunay
    number("Mesh.RecombinationAlgorithm", 1)  # Blossom
    number("Mesh.RecombineAll", 0)
    number("Mesh.MeshSizeFromPoints", 0)
    number("Mesh.MeshSizeFromCurvature", 0)
    number("Mesh.MeshSizeExtendFromBoundary", 0)
    if section.sizes is not None:
        number("Mesh.MeshSizeMax", section.sizes.maximum)
        number("Mesh.MeshSizeMin", section.sizes.minimum)
    number("Mesh.ElementOrder", section.element_order)
    number("Mesh.MshFileVersion", 2.2)
    number("Mesh.Binary", 0)
    number("Mesh.SaveAll", 0)
    number("Mesh.SaveParametric", 0)
    number("Mesh.ScalingFactor", section.scale)


def _transfinite(section: Section, surface: int, corners: list[int]) -> None:
    gmsh.model.mesh.setTransfiniteSurface(
        surface, arrangement=ARRANGEMENTS[section.arrangement], cornerTags=corners
    )
    if section.element_type == "quadrangles":
        gmsh.model.mesh.setRecombine(2, surface)


@dataclasses.dataclass
class _Entities:
    """The entity tags of a section model, in the order of creation."""

    pb: list[int]  # core points on y0, left to right
    pt: list[int]  # core points on y1
    hb: list[int]  # core curves on y0
    ht: list[int]  # core curves on y1
    vl: list[int]  # vertical core curves, left to right
    core: list[int]  # column surfaces
    # (layer, surface, (side, across, other side), corner points)
    layers: list[tuple[Layer, int, tuple[int, int, int], list[int]]]
    outer: list[int]  # outer curves of the layers


def _geometry(section: Section) -> _Entities:
    geo = gmsh.model.geo
    y0, y1 = section.y
    xs = section.x
    n = len(section.columns)
    pb = [geo.addPoint(x, y0, 0.0) for x in xs]
    pt = [geo.addPoint(x, y1, 0.0) for x in xs]
    hb = [geo.addLine(pb[i], pb[i + 1]) for i in range(n)]
    ht = [geo.addLine(pt[i], pt[i + 1]) for i in range(n)]
    vl = [geo.addLine(pb[i], pt[i]) for i in range(n + 1)]
    core = []
    for i in range(n):
        loop = geo.addCurveLoop([hb[i], vl[i + 1], -ht[i], -vl[i]])
        core.append(geo.addPlaneSurface([loop]))
    e = _Entities(pb, pt, hb, ht, vl, core, [], [])
    if section.above is not None:
        y = y1 + section.above.height
        ptl, ptr = geo.addPoint(xs[0], y, 0.0), geo.addPoint(xs[-1], y, 0.0)
        right, top, left = geo.addLine(pt[-1], ptr), geo.addLine(ptr, ptl), geo.addLine(ptl, pt[0])
        surface = geo.addPlaneSurface([geo.addCurveLoop([*ht, right, top, left])])
        e.outer += [right, top, left]
        e.layers.append((section.above, surface, (right, top, left), [pt[0], pt[-1], ptr, ptl]))
    if section.below is not None:
        y = y0 - section.below.height
        pbl, pbr = geo.addPoint(xs[0], y, 0.0), geo.addPoint(xs[-1], y, 0.0)
        left, bottom, right = (
            geo.addLine(pb[0], pbl),
            geo.addLine(pbl, pbr),
            geo.addLine(pbr, pb[-1]),
        )
        loop = geo.addCurveLoop([left, bottom, right, *[-c for c in reversed(hb)]])
        surface = geo.addPlaneSurface([loop])
        e.outer += [left, bottom, right]
        e.layers.append((section.below, surface, (left, bottom, right), [pbl, pbr, pb[-1], pb[0]]))
    geo.synchronize()
    return e


def _mesh_controls(section: Section, e: _Entities) -> None:
    mesh = gmsh.model.mesh
    for i, column in enumerate(section.columns):
        for curve in (e.hb[i], e.ht[i]):
            mesh.setTransfiniteCurve(curve, column.cells + 1, "Progression", column.progression)
    for curve in e.vl:
        mesh.setTransfiniteCurve(curve, section.rows + 1)
    for i, surface in enumerate(e.core):
        _transfinite(section, surface, [e.pb[i], e.pb[i + 1], e.pt[i + 1], e.pt[i]])
    for layer, surface, (side_a, across, side_b), corners in e.layers:
        if layer.structured:
            mesh.setTransfiniteCurve(side_a, layer.rows + 1)
            mesh.setTransfiniteCurve(side_b, layer.rows + 1)
            mesh.setTransfiniteCurve(across, section.x_cells + 1)
            _transfinite(section, surface, corners)
    free = [surface for layer, surface, _, _ in e.layers if not layer.structured]
    if free and section.sizes is not None:  # parse() refuses free layers without sizes
        field = mesh.field.add("MathEval")
        mesh.field.setString(field, "F", section.sizes.expression(*section.y))
        mesh.field.setAsBackgroundMesh(field)
        if section.element_type == "quadrangles":
            for surface in free:
                mesh.setRecombine(2, surface)


def _physical_groups(section: Section, e: _Entities) -> dict[str, tuple[int, int]]:
    entities: dict[tuple[int, str], list[int]] = {}
    for column, surface in zip(section.columns, e.core, strict=True):
        entities.setdefault((2, column.name), []).append(surface)
    for layer, surface, _, _ in e.layers:
        entities[(2, layer.name)] = [surface]
    curves = {
        "core_top": e.ht,
        "core_bottom": e.hb,
        "core_left": [e.vl[0]],
        "core_right": [e.vl[-1]],
        "outer": e.outer,
    }
    for side in SIDES:
        if side in section.boundaries and curves[side]:
            entities[(1, section.boundaries[side])] = curves[side]
    xs = np.asarray(section.x)
    for line in section.lines:
        entities[(1, line.name)] = [e.vl[int(np.argmin(np.abs(xs - x)))] for x in line.x]

    given = dict(section.tags)
    next_tag = max(given.values(), default=0) + 1
    groups: dict[str, tuple[int, int]] = {}
    for (dim, name), ents in entities.items():
        tag = given.get(name)
        if tag is None:
            tag, next_tag = next_tag, next_tag + 1
        gmsh.model.addPhysicalGroup(dim, ents, tag, name)
        groups[name] = (dim, tag)
    return groups


def build(section: Section) -> dict[str, tuple[int, int]]:
    """Makes the model of ``section`` and its mesh controls in the running Gmsh.

    The mesh is then generated by ``gmsh.model.mesh.generate(2)``.

    Returns:
        Group name -> ``(dimension, physical tag)``, in the order of creation.
    """
    gmsh.model.add("section")
    _options(section)
    entities = _geometry(section)
    _mesh_controls(section, entities)
    return _physical_groups(section, entities)


def edge_lengths(scale: float = 1.0) -> tuple[float, float, int]:
    """Shortest and longest corner-to-corner edge of the surface elements, and their count.

    Args:
        scale: Factor on the coordinates (output length per model length),
            applied before the lengths are measured, as on writing.
    """
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    xy = coords.reshape(-1, 3)[:, :2] * scale
    order = np.argsort(node_tags)
    sorted_tags = node_tags[order]
    shortest, longest, count = math.inf, 0.0, 0
    types, _, nodes = gmsh.model.mesh.getElements(2)
    for etype, enodes in zip(types, nodes, strict=True):
        _, _, _, n_nodes, _, n_primary = gmsh.model.mesh.getElementProperties(etype)
        conn = enodes.reshape(-1, n_nodes)[:, :n_primary]
        pts = xy[order[np.searchsorted(sorted_tags, conn)]]
        lengths = np.linalg.norm(pts - np.roll(pts, -1, axis=1), axis=2)
        shortest = min(shortest, float(lengths.min()))
        longest = max(longest, float(lengths.max()))
        count += conn.shape[0]
    return shortest, longest, count
