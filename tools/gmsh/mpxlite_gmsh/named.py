# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the GNU General Public
# License (the file COPYING of the directory above) for more details.

"""A named geometry: a Gmsh OpenCASCADE model whose parts are known by name.

The parts of a geometry are the physical groups of the model, volumes
(dimension 3) and surfaces (dimension 2), each with a name. Entity numbers
are never an identity across files: they change when a file is read back.
:class:`NamedGeometry` therefore addresses everything by name:

* :meth:`NamedGeometry.load` / :meth:`NamedGeometry.write`: an XAO file (the
  format that keeps the names);
* :meth:`NamedGeometry.conform`: one Boolean fragment of all the volumes and
  the named surfaces, every piece keeping the name of the part it came from;
  what is neither a volume nor part of a named surface is removed;
* :meth:`NamedGeometry.scale`: a scaling about the origin, names kept;
* :meth:`NamedGeometry.fragment_with`: new volumes fragmented with the model,
  the parts keeping their names (the parts must not be split);
* questions: measures, centres, bounding boxes, corner points, the faces on
  a plane, the volumes on each side of a face, whether two parts touch,
  whether faces match under an affine map, point-in-part, closest point.

The functions below the class work on the current model and its entity
tags; the class is the interface by names.

Geometric tolerance: lengths are compared to within ``1e-6`` times the
largest extent of the model (:data:`RTOL`), areas and volumes to within
``1e-6`` relative; entities are matched after an operation to within
``1e-7`` of the scale (the precision of OpenCASCADE).
"""

from __future__ import annotations

import itertools
import pathlib
import tempfile
from collections.abc import Mapping, Sequence

import gmsh
import numpy as np

from mpxlite.gmshjob import GmshJobError

__all__ = [
    "RTOL",
    "SAMPLES",
    "NamedGeometry",
    "boundary_curves",
    "curve_points",
    "curve_samples",
    "length_text",
    "model_extent",
    "on_surface",
    "point_text",
    "signature",
    "surface_samples",
    "vertices",
]

RTOL = 1.0e-6
"""Relative geometric tolerance (lengths: of the extent of the model).

A job may set it (``named.RTOL = value``) before it reads a geometry; the
functions of the toolkit read it when they run."""

SAMPLES = (0.25, 0.5, 0.75)
"""Parameters (fractions of the range) of the interior samples of a curve or a surface."""

MATCH_RTOL = 1.0e-7
"""Relative tolerance of the matching of entities after an operation."""


def length_text(value: float, unit: str) -> str:
    """Formats a length given in the unit of the model."""
    return "0" if value == 0.0 else f"{value:.6g} {unit}".rstrip()


def point_text(xyz: Sequence[float], unit: str) -> str:
    """Formats a point given in the unit of the model."""
    values = ", ".join(f"{c + 0.0:.6g}" for c in xyz)
    return f"({values}) {unit}".rstrip()


def vertices(dim: int, tag: int) -> np.ndarray:
    """Coordinates of the corner points of an entity, shape (n, 3)."""
    if dim == 0:
        return np.array([gmsh.model.getValue(0, tag, [])])
    points = gmsh.model.getBoundary([(dim, tag)], combined=False, oriented=False, recursive=True)
    if not points:
        return np.zeros((0, 3))
    return np.array([gmsh.model.getValue(0, t, []) for _, t in points])


def curve_samples(tag: int) -> np.ndarray:
    """Interior points of a curve, shape (3, 3)."""
    low, high = gmsh.model.getParametrizationBounds(1, tag)
    params = [low[0] + s * (high[0] - low[0]) for s in SAMPLES]
    return np.reshape(gmsh.model.getValue(1, tag, params), (-1, 3))


def curve_points(tag: int) -> np.ndarray:
    """Corner and interior points of a curve."""
    return np.vstack([vertices(1, tag), curve_samples(tag)])


def surface_samples(tag: int) -> tuple[np.ndarray, np.ndarray]:
    """Points and unit normals of a surface on a 3 x 3 parametric grid."""
    low, high = gmsh.model.getParametrizationBounds(2, tag)
    uv = []
    for su, sv in itertools.product(SAMPLES, SAMPLES):
        uv += [
            low[0] + su * (high[0] - low[0]),
            low[1] + sv * (high[1] - low[1]),
        ]
    points = np.reshape(gmsh.model.getValue(2, tag, uv), (-1, 3))
    normals = np.reshape(gmsh.model.getNormal(tag, uv), (-1, 3))
    return points, normals


def boundary_curves(tags: Sequence[int]) -> list[int]:
    """Curves bounding the union of the given surfaces."""
    curves = gmsh.model.getBoundary([(2, t) for t in tags], combined=True, oriented=False)
    return sorted({abs(t) for _, t in curves})


def model_extent() -> float:
    """Largest extent of the corner points of the model."""
    points = np.array([gmsh.model.getValue(0, t, []) for _, t in gmsh.model.getEntities(0)])
    if points.size == 0:
        return 1.0
    return float(np.max(np.ptp(points, axis=0))) or 1.0


def on_surface(point: np.ndarray, tag: int, tol: float) -> bool:
    """Whether a point lies on the closure of a (trimmed) surface."""
    closest, _ = gmsh.model.getClosestPoint(2, tag, list(point))
    if np.linalg.norm(np.asarray(closest) - point) > tol:
        return False
    if gmsh.model.isInside(2, tag, list(point)):
        return True
    for curve in boundary_curves([tag]):
        closest, _ = gmsh.model.getClosestPoint(1, curve, list(point))
        if np.linalg.norm(np.asarray(closest) - point) <= tol:
            return True
    return False


def signature(dim: int, tag: int, scale: float = 1.0) -> np.ndarray:
    """Centre of mass and measure of an entity, scaled by ``scale``."""
    occ = gmsh.model.occ
    centre = np.asarray(occ.getCenterOfMass(dim, tag)) * scale
    return np.array([*centre, occ.getMass(dim, tag) * scale**dim])


def _kind(dim: int) -> str:
    return "volume" if dim == 3 else "surface"


def _overlap(named: Mapping[str, list[tuple[int, int]]], kind: str) -> None:
    """Refuses two names sharing an entity after the fragment."""
    owner: dict[tuple[int, int], str] = {}
    for name, entities in named.items():
        for entity in entities:
            other = owner.setdefault(entity, name)
            if other != name:
                hint = (
                    "cut one by the other before naming them"
                    if kind == "volume"
                    else "a part of a surface can have one name only"
                )
                raise GmshJobError(
                    "rule",
                    "names.overlap",
                    f"the {kind}s '{other}' and '{name}' overlap; {hint}",
                    {"kind": kind, "names": [other, name]},
                )


def _close_downwards(entities: set[tuple[int, int]]) -> None:
    """Adds the boundary entities (curves, points) of every entity in place."""
    stack = list(entities)
    while stack:
        entity = stack.pop()
        if entity[0] == 0:
            continue
        for sub in gmsh.model.getBoundary(
            [entity], combined=False, oriented=False, recursive=False
        ):
            sub = (sub[0], abs(sub[1]))
            if sub not in entities:
                entities.add(sub)
                stack.append(sub)
        if entity[0] >= 2:
            # Embedded entities and internal edges are adjacencies too.
            _, down = gmsh.model.getAdjacencies(*entity)
            for tag in down:
                sub = (entity[0] - 1, int(tag))
                if sub not in entities:
                    entities.add(sub)
                    stack.append(sub)


def _remove_groups() -> None:
    """Removes every physical group and its name.

    ``removePhysicalGroups`` keeps the names; a name left behind would not be
    given to a group added again under another tag.
    """
    for dim, tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, tag)
        if name:
            gmsh.model.removePhysicalName(name)
    gmsh.model.removePhysicalGroups()


def _match_entities(
    before: Mapping[tuple[int, int], np.ndarray], scale: float, what: str
) -> dict[tuple[int, int], int]:
    """The entity of the current model matching each entity of ``before`` by signature."""
    now = {
        dim: [(t, signature(dim, t)) for _, t in gmsh.model.getEntities(dim)]
        for dim in sorted({d for d, _ in before})
    }
    out = {}
    for (dim, e), sig in before.items():
        hits = [
            t for t, s in now[dim]
            if np.max(np.abs(s[:3] - sig[:3])) <= MATCH_RTOL * scale
            and abs(s[3] - sig[3]) <= MATCH_RTOL * scale**dim
        ]  # fmt: skip
        if len(hits) != 1:
            raise GmshJobError(
                "gmsh",
                "names.lost",
                f"{what} lost the entity ({dim}, {e}) ({len(hits)} matches)",
                {"dim": dim, "matches": len(hits)},
            )
        out[(dim, e)] = hits[0]
    return out


class NamedGeometry:
    """The current Gmsh OpenCASCADE model, its parts known by name.

    Args:
        unit: Name of the length unit of the model, for messages only
            (``"mm"``; empty: lengths printed as numbers).
    """

    def __init__(self, unit: str = "") -> None:
        self.unit = unit

    # -- files ------------------------------------------------------------

    @classmethod
    def load(cls, path: pathlib.Path | str, unit: str = "") -> NamedGeometry:
        """Opens an XAO file in a new model (its physical groups are the parts).

        Raises:
            GmshJobError: kind ``input``: not an XAO file, or unreadable.
        """
        path = pathlib.Path(path)
        if path.suffix.lower() != ".xao":
            raise GmshJobError(
                "input",
                "input.format",
                f"a geometry file must be an XAO file (.xao); found {path.name}",
                {"path": str(path)},
            )
        try:
            gmsh.open(str(path))
        except Exception as error:  # the Gmsh API raises Exception
            raise GmshJobError(
                "input",
                "input.unreadable",
                f"the geometry file {path.name} cannot be read: {error}",
                {"path": str(path)},
            ) from None
        gmsh.model.occ.synchronize()
        return cls(unit)

    def write(self, path: pathlib.Path | str) -> pathlib.Path:
        """Writes the model, its names included, to an XAO file."""
        path = pathlib.Path(path)
        if path.suffix.lower() != ".xao":
            raise ValueError(f"a named geometry is written as XAO; got {path.name}")
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
        return path

    # -- names --------------------------------------------------------------

    @staticmethod
    def groups(dim: int | None = None) -> dict[str, tuple[int, tuple[int, ...]]]:
        """Name -> ``(dim, entity tags)`` of the parts (physical groups).

        Raises:
            GmshJobError: kind ``rule``: a group has no name (``names.unnamed``),
                is of dimension 0 or 1 (``names.dimension``), or one name is
                used in two dimensions (``names.twice``).
        """
        out: dict[str, tuple[int, set[int]]] = {}
        for d, tag in gmsh.model.getPhysicalGroups(-1 if dim is None else dim):
            name = gmsh.model.getPhysicalName(d, tag)
            if d < 2:
                raise GmshJobError(
                    "rule",
                    "names.dimension",
                    f"the physical group {name or tag!r} is of dimension {d}: the parts of "
                    "a geometry are volumes and surfaces",
                    {"name": name, "dim": d},
                )
            if not name:
                raise GmshJobError(
                    "rule",
                    "names.unnamed",
                    f"the physical group {tag} of dimension {d} has no name; "
                    f"every named {_kind(d)} needs a name",
                    {"dim": d, "tag": tag},
                )
            known = out.get(name)
            if known is not None and known[0] != d:
                raise GmshJobError(
                    "rule",
                    "names.twice",
                    f"the name '{name}' is used twice, for a volume and for a "
                    "surface; every part needs its own name",
                    {"name": name},
                )
            entities = gmsh.model.getEntitiesForPhysicalGroup(d, tag)
            out.setdefault(name, (d, set()))[1].update(int(t) for t in entities)
        return {n: (d, tuple(sorted(ts))) for n, (d, ts) in out.items()}

    def entities(self, name: str) -> tuple[int, ...]:
        """Entity tags of the part ``name`` (in the current model)."""
        return self.groups()[name][1]

    @staticmethod
    def set_groups(groups: Mapping[str, tuple[int, Sequence[int]]]) -> None:
        """Replaces the physical groups by ``name -> (dim, entities)`` (in order)."""
        _remove_groups()
        for tag, (name, (dim, entities)) in enumerate(groups.items(), start=1):
            gmsh.model.addPhysicalGroup(dim, sorted(set(entities)), tag, name)

    # -- operations ---------------------------------------------------------

    def conform(self) -> None:
        """Makes the parts conform, every piece keeping the name of its part.

        One Boolean fragment of every volume of the model and of the
        entities of the named surfaces; each name is carried to the entities
        its part became and the physical groups are rewritten. A surface
        that coincides with a face of a volume keeps its name. Points,
        curves and surfaces that are neither named nor part of a volume or
        of a named surface are removed.

        Raises:
            GmshJobError: kind ``rule``: an entity is in two groups of one
                dimension (``names.shared_entity``), no volume
                (``geometry.no_volume``), two named volumes or two named
                surfaces overlap (``names.overlap``), or a group is
                malformed (see :meth:`groups`).
        """
        occ = gmsh.model.occ
        named = self.groups()
        volume_tags = {n: list(ts) for n, (d, ts) in named.items() if d == 3}
        surface_tags = {n: list(ts) for n, (d, ts) in named.items() if d == 2}
        for tags_of, dim in ((volume_tags, 3), (surface_tags, 2)):
            owner: dict[int, str] = {}
            for name, tags in tags_of.items():
                for tag in tags:
                    other = owner.setdefault(tag, name)
                    if other != name:
                        raise GmshJobError(
                            "rule",
                            "names.shared_entity",
                            f"the {_kind(dim)} tag {tag} is given two names, "
                            f"'{other}' and '{name}'; a part has one name",
                            {"kind": _kind(dim), "names": [other, name]},
                        )
        inputs = [(3, t) for _, t in occ.getEntities(3)]
        inputs += [(2, t) for tags in surface_tags.values() for t in tags]
        inputs = list(dict.fromkeys(inputs))
        if not inputs or not any(d == 3 for d, _ in inputs):
            raise GmshJobError("rule", "geometry.no_volume", "the model has no volume to name")
        if len(inputs) > 1:
            _, out_map = occ.fragment(inputs[:1], inputs[1:])
            images = {e: list(out) for e, out in zip(inputs, out_map, strict=True)}
        else:
            images = {inputs[0]: [inputs[0]]}
        occ.synchronize()

        named_volumes = {
            name: sorted({e for t in tags for e in images[(3, t)]})
            for name, tags in volume_tags.items()
        }
        named_surfaces = {
            name: sorted({e for t in tags for e in images[(2, t)]})
            for name, tags in surface_tags.items()
        }
        _overlap(named_volumes, "volume")
        _overlap(named_surfaces, "surface")

        keep = set(gmsh.model.getEntities(3))
        keep.update(e for es in named_surfaces.values() for e in es)
        _close_downwards(keep)
        stray = [e for d in (2, 1, 0) for e in gmsh.model.getEntities(d) if e not in keep]
        if stray:
            occ.remove(stray)
            occ.synchronize()

        _remove_groups()
        tag = 0
        for groups, dim in ((named_volumes, 3), (named_surfaces, 2)):
            for name, entities in groups.items():
                tag += 1
                gmsh.model.addPhysicalGroup(dim, [t for _, t in entities], tag, name)

    def scale(self, factor: float) -> None:
        """Scales the model about the origin, keeping the names.

        The model is written to a BREP file and read back with the
        OpenCASCADE import scaling: the surfaces keep their type (a plane
        stays a plane). A scaling in place (``occ.dilate``) turns every
        surface into a B-spline, on which Booleans can fail silently (empty
        results). The entity tags may change; each physical group is
        carried to the entities of the same scaled centre of mass and
        measure.

        Raises:
            GmshJobError: kind ``gmsh``, code ``names.lost``: a part was not
                found again.
        """
        occ = gmsh.model.occ
        occ.synchronize()
        groups = [
            (dim, tag, gmsh.model.getPhysicalName(dim, tag),
             [int(e) for e in gmsh.model.getEntitiesForPhysicalGroup(dim, tag)])
            for dim, tag in gmsh.model.getPhysicalGroups()
        ]  # fmt: skip
        before = {(d, e): signature(d, e, factor) for d, _, _, ents in groups for e in ents}
        extent = model_extent() * factor
        with tempfile.TemporaryDirectory(prefix="mpxlite_gmsh_scale_") as folder:
            path = pathlib.Path(folder) / "cell.brep"
            gmsh.write(str(path))
            gmsh.clear()
            gmsh.option.setNumber("Geometry.OCCScaling", factor)
            try:
                occ.importShapes(str(path))
            finally:
                gmsh.option.setNumber("Geometry.OCCScaling", 1.0)
        occ.synchronize()
        became = _match_entities(before, extent, "the scaling of the model")
        _remove_groups()
        for dim, tag, name, ents in groups:
            gmsh.model.addPhysicalGroup(dim, [became[(dim, e)] for e in ents], tag=tag, name=name)

    @staticmethod
    def fragment_with(
        tools: Sequence[tuple[int, int]], scale: float
    ) -> list[list[tuple[int, int]]]:
        """Fragments the volumes and named surfaces with ``tools``, names kept.

        The parts must not be split by the tools (they touch the model on
        whole faces only): every entity of a part is found again by its
        centre of mass and its measure (the entity maps of a fragment are
        not reliable for unchanged entities). The physical groups keep their
        tags and names.

        Args:
            tools: The ``(dim, tag)`` entities fragmented with the model.
            scale: Length scale of the model, for the matching tolerance.

        Returns:
            For each tool, the entities it became.

        Raises:
            GmshJobError: kind ``gmsh``, code ``names.lost``: a part was split.
        """
        occ = gmsh.model.occ
        groups = []
        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag)
            ents = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            groups.append((dim, tag, name, [int(e) for e in ents]))
        objects = [e for e in gmsh.model.getEntities(3) if e not in set(tools)]
        objects += [(2, e) for d, _, _, ents in groups if d == 2 for e in ents]
        before = {(d, e): signature(d, e) for d, _, _, ents in groups for e in ents}
        _, children = occ.fragment(objects, list(tools))
        occ.synchronize()
        became = _match_entities(before, scale, "the fragment with new volumes")
        _remove_groups()
        for dim, tag, name, ents in groups:
            gmsh.model.addPhysicalGroup(dim, sorted(became[(dim, e)] for e in ents), tag, name)
        return [list(children[len(objects) + k]) for k in range(len(tools))]

    # -- questions ----------------------------------------------------------

    def measure(self, name: str) -> float:
        """Volume or area of the part ``name``."""
        dim, ents = self.groups()[name]
        return float(sum(gmsh.model.occ.getMass(dim, e) for e in ents))

    def centre(self, name: str) -> np.ndarray:
        """Centre of mass of the part ``name``."""
        dim, ents = self.groups()[name]
        masses = np.array([gmsh.model.occ.getMass(dim, e) for e in ents])
        centres = np.array([gmsh.model.occ.getCenterOfMass(dim, e) for e in ents])
        return np.average(centres, axis=0, weights=masses)

    def bounding_box(self, name: str) -> np.ndarray:
        """``[[x, y, z] min, [x, y, z] max]`` of the part ``name``."""
        dim, ents = self.groups()[name]
        boxes = np.array([np.reshape(gmsh.model.getBoundingBox(dim, e), (2, 3)) for e in ents])
        return np.array([boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)])

    def corner_points(self, name: str | None = None) -> np.ndarray:
        """Corner points of the part ``name`` (of the model: ``None``), shape (n, 3)."""
        if name is None:
            return np.array([gmsh.model.getValue(0, t, []) for _, t in gmsh.model.getEntities(0)])
        dim, ents = self.groups()[name]
        return np.unique(np.vstack([vertices(dim, e) for e in ents]), axis=0)

    @staticmethod
    def extent() -> float:
        """Largest extent of the corner points of the model."""
        return model_extent()

    @staticmethod
    def faces_on_plane(normal: Sequence[float], offset: float, tol: float) -> list[int]:
        """Surfaces of the model lying in the plane ``x . normal = offset``.

        A surface lies in the plane when its corner points and a 3 x 3 grid
        of its points are within ``tol`` of it and its normals are parallel
        to ``normal``.
        """
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        out = []
        for _, tag in gmsh.model.getEntities(2):
            samples, normals = surface_samples(tag)
            points = np.vstack([vertices(2, tag), samples])
            on = np.all(np.abs(points @ n - offset) <= tol)
            parallel = np.all(np.linalg.norm(np.cross(normals, n), axis=1) <= 1e-6)
            if on and parallel:
                out.append(tag)
        return out

    def volumes_of_face(self, face: int) -> list[str]:
        """Names of the volumes on each side of a surface entity."""
        up, _ = gmsh.model.getAdjacencies(2, face)
        owner = {e: n for n, (d, ents) in self.groups(3).items() if d == 3 for e in ents}
        return sorted({owner.get(int(v), "") for v in up})

    def touching(self, first: str, second: str, tol: float | None = None) -> bool:
        """Whether two parts share a point of their closures (within ``tol``)."""
        tol = RTOL * model_extent() if tol is None else tol
        d1, e1 = self.groups()[first]
        d2, e2 = self.groups()[second]
        for a, b in itertools.product(e1, e2):
            dist = gmsh.model.occ.getDistance(d1, a, d2, b)[0]
            if dist <= tol:
                return True
        return False

    @staticmethod
    def faces_match(
        faces: Sequence[int], images: Sequence[int], affine: np.ndarray, tol: float
    ) -> list[tuple[int, int]] | None:
        """Pairs ``(face, image)`` with ``image = A face`` (``A`` a 4 x 4 affine map).

        Matched by centre of mass and area; ``None`` when a face has no
        image or the counts differ.
        """
        if len(faces) != len(images):
            return None
        a = np.asarray(affine, dtype=float).reshape(4, 4)
        left = list(images)
        pairs = []
        for f in faces:
            sig = signature(2, f)
            c = a[:3, :3] @ sig[:3] + a[:3, 3]
            hit = [
                g for g in left
                if np.max(np.abs(signature(2, g)[:3] - c)) <= tol
                and abs(signature(2, g)[3] - sig[3]) <= RTOL * max(sig[3], 1e-300)
            ]  # fmt: skip
            if len(hit) != 1:
                return None
            left.remove(hit[0])
            pairs.append((f, hit[0]))
        return pairs

    def point_in(self, name: str, point: Sequence[float]) -> bool:
        """Whether a point is inside (or on) the part ``name``."""
        dim, ents = self.groups()[name]
        tol = RTOL * model_extent()
        p = np.asarray(point, dtype=float)
        for e in ents:
            if dim == 2:
                if on_surface(p, e, tol):
                    return True
            elif gmsh.model.isInside(dim, e, p.tolist()):
                return True
        return False

    def closest_point(self, name: str, point: Sequence[float]) -> np.ndarray:
        """The point of the part ``name`` closest to ``point`` (``point`` inside a volume: itself).

        Exact to the precision of OpenCASCADE (a distance between shapes).
        """
        dim, ents = self.groups()[name]
        p = np.asarray(point, dtype=float)
        occ = gmsh.model.occ
        if dim == 3 and any(gmsh.model.isInside(3, e, p.tolist()) for e in ents):
            return p
        probe = occ.addPoint(*p.tolist())
        try:
            best, gap = p, np.inf
            for e in ents:
                found = occ.getDistance(0, probe, dim, e)
                if found[0] < gap:
                    best, gap = np.asarray(found[4:7], dtype=float), found[0]
        finally:
            occ.remove([(0, probe)])
        return best
