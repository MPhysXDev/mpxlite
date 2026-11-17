# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the GNU General Public
# License (the file COPYING of the directory above) for more details.

"""A periodic prism cell: the rules of its geometry, the facts read from it.

Conventions of a periodic prism cell:

* the cell occupies a prism whose bottom face lies in the plane z = 0, with
  one corner at the origin and its first lattice vector along +x;
* the lattice vectors are ``(pu, 0, 0)`` and
  ``(pv cos(alpha), pv sin(alpha), 0)``, ``0 < alpha < pi``;
* the volumes fill the prism exactly; the geometry is periodic in (x, y).

:func:`accept_cell` checks the rules (each refusal is a
:class:`~mpxlite.GmshJobError` of kind ``rule`` with its code):

* R1 every volume is named, names are distinct and named parts do not
  overlap (``cell.unnamed_volume``, ``geometry.no_volume``, ``names.*``);
* R2 the bounding prism follows the convention (``prism.no_point``,
  ``prism.bottom``, ``prism.corner``, ``prism.first_side``, ``prism.flat``,
  ``prism.footprint``);
* R3 every outer face lies on a face of the prism or inside it
  (``prism.outside``);
* R4 the volumes fill the prism exactly (``cell.not_filled``);
* R5 no outer face of the volumes lies inside the prism (a gap, or touching
  volumes that do not conform: ``cell.inner_face``);
* R6 the trace of the parts on each side face matches that on the opposite
  face under the lattice translation, names included (``cell.not_periodic``).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence

import gmsh
import numpy as np

from mpxlite.gmshjob import GmshJobError
from mpxlite_gmsh import named as _named
from mpxlite_gmsh.named import (
    NamedGeometry,
    boundary_curves,
    curve_points,
    length_text,
    model_extent,
    point_text,
    surface_samples,
    vertices,
)

__all__ = ["CellFacts", "PeriodicPrism", "accept_cell", "classify_faces", "find_prism"]


@dataclasses.dataclass(frozen=True)
class PeriodicPrism:
    """The bounding prism and its faces, for the rules and the messages."""

    pu: float
    pv: float
    alpha: float
    height: float
    tol: float
    unit: str

    @property
    def t1(self) -> np.ndarray:
        """First lattice vector (3D)."""
        return np.array([self.pu, 0.0, 0.0])

    @property
    def t2(self) -> np.ndarray:
        """Second lattice vector (3D)."""
        return np.array([self.pv * math.cos(self.alpha), self.pv * math.sin(self.alpha), 0])

    @property
    def rectangular(self) -> bool:
        """Whether the footprint is a rectangle."""
        return abs(math.cos(self.alpha)) < _named.RTOL

    @property
    def lattice(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """The two lattice vectors ``((pu, 0), (pv cos a, pv sin a))``.

        The x component of the second vector is exactly 0 on a rectangular
        lattice (``cos a`` below ``1e-12``).
        """
        t2_x = self.pv * math.cos(self.alpha)
        if abs(t2_x) <= 1e-12 * self.pv:
            t2_x = 0.0
        return ((self.pu, 0.0), (t2_x, self.pv * math.sin(self.alpha)))

    def faces(self) -> list[tuple[str, np.ndarray, float, np.ndarray]]:
        """Side faces: (label, unit normal, offset, translation to opposite).

        The first two faces are the faces through the origin; the face
        opposite each is reached by the translation.
        """
        sin_a, cos_a = math.sin(self.alpha), math.cos(self.alpha)
        n_y = np.array([0.0, 1.0, 0.0])
        n_v = np.array([sin_a, -cos_a, 0.0])
        if self.rectangular:
            labels = (
                "y = 0",
                f"y = {length_text(self.pv, self.unit)}",
                "x = 0",
                f"x = {length_text(self.pu, self.unit)}",
            )
        else:
            labels = (
                "y = 0",
                f"y = {length_text(self.pv * sin_a, self.unit)}",
                "through the origin along the second lattice vector",
                f"through ({length_text(self.pu, self.unit)}, 0, 0) along the second "
                "lattice vector",
            )
        return [
            (labels[0], n_y, 0.0, self.t2),
            (labels[2], n_v, 0.0, self.t1),
            (labels[1], n_y, self.pv * sin_a, -self.t2),
            (labels[3], n_v, self.pu * sin_a, -self.t1),
        ]

    def side_of(self, points: np.ndarray, every: bool = False) -> str | None:
        """Label of a side face holding the points, if any.

        Args:
            points: Points, shape (n, 3).
            every: Require all the points on the face, not just one.

        Returns:
            The label of the first such face, or None.
        """
        test = np.all if every else np.any
        for label, normal, offset, _ in self.faces():
            if test(np.abs(points @ normal - offset) <= self.tol):
                return label
        return None

    def contains(self, points: np.ndarray) -> bool:
        """Whether all points lie in the closed prism."""
        result = True
        for _, normal, offset, _ in self.faces()[:2]:
            far = offset + (self.t2 if normal[0] == 0 else self.t1) @ normal
            proj = points @ normal
            result &= bool(np.all(proj >= -self.tol))
            result &= bool(np.all(proj <= far + self.tol))
        z = points[:, 2]
        return result and bool(np.all(z >= -self.tol) and np.all(z <= self.height + self.tol))


@dataclasses.dataclass(frozen=True)
class CellFacts:
    """What is read from an accepted cell (lengths in the unit of the model).

    Attributes:
        prism: The bounding prism.
        volumes: Volume name -> entity tags.
        surfaces: Surface name -> entity tags.
        bottom: Tags of the surfaces forming the bottom face (z = 0).
        top: Tags of the surfaces forming the top face (z = height).
        sides: Side face label -> tags of the surfaces forming it.
    """

    prism: PeriodicPrism
    volumes: Mapping[str, tuple[int, ...]]
    surfaces: Mapping[str, tuple[int, ...]]
    bottom: tuple[int, ...]
    top: tuple[int, ...]
    sides: Mapping[str, tuple[int, ...]]


def _rule(code: str, message: str, **data: object) -> GmshJobError:
    return GmshJobError("rule", code, message, data)


def _check_names(
    volumes: Mapping[str, tuple[int, ...]],
    surfaces: Mapping[str, tuple[int, ...]],
    unit: str,
) -> None:
    """Every volume named once; named surfaces disjoint; names distinct (R1)."""
    for groups, kind in ((volumes, "volume"), (surfaces, "surface")):
        owner: dict[int, str] = {}
        for name, tags in groups.items():
            for tag in tags:
                other = owner.setdefault(tag, name)
                if other != name:
                    raise _rule(
                        "names.overlap",
                        f"the {kind}s '{other}' and '{name}' overlap; a part has one name",
                        kind=kind,
                        names=[other, name],
                    )
    named = {t for tags in volumes.values() for t in tags}
    all_volumes = [t for _, t in gmsh.model.getEntities(3)]
    if not all_volumes:
        raise _rule("geometry.no_volume", "the geometry has no volume")
    for tag in all_volumes:
        if tag not in named:
            centre = gmsh.model.occ.getCenterOfMass(3, tag)
            raise _rule(
                "cell.unnamed_volume",
                f"the volume centred at {point_text(centre, unit)} has no name; "
                "every volume must be in a named physical group",
                centre=list(centre),
            )


def find_prism(unit: str) -> PeriodicPrism:
    """Reads the bounding prism from the corner points of the model (R2)."""
    points = np.array([gmsh.model.getValue(0, t, []) for _, t in gmsh.model.getEntities(0)])
    if points.size == 0:
        raise _rule("prism.no_point", "the geometry has no corner point: not a prism")
    tol = _named.RTOL * model_extent()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    if abs(z_min) > tol:
        raise _rule(
            "prism.bottom",
            "the bottom face of the cell must lie in the plane z = 0; the "
            f"lowest point of the geometry is at z = {length_text(z_min, unit)}",
            z_min=float(z_min),
        )
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    low = points[np.abs(points[:, 1] - y_min) <= tol]
    high = points[np.abs(points[:, 1] - y_max) <= tol]
    corner = low[np.lexsort((low[:, 2], low[:, 0]))[0]]
    if abs(y_min) > tol or abs(corner[0]) > tol:
        raise _rule(
            "prism.corner",
            "one corner of the cell footprint must be at the origin, with "
            "the first side along +x; the lowest corner found is at "
            f"{point_text(corner, unit)}",
            corner=corner.tolist(),
        )
    pu = float(low[:, 0].max())
    if pu <= tol:
        raise _rule(
            "prism.first_side",
            "the first side of the cell footprint must lie along +x from "
            "the origin; the geometry has no side along +x in the plane "
            "y = 0",
        )
    t2 = high[np.argmin(high[:, 0])]
    pv = float(math.hypot(t2[0], t2[1]))
    alpha = float(math.atan2(t2[1], t2[0]))
    if y_max <= tol or z_max <= tol:
        raise _rule(
            "prism.flat",
            "the geometry is flat: the cell must be a prism of non-zero footprint and height",
        )
    far = float(high[:, 0].max())
    if abs(far - (t2[0] + pu)) > tol:
        raise _rule(
            "prism.footprint",
            "the cell footprint must be a parallelogram with its first side "
            f"along +x: the side at y = 0 is {length_text(pu, unit)} long, the "
            f"side at y = {length_text(y_max, unit)} is "
            f"{length_text(far - t2[0], unit)} long",
            first_side=pu,
            opposite_side=far - float(t2[0]),
        )
    return PeriodicPrism(pu, pv, alpha, float(z_max), tol, unit)


def classify_faces(
    prism: PeriodicPrism,
) -> tuple[list[int], list[int], dict[str, list[int]]]:
    """Sorts the outer faces into bottom, top and the four side faces (R3 to R5)."""
    volumes = gmsh.model.getEntities(3)
    outer = sorted(
        {abs(t) for _, t in gmsh.model.getBoundary(volumes, combined=True, oriented=False)}
    )
    planes = [("bottom", np.array([0.0, 0.0, 1.0]), 0.0)]
    planes.append(("top", np.array([0.0, 0.0, 1.0]), prism.height))
    planes += [(label, n, d) for label, n, d, _ in prism.faces()]
    bottom: list[int] = []
    top: list[int] = []
    sides: dict[str, list[int]] = {label: [] for label, *_ in prism.faces()}
    inside = []
    for tag in outer:
        samples, normals = surface_samples(tag)
        points = np.vstack([vertices(2, tag), samples])
        for label, normal, offset in planes:
            on_plane = np.all(np.abs(points @ normal - offset) <= prism.tol)
            parallel = np.all(np.linalg.norm(np.cross(normals, normal), axis=1) <= 1e-6)
            if on_plane and parallel:
                {"bottom": bottom, "top": top}.get(label, sides.get(label)).append(tag)
                break
        else:
            centre = gmsh.model.occ.getCenterOfMass(2, tag)
            if prism.contains(points):
                inside.append(centre)
                continue
            raise _rule(
                "prism.outside",
                "the geometry must be bounded by a prism with one corner "
                f"at the origin, of sides {length_text(prism.pu, prism.unit)}"
                f" along +x and {length_text(prism.pv, prism.unit)} at "
                f"{math.degrees(prism.alpha):.6g} degrees, and height "
                f"{length_text(prism.height, prism.unit)}; the face centred "
                f"at {point_text(centre, prism.unit)} leaves this prism",
                centre=list(centre),
                sides=[prism.pu, prism.pv],
                angle=prism.alpha,
                height=prism.height,
            )
    _check_filled(prism)
    if inside:
        raise _rule(
            "cell.inner_face",
            "the volumes must fill the prism of the cell exactly: the "
            f"face centred at {point_text(inside[0], prism.unit)} bounds a "
            "volume but lies inside the prism (a gap between volumes, "
            "or touching volumes that do not share their common face)",
            centre=list(inside[0]),
        )
    return bottom, top, sides


def _check_filled(prism: PeriodicPrism) -> None:
    """The volumes of the parts add up to the volume of the prism (R4)."""
    total = sum(gmsh.model.occ.getMass(3, t) for _, t in gmsh.model.getEntities(3))
    expected = prism.pu * prism.pv * math.sin(prism.alpha) * prism.height
    if abs(total - expected) <= _named.RTOL * expected:
        return
    cube = f"{prism.unit}^3" if prism.unit else ""
    what = "overlap" if total > expected else "leave a gap"
    raise _rule(
        "cell.not_filled",
        f"the volumes must fill the prism of the cell exactly; they {what}: "
        f"their volumes add up to {total:.6g} {cube}, the prism holds "
        f"{expected:.6g} {cube}".replace("  ", " ").rstrip(),
        total=total,
        expected=expected,
        overlap=total > expected,
    )


def _labels_of_face(
    tag: int,
    volume_of: Mapping[int, str],
    surface_of: Mapping[int, str],
) -> tuple[str, ...]:
    """Names attached to a piece of a side face."""
    up, _ = gmsh.model.getAdjacencies(2, tag)
    names = {volume_of.get(int(v), "?") for v in up}
    if tag in surface_of:
        names.add(surface_of[tag])
    return tuple(sorted(names))


def _pieces(
    prism: PeriodicPrism,
    faces: Sequence[int],
    normal: np.ndarray,
    offset: float,
    volume_of: Mapping[int, str],
    surface_of: Mapping[int, str],
) -> list[tuple[tuple[str, ...], float, np.ndarray]]:
    """Partition of a side face: (names, measure, centre) of each piece.

    The pieces are the faces of the volumes on the side face, and the
    traces on it of the named surfaces that are not on the face.
    """
    occ = gmsh.model.occ
    result = []
    for tag in faces:
        labels = _labels_of_face(tag, volume_of, surface_of)
        centre = np.array(occ.getCenterOfMass(2, tag))
        result.append((labels, occ.getMass(2, tag), centre))
    on_face = set(faces)
    for tag, name in surface_of.items():
        if tag in on_face:
            continue
        for curve in boundary_curves([tag]):
            points = curve_points(curve)
            if np.all(np.abs(points @ normal - offset) <= prism.tol):
                centre = np.array(occ.getCenterOfMass(1, curve))
                result.append(((name,), occ.getMass(1, curve), centre))
    return result


def _check_periodic(
    prism: PeriodicPrism,
    sides: Mapping[str, list[int]],
    volumes: Mapping[str, tuple[int, ...]],
    surfaces: Mapping[str, tuple[int, ...]],
) -> list[dict[str, object]]:
    """Opposite side faces match under the lattice translation (R6).

    Returns:
        Per pair of opposite faces: the labels and the number of pieces matched.
    """
    volume_of = {t: n for n, tags in volumes.items() for t in tags}
    surface_of = {t: n for n, tags in surfaces.items() for t in tags}
    faces = prism.faces()
    matched = []
    for (label, normal, offset, shift), (other, _, offset2, _) in zip(
        faces[:2], faces[2:], strict=True
    ):
        here = _pieces(prism, sides[label], normal, offset, volume_of, surface_of)
        there = _pieces(prism, sides[other], normal, offset2, volume_of, surface_of)
        unmatched = list(there)
        for names, measure, centre in here:
            for k, (names2, measure2, centre2) in enumerate(unmatched):
                if (
                    names == names2
                    and abs(measure - measure2) <= _named.RTOL * max(measure, 1e-300)
                    and np.linalg.norm(centre + shift - centre2) <= prism.tol
                ):
                    del unmatched[k]
                    break
            else:
                raise _not_periodic(prism, label, other, names, centre)
        if unmatched:
            names, _, centre = unmatched[0]
            raise _not_periodic(prism, other, label, names, centre)
        matched.append({"face": label, "opposite": other, "pieces": len(here)})
    return matched


def _not_periodic(
    prism: PeriodicPrism,
    label: str,
    other: str,
    names: tuple[str, ...],
    centre: np.ndarray,
) -> GmshJobError:
    """The refusal for a side face without its periodic image."""
    parts = " and ".join(f"'{n}'" for n in names)
    return _rule(
        "cell.not_periodic",
        "the geometry must be periodic in (x, y): the side faces of the cell "
        "must match under the lattice translation, parts and names "
        f"included; {parts} at {point_text(centre, prism.unit)} on the side "
        f"face {label} has no image on the opposite side face {other}",
        names=list(names),
        centre=centre.tolist(),
        face=label,
        opposite=other,
    )


def accept_cell(geometry: NamedGeometry) -> tuple[CellFacts, list[dict[str, object]]]:
    """Checks the rules R1 to R6 on the current (conformal) model.

    Returns:
        The facts of the cell and the periodic match (per pair of opposite
        side faces, its labels and number of pieces).

    Raises:
        GmshJobError: kind ``rule``, code of the first rule broken.
    """
    gmsh.model.occ.synchronize()
    groups = geometry.groups()
    volumes = {n: ts for n, (d, ts) in groups.items() if d == 3}
    surfaces = {n: ts for n, (d, ts) in groups.items() if d == 2}
    _check_names(volumes, surfaces, geometry.unit)
    prism = find_prism(geometry.unit)
    bottom, top, sides = classify_faces(prism)
    matched = _check_periodic(prism, sides, volumes, surfaces)
    facts = CellFacts(
        prism=prism,
        volumes=volumes,
        surfaces=surfaces,
        bottom=tuple(sorted(bottom)),
        top=tuple(sorted(top)),
        sides={k: tuple(sorted(v)) for k, v in sides.items()},
    )
    return facts, matched


def read_cell(geometry: NamedGeometry) -> CellFacts:
    """The facts of an accepted cell, without the rules R1 and R6 (a model read again)."""
    groups = geometry.groups()
    prism = find_prism(geometry.unit)
    bottom, top, sides = classify_faces(prism)
    return CellFacts(
        prism=prism,
        volumes={n: ts for n, (d, ts) in groups.items() if d == 3},
        surfaces={n: ts for n, (d, ts) in groups.items() if d == 2},
        bottom=tuple(sorted(bottom)),
        top=tuple(sorted(top)),
        sides={k: tuple(sorted(v)) for k, v in sides.items()},
    )
