# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the GNU General Public
# License (the file COPYING of the directory above) for more details.

"""Bridging strips: a named surface that joins two pieces of a surface set.

A bridging strip is a flat named surface whose boundary is made of two
*ends*, each on a surface of a given set (its *end surfaces*), and two
straight parallel *free sides*. :func:`measure_strip` measures it (direction
from one end to the other, width between the free sides, length between the
ends, centre, the end surfaces it joins) and checks its rules (each refusal
is a :class:`~mpxlite.GmshJobError` of kind ``rule``):

* L1 the strip is a named surface (``strip.unknown``);
* L2 it touches no side face of the cell (``strip.touches_side``);
* L3 it has exactly two ends, each on an end surface (``strip.ends``);
* L4 its free sides are straight (``strip.curved_side``);
* L5 it has exactly two free sides, parallel (``strip.free_sides``);
* L6 its two ends are on two pieces of the end surfaces that do not touch
  inside the cell (``strip.short_circuit``). Two end surfaces are one piece
  when they share a curve inside the cell; a contact through the periodic
  boundary does not join them, so a strip may join a surface to a periodic
  image of itself.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Mapping, Sequence

import gmsh
import numpy as np

from mpxlite.gmshjob import GmshJobError
from mpxlite_gmsh.named import (
    boundary_curves,
    curve_points,
    curve_samples,
    on_surface,
)
from mpxlite_gmsh.prism import PeriodicPrism

__all__ = ["StripShape", "measure_strip"]


@dataclasses.dataclass(frozen=True)
class StripShape:
    """Measured shape of a bridging strip.

    Attributes:
        direction: Unit vector from one end to the other, with the sign
            convention (first non-zero component positive, in the order
            x, y, z).
        width: Distance between the two free sides.
        length: Distance between the two ends.
        centre: Centre of the surface.
        ends: Names of the two end surfaces it joins, the end at the lower
            coordinate along ``direction`` first.
        area: Area of the surface.
    """

    direction: tuple[float, float, float]
    width: float
    length: float
    centre: tuple[float, float, float]
    ends: tuple[str, str]
    area: float


def _rule(code: str, message: str, **data: object) -> GmshJobError:
    return GmshJobError("rule", code, message, data)


def _in_box(points: np.ndarray, box: np.ndarray, tol: float) -> bool:
    """Whether points lie in a bounding box ``[[min], [max]]``."""
    return bool(np.all(points >= box[0] - tol) and np.all(points <= box[1] + tol))


def _curve_on(curve: int, surface: int, shift: np.ndarray, tol: float) -> bool:
    """Whether a curve, translated by ``shift``, lies on a surface."""
    if not shift.any():
        up, _ = gmsh.model.getAdjacencies(1, curve)
        if surface in {int(t) for t in up}:
            return True
    return all(on_surface(point + shift, surface, tol) for point in curve_samples(curve))


def _components(surfaces: Mapping[str, tuple[int, ...]], tol: float) -> dict[int, int]:
    """Connected end surfaces inside the cell: surface tag -> component index."""
    tags = sorted({t for ts in surfaces.values() for t in ts})
    parent = {t: t for t in tags}

    def find(tag: int) -> int:
        while parent[tag] != tag:
            parent[tag] = parent[parent[tag]]
            tag = parent[tag]
        return tag

    curves = {t: boundary_curves([t]) for t in tags}
    boxes = {t: np.reshape(gmsh.model.getBoundingBox(2, t), (2, 3)) for t in tags}
    for a, b in itertools.combinations(tags, 2):
        if find(a) == find(b):
            continue
        candidates = [c for c in curves[a] if _in_box(curve_samples(c), boxes[b], tol)]
        if any(_curve_on(c, b, np.zeros(3), tol) for c in candidates):
            parent[find(a)] = find(b)
    return {t: find(t) for t in tags}


def _end_groups(curves: Sequence[int]) -> list[list[int]]:
    """Groups curves into chains connected by their end points."""
    ends = {
        c: {abs(t) for _, t in gmsh.model.getBoundary([(1, c)], combined=False, oriented=False)}
        for c in curves
    }
    groups: list[list[int]] = []
    for curve in curves:
        joined = [g for g in groups if any(ends[c] & ends[curve] for c in g)]
        merged = [curve] + [c for g in joined for c in g]
        groups = [g for g in groups if g not in joined] + [merged]
    return groups


def _line_of(curve: int, tol: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Point and unit direction of a straight curve, or None if curved."""
    points = curve_points(curve)
    start, end = points[0], points[1] if len(points) > 1 else points[0]
    chord = end - start
    size = np.linalg.norm(chord)
    if size <= tol:
        return None
    direction = chord / size
    offsets = points - start
    distances = np.linalg.norm(offsets - np.outer(offsets @ direction, direction), axis=1)
    return (start, direction) if np.all(distances <= tol) else None


def _signed(direction: np.ndarray) -> np.ndarray:
    """The sign convention: first non-zero component positive (x, y, z)."""
    for value in direction:
        if abs(value) > 1e-9:
            return direction if value > 0 else -direction
    return direction


def _free_sides(name: str, curves: Sequence[int], tol: float) -> tuple[np.ndarray, float]:
    """Direction and width from the free sides of a strip (L4, L5)."""
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for curve in curves:
        line = _line_of(curve, tol)
        if line is None:
            raise _rule(
                "strip.curved_side",
                f"the strip '{name}' must have two parallel straight free sides "
                "from one end to the other; one of its free sides is curved",
                strip=name,
            )
        point, direction = line
        same = [
            k
            for k, (p, d) in enumerate(lines)
            if np.linalg.norm(np.cross(d, direction)) <= 1e-6
            and np.linalg.norm(np.cross(point - p, d)) <= tol
        ]
        if not same:
            lines.append(line)
    parallel = len(lines) == 2 and (np.linalg.norm(np.cross(lines[0][1], lines[1][1])) <= 1e-6)
    if not parallel:
        raise _rule(
            "strip.free_sides",
            f"the strip '{name}' must have two parallel straight free sides "
            "(sides not on an end surface); "
            f"found {len(lines)} free side(s)" + ("" if len(lines) != 2 else ", not parallel"),
            strip=name,
            free_sides=len(lines),
            parallel=False,
        )
    direction = _signed(lines[0][1])
    offset = lines[1][0] - lines[0][0]
    width = float(np.linalg.norm(offset - (offset @ direction) * direction))
    return direction, width


def measure_strip(
    prism: PeriodicPrism,
    surfaces: Mapping[str, tuple[int, ...]],
    name: str,
    end_surfaces: Mapping[str, tuple[int, ...]],
) -> StripShape:
    """Measures a bridging strip and checks its rules (L1 to L6).

    The free sides are the boundary curves of the surface that are not on an
    end surface; the ends are the others.

    Args:
        prism: The prism of the accepted cell (its tolerance).
        surfaces: Surface name -> entity tags of the current model.
        name: Name of the strip surface.
        end_surfaces: End surface name -> entity tags.

    Returns:
        The measured shape.

    Raises:
        GmshJobError: kind ``rule``, code ``strip.*``.
    """
    if name not in surfaces:
        raise _rule(
            "strip.unknown",
            f"the strip '{name}' is not a named surface of the geometry; "
            f"named surfaces: {sorted(surfaces)}",
            strip=name,
            surfaces=sorted(surfaces),
        )
    tags = surfaces[name]
    curves = boundary_curves(tags)
    points = np.vstack([curve_points(c) for c in curves])
    face = prism.side_of(points)
    if face is not None:
        raise _rule(
            "strip.touches_side",
            f"the strip '{name}' touches the side face {face} of the cell; a strip may "
            "not touch a side face: choose another cell of the same periodic structure "
            "(the same pattern shifted) in which no strip touches a side face",
            strip=name,
            face=face,
        )
    end_tags = {t: n for n, ts in end_surfaces.items() for t in ts if t not in tags}
    on: dict[int, set[int]] = {}
    for curve in curves:
        touched = {t for t in end_tags if _curve_on(curve, t, np.zeros(3), prism.tol)}
        if touched:
            on[curve] = touched
    groups = _end_groups(list(on))
    if len(groups) != 2:
        raise _rule(
            "strip.ends",
            f"the strip '{name}' must have two ends, each on an end surface; "
            f"found {len(groups)} end(s) on an end surface",
            strip=name,
            ends=len(groups),
        )
    free = [c for c in curves if c not in on]
    direction, width = _free_sides(name, free, prism.tol)
    component = _components(end_surfaces, prism.tol)
    ends = []
    for group in groups:
        touched = sorted({t for c in group for t in on[c]})
        position = float(np.mean(np.vstack([curve_points(c) for c in group]) @ direction))
        ends.append((position, touched))
    ends.sort(key=lambda end: end[0])
    first = {component[t] for t in ends[0][1]}
    second = {component[t] for t in ends[1][1]}
    if first & second:
        names = sorted({end_tags[t] for _, ts in ends for t in ts})
        raise _rule(
            "strip.short_circuit",
            f"the two ends of the strip '{name}' are on one piece of the end surfaces "
            f"({', '.join(repr(n) for n in names)}, connected inside the cell): a strip "
            "must join two pieces that do not touch inside the cell",
            strip=name,
            surfaces=names,
        )
    projection = points @ direction
    occ = gmsh.model.occ
    areas = [occ.getMass(2, t) for t in tags]
    centres = np.array([occ.getCenterOfMass(2, t) for t in tags])
    centre = np.average(centres, axis=0, weights=areas)
    return StripShape(
        direction=tuple(float(c) for c in direction),
        width=width,
        length=float(projection.max() - projection.min()),
        centre=tuple(float(c) for c in centre),
        ends=(end_tags[ends[0][1][0]], end_tags[ends[1][1][0]]),
        area=float(sum(areas)),
    )
