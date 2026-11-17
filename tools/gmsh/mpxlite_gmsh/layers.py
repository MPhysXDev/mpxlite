# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the GNU General Public
# License (the file COPYING of the directory above) for more details.

"""Layers above and below a periodic prism cell.

:func:`add_layers` adds a slab on the footprint of the cell above its top
face (``z = height`` to ``height + thickness``) and, optionally, one below
its bottom face; each layer is a named volume and its outer face a named
surface. The parts of the cell keep their names: the layers touch the cell
on whole planes only, so the fragment splits no entity of the cell. Without
a layer below, the faces of the bottom plane that carry no name can be
given one (every face of the outer boundary then belongs to a group).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence

import gmsh
import numpy as np

from mpxlite_gmsh.named import NamedGeometry

__all__ = ["Layer", "add_layers", "faces_at", "slab"]


@dataclasses.dataclass(frozen=True)
class Layer:
    """A layer added outside the cell.

    Attributes:
        thickness: Thickness (positive, lengths of the model).
        volume: Name of its volume.
        face: Name of its outer face (``z = height + thickness`` above,
            ``z = -thickness`` below).
    """

    thickness: float
    volume: str
    face: str


def _scale(lattice: np.ndarray) -> float:
    return math.sqrt(abs(float(np.linalg.det(lattice))))


def slab(lattice: np.ndarray, z0: float, z1: float) -> int:
    """An OpenCASCADE prism on the footprint of the cell between ``z0`` and ``z1``."""
    occ = gmsh.model.occ
    corners = [(0.0, 0.0), lattice[0], lattice[0] + lattice[1], lattice[1]]
    pts = [occ.addPoint(float(x), float(y), z0) for x, y in corners]
    lines = [occ.addLine(pts[k], pts[(k + 1) % 4]) for k in range(4)]
    face = occ.addPlaneSurface([occ.addCurveLoop(lines)])
    out = occ.extrude([(2, face)], 0.0, 0.0, z1 - z0)
    return next(t for d, t in out if d == 3)


def faces_at(lattice: np.ndarray, z: float) -> list[int]:
    """Surfaces lying in the plane ``z`` within the footprint."""
    xy = np.array([[0.0, 0.0], lattice[0], lattice[0] + lattice[1], lattice[1]])
    eps = max(1e-6 * _scale(lattice), 2e-7)
    lo, hi = xy.min(axis=0) - eps, xy.max(axis=0) + eps
    ents = gmsh.model.getEntitiesInBoundingBox(lo[0], lo[1], z - eps, hi[0], hi[1], z + eps, 2)
    return sorted(t for _, t in ents)


def _name_bare_plane(
    lattice: np.ndarray, z: float, name: str, add: Callable[[int, Sequence[int], str], None]
) -> None:
    """Names the faces of the plane ``z`` that are in no surface group."""
    owner = {}
    for dim, tag in gmsh.model.getPhysicalGroups(2):
        group = gmsh.model.getPhysicalName(dim, tag)
        for e in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
            owner[int(e)] = group
    bare = [f for f in faces_at(lattice, z) if f not in owner]
    if bare:
        add(2, bare, name)


def add_layers(
    geometry: NamedGeometry,
    lattice: np.ndarray,
    height: float,
    above: Layer | None,
    below: Layer | None,
    bare_bottom: str | None = None,
) -> dict[str, list[str]]:
    """Adds the layers to the current model; the names added, by role.

    The physical groups of the cell keep their tags and names; the groups
    added get the next free tags, in the order: the volume and the face of
    the layer above, then those of the layer below, or the group of the bare
    bottom faces.

    Args:
        geometry: The cell.
        lattice: Lattice rows ``T_1``, ``T_2``.
        height: Height of the cell.
        above: The layer above, or ``None``.
        below: The layer below, or ``None``.
        bare_bottom: Without a layer below: the name given to the faces of
            the plane ``z = 0`` in no surface group (``None``: left bare).

    Returns:
        ``{"above": [volume], "top": [face], "below": [volume],
        "bottom": [face], "bare_bottom": [name]}`` for what was added.
    """
    tools = []
    if above is not None:
        tools.append((3, slab(lattice, height, height + above.thickness)))
    if below is not None:
        tools.append((3, slab(lattice, -below.thickness, 0.0)))
    used = {t for _, t in gmsh.model.getPhysicalGroups()}
    roles: dict[str, list[str]] = {}
    children = geometry.fragment_with(tools, _scale(lattice)) if tools else []

    def add(dim: int, ents: Sequence[int], name: str) -> None:
        tag = max(used, default=0) + 1
        used.add(tag)
        gmsh.model.addPhysicalGroup(dim, sorted(set(ents)), tag, name)

    k = 0
    if above is not None:
        add(3, [t for d, t in children[k] if d == 3], above.volume)
        add(2, faces_at(lattice, height + above.thickness), above.face)
        roles |= {"top": [above.face], "above": [above.volume]}
        k += 1
    if below is not None:
        add(3, [t for d, t in children[k] if d == 3], below.volume)
        add(2, faces_at(lattice, -below.thickness), below.face)
        roles |= {"bottom": [below.face], "below": [below.volume]}
    elif bare_bottom is not None:
        before = len(used)
        _name_bare_plane(lattice, 0.0, bare_bottom, add)
        if len(used) > before:
            roles["bare_bottom"] = [bare_bottom]
    return roles
