# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the GNU General Public
# License (the file COPYING of the directory above) for more details.

"""Mesh sizes of a periodic cell from controls given by name.

:func:`cell_sizing` turns sizes given per named volume and the named surface
groups whose edges are refined into the
:class:`~mpxlite.cells.autosym.MeshSizing` of the cell (all lengths in the
unit of the model):

* bulk size of a volume: the size given for it, capped by
  ``extent_fraction`` times the smallest extent of the volume (of each of
  its pieces; no cap when ``extent_fraction`` is ``None``);
* edge size on a curve bounding a refined surface group:
  ``factor * min(f / divisions, bulk)``, at least ``min_fraction`` times the
  cell size (the square root of the footprint area), on each of its pieces
  (``line_pieces`` per straight line, ``curve_pieces`` per curve); ``f`` is
  the local feature size of the piece (:func:`feature_sizes`): its distance
  to the nearest refined curve that shares no end with its curve (periodic
  images included), at most the smallest radius of curvature of its curve;
  ``bulk`` is the smallest bulk size of the volumes the curve touches;
  ``factor`` the smallest factor of the refined groups the curve bounds;
  curves on the side faces of the cell are cuts, not edges;
* grading: the size grows linearly by ``slope`` per unit distance from a
  refined curve, up to the bulk size.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence

import gmsh
import numpy as np

from mpxlite.cells.autosym import MeshSizing, on_window_line
from mpxlite.gmshjob import GmshJobError
from mpxlite_gmsh.symmetric import curve_pieces

__all__ = ["EdgeRule", "cell_sizing", "feature_sizes"]


@dataclasses.dataclass(frozen=True)
class EdgeRule:
    """The rule of the refined edges (see the module).

    Attributes:
        divisions: Elements across the local feature size of a refined curve.
        slope: Growth of the size per unit distance from a refined curve.
        min_fraction: Smallest edge size, as a fraction of the cell size.
        line_pieces: Pieces of a straight curve that carry their own size.
        curve_pieces: Pieces of a curved curve that carry their own size.
        extent_fraction: Largest bulk size, as a fraction of the smallest
            extent of a volume (``None``: no cap).
    """

    divisions: float = 6.0
    slope: float = 0.5
    min_fraction: float = 1e-3
    line_pieces: int = 8
    curve_pieces: int = 32
    extent_fraction: float | None = 0.5


def _scale(lattice: np.ndarray) -> float:
    return math.sqrt(abs(float(np.linalg.det(lattice))))


def _volume_extent(ents: Sequence[int]) -> float:
    """Smallest extent of the pieces of a volume group."""
    out = math.inf
    for e in ents:
        bb = np.asarray(gmsh.model.getBoundingBox(3, e)).reshape(2, 3)
        out = min(out, float(np.min(bb[1] - bb[0])))
    return out


def _volumes_of_curve(curve: int) -> set[int]:
    out = set()
    for s in gmsh.model.getAdjacencies(1, curve)[0]:
        out.update(int(v) for v in gmsh.model.getAdjacencies(2, int(s))[0])
    return out


def _pieces(curve: int, rule: EdgeRule) -> np.ndarray:
    """The ends of the pieces of a curve (``line_pieces`` or ``curve_pieces``)."""
    line = gmsh.model.getType(1, curve) == "Line"
    return curve_pieces(curve, rule.line_pieces if line else rule.curve_pieces)


def _refined_curves(group_tags: Sequence[int], lattice: np.ndarray, rule: EdgeRule) -> list[int]:
    """Boundary curves of the surface groups, the cuts on the side faces excluded."""
    curves = set()
    for tag in group_tags:
        ents = gmsh.model.getEntitiesForPhysicalGroup(2, tag)
        bnd = gmsh.model.getBoundary([(2, int(e)) for e in ents], combined=False, oriented=False)
        curves.update(abs(c) for _, c in bnd)
    tol = 1e-7 * _scale(lattice)
    lo = np.zeros(2)
    return [
        c for c in sorted(curves)
        if not on_window_line(_pieces(c, rule), lattice, lo, tol)
    ]  # fmt: skip


def _min_radius(curve: int, rule: EdgeRule) -> float:
    """Smallest radius of curvature of a curve (inf for a straight line)."""
    if gmsh.model.getType(1, curve) == "Line":
        return math.inf
    lo, hi = gmsh.model.getParametrizationBounds(1, curve)
    par = np.linspace(lo[0], hi[0], rule.curve_pieces + 1)
    k = np.abs(np.asarray(gmsh.model.getCurvature(1, curve, par.tolist())))
    kmax = float(k.max(initial=0.0))
    return 1.0 / kmax if kmax > 0 else math.inf


def _point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distances of the points ``p`` to the segments ``a b``; shape ``(len(p), len(a))``."""
    ab = b - a
    ab2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300)
    ap = p[:, None, :] - a[None, :, :]
    w = np.clip(np.einsum("pij,ij->pi", ap, ab) / ab2, 0.0, 1.0)
    d = ap - w[..., None] * ab[None]
    return np.sqrt(np.einsum("pij,pij->pi", d, d))


def _piece_distance(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Distance of each piece of the polyline ``p`` to the polyline ``q``."""
    # the points of q to the pieces of p, and the ends of the pieces of p to q
    to_q = _point_segment(p, q[:-1], q[1:]).min(axis=1)
    from_q = _point_segment(q, p[:-1], p[1:]).min(axis=0)
    return np.minimum(np.minimum(to_q[:-1], to_q[1:]), from_q)


def feature_sizes(
    curves: Sequence[int], lattice: np.ndarray, rule: EdgeRule | None = None
) -> dict[int, np.ndarray]:
    """Local feature size along each curve of the current model.

    For each piece of a curve (``line_pieces`` of a straight line,
    ``curve_pieces`` of a curved one, of equal parameter length): the
    distance to the nearest curve of the list, or periodic image of one,
    that shares no end with the curve; at most the smallest radius of
    curvature of the curve. A curve alone in the cell has an infinite
    feature size.

    Args:
        curves: Gmsh curve tags.
        lattice: Lattice rows ``T_1``, ``T_2``.
        rule: The pieces (default: :class:`EdgeRule`).

    Returns:
        Curve tag -> feature size of each of its pieces.
    """
    rule = EdgeRule() if rule is None else rule
    tol = 1e-6 * _scale(lattice)
    pts = {c: _pieces(c, rule) for c in curves}
    images = []
    for c in curves:
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                shift = np.zeros(3)
                shift[:2] = np.array([i, j], dtype=float) @ lattice
                images.append((c, (i, j), pts[c] + shift))
    out = {}
    for c in curves:
        p = pts[c]
        ends = p[[0, -1]]
        best = np.full(len(p) - 1, _min_radius(c, rule))
        for d, shift, q in images:
            if d == c and shift == (0, 0):
                continue
            q_ends = q[[0, -1]]
            gaps = np.linalg.norm(ends[:, None] - q_ends[None], axis=2)
            if np.min(gaps) <= tol:
                continue  # adjacent: they share an end
            best = np.minimum(best, _piece_distance(p, q))
        out[c] = best
    return out


def cell_sizing(
    volume_sizes: Mapping[str, float],
    refined: Mapping[str, float],
    lattice: np.ndarray,
    rule: EdgeRule | None = None,
) -> MeshSizing:
    """The :class:`~mpxlite.cells.autosym.MeshSizing` of the current model (module rule).

    Args:
        volume_sizes: Volume name -> size, for every named volume.
        refined: Surface group name -> factor on its edge sizes.
        lattice: Lattice rows ``T_1``, ``T_2``.
        rule: The edge rule (default: :class:`EdgeRule`).

    Returns:
        The sizing.

    Raises:
        GmshJobError: kind ``input``: a named volume has no size
            (``sizes.missing``), a size names no volume, a refined group no
            surface (``sizes.unknown``).
    """
    rule = EdgeRule() if rule is None else rule
    groups = {gmsh.model.getPhysicalName(d, t): (d, t) for d, t in gmsh.model.getPhysicalGroups()}
    unknown = sorted(
        [n for n in volume_sizes if groups.get(n, (0, 0))[0] != 3]
        + [n for n in refined if groups.get(n, (0, 0))[0] != 2]
    )
    if unknown:
        raise GmshJobError(
            "input",
            "sizes.unknown",
            f"sizes are given for {unknown}, which are not named volumes (sizes) or named "
            "surfaces (refined edges) of the geometry",
            {"names": unknown},
        )
    bulk = {}
    vol_bulk = {}
    for name, size in volume_sizes.items():
        ents = [int(e) for e in gmsh.model.getEntitiesForPhysicalGroup(3, groups[name][1])]
        h = float(size)
        if rule.extent_fraction is not None:
            h = min(h, rule.extent_fraction * _volume_extent(ents))
        bulk[name] = h
        vol_bulk.update({e: h for e in ents})
    missing = [n for n, (d, _) in groups.items() if d == 3 and n not in bulk]
    if missing:
        raise GmshJobError(
            "input",
            "sizes.missing",
            f"volumes without a size: {', '.join(missing)}",
            {"names": missing},
        )
    factor_of: dict[int, float] = {}
    for name, factor in refined.items():
        for c in _refined_curves([groups[name][1]], lattice, rule):
            factor_of[c] = min(float(factor), factor_of.get(c, math.inf))
    curves = _refined_curves([groups[n][1] for n in refined], lattice, rule)
    floor = rule.min_fraction * _scale(lattice)
    edges = {}
    for c, f in feature_sizes(curves, lattice, rule).items():
        cap = min((vol_bulk[v] for v in _volumes_of_curve(c)
                   if v in vol_bulk), default=min(bulk.values()))  # fmt: skip
        h = factor_of[c] * np.minimum(f / rule.divisions, cap)
        edges[c] = tuple(np.maximum(h, floor).tolist())
    h_min = min(bulk.values())
    return MeshSizing(
        bulk=bulk,
        plane=h_min,
        transition=h_min,
        planes=(),
        curve_edges=edges,
        edge_slope=rule.slope,
    )
