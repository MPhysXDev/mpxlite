# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the GNU General Public
# License (the file COPYING of the directory above) for more details.

r"""Symmetric periodic mesh of a tagged OpenCASCADE cell: the Gmsh side.

Input: the current Gmsh model, built with the OpenCASCADE kernel -- the cell
window ``W`` (lattice rows ``T_1``, ``T_2``; orthogonal for a group other
than ``p1`` / ``p2``) times ``[z_0, z_1]``, its volumes and surfaces tagged
by physical groups (dimensions 2 and 3; the window faces may carry the
periodic-face groups) -- and a :class:`~mpxlite.cells.autosym.MeshSizing`.
:func:`build_symmetric_mesh` then

1. detects the space group (:func:`mpxlite.cells.detect.detect_symmetry` on
   a throw-away mesh of the model that resolves the curved entities, with the
   tolerance raised to twice its measured chord sag; or takes a given
   :class:`~mpxlite.cells.detect.DetectedSymmetry`), and keeps its largest
   subgroup that is admissible and has an asymmetric unit
   (:func:`mpxlite.cells.autosym.supported_subgroup`; ``p1`` at worst: a
   missing symmetry costs time, never a refusal; the report's ``notes`` say
   so, and report a ``p1`` with near symmetries);
2. takes its asymmetric unit :math:`F`
   (:func:`mpxlite.cells.asymmetric.asymmetric_unit`, ITA domain about the
   detected centre) and cuts the prism :math:`F \times [z_0, z_1]` from the
   geometry by OpenCASCADE Booleans: the tagged entities, translated by every
   lattice vector ``R`` whose window meets :math:`F`, are intersected with
   the prism (``occ.intersect``), and the pieces and the vertical cut faces
   on the preimages of the window lines are fragmented together
   (``occ.fragment``, which glues the pieces of neighbouring windows); each
   piece is classified to the tag of the entity holding it;
3. constrains the faces of the prism paired by an operation of the group
   (rotation, glide, lattice translation: ``gmsh.model.mesh.setPeriodic``
   with that affine map; mirror faces need nothing) and meshes the unit with
   the size field of the :class:`~mpxlite.cells.autosym.MeshSizing`, after
   an estimate of the number of elements against a limit
   (:class:`MeshLimits`);
4. bisects every edge that joins two nodes of one orbit
   (:func:`~mpxlite.cells.symmetric.split_orbit_edges`; such edges occur at
   the rotation axes where two paired faces meet, ``p4`` / ``p4g``), so
   that no element holds two nodes of one orbit;
5. replicates the unit by the operations of the group
   (:func:`~mpxlite.cells.symmetric.replicate_group`: images folded into the
   window, tags permuted by the detected permutation -- a tag cut by the
   unit is rebuilt with the tag of each image --, nodes merged), extracts
   the periodic faces from the boundary of the tetrahedra on the window
   faces, writes MSH 2.2 with the ``$Periodic`` links by ``T_1``, ``T_2``,
   checks the topology and renumbers the nodes so that every operation
   preserves the node order of every element
   (:func:`~mpxlite.cells.spacegroup.renumber_symmetric`).

The mesh is invariant under the group to round-off, whatever the size
field. The model is consumed (the cut replaces its entities).
:func:`tagged_cell` reads the physical groups of the current model's mesh
as a :class:`~mpxlite.cells.detect.TaggedCell`; :func:`element_blocks`
collects the elements of an entity as a block of
:mod:`mpxlite.cells.symmetric`.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import gmsh
import numpy as np

from mpxlite.cells.asymmetric import AsymmetricUnit, UnitEdge
from mpxlite.cells.autosym import (
    DETECT_CIRCLE_POINTS,
    CurveMap,
    MeshSizing,
    SizeField,
    SymmetricMeshError,
    SymmetricMeshReport,
    TaggedModel,
    acyclic,
    affine,
    curve_root,
    near_note,
    on_window_line,
    renumber,
    supported_subgroup,
    write_replicated,
)
from mpxlite.cells.detect import (
    SIMPLEX_NODES,
    DetectedSymmetry,
    TaggedCell,
    TagGeometry,
    default_class,
    detect_symmetry,
)
from mpxlite.cells.spacegroup import _lattice
from mpxlite.cells.symmetric import Block, replicate_group, split_orbit_edges
from mpxlite.gmshjob import GmshJobError

__all__ = [
    "DEFAULT_MESH_OPTIONS",
    "MeshLimits",
    "build_symmetric_mesh",
    "curve_pieces",
    "element_blocks",
    "estimate_elements",
    "read_tagged_cell",
    "tagged_cell",
]

DEFAULT_MESH_OPTIONS: Mapping[str, float] = MappingProxyType(
    {"Mesh.Algorithm": 6, "Mesh.Algorithm3D": 1, "Mesh.Optimize": 0}
)
"""Mesh options of the unit mesh (Frontal-Delaunay 2-D, Delaunay 3-D, no optimisation)."""

_TET_PER_VOLUME = 6.0 * math.sqrt(2.0)
"""Regular tetrahedra of edge ``h`` per volume ``h^3``."""


@dataclass(frozen=True)
class MeshLimits:
    """Bounds of a mesh generation.

    Attributes:
        max_elements: Largest number of tetrahedra of the unit mesh. It is
            estimated from the size field before the generation (refused
            above it, a mesh that would run for ever) and checked after.
        options: Gmsh mesh options of the unit mesh (over
            :data:`DEFAULT_MESH_OPTIONS`).
        check: Called with the name of each step after it ran (the job
            checks the messages of Gmsh there), or ``None``.
    """

    max_elements: int = 5_000_000
    options: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_MESH_OPTIONS))
    check: Callable[[str], None] | None = None

    def after(self, step: str) -> None:
        """Run :attr:`check` after ``step``."""
        if self.check is not None:
            self.check(step)


def tagged_cell(lattice: object, *, generate: bool = True) -> TaggedCell:
    """The physical groups of the current Gmsh model (meshed first if it has no mesh)."""
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    if not len(node_tags) and generate:
        gmsh.model.mesh.generate(max(d for d, _ in gmsh.model.getPhysicalGroups()))
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    index = np.full(int(np.max(node_tags)) + 1, -1, dtype=np.int64)
    index[np.asarray(node_tags, dtype=np.int64)] = np.arange(len(node_tags))
    groups: dict[int, TagGeometry] = {}
    for dim, tag in gmsh.model.getPhysicalGroups():
        if tag in groups:
            raise ValueError(f"physical tag {tag} used in dimensions {groups[tag].dim}, {dim}")
        cells = []
        for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
            etypes, _, enodes = gmsh.model.mesh.getElements(dim, int(ent))
            for et, nodes in zip(etypes, enodes, strict=True):
                if int(et) not in SIMPLEX_NODES:
                    raise ValueError(f"element type {et} (tag {tag}) is not a simplex")
                _, nn = SIMPLEX_NODES[int(et)]
                nper = gmsh.model.mesh.getElementProperties(int(et))[3]
                conn = np.asarray(nodes, dtype=np.int64).reshape(-1, nper)[:, :nn]
                cells.append(index[conn])
        if cells:
            groups[tag] = TagGeometry(
                dim, np.concatenate(cells), gmsh.model.getPhysicalName(dim, tag)
            )
    return TaggedCell(lattice, xyz, groups)


def read_tagged_cell(path: Path | str, lattice: object) -> TaggedCell:
    """The physical groups of a mesh file of any format Gmsh reads.

    The file is opened in a new model, which is removed afterwards; the
    current model of the caller (if any) is current again.
    """
    caller = gmsh.model.getCurrent() if gmsh.model.list() else None
    gmsh.open(str(Path(path).expanduser().resolve()))
    try:
        return tagged_cell(lattice, generate=False)
    finally:
        if caller is not None and gmsh.model.getCurrent() != caller:
            gmsh.model.remove()
            gmsh.model.setCurrent(caller)


def element_blocks(
    blocks: list[Block], kind: str, etype: int, dim: int, tag: int, index: Mapping[int, int]
) -> None:
    """Append the elements of Gmsh entity ``(dim, tag)`` to ``blocks`` as ``kind``.

    Needs a generated mesh. ``index`` maps the Gmsh node tags to rows of the
    node array; every element of the entity must be of type ``etype``
    (4: tetrahedron, 2: triangle).

    Raises:
        RuntimeError: on an element of another type.
    """
    types, _, conn = gmsh.model.mesh.getElements(dim, tag)
    for t, c in zip(types, conn, strict=True):
        if int(t) != etype:
            raise RuntimeError(f"unexpected element type {t} on entity ({dim}, {tag})")
        nn = 4 if etype == 4 else 3
        arr = np.array([index[int(n)] for n in c], dtype=np.int64).reshape(-1, nn)
        blocks.append((kind, etype, arr))


def _points(dim: int, tag: int) -> np.ndarray:
    """``(k, 3)`` coordinates of the corner points of an entity."""
    if dim == 0:
        return np.asarray(gmsh.model.getValue(0, tag, []), dtype=float).reshape(1, 3)
    pts = gmsh.model.getBoundary([(dim, tag)], combined=False, oriented=False, recursive=True)
    return np.array([gmsh.model.getValue(0, abs(p), []) for _, p in pts], dtype=float).reshape(
        -1, 3
    )


def curve_pieces(tag: int, k: int) -> np.ndarray:
    """The ``k + 1`` ends of the ``k`` pieces of equal parameter length of a curve."""
    lo, hi = gmsh.model.getParametrizationBounds(1, tag)
    par = np.linspace(lo[0], hi[0], k + 1)
    return np.asarray(gmsh.model.getValue(1, tag, par.tolist()), dtype=float).reshape(-1, 3)


def _curve_samples(tag: int, n: int) -> np.ndarray:
    """Points along a curve (its end points for a straight line)."""
    lo, hi = gmsh.model.getParametrizationBounds(1, tag)
    k = 2 if gmsh.model.getType(1, tag) == "Line" else max(2, n)
    par = np.linspace(lo[0], hi[0], k)
    return np.asarray(gmsh.model.getValue(1, tag, par.tolist()), dtype=float).reshape(-1, 3)


def _read_model(lattice: np.ndarray, window_lo: Sequence[float] | None) -> TaggedModel:
    groups: dict[int, tuple[int, str, list[int]]] = {}
    owner: dict[tuple[int, int], int] = {}
    for dim, tag in gmsh.model.getPhysicalGroups():
        if dim < 2:
            raise SymmetricMeshError(f"physical group {tag} of dimension {dim}: only 2 and 3")
        if tag in groups:
            raise SymmetricMeshError(f"physical tag {tag} used in two dimensions")
        ents = [int(e) for e in gmsh.model.getEntitiesForPhysicalGroup(dim, tag)]
        for e in ents:
            if (dim, e) in owner:
                raise SymmetricMeshError(
                    f"entity ({dim}, {e}) in groups {owner[(dim, e)]} and {tag}"
                )
            owner[(dim, e)] = tag
        groups[tag] = (dim, gmsh.model.getPhysicalName(dim, tag), ents)
    vols = [t for _, t in gmsh.model.getEntities(3)]
    if not vols:
        raise SymmetricMeshError("the model has no volume")
    for v in vols:
        if (3, v) not in owner:
            raise SymmetricMeshError(f"volume {v} is in no physical group")
    pts = np.concatenate([_points(3, v) for v in vols])
    lo = (
        np.min(pts[:, :2] @ np.linalg.inv(lattice), axis=0)
        if window_lo is None
        else np.asarray(window_lo, dtype=float).reshape(2)
    )
    return TaggedModel(groups, lo, (float(pts[:, 2].min()), float(pts[:, 2].max())))


def _chord_sag() -> float:
    """Largest distance (m) of the current mesh to the exact OCC geometry it discretises.

    Sampled at the midpoints of the mesh edges on every curve that is not a
    straight line, and at the edge midpoints and centroids of the triangles
    on every surface that is not a plane (``gmsh.model.getClosestPoint``:
    exact to the OCC precision). Polygonal / planar geometry: 0.
    """
    worst = 0.0
    for dim, kind in ((1, "Line"), (2, "Plane")):
        for _, ent in gmsh.model.getEntities(dim):
            if gmsh.model.getType(dim, ent) == kind:
                continue
            types, _, conn = gmsh.model.mesh.getElements(dim, ent)
            tags, xyz, _ = gmsh.model.mesh.getNodes(dim, ent, includeBoundary=True)
            if not len(tags):
                continue
            row = np.full(int(np.max(tags)) + 1, -1, dtype=np.int64)
            row[np.asarray(tags, dtype=np.int64)] = np.arange(len(tags))
            pts = np.asarray(xyz, dtype=float).reshape(-1, 3)
            for ty, c in zip(types, conn, strict=True):
                if int(ty) not in (1, 2):
                    continue
                v = pts[row[np.asarray(c, dtype=np.int64).reshape(-1, dim + 1)]]
                samples = [0.5 * (v[:, i] + v[:, (i + 1) % (dim + 1)]) for i in range(dim)]
                if dim == 2:
                    samples += [0.5 * (v[:, 2] + v[:, 0]), v.mean(axis=1)]
                s = np.concatenate(samples)
                cp, _ = gmsh.model.getClosestPoint(dim, ent, s.ravel().tolist())
                d = np.linalg.norm(np.asarray(cp, dtype=float).reshape(-1, 3) - s, axis=1)
                worst = max(worst, float(d.max(initial=0.0)))
    return worst


def _detect(
    lattice: np.ndarray, model: TaggedModel, h: float, tol: float | None = None, **kw: object
) -> DetectedSymmetry:
    """Space group of the model on a curvature-adapted throw-away mesh.

    The verification of :func:`~mpxlite.cells.detect.detect_symmetry`
    compares mesh nodes (on the exact geometry) with mesh simplices (chords):
    a curved boundary matches its image only to the chord sag of the mesh,
    and gmsh discretises two congruent curved entities as translates, not as
    rotations / reflections of each other. The throw-away mesh therefore
    resolves every curved entity with :data:`DETECT_CIRCLE_POINTS` nodes per
    turn, the sag achieved is measured against the OCC geometry
    (:func:`_chord_sag`) and the detection tolerance is at least twice that
    sag (``tol`` default: ``1e-6`` of the cell size, unchanged on polygonal
    geometry).

    A ``p1`` result with operations broken by at most ``near_tol`` (a
    symmetry possibly missed) is kept: :func:`near_note` reports it.
    """
    keys = ("Mesh.MeshSizeMax", "Mesh.MeshSizeMin", "Mesh.MeshSizeFromCurvature",
            "Mesh.MinimumCirclePoints", "Mesh.MinimumCurvePoints",
            "Mesh.MeshSizeExtendFromBoundary")  # fmt: skip
    saved = {k: gmsh.option.getNumber(k) for k in keys}
    scale = math.sqrt(abs(float(np.linalg.det(lattice))))
    try:
        for k, v in zip(keys, (h, 0.0, DETECT_CIRCLE_POINTS, DETECT_CIRCLE_POINTS, 3, 0),
                        strict=True):  # fmt: skip
            gmsh.option.setNumber(k, v)
        gmsh.model.mesh.generate(3)
        sag = _chord_sag()
        cell = tagged_cell(lattice, generate=False)
        cell = TaggedCell(cell.lattice, cell.points, cell.groups, window_lo=model.lo)
    finally:
        gmsh.model.mesh.clear()
        for k, v in saved.items():
            gmsh.option.setNumber(k, v)
    base = 1e-6 * scale if tol is None else float(tol)
    return detect_symmetry(cell, tol=max(base, 2.0 * sag), **kw)  # type: ignore[arg-type]


def _window_sides(
    model: TaggedModel, tags: Sequence[int], lattice: np.ndarray, tol: float
) -> dict[int, int]:
    """Window tag -> side ``k`` (``u_1 = lo_1``, ``lo_1 + 1``, ``u_2 = lo_2``, ``lo_2 + 1``)."""
    tinv = np.linalg.inv(lattice)
    utol = tol / float(np.min(np.linalg.norm(lattice, axis=1)))
    out = {}
    for tag in tags:
        dim, name, ents = model.groups[tag]
        u = np.concatenate([_points(dim, e) for e in ents])[:, :2] @ tinv
        side = None
        for k in range(4):
            axis, s = divmod(k, 2)
            if np.all(np.abs(u[:, axis] - model.lo[axis] - s) <= utol):
                side = k
        if side is None:
            raise SymmetricMeshError(f"window group {tag} ({name}) is not on one window face")
        out[tag] = side
    return out


def _matches(name: str, tag: int, wanted: Sequence[str]) -> bool:
    return name in wanted or default_class(tag, name) in wanted


def _edge_segments(
    model: TaggedModel, sizing: MeshSizing, lattice: np.ndarray, window: set[int], tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Segments ``(n, 2, 3)`` of the refined edges in the window and their edge sizes ``(n,)``.

    The curves on the window lines are excluded.
    """
    if sizing.curve_edges is not None:
        size_of = {int(c): h for c, h in sizing.curve_edges.items()}
    else:
        size_of = {}
        for tag, (dim, name, ents) in model.groups.items():
            if dim != 2 or tag in window or not _matches(name, tag, sizing.edge_groups):
                continue
            bnd = gmsh.model.getBoundary([(2, e) for e in ents], combined=False, oriented=False)
            size_of.update({abs(c): float(sizing.edge or sizing.plane) for _, c in bnd})
    segs, sizes = [], []
    for c in sorted(size_of):
        h = np.ravel(np.asarray(size_of[c], dtype=float))
        p = _curve_samples(c, sizing.curve_samples) if len(h) == 1 else curve_pieces(c, len(h))
        if not on_window_line(p, lattice, model.lo, tol):
            segs.append(np.stack([p[:-1], p[1:]], axis=1))
            sizes.append(np.broadcast_to(h, (len(p) - 1,)))
    if not segs:
        return np.zeros((0, 2, 3)), np.zeros(0)
    return np.concatenate(segs), np.concatenate(sizes)


def _prism(verts: np.ndarray, z0: float, z1: float) -> int:
    occ = gmsh.model.occ
    pts = [occ.addPoint(x, y, z0) for x, y in verts]
    lines = [occ.addLine(pts[k], pts[(k + 1) % len(pts)]) for k in range(len(pts))]
    surf = occ.addPlaneSurface([occ.addCurveLoop(lines)])
    out = occ.extrude([(2, surf)], 0, 0, z1 - z0)
    return next(t for d, t in out if d == 3)


def _rectangle(a: np.ndarray, b: np.ndarray, z0: float, z1: float) -> int:
    occ = gmsh.model.occ
    pts = [occ.addPoint(a[0], a[1], z0), occ.addPoint(b[0], b[1], z0),
           occ.addPoint(b[0], b[1], z1), occ.addPoint(a[0], a[1], z1)]  # fmt: skip
    lines = [occ.addLine(pts[k], pts[(k + 1) % 4]) for k in range(4)]
    return occ.addPlaneSurface([occ.addCurveLoop(lines)])


def _interior_points(dim: int, tag: int, n: int = 12) -> list[np.ndarray]:
    """Points inside an entity: its centre of mass if inside, else a parametric / box sample."""
    c = np.asarray(gmsh.model.occ.getCenterOfMass(dim, tag))
    if gmsh.model.isInside(dim, tag, c.tolist()) and _on_entity(dim, tag, c):
        return [c]
    out = []
    if dim == 2:
        lo, hi = gmsh.model.getParametrizationBounds(2, tag)
        uv = [(lo[0] + (hi[0] - lo[0]) * (i + 0.5) / n, lo[1] + (hi[1] - lo[1]) * (j + 0.5) / n)
              for i in range(n) for j in range(n)]  # fmt: skip
        pts = np.asarray(gmsh.model.getValue(2, tag, np.ravel(uv).tolist())).reshape(-1, 3)
    else:
        bb = np.asarray(gmsh.model.getBoundingBox(dim, tag)).reshape(2, 3)
        g = (np.arange(n) + 0.5) / n
        pts = bb[0] + np.stack(np.meshgrid(g, g, g), axis=-1).reshape(-1, 3) * (bb[1] - bb[0])
    for p in pts:
        if gmsh.model.isInside(dim, tag, p.tolist()) and _on_entity(dim, tag, p):
            out.append(p)
    return out


def _on_entity(dim: int, tag: int, p: np.ndarray) -> bool:
    """``p`` on the surface (``isInside`` projects onto it), or any point for a volume."""
    if dim != 2:
        return True
    q = np.asarray(gmsh.model.getClosestPoint(2, tag, p.tolist())[0])
    size = float(np.linalg.norm(np.ptp(np.asarray(gmsh.model.getBoundingBox(2, tag)).reshape(2, 3),
                                       axis=0)))  # fmt: skip
    return float(np.linalg.norm(q - p)) <= 1e-7 * max(size, 1e-300)  # OCC precision


def _classify(
    dim: int, tag: int, originals: _Tagged, lattice: np.ndarray, lo: np.ndarray
) -> int | None:
    """Tag of the original entity holding ``(dim, tag)`` (folded into the window), or ``None``.

    The original entities are kept in the model during the cut: the maps of
    the Boolean operations are not used (gmsh's ``fragment`` map of an
    entity it leaves unchanged can carry a stale tag). ``isInside`` may
    ignore the holes of a face (a patch inside a bare plane): of several
    claimants, the smallest holds the point. Majority over up to 3 points.
    """
    votes: dict[int | None, int] = {}
    tinv = np.linalg.inv(lattice)
    for p in _interior_points(dim, tag):
        r = np.floor(p[:2] @ tinv - lo + 1e-12)
        q = p.copy()
        q[:2] -= r @ lattice
        hits = [
            (mass, t_) for d, e, (bmin, bmax, t_, mass) in originals
            if d == dim and np.all(q >= bmin) and np.all(q <= bmax)
            and gmsh.model.isInside(d, e, q.tolist()) and _on_entity(d, e, q)
        ]  # fmt: skip
        # isInside may ignore the holes of a face: the innermost (smallest) claimant holds q
        key = min(hits)[1] if hits else None
        votes[key] = votes.get(key, 0) + 1
        if max(votes.values()) >= 3:
            break
    if not votes:
        raise SymmetricMeshError(f"Boolean cut: entity ({dim}, {tag}) cannot be classified")
    best = max(votes.items(), key=lambda kv: kv[1])
    if sum(votes.values()) > 1 and best[1] < 0.75 * sum(votes.values()):
        raise SymmetricMeshError(f"Boolean cut: entity ({dim}, {tag}) straddles tags {votes}")
    return best[0]


_Tagged = list[tuple[int, int, tuple[np.ndarray, np.ndarray, int, float]]]


"""``(dim, entity, (bbox min, bbox max, tag, measure))`` of the user's tagged entities."""


def _originals(model: TaggedModel, window: set[int], tol: float) -> tuple[_Tagged, list]:
    """The tagged entities (window faces excluded) and the entities to copy (volumes, free faces).

    ``free``: the tagged surfaces on no volume boundary (copied on their own).
    """
    tagged: _Tagged = []
    free: list[tuple[int, int]] = []
    for tag, (dim, _, ents) in sorted(model.groups.items()):
        if tag in window:
            continue
        for e in ents:
            bb = np.asarray(gmsh.model.getBoundingBox(dim, e)).reshape(2, 3)
            mass = gmsh.model.occ.getMass(dim, e)
            tagged.append((dim, e, (bb[0] - 10 * tol, bb[1] + 10 * tol, tag, mass)))
            if dim == 2 and not gmsh.model.getAdjacencies(2, e)[0].size:
                free.append((2, e))
    return tagged, gmsh.model.getEntities(3) + free


def _pieces(
    unit: AsymmetricUnit, originals: Sequence[tuple[int, int]], z: tuple[float, float],
    scale: float,
) -> list[tuple[int, int]]:  # fmt: skip
    """Copies of the cell translated into the unit, cut by the prism and the cut faces."""
    occ = gmsh.model.occ
    z0, z1 = z
    dz = 1e-2 * (z1 - z0)
    pieces: list[tuple[int, int]] = []
    for r in unit.translations:
        shift = np.asarray(r, dtype=float) @ unit.lattice
        cp = occ.copy(originals)
        occ.translate(cp, float(shift[0]), float(shift[1]), 0.0)
        prism = _prism(unit.vertices, z0 - dz, z1 + dz)
        out, _ = occ.intersect(cp, [(3, prism)], removeObject=True, removeTool=True)
        pieces += [c for c in dict.fromkeys(out) if c not in pieces]
    ext = 1e-3 * scale
    rects = []
    for seg in unit.cuts:
        d = (seg[1] - seg[0]) / np.linalg.norm(seg[1] - seg[0])
        rects.append((2, _rectangle(seg[0] - ext * d, seg[1] + ext * d, z0 - dz, z1 + dz)))
    if len(unit.translations) > 1 or rects:
        out, _ = occ.fragment(pieces, rects)
        pieces = list(dict.fromkeys(out))
    occ.synchronize()
    return pieces


def _tags_of(
    pieces: Sequence[tuple[int, int]], tagged: _Tagged, free: bool, model: TaggedModel,
    lattice: np.ndarray,
) -> tuple[dict[int, int], dict[int, int]]:  # fmt: skip
    """``(volume -> tag, tagged surface -> tag)`` of the unit, by :func:`_classify`."""
    keep_v = sorted({t_ for d, t_ in pieces if d == 3})
    vol_tag: dict[int, int] = {}
    for v in keep_v:
        tag = _classify(3, v, tagged, lattice, model.lo)
        if tag is None:
            raise SymmetricMeshError(f"Boolean cut: volume {v} of the unit has no tag")
        vol_tag[v] = tag
    bnd = {abs(s_) for _, s_ in gmsh.model.getBoundary(
        [(3, v) for v in keep_v], combined=False, oriented=False)}  # fmt: skip
    surf_tag: dict[int, int] = {}
    for s_ in sorted(bnd):
        tag = _classify(2, s_, tagged, lattice, model.lo)
        if tag is not None:
            surf_tag[s_] = tag
    if free:  # free tagged surfaces: their pieces inside the unit (the loose cut pieces are not)
        for s_ in sorted({t_ for d, t_ in pieces if d == 2} - bnd):
            try:
                tag = _classify(2, s_, tagged, lattice, model.lo)
            except SymmetricMeshError:
                continue
            if tag is not None:
                surf_tag[s_] = tag
    return vol_tag, surf_tag


def _cut_unit(
    model: TaggedModel, unit: AsymmetricUnit, window: set[int], scale: float, tol: float
) -> tuple[dict[int, int], dict[int, int]]:
    """Replace the model by the unit prism; ``(volume -> tag, tagged surface -> tag)``."""
    occ = gmsh.model.occ
    tagged, originals = _originals(model, window, tol)
    pieces = _pieces(unit, originals, model.z, scale)
    vol_tag, surf_tag = _tags_of(pieces, tagged, any(d == 2 for d, _ in originals), model,
                                 unit.lattice)  # fmt: skip
    # drop the originals and the parts of the cut faces outside the volumes
    occ.remove(originals, recursive=True)
    occ.synchronize()
    loose = [(2, s_) for _, s_ in gmsh.model.getEntities(2)
             if s_ not in surf_tag and not gmsh.model.getAdjacencies(2, s_)[0].size]  # fmt: skip
    if loose:
        occ.remove(loose, recursive=True)
        occ.synchronize()
    vols = {t_ for _, t_ in gmsh.model.getEntities(3)}
    if vols != set(vol_tag):
        raise SymmetricMeshError(f"Boolean cut: {len(vols ^ set(vol_tag))} volume(s) lost")
    if not set(surf_tag) <= {t_ for _, t_ in gmsh.model.getEntities(2)}:
        raise SymmetricMeshError("Boolean cut: tagged surface(s) lost")
    gmsh.model.removePhysicalGroups()
    return vol_tag, surf_tag


def _unit_faces(unit: AsymmetricUnit, tol: float) -> dict[int, list[int]]:
    """Edge index of the unit -> the surfaces of the model on its vertical face."""
    out: dict[int, list[int]] = {e.index: [] for e in unit.edges}
    for _, s in gmsh.model.getEntities(2):
        p = np.vstack([_points(2, s), gmsh.model.occ.getCenterOfMass(2, s)])
        for e in unit.edges:
            d = e.b - e.a
            ln = float(np.linalg.norm(d))
            q = p[:, :2] - e.a
            perp = np.abs(q @ np.array([d[1], -d[0]])) / ln
            along = q @ d / ln
            if np.all(perp <= tol) and np.all(along >= -tol) and np.all(along <= ln + tol):
                out[e.index].append(s)
                break
    return out


def _curves(surf: int) -> list[int]:
    return sorted({abs(c) for _, c in gmsh.model.getBoundary(
        [(2, surf)], combined=False, oriented=False)})  # fmt: skip


def _pair_plan(
    e: UnitEdge, faces: dict[int, list[int]], com: Mapping[int, np.ndarray], tol: float
) -> tuple[list[tuple[int, int]], np.ndarray, list[tuple[int, int]]]:
    """``(surface pairs (slave, master), affine, curve pairs (slave, master))`` of a paired edge."""
    assert e.g is not None and e.t is not None and e.partner is not None
    occ = gmsh.model.occ
    slaves = list(faces[e.index])
    masters = faces[e.partner]
    if len(slaves) != len(masters):
        raise SymmetricMeshError(
            f"edges {e.partner} -> {e.index} ({e.op}): {len(masters)} / {len(slaves)} "
            "surfaces (the geometry is not invariant at the unit boundary)"
        )
    pairs, cpairs = [], []
    for m in masters:
        c = com[m].copy()
        c[:2] = e.g @ c[:2] + e.t
        hit = [s for s in slaves if np.max(np.abs(com[s] - c)) <= tol]
        if len(hit) != 1:
            raise SymmetricMeshError(
                f"edges {e.partner} -> {e.index} ({e.op}): surface {m} has no image"
            )
        slaves.remove(hit[0])
        pairs.append((hit[0], m))
        cs = {c_: np.asarray(occ.getCenterOfMass(1, c_)) for c_ in _curves(hit[0])}
        for cm in _curves(m):
            x = np.asarray(occ.getCenterOfMass(1, cm))
            x[:2] = e.g @ x[:2] + e.t
            img = [c_ for c_, y in cs.items() if np.max(np.abs(y - x)) <= tol]
            if len(img) != 1:
                raise SymmetricMeshError(f"curve {cm} of surface {m} has no image")
            if img[0] != cm:
                cpairs.append((img[0], cm))
    return pairs, affine(e.g, e.t), cpairs


def _same_on_curve(c: int, a1: np.ndarray, a2: np.ndarray, tol: float) -> bool:
    """``a1`` and ``a2`` map the points of curve ``c`` alike (point by point, within ``tol``)."""
    p = _curve_samples(c, 17)
    q1 = p @ a1[:3, :3].T + a1[:3, 3]
    q2 = p @ a2[:3, :3].T + a2[:3, 3]
    return bool(np.max(np.linalg.norm(q1 - q2, axis=1)) <= tol)


def _unify_curve_masters(
    cmap: CurveMap, expect: Sequence[tuple[int, int, np.ndarray]], tol: float
) -> list[tuple[int, int, np.ndarray]]:
    """Curve links that make every expected copy ``slave = A master`` hold; refuse a conflict.

    ``cmap``: the curve masters gmsh ends with (the last pair wins);
    ``expect``: every ``(slave, master, A)`` a pair of faces relies on. A
    slave whose final master descends from another root curve is made
    consistent by linking that root to the expected one (one more
    ``setPeriodic`` on the curves); a slave whose final master descends from
    the expected root by another map cannot be (``SymmetricMeshError``).
    """
    links: list[tuple[int, int, np.ndarray]] = []
    changed = True
    while changed:
        changed = False
        for sl, ma, a in expect:
            r1, a1 = curve_root(cmap, sl)
            r2, a2 = curve_root(cmap, ma)
            a2 = a @ a2
            if r1 == r2:
                if not _same_on_curve(r1, a1, a2, tol):
                    raise SymmetricMeshError(
                        f"curve {sl}: two pairs of faces copy its mesh from curve {r1} by "
                        "different maps (inconsistent curve masters across the paired faces "
                        "of the unit)"
                    )
                continue
            b = np.linalg.inv(a1) @ a2  # mesh(r1) = b mesh(r2)
            ends = _curve_samples(r2, 2) @ b[:3, :3].T + b[:3, 3]
            if not np.allclose(np.sort(ends, axis=0), np.sort(_curve_samples(r1, 2), axis=0),
                               rtol=0, atol=tol):  # fmt: skip
                raise SymmetricMeshError(
                    f"curve {sl}: its final master curve {r1} is not the image of the expected "
                    f"curve {r2} (the paired faces of the unit do not match)"
                )
            cmap[r1] = (r2, b)
            links.append((r1, r2, b))
            changed = True
    return links


def _pair_faces(unit: AsymmetricUnit, faces: dict[int, list[int]], tol: float) -> int:
    """``setPeriodic`` on every surface of a paired edge; count.

    gmsh gives the curves of a slave surface the curves of its master as
    masters (the last setting wins). Curves of one orbit may then form a
    cycle of masters (``pg``: the glide twice is a lattice translation),
    which gmsh does not mesh; each pair is therefore oriented (which side is
    the master) so that the curve masters stay acyclic. A curve that a later
    pair gives another master than an earlier pair expects is made
    consistent by a curve link between the two root curves
    (:func:`_unify_curve_masters`), or refused.

    Raises:
        SymmetricMeshError: no orientation of a pair avoids a cycle, or two
            pairs copy a curve mesh by different maps.
    """
    com = {s: np.asarray(gmsh.model.occ.getCenterOfMass(2, s)) for f in faces.values() for s in f}
    cmap: CurveMap = {}
    expect: list[tuple[int, int, np.ndarray]] = []
    n = 0
    for e in unit.pairs():
        pairs, aff, cpairs = _pair_plan(e, faces, com, tol)
        for flip in (False, True):
            oriented = [(ma, sl) if flip else (sl, ma) for sl, ma in cpairs]
            if acyclic({k: v[0] for k, v in cmap.items()} | dict(oriented)):
                break
        else:
            raise SymmetricMeshError("no orientation of the paired faces avoids a cycle of curves")
        a = np.linalg.inv(aff) if flip else aff
        for sl, ma in oriented:
            cmap[sl] = (ma, a)
            expect.append((sl, ma, a))
        for sl, ma in pairs:
            if flip:
                sl, ma = ma, sl
            gmsh.model.mesh.setPeriodic(2, [sl], [ma], a.ravel().tolist())
            n += 1
    for sl, ma, b in _unify_curve_masters(cmap, expect, tol):
        gmsh.model.mesh.setPeriodic(1, [sl], [ma], b.ravel().tolist())
    return n


def _bulk_of(
    vol_tag: Mapping[int, int], names: Mapping[int, str], sizing: MeshSizing
) -> dict[tuple[int, int], float]:
    """``(dim, tag) -> bulk size`` of every entity (the smallest of the adjacent volumes)."""
    out: dict[tuple[int, int], float] = {}
    for v, tag in vol_tag.items():
        name = names[tag]
        if name not in sizing.bulk:
            raise SymmetricMeshError(f"no bulk size for volume group {name!r}")
        h = float(sizing.bulk[name])
        level = [(3, v)]
        while level:
            for ent in level:
                out[ent] = min(h, out.get(ent, math.inf))
            if level[0][0] == 0:
                break
            bnd = gmsh.model.getBoundary(level, combined=False, oriented=False)
            level = sorted({(d, abs(t)) for d, t in bnd})
    return out


def _unit_mesh(
    vol_tag: Mapping[int, int], surf_tag: Mapping[int, int], names: Mapping[int, str]
) -> tuple[np.ndarray, list[Block]]:
    node_tags, xyz, _ = gmsh.model.mesh.getNodes()
    index = np.full(int(np.max(node_tags)) + 1, -1, dtype=np.int64)
    index[np.asarray(node_tags, dtype=np.int64)] = np.arange(len(node_tags))
    coords = np.asarray(xyz, dtype=float).reshape(-1, 3)
    per: dict[tuple[str, int], list[np.ndarray]] = {}
    for dim, etype, nn, ents in ((3, 4, 4, vol_tag), (2, 2, 3, surf_tag)):
        for e, tag in ents.items():
            types, _, conn = gmsh.model.mesh.getElements(dim, e)
            for ty, c in zip(types, conn, strict=True):
                if int(ty) != etype:
                    raise SymmetricMeshError(f"element type {ty} on entity ({dim}, {e})")
                per.setdefault((names[tag], etype), []).append(
                    index[np.asarray(c, dtype=np.int64)].reshape(-1, nn)
                )
    blocks = [(k, et, np.concatenate(v)) for (k, et), v in per.items()]
    return coords, blocks


def estimate_elements(size: SizeField, volumes: Sequence[int], samples: int = 6) -> float:
    """Estimated number of tetrahedra of the mesh of ``volumes`` with the size field.

    For each volume entity: its measure times the mean of ``1 / h^3`` on a
    ``samples^3`` grid of its bounding box, times the number of regular
    tetrahedra of edge ``h`` in ``h^3``.
    """
    g = (np.arange(samples) + 0.5) / samples
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    total = 0.0
    for v in volumes:
        bb = np.asarray(gmsh.model.getBoundingBox(3, v)).reshape(2, 3)
        pts = bb[0] + grid * (bb[1] - bb[0])
        inv = [1.0 / size(3, v, *p) ** 3 for p in pts]
        total += gmsh.model.occ.getMass(3, v) * float(np.mean(inv))
    return _TET_PER_VOLUME * total


def _mesh_unit(size: SizeField, sizing: MeshSizing, limits: MeshLimits) -> None:
    """Mesh the unit with the size field, within the limits.

    Raises:
        GmshJobError: kind ``gmsh``, code ``mesh.too_many_elements``: the
            estimate before, or the count after, the generation exceeds
            ``limits.max_elements``.
    """
    volumes = [t for _, t in gmsh.model.getEntities(3)]
    estimate = estimate_elements(size, volumes)
    if estimate > limits.max_elements:
        raise GmshJobError(
            "gmsh",
            "mesh.too_many_elements",
            f"the sizes would give about {estimate:.3g} tetrahedra in the unit of the cell, "
            f"more than the limit of {limits.max_elements}",
            {"estimate": estimate, "limit": limits.max_elements, "when": "before"},
        )
    gmsh.model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: size(dim, tag, x, y, z))
    options = {**DEFAULT_MESH_OPTIONS, **dict(limits.options)}
    for key, val in (("Mesh.MeshSizeFromPoints", 0), ("Mesh.MeshSizeFromCurvature", 0),
                     ("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.MeshSizeMin", sizing.h_min),
                     ("Mesh.MeshSizeMax", sizing.h_max), *options.items()):  # fmt: skip
        gmsh.option.setNumber(key, val)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.removeSizeCallback()
    limits.after("the mesh of the unit")
    _, tets, _ = gmsh.model.mesh.getElements(3)
    count = sum(len(t) for t in tets)
    if count > limits.max_elements:
        raise GmshJobError(
            "gmsh",
            "mesh.too_many_elements",
            f"the mesh of the unit of the cell has {count} elements, more than the limit "
            f"of {limits.max_elements}",
            {"count": count, "limit": limits.max_elements, "when": "after"},
        )


def _size_data(
    model: TaggedModel, sizing: MeshSizing, unit: AsymmetricUnit, window: set[int], tol: float
) -> tuple[list[float], np.ndarray, np.ndarray]:
    """The refined planes, edge segments of the periodic pattern near the unit, their sizes."""
    t = unit.lattice
    planes = sizing.planes
    if planes is None:
        lv = set()
        for tag, (dim, _, ents) in model.groups.items():
            if dim != 2 or tag in window:
                continue
            for e in ents:
                z = _points(2, e)[:, 2]
                if np.ptp(z) <= tol and model.z[0] + tol < z[0] < model.z[1] - tol:
                    lv.add(round(float(z[0]) / tol) * tol)
        planes = sorted(lv)
    segs, sizes = _edge_segments(model, sizing, t, window, tol)
    if not sizing.refines_edges or not len(segs):
        return list(planes), np.zeros((0, 2, 3)), np.zeros(0)
    cen = unit.vertices.mean(axis=0)
    rad = float(np.max(np.linalg.norm(unit.vertices - cen, axis=1))) + sizing.reach
    rep, rep_h = [], []
    for i in range(-2, 3):
        for j in range(-2, 3):
            sg = segs.copy()
            sg[:, :, :2] += np.array([i, j], dtype=float) @ t
            mid = sg.mean(axis=1)[:, :2]
            half = 0.5 * np.linalg.norm(sg[:, 1, :2] - sg[:, 0, :2], axis=1)
            near = np.linalg.norm(mid - cen, axis=1) <= rad + half
            rep.append(sg[near])
            rep_h.append(sizes[near])
    return list(planes), np.concatenate(rep), np.concatenate(rep_h)


def build_symmetric_mesh(
    output_path: Path | str,
    lattice: object,
    sizing: MeshSizing,
    *,
    symmetry: DetectedSymmetry | None = None,
    window_lo: Sequence[float] | None = None,
    classes: Mapping[int, Hashable] | None = None,
    directions: Mapping[int, Sequence[float]] | None = None,
    tol: float | None = None,
    numbering: bool = True,
    admissible: Callable[[np.ndarray], bool] | None = None,
    limits: MeshLimits | None = None,
) -> SymmetricMeshReport:
    """Build the symmetric periodic mesh of the current gmsh (OCC) model (see the module).

    Args:
        output_path: The MSH 2.2 file to write.
        lattice: Rows ``T_1``, ``T_2`` (m); a non-orthogonal lattice is meshed
            with ``p1`` or ``p2`` only.
        sizing: The density policy.
        symmetry: The space group (default: detected on a curvature-adapted
            throw-away mesh, with ``classes`` / ``directions`` / ``tol`` passed
            to :func:`~mpxlite.cells.detect.detect_symmetry`; see :func:`_detect`).
        window_lo: Lower window corner (lattice coordinates; default the
            smallest lattice coordinates of the volumes).
        classes: Tag classes for the detection (tags are exchanged only within
            a class; default from the physical names).
        directions: Tag -> direction of the tag for the detection.
        tol: Detection tolerance (m; default ``1e-6`` of the cell size), raised
            to twice the measured chord sag of the detection mesh on curved
            geometry.
        numbering: Renumber for the group
            (:func:`~mpxlite.cells.spacegroup.renumber_symmetric`, default);
            ``False``: the periodic numbering only.
        admissible: Cartesian ``(2, 2)`` matrix of an operation -> whether it
            may be used (for instance whether it preserves an anisotropic
            property of the parts); default: all.
        limits: Element limit, mesh options and the check of each step
            (default: :class:`MeshLimits`).

    The group used is the largest admissible subgroup of the detected one
    that has an asymmetric unit (preferring a half turn at equal order; at
    worst ``p1``); a ``p1`` with near symmetries is meshed as ``p1``. Both
    are stated in :attr:`SymmetricMeshReport.notes`, never refused.

    Returns:
        The :class:`SymmetricMeshReport`.

    Raises:
        SymmetricMeshError: unsupported model (dimension-0/1 groups, an
            untagged volume), a Boolean or face-matching failure.
        MeshTopologyError: the replicated mesh is not conforming.
        GmshJobError: the element limit is exceeded, or a check of
            ``limits`` fails.
    """
    limits = MeshLimits() if limits is None else limits
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    t = _lattice(lattice)
    scale = math.sqrt(abs(float(np.linalg.det(t))))
    gtol = 1e-7 * scale  # geometric matching of OCC entities (OCC tolerance 1e-7)
    model = _read_model(t, window_lo)
    names = {tag: (name or f"tag_{tag}") for tag, (_, name, _) in model.groups.items()}
    if symmetry is None:
        symmetry = _detect(t, model, scale / 4, classes=classes, directions=directions, tol=tol)
        timings["detect"] = time.perf_counter() - t0
        limits.after("the detection of the symmetry")
    # the conformity warning is about the throw-away detection mesh, not about the cell
    notes = [w for w in symmetry.warnings if "periodic-conforming" not in w]
    near = near_note(symmetry)
    if near:
        notes.append(near)
    detected = symmetry
    symmetry, unit, sub_notes = supported_subgroup(symmetry, t, model.lo, admissible)
    notes += sub_notes
    window = set(symmetry.window_tags) & set(model.groups)
    sides = _window_sides(model, sorted(window), t, gtol)
    planes, segs, seg_sizes = _size_data(model, sizing, unit, window, gtol)
    t1 = time.perf_counter()
    vol_tag, surf_tag = _cut_unit(model, unit, window, scale, gtol)
    faces = _unit_faces(unit, gtol)
    n_pairs = _pair_faces(unit, faces, gtol)
    timings["cut"] = time.perf_counter() - t1
    limits.after("the cut of the asymmetric unit")
    t1 = time.perf_counter()
    bulk_of = _bulk_of(vol_tag, names, sizing)
    _mesh_unit(SizeField(sizing, planes, segs, seg_sizes, bulk_of), sizing, limits)
    coords, blocks = _unit_mesh(vol_tag, surf_tag, names)
    n_unit = sum(len(c) for _, e, c in blocks if e == 4)
    timings["mesh"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    ops = symmetry.ops
    coords, blocks, n_split = split_orbit_edges(
        coords, blocks, t, ops,
        interior=lambda p: unit.contains(p[:, :2], -1e-9 * scale),
    )  # fmt: skip
    kind_maps = [
        {names[a]: names[b] for a, b in perm.items() if a in names and b in names}
        for perm in symmetry.permutations
    ]
    nodes, elems = replicate_group(coords, blocks, t, ops, window_lo=model.lo, kind_maps=kind_maps)
    out, physical = write_replicated(output_path, nodes, elems, model, names, sides, t, gtol)
    timings["replicate"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    renumber(out, t, ops, model.lo, numbering)
    timings["numbering"] = time.perf_counter() - t1
    return SymmetricMeshReport(
        out, symmetry, unit, physical, len(nodes),
        sum(len(c) for _, e, c in elems if e == 4), n_unit, n_split, n_pairs, timings,
        None if detected is symmetry else detected, tuple(notes),
    )  # fmt: skip
