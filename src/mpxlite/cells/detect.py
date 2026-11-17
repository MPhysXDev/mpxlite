# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

r"""Space-group detection of a tagged periodic cell.

Input
    A :class:`TaggedCell`: the real-space lattice rows ``T_1``, ``T_2`` and
    the physical groups of a mesh as simplices -- points, segments,
    triangles, tetrahedra (:meth:`TaggedCell.from_msh`, or built directly;
    the toolkit of the Gmsh jobs builds one from a meshed Gmsh model). The
    tags are parameter labels: regions varied independently carry distinct
    tags; tags of one *class* (default: the physical name
    without its trailing ``_<int>`` indices, ``gap_0_0_1 -> gap``; an
    unnamed tag is its own class) may be exchanged by an operation, tags of
    different classes never. ``z`` is the stacking direction: the
    operations act in plane with ``z`` fixed (the ``z`` mirror of a
    substrate-backed cell is not a symmetry and is not looked for).

Method
    1. Lattice holohedry (:func:`lattice_holohedry`): every integer ``M``
       with :math:`M^T G M = G`, :math:`G = T T^T` (at most 12).
    2. Translation candidates for every ``M``: the components of the tags
       (connected on the torus, unwrapped across the window) that do not
       wrap around it carry a measure, a centroid and a second-moment
       tensor; the reference component (the rarest kind) maps onto a
       compatible component (same class, measure, tensor rotated by
       ``g``), which fixes :math:`t = c_1 - g c_0`. Without such components
       (stripes, layers only) or on a mesh that is not periodic-conforming,
       the candidates come from the nodes.
    3. Verification on ALL tags: the image of every tag must coincide with a
       tag of its class modulo the lattice -- symmetric Hausdorff distance
       of the nodes to the piecewise-linear sets (point-to-simplex,
       independent of the mesh; the nodes and the centroids of the
       simplices are tested), at most ``tol``. A volume is represented
       by its boundary on the torus (its faces on the window faces that
       meet their periodic partner are interior); the 2-D groups lying on
       the window faces (the Bloch faces) are excluded. Optional current
       directions of the tags (a tag with a direction) must map onto ``+-`` the image
       direction (sign reported).
    4. The verified operations (checked to close modulo the lattice) are
       reported as
       :class:`~mpxlite.cells.spacegroup.LatticeOp` objects (named and shifted
       relative to a centre: a site of the highest-order rotation, the one
       carrying the centroid of the lowest-tag component when there is a
       choice), with the induced tag permutation, the wallpaper-group symbol
       of the in-plane action (:func:`wallpaper_name`) and the subgroup that
       fixes every tag.

Continuous symmetry
    When a generic translation verifies (a laterally uniform stack, or
    stripes), the shifts are not discrete: the group is then the lattice
    point operations about the window centre that verify (no fractional
    translation), with a warning.

Near symmetries
    Candidates (and lattice operations of a nearly symmetric lattice) broken
    by more than ``tol`` but by at most ``near_tol`` are reported as
    :class:`NearSymmetry`, never symmetrised.

Tolerance
    Default ``tol = 1e-6`` and ``near_tol = 1e-2`` of the cell size
    :math:`\sqrt{|\det T|}`. Flat (polygonal / planar) geometry matches to
    round-off on any mesh; a curved boundary matches to the chord sag of
    its discretisation, which ``tol`` must then exceed.

Non-rectangular lattices
    The holohedry and the group are found for any lattice (oblique,
    rectangular, centred rectangular, square, hexagonal); the symmetric
    meshing (:func:`~mpxlite.cells.asymmetric.asymmetric_unit`, the job
    ``periodic_cell_mesh``) supports every
    rectangular / square group and ``p1`` / ``p2`` on any lattice, so another
    group on a lattice with non-orthogonal vectors is flagged
    (:attr:`DetectedSymmetry.warnings`).
"""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from mpxlite.cells.periodic import UnsupportedMeshFormatError
from mpxlite.cells.spacegroup import LatticeOp, _lattice, _read_msh2

__all__ = [
    "POINT_GROUP",
    "SIMPLEX_NODES",
    "DetectedSymmetry",
    "NearSymmetry",
    "TagGeometry",
    "TaggedCell",
    "default_class",
    "detect_symmetry",
    "lattice_holohedry",
    "point_group",
    "wallpaper_name",
]

SIMPLEX_NODES = {15: (0, 1), 1: (1, 2), 8: (1, 2), 2: (2, 3), 9: (2, 3), 4: (3, 4), 11: (3, 4)}
"""Gmsh element type -> (dimension, corner nodes) of the simplices read (orders 1 and 2)."""

_ANGLE_ORDER = (90.0, 0.0, 45.0, 135.0)
"""Mirror axes listed first, in the order of ``C4V_NAMES`` (``mx``, ``my``, ``md``, ``md'``)."""


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class TagGeometry:
    """One physical group: ``cells`` ``(m, dim + 1)`` rows of :attr:`TaggedCell.points`."""

    dim: int
    cells: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        if self.dim not in (0, 1, 2, 3):
            raise ValueError(f"dim must be 0..3, got {self.dim}")
        c = np.asarray(self.cells, dtype=np.int64).reshape(-1, self.dim + 1)
        if not len(c):
            raise ValueError(f"group {self.name!r} has no cells")
        c.setflags(write=False)
        object.__setattr__(self, "cells", c)


@dataclass(frozen=True, eq=False)
class TaggedCell:
    """A periodic cell: lattice rows ``T_1``, ``T_2`` (m), nodes ``(n, 3)``, groups by tag.

    ``window_lo``: the lower corner of the window in lattice coordinates
    (default: the smallest lattice coordinates of the nodes of the volumes,
    of all nodes without volumes); its faces
    ``u_i = lo_i``, ``lo_i + 1`` are the cut of the periodic cell.
    """

    lattice: np.ndarray
    points: np.ndarray
    groups: Mapping[int, TagGeometry]
    window_lo: np.ndarray | None = None

    def __post_init__(self) -> None:
        t = _lattice(self.lattice)
        p = np.asarray(self.points, dtype=float).reshape(-1, 3)
        if not self.groups:
            raise ValueError("no physical groups")
        for tag, g in self.groups.items():
            if g.cells.min() < 0 or g.cells.max() >= len(p):
                raise ValueError(f"group {tag}: node index out of range")
        if self.window_lo is None:
            vols = [g.cells.ravel() for g in self.groups.values() if g.dim == 3]
            vols = vols or [g.cells.ravel() for g in self.groups.values()]
            used = np.unique(np.concatenate(vols))
            lo = np.min(p[used, :2] @ np.linalg.inv(t), axis=0)
        else:
            lo = np.asarray(self.window_lo, dtype=float).reshape(2)
        object.__setattr__(self, "lattice", t)
        object.__setattr__(self, "points", p)
        object.__setattr__(self, "groups", dict(self.groups))
        object.__setattr__(self, "window_lo", lo)

    @classmethod
    def from_msh(cls, path: Path | str, lattice: object) -> TaggedCell:
        """The physical groups of a mesh file (MSH 2.x ASCII).

        Simplices of order 1 and 2 (their corner nodes); the tags of other
        element kinds raise ``ValueError``.

        Raises:
            UnsupportedMeshFormatError: the file is not MSH 2.x ASCII.
            ValueError: an element of a physical group is not a simplex, or
                a physical tag is used in two dimensions.
        """
        path = Path(path).expanduser().resolve()
        head = path.read_text(encoding="utf-8", errors="replace")[:200].split()
        if len(head) >= 2 and head[0] == "$MeshFormat" and head[1].startswith("2"):
            tags, xyz, elems = _read_msh2(path)
            names = _msh2_names(path)
            row = {int(v): k for k, v in enumerate(tags.tolist())}
            per: dict[int, tuple[int, list[list[int]]]] = {}
            for _num, etype, phys, nodes in elems:
                if phys == 0:
                    continue
                if etype not in SIMPLEX_NODES:
                    raise ValueError(f"{path}: element type {etype} (tag {phys}) is not a simplex")
                dim, nn = SIMPLEX_NODES[etype]
                d, cells = per.setdefault(phys, (dim, []))
                if d != dim:
                    raise ValueError(f"{path}: physical tag {phys} used in dimensions {d}, {dim}")
                cells.append([row[v] for v in nodes[:nn]])
            groups = {
                t: TagGeometry(d, np.array(c), names.get((d, t), "")) for t, (d, c) in per.items()
            }
            return cls(lattice, xyz, groups)
        raise UnsupportedMeshFormatError(
            f"{path}: TaggedCell.from_msh reads MSH 2.x ASCII only; write the mesh as MSH 2.2 "
            "ASCII (Gmsh options Mesh.MshFileVersion = 2.2, Mesh.Binary = 0)"
        )


def _msh2_names(path: Path) -> dict[tuple[int, int], str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if "$PhysicalNames" not in lines:
        return {}
    i = lines.index("$PhysicalNames")
    out = {}
    for line in lines[i + 2 : i + 2 + int(lines[i + 1])]:
        d, t, name = line.split(maxsplit=2)
        out[(int(d), int(t))] = name.strip().strip('"')
    return out


def default_class(tag: int, name: str) -> Hashable:
    """``gap_0_0_1 -> gap``; an unnamed tag is its own class."""
    if not name:
        return ("tag", tag)
    base = re.sub(r"(_\d+)+$", "", name)
    return base or name


# ---------------------------------------------------------------------------
# Lattice
# ---------------------------------------------------------------------------


def _lattice_vectors_of_norm(gram: np.ndarray, n2: float, rtol: float) -> list[np.ndarray]:
    ginv = np.linalg.inv(gram)
    b = [math.floor(math.sqrt(n2 * (1 + rtol) * ginv[k, k]) + 1e-9) for k in (0, 1)]
    out = []
    for i in range(-b[0], b[0] + 1):
        for j in range(-b[1], b[1] + 1):
            v = np.array([i, j], dtype=float)
            if abs(v @ gram @ v - n2) <= rtol * n2:
                out.append(v)
    return out


def lattice_holohedry(lattice: object, rtol: float = 1e-9) -> tuple[np.ndarray, ...]:
    """Every integer ``M`` with ``M^T G M = G`` (``G`` the Gram matrix), identity first.

    ``rtol``: relative tolerance on ``G``. The order is 2 (oblique), 4
    (rectangular, centred rectangular), 8 (square) or 12 (hexagonal).
    """
    t = _lattice(lattice)
    gram = t @ t.T
    scale = math.sqrt(gram[0, 0] * gram[1, 1])
    out = []
    for c1 in _lattice_vectors_of_norm(gram, gram[0, 0], rtol):
        for c2 in _lattice_vectors_of_norm(gram, gram[1, 1], rtol):
            if abs(c1 @ gram @ c2 - gram[0, 1]) <= rtol * scale:
                out.append(np.column_stack([c1, c2]).astype(np.int64))
    out.sort(key=lambda m: (not np.array_equal(m, np.eye(2)), m.ravel().tolist()))
    return tuple(out)


def _lattice_type(lattice: np.ndarray, order: int) -> str:
    if order == 12:
        return "hexagonal"
    if order == 8:
        return "square"
    if order == 2:
        return "oblique"
    centred = _centring(lattice, _reflection_matrices(lattice)[0]) == 2
    return "centred rectangular" if centred else "rectangular"


def _reflection_matrices(lattice: np.ndarray) -> list[np.ndarray]:
    return [m for m in lattice_holohedry(lattice) if round(np.linalg.det(m)) == -1]


def _axis(m: np.ndarray) -> np.ndarray:
    """Primitive integer vector (lattice coordinates) along the axis of a reflection ``M``."""
    a = m - np.eye(2, dtype=np.int64)
    for r in a:
        if np.any(r):
            v = np.array([r[1], -r[0]], dtype=np.int64)
            return v // math.gcd(int(v[0]), int(v[1]))
    raise ValueError("not a reflection")  # pragma: no cover


def _centring(lattice: np.ndarray, m: np.ndarray) -> int:
    """``|w| / s``: 1 (primitive) or 2 (centred) along the axis of ``M``.

    ``w`` is the primitive lattice vector along the axis, ``s`` the spacing
    of the projections of the lattice onto it.
    """
    w = _axis(m) @ lattice
    p = [float(lattice[k] @ w / (w @ w)) for k in (0, 1)]
    q = [Fraction(v).limit_denominator(12) for v in p]
    den = math.lcm(*(f.denominator for f in q))
    return den


# ---------------------------------------------------------------------------
# Geometry kernels
# ---------------------------------------------------------------------------


def _pt_seg(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    nn = np.einsum("ij,ij->i", ab, ab)
    s = np.einsum("ij,ij->i", p - a, ab) / np.where(nn > 0, nn, 1.0)
    s = np.clip(s, 0.0, 1.0)
    return np.linalg.norm(p - a - s[:, None] * ab, axis=1)


def _pt_tri(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ab, ac, ap = b - a, c - a, p - a
    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    d20 = np.einsum("ij,ij->i", ap, ab)
    d21 = np.einsum("ij,ij->i", ap, ac)
    den = d00 * d11 - d01 * d01
    ok = den > 1e-30 * np.maximum(d00 * d11, 1e-300)
    sden = np.where(ok, den, 1.0)
    v = (d11 * d20 - d01 * d21) / sden
    w = (d00 * d21 - d01 * d20) / sden
    inside = ok & (v >= 0) & (w >= 0) & (v + w <= 1)
    n = np.cross(ab, ac)
    nrm = np.linalg.norm(n, axis=1)
    dplane = np.abs(np.einsum("ij,ij->i", ap, n)) / np.where(nrm > 0, nrm, 1.0)
    out = dplane
    o = ~inside
    if o.any():
        po, ao, bo, co = p[o], a[o], b[o], c[o]
        e = np.minimum(_pt_seg(po, ao, bo), _pt_seg(po, bo, co))
        out[o] = np.minimum(e, _pt_seg(po, co, ao))
    return out


def _fold(x: np.ndarray, lattice: np.ndarray, lo: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``x`` translated by the lattice vector that brings ``ref`` into the window."""
    u = ref[..., :2] @ np.linalg.inv(lattice)
    d = np.floor(u - lo)
    out = np.array(x, dtype=float, copy=True)
    out[..., :2] -= d @ lattice if out.ndim == 2 else (d @ lattice)[:, None, :]
    return out


class _SimplexSet:
    """Distance (capped at ``cap``) from points of the window to a set of simplices on the torus."""

    def __init__(self, simp: np.ndarray, lattice: np.ndarray, lo: np.ndarray, cap: float) -> None:
        s = _fold(simp, lattice, lo, simp.mean(axis=1))
        reps = [s]
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if i or j:
                    r = s.copy()
                    r[..., :2] += np.array([i, j], dtype=float) @ lattice
                    reps.append(r)
        self.v = np.concatenate(reps)  # (m, k, 3)
        self.cap = float(cap)
        bmin = self.v.min(axis=1) - cap
        bmax = self.v.max(axis=1) + cap
        ext = float(np.percentile(np.max(bmax - bmin, axis=1), 75))
        h = max(ext, 2.0 * cap, 1e-300)
        while True:
            self.origin = bmin.min(axis=0)
            i0 = np.floor((bmin - self.origin) / h).astype(np.int64)
            i1 = np.floor((bmax - self.origin) / h).astype(np.int64)
            n = i1 - i0 + 1
            cnt = np.prod(n, axis=1)
            if cnt.sum() <= 40 * len(cnt) + 10**6:
                break
            h *= 1.5
        self.h = h
        self.dims = i1.max(axis=0) + 1
        sid = np.repeat(np.arange(len(cnt)), cnt)
        k = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        nx, ny = n[sid, 0], n[sid, 1]
        ix = i0[sid, 0] + k % nx
        iy = i0[sid, 1] + (k // nx) % ny
        iz = i0[sid, 2] + k // (nx * ny)
        cell = (ix * self.dims[1] + iy) * self.dims[2] + iz
        order = np.argsort(cell, kind="stable")
        self.cell = cell[order]
        self.sid = sid[order]

    def distance(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float).reshape(-1, 3)
        out = np.full(len(p), self.cap)
        ip = np.floor((p - self.origin) / self.h).astype(np.int64)
        inb = np.all((ip >= 0) & (ip < self.dims), axis=1)
        idx = np.flatnonzero(inb)
        if not len(idx):
            return out
        c = (ip[idx, 0] * self.dims[1] + ip[idx, 1]) * self.dims[2] + ip[idx, 2]
        st = np.searchsorted(self.cell, c, side="left")
        en = np.searchsorted(self.cell, c, side="right")
        cnt = en - st
        chunk = max(1, 2_000_000 // max(1, int(cnt.max(initial=1))))
        k = self.v.shape[1]
        for a in range(0, len(idx), chunk):
            sl = slice(a, a + chunk)
            cc = cnt[sl]
            if not cc.sum():
                continue
            pi = np.repeat(np.arange(len(cc)), cc)
            loc = np.arange(int(cc.sum())) - np.repeat(np.cumsum(cc) - cc, cc)
            si = self.sid[np.repeat(st[sl], cc) + loc]
            q = p[idx[sl]][pi]
            vv = self.v[si]
            if k == 1:
                d = np.linalg.norm(q - vv[:, 0], axis=1)
            elif k == 2:
                d = _pt_seg(q, vv[:, 0], vv[:, 1])
            else:
                d = _pt_tri(q, vv[:, 0], vv[:, 1], vv[:, 2])
            best = np.full(len(cc), self.cap)
            np.minimum.at(best, pi, d)
            out[idx[sl]] = np.minimum(out[idx[sl]], best)
        return out


def _simplex_measure(v: np.ndarray, dim: int) -> np.ndarray:
    if dim == 0:
        return np.ones(len(v))
    if dim == 1:
        return np.linalg.norm(v[:, 1] - v[:, 0], axis=1)
    if dim == 2:
        return 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    e = v[:, 1:] - v[:, :1]
    return np.abs(np.linalg.det(e)) / 6.0


@dataclass(frozen=True)
class _Component:
    tag: int
    measure: float
    centroid: np.ndarray
    moment: np.ndarray  # second moment about the centroid, per unit measure


@dataclass
class _Tag:
    tag: int
    dim: int
    name: str
    klass: Hashable
    measure: float
    nodes: np.ndarray  # (k, 3) nodes of the shape
    simplices: np.ndarray  # (m, s, 3) the shape (a volume: its boundary)
    components: list[_Component]
    direction: np.ndarray | None = None
    _set: _SimplexSet | None = None
    edge_points: np.ndarray | None = None  # (k, 3) on the feature boundary edges of the shape
    edges: np.ndarray | None = None  # (m, 2, 3) every boundary edge of the shape
    _edge_set: _SimplexSet | None = None


def _boundary_edges(
    shape: np.ndarray, xyz: np.ndarray, u: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """``(points, edges)`` of the boundary of a triangle set (``None`` without boundary).

    ``edges``: every edge used by one triangle; ``points``: the nodes and
    midpoints of those not on a window line (the cuts of the window are not
    features). A false operation that maps the nodes and the centroids of a
    coarse, regular mesh into the set still moves its boundary off the
    boundary of the image.
    """
    if shape.ndim != 2 or shape.shape[1] != 3 or not len(shape):
        return None, None
    e = np.sort(np.concatenate([shape[:, [0, 1]], shape[:, [1, 2]], shape[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    bnd = uniq[cnt == 1]
    if not len(bnd):
        return None, None
    ue = u[bnd]  # (m, 2, 2)
    on_line = np.zeros(len(bnd), dtype=bool)
    for i in (0, 1):
        f = ue[..., i]
        on_line |= np.all(np.abs(f - np.rint(f)) < 1e-9, axis=1) & (
            np.abs(f[:, 0] - f[:, 1]) < 1e-9
        )
    edges = xyz[bnd]
    feat = edges[~on_line]
    if not len(feat):
        return None, edges
    return np.concatenate([feat[:, 0], feat.mean(axis=1)]), edges


def _components(
    tag: int, dim: int, cells: np.ndarray, xyz: np.ndarray, u: np.ndarray, fid: np.ndarray,
    lattice: np.ndarray,
) -> list[_Component]:  # fmt: skip
    """The components that do not wrap around the torus (unwrapped across the window)."""
    if dim == 0:
        return [_Component(tag, 1.0, xyz[c[0]].copy(), np.zeros((3, 3))) for c in cells]
    f = fid[cells]
    a = np.repeat(f[:, 0], dim)
    b = f[:, 1:].ravel()
    d = (u[cells[:, 1:]] - u[cells[:, :1]]).reshape(-1, 2)
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    dd = np.concatenate([d, -d])
    order = np.argsort(src, kind="stable")
    src, dst, dd = src[order], dst[order], dd[order]
    nodes = np.unique(f)
    start = np.searchsorted(src, nodes, side="left")
    end = np.searchsorted(src, nodes, side="right")
    pos = np.full((len(nodes), 2), np.nan)
    comp = np.full(len(nodes), -1, dtype=np.int64)
    ncomp = 0
    dst_l = np.searchsorted(nodes, dst)
    first_row = np.full(len(nodes), -1, dtype=np.int64)
    first_row[np.searchsorted(nodes, fid[cells.ravel()][::-1])] = cells.ravel()[::-1]
    for seed in range(len(nodes)):
        if comp[seed] >= 0:
            continue
        comp[seed] = ncomp
        pos[seed] = u[first_row[seed]]
        queue = deque([seed])
        while queue:
            n = queue.popleft()
            for e in range(start[n], end[n]):
                m = dst_l[e]
                if comp[m] < 0:
                    comp[m] = ncomp
                    pos[m] = pos[n] + dd[e]
                    queue.append(m)
        ncomp += 1
    sl = np.searchsorted(nodes, src)
    err = np.abs(pos[dst_l] - pos[sl] - dd).max(axis=1)
    wrapped = np.zeros(ncomp, dtype=bool)
    wrapped[comp[sl[err > 0.5]]] = True
    v0 = np.searchsorted(nodes, f[:, 0])
    shift = np.rint(pos[v0] - u[cells[:, 0]])
    vx = xyz[cells].copy()
    vx[..., :2] += (shift @ lattice)[:, None, :]
    meas = _simplex_measure(vx, dim)
    cid = comp[v0]
    out = []
    for k in range(ncomp):
        if wrapped[k]:
            continue
        sel = cid == k
        m = meas[sel]
        tot = float(m.sum())
        if tot <= 0:
            continue
        v = vx[sel]
        sv = v.sum(axis=1)
        m2 = np.einsum("e,eki,ekj->ij", m, v, v) + np.einsum("e,ei,ej->ij", m, sv, sv)
        m2 /= (dim + 1) * (dim + 2)
        c = (m[:, None] * v.mean(axis=1)).sum(axis=0) / tot
        out.append(_Component(tag, tot, c, m2 / tot - np.outer(c, c)))
    return out


def _volume_boundary(
    cells: np.ndarray, xyz: np.ndarray, fid: np.ndarray, u: np.ndarray, lo: np.ndarray,
    lattice: np.ndarray, tol: float, cap: float,
) -> np.ndarray:  # fmt: skip
    """Boundary triangles (node rows) of a tetrahedral region on the torus."""
    faces = np.concatenate([cells[:, [1, 2, 3]], cells[:, [0, 2, 3]], cells[:, [0, 1, 3]],
                            cells[:, [0, 1, 2]]])  # fmt: skip
    key = np.sort(fid[faces], axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bnd = faces[cnt[inv.ravel()] == 1]
    # faces on the window planes whose periodic partner is a face of the region: interior
    uf = u[bnd]
    eps = 1e-9
    keep = np.ones(len(bnd), dtype=bool)
    for i in (0, 1):
        for side in (0.0, 1.0):
            on = np.all(np.abs(uf[:, :, i] - lo[i] - side) < eps, axis=1)
            other = np.all(np.abs(uf[:, :, i] - lo[i] - (1.0 - side)) < eps, axis=1)
            if not on.any() or not other.any():
                continue
            ss = _SimplexSet(xyz[bnd[other]], lattice, lo, cap)
            k = np.flatnonzero(on)
            p = xyz[bnd[k]].reshape(-1, 3)
            d = ss.distance(_fold(p, lattice, lo, p)).reshape(-1, 3)
            keep[k[d.max(axis=1) <= tol]] = False
    return bnd[keep]


def _prepare(
    cell: TaggedCell, classes: Mapping[int, Hashable], directions: Mapping[int, object],
    tol: float, cap: float, window_tags: set[int],
) -> tuple[dict[int, _Tag], bool]:  # fmt: skip
    """Per-tag shapes and components; second value: the mesh is periodic-conforming.

    Node identity on the torus (``fid``): exact 1e-9 keys of ``(u mod 1, z)``,
    no neighbour search -- the periodic copies of one node agree to round-off
    unless a coordinate straddles a rounding boundary; it assumes distinct
    nodes ``>> 1e-9`` of the lattice apart (a split copy then only marks the
    mesh non-conforming, which falls back to the node candidates).
    """
    t = cell.lattice
    xyz = cell.points
    u = xyz[:, :2] @ np.linalg.inv(t)
    zq = 1e-9 * math.sqrt(abs(np.linalg.det(t)))
    q = np.column_stack([
        np.mod(np.rint(u * 1e9).astype(np.int64), 10**9),
        np.rint(xyz[:, 2] / zq).astype(np.int64),
    ])  # fmt: skip
    _, fid = np.unique(q, axis=0, return_inverse=True)
    fid = fid.ravel()
    lo = cell.window_lo
    # periodic conformity: every node on a window face has a partner
    used = np.unique(np.concatenate([g.cells.ravel() for g in cell.groups.values()]))
    conforming = True
    for i in (0, 1):
        a = used[np.abs(u[used, i] - lo[i]) < 1e-9]
        b = used[np.abs(u[used, i] - lo[i] - 1.0) < 1e-9]
        if len(a) and len(b) and set(fid[a].tolist()) != set(fid[b].tolist()):
            conforming = False
    tags: dict[int, _Tag] = {}
    for tag, g in sorted(cell.groups.items()):
        if tag in window_tags:
            continue
        cells = g.cells
        comps = _components(tag, g.dim, cells, xyz, u, fid, t)
        meas = float(_simplex_measure(xyz[cells], g.dim).sum())
        shape = _volume_boundary(cells, xyz, fid, u, lo, t, tol, cap) if g.dim == 3 else cells
        if not len(shape):
            shape = cells[:, :3] if g.dim == 3 else cells
        d = directions.get(tag)
        if d is not None:
            d = np.asarray(d, dtype=float).ravel()
            d = np.concatenate([d, np.zeros(3 - len(d))]) if len(d) < 3 else d[:3]
            d = d / np.linalg.norm(d)
        # the nodes, then the centroids of the simplices: a coarse, regular mesh can map its
        # nodes into the set under a false shift; the centroids then fall outside
        pts = xyz[np.unique(shape)]
        if shape.shape[1] >= 3:
            pts = np.concatenate([pts, xyz[shape].mean(axis=1)])
        u_lo = u - lo
        epts, edges = _boundary_edges(shape, xyz, u_lo)
        tags[tag] = _Tag(tag, g.dim, g.name, classes[tag], meas, pts, xyz[shape], comps, d,
                         edge_points=epts, edges=edges)  # fmt: skip
    return tags, conforming


def _window_tags(cell: TaggedCell) -> set[int]:
    """2-D groups lying on the window faces (Bloch faces)."""
    u = cell.points[:, :2] @ np.linalg.inv(cell.lattice)
    out = set()
    for tag, g in cell.groups.items():
        if g.dim != 2:
            continue
        uu = u[g.cells]
        for i in (0, 1):
            if any(np.all(np.abs(uu[:, :, i] - cell.window_lo[i] - s) < 1e-9) for s in (0, 1)):
                out.add(tag)
    return out


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class NearSymmetry:
    """An operation broken by more than ``tol`` but by at most ``near_tol``.

    ``deviation``: the largest distance (m) between a tag and its best
    image; ``tag``: the tag where it is reached (``None`` for a lattice
    operation: ``deviation`` is then the relative Gram mismatch).
    """

    op: LatticeOp | None
    matrix: np.ndarray
    deviation: float
    tag: int | None
    reason: str


@dataclass(frozen=True, eq=False)
class DetectedSymmetry:
    """The space group of a tagged cell (:func:`detect_symmetry`)."""

    lattice: np.ndarray
    lattice_type: str
    holohedry: tuple[np.ndarray, ...]
    ops: tuple[LatticeOp, ...]
    """The group modulo the lattice, identity first."""
    permutations: tuple[dict[int, int], ...]
    """Per operation: tag -> image tag (every verified tag)."""
    direction_signs: tuple[dict[int, int], ...]
    """Per operation: tag -> ``s`` with ``G d_tag = s d_image`` (the tags with a direction)."""
    deviations: tuple[float, ...]
    """Per operation: measured largest distance (m) of a tag to its image."""
    name: str
    """Wallpaper-group symbol of the in-plane action."""
    point_group: str
    centre: np.ndarray
    """Cartesian centre (m) the names and shifts refer to."""
    tol: float
    near_tol: float
    near: tuple[NearSymmetry, ...] = ()
    window_tags: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()
    classes: dict[int, Hashable] = field(default_factory=dict)

    @property
    def order(self) -> int:
        """Number of operations modulo the lattice."""
        return len(self.ops)

    @property
    def primitive(self) -> bool:
        """No translation of the group but the lattice ones (the cell is primitive)."""
        return not any(np.array_equal(op.matrix, np.eye(2)) for op in self.ops[1:])

    @property
    def orthogonal(self) -> bool:
        """Orthogonal lattice vectors (the symmetric meshing supports rectangular / square only)."""
        t = self.lattice
        return abs(float(t[0] @ t[1])) <= 1e-9 * float(np.linalg.norm(t[0]) * np.linalg.norm(t[1]))

    def tag_preserving(self) -> tuple[LatticeOp, ...]:
        """The subgroup that fixes every tag (symmetry for any parameter values)."""
        return tuple(
            op for op, p in zip(self.ops, self.permutations, strict=True)
            if all(a == b for a, b in p.items())
        )  # fmt: skip

    def op(self, name: str) -> LatticeOp:
        """The operation called ``name``."""
        for o in self.ops:
            if o.name == name:
                return o
        raise KeyError(name)

    def summary(self) -> str:
        """Multi-line report."""
        tp = self.tag_preserving()
        c = self.centre
        lines = [
            f"{self.name} ({self.point_group}, order {self.order}) on a {self.lattice_type} "
            f"lattice, centre ({c[0]:.6g}, {c[1]:.6g}); tag-preserving subgroup "
            f"{wallpaper_name(tp, self.lattice)} (order {len(tp)}); tol {self.tol:.3g} m",
        ]
        for op, p, s, d in zip(
            self.ops, self.permutations, self.direction_signs, self.deviations, strict=True
        ):
            moved = ", ".join(f"{a}->{b}" for a, b in p.items() if a != b) or "-"
            sg = ", ".join(f"{a}:{v:+d}" for a, v in s.items() if v < 0)
            lines.append(
                f"  {op.name:<8} M={op.matrix.tolist()} tau={np.round(op.shift, 9).tolist()} "
                f"dev={d:.2e} moved {moved}" + (f"; reversed {sg}" if sg else "")
            )
        for n in self.near:
            lines.append(
                f"  NEAR {n.op.name if n.op else 'lattice'} M={n.matrix.tolist()}: "
                f"deviation {n.deviation:.3g} ({n.reason})"
            )
        lines.extend(f"  warning: {w}" for w in self.warnings)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def _translation_lattice(ops: Sequence[LatticeOp], lattice: np.ndarray) -> np.ndarray:
    """Cartesian basis of the translations of the group (lattice + pure fractional ones)."""
    vecs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    vecs += [op.shift % 1.0 for op in ops if np.array_equal(op.matrix, np.eye(2))]
    fr = [[Fraction(float(x)).limit_denominator(24) for x in v] for v in vecs]
    den = math.lcm(*(f.denominator for v in fr for f in v))
    a = [[int(f * den) for f in v] for v in fr]
    # Hermite normal form of the integer rows (2-D, by gcd steps)
    rows = [r for r in a if any(r)]
    basis: list[list[int]] = []
    for col in (0, 1):
        piv = [r for r in rows if r[col] != 0]
        rest = [r for r in rows if r[col] == 0]
        while len(piv) > 1:
            piv.sort(key=lambda r: abs(r[col]))
            p = piv[0]
            new = [p]
            for r in piv[1:]:
                k = r[col] // p[col]
                rr = [r[0] - k * p[0], r[1] - k * p[1]]
                (new if rr[col] != 0 else rest).append(rr)
            piv = new
        if piv:
            basis.append(piv[0])
        rows = [r for r in rest if any(r)]
    b = np.array(basis, dtype=float) / den
    return b @ lattice


def _reflection_kind(op: LatticeOp, lattice: np.ndarray, tlat: np.ndarray) -> str:
    """``mirror``, ``glide`` or ``both`` (centred along the axis) for a reflection coset."""
    _, t = op.cartesian(lattice)
    w_dir = _axis(op.matrix) @ lattice
    wh = w_dir / np.linalg.norm(w_dir)
    # the shortest translation of the group along the axis, and the spacing of the projections
    proj = [float(v @ wh) for v in tlat]
    perp = [float(wh[0] * v[1] - wh[1] * v[0]) for v in tlat]
    # translations along the axis: integer combinations with zero perpendicular part
    along = None
    for i in range(-4, 5):
        for j in range(-4, 5):
            if (i or j) and abs(i * perp[0] + j * perp[1]) < 1e-9 * np.linalg.norm(w_dir):
                ln = abs(i * proj[0] + j * proj[1])
                along = ln if along is None else min(along, ln)
    assert along is not None
    fr = [Fraction(p / along).limit_denominator(12) for p in proj]
    s = along / math.lcm(*(f.denominator for f in fr))
    if s < along * (1 - 1e-6):
        return "both"
    q = (float(t @ wh) / along) % 1.0
    return "mirror" if min(q, 1 - q) < 1e-6 else "glide"


def _rotation(g: np.ndarray) -> tuple[int, int]:
    """``(n, k)``: rotation by ``2 pi k / n`` (reduced)."""
    ang = math.atan2(g[1, 0], g[0, 0]) % (2 * math.pi)
    fr = Fraction(ang / (2 * math.pi)).limit_denominator(12)
    return fr.denominator, fr.numerator


POINT_GROUP = {(1, False): "C1", (2, False): "C2", (3, False): "C3", (4, False): "C4",
                (6, False): "C6", (1, True): "Cs", (2, True): "C2v", (3, True): "C3v",
                (4, True): "C4v", (6, True): "C6v"}  # fmt: skip


def point_group(ops: Sequence[LatticeOp], lattice: np.ndarray) -> tuple[int, bool]:
    """``(rotation order, has a reflection)`` of the point group of ``ops``."""
    n = 1
    refl = False
    for op in ops:
        g, _ = op.cartesian(lattice)
        if op.det == 1:
            n = max(n, _rotation(g)[0])
        else:
            refl = True
    return n, refl


def wallpaper_name(ops: Sequence[LatticeOp], lattice: object) -> str:
    """Wallpaper-group symbol (``p1`` ... ``p6m``) of a group of operations on ``lattice``.

    The translations of the group (the lattice and any pure fractional
    translation) decide between mirrors and glides.
    """
    t = _lattice(lattice)
    ops = list(ops)
    n, refl = point_group(ops, t)
    if not refl:
        return f"p{n}"
    tlat = _translation_lattice(ops, t)
    refs = [op for op in ops if op.det == -1]
    kinds: dict[tuple[int, ...], str] = {}
    axis_of: dict[tuple[int, ...], np.ndarray] = {}
    for op in refs:
        k = tuple(op.matrix.ravel().tolist())
        kind = _reflection_kind(op, t, tlat)
        if kinds.get(k) in ("mirror", "both") and kind == "glide":
            continue
        kinds[k] = kind if k not in kinds else ("both" if kinds[k] != kind else kind)
        axis_of[k] = _axis(op.matrix) @ t
    ks = list(kinds.values())
    if n == 1:
        return {"mirror": "pm", "glide": "pg", "both": "cm"}[ks[0]]
    if n == 2:
        if all(k == "both" for k in ks):
            return "cmm"
        nm = sum(k == "mirror" for k in ks)
        return {2: "pmm", 1: "pmg", 0: "pgg"}[nm]
    if n == 3:
        along = any(_along_shortest(w, tlat) for w in axis_of.values())
        return "p31m" if along else "p3m1"
    if n == 4:
        for k, w in axis_of.items():
            if _along_shortest(w, tlat):
                return "p4m" if kinds[k] == "mirror" else "p4g"
        raise AssertionError("no axial reflection")  # pragma: no cover
    return "p6m"


def _along_shortest(w: np.ndarray, tlat: np.ndarray) -> bool:
    """The direction ``w`` is that of a shortest translation of ``tlat``.

    ``tlat`` is any basis of the translations of the group: with a pure
    fractional translation (a centred cell) its shortest vectors are not
    the lattice vectors of the cell, and an axis is compared by direction.
    """
    vecs = [i * tlat[0] + j * tlat[1] for i in range(-2, 3) for j in range(-2, 3) if i or j]
    shortest = min(float(np.linalg.norm(v)) for v in vecs)
    wh = np.asarray(w, dtype=float) / float(np.linalg.norm(w))
    return any(
        abs(float(np.linalg.norm(v)) - shortest) < 1e-6 * shortest
        and abs(float(wh[0] * v[1] - wh[1] * v[0])) < 1e-6 * shortest
        for v in vecs
    )


def _name_ops(
    ops: Sequence[LatticeOp], lattice: np.ndarray, centre: np.ndarray, tol: float
) -> list[LatticeOp]:
    """Representatives shifted to move ``centre`` least, named relative to it, sorted."""
    out = []
    for op in ops:
        g, _ = op.cartesian(lattice)
        best = None
        for i in range(-3, 4):
            for j in range(-3, 4):
                tau = op.shift + np.array([i, j], dtype=float)
                tau = np.where(np.abs(tau - np.rint(tau)) < 1e-12, np.rint(tau), tau)
                disp = float(np.linalg.norm(g @ centre + tau @ lattice - centre))
                key = (round(disp / max(tol, 1e-300)), abs(tau).sum(), tuple(-tau))
                if best is None or key < best[0]:
                    best = (key, tau)
        tau = best[1]  # type: ignore[index]
        t = tau @ lattice
        if op.det == 1:
            n, k = _rotation(g)
            if n == 1:
                if np.allclose(tau, np.rint(tau)):
                    name, sk = "E", (0, 0.0)
                else:
                    fr = [str(Fraction(float(x % 1)).limit_denominator(24)) for x in tau]
                    name, sk = f"t({','.join(fr)})", (1, 0.0)
            else:
                name = f"C{n}" if k == 1 else f"C{n}^{k}"
                sk = (2, k / n)
        else:
            w = _axis(op.matrix) @ lattice
            ang = math.degrees(math.atan2(w[1], w[0])) % 180.0
            ang = round(ang, 6) % 180.0
            label = {90.0: "x", 0.0: "y", 45.0: "d", 135.0: "d'"}.get(ang, f"[{ang:g}]")
            glide = abs(float(t @ w) / float(w @ w)) > tol / float(np.linalg.norm(w))
            name = ("g" if glide else "m") + label
            pri = _ANGLE_ORDER.index(ang) if ang in _ANGLE_ORDER else 4 + ang / 180.0
            sk = (3 + int(glide), pri)
        out.append((sk, LatticeOp(name, op.matrix, tau + 0.0)))
    out.sort(key=lambda e: e[0])
    seen: dict[str, int] = {}
    named = []
    for _, op in out:
        k = seen.get(op.name, 0)
        seen[op.name] = k + 1
        named.append(op if k == 0 else LatticeOp(op.name + "'" * k, op.matrix, op.shift))
    return named


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class _Verifier:
    def __init__(self, tags: dict[int, _Tag], lattice: np.ndarray, lo: np.ndarray,
                 tol: float, cap: float) -> None:  # fmt: skip
        self.tags = tags
        self.t = lattice
        self.lo = lo
        self.tol = tol
        self.cap = cap
        self.order = sorted(tags, key=lambda k: (len(tags[k].nodes), k))
        self.compat: dict[int, list[int]] = {}
        for a, ta in tags.items():
            self.compat[a] = [
                b for b, tb in tags.items()
                if tb.dim == ta.dim and tb.klass == ta.klass
                and abs(tb.measure - ta.measure) <= 5e-2 * max(ta.measure, tb.measure)
                and (tb.direction is None) == (ta.direction is None)
            ]  # fmt: skip

    def _set(self, b: int) -> _SimplexSet:
        tb = self.tags[b]
        if tb._set is None:
            tb._set = _SimplexSet(tb.simplices, self.t, self.lo, self.cap)
        return tb._set

    def _image(self, x: np.ndarray, g: np.ndarray, tr: np.ndarray) -> np.ndarray:
        y = np.array(x, dtype=float, copy=True)
        y[:, :2] = x[:, :2] @ g.T + tr
        return _fold(y, self.t, self.lo, y)

    def dist(self, a: int, b: int, g: np.ndarray, tr: np.ndarray, sample: bool = False) -> float:
        xa = self.tags[a].nodes
        if sample and len(xa) > 24:
            xa = xa[np.linspace(0, len(xa) - 1, 24).astype(int)]
        d = float(self._set(b).distance(self._image(xa, g, tr)).max())
        if sample or d > self.cap * 0.999:
            return d
        gi = g.T
        xb = self.tags[b].nodes
        d = max(d, float(self._set(a).distance(self._image(xb, gi, -(tr @ gi.T))).max()))
        if d > self.tol:
            return d
        return max(d, self._edge_dist(a, b, g, tr), self._edge_dist(b, a, gi, -(tr @ gi.T)))

    def _edge_dist(self, a: int, b: int, g: np.ndarray, tr: np.ndarray) -> float:
        """Distance of the image of the feature boundary of ``a`` to the boundary of ``b``."""
        ta, tb = self.tags[a], self.tags[b]
        if ta.edge_points is None:
            return 0.0
        if tb.edges is None:
            return self.cap
        if tb._edge_set is None:
            tb._edge_set = _SimplexSet(tb.edges, self.t, self.lo, self.cap)
        return float(tb._edge_set.distance(self._image(ta.edge_points, g, tr)).max())

    def evaluate(self, g: np.ndarray, tr: np.ndarray) -> tuple[float, int | None, dict, dict, str]:
        """Deviation, worst tag, permutation, direction signs, failure reason."""
        perm: dict[int, int] = {}
        signs: dict[int, int] = {}
        dev, worst = 0.0, None
        gg = np.eye(3)
        gg[:2, :2] = g
        for a in self.order:
            quick = sorted((self.dist(a, b, g, tr, sample=True), b) for b in self.compat[a])
            best = (math.inf, None)
            for dq, b in quick[:3]:
                if dq >= self.cap * 0.999:
                    break
                d = self.dist(a, b, g, tr)
                if d < best[0]:
                    best = (d, b)
                if d <= self.tol:
                    break
            if best[1] is None or best[0] >= self.cap * 0.999:
                return math.inf, a, perm, signs, "geometry"
            perm[a] = best[1]
            if best[0] > dev:
                dev, worst = best[0], a
        if len(set(perm.values())) != len(perm):
            return math.inf, None, perm, signs, "not a permutation"
        for a, b in perm.items():
            da, db = self.tags[a].direction, self.tags[b].direction
            if da is None:
                continue
            c = float((gg @ da) @ db)
            if abs(abs(c) - 1.0) > 1e-6:
                return max(dev, self.tol * 2), a, perm, signs, "direction"
            signs[a] = 1 if c > 0 else -1
        return dev, worst, perm, signs, "geometry"


def _candidates(
    tags: dict[int, _Tag], ver: _Verifier, g: np.ndarray, lattice: np.ndarray, use_nodes: bool,
) -> list[np.ndarray]:  # fmt: skip
    """Fractional translations ``tau`` for the in-plane matrix ``g`` (Cartesian)."""
    tinv_t = np.linalg.inv(lattice.T)
    gg = np.eye(3)
    gg[:2, :2] = g
    taus = []
    comps = [c for t in tags.values() for c in t.components]

    if comps:
        best = None
        for c0 in comps:
            n = sum(_congruent(c0, c1, ver) for c1 in comps)
            key = (n, c0.tag, -c0.measure)
            if best is None or key < best[0]:
                best = (key, c0)
        c0 = best[1]  # type: ignore[index]
        for c1 in comps:
            if _maps_onto(c0, c1, ver, gg):
                taus.append(tinv_t @ (c1.centroid[:2] - g @ c0.centroid[:2]))
    if use_nodes or not comps:
        ref = min(tags.values(), key=lambda t: (len(t.nodes), t.tag))
        x0 = ref.nodes[0, :2]
        for b in ver.compat[ref.tag]:
            for y in tags[b].nodes:
                if abs(y[2] - ref.nodes[0, 2]) <= ver.tol:
                    taus.append(tinv_t @ (y[:2] - g @ x0))
    # unique modulo the lattice
    out: dict[tuple[int, int], np.ndarray] = {}
    res = max(ver.tol / math.sqrt(abs(np.linalg.det(lattice))), 1e-12)
    for tau in taus:
        k = tuple(np.mod(np.rint(tau / res).astype(np.int64), round(1 / res)).tolist())
        out.setdefault(k, np.mod(tau, 1.0))  # type: ignore[arg-type]
    return list(out.values())


def _maps_onto(c0: _Component, c1: _Component, ver: _Verifier, gg: np.ndarray) -> bool:
    """``c1`` may be the image of ``c0`` under ``gg`` (class, measure, rotated moment)."""
    if c1.tag not in ver.compat[c0.tag]:
        return False
    if abs(c1.measure - c0.measure) > 1e-2 * max(c0.measure, c1.measure):
        return False
    m0 = gg @ c0.moment @ gg.T
    sc = max(float(np.abs(c0.moment).max()), float(np.abs(c1.moment).max()), 1e-300)
    return float(np.abs(m0 - c1.moment).max()) <= 1e-2 * sc + 1e-12


def _congruent(c0: _Component, c1: _Component, ver: _Verifier) -> bool:
    """``c1`` may be the image of ``c0`` under some operation (class, measure, moment spectrum)."""
    if c1.tag not in ver.compat[c0.tag]:
        return False
    if abs(c1.measure - c0.measure) > 1e-2 * max(c0.measure, c1.measure):
        return False
    e0 = np.linalg.eigvalsh(c0.moment)
    e1 = np.linalg.eigvalsh(c1.moment)
    return float(np.abs(e0 - e1).max()) <= 1e-2 * max(float(np.abs(e0).max()), 1e-300) + 1e-12


_Found = tuple[LatticeOp, dict[int, int], dict[int, int], float]
"""A verified operation: op, tag permutation, direction signs, deviation."""


def _one_per_coset(found: list[_Found], lattice: np.ndarray, tol: float) -> list[_Found]:
    """One operation per coset (the candidates of one op may differ by up to ``tol``)."""
    group: list[_Found] = []
    for f in sorted(found, key=lambda f: f[3]):
        if not any(_same_op(f[0], h[0], lattice, 2 * tol) for h in group):
            group.append(f)
    ident = LatticeOp("E", np.eye(2), np.zeros(2))
    if not any(_same_op(f[0], ident, lattice, 2 * tol) for f in group):
        raise ValueError("the identity does not verify: tags overlap, or tol is too small")
    return group


_GENERIC_DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1), (0.7071067811865476, 0.5773502691896258))
"""Directions (lattice coordinates) of the generic translations tried."""


def _continuous(ver: _Verifier, lattice: np.ndarray, tol: float) -> list[tuple[float, float]]:
    """The directions of :data:`_GENERIC_DIRECTIONS` along which a generic translation verifies."""
    out = []
    for v in _GENERIC_DIRECTIONS:
        tr = 0.1234567890123 * np.asarray(v, dtype=float) @ lattice
        if ver.evaluate(np.eye(2), tr)[0] <= tol:
            out.append(v)
    return out


def _point_ops_about(
    ver: _Verifier, holo: Sequence[np.ndarray], lattice: np.ndarray, centre: np.ndarray,
    tol: float,
) -> list[_Found]:  # fmt: skip
    """The lattice point operations about ``centre`` (Cartesian, m) that verify."""
    found: list[_Found] = []
    for k, m in enumerate(holo):
        g = lattice.T @ m @ np.linalg.inv(lattice.T)
        tr = centre - g @ centre
        dev, _, perm, signs, _ = ver.evaluate(g, tr)
        if dev <= tol:
            found.append((LatticeOp(f"op{k}", m, tr @ np.linalg.inv(lattice)), perm, signs, dev))
    return found


def _closure_warning(group: list[_Found], lattice: np.ndarray, tol: float) -> list[str]:
    for a in group:
        for b in group:
            c = a[0].compose(b[0])
            if not any(_same_op(c, h[0], lattice, 4 * tol) for h in group):
                return [
                    f"the {len(group)} verified operations do not close ({a[0].name} o "
                    f"{b[0].name} missing): the candidate translations missed some"
                ]
    return []


def _choose_centre(
    ops: Sequence[LatticeOp], tags: dict[int, _Tag], cell: TaggedCell, tol: float
) -> np.ndarray:
    """A site of the highest-order rotation, preferring the centroid of the lowest-tag component.

    Without rotation: that centroid (the window centre without components).
    """
    t = cell.lattice
    size = math.sqrt(abs(float(np.linalg.det(t))))
    n, _ = point_group(ops, t)
    comps = sorted((c for tg in tags.values() for c in tg.components), key=lambda c: c.tag)
    ref = comps[0].centroid[:2] if comps else (cell.window_lo + 0.5) @ t
    if n == 1:
        return np.asarray(ref, dtype=float)
    sites = []
    for op in ops:
        g, tr = op.cartesian(t)
        if op.det == 1 and _rotation(g)[0] == n:
            for i in range(-2, 3):
                for j in range(-2, 3):
                    sites.append(np.linalg.solve(np.eye(2) - g, tr + np.array([i, j]) @ t))

    def score(s: np.ndarray) -> tuple[float, float]:
        hit = [c.tag for c in comps if _torus_dist(c.centroid[:2], s, t) <= max(tol, 1e-9 * size)]
        return (min(hit) if hit else math.inf, _torus_dist(s, ref, t))

    best = min(sites, key=score)
    return best - np.rint((best - ref) @ np.linalg.inv(t)) @ t


def _near_lattice(lattice: np.ndarray, holo: Sequence[np.ndarray]) -> list[NearSymmetry]:
    """Operations of a nearly symmetric lattice (relative Gram mismatch <= 1e-3)."""
    gram = lattice @ lattice.T
    out = []
    for m in lattice_holohedry(lattice, rtol=1e-3):
        if not any(np.array_equal(m, h) for h in holo):
            dev = float(np.abs(m.T @ gram @ m - gram).max() / np.abs(gram).max())
            out.append(NearSymmetry(None, m, dev, None, "lattice"))
    return out


def _near_cosets(
    cand: list[tuple[LatticeOp, float, int | None, str]], named: Sequence[LatticeOp],
    lattice: np.ndarray, centre: np.ndarray, tol: float,
) -> list[NearSymmetry]:  # fmt: skip
    """The near candidates, one per coset of the exact group."""
    kept: list[tuple[LatticeOp, float]] = []
    out = []
    for op, dev, worst, why in sorted(cand, key=lambda e: e[1]):
        if any(_same_op(op, h, lattice, 2 * tol) for h in named):
            continue
        if any(
            _same_op(h.compose(op), k, lattice, 2 * (dev + kd + tol))
            for h in named
            for k, kd in kept
        ):
            continue
        kept.append((op, dev))
        out.append(
            NearSymmetry(_name_ops([op], lattice, centre, tol)[0], op.matrix, dev, worst, why)
        )
    return out


def detect_symmetry(
    cell: TaggedCell,
    *,
    tol: float | None = None,
    near_tol: float | None = None,
    classes: Mapping[int, Hashable] | None = None,
    directions: Mapping[int, Sequence[float]] | None = None,
    window_tags: Sequence[int] | None = None,
) -> DetectedSymmetry:
    """The space group of a tagged periodic cell (see the module).

    Args:
        cell: The tagged geometry and the lattice.
        tol: Largest distance (m) of a tag to its image for an exact
            operation (default ``1e-6`` of the cell size).
        near_tol: Largest distance (m) reported as a near symmetry
            (default ``1e-2`` of the cell size).
        classes: tag -> class; tags are exchanged only within a class
            (default from the names, see the module). Unlisted tags keep
            the default.
        directions: tag -> direction of the tag: it must map
            onto ``+-`` the direction of the image tag.
        window_tags: Tags excluded as the Bloch faces (default: the 2-D
            groups lying on the window faces).

    Returns:
        The :class:`DetectedSymmetry`.
    """
    t = cell.lattice
    size = math.sqrt(abs(float(np.linalg.det(t))))
    tol = 1e-6 * size if tol is None else float(tol)
    near_tol = max(1e-2 * size, 10 * tol) if near_tol is None else float(near_tol)
    cls_map = {
        tag: default_class(tag, g.name) for tag, g in cell.groups.items()
    } | dict(classes or {})  # fmt: skip
    win = _window_tags(cell) if window_tags is None else set(window_tags)
    tags, conforming = _prepare(cell, cls_map, dict(directions or {}), tol, near_tol, win)
    if not tags:
        raise ValueError("no tag left to verify (all on the window faces)")
    warnings = []
    if not conforming:
        warnings.append(
            "the mesh is not periodic-conforming across the window: the components cut by "
            "the window are split and the candidate translations also come from the nodes"
        )
    ver = _Verifier(tags, t, cell.window_lo, tol, near_tol)
    holo = lattice_holohedry(t)
    found: list[_Found] = []
    near_cand: list[tuple[LatticeOp, float, int | None, str]] = []
    use_nodes = not conforming or not any(tg.components for tg in tags.values())
    continuous = _continuous(ver, t, tol)
    for k, m in enumerate(holo if not continuous else ()):
        g = t.T @ m @ np.linalg.inv(t.T)
        for tau in _candidates(tags, ver, g, t, use_nodes):
            dev, worst, perm, signs, why = ver.evaluate(g, tau @ t)
            op = LatticeOp(f"op{k}", m, tau)
            if dev <= tol:
                found.append((op, perm, signs, dev))
            elif dev <= near_tol:
                near_cand.append((op, dev, worst, why))
    if continuous:
        # a laterally uniform (or striped) cell: every shift verifies; keep the point operations
        # about the window centre, a group without fractional translations
        found = _point_ops_about(ver, holo, t, (cell.window_lo + 0.5) @ t, tol)
        near_cand = []
        along = (
            "in every direction" if len(continuous) == len(_GENERIC_DIRECTIONS)
            else "along " + ", ".join(f"({v[0]:.3g}, {v[1]:.3g})" for v in continuous)
            + " (lattice coordinates)"
        )  # fmt: skip
        warnings.append(
            f"the geometry is invariant under continuous translations {along}: only the point "
            "operations about the window centre are used"
        )
    group = _one_per_coset(found, t, tol)
    warnings.extend(_closure_warning(group, t, tol))
    centre = _choose_centre([f[0] for f in group], tags, cell, tol)
    named = _name_ops([f[0] for f in group], t, centre, tol)
    perms, signs, devs = [], [], []
    for op in named:
        _, p, s, d = next(f for f in group if _same_op(op, f[0], t, 2 * tol))
        perms.append(dict(sorted(p.items())))
        signs.append(dict(sorted(s.items())))
        devs.append(d)
    near = _near_lattice(t, holo) + _near_cosets(near_cand, named, t, centre, tol)
    ltype = _lattice_type(t, len(holo))
    res = DetectedSymmetry(
        lattice=t,
        lattice_type=ltype,
        holohedry=holo,
        ops=tuple(named),
        permutations=tuple(perms),
        direction_signs=tuple(signs),
        deviations=tuple(devs),
        name=wallpaper_name(named, t),
        point_group=POINT_GROUP[point_group(named, t)],
        centre=centre,
        tol=tol,
        near_tol=near_tol,
        near=tuple(near),
        window_tags=tuple(sorted(win)),
        classes=cls_map,
    )
    extra = []
    from mpxlite.cells.asymmetric import LATTICE_FREE

    if not res.orthogonal and res.name not in LATTICE_FREE:
        extra.append(
            f"{ltype} lattice with non-orthogonal vectors: the symmetric meshing "
            "(asymmetric_unit, build_symmetric_mesh) supports p1 and p2 only on such a lattice"
        )
    if not res.primitive:
        extra.append(
            "the cell is not primitive (the group holds fractional translations); the "
            "symbol refers to the full translation lattice"
        )
    object.__setattr__(res, "warnings", tuple(warnings + extra))
    return res


def _same_op(a: LatticeOp, b: LatticeOp, lattice: np.ndarray, tol: float) -> bool:
    """Same matrix and shifts within ``tol`` (m) modulo the lattice."""
    if not np.array_equal(a.matrix, b.matrix):
        return False
    d = a.shift - b.shift
    return float(np.linalg.norm((d - np.rint(d)) @ lattice)) <= tol


def _torus_dist(a: np.ndarray, b: np.ndarray, lattice: np.ndarray) -> float:
    u = (np.asarray(a) - np.asarray(b)) @ np.linalg.inv(lattice)
    best = math.inf
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            best = min(best, float(np.linalg.norm((u - np.rint(u) + [i, j]) @ lattice)))
    return best
