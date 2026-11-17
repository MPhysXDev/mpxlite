# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

r"""Asymmetric unit (fundamental domain) of a plane group on a rectangular cell.

:func:`asymmetric_unit` returns a convex polygon :math:`F` whose images
under the operations of the group, translated by lattice vectors, tile the
plane without overlap, for the 12 plane groups of the rectangular and square
lattices (``p1 p2 pm pg cm pmm pmg pgg cmm p4 p4m p4g``; ``cm`` / ``cmm`` on
their centred conventional cell). The polygons are those of the
International Tables (ITA, standard setting, fractional coordinates of the
conventional cell; :data:`ITA`), carried into the user's frame: a signed
permutation ``P`` of the fractional axes and an origin ``o``,

.. math::
   u = P v + o, \qquad x = u_1 T_1 + u_2 T_2 ,

are searched so that the ITA operations conjugated by the frame are the
given ones modulo the lattice; the origin nearest to the detected centre
(:attr:`mpxlite.cells.detect.DetectedSymmetry.centre`) is kept and the
polygon is translated by a lattice vector so that its centroid lies in the
window. Where ITA's unit has an edge mapped onto itself by a rotation or a
glide (``p2``, ``pg``, ``pmg``, ``cmm``) the edge is split at the image
point (:data:`ITA` lists the split vertices), so that every edge piece is

* a **mirror** edge: fixed point-wise by a reflection of the group (the
  images on both sides share it point-wise: no constraint), or
* **paired** with another edge piece by an operation ``A = (g, t)`` of the
  group composed with a lattice translation, ``A(partner) = edge``
  (:class:`UnitEdge`): the mesh of the edge must be the image of the mesh of
  its partner (a translation for the lattice pairs, a rotation, a glide).

The pairs are found generically from the tiling (the image of :math:`F`
across each edge), not from the table. :attr:`AsymmetricUnit.cuts` are the
segments of :math:`F` on the preimages :math:`g^{-1}(\partial W + \Lambda)`
of the window lines for the operations ``g`` other than the identity
(:math:`W` the window ``[lo, lo + 1)`` in lattice coordinates): a mesh of
:math:`F` conforming along them, and along the window lines themselves,
replicates into the window element by element (no element straddles a
window face). :attr:`AsymmetricUnit.translations` are the lattice
translations ``R`` of the window that meet :math:`F`: the user's cell
geometry, translated by ``R``, covers :math:`F`.

Orthogonal lattice vectors (rectangular and square cells) are supported
for every group of :data:`ITA`, any lattice for ``p1`` and ``p2``
(:data:`LATTICE_FREE`); a supercell (a group with pure fractional translations other
than the ``cm`` / ``cmm`` centring) is refused.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mpxlite.cells.spacegroup import LatticeOp, _lattice

__all__ = [
    "ITA",
    "AsymmetricUnit",
    "AsymmetricUnitError",
    "UnitEdge",
    "asymmetric_unit",
]

_F = 1.0 / 2.0
_Q = 1.0 / 4.0

# (matrix rows, shift) of the ITA standard operations, fractional coordinates
_E = ((1, 0), (0, 1))
_C2 = ((-1, 0), (0, -1))
_C4 = ((0, -1), (1, 0))
_C4I = ((0, 1), (-1, 0))
_MX = ((-1, 0), (0, 1))  # x -> -x
_MY = ((1, 0), (0, -1))
_MD = ((0, 1), (1, 0))
_MDI = ((0, -1), (-1, 0))
_O = (0.0, 0.0)

ITA: dict[str, tuple[tuple[tuple[tuple[int, int], tuple[int, int]], tuple[float, float]], ...]] = {
    "p1": ((_E, _O),),
    "p2": ((_E, _O), (_C2, _O)),
    "pm": ((_E, _O), (_MX, _O)),
    "pg": ((_E, _O), (_MX, (0.0, _F))),
    "cm": ((_E, _O), (_MX, _O), (_E, (_F, _F)), (_MX, (_F, _F))),
    "pmm": ((_E, _O), (_C2, _O), (_MX, _O), (_MY, _O)),
    "pmg": ((_E, _O), (_C2, _O), (_MX, (_F, 0.0)), (_MY, (_F, 0.0))),
    "pgg": ((_E, _O), (_C2, _O), (_MX, (_F, _F)), (_MY, (_F, _F))),
    "cmm": ((_E, _O), (_C2, _O), (_MX, _O), (_MY, _O),
            (_E, (_F, _F)), (_C2, (_F, _F)), (_MX, (_F, _F)), (_MY, (_F, _F))),
    "p4": ((_E, _O), (_C2, _O), (_C4, _O), (_C4I, _O)),
    "p4m": ((_E, _O), (_C2, _O), (_C4, _O), (_C4I, _O),
            (_MX, _O), (_MY, _O), (_MD, _O), (_MDI, _O)),
    "p4g": ((_E, _O), (_C2, _O), (_C4, _O), (_C4I, _O),
            (_MX, (_F, _F)), (_MY, (_F, _F)), (_MD, (_F, _F)), (_MDI, (_F, _F))),
}  # fmt: skip
"""Operations of the ITA standard setting (``(M rows, tau)``, fractional, identity first)."""

LATTICE_FREE = ("p1", "p2")
"""Groups whose fractional tables hold on any lattice (oblique included)."""

UNITS: dict[str, tuple[tuple[float, float], ...]] = {
    "p1": ((0, 0), (1, 0), (1, 1), (0, 1)),
    "p2": ((0, 0), (_F, 0), (_F, _F), (_F, 1), (0, 1), (0, _F)),
    "pm": ((0, 0), (_F, 0), (_F, 1), (0, 1)),
    "pg": ((0, 0), (_F, 0), (_F, _F), (_F, 1), (0, 1), (0, _F)),
    "cm": ((0, 0), (_F, 0), (_F, _F), (0, _F)),
    "pmm": ((0, 0), (_F, 0), (_F, _F), (0, _F)),
    "pmg": ((0, 0), (_Q, 0), (_Q, 1), (0, 1), (0, _F)),
    "pgg": ((0, 0), (_Q, 0), (_Q, _F), (_Q, 1), (0, 1), (0, _F)),
    "cmm": ((0, 0), (_Q, 0), (_Q, _Q), (_Q, _F), (0, _F)),
    "p4": ((0, 0), (_F, 0), (_F, _F), (0, _F)),
    "p4m": ((0, 0), (_F, 0), (_F, _F)),
    "p4g": ((0, 0), (_F, 0), (0, _F)),
}  # fmt: skip
"""ITA asymmetric units (CCW vertices, fractional), split where an edge maps onto itself.

``p2`` / ``pg``: ``0 <= x <= 1/2, 0 <= y <= 1`` (edges ``x = 0, 1/2`` split at
``y = 1/2``: 2-fold point / glide image); ``pm``: same, unsplit; ``cm``,
``pmm``, ``p4``: ``0 <= x, y <= 1/2``; ``pmg``: ``0 <= x <= 1/4,
0 <= y <= 1`` (``x = 0`` split at the 2-fold point ``y = 1/2``); ``pgg``:
``0 <= x <= 1/4, 0 <= y <= 1`` (both edges split at ``y = 1/2``; the
square ``0 <= x, y <= 1/2`` has its two glide pairs on the same corner
curves with crossed orientations, which gmsh's periodic masters cannot
express); ``cmm``:
``0 <= x <= 1/4, 0 <= y <= 1/2`` (``x = 1/4`` split at the 2-fold point
``y = 1/4``); ``p4m``: ``0 <= y <= x <= 1/2``; ``p4g``: ``x, y >= 0,
x + y <= 1/2``.
"""

_TOL = 1e-9
"""Relative tolerance (of the lattice scale / fractional units) of the geometric tests."""


class AsymmetricUnitError(ValueError):
    """No asymmetric unit for this group / lattice (unsupported, or inconsistent operations)."""


@dataclass(frozen=True, eq=False)
class UnitEdge:
    """One edge piece ``a -> b`` (Cartesian, CCW) of the unit.

    ``kind``: ``"mirror"`` (fixed point-wise by a reflection) or
    ``"paired"``: ``partner`` is the index of the edge piece with
    ``g @ x_partner + t = x_edge`` (``g`` ``(2, 2)``, ``t`` ``(2,)``, m),
    ``op`` the group operation (its lattice translation included in ``t``).
    """

    index: int
    a: np.ndarray
    b: np.ndarray
    kind: str
    partner: int | None = None
    g: np.ndarray | None = None
    t: np.ndarray | None = None
    op: str = ""


@dataclass(frozen=True, eq=False)
class AsymmetricUnit:
    """A fundamental domain of a plane group (see the module)."""

    name: str
    """Wallpaper symbol."""
    lattice: np.ndarray
    ops: tuple[LatticeOp, ...]
    """The group modulo the lattice (the given operations)."""
    frame: np.ndarray
    """``P``: ITA fractional axes -> lattice coordinates (signed permutation)."""
    origin: np.ndarray
    """``o`` (lattice coordinates) of the ITA origin, translation of the unit included."""
    vertices: np.ndarray
    """``(k, 2)`` CCW Cartesian vertices (m), split vertices included."""
    edges: tuple[UnitEdge, ...]
    cuts: tuple[np.ndarray, ...]
    """``(2, 2)`` segments (m) of the unit on the preimages of the window lines."""
    translations: tuple[tuple[int, int], ...]
    """Lattice translations ``R`` with ``(W + R) cap F`` of positive area."""
    window_lo: np.ndarray

    @property
    def area(self) -> float:
        """Area of the unit (m^2): ``|det T| / order``."""
        v = self.vertices
        return 0.5 * float(np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1]))

    def pairs(self) -> list[UnitEdge]:
        """The paired edges, once each (the slave of the pair: ``index > partner``)."""
        return [e for e in self.edges if e.kind == "paired" and e.index > e.partner]

    def contains(self, xy: np.ndarray, tol: float = 0.0) -> np.ndarray:
        """Points ``(n, 2)`` in the closed unit (within ``tol`` m)."""
        p = np.atleast_2d(np.asarray(xy, dtype=float))
        v = self.vertices
        inside = np.ones(len(p), dtype=bool)
        for k in range(len(v)):
            a, b = v[k], v[(k + 1) % len(v)]
            d = b - a
            n = np.array([d[1], -d[0]]) / np.linalg.norm(d)  # outward (CCW)
            inside &= (p - a) @ n <= tol
        return inside

    def summary(self) -> str:
        """Multi-line report."""
        pts = " ".join(f"({x:.6g}, {y:.6g})" for x, y in self.vertices)
        lines = [
            f"{self.name}: unit of {len(self.vertices)} vertices, area {self.area:.6g} m^2 "
            f"(1/{len(self.ops)} of the cell); frame P={self.frame.tolist()} "
            f"o={np.round(self.origin, 9).tolist()}",
            f"  vertices {pts}",
        ]
        for e in self.edges:
            tail = "" if e.kind == "mirror" else f" <- edge {e.partner} by {e.op}"
            lines.append(f"  edge {e.index}: {e.kind}{tail}")
        lines.append(f"  {len(self.cuts)} cut segment(s); window translations {self.translations}")
        return "\n".join(lines)


def _signed_perms() -> list[np.ndarray]:
    out = []
    for perm in ((0, 1), (1, 0)):
        for sx, sy in itertools.product((1, -1), repeat=2):
            p = np.zeros((2, 2), dtype=np.int64)
            p[0, perm[0]] = sx
            p[1, perm[1]] = sy
            out.append(p)
    return out


def _mod1(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) - np.rint(x)


def _same_group(a: Sequence[tuple[np.ndarray, np.ndarray]], b: Sequence[LatticeOp]) -> bool:
    """Same set of ``(M, tau)`` modulo the lattice."""
    if len(a) != len(b):
        return False
    for m, tau in a:
        if not any(
            np.array_equal(m, op.matrix) and np.max(np.abs(_mod1(tau - op.shift))) <= _TOL
            for op in b
        ):
            return False
    return True


_ItaOps = tuple[tuple[tuple[tuple[int, int], tuple[int, int]], tuple[float, float]], ...]


def _conjugate(
    p: np.ndarray, o: np.ndarray, m: np.ndarray, tau: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """ITA op ``v -> m v + tau`` in the frame ``u = P v + o``."""
    pi = np.round(np.linalg.inv(p)).astype(np.int64)
    mu = p @ m @ pi
    return mu, p @ tau + o - mu @ o


def _origins(
    p: np.ndarray, ita: _ItaOps, ops: Sequence[LatticeOp], c: np.ndarray, lo: np.ndarray
) -> list[np.ndarray]:
    """Candidate origins ``o`` (lattice coordinates) for the frame ``P``.

    A free coordinate (along a mirror / glide axis, or both for ``p1``) is
    taken from the window corner ``lo`` (the unit then meets fewer windows).
    """
    pi = np.round(np.linalg.inv(p)).astype(np.int64)
    points: list[np.ndarray] | None = None
    lines: list[tuple[np.ndarray, np.ndarray]] = []  # (point, direction) in lattice coords
    shifts = [np.array(k, dtype=float) for k in itertools.product(range(-2, 3), repeat=2)]
    for rows, tau_i in ita[1:]:
        mi = np.array(rows, dtype=np.int64)
        mu = p @ mi @ pi
        match = [op for op in ops if np.array_equal(op.matrix, mu)]
        if not match:
            return []
        a = np.eye(2) - mu
        rhs = [op.shift - p @ np.asarray(tau_i) for op in match]
        rank = np.linalg.matrix_rank(a)
        if rank == 2:
            sols = [np.linalg.solve(a, r + k) for r in rhs for k in shifts]
            if points is None:
                points = sols
            else:
                points = [q for q in points if any(np.max(np.abs(_mod1(q - s))) <= _TOL
                                                   for s in sols)]  # fmt: skip
        elif rank == 1:
            null = np.linalg.svd(a)[2][-1]
            for r in rhs:
                for k in shifts:
                    o, *_ = np.linalg.lstsq(a, r + k, rcond=None)
                    if np.max(np.abs(a @ o - r - k)) <= _TOL:
                        lines.append((o, null))
    if points is None:
        if not lines:
            return [np.asarray(lo, dtype=float)]
        points = [o + float((lo - o) @ d) * d for o, d in lines]
    return points


def _clip_area(poly: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Area of a convex polygon (lattice coordinates) inside the box ``[lo, hi]``."""
    out = [tuple(v) for v in poly]
    for ax in (0, 1):
        for sgn, bound in ((1.0, lo[ax]), (-1.0, hi[ax])):
            src, out = out, []
            for k, cur in enumerate(src):
                prv = src[k - 1]
                sc, sp = sgn * (cur[ax] - bound), sgn * (prv[ax] - bound)
                if sc >= 0:
                    if sp < 0:
                        s = sp / (sp - sc)
                        out.append(tuple(np.add(prv, s * np.subtract(cur, prv))))
                    out.append(cur)
                elif sp >= 0:
                    s = sp / (sp - sc)
                    out.append(tuple(np.add(prv, s * np.subtract(cur, prv))))
            if not out:
                return 0.0
    v = np.array(out)
    return 0.5 * abs(float(np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1])))


def _translations(uv: np.ndarray, lo: np.ndarray) -> list[tuple[int, int]]:
    """Lattice translations ``R`` with ``(W + R) cap F`` of positive area (``uv``: lattice)."""
    umin, umax = uv.min(axis=0), uv.max(axis=0)
    out = []
    for r in itertools.product(
        range(math.floor(umin[0] - lo[0]) - 1, math.ceil(umax[0] - lo[0]) + 1),
        range(math.floor(umin[1] - lo[1]) - 1, math.ceil(umax[1] - lo[1]) + 1),
    ):
        ra = np.array(r, dtype=float)
        if _clip_area(uv, lo + ra, lo + ra + 1) > 1e-9:
            out.append((int(r[0]), int(r[1])))
    return out


def _in_polygon(q: np.ndarray, verts: np.ndarray, tol: float) -> bool:
    """``q`` in the closed CCW convex polygon (within ``tol``)."""
    n = len(verts)
    for k in range(n):
        a, b = verts[k], verts[(k + 1) % n]
        d = b - a
        if (q - a) @ np.array([d[1], -d[0]]) / np.linalg.norm(d) > tol:
            return False
    return True


def _neighbour(
    q: np.ndarray, verts: np.ndarray, ops: Sequence[LatticeOp], t: np.ndarray, scale: float
) -> tuple[LatticeOp, np.ndarray, np.ndarray] | None:
    """``(op, g, t')`` with ``q`` in the image ``g F + t'`` (lattice translation in ``t'``)."""
    tinv = np.linalg.inv(t)
    for op in ops:
        g, tr = op.cartesian(t)
        u0 = ((q - tr) @ g) @ tinv  # q = g p + tr + R t, p in F (g^-1 = g^T)
        for r in itertools.product(range(-3, 4), repeat=2):
            ra = np.array(r, dtype=float)
            if _in_polygon((u0 - ra) @ t, verts, _TOL * scale):
                return op, g, tr + g @ (ra @ t)
    return None


def _edges(
    verts: np.ndarray, ops: Sequence[LatticeOp], t: np.ndarray, scale: float
) -> list[UnitEdge]:
    """Classify every edge piece by the image of the unit across it."""
    n = len(verts)
    out = []

    def vertex_index(x: np.ndarray) -> int | None:
        d = np.linalg.norm(verts - x, axis=1)
        k = int(np.argmin(d))
        return k if d[k] <= 1e-7 * scale else None

    for k in range(n):
        a, b = verts[k], verts[(k + 1) % n]
        d = b - a
        q = 0.5 * (a + b) + 1e-4 * scale * np.array([d[1], -d[0]]) / np.linalg.norm(d)
        hit = _neighbour(q, verts, ops, t, scale)  # the image just outside the edge
        if hit is None:
            raise AsymmetricUnitError(f"edge {k}: no image of the unit across it (not a tiling)")
        op, g, tt = hit
        ia, ib = vertex_index((a - tt) @ g), vertex_index((b - tt) @ g)
        if ia is None or ib is None or (ib - ia) % n not in (1, n - 1):
            raise AsymmetricUnitError(
                f"edge {k}: its preimage under {op.name} is not an edge of the unit "
                "(an edge piece must be split)"
            )
        partner = ia if (ib - ia) % n == 1 else ib
        if partner != k:
            out.append(UnitEdge(k, a, b, "paired", partner, g, tt, op.name))
        elif max(np.linalg.norm(g @ x + tt - x) for x in (a, b)) <= 1e-7 * scale:
            out.append(UnitEdge(k, a, b, "mirror", op=op.name))
        else:
            raise AsymmetricUnitError(
                f"edge {k} is mapped onto itself by {op.name} but not point-wise "
                "(it must be split at the image point)"
            )
    return out


def _cuts(
    uverts: np.ndarray, ops: Sequence[LatticeOp], lo: np.ndarray, t: np.ndarray
) -> list[np.ndarray]:
    """Segments of the unit (lattice coordinates ``uverts``) on ``g^-1`` of the window lines."""
    n = len(uverts)
    seen: list[tuple[np.ndarray, float]] = []
    ident = []
    out = []
    for op in ops:
        for i in (0, 1):
            e = np.zeros(2)
            e[i] = 1.0
            a = op.matrix.T @ e  # (M u + tau)_i = a . u + tau_i
            proj = uverts @ a
            k0 = math.floor(proj.min() - lo[i] + op.shift[i]) - 1
            k1 = math.ceil(proj.max() - lo[i] + op.shift[i]) + 1
            for k in range(k0, k1 + 1):
                b = lo[i] + k - op.shift[i]
                s = uverts @ a - b
                if s.min() >= -_TOL or s.max() <= _TOL:
                    continue  # does not cross the interior
                key = (a / np.linalg.norm(a), b / np.linalg.norm(a))
                if np.dot(key[0], [1.0, 0.5]) < 0:
                    key = (-key[0], -key[1])
                is_ident = np.array_equal(op.matrix, np.eye(2)) and np.allclose(
                    _mod1(op.shift), 0.0, atol=_TOL
                )
                dup = [j for j, (na, nb) in enumerate(seen)
                       if np.allclose(na, key[0], atol=_TOL)
                       and abs(nb - key[1]) <= _TOL]  # fmt: skip
                if dup:
                    if is_ident:
                        ident.append(dup[0])
                    continue
                seen.append(key)
                if is_ident:
                    ident.append(len(seen) - 1)
                pts = []
                for j in range(n):
                    s0, s1 = s[j], s[(j + 1) % n]
                    if abs(s0) <= _TOL:
                        pts.append(uverts[j])
                    elif s0 * s1 < 0 and abs(s1) > _TOL:
                        w = s0 / (s0 - s1)
                        pts.append(uverts[j] + w * (uverts[(j + 1) % n] - uverts[j]))
                out.append((len(seen) - 1, np.array(pts[:2]) @ t))
    return [seg for j, seg in out if j not in ident]


def _best_frame(
    sym: str, ops: Sequence[LatticeOp], t: np.ndarray, c: np.ndarray, lo: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """``(P, o, unit vertices (lattice), translations)``: origin nearest ``c``, fewest windows."""
    scale = math.sqrt(abs(float(np.linalg.det(t))))
    ita = ITA[sym]
    best = None
    for pk, p in enumerate(_signed_perms()):
        for o in _origins(p, ita, ops, c, lo):
            conj = [_conjugate(p, o, np.array(m), np.asarray(tau)) for m, tau in ita]
            if not _same_group(conj, ops):
                continue
            dist = round(float(np.linalg.norm(_mod1(o - c) @ t)) / (1e-7 * scale))
            uv0 = np.array(UNITS[sym], dtype=float) @ p.T + o
            if np.linalg.det(p) < 0:
                uv0 = uv0[::-1]
            base = np.floor(uv0.mean(axis=0) - lo)
            for dk in itertools.product((-1, 0, 1), repeat=2):
                shift = base + np.array(dk, dtype=float)
                uv = uv0 - shift
                cen = uv.mean(axis=0) - lo
                if not np.all((cen >= -0.5) & (cen < 1.5)):
                    continue
                trans = _translations(uv, lo)
                key = (dist, len(trans), round(float(np.linalg.norm((cen - 0.5) @ t))
                                               / (1e-7 * scale)), pk)  # fmt: skip
                if best is None or key < best[0]:
                    best = (key, p, o - shift, uv, trans)
    if best is None:
        raise AsymmetricUnitError(f"{sym}: no ITA frame reproduces the operations")
    return best[1], best[2], best[3], best[4]


def asymmetric_unit(
    ops: Sequence[LatticeOp],
    lattice: object,
    *,
    name: str | None = None,
    centre: Sequence[float] | None = None,
    window_lo: Sequence[float] = (0.0, 0.0),
) -> AsymmetricUnit:
    """The asymmetric unit of the group ``ops`` (modulo the lattice) on ``lattice``.

    Args:
        ops: The group modulo the lattice
            (:attr:`mpxlite.cells.detect.DetectedSymmetry.ops`).
        lattice: Rows ``T_1``, ``T_2`` (m); orthogonal unless the group is
            in :data:`LATTICE_FREE`.
        name: Wallpaper symbol (default :func:`mpxlite.cells.detect.wallpaper_name`).
        centre: Cartesian point (m) the ITA origin is placed nearest to
            (default the window centre).
        window_lo: Lower corner of the window (lattice coordinates).

    Raises:
        AsymmetricUnitError: a non-rectangular lattice with a group outside
            :data:`LATTICE_FREE`, a group outside
            :data:`ITA`, a supercell, or operations no ITA frame reproduces.
    """
    from mpxlite.cells.detect import wallpaper_name

    t = _lattice(lattice)
    scale = math.sqrt(abs(float(np.linalg.det(t))))
    ops = tuple(ops)
    sym = wallpaper_name(ops, t) if name is None else name
    oblique = abs(float(t[0] @ t[1])) > 1e-9 * float(np.linalg.norm(t[0]) * np.linalg.norm(t[1]))
    if oblique and sym not in LATTICE_FREE:
        raise AsymmetricUnitError(
            f"{sym} on a non-rectangular lattice (non-orthogonal vectors: oblique, hexagonal or "
            "rhombic): only p1 and p2 are supported on such a lattice"
        )
    if sym not in ITA:
        raise AsymmetricUnitError(f"plane group {sym} is not supported (rectangular / square only)")
    ita = ITA[sym]
    if len(ita) != len(ops):
        raise AsymmetricUnitError(
            f"{sym}: {len(ops)} operations modulo the lattice, ITA's conventional cell has "
            f"{len(ita)} (a supercell, or a primitive centred cell: give the conventional one)"
        )
    lo = np.asarray(window_lo, dtype=float).reshape(2)
    tinv = np.linalg.inv(t)
    c = (lo + 0.5) if centre is None else np.asarray(centre, dtype=float).reshape(2) @ tinv
    p, o, uv, trans = _best_frame(sym, ops, t, c, lo)
    verts = uv @ t
    edges = _edges(verts, ops, t, scale)
    cuts = _cuts(uv, ops, lo, t)
    unit = AsymmetricUnit(sym, t, ops, p, o, verts, tuple(edges), tuple(cuts), tuple(trans), lo)
    expect = abs(float(np.linalg.det(t))) / len(ops)
    if abs(unit.area - expect) > 1e-9 * expect:  # pragma: no cover - table error
        raise AsymmetricUnitError(f"{sym}: unit area {unit.area} != cell / order {expect}")
    return unit
