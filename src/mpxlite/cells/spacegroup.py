# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

r"""Space-group operations of a periodic cell and a node numbering they preserve.

Why a numbering
    For second-order elements, GetDP's H(curl) basis depends on the global
    node numbers: an edge is oriented from its smaller to its larger node number
    (sign of ``BF_Edge`` / ``BF_Edge_4E``) and the two facet functions
    ``BF_Edge_3F_b`` / ``BF_Edge_3F_c`` of a triangle are attached to its
    2nd / 3rd smallest node number (``Get_FacetFunctionIndex``). On a mesh
    invariant under an operation :math:`\mathcal G` the discrete problem is
    then invariant -- :math:`\mathcal G` maps every basis function onto a
    basis function, so the system at :math:`g\kappa` is the system at
    :math:`\kappa` permuted with phases -- exactly when :math:`\mathcal G`
    preserves the order of the node numbers inside every element; otherwise
    the images are other functions of the same space and the symmetry holds
    only to the discretisation error.

The numbering
    :func:`symmetric_node_order` sorts the nodes by a key constant on every
    orbit of the group (and of the lattice translations): the smallest
    image of the node folded onto the reference cell, then ``z``; ties
    (the members of one orbit, the periodic copies of one point) are
    broken by the position. Two distinct nodes of one element then never
    tie on the key **provided no element holds two nodes of one orbit**;
    every operation of the group, and every lattice translation (the
    order the ``$Periodic`` links need, see
    :func:`~mpxlite.cells.periodic.renumber_periodic`), then preserves the
    node order of every element. The condition holds on a mesh built from
    one fundamental domain of the group by replication
    (:func:`~mpxlite.cells.symmetric.build_c4v_mesh`): its images are
    separated by mesh faces, so the rotation axes and mirror planes are
    made of mesh edges / faces and an element lies in one image of the
    closed domain, which meets an orbit once. :func:`renumber_symmetric`
    rewrites an MSH 2.2 file with that numbering (and refuses an element
    with two nodes of one orbit); :func:`check_symmetric_numbering` checks
    a mesh against one operation.

Operations
    :class:`LatticeOp` is an affine map in the lattice coordinates ``u``
    (``x = u_1 T_1 + u_2 T_2`` in plane, ``z`` unchanged): an integer
    matrix ``M`` (``M = T^{-T} g T^{T}``, ``T`` the rows ``T_1``, ``T_2``)
    and a fractional translation ``tau``,

    .. math::
       u \mapsto M u + \tau, \qquad x \mapsto g x + t, \quad t = T^T \tau ,

    so any 2-D space group is described the same way (rotations, mirrors,
    glides; square, rectangular, oblique or hexagonal lattices).
    :func:`lattice_group` closes a set of generators modulo the lattice.

Mirrors and glides need the same numbering as rotations.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mpxlite.cells.periodic import (
    PeriodicFacetReport,
    UnsupportedMeshFormatError,
    _msh2_only,
    _read_format,
    _renumber_rows,
    _sections,
    align_periodic_facets,
)

__all__ = [
    "LatticeOp",
    "SymmetricNumberingError",
    "SymmetricNumberingReport",
    "check_symmetric_numbering",
    "lattice_group",
    "renumber_symmetric",
    "symmetric_node_order",
]

QUANTUM = 10**9
"""Quantisation of the lattice coordinates (1e-9 of a lattice vector), as ``renumber_periodic``."""

_LINEAR = frozenset({1, 2, 4, 15})
"""Element types :func:`renumber_symmetric` accepts (those of the symmetry gate)."""

MAX_GROUP = 48
"""Largest group accepted by :func:`lattice_group` (a 2-D space group has <= 12 mod lattice)."""


class SymmetricNumberingError(ValueError):
    """No numbering makes the operations order-preserving on this mesh (see the module)."""


def _lattice(lattice: object) -> np.ndarray:
    t = np.asarray(lattice, dtype=float).reshape(2, 2)
    det = float(np.linalg.det(t))
    if not math.isfinite(det) or abs(det) <= 1e-12 * float(np.prod(np.linalg.norm(t, axis=1))):
        raise ValueError(f"lattice vectors {t.tolist()} are not independent")
    return t


@dataclass(frozen=True, eq=False)
class LatticeOp:
    """An in-plane affine operation ``u -> M u + tau`` in lattice coordinates (``z`` kept).

    Attributes:
        name: Label.
        matrix: ``M``, ``(2, 2)`` integer, ``det M = +-1``.
        shift: ``tau``, ``(2,)`` fractional translation (lattice units). It is
            part of the operation: two operations that differ by a lattice
            translation act alike on a periodic mesh but give Bloch fields
            that differ by a phase.
    """

    name: str
    matrix: np.ndarray
    shift: np.ndarray

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=float).reshape(2, 2)
        if not np.array_equal(m, np.round(m)) or round(abs(np.linalg.det(m))) != 1:
            raise ValueError(f"{self.name}: matrix {m.tolist()} is not unimodular integer")
        mi = np.round(m).astype(np.int64)
        tau = np.asarray(self.shift, dtype=float).reshape(2)
        if not np.all(np.isfinite(tau)):
            raise ValueError(f"{self.name}: shift must be finite")
        for name, value in (("matrix", mi), ("shift", tau.copy())):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @classmethod
    def from_cartesian(
        cls,
        name: str,
        g: object,
        lattice: object,
        *,
        centre: Sequence[float] | None = None,
        translation: Sequence[float] | None = None,
    ) -> LatticeOp:
        """The operation ``x -> g x + t`` (``g`` orthogonal ``(2, 2)``).

        ``t`` is ``translation`` (m), or ``c - g c`` for a fixed point
        ``centre`` (default the origin). Raises ``ValueError`` if ``g`` does
        not map the lattice onto itself.
        """
        t = _lattice(lattice)
        gm = np.asarray(g, dtype=float).reshape(2, 2)
        if not np.allclose(gm @ gm.T, np.eye(2), rtol=0, atol=1e-12):
            raise ValueError(f"{name}: g is not orthogonal")
        tinv_t = np.linalg.inv(t.T)
        m = tinv_t @ gm @ t.T
        if np.max(np.abs(m - np.round(m))) > 1e-9:
            raise ValueError(f"{name}: g does not map the lattice onto itself")
        if translation is not None and centre is not None:
            raise ValueError("give centre or translation, not both")
        if translation is not None:
            tr = np.asarray(translation, dtype=float).reshape(2)
        else:
            c = np.zeros(2) if centre is None else np.asarray(centre, dtype=float).reshape(2)
            tr = c - gm @ c
        return cls(name, np.round(m), tinv_t @ tr)

    def cartesian(self, lattice: object) -> tuple[np.ndarray, np.ndarray]:
        """``(g, t)`` of ``x -> g x + t`` for the lattice rows ``T_1``, ``T_2`` (m)."""
        t = _lattice(lattice)
        g = t.T @ self.matrix @ np.linalg.inv(t.T)
        return g, t.T @ self.shift

    @property
    def det(self) -> int:
        """``+1`` for a rotation (or translation), ``-1`` for a mirror / glide."""
        return round(float(np.linalg.det(self.matrix)))

    def apply(self, u: np.ndarray) -> np.ndarray:
        """Image of lattice coordinates ``u`` ``(..., 2)``."""
        return np.asarray(u, dtype=float) @ self.matrix.T + self.shift

    def compose(self, other: LatticeOp, name: str | None = None) -> LatticeOp:
        """``self o other`` (``other`` first)."""
        return LatticeOp(
            name or f"{self.name}*{other.name}",
            self.matrix @ other.matrix,
            self.matrix @ other.shift + self.shift,
        )

    def inverse(self, name: str | None = None) -> LatticeOp:
        """The inverse operation."""
        mi = np.round(np.linalg.inv(self.matrix))
        return LatticeOp(name or f"{self.name}^-1", mi, -(mi @ self.shift))

    def key(self) -> tuple[int, ...]:
        """Identity of the operation modulo the lattice (matrix, quantised shift mod 1)."""
        q = np.mod(np.rint(self.shift * QUANTUM).astype(np.int64), QUANTUM)
        return (*self.matrix.ravel().tolist(), *q.tolist())


def _reduced(op: LatticeOp, name: str | None = None) -> LatticeOp:
    """``op`` with its shift reduced to ``[0, 1)`` (to 1e-9: ``-1e-9 <= tau < 1 - 1e-9``)."""
    tau = op.shift - np.floor(op.shift + 1e-9)
    return LatticeOp(op.name if name is None else name, op.matrix, tau + 0.0)


def lattice_group(generators: Iterable[LatticeOp]) -> tuple[LatticeOp, ...]:
    """Closure of ``generators`` modulo the lattice, the identity first.

    One representative per coset, its shift reduced to ``[0, 1)``: the
    representatives fix the mesh map, not a Bloch phase convention (two
    operations that differ by a lattice translation give Bloch fields that
    differ by a phase; take the phase from the operations themselves).

    Raises ``ValueError`` above :data:`MAX_GROUP` operations (a translation
    that is not of finite order modulo the lattice, e.g. an irrational one).
    """
    ident = LatticeOp("E", np.eye(2), np.zeros(2))
    gens = [_reduced(g) for g in generators]
    out = {ident.key(): ident}
    frontier = [ident]
    while frontier:
        nxt = []
        for a in frontier:
            for b in gens:
                c = _reduced(b.compose(a, name=b.name if a is ident else f"{b.name}*{a.name}"))
                k = c.key()
                if k not in out:
                    out[k] = c
                    nxt.append(c)
                    if len(out) > MAX_GROUP:
                        raise ValueError(f"the generators do not close within {MAX_GROUP} ops")
        frontier = nxt
    return tuple(out.values())


def _quantised(u: np.ndarray) -> np.ndarray:
    """``(n, 2)`` lattice coordinates folded onto ``[0, 1)``, as int64 multiples of 1e-9.

    Exact keys, no neighbour search: a node and its image whose coordinate
    straddles a rounding boundary get keys one unit apart (round-off). The
    orders built on these keys (:func:`orbit_keys`,
    :func:`symmetric_node_order`) assume distinct mesh nodes ``>> 1e-9`` of
    the lattice apart, so that no other node can hold the adjacent key: the
    tie then falls to the next sort key and the orders stay consistent.
    """
    return np.mod(np.rint(u * QUANTUM).astype(np.int64), QUANTUM)


def orbit_keys(
    coords: np.ndarray, lattice: object, group: Sequence[LatticeOp], z_quantum: float
) -> np.ndarray:
    """``(n, 2)`` int64 key of the orbit of every node: smallest folded image, quantised ``z``.

    Exact 1e-9 keys (see :func:`_quantised`: nodes assumed ``>> 1e-9`` of
    the lattice apart).
    """
    t = _lattice(lattice)
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    u = xyz[:, :2] @ np.linalg.inv(t)
    best = None
    for op in group:
        q = _quantised(op.apply(u))
        k = q[:, 0] * QUANTUM + q[:, 1]
        best = k if best is None else np.minimum(best, k)
    zq = np.rint(xyz[:, 2] / z_quantum).astype(np.int64)
    return np.column_stack([best, zq])


def symmetric_node_order(
    coords: np.ndarray, lattice: object, group: Sequence[LatticeOp]
) -> np.ndarray:
    """Node order (``argsort``) constant on the orbits of ``group`` and of the lattice.

    Sort key: :func:`orbit_keys` (the orbit), then the node's own folded
    lattice coordinates (the orbit member), then its unfolded ones (the
    periodic copy). ``group`` should contain the identity
    (:func:`lattice_group`).
    """
    t = _lattice(lattice)
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    zq = 1e-9 * float(np.max(np.abs(t)))
    orb = orbit_keys(xyz, t, group, zq)
    u = xyz[:, :2] @ np.linalg.inv(t)
    own = _quantised(u)
    return np.lexsort((u[:, 1], u[:, 0], own[:, 1], own[:, 0], orb[:, 1], orb[:, 0]))


def _element_conflicts(conn: np.ndarray, orbit: np.ndarray) -> int:
    """Elements (rows of node indices) holding two distinct nodes of one orbit."""
    bad = 0
    for row in conn:
        nodes = np.unique(row)
        if len({tuple(orbit[n]) for n in nodes.tolist()}) != len(nodes):
            bad += 1
    return bad


def _read_msh2(
    path: Path | str,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, list[int]]]]:
    """MSH 2.x ASCII: node tags, coordinates, elements ``(num, type, phys, nodes)`` by number."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    sections = _sections(lines)
    version, major = _read_format(path, lines, sections)
    if major != 2:
        raise _msh2_only(path, "the symmetric numbering reads", version)
    i = lines.index("$Nodes")
    n = int(lines[i + 1])
    arr = np.array(" ".join(lines[i + 2 : i + 2 + n]).split(), dtype=float).reshape(n, 4)
    j = lines.index("$Elements")
    elems = []
    for line in lines[j + 2 : j + 2 + int(lines[j + 1])]:
        t = line.split()
        ntags = int(t[2])
        phys = int(t[3]) if ntags >= 1 else 0
        elems.append((int(t[0]), int(t[1]), phys, [int(v) for v in t[3 + ntags :]]))
    elems.sort(key=lambda e: e[0])
    return arr[:, 0].astype(np.int64), np.ascontiguousarray(arr[:, 1:4]), elems


def renumber_symmetric(
    msh_path: Path | str,
    lattice: object,
    group: Sequence[LatticeOp],
    *,
    output: Path | str | None = None,
    align: bool = True,
) -> PeriodicFacetReport:
    """Renumber an MSH 2.x ASCII mesh by :func:`symmetric_node_order` of ``group``.

    The numbering is also the periodic one (every lattice translation
    preserves it), so the ``$Periodic`` links stay order-consistent;
    ``align``: the slave triangles are then stored in their master's order
    (:func:`~mpxlite.cells.periodic.align_periodic_facets`, default).

    Args:
        msh_path: The mesh (MSH 2.x ASCII, ``$Periodic`` links by lattice vectors).
        lattice: Rows ``T_1``, ``T_2`` (m) of the in-plane lattice.
        group: The operations to preserve (closed with :func:`lattice_group`
            when not already a group).
        output: Destination (default: rewrite ``msh_path`` in place).
        align: Align the slave periodic triangles (default).

    Returns:
        The :func:`~mpxlite.cells.periodic.check_periodic_facet_order` report.

    Raises:
        SymmetricNumberingError: an element holds two nodes of one orbit
            (no numbering preserves the order there).
        UnsupportedMeshFormatError: not MSH 2.x ASCII, or element types other
            than linear tetrahedra, triangles, segments and points (a
            second-order mesh: the conflict check and the gate do not cover it).
    """
    path = Path(msh_path).expanduser().resolve()
    t = _lattice(lattice)
    ops = lattice_group(group)
    tags, xyz, elems = _read_msh2(path)
    zq = 1e-9 * float(np.max(np.abs(t)))
    orbit = orbit_keys(xyz, t, ops, zq)
    other = sorted({e[1] for e in elems} - _LINEAR)
    if other:
        raise UnsupportedMeshFormatError(
            f"{path}: element type(s) {other} (linear tetrahedra and triangles only: the orbit "
            "conflict check and the symmetry gate cover those)"
        )
    row = {int(v): k for k, v in enumerate(tags.tolist())}
    vols = [np.array([row[v] for v in e[3][:4]]) for e in elems if e[1] == 4]
    bad = _element_conflicts(np.array(vols, dtype=np.int64).reshape(-1, 4), orbit)
    if bad:
        raise SymmetricNumberingError(
            f"{path}: {bad} tetrahedra hold two nodes of one orbit of the {len(ops)}-element "
            "group, whose order no numbering can preserve (the rotation axes / mirror planes "
            "must be made of mesh edges / faces, e.g. a mesh replicated from a fundamental "
            "domain)"
        )
    report = _renumber_rows(path, lambda c: symmetric_node_order(c, t, ops), output)
    if align and report.misoriented:
        report = align_periodic_facets(report.path)
    return report


@dataclass(frozen=True)
class SymmetricNumberingReport:
    """Outcome of :func:`check_symmetric_numbering` for one operation."""

    path: Path
    op: str
    n_elements: int
    """Tetrahedra and triangles checked (the triangles of the Bloch faces excluded)."""
    n_unmapped: int = 0
    """Elements whose image is not an element of the mesh (the mesh is not invariant)."""
    n_reordered: int = 0
    """Elements whose image has another node-number order (the basis is not invariant)."""
    tag_map: tuple[tuple[int, int, int], ...] = ()
    """``(dim, tag, image tag)`` of every physical tag that the operation moves."""

    @property
    def ok(self) -> bool:
        """Mesh invariant and every node order preserved (tags may be permuted)."""
        return self.n_unmapped == 0 and self.n_reordered == 0

    def summary(self) -> str:
        """One line."""
        moved = ", ".join(f"{d}:{a}->{b}" for d, a, b in self.tag_map) or "none"
        return (
            f"{self.path.name} under {self.op}: {self.n_elements} elements, "
            f"{self.n_unmapped} unmapped, {self.n_reordered} with another node order; "
            f"moved tags {moved}"
        )


def element_images(
    coords: np.ndarray,
    conn: Sequence[Sequence[int]],
    lattice: object,
    op: LatticeOp,
    find: Callable[[np.ndarray], np.ndarray],
    window_lo: np.ndarray,
) -> tuple[list[list[int] | None], np.ndarray]:
    """Image nodes (rows of ``coords``) of every element, folded into the window.

    The element ``P`` maps to ``g P + t - D`` with ``D`` the lattice vector
    that brings its centroid into the window ``[lo, lo + 1)`` (lattice
    coordinates); ``find`` maps ``(m, 3)`` absolute coordinates to rows
    (``-1``: no node there). Returns the image rows (``None`` for an element
    with a missing node) and ``D`` in lattice coordinates ``(n_el, 2)``.
    """
    t = _lattice(lattice)
    c = np.asarray(conn, dtype=np.int64).reshape(len(conn), -1)
    p = np.asarray(coords, dtype=float)[c]  # (n_el, k, 3)
    u = op.apply(p[..., :2] @ np.linalg.inv(t))
    shifts = np.floor(u.mean(axis=1) - window_lo + 1e-9)
    q = np.concatenate([(u - shifts[:, None, :]) @ t, p[..., 2:]], axis=2)
    rows = np.asarray(find(q.reshape(-1, 3))).reshape(c.shape)
    out = [None if np.any(r < 0) else r.tolist() for r in rows]
    return out, shifts


def check_symmetric_numbering(
    msh_path: Path | str, lattice: object, op: LatticeOp
) -> SymmetricNumberingReport:
    """Check that ``op`` maps the mesh onto itself with every node order preserved.

    Every tetrahedron and every triangle (but those of the 2-D ``$Periodic``
    surfaces) is mapped by ``op``, translated by a lattice vector into the
    window (the box of the nodes in lattice coordinates), its nodes found
    within ``1e-9`` of the lattice scale (the gate's 8-corner index, robust
    to rounding boundaries) and the element looked up among those of its
    kind; its nodes must have, in the image, the same order of node numbers.
    The physical tags may be permuted (tags exchanged by the operation); the
    report lists the moved ones.
    """
    from mpxlite.cells.periodic import _parse  # the 2-D link surfaces

    path = Path(msh_path).expanduser().resolve()
    t = _lattice(lattice)
    tags, xyz, elems = _read_msh2(path)
    row = {int(v): k for k, v in enumerate(tags.tolist())}
    mesh = _parse(path, path.read_text(encoding="utf-8"))
    skip = {e for lk in mesh.links if lk.dim == 2 for e in (lk.slave, lk.master)}
    text = path.read_text(encoding="utf-8").splitlines()
    j = text.index("$Elements")
    entity = {}
    for line in text[j + 2 : j + 2 + int(text[j + 1])]:
        tk = line.split()
        entity[int(tk[0])] = int(tk[4]) if int(tk[2]) >= 2 else 0
    from mpxlite.cells.symmetry_gate import _PointIndex  # the gate's robust node lookup

    points = _PointIndex(xyz, 1e-9 * float(np.max(np.abs(t))))
    window_lo = np.min(xyz[:, :2] @ np.linalg.inv(t), axis=0)
    n_el = n_unmapped = n_reordered = 0
    moved: dict[tuple[int, int], int] = {}
    for etype, nn, dim in ((4, 4, 3), (2, 3, 2)):
        sel = [e for e in elems if e[1] == etype and (dim == 3 or entity[e[0]] not in skip)]
        conn = [[row[v] for v in e[3][:nn]] for e in sel]
        index = {frozenset(c): k for k, c in enumerate(conn)}
        images, _ = element_images(xyz, conn, t, op, lambda q: points.find(q)[0], window_lo)
        for k, img in enumerate(images):
            n_el += 1
            hit = None if img is None else index.get(frozenset(img))
            if hit is None:
                n_unmapped += 1
                continue
            num = tags[np.array(conn[k])]
            num_img = tags[np.array(img)]  # type: ignore[arg-type]
            if not np.array_equal(np.argsort(num), np.argsort(num_img)):
                n_reordered += 1
            a, b = sel[k][2], sel[hit][2]
            if moved.setdefault((dim, a), b) != b:
                n_unmapped += 1  # a tag split between two images: not a symmetry
    tag_map = tuple(sorted((d, a, b) for (d, a), b in moved.items() if a != b))
    return SymmetricNumberingReport(path, op.name, n_el, n_unmapped, n_reordered, tag_map)
