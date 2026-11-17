# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

r"""Periodic tiling of a cell mesh into an ``N x M`` macrocell mesh (MSH 2.2 ASCII).

:func:`tile_periodic` writes the mesh of the macrocell made of ``N x M``
translated copies of a periodic cell mesh: copy ``(i, j)`` is the cell
translated by

.. math::

    R_{ij} = (o_1 + i)\, T_1 + (o_2 + j)\, T_2

(``origin`` = :math:`(o_1, o_2)`; default ``(0, 0)``, the copy ``(0, 0)`` is
the cell), the
nodes shared by adjacent copies (the facing Bloch faces) are merged, the
inner Bloch faces are dropped and the outer ones are linked by new
``$Periodic`` sections (translations ``N T_1`` and ``M T_2``). The cell
mesh being periodic (node-identical opposite faces), the result is a
conforming mesh whose every tetrahedron is a translated cell tetrahedron:
the macrocell discrete problem is exactly the cell one tiled, so a cell
Bloch-set decomposition and a direct macrocell solve agree to round-off
(an independently generated macrocell mesh would only agree to the mesh
error).

Node numbering: every macrocell element ranks its nodes as the cell
element it copies (the GetDP higher-order facet functions, attached by
node rank, are then the cell's on every copy, so the macrocell discrete
problem is the cell one tiled), and the outer periodic maps preserve the
facet node order (checked before second-order runs). The numbering is the
topological order of the node-rank constraints of all the copied elements,
which exists iff the cell numbering is consistent across its facing Bloch
faces: every slave Bloch facet ranks its nodes as its master image
(:func:`~mpxlite.cells.periodic.check_periodic_facet_order`, checked first)
and the copies glued on a Bloch face impose no contradictory order (a
cycle: refused). Any translation-invariant numbering satisfies this --
:func:`~mpxlite.cells.periodic.renumber_periodic` with any origin,
:func:`~mpxlite.cells.spacegroup.renumber_symmetric` -- and so does the
numbering of a gmsh mesh with ``setPeriodic`` faces whose facet check
passes. The slave facets of the outer links are then stored in the node
order of their masters (:func:`~mpxlite.cells.periodic.align_periodic_facets`).

Physical groups: every group of the cell is kept on every copy with its
tag, except that ``retag(dim, tag, name, i, j)`` may rename it per copy
(return ``(tag, name)``) or drop it (return ``None``); the Bloch-face groups
named in ``bloch_tags`` keep only their outer copies. Elementary entities
are made distinct per copy.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mpxlite.cells.periodic import (
    PeriodicFacetReport,
    _msh2_only,
    _write_mesh_text,
    align_periodic_facets,
    check_periodic_facet_order,
)

Retag = Callable[[int, int, str, int, int], "tuple[int, str] | None"]
"""``retag(dim, tag, name, i, j) -> (tag, name) | None`` of a physical group on copy ``(i, j)``."""

MERGE_RTOL = 1e-9
"""Distance (relative to the lattice scale) below which two copy nodes are the same node."""


@dataclass(frozen=True)
class TilingReport:
    """Summary of a tiling."""

    path: Path
    """The macrocell mesh written."""
    shape: tuple[int, int]
    """``(N, M)``."""
    n_nodes: int
    """Number of macrocell nodes (after merging)."""
    n_merged: int
    """Number of copy nodes merged into a node of another copy."""
    n_elements: int
    """Number of elements written."""
    periodic: PeriodicFacetReport
    """:func:`~mpxlite.cells.periodic.check_periodic_facet_order` of the output."""


@dataclass
class _Cell:
    names: dict[tuple[int, int], str]
    nodes: dict[int, tuple[float, float, float]]
    elements: list[tuple[int, int, int, tuple[int, ...]]]  # type, phys, entity, nodes
    links: list[tuple[int, int, int]]  # dim, slave entity, master entity


def _read_cell(path: Path) -> _Cell:
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = {s: k for k, s in enumerate(lines) if s.startswith("$")}
    fmt = lines[idx["$MeshFormat"] + 1].split()
    if not fmt[0].startswith("2.") or fmt[1] != "0":
        raise _msh2_only(path, "tile_periodic reads", " ".join(fmt[:2]))
    names: dict[tuple[int, int], str] = {}
    if "$PhysicalNames" in idx:
        k0 = idx["$PhysicalNames"]
        for s in lines[k0 + 2 : k0 + 2 + int(lines[k0 + 1])]:
            d, t, n = s.split(maxsplit=2)
            names[int(d), int(t)] = n.strip().strip('"')
    k0 = idx["$Nodes"]
    nodes = {}
    for s in lines[k0 + 2 : k0 + 2 + int(lines[k0 + 1])]:
        t = s.split()
        nodes[int(t[0])] = (float(t[1]), float(t[2]), float(t[3]))
    k0 = idx["$Elements"]
    elements = []
    for s in lines[k0 + 2 : k0 + 2 + int(lines[k0 + 1])]:
        t = [int(v) for v in s.split()]
        ntags = t[2]
        if ntags < 2:
            raise ValueError(f"{path}: element {t[0]} has no elementary tag")
        elements.append((t[1], t[3], t[4], tuple(t[3 + ntags :])))
    links = []
    if "$Periodic" in idx:
        k = idx["$Periodic"] + 2
        for _ in range(int(lines[idx["$Periodic"] + 1])):
            d, s, m = (int(v) for v in lines[k].split())
            k += 1
            if lines[k].startswith("Affine"):
                k += 1
            k += 1 + int(lines[k])
            links.append((d, s, m))
    return _Cell(names, nodes, elements, links)


class _Merger:
    """Geometric node merging by bucket hashing (distance <= ``tol``)."""

    def __init__(self, tol: float) -> None:
        self.tol = tol
        self.h = 4.0 * tol
        self.buckets: dict[tuple[int, int, int], list[int]] = {}
        self.xyz: list[tuple[float, float, float]] = []

    def _key(self, p: tuple[float, float, float]) -> tuple[int, int, int]:
        return (math.floor(p[0] / self.h), math.floor(p[1] / self.h), math.floor(p[2] / self.h))

    def find(self, p: tuple[float, float, float]) -> int | None:
        kx, ky, kz = self._key(p)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for n in self.buckets.get((kx + dx, ky + dy, kz + dz), ()):
                        q = self.xyz[n]
                        if max(abs(p[0] - q[0]), abs(p[1] - q[1]), abs(p[2] - q[2])) <= self.tol:
                            return n
        return None

    def add(self, p: tuple[float, float, float]) -> tuple[int, bool]:
        """Index of ``p`` (0-based) and whether it is new."""
        found = self.find(p)
        if found is not None:
            return found, False
        self.xyz.append(p)
        n = len(self.xyz) - 1
        self.buckets.setdefault(self._key(p), []).append(n)
        return n, True


_DIMS = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3, 15: 0, 8: 1, 9: 2, 11: 3}
"""Dimension of the Gmsh element types accepted."""

_Element = tuple[int, int, int, tuple[int, ...]]  # type, physical tag, entity, nodes
_Link = tuple[int, int, tuple[float, ...], list[tuple[int, int]]]  # slave, master, affine, pairs


@dataclass
class _Tiling:
    """Inputs of one tiling, shared by the steps of :func:`tile_periodic`."""

    src: Path
    cell: _Cell
    n: tuple[int, int]
    t: tuple[tuple[float, float], tuple[float, float]]
    origin: tuple[float, float]
    bloch: tuple[tuple[int, int], tuple[int, int]]
    tol: float

    @property
    def copies(self) -> list[tuple[int, int]]:
        return [(i, j) for j in range(self.n[1]) for i in range(self.n[0])]

    @property
    def n_ent(self) -> int:
        return 1 + max(e[2] for e in self.cell.elements)

    def is_bloch(self, phys: int) -> bool:
        return phys in (*self.bloch[0], *self.bloch[1])

    def is_outer(self, phys: int, i: int, j: int) -> bool:
        (xm, xp), (ym, yp) = self.bloch
        n1, n2 = self.n
        return (
            (phys == xm and i == 0)
            or (phys == xp and i == n1 - 1)
            or (phys == ym and j == 0)
            or (phys == yp and j == n2 - 1)
        )


def _check_cell_numbering(t: _Tiling) -> None:
    """Every slave Bloch facet of the cell ranks its nodes as its master image (in memory)."""
    report = check_periodic_facet_order(t.src)
    if not report.has_periodic:
        raise ValueError(f"{t.src}: the cell mesh has no 2-D $Periodic link")
    if report.bad or report.n_unmapped:
        raise ValueError(
            f"{t.src}: the cell node numbering does not preserve the node ranking of its "
            f"periodic facets ({report.summary()}); tile_periodic requires every slave Bloch "
            "facet to rank its nodes as its master image: apply "
            "mpxlite.cells.renumber_periodic(cell, t1, t2) (or renumber_symmetric) first"
        )


_CORNERS = {1: 2, 2: 3, 3: 4, 4: 4, 5: 8, 6: 6, 7: 5, 15: 1, 8: 2, 9: 3, 11: 4}
"""Corner nodes (the ones GetDP ranks) of the Gmsh element types accepted."""


def _macro_order(t: _Tiling, node_of: dict[tuple[int, int, int], int], n_nodes: int) -> list[int]:
    """Macrocell nodes in an order that ranks every copied element as the cell element.

    The corners of each cell element, sorted by cell number, give the
    constraints ``a < b`` of consecutive corners on every copy; the order is
    a topological sort (ties: first copy, then cell number).
    """
    nums = sorted(t.cell.nodes)
    row = {n: k for k, n in enumerate(nums)}
    copies = t.copies
    idx = np.empty((len(copies), len(nums)), dtype=np.int64)
    for c, (i, j) in enumerate(copies):
        idx[c] = [node_of[i, j, n] for n in nums]
    prio = np.full(n_nodes, np.iinfo(np.int64).max, dtype=np.int64)
    for c in range(len(copies)):
        np.minimum.at(prio, idx[c], c * len(nums) + np.arange(len(nums)))
    per_k: dict[int, list[list[int]]] = {}
    for etype, _, _, verts in t.cell.elements:
        k = _CORNERS[etype]
        if k > 1:
            per_k.setdefault(k, []).append(sorted(row[v] for v in verts[:k]))
    src, dst = [], []
    for rows in per_k.values():
        r = np.asarray(rows, dtype=np.int64)
        for c in range(len(copies)):
            m = idx[c][r]
            src.append(m[:, :-1].ravel())
            dst.append(m[:, 1:].ravel())
    order: list[int] = []
    if src:
        e = np.unique(np.column_stack([np.concatenate(src), np.concatenate(dst)]), axis=0)
        e = e[e[:, 0] != e[:, 1]]
        e = e[np.argsort(e[:, 0], kind="stable")]
        start = np.searchsorted(e[:, 0], np.arange(n_nodes + 1))
        succ = e[:, 1].tolist()
        indeg = np.bincount(e[:, 1], minlength=n_nodes).tolist()
    else:
        start, succ, indeg = np.zeros(n_nodes + 1, dtype=np.int64), [], [0] * n_nodes
    pr = prio.tolist()
    st = start.tolist()
    heap = [(pr[v], v) for v in range(n_nodes) if indeg[v] == 0]
    heapq.heapify(heap)
    while heap:
        _, v = heapq.heappop(heap)
        order.append(v)
        for w in succ[st[v] : st[v + 1]]:
            indeg[w] -= 1
            if indeg[w] == 0:
                heapq.heappush(heap, (pr[w], w))
    if len(order) != n_nodes:
        raise ValueError(
            f"{t.src}: no macrocell numbering ranks every copied element as the cell element "
            f"({n_nodes - len(order)} nodes on a cycle of contradictory ranks across the glued "
            "Bloch faces): the cell numbering is not consistent across its facing Bloch faces; "
            "apply mpxlite.cells.renumber_periodic(cell, t1, t2) (or renumber_symmetric) first"
        )
    return order


def _tile_nodes(t: _Tiling) -> tuple[_Merger, dict[tuple[int, int, int], int], int]:
    """Translated copy nodes, merged: (nodes, (i, j, cell node) -> index, merged count)."""
    (v1, v2), (o1, o2) = t.t, t.origin
    merger = _Merger(t.tol)
    node_of: dict[tuple[int, int, int], int] = {}
    n_merged = 0
    for i, j in t.copies:
        rx = (o1 + i) * v1[0] + (o2 + j) * v2[0]
        ry = (o1 + i) * v1[1] + (o2 + j) * v2[1]
        for n, (x, y, z) in t.cell.nodes.items():
            k, new = merger.add((x + rx, y + ry, z))
            node_of[i, j, n] = k
            n_merged += not new
    return merger, node_of, n_merged


def _tile_elements(
    t: _Tiling, node_of: dict[tuple[int, int, int], int], retag: Retag | None
) -> tuple[dict[tuple[int, int], str], list[_Element], dict[tuple[int, int, int], set[int]]]:
    """Copy elements: (physical names, elements, (Bloch tag, i, j) -> entities)."""
    groups: dict[tuple[int, int], str] = {}
    elements: list[_Element] = []
    faces: dict[tuple[int, int, int], set[int]] = {}
    n_ent = t.n_ent
    for c, (i, j) in enumerate(t.copies):
        for etype, phys, ent, verts in t.cell.elements:
            dim = _DIMS.get(etype)
            if dim is None:
                raise ValueError(f"{t.src}: unsupported element type {etype}")
            name0 = t.cell.names.get((dim, phys), str(phys))
            if t.is_bloch(phys):
                got: tuple[int, str] | None = (phys, name0) if t.is_outer(phys, i, j) else None
            else:
                got = (phys, name0) if retag is None else retag(dim, phys, name0, i, j)
            if got is None:
                continue
            tag, name = got
            prev = groups.setdefault((dim, tag), name)
            if prev != name:
                raise ValueError(f"physical group ({dim}, {tag}) named both {prev!r} and {name!r}")
            if t.is_bloch(phys):
                faces.setdefault((phys, i, j), set()).add(ent + c * n_ent)
            elements.append((etype, tag, ent + c * n_ent, tuple(node_of[i, j, v] for v in verts)))
    return groups, elements, faces


def _outer_links(
    t: _Tiling,
    xyz: list[tuple[float, float, float]],
    elements: list[_Element],
    faces: dict[tuple[int, int, int], set[int]],
) -> list[_Link]:
    """``$Periodic`` links: plus face of the last copy -> minus face of the first."""
    face_nodes: dict[int, set[int]] = {}
    for _, phys, ent, verts in elements:
        if t.is_bloch(phys):
            face_nodes.setdefault(ent, set()).update(verts)
    cell_pair = {s: m for d, s, m in t.cell.links if d == 2}  # slave -> master entity
    (_, xp), (_, yp) = t.bloch
    (n1, n2), (v1, v2), n_ent, copies = t.n, t.t, t.n_ent, t.copies
    links: list[_Link] = []
    for plus, axis in ((xp, 0), (yp, 1)):
        tx, ty = (n1 * v1[0], n1 * v1[1]) if axis == 0 else (n2 * v2[0], n2 * v2[1])
        for k in range(n2 if axis == 0 else n1):
            cp = copies.index((n1 - 1, k) if axis == 0 else (k, n2 - 1))
            cm = copies.index((0, k) if axis == 0 else (k, 0))
            for s_ent in sorted(faces.get((plus, *copies[cp]), ())):
                m_cell = cell_pair.get(s_ent - cp * n_ent)
                if m_cell is None:
                    raise ValueError(f"{t.src}: no periodic link from entity {s_ent - cp * n_ent}")
                m_ent = m_cell + cm * n_ent
                master = _Merger(t.tol)
                ids = sorted(face_nodes[m_ent])
                for n in ids:
                    master.add(xyz[n])
                pairs = []
                for n in sorted(face_nodes[s_ent]):
                    x, y, z = xyz[n]
                    hit = master.find((x - tx, y - ty, z))
                    if hit is None:
                        raise ValueError(f"outer periodic node {xyz[n]} has no image on the master")
                    pairs.append((n, ids[hit]))
                affine = (1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, 0, 0, 0, 0, 1)
                links.append((s_ent, m_ent, affine, pairs))
    return links


def _write_msh22(
    out: Path,
    groups: dict[tuple[int, int], str],
    xyz: list[tuple[float, float, float]],
    elements: list[_Element],
    links: list[_Link],
    order: list[int],
) -> None:
    """Write the macrocell; node ``order[k]`` gets the number ``k + 1``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    num = [0] * len(xyz)
    for k, v in enumerate(order):
        num[v] = k + 1
    w = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(groups))]
    w += [f'{d} {t} "{n}"' for (d, t), n in sorted(groups.items())]
    w += ["$EndPhysicalNames", "$Nodes", str(len(xyz))]
    w += [f"{k + 1} {xyz[v][0]!r} {xyz[v][1]!r} {xyz[v][2]!r}" for k, v in enumerate(order)]
    w += ["$EndNodes", "$Elements", str(len(elements))]
    w += [
        f"{k + 1} {t} 2 {p} {e} " + " ".join(str(num[v]) for v in verts)
        for k, (t, p, e, verts) in enumerate(elements)
    ]
    w += ["$EndElements", "$Periodic", str(len(links))]
    for s_ent, m_ent, affine, pairs in links:
        w.append(f"2 {s_ent} {m_ent}")
        w.append("Affine " + " ".join(repr(float(a)) if a else "0" for a in affine))
        w.append(str(len(pairs)))
        w += [f"{num[a]} {num[b]}" for a, b in pairs]
    w += ["$EndPeriodic", ""]
    _write_mesh_text(out, "\n".join(w))


def tile_periodic(
    cell_mesh: Path | str,
    output: Path | str,
    N: int,  # noqa: N803 -- spec notation
    M: int,  # noqa: N803
    t1: tuple[float, float],
    t2: tuple[float, float],
    *,
    bloch_tags: tuple[tuple[int, int], tuple[int, int]],
    retag: Retag | None = None,
    origin: tuple[float, float] = (0.0, 0.0),
) -> TilingReport:
    """Write the ``N x M`` periodic tiling of a cell mesh (see the module docstring).

    Args:
        cell_mesh: Periodic cell mesh, MSH 2.x ASCII with ``$Periodic`` links,
            numbered consistently across its facing Bloch faces (see the
            module; checked).
        output: The macrocell mesh to write (MSH 2.2 ASCII).
        N: Copies along ``t1`` (``>= 1``).
        M: Copies along ``t2`` (``>= 1``).
        t1: First cell lattice vector ``(x, y)`` (m).
        t2: Second cell lattice vector ``(x, y)`` (m).
        bloch_tags: Physical tags of the Bloch faces ``((T1 minus, T1 plus),
            (T2 minus, T2 plus))``; only their outer copies are kept and
            linked.
        retag: Per-copy renaming of the physical groups (default: keep).
        origin: Lattice coordinates ``(o1, o2)`` of the copy ``(0, 0)``
            (``(-(N-1)/2, -(M-1)/2)`` centres the macrocell on the cell).

    Returns:
        A :class:`TilingReport`.

    Raises:
        ValueError: invalid sizes, a mesh that is not MSH 2.x ASCII, has no
            periodic link or is not numbered consistently across its Bloch
            faces, unmatched outer periodic nodes, or a renamed group
            colliding with another one.
    """
    if int(N) != N or int(M) != M or N < 1 or M < 1:
        raise ValueError(f"N, M must be positive integers, got {(N, M)}")
    src = Path(cell_mesh).expanduser().resolve()
    out = Path(output).expanduser().resolve()
    v1 = (float(t1[0]), float(t1[1]))
    v2 = (float(t2[0]), float(t2[1]))
    o = (float(origin[0]), float(origin[1]))
    tiling = _Tiling(
        src, _read_cell(src), (int(N), int(M)), (v1, v2), o, bloch_tags,
        MERGE_RTOL * max(abs(c) for c in (*v1, *v2)),
    )  # fmt: skip
    _check_cell_numbering(tiling)
    merger, node_of, n_merged = _tile_nodes(tiling)
    order = _macro_order(tiling, node_of, len(merger.xyz))
    groups, elements, faces = _tile_elements(tiling, node_of, retag)
    links = _outer_links(tiling, merger.xyz, elements, faces)
    _write_msh22(out, groups, merger.xyz, elements, links, order)
    report = check_periodic_facet_order(out)
    if report.misoriented:
        report = align_periodic_facets(out)
    if not report.ok:
        raise RuntimeError(f"{out}: inconsistent outer periodic links: {report.summary()}")
    return TilingReport(out, (int(N), int(M)), len(merger.xyz), n_merged, len(elements), report)
