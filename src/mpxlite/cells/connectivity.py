# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""Element connectivity of a Gmsh mesh as numpy arrays (tetrahedra, triangles, Bloch links).

:func:`read_connectivity` parses a Gmsh MSH 2.x / 4.1 ASCII mesh into the
arrays a sparsity count needs, with no knowledge of what the tags mean
(physical tags only):

* the vertex nodes of every tetrahedron and of every triangle, with their
  physical tag (an element listed under several physicals appears once
  per physical, as in the file; the elementary entity of each triangle is
  kept too);
* the node coordinates;
* the ``$Periodic`` node correspondences of every link (all dimensions)
  as (slave, master) index pairs, and the elementary entities of the 2-D
  links (the slave and master Bloch surfaces).

Node numbers are replaced by compact indices ``0 .. n_nodes - 1`` (the
row of :attr:`MeshConnectivity.coords`). The result is cached per mesh
content (sha256, shared memo with :mod:`mpxlite.cells.periodic`).

Only tetrahedral volumes are supported (the first four nodes of a
higher-order tetrahedron are its vertices); a mesh with another volume
element type raises :class:`~mpxlite.cells.periodic.UnsupportedMeshFormatError`.
"""

from __future__ import annotations

import dataclasses
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mpxlite.cells.periodic import (
    _TRIANGLES,
    _VOLUMES,
    UnsupportedMeshFormatError,
    _digest_of,
    _read_format,
    _read_links,
    _sections,
)

__all__ = ["MeshConnectivity", "periodic_node_classes", "read_connectivity"]

_TETS = frozenset({4, 11, 29, 30, 31})
"""Gmsh tetrahedron types (first 4 nodes are the vertices)."""


@dataclass(frozen=True)
class MeshConnectivity:
    """Outcome of :func:`read_connectivity` (compact node indices throughout)."""

    path: Path
    """The mesh read."""
    coords: np.ndarray
    """``(n_nodes, 3)`` node coordinates."""
    tets: np.ndarray
    """``(n_tets, 4)`` vertex indices of the tetrahedra (one row per physical listing)."""
    tet_tags: np.ndarray
    """``(n_tets,)`` physical tag of each row of :attr:`tets` (0: none)."""
    tris: np.ndarray
    """``(n_tris, 3)`` vertex indices of the triangles (one row per physical listing)."""
    tri_tags: np.ndarray
    """``(n_tris,)`` physical tag of each row of :attr:`tris` (0: none)."""
    tri_entities: np.ndarray
    """``(n_tris,)`` elementary entity of each row of :attr:`tris`."""
    periodic_pairs: np.ndarray
    """``(n_pairs, 2)`` (slave, master) node indices of every ``$Periodic`` link."""
    periodic_surfaces: frozenset[int] = frozenset()
    """Elementary entities of the 2-D links (slave and master surfaces)."""
    sha256: str = field(default="", compare=False)
    """sha256 of the mesh bytes (cache key)."""

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return int(self.coords.shape[0])


_lock = threading.Lock()
_memo: dict[str, MeshConnectivity] = {}


def read_connectivity(msh_path: Path | str, *, use_cache: bool = True) -> MeshConnectivity:
    """Tetrahedra, triangles and Bloch node links of a mesh (see the module).

    Args:
        msh_path: Gmsh mesh, MSH 2.x or 4.1 ASCII.
        use_cache: Reuse the arrays of a mesh with the same bytes (default).

    Raises:
        UnsupportedMeshFormatError: not MSH 2.x / 4.1 ASCII, a missing
            section, a volume element that is not a tetrahedron, or an
            element node that is not in ``$Nodes``.
        FileNotFoundError: if the file does not exist.
    """
    path = Path(msh_path).expanduser().resolve()
    digest, data = _digest_of(path)
    if use_cache:
        with _lock:
            found = _memo.get(digest)
        if found is not None:
            return found if found.path == path else dataclasses.replace(found, path=path)
    if data is None:
        data = path.read_bytes()
    conn = _parse(path, data.decode("utf-8", errors="replace"), digest)
    with _lock:
        _memo[digest] = conn
    return conn


def periodic_node_classes(conn: MeshConnectivity) -> np.ndarray:
    """Representative node of the periodic class of every node, ``(n_nodes,)``.

    The classes are the connected components of the graph of every
    ``$Periodic`` (slave, master) pair (all links, all dimensions), so the
    corner copies chained through two links (``x+ y+ -> x- y+ -> x- y-``)
    share one class; the representative is the smallest index of the
    class. A node of no link is its own class.
    """
    label = np.arange(conn.n_nodes, dtype=np.int64)
    pairs = conn.periodic_pairs
    if pairs.size == 0:
        return label
    a, b = pairs[:, 0], pairs[:, 1]
    while True:  # min-label propagation + pointer jumping (a few sweeps)
        low = np.minimum(label[a], label[b])
        before = label.copy()
        np.minimum.at(label, a, low)
        np.minimum.at(label, b, low)
        while True:
            jumped = label[label]
            if np.array_equal(jumped, label):
                break
            label = jumped
        if np.array_equal(label, before):
            return label


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse(path: Path, text: str, digest: str) -> MeshConnectivity:
    lines = text.splitlines()
    sections = _sections(lines)
    _, major = _read_format(path, lines, sections)
    v41 = major == 4
    for name in ("Nodes", "Elements"):
        if name not in sections:
            raise UnsupportedMeshFormatError(f"{path}: no ${name} section")
    tags, xyz = (_nodes_v41 if v41 else _nodes_v2)(lines, sections["Nodes"])
    if tags.size == 0:
        raise UnsupportedMeshFormatError(f"{path}: the mesh has no node")
    index = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
    index[tags] = np.arange(tags.size)

    def compact(nodes: list[list[int]], width: int) -> np.ndarray:
        raw = np.array(nodes, dtype=np.int64).reshape(-1, width)
        if raw.size and (raw.max() >= index.size or raw.min() < 0):
            raise UnsupportedMeshFormatError(f"{path}: element node not in $Nodes")
        out = index[raw]
        if out.size and out.min() < 0:
            raise UnsupportedMeshFormatError(f"{path}: element node not in $Nodes")
        return out

    if v41:
        elements = _elements_v41(lines, sections["Elements"], sections.get("Entities"))
    else:
        elements = _elements_v2(lines, sections["Elements"])
    tets, tet_tags, tris, tri_tags, tri_ent, other = elements
    if other:
        raise UnsupportedMeshFormatError(
            f"{path}: {other} volume element(s) that are not tetrahedra (unsupported here)"
        )
    links = _read_links(lines, sections.get("Periodic"), v41=v41)
    pairs = [(s, m) for link in links for s, m in link.nodes.items()]
    known = [(s, m) for s, m in pairs if s < index.size and m < index.size]
    pair_arr = index[np.array(known, dtype=np.int64).reshape(-1, 2)]
    pair_arr = pair_arr[np.all(pair_arr >= 0, axis=1)]
    surfaces = frozenset(e for link in links if link.dim == 2 for e in (link.slave, link.master))
    return MeshConnectivity(
        path=path,
        coords=xyz,
        tets=compact(tets, 4),
        tet_tags=np.array(tet_tags, dtype=np.int64),
        tris=compact(tris, 3),
        tri_tags=np.array(tri_tags, dtype=np.int64),
        tri_entities=np.array(tri_ent, dtype=np.int64),
        periodic_pairs=pair_arr,
        periodic_surfaces=surfaces,
        sha256=digest,
    )


def _nodes_v2(lines: list[str], start: int) -> tuple[np.ndarray, np.ndarray]:
    count = int(lines[start])
    body = " ".join(lines[start + 1 : start + 1 + count]).split()
    arr = np.array(body, dtype=float).reshape(count, -1)[:, :4] if count else np.zeros((0, 4))
    return arr[:, 0].astype(np.int64), np.ascontiguousarray(arr[:, 1:4])


def _nodes_v41(lines: list[str], start: int) -> tuple[np.ndarray, np.ndarray]:
    tags: list[int] = []
    xyz: list[list[float]] = []
    j = start + 1
    for _ in range(int(lines[start].split()[0])):
        count = int(lines[j].split()[3])
        tags.extend(int(t) for t in lines[j + 1 : j + 1 + count])
        xyz.extend(
            [float(v) for v in ln.split()[:3]] for ln in lines[j + 1 + count : j + 1 + 2 * count]
        )
        j += 2 * count + 1
    return np.array(tags, dtype=np.int64), np.array(xyz, dtype=float).reshape(-1, 3)


_Elements = tuple[list[list[int]], list[int], list[list[int]], list[int], list[int], int]


def _elements_v2(lines: list[str], start: int) -> _Elements:
    tets: list[list[int]] = []
    tet_tags: list[int] = []
    tris: list[list[int]] = []
    tri_tags: list[int] = []
    tri_ent: list[int] = []
    other = 0
    for line in lines[start + 1 : start + 1 + int(lines[start])]:
        t = line.split()
        etype, ntags = int(t[1]), int(t[2])
        if etype in _TRIANGLES:
            k = 3 + ntags
            tris.append([int(t[k]), int(t[k + 1]), int(t[k + 2])])
            tri_tags.append(int(t[3]) if ntags >= 1 else 0)
            tri_ent.append(int(t[4]) if ntags >= 2 else 0)
        elif etype in _TETS:
            k = 3 + ntags
            tets.append([int(t[k]), int(t[k + 1]), int(t[k + 2]), int(t[k + 3])])
            tet_tags.append(int(t[3]) if ntags >= 1 else 0)
        elif etype in _VOLUMES:
            other += 1
    return tets, tet_tags, tris, tri_tags, tri_ent, other


def _elements_v41(lines: list[str], start: int, entities: int | None) -> _Elements:
    phys: dict[tuple[int, int], list[int]] = {}
    if entities is not None:
        counts = [int(v) for v in lines[entities].split()[:4]]
        j = entities + 1
        for dim, n in enumerate(counts):
            for _ in range(n):
                t = lines[j].split()
                j += 1
                k = 4 if dim == 0 else 7
                n_phys = int(t[k])
                phys[dim, int(t[0])] = [int(v) for v in t[k + 1 : k + 1 + n_phys]]
    tets: list[list[int]] = []
    tet_tags: list[int] = []
    tris: list[list[int]] = []
    tri_tags: list[int] = []
    tri_ent: list[int] = []
    other = 0
    j = start + 1
    for _ in range(int(lines[start].split()[0])):
        dim, entity, etype, count = (int(v) for v in lines[j].split()[:4])
        body = lines[j + 1 : j + 1 + count]
        j += count + 1
        if etype in _TRIANGLES:
            verts = [[int(v) for v in ln.split()[1:4]] for ln in body]
            for tag in phys.get((dim, entity), [0]):
                tris.extend(verts)
                tri_tags.extend([tag] * len(verts))
                tri_ent.extend([entity] * len(verts))
        elif etype in _TETS:
            verts = [[int(v) for v in ln.split()[1:5]] for ln in body]
            for tag in phys.get((dim, entity), [0]):
                tets.extend(verts)
                tet_tags.extend([tag] * len(verts))
        elif etype in _VOLUMES:
            other += count
    return tets, tet_tags, tris, tri_tags, tri_ent, other
