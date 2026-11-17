# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""Node-order consistency of the periodic (Bloch) facet pairs of a Gmsh mesh.

Why it matters
    For second-order elements the Hcurl basis of GetDP carries two facet functions per
    triangle, ``BF_Edge_3F_b`` / ``BF_Edge_3F_c``. GetDP attaches them to
    the facet's vertices by the RANK of their GLOBAL node numbers (the 2nd
    / 3rd smallest; ``Get_FacetFunctionIndex``, src/functions/BF_Edge_3.cpp),
    and the Bloch constraint ``LinkCplx`` on ``FacetsOf`` ties the
    coefficient of a slave facet to the SAME-NAMED coefficient of its
    image facet. The link is right only when the periodic map preserves
    the node-number ranking inside every linked facet; otherwise the
    linked facet DOFs belong to different functions and the run silently
    loses its second-order accuracy (measured on a hexagonal lattice:
    4.3e-3 amplitude error instead of 6e-6). Gmsh does not guarantee the
    property: it depends on the orientation in which opposite periodic
    sides are drawn.

What is checked
    For every 2-D ``$Periodic`` link (slave surface -> master surface) of
    the mesh, every triangle of the slave surface is mapped node by node to
    the master side (through the link's node correspondence; nodes it
    does not list -- MSH 4.1 lists boundary nodes under the curve / point
    links -- through any link of the same affine transformation, then
    geometrically by that transformation) and two properties are checked:

    * RANK: the order of the three vertex numbers is preserved (else
      the facet is in ``report.bad``: silent loss of accuracy);
    * STORED ORIENTATION: the image of the slave's stored node list is
      the stored node list of its master triangle (else the facet is in
      ``report.misoriented``: GetDP's facet ``Link`` maps the stored
      lists and stops in pre-processing with "7-Constraint Link: bad
      correspondance of facets"). A slave whose image is not a triangle
      of the master surface counts as unmapped.

    Only the vertex nodes (first three) of curved triangles are compared;
    other 2-D elements on a slave surface (quadrangles) are counted as
    unchecked.

Formats: Gmsh MSH 2.x ASCII (what GetDP reads and what the writers of
:mod:`mpxlite.cells` write) and MSH 4.1 ASCII; anything else raises
:class:`UnsupportedMeshFormatError`. :func:`renumber_periodic` rewrites an
MSH 2.x ASCII file with an order-preserving numbering and (by default)
:func:`align_periodic_facets` the slave triangles in their master's stored
order.

Face mesh size
    :func:`measure_z_faces` measures, from the same parse, the triangles
    lying in the planes of the lowest and highest node -- the top and
    bottom faces of a periodic box -- (count, mean edge length, area).

The results of :func:`check_periodic_facet_order` and
:func:`measure_z_faces` are cached per mesh CONTENT (sha256 of the bytes,
itself memoised on path, size and mtime); the first call of either parses
the mesh once and fills both, so repeated runs on one mesh (sweeps, cached
studies) parse it once.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "BadFacet",
    "MeshGroups",
    "PeriodicFacetOrderError",
    "PeriodicFacetReport",
    "PhysicalGroupMesh",
    "UnsupportedMeshFormatError",
    "ZFaceMesh",
    "ZPlaneMesh",
    "align_periodic_facets",
    "check_periodic_facet_order",
    "measure_physical_groups",
    "measure_z_faces",
    "renumber_periodic",
]

# Gmsh element types whose first three nodes are the vertices of a triangle
# (3-, 6-, 9-, 10-, 12-, 15-, 15-, 21-node triangles), and the other 2-D types.
_TRIANGLES = frozenset({2, 9, 20, 21, 22, 23, 24, 25})
_OTHER_2D = frozenset({3, 10, 16, 36, 37, 38, 47, 48, 49, 50, 51})
# 3-D element types (tetrahedra, hexahedra, prisms, pyramids of any order).
_VOLUMES = frozenset({4, 5, 6, 7, 11, 12, 13, 14, 17, 18, 19, 29, 30, 31, 92, 93})

_GEOM_REL_TOL = 1e-8
"""Geometric node matching tolerance, relative to the mesh extent."""


MSH2_CONVERSION = (
    "convert it with `gmsh mesh.msh -save -format msh22 -o mesh22.msh` (or write it with the "
    "gmsh option Mesh.MshFileVersion = 2.2)"
)
"""How to get the MSH 2.2 ASCII file the rewriting and symmetry tools need."""


def _msh2_only(path: Path, what: str, version: object) -> UnsupportedMeshFormatError:
    """The refusal of a non-2.x mesh by a tool that reads MSH 2.x ASCII only."""
    return UnsupportedMeshFormatError(
        f"{path}: {what} MSH 2.x ASCII only (got {version}); {MSH2_CONVERSION}"
    )


class UnsupportedMeshFormatError(ValueError):
    """The mesh is not in a format this module can read (MSH 2.x / 4.1 ASCII)."""


@dataclass(frozen=True)
class BadFacet:
    """A slave facet whose node-number order the periodic map does not preserve."""

    slave_entity: int
    """Elementary tag of the slave surface."""
    master_entity: int
    """Elementary tag of its master surface."""
    nodes: tuple[int, int, int]
    """Vertex node numbers of the slave triangle (file order)."""
    image: tuple[int, int, int]
    """Their images on the master side (same order)."""
    master_nodes: tuple[int, int, int] | None = None
    """Stored node order of the master triangle (set for a misoriented facet)."""


@dataclass(frozen=True)
class PeriodicFacetReport:
    """Outcome of :func:`check_periodic_facet_order` for one mesh."""

    path: Path
    """The mesh checked."""
    msh_version: str
    """MSH format version read from ``$MeshFormat``."""
    n_links: int
    """Number of 2-D periodic links (slave surface -> master surface)."""
    n_facets: int
    """Number of slave triangles checked."""
    bad: tuple[BadFacet, ...] = ()
    """Slave triangles whose node-number order is NOT preserved."""
    n_unmapped: int = 0
    """Slave triangles with a vertex that has no image on the master side, or whose image
    is not a triangle of the master surface."""
    misoriented: tuple[BadFacet, ...] = ()
    """Slave triangles whose STORED node order, mapped to the master side, differs from
    the stored node order of their master triangle (GetDP stops in pre-processing with
    "7-Constraint Link: bad correspondance of facets")."""
    n_unchecked: int = 0
    """Non-triangular 2-D elements on slave surfaces (not checked)."""
    sha256: str = field(default="", compare=False)
    """sha256 of the mesh bytes (cache key)."""

    @property
    def n_bad(self) -> int:
        """Number of order-violating facets."""
        return len(self.bad)

    @property
    def n_misoriented(self) -> int:
        """Number of slave facets stored with another node order than their master."""
        return len(self.misoriented)

    @property
    def has_periodic(self) -> bool:
        """Whether the mesh declares any 2-D periodic link."""
        return self.n_links > 0

    @property
    def ok(self) -> bool:
        """No order-violating, misoriented or unmapped facet (vacuously true without links)."""
        return not self.bad and not self.misoriented and self.n_unmapped == 0

    def summary(self) -> str:
        """One-line human-readable summary."""
        head = (
            f"{self.path.name}: {self.n_links} periodic surface link(s), "
            f"{self.n_facets} slave facet(s), {self.n_bad} with a node order not preserved"
        )
        if self.misoriented:
            head += f", {self.n_misoriented} stored with another orientation than their master"
        if self.n_unmapped:
            head += f", {self.n_unmapped} unmapped"
        if self.n_unchecked:
            head += f", {self.n_unchecked} non-triangular unchecked"
        return head


class PeriodicFacetOrderError(ValueError):
    """A run was refused: the mesh's periodic maps do not preserve facet node order.

    Its ``report`` attribute is the :class:`PeriodicFacetReport` of the mesh.
    """

    def __init__(self, message: str, report: PeriodicFacetReport) -> None:
        super().__init__(message)
        self.report = report


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass
class _Link:
    dim: int
    slave: int
    master: int
    affine: tuple[float, ...] | None
    nodes: dict[int, int]


@dataclass
class _Mesh:
    version: str
    links: list[_Link]
    facets: dict[int, list[tuple[int, int, int]]]  # slave / master entity -> vertex triples
    n_unchecked: int
    triangles: list[tuple[int, int, int]]  # every triangle of the mesh (vertex triples)
    coord_reader: Callable[[], dict[int, tuple[float, float, float]]]
    group_reader: Callable[[], dict[tuple[int, int], tuple[int, set[int], list[tuple[int, ...]]]]]
    """(dim, physical tag) -> (element count, node set, vertex triples of its triangles)."""
    _coords: dict[int, tuple[float, float, float]] | None = None

    @property
    def coords(self) -> dict[int, tuple[float, float, float]]:
        if self._coords is None:
            self._coords = self.coord_reader()
        return self._coords


def _sections(lines: Sequence[str]) -> dict[str, int]:
    """First line index of the content of every ``$Section``."""
    out: dict[str, int] = {}
    for i, line in enumerate(lines):
        if line.startswith("$") and not line.startswith("$End"):
            out.setdefault(line.strip()[1:], i + 1)
    return out


def _read_format(path: Path, lines: Sequence[str], sections: dict[str, int]) -> tuple[str, int]:
    """``(version string, major)`` of an ASCII MSH 2.x / 4.1 file."""
    if "MeshFormat" not in sections:
        first = lines[0] if lines else ""
        raise UnsupportedMeshFormatError(
            f"{path}: no $MeshFormat header (first line {first!r}); only Gmsh MSH 2.x and "
            "4.1 ASCII meshes are supported"
        )
    tokens = lines[sections["MeshFormat"]].split()
    try:
        version, file_type = tokens[0], int(tokens[1])
        major = int(float(version))
    except (IndexError, ValueError) as exc:
        raise UnsupportedMeshFormatError(f"{path}: unreadable $MeshFormat {tokens}") from exc
    if file_type != 0:
        raise UnsupportedMeshFormatError(f"{path}: binary MSH {version}; only ASCII is supported")
    if major != 2 and version != "4.1":
        raise UnsupportedMeshFormatError(
            f"{path}: MSH {version} is not supported (MSH 2.x or 4.1 ASCII)"
        )
    return version, major


def _read_links(lines: Sequence[str], start: int | None, *, v41: bool) -> list[_Link]:
    """Every ``$Periodic`` link (all dimensions)."""
    if start is None:
        return []
    links: list[_Link] = []
    j = start + 1
    for _ in range(int(lines[start])):
        dim, slave, master = (int(t) for t in lines[j].split()[:3])
        j += 1
        affine: tuple[float, ...] | None = None
        head = lines[j].split()
        if v41:
            if int(head[0]) > 0:
                affine = tuple(float(t) for t in head[1:])
            j += 1
        elif head and head[0] == "Affine":
            affine = tuple(float(t) for t in head[1:])
            j += 1
        count = int(lines[j])
        nodes = {}
        for line in lines[j + 1 : j + 1 + count]:
            a, b = line.split()[:2]
            nodes[int(a)] = int(b)
        j += count + 1
        links.append(_Link(dim, slave, master, affine, nodes))
    return links


def _elements_v2(
    lines: Sequence[str], start: int, slaves: set[int], wanted: set[int]
) -> tuple[dict[int, list[tuple[int, int, int]]], int, list[tuple[int, int, int]]]:
    """Wanted-surface triangles, slave unchecked 2-D count, every triangle (MSH 2.x).

    The triangles of the ``wanted`` (slave + master) surfaces keep their
    stored node order (first occurrence of an element repeated across
    physicals, as GetDP); every triangle of the mesh is deduplicated.
    """
    seen: dict[int, dict[tuple[int, ...], tuple[int, int, int]]] = {}
    every: dict[tuple[int, ...], tuple[int, int, int]] = {}
    unchecked: set[tuple[int, ...]] = set()
    for line in lines[start + 1 : start + 1 + int(lines[start])]:
        t = line.split()
        etype, ntags = int(t[1]), int(t[2])
        if etype not in _TRIANGLES and etype not in _OTHER_2D:
            continue
        entity = int(t[4] if ntags >= 2 else t[3])
        verts = tuple(int(n) for n in t[3 + ntags :])
        if etype in _TRIANGLES:
            tri = (verts[0], verts[1], verts[2])
            every.setdefault(tuple(sorted(tri)), tri)
        if entity not in wanted:
            continue
        if etype in _TRIANGLES:
            seen.setdefault(entity, {}).setdefault(tuple(sorted(tri)), tri)
        elif entity in slaves:
            unchecked.add((entity, *sorted(verts)))
    return {e: list(d.values()) for e, d in seen.items()}, len(unchecked), list(every.values())


def _elements_v41(
    lines: Sequence[str], start: int, slaves: set[int], wanted: set[int]
) -> tuple[dict[int, list[tuple[int, int, int]]], int, list[tuple[int, int, int]]]:
    """As :func:`_elements_v2`, MSH 4.1."""
    facets: dict[int, list[tuple[int, int, int]]] = {}
    every: dict[tuple[int, ...], tuple[int, int, int]] = {}
    unchecked = 0
    j = start + 1
    for _ in range(int(lines[start].split()[0])):
        dim, entity, etype, count = (int(t) for t in lines[j].split()[:4])
        body = lines[j + 1 : j + 1 + count]
        j += count + 1
        if dim != 2:
            continue
        if etype not in _TRIANGLES:
            if entity in slaves:
                unchecked += count
            continue
        for line in body:
            t = line.split()
            tri = (int(t[1]), int(t[2]), int(t[3]))
            every.setdefault(tuple(sorted(tri)), tri)
            if entity in wanted:
                facets.setdefault(entity, []).append(tri)
    return facets, unchecked, list(every.values())


def _coords_v2(lines: Sequence[str], start: int) -> dict[int, tuple[float, float, float]]:
    out = {}
    for line in lines[start + 1 : start + 1 + int(lines[start])]:
        t = line.split()
        out[int(t[0])] = (float(t[1]), float(t[2]), float(t[3]))
    return out


def _coords_v41(lines: Sequence[str], start: int) -> dict[int, tuple[float, float, float]]:
    out = {}
    j = start + 1
    for _ in range(int(lines[start].split()[0])):
        count = int(lines[j].split()[3])
        tags = [int(t) for t in lines[j + 1 : j + 1 + count]]
        for k, tag in enumerate(tags):
            t = lines[j + 1 + count + k].split()
            out[tag] = (float(t[0]), float(t[1]), float(t[2]))
        j += 2 * count + 1
    return out


def _parse(path: Path, text: str) -> _Mesh:
    lines = text.splitlines()
    sections = _sections(lines)
    version, major = _read_format(path, lines, sections)
    v41 = major == 4
    for name in ("Nodes", "Elements"):
        if name not in sections:
            raise UnsupportedMeshFormatError(f"{path}: no ${name} section")
    links = _read_links(lines, sections.get("Periodic"), v41=v41)
    slaves = {link.slave for link in links if link.dim == 2}
    wanted = slaves | {link.master for link in links if link.dim == 2}
    reader = _elements_v41 if v41 else _elements_v2
    facets, unchecked, triangles = reader(lines, sections["Elements"], slaves, wanted)
    coords_of = _coords_v41 if v41 else _coords_v2
    start = sections["Nodes"]
    if v41:
        groups_of = lambda: _groups_v41(lines, sections["Elements"], sections.get("Entities"))  # noqa: E731
    else:
        groups_of = lambda: _groups_v2(lines, sections["Elements"])  # noqa: E731
    return _Mesh(
        version,
        links,
        facets,
        unchecked,
        triangles,
        lambda: coords_of(lines, start),
        groups_of,
    )


_Groups = dict[tuple[int, int], tuple[int, set[int], list[tuple[int, ...]]]]


def _add_element(groups: _Groups, dim: int, phys: int, etype: int, verts: Sequence[int]) -> None:
    count, nodes, tris = groups.setdefault((dim, phys), (0, set(), []))
    nodes.update(verts)
    if etype in _TRIANGLES:
        tris.append((verts[0], verts[1], verts[2]))
    groups[dim, phys] = (count + 1, nodes, tris)


def _dim_of(etype: int) -> int | None:
    if etype in _TRIANGLES or etype in _OTHER_2D:
        return 2
    if etype in _VOLUMES:
        return 3
    return None


def _groups_v2(lines: Sequence[str], start: int) -> _Groups:
    """Per (dim, physical tag) element data of the 2-D / 3-D elements (MSH 2.x)."""
    groups: _Groups = {}
    for line in lines[start + 1 : start + 1 + int(lines[start])]:
        t = line.split()
        etype, ntags = int(t[1]), int(t[2])
        dim = _dim_of(etype)
        if dim is None or ntags < 1:
            continue
        _add_element(groups, dim, int(t[3]), etype, [int(n) for n in t[3 + ntags :]])
    return groups


def _groups_v41(lines: Sequence[str], start: int, entities: int | None) -> _Groups:
    """As :func:`_groups_v2`, MSH 4.1 (physical tags from ``$Entities``)."""
    phys: dict[tuple[int, int], list[int]] = {}
    if entities is not None:
        counts = [int(v) for v in lines[entities].split()[:4]]
        j = entities + 1
        for dim, n in enumerate(counts):
            for _ in range(n):
                t = lines[j].split()
                j += 1
                k = 4 if dim == 0 else 7  # tag + x y z (points) or bounding box
                n_phys = int(t[k])
                phys[dim, int(t[0])] = [int(v) for v in t[k + 1 : k + 1 + n_phys]]
    groups: _Groups = {}
    j = start + 1
    for _ in range(int(lines[start].split()[0])):
        dim, entity, etype, count = (int(v) for v in lines[j].split()[:4])
        body = lines[j + 1 : j + 1 + count]
        j += count + 1
        if dim not in (2, 3) or _dim_of(etype) is None:
            continue
        for tag in phys.get((dim, entity), []):
            for line in body:
                _add_element(groups, dim, tag, etype, [int(v) for v in line.split()[1:]])
    return groups


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------


def _affine_key(affine: tuple[float, ...] | None) -> tuple[float, ...] | None:
    return None if affine is None else tuple(round(v, 9) for v in affine)


class _Mapper:
    """Image of a slave node on the master side of one link."""

    def __init__(self, mesh: _Mesh) -> None:
        self.mesh = mesh
        self._union: dict[tuple[float, ...] | None, dict[int, int]] = {}
        for link in mesh.links:
            self._union.setdefault(_affine_key(link.affine), {}).update(link.nodes)
        self._grid: dict[tuple[int, int, int], list[int]] | None = None
        self._h = 1.0

    def __call__(self, link: _Link, node: int) -> int | None:
        found = link.nodes.get(node)
        if found is None and link.affine is not None:
            found = self._union[_affine_key(link.affine)].get(node)
        if found is None:
            found = self._geometric(link, node)
        return found

    def _inverse(self, link: _Link) -> np.ndarray | None:
        coords = self.mesh.coords
        if link.affine is not None and len(link.affine) == 16:
            return np.linalg.inv(np.array(link.affine, dtype=float).reshape(4, 4))
        for a, b in link.nodes.items():  # derive a translation from one pair
            if a in coords and b in coords:
                inv = np.eye(4)
                inv[:3, 3] = np.subtract(coords[b], coords[a])
                return inv
        return None

    def _geometric(self, link: _Link, node: int) -> int | None:
        coords = self.mesh.coords
        inv = self._inverse(link)
        if inv is None or node not in coords:
            return None
        target = inv @ np.array([*coords[node], 1.0])
        grid = self._index()
        h = self._h
        cell = tuple(math.floor(c / h) for c in target[:3])
        for offset in np.ndindex(3, 3, 3):
            key = (cell[0] + offset[0] - 1, cell[1] + offset[1] - 1, cell[2] + offset[2] - 1)
            for cand in grid.get(key, ()):
                if math.dist(coords[cand], target[:3]) < h:
                    return cand
        return None

    def _index(self) -> dict[tuple[int, int, int], list[int]]:
        if self._grid is None:
            coords = self.mesh.coords
            pts = np.array(list(coords.values())) if coords else np.zeros((1, 3))
            extent = float(np.max(np.ptp(pts, axis=0))) or 1.0
            self._h = _GEOM_REL_TOL * extent
            self._grid = {}
            for tag, xyz in coords.items():
                key = (
                    math.floor(xyz[0] / self._h),
                    math.floor(xyz[1] / self._h),
                    math.floor(xyz[2] / self._h),
                )
                self._grid.setdefault(key, []).append(tag)
        return self._grid


def _rank(nodes: Sequence[int]) -> list[int]:
    return sorted(range(len(nodes)), key=nodes.__getitem__)


def _check(path: Path, mesh: _Mesh, digest: str) -> PeriodicFacetReport:
    mapper = _Mapper(mesh)
    bad: list[BadFacet] = []
    misoriented: list[BadFacet] = []
    n_facets = n_unmapped = 0
    surface_links = [link for link in mesh.links if link.dim == 2]
    for link in surface_links:
        masters = {frozenset(t): t for t in mesh.facets.get(link.master, [])}
        for tri in mesh.facets.get(link.slave, []):
            n_facets += 1
            a, b, c = (mapper(link, n) for n in tri)
            if a is None or b is None or c is None or len({a, b, c}) != 3:
                n_unmapped += 1
                continue
            img = (a, b, c)
            stored = masters.get(frozenset(img))
            if stored is None:
                n_unmapped += 1
                continue
            if _rank(tri) != _rank(img):
                bad.append(BadFacet(link.slave, link.master, tri, img))
            if img != stored:
                misoriented.append(BadFacet(link.slave, link.master, tri, img, stored))
    return PeriodicFacetReport(
        path=path,
        msh_version=mesh.version,
        n_links=len(surface_links),
        n_facets=n_facets,
        bad=tuple(bad),
        n_unmapped=n_unmapped,
        misoriented=tuple(misoriented),
        n_unchecked=mesh.n_unchecked,
        sha256=digest,
    )


# ---------------------------------------------------------------------------
# Mesh size of the top / bottom faces of the periodic box
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZPlaneMesh:
    """The triangles of a mesh lying in one plane ``z = const``."""

    z: float
    """Height of the plane (m)."""
    n_triangles: int
    """Number of triangles whose three vertices lie in the plane."""
    mean_edge: float
    """Mean edge length of those triangles (each triangle's three edges); NaN if none."""
    area: float
    """Total area of those triangles."""


@dataclass(frozen=True)
class ZFaceMesh:
    """Outcome of :func:`measure_z_faces`: the faces ``z = z_min`` and ``z = z_max``."""

    path: Path
    """The mesh measured."""
    bottom: ZPlaneMesh
    """The plane of the lowest node."""
    top: ZPlaneMesh
    """The plane of the highest node."""
    sha256: str = field(default="", compare=False)
    """sha256 of the mesh bytes (cache key)."""

    @property
    def found(self) -> bool:
        """Whether both planes carry triangles."""
        return self.bottom.n_triangles > 0 and self.top.n_triangles > 0

    @property
    def mean_edge(self) -> float:
        """Mean edge length over the triangles of BOTH planes (NaN if none)."""
        n = self.bottom.n_triangles + self.top.n_triangles
        if n == 0:
            return math.nan
        total = sum(p.mean_edge * p.n_triangles for p in (self.bottom, self.top) if p.n_triangles)
        return total / n

    def summary(self) -> str:
        """One-line human-readable summary."""
        parts = [
            f"{name} z = {p.z:.6g}: {p.n_triangles} triangle(s), mean edge {p.mean_edge:.4g}, "
            f"area {p.area:.4g}"
            for name, p in (("bottom", self.bottom), ("top", self.top))
        ]
        return f"{self.path.name}: " + "; ".join(parts)


def _z_faces(path: Path, mesh: _Mesh, digest: str) -> ZFaceMesh:
    coords = mesh.coords
    if not coords:
        raise UnsupportedMeshFormatError(f"{path}: the mesh has no node")
    xyz = np.array(list(coords.values()), dtype=float)
    tol = _GEOM_REL_TOL * (float(np.max(np.ptp(xyz, axis=0))) or 1.0)
    z_lo, z_hi = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    if mesh.triangles:
        tri = np.array(
            [[coords[n] for n in t] for t in mesh.triangles if all(n in coords for n in t)],
            dtype=float,
        ).reshape(-1, 3, 3)
    else:
        tri = np.zeros((0, 3, 3))

    def plane(z: float) -> ZPlaneMesh:
        sel = tri[np.all(np.abs(tri[:, :, 2] - z) <= tol, axis=1)]
        if len(sel) == 0:
            return ZPlaneMesh(z, 0, math.nan, 0.0)
        e = sel[:, [1, 2, 0], :] - sel  # the three edges of every triangle
        lengths = np.linalg.norm(e, axis=2)
        area = 0.5 * np.linalg.norm(np.cross(e[:, 0], -e[:, 2]), axis=1)
        return ZPlaneMesh(z, len(sel), float(lengths.mean()), float(area.sum()))

    return ZFaceMesh(path=path, bottom=plane(z_lo), top=plane(z_hi), sha256=digest)


@dataclass(frozen=True)
class PhysicalGroupMesh:
    """The 2-D or 3-D elements of one physical group (:func:`measure_physical_groups`)."""

    dim: int
    """Dimension of the elements (2: surface, 3: volume)."""
    tag: int
    """Physical (level-1) tag."""
    n_elements: int
    """Number of elements of that dimension carrying the tag."""
    z_min: float
    """Lowest node height of those elements (m)."""
    z_max: float
    """Highest node height (m)."""
    n_triangles: int = 0
    """Number of triangles (2-D groups)."""
    mean_edge: float = math.nan
    """Mean edge length of the triangles (NaN if none)."""
    area: float = 0.0
    """Total area of the triangles."""
    nodes: frozenset[int] = field(default=frozenset(), compare=False, repr=False)
    """Node numbers of the elements (2-D groups only; empty for volumes)."""
    normals: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)), compare=False, repr=False)
    """Unit normals of the triangles, ``(n_triangles, 3)`` (2-D groups; sign arbitrary)."""

    @property
    def planar_z(self) -> bool:
        """Whether every node of the group lies in one plane ``z = const`` (to round-off)."""
        return self.z_max - self.z_min <= 1e-9 * max(1.0, abs(self.z_max), abs(self.z_min))


@dataclass(frozen=True)
class MeshGroups:
    """Outcome of :func:`measure_physical_groups`: per-physical-group extents and face sizes."""

    path: Path
    """The mesh measured."""
    z_min: float
    """Lowest node height of the mesh."""
    z_max: float
    """Highest node height of the mesh."""
    groups: tuple[PhysicalGroupMesh, ...]
    """One entry per (dimension, physical tag) of the 2-D and 3-D elements, sorted."""
    tol: float = 0.0
    """Geometric tolerance of the plane tests (1e-8 of the mesh extent)."""
    sha256: str = field(default="", compare=False)
    """sha256 of the mesh bytes (cache key)."""

    def of(self, dim: int, tags: Iterable[int]) -> list[PhysicalGroupMesh]:
        """The groups of dimension ``dim`` whose tag is in ``tags`` (present in the mesh)."""
        wanted = set(tags)
        return [g for g in self.groups if g.dim == dim and g.tag in wanted]

    def face(self, tags: Iterable[int]) -> PhysicalGroupMesh | None:
        """The union of the 2-D groups ``tags`` as one face (``None`` if none is in the mesh).

        Element counts, triangle counts and areas add up; the mean edge is
        the triangle-weighted mean; ``tag`` is -1.
        """
        found = self.of(2, tags)
        if not found:
            return None
        n_tri = sum(g.n_triangles for g in found)
        edge = (
            sum(g.mean_edge * g.n_triangles for g in found if g.n_triangles) / n_tri
            if n_tri
            else math.nan
        )
        return PhysicalGroupMesh(
            dim=2,
            tag=-1,
            n_elements=sum(g.n_elements for g in found),
            z_min=min(g.z_min for g in found),
            z_max=max(g.z_max for g in found),
            n_triangles=n_tri,
            mean_edge=edge,
            area=sum(g.area for g in found),
            nodes=frozenset().union(*(g.nodes for g in found)),
            normals=np.concatenate([g.normals for g in found]),
        )


def _physical_groups(path: Path, mesh: _Mesh, digest: str) -> MeshGroups:
    coords = mesh.coords
    if not coords:
        raise UnsupportedMeshFormatError(f"{path}: the mesh has no node")
    xyz = np.array(list(coords.values()), dtype=float)
    tol = _GEOM_REL_TOL * (float(np.max(np.ptp(xyz, axis=0))) or 1.0)
    out = []
    for (dim, tag), (count, nodes, tris) in sorted(mesh.group_reader().items()):
        z = np.array([coords[n][2] for n in nodes if n in coords], dtype=float)
        if z.size == 0:
            continue
        extra: dict[str, object] = {}
        if dim == 2:
            extra["nodes"] = frozenset(nodes)
        if tris:
            tri = np.array(
                [[coords[n] for n in t] for t in tris if all(n in coords for n in t)],
                dtype=float,
            ).reshape(-1, 3, 3)
            e = tri[:, [1, 2, 0], :] - tri
            cross = np.cross(e[:, 0], -e[:, 2])
            twice = np.linalg.norm(cross, axis=1)
            area = 0.5 * twice
            normals = cross / np.where(twice > 0.0, twice, 1.0)[:, None]
            normals.setflags(write=False)
            extra.update(
                n_triangles=len(tri),
                mean_edge=float(np.linalg.norm(e, axis=2).mean()),
                area=float(area.sum()),
                normals=normals,
            )
        out.append(
            PhysicalGroupMesh(dim, tag, count, float(z.min()), float(z.max()), **extra)  # type: ignore[arg-type]
        )
    return MeshGroups(
        path=path,
        z_min=float(xyz[:, 2].min()),
        z_max=float(xyz[:, 2].max()),
        groups=tuple(out),
        tol=tol,
        sha256=digest,
    )


# ---------------------------------------------------------------------------
# Content-hash cache (one parse serves both the check and the face measure)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_digest_memo: dict[tuple[str, int, int], str] = {}
_report_memo: dict[str, PeriodicFacetReport] = {}
_faces_memo: dict[str, ZFaceMesh] = {}
_groups_memo: dict[str, MeshGroups] = {}


def _digest_of(path: Path) -> tuple[str, bytes | None]:
    """sha256 of the file (memoised on path, size, mtime) and its bytes if read."""
    st = path.stat()
    memo_key = (str(path), st.st_size, st.st_mtime_ns)
    with _lock:
        digest = _digest_memo.get(memo_key)
    if digest is not None:
        return digest, None
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    with _lock:
        _digest_memo[memo_key] = digest
    return digest, data


def _write_mesh_text(path: Path, text: str) -> None:
    """Write a mesh file and memoise the digest of the bytes just written.

    Every writer of :mod:`mpxlite.cells` goes through here: the digest memo
    is keyed on ``(path, size, mtime_ns)``, and an in-place rewrite of the
    same size within one timestamp tick of the previous ``stat`` (coarse
    inode timestamps) would otherwise return the previous content's digest,
    hence its stale report. The entries of ``path`` are dropped and the
    fresh digest is stored under the new key.
    """
    data = text.encode("utf-8")
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    st = path.stat()
    name = str(path)
    with _lock:
        for k in [k for k in _digest_memo if k[0] == name]:
            del _digest_memo[k]
        _digest_memo[(name, st.st_size, st.st_mtime_ns)] = digest


def _analyse(
    path: Path, digest: str, data: bytes | None
) -> tuple[PeriodicFacetReport, ZFaceMesh, MeshGroups]:
    """Parse once; compute (and memoise) the facet report, z-face and group measures."""
    if data is None:
        data = path.read_bytes()
    mesh = _parse(path, data.decode("utf-8", errors="replace"))
    report = _check(path, mesh, digest)
    faces = _z_faces(path, mesh, digest)
    groups = _physical_groups(path, mesh, digest)
    with _lock:
        _report_memo[digest] = report
        _faces_memo[digest] = faces
        _groups_memo[digest] = groups
    return report, faces, groups


def _with_path[R: (PeriodicFacetReport, ZFaceMesh, MeshGroups)](found: R, path: Path) -> R:
    return found if found.path == path else dataclasses.replace(found, path=path)


def check_periodic_facet_order(
    msh_path: Path | str, *, use_cache: bool = True
) -> PeriodicFacetReport:
    """Check that the ``$Periodic`` maps of a mesh preserve the facet node order.

    Args:
        msh_path: Gmsh mesh, MSH 2.x or 4.1 ASCII.
        use_cache: Reuse the report of a mesh with the same bytes (sha256)
            checked (or measured by :func:`measure_z_faces`) before in this
            process (default). The hash itself is memoised on (path, size,
            mtime), so a repeated check of an unchanged file costs one
            ``stat``.

    Returns:
        The :class:`PeriodicFacetReport`; ``report.ok`` is the verdict
        (vacuously true for a mesh without periodic links: see
        ``report.has_periodic``).

    Raises:
        UnsupportedMeshFormatError: legacy MSH 1, binary files, MSH 3 / 4.0,
            missing sections.
        FileNotFoundError: if the file does not exist.
    """
    path = Path(msh_path).expanduser().resolve()
    digest, data = _digest_of(path)
    if use_cache:
        with _lock:
            found = _report_memo.get(digest)
        if found is not None:
            return _with_path(found, path)
    return _analyse(path, digest, data)[0]


def measure_z_faces(msh_path: Path | str, *, use_cache: bool = True) -> ZFaceMesh:
    """Mesh size of the bottom and top faces (``z = z_min``, ``z = z_max``) of a mesh.

    For a periodic box (Bloch faces lateral, z out of plane) these planes
    are its top and bottom faces (the outer faces of the box, or a
    boundary face that terminates the cell). The faces are found
    GEOMETRICALLY: every triangle of the mesh (any physical tag; elements
    duplicated across physicals counted once) whose three vertices lie in
    the plane of the lowest / highest node (within 1e-8 of the mesh
    extent). No tag knowledge is needed, so the measure applies to any
    mesh whatever its physical numbering; a face that
    is not exported as 2-D elements is reported with ``n_triangles = 0``.

    The measure shares the parse and the content-hash cache of
    :func:`check_periodic_facet_order` (one parse of a mesh serves both).

    Args:
        msh_path: Gmsh mesh, MSH 2.x or 4.1 ASCII.
        use_cache: Reuse the measure of a mesh with the same bytes (default).

    Returns:
        The :class:`ZFaceMesh` (``mean_edge``: the mean edge length over
        both faces; ``found``: both faces carry triangles).

    Raises:
        UnsupportedMeshFormatError: as :func:`check_periodic_facet_order`,
            or a mesh without nodes.
        FileNotFoundError: if the file does not exist.
    """
    path = Path(msh_path).expanduser().resolve()
    digest, data = _digest_of(path)
    if use_cache:
        with _lock:
            found = _faces_memo.get(digest)
        if found is not None:
            return _with_path(found, path)
    return _analyse(path, digest, data)[1]


def measure_physical_groups(msh_path: Path | str, *, use_cache: bool = True) -> MeshGroups:
    """Per-physical-group z-extent and triangle size of a mesh's 2-D / 3-D elements.

    The tag-aware counterpart of :func:`measure_z_faces`: for every
    (dimension, physical tag) of the surface and volume elements, the
    element count, the node height range and, for triangles, count, mean
    edge length and area. Combined with the role of each tag (the top and
    bottom faces of the box, the periodic faces, boundary surfaces), it
    measures the mesh size of the outer faces and their distance to the
    structure, and checks the box before a run.

    Shares the parse and the content-hash cache of
    :func:`check_periodic_facet_order` / :func:`measure_z_faces`.

    Args:
        msh_path: Gmsh mesh, MSH 2.x or 4.1 ASCII.
        use_cache: Reuse the measure of a mesh with the same bytes (default).

    Raises:
        UnsupportedMeshFormatError: as :func:`check_periodic_facet_order`,
            or a mesh without nodes.
        FileNotFoundError: if the file does not exist.
    """
    path = Path(msh_path).expanduser().resolve()
    digest, data = _digest_of(path)
    if use_cache:
        with _lock:
            found = _groups_memo.get(digest)
        if found is not None:
            return _with_path(found, path)
    return _analyse(path, digest, data)[2]


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------


def _lattice_uv(
    t1: tuple[float, float], t2: tuple[float, float]
) -> Callable[[float, float], tuple[float, float]]:
    """``(x, y) -> (u, v)`` with ``(x, y) = u T_1 + v T_2``."""
    (a, b), (c, d) = t1, t2
    det = a * d - b * c
    if not math.isfinite(det) or abs(det) <= 1e-12 * (math.hypot(a, b) * math.hypot(c, d)):
        raise ValueError(f"lattice vectors {t1}, {t2} are not independent")
    if b == 0.0:  # T_1 along x (the usual convention): the direct formula

        def uv(x: float, y: float) -> tuple[float, float]:
            v = y / d
            return (x - v * c) / a, v

        return uv

    def uv_general(x: float, y: float) -> tuple[float, float]:
        return (x * d - y * c) / det, (a * y - b * x) / det

    return uv_general


def _node_rows(lines: list[str], path: Path) -> tuple[int, list[list[str]]]:
    if "$NodeData" in lines:
        raise UnsupportedMeshFormatError(f"{path}: $NodeData sections cannot be renumbered")
    i_nodes = lines.index("$Nodes")
    n_nodes = int(lines[i_nodes + 1])
    return i_nodes, [lines[i_nodes + 2 + k].split() for k in range(n_nodes)]


def _periodic_lines(lines: list[str]) -> Iterator[int]:
    """Indices of the node-pair lines of the MSH 2.x ``$Periodic`` section."""
    if "$Periodic" not in lines:
        return
    j = lines.index("$Periodic") + 1
    n_links = int(lines[j])
    j += 1
    for _ in range(n_links):
        j += 2 if lines[j + 1].startswith("Affine") else 1
        count = int(lines[j])
        yield from range(j + 1, j + 1 + count)
        j += count + 1


def renumber_periodic(
    msh_path: Path | str,
    t1: tuple[float, float],
    t2: tuple[float, float],
    *,
    output: Path | str | None = None,
    origin: tuple[float, float] = (-0.5, -0.5),
    align: bool = True,
) -> PeriodicFacetReport:
    """Renumber the nodes of an MSH 2.x ASCII mesh so its periodic maps preserve node order.

    The new numbering sorts the nodes by a key invariant under the lattice
    translations ``T_1``, ``T_2`` (in-plane, the cell spanned by them in
    the (x, y) plane, z out of plane): the lattice coordinates ``(u, v)``
    of the node (``(x, y) = u T_1 + v T_2``) folded onto the reference cell
    ``[origin, origin + 1)`` (rounded to 1e-9), then z (rounded to 1e-9 of
    the lattice scale); images of one point tie on that key and are ordered
    by their unfolded coordinates. Every periodic map by ``T_1`` / ``T_2``
    then preserves the order of the node numbers, hence of every linked
    facet. The keys are exact (no neighbour search): two images of a point
    whose coordinate straddles a rounding boundary get keys one unit apart,
    harmless as long as distinct nodes are ``>> 1e-9`` of the lattice apart
    (no other node holds the adjacent key, so their order relative to every
    other node of a facet is still the same). The ``$Nodes``, ``$Elements``
    and ``$Periodic`` sections are rewritten (the MSH 2.2 writer of gmsh
    renumbers nodes sequentially whatever the in-memory tags, hence a
    rewrite of the file itself).

    Args:
        msh_path: The mesh (MSH 2.x ASCII).
        t1: First in-plane lattice vector ``(x, y)`` (m) of the Bloch maps,
            e.g. ``(t1_x, 0)``.
        t2: Second in-plane lattice vector, e.g. ``(t2_x, t2_y)``.
        output: Destination (default: rewrite ``msh_path`` in place).
        origin: Lattice coordinates of the reference-cell corner: ``(-1/2,
            -1/2)`` for a cell centred on ``x = y = 0``. Any value works; it only fixes which of the
            images is folded onto which.
        align: Also give every slave triangle the stored node order of its
            master (:func:`align_periodic_facets`; default). Renumbering
            alone never changes the stored orientation of a facet.

    Returns:
        The :func:`check_periodic_facet_order` report of the rewritten mesh.

    Raises:
        UnsupportedMeshFormatError: not MSH 2.x ASCII, or a ``$NodeData``
            section is present.
        ValueError: on dependent lattice vectors.
    """
    uv_of = _lattice_uv((float(t1[0]), float(t1[1])), (float(t2[0]), float(t2[1])))
    quantum = 1e-9 * max(abs(float(c)) for c in (*t1, *t2))

    def fold(w: float, o: float) -> int:
        return round((w - math.floor(w - o + 1e-9)) * 1e9)

    def key(x: float, y: float, z: float) -> tuple[int, int, int, float, float]:
        u, v = uv_of(x, y)
        return (fold(u, origin[0]), fold(v, origin[1]), round(z / quantum), u, v)

    report = _renumber_by(msh_path, key, output)
    if align and report.misoriented:
        report = align_periodic_facets(report.path)
    return report


def align_periodic_facets(
    msh_path: Path | str, *, output: Path | str | None = None
) -> PeriodicFacetReport:
    """Store every slave periodic triangle in the node order of its master (MSH 2.x ASCII).

    GetDP's ``Link`` constraint on ``FacetsOf`` maps the nodes of every
    slave facet IN THEIR STORED ORDER to the master side and requires the
    result to be the stored node list of the master facet; a mesh whose
    opposite periodic surfaces are oriented differently (common with the
    built-in geo kernel) stops GetDP with "7-Constraint Link: bad
    correspondance of facets" with second-order elements. This rewrites the node
    list of each such 3-node slave triangle (every occurrence of the
    element) as the pre-image of its master's: same nodes, same numbering,
    no other element touched, so the node-number ranks checked by
    :func:`check_periodic_facet_order` are unchanged. Curved triangles
    are left as they are (still reported).

    Args:
        msh_path: The mesh (MSH 2.x ASCII).
        output: Destination (default: rewrite ``msh_path`` in place).

    Returns:
        The :func:`check_periodic_facet_order` report of the rewritten mesh.

    Raises:
        UnsupportedMeshFormatError: not MSH 2.x ASCII.
    """
    path = Path(msh_path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    plain = text.splitlines()
    version, major = _read_format(path, plain, _sections(plain))
    if major != 2:
        raise _msh2_only(path, "align_periodic_facets rewrites", version)
    mesh = _parse(path, text)
    mapper = _Mapper(mesh)
    # slave entity -> {node set: node order to store}
    fixes: dict[int, dict[frozenset[int], tuple[int, int, int]]] = {}
    for link in (lk for lk in mesh.links if lk.dim == 2):
        masters = {frozenset(t): t for t in mesh.facets.get(link.master, [])}
        for tri in mesh.facets.get(link.slave, []):
            image = {mapper(link, n): n for n in tri}
            stored = masters.get(frozenset(image))
            if None in image or stored is None or len(image) != 3:
                continue
            fixes.setdefault(link.slave, {})[frozenset(tri)] = tuple(image[m] for m in stored)  # type: ignore[assignment]
    lines = text.split("\n")
    i_el = lines.index("$Elements")
    for k in range(int(lines[i_el + 1])):
        t = lines[i_el + 2 + k].split()
        if t[1] != "2":
            continue
        ntags = int(t[2])
        entity = int(t[4] if ntags >= 2 else t[3])
        head = 3 + ntags
        order = fixes.get(entity, {}).get(frozenset(int(n) for n in t[head:]))
        if order is not None:
            lines[i_el + 2 + k] = " ".join(t[:head] + [str(n) for n in order])
    target = Path(output).expanduser().resolve() if output is not None else path
    _write_mesh_text(target, "\n".join(lines))
    return check_periodic_facet_order(target)


def _renumber_by(
    msh_path: Path | str,
    key: Callable[[float, float, float], tuple],
    output: Path | str | None = None,
) -> PeriodicFacetReport:
    """Renumber the nodes of an MSH 2.x ASCII mesh by ascending ``key(x, y, z)``."""

    def order(xyz: np.ndarray) -> np.ndarray:
        keys = [key(float(x), float(y), float(z)) for x, y, z in xyz.tolist()]
        return np.array(sorted(range(len(keys)), key=keys.__getitem__), dtype=np.int64)

    return _renumber_rows(msh_path, order, output)


def _renumber_rows(
    msh_path: Path | str,
    order_of: Callable[[np.ndarray], np.ndarray],
    output: Path | str | None = None,
) -> PeriodicFacetReport:
    """Renumber the nodes of an MSH 2.x ASCII mesh: number ``k + 1`` to row ``order_of(xyz)[k]``.

    ``order_of`` receives the ``(n, 3)`` coordinates of the ``$Nodes`` rows
    (file order) and returns their new order (a permutation).
    """
    path = Path(msh_path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    plain = text.splitlines()
    version, major = _read_format(path, plain, _sections(plain))
    if major != 2:
        raise _msh2_only(path, "renumbering rewrites", version)

    i_nodes, rows = _node_rows(lines, path)
    xyz = np.array([[float(v) for v in row[1:4]] for row in rows], dtype=float).reshape(-1, 3)
    perm = np.asarray(order_of(xyz), dtype=np.int64)
    if not np.array_equal(np.sort(perm), np.arange(len(rows))):
        raise ValueError(f"{path}: the node order is not a permutation of the {len(rows)} nodes")
    rows = [rows[k] for k in perm.tolist()]
    new = {row[0]: str(k + 1) for k, row in enumerate(rows)}
    for k, row in enumerate(rows):
        lines[i_nodes + 2 + k] = " ".join([new[row[0]], *row[1:]])

    i_el = lines.index("$Elements")
    for k in range(int(lines[i_el + 1])):
        t = lines[i_el + 2 + k].split()
        head = 3 + int(t[2])
        lines[i_el + 2 + k] = " ".join(t[:head] + [new[n] for n in t[head:]])

    for k in _periodic_lines(lines):
        a, b = lines[k].split()
        lines[k] = f"{new[a]} {new[b]}"

    target = Path(output).expanduser().resolve() if output is not None else path
    _write_mesh_text(target, "\n".join(lines))
    return check_periodic_facet_order(target)
