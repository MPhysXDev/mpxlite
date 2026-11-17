# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""Conformity of a 3-D mesh: orphan nodes and detached boundary triangles.

A mesh assembled or edited outside gmsh (tiled, replicated, renumbered) can
carry triangles that are not a face of any tetrahedron (a periodic,
outer or tagged boundary face detached from the volume) or nodes that belong to no
volume element. GetDP then integrates boundary terms on a surface the
volume unknowns never see: no error, wrong physics. The periodic facet
check (:mod:`mpxlite.cells.periodic`) only compares the slave and master
faces with each other, so it does not see such faces.

:func:`check_mesh_topology` (MSH 2.x / 4.1 ASCII, ``O(elements)``,
content-hash cached) counts

* ``orphan_nodes``: nodes of no 3-D element;
* ``detached_triangles``: triangles (the vertex triple of any triangle
  type) that are not a face of a tetrahedron.

The face check runs on tetrahedral meshes only (other volume element
types: skipped, ``checked_faces = False``, the reason in
``faces_skipped``); a mesh without any 3-D element is not checked at all.
A solver runs it before every solve, refuses a mesh with a defect
(:class:`MeshTopologyError`) and warns when the face check was skipped
(:class:`MeshTopologyWarning`).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from mpxlite.cells.periodic import (
    _TRIANGLES,
    _VOLUMES,
    UnsupportedMeshFormatError,
    _digest_of,
    _read_format,
    _sections,
)

_TETS = frozenset({4, 11, 29, 30, 31})
"""Gmsh tetrahedron types (first 4 nodes are the vertices)."""


class MeshTopologyError(ValueError):
    """The mesh has orphan nodes or triangles detached from the volume."""


class MeshTopologyWarning(UserWarning):
    """The detached-face check of a mesh was skipped (non-tetrahedral volumes, no volume)."""


@dataclass(frozen=True)
class MeshTopologyReport:
    """Result of :func:`check_mesh_topology`."""

    path: Path
    sha256: str
    n_nodes: int
    n_volumes: int
    n_triangles: int
    orphan_nodes: tuple[int, ...]
    """Node tags of no 3-D element (at most the first 20)."""
    n_orphan_nodes: int
    detached_triangles: tuple[tuple[int, int, int], ...]
    """Vertex triples of triangles that are no tetrahedron face (at most the first 20)."""
    n_detached_triangles: int
    checked_faces: bool
    """False when the face check was skipped (non-tetrahedral volumes, or no volume)."""
    faces_skipped: str = ""
    """Why the face check was skipped (``""`` when it ran)."""

    @property
    def ok(self) -> bool:
        """No orphan node and no detached triangle."""
        return self.n_orphan_nodes == 0 and self.n_detached_triangles == 0

    def summary(self) -> str:
        """One-line description of the defects."""
        if self.ok:
            skipped = f"; detached-face check skipped: {self.faces_skipped}" * (
                not self.checked_faces
            )
            return (
                f"{self.path}: conforming ({self.n_volumes} volumes, {self.n_triangles} "
                f"triangles){skipped}"
            )
        return (
            f"{self.path}: {self.n_orphan_nodes} node(s) of no volume element "
            f"(e.g. {list(self.orphan_nodes[:5])}), {self.n_detached_triangles} of "
            f"{self.n_triangles} triangle(s) not a face of any tetrahedron "
            f"(e.g. {list(self.detached_triangles[:3])})"
        )


_lock = threading.Lock()
_memo: dict[str, MeshTopologyReport] = {}


def check_mesh_topology(msh_path: Path | str, *, use_cache: bool = True) -> MeshTopologyReport:
    """Orphan nodes and detached triangles of a mesh (see the module).

    Raises:
        UnsupportedMeshFormatError: not MSH 2.x / 4.1 ASCII, or no
            ``$Nodes`` / ``$Elements`` section.
        FileNotFoundError: if the file does not exist.
    """
    path = Path(msh_path).expanduser().resolve()
    digest, data = _digest_of(path)
    if use_cache:
        with _lock:
            found = _memo.get(digest)
        if found is not None:
            return found if found.path == path else _replace_path(found, path)
    if data is None:
        data = path.read_bytes()
    lines = data.decode("utf-8", errors="replace").splitlines()
    sections = _sections(lines)
    _, major = _read_format(path, lines, sections)
    for name in ("Nodes", "Elements"):
        if name not in sections:
            raise UnsupportedMeshFormatError(f"{path}: no ${name} section")
    nodes = _node_tags(lines, sections["Nodes"], v41=major == 4)
    elements = _elements(lines, sections["Elements"], v41=major == 4)
    faces: set[frozenset[int]] = set()
    used: set[int] = set()
    triangles: list[tuple[int, int, int]] = []
    other_volume: set[int] = set()
    n_vol = 0
    for etype, verts in elements:
        if etype in _VOLUMES:
            n_vol += 1
            used.update(verts)
            if etype in _TETS:
                a, b, c, d = verts[:4]
                faces.update(
                    (frozenset((a, b, c)), frozenset((a, b, d)), frozenset((a, c, d)),
                     frozenset((b, c, d)))
                )  # fmt: skip
            else:
                other_volume.add(etype)
        elif etype in _TRIANGLES:
            triangles.append((verts[0], verts[1], verts[2]))
    if n_vol == 0:  # not a volume mesh (surface-only test meshes): nothing to check
        orphans: list[int] = []
        detached: list[tuple[int, int, int]] = []
        skipped = "no volume element"
    else:
        orphans = [n for n in nodes if n not in used]
        detached = [] if other_volume else [t for t in triangles if frozenset(t) not in faces]
        skipped = (
            f"non-tetrahedral volume elements (gmsh types {sorted(other_volume)})"
            if other_volume
            else ""
        )
    report = MeshTopologyReport(
        path=path,
        sha256=digest,
        n_nodes=len(nodes),
        n_volumes=n_vol,
        n_triangles=len(triangles),
        orphan_nodes=tuple(orphans[:20]),
        n_orphan_nodes=len(orphans),
        detached_triangles=tuple(detached[:20]),
        n_detached_triangles=len(detached),
        checked_faces=not skipped,
        faces_skipped=skipped,
    )
    with _lock:
        _memo[digest] = report
    return report


def _replace_path(found: MeshTopologyReport, path: Path) -> MeshTopologyReport:
    import dataclasses

    return dataclasses.replace(found, path=path)


def _node_tags(lines: list[str], start: int, *, v41: bool) -> list[int]:
    if not v41:
        return [
            int(ln.split(maxsplit=1)[0]) for ln in lines[start + 1 : start + 1 + int(lines[start])]
        ]
    out: list[int] = []
    j = start + 1
    for _ in range(int(lines[start].split()[0])):
        count = int(lines[j].split()[3])
        out.extend(int(t) for t in lines[j + 1 : j + 1 + count])
        j += 2 * count + 1
    return out


def _elements(lines: list[str], start: int, *, v41: bool) -> list[tuple[int, list[int]]]:
    out: list[tuple[int, list[int]]] = []
    if not v41:
        for ln in lines[start + 1 : start + 1 + int(lines[start])]:
            t = ln.split()
            etype, ntags = int(t[1]), int(t[2])
            if etype in _VOLUMES or etype in _TRIANGLES:
                out.append((etype, [int(n) for n in t[3 + ntags :]]))
        return out
    j = start + 1
    for _ in range(int(lines[start].split()[0])):
        _, _, etype, count = (int(v) for v in lines[j].split()[:4])
        body = lines[j + 1 : j + 1 + count]
        j += count + 1
        if etype in _VOLUMES or etype in _TRIANGLES:
            out.extend((etype, [int(v) for v in ln.split()[1:]]) for ln in body)
    return out
