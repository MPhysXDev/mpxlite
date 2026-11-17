# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

r"""Exactly C4v-symmetric meshes of a square periodic cell by wedge replication.

A point-group reduction of a periodic problem holds at round-off only if
the *discrete* problem is invariant, i.e. if the
mesh itself is mapped onto itself by the group. A mesher does not guarantee
that, so the symmetric mesh is built from one fundamental domain of the
space group:

* the caller meshes the wedge

  .. math::
     T_0 = \{(u, v) = (x, y) - c : 0 \le v \le u \le L/2\} \times [z_0, z_1]

  (``c`` the symmetry centre, ``L`` the side of the square cell), conforming
  along the planes where the images of the window faces cross the wedge
  (``u = L/4`` and ``v = L/4`` for the window ``[-L/4, 3L/4]^2`` with
  ``c = (L/2, L/2)``), and labels those faces :data:`CUT`; the wedge faces
  themselves become interior and are not returned (the toolkit of the Gmsh
  jobs collects the blocks of a meshed Gmsh entity);
* :func:`replicate_c4v` maps the wedge by the 8 operations of
  :data:`C4V` about ``c``, translates every element by a lattice vector into
  the window (by its centroid; no element may straddle a window face),
  swaps two nodes of every mirrored element (positive volumes, outward
  normals kept), turns the ``"cut"`` faces on the upper window faces into
  the Bloch faces ``periodic_xp`` / ``periodic_yp`` and their translates by
  ``-L`` into ``periodic_xm`` / ``periodic_ym``, and merges coincident
  nodes (tolerance ``1e-9 L``); a ``relabel`` hook splits replicated
  blocks by position (e.g. one tag per copy of a surface);
* :func:`write_msh22_periodic` writes MSH 2.2 ASCII with a ``$Periodic``
  section (affine translation by ``L`` along x and y);
* :func:`build_c4v_mesh` chains the two, then checks the replicated mesh
  (:func:`~mpxlite.cells.topology.check_mesh_topology`: no orphan node, no
  detached triangle; extent ``<= L``) and renumbers it to the periodic node
  order required by second-order elements
  (:func:`~mpxlite.cells.periodic.renumber_periodic`), or, with
  ``numbering=``, to a numbering that every operation of a group also
  preserves (:func:`~mpxlite.cells.spacegroup.renumber_symmetric`;
  :func:`c4v_lattice_ops` gives the 8 operations of this construction).

Any plane group (:mod:`mpxlite.cells.autosym`):
:func:`replicate_group` generalises :func:`replicate_c4v` to the operations
of any group (:class:`~mpxlite.cells.spacegroup.LatticeOp`, any lattice,
kinds mapped per operation: the tag permutation),
:func:`split_orbit_edges` bisects the edges that join two nodes of one
orbit, :func:`bloch_faces` extracts the Bloch faces from the boundary of
the tetrahedra, :func:`merge_points` merges coincident nodes (vectorised)
and :func:`write_msh22_periodic` takes any lattice vectors.

The result is invariant under C4v about ``c`` modulo the lattice to
round-off, whatever the size field of the wedge mesh. The second-order
H(curl) basis of GetDP depends on the global node numbers (edge
orientations, facet functions): with the periodic numbering the discrete
problem is symmetric only up to the discretisation error; with
``numbering=`` the operations of the group map the basis onto itself and it
is symmetric to round-off.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from mpxlite.cells.periodic import _write_mesh_text, check_periodic_facet_order, renumber_periodic
from mpxlite.cells.spacegroup import LatticeOp, renumber_symmetric
from mpxlite.cells.topology import MeshTopologyError, check_mesh_topology

C4V: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((1, 0), (0, 1)),
    ((0, -1), (1, 0)),
    ((-1, 0), (0, -1)),
    ((0, 1), (-1, 0)),
    ((-1, 0), (0, 1)),
    ((1, 0), (0, -1)),
    ((0, 1), (1, 0)),
    ((0, -1), (-1, 0)),
)
"""The 8 in-plane matrices of C4v: E, C4, C2, C4^3, m_x, m_y, m_d, m_d'."""

C4V_NAMES = ("E", "C4", "C2", "C4^3", "mx", "my", "md", "md'")
"""Names of the :data:`C4V` matrices."""

CUT = "cut"
"""Block kind of the wedge faces that lie on the images of the window faces."""

PERIODIC_NAMES = ("periodic_xm", "periodic_xp", "periodic_ym", "periodic_yp")
"""Physical-group names of the Bloch faces written by :func:`replicate_c4v`."""

Block = tuple[str, int, np.ndarray]
"""``(kind, gmsh element type, connectivity (k, nn) into the node array)``."""

Relabel = Callable[[str, np.ndarray], "np.ndarray | None"]
"""``relabel(kind, xyz)``: per-element kinds ``(k,)``, ``xyz`` ``(k, nn, 3)``; ``None``: keep."""


def replicate_c4v(
    coords: np.ndarray,
    blocks: Sequence[Block],
    cell: float,
    *,
    centre: Sequence[float] | None = None,
    window_hi: float | None = None,
    relabel: Relabel | None = None,
) -> tuple[np.ndarray, list[Block]]:
    """The 8 C4v images of a wedge mesh, folded into the window, nodes merged.

    Args:
        coords: ``(n, 3)`` wedge node coordinates [m].
        blocks: Wedge element blocks; kind :data:`CUT` marks the faces on
            the images of the window faces.
        cell: Side ``L`` of the square cell (lattice ``(L, 0), (0, L)``).
        centre: Symmetry centre ``c`` (x, y); default ``(L/2, L/2)``.
        window_hi: Upper window face coordinate (same along x and y):
            elements whose centroid lies above it are translated by ``-L``;
            default ``3L/4`` (window ``[-L/4, 3L/4]^2``).
        relabel: Optional split of a replicated, folded block by position.

    Returns:
        ``(nodes, blocks)``: merged node coordinates and the replicated
        blocks (Bloch faces as :data:`PERIODIC_NAMES`).
    """
    big_l = float(cell)
    cxy = (big_l / 2, big_l / 2) if centre is None else tuple(float(v) for v in centre)
    c = np.array([cxy[0], cxy[1], 0.0])
    hi = 0.75 * big_l if window_hi is None else float(window_hi)
    eps = 1e-9 * big_l
    pts: list[np.ndarray] = []
    elems: list[Block] = []
    n0 = 0

    def emit(kind: str, etype: int, xyz: np.ndarray) -> None:
        nonlocal n0
        k, nn = xyz.shape[:2]
        pts.append(xyz.reshape(-1, 3))
        elems.append((kind, etype, n0 + np.arange(k * nn).reshape(k, nn)))
        n0 += k * nn

    for gm in C4V:
        g3 = np.eye(3)
        g3[:2, :2] = np.array(gm, dtype=float)
        img = c + (coords - c) @ g3.T
        mirror = np.linalg.det(g3) < 0
        for kind, etype, conn in blocks:
            if mirror:
                conn = conn[:, [1, 0, *range(2, conn.shape[1])]]
            xyz = img[conn]  # (k, nn, 3)
            cen = xyz.mean(axis=1)
            if kind == CUT:
                for axis, name in ((0, "x"), (1, "y")):
                    on = np.all(np.abs(xyz[:, :, axis] - hi) < eps, axis=1)
                    if np.any(on):
                        face = xyz[on].copy()
                        # wrap along the other axis into the window, as the tets
                        other = 1 - axis
                        wrap = face[:, :, other].mean(axis=1) > hi
                        face[wrap, :, other] -= big_l
                        emit(f"periodic_{name}p", etype, face)
                        lo = face.copy()
                        lo[:, :, axis] -= big_l
                        emit(f"periodic_{name}m", etype, lo)
                continue
            shift = np.zeros_like(cen)
            shift[:, 0] = np.where(cen[:, 0] > hi, -big_l, 0.0)
            shift[:, 1] = np.where(cen[:, 1] > hi, -big_l, 0.0)
            xyz = xyz + shift[:, None, :]
            names = None if relabel is None else relabel(kind, xyz)
            if names is None:
                emit(kind, etype, xyz)
                continue
            names = np.asarray(names)
            for name in sorted(set(names.tolist())):  # deterministic block order
                emit(str(name), etype, xyz[names == name])
    allp = np.concatenate(pts)
    grid = PointGrid(eps)
    rep = np.array([grid.add(x, i) for i, x in enumerate(allp.tolist())])
    uniq, inverse = np.unique(rep, return_inverse=True)
    nodes = allp[uniq]
    return nodes, [(k, t, inverse[cn]) for k, t, cn in elems]


class PointGrid:
    """Points merged within ``tol`` (bucket hash, neighbour buckets searched)."""

    def __init__(self, tol: float) -> None:
        self.tol = tol
        self.size = 4.0 * tol
        self.buckets: dict[tuple[int, int, int], list[tuple[tuple[float, ...], int]]] = {}
        self.count = 0

    def _key(self, x: list[float]) -> tuple[int, int, int]:
        return tuple(int(np.floor(c / self.size)) for c in x)  # type: ignore[return-value]

    def find(self, x: list[float]) -> int | None:
        """Index of a stored point equal to ``x`` within ``tol``, else ``None``."""
        k = self._key(x)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for y, idx in self.buckets.get((k[0] + dx, k[1] + dy, k[2] + dz), ()):
                        if max(abs(a - b) for a, b in zip(x, y, strict=True)) <= self.tol:
                            return idx
        return None

    def add(self, x: list[float], idx: int | None = None) -> int:
        """Index of the point equal to ``x`` within ``tol`` (``x`` added when new)."""
        hit = self.find(x)
        if hit is not None:
            return hit
        if idx is None:
            idx = self.count
        self.count += 1
        self.buckets.setdefault(self._key(x), []).append((tuple(x), idx))
        return idx


def write_msh22_periodic(
    path: Path | str,
    nodes: np.ndarray,
    blocks: Sequence[Block],
    physical: Mapping[str, tuple[int, int]],
    cell: float,
    *,
    periodic: Sequence[tuple[str, str, Sequence[float]]] | None = None,
) -> Path:
    """Write MSH 2.2 ASCII with the ``$Periodic`` maps ``x_p = x_m + L``, ``y_p = y_m + L``.

    ``physical``: ``name -> (dim, tag)`` of every block kind (and of the
    :data:`PERIODIC_NAMES`); only the groups with elements are written to
    ``$PhysicalNames``. Elements are written by decreasing dimension.
    ``periodic``: the links ``(slave, master, vector)`` (``x_slave = x_master
    + vector``, m), default the two of :data:`PERIODIC_NAMES` on the square
    lattice of side ``cell`` (any lattice vectors otherwise).

    Raises:
        RuntimeError: if a slave Bloch-face node has no master image.
    """
    path = Path(path)
    big_l = float(cell)
    if periodic is None:
        periodic = (
            ("periodic_xp", "periodic_xm", (big_l, 0.0)),
            ("periodic_yp", "periodic_ym", (0.0, big_l)),
        )
    links = [(sl, ma, np.array([*np.asarray(v, dtype=float)[:2], 0.0])) for sl, ma, v in periodic]
    linked = {n for sl, ma, _ in links for n in (sl, ma)}
    used = {kind for kind, _, conn in blocks if len(conn)}
    named = [(n, dt) for n, dt in physical.items() if n in used]
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(named))]
    lines += [f'{d} {t} "{n}"' for n, (d, t) in named]
    lines += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    lines += [f"{i + 1} {x!r} {y!r} {z!r}" for i, (x, y, z) in enumerate(nodes.tolist())]
    lines.append("$EndNodes")
    rows = []
    face_nodes: dict[str, set[int]] = {}
    for kind, etype, conn in sorted(blocks, key=lambda b: -physical[b[0]][0]):
        tag = physical[kind][1]
        for row in conn.tolist():
            rows.append(f"{etype} 2 {tag} {tag} " + " ".join(str(n + 1) for n in row))
        if kind in linked:
            face_nodes.setdefault(kind, set()).update(int(n) for n in conn.ravel())
    lines += ["$Elements", str(len(rows))]
    lines += [f"{i + 1} {r}" for i, r in enumerate(rows)]
    lines += ["$EndElements", "$Periodic", str(len(links))]
    eps = 1e-9 * max(float(np.linalg.norm(v)) for _, _, v in links)
    for slave, master, vec in links:
        s_ids = sorted(face_nodes[slave])
        m_ids = np.array(sorted(face_nodes[master]))
        shifted = nodes[m_ids] + vec
        grid = PointGrid(eps)
        for k, x in enumerate(shifted.tolist()):
            grid.add(x, k)
        j = [grid.find(x) for x in nodes[s_ids].tolist()]
        if any(k is None for k in j) or len(s_ids) != len(m_ids):
            bad = sum(k is None for k in j)
            raise RuntimeError(f"{slave}: no periodic match ({bad} of {len(s_ids)}, {len(m_ids)})")
        aff = np.eye(4)
        aff[:3, 3] = vec
        lines.append(f"2 {physical[slave][1]} {physical[master][1]}")
        lines.append("Affine " + " ".join(repr(float(v)) for v in aff.ravel()))
        lines.append(str(len(s_ids)))
        lines += [f"{s + 1} {int(m_ids[k]) + 1}" for s, k in zip(s_ids, j, strict=True)]
    lines += ["$EndPeriodic", ""]
    _write_mesh_text(path, "\n".join(lines))
    return path


def build_c4v_mesh(
    output_path: Path | str,
    coords: np.ndarray,
    blocks: Sequence[Block],
    cell: float,
    physical: Mapping[str, tuple[int, int]],
    *,
    centre: Sequence[float] | None = None,
    window_hi: float | None = None,
    relabel: Relabel | None = None,
    origin: tuple[float, float] = (-0.25, -0.25),
    numbering: Sequence[LatticeOp] | None = None,
) -> Path:
    """Replicate a wedge mesh (:func:`replicate_c4v`), write it, check it, renumber it.

    ``origin``: window origin in lattice coordinates for
    :func:`~mpxlite.cells.periodic.renumber_periodic` (default the
    ``[-L/4, 3L/4]^2`` window). ``numbering``: operations (in the lattice
    ``(L, 0), (0, L)``) whose group must preserve the node order, e.g.
    :func:`c4v_lattice_ops` or a subgroup (the rotations); the mesh is then
    renumbered by :func:`~mpxlite.cells.spacegroup.renumber_symmetric`
    (also periodic-consistent). Default ``None``: the periodic numbering
    only (unchanged behaviour). Returns the absolute path.

    Raises:
        MeshTopologyError: orphan nodes, detached triangles, or an extent
            beyond ``L`` (an element folded the wrong way).
        RuntimeError: periodic matching or facet-order repair failed.
        mpxlite.cells.spacegroup.SymmetricNumberingError: no numbering
            preserves the order under ``numbering`` (not raised by the wedge
            construction about its own centre).
    """
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    big_l = float(cell)
    nodes, elems = replicate_c4v(
        coords, blocks, big_l, centre=centre, window_hi=window_hi, relabel=relabel
    )
    ext = np.ptp(nodes[:, :2], axis=0)
    if np.any(ext > big_l * (1 + 1e-9)):
        raise MeshTopologyError(f"{out}: replicated mesh extent {ext.tolist()} exceeds L = {big_l}")
    write_msh22_periodic(out, nodes, elems, physical, big_l)
    topo = check_mesh_topology(out)
    if not topo.ok:
        raise MeshTopologyError(f"non-conforming replicated mesh: {topo.summary()}")
    if numbering is not None:
        report = renumber_symmetric(out, ((big_l, 0.0), (0.0, big_l)), numbering)
        if not report.ok:
            raise RuntimeError(f"periodic facet repair failed: {report.summary()}")
    elif not check_periodic_facet_order(out).ok:
        report = renumber_periodic(out, (big_l, 0.0), (0.0, big_l), origin=origin)
        if not report.ok:
            raise RuntimeError(f"periodic facet repair failed: {report.summary()}")
    return out


def c4v_lattice_ops(
    cell: float, centre: Sequence[float] | None = None, names: Sequence[str] | None = None
) -> tuple[LatticeOp, ...]:
    """The operations of :data:`C4V` about ``centre`` (default ``(L/2, L/2)``) as lattice maps.

    Lattice ``(L, 0), (0, L)``; ``names``: a subset of :data:`C4V_NAMES`
    (default all), e.g. ``("C2",)`` or ``("C4",)`` for
    :func:`build_c4v_mesh` ``numbering=`` (the group they generate is used).
    """
    big_l = float(cell)
    c = (big_l / 2, big_l / 2) if centre is None else tuple(float(v) for v in centre)
    lat = ((big_l, 0.0), (0.0, big_l))
    want = C4V_NAMES if names is None else tuple(names)
    unknown = sorted(set(want) - set(C4V_NAMES))
    if unknown:
        raise ValueError(f"unknown C4v operations {unknown} (known: {C4V_NAMES})")
    return tuple(
        LatticeOp.from_cartesian(n, g, lat, centre=c)
        for n, g in zip(C4V_NAMES, C4V, strict=True)
        if n in want
    )


def c4v_images(points: np.ndarray, cell: float, centre: Sequence[float] | None = None,
               window_lo: float | None = None) -> list[np.ndarray]:  # fmt: skip
    """The 8 images of ``points`` ``(n, 2|3)`` under C4v about ``centre``, folded into the window.

    Folding is modulo ``L`` into ``[window_lo, window_lo + L)`` (default
    ``-L/4``). A mesh is C4v-invariant iff each image of its node set is
    the node set itself (up to a tolerance).
    """
    big_l = float(cell)
    p = np.asarray(points, dtype=float)
    c = np.array((big_l / 2, big_l / 2) if centre is None else tuple(centre), dtype=float)
    lo = -0.25 * big_l if window_lo is None else float(window_lo)
    out = []
    for gm in C4V:
        q = p.copy()
        q[:, :2] = c + (p[:, :2] - c) @ np.array(gm, dtype=float).T
        q[:, :2] = lo + np.mod(q[:, :2] - lo + 1e-9 * big_l, big_l) - 1e-9 * big_l
        out.append(q)
    return out


# ---------------------------------------------------------------------------
# Any plane group: replication of an asymmetric unit (mpxlite.cells.autosym)
# ---------------------------------------------------------------------------

_SPLIT = {4: ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), 2: ((0, 1), (0, 2), (1, 2))}
"""Edges (local node pairs) of a tetrahedron (4) and a triangle (2)."""


def merge_points(points: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Merge the points ``(n, 3)`` closer than ``tol`` (max norm): ``(unique, inverse)``.

    Vectorised bucket hash (buckets of ``1000 tol``); the points near a
    bucket face are matched against the neighbouring buckets. Distinct points
    must be farther apart than the bucket size.

    Raises:
        RuntimeError: two points of one bucket farther apart than ``tol``.
    """
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    h = 1000.0 * tol
    q = np.floor(p / h).astype(np.int64)
    _, first, group = np.unique(q, axis=0, return_index=True, return_inverse=True)
    group = group.ravel()
    if np.max(np.abs(p - p[first[group]])) > tol:
        raise RuntimeError("merge_points: distinct points closer than the bucket size")
    frac = p / h - q
    near = np.where(np.any((frac < tol / h) | (frac > 1 - tol / h), axis=1))[0]
    if len(near):
        parent = np.arange(len(first))

        def root(k: int) -> int:
            while parent[k] != k:
                parent[k] = parent[parent[k]]
                k = parent[k]
            return k

        bucket = {tuple(k): g for g, k in enumerate(q[first].tolist())}
        for i in near.tolist():
            lo = np.where(frac[i] < tol / h, -1, 0)
            hi = np.where(frac[i] > 1 - tol / h, 1, 0)
            for d in np.ndindex(*(hi - lo + 1)):
                key = tuple((q[i] + lo + np.array(d)).tolist())
                g = bucket.get(key)
                if g is not None and np.max(np.abs(p[first[g]] - p[i])) <= tol:
                    a, b = root(g), root(int(group[i]))
                    if a != b:
                        parent[max(a, b)] = min(a, b)
        roots = np.array([root(k) for k in range(len(first))])
        group = roots[group]
    uniq, inverse = np.unique(group, return_inverse=True)
    del uniq
    rep = np.zeros(inverse.max() + 1, dtype=np.int64)
    rep[inverse[::-1]] = np.arange(len(p))[::-1]  # first point of every class
    return p[rep], inverse.ravel()


def _orbit_pairs(
    coords: np.ndarray, a: np.ndarray, b: np.ndarray, lattice: np.ndarray,
    ops: Sequence[LatticeOp], tol: float,
) -> np.ndarray:  # fmt: skip
    """Mask of the pairs ``(a, b)`` of distinct nodes in one orbit (an op maps ``a`` to ``b``)."""
    tinv = np.linalg.inv(lattice)
    ua = coords[a, :2] @ tinv
    ub = coords[b, :2] @ tinv
    same_z = np.abs(coords[a, 2] - coords[b, 2]) <= tol
    distinct = np.max(np.abs(coords[a] - coords[b]), axis=1) > tol
    bad = np.zeros(len(a), dtype=bool)
    for op in ops:
        d = op.apply(ua) - ub
        d -= np.rint(d)
        bad |= np.max(np.abs(d @ lattice), axis=1) <= tol
    return bad & same_z & distinct


def split_orbit_edges(
    coords: np.ndarray,
    blocks: Sequence[Block],
    lattice: object,
    ops: Sequence[LatticeOp],
    *,
    tol: float | None = None,
    interior: Callable[[np.ndarray], np.ndarray] | None = None,
    max_rounds: int = 20,
) -> tuple[np.ndarray, list[Block], int]:
    """Bisect every edge joining two nodes of one orbit (``renumber_symmetric``'s condition).

    An element holds two nodes of one orbit of the group (and of the
    lattice) exactly when one of its edges joins them; no numbering can
    then preserve its node order (:mod:`mpxlite.cells.spacegroup`). On a
    unit bounded by mirrors it never happens; on a unit whose edges are
    paired by a rotation it does at the rotation axis where the paired
    faces meet (``p4``, ``p4g``: an edge from ``(h, 0)`` to ``(0, h)`` about
    the axis). Every such edge is split at its midpoint (all tetrahedra and
    triangles holding it), in rounds of edges no element shares, until none
    is left. The midpoint of an edge between two faces of the unit is
    interior (its orbit meets the unit once); ``interior(points (k, 3)) ->
    (k,) bool`` refuses a midpoint on the boundary of the unit (an edge lying
    in a paired face: bisecting it would break the matching).

    Returns:
        ``(coords, blocks, n_split)``.

    Raises:
        RuntimeError: an offending edge on the boundary of the unit, or still
            offending edges after ``max_rounds``.
    """
    t = np.asarray(lattice, dtype=float).reshape(2, 2)
    tol = 1e-9 * float(np.max(np.abs(t))) if tol is None else float(tol)
    xyz = np.array(coords, dtype=float).reshape(-1, 3)
    out = [(k, e, np.asarray(c, dtype=np.int64)) for k, e, c in blocks]
    total = 0
    for _ in range(max_rounds):
        pairs = np.concatenate(
            [np.sort(c[:, list(p)], axis=1) for _, e, c in out for p in _SPLIT[e]]
        ) if out else np.zeros((0, 2), dtype=np.int64)  # fmt: skip
        pairs = np.unique(pairs, axis=0)
        bad = pairs[_orbit_pairs(xyz, pairs[:, 0], pairs[:, 1], t, ops, tol)]
        if not len(bad):
            return xyz, out, total
        n = len(xyz)
        holders = _edge_holders(out, bad[:, 0] * n + bad[:, 1], n)
        taken: set[tuple[int, int]] = set()
        chosen = []  # edges no element shares
        for k in (bad[:, 0] * n + bad[:, 1]).tolist():
            if not any(x in taken for x in holders[k]):
                taken.update(holders[k])
                chosen.append(k)
        mids = np.array([(xyz[k // n] + xyz[k % n]) / 2 for k in chosen])
        if interior is not None and not np.all(interior(mids)):
            raise RuntimeError(
                "an edge joining two nodes of one orbit lies on the boundary of the unit: "
                "bisecting it would break the matching of the paired faces"
            )
        xyz = np.vstack([xyz, mids])
        out = _bisect(out, holders, chosen, n)
        total += len(chosen)
    raise RuntimeError(f"split_orbit_edges: offending edges left after {max_rounds} rounds")


def _edge_holders(
    blocks: Sequence[Block], keys: np.ndarray, n: int
) -> dict[int, list[tuple[int, int]]]:
    """Edge key ``a n + b`` -> the ``(block, row)`` of the elements holding it."""
    holders: dict[int, list[tuple[int, int]]] = {}
    for bi, (_, e, c) in enumerate(blocks):
        for p in _SPLIT[e]:
            kk = np.sort(c[:, list(p)], axis=1)
            key = kk[:, 0] * n + kk[:, 1]
            for r in np.where(np.isin(key, keys))[0].tolist():
                holders.setdefault(int(key[r]), []).append((bi, r))
    return holders


def _bisect(
    blocks: Sequence[Block], holders: Mapping[int, list[tuple[int, int]]], chosen: Sequence[int],
    n: int,
) -> list[Block]:  # fmt: skip
    """Split every element holding a chosen edge at its midpoint (node ``n + j``)."""
    split: dict[int, dict[int, tuple[int, int, int]]] = {}
    for j, k in enumerate(chosen):
        for bi, r in holders[k]:
            split.setdefault(bi, {})[r] = (k // n, k % n, n + j)
    out = []
    for bi, (kind, e, c) in enumerate(blocks):
        rows = split.get(bi)
        if not rows:
            out.append((kind, e, c))
            continue
        keep = np.ones(len(c), dtype=bool)
        extra = []
        for r, (a, b, m) in rows.items():
            keep[r] = False
            for old in (a, b):
                row = c[r].copy()
                row[row == old] = m
                extra.append(row)
        out.append((kind, e, np.vstack([c[keep], np.array(extra, dtype=np.int64)])))
    return out


def replicate_group(
    coords: np.ndarray,
    blocks: Sequence[Block],
    lattice: object,
    ops: Sequence[LatticeOp],
    *,
    window_lo: Sequence[float] = (0.0, 0.0),
    kind_maps: Sequence[Mapping[str, str]] | None = None,
    tol: float | None = None,
) -> tuple[np.ndarray, list[Block]]:
    """The images of a unit mesh under ``ops``, folded into the window, nodes merged.

    Generalises :func:`replicate_c4v` to any group of :class:`LatticeOp`
    (lattice rows ``T_1``, ``T_2``): every element of every image is
    translated by the lattice vector that brings its centroid into the
    window ``[lo, lo + 1)`` (lattice coordinates; no element may straddle a
    window face), the elements of a reflected image get two nodes swapped
    (positive volumes), the kind of a block is mapped by ``kind_maps[k]``
    for the operation ``k`` (the tag permutation: surfaces split across
    the unit get the tag of their image), coincident nodes are merged
    (:func:`merge_points`, ``tol`` default ``1e-9`` of the lattice scale)
    and repeated triangles of one kind (a tagged surface on a mirror) are
    dropped.

    Raises:
        RuntimeError: two coincident triangles of different kinds.
    """
    t = np.asarray(lattice, dtype=float).reshape(2, 2)
    tinv = np.linalg.inv(t)
    lo = np.asarray(window_lo, dtype=float).reshape(2)
    tol = 1e-9 * float(np.max(np.abs(t))) if tol is None else float(tol)
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    pts: list[np.ndarray] = []
    elems: list[tuple[str, int, np.ndarray]] = []
    n0 = 0
    for k, op in enumerate(ops):
        g, tr = op.cartesian(t)
        img = xyz.copy()
        img[:, :2] = xyz[:, :2] @ g.T + tr
        kmap = {} if kind_maps is None else kind_maps[k]
        keys, conns = [], []
        for kind, etype, conn in blocks:
            c = np.asarray(conn, dtype=np.int64)
            if op.det < 0:
                c = c[:, [1, 0, *range(2, c.shape[1])]]
            u = img[c][..., :2] @ tinv
            d = np.floor(u.mean(axis=1) - lo + 1e-9).astype(np.int64)
            keys.append(np.column_stack([c.ravel(), np.repeat(d, c.shape[1], axis=0)]))
            conns.append((kmap.get(kind, kind), etype, c.shape))
        allk = np.concatenate(keys)
        uk, inv = np.unique(allk, axis=0, return_inverse=True)
        inv = inv.ravel()
        pts.append(img[uk[:, 0]] - np.column_stack([uk[:, 1:] @ t, np.zeros(len(uk))]))
        pos = 0
        for kind, etype, shape in conns:
            m = shape[0] * shape[1]
            elems.append((kind, etype, n0 + inv[pos : pos + m].reshape(shape)))
            pos += m
        n0 += len(uk)
    nodes, inverse = merge_points(np.concatenate(pts), tol)
    merged: dict[tuple[str, int], list[np.ndarray]] = {}
    for kind, etype, conn in elems:
        merged.setdefault((kind, etype), []).append(inverse[conn])
    out: list[Block] = []
    seen_tri: dict[tuple[int, ...], str] = {}
    for (kind, etype), parts in merged.items():
        conn = np.concatenate(parts)
        if etype == 2:
            key = np.sort(conn, axis=1)
            _, first = np.unique(key, axis=0, return_index=True)
            conn = conn[np.sort(first)]
            for row in np.sort(conn, axis=1).tolist():
                other = seen_tri.setdefault(tuple(row), kind)
                if other != kind:
                    raise RuntimeError(f"coincident triangles of {other!r} and {kind!r}")
        out.append((kind, etype, conn))
    return nodes, out


def bloch_faces(
    nodes: np.ndarray,
    blocks: Sequence[Block],
    lattice: object,
    *,
    window_lo: Sequence[float] = (0.0, 0.0),
    names: Sequence[str] = PERIODIC_NAMES,
    tol: float | None = None,
) -> list[Block]:
    """The boundary triangles of the tetrahedra on the window faces, as Bloch-face blocks.

    ``names``: the faces ``u_1 = lo_1``, ``u_1 = lo_1 + 1``, ``u_2 = lo_2``,
    ``u_2 = lo_2 + 1`` (default :data:`PERIODIC_NAMES`); triangles outward.
    """
    t = np.asarray(lattice, dtype=float).reshape(2, 2)
    lo = np.asarray(window_lo, dtype=float).reshape(2)
    tol = 1e-9 * float(np.max(np.abs(t))) if tol is None else float(tol)
    tets = np.concatenate([c for _, e, c in blocks if e == 4])
    faces = np.concatenate([tets[:, [1, 2, 3]], tets[:, [0, 3, 2]], tets[:, [0, 1, 3]],
                            tets[:, [0, 2, 1]]])  # fmt: skip
    key = np.sort(faces, axis=1)
    _, first, inv, cnt = np.unique(
        key, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    bnd = faces[first[cnt == 1]]
    u = nodes[:, :2] @ np.linalg.inv(t)
    out: list[Block] = []
    utol = tol / float(np.min(np.linalg.norm(t, axis=1)))
    for k, name in enumerate(names):
        axis, side = divmod(k, 2)
        level = lo[axis] + side
        on = np.all(np.abs(u[bnd][:, :, axis] - level) <= utol, axis=1)
        out.append((name, 2, bnd[on]))
    del inv
    return out


__all__ = [
    "C4V",
    "C4V_NAMES",
    "CUT",
    "PERIODIC_NAMES",
    "PointGrid",
    "bloch_faces",
    "build_c4v_mesh",
    "c4v_images",
    "c4v_lattice_ops",
    "merge_points",
    "replicate_c4v",
    "replicate_group",
    "split_orbit_edges",
    "write_msh22_periodic",
]
