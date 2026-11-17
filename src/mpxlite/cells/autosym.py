# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

r"""Symmetric periodic mesh of a tagged cell: the NumPy side.

The symmetric periodic mesh of a tagged OpenCASCADE cell is built by the Gmsh
job ``periodic_cell_mesh`` of mpxlite (its toolkit calls Gmsh). The job

1. detects the space group (:func:`mpxlite.cells.detect.detect_symmetry` on a
   throw-away mesh of the model that resolves the curved entities, with the
   tolerance raised to twice its measured chord sag; or takes a given
   :class:`~mpxlite.cells.detect.DetectedSymmetry`), and keeps its largest
   subgroup that is admissible (a test on the linear part of an operation)
   and has an asymmetric unit (:func:`supported_subgroup`; ``p1`` at worst: a
   missing symmetry costs time, never a refusal; :func:`near_note` reports a
   ``p1`` with near symmetries);
2. takes its asymmetric unit :math:`F`
   (:func:`mpxlite.cells.asymmetric.asymmetric_unit`, ITA domain about the
   detected centre) and cuts the prism :math:`F \times [z_0, z_1]` from the
   geometry by OpenCASCADE Booleans, the physical tags following the pieces;
3. constrains the faces of the prism paired by an operation of the group
   (rotation, glide, lattice translation: periodic meshing with that affine
   map, :func:`affine`; the curve masters kept acyclic, :func:`acyclic`,
   :func:`curve_root`; mirror faces need nothing) and meshes the unit with the
   size field :class:`SizeField` of a :class:`MeshSizing`;
4. bisects every edge that joins two nodes of one orbit
   (:func:`~mpxlite.cells.symmetric.split_orbit_edges`);
5. replicates the unit by the operations of the group
   (:func:`~mpxlite.cells.symmetric.replicate_group`), and
   :func:`write_replicated` extracts the periodic faces from the boundary of
   the tetrahedra on the window faces
   (:func:`~mpxlite.cells.symmetric.bloch_faces`), checks that the replicated
   mesh is closed (:func:`check_closed`), writes MSH 2.2 with the
   ``$Periodic`` links by ``T_1``, ``T_2`` and checks its topology;
   :func:`renumber` then renumbers the nodes so that every operation
   preserves the node order of every element
   (:func:`~mpxlite.cells.spacegroup.renumber_symmetric`).

This module holds what needs no Gmsh: the size policy, the report, the
subgroup choice, the size field, the affine maps of the pairing, the checks,
the writing and the numbering.

Density policy (:class:`MeshSizing`): the bulk size of each volume group;
``plane`` on the horizontal planes of the 2-D groups inside the stack,
graded linearly to the bulk size over ``transition``; with ``edge`` set,
``edge`` on the boundary curves of the ``edge_groups`` surfaces (the curves
on the window lines excluded: they cut a feature, they do not bound it),
graded linearly to the largest size over ``edge_grading * h_max / plane``
(3-D distance to the periodic pattern of those curves); or, with
``curve_edges``, sizes given per curve (or per piece of a curve), graded with
the slope ``edge_slope``.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mpxlite.cells.asymmetric import AsymmetricUnit, AsymmetricUnitError, asymmetric_unit
from mpxlite.cells.detect import DetectedSymmetry
from mpxlite.cells.spacegroup import LatticeOp, renumber_symmetric
from mpxlite.cells.symmetric import PERIODIC_NAMES, Block, bloch_faces, write_msh22_periodic
from mpxlite.cells.topology import MeshTopologyError, check_mesh_topology

__all__ = [
    "DEFAULT_PERIODIC_TAGS",
    "DETECT_CIRCLE_POINTS",
    "CurveMap",
    "MeshSizing",
    "SizeField",
    "SymmetricMeshError",
    "SymmetricMeshReport",
    "TaggedModel",
    "acyclic",
    "affine",
    "check_closed",
    "curve_root",
    "has_half_turn",
    "near_note",
    "on_window_line",
    "renumber",
    "restricted",
    "subgroups",
    "supported_subgroup",
    "write_replicated",
]

DEFAULT_PERIODIC_TAGS = (301, 302, 303, 304)
"""Tags of the Bloch faces when the model has none (names :data:`PERIODIC_NAMES`)."""


class SymmetricMeshError(RuntimeError):
    """The symmetric mesh could not be built (unsupported model, Boolean or matching failure)."""


@dataclass(frozen=True)
class MeshSizing:
    """Mesh-size policy of the unit (lengths in metres; see the module).

    Attributes:
        bulk: Volume group name -> bulk size.
        plane: Size on the horizontal planes of the 2-D groups inside the stack.
        transition: Distance over which ``plane`` grades to the bulk size.
        edge: Size on the edges of the ``edge_groups`` surfaces (``None``: off).
        edge_grading: Distance (m) over which ``edge`` grades to ``plane``
            (the same slope continues up to the largest size). The default
            ``0.25e-3`` m suits a cell of a few millimetres; scale it with
            the cell.
        edge_groups: 2-D group names, or name classes (``gap`` for
            ``gap_0_0_1``), whose boundary curves are refined.
        planes: z of the refined planes (default: those of the horizontal
            2-D groups strictly inside ``(z_0, z_1)``; ``()``: none).
        curve_samples: Points per curved edge (straight lines: 2).
        curve_edges: Gmsh curve tag of the model -> edge size on
            that curve, or the sizes of its ``k`` pieces of equal parameter
            length, graded with the slope ``edge_slope``; replaces
            ``edge``, ``edge_grading`` and ``edge_groups`` (the sizes the
            job ``periodic_cell_mesh`` derives from the refined edges).
        edge_slope: Growth of the size per unit distance from a refined
            curve (with ``curve_edges``).
    """

    bulk: Mapping[str, float]
    plane: float
    transition: float
    edge: float | None = None
    edge_grading: float = 0.25e-3
    edge_groups: Sequence[str] = ()
    planes: Sequence[float] | None = None
    curve_samples: int = 64
    curve_edges: Mapping[int, float | Sequence[float]] | None = None
    edge_slope: float = 0.5

    def __post_init__(self) -> None:
        for name in ("plane", "transition", "edge_grading", "edge_slope"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if self.edge is not None and not self.edge > 0:
            raise ValueError("edge must be positive or None")
        if not self.bulk or min(self.bulk.values()) <= 0:
            raise ValueError("bulk sizes must be positive")
        if self.curve_edges is not None:
            if self.edge is not None:
                raise ValueError("give edge or curve_edges, not both")
            if not np.all(self._edge_values() > 0):
                raise ValueError("curve_edges sizes must be positive")

    def _edge_values(self) -> np.ndarray:
        vals = [np.ravel(v) for v in (self.curve_edges or {}).values()]
        return np.concatenate(vals) if vals else np.zeros(0)

    @property
    def h_max(self) -> float:
        """Largest size."""
        return max(self.plane, *self.bulk.values())

    @property
    def h_min(self) -> float:
        """Smallest size."""
        edges = self._edge_values().tolist()
        return min(self.plane, *self.bulk.values(), *edges, self.edge or self.plane)

    @property
    def slope(self) -> float:
        """Growth of the edge size per unit distance from a refined curve."""
        if self.curve_edges is not None:
            return self.edge_slope
        edge = self.edge or self.plane
        return (self.h_max - edge) / (self.edge_grading * self.h_max / self.plane)

    @property
    def reach(self) -> float:
        """Distance beyond which the edge refinement no longer acts."""
        if self.curve_edges is not None:
            return (self.h_max - float(self._edge_values().min(initial=self.h_max))) / self.slope
        return self.edge_grading * self.h_max / self.plane

    @property
    def refines_edges(self) -> bool:
        """True when edges are refined."""
        return self.edge is not None or bool(self.curve_edges)


@dataclass(frozen=True, eq=False)
class SymmetricMeshReport:
    """Outcome of :func:`build_symmetric_mesh`."""

    path: Path
    symmetry: DetectedSymmetry
    unit: AsymmetricUnit
    physical: dict[str, tuple[int, int]]
    """``name -> (dim, tag)`` of the groups written."""
    n_nodes: int
    n_tetrahedra: int
    n_unit_tetrahedra: int
    n_split: int
    """Edges bisected in the unit mesh (two nodes of one orbit)."""
    n_paired: int = 0
    """Surfaces of the unit constrained to the image of their partner (``setPeriodic``)."""
    timings: dict[str, float] = field(default_factory=dict)
    detected: DetectedSymmetry | None = None
    """The group detected, when the mesh uses a subgroup of it (else ``None``)."""
    notes: tuple[str, ...] = ()
    """What the user should know: near symmetries, a subgroup used, detection warnings."""

    def summary(self) -> str:
        """One line."""
        tm = ", ".join(f"{k} {v:.2f} s" for k, v in self.timings.items())
        return (
            f"{self.path.name}: {self.symmetry.name} (order {self.symmetry.order}), "
            f"{self.n_nodes} nodes, {self.n_tetrahedra} tetrahedra ({self.n_unit_tetrahedra} "
            f"in the unit, {self.n_paired} paired surfaces, {self.n_split} edges split); {tm}"
        )


@dataclass
class TaggedModel:
    """The physical groups of a model, its window corner and its height range.

    Attributes:
        groups: Physical tag -> ``(dim, name, entity tags)``.
        lo: Lower window corner (lattice coordinates).
        z: ``(z_0, z_1)`` of the volumes.
    """

    groups: dict[int, tuple[int, str, list[int]]]  # tag -> (dim, name, entities)
    lo: np.ndarray
    z: tuple[float, float]


DETECT_CIRCLE_POINTS = 96
"""Mesh nodes per full turn of a curved entity in the throw-away detection mesh.

The chord sag of a circle of radius ``r`` is then
:math:`r (1 - \\cos(\\pi / 96)) \\approx 5.4 \\cdot 10^{-4} r`.
"""


def near_note(sym: DetectedSymmetry) -> str | None:
    """The note of a ``p1`` result with near symmetries (a symmetry possibly missed)."""
    if sym.order != 1 or not sym.near:
        return None
    near = "; ".join(
        f"{n.op.name if n.op else 'lattice'} M={n.matrix.tolist()} broken by "
        f"{n.deviation:.3g}" + (" m" if n.op else " (relative Gram mismatch)")
        for n in sym.near
    )
    return (
        f"no symmetry found within {sym.tol:.3g} m, but operations are broken by at most "
        f"{sym.near_tol:.3g} m ({near}): the mesh is plain periodic (p1); if the geometry is "
        "meant to be symmetric, a symmetry was missed and only time is lost"
    )


def subgroups(ops: Sequence[LatticeOp], allowed: Sequence[int]) -> list[frozenset[int]]:
    """The subgroups (index sets into ``ops``) generated by up to three of ``allowed``.

    ``ops`` is a group modulo the lattice, identity first; ``allowed`` a
    subset closed under composition that holds the identity.
    """

    def product(i: int, j: int) -> int:
        c = ops[i].compose(ops[j])
        best, gap = -1, math.inf
        for k, op in enumerate(ops):
            if np.array_equal(op.matrix, c.matrix):
                d = op.shift - c.shift
                d = float(np.max(np.abs(d - np.rint(d))))
                if d < gap:
                    best, gap = k, d
        if gap > 1e-6:
            raise SymmetricMeshError("the detected operations do not close modulo the lattice")
        return best

    out: set[frozenset[int]] = set()
    gens = [i for i in allowed if i != 0]
    for k in range(0, min(3, len(gens)) + 1):
        for combo in itertools.combinations(gens, k):
            sub = {0, *combo}
            grown = True
            while grown:
                new = {product(i, j) for i in sub for j in sub} - sub
                grown = bool(new)
                sub |= new
            out.add(frozenset(sub))
    return sorted(out, key=lambda g: (-len(g), sorted(g)))


def restricted(sym: DetectedSymmetry, keep: Sequence[int], lattice: np.ndarray) -> DetectedSymmetry:
    """``sym`` restricted to the operations ``keep`` (a subgroup, identity first)."""
    from mpxlite.cells.detect import POINT_GROUP, point_group, wallpaper_name

    ops = tuple(sym.ops[i] for i in keep)
    return dataclasses.replace(
        sym,
        ops=ops,
        permutations=tuple(sym.permutations[i] for i in keep),
        direction_signs=tuple(sym.direction_signs[i] for i in keep),
        deviations=tuple(sym.deviations[i] for i in keep),
        name=wallpaper_name(ops, lattice),
        point_group=POINT_GROUP[point_group(ops, lattice)],
    )


def has_half_turn(sym: DetectedSymmetry, lattice: np.ndarray) -> bool:
    """Whether the group holds the half turn (in-plane matrix ``-I``)."""
    return any(op.det == 1 and np.allclose(op.cartesian(lattice)[0], -np.eye(2)) for op in sym.ops)


def supported_subgroup(
    sym: DetectedSymmetry, lattice: np.ndarray, lo: np.ndarray,
    admissible: Callable[[np.ndarray], bool] | None,
) -> tuple[DetectedSymmetry, AsymmetricUnit, list[str]]:  # fmt: skip
    """The largest subgroup of ``sym`` that is admissible and has an asymmetric unit.

    Of two subgroups of one order, the one with the half turn is preferred.
    Returns the group used, its unit and the notes (empty when ``sym`` is
    used whole).
    """
    allowed = [
        i for i, op in enumerate(sym.ops)
        if admissible is None or admissible(op.cartesian(lattice)[0])
    ]  # fmt: skip
    notes: list[str] = []
    if len(allowed) < sym.order:
        dropped = ", ".join(sym.ops[i].name for i in range(sym.order) if i not in allowed)
        notes.append(
            f"operations {dropped} of {sym.name} are not symmetries of the materials "
            "(anisotropy): not used"
        )
    first_error = None
    candidates = subgroups(sym.ops, allowed)
    candidates.sort(key=lambda g: (-len(g), not has_half_turn(restricted(sym, sorted(g), lattice),
                                                              lattice)))  # fmt: skip
    for sub in candidates:
        cand = sym if len(sub) == sym.order else restricted(sym, sorted(sub), lattice)
        try:
            unit = asymmetric_unit(cand.ops, lattice, name=cand.name, centre=cand.centre,
                                   window_lo=lo)  # fmt: skip
        except AsymmetricUnitError as err:
            first_error = first_error or str(err)
            continue
        if cand is not sym:
            if first_error:
                notes.append(f"{sym.name} cannot be meshed symmetrically ({first_error})")
            notes.append(
                f"the mesh uses the subgroup {cand.name} (order {cand.order} of {sym.order}); "
                "only time is lost"
            )
        return cand, unit, notes
    raise AssertionError("p1 always has a unit")  # pragma: no cover


def on_window_line(p: np.ndarray, lattice: np.ndarray, lo: np.ndarray, tol: float) -> bool:
    """All the points ``p`` ``(k, 3)`` lie on one window line (a line of the lattice grid)."""
    u = p[:, :2] @ np.linalg.inv(lattice) - lo
    utol = tol / float(np.min(np.linalg.norm(lattice, axis=1)))
    return any(np.all(np.abs(u[:, i] - np.rint(u[:, i])) <= utol) for i in (0, 1))


class SizeField:
    """``h(entity, x)``: plane grading, bulk size by entity, edge refinement (see the module).

    Near the refined edges: ``min_i(h_i + slope d_i)`` over the segments
    ``i`` of edge size ``h_i`` at the distance ``d_i``, capped by the bulk.
    """

    def __init__(
        self, sizing: MeshSizing, planes: Sequence[float], segs: np.ndarray,
        seg_sizes: np.ndarray, bulk_of: Mapping[tuple[int, int], float],
    ) -> None:  # fmt: skip
        self.s = sizing
        self.planes = np.asarray(planes, dtype=float)
        self.a = segs[:, 0]
        self.ab = segs[:, 1] - segs[:, 0]
        self.ab2 = np.maximum(np.einsum("ij,ij->i", self.ab, self.ab), 1e-300)
        self.h_seg = np.asarray(seg_sizes, dtype=float)
        self.slope = sizing.slope
        self.bulk_of = dict(bulk_of)
        self.hmax = sizing.h_max

    def __call__(self, dim: int, tag: int, x: float, y: float, z: float) -> float:
        """The size at ``(x, y, z)`` on the entity ``(dim, tag)``."""
        s = self.s
        hb = self.bulk_of.get((dim, tag), self.hmax)
        h = hb
        if len(self.planes):
            d = float(np.min(np.abs(self.planes - z)))
            h = s.plane + (hb - s.plane) * min(1.0, d / s.transition)
        if len(self.a):
            ap = np.array((x, y, z)) - self.a
            w = np.clip(np.einsum("ij,ij->i", ap, self.ab) / self.ab2, 0.0, 1.0)
            d = np.sqrt(np.einsum("ij,ij->i", *(2 * [ap - w[:, None] * self.ab])))
            h = min(h, float(np.min(self.h_seg + self.slope * d)))
        return float(h)


def affine(g: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The ``4 x 4`` affine map ``x -> g x + t`` (in plane, ``z`` fixed)."""
    a = np.eye(4)
    a[:2, :2] = g
    a[:2, 3] = t
    return a


def acyclic(master: Mapping[int, int]) -> bool:
    """Whether the map slave -> master has no cycle."""
    for c0 in master:
        seen, c = {c0}, c0
        while c in master:
            c = master[c]
            if c in seen:
                return False
            seen.add(c)
    return True


CurveMap = dict[int, tuple[int, np.ndarray]]
"""Slave curve -> (master curve, affine ``4 x 4`` master -> slave)."""


def curve_root(cmap: CurveMap, c: int) -> tuple[int, np.ndarray]:
    """The curve whose mesh ``c`` copies, and the affine map root -> ``c``."""
    a = np.eye(4)
    seen = {c}
    while c in cmap:
        master, step = cmap[c]
        a = a @ step  # x_c = step x_master
        c = master
        if c in seen:
            break
        seen.add(c)
    return c, a


def check_closed(
    nodes: np.ndarray, blocks: Sequence[Block], lattice: np.ndarray, model: TaggedModel, tol: float
) -> None:
    """Every free face of the tetrahedra lies on a window face or on ``z_0`` / ``z_1``.

    A free face inside the cell means two images of the unit that do not
    conform (unmatched paired faces).
    """
    tets = np.concatenate([c for _, e, c in blocks if e == 4])
    faces = np.sort(np.concatenate([tets[:, [1, 2, 3]], tets[:, [0, 2, 3]], tets[:, [0, 1, 3]],
                                    tets[:, [0, 1, 2]]]), axis=1)  # fmt: skip
    uniq, cnt = np.unique(faces, axis=0, return_counts=True)
    free = uniq[cnt == 1]
    xyz = nodes[free]
    u = xyz[..., :2] @ np.linalg.inv(lattice) - model.lo
    utol = tol / float(np.min(np.linalg.norm(lattice, axis=1)))
    on = np.zeros(len(free), dtype=bool)
    for axis in (0, 1):
        for level in (0.0, 1.0):
            on |= np.all(np.abs(u[..., axis] - level) <= utol, axis=1)
    for z in model.z:
        on |= np.all(np.abs(xyz[..., 2] - z) <= tol, axis=1)
    if not np.all(on):
        raise MeshTopologyError(
            f"non-conforming replicated mesh: {int((~on).sum())} free tetrahedron faces inside "
            "the cell (the paired faces of the unit do not match)"
        )


def write_replicated(
    output_path: Path | str, nodes: np.ndarray, elems: list[Block], model: TaggedModel,
    names: Mapping[int, str], sides: Mapping[int, int], lattice: np.ndarray, tol: float,
) -> tuple[Path, dict[str, tuple[int, int]]]:  # fmt: skip
    """Bloch faces, checks, MSH 2.2 with the ``$Periodic`` links; ``(path, physical)``."""
    side_name = {k: names[tag] for tag, k in sides.items()}
    pnames = [side_name.get(k, PERIODIC_NAMES[k]) for k in range(4)]
    elems = elems + bloch_faces(nodes, elems, lattice, window_lo=model.lo, names=pnames)
    check_closed(nodes, elems, lattice, model, tol)
    physical = {names[tag]: (dim, tag) for tag, (dim, _, _) in sorted(model.groups.items())}
    used = {tg for _, tg in physical.values()}
    for k, name in enumerate(pnames):
        if name not in physical:
            tag = DEFAULT_PERIODIC_TAGS[k]
            while tag in used:
                tag += 1000
            physical[name] = (2, tag)
            used.add(tag)
    kinds = {k for k, _, _ in elems}
    physical = {n: v for n, v in physical.items() if n in kinds}
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = np.ptp(nodes[:, :2] @ np.linalg.inv(lattice), axis=0)
    if np.any(ext > 1 + 1e-9):
        raise MeshTopologyError(f"{out}: replicated mesh extent {ext.tolist()} exceeds the cell")
    scale = math.sqrt(abs(float(np.linalg.det(lattice))))
    write_msh22_periodic(
        out, nodes, elems, physical, scale,
        periodic=((pnames[1], pnames[0], lattice[0]), (pnames[3], pnames[2], lattice[1])),
    )  # fmt: skip
    topo = check_mesh_topology(out)
    if not topo.ok:
        raise MeshTopologyError(f"non-conforming replicated mesh: {topo.summary()}")
    return out, physical


def renumber(
    out: Path, lattice: np.ndarray, ops: Sequence, lo: np.ndarray, numbering: bool
) -> None:
    """Symmetric numbering of ``ops`` (or the periodic one), slave facets aligned."""
    from mpxlite.cells.periodic import check_periodic_facet_order, renumber_periodic

    if numbering:
        report = renumber_symmetric(out, lattice, ops)
    elif not check_periodic_facet_order(out).ok:
        report = renumber_periodic(out, lattice[0], lattice[1], origin=tuple(lo))
    else:
        return
    if not report.ok:
        raise SymmetricMeshError(f"periodic facet repair failed: {report.summary()}")
