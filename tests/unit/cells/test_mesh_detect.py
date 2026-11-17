# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""T0: space-group detection of tagged cells (:mod:`mpxlite.cells.detect`).

Synthetic cells: a generic quadrilateral motif replicated by the
operations of a wallpaper group (each copy triangulated along one diagonal
or the other: the detection must not depend on the mesh), optionally on a
tetrahedral slab with its Bloch faces. A mesh file that is not MSH 2.x is
refused by :meth:`TaggedCell.from_msh` (its reading of a Gmsh mesh written by
a user's script: ``tests/integration/cells``).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from mpxlite.cells.detect import (
    TaggedCell,
    TagGeometry,
    detect_symmetry,
    lattice_holohedry,
    wallpaper_name,
)
from mpxlite.cells.periodic import UnsupportedMeshFormatError
from mpxlite.cells.spacegroup import LatticeOp, lattice_group
from mpxlite.cells.symmetric import c4v_lattice_ops

S3 = math.sqrt(3.0)
OBLIQUE = ((1.0, 0.0), (0.31, 1.13))
RECT = ((1.0, 0.0), (0.0, 1.37))
RHOMB = ((0.5, 0.8), (0.5, -0.8))  # centred rectangular, conventional 1 x 1.6
SQUARE = ((1.0, 0.0), (0.0, 1.0))
HEX = ((1.0, 0.0), (0.5, S3 / 2))

I2 = np.eye(2)
MX = np.diag([-1.0, 1.0])  # x -> -x (axis along y)
MY = np.diag([1.0, -1.0])


def _rot(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


# name: (lattice, generators (g, Cartesian translation), expected order mod the lattice)
GROUPS = {
    "p1": (OBLIQUE, [], 1),
    "p2": (OBLIQUE, [(-I2, (0, 0))], 2),
    "pm": (RECT, [(MX, (0, 0))], 2),
    "pg": (RECT, [(MX, (0, 1.37 / 2))], 2),
    "cm": (RHOMB, [(MX, (0, 0))], 2),
    "pmm": (RECT, [(MX, (0, 0)), (MY, (0, 0))], 4),
    "pmg": (RECT, [(-I2, (0, 0)), (MX, (0.5, 0))], 4),
    "pgg": (RECT, [(-I2, (0, 0)), (MX, (0.5, 1.37 / 2))], 4),
    "cmm": (RHOMB, [(MX, (0, 0)), (MY, (0, 0))], 4),
    "p4": (SQUARE, [(_rot(90), (0, 0))], 4),
    "p4m": (SQUARE, [(_rot(90), (0, 0)), (MX, (0, 0))], 8),
    "p4g": (SQUARE, [(_rot(90), (0, 0)), (MX, (0.5, 0.5))], 8),
    "p3": (HEX, [(_rot(120), (0, 0))], 3),
    "p3m1": (HEX, [(_rot(120), (0, 0)), (MX, (0, 0))], 6),
    "p31m": (HEX, [(_rot(120), (0, 0)), (MY, (0, 0))], 6),
    "p6": (HEX, [(_rot(60), (0, 0))], 6),
    "p6m": (HEX, [(_rot(60), (0, 0)), (MY, (0, 0))], 12),
}

MOTIF = np.array([(0.0, 0.0), (1.0, 0.15), (0.8, 0.9), (0.1, 0.7)])
"""A generic convex quadrilateral (no symmetry of its own), scaled to the cell."""


def _group(lattice, gens) -> tuple[LatticeOp, ...]:
    return lattice_group(
        LatticeOp.from_cartesian(f"g{k}", g, lattice, translation=t)
        for k, (g, t) in enumerate(gens)
    )


class _Builder:
    """Points and tagged simplices of a synthetic cell."""

    def __init__(self, lattice) -> None:
        self.t = np.asarray(lattice, dtype=float)
        self.points: list[np.ndarray] = []
        self.cells: dict[int, tuple[int, list[list[int]], str]] = {}

    def _add(self, xyz: np.ndarray) -> list[int]:
        k = len(self.points)
        self.points.extend(np.asarray(xyz, dtype=float))
        return list(range(k, k + len(xyz)))

    def cell(self, tag: int, dim: int, rows: list[int], name: str = "") -> None:
        self.cells.setdefault(tag, (dim, [], name))[1].append(rows)

    def polygon(self, tag: int, xy: np.ndarray, start: int = 0, name: str = "") -> None:
        """A convex polygon at z = 0, fan-triangulated from vertex ``start``."""
        xy = np.roll(np.asarray(xy, dtype=float), -start, axis=0)
        ids = self._add(np.column_stack([xy, np.zeros(len(xy))]))
        for k in range(1, len(ids) - 1):
            self.cell(tag, 2, [ids[0], ids[k], ids[k + 1]], name)

    def orbit(self, ops, centre_frac=(0.137, 0.291), size=0.07, tags=None, names=None) -> None:
        """The motif and its images under ``ops``, folded into ``[0, 1)^2``."""
        scale = math.sqrt(abs(np.linalg.det(self.t)))
        base = np.asarray(centre_frac) @ self.t + size * scale * (MOTIF - MOTIF.mean(axis=0))
        tinv = np.linalg.inv(self.t)
        for k, op in enumerate(ops):
            g, tr = op.cartesian(self.t)
            img = base @ g.T + tr
            shift = np.floor(img.mean(axis=0) @ tinv)
            img = img - shift @ self.t
            tag = 1 if tags is None else tags[k]
            self.polygon(tag, img, start=k % 2, name="" if names is None else names[k])

    def slab(self, z0: float, z1: float, n: int = 4, tag: int = 11, faces=True) -> None:
        """A tetrahedral layer over the window ``[0, 1]^2`` (conforming, Kuhn split)."""
        grid = {}
        for i in range(n + 1):
            for j in range(n + 1):
                for k, z in enumerate((z0, z1)):
                    x = np.array([i / n, j / n]) @ self.t
                    grid[i, j, k] = self._add(np.array([[x[0], x[1], z]]))[0]
        kuhn = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
        for i in range(n):
            for j in range(n):
                c = [grid[i + (b & 1), j + ((b >> 1) & 1), (b >> 2) & 1] for b in range(8)]
                for tet in kuhn:
                    self.cell(tag, 3, [c[v] for v in tet], "slab")
        if faces:
            for side, (tag_f, name) in enumerate(((301, "periodic_xm"), (302, "periodic_xp"))):
                i = side * n
                for j in range(n):
                    q = [grid[i, j, 0], grid[i, j + 1, 0], grid[i, j + 1, 1], grid[i, j, 1]]
                    self.cell(tag_f, 2, [q[0], q[1], q[2]], name)
                    self.cell(tag_f, 2, [q[0], q[2], q[3]], name)

    def build(self, **kw) -> TaggedCell:
        groups = {
            t: TagGeometry(d, np.array(c), name) for t, (d, c, name) in self.cells.items()
        }  # fmt: skip
        return TaggedCell(self.t, np.array(self.points), groups, **kw)


# ----- lattice -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lattice", "order"),
    [(OBLIQUE, 2), (RECT, 4), (RHOMB, 4), (SQUARE, 8), (HEX, 12), (((1, 0), (3, 1)), 8),
     (((2.0, 0.0), (7.0, S3)), 12)],
)  # fmt: skip
def test_lattice_holohedry(lattice, order):
    t = np.asarray(lattice, dtype=float)
    holo = lattice_holohedry(t)
    assert len(holo) == order
    assert np.array_equal(holo[0], np.eye(2))
    gram = t @ t.T
    for m in holo:
        assert np.allclose(m.T @ gram @ m, gram)


# ----- the 17 groups (12 of the brief + the trigonal / hexagonal ones) ---------------------


@pytest.mark.parametrize("name", list(GROUPS))
def test_detect_wallpaper_groups(name):
    lattice, gens, order = GROUPS[name]
    ops = _group(lattice, gens)
    b = _Builder(lattice)
    b.orbit(ops)
    res = detect_symmetry(b.build())
    assert res.name == name, res.summary()
    assert res.order == order == len(ops)
    assert {op.key() for op in res.ops} == {op.key() for op in ops}
    assert max(res.deviations) <= res.tol
    assert not res.near
    assert wallpaper_name(ops, lattice) == name
    assert res.primitive
    assert res.orthogonal == (lattice in (RECT, SQUARE))
    if not res.orthogonal and name not in ("p1", "p2"):  # p1, p2: any lattice
        # mesh-m11: the warning names the lattice type (hexagonal is not called oblique)
        assert any(f"{res.lattice_type} lattice with non-orthogonal" in w for w in res.warnings)
        assert res.lattice_type != "hexagonal" or not any("oblique" in w for w in res.warnings)


def test_detect_centred_conventional_cell():
    """cm on its conventional (non-primitive) rectangular cell: the centring is found."""
    rect = ((1.0, 0.0), (0.0, 1.6))
    ops = _group(rect, [(MX, (0, 0)), (I2, (0.5, 0.8))])
    b = _Builder(rect)
    b.orbit(ops)
    res = detect_symmetry(b.build())
    assert res.name == "cm" and res.order == 4 and not res.primitive
    assert any("not primitive" in w for w in res.warnings)
    assert "t(1/2,1/2)" in [op.name for op in res.ops]


# ----- tags, classes, directions ---------------------------------------------------------


def test_tag_permutation_follows_classes():
    ops = _group(SQUARE, [(_rot(90), (0, 0))])
    b = _Builder(SQUARE)
    b.orbit(ops, tags=[11, 12, 13, 14], names=[f"port_{k}" for k in range(4)])
    res = detect_symmetry(b.build())
    assert res.name == "p4"
    c4 = res.op("C4")
    perm = res.permutations[res.ops.index(c4)]
    assert sorted(perm) == sorted(perm.values()) == [11, 12, 13, 14]
    assert all(perm[k] != k for k in perm)
    assert [op.name for op in res.tag_preserving()] == ["E"]
    # one class per tag: nothing may be exchanged
    res1 = detect_symmetry(b.build(), classes={k: k for k in (11, 12, 13, 14)})
    assert res1.name == "p1" and res1.order == 1


def _checkerboard_2d() -> _Builder:
    """Patch diamond centred on (1/2, 1/2), square gaps at (0, 1/2) and (1/2, 0)."""
    b = _Builder(SQUARE)
    r = 0.04
    d = np.array([(0.5, r), (1 - r, 0.5), (0.5, 1 - r), (r, 0.5)])
    b.polygon(1, d, name="tile_patch")
    for tag, (cx, cy) in ((1000, (0.0, 0.5)), (1001, (0.5, 0.0))):
        sq = np.array([(-r, -r), (r, -r), (r, r), (-r, r)]) + np.array([cx, cy])
        b.polygon(tag, sq, name=f"gap_0_0_{tag - 1000}")
    return b


def test_checkerboard_c4v_and_direction_signs():
    b = _checkerboard_2d()
    res = detect_symmetry(b.build(), directions={1000: (1, 0, 0), 1001: (0, 1, 0)})
    assert res.name == "p4m" and np.allclose(res.centre, (0.5, 0.5))
    ref = c4v_lattice_ops(1.0)
    assert [op.name for op in res.ops] == [op.name for op in ref]
    for got, want in zip(res.ops, ref, strict=True):
        assert np.array_equal(got.matrix, want.matrix) and np.allclose(got.shift, want.shift)
    k = res.ops.index(res.op("C4"))
    assert res.permutations[k] == {1: 1, 1000: 1001, 1001: 1000}
    assert res.direction_signs[k] == {1000: 1, 1001: -1}
    assert res.direction_signs[res.ops.index(res.op("mx"))] == {1000: -1, 1001: 1}
    assert [op.name for op in res.tag_preserving()] == ["E", "C2", "mx", "my"]
    assert wallpaper_name(res.tag_preserving(), SQUARE) == "pmm"
    # a direction that breaks the rotation
    res2 = detect_symmetry(b.build(), directions={1000: (1, 0, 0), 1001: (1, 1, 0)})
    assert res2.name == "p2" and res2.order == 2  # only C2 maps (1, 1) onto -(1, 1)
    assert any(n.reason == "direction" for n in res2.near)


def test_near_symmetry_reported_not_symmetrised():
    ops = _group(SQUARE, [(_rot(90), (0, 0)), (MX, (0, 0))])
    b = _Builder(SQUARE)
    b.orbit(ops)
    cell = b.build()
    pts = cell.points.copy()
    pts[:4, 0] += 1e-4  # first copy moved by 1e-4 of the cell
    res = detect_symmetry(TaggedCell(cell.lattice, pts, cell.groups))
    assert res.order == 1 and res.name == "p1"
    assert len(res.near) == 7
    assert all(5e-5 < n.deviation < 2e-4 and n.tag == 1 for n in res.near)
    loose = detect_symmetry(TaggedCell(cell.lattice, pts, cell.groups), tol=2e-4)
    assert loose.name == "p4m"


def test_near_lattice_symmetry():
    lat = ((1.0, 0.0), (0.0, 1.0002))
    b = _Builder(lat)
    b.orbit(_group(lat, [(MX, (0, 0)), (MY, (0, 0))]))
    res = detect_symmetry(b.build())
    assert res.name == "pmm" and res.lattice_type == "rectangular"
    assert sum(n.reason == "lattice" for n in res.near) == 4


# ----- volumes, Bloch faces, files ---------------------------------------------------------


def test_slab_and_window_faces():
    ops = _group(SQUARE, [(_rot(90), (0, 0))])
    b = _Builder(SQUARE)
    b.orbit(ops, centre_frac=(0.3, 0.2))
    b.slab(-0.2, 0.0)
    res = detect_symmetry(b.build())
    assert res.name == "p4" and res.window_tags == (301, 302)
    assert all(p[11] == 11 for p in res.permutations)


def test_continuous_translation_point_ops():
    # a laterally uniform slab: the point operations about the window centre, with a warning
    b = _Builder(SQUARE)
    b.slab(-0.2, 0.0, faces=False)
    res = detect_symmetry(b.build())
    assert res.name == "p4m" and res.order == 8 and res.primitive
    assert any("continuous translations" in w for w in res.warnings)


def _write_msh22(cell: TaggedCell, path: Path) -> None:
    etype = {0: 15, 1: 1, 2: 2, 3: 4}
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames",
             str(len(cell.groups))]  # fmt: skip
    lines += [f'{g.dim} {t} "{g.name or f"g{t}"}"' for t, g in cell.groups.items()]
    lines += ["$EndPhysicalNames", "$Nodes", str(len(cell.points))]
    lines += [f"{k + 1} {x:.17g} {y:.17g} {z:.17g}" for k, (x, y, z) in enumerate(cell.points)]
    rows = [(t, g.dim, c) for t, g in cell.groups.items() for c in g.cells]
    lines += ["$EndNodes", "$Elements", str(len(rows))]
    lines += [
        f"{k + 1} {etype[d]} 2 {t} {t} " + " ".join(str(v + 1) for v in c)
        for k, (t, d, c) in enumerate(rows)
    ]
    lines += ["$EndElements"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_from_msh_round_trip(tmp_path):
    b = _checkerboard_2d()
    b.slab(-0.2, 0.0)
    path = tmp_path / "cb.msh"
    _write_msh22(b.build(), path)
    cell = TaggedCell.from_msh(path, SQUARE)
    assert cell.groups[1000].name == "gap_0_0_0" and cell.groups[11].dim == 3
    res = detect_symmetry(cell)
    assert res.name == "p4m" and res.classes[1000] == res.classes[1001] == "gap"
    assert res.window_tags == (301, 302)


def test_from_msh_refuses_another_format(tmp_path):
    path = tmp_path / "v41.msh"
    path.write_text("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n", encoding="utf-8")
    with pytest.raises(UnsupportedMeshFormatError, match=r"MSH 2\.2"):
        TaggedCell.from_msh(path, SQUARE)
