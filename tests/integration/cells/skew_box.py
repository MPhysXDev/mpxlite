# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""Parallelepiped (general-lattice) box mesh generator (a user's script: it imports Gmsh).

The in-plane cell is the parallelogram spanned by the lattice vectors
``T_1 = (t1_x, 0)`` and ``T_2 = (t2_x, t2_y)``, centred on ``x = y = 0``;
``t2_x = 0`` recovers the rectangular cell of ``box.py``. Vertical stack and
tags are those of ``box.py`` (2-layer box, or the 3-layer box with a
centred ``grating_layer`` slab when ``grating_layer_thickness`` is set):

    2   vol_bot           301  periodic_xm   (u = -1/2, the -T_1 side)
    3   vol_top           302  periodic_xp   (u = +1/2) = periodic_xm + T_1
    5   grating_layer     303  periodic_ym   (v = -1/2, the -T_2 side)
    102 outer_bot         304  periodic_yp   (v = +1/2) = periodic_ym + T_2
    103 outer_top

with ``(u, v)`` the lattice coordinates of a point, ``r = u T_1 + v T_2``.

Mesh sizing: uniform ``mesh_size``. MSH 2.2 ASCII, no optimisation pass
(periodic node positions are preserved).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gmsh

from mpxlite.cells.periodic import renumber_periodic

TAGS: dict[str, int] = {
    "vol_bot": 2,
    "vol_top": 3,
    "grating_layer": 5,
    "outer_bot": 102,
    "outer_top": 103,
    "periodic_xm": 301,
    "periodic_xp": 302,
    "periodic_ym": 303,
    "periodic_yp": 304,
}


@dataclass(frozen=True)
class SkewBoxParams:
    """Geometric and meshing parameters (metres)."""

    t1_x: float
    t2_x: float
    t2_y: float
    substrate_height: float
    superstrate_height: float
    mesh_size: float
    grating_layer_thickness: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "t1_x",
            "t2_y",
            "substrate_height",
            "superstrate_height",
            "mesh_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.grating_layer_thickness is not None:
            half = self.grating_layer_thickness / 2.0
            if not 0 < half < min(self.substrate_height, self.superstrate_height):
                raise ValueError(
                    "grating_layer_thickness must be positive and its half smaller than "
                    "both substrate_height and superstrate_height"
                )

    def z_layers(self) -> list[tuple[str, float, float]]:
        """``(volume name, z_low, z_high)`` of every layer, bottom to top."""
        zs = -self.substrate_height
        zt = self.superstrate_height
        if self.grating_layer_thickness is None:
            return [("vol_bot", zs, 0.0), ("vol_top", 0.0, zt)]
        h = self.grating_layer_thickness / 2.0
        return [("vol_bot", zs, -h), ("grating_layer", -h, h), ("vol_top", h, zt)]

    @property
    def z_top(self) -> float:
        """z of the top face of the box."""
        return self.superstrate_height

    @property
    def z_bot(self) -> float:
        """z of the bottom face of the box."""
        return -self.substrate_height


def generate_mesh(params: SkewBoxParams, output_path: Path | str) -> Path:
    """Build the parallelepiped box mesh and write it as MSH 2.2 ASCII.

    Args:
        params: Lattice, stack and meshing parameters.
        output_path: Destination ``.msh`` file (parents are created if missing).

    Returns:
        The absolute path of the file just written.
    """
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("skew_box")
        _build(params)
        _mesh(params, output_path)
    finally:
        gmsh.finalize()
    return output_path


def _lattice_uv(params: SkewBoxParams, x: float, y: float) -> tuple[float, float]:
    v = y / params.t2_y
    u = (x - v * params.t2_x) / params.t1_x
    return u, v


def _build(params: SkewBoxParams) -> None:
    occ = gmsh.model.occ
    t1 = (params.t1_x, 0.0)
    t2 = (params.t2_x, params.t2_y)
    corners = [
        (-0.5 * t1[0] - 0.5 * t2[0], -0.5 * t2[1]),
        (+0.5 * t1[0] - 0.5 * t2[0], -0.5 * t2[1]),
        (+0.5 * t1[0] + 0.5 * t2[0], +0.5 * t2[1]),
        (-0.5 * t1[0] + 0.5 * t2[0], +0.5 * t2[1]),
    ]
    layers = params.z_layers()
    for _, z_lo, z_hi in layers:
        pts = [occ.addPoint(x, y, z_lo) for x, y in corners]
        lines = [occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        surf = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        occ.extrude([(2, surf)], 0.0, 0.0, z_hi - z_lo)
    occ.removeAllDuplicates()
    occ.synchronize()

    # Volumes, by z centre of mass.
    volumes = [t for _, t in gmsh.model.getEntities(3)]
    if len(volumes) != len(layers):
        raise RuntimeError(f"expected {len(layers)} volumes, got {len(volumes)}")
    for name, z_lo, z_hi in layers:
        zc = 0.5 * (z_lo + z_hi)
        found = [v for v in volumes if abs(occ.getCenterOfMass(3, v)[2] - zc) < 1e-9]
        if len(found) != 1:
            raise RuntimeError(f"expected one volume for {name}, got {found}")
        gmsh.model.addPhysicalGroup(3, found, TAGS[name], name)
    _tag_surfaces(params, layers, t1, t2)


def _classify_surfaces(
    params: SkewBoxParams, layers: list[tuple[str, float, float]]
) -> dict[str, list[tuple[float, int]]]:
    """``(z centre, tag)`` of the surfaces of each group (horizontal ones by z,
    lateral ones by their lattice coordinates)."""
    occ = gmsh.model.occ
    z_bot = layers[0][1]
    z_top = layers[-1][2]
    groups: dict[str, list[tuple[float, int]]] = {
        k: [] for k in ("obot", "otop", "xm", "xp", "ym", "yp")
    }
    tol = 1e-6  # OCC bounding boxes are inflated by the geometric tolerance
    for _, s in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(2, s)
        cx, cy, cz = occ.getCenterOfMass(2, s)
        if abs(bb[5] - bb[2]) < tol:  # horizontal face
            z = cz
            if abs(z - z_bot) < tol:
                groups["obot"].append((z, s))
            elif abs(z - z_top) < tol:
                groups["otop"].append((z, s))
            continue
        u, v = _lattice_uv(params, cx, cy)
        for key, ok in (
            ("xm", abs(u + 0.5) < 1e-6),
            ("xp", abs(u - 0.5) < 1e-6),
            ("ym", abs(v + 0.5) < 1e-6),
            ("yp", abs(v - 0.5) < 1e-6),
        ):
            if ok:
                groups[key].append((cz, s))
    return groups


def _tag_surfaces(
    params: SkewBoxParams,
    layers: list[tuple[str, float, float]],
    t1: tuple[float, float],
    t2: tuple[float, float],
) -> None:
    """Surface physical groups, and the T_1 / T_2 periodic pairings."""
    groups = _classify_surfaces(params, layers)
    for key, name in (
        ("obot", "outer_bot"),
        ("otop", "outer_top"),
        ("xm", "periodic_xm"),
        ("xp", "periodic_xp"),
        ("ym", "periodic_ym"),
        ("yp", "periodic_yp"),
    ):
        tags = [s for _, s in sorted(groups[key])]
        if not tags:
            raise RuntimeError(f"no surface found for {name}")
        gmsh.model.addPhysicalGroup(2, tags, TAGS[name], name)

    # Periodicity: +T_1 face = -T_1 face translated by T_1, same for T_2.
    for minus, plus, (dx, dy) in (("xm", "xp", t1), ("ym", "yp", t2)):
        lo = [s for _, s in sorted(groups[minus])]
        hi = [s for _, s in sorted(groups[plus])]
        if len(lo) != len(hi):
            raise RuntimeError(f"periodic-face count mismatch {minus}/{plus}: {lo} vs {hi}")
        affine = [1, 0, 0, dx, 0, 1, 0, dy, 0, 0, 1, 0, 0, 0, 0, 1]
        for s_lo, s_hi in zip(lo, hi, strict=True):
            gmsh.model.mesh.setPeriodic(2, [s_hi], [s_lo], affine)


def _mesh(params: SkewBoxParams, output_path: Path) -> None:
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.Optimize", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", params.mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", params.mesh_size)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(output_path))
    _renumber_periodic(params, output_path)


def _renumber_periodic(params: SkewBoxParams, path: Path) -> None:
    """Renumber the written MSH 2.2 file so both periodic maps preserve the node ORDER.

    GetDP picks the second-order facet functions ``BF_Edge_3F_b`` / ``_c`` of a
    facet by the rank of its GLOBAL node numbers, and ``LinkCplx`` on
    ``FacetsOf`` ties same-named coefficients of a slave facet and its
    image: the link is right only when the periodic map preserves the
    node-number ranking inside every linked facet. gmsh does not guarantee
    it here (the curve loop of the parallelogram runs anti-parallel on
    opposite sides), hence the core repair
    :func:`mpxlite.cells.periodic.renumber_periodic` (cell centred on
    ``x = y = 0``: default ``origin``), checked afterwards.
    """
    report = renumber_periodic(path, (params.t1_x, 0.0), (params.t2_x, params.t2_y))
    if not report.ok or not report.has_periodic:
        raise RuntimeError(f"skew_box: periodic facet order not repaired: {report.summary()}")
