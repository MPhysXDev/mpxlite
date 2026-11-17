# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""Box mesh generator of the integration tests (a user's script: it imports Gmsh).

A rectangular periodic cell ``period_x x period_y`` centred on ``x = y = 0``,
closed at top and bottom by its outer faces. Two variants:

  * **default 2-layer box** (``grating_layer_thickness=None``)::

        z = +Ht   ----  outer_top     (top face of the box)
                    |   vol_top       (halfspace above z = 0)
        z = 0
                    |   vol_bot       (halfspace below z = 0)
        z = -Hs   ----  outer_bot     (bottom face of the box)

  * **grating_layer_thickness=t > 0** -- 3-layer box with a thin centred
    slab ``grating_layer`` (|z| < t/2) between vol_bot and vol_top, meshed
    at the bulk size.

``superstrate_height`` / ``substrate_height`` are the distances from
``z = 0`` to the outer faces. Tags:

    2   vol_bot           102  outer_bot
    3   vol_top           103  outer_top
    5   grating_layer     301  periodic_xm   302  periodic_xp
                          303  periodic_ym   304  periodic_yp

Mesh sizing: uniform ``mesh_size``; with ``mesh_size_bot`` set, the
substrate side (vol_bot, and the grating layer) uses ``mesh_size_bot``.
MSH 2.2 ASCII, no optimisation pass (periodic node positions are
preserved). Node numbering: gmsh's, unless a periodic map breaks the node
order of a linked facet (coarse meshes, where one triangle spans a whole
cell edge): the file is then renumbered by
:func:`mpxlite.cells.periodic.renumber_periodic`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gmsh

from mpxlite.cells.periodic import check_periodic_facet_order, renumber_periodic

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
class BoxParams:
    """Geometric and meshing parameters (all distances in metres).

    ``mesh_size`` is the characteristic length on the superstrate side
    (``vol_top``); ``mesh_size_bot`` (default: ``mesh_size``) the one on the
    substrate side (``vol_bot`` and the grating layer).
    """

    period_x: float
    period_y: float
    substrate_height: float
    superstrate_height: float
    mesh_size: float
    mesh_size_bot: float | None = None
    grating_layer_thickness: float | None = None  # carves a centred slab on z=0

    def __post_init__(self) -> None:
        for name in ("period_x", "period_y", "substrate_height", "superstrate_height", "mesh_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.mesh_size_bot is not None and self.mesh_size_bot <= 0:
            raise ValueError(f"mesh_size_bot must be positive, got {self.mesh_size_bot}")
        if self.grating_layer_thickness is not None:
            half = self.grating_layer_thickness / 2.0
            if not 0 < half < min(self.substrate_height, self.superstrate_height):
                raise ValueError(
                    "grating_layer_thickness must be positive and its half strictly less "
                    "than both substrate_height and superstrate_height"
                )

    @property
    def effective_mesh_size_bot(self) -> float:
        """Mesh size on the substrate side (falls back to ``mesh_size``)."""
        return self.mesh_size if self.mesh_size_bot is None else self.mesh_size_bot

    @property
    def z_top(self) -> float:
        """z of the top face of the box."""
        return self.superstrate_height

    @property
    def z_bot(self) -> float:
        """z of the bottom face of the box."""
        return -self.substrate_height

    def z_layers(self) -> list[tuple[str, float, float]]:
        """``(volume name, z_low, z_high)`` of every layer, bottom to top."""
        zs, zt = -self.substrate_height, self.superstrate_height
        if self.grating_layer_thickness is None:
            return [("vol_bot", zs, 0.0), ("vol_top", 0.0, zt)]
        h = self.grating_layer_thickness / 2.0
        return [("vol_bot", zs, -h), ("grating_layer", -h, h), ("vol_top", h, zt)]


def generate_mesh(params: BoxParams, output_path: Path | str) -> Path:
    """Build the box mesh and write it as MSH 2.2 ASCII.

    Args:
        params: Geometric and meshing parameters.
        output_path: Destination ``.msh`` file (parents are created if missing).

    Returns:
        The absolute path of the file just written.
    """
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box")
        _build(params)
        _mesh(params, output_path)
    finally:
        gmsh.finalize()
    # A coarse mesh may link periodic facets whose node-number order the
    # periodic map does not preserve (wrong for second-order elements, see
    # mpxlite.cells.periodic): renumber only then, so clean meshes stay
    # byte-identical.
    if not check_periodic_facet_order(output_path).ok:
        renumber_periodic(output_path, (params.period_x, 0.0), (0.0, params.period_y))
    return output_path


def _build(params: BoxParams) -> None:
    occ = gmsh.model.occ
    px, py = params.period_x, params.period_y
    hx, hy = px / 2, py / 2
    layers = params.z_layers()
    for _, z0, z1 in layers:
        occ.addBox(-hx, -hy, z0, px, py, z1 - z0)
    occ.fragment([(3, t) for t in range(1, len(layers) + 1)], [])
    occ.synchronize()
    e = 1e-6

    def inbox(
        x0: float, y0: float, z0: float, x1: float, y1: float, z1: float, dim: int
    ) -> list[int]:
        return [
            t
            for _, t in gmsh.model.getEntitiesInBoundingBox(
                x0 - e, y0 - e, z0 - e, x1 + e, y1 + e, z1 + e, dim=dim
            )
        ]

    for name, z0, z1 in layers:
        vols = inbox(-hx, -hy, z0, hx, hy, z1, 3)
        if len(vols) != 1:
            raise RuntimeError(f"expected 1 volume in z=({z0}, {z1}), got {len(vols)}")
        gmsh.model.addPhysicalGroup(3, vols, TAGS[name], name)
    zb, zt = params.z_bot, params.z_top
    gmsh.model.addPhysicalGroup(
        2, inbox(-hx, -hy, zb, hx, hy, zb, 2), TAGS["outer_bot"], "outer_bot"
    )
    gmsh.model.addPhysicalGroup(
        2, inbox(-hx, -hy, zt, hx, hy, zt, 2), TAGS["outer_top"], "outer_top"
    )

    def zsort(tags: list[int]) -> list[int]:
        keyed = [(sum(gmsh.model.getBoundingBox(2, t)[2::3]) / 2, t) for t in tags]
        return [t for _, t in sorted(keyed)]

    xm = zsort(inbox(-hx, -hy, zb, -hx, hy, zt, 2))
    xp = zsort(inbox(hx, -hy, zb, hx, hy, zt, 2))
    ym = zsort(inbox(-hx, -hy, zb, hx, -hy, zt, 2))
    yp = zsort(inbox(-hx, hy, zb, hx, hy, zt, 2))
    for name, tags in (
        ("periodic_xm", xm),
        ("periodic_xp", xp),
        ("periodic_ym", ym),
        ("periodic_yp", yp),
    ):
        gmsh.model.addPhysicalGroup(2, tags, TAGS[name], name)
    if len(xm) != len(xp) or len(ym) != len(yp):
        raise RuntimeError("periodic-face count mismatch")
    tx = [1, 0, 0, px, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    ty = [1, 0, 0, 0, 0, 1, 0, py, 0, 0, 1, 0, 0, 0, 0, 1]
    for sm, sp in zip(xm, xp, strict=True):
        gmsh.model.mesh.setPeriodic(2, [sp], [sm], tx)
    for sm, sp in zip(ym, yp, strict=True):
        gmsh.model.mesh.setPeriodic(2, [sp], [sm], ty)


def _mesh(params: BoxParams, output_path: Path) -> None:
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay 2D
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay 3D (built-in)
    gmsh.option.setNumber("Mesh.Optimize", 0)  # preserve periodic node positions
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    h_top, h_bot = params.mesh_size, params.effective_mesh_size_bot
    if abs(h_top - h_bot) < 1e-15:
        gmsh.option.setNumber("Mesh.MeshSizeMin", h_top)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h_top)
    else:
        top = [(3, t) for t in gmsh.model.getEntitiesForPhysicalGroup(3, TAGS["vol_top"])]
        bot = [
            (3, t)
            for name in ("vol_bot", "grating_layer")
            if (name != "grating_layer" or params.grating_layer_thickness is not None)
            for t in gmsh.model.getEntitiesForPhysicalGroup(3, TAGS[name])
        ]
        pts_bot = gmsh.model.getBoundary(bot, combined=False, recursive=True)
        pts_top = gmsh.model.getBoundary(top, combined=False, recursive=True)
        gmsh.model.mesh.setSize(pts_top, h_top)
        gmsh.model.mesh.setSize(pts_bot, h_bot)  # the shared interface takes the finer side
        if h_top < h_bot:
            shared = [p for p in pts_top if p in pts_bot]
            gmsh.model.mesh.setSize(shared, h_top)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(h_top, h_bot))
        gmsh.option.setNumber("Mesh.MeshSizeMax", max(h_top, h_bot))
    gmsh.model.mesh.generate(3)
    gmsh.write(str(output_path))
