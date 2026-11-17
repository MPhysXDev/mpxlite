# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""Periodic cells: lattices, symmetry, asymmetric units, periodic meshes (pure NumPy).

Nothing here calls Gmsh: these modules work on lattices, on tagged meshes and
on mesh files. The Gmsh side of a periodic cell (making its parts conform,
measuring it, cutting and meshing its asymmetric unit) is done by the Gmsh
jobs of mpxlite (:func:`mpxlite.run_gmsh_job`), which use these modules.

* :mod:`mpxlite.cells.periodic` -- node-order consistency of the periodic
  (``$Periodic``) facet pairs of a Gmsh mesh, required by second-order
  elements whose basis depends on the global node numbering:
  :func:`check_periodic_facet_order` (diagnosis, content-hash cached),
  :func:`renumber_periodic` and :func:`align_periodic_facets` (repair);
  :func:`measure_physical_groups`, the per-physical-group z-extent and
  triangle size (same parse and cache); :func:`measure_z_faces`, the mesh
  size of the lowest / highest planes (the top and bottom faces of the box).
* :mod:`mpxlite.cells.topology` -- :func:`check_mesh_topology`: orphan nodes
  and boundary triangles detached from the volume
  (:class:`MeshTopologyError`).
* :mod:`mpxlite.cells.connectivity` -- the element connectivity of a mesh
  as arrays (``read_connectivity``).
* :mod:`mpxlite.cells.symmetric` -- exactly C4v-symmetric meshes of a square
  periodic cell by wedge replication (:func:`build_c4v_mesh`), and the
  generic replication helpers of any plane group and lattice
  (:func:`replicate_group`, ``bloch_faces``, ``write_msh22_periodic``).
* :mod:`mpxlite.cells.spacegroup` -- space-group operations of a periodic
  cell (:class:`~mpxlite.cells.spacegroup.LatticeOp`) and a node numbering
  they preserve (:func:`~mpxlite.cells.spacegroup.renumber_symmetric`).
* :mod:`mpxlite.cells.symmetry_gate` -- :func:`require_mesh_symmetry`, the
  check that a mesh has its declared symmetry (nodes, elements, tags, node
  order), run before every exact use of the symmetry.
* :mod:`mpxlite.cells.detect` --
  :func:`~mpxlite.cells.detect.detect_symmetry`, the space group of a tagged
  cell (lattice holohedry, translations from the tagged components,
  verification on every tag, tag permutation, wallpaper symbol, near
  symmetries reported).
* :mod:`mpxlite.cells.asymmetric` --
  :func:`~mpxlite.cells.asymmetric.asymmetric_unit`, the ITA asymmetric unit
  of a rectangular / square plane group in the user's frame (edges: mirrors
  or paired by an operation; cut segments on the window-line preimages).
* :mod:`mpxlite.cells.autosym` -- the NumPy side of the symmetric periodic
  mesh of a tagged cell (:class:`MeshSizing`, the subgroup used, the size
  field, the writing and numbering of the replicated mesh,
  :class:`SymmetricMeshReport`); the job ``periodic_cell_mesh`` does the
  rest.
* :mod:`mpxlite.cells.tiling` -- :func:`~mpxlite.cells.tiling.tile_periodic`,
  macrocell meshes as periodic tilings of a cell mesh.

Formats: Gmsh MSH 2.2 ASCII is the production format (what GetDP reads and
what every writer of this package emits). The checks that only read --
:func:`check_periodic_facet_order`, :func:`measure_physical_groups`,
:func:`measure_z_faces`, :func:`check_mesh_topology`,
``mpxlite.cells.connectivity.read_connectivity`` -- also read MSH 4.1
ASCII. :meth:`TaggedCell.from_msh` and every tool that rewrites a mesh or
checks its symmetry node by node -- :func:`renumber_periodic`,
:func:`align_periodic_facets`, :func:`renumber_symmetric`,
:func:`check_symmetric_numbering`, the symmetry gate
(:func:`check_mesh_symmetry`, :func:`require_mesh_symmetry`),
:func:`~mpxlite.cells.tiling.tile_periodic` -- read MSH 2.x ASCII only and
refuse another format with :class:`UnsupportedMeshFormatError`, whose
message gives the conversion.
"""

from __future__ import annotations

from mpxlite.cells.asymmetric import AsymmetricUnit, AsymmetricUnitError, asymmetric_unit
from mpxlite.cells.autosym import MeshSizing, SymmetricMeshError, SymmetricMeshReport
from mpxlite.cells.detect import DetectedSymmetry, TaggedCell, detect_symmetry
from mpxlite.cells.periodic import (
    BadFacet,
    MeshGroups,
    PeriodicFacetOrderError,
    PeriodicFacetReport,
    PhysicalGroupMesh,
    UnsupportedMeshFormatError,
    ZFaceMesh,
    ZPlaneMesh,
    align_periodic_facets,
    check_periodic_facet_order,
    measure_physical_groups,
    measure_z_faces,
    renumber_periodic,
)
from mpxlite.cells.spacegroup import (
    LatticeOp,
    check_symmetric_numbering,
    lattice_group,
    renumber_symmetric,
)
from mpxlite.cells.symmetric import build_c4v_mesh, replicate_c4v, replicate_group
from mpxlite.cells.symmetry_gate import (
    MeshSymmetryError,
    MeshSymmetryReport,
    SymmetryGateSkipped,
    check_mesh_symmetry,
    require_mesh_symmetry,
)
from mpxlite.cells.tiling import TilingReport, tile_periodic
from mpxlite.cells.topology import (
    MeshTopologyError,
    MeshTopologyReport,
    MeshTopologyWarning,
    check_mesh_topology,
)

__all__ = [
    "AsymmetricUnit",
    "AsymmetricUnitError",
    "BadFacet",
    "DetectedSymmetry",
    "LatticeOp",
    "MeshGroups",
    "MeshSizing",
    "MeshSymmetryError",
    "MeshSymmetryReport",
    "MeshTopologyError",
    "MeshTopologyReport",
    "MeshTopologyWarning",
    "PeriodicFacetOrderError",
    "PeriodicFacetReport",
    "PhysicalGroupMesh",
    "SymmetricMeshError",
    "SymmetricMeshReport",
    "SymmetryGateSkipped",
    "TaggedCell",
    "TilingReport",
    "UnsupportedMeshFormatError",
    "ZFaceMesh",
    "ZPlaneMesh",
    "align_periodic_facets",
    "asymmetric_unit",
    "build_c4v_mesh",
    "check_mesh_symmetry",
    "check_mesh_topology",
    "check_periodic_facet_order",
    "check_symmetric_numbering",
    "detect_symmetry",
    "lattice_group",
    "measure_physical_groups",
    "measure_z_faces",
    "renumber_periodic",
    "renumber_symmetric",
    "replicate_c4v",
    "replicate_group",
    "require_mesh_symmetry",
    "tile_periodic",
]
