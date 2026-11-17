# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""A tagged cell read from a mesh written by a user's script with Gmsh.

:meth:`~mpxlite.cells.detect.TaggedCell.from_msh` reads MSH 2.x ASCII and
never touches Gmsh: the caller's live model and current model are unchanged
by the read. An MSH 4.1 file is refused with the advice to write MSH 2.2;
the same mesh written as MSH 2.2 reads with its physical group.
"""

from __future__ import annotations

from pathlib import Path

import gmsh
import pytest

from mpxlite.cells import UnsupportedMeshFormatError
from mpxlite.cells.detect import TaggedCell

SQUARE = ((1.0, 0.0), (0.0, 1.0))


def test_from_msh_leaves_the_callers_gmsh_model(tmp_path: Path) -> None:
    v41, v22 = tmp_path / "c41.msh", tmp_path / "c22.msh"
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("cell")
        box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [box], 1, "v")
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
        gmsh.model.mesh.generate(3)
        n_tets = sum(len(t) for t in gmsh.model.mesh.getElements(3)[1])
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.write(str(v41))
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(v22))
        gmsh.model.add("mine")
        gmsh.model.occ.addBox(0, 0, 0, 2, 2, 2)
        gmsh.model.occ.synchronize()
        models = gmsh.model.list()
        with pytest.raises(UnsupportedMeshFormatError, match=r"MSH 2\.2"):
            TaggedCell.from_msh(v41, SQUARE)
        cell = TaggedCell.from_msh(v22, SQUARE)
        assert cell.groups[1].dim == 3 and cell.groups[1].name == "v"
        assert len(cell.groups[1].cells) == n_tets
        assert gmsh.model.getCurrent() == "mine" and gmsh.model.list() == models
        assert gmsh.model.getBoundingBox(3, 1)[3:] == pytest.approx((2.0, 2.0, 2.0))
    finally:
        gmsh.finalize()
