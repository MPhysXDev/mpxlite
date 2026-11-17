# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""The runner of Gmsh jobs (:func:`mpxlite.run_gmsh_job`) on the real jobs.

A time limit, invalid input files, an element limit, a memory cap, and what
a successful run reports (Gmsh version, messages, the run folder).
"""

from __future__ import annotations

import json
from pathlib import Path

import gmsh
import pytest

import mpxlite
from tests.integration.gmsh_cells import (
    FR4,
    geometry_job,
    mesh_job,
    mesh_parameters,
    name,
    write_xao,
)

MM = 1e-3


def _slab() -> None:
    occ = gmsh.model.occ
    name(
        {"substrate": occ.addBox(0, 0, 0, 6 * MM, 6 * MM, 1 * MM)},
        {"ground": occ.addRectangle(0, 0, 0, 6 * MM, 6 * MM)},
    )


def test_successful_run_reports(tmp_path: Path) -> None:
    xao = write_xao(tmp_path / "slab.xao", _slab)
    result = geometry_job(tmp_path, xao)
    assert result.gmsh_version == "4.15.2"
    assert isinstance(result.messages, tuple)
    assert all(level in ("warning", "error", "info") for level, _ in result.messages)
    assert result.log_path.is_file() and result.elapsed_s > 0
    doc = json.loads((result.run_dir / "report.json").read_text())
    assert doc["format"] == "mpxlite-gmsh-job-report" and doc["version"] == 1
    assert doc["status"] == "ok" and doc["gmsh"] == "4.15.2"
    request = json.loads((result.run_dir / "request.json").read_text())
    assert request["inputs"] == {"geometry": str(xao.resolve())}
    assert set(result.outputs) == {"geometry"}
    assert result.outputs["geometry"].parent == result.run_dir


def test_time_limit(tmp_path: Path) -> None:
    xao = write_xao(tmp_path / "slab.xao", _slab)
    with pytest.raises(mpxlite.GmshJobError) as caught:
        mpxlite.run_gmsh_job(
            "periodic_cell_geometry",
            inputs={"geometry": xao},
            runs_dir=tmp_path / "jobs",
            timeout=0.001,
        )
    assert (caught.value.kind, caught.value.code) == ("timeout", "job.timeout")
    assert caught.value.run_dir is not None and caught.value.run_dir.is_dir()


def test_input_not_xao(tmp_path: Path) -> None:
    path = tmp_path / "cell.brep"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(mpxlite.GmshJobError) as caught:
        geometry_job(tmp_path, path)
    assert (caught.value.kind, caught.value.code) == ("input", "input.format")
    assert caught.value.data["input"] == "geometry"


def test_input_unreadable_xao(tmp_path: Path) -> None:
    path = tmp_path / "cell.xao"
    path.write_text("<not an XAO document", encoding="utf-8")
    with pytest.raises(mpxlite.GmshJobError) as caught:
        geometry_job(tmp_path, path)
    assert (caught.value.kind, caught.value.code) == ("input", "input.unreadable")


def test_missing_input(tmp_path: Path) -> None:
    with pytest.raises(mpxlite.GmshJobError) as caught:
        mpxlite.run_gmsh_job("periodic_cell_geometry", runs_dir=tmp_path / "jobs")
    assert (caught.value.kind, caught.value.code) == ("input", "input.missing")


def test_element_limit(tmp_path: Path) -> None:
    xao = geometry_job(tmp_path, write_xao(tmp_path / "slab.xao", _slab)).outputs["geometry"]
    params = mesh_parameters({"substrate": FR4}, ["ground"])
    params["mesh"]["max_elements"] = 10
    with pytest.raises(mpxlite.GmshJobError) as caught:
        mesh_job(tmp_path, xao, **params)
    err = caught.value
    assert (err.kind, err.code) == ("gmsh", "mesh.too_many_elements")
    assert err.data["limit"] == 10 and err.data["when"] in ("before", "after")


def test_absurd_size_is_refused_before_meshing(tmp_path: Path) -> None:
    """A size a thousand times too small fails at once, before the generation."""
    xao = geometry_job(tmp_path, write_xao(tmp_path / "slab.xao", _slab)).outputs["geometry"]
    params = mesh_parameters({"substrate": FR4 / 1000}, ["ground"])
    with pytest.raises(mpxlite.GmshJobError) as caught:
        mesh_job(tmp_path, xao, **params)
    err = caught.value
    assert (err.kind, err.code) == ("gmsh", "mesh.too_many_elements")
    assert err.data["when"] == "before" and err.data["estimate"] > err.data["limit"]


def test_memory_cap(tmp_path: Path) -> None:
    if not mpxlite.hard_cap_available():  # probed when the test runs, not at collection
        pytest.skip("no hard memory cap here")
    xao = write_xao(tmp_path / "slab.xao", _slab)
    result = mpxlite.run_gmsh_job(
        "periodic_cell_geometry",
        inputs={"geometry": xao},
        runs_dir=tmp_path / "jobs",
        memory_max=2**30,
        timeout=600,
    )
    assert result.report["cell"]["sides"] == pytest.approx((6 * MM, 6 * MM))
