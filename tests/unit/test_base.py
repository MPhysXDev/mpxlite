# Copyright (c) 2025 Thierry Valet
# SPDX-License-Identifier: MIT

"""Unit tests for mpxlite.base.SolverLite."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from mpxlite.base import (
    _CASE_EXTENSIONS,
    _MANIFEST_NAME,
    _WORKFLOW_NAME,
    GetDPError,
    SolverLite,
)
from mpxlite.workflow import WorkflowData

# ----- helpers ------------------------------------------------------------


def _write_msh22(path: Path) -> None:
    path.write_text(
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n0\n$EndNodes\n$Elements\n0\n$EndElements\n",
        encoding="utf-8",
    )


def _scaffold_case(directory: Path, name: str) -> Path:
    """Create case files (geo/physprop/cond) and a valid .msh; return the mesh path."""
    for ext in _CASE_EXTENSIONS:
        (directory / f"{name}{ext}").write_text(f"// {name}{ext}\n", encoding="utf-8")
    mesh = directory / f"{name}.msh"
    _write_msh22(mesh)
    return mesh


@pytest.fixture
def case(tmp_path: Path) -> tuple[Path, Path]:
    mesh = _scaffold_case(tmp_path, "demo")
    return tmp_path, mesh


def _patch_run_ok(mocker: MockerFixture) -> object:
    return mocker.patch(
        "mpxlite.base.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
    )


def _patch_resolve_getdp(mocker: MockerFixture, path: str = "/usr/bin/getdp") -> None:
    mocker.patch(
        "mpxlite.base.SolverLite._resolve_getdp",
        return_value=path,
    )


# ----- construction & validation ------------------------------------------


def test_construct_resolves_paths(case: tuple[Path, Path]) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    assert s.project_dir == project_dir.resolve()
    assert s.mesh == mesh.resolve()
    assert s.resolution_id == "myres"
    assert isinstance(s.workflow, WorkflowData)


def test_workflow_can_be_supplied(case: tuple[Path, Path]) -> None:
    project_dir, mesh = case
    wf = WorkflowData({"X": 1.0})
    s = SolverLite(project_dir, "demo", mesh.name, "myres", workflow=wf)
    assert s.workflow is wf


def test_missing_project_dir_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "nope"
    with pytest.raises(FileNotFoundError, match="Project directory"):
        SolverLite(bogus, "demo", "mesh.msh", "myres")


def test_missing_case_file_raises(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.msh"
    _write_msh22(mesh)
    with pytest.raises(FileNotFoundError, match="case file missing"):
        SolverLite(tmp_path, "demo", "mesh.msh", "myres")


def test_missing_mesh_raises(tmp_path: Path) -> None:
    for ext in _CASE_EXTENSIONS:
        (tmp_path / f"demo{ext}").write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Mesh file"):
        SolverLite(tmp_path, "demo", "nonexistent.msh", "myres")


def test_unsupported_mesh_version_raises(case: tuple[Path, Path]) -> None:
    project_dir, mesh = case
    mesh.write_text("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported by GetDP"):
        SolverLite(project_dir, "demo", mesh.name, "myres")


def test_unrecognized_mesh_format_raises(case: tuple[Path, Path]) -> None:
    project_dir, mesh = case
    mesh.write_text("not a mesh\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unrecognized mesh format"):
        SolverLite(project_dir, "demo", mesh.name, "myres")


def test_unparseable_mesh_version_raises(case: tuple[Path, Path]) -> None:
    project_dir, mesh = case
    mesh.write_text("$MeshFormat\nNOT_A_NUMBER 0 8\n$EndMeshFormat\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot parse MSH version"):
        SolverLite(project_dir, "demo", mesh.name, "myres")


def test_legacy_msh_v1_format_accepted(case: tuple[Path, Path]) -> None:
    project_dir, mesh = case
    mesh.write_text("$NOD\n0\n$ENDNOD\n", encoding="utf-8")
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    assert s.mesh.is_file()


def test_missing_resource_file_raises(tmp_path: Path) -> None:
    mesh = _scaffold_case(tmp_path, "demo")
    bogus = tmp_path / "no_such_resource.pro"
    with pytest.raises(FileNotFoundError, match=r"Resource \.pro file"):
        SolverLite(
            tmp_path,
            "demo",
            mesh.name,
            "myres",
            resource_files=[bogus],
        )


def test_absolute_mesh_path_accepted(case: tuple[Path, Path]) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", str(mesh), "myres")
    assert s.mesh == mesh.resolve()


# ----- manifest writing ---------------------------------------------------


def test_manifest_includes_in_correct_order(tmp_path: Path) -> None:
    mesh = _scaffold_case(tmp_path, "demo")
    sub = tmp_path / "sub"
    sub.mkdir()
    res1 = sub / "fs.pro"
    res1.write_text("// functionspaces\n", encoding="utf-8")
    res2 = sub / "extra.pro"
    res2.write_text("// extra\n", encoding="utf-8")

    s = SolverLite(
        tmp_path,
        "demo",
        mesh.name,
        "myres",
        resource_files=[res1, res2],
    )
    s._write_workflow_pro()
    manifest = s._write_manifest()

    text = manifest.read_text(encoding="utf-8")
    expected_order = [
        f'Include "{_WORKFLOW_NAME}"',
        'Include "demo.geo.pro"',
        f'Include "{res1.resolve()}"',
        f'Include "{res2.resolve()}"',
        'Include "demo.physprop.pro"',
        'Include "demo.cond.pro"',
    ]
    last_pos = -1
    for line in expected_order:
        pos = text.find(line)
        assert pos > last_pos, f"Manifest line not in expected order:\n{text}"
        last_pos = pos


def test_workflow_pro_written(case: tuple[Path, Path]) -> None:
    project_dir, mesh = case
    wf = WorkflowData({"FOO": 1.5})
    s = SolverLite(project_dir, "demo", mesh.name, "myres", workflow=wf)
    path = s._write_workflow_pro()
    assert "FOO = 1.5;" in path.read_text(encoding="utf-8")


# ----- run() end-to-end API ------------------------------------------------


def test_run_no_postop_emits_pre_cal_only(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake_run = _patch_run_ok(mocker)
    _patch_resolve_getdp(mocker)

    result = s.run()

    cmd = fake_run.call_args[0][0]
    assert "-pre" in cmd and "myres" in cmd and "-cal" in cmd
    assert "-pos" not in cmd
    # Returned SolverResult is well-formed.
    from mpxlite import SolverResult

    assert isinstance(result, SolverResult)
    assert result.work_dir == project_dir.resolve()
    assert result.completed.returncode == 0


def test_run_single_postop(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake_run = _patch_run_ok(mocker)
    _patch_resolve_getdp(mocker)

    s.run("po_orders")

    cmd = fake_run.call_args[0][0]
    pos_indices = [i for i, t in enumerate(cmd) if t == "-pos"]
    # Exactly one -pos: GetDP CLI accepts only one per invocation, so the
    # API takes a single str; multiple Print[] calls are bundled at the
    # .pro level via composite PostOperations.
    assert len(pos_indices) == 1
    assert cmd[pos_indices[0] + 1] == "po_orders"
    assert cmd.index("-cal") < pos_indices[0]


def test_run_explicit_none_means_no_postop(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake_run = _patch_run_ok(mocker)
    _patch_resolve_getdp(mocker)

    s.run(None)

    cmd = fake_run.call_args[0][0]
    assert "-pos" not in cmd


def test_run_writes_manifest_and_workflow(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_run_ok(mocker)
    _patch_resolve_getdp(mocker)

    s.run("po_a")

    assert (project_dir / _MANIFEST_NAME).is_file()
    assert (project_dir / _WORKFLOW_NAME).is_file()


def test_run_single_subprocess_call_per_invocation(
    case: tuple[Path, Path], mocker: MockerFixture
) -> None:
    """One solver.run() == one subprocess.run() == one mpirun launch."""
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake_run = _patch_run_ok(mocker)
    _patch_resolve_getdp(mocker)

    s.run("po_orders")

    assert fake_run.call_count == 1


def test_getdp_failure_raises_with_diagnostic(
    case: tuple[Path, Path], mocker: MockerFixture
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    mocker.patch(
        "mpxlite.base.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["getdp"],
            returncode=2,
            stdout="",
            stderr="Bad input on line 42",
        ),
    )
    _patch_resolve_getdp(mocker)
    with pytest.raises(GetDPError, match="exited with code 2") as exc_info:
        s.run()
    assert exc_info.value.returncode == 2
    assert "line 42" in exc_info.value.stderr


def test_verbosity_flag_passed(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres", verbosity=5)
    fake_run = _patch_run_ok(mocker)
    _patch_resolve_getdp(mocker)
    s.run()
    cmd = fake_run.call_args[0][0]
    v_index = cmd.index("-v")
    assert cmd[v_index + 1] == "5"


# ----- _resolve_getdp -----------------------------------------------------


def test_resolve_getdp_via_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "fakedp"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setenv("MPXLITE_GETDP", str(fake))
    assert SolverLite._resolve_getdp() == str(fake)


def test_resolve_getdp_env_must_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPXLITE_GETDP", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="not an executable"):
        SolverLite._resolve_getdp()


def test_resolve_getdp_defaults_to_bundled_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """When MPXLITE_GETDP is not set, the bundled getdp_runner.sh is used."""
    monkeypatch.delenv("MPXLITE_GETDP", raising=False)
    import mpxlite
    assert SolverLite._resolve_getdp() == str(mpxlite.WRAPPER_PATH)


def test_resolve_getdp_falls_back_to_path_when_wrapper_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the wrapper is somehow not on disk, fall back to bare getdp on PATH."""
    monkeypatch.delenv("MPXLITE_GETDP", raising=False)
    # Simulate a missing wrapper (e.g. wheel unpacked, shell bit lost).
    monkeypatch.setattr("mpxlite.WRAPPER_PATH", tmp_path / "missing.sh")
    monkeypatch.setattr(
        "mpxlite.base.shutil.which",
        lambda _name: "/usr/bin/getdp",
    )
    assert SolverLite._resolve_getdp() == "/usr/bin/getdp"


def test_resolve_getdp_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither wrapper nor bare getdp -> raise."""
    monkeypatch.delenv("MPXLITE_GETDP", raising=False)
    monkeypatch.setattr("mpxlite.WRAPPER_PATH", tmp_path / "missing.sh")
    monkeypatch.setattr("mpxlite.base.shutil.which", lambda _name: None)
    with pytest.raises(FileNotFoundError, match="bundled getdp_runner.sh"):
        SolverLite._resolve_getdp()
