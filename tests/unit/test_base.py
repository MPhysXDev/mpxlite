# Copyright (c) 2025 MPhysX
# SPDX-License-Identifier: MIT

"""Unit tests for mpxlite.base.SolverLite (no real getdp: Popen is mocked)."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pytest_mock import MockerFixture

import mpxlite.base as base_mod
from mpxlite.base import (
    CASE_EXTENSIONS,
    LOG_NAME,
    MANIFEST_NAME,
    PETSC_OPTIONS_NAME,
    WORKFLOW_NAME,
    GetDPCancelledError,
    GetDPError,
    GetDPTimeoutError,
    SolverLite,
    _session_pids,
    _terminate_session,
)
from mpxlite.launch import LaunchSpec
from mpxlite.result import SolverResult
from mpxlite.workflow import WorkflowData

# ----- helpers ------------------------------------------------------------


def _write_msh22(path: Path) -> None:
    path.write_text(
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n0\n$EndNodes\n$Elements\n0\n$EndElements\n",
        encoding="utf-8",
    )


def _scaffold_case(directory: Path, name: str) -> Path:
    """Create case files (geo/physprop/cond) and a valid .msh; return the mesh path."""
    for ext in CASE_EXTENSIONS:
        (directory / f"{name}{ext}").write_text(f"// {name}{ext}\n", encoding="utf-8")
    mesh = directory / f"{name}.msh"
    _write_msh22(mesh)
    return mesh


@pytest.fixture
def case(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "proj"
    project.mkdir()
    mesh = _scaffold_case(project, "demo")
    return project, mesh


class FakePopen:
    """Stand-in for subprocess.Popen: writes ``output`` to stdout, exits ``returncode``.

    ``hang=True`` keeps it running until a signal is delivered to it through
    the :class:`FakeSession` (``ignore_term=True``: only SIGKILL is effective).
    """

    instances: list[FakePopen] = []  # noqa: RUF012

    def __init__(
        self,
        cmd: list[str],
        *,
        output: str = "",
        returncode: int = 0,
        hang: bool = False,
        ignore_term: bool = False,
        **kwargs: Any,
    ) -> None:
        self.cmd = cmd
        self.args = cmd
        self.kwargs = kwargs
        self.pid = 4242
        self.returncode: int | None = None
        self._rc = returncode
        self.hang = hang
        self.ignore_term = ignore_term
        self.signals: list[int] = []
        stdout = kwargs["stdout"]
        stdout.write(output.encode())
        stdout.flush()
        FakePopen.instances.append(self)

    def poll(self) -> int | None:
        if self.returncode is None:
            if not self.hang:
                self.returncode = self._rc
            elif self.signals:
                self.returncode = -self.signals[-1]
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        rc = self.poll()
        if rc is not None:
            return rc
        assert timeout is not None, "an unbounded wait on a hanging process"
        time.sleep(min(timeout, 0.01))
        rc = self.poll()
        if rc is None:
            raise subprocess.TimeoutExpired(self.cmd, timeout)
        return rc


class FakeSession:
    """The session of the latest FakePopen (pid 4242) plus extra ``ranks``.

    ``ranks`` maps the pid of a rank (own process group, same session) to
    whether it ignores SIGTERM. Every delivered signal is logged in
    ``received`` as ``(pid, sig)``; ``killpg`` is the no-/proc fallback.
    """

    def __init__(self) -> None:
        self.ranks: dict[int, bool] = {}
        self.received: list[tuple[int, int]] = []
        self.killpg_calls: list[tuple[int, int]] = []
        self.proc_available = True

    @property
    def leader(self) -> FakePopen:
        return FakePopen.instances[-1]

    def pids(self, sid: int) -> list[int] | None:
        if not self.proc_available:
            return None
        assert sid == self.leader.pid
        alive = [sid] if self.leader.poll() is None else []
        return alive + list(self.ranks)

    def kill(self, pid: int, sig: int) -> None:
        self.received.append((pid, sig))
        if pid == self.leader.pid:
            if sig == signal.SIGKILL or not self.leader.ignore_term:
                self.leader.signals.append(sig)
            return
        if pid not in self.ranks:
            raise ProcessLookupError
        if sig == signal.SIGKILL or not self.ranks[pid]:
            del self.ranks[pid]

    def killpg(self, pgid: int, sig: int) -> None:
        self.killpg_calls.append((pgid, sig))
        self.kill(pgid, sig)


@pytest.fixture
def session(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> FakeSession:
    """Fake /proc session scan and signal delivery; short grace periods."""
    fake = FakeSession()
    mocker.patch("mpxlite.base._session_pids", side_effect=fake.pids)
    mocker.patch("mpxlite.base._pid_in_use", return_value=False)  # pid 4242 never recycled
    mocker.patch("mpxlite.base.os.kill", side_effect=fake.kill)
    mocker.patch("mpxlite.base.os.killpg", side_effect=fake.killpg)
    monkeypatch.setattr(base_mod, "_TERM_GRACE_S", 0.3)
    monkeypatch.setattr(base_mod, "_KILL_WAIT_S", 0.5)
    monkeypatch.setattr(base_mod, "_POLL_S", 0.02)
    return fake


def _patch_popen(mocker: MockerFixture, **behaviour: Any) -> Any:
    FakePopen.instances = []
    return mocker.patch(
        "mpxlite.base.subprocess.Popen",
        side_effect=lambda cmd, **kw: FakePopen(cmd, **behaviour, **kw),
    )


def _patch_resolve_getdp(mocker: MockerFixture, path: str = "/usr/bin/getdp") -> Any:
    return mocker.patch("mpxlite.base.SolverLite._resolve_getdp", return_value=path)


def _cmd(fake: Any) -> list[str]:
    return fake.call_args[0][0]


# ----- construction & validation ------------------------------------------


def test_construct_resolves_paths(case: tuple[Path, Path]) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    assert s.project_dir == project_dir.resolve()
    assert s.mesh == mesh.resolve()
    assert s.resolution_id == "myres"
    assert s.runs_dir == project_dir.resolve() / "runs"
    assert isinstance(s.workflow, WorkflowData)


def test_runs_dir_not_created_at_construction(case: tuple[Path, Path]) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    assert not s.runs_dir.exists()


def test_custom_runs_dir(case: tuple[Path, Path], tmp_path: Path) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres", runs_dir=tmp_path / "out")
    assert s.runs_dir == (tmp_path / "out").resolve()


def test_workflow_can_be_supplied(case: tuple[Path, Path]) -> None:
    project_dir, mesh = case
    wf = WorkflowData({"X": 1.0})
    s = SolverLite(project_dir, "demo", mesh.name, "myres", workflow=wf)
    assert s.workflow is wf


def test_missing_project_dir_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "nope"
    with pytest.raises(FileNotFoundError, match="Project directory"):
        SolverLite(bogus, "demo", "mesh.msh", "myres")


def test_missingcase_file_raises(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.msh"
    _write_msh22(mesh)
    with pytest.raises(FileNotFoundError, match="case file missing"):
        SolverLite(tmp_path, "demo", "mesh.msh", "myres")


def test_missing_mesh_raises(tmp_path: Path) -> None:
    for ext in CASE_EXTENSIONS:
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


def test_missing_resource_file_raises(case: tuple[Path, Path]) -> None:
    project_dir, mesh = case
    bogus = project_dir / "no_such_resource.pro"
    with pytest.raises(FileNotFoundError, match=r"Resource \.pro file"):
        SolverLite(project_dir, "demo", mesh.name, "myres", resource_files=[bogus])


def test_absolute_mesh_path_accepted(case: tuple[Path, Path]) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", str(mesh), "myres")
    assert s.mesh == mesh.resolve()


def test_quote_in_include_path_rejected(tmp_path: Path) -> None:
    project = tmp_path / 'we"ird'
    project.mkdir()
    mesh = _scaffold_case(project, "demo")
    with pytest.raises(ValueError, match="Include string"):
        SolverLite(project, "demo", mesh.name, "myres")


# ----- run(): layout and manifest -----------------------------------------


def test_run_dir_layout(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake = _patch_popen(mocker, output="hello from getdp\n")
    _patch_resolve_getdp(mocker)

    result = s.run("po_a", run_id="r1")

    run_dir = project_dir.resolve() / "runs" / "r1"
    assert isinstance(result, SolverResult)
    assert result.work_dir == run_dir
    assert result.run_id == "r1"
    assert result.returncode == 0
    assert result.log_path == run_dir / LOG_NAME
    assert result.log_path.read_text() == "hello from getdp\n"
    assert result.path("eff.dat") == run_dir / "eff.dat"
    assert result.path("sub/eff.dat") == run_dir / "sub" / "eff.dat"
    for escaping in ("../eff.dat", "sub/../../eff.dat", "/tmp/eff.dat"):
        with pytest.raises(ValueError, match="inside the run directory"):
            result.path(escaping)
    assert result.elapsed_s >= 0.0
    assert result.cmd == tuple(_cmd(fake))
    assert sorted(p.name for p in run_dir.iterdir()) == sorted(
        [LOG_NAME, MANIFEST_NAME, WORKFLOW_NAME]
    )
    # Shared inputs untouched: nothing new in project_dir besides runs/.
    assert sorted(p.name for p in project_dir.iterdir()) == sorted(
        ["demo.cond.pro", "demo.geo.pro", "demo.physprop.pro", "demo.msh", "runs"]
    )
    kwargs = fake.call_args.kwargs
    assert kwargs["cwd"] == str(run_dir)
    assert kwargs["stderr"] is subprocess.STDOUT
    assert kwargs["start_new_session"] is True
    # mpirun forwards (and consumes) the parent's stdin: never give it ours.
    assert kwargs["stdin"] is subprocess.DEVNULL


def test_run_dir_under_custom_runs_dir(
    case: tuple[Path, Path], tmp_path: Path, mocker: MockerFixture
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres", runs_dir=tmp_path / "a" / "b")
    _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    result = s.run(run_id="x")
    assert result.work_dir == (tmp_path / "a" / "b" / "x").resolve()
    assert not (project_dir / "runs").exists()


def test_manifest_absolute_includes_in_order(
    case: tuple[Path, Path], mocker: MockerFixture
) -> None:
    project_dir, mesh = case
    sub = project_dir / "sub"
    sub.mkdir()
    res1 = sub / "fs.pro"
    res1.write_text("// functionspaces\n", encoding="utf-8")
    res2 = sub / "extra.pro"
    res2.write_text("// extra\n", encoding="utf-8")
    s = SolverLite(project_dir, "demo", mesh.name, "myres", resource_files=[res1, res2])
    _patch_popen(mocker)
    _patch_resolve_getdp(mocker)

    result = s.run(run_id="r")

    text = result.path(MANIFEST_NAME).read_text(encoding="utf-8")
    pd = project_dir.resolve()
    expected = [
        f'Include "{result.work_dir / WORKFLOW_NAME}";',
        f'Include "{pd / "demo.geo.pro"}";',
        f'Include "{res1.resolve()}";',
        f'Include "{res2.resolve()}";',
        f'Include "{pd / "demo.physprop.pro"}";',
        f'Include "{pd / "demo.cond.pro"}";',
    ]
    include_lines = [ln for ln in text.splitlines() if ln.startswith("Include")]
    assert include_lines == expected


def test_default_run_id_format(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    r1 = s.run()
    r2 = s.run()
    import re

    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{8}", r1.run_id)
    assert r1.run_id != r2.run_id
    assert r1.work_dir.name == r1.run_id


def test_existing_run_id_raises(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake = _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    s.run(run_id="same")
    marker = s.runs_dir / "same" / "keep.dat"
    marker.write_text("x")
    with pytest.raises(FileExistsError):
        s.run(run_id="same")
    assert marker.read_text() == "x"
    assert fake.call_count == 1


@pytest.mark.parametrize(
    "bad", ["", ".", "..", "../escape", "a/b", "-a", "_a", "a b", "a\n", "é", ".hidden"]
)
def test_bad_run_id_rejected(case: tuple[Path, Path], mocker: MockerFixture, bad: str) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake = _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    with pytest.raises(ValueError, match="Invalid run_id"):
        s.run(run_id=bad)
    assert fake.call_count == 0
    assert not s.runs_dir.exists()


@pytest.mark.parametrize("good", ["a", "A1", "0", "run_1.v2-x", "9.a"])
def test_good_run_id_accepted(case: tuple[Path, Path], mocker: MockerFixture, good: str) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    assert s.run(run_id=good).work_dir.name == good


# ----- run(): workflow overrides & no mutation ------------------------------


def test_run_does_not_mutate_solver(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    wf = WorkflowData({"A": 1.0, "B": 2})
    s = SolverLite(project_dir, "demo", mesh.name, "myres", workflow=wf)
    _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    before = dict(s.__dict__)
    before_wf = dict(wf)

    s.run("po", workflow={"A": 5.0, "C": 3}, run_id="r", launch=LaunchSpec(np=1))

    assert dict(s.workflow) == before_wf
    assert s.workflow is wf
    assert s.__dict__ == before


def test_workflow_overrides_merged(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(
        project_dir, "demo", mesh.name, "myres", workflow=WorkflowData({"A": 1.0, "B": 2})
    )
    _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    result = s.run(workflow={"A": 5.0, "C": 3}, run_id="r")
    text = result.path(WORKFLOW_NAME).read_text(encoding="utf-8")
    assert "A = 5.0;" in text
    assert "B = 2;" in text
    assert "C = 3;" in text
    assert "A = 1.0;" not in text


def test_invalid_override_rejected_before_run_dir(
    case: tuple[Path, Path], mocker: MockerFixture
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake = _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    with pytest.raises(ValueError):
        s.run(workflow={"1bad": 1.0}, run_id="r")
    assert fake.call_count == 0
    assert not (s.runs_dir / "r").exists()


def test_concurrent_runs_isolated(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres", workflow=WorkflowData({"F": 0.0}))
    _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    results: dict[int, SolverResult] = {}

    def work(i: int) -> None:
        results[i] = s.run(workflow={"F": float(i)}, run_id=f"r{i}")

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i, r in results.items():
        assert f"F = {float(i)!r};" in r.path(WORKFLOW_NAME).read_text()
    assert dict(s.workflow) == {"F": 0.0}


# ----- run(): command line ------------------------------------------------


def test_run_no_postop_emits_pre_cal_only(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake = _patch_popen(mocker)
    _patch_resolve_getdp(mocker, "/drv")

    result = s.run(run_id="r")

    assert _cmd(fake) == [
        "/drv",
        str(result.work_dir / MANIFEST_NAME),
        "-msh",
        str(mesh.resolve()),
        "-v",
        "3",
        "-pre",
        "myres",
        "-cal",
    ]


def test_run_single_postop(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake = _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    s.run("po_orders")
    cmd = _cmd(fake)
    assert cmd[-3:] == ["-cal", "-pos", "po_orders"]


def test_run_multiple_postops(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake = _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    s.run(["po_a", "po_b", "po_c"])
    cmd = _cmd(fake)
    assert cmd.count("-pos") == 1
    assert cmd[-5:] == ["-cal", "-pos", "po_a", "po_b", "po_c"]


@pytest.mark.parametrize("empty", [None, [], ()])
def test_run_empty_postops_means_none(
    case: tuple[Path, Path], mocker: MockerFixture, empty: Any
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake = _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    s.run(empty)
    assert "-pos" not in _cmd(fake)


@pytest.mark.parametrize("bad", ["", "-x", "a b", ["ok", ""]])
def test_bad_postop_rejected(case: tuple[Path, Path], mocker: MockerFixture, bad: Any) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    with pytest.raises(ValueError, match="PostOperation"):
        s.run(bad)


def test_non_str_postop_rejected(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    with pytest.raises(TypeError):
        s.run(["a", 3])  # type: ignore[list-item]


def test_verbosity_flag_passed(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres", verbosity=5)
    fake = _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    s.run()
    cmd = _cmd(fake)
    assert cmd[cmd.index("-v") + 1] == "5"


def test_single_subprocess_per_run(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake = _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    s.run(["a", "b"])
    assert fake.call_count == 1


# ----- run(): environment -------------------------------------------------


def test_env_from_launch_spec(
    case: tuple[Path, Path], mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir, mesh = case
    monkeypatch.setenv("MPXLITE_GETDP_NP", "7")
    monkeypatch.setenv("SOME_PARENT_VAR", "kept")
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake = _patch_popen(mocker)
    resolve = _patch_resolve_getdp(mocker)

    s.run(launch=LaunchSpec(np=2, threads=3, cpus=(4, 5, 6, 7, 8, 9), env={"FOO": "bar"}))

    env = fake.call_args.kwargs["env"]
    assert env["MPXLITE_GETDP_NP"] == "2"
    assert env["MPXLITE_GETDP_THREADS"] == "3"
    assert env["MPXLITE_GETDP_CPUS"] == "4,5,6,7,8,9"
    assert env["FOO"] == "bar"
    assert env["SOME_PARENT_VAR"] == "kept"
    assert resolve.call_args[0][0]["MPXLITE_GETDP_NP"] == "2"  # the driver sees the run's env
    # Parent environment untouched.
    import os

    assert os.environ["MPXLITE_GETDP_NP"] == "7"
    assert "FOO" not in os.environ


def test_env_defaults_without_launch(
    case: tuple[Path, Path], mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir, mesh = case
    monkeypatch.setenv("MPXLITE_GETDP_NP", "3")
    monkeypatch.delenv("MPXLITE_GETDP_CPUS", raising=False)
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake = _patch_popen(mocker)
    resolve = _patch_resolve_getdp(mocker)
    s.run()
    env = fake.call_args.kwargs["env"]
    assert env["MPXLITE_GETDP_NP"] == "3"
    assert "MPXLITE_GETDP_CPUS" not in env
    assert resolve.call_args[0][0]["MPXLITE_GETDP_NP"] == "3"


# ----- run(): failures ----------------------------------------------------


def test_getdp_failure_raises_with_log_tail(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    lines = "".join(f"line {i}\n" for i in range(100)) + "Bad input on line 42\n"
    _patch_popen(mocker, output=lines, returncode=2)
    _patch_resolve_getdp(mocker)

    with pytest.raises(GetDPError, match="exited with code 2") as exc_info:
        s.run(run_id="bad")

    err = exc_info.value
    run_dir = s.runs_dir / "bad"
    assert not isinstance(err, GetDPTimeoutError)
    assert err.returncode == 2
    assert err.run_dir == run_dir
    assert err.log_path == run_dir / LOG_NAME
    assert "Bad input on line 42" in str(err)
    assert "line 99" in str(err)
    assert "line 50\n" not in str(err)  # only the last ~40 lines
    assert "Bad input on line 42" in err.log_tail
    assert run_dir.is_dir()  # kept for diagnosis


def test_timeout_kills_whole_session(
    case: tuple[Path, Path], mocker: MockerFixture, session: FakeSession
) -> None:
    """Ranks in their own process group (Open MPI setpgid) are signalled too."""
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker, output="started\n", hang=True)
    _patch_resolve_getdp(mocker)
    session.ranks = {5001: False, 5002: False}

    with pytest.raises(GetDPTimeoutError, match=r"timed out after 0\.2 s") as exc_info:
        s.run(run_id="slow", timeout=0.2)

    err = exc_info.value
    assert isinstance(err, GetDPError)
    assert err.timeout == 0.2
    assert err.run_dir == s.runs_dir / "slow"
    assert "started" in str(err)
    assert {(pid, signal.SIGTERM) for pid in (4242, 5001, 5002)} <= set(session.received)
    assert session.ranks == {}
    assert err.returncode == -signal.SIGTERM
    assert session.killpg_calls == []


def test_timeout_escalates_to_sigkill(
    case: tuple[Path, Path], mocker: MockerFixture, session: FakeSession
) -> None:
    """Processes ignoring SIGTERM are SIGKILLed after the grace period."""
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker, hang=True, ignore_term=True)
    _patch_resolve_getdp(mocker)
    session.ranks = {5001: True, 5002: False}
    t0 = time.monotonic()
    with pytest.raises(GetDPTimeoutError) as exc_info:
        s.run(timeout=0.1)
    assert time.monotonic() - t0 >= 0.1 + base_mod._TERM_GRACE_S
    assert (4242, signal.SIGKILL) in session.received
    assert (5001, signal.SIGKILL) in session.received
    assert (5002, signal.SIGKILL) not in session.received  # died on SIGTERM
    assert session.ranks == {}
    assert exc_info.value.returncode == -signal.SIGKILL


def test_kill_rescans_until_session_empty(
    case: tuple[Path, Path], mocker: MockerFixture, session: FakeSession
) -> None:
    """A survivor that appears after the first SIGKILL sweep is killed too."""
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker, hang=True, ignore_term=True)
    _patch_resolve_getdp(mocker)
    real_kill = session.kill
    spawned: list[int] = []

    def kill(pid: int, sig: int) -> None:
        real_kill(pid, sig)
        if pid == 4242 and sig == signal.SIGKILL and not spawned:
            spawned.append(6000)
            session.ranks[6000] = True  # e.g. forked while we were killing

    mocker.patch("mpxlite.base.os.kill", side_effect=kill)
    with pytest.raises(GetDPTimeoutError):
        s.run(timeout=0.05)
    assert (6000, signal.SIGKILL) in session.received
    assert session.ranks == {}


def test_kill_phase_is_bounded(
    case: tuple[Path, Path], mocker: MockerFixture, session: FakeSession
) -> None:
    """An unkillable process (e.g. D state) does not hang run() forever."""
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker, hang=True)
    _patch_resolve_getdp(mocker)
    session.ranks = {7000: True}
    mocker.patch(
        "mpxlite.base.os.kill",
        side_effect=lambda pid, sig: (
            session.received.append((pid, sig)) if pid == 7000 else session.kill(pid, sig)
        ),
    )
    t0 = time.monotonic()
    with pytest.raises(GetDPTimeoutError):
        s.run(timeout=0.05)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.05 + base_mod._TERM_GRACE_S + base_mod._KILL_WAIT_S + 1.0
    assert session.ranks == {7000: True}


def test_timeout_with_already_dead_session(
    case: tuple[Path, Path], mocker: MockerFixture, session: FakeSession
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker, hang=True)
    _patch_resolve_getdp(mocker)

    def gone(pid: int, sig: int) -> None:
        session.leader.signals.append(sig)
        raise ProcessLookupError

    mocker.patch("mpxlite.base.os.kill", side_effect=gone)
    with pytest.raises(GetDPTimeoutError):
        s.run(timeout=0.1)


def test_timeout_falls_back_to_killpg_without_proc(
    case: tuple[Path, Path], mocker: MockerFixture, session: FakeSession
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker, hang=True, ignore_term=True)
    _patch_resolve_getdp(mocker)
    session.proc_available = False
    with pytest.raises(GetDPTimeoutError) as exc_info:
        s.run(timeout=0.05)
    assert session.killpg_calls[0] == (4242, signal.SIGTERM)
    assert session.killpg_calls[-1] == (4242, signal.SIGKILL)
    assert exc_info.value.returncode == -signal.SIGKILL


def test_interrupt_kills_session(
    case: tuple[Path, Path], mocker: MockerFixture, session: FakeSession
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker, hang=True)
    _patch_resolve_getdp(mocker)
    session.ranks = {5001: False}
    mocker.patch("mpxlite.base._wait", side_effect=KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        s.run()
    assert session.received[:2] == [(4242, signal.SIGTERM), (5001, signal.SIGTERM)]
    assert session.ranks == {}


def test_interrupt_right_after_start_kills_session(
    case: tuple[Path, Path], mocker: MockerFixture, session: FakeSession
) -> None:
    """A signal between Popen returning and the run's registration still cleans up (F-13)."""
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker, hang=True)
    _patch_resolve_getdp(mocker)
    session.ranks = {5001: False}
    mocker.patch("mpxlite.base._register_live_run", side_effect=KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        s.run()
    assert session.received[:2] == [(4242, signal.SIGTERM), (5001, signal.SIGTERM)]
    assert session.ranks == {}
    assert base_mod._live_runs == {}


def test_interrupt_before_start_starts_nothing(
    case: tuple[Path, Path], mocker: MockerFixture, session: FakeSession
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_resolve_getdp(mocker)
    mocker.patch("mpxlite.base.subprocess.Popen", side_effect=KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        s.run()
    assert session.received == []


def test_second_interrupt_escalates_to_sigkill(
    case: tuple[Path, Path],
    mocker: MockerFixture,
    session: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C during the grace period: SIGKILL at once, cleanup not abandoned."""
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker, hang=True, ignore_term=True)
    _patch_resolve_getdp(mocker)
    session.ranks = {5001: True}
    monkeypatch.setattr(base_mod, "_TERM_GRACE_S", 60.0)
    real_wait_session = base_mod._wait_session
    calls: list[float] = []

    def wait_session(proc: Any, seconds: float) -> bool:
        calls.append(seconds)
        if len(calls) == 1:
            raise KeyboardInterrupt  # the second Ctrl-C, during the grace period
        return real_wait_session(proc, seconds)

    mocker.patch("mpxlite.base._wait_session", side_effect=wait_session)
    t0 = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        s.run(timeout=0.05)
    assert time.monotonic() - t0 < 5.0  # did not sit out the 60 s grace
    assert calls[0] == 60.0
    assert (4242, signal.SIGKILL) in session.received
    assert (5001, signal.SIGKILL) in session.received
    assert session.ranks == {}
    assert session.leader.returncode == -signal.SIGKILL


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_termination_signal_during_timeout_cleanup_still_escalates(
    case: tuple[Path, Path],
    mocker: MockerFixture,
    session: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
    sig: int,
) -> None:
    """SIGTERM / SIGHUP in the grace period of a timeout cleanup: SIGKILL still sent (F-2).

    The signal is handled once the cleanup is done: the process then dies of it.
    """
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker, hang=True, ignore_term=True)
    _patch_resolve_getdp(mocker)
    session.ranks = {5001: True}
    monkeypatch.setattr(base_mod, "_TERM_GRACE_S", 60.0)
    self_signals: list[int] = []

    def kill(pid: int, signum: int) -> None:
        if pid == os.getpid():
            self_signals.append(signum)  # the guard re-delivering the signal to itself
            return
        session.kill(pid, signum)

    mocker.patch("mpxlite.base.os.kill", side_effect=kill)
    real_wait_session = base_mod._wait_session
    calls: list[float] = []

    def wait_session(proc: Any, seconds: float) -> bool:
        calls.append(seconds)
        if len(calls) == 1:  # the signal arrives during the grace period
            handler = signal.getsignal(sig)
            assert callable(handler)
            handler(sig, None)
            return False
        return real_wait_session(proc, seconds)

    mocker.patch("mpxlite.base._wait_session", side_effect=wait_session)
    with pytest.raises(SystemExit) as exc_info:
        s.run(timeout=0.05)
    assert exc_info.value.code == 128 + sig
    assert self_signals == [sig]
    assert calls[0] == 60.0
    assert (4242, signal.SIGKILL) in session.received
    assert (5001, signal.SIGKILL) in session.received
    assert session.ranks == {}
    assert base_mod._cleanup_depth == 0
    assert base_mod._deferred_signal is None
    assert signal.getsignal(sig) is signal.SIG_DFL


# ----- run(): cancellation ------------------------------------------------


def test_cancel_stops_running_process(
    case: tuple[Path, Path], mocker: MockerFixture, session: FakeSession
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker, output="working\n", hang=True)
    _patch_resolve_getdp(mocker)
    session.ranks = {5001: False}
    cancel = threading.Event()
    timer = threading.Timer(0.2, cancel.set)
    timer.start()
    t0 = time.monotonic()
    try:
        with pytest.raises(GetDPCancelledError, match="cancelled") as exc_info:
            s.run(run_id="c", cancel=cancel)
    finally:
        timer.cancel()
    assert time.monotonic() - t0 < 2.0
    err = exc_info.value
    assert isinstance(err, GetDPError)
    assert not isinstance(err, GetDPTimeoutError)
    assert err.run_dir == s.runs_dir / "c"
    assert "working" in err.log_tail
    assert err.returncode == -signal.SIGTERM
    assert (5001, signal.SIGTERM) in session.received
    assert session.ranks == {}


def test_cancel_already_set_starts_nothing(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake = _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(GetDPCancelledError, match="before it was started") as exc_info:
        s.run(run_id="never", cancel=cancel)
    assert fake.call_count == 0
    assert exc_info.value.run_dir is None
    assert not (s.runs_dir / "never").exists()


def test_cancel_unset_run_completes(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    assert s.run(cancel=threading.Event()).returncode == 0


# ----- session cleanup on real processes (Linux /proc) ----------------------

_has_proc = sys.platform.startswith("linux") and os.path.isdir("/proc/self")


@pytest.mark.skipif(not _has_proc, reason="needs Linux /proc")
def test_session_pids_real_process(tmp_path: Path) -> None:
    # A comm with a space and parentheses exercises the /proc/<pid>/stat parsing.
    odd = tmp_path / "we ird) (x"
    odd.symlink_to("/bin/sleep")
    proc = subprocess.Popen([str(odd), "30"], start_new_session=True)
    try:
        assert _session_pids(proc.pid) == [proc.pid]
        assert os.getpid() not in (_session_pids(proc.pid) or [])
        assert _session_pids(os.getsid(0)) is not None
    finally:
        proc.kill()
        proc.wait()
    assert _session_pids(proc.pid) == []


@pytest.mark.skipif(not _has_proc, reason="needs Linux /proc")
def test_terminate_session_kills_other_process_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """A child in its own process group (like an Open MPI rank) is killed too."""
    monkeypatch.setattr(base_mod, "_TERM_GRACE_S", 1.0)
    # The child ignores SIGTERM: it must be SIGKILLed after the grace period.
    script = "set -m; (trap '' TERM; exec sleep 300) & echo $!; exec sleep 300"
    proc = subprocess.Popen(
        ["bash", "-c", script], start_new_session=True, stdout=subprocess.PIPE, text=True
    )
    assert proc.stdout is not None
    child = int(proc.stdout.readline())
    try:
        # Wait until the child ignores SIGTERM (trap set, sleep exec-ed): on a loaded
        # machine the signal could otherwise arrive first and end the child early.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            status = Path(f"/proc/{child}/status").read_text(encoding="ascii")
            ignored = int(status.split("SigIgn:")[1].split()[0], 16)
            if "Name:\tsleep" in status and ignored & (1 << (signal.SIGTERM - 1)):
                break
            time.sleep(0.01)
        assert os.getpgid(child) == child != proc.pid  # own group, same session
        assert os.getsid(child) == proc.pid
        t0 = time.monotonic()
        _terminate_session(proc)
        assert 1.0 <= time.monotonic() - t0 < 1.0 + 2.0
        assert _session_pids(proc.pid) == []
        assert proc.returncode == -signal.SIGTERM
    finally:
        for pid in _session_pids(proc.pid) or []:
            os.kill(pid, signal.SIGKILL)
        proc.stdout.close()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="prctl is Linux-only")
def test_libc_prctl_has_declared_signature() -> None:
    """prctl's variadic unsigned long arguments are declared, not passed as C int (F-7)."""
    import ctypes

    base_mod._libc_prctl.cache_clear()
    prctl = base_mod._libc_prctl()
    assert prctl is not None
    assert prctl.argtypes == [ctypes.c_int, *[ctypes.c_ulong] * 4]  # type: ignore[attr-defined]
    assert prctl.restype is ctypes.c_int  # type: ignore[attr-defined]
    assert callable(base_mod._parent_death_preexec())


# ----- pid reuse after the session emptied (fake /proc) ----------------------


class ReapedLeader:
    """A Popen-like leader that has exited and been reaped (``returncode`` set)."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.args = ["getdp"]
        self.returncode: int | None = 0

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return 0


def _fake_proc(root: Path, entries: dict[int, tuple[str, int]]) -> None:
    """Write ``/proc/<pid>/stat`` files: ``{pid: (state, session)}``."""
    for pid, (state, sid) in entries.items():
        (root / str(pid)).mkdir()
        (root / str(pid) / "stat").write_bytes(
            f"{pid} (proc x) {state} 1 {sid} {sid} 0 -1 4194560 0 0 0 0 0 0 0 0 20 0 1 0 "
            "123456 0 0\n".encode()
        )
    (root / "self").mkdir()  # non-numeric entries are skipped


def test_recycled_leader_pid_is_not_our_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """After the reap, a new session that got the leader's pid is left alone (F-1)."""
    _fake_proc(tmp_path, {4242: ("S", 4242), 4243: ("R", 4242), 9: ("S", 9)})
    monkeypatch.setattr(base_mod, "_PROC", str(tmp_path))
    kill = mocker.patch("mpxlite.base.os.kill")
    killpg = mocker.patch("mpxlite.base.os.killpg")
    monkeypatch.setattr(base_mod, "_TERM_GRACE_S", 0.05)
    monkeypatch.setattr(base_mod, "_KILL_WAIT_S", 0.1)
    proc: Any = ReapedLeader(4242)
    assert sorted(_session_pids(4242) or []) == [4242, 4243]
    base_mod._cleanup_after_exit(proc, tmp_path)
    kill.assert_not_called()
    assert base_mod._session_members(proc) == []
    assert not base_mod._signal_session(proc, signal.SIGKILL)
    _terminate_session(proc)
    kill.assert_not_called()
    killpg.assert_not_called()


def test_reaped_leader_leftovers_still_terminated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """Members left after the leader was reaped hold its pid: they are ours."""
    _fake_proc(tmp_path, {5001: ("S", 4242), 5002: ("Z", 4242), 9: ("S", 9)})
    monkeypatch.setattr(base_mod, "_PROC", str(tmp_path))
    monkeypatch.setattr(base_mod, "_TERM_GRACE_S", 0.05)
    monkeypatch.setattr(base_mod, "_KILL_WAIT_S", 0.1)
    monkeypatch.setattr(base_mod, "_POLL_S", 0.02)
    proc: Any = ReapedLeader(4242)

    def kill(pid: int, sig: int) -> None:
        if sig == signal.SIGKILL:
            (tmp_path / str(pid) / "stat").unlink()
            (tmp_path / str(pid)).rmdir()

    sent = mocker.patch("mpxlite.base.os.kill", side_effect=kill)
    assert base_mod._session_members(proc) == [5001]
    base_mod._cleanup_after_exit(proc, tmp_path)
    assert [c.args for c in sent.call_args_list] == [
        (5001, signal.SIGTERM),
        (5001, signal.SIGKILL),
    ]
    assert base_mod._session_members(proc) == []


def test_empty_session_never_rescanned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """Once a reaped leader's session was seen empty, /proc is not scanned again."""
    _fake_proc(tmp_path, {9: ("S", 9)})
    monkeypatch.setattr(base_mod, "_PROC", str(tmp_path))
    scan = mocker.patch("mpxlite.base._session_pids", wraps=base_mod._session_pids)
    proc: Any = ReapedLeader(4242)
    assert base_mod._session_members(proc) == []
    # A new session leader gets the free pid later: still not ours, not even scanned.
    (tmp_path / "4242").mkdir()
    (tmp_path / "4242" / "stat").write_bytes(b"4242 (x) S 1 4242 4242 0\n")
    assert not base_mod._session_alive(proc)
    assert base_mod._session_members(proc) == []
    assert scan.call_count == 1


def test_resolve_failure_creates_no_run_dir(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake = _patch_popen(mocker)
    mocker.patch(
        "mpxlite.base.SolverLite._resolve_getdp", side_effect=FileNotFoundError("no getdp")
    )
    with pytest.raises(FileNotFoundError, match="no getdp"):
        s.run(run_id="r")
    assert fake.call_count == 0
    assert not s.runs_dir.exists()


def test_getdp_error_without_log() -> None:
    err = GetDPError(3, ["getdp", "x.pro"])
    assert err.returncode == 3
    assert err.log_path is None and err.run_dir is None
    assert err.log_tail == ""
    assert err.warnings == ()
    assert "getdp x.pro" in str(err)


def test_getdp_error_unreadable_log(tmp_path: Path) -> None:
    err = GetDPError(1, ["getdp"], log_path=tmp_path / "missing.log", run_dir=tmp_path)
    assert err.log_tail == ""
    assert "(empty)" in str(err)


# ----- run(): PETSc options -----------------------------------------------


def _opts_text(run_dir: Path) -> list[str]:
    return (run_dir / PETSC_OPTIONS_NAME).read_text(encoding="utf-8").splitlines()


def test_no_petsc_options_no_file_no_flag(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres", petsc_options={})
    fake = _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    result = s.run("po", run_id="r", petsc_options={})
    assert "-options_file" not in _cmd(fake)
    assert not (result.work_dir / PETSC_OPTIONS_NAME).exists()
    assert dict(s.petsc_options) == {}


def test_template_petsc_options_written_and_passed(
    case: tuple[Path, Path], mocker: MockerFixture
) -> None:
    project_dir, mesh = case
    opts = {"pc_factor_mat_solver_type": "mumps", "ksp_view": None, "mat_mumps_icntl_14": 50}
    s = SolverLite(project_dir, "demo", mesh.name, "myres", petsc_options=opts)
    fake = _patch_popen(mocker)
    _patch_resolve_getdp(mocker, "/drv")

    result = s.run(["po_a", "po_b"], run_id="r")

    run_dir = result.work_dir
    lines = _opts_text(run_dir)
    assert lines[0].startswith("# ")
    assert lines[1:] == [
        "-ksp_view",
        "-mat_mumps_icntl_14 50",
        "-pc_factor_mat_solver_type mumps",
    ]
    cmd = _cmd(fake)
    assert cmd == [
        "/drv",
        str(run_dir / MANIFEST_NAME),
        "-msh",
        str(mesh.resolve()),
        "-v",
        "3",
        "-options_file",
        str(run_dir / PETSC_OPTIONS_NAME),
        "-pre",
        "myres",
        "-cal",
        "-pos",
        "po_a",
        "po_b",
    ]
    assert Path(cmd[cmd.index("-options_file") + 1]).is_absolute()
    assert result.cmd == tuple(cmd)


def test_per_run_petsc_options_merged_over_template(
    case: tuple[Path, Path], mocker: MockerFixture
) -> None:
    project_dir, mesh = case
    template = {"pc_factor_mat_solver_type": "mumps", "ksp_view": True, "mat_mumps_icntl_35": 1}
    s = SolverLite(project_dir, "demo", mesh.name, "myres", petsc_options=template)
    _patch_popen(mocker)
    _patch_resolve_getdp(mocker)

    over = s.run(
        run_id="over",
        petsc_options={"pc_factor_mat_solver_type": "petsc", "ksp_view": False},
    )
    base = s.run(run_id="base")

    assert _opts_text(over.work_dir)[1:] == [
        "-mat_mumps_icntl_35 1",
        "-pc_factor_mat_solver_type petsc",
    ]
    assert _opts_text(base.work_dir)[1:] == [
        "-ksp_view",
        "-mat_mumps_icntl_35 1",
        "-pc_factor_mat_solver_type mumps",
    ]
    assert dict(s.petsc_options) == template


def test_petsc_option_value_formatting(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    opts = {
        "a_flag": None,
        "b_true": True,
        "c_false": False,
        "d_int": -3,
        "e_float": 1e-8,
        "f_np_float": np.float64(0.25),
        "g_big": 1.5e300,
        "h_str": "lu",
        "i_path": "/x/y.txt",
    }
    result = s.run(run_id="r", petsc_options=opts)
    assert _opts_text(result.work_dir)[1:] == [
        "-a_flag",
        "-b_true",
        "-d_int -3",
        "-e_float 1e-08",
        "-f_np_float 0.25",
        "-g_big 1.5e+300",
        "-h_str lu",
        "-i_path /x/y.txt",
    ]


def test_all_false_petsc_options_write_nothing(
    case: tuple[Path, Path], mocker: MockerFixture
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres", petsc_options={"ksp_view": None})
    fake = _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    result = s.run(run_id="r", petsc_options={"ksp_view": False})
    assert "-options_file" not in _cmd(fake)
    assert not (result.work_dir / PETSC_OPTIONS_NAME).exists()


def test_petsc_options_template_read_only(case: tuple[Path, Path]) -> None:
    project_dir, mesh = case
    opts: dict[str, Any] = {"ksp_view": None}
    s = SolverLite(project_dir, "demo", mesh.name, "myres", petsc_options=opts)
    opts["pc_type"] = "lu"  # the caller's mapping is copied
    assert dict(s.petsc_options) == {"ksp_view": None}
    with pytest.raises(TypeError):
        s.petsc_options["pc_type"] = "lu"  # type: ignore[index]
    with pytest.raises(AttributeError):
        s.petsc_options = {}  # type: ignore[misc]


_BAD_PETSC_KEYS = ["-ksp_view", "1abc", "a-b", "a b", "", "_x", "a.b"]
_BAD_PETSC_VALUES = ["a b", "x\n", "a\tb", "a#b", "", float("nan"), float("inf")]


@pytest.mark.parametrize("key", _BAD_PETSC_KEYS)
def test_bad_petsc_key_rejected(case: tuple[Path, Path], key: str) -> None:
    project_dir, mesh = case
    match = "leading '-'" if key.startswith("-") else "Invalid PETSc option name"
    with pytest.raises(ValueError, match=match):
        SolverLite(project_dir, "demo", mesh.name, "myres", petsc_options={key: None})


@pytest.mark.parametrize("value", _BAD_PETSC_VALUES)
def test_bad_petsc_value_rejected(case: tuple[Path, Path], value: Any) -> None:
    project_dir, mesh = case
    with pytest.raises(ValueError, match="pc_type"):
        SolverLite(project_dir, "demo", mesh.name, "myres", petsc_options={"pc_type": value})


@pytest.mark.parametrize("opts", [{3: None}, {"pc_type": [1]}, {"pc_type": 1j}])
def test_bad_petsc_type_rejected(case: tuple[Path, Path], opts: Any) -> None:
    project_dir, mesh = case
    with pytest.raises(TypeError):
        SolverLite(project_dir, "demo", mesh.name, "myres", petsc_options=opts)


@pytest.mark.parametrize("opts", [{"-pc_type": "lu"}, {"pc_type": "a b"}, {"pc_type": object()}])
def test_bad_per_run_petsc_options_rejected_before_run_dir(
    case: tuple[Path, Path], mocker: MockerFixture, opts: Any
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake = _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    with pytest.raises((ValueError, TypeError)):
        s.run(run_id="r", petsc_options=opts)
    assert not s.runs_dir.exists()
    assert fake.call_count == 0


# ----- run(): GetDP warnings ----------------------------------------------

_NOISY_LOG = (
    "Authorization required, but no authorization protocol specified\n"
    "\n"
    "Info    : Started\n"
    "Warning : 'case.pro', line 28 : Constraint 'NoSuch' is not provided\n"
    "\x1b[35mWarning : [rank   1] 'case.pro', line 28 : Constraint 'NoSuch' is not provided"
    "\x1b[0m\n"
    "\x1b[35mWarning : [rank   0] Second warning\x1b[0m\n"
    "   Warning : [rank  12] Second warning   \n"
    "WARNING: There was an error initializing an OpenFabrics device.\n"
    "[getdp_runner] warning: MPXLITE_GETDP_CPUS=0-1 ignored under SLURM\n"
    "Info    : Warning count is not a warning\n"
    "Warnings: not a GetDP warning either\n"
    "WarningCount = 3\n"
    "Warning, PETSc banner without the message-type colon\n"
    "Warning : 'case.pro', line 28 : Constraint 'NoSuch' is not provided\n"
)
_EXPECTED_WARNINGS = (
    "Warning : 'case.pro', line 28 : Constraint 'NoSuch' is not provided",
    "Warning : Second warning",
)


def test_warnings_collected_and_logged_once(
    case: tuple[Path, Path], mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker, output=_NOISY_LOG)
    _patch_resolve_getdp(mocker)
    with caplog.at_level(logging.WARNING, logger="mpxlite"):
        result = s.run(run_id="w")
    assert result.warnings == _EXPECTED_WARNINGS
    records = [r for r in caplog.records if r.name == "mpxlite"]
    assert [r.levelno for r in records] == [logging.WARNING] * 2
    assert [r.getMessage() for r in records] == [f"getdp run w: {w}" for w in _EXPECTED_WARNINGS]


def test_no_warnings_by_default(
    case: tuple[Path, Path], mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker, output="Authorization required, but no authorization protocol\nInfo : x\n")
    _patch_resolve_getdp(mocker)
    with caplog.at_level(logging.WARNING, logger="mpxlite"):
        result = s.run()
    assert result.warnings == ()
    assert not [r for r in caplog.records if r.name == "mpxlite"]
    assert SolverResult(Path("/x"), "x", Path("/x/l"), 0.0, 0, ()).warnings == ()


def test_warnings_on_getdp_error(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker, output=_NOISY_LOG + "Error   : boom\n", returncode=1)
    _patch_resolve_getdp(mocker)
    with pytest.raises(GetDPError) as info:
        s.run(run_id="bad")
    assert info.value.warnings == _EXPECTED_WARNINGS


def test_warnings_tolerate_undecodable_log(tmp_path: Path) -> None:
    log = tmp_path / "getdp.log"
    log.write_bytes(b"\xff\xfe garbage\nWarning : caf\xe9\n")
    err = GetDPError(1, ["getdp"], log_path=log)
    assert err.warnings == ("Warning : caf�",)


# ----- _resolve_getdp -----------------------------------------------------


def _fake_exe(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def test_resolve_getdp_via_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_exe(tmp_path / "fakedp")
    monkeypatch.setenv("MPXLITE_GETDP", str(fake))
    assert SolverLite._resolve_getdp() == str(fake)


def test_resolve_getdp_relative_env_made_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative MPXLITE_GETDP is resolved against the cwd, not the run dir (F-4)."""
    (tmp_path / "drivers").mkdir()
    fake = _fake_exe(tmp_path / "drivers" / "fakedp")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPXLITE_GETDP", "./drivers/fakedp")
    resolved = SolverLite._resolve_getdp()
    assert resolved == str(fake.resolve())
    assert Path(resolved).is_absolute()
    monkeypatch.chdir(tmp_path / "drivers")  # e.g. Popen(cwd=run_dir)
    assert Path(resolved).is_file()


def test_resolve_getdp_via_explicit_env_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MPXLITE_GETDP", raising=False)
    fake = _fake_exe(tmp_path / "fakedp")
    assert SolverLite._resolve_getdp({"MPXLITE_GETDP": str(fake)}) == str(fake)


def test_resolve_getdp_env_must_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPXLITE_GETDP", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="not an executable"):
        SolverLite._resolve_getdp()


def test_resolve_getdp_defaults_to_bundled_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MPXLITE_GETDP", raising=False)
    import mpxlite

    assert SolverLite._resolve_getdp() == str(mpxlite.WRAPPER_PATH)


def test_resolve_getdp_never_uses_a_getdp_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the wrapper there is no driver: a getdp on PATH is not one."""
    monkeypatch.delenv("MPXLITE_GETDP", raising=False)
    monkeypatch.setattr("mpxlite.WRAPPER_PATH", tmp_path / "missing.sh")
    _fake_exe(tmp_path / "getdp")
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match=r"getdp_runner\.sh wrapper"):
        SolverLite._resolve_getdp()
    with pytest.raises(FileNotFoundError, match=r"getdp_runner\.sh wrapper"):
        SolverLite._resolve_getdp({"PATH": str(tmp_path)})


# ----- the engine of a run -------------------------------------------------


def test_run_names_the_engine_to_the_driver(
    case: tuple[Path, Path], mocker: MockerFixture, fake_engine: Path
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake = _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    s.run()
    assert fake.call_args.kwargs["env"]["MPXLITE_GETDP_BINARY"] == str(fake_engine.resolve())


def test_run_engine_of_the_launch_wins(
    case: tuple[Path, Path], mocker: MockerFixture, tmp_path: Path
) -> None:
    project_dir, mesh = case
    other = _fake_exe(tmp_path / "other_getdp")
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake = _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    s.run(launch=LaunchSpec(env={"MPXLITE_GETDP_BINARY": str(other)}))
    assert fake.call_args.kwargs["env"]["MPXLITE_GETDP_BINARY"] == str(other.resolve())


def test_run_without_engine_creates_no_run_dir(
    case: tuple[Path, Path], mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No engine named and none installed: refused before anything is created."""
    project_dir, mesh = case
    monkeypatch.delenv("MPXLITE_GETDP_BINARY")
    monkeypatch.setenv("MPXLITE_HOME", str(tmp_path / "no_installation"))
    _fake_exe(tmp_path / "getdp")
    monkeypatch.setenv("PATH", str(tmp_path))
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake = _patch_popen(mocker)
    with pytest.raises(FileNotFoundError, match="No engine at"):
        s.run(run_id="r")
    assert fake.call_count == 0
    assert not s.runs_dir.exists()


# ----- hard memory cap (memory_max) ---------------------------------------


class FakeScope:
    """Stand-in for MemoryScope: records calls, reports ``oom`` when asked."""

    def __init__(self, oom: bool) -> None:
        self.oom = oom
        self.unit = "mpxlite-x-1.scope"
        self.memory_max = 0
        self.settles: list[bool] = []
        self.killed = 0

    def command_prefix(self) -> list[str]:
        return ["systemd-run", "--scope", f"--unit={self.unit}"]

    def oom_killed(self, settle: bool) -> bool:
        self.settles.append(settle)
        return self.oom

    def kill(self) -> None:
        self.killed += 1


@pytest.fixture
def fake_scope(mocker: MockerFixture) -> Any:
    """Hard caps available; ``fake_scope(oom)`` installs the scope of the next run."""
    mocker.patch("mpxlite.base.require_hard_cap")

    def install(oom: bool = False) -> FakeScope:
        scope = FakeScope(oom)

        def for_run(run_id: str, memory_max: int) -> FakeScope:
            scope.memory_max = memory_max
            return scope

        mocker.patch("mpxlite.base.MemoryScope.for_run", side_effect=for_run)
        return scope

    return install


@pytest.mark.parametrize(
    ("bad", "exc"), [(0, ValueError), (-1, ValueError), (True, TypeError), ("1G", TypeError)]
)
def test_bad_memory_max_rejected_before_run_dir(
    case: tuple[Path, Path], mocker: MockerFixture, bad: Any, exc: type[Exception]
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    popen = _patch_popen(mocker)
    with pytest.raises(exc, match="memory_max"):
        s.run(run_id="m", memory_max=bad)
    popen.assert_not_called()
    assert not (s.runs_dir / "m").exists()


def test_memory_max_unavailable_raises_before_run_dir(
    case: tuple[Path, Path], mocker: MockerFixture
) -> None:
    from mpxlite.memory import HardCapUnavailableError

    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    popen = _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    mocker.patch("mpxlite.base.require_hard_cap", side_effect=HardCapUnavailableError("no systemd"))
    with pytest.raises(HardCapUnavailableError, match="no systemd"):
        s.run(run_id="m", memory_max=1 << 30)
    popen.assert_not_called()
    assert not (s.runs_dir / "m").exists()


def test_no_memory_max_no_scope(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    popen = _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    req = mocker.patch("mpxlite.base.require_hard_cap")
    res = s.run()
    req.assert_not_called()
    assert _cmd(popen)[0] == "/usr/bin/getdp"
    assert res.memory_max is None


def test_capped_run_wrapped_in_scope(
    case: tuple[Path, Path], mocker: MockerFixture, fake_scope: Any
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    popen = _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    scope = fake_scope(oom=False)
    res = s.run("po", memory_max=123_456)
    cmd = _cmd(popen)
    assert cmd[:3] == scope.command_prefix()
    assert cmd[3] == "/usr/bin/getdp"
    assert res.cmd[0] == "/usr/bin/getdp"  # prefix not in the result's cmd
    assert res.memory_max == 123_456 == scope.memory_max
    assert popen.call_args.kwargs["start_new_session"] is True
    assert scope.settles == [False]  # success: a single query
    assert scope.killed == 0


def test_oom_killed_run_raises_memory_error(
    case: tuple[Path, Path], mocker: MockerFixture, fake_scope: Any
) -> None:
    from mpxlite.base import GetDPMemoryError

    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker, output="allocating\n", returncode=-9)
    _patch_resolve_getdp(mocker)
    scope = fake_scope(oom=True)
    with pytest.raises(GetDPMemoryError) as info:
        s.run(run_id="hog", memory_max=2 << 30)
    err = info.value
    assert isinstance(err, GetDPError)
    assert err.memory_max == 2 << 30
    assert err.unit == scope.unit
    assert err.returncode == -9
    assert err.run_dir == s.runs_dir / "hog"
    text = str(err)
    assert "MemoryMax=2147483648" in text and "2 GiB" in text
    assert str(s.runs_dir / "hog") in text.splitlines()[0]
    assert "allocating" in text
    assert scope.settles == [True]


def test_failed_capped_run_not_oom_is_plain_error(
    case: tuple[Path, Path], mocker: MockerFixture, fake_scope: Any
) -> None:
    from mpxlite.base import GetDPMemoryError

    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker, returncode=1)
    _patch_resolve_getdp(mocker)
    fake_scope(oom=False)
    with pytest.raises(GetDPError) as info:
        s.run(memory_max=1 << 30)
    assert not isinstance(info.value, GetDPMemoryError)
    assert info.value.returncode == 1


def test_timeout_of_capped_run_kills_scope(
    case: tuple[Path, Path], mocker: MockerFixture, session: FakeSession, fake_scope: Any
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker, hang=True)
    _patch_resolve_getdp(mocker)
    session.ranks = {5001: False}
    scope = fake_scope(oom=False)
    with pytest.raises(GetDPTimeoutError):
        s.run(timeout=0.1, memory_max=1 << 30)
    assert session.ranks == {}  # session cleanup still applies
    assert scope.killed == 1
    assert scope.settles == []


# ----- preprocess() -------------------------------------------------------


def test_preprocess_runs_pre_only(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(
        project_dir, "demo", mesh.name, "myres", workflow=WorkflowData({"A": 1.0}), verbosity=4
    )
    fake = _patch_popen(mocker, output="pre ok\n")
    _patch_resolve_getdp(mocker, "/drv")

    result = s.preprocess(workflow={"A": 2.0}, run_id="chk")

    run_dir = project_dir.resolve() / "runs" / "chk"
    assert _cmd(fake) == [
        "/drv",
        str(run_dir / MANIFEST_NAME),
        "-msh",
        str(mesh.resolve()),
        "-v",
        "4",
        "-pre",
        "myres",
    ]
    assert fake.call_count == 1
    assert isinstance(result, SolverResult)
    assert result.work_dir == run_dir
    assert result.run_id == "chk"
    assert result.returncode == 0
    assert result.cmd == tuple(_cmd(fake))
    assert result.log_path.read_text() == "pre ok\n"
    assert sorted(p.name for p in run_dir.iterdir()) == sorted(
        [LOG_NAME, MANIFEST_NAME, WORKFLOW_NAME]
    )
    assert "A = 2.0;" in result.path(WORKFLOW_NAME).read_text(encoding="utf-8")
    assert dict(s.workflow) == {"A": 1.0}  # template untouched
    assert fake.call_args.kwargs["cwd"] == str(run_dir)
    assert fake.call_args.kwargs["stdin"] is subprocess.DEVNULL


def test_preprocess_manifest_identical_to_run(
    case: tuple[Path, Path], mocker: MockerFixture
) -> None:
    project_dir, mesh = case
    res = project_dir / "fs.pro"
    res.write_text("// fs\n", encoding="utf-8")
    s = SolverLite(project_dir, "demo", mesh.name, "myres", resource_files=[res])
    _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    pre = s.preprocess(run_id="p")
    full = s.run(run_id="r")
    for name in (MANIFEST_NAME, WORKFLOW_NAME):
        text_pre = pre.path(name).read_text(encoding="utf-8")
        text_run = full.path(name).read_text(encoding="utf-8")
        assert text_pre.replace(str(pre.work_dir), "<RUN>") == text_run.replace(
            str(full.work_dir), "<RUN>"
        )


def test_preprocess_petsc_options_and_launch_env(
    case: tuple[Path, Path], mocker: MockerFixture
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres", petsc_options={"ksp_type": "preonly"})
    fake = _patch_popen(mocker)
    _patch_resolve_getdp(mocker, "/drv")
    result = s.preprocess(
        run_id="p", petsc_options={"petsc_prealloc": 400}, launch=LaunchSpec(np=2, threads=1)
    )
    opts = result.path(PETSC_OPTIONS_NAME)
    assert opts.read_text(encoding="utf-8").splitlines()[1:] == [
        "-ksp_type preonly",
        "-petsc_prealloc 400",
    ]
    assert _cmd(fake)[-4:] == ["-options_file", str(opts), "-pre", "myres"]
    env = fake.call_args.kwargs["env"]
    assert env["MPXLITE_GETDP_NP"] == "2"
    assert env["MPXLITE_GETDP_THREADS"] == "1"


def test_preprocess_warnings_collected_and_logged(
    case: tuple[Path, Path], mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker, output=_NOISY_LOG)
    _patch_resolve_getdp(mocker)
    with caplog.at_level(logging.WARNING, logger="mpxlite"):
        result = s.preprocess(run_id="w")
    assert result.warnings == _EXPECTED_WARNINGS
    logged = [r.getMessage() for r in caplog.records if r.name == "mpxlite"]
    assert logged == [f"getdp run w: {w}" for w in _EXPECTED_WARNINGS]


def test_preprocess_failure_raises_getdp_error(
    case: tuple[Path, Path], mocker: MockerFixture
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker, output=_NOISY_LOG + "Error   : Unknown Group 'X'\n", returncode=1)
    _patch_resolve_getdp(mocker)
    with pytest.raises(GetDPError) as info:
        s.preprocess(run_id="bad")
    err = info.value
    assert err.returncode == 1
    assert err.run_dir == s.runs_dir / "bad"
    assert "Unknown Group 'X'" in err.log_tail
    assert err.warnings == _EXPECTED_WARNINGS
    assert err.cmd[-2:] == ["-pre", "myres"]


def test_preprocess_validates_before_run_dir(
    case: tuple[Path, Path], mocker: MockerFixture
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake = _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    with pytest.raises(ValueError, match="Invalid run_id"):
        s.preprocess(run_id="../x")
    with pytest.raises(ValueError):
        s.preprocess(workflow={"1bad": 1.0}, run_id="r")
    with pytest.raises(ValueError, match="memory_max"):
        s.preprocess(run_id="r", memory_max=0)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(GetDPCancelledError):
        s.preprocess(run_id="r", cancel=cancel)
    assert fake.call_count == 0
    assert not s.runs_dir.exists()


def test_preprocess_is_keyword_only(case: tuple[Path, Path]) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    with pytest.raises(TypeError):
        s.preprocess("po")  # type: ignore[misc]


def test_preprocess_timeout_kills_session(
    case: tuple[Path, Path], mocker: MockerFixture, session: FakeSession
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker, hang=True)
    _patch_resolve_getdp(mocker)
    session.ranks = {5001: False}
    with pytest.raises(GetDPTimeoutError):
        s.preprocess(timeout=0.1)
    assert session.ranks == {}


def test_preprocess_capped_in_scope(
    case: tuple[Path, Path], mocker: MockerFixture, fake_scope: Any
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    popen = _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    scope = fake_scope(oom=False)
    res = s.preprocess(memory_max=1 << 30)
    cmd = _cmd(popen)
    assert cmd[:3] == scope.command_prefix()
    assert cmd[-2:] == ["-pre", "myres"]
    assert res.memory_max == 1 << 30 == scope.memory_max


# ----- the interface for plug-ins -----------------------------------------


def test_case_file_paths(case: tuple[Path, Path]) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    assert CASE_EXTENSIONS == (".geo.pro", ".physprop.pro", ".cond.pro")
    for ext in CASE_EXTENSIONS:
        assert s.case_file(ext) == project_dir / f"demo{ext}"
        assert s.case_file(ext).is_file()
    with pytest.raises(ValueError, match="is not one of"):
        s.case_file(".msh")


def test_effective_scalars_and_options_are_public_and_pure(case: tuple[Path, Path]) -> None:
    project_dir, mesh = case
    template = WorkflowData({"A": 1.0, "B": 2})
    s = SolverLite(
        project_dir, "demo", mesh.name, "myres", workflow=template, petsc_options={"ksp_rtol": 1e-8}
    )
    scalars = s.effective_workflow({"B": 5, "C": "x"})
    assert dict(scalars) == {"A": 1.0, "B": 5, "C": "x"}
    assert dict(s.workflow) == {"A": 1.0, "B": 2}
    assert dict(s.effective_workflow()) == {"A": 1.0, "B": 2}
    options = s.effective_petsc_options({"pc_type": "lu", "ksp_rtol": False})
    assert options == {"ksp_rtol": False, "pc_type": "lu"}
    assert s.effective_petsc_options() == {"ksp_rtol": 1e-8}
    assert dict(s.petsc_options) == {"ksp_rtol": 1e-8}


def test_run_uses_the_overridden_effective_methods(
    case: tuple[Path, Path], mocker: MockerFixture
) -> None:
    """What a subclass returns from the two methods is what the run writes."""

    class Derived(SolverLite):
        def effective_workflow(self, overrides: Any = None) -> WorkflowData:
            scalars = super().effective_workflow(overrides)
            scalars["TWICE"] = 2 * scalars["A"]  # type: ignore[operator]
            return scalars

        def effective_petsc_options(self, overrides: Any = None) -> dict[str, Any]:
            return {"petsc_prealloc": 400, **super().effective_petsc_options(overrides)}

    project_dir, mesh = case
    s = Derived(project_dir, "demo", mesh.name, "myres", workflow=WorkflowData({"A": 1.5}))
    _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    result = s.run(workflow={"A": 4.0}, petsc_options={"ksp_monitor": None})
    assert dict(WorkflowData.read_pro(result.workflow_path)) == {"A": 4.0, "TWICE": 8.0}
    text = result.petsc_options_path.read_text(encoding="utf-8")
    assert "-petsc_prealloc 400" in text and "-ksp_monitor" in text


def test_run_reads_a_solution_file(
    case: tuple[Path, Path], mocker: MockerFixture, tmp_path: Path
) -> None:
    project_dir, mesh = case
    solution = tmp_path / "earlier" / "mpxlite.res"
    solution.parent.mkdir()
    solution.write_text("$ResFormat\n", encoding="utf-8")
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake = _patch_popen(mocker)
    _patch_resolve_getdp(mocker, "/drv")
    result = s.run("po", run_id="r", read_solution=solution)
    assert _cmd(fake) == [
        "/drv",
        str(result.manifest_path),
        "-msh",
        str(mesh.resolve()),
        "-v",
        "3",
        "-res",
        str(solution.resolve()),
        "-pre",
        "myres",
        "-cal",
        "-pos",
        "po",
    ]
    result = s.preprocess(run_id="p", read_solution=str(solution))
    assert _cmd(fake)[6:] == ["-res", str(solution.resolve()), "-pre", "myres"]


def test_missing_solution_file_creates_no_run_dir(
    case: tuple[Path, Path], mocker: MockerFixture, tmp_path: Path
) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    fake = _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    with pytest.raises(FileNotFoundError, match="Solution file to read not found"):
        s.run(run_id="r", read_solution=tmp_path / "nowhere.res")
    assert fake.call_count == 0
    assert not s.runs_dir.exists()


def test_paths_of_the_run_folder(case: tuple[Path, Path], mocker: MockerFixture) -> None:
    project_dir, mesh = case
    s = SolverLite(project_dir, "demo", mesh.name, "myres")
    _patch_popen(mocker)
    _patch_resolve_getdp(mocker)
    result = s.run(run_id="r")
    run_dir = s.runs_dir / "r"
    assert result.manifest_path == run_dir / "mpxlite.pro" and result.manifest_path.is_file()
    assert result.workflow_path == run_dir / "workflow.pro" and result.workflow_path.is_file()
    assert result.log_path == run_dir / "getdp.log"
    assert result.petsc_options_path == run_dir / "petsc.opts"
    assert not result.petsc_options_path.exists()  # no option was set
    assert result.pre_path == run_dir / "mpxlite.pre"
    assert result.solution_path == run_dir / "mpxlite.res"


def test_the_replaced_private_names_are_gone() -> None:
    for name in ("_case_file", "_effective_workflow", "_effective_petsc_options"):
        assert not hasattr(SolverLite, name), name
    for name in ("_CASE_EXTENSIONS", "_MANIFEST_NAME", "_WORKFLOW_NAME", "_LOG_NAME"):
        assert not hasattr(base_mod, name), name
