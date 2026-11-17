# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""The engine of an installation: one rule to find it, its build information."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mpxlite import engine, engine_info, find_engine, installation_home

_INFO = """\
Authorization required, but no authorization protocol specified
Version          : 3.5.0-mpx3-git-864875cd
License          : GNU General Public License
Build options    : 64Bit Blas[petsc] Kernel Lapack[petsc] MPI NoFortran PETSc
PETSc version    : 3.26.0 (complex arithmetic)
Web site         : http://getdp.info
"""


def _program(path: Path, body: str = "exit 0") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _installation(home: Path, body: str = "exit 0") -> Path:
    """A directory laid out as an installation; returns its engine."""
    return _program(home / "engine" / "bin" / "getdp", body)


# ----- installation_home ----------------------------------------------------


def test_home_named_by_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPXLITE_HOME", str(tmp_path / "from_environ"))
    assert installation_home() == tmp_path / "from_environ"
    assert installation_home({"MPXLITE_HOME": str(tmp_path / "given")}) == tmp_path / "given"


def test_home_of_the_installation_the_package_is_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Python environment of an installation is <home>/venv."""
    _installation(tmp_path / "home")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "home" / "venv"))
    assert installation_home({}) == tmp_path / "home"


def test_home_defaults_to_the_user_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "elsewhere" / "venv"))
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    assert installation_home({}) == tmp_path / "user" / "mpxlite"


# ----- find_engine ----------------------------------------------------------


def test_engine_named_by_the_environment(fake_engine: Path, tmp_path: Path) -> None:
    assert find_engine() == fake_engine.resolve()
    other = _program(tmp_path / "other")
    assert find_engine({"MPXLITE_GETDP_BINARY": str(other)}) == other.resolve()


def test_named_engine_must_be_an_executable_file(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.write_text("not a program", encoding="utf-8")
    for bad in (tmp_path / "missing", plain, tmp_path):
        with pytest.raises(FileNotFoundError, match="is not an executable file"):
            find_engine({"MPXLITE_GETDP_BINARY": str(bad)})


def test_a_named_engine_is_never_replaced_by_the_installed_one(tmp_path: Path) -> None:
    """A wrong name is an error, not a reason to look elsewhere."""
    _installation(tmp_path / "home")
    env = {
        "MPXLITE_GETDP_BINARY": str(tmp_path / "missing"),
        "MPXLITE_HOME": str(tmp_path / "home"),
    }
    with pytest.raises(FileNotFoundError, match="is not an executable file"):
        find_engine(env)


def test_engine_of_the_installation(tmp_path: Path) -> None:
    program = _installation(tmp_path / "home")
    assert find_engine({"MPXLITE_HOME": str(tmp_path / "home")}) == program.resolve()


def test_path_is_never_searched(tmp_path: Path) -> None:
    _program(tmp_path / "bin" / "getdp")
    env = {"PATH": str(tmp_path / "bin"), "MPXLITE_HOME": str(tmp_path / "empty")}
    with pytest.raises(FileNotFoundError, match=r"No engine at .*empty/engine/bin/getdp"):
        find_engine(env)


# ----- engine_info ----------------------------------------------------------


def test_info_keeps_the_key_value_lines(tmp_path: Path) -> None:
    body = "cat <<'TXT'\n" + _INFO + "TXT"
    _installation(tmp_path / "home", body)
    info = engine_info({"MPXLITE_HOME": str(tmp_path / "home")})
    assert info == {
        "Version": "3.5.0-mpx3-git-864875cd",
        "License": "GNU General Public License",
        "Build options": "64Bit Blas[petsc] Kernel Lapack[petsc] MPI NoFortran PETSc",
        "PETSc version": "3.26.0 (complex arithmetic)",
        "Web site": "http://getdp.info",
    }
    assert next(iter(info)) == "Version"


def test_info_is_asked_once_per_file(tmp_path: Path) -> None:
    count = tmp_path / "count"
    body = f"echo x >> {count}\necho 'Version : 1'"
    _installation(tmp_path / "home", body)
    env = {"MPXLITE_HOME": str(tmp_path / "home")}
    first = engine_info(env)
    first["Version"] = "changed by the caller"
    assert engine_info(env) == {"Version": "1"}
    assert count.read_text(encoding="utf-8").count("x") == 1


def test_info_of_a_failing_program(tmp_path: Path) -> None:
    _installation(tmp_path / "failing", "echo 'Version : 1'\nexit 3")
    with pytest.raises(RuntimeError, match=r"--info failed \(exit 3\)"):
        engine_info({"MPXLITE_HOME": str(tmp_path / "failing")})
    _installation(tmp_path / "silent", "echo nothing useful")
    with pytest.raises(RuntimeError, match="--info failed"):
        engine_info({"MPXLITE_HOME": str(tmp_path / "silent")})


def test_names_of_the_variables() -> None:
    assert engine.ENGINE_ENV == "MPXLITE_GETDP_BINARY"
    assert engine.HOME_ENV == "MPXLITE_HOME"
