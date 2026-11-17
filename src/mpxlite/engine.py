# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""The engine of an mpxlite installation: where it is and what it is.

The engine is the GetDP program an mpxlite installation holds, with its
libraries, in ``<home>/engine``. One rule names the program a run executes,
and it is applied in one place, :func:`find_engine`:

1. ``MPXLITE_GETDP_BINARY``, when it is set; it must then name an executable
   file. The ``activate.sh`` of an installation sets it to its engine.
2. Otherwise ``engine/bin/getdp`` of the installation (:func:`installation_home`).

``PATH`` is never searched: a ``getdp`` that happens to be on it is not the
engine. :func:`engine_info` gives the build information of the engine.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from collections.abc import Mapping
from pathlib import Path

__all__ = [
    "ENGINE_ENV",
    "HOME_ENV",
    "engine_info",
    "find_engine",
    "installation_home",
]

ENGINE_ENV = "MPXLITE_GETDP_BINARY"
"""Environment variable naming the GetDP program to run."""

HOME_ENV = "MPXLITE_HOME"
"""Environment variable naming the mpxlite installation."""

_ENGINE_IN_HOME = Path("engine") / "bin" / "getdp"
_DEFAULT_HOME = "mpxlite"  # under the home directory of the user

_INFO_TIMEOUT_S = 30.0
_INFO_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z ]*?)\s*:\s+(\S.*?)\s*$")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

_info_lock = threading.Lock()
_info_memo: dict[tuple[str, int, int], dict[str, str]] = {}


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def installation_home(env: Mapping[str, str] | None = None) -> Path:
    """The mpxlite installation whose engine is used.

    ``MPXLITE_HOME`` when it is set; otherwise the installation this package
    is installed in (its Python environment is ``<home>/venv``), when it has
    an engine; otherwise ``~/mpxlite``.

    Args:
        env: The environment read (default: :data:`os.environ`).

    Returns:
        The directory of the installation. It is not required to exist.
    """
    environ = os.environ if env is None else env
    named = environ.get(HOME_ENV, "")
    if named:
        return Path(named).expanduser()
    own = Path(sys.prefix).parent
    if (own / _ENGINE_IN_HOME).is_file():
        return own
    return Path.home() / _DEFAULT_HOME


def find_engine(env: Mapping[str, str] | None = None) -> Path:
    """The GetDP program a run executes (see the module documentation).

    Args:
        env: The environment of the run (default: :data:`os.environ`), read
            for ``MPXLITE_GETDP_BINARY`` and ``MPXLITE_HOME``.

    Returns:
        The absolute path of the executable.

    Raises:
        FileNotFoundError: ``MPXLITE_GETDP_BINARY`` does not name an
            executable file, or it is unset and the installation has no
            engine.
    """
    environ = os.environ if env is None else env
    named = environ.get(ENGINE_ENV, "")
    if named:
        path = Path(named).expanduser()
        if _is_executable(path):
            return path.resolve()
        raise FileNotFoundError(f"{ENGINE_ENV}={named!r} is not an executable file.")
    home = installation_home(environ)
    path = home / _ENGINE_IN_HOME
    if _is_executable(path):
        return path.resolve()
    raise FileNotFoundError(
        f"No engine at {path}: mpxlite is not installed at {home}. Install mpxlite from "
        f"its bundle (install.sh), or set {HOME_ENV} to its installation, or "
        f"{ENGINE_ENV} to the GetDP program to run."
    )


def engine_info(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build information of the engine: the ``Key : value`` lines of ``getdp --info``.

    For instance ``"Version"``, ``"Build options"`` and ``"PETSc version"``.
    The program is run once per file (path, size and modification time);
    later calls return the stored answer.

    Args:
        env: The environment of the run, as for :func:`find_engine`.

    Returns:
        A new mapping from key to value, in the order printed.

    Raises:
        FileNotFoundError: as :func:`find_engine`.
        RuntimeError: the program fails, or prints no such line.
    """
    path = find_engine(env)
    stat = path.stat()
    memo_key = (str(path), stat.st_size, stat.st_mtime_ns)
    with _info_lock:
        known = _info_memo.get(memo_key)
    if known is not None:
        return dict(known)
    # hwloc (Open MPI) is kept off X11 / OpenGL, whose probing prints noise.
    run_env = {**(os.environ if env is None else env), "HWLOC_COMPONENTS": "-gl"}
    try:
        proc = subprocess.run(
            [str(path), "--info"],
            capture_output=True,
            text=True,
            env=run_env,
            timeout=_INFO_TIMEOUT_S,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"{path} --info could not be run: {exc}") from exc
    text = _ANSI_RE.sub("", proc.stdout + proc.stderr)
    info: dict[str, str] = {}
    for line in text.splitlines():
        match = _INFO_LINE_RE.match(line)
        if match:
            info.setdefault(match.group(1), match.group(2))
    if proc.returncode != 0 or not info:
        raise RuntimeError(f"{path} --info failed (exit {proc.returncode}):\n{text}")
    with _info_lock:
        _info_memo[memo_key] = dict(info)
    return info
