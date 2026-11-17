# Copyright (c) 2025 MPhysX
# SPDX-License-Identifier: MIT

"""Helpers of the integration suite: the ``rod`` test case and its exact results.

The case ``data/rod.*`` solves ``-u'' = S`` on the unit segment with
``u(0) = 0`` and ``u(1) = V`` (``S`` and ``V`` are workflow scalars), on a
committed 10-element MSH 2.2 mesh (``data/rod.msh``, generated from
``data/rod.geo`` with ``gmsh -1 -format msh22``). P1 nodal values are exact:
``u(0.5) = V / 2 + S / 8``; the integral of the P1 solution over the segment
is ``V / 2 + S (1 - h^2) / 12`` with ``h = 0.1``.
"""

from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path

from mpxlite import find_engine

DATA_DIR = Path(__file__).parent / "data"
MPI_GETDP_ENV = "MPXLITE_TEST_MPI_GETDP"


@functools.cache
def mpi_getdp() -> Path:
    """getdp binary of the MPI tests: ``$MPXLITE_TEST_MPI_GETDP``, else the binary under test.

    Resolved lazily (at test setup, never at collection) and cached.
    """
    override = os.environ.get(MPI_GETDP_ENV)
    if override:
        return Path(override)
    try:
        return find_engine()
    except FileNotFoundError:
        return Path(os.devnull)  # not a regular file: MPI tests are skipped


H = 0.1


def expected_mid(v: float, s: float) -> float:
    """Exact ``u(0.5)``."""
    return v / 2 + s / 8


def expected_integral(v: float, s: float) -> float:
    """Integral over the segment of the P1 solution."""
    return v / 2 + s * (1 - H**2) / 12


@functools.cache
def build_options(binary: Path) -> list[str]:
    """``Build options`` of ``binary --info`` (empty if not an executable getdp).

    Runs ``binary --info`` once per binary; call it only from test setup or
    bodies (string-form ``skipif`` conditions), never at import.
    """
    if not (binary.is_file() and os.access(binary, os.X_OK)):
        return []
    try:
        info = subprocess.run(
            [str(binary), "--info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    for line in info.splitlines():
        if line.strip().startswith("Build options"):
            return line.split(":", 1)[-1].split()
    return []


def mpi_getdp_available() -> bool:
    """Whether :func:`mpi_getdp` exists and was built with MPI."""
    return "MPI" in build_options(mpi_getdp())


def snapshot(directory: Path) -> dict[str, tuple[int, int]]:
    """``{relative path: (size, mtime_ns)}`` of every entry under ``directory``."""
    out = {}
    for path in sorted(directory.rglob("*")):
        st = path.stat()
        out[str(path.relative_to(directory))] = (st.st_size, st.st_mtime_ns)
    return out
