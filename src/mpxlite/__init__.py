# Copyright (c) 2025 Thierry Valet
# SPDX-License-Identifier: MIT

"""mpxlite — lightweight Python encapsulation of the GetDP FE solver.

Generic, solver-agnostic platform that drives the native ``getdp``
binary through a clean Python API: a typed scalar container
(:class:`WorkflowData`), a single end-to-end invocation entry point
(:meth:`SolverLite.run`), a result wrapper (:class:`SolverResult`),
and parsers for the standard ``.dat`` output formats.

Concrete physics solvers (e.g. ``metalab.Meta3D`` for 3D periodic
metasurfaces) plug in by subclassing :class:`SolverLite` and bundling
their own ``.pro`` resources via the ``resource_files`` attribute.

The bundled ``tools/getdp_runner.sh`` wrapper auto-detects the MPI
launcher (mpirun / srun / mpiexec) from the surrounding scheduler
environment so the same Python code runs unchanged on a laptop,
on a single-node SLURM/PBS allocation, and on a multi-node allocation.
:data:`WRAPPER_PATH` is its location inside the installed package,
exposed for diagnostic and override purposes.
"""

from __future__ import annotations

import os
import shutil
from importlib.resources import files
from pathlib import Path

from mpxlite.base import GetDPError, SolverLite
from mpxlite.parsers import parse_complex_scalar, parse_complex_table, parse_real_table
from mpxlite.result import SolverResult
from mpxlite.workflow import WorkflowData

__version__ = "0.1.0"


def _bundled_wrapper_path() -> Path:
    """Path to the bundled ``getdp_runner.sh`` inside the installed package."""
    return Path(str(files("mpxlite").joinpath("tools/getdp_runner.sh")))


WRAPPER_PATH: Path = _bundled_wrapper_path()
"""Absolute path of the bundled ``getdp_runner.sh`` launcher."""


def find_getdp() -> Path:
    """Locate the actual ``getdp`` binary on the host.

    Search order:

        1. ``MPXLITE_GETDP_BINARY`` environment variable (if set);
        2. ``getdp`` on ``PATH`` (via :func:`shutil.which`).

    Returns:
        Absolute path of an executable ``getdp`` binary.

    Raises:
        FileNotFoundError: if no usable binary is found.
    """
    override = os.environ.get("MPXLITE_GETDP_BINARY")
    if override:
        path = Path(override).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
        raise FileNotFoundError(
            f"MPXLITE_GETDP_BINARY={override!r} is not an executable file."
        )
    found = shutil.which("getdp")
    if found:
        return Path(found).resolve()
    raise FileNotFoundError(
        "getdp binary not found. Install getdp and put it on PATH, or set "
        "MPXLITE_GETDP_BINARY to its absolute path."
    )


__all__ = [
    "GetDPError",
    "SolverLite",
    "SolverResult",
    "WRAPPER_PATH",
    "WorkflowData",
    "find_getdp",
    "parse_complex_scalar",
    "parse_complex_table",
    "parse_real_table",
]
