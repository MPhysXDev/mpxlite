# Copyright (c) 2025 MPhysX
# SPDX-License-Identifier: MIT

"""mpxlite: platform that runs an engine of external programs.

Generic, solver-agnostic platform that drives the native ``getdp``
binary through a clean Python API: a typed scalar container
(:class:`WorkflowData`), a single end-to-end invocation entry point
(:meth:`SolverLite.run`), a result wrapper (:class:`SolverResult`),
and parsers for the standard ``.dat`` output formats.

Concrete physics solvers plug in by subclassing :class:`SolverLite`
and bundling their own ``.pro`` resources via the ``resource_files``
attribute.

The bundled ``tools/getdp_runner.sh`` wrapper auto-detects the MPI
launcher (srun / mpirun) from the surrounding scheduler
environment so the same Python code runs unchanged on a laptop,
on a single-node SLURM/PBS allocation, and on a multi-node allocation.

Gmsh jobs (:mod:`mpxlite.gmshjob`): a plug-in never imports Gmsh; what it
needs Gmsh to do is done by the jobs of the installation, programs started by
:func:`run_gmsh_job` (files and a parameter document in, files and a report
out, :exc:`GmshJobError` on failure). The pure-NumPy tools of periodic cells
(lattices, symmetry detection on a tagged mesh, asymmetric units, periodic
mesh checks and writing) are in :mod:`mpxlite.cells`.

Memory safety (:mod:`mpxlite.memory`): concurrent runs reserve a memory
estimate out of a budget and, where systemd and cgroup v2 allow it, each
run is hard-capped in its own cgroup scope (:exc:`GetDPMemoryError`).
:data:`WRAPPER_PATH` is its location inside the installed package,
exposed for diagnostic and override purposes.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from mpxlite.base import (
    CASE_EXTENSIONS,
    GetDPCancelledError,
    GetDPError,
    GetDPMemoryError,
    GetDPTimeoutError,
    PetscOptionValue,
    SolverLite,
)
from mpxlite.engine import engine_info, find_engine, installation_home
from mpxlite.executor import (
    Executor,
    LocalExecutor,
    RunOutcome,
    RunRequest,
    SerialExecutor,
)
from mpxlite.gmshjob import (
    GMSH_JOBS_ENV,
    GmshJobError,
    GmshJobResult,
    find_gmsh_jobs,
    run_gmsh_job,
)
from mpxlite.launch import ENV_CPUS, ENV_LAUNCHER, ENV_NP, ENV_THREADS, LaunchSpec
from mpxlite.memory import (
    UNBOUNDED,
    HardCapUnavailableError,
    Unbounded,
    hard_cap_available,
    resolve_hard_cap,
)
from mpxlite.parsers import (
    parse_complex_scalar,
    parse_complex_table,
    parse_complex_vector_table,
    parse_real_table,
)
from mpxlite.result import (
    LOG_NAME,
    MANIFEST_NAME,
    PETSC_OPTIONS_NAME,
    PRE_NAME,
    SOLUTION_NAME,
    WORKFLOW_NAME,
    SolverResult,
)
from mpxlite.workflow import Scalar, WorkflowData

__version__ = "1.1.0"


def _bundled_wrapper_path() -> Path:
    """Path to the bundled ``getdp_runner.sh`` inside the installed package."""
    return Path(str(files("mpxlite").joinpath("tools/getdp_runner.sh")))


WRAPPER_PATH: Path = _bundled_wrapper_path()
"""Absolute path of the bundled ``getdp_runner.sh`` launcher."""

GETDP_NP_ENV: str = ENV_NP
"""Environment variable of the number of MPI ranks of a run, ``MPXLITE_GETDP_NP``."""

GETDP_THREADS_ENV: str = ENV_THREADS
"""Environment variable of the OpenMP threads per rank, ``MPXLITE_GETDP_THREADS``."""

GETDP_CPUS_ENV: str = ENV_CPUS
"""Environment variable of the CPU list a run is confined to, ``MPXLITE_GETDP_CPUS``."""

GETDP_LAUNCHER_ENV: str = ENV_LAUNCHER
"""Environment variable of the explicit MPI launcher of a run, ``MPXLITE_GETDP_LAUNCHER``."""


__all__ = [
    "CASE_EXTENSIONS",
    "GETDP_CPUS_ENV",
    "GETDP_LAUNCHER_ENV",
    "GETDP_NP_ENV",
    "GETDP_THREADS_ENV",
    "GMSH_JOBS_ENV",
    "LOG_NAME",
    "MANIFEST_NAME",
    "PETSC_OPTIONS_NAME",
    "PRE_NAME",
    "SOLUTION_NAME",
    "UNBOUNDED",
    "WORKFLOW_NAME",
    "WRAPPER_PATH",
    "Executor",
    "GetDPCancelledError",
    "GetDPError",
    "GetDPMemoryError",
    "GetDPTimeoutError",
    "GmshJobError",
    "GmshJobResult",
    "HardCapUnavailableError",
    "LaunchSpec",
    "LocalExecutor",
    "PetscOptionValue",
    "RunOutcome",
    "RunRequest",
    "Scalar",
    "SerialExecutor",
    "SolverLite",
    "SolverResult",
    "Unbounded",
    "WorkflowData",
    "engine_info",
    "find_engine",
    "find_gmsh_jobs",
    "hard_cap_available",
    "installation_home",
    "parse_complex_scalar",
    "parse_complex_table",
    "parse_complex_vector_table",
    "parse_real_table",
    "resolve_hard_cap",
    "run_gmsh_job",
]
