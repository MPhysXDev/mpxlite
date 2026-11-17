# Copyright (c) 2025 MPhysX
# SPDX-License-Identifier: MIT

"""Smoke tests on the mpxlite package itself."""

from __future__ import annotations

import re

import mpxlite


def test_package_importable() -> None:
    """The package can be imported."""
    assert mpxlite is not None


def test_package_has_version() -> None:
    """``__version__`` is set and matches PEP 440 (rough check)."""
    version = mpxlite.__version__
    assert isinstance(version, str)
    assert re.match(r"^\d+\.\d+\.\d+", version), version


def test_top_level_exports() -> None:
    """The public API is exported at the package top level."""
    expected = {
        "CASE_EXTENSIONS",
        "GETDP_CPUS_ENV",
        "GETDP_LAUNCHER_ENV",
        "GETDP_NP_ENV",
        "GETDP_THREADS_ENV",
        "GMSH_JOBS_ENV",
        "GmshJobError",
        "GmshJobResult",
        "find_gmsh_jobs",
        "run_gmsh_job",
        "LOG_NAME",
        "MANIFEST_NAME",
        "PETSC_OPTIONS_NAME",
        "PRE_NAME",
        "SOLUTION_NAME",
        "WORKFLOW_NAME",
        "PetscOptionValue",
        "Unbounded",
        "resolve_hard_cap",
        "GetDPCancelledError",
        "GetDPError",
        "GetDPMemoryError",
        "GetDPTimeoutError",
        "HardCapUnavailableError",
        "UNBOUNDED",
        "Executor",
        "LaunchSpec",
        "LocalExecutor",
        "RunOutcome",
        "RunRequest",
        "Scalar",
        "SerialExecutor",
        "SolverLite",
        "SolverResult",
        "WRAPPER_PATH",
        "WorkflowData",
        "engine_info",
        "find_engine",
        "hard_cap_available",
        "installation_home",
        "parse_complex_scalar",
        "parse_complex_table",
        "parse_complex_vector_table",
        "parse_real_table",
    }
    assert set(mpxlite.__all__) == expected
    for name in expected:
        assert hasattr(mpxlite, name), f"{name} not exported"


def test_launch_variables_at_the_top_level() -> None:
    """The launch variables have public names, the values read by LaunchSpec."""
    from mpxlite import launch

    assert mpxlite.GETDP_NP_ENV == launch.ENV_NP == "MPXLITE_GETDP_NP"
    assert mpxlite.GETDP_THREADS_ENV == launch.ENV_THREADS == "MPXLITE_GETDP_THREADS"
    assert mpxlite.GETDP_CPUS_ENV == launch.ENV_CPUS == "MPXLITE_GETDP_CPUS"
    assert mpxlite.GETDP_LAUNCHER_ENV == launch.ENV_LAUNCHER == "MPXLITE_GETDP_LAUNCHER"


def test_wrapper_path_is_bundled_and_executable() -> None:
    """The bundled launcher is reachable and has the executable bit set."""
    import os

    assert mpxlite.WRAPPER_PATH.is_file(), mpxlite.WRAPPER_PATH
    assert os.access(mpxlite.WRAPPER_PATH, os.X_OK)
