# Copyright (c) 2025 MPhysX
# SPDX-License-Identifier: MIT

"""Fixtures of the real-GetDP integration suite (skipped when no getdp is found)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from mpxlite import SolverLite, WorkflowData, find_engine
from tests.integration.rod import DATA_DIR, snapshot

try:
    import gmsh  # noqa: F401
except ImportError:
    # Without the gmsh module (a Python without an mpxlite installation, e.g. the
    # unit-test job), the modules that import it cannot be collected; they need
    # the engine anyway. The report header says so.
    collect_ignore = [
        "gmsh_cells.py",
        "test_gmsh_job_geometry.py",
        "test_gmsh_job_mesh.py",
        "test_gmsh_job_runner.py",
        "cells",
    ]


def pytest_report_header(config: pytest.Config) -> str | None:
    """Names the integration modules left out for want of the gmsh module."""
    if "collect_ignore" in globals():
        return "no gmsh module: the Gmsh integration tests are not collected"
    return None


def _getdp_available() -> bool:
    try:
        find_engine()
    except FileNotFoundError:
        return False
    return True


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark every test of this directory ``integration``; skip them all without getdp."""
    here = Path(__file__).parent
    available = _getdp_available()
    skip = pytest.mark.skip(reason="no engine (MPXLITE_GETDP_BINARY, or an installed mpxlite)")
    for item in items:
        if here in Path(item.path).parents:
            item.add_marker(pytest.mark.integration)
            if not available:
                item.add_marker(skip)


@pytest.fixture
def make_solver(tmp_path: Path) -> Iterator[Callable[..., SolverLite]]:
    """Factory of rod solvers on the shared, committed data dir; runs go to tmp_path.

    Extra keyword arguments (e.g. ``petsc_options``, ``resource_files``) go
    to :class:`SolverLite`. Asserts at teardown that the shared data
    directory was left untouched.
    """
    before = snapshot(DATA_DIR)

    def factory(v: float = 2.0, s: float = 8.0, verbosity: int = 3, **kwargs: Any) -> SolverLite:
        kwargs.setdefault("resource_files", [DATA_DIR / "rod.formulation.pro"])
        return SolverLite(
            DATA_DIR,
            "rod",
            "rod.msh",
            "Static",
            workflow=WorkflowData({"V": v, "S": s}),
            verbosity=verbosity,
            runs_dir=tmp_path / "runs",
            **kwargs,
        )

    yield factory
    assert snapshot(DATA_DIR) == before, "a run wrote into the shared data directory"
