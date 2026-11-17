# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""Hermetic environment for the unit tests.

The memory-safety code reads the machine's cgroup (memory headroom, whether
the process lives under the systemd user manager) and the scheduler
environment. The unit tests must not depend on where they run (a CI runner,
a capped scope, a SLURM job), so these probes are neutralised by default;
the tests of the probes themselves use the functions imported at module
import time, with explicit inputs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mpxlite import engine, gmshjob, memory


@pytest.fixture(scope="session")
def fake_engine(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An executable file standing for the engine (it is never run by a unit test)."""
    path = tmp_path_factory.mktemp("engine") / "getdp"
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture(autouse=True)
def _hermetic_engine_env(monkeypatch: pytest.MonkeyPatch, fake_engine: Path) -> None:
    """One named engine, whatever is installed on the machine that runs the tests."""
    monkeypatch.setenv(engine.ENGINE_ENV, str(fake_engine))
    monkeypatch.delenv(engine.HOME_ENV, raising=False)
    monkeypatch.delenv("MPXLITE_GETDP", raising=False)
    monkeypatch.delenv(gmshjob.GMSH_JOBS_ENV, raising=False)


@pytest.fixture(autouse=True)
def _hermetic_memory_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in memory.SCHEDULER_JOB_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(memory, "cgroup_memory_headroom", lambda *a, **k: None)
    monkeypatch.setattr(memory, "scheduler_job_reason", lambda: None)
    monkeypatch.setattr(memory, "reap_stale_scopes_once", lambda: None)
    monkeypatch.setattr(memory, "_warned_escape", False)
