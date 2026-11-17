# Copyright (c) 2025 MPhysX
# SPDX-License-Identifier: MIT

"""Hard per-run memory caps with the real cgroup v2 / systemd user manager.

Skipped when caps are unavailable (e.g. CI runners without a systemd user
manager). Every cap is small (<= 512 MiB) and the memory hog is a Python
one-liner standing in for ``getdp`` (``MPXLITE_GETDP_BINARY`` shim), so a
failure of the cap mechanism cannot endanger the machine: the hog never
allocates more than ~600 MB.
"""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from mpxlite import (
    GetDPMemoryError,
    GetDPTimeoutError,
    LaunchSpec,
    LocalExecutor,
    RunRequest,
    SolverLite,
    SolverResult,
    hard_cap_available,  # noqa: F401 (string-form skipif condition)
    parse_real_table,
)
from mpxlite.base import _session_pids
from tests.integration.rod import (  # noqa: F401 (string-form skipif condition)
    MPI_GETDP_ENV,
    expected_mid,
    mpi_getdp_available,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(  # string form: evaluated at setup, never at collection
        "not hard_cap_available()",
        reason="hard memory caps unavailable (no systemd user manager / cgroup v2 memory)",
    ),
]

SolverFactory = Callable[..., SolverLite]
MiB = 1024**2

_requires_mpi = pytest.mark.skipif(
    "not mpi_getdp_available()", reason=f"no MPI getdp (set {MPI_GETDP_ENV})"
)


def _mid(result: SolverResult) -> float:
    return float(parse_real_table(result.path("u_mid.dat"))[-1, -1])


def _unit_state(unit: str) -> str:
    out = subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", "LoadState", "--value"],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip()


def _write_shim(path: Path, body: str) -> Path:
    """A fake getdp that records its session id and cgroup in its cwd, then runs ``body``."""
    path.write_text(
        "#!/bin/bash\n"
        'read -r -a st < /proc/self/stat; echo "${st[5]}" > "sid.$$"\n'
        'cat /proc/self/cgroup > "cgroup.$$"\n' + body,
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _hog_shim(tmp_path: Path) -> Path:
    """Allocates (and touches) ``$HOG_MB`` MiB, holds it 3 s, exits 0."""
    code = 'import os, time; x = b"\\1" * (int(os.environ["HOG_MB"]) << 20); time.sleep(3)'
    return _write_shim(tmp_path / "hog_shim.sh", f"exec {sys.executable} -c '{code}'\n")


@pytest.fixture
def reap_sessions() -> Iterator[list[int]]:
    """Session ids to SIGKILL at teardown, should a test leave survivors."""
    sids: list[int] = []
    yield sids
    for sid in sids:
        for pid in _session_pids(sid) or []:
            os.kill(pid, signal.SIGKILL)


def _sids(run_dir: Path) -> set[int]:
    return {int(f.read_text()) for f in run_dir.glob("sid.*")}


def test_capped_run_that_fits_succeeds(make_solver: SolverFactory) -> None:
    result = make_solver().run("mid", run_id="fits", memory_max=256 * MiB)
    assert result.memory_max == 256 * MiB
    assert _mid(result) == pytest.approx(expected_mid(2.0, 8.0), rel=1e-9)


def test_oom_run_fails_alone_in_batch(make_solver: SolverFactory, tmp_path: Path) -> None:
    """One run exceeds its cap: GetDPMemoryError for it, the siblings succeed."""
    hog = _hog_shim(tmp_path)
    solver = make_solver(verbosity=1)
    hog_launch = LaunchSpec(env={"MPXLITE_GETDP_BINARY": str(hog), "HOG_MB": "600"})
    reqs = [
        RunRequest(solver, "mid", run_id="hog", launch=hog_launch, memory_max=192 * MiB),
        *(
            RunRequest(solver, "mid", workflow={"V": v}, run_id=f"ok{i}")
            for i, v in enumerate((1.0, 2.0, 3.0))
        ),
    ]
    cpus = sorted(os.sched_getaffinity(0))[:2]
    ex = LocalExecutor(cpus=cpus, memory_bytes=1024 * MiB, memory_per_run=256 * MiB, hard_cap=True)
    outcomes = ex.submit(reqs)

    hog_out, *others = outcomes
    assert all(o.ok for o in others), [str(o.error) for o in others]
    for o, v in zip(others, (1.0, 2.0, 3.0), strict=True):
        assert o.result is not None and o.memory_max == 256 * MiB
        assert _mid(o.result) == pytest.approx(expected_mid(v, 8.0), rel=1e-9)
    err = hog_out.error
    assert isinstance(err, GetDPMemoryError), err
    assert err.memory_max == 192 * MiB and hog_out.memory_max == 192 * MiB
    assert err.run_dir == solver.runs_dir / "hog"
    assert "MemoryMax=201326592" in str(err) and str(err.run_dir) in str(err)
    assert err.unit is not None and _unit_state(err.unit) == "not-found"  # reset, not leaked
    # the hog really ran inside its own scope
    (cgroup,) = (solver.runs_dir / "hog").glob("cgroup.*")
    assert err.unit in cgroup.read_text()


@_requires_mpi
def test_mpi_ranks_together_exceed_cap(
    make_solver: SolverFactory, tmp_path: Path, reap_sessions: list[int]
) -> None:
    """Two ranks of 160 MiB each fit alone but not together in 256 MiB: all killed."""
    hog = _hog_shim(tmp_path)
    launch = LaunchSpec(np=2, env={"MPXLITE_GETDP_BINARY": str(hog), "HOG_MB": "160"})
    solver = make_solver(verbosity=0)
    with pytest.raises(GetDPMemoryError) as info:
        solver.run("mid", run_id="mpihog", launch=launch, memory_max=256 * MiB, timeout=60)
    run_dir = solver.runs_dir / "mpihog"
    assert info.value.run_dir == run_dir
    sids = _sids(run_dir)
    reap_sessions.extend(sids)
    assert len(list(run_dir.glob("sid.*"))) == 2  # both ranks had started
    assert len(sids) == 1
    assert _session_pids(sids.pop()) == []  # every rank and mpirun are gone


@_requires_mpi
def test_capped_mpi_timeout_kills_whole_session(
    make_solver: SolverFactory, tmp_path: Path, reap_sessions: list[int]
) -> None:
    """Under systemd-run the ranks keep the run's session: the cleanup finds them all."""
    shim = _write_shim(tmp_path / "sleep_shim.sh", "set -m\n(exec sleep 300) &\nexec sleep 300\n")
    launch = LaunchSpec(np=2, env={"MPXLITE_GETDP_BINARY": str(shim)})
    solver = make_solver(verbosity=0)
    run_dir = solver.runs_dir / "hang"
    with pytest.raises(GetDPTimeoutError):
        solver.run("mid", run_id="hang", launch=launch, timeout=3.0, memory_max=256 * MiB)
    assert len(list(run_dir.glob("sid.*"))) == 2
    sids = _sids(run_dir)
    reap_sessions.extend(sids)
    assert len(sids) == 1
    sid = sids.pop()
    assert _session_pids(sid) == []
    # every rank ran in the run's scope, and the scope is gone
    units = {line.rsplit("/", 1)[-1] for f in run_dir.glob("cgroup.*") for line in [f.read_text()]}
    assert len(units) == 1
    (unit,) = units
    unit = unit.strip()
    assert unit.startswith("mpxlite-hang-") and unit.endswith(".scope")
    assert _unit_state(unit) == "not-found"
