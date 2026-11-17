# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""No run outlives a Python process killed with SIGKILL (real wrapper, mpirun, systemd).

The Python process driving the run is a child ``python -c`` script; the
"getdp" is a sleeping shim (``MPXLITE_GETDP_BINARY``) started through the
bundled wrapper, so the real ``mpirun`` / ``setpriv`` / ``systemd-run``
chain is exercised without any solver memory. Only scopes created by
these tests are ever killed: the reaper is checked to leave a live
owner's scope alone (other users of mpxlite may share the node).
"""

from __future__ import annotations

import os
import shutil  # noqa: F401 (string-form skipif condition)
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from mpxlite import hard_cap_available  # noqa: F401 (string-form skipif condition)
from mpxlite.memory import reap_stale_scopes
from tests.integration.rod import (  # noqa: F401 (string-form skipif condition)
    DATA_DIR,
    MPI_GETDP_ENV,
    mpi_getdp_available,
)

pytestmark = pytest.mark.integration

SRC = Path(__file__).resolve().parents[2] / "src"
MiB = 1024**2

# String-form conditions: evaluated at test setup, never at collection.
_requires_caps = pytest.mark.skipif(
    "not hard_cap_available()", reason="hard memory caps unavailable (no systemd user manager)"
)
_requires_mpi = pytest.mark.skipif(
    "not (mpi_getdp_available() and shutil.which('mpirun') and shutil.which('setpriv'))",
    reason=f"no MPI getdp (set {MPI_GETDP_ENV}), mpirun or setpriv",
)


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            state = f.read().rsplit(b")", 1)[1].split()[0]
    except OSError:
        return False
    return state not in (b"Z", b"X")


def _wait_for(predicate: object, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():  # type: ignore[operator]
            return True
        time.sleep(0.05)
    return bool(predicate())  # type: ignore[operator]


def _shim(tmp_path: Path, body: str) -> Path:
    """Fake getdp: records ``<pid>`` in ``pids/`` of its cwd (the run dir), then ``body``."""
    path = tmp_path / "shim.sh"
    path.write_text(
        '#!/bin/bash\nmkdir -p "$PWD/pids"; echo $$ > "$PWD/pids/$$"\n' + body, encoding="utf-8"
    )
    path.chmod(0o755)
    return path


def _spawn_driver(tmp_path: Path, shim: Path, np: int, cap: int | None) -> subprocess.Popen[str]:
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(SRC)!r})
        from mpxlite import LaunchSpec, SolverLite
        solver = SolverLite({str(DATA_DIR)!r}, "rod", "rod.msh", "Static",
                            resource_files=[{str(DATA_DIR / "rod.formulation.pro")!r}],
                            runs_dir={str(tmp_path / "runs")!r})
        solver.run("mid", run_id="r", memory_max={cap!r},
                   launch=LaunchSpec(np={np}, env={{"MPXLITE_GETDP_BINARY": {str(shim)!r}}}))
        """
    )
    return subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _pids(run_dir: Path, n: int) -> list[int]:
    pid_dir = run_dir / "pids"
    assert _wait_for(lambda: pid_dir.is_dir() and len(list(pid_dir.iterdir())) >= n)
    time.sleep(0.3)
    return [int(p.name) for p in pid_dir.iterdir()]


def _kill(pids: list[int]) -> None:
    for pid in pids:
        if _alive(pid):
            os.kill(pid, signal.SIGKILL)


def _scope_of(pid: int) -> str:
    return Path(f"/proc/{pid}/cgroup").read_text().strip().rsplit("/", 1)[-1]


def _unit_state(unit: str) -> str:
    out = subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", "LoadState", "--value"],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip()


@pytest.mark.parametrize("capped", [False, pytest.param(True, marks=_requires_caps)])
@pytest.mark.parametrize("np", [1, pytest.param(2, marks=_requires_mpi)])
def test_sigkilled_python_leaves_no_run_behind(tmp_path: Path, np: int, capped: bool) -> None:
    """PR_SET_PDEATHSIG on the run, ``setpriv --pdeathsig`` on every MPI rank."""
    shim = _shim(tmp_path, "exec sleep 7381\n")
    driver = _spawn_driver(tmp_path, shim, np, 256 * MiB if capped else None)
    pids: list[int] = []
    try:
        pids = _pids(tmp_path / "runs" / "r", np)
        assert len(pids) == np and all(_alive(p) for p in pids)
        driver.kill()
        driver.communicate(timeout=30)
        assert _wait_for(lambda: not any(_alive(p) for p in pids), 10), "a run process survived"
    finally:
        driver.kill()
        _kill(pids)


@_requires_caps
def test_reaper_spares_live_owner_and_kills_dead_owners_scope(tmp_path: Path) -> None:
    """A process that escaped PDEATHSIG keeps the scope of a dead owner alive: reaped.

    The shim starts a background ``sleep`` (a grandchild: no parent-death
    signal) inside the run's scope, then sleeps itself.
    """
    shim = _shim(tmp_path, 'sleep 7382 &\necho $! > "$PWD/pids/bg"\nexec sleep 7383\n')
    driver = _spawn_driver(tmp_path, shim, 1, 256 * MiB)
    run_dir = tmp_path / "runs" / "r"
    leftovers: list[int] = []
    try:
        assert _wait_for(lambda: (run_dir / "pids" / "bg").is_file())
        bg = int((run_dir / "pids" / "bg").read_text())
        leftovers = [bg, *(int(p.name) for p in (run_dir / "pids").iterdir() if p.name.isdigit())]
        unit = _scope_of(bg)
        assert unit.startswith("mpxlite-r-") and unit.endswith(".scope")

        assert unit not in reap_stale_scopes()  # its owner is alive: never touched
        assert _alive(bg) and _unit_state(unit) == "loaded"

        driver.kill()
        driver.communicate(timeout=30)
        time.sleep(0.5)
        assert _alive(bg) and _unit_state(unit) == "loaded"  # the orphan the reaper is for

        assert unit in reap_stale_scopes()
        assert _wait_for(lambda: not _alive(bg), 10)
        assert _wait_for(lambda: _unit_state(unit) == "not-found", 10)
    finally:
        driver.kill()
        _kill(leftovers)
