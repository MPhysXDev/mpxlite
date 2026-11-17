# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""Process-lifetime safety of runs, with real subprocesses but no getdp.

The driver is a stub shell script (``MPXLITE_GETDP``) that records its pid
in the run directory and sleeps; the Python process driving it is a child
``python -c`` script, so the tests can SIGKILL / SIGTERM it and check that
no run process outlives it. Everything is tiny (``sleep`` and a Python
interpreter) and bounded by timeouts.
"""

from __future__ import annotations

import logging
import math
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from mpxlite import GetDPError, SolverLite
from mpxlite.base import CASE_EXTENSIONS

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux process semantics (/proc, prctl)"
)

SRC = Path(__file__).resolve().parents[2] / "src"
_MSH = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"


def _case(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    for ext in CASE_EXTENSIONS:
        (project / f"demo{ext}").write_text("// x\n", encoding="utf-8")
    (project / "demo.msh").write_text(_MSH, encoding="utf-8")
    return project


def _stub(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "stub-getdp"
    path.write_text('#!/bin/bash\necho $$ > "$PWD/stub.pid"\n' + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            state = f.read().rsplit(b")", 1)[1].split()[0]
    except OSError:
        return False
    return state not in (b"Z", b"X")


def _wait_for(predicate: Any, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


def _read_pid(path: Path) -> int:
    assert _wait_for(lambda: path.is_file() and path.read_text().strip()), f"no {path}"
    return int(path.read_text())


def _driver_script(project: Path, runs: Path, body: str) -> str:
    return textwrap.dedent(
        f"""
        import os, signal, sys, threading, time
        sys.path.insert(0, {str(SRC)!r})
        from mpxlite import SolverLite, LocalExecutor, RunRequest, UNBOUNDED
        solver = SolverLite({str(project)!r}, "demo", "demo.msh", "Res", runs_dir={str(runs)!r})
        """
    ) + textwrap.dedent(body)


def _spawn(tmp_path: Path, stub: Path, body: str) -> tuple[subprocess.Popen[str], Path]:
    project = tmp_path / "proj" if (tmp_path / "proj").is_dir() else _case(tmp_path)
    runs = tmp_path / "runs"
    env = {**os.environ, "MPXLITE_GETDP": str(stub)}
    proc = subprocess.Popen(
        [sys.executable, "-c", _driver_script(project, runs, body)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, runs


def _finish(proc: subprocess.Popen[str]) -> tuple[int, str]:
    try:
        out, _ = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    return proc.returncode, out


# ----- timeout validation (review nit) -----------------------------------------


@pytest.mark.parametrize(
    ("bad", "exc"),
    [(math.nan, ValueError), (-1, ValueError), (True, ValueError), ("10", TypeError)],
)
def test_bad_timeout_rejected_before_anything(tmp_path: Path, bad: Any, exc: type) -> None:
    solver = SolverLite(_case(tmp_path), "demo", "demo.msh", "Res", runs_dir=tmp_path / "runs")
    with pytest.raises(exc, match="timeout"):
        solver.run(timeout=bad)
    assert not (tmp_path / "runs").exists()


# ----- review S2: cleanup after a normal exit ----------------------------------


@pytest.mark.parametrize("code", [0, 3])
def test_leftover_session_processes_terminated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    code: int,
) -> None:
    """A driver that exits leaving a child behind must not leak it (reviewer's leftover.py)."""
    stub = _stub(
        tmp_path, f'sleep 7361 >/dev/null 2>&1 &\necho $! > "$PWD/child.pid"\nexit {code}\n'
    )
    monkeypatch.setenv("MPXLITE_GETDP", str(stub))
    solver = SolverLite(_case(tmp_path), "demo", "demo.msh", "Res", runs_dir=tmp_path / "runs")
    start = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="mpxlite"):
        if code:
            with pytest.raises(GetDPError):
                solver.run(run_id="r")
        else:
            solver.run(run_id="r")
    child = int((tmp_path / "runs" / "r" / "child.pid").read_text())
    assert not _alive(child)
    assert time.monotonic() - start < 5  # SIGTERM ends `sleep` at once: no grace wait
    assert any("left processes in its session" in r.message for r in caplog.records)


def test_clean_run_logs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("MPXLITE_GETDP", str(_stub(tmp_path, "exit 0\n")))
    solver = SolverLite(_case(tmp_path), "demo", "demo.msh", "Res", runs_dir=tmp_path / "runs")
    with caplog.at_level(logging.WARNING, logger="mpxlite"):
        solver.run(run_id="r")
    assert not caplog.records


# ----- review S1: the Python process dies --------------------------------------


def test_sigkill_of_python_kills_the_run(tmp_path: Path) -> None:
    """PR_SET_PDEATHSIG: the run dies with the Python process (reviewer's orphan.py)."""
    stub = _stub(tmp_path, "exec sleep 7362\n")
    proc, runs = _spawn(tmp_path, stub, 'solver.run(run_id="r")\n')
    pid = _read_pid(runs / "r" / "stub.pid")
    assert _alive(pid)
    proc.kill()
    _finish(proc)
    assert _wait_for(lambda: not _alive(pid), 5), "run survived its SIGKILLed parent"


@pytest.mark.parametrize("signame", ["SIGTERM", "SIGHUP"])
def test_termination_signal_cleans_up_then_kills_python(tmp_path: Path, signame: str) -> None:
    """SIGTERM / SIGHUP clean up like Ctrl-C, then the process still dies of the signal.

    The stub records the SIGTERM of the graceful cleanup; the parent-death
    backstop would SIGKILL it without a trace.
    """
    stub = _stub(tmp_path, "trap 'echo term > \"$PWD/term\"; exit 0' TERM\nsleep 7363 &\nwait\n")
    proc, runs = _spawn(tmp_path, stub, 'solver.run(run_id="r")\nprint("not reached")\n')
    pid = _read_pid(runs / "r" / "stub.pid")
    time.sleep(0.2)  # let bash install its trap
    sig = getattr(signal, signame)
    proc.send_signal(sig)
    code, out = _finish(proc)
    assert code == -sig, out
    assert "not reached" not in out
    assert (runs / "r" / "term").is_file(), "the run was not terminated gracefully"
    assert not _alive(pid)


def test_termination_during_executor_submit(tmp_path: Path) -> None:
    stub = _stub(tmp_path, "exec sleep 7364\n")
    body = """
    ex = LocalExecutor(cpus=[0, 1], memory_per_run=UNBOUNDED, hard_cap=False)
    ex.submit([RunRequest(solver, run_id=f"r{i}") for i in range(2)])
    """
    proc, runs = _spawn(tmp_path, stub, body)
    pids = [_read_pid(runs / f"r{i}" / "stub.pid") for i in range(2)]
    proc.terminate()
    code, out = _finish(proc)
    assert code == -signal.SIGTERM, out
    assert not any(_alive(p) for p in pids)


def test_user_handler_is_chained(tmp_path: Path) -> None:
    """A user SIGTERM handler that returns keeps the run going; one that raises cleans up."""
    stub = _stub(tmp_path, 'while [ ! -e "$PWD/go" ]; do sleep 0.05; done\nexit 0\n')
    body = """
    calls = []
    def handler(signum, frame):
        calls.append(signum)
        if len(calls) == 2:
            raise SystemExit(7)
    signal.signal(signal.SIGTERM, handler)
    res = solver.run(run_id="r")
    print("first run ok", calls, flush=True)
    assert signal.getsignal(signal.SIGTERM) is handler  # restored
    solver.run(run_id="r2")
    """
    proc, runs = _spawn(tmp_path, stub, body)
    _read_pid(runs / "r" / "stub.pid")
    proc.terminate()  # handler returns: the run goes on
    time.sleep(0.5)
    (runs / "r" / "go").touch()
    pid2 = _read_pid(runs / "r2" / "stub.pid")
    proc.terminate()  # handler raises SystemExit(7): cleanup, then exit 7
    code, out = _finish(proc)
    assert "first run ok [15]" in out, out
    assert code == 7, out
    assert not _alive(pid2)


def test_ignored_signal_stays_ignored(tmp_path: Path) -> None:
    stub = _stub(tmp_path, 'while [ ! -e "$PWD/go" ]; do sleep 0.05; done\nexit 0\n')
    body = """
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    solver.run(run_id="r")
    print("done", flush=True)
    """
    proc, runs = _spawn(tmp_path, stub, body)
    _read_pid(runs / "r" / "stub.pid")
    proc.terminate()
    time.sleep(0.3)
    (runs / "r" / "go").touch()
    code, out = _finish(proc)
    assert code == 0 and "done" in out, out


def test_atexit_terminates_runs_of_daemon_threads(tmp_path: Path) -> None:
    stub = _stub(tmp_path, "exec sleep 7365\n")
    body = """
    t = threading.Thread(target=solver.run, kwargs={"run_id": "r"}, daemon=True)
    t.start()
    while not os.path.exists(os.path.join(solver.runs_dir, "r", "stub.pid")):
        time.sleep(0.05)
    time.sleep(0.2)
    sys.exit(0)
    """
    proc, runs = _spawn(tmp_path, stub, body)
    code, out = _finish(proc)
    pid = int((runs / "r" / "stub.pid").read_text())
    assert code == 0, out
    assert "interpreter exiting with 1 live getdp run" in out
    assert not _alive(pid)
