# Copyright (c) 2025 MPhysX
# SPDX-License-Identifier: MIT

"""Unit tests for mpxlite.memory (no real systemd: subprocess calls are mocked)."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pytest_mock import MockerFixture

import mpxlite
from mpxlite import memory
from mpxlite.memory import (
    DEFAULT_MEMORY_FRACTION,
    UNBOUNDED,
    HardCapUnavailableError,
    MemoryScope,
    cgroup_memory_headroom,
    default_memory_budget,
    hard_cap_available,
    mem_available,
    reap_stale_scopes,
    reap_stale_scopes_once,
    require_hard_cap,
    resolve_hard_cap,
    scheduler_job_reason,
)


@pytest.fixture(autouse=True)
def fresh_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with no cached probe result and no warning issued."""
    monkeypatch.setattr(memory, "_probe_state", None)
    monkeypatch.setattr(memory, "_warned_unavailable", False)


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> Any:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


# ----- /proc/meminfo ----------------------------------------------------------


def test_mem_available_parses_kb(tmp_path: Path) -> None:
    f = tmp_path / "meminfo"
    f.write_text("MemTotal:  32000000 kB\nMemFree: 1 kB\nMemAvailable:   27237076 kB\n")
    assert mem_available(f) == 27237076 * 1024


@pytest.mark.parametrize(
    "text", ["MemTotal: 1 kB\n", "MemAvailable: lots kB\n", "MemAvailable:\n", ""]
)
def test_mem_available_missing_or_malformed(tmp_path: Path, text: str) -> None:
    f = tmp_path / "meminfo"
    f.write_text(text)
    assert mem_available(f) is None


def test_mem_available_unreadable(tmp_path: Path) -> None:
    assert mem_available(tmp_path / "nope") is None


def test_mem_available_real_proc() -> None:
    if not memory.MEMINFO_PATH.is_file():
        pytest.skip("no /proc/meminfo")
    value = mem_available()
    assert value is not None and value > 0


def test_default_budget_is_fraction_of_available(monkeypatch: pytest.MonkeyPatch) -> None:
    assert DEFAULT_MEMORY_FRACTION == 0.8
    monkeypatch.setattr(memory, "mem_available", lambda: 10_000)
    assert default_memory_budget() == 8_000
    assert default_memory_budget(0.5) == 5_000


def test_default_budget_falls_back_to_physical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "mem_available", lambda: None)
    monkeypatch.setattr(memory.os, "sysconf", lambda name: 1000)
    assert default_memory_budget() == 800_000


# ----- sentinel ---------------------------------------------------------------


def test_unbounded_sentinel() -> None:
    assert mpxlite.UNBOUNDED is UNBOUNDED is memory.Unbounded.UNBOUNDED
    assert repr(UNBOUNDED) == "mpxlite.UNBOUNDED"
    assert UNBOUNDED != 0 and UNBOUNDED is not None


# ----- availability probe -----------------------------------------------------


@pytest.fixture
def which(mocker: MockerFixture) -> Iterator[Any]:
    yield mocker.patch("mpxlite.memory.shutil.which", side_effect=lambda name: f"/usr/bin/{name}")


def test_probe_without_systemd_run(mocker: MockerFixture) -> None:
    mocker.patch("mpxlite.memory.shutil.which", return_value=None)
    run = mocker.patch("mpxlite.memory.subprocess.run")
    assert hard_cap_available() is False
    run.assert_not_called()
    with pytest.raises(HardCapUnavailableError, match="not found on PATH"):
        require_hard_cap()


def test_probe_success_and_cached(mocker: MockerFixture, which: Any) -> None:
    run = mocker.patch(
        "mpxlite.memory.subprocess.run", return_value=_completed(stdout=f"{memory._PROBE_CAP}\n")
    )
    assert hard_cap_available() is True
    assert hard_cap_available() is True
    assert run.call_count == 1
    cmd = run.call_args[0][0]
    assert cmd[:4] == ["/usr/bin/systemd-run", "--user", "--scope", "-q"]
    assert f"MemoryMax={memory._PROBE_CAP}" in cmd and "MemorySwapMax=0" in cmd
    assert "OOMPolicy=stop" in cmd
    assert run.call_args.kwargs["stdin"] is subprocess.DEVNULL
    require_hard_cap()  # no raise
    assert hard_cap_available(refresh=True) is True
    assert run.call_count == 2


def test_probe_no_user_manager(mocker: MockerFixture, which: Any) -> None:
    mocker.patch(
        "mpxlite.memory.subprocess.run",
        return_value=_completed(1, stderr="Failed to connect to bus: No medium found\n"),
    )
    assert hard_cap_available() is False
    with pytest.raises(HardCapUnavailableError, match="No medium found"):
        require_hard_cap()


def test_probe_memory_controller_missing(mocker: MockerFixture, which: Any) -> None:
    mocker.patch("mpxlite.memory.subprocess.run", return_value=_completed(stdout="max\n"))
    assert hard_cap_available() is False
    with pytest.raises(HardCapUnavailableError, match="memory controller"):
        require_hard_cap()


@pytest.mark.parametrize("exc", [OSError("boom"), subprocess.TimeoutExpired("x", 1)])
def test_probe_cannot_start(mocker: MockerFixture, which: Any, exc: Exception) -> None:
    mocker.patch("mpxlite.memory.subprocess.run", side_effect=exc)
    assert hard_cap_available() is False


def test_resolve_hard_cap_modes(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[int] = []

    def probe() -> tuple[bool, str]:
        calls.append(1)
        return False, "nope"

    monkeypatch.setattr(memory, "_probe", probe)
    assert resolve_hard_cap(False) is False
    assert calls == []  # "off" never probes
    with caplog.at_level(logging.WARNING, logger="mpxlite"):
        assert resolve_hard_cap(None) is False
        assert resolve_hard_cap(None) is False
    assert [r.message.count("NOT capped") for r in caplog.records] == [1]  # warned once
    assert "nope" in caplog.records[0].message
    with pytest.raises(HardCapUnavailableError, match="nope"):
        resolve_hard_cap(True)
    assert calls == [1]  # cached


def test_resolve_hard_cap_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "_probe", lambda: (True, ""))
    assert resolve_hard_cap(None) is True
    assert resolve_hard_cap(True) is True


@pytest.mark.parametrize("bad", [1, "auto", 0.5])
def test_resolve_hard_cap_type(bad: Any) -> None:
    with pytest.raises(TypeError):
        resolve_hard_cap(bad)


# ----- MemoryScope ------------------------------------------------------------


def test_scope_names_unique_and_prefix(which: Any) -> None:
    a = MemoryScope.for_run("r1", 1 << 20)
    b = MemoryScope.for_run("r1", 1 << 20)
    assert a.unit != b.unit
    assert a.unit.startswith("mpxlite-r1-") and a.unit.endswith(".scope")
    assert len(MemoryScope.for_run("x" * 500, 1).unit) < 200
    prefix = a.command_prefix()
    assert prefix[5] == f"--description=mpxlite run {memory._owner_tag()}"
    assert prefix[:5] + prefix[6:] == [
        "/usr/bin/systemd-run",
        "--user",
        "--scope",
        "-q",
        f"--unit={a.unit}",
        "-p",
        "MemoryMax=1048576",
        "-p",
        "MemorySwapMax=0",
        "-p",
        "OOMPolicy=stop",
    ]


class FakeSystemctl:
    """``systemctl --user`` stand-in: ``show`` returns the queued states in turn."""

    def __init__(self, states: list[str | None]) -> None:
        self.states = states
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, *args: str) -> Any:
        self.calls.append(args)
        if args[0] != "show":
            return _completed()
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return None if state is None else _completed(stdout=state)


@pytest.fixture
def systemctl(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(memory, "_SETTLE_S", 0.5)
    monkeypatch.setattr(memory, "_SETTLE_POLL_S", 0.001)

    def install(states: list[str | None]) -> FakeSystemctl:
        fake = FakeSystemctl(states)
        monkeypatch.setattr(memory, "_systemctl", fake)
        return fake

    return install


_GONE = "LoadState=not-found\nActiveState=inactive\nResult=success\n"
_ACTIVE = "LoadState=loaded\nActiveState=active\nResult=success\n"
_DEACT_OOM = "LoadState=loaded\nActiveState=deactivating\nResult=oom-kill\n"
_FAILED_OOM = "LoadState=loaded\nActiveState=failed\nResult=oom-kill\n"
_FAILED_OTHER = "LoadState=loaded\nActiveState=failed\nResult=signal\n"


def _resets(fake: FakeSystemctl) -> list[tuple[str, ...]]:
    return [c for c in fake.calls if c[0] == "reset-failed"]


def test_oom_detected_after_settling(systemctl: Any) -> None:
    fake = systemctl([_ACTIVE, _ACTIVE, _DEACT_OOM, _FAILED_OOM])
    scope = MemoryScope("u.scope", 1)
    assert scope.oom_killed(settle=True) is True
    assert _resets(fake) == [("reset-failed", "u.scope")]
    assert len([c for c in fake.calls if c[0] == "show"]) == 4


def test_settle_bounded_number_of_systemctl_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scope systemd never finalises: the 5 s settle forks ~20 systemctl, not ~100 (F-11)."""
    fake = FakeSystemctl([_ACTIVE])
    monkeypatch.setattr(memory, "_systemctl", fake)
    clock = [0.0]
    monkeypatch.setattr(memory.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(memory.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    assert MemoryScope("u.scope", 1).oom_killed(settle=True) is False
    shows = len([c for c in fake.calls if c[0] == "show"])
    assert clock[0] >= memory._SETTLE_S  # the bound is kept
    assert shows <= memory._SETTLE_S / 0.25 + 1


def test_oom_seen_while_deactivating_waits_for_failed(systemctl: Any) -> None:
    fake = systemctl([_DEACT_OOM, _FAILED_OOM])
    assert MemoryScope("u.scope", 1).oom_killed(settle=False) is True
    assert _resets(fake) == [("reset-failed", "u.scope")]


def test_clean_end_is_not_oom(systemctl: Any) -> None:
    fake = systemctl([_GONE])
    assert MemoryScope("u.scope", 1).oom_killed(settle=True) is False
    assert _resets(fake) == []


def test_failed_other_reason_is_not_oom_but_reset(systemctl: Any) -> None:
    fake = systemctl([_FAILED_OTHER])
    assert MemoryScope("u.scope", 1).oom_killed(settle=True) is False
    assert _resets(fake) == [("reset-failed", "u.scope")]


def test_no_settle_returns_at_once(systemctl: Any) -> None:
    fake = systemctl([_ACTIVE])
    assert MemoryScope("u.scope", 1).oom_killed(settle=False) is False
    assert len(fake.calls) == 1


def test_settle_is_bounded(systemctl: Any) -> None:
    systemctl([_ACTIVE])
    assert MemoryScope("u.scope", 1).oom_killed(settle=True) is False


def test_systemctl_unusable_is_not_oom(systemctl: Any) -> None:
    systemctl([None])
    assert MemoryScope("u.scope", 1).oom_killed(settle=True) is False


def test_kill_signals_scope_and_resets(systemctl: Any) -> None:
    fake = systemctl([_FAILED_OTHER])
    MemoryScope("u.scope", 1).kill()
    assert fake.calls[0] == ("kill", "--signal=SIGKILL", "u.scope")
    assert _resets(fake) == [("reset-failed", "u.scope")]


def test_systemctl_helper_handles_errors(mocker: MockerFixture) -> None:
    mocker.patch("mpxlite.memory.subprocess.run", side_effect=OSError)
    assert memory._systemctl("show", "x") is None
    run = mocker.patch("mpxlite.memory.subprocess.run", return_value=_completed())
    assert memory._systemctl("show", "x") is not None
    assert run.call_args[0][0][1:] == ["--user", "show", "x"]


def test_scope_unusable_cgroup_keeps_settle_semantics(systemctl: Any) -> None:
    fake = systemctl([_ACTIVE])  # no ControlGroup property: no leftover detection
    assert MemoryScope("u.scope", 1).oom_killed(settle=False) is False
    assert [c[0] for c in fake.calls] == ["show"]


# ----- review S2: leftovers, synchronous OOM, interrupts ----------------------


@pytest.fixture
def cgroup_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake /sys/fs/cgroup with the scope cgroup ``/app.slice/u.scope``."""
    monkeypatch.setattr(memory, "CGROUP_ROOT", tmp_path)
    scope = tmp_path / "app.slice" / "u.scope"
    scope.mkdir(parents=True)
    (scope / "memory.events").write_text("low 0\nhigh 0\nmax 3\noom 1\noom_kill 0\n")
    (scope / "cgroup.procs").write_text("")
    return scope


_CG = "ControlGroup=/app.slice/u.scope\n"


def test_leftover_processes_in_scope_are_killed(systemctl: Any, cgroup_tree: Path) -> None:
    """Regression (review S2): a process left in the scope after a clean exit is killed."""
    (cgroup_tree / "cgroup.procs").write_text("4242\n")
    fake = systemctl([_ACTIVE + _CG, _ACTIVE + _CG, _FAILED_OTHER + _CG])
    assert MemoryScope("u.scope", 1).oom_killed(settle=False) is False
    assert ("kill", "--signal=SIGKILL", "u.scope") in fake.calls
    assert _resets(fake) == [("reset-failed", "u.scope")]


def test_clean_scope_with_empty_cgroup_returns_at_once(systemctl: Any, cgroup_tree: Path) -> None:
    """Regression (review S2): no settle wait when nothing is left (even after a failure)."""
    fake = systemctl([_ACTIVE + _CG, _GONE])
    assert MemoryScope("u.scope", 1).oom_killed(settle=True) is False
    assert [c[0] for c in fake.calls] == ["show", "show"]


def test_oom_read_from_memory_events_before_systemd(systemctl: Any, cgroup_tree: Path) -> None:
    """The kernel counter is authoritative while systemd's Result still says success."""
    (cgroup_tree / "memory.events").write_text("oom 1\noom_kill 1\n")
    (cgroup_tree / "cgroup.procs").write_text("4243\n")  # ranks being stopped: not leftovers
    fake = systemctl([_ACTIVE + _CG, _DEACT_OOM + _CG, _FAILED_OOM + _CG])
    assert MemoryScope("u.scope", 1).oom_killed(settle=False) is True
    assert not [c for c in fake.calls if c[0] == "kill"]
    assert _resets(fake) == [("reset-failed", "u.scope")]


def test_interrupted_settle_still_resets_the_unit(systemctl: Any) -> None:
    """Regression (review nit): Ctrl-C while settling must not leave a failed unit."""
    fake = systemctl([_DEACT_OOM])

    def interrupt(_: float) -> None:
        raise KeyboardInterrupt

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(memory.time, "sleep", interrupt)
        with pytest.raises(KeyboardInterrupt):
            MemoryScope("u.scope", 1).oom_killed(settle=True)
    assert fake.calls[-2:] == [
        ("kill", "--signal=SIGKILL", "u.scope"),
        ("reset-failed", "u.scope"),
    ]


# ----- review S6: cgroup memory headroom --------------------------------------


def _cg(root: Path, rel: str, limit: str | None, current: int) -> None:
    d = root / rel if rel else root
    d.mkdir(parents=True, exist_ok=True)
    if limit is not None:
        (d / "memory.max").write_text(limit + "\n")
    (d / "memory.current").write_text(f"{current}\n")


def test_cgroup_headroom_smallest_limited_ancestor(tmp_path: Path) -> None:
    _cg(tmp_path, "", None, 0)  # the root cgroup has no memory.max
    _cg(tmp_path, "job_42", "1000000", 300000)  # 700000 left
    _cg(tmp_path, "job_42/step_0", "max", 290000)
    _cg(tmp_path, "job_42/step_0/task_0", "800000", 0)  # 800000 left
    assert cgroup_memory_headroom("/job_42/step_0/task_0", tmp_path) == 700000


def test_cgroup_headroom_without_limit_or_v2(tmp_path: Path) -> None:
    _cg(tmp_path, "a/b", "max", 5)
    assert cgroup_memory_headroom("/a/b", tmp_path) is None
    _cg(tmp_path, "c", "100", 500)
    assert cgroup_memory_headroom("/c", tmp_path) == 0  # over its limit: nothing left
    assert memory._own_cgroup(tmp_path / "missing") is None
    v1 = tmp_path / "cgroup.v1"
    v1.write_text("12:memory:/user.slice\n1:name=systemd:/x\n")
    assert memory._own_cgroup(v1) is None
    v2 = tmp_path / "cgroup.v2"
    v2.write_text("0::/user.slice/u.scope\n")
    assert memory._own_cgroup(v2) == "/user.slice/u.scope"


def test_default_budget_bounded_by_cgroup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression (review S6): the budget never exceeds the enclosing cgroup's headroom."""
    monkeypatch.setattr(memory, "mem_available", lambda: 10_000)
    monkeypatch.setattr(memory, "cgroup_memory_headroom", lambda: 3_000)
    assert default_memory_budget() == 3_000
    monkeypatch.setattr(memory, "cgroup_memory_headroom", lambda: 50_000)
    assert default_memory_budget() == 8_000
    monkeypatch.setattr(memory, "cgroup_memory_headroom", lambda: 0)
    assert default_memory_budget() == 1


# ----- review S3: scheduler jobs / foreign cgroups ----------------------------


def test_scheduler_job_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    uid = os.getuid()
    user = f"/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/x.scope"
    monkeypatch.setattr(memory, "_own_cgroup", lambda: user)
    assert scheduler_job_reason() is None
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    assert "SLURM_JOB_ID=42" in (scheduler_job_reason() or "")
    monkeypatch.delenv("SLURM_JOB_ID")
    monkeypatch.setenv("PBS_JOBID", "7.pbs")
    assert "PBS_JOBID" in (scheduler_job_reason() or "")
    monkeypatch.delenv("PBS_JOBID")
    monkeypatch.setattr(memory, "_own_cgroup", lambda: "/system.slice/slurmstepd.scope/job_42")
    assert "not under the systemd user manager" in (scheduler_job_reason() or "")
    monkeypatch.setattr(memory, "_own_cgroup", lambda: None)
    assert scheduler_job_reason() is None  # unknown: left to the probe


def test_auto_caps_disabled_in_job(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Regression (review S3): auto mode never escapes a job's cgroup, explicit True warns."""
    monkeypatch.setattr(memory, "_probe", lambda: (True, ""))
    monkeypatch.setattr(memory, "scheduler_job_reason", lambda: "inside a scheduler job (X=1)")
    with caplog.at_level(logging.WARNING, logger="mpxlite"):
        assert resolve_hard_cap(None) is False
        assert resolve_hard_cap(None) is False
        assert resolve_hard_cap(True) is True
        require_hard_cap("run")
    messages = [r.message for r in caplog.records]
    assert sum("disabled: inside a scheduler job" in m for m in messages) == 1
    assert sum("requested explicitly while inside a scheduler job" in m for m in messages) == 1


def test_unavailable_message_depends_on_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "_probe", lambda: (False, "nope"))
    with pytest.raises(HardCapUnavailableError, match="without memory_max") as run_err:
        require_hard_cap("run")
    assert "hard_cap=False" not in str(run_err.value)
    with pytest.raises(HardCapUnavailableError, match="hard_cap=False"):
        resolve_hard_cap(True)


def test_probe_reports_cgroup_v1_or_hybrid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    mounts = tmp_path / "mounts"
    mounts.write_text(
        "tmpfs /sys/fs/cgroup tmpfs ro 0 0\ncgroup2 /sys/fs/cgroup/unified cgroup2 rw 0 0\n"
    )
    monkeypatch.setattr(memory, "MOUNTS_PATH", mounts)
    run = mocker.patch("mpxlite.memory.subprocess.run")
    ok, reason = memory._probe()
    assert not ok and "cgroup v1 or hybrid" in reason and "'tmpfs'" in reason
    run.assert_not_called()
    mounts.write_text("cgroup2 /sys/fs/cgroup cgroup2 rw 0 0\n")
    assert memory._cgroup_fs_type(mounts) == "cgroup2"


# ----- review S1: stale scopes of dead owners ---------------------------------


def _block(unit: str, description: str) -> str:
    return f"Id={unit}\nDescription={description}\n"


def test_reaper_kills_only_scopes_of_verifiably_dead_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never touch a live owner's scope (other mpxlite users run on the same node)."""
    ns = memory._pid_namespace()
    if ns is None or not Path("/proc/self/stat").exists():
        pytest.skip("needs /proc")
    live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        live_start = memory._proc_start_time(live.pid)
        dead_start = memory._proc_start_time(dead.pid)
        dead.wait()

        def tag(pid: int, start: Any, pidns: Any = ns) -> str:
            return f"mpxlite run [owner pid={pid} start={start} pidns={pidns}]"

        show = "\n".join(
            [
                _block("mpxlite-live-1.scope", tag(live.pid, live_start)),
                _block("mpxlite-dead-1.scope", tag(dead.pid, dead_start)),
                _block("mpxlite-reused-1.scope", tag(live.pid, (live_start or 0) + 1)),
                _block("mpxlite-otherns-1.scope", tag(dead.pid, dead_start, ns + 1)),
                _block("mpxlite-self-1.scope", tag(os.getpid(), memory._proc_start_time("self"))),
                _block("mpxlite-untagged-1.scope", "mpxlite run"),
                _block("mpxlite-probe-abc.scope", "/bin/sh -c ..."),
                _block("other-1.scope", tag(dead.pid, dead_start)),
            ]
        )
        calls: list[tuple[str, ...]] = []

        def fake(*args: str) -> Any:
            calls.append(args)
            if args[0] == "show" and "Description" in args:
                return _completed(stdout=show)
            return _completed(stdout=_GONE)

        monkeypatch.setattr(memory, "_systemctl", fake)
        reaped = reap_stale_scopes()
    finally:
        live.kill()
        live.wait()
    assert reaped == ["mpxlite-dead-1.scope", "mpxlite-reused-1.scope"]
    killed = [c[2] for c in calls if c[0] == "kill"]
    assert killed == reaped
    assert live.returncode is not None  # (killed by the test itself, not by the reaper)


def test_reaper_is_best_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "_systemctl", lambda *a: None)
    assert reap_stale_scopes() == []
    monkeypatch.setattr(memory, "_reaped_once", False)

    calls: list[int] = []

    def boom() -> list[str]:
        calls.append(1)
        raise RuntimeError("x")

    monkeypatch.setattr(memory, "reap_stale_scopes", boom)
    reap_stale_scopes_once()  # swallowed
    reap_stale_scopes_once()  # once per process
    assert calls == [1]
