# Copyright (c) 2025 MPhysX
# SPDX-License-Identifier: MIT

"""Unit tests for mpxlite.executor (no real getdp: SolverLite.run is mocked)."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pytest_mock import MockerFixture

from mpxlite import memory
from mpxlite.base import (
    CASE_EXTENSIONS,
    GetDPCancelledError,
    GetDPError,
    GetDPMemoryError,
    SolverLite,
)
from mpxlite.executor import (
    Executor,
    LocalExecutor,
    RunOutcome,
    RunRequest,
    SerialExecutor,
    _find_block,
)
from mpxlite.launch import LaunchSpec
from mpxlite.memory import UNBOUNDED, HardCapUnavailableError
from mpxlite.result import SolverResult

# ----- helpers ------------------------------------------------------------


@pytest.fixture(autouse=True)
def caps_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real systemd probe: hard caps are unavailable unless a test says otherwise."""
    _set_caps(monkeypatch, False)


def _set_caps(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    monkeypatch.setattr(memory, "_probe_state", (available, "" if available else "test: no caps"))
    monkeypatch.setattr(memory, "_warned_unavailable", False)


@pytest.fixture
def solver(tmp_path: Path) -> SolverLite:
    project = tmp_path / "proj"
    project.mkdir()
    for ext in CASE_EXTENSIONS:
        (project / f"demo{ext}").write_text("// x\n", encoding="utf-8")
    (project / "demo.msh").write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n", encoding="utf-8")
    return SolverLite(project, "demo", "demo.msh", "Res")


class Recorder:
    """Fake ``SolverLite.run``: records calls, concurrency and CPU blocks."""

    def __init__(
        self,
        delay: float = 0.0,
        fail: Callable[[str | None], bool] | None = None,
        barrier: threading.Barrier | None = None,
        block_until_cancel: bool = False,
        interrupt: Callable[[str | None], bool] | None = None,
    ) -> None:
        self.delay = delay
        self.fail = fail
        self.barrier = barrier
        self.block_until_cancel = block_until_cancel
        self.interrupt = interrupt
        self.cancels: dict[str, threading.Event | None] = {}
        self.cancelled: list[str] = []
        self.lock = threading.Lock()
        self.active: dict[str, tuple[int, ...] | None] = {}
        self.max_active = 0
        self.starts: list[str] = []
        self.launches: dict[str, LaunchSpec | None] = {}
        self.overlaps: list[tuple[str, str]] = []
        self.petsc: dict[str, Any] = {}
        self.memory: dict[str, int | None] = {}
        self.read_solutions: dict[str, Any] = {}

    def _enter(self, rid: str, launch: LaunchSpec | None) -> None:
        cpus = launch.cpus if launch is not None else None
        with self.lock:
            for other, other_cpus in self.active.items():
                if cpus and other_cpus and set(cpus) & set(other_cpus):
                    self.overlaps.append((rid, other))
            self.active[rid] = cpus
            self.max_active = max(self.max_active, len(self.active))
            self.starts.append(rid)
            self.launches[rid] = launch

    def __call__(
        self,
        solver: SolverLite,
        postop: Any = None,
        *,
        workflow: Any = None,
        run_id: str | None = None,
        launch: LaunchSpec | None = None,
        timeout: float | None = None,
        cancel: threading.Event | None = None,
        petsc_options: Any = None,
        memory_max: int | None = None,
        read_solution: Any = None,
    ) -> SolverResult:
        rid = run_id or "anon"
        self.petsc[rid] = petsc_options
        self.memory[rid] = memory_max
        self.read_solutions[rid] = read_solution
        if cancel is not None and cancel.is_set():
            with self.lock:
                self.cancelled.append(rid)
            raise GetDPCancelledError(["getdp"], started=False)
        self._enter(rid, launch)
        self.cancels[rid] = cancel
        try:
            if self.interrupt is not None and self.interrupt(rid):
                raise KeyboardInterrupt
            if self.block_until_cancel:
                assert cancel is not None and cancel.wait(timeout=10)
                time.sleep(0.05)  # session cleanup
                with self.lock:
                    self.cancelled.append(rid)
                raise GetDPCancelledError(["getdp"])
            if self.barrier is not None:
                self.barrier.wait()
            time.sleep(self.delay)
            if self.fail is not None and self.fail(rid):
                raise GetDPError(1, ["getdp"], reason=f"boom {rid}")
            work = solver.runs_dir / rid
            return SolverResult(work, rid, work / "getdp.log", self.delay, 0, ("getdp",))
        finally:
            with self.lock:
                del self.active[rid]


def _patch(mocker: MockerFixture, rec: Recorder) -> None:
    mocker.patch.object(SolverLite, "run", autospec=True, side_effect=rec)


def _reqs(solver: SolverLite, n: int, **kw: Any) -> list[RunRequest]:
    return [RunRequest(solver, run_id=f"r{i}", **kw) for i in range(n)]


# ----- data classes -------------------------------------------------------


def test_outcome_ok(solver: SolverLite) -> None:
    req = RunRequest(solver)
    res = SolverResult(Path("/x"), "x", Path("/x/getdp.log"), 0.0, 0, ())
    assert RunOutcome(req, res, None).ok
    assert not RunOutcome(req, None, ValueError()).ok
    assert not RunOutcome(req, None, None).ok


def test_executors_satisfy_protocol() -> None:
    executors: list[Executor] = [
        SerialExecutor(),
        LocalExecutor(cpus=[0], memory_per_run=UNBOUNDED),
    ]
    assert all(hasattr(e, "submit") for e in executors)


def test_find_block() -> None:
    assert _find_block([True, True, True], 2) == 0
    assert _find_block([False, True, False, True, True], 2) == 3
    assert _find_block([True, False, True], 2) is None
    assert _find_block([], 1) is None


# ----- SerialExecutor -----------------------------------------------------


def test_serial_order_errors_and_launch_untouched(
    solver: SolverLite, mocker: MockerFixture
) -> None:
    rec = Recorder(fail=lambda rid: rid == "r1")
    _patch(mocker, rec)
    spec = LaunchSpec(np=2)
    out = SerialExecutor().submit(_reqs(solver, 3, launch=spec, postop="po"))
    assert [o.request.run_id for o in out] == ["r0", "r1", "r2"]
    assert [o.ok for o in out] == [True, False, True]
    assert isinstance(out[1].error, GetDPError)
    assert rec.starts == ["r0", "r1", "r2"]
    assert rec.max_active == 1
    assert all(rec.launches[r] is spec for r in rec.starts)


@pytest.mark.parametrize(
    "make", [SerialExecutor, lambda: LocalExecutor(cpus=range(2), memory_per_run=UNBOUNDED)]
)
def test_petsc_options_passed_through(
    solver: SolverLite, mocker: MockerFixture, make: Callable[[], Any]
) -> None:
    rec = Recorder()
    _patch(mocker, rec)
    opts = {"pc_factor_mat_solver_type": "mumps", "ksp_view": None}
    reqs = [RunRequest(solver, run_id="a", petsc_options=opts), RunRequest(solver, run_id="b")]
    out = make().submit(reqs)
    assert all(o.ok for o in out)
    assert rec.petsc == {"a": opts, "b": None}
    assert RunRequest(solver).petsc_options is None


def test_serial_and_local_parity(solver: SolverLite, mocker: MockerFixture) -> None:
    _patch(mocker, Recorder(fail=lambda rid: rid in {"r2", "r4"}))
    reqs = _reqs(solver, 6)
    serial = SerialExecutor().submit(reqs)
    local = LocalExecutor(cpus=range(4), memory_per_run=UNBOUNDED).submit(reqs)
    for s, loc, req in zip(serial, local, reqs, strict=True):
        assert s.request is req and loc.request is req
        assert s.ok == loc.ok
        if s.ok:
            assert s.result == loc.result
        else:
            assert type(s.error) is type(loc.error)


# ----- LocalExecutor: construction -----------------------------------------


def test_local_default_cpus_is_affinity() -> None:
    import os

    assert LocalExecutor(memory_per_run=1).cpus == tuple(sorted(os.sched_getaffinity(0)))


def test_local_default_cpus_without_affinity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Platforms without sched_getaffinity (macOS): 0 .. cpu_count-1."""
    import os

    monkeypatch.delattr(os, "sched_getaffinity")
    monkeypatch.setattr(os, "cpu_count", lambda: 3)
    assert LocalExecutor(memory_per_run=1).cpus == (0, 1, 2)
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert LocalExecutor(memory_per_run=1).cpus == (0,)


def test_local_numpy_cpu_ids(solver: SolverLite, mocker: MockerFixture) -> None:
    """Regression (review S4): numpy CPU ids no longer abort the whole batch."""
    np = pytest.importorskip("numpy")
    rec = Recorder()
    _patch(mocker, rec)
    ex = LocalExecutor(cpus=np.arange(12, 14), memory_per_run=64 << 20, hard_cap=False)
    assert ex.cpus == (12, 13) and all(type(c) is int for c in ex.cpus)
    out = ex.submit(_reqs(solver, 3))
    assert all(o.ok for o in out), [o.error for o in out]
    assert {o.launch.cpus for o in out if o.launch is not None} <= {(12,), (13,)}
    assert ex._free == [True, True]


@pytest.mark.parametrize("bad", [[0, True], [0, 1.0], "01"])
def test_local_bad_cpu_ids_rejected(bad: Any) -> None:
    with pytest.raises(TypeError):
        LocalExecutor(cpus=bad, memory_per_run=1)
    np = pytest.importorskip("numpy")
    with pytest.raises(TypeError):
        LocalExecutor(cpus=[np.True_], memory_per_run=1)


def test_local_bad_pinned_spec_fails_alone(
    solver: SolverLite, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An error while building the pinned spec is that request's outcome, not the batch's."""
    import mpxlite.executor as executor_mod

    rec = Recorder()
    _patch(mocker, rec)
    real_replace = executor_mod.dataclasses.replace

    def replace(obj: Any, **changes: Any) -> Any:
        if changes.get("cpus") == (0, 1):  # the pinned spec of the 2-wide request
            raise TypeError("boom")
        return real_replace(obj, **changes)

    monkeypatch.setattr(executor_mod.dataclasses, "replace", replace)
    ex = LocalExecutor(cpus=[0, 1], memory_per_run=UNBOUNDED)
    reqs = [
        RunRequest(solver, run_id=f"r{i}", launch=LaunchSpec(np=2 if i else 1)) for i in range(2)
    ]
    out = ex.submit(reqs)
    assert out[0].ok
    assert isinstance(out[1].error, TypeError) and "boom" in str(out[1].error)
    assert out[1].launch is None
    assert ex._free == [True, True]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cpus": []},
        {"cpus": [0, 0]},
        {"cpus": [-1]},
        {"cpus": [0], "memory_bytes": 10},
        {"cpus": [0], "memory_bytes": 10, "memory_per_run": 11},
        {"cpus": [0], "memory_bytes": 10, "memory_per_run": 0},
    ],
)
def test_local_bad_construction(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        LocalExecutor(**kwargs)


def test_local_memory_default_is_80pc_of_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "mem_available", lambda: 10_000_000)
    ex = LocalExecutor(cpus=[0], memory_per_run=1)
    assert ex.memory_bytes == 8_000_000
    assert LocalExecutor(cpus=[0], memory_bytes=5, memory_per_run=1).memory_bytes == 5


def test_local_memory_per_run_required() -> None:
    with pytest.raises(ValueError, match="requires memory_per_run") as info:
        LocalExecutor(cpus=[0])
    assert "UNBOUNDED" in str(info.value) and "OOM" in str(info.value)


def test_local_unbounded_warns_and_disables_accounting(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="mpxlite"):
        ex = LocalExecutor(cpus=[0], memory_per_run=UNBOUNDED)
    assert ex.memory_bytes is None and ex.memory_per_run is UNBOUNDED
    assert any("UNBOUNDED" in r.message for r in caplog.records)
    with pytest.raises(ValueError, match="meaningless"):
        LocalExecutor(cpus=[0], memory_bytes=10, memory_per_run=UNBOUNDED)


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"memory_per_run": 1.5}, TypeError),
        ({"memory_per_run": True}, TypeError),
        ({"memory_per_run": 1, "memory_bytes": 2.0}, TypeError),
        ({"memory_per_run": 1, "memory_bytes": 0}, ValueError),
        ({"memory_per_run": UNBOUNDED, "hard_cap": "yes"}, TypeError),
    ],
)
def test_local_bad_memory_types(kwargs: dict[str, Any], exc: type[Exception]) -> None:
    with pytest.raises(exc):
        LocalExecutor(cpus=[0], **kwargs)


# ----- memory: per-request estimates and hard caps -------------------------


def test_local_request_memory_overrides_default(solver: SolverLite, mocker: MockerFixture) -> None:
    rec = Recorder(delay=0.03)
    _patch(mocker, rec)
    ex = LocalExecutor(cpus=range(8), memory_bytes=3_000, memory_per_run=1_000)
    reqs = [RunRequest(solver, run_id=f"big{i}", memory_max=3_000) for i in range(2)]
    reqs += _reqs(solver, 3)
    out = ex.submit(reqs)
    assert all(o.ok for o in out)
    assert rec.max_active <= 3
    assert ex._memory_free == 3_000


def test_local_request_memory_alone_serializes(solver: SolverLite, mocker: MockerFixture) -> None:
    rec = Recorder(delay=0.03)
    _patch(mocker, rec)
    ex = LocalExecutor(cpus=range(8), memory_bytes=3_000, memory_per_run=1_000)
    out = ex.submit([RunRequest(solver, run_id=f"r{i}", memory_max=2_000) for i in range(3)])
    assert all(o.ok for o in out)
    assert rec.max_active == 1


@pytest.mark.parametrize(("value", "exc"), [(5_000, ValueError), (0, ValueError), (1.0, TypeError)])
def test_local_bad_request_memory_rejected(
    solver: SolverLite, mocker: MockerFixture, value: Any, exc: type[Exception]
) -> None:
    rec = Recorder()
    _patch(mocker, rec)
    ex = LocalExecutor(cpus=range(2), memory_bytes=3_000, memory_per_run=1_000)
    out = ex.submit([RunRequest(solver, run_id="bad", memory_max=value), *_reqs(solver, 1)])
    assert isinstance(out[0].error, exc) and out[0].launch is None
    assert out[1].ok
    assert rec.starts == ["r0"]


def test_local_caps_passed_when_available(
    solver: SolverLite, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_caps(monkeypatch, True)
    rec = Recorder()
    _patch(mocker, rec)
    ex = LocalExecutor(cpus=range(2), memory_bytes=3_000, memory_per_run=1_000)
    out = ex.submit([RunRequest(solver, run_id="a"), RunRequest(solver, run_id="b", memory_max=7)])
    assert rec.memory == {"a": 1_000, "b": 7}
    assert [o.memory_max for o in out] == [1_000, 7]


def test_local_caps_unavailable_auto_warns_once_runs_uncapped(
    solver: SolverLite, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    rec = Recorder()
    _patch(mocker, rec)
    with caplog.at_level(logging.WARNING, logger="mpxlite"):
        ex = LocalExecutor(cpus=range(2), memory_bytes=3_000, memory_per_run=1_000)
        LocalExecutor(cpus=range(2), memory_bytes=3_000, memory_per_run=1_000)
    assert sum("NOT capped" in r.message for r in caplog.records) == 1
    out = ex.submit(_reqs(solver, 2))
    assert all(o.ok and o.memory_max is None for o in out)
    assert rec.memory == {"r0": None, "r1": None}


def test_local_hard_cap_required_but_unavailable() -> None:
    with pytest.raises(HardCapUnavailableError, match="test: no caps"):
        LocalExecutor(cpus=[0], memory_per_run=1, hard_cap=True)


def test_local_hard_cap_off_never_probes(
    solver: SolverLite, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory, "_probe_state", None)
    probe = mocker.patch("mpxlite.memory._probe", return_value=(True, ""))
    rec = Recorder()
    _patch(mocker, rec)
    ex = LocalExecutor(cpus=range(2), memory_bytes=3_000, memory_per_run=1_000, hard_cap=False)
    assert all(o.ok for o in ex.submit(_reqs(solver, 2)))
    probe.assert_not_called()
    assert rec.memory == {"r0": None, "r1": None}


def test_local_unbounded_request_memory_caps_without_accounting(
    solver: SolverLite, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_caps(monkeypatch, True)
    rec = Recorder()
    _patch(mocker, rec)
    ex = LocalExecutor(cpus=range(2), memory_per_run=UNBOUNDED)
    out = ex.submit([RunRequest(solver, run_id="a", memory_max=10**15), *_reqs(solver, 1)])
    assert all(o.ok for o in out)
    assert rec.memory == {"a": 10**15, "r0": None}


def test_memory_error_fails_alone(solver: SolverLite, mocker: MockerFixture) -> None:
    def run(solver: SolverLite, postop: Any = None, **kw: Any) -> SolverResult:
        if kw["run_id"] == "hog":
            raise GetDPMemoryError(kw["memory_max"] or 1, ["getdp"])
        time.sleep(0.02)
        work = solver.runs_dir / kw["run_id"]
        return SolverResult(work, kw["run_id"], work / "getdp.log", 0.0, 0, ("getdp",))

    mocker.patch.object(SolverLite, "run", autospec=True, side_effect=run)
    ex = LocalExecutor(cpus=range(2), memory_bytes=3_000, memory_per_run=1_000)
    out = ex.submit([RunRequest(solver, run_id="hog"), *_reqs(solver, 3)])
    assert [o.ok for o in out] == [False, True, True, True]
    assert isinstance(out[0].error, GetDPMemoryError)
    assert ex._memory_free == 3_000 and ex._free == [True, True]


def test_serial_memory_caps(
    solver: SolverLite, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = Recorder()
    _patch(mocker, rec)
    SerialExecutor().submit([RunRequest(solver, run_id="a", memory_max=5)])
    assert rec.memory == {"a": None}  # caps unavailable: auto mode runs uncapped
    _set_caps(monkeypatch, True)
    ex = SerialExecutor(memory_per_run=100)
    out = ex.submit([RunRequest(solver, run_id="b"), RunRequest(solver, run_id="c", memory_max=5)])
    assert rec.memory == {"a": None, "b": 100, "c": 5}
    assert [o.memory_max for o in out] == [100, 5]
    bad = ex.submit([RunRequest(solver, run_id="d", memory_max=-1)])
    assert isinstance(bad[0].error, ValueError) and "d" not in rec.starts
    assert (
        SerialExecutor(memory_per_run=100, hard_cap=False)
        .submit([RunRequest(solver, run_id="e")])[0]
        .ok
    )
    assert rec.memory["e"] is None


def test_serial_hard_cap_required_but_unavailable() -> None:
    with pytest.raises(HardCapUnavailableError):
        SerialExecutor(hard_cap=True)
    with pytest.raises(ValueError):
        SerialExecutor(memory_per_run=0)


# ----- LocalExecutor: scheduling -------------------------------------------


def test_local_runs_concurrently_in_input_order(solver: SolverLite, mocker: MockerFixture) -> None:
    rec = Recorder(barrier=threading.Barrier(4, timeout=5))
    _patch(mocker, rec)
    out = LocalExecutor(cpus=[10, 11, 12, 13], memory_per_run=UNBOUNDED).submit(_reqs(solver, 4))
    assert all(o.ok for o in out), [o.error for o in out]
    assert [o.request.run_id for o in out] == ["r0", "r1", "r2", "r3"]
    assert rec.max_active <= 4
    assert sorted(o.launch.cpus for o in out if o.launch) == [(10,), (11,), (12,), (13,)]


def test_local_blocks_never_overlap(solver: SolverLite, mocker: MockerFixture) -> None:
    rec = Recorder(delay=0.02)
    _patch(mocker, rec)
    widths = [1, 2, 3, 1, 4, 2, 1, 1, 3, 2, 2, 1]
    reqs = [
        RunRequest(solver, run_id=f"r{i}", launch=LaunchSpec(np=w)) for i, w in enumerate(widths)
    ]
    out = LocalExecutor(cpus=range(6), memory_per_run=UNBOUNDED).submit(reqs)
    assert all(o.ok for o in out)
    assert rec.overlaps == []
    assert rec.max_active >= 2
    for o, w in zip(out, widths, strict=True):
        assert o.launch is not None and o.launch.cpus is not None
        assert len(o.launch.cpus) == w
        # contiguous in the executor's CPU list
        assert list(o.launch.cpus) == list(range(o.launch.cpus[0], o.launch.cpus[0] + w))


def test_local_width_accounting_ranks_times_threads(
    solver: SolverLite, mocker: MockerFixture
) -> None:
    rec = Recorder(delay=0.05)
    _patch(mocker, rec)
    out = LocalExecutor(cpus=range(8), memory_per_run=UNBOUNDED).submit(
        _reqs(solver, 4, launch=LaunchSpec(np=2, threads=2))
    )
    assert all(o.ok for o in out)
    assert rec.max_active == 2
    assert rec.overlaps == []
    assert all(o.launch and o.launch.cpus and len(o.launch.cpus) == 4 for o in out)


def test_local_width_from_env_default(
    solver: SolverLite, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MPXLITE_GETDP_NP", "3")
    _patch(mocker, Recorder())
    out = LocalExecutor(cpus=range(4), memory_per_run=UNBOUNDED).submit(_reqs(solver, 1))
    assert out[0].launch is not None and out[0].launch.cpus == (0, 1, 2)


def test_local_threads_none_becomes_one(
    solver: SolverLite, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MPXLITE_GETDP_THREADS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    rec = Recorder()
    _patch(mocker, rec)
    reqs = [
        RunRequest(solver, run_id="a", launch=LaunchSpec(np=2)),
        RunRequest(solver, run_id="b"),
        RunRequest(solver, run_id="c", launch=LaunchSpec(np=1, threads=3)),
    ]
    out = LocalExecutor(cpus=range(4), memory_per_run=UNBOUNDED).submit(reqs)
    assert all(o.ok for o in out)
    threads = {
        r: spec.to_env({})["MPXLITE_GETDP_THREADS"]
        for r, spec in rec.launches.items()
        if spec is not None
    }
    assert threads == {"a": "1", "b": "1", "c": "3"}
    assert rec.launches["c"] is not None and len(rec.launches["c"].cpus or ()) == 3


def test_local_respects_inherited_thread_count(
    solver: SolverLite, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MPXLITE_GETDP_THREADS", "2")
    rec = Recorder()
    _patch(mocker, rec)
    req = RunRequest(solver, run_id="x", launch=LaunchSpec(np=2))
    out = LocalExecutor(cpus=range(4), memory_per_run=UNBOUNDED).submit([req])
    assert out[0].ok
    spec = rec.launches["x"]
    assert spec is not None and len(spec.cpus or ()) == 4  # width = 2 ranks x 2 threads
    assert spec.to_env({"MPXLITE_GETDP_THREADS": "2"})["MPXLITE_GETDP_THREADS"] == "2"


def test_local_overrides_submitted_cpus(solver: SolverLite, mocker: MockerFixture) -> None:
    _patch(mocker, Recorder())
    spec = LaunchSpec(np=1, cpus=(99,), env={"FOO": "bar"})
    out = LocalExecutor(cpus=[5], memory_per_run=UNBOUNDED).submit(
        [RunRequest(solver, launch=spec)]
    )
    assert out[0].launch is not None
    assert out[0].launch.cpus == (5,)
    assert dict(out[0].launch.env) == {"FOO": "bar"}
    assert out[0].request.launch is spec


def test_local_memory_cap(solver: SolverLite, mocker: MockerFixture) -> None:
    rec = Recorder(delay=0.03)
    _patch(mocker, rec)
    ex = LocalExecutor(cpus=range(8), memory_bytes=2_500, memory_per_run=1_000)
    out = ex.submit(_reqs(solver, 6))
    assert all(o.ok for o in out)
    assert rec.max_active == 2


def test_local_over_wide_rejected_immediately(solver: SolverLite, mocker: MockerFixture) -> None:
    rec = Recorder()
    _patch(mocker, rec)
    reqs = [
        RunRequest(solver, run_id="ok0"),
        RunRequest(solver, run_id="wide", launch=LaunchSpec(np=5)),
        RunRequest(solver, run_id="ok1", launch=LaunchSpec(np=2, threads=2)),
    ]
    out = LocalExecutor(cpus=range(4), memory_per_run=UNBOUNDED).submit(reqs)
    assert [o.ok for o in out] == [True, False, True]
    assert isinstance(out[1].error, ValueError)
    assert "5 CPUs" in str(out[1].error)
    assert out[1].launch is None
    assert "wide" not in rec.starts


def test_local_invalid_env_np_is_error_outcome(
    solver: SolverLite, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MPXLITE_GETDP_NP", "zero")
    rec = Recorder()
    _patch(mocker, rec)
    out = LocalExecutor(cpus=range(2), memory_per_run=UNBOUNDED).submit(_reqs(solver, 2))
    assert all(isinstance(o.error, ValueError) for o in out)
    assert rec.starts == []


def test_local_errors_captured_batch_continues(solver: SolverLite, mocker: MockerFixture) -> None:
    rec = Recorder(delay=0.01, fail=lambda rid: rid in {"r0", "r3"})
    _patch(mocker, rec)
    out = LocalExecutor(cpus=range(2), memory_per_run=UNBOUNDED).submit(_reqs(solver, 6))
    assert [o.ok for o in out] == [False, True, True, False, True, True]
    assert all(isinstance(o.error, GetDPError) for o in out if not o.ok)
    assert all(o.result is None for o in out if not o.ok)
    assert sorted(rec.starts) == [f"r{i}" for i in range(6)]


def test_local_resources_released_after_failures(solver: SolverLite, mocker: MockerFixture) -> None:
    _patch(mocker, Recorder(fail=lambda rid: True))
    ex = LocalExecutor(cpus=range(2), memory_bytes=2, memory_per_run=1)
    ex.submit(_reqs(solver, 4, launch=LaunchSpec(np=2)))
    assert ex._free == [True, True]
    assert ex._memory_free == 2


def test_local_fifo_wide_not_starved(solver: SolverLite, mocker: MockerFixture) -> None:
    rec = Recorder(delay=0.03)
    _patch(mocker, rec)
    reqs = [RunRequest(solver, run_id=f"n{i}") for i in range(3)]
    reqs.append(RunRequest(solver, run_id="wide", launch=LaunchSpec(np=4)))
    reqs += [RunRequest(solver, run_id=f"m{i}") for i in range(6)]
    out = LocalExecutor(cpus=range(4), memory_per_run=UNBOUNDED).submit(reqs)
    assert all(o.ok for o in out)
    # Strict FIFO: the runs start in submission order, wave by wave; the wide
    # run starts before any later narrow one and runs alone. Within a wave the
    # runs are dispatched in order, but the worker threads record their start
    # in whatever order the scheduler runs them, so a wave is compared as a set.
    starts = rec.starts
    assert set(starts[:3]) == {"n0", "n1", "n2"}
    assert starts[3] == "wide"
    assert set(starts[4:8]) == {"m0", "m1", "m2", "m3"}
    assert set(starts[8:]) == {"m4", "m5"}
    assert rec.max_active <= 4
    assert rec.overlaps == []


def test_local_empty_batch() -> None:
    assert LocalExecutor(cpus=[0], memory_per_run=UNBOUNDED).submit([]) == []
    assert SerialExecutor().submit([]) == []


def test_local_shared_between_threads(solver: SolverLite, mocker: MockerFixture) -> None:
    rec = Recorder(delay=0.01)
    _patch(mocker, rec)
    ex = LocalExecutor(cpus=range(3), memory_per_run=UNBOUNDED)
    results: dict[int, list[RunOutcome]] = {}

    def batch(k: int) -> None:
        results[k] = ex.submit(
            [RunRequest(solver, run_id=f"t{k}-{i}", launch=LaunchSpec(np=2)) for i in range(3)]
        )

    threads = [threading.Thread(target=batch, args=(k,)) for k in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert all(len(results[k]) == 3 and all(o.ok for o in results[k]) for k in range(3))
    assert rec.overlaps == []


# ----- interruption (Ctrl-C) -----------------------------------------------


def test_each_submit_passes_one_cancel_event(solver: SolverLite, mocker: MockerFixture) -> None:
    rec = Recorder()
    _patch(mocker, rec)
    SerialExecutor().submit(_reqs(solver, 2))
    serial = {id(rec.cancels[r]) for r in ("r0", "r1")}
    LocalExecutor(cpus=range(2), memory_per_run=UNBOUNDED).submit(_reqs(solver, 2))
    local = {id(rec.cancels[r]) for r in ("r0", "r1")}
    assert len(serial) == 1 and len(local) == 1 and serial != local
    assert all(isinstance(e, threading.Event) and not e.is_set() for e in rec.cancels.values())


def test_local_interrupt_cancels_in_flight_runs(solver: SolverLite, mocker: MockerFixture) -> None:
    rec = Recorder(block_until_cancel=True)
    _patch(mocker, rec)
    ex = LocalExecutor(cpus=range(2), memory_per_run=UNBOUNDED)
    real_acquire = ex._acquire
    calls: list[int] = []

    def acquire(width: int, reserve: int) -> tuple[int, ...]:
        calls.append(width)
        if len(calls) == 3:  # Ctrl-C while the head of the queue waits for CPUs
            raise KeyboardInterrupt
        return real_acquire(width, reserve)

    mocker.patch.object(ex, "_acquire", side_effect=acquire)
    t0 = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        ex.submit(_reqs(solver, 4))
    assert time.monotonic() - t0 < 5.0
    # In-flight runs were cancelled and had finished before submit re-raised.
    assert sorted(rec.cancelled) == ["r0", "r1"]
    assert rec.active == {}
    assert all(e is not None and e.is_set() for e in rec.cancels.values())
    # Later requests were never started; every CPU block was released.
    assert rec.starts == ["r0", "r1"]
    assert ex._free == [True, True]


def test_local_interrupt_while_waiting_for_results(
    solver: SolverLite, mocker: MockerFixture
) -> None:
    rec = Recorder(block_until_cancel=True)
    _patch(mocker, rec)
    ex = LocalExecutor(cpus=range(4), memory_per_run=UNBOUNDED)
    mocker.patch("mpxlite.executor.Future.result", autospec=True, side_effect=KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        ex.submit(_reqs(solver, 2))
    assert sorted(rec.cancelled) == ["r0", "r1"]
    assert ex._free == [True] * 4


def test_serial_interrupt_stops_batch(solver: SolverLite, mocker: MockerFixture) -> None:
    rec = Recorder(interrupt=lambda rid: rid == "r1")
    _patch(mocker, rec)
    with pytest.raises(KeyboardInterrupt):
        SerialExecutor().submit(_reqs(solver, 3))
    assert rec.starts == ["r0", "r1"]
    event = rec.cancels["r0"]
    assert event is not None and event.is_set()


def test_request_passes_the_solution_file_to_read(
    solver: SolverLite, mocker: MockerFixture, tmp_path: Path
) -> None:
    rec = Recorder()
    _patch(mocker, rec)
    out = SerialExecutor().submit(
        [RunRequest(solver, run_id="a", read_solution=tmp_path / "x.res"), *_reqs(solver, 1)]
    )
    assert all(o.ok for o in out)
    assert rec.read_solutions == {"a": tmp_path / "x.res", "r0": None}
