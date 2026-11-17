# Copyright (c) 2025 MPhysX
# SPDX-License-Identifier: MIT

"""Executors: run batches of :meth:`SolverLite.run` requests.

A :class:`RunRequest` captures the arguments of one :meth:`SolverLite.run`
call; an :class:`Executor` turns a batch of requests into a list of
:class:`RunOutcome` (same order as the input). A failing run never aborts
the batch: its exception is captured in the corresponding outcome.

Two executors are provided:

* :class:`SerialExecutor` runs the requests one after another, in the
  calling thread, with their launch specs untouched;
* :class:`LocalExecutor` runs them concurrently on one many-core machine,
  each on its own block of logical CPUs and its own memory share.

Memory safety (see :mod:`mpxlite.memory`): :class:`LocalExecutor` requires
``memory_per_run``, a peak-memory estimate of one run that is both the
share it reserves out of the memory budget ``memory_bytes`` (default:
:data:`~mpxlite.memory.DEFAULT_MEMORY_FRACTION` of ``MemAvailable`` at
construction) and, when available, the run's *hard cap* (a cgroup v2
``MemoryMax`` of a transient systemd scope). A run that exceeds its cap
fails alone with :exc:`~mpxlite.base.GetDPMemoryError`; the batch goes on.
``RunRequest.memory_max`` overrides the estimate for one run. Packing by
CPU count only requires the explicit opt-out
``memory_per_run=mpxlite.UNBOUNDED``.

Scheduling of :class:`LocalExecutor` (strict FIFO, first-fit):

* the *width* of a request is ``ranks * threads_per_rank`` as the pinned run
  will see it (see :meth:`~mpxlite.launch.LaunchSpec.effective_threads`);
* requests are dispatched in input order. The request at the head of the
  queue waits until a block of ``width`` *contiguous* free CPUs (contiguous
  positions in the executor's sorted CPU list) and, unless the executor is
  :data:`~mpxlite.memory.UNBOUNDED`, the run's memory share is free; later
  requests never overtake it.
  A wide request therefore cannot starve behind narrow ones, at the price
  of idle CPUs while it waits (no backfilling: run durations are unknown);
* the block is the lowest-numbered contiguous free one (first fit). With
  the default CPU set ``sorted(os.sched_getaffinity(0))`` and the usual
  Linux numbering (physical cores first, hyper-threading siblings after),
  blocks land on distinct physical cores before HT siblings are used;
* the scheduler is deadlock-free: every admitted request fits in the whole
  CPU set and memory budget, every running request releases its share when
  it ends (successfully or not), so the head of the queue eventually fits.
  A request that can never fit (wider than the CPU set, or needing more
  memory than the whole budget) is rejected at once with a
  :class:`ValueError` outcome and does not block the queue.

Interruption (``Ctrl-C``): each :meth:`submit` call owns one
:class:`threading.Event`, passed as ``cancel`` to every
:meth:`SolverLite.run` of the batch. When ``submit`` is interrupted by any
:class:`BaseException` (typically :class:`KeyboardInterrupt`), it sets the
event, waits for the in-flight runs to terminate their processes (they end
with :class:`~mpxlite.base.GetDPCancelledError` within the SIGTERM grace
period), then re-raises: no outcome list is returned. Requests not yet
dispatched are never started; a request dispatched but not yet started
raises :class:`~mpxlite.base.GetDPCancelledError` without creating its run
directory. A second ``Ctrl-C`` while waiting propagates at once; the worker
threads still finish their cleanup.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
import threading
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mpxlite.base import PetscOptionValue, SolverLite, _check_memory_max, _termination_signals
from mpxlite.launch import LaunchSpec, as_int
from mpxlite.memory import UNBOUNDED, Unbounded, default_memory_budget, resolve_hard_cap
from mpxlite.result import SolverResult
from mpxlite.workflow import Scalar

logger = logging.getLogger("mpxlite")


@dataclass(frozen=True)
class RunRequest:
    """Arguments of one :meth:`SolverLite.run` call.

    Attributes:
        solver: The (template) solver to run; it is never modified.
        postop: ``PostOperation`` name(s), as for :meth:`SolverLite.run`.
        workflow: Per-run scalar overrides.
        run_id: Name of the run directory (default: generated). Run ids must
            be unique within a batch sharing a ``runs_dir``.
        launch: Parallel layout of the run (default: ``LaunchSpec()``).
        timeout: Wall-clock limit in seconds (default: none).
        petsc_options: Per-run PETSc options, merged over the solver's
            template ones (see :meth:`SolverLite.run`).
        memory_max: Peak-memory estimate of this run, in bytes, overriding
            the executor's ``memory_per_run``: the memory share it reserves
            and its hard cap (see the executors).
        read_solution: A solution file the run reads first, as for
            :meth:`SolverLite.run`.
    """

    solver: SolverLite
    postop: str | Sequence[str] | None = None
    workflow: Mapping[str, Scalar] | None = None
    run_id: str | None = None
    launch: LaunchSpec | None = None
    timeout: float | None = None
    petsc_options: Mapping[str, PetscOptionValue] | None = None
    memory_max: int | None = None
    read_solution: Path | str | None = None


@dataclass(frozen=True)
class RunOutcome:
    """Outcome of one :class:`RunRequest`.

    Exactly one of ``result`` and ``error`` is set.

    Attributes:
        request: The request, as submitted.
        result: The :class:`SolverResult` of a successful run.
        error: The exception raised by a failed (or rejected) run.
        launch: The launch spec the run was actually started with (e.g.
            with the CPU block assigned by :class:`LocalExecutor`); ``None``
            if the run was rejected before being started.
        memory_max: The hard memory cap the run was started with, in bytes
            (``None``: not capped).
    """

    request: RunRequest
    result: SolverResult | None
    error: BaseException | None
    launch: LaunchSpec | None = None
    memory_max: int | None = None

    @property
    def ok(self) -> bool:
        """Whether the run completed successfully."""
        return self.error is None and self.result is not None


class Executor(Protocol):
    """Anything that runs a batch of :class:`RunRequest`."""

    def submit(self, requests: Iterable[RunRequest]) -> list[RunOutcome]:
        """Run ``requests`` and return their outcomes in input order."""
        ...


def _execute(
    request: RunRequest,
    launch: LaunchSpec | None,
    cancel: threading.Event | None,
    memory_max: int | None = None,
    cpus: tuple[int, ...] | None = None,
) -> RunOutcome:
    """Run ``request`` with ``launch`` (and cap), capturing any exception in the outcome.

    With ``cpus``, the run is pinned to them (``launch`` with ``cpus``
    replaced); building that spec is part of the request, so an error
    there fails this request alone.
    """
    try:
        if cpus is not None:
            launch = dataclasses.replace(launch or LaunchSpec(), cpus=cpus)
    except Exception as exc:
        return RunOutcome(request=request, result=None, error=exc, memory_max=memory_max)
    done = functools.partial(RunOutcome, request=request, launch=launch, memory_max=memory_max)
    try:
        result = request.solver.run(
            request.postop,
            workflow=request.workflow,
            run_id=request.run_id,
            launch=launch,
            timeout=request.timeout,
            cancel=cancel,
            petsc_options=request.petsc_options,
            memory_max=memory_max,
            read_solution=request.read_solution,
        )
    except Exception as exc:
        return done(result=None, error=exc)
    return done(result=result, error=None)


class _CapPolicy:
    """Executor ``hard_cap`` option: whether a run's memory value becomes a hard cap."""

    def __init__(self, mode: bool | None, eager: bool) -> None:
        if mode is not None and not isinstance(mode, bool):
            raise TypeError(f"hard_cap must be True, False or None, got {mode!r}")
        self.mode = mode
        self._on: bool | None = None
        if eager or mode is True:
            self._on = resolve_hard_cap(mode)

    def cap(self, memory: int | None) -> int | None:
        """The cap to enforce for a run of estimate ``memory`` (``None``: uncapped)."""
        if memory is None:
            return None
        if self._on is None:  # probed lazily: no cap was needed so far
            self._on = resolve_hard_cap(self.mode)
        return memory if self._on else None


class SerialExecutor:
    """Run requests one after another in the calling thread.

    Launch specs are passed through unchanged (no CPU assignment). Only one
    run is alive at a time, so no memory estimate is required; one given
    (``memory_per_run`` or ``RunRequest.memory_max``) becomes the run's
    hard cap, as for :class:`LocalExecutor`.

    Args:
        memory_per_run: Default hard cap of every run, in bytes (``None``:
            no cap). ``RunRequest.memory_max`` overrides it.
        hard_cap: ``None`` (default) caps runs when caps are available on
            this machine, else logs one warning and runs uncapped;
            ``True`` requires them; ``False`` never caps.

    Raises:
        TypeError: on a ``memory_per_run`` that is not an int, or an
            invalid ``hard_cap``.
        ValueError: on a non-positive ``memory_per_run``.
        HardCapUnavailableError: ``hard_cap=True`` and caps are unavailable.
    """

    def __init__(self, memory_per_run: int | None = None, *, hard_cap: bool | None = None) -> None:
        self.memory_per_run = _check_memory_max(memory_per_run)
        self._caps = _CapPolicy(hard_cap, eager=memory_per_run is not None)
        self.hard_cap = hard_cap

    def _run_one(self, request: RunRequest, cancel: threading.Event) -> RunOutcome:
        try:
            memory = _check_memory_max(request.memory_max)
        except (TypeError, ValueError) as exc:
            return RunOutcome(request=request, result=None, error=exc)
        cap = self._caps.cap(memory if memory is not None else self.memory_per_run)
        return _execute(request, request.launch, cancel, cap)

    def submit(self, requests: Iterable[RunRequest]) -> list[RunOutcome]:
        """Run ``requests`` sequentially.

        On ``Ctrl-C`` the current run terminates its processes and the
        interrupt is re-raised; the remaining requests are not started (see
        the module docstring).

        Args:
            requests: The runs to perform.

        Returns:
            One :class:`RunOutcome` per request, in input order.
        """
        cancel = threading.Event()
        with _termination_signals():
            try:
                return [self._run_one(req, cancel) for req in requests]
            except BaseException:
                cancel.set()
                raise


@dataclass(frozen=True)
class _Prepared:
    """A request admitted by :class:`LocalExecutor`: its resources and cap."""

    spec: LaunchSpec
    width: int
    reserve: int  # bytes reserved out of the budget while it runs
    cap: int | None  # hard memory cap (None: uncapped)


def _default_cpus() -> tuple[int, ...]:
    """CPUs this process may use: its affinity mask, else ``0 .. os.cpu_count() - 1``.

    ``os.sched_getaffinity`` does not exist on every platform (e.g. macOS).
    """
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        return tuple(sorted(getaffinity(0)))
    return tuple(range(os.cpu_count() or 1))


def _cpu_ids(cpus: Sequence[int]) -> tuple[int, ...]:
    """``cpus`` as a tuple of plain ints (numpy integers accepted, bools rejected).

    Raises:
        TypeError: on a string, a bool or a non-integer entry.
    """
    if isinstance(cpus, (str, bytes)):
        raise TypeError("LocalExecutor cpus must be a sequence of ints, not a string")
    return tuple(as_int(cpu, "LocalExecutor cpus entries") for cpu in cpus)


def _find_block(free: Sequence[bool], width: int) -> int | None:
    """Index of the first run of ``width`` consecutive ``True`` in ``free``."""
    run = 0
    for i, is_free in enumerate(free):
        run = run + 1 if is_free else 0
        if run == width:
            return i - width + 1
    return None


class LocalExecutor:
    """Run requests concurrently on the local machine, one CPU block each.

    See the module docstring for the scheduling policy (strict FIFO,
    contiguous first-fit CPU blocks, memory budget).

    Each started run gets ``dataclasses.replace(launch, cpus=block)``: any
    ``cpus`` of the submitted spec is overridden. Because the run is pinned,
    :meth:`LaunchSpec.to_env` defaults its thread count to 1 unless
    ``threads``, ``MPXLITE_GETDP_THREADS`` or ``OMP_NUM_THREADS`` sets it, and
    the width accounting uses the same rule, so it matches what actually
    runs (an unset ``OMP_NUM_THREADS`` lets every rank start a full OpenMP
    pool).

    Memory: every run reserves its memory estimate (``RunRequest.memory_max``,
    else ``memory_per_run``) out of ``memory_bytes`` while it runs, and that
    estimate is also its hard cap (``MemoryMax`` of its own systemd scope,
    see :mod:`mpxlite.memory`), so a wrong estimate makes that run fail
    with :exc:`~mpxlite.base.GetDPMemoryError` instead of exhausting the
    machine. The scopes are siblings of the Python process's own cgroup:
    an outer cap around the Python process does not bound them; the sum of
    the per-run caps, kept within ``memory_bytes`` by the accounting, does.

    One instance may be shared by several threads calling :meth:`submit`
    concurrently: they share its CPU set and memory budget (FIFO order then
    holds within each batch only).

    Args:
        cpus: Logical CPU ids available to the executor (ints; numpy
            integers are accepted). Default: ``sorted(os.sched_getaffinity(0))``,
            or ``range(os.cpu_count())`` where the affinity mask is not
            available (e.g. macOS).
        memory_bytes: Total memory budget, in bytes. Default:
            :data:`~mpxlite.memory.DEFAULT_MEMORY_FRACTION` (80 %) of the
            ``MemAvailable`` of ``/proc/meminfo`` measured now (the
            physical memory where ``/proc/meminfo`` is unusable).
        memory_per_run: Peak-memory estimate of one run, in bytes:
            REQUIRED, since several runs share the machine. It is reserved
            per run and is the run's hard cap. :data:`~mpxlite.UNBOUNDED`
            explicitly disables memory accounting (runs packed by CPU count
            only, at the caller's risk; a warning is logged); a request's
            ``memory_max`` is then never accounted, only applied as its
            hard cap when caps are on (``hard_cap``).
        hard_cap: ``None`` (default) caps runs when caps are available on
            this machine, else logs one warning and runs uncapped (the
            budget still applies); ``True`` requires caps (error now if
            unavailable); ``False`` never caps.

    Raises:
        ValueError: on an empty or duplicated CPU set, a negative CPU id, a
            missing ``memory_per_run``, a non-positive memory amount,
            ``memory_per_run > memory_bytes``, or ``memory_bytes`` with
            :data:`~mpxlite.UNBOUNDED`.
        TypeError: on a CPU id that is not an int (or is a bool), a memory
            amount that is not an int, or an invalid ``hard_cap``.
        HardCapUnavailableError: ``hard_cap=True`` and caps are unavailable.
    """

    def __init__(
        self,
        cpus: Sequence[int] | None = None,
        memory_bytes: int | None = None,
        memory_per_run: int | Unbounded | None = None,
        *,
        hard_cap: bool | None = None,
    ) -> None:
        ids = _default_cpus() if cpus is None else _cpu_ids(cpus)
        if not ids:
            raise ValueError("LocalExecutor needs at least one CPU")
        if len(set(ids)) != len(ids) or any(c < 0 for c in ids):
            raise ValueError(f"LocalExecutor cpus must be unique ids >= 0, got {ids}")
        self.cpus: tuple[int, ...] = ids
        self.memory_bytes = self._check_memory(memory_bytes, memory_per_run)
        self.memory_per_run: int | Unbounded = memory_per_run
        self.hard_cap = hard_cap
        self._caps = _CapPolicy(hard_cap, eager=memory_per_run is not UNBOUNDED)
        self._free = [True] * len(ids)
        self._memory_free = self.memory_bytes or 0
        self._cond = threading.Condition()

    @staticmethod
    def _check_memory(
        memory_bytes: int | None, memory_per_run: int | Unbounded | None
    ) -> int | None:
        """The memory budget (``None``: no accounting); validates both arguments."""
        if memory_per_run is None:
            raise ValueError(
                "LocalExecutor requires memory_per_run (peak memory of one run, in bytes): "
                "it runs several getdp processes at once, and packing them by CPU count "
                "only can exhaust the machine's memory and wake the kernel OOM killer, "
                "which then kills arbitrary processes (desktop, editor...). The estimate "
                "is reserved per run and enforced as its hard cap. Pass "
                "memory_per_run=mpxlite.UNBOUNDED to pack by CPU count only at your own risk."
            )
        if memory_per_run is UNBOUNDED:
            if memory_bytes is not None:
                raise ValueError("memory_bytes is meaningless with memory_per_run=UNBOUNDED")
            logger.warning(
                "LocalExecutor(memory_per_run=UNBOUNDED): runs are packed by CPU count only; "
                "concurrent runs may exhaust the machine's memory"
            )
            return None
        per_run = _check_memory_max(memory_per_run)
        budget = default_memory_budget() if memory_bytes is None else memory_bytes
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise TypeError(f"memory_bytes must be an int (bytes) or None, got {budget!r}")
        if budget <= 0:
            raise ValueError("memory amounts must be > 0")
        if per_run is not None and per_run > budget:
            raise ValueError(f"memory_per_run={per_run} exceeds memory_bytes={budget}")
        return budget

    def _memory_of(self, request: RunRequest) -> tuple[int, int | None]:
        """``(reservation, hard cap)`` of ``request``, in bytes.

        Raises:
            TypeError, ValueError: on an invalid ``memory_max``, or one larger
                than the whole budget.
        """
        memory = _check_memory_max(request.memory_max)
        if memory is None and self.memory_per_run is not UNBOUNDED:
            memory = self.memory_per_run
        reserve = 0
        if self.memory_bytes is not None and memory is not None:
            if memory > self.memory_bytes:
                raise ValueError(
                    f"Run needs memory_max={memory} bytes but the executor's budget is "
                    f"memory_bytes={self.memory_bytes}"
                )
            reserve = memory
        return reserve, self._caps.cap(memory)

    # -- resource accounting (call with self._cond held) ----------------

    def _try_acquire(self, width: int, reserve: int) -> tuple[int, ...] | None:
        if self._memory_free < reserve:
            return None
        start = _find_block(self._free, width)
        if start is None:
            return None
        for i in range(start, start + width):
            self._free[i] = False
        self._memory_free -= reserve
        return self.cpus[start : start + width]

    def _release(self, block: tuple[int, ...], reserve: int) -> None:
        with self._cond:
            for cpu in block:
                self._free[self.cpus.index(cpu)] = True
            self._memory_free += reserve
            self._cond.notify_all()

    def _acquire(self, width: int, reserve: int) -> tuple[int, ...]:
        with self._cond:
            block = self._try_acquire(width, reserve)
            while block is None:
                self._cond.wait()
                block = self._try_acquire(width, reserve)
            return block

    # -- dispatch -------------------------------------------------------

    def _prepare(self, request: RunRequest) -> _Prepared | RunOutcome:
        """Per-run launch spec, width and memory, or a rejection outcome."""
        try:
            spec = request.launch if request.launch is not None else LaunchSpec()
            # Width as the pinned run will see it: to_env() defaults a pinned
            # run to 1 thread per rank unless a thread count is set anywhere.
            width = dataclasses.replace(spec, cpus=(0,)).width(os.environ)
            if width > len(self.cpus):
                raise ValueError(
                    f"Run needs {width} CPUs but the executor only has {len(self.cpus)} "
                    f"({self.cpus})"
                )
            reserve, cap = self._memory_of(request)
        except Exception as exc:
            return RunOutcome(request=request, result=None, error=exc)
        return _Prepared(spec, width, reserve, cap)

    def _run_on_block(
        self,
        request: RunRequest,
        prep: _Prepared,
        block: tuple[int, ...],
        cancel: threading.Event,
    ) -> RunOutcome:
        try:
            return _execute(request, prep.spec, cancel, prep.cap, cpus=block)
        finally:
            self._release(block, prep.reserve)

    def submit(self, requests: Iterable[RunRequest]) -> list[RunOutcome]:
        """Run ``requests`` concurrently, each on its own CPU block.

        The calling thread dispatches the requests in order (blocking while
        the head of the queue does not fit) and then waits for all of them.
        On ``Ctrl-C`` (any :class:`BaseException` in the calling thread), the
        in-flight runs are cancelled and terminate their processes, then the
        exception is re-raised (see the module docstring).

        Args:
            requests: The runs to perform.

        Returns:
            One :class:`RunOutcome` per request, in input order. Rejected
            requests (wider than the CPU set, invalid launch spec) and
            failed runs carry their exception in ``error``.
        """
        reqs = list(requests)
        cancel = threading.Event()
        pool = ThreadPoolExecutor(
            max_workers=max(1, min(len(reqs), len(self.cpus))),
            thread_name_prefix="mpxlite-run",
        )
        with _termination_signals():
            try:
                outcomes = self._dispatch(pool, reqs, cancel)
            except BaseException:
                cancel.set()
                # Queued tasks still run: they see the event, start nothing and
                # release their CPU block (cancel_futures would leak the blocks).
                pool.shutdown(wait=True)
                raise
        pool.shutdown(wait=True)
        return outcomes

    def _dispatch(
        self, pool: ThreadPoolExecutor, reqs: list[RunRequest], cancel: threading.Event
    ) -> list[RunOutcome]:
        """Dispatch ``reqs`` in FIFO order onto ``pool`` and collect the outcomes."""
        slots: list[RunOutcome | Future[RunOutcome]] = []
        for req in reqs:
            prepared = self._prepare(req)
            if isinstance(prepared, RunOutcome):
                slots.append(prepared)
                continue
            block = self._acquire(prepared.width, prepared.reserve)
            try:
                slots.append(pool.submit(self._run_on_block, req, prepared, block, cancel))
            except BaseException:
                self._release(block, prepared.reserve)
                raise
        return [s if isinstance(s, RunOutcome) else s.result() for s in slots]
