# Copyright (c) 2025 MPhysX
# SPDX-License-Identifier: MIT

r"""Memory safety: default memory budget and hard per-run memory caps.

Two mechanisms keep a batch of runs from exhausting the machine's memory:

* a *budget*: :class:`~mpxlite.executor.LocalExecutor` reserves an estimate
  of each run's peak memory (``memory_per_run``) out of a total
  (``memory_bytes``, default :data:`DEFAULT_MEMORY_FRACTION` of the
  ``MemAvailable`` of ``/proc/meminfo`` measured when the executor is
  built) and starts a run only when its share is free;
* a *hard cap*: each capped run executes inside its own transient systemd
  scope (a cgroup v2 leaf), started as::

      systemd-run --user --scope -q --unit=<unique> \
          -p MemoryMax=<bytes> -p MemorySwapMax=0 -p OOMPolicy=stop <command>

  If the run's processes together exceed ``MemoryMax``, the kernel's
  cgroup OOM killer acts *inside the scope only* and systemd stops the
  whole scope (``OOMPolicy=stop``: every MPI rank and ``mpirun``), so a
  wrong estimate fails that run alone
  (:exc:`~mpxlite.base.GetDPMemoryError`) instead of triggering the
  global OOM killer.

``systemd-run --scope`` registers the scope for its own pid and then
exec-s the command, so the run keeps the pid and session that
:meth:`~mpxlite.base.SolverLite.run` created: the session-based process
cleanup (timeout, cancel) is unchanged. A ``--user --scope`` is always
created under the user manager's ``app.slice``, *beside* (not inside)
the scope or service the Python process runs in: an outer ``MemoryMax``
around the Python process does not bound the runs. The per-run caps do,
which is why an executor's ``memory_per_run`` (reservation) and the
cap are the same number and their sum over concurrent runs never exceeds
the budget.

Availability of the hard cap (``systemd-run`` and ``systemctl`` on
``PATH``, a reachable user manager, a delegated cgroup v2 memory
controller) is probed once per process by :func:`hard_cap_available`
(a trivial capped scope that reads back its own ``memory.max``) and
cached. Inside a SLURM / PBS job (``SLURM_JOB_ID`` / ``PBS_JOBID`` set), or
whenever the Python process does not live under the systemd user manager
(``user@<uid>.service`` in ``/proc/self/cgroup``), the automatic mode does
not cap: a ``--user`` scope would escape the job's (or service's) memory
limit, which then protects the node instead. An explicit
``hard_cap=True`` / ``run(memory_max=...)`` is still honoured, with a
warning.

Every run scope records its owner (pid, start time and pid namespace of
the Python process) in its unit description. :func:`reap_stale_scopes`
(run once per process, before the first capped run) kills the
``mpxlite-*`` scopes whose owner is verifiably dead -- left behind by a
Python process killed with ``SIGKILL`` -- and never touches a scope whose
owner is alive or cannot be identified.
"""

from __future__ import annotations

import contextlib
import enum
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Final

logger = logging.getLogger("mpxlite")

DEFAULT_MEMORY_FRACTION: Final = 0.8
"""Fraction of ``MemAvailable`` used as the default ``LocalExecutor`` budget."""

MEMINFO_PATH: Final = Path("/proc/meminfo")
PROC_SELF_CGROUP: Final = Path("/proc/self/cgroup")
CGROUP_ROOT: Final = Path("/sys/fs/cgroup")
MOUNTS_PATH: Final = Path("/proc/self/mounts")
SCHEDULER_JOB_VARS: Final = ("SLURM_JOB_ID", "PBS_JOBID")

_PROBE_CAP: Final = 64 * 1024 * 1024  # MemoryMax of the probe scope (page multiple)
_PROBE_TIMEOUT_S: Final = 15.0
_SYSTEMCTL_TIMEOUT_S: Final = 15.0
_SETTLE_S: Final = 5.0  # bound on the wait for systemd to finalise a scope's state
_SETTLE_POLL_S: Final = 0.25  # each lap forks a systemctl: at most ~20 per settle
_UNIT_PREFIX: Final = "mpxlite"
_MAX_RUN_ID_IN_UNIT: Final = 100
# ActiveState values after which a scope's Result no longer changes.
_TERMINAL_STATES: Final = frozenset({"inactive", "failed"})
_OOM_RESULT: Final = "oom-kill"
# Owner tag written into a run scope's Description, parsed by the reaper.
_OWNER_RE: Final = re.compile(r"\[owner pid=(\d+) start=(\d+) pidns=(\d+)\]")
# Reads back the memory.max of the cgroup the probe runs in (cgroup v2 line "0::<path>").
_PROBE_SCRIPT: Final = (
    'p=$(sed -n "s/^0:://p" /proc/self/cgroup) && cat "/sys/fs/cgroup$p/memory.max"'
)


class Unbounded(enum.Enum):
    """Type of the :data:`UNBOUNDED` sentinel."""

    UNBOUNDED = "UNBOUNDED"

    def __repr__(self) -> str:
        return "mpxlite.UNBOUNDED"


UNBOUNDED: Final = Unbounded.UNBOUNDED
"""``LocalExecutor(memory_per_run=UNBOUNDED)``: pack runs by CPU count only.

Explicit opt-out of memory accounting: the caller accepts that concurrent
runs may exhaust the machine's memory. A warning is logged when used.
"""


class HardCapUnavailableError(RuntimeError):
    """A hard memory cap was required but cannot be enforced on this machine."""


def mem_available(path: Path = MEMINFO_PATH) -> int | None:
    """``MemAvailable`` of ``/proc/meminfo``, in bytes.

    Args:
        path: The meminfo file to read.

    Returns:
        The value in bytes, or ``None`` if the file or the field is missing
        or malformed.
    """
    try:
        text = path.read_text(encoding="ascii", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key == "MemAvailable":
            fields = rest.split()
            if fields and fields[0].isdigit():
                scale = 1024 if fields[1:2] == ["kB"] else 1
                return int(fields[0]) * scale
            return None
    return None


def _own_cgroup(path: Path | None = None) -> str | None:
    """The cgroup v2 path of this process (``/proc/self/cgroup`` line ``0::<path>``).

    Returns:
        The path (e.g. ``"/user.slice/.../app.slice/x.scope"``), or ``None``
        if the file is unreadable or has no unified (v2) entry.
    """
    try:
        text = (path or PROC_SELF_CGROUP).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("0::"):
            return line[3:].strip() or "/"
    return None


def _read_int(path: Path) -> int | None:
    """Integer content of a cgroup file; ``None`` if missing, unreadable or ``max``."""
    try:
        text = path.read_text(encoding="ascii", errors="replace").strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def cgroup_memory_headroom(cgroup: str | None = None, root: Path | None = None) -> int | None:
    """Memory still usable under the cgroup v2 limits enclosing this process.

    Walks from the process's own cgroup up to the root and returns the
    smallest ``memory.max - memory.current`` over the levels that set a
    limit (``memory.max`` other than ``max``), e.g. the job cgroup of a
    SLURM step or a ``systemd-run --scope -p MemoryMax=...`` around Python.

    Args:
        cgroup: The cgroup path (default: :func:`_own_cgroup`).
        root: Mount point of the cgroup v2 hierarchy (default ``/sys/fs/cgroup``).

    Returns:
        The headroom in bytes (``>= 0``), or ``None`` when no level sets a
        limit or cgroup v2 is unavailable.
    """
    cg = _own_cgroup() if cgroup is None else cgroup
    if cg is None:
        return None
    base = CGROUP_ROOT if root is None else root
    parts = [p for p in cg.split("/") if p]
    best: int | None = None
    for depth in range(len(parts), -1, -1):
        level = base.joinpath(*parts[:depth])
        limit = _read_int(level / "memory.max")
        if limit is None:
            continue
        current = _read_int(level / "memory.current") or 0
        headroom = max(0, limit - current)
        best = headroom if best is None else min(best, headroom)
    return best


def default_memory_budget(fraction: float = DEFAULT_MEMORY_FRACTION) -> int:
    """Default total memory budget: ``fraction`` of the memory available now.

    Uses ``MemAvailable`` (:func:`mem_available`); falls back to the
    physical memory when ``/proc/meminfo`` is unusable (non-Linux). When
    the process runs under a cgroup v2 memory limit (a scheduler job, a
    capped service or scope), the budget is at most the headroom left
    under that limit (:func:`cgroup_memory_headroom`, not scaled by
    ``fraction``).

    Args:
        fraction: Fraction of the available memory, in ``(0, 1]``.

    Returns:
        The budget in bytes (at least 1).
    """
    available = mem_available()
    if available is None:
        available = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    budget = int(available * fraction)
    headroom = cgroup_memory_headroom()
    if headroom is not None:
        budget = min(budget, headroom)
    return max(1, budget)


# -- availability probe ---------------------------------------------------

_probe_lock = threading.Lock()
_probe_state: tuple[bool, str] | None = None  # (available, reason when not)
_warned_unavailable = False


def _cgroup_fs_type(mounts: Path | None = None) -> str | None:
    """File system type mounted at ``/sys/fs/cgroup`` (``None`` if unknown)."""
    try:
        text = (mounts or MOUNTS_PATH).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    found = None
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[1] == str(CGROUP_ROOT):
            found = fields[2]  # the last mount on the point wins
    return found


def _probe() -> tuple[bool, str]:
    """Start a trivial capped scope; check it really runs under ``MemoryMax``."""
    fs_type = _cgroup_fs_type()
    if fs_type is not None and fs_type != "cgroup2":
        return False, (
            f"{CGROUP_ROOT} is {fs_type!r}, not the unified cgroup v2 hierarchy (cgroup v1 or "
            "hybrid mode): boot with systemd.unified_cgroup_hierarchy=1"
        )
    systemd_run = shutil.which("systemd-run")
    if systemd_run is None or shutil.which("systemctl") is None:
        return False, "systemd-run / systemctl not found on PATH"
    unit = f"{_UNIT_PREFIX}-probe-{uuid.uuid4().hex[:12]}"
    cmd = [*_scope_prefix(systemd_run, unit, _PROBE_CAP), "/bin/sh", "-c", _PROBE_SCRIPT]
    try:
        out = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"probe scope could not be started ({exc})"
    if out.returncode != 0:
        detail = (out.stderr or out.stdout).strip().splitlines()
        return False, (
            f"systemd-run --user --scope failed (code {out.returncode}"
            f"{': ' + detail[-1] if detail else ''}; is a systemd user manager reachable?)"
        )
    value = out.stdout.strip()
    if value != str(_PROBE_CAP):
        return False, (
            f"the probe scope's memory.max is {value!r}, not {_PROBE_CAP}: no delegated "
            "cgroup v2 memory controller"
        )
    return True, ""


def hard_cap_available(*, refresh: bool = False) -> bool:
    """Whether hard per-run memory caps can be enforced on this machine.

    The probe runs once per process (thread-safe) and its result is cached.

    Args:
        refresh: Probe again instead of returning the cached result.

    Returns:
        ``True`` if capped scopes work (see the module docstring).
    """
    global _probe_state
    with _probe_lock:
        if _probe_state is None or refresh:
            _probe_state = _probe()
            if not _probe_state[0]:
                logger.debug("hard memory cap unavailable: %s", _probe_state[1])
        return _probe_state[0]


def _unavailable_reason() -> str:
    hard_cap_available()
    return _probe_state[1] if _probe_state is not None else ""


def scheduler_job_reason() -> str | None:
    """Why a ``--user`` scope would escape this process's memory confinement, if it would.

    Returns:
        A reason when ``SLURM_JOB_ID`` / ``PBS_JOBID`` is set, or when the
        process's cgroup v2 path is not under ``user@<uid>.service`` (the
        systemd user manager, where ``--user`` scopes are created); else
        ``None`` (also when the cgroup cannot be read).
    """
    for name in SCHEDULER_JOB_VARS:
        if os.environ.get(name):
            return f"inside a scheduler job ({name}={os.environ[name]})"
    cgroup = _own_cgroup()
    if cgroup is not None and f"/user@{os.getuid()}.service/" not in f"{cgroup}/":
        return (
            f"this process's cgroup {cgroup} is not under the systemd user manager "
            f"(user@{os.getuid()}.service)"
        )
    return None


_warned_escape = False


def _warn_escape(reason: str) -> None:
    global _warned_escape
    with _probe_lock:
        if _warned_escape:
            return
        _warned_escape = True
    logger.warning(
        "hard memory caps requested explicitly while %s: each capped run lives in a "
        "systemd --user scope OUTSIDE this process's cgroup, so it is not bound by the "
        "job's / service's memory limit",
        reason,
    )


def require_hard_cap(context: str = "run") -> None:
    """Raise unless hard per-run memory caps are available.

    Warns once per process when caps are available but would escape the
    process's own confinement (:func:`scheduler_job_reason`).

    Args:
        context: Who requires the cap, for the advice of the message:
            ``"run"`` (``SolverLite.run(memory_max=...)``) or
            ``"executor"`` (``hard_cap=True``).

    Raises:
        HardCapUnavailableError: naming why the probe failed.
    """
    if not hard_cap_available():
        advice = (
            "pass hard_cap=False (under SLURM / PBS the scheduler's memory limit applies)"
            if context == "executor"
            else "call run() without memory_max, or run it through an executor whose "
            "hard_cap is None / False"
        )
        raise HardCapUnavailableError(
            f"hard per-run memory cap required but unavailable: {_unavailable_reason()}. "
            "It needs systemd-run, a systemd user manager and a cgroup v2 memory controller; "
            f"to run uncapped, {advice}."
        )
    reason = scheduler_job_reason()
    if reason is not None:
        _warn_escape(reason)


def resolve_hard_cap(mode: bool | None) -> bool:
    """Whether runs will be hard-capped under the executor option ``mode``.

    Args:
        mode: ``None`` = automatic (cap when available and the process is
            not confined by a scheduler job or a non-user cgroup, see
            :func:`scheduler_job_reason`; else warn once per process and
            run uncapped); ``True`` = required (warns inside a job);
            ``False`` = off.

    Returns:
        Whether caps are applied.

    Raises:
        HardCapUnavailableError: ``mode`` is ``True`` and caps are unavailable.
        TypeError: ``mode`` is not a bool nor ``None``.
    """
    global _warned_unavailable
    if mode is not None and not isinstance(mode, bool):
        raise TypeError(f"hard_cap must be True, False or None, got {mode!r}")
    if mode is False:
        return False
    if mode is True:
        require_hard_cap("executor")
        return True
    reason = scheduler_job_reason()
    if reason is None:
        if hard_cap_available():
            return True
        reason = _probe_state[1] if _probe_state is not None else ""
        why = f"hard per-run memory caps unavailable ({reason})"
    else:
        why = (
            f"hard per-run memory caps disabled: {reason}; a systemd --user scope would "
            "escape its memory limit"
        )
    with _probe_lock:
        if not _warned_unavailable:
            _warned_unavailable = True
            logger.warning(
                "%s: runs are NOT capped, only the memory budget (and any scheduler / cgroup "
                "limit around this process) protects the machine. Pass hard_cap=False to "
                "silence this warning.",
                why,
            )
    return False


# -- capped scope of one run ------------------------------------------------


def _scope_prefix(
    systemd_run: str, unit: str, memory_max: int, description: str | None = None
) -> list[str]:
    return [
        systemd_run,
        "--user",
        "--scope",
        "-q",
        f"--unit={unit}",
        *([f"--description={description}"] if description else []),
        "-p",
        f"MemoryMax={memory_max}",
        "-p",
        "MemorySwapMax=0",
        "-p",
        "OOMPolicy=stop",
    ]


def _systemctl(*args: str) -> subprocess.CompletedProcess[str] | None:
    """Run ``systemctl --user <args>``; ``None`` if it could not be run."""
    systemctl = shutil.which("systemctl") or "systemctl"
    try:
        return subprocess.run(
            [systemctl, "--user", *args],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_SYSTEMCTL_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _parse_show(text: str) -> list[dict[str, str]]:
    """Property blocks of a ``systemctl show`` output (one block per unit)."""
    blocks: list[dict[str, str]] = [{}]
    for line in text.splitlines():
        if not line.strip():
            if blocks[-1]:
                blocks.append({})
            continue
        key, sep, value = line.partition("=")
        if sep:
            blocks[-1][key] = value
    return [b for b in blocks if b]


def _proc_start_time(pid: int | str) -> int | None:
    """Start time (clock ticks since boot, ``/proc/<pid>/stat`` field 22) of ``pid``.

    Raises:
        FileNotFoundError: no such process.
    """
    with open(f"/proc/{pid}/stat", "rb") as f:
        stat = f.read()
    fields = stat[stat.rfind(b")") + 2 :].split()
    return int(fields[19]) if len(fields) > 19 else None


def _pid_namespace() -> int | None:
    try:
        return os.stat("/proc/self/ns/pid").st_ino
    except OSError:
        return None


def _owner_tag() -> str | None:
    """``[owner pid=.. start=.. pidns=..]`` of this process, or ``None`` if unknown."""
    try:
        start = _proc_start_time("self")
    except OSError:
        return None
    ns = _pid_namespace()
    if start is None or ns is None:
        return None
    return f"[owner pid={os.getpid()} start={start} pidns={ns}]"


def _owner_alive(pid: int, start: int) -> bool:
    """Whether the process ``pid`` started at ``start`` still exists (when unsure: True)."""
    try:
        return _proc_start_time(pid) == start
    except FileNotFoundError:
        return False
    except (OSError, ValueError):
        return True


def _cgroup_file(cgroup: str, name: str) -> Path | None:
    return CGROUP_ROOT / cgroup.lstrip("/") / name if cgroup.startswith("/") else None


def _cgroup_oom_kills(cgroup: str) -> int:
    """``oom_kill`` counter of the scope's ``memory.events`` (0 if unreadable).

    Read straight from the kernel, so it is already up to date when the
    run's main process is seen to exit (systemd's ``Result`` lags behind).
    """
    path = _cgroup_file(cgroup, "memory.events")
    if path is None:
        return 0
    try:
        text = path.read_text(encoding="ascii", errors="replace")
    except OSError:
        return 0
    for line in text.splitlines():
        key, _, value = line.partition(" ")
        if key == "oom_kill" and value.strip().isdigit():
            return int(value)
    return 0


def _cgroup_has_procs(cgroup: str) -> bool | None:
    """Whether processes are left in the scope's cgroup (``None``: unknown)."""
    path = _cgroup_file(cgroup, "cgroup.procs")
    if path is None:
        return None
    try:
        return bool(path.read_text(encoding="ascii", errors="replace").strip())
    except FileNotFoundError:
        return False  # cgroup already removed: empty
    except OSError:
        return None


@dataclass(frozen=True)
class MemoryScope:
    """The transient systemd scope that hard-caps one run.

    Attributes:
        unit: Full unit name (``mpxlite-<run_id>-<random>.scope``), unique.
        memory_max: The cap, in bytes (``MemoryMax``; swap is forbidden).
    """

    unit: str
    memory_max: int

    @classmethod
    def for_run(cls, run_id: str, memory_max: int) -> MemoryScope:
        """A new uniquely named scope for the run ``run_id``."""
        name = f"{_UNIT_PREFIX}-{run_id[:_MAX_RUN_ID_IN_UNIT]}-{uuid.uuid4().hex[:12]}"
        return cls(unit=f"{name}.scope", memory_max=memory_max)

    def command_prefix(self) -> list[str]:
        """The ``systemd-run`` arguments to put in front of the run's command.

        The unit description carries the owner tag of this process (see
        :func:`reap_stale_scopes`).
        """
        tag = _owner_tag()
        return _scope_prefix(
            shutil.which("systemd-run") or "systemd-run",
            self.unit,
            self.memory_max,
            f"mpxlite run {tag}" if tag else "mpxlite run",
        )

    def _show(self) -> dict[str, str] | None:
        out = _systemctl(
            "show",
            self.unit,
            "-p",
            "LoadState",
            "-p",
            "ActiveState",
            "-p",
            "Result",
            "-p",
            "ControlGroup",
        )
        if out is None or out.returncode != 0:
            return None
        return dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line)

    def oom_killed(self, settle: bool) -> bool:
        """Whether the scope was stopped by its memory cap; clean up its unit.

        Call once the run's main process has exited. The OOM kill is read
        from the scope's ``memory.events`` (synchronous with the kernel)
        and from its systemd ``Result`` (``oom-kill``). Processes still
        left in the scope (they escaped the run's session) are killed with
        a warning. A failed scope stays loaded in the user manager until
        reset: this resets it, also when the wait is interrupted (e.g. by
        ``Ctrl-C``). Every wait is bounded by a few seconds.

        Args:
            settle: Wait for the final state (use after a failed run; a
                successful, clean run returns after a single query).

        Returns:
            ``True`` if the cgroup OOM killer fired in the scope.
        """
        start = time.monotonic()
        oom = killed = False
        done = False
        try:
            while True:
                props = self._show()
                if props is None or props.get("LoadState") == "not-found":
                    done = True
                    return oom  # gone (collected after a clean end) or unknown
                state = props.get("ActiveState", "")
                cgroup = props.get("ControlGroup", "")
                oom = oom or props.get("Result") == _OOM_RESULT or _cgroup_oom_kills(cgroup) > 0
                if state in _TERMINAL_STATES:
                    if state == "failed":
                        _systemctl("reset-failed", self.unit)
                    done = True
                    return oom
                procs = _cgroup_has_procs(cgroup)  # None: unknown, no leftover check
                if procs and not (oom or killed):
                    logger.warning(
                        "processes left in scope %s after the run ended: killing them", self.unit
                    )
                    _systemctl("kill", "--signal=SIGKILL", self.unit)
                    killed, start = True, time.monotonic()
                    continue
                # Once oom-kill is seen (or leftovers killed), wait for the end
                # so that the reset sticks.
                waited = time.monotonic() - start
                if waited >= (_SETTLE_S if settle or oom or killed else 0.0):
                    done = True
                    return oom
                time.sleep(_SETTLE_POLL_S)
        finally:
            if not done:  # interrupted: never leave a failed unit (nor leftovers) behind
                _systemctl("kill", "--signal=SIGKILL", self.unit)
                _systemctl("reset-failed", self.unit)

    def kill(self) -> None:
        """Best effort: SIGKILL whatever is left in the scope and forget it."""
        _systemctl("kill", "--signal=SIGKILL", self.unit)
        with contextlib.suppress(Exception):
            self.oom_killed(settle=True)


# -- stale scopes of dead owners ----------------------------------------------


def reap_stale_scopes() -> list[str]:
    """Kill the ``mpxlite-*`` run scopes whose owning Python process is dead.

    A Python process killed with ``SIGKILL`` cannot clean up: a run whose
    processes survived it (e.g. MPI ranks started by a custom launcher)
    keeps its scope alive. Each run scope records its owner (pid, start
    time, pid namespace) in its description; a scope is reaped only when
    that owner is verifiably gone: same pid namespace as this process and
    no process with that pid and start time. Scopes without an owner tag
    (other tools, older mpxlite versions), of another pid namespace, or of
    a live owner are never touched.

    Returns:
        The units reaped (killed and reset).
    """
    out = _systemctl("show", "-p", "Id", "-p", "Description", f"{_UNIT_PREFIX}-*.scope")
    my_ns = _pid_namespace()
    if out is None or out.returncode != 0 or my_ns is None:
        return []
    reaped = []
    for props in _parse_show(out.stdout):
        unit = props.get("Id", "")
        match = _OWNER_RE.search(props.get("Description", ""))
        if not unit.startswith(f"{_UNIT_PREFIX}-") or match is None:
            continue
        pid, start, ns = (int(g) for g in match.groups())
        if ns != my_ns or pid == os.getpid() or _owner_alive(pid, start):
            continue
        logger.warning("killing stale run scope %s (its owner, pid %d, is gone)", unit, pid)
        MemoryScope(unit, 0).kill()
        reaped.append(unit)
    return reaped


_reap_lock = threading.Lock()
_reaped_once = False


def reap_stale_scopes_once() -> None:
    """:func:`reap_stale_scopes` once per process; never raises."""
    global _reaped_once
    with _reap_lock:
        if _reaped_once:
            return
        _reaped_once = True
        try:
            reap_stale_scopes()
        except Exception as exc:  # best effort only
            logger.debug("stale scope reaping failed: %s", exc)
