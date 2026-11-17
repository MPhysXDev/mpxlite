# Copyright (c) 2025 MPhysX
# SPDX-License-Identifier: MIT

"""SolverLite: subprocess wrapper around the native getdp binary.

A :class:`SolverLite` instance binds a GetDP project (a triplet of
``<project>.geo.pro`` / ``.physprop.pro`` / ``.cond.pro``) to a mesh and
a named GetDP Resolution, holds a template :class:`WorkflowData` of
runtime scalars, and runs the end-to-end pre-process / solve / optional
post-op pipeline through the :meth:`SolverLite.run` method as a single
``getdp`` invocation (one ``mpirun`` launch when the wrapper script is in
MPI mode).

Every run is isolated in its own run directory: the shared case files and
mesh are only read, never written nor copied, so many runs (possibly
concurrent) can share one read-only input directory.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import logging
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import FrameType, MappingProxyType
from typing import Any

from mpxlite import memory as _memory
from mpxlite.engine import ENGINE_ENV, find_engine
from mpxlite.launch import LaunchSpec
from mpxlite.memory import MemoryScope, require_hard_cap
from mpxlite.result import (
    LOG_NAME,
    MANIFEST_NAME,
    PETSC_OPTIONS_NAME,
    WORKFLOW_NAME,
    SolverResult,
)
from mpxlite.workflow import Scalar, WorkflowData

logger = logging.getLogger("mpxlite")

CASE_EXTENSIONS: tuple[str, ...] = (".geo.pro", ".physprop.pro", ".cond.pro")
"""Extensions of the three case files of a project, in ``Include`` order
(:meth:`SolverLite.case_file`)."""

_RUNS_DIRNAME = "runs"

_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# Characters a path cannot contain to be written in a GetDP ``Include "..."``
# string (GetDP strings have no usable escape for them).
_FORBIDDEN_INCLUDE_CHARS = ('"', "\\", "\n", "\r")

_PETSC_KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
# ANSI SGR sequences: GetDP colours its messages when stdout is a terminal,
# which it is under ``mpirun`` (Open MPI gives the ranks a pty).
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Rank tag GetDP inserts after the message type under MPI: "[rank   1] ".
_RANK_TAG_RE = re.compile(r"\[rank\s+\d+\]\s*")
# GetDP message type of a warning: "Warning : ..." (not "Warnings: ...", "WarningCount").
_WARNING_RE = re.compile(r"Warning\s*:")

PetscOptionValue = str | int | float | bool | None
"""Value of one PETSc option of ``petsc_options`` (see :class:`SolverLite`)."""

_LOG_TAIL_LINES = 40
_LOG_TAIL_BYTES = 64 * 1024
_TERM_GRACE_S = 10.0  # SIGTERM -> SIGKILL grace period of the run's session
_KILL_WAIT_S = 5.0  # bound on the SIGKILL / re-scan phase and on the final reap
_POLL_S = 0.2  # wait-loop period (cancel check, session re-scan)


def _read_log_tail(path: Path | None, n_lines: int = _LOG_TAIL_LINES) -> str:
    """Return the last ``n_lines`` lines of ``path`` (empty if unreadable)."""
    if path is None:
        return ""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - _LOG_TAIL_BYTES))
            data = f.read()
    except OSError:
        return ""
    lines = data.decode("utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])


def _normalize_warning(line: str) -> str | None:
    """The GetDP warning carried by a log ``line``, or ``None`` if it is none.

    ANSI colour sequences and the ``[rank N]`` tag are removed, so that the
    copies printed by every MPI rank collapse into one message.
    """
    text = _ANSI_RE.sub("", line).strip()
    if _WARNING_RE.match(text) is None:
        return None
    head, sep, rest = text.partition(":")
    rest = rest.lstrip()
    tag = _RANK_TAG_RE.match(rest)
    if tag is not None:
        rest = rest[tag.end() :]
    return f"{head}{sep} {rest}".rstrip()


def _read_warnings(path: Path | None) -> tuple[str, ...]:
    """GetDP ``Warning`` lines of the log ``path``, de-duplicated in order.

    Returns:
        The normalised warnings (see :func:`_normalize_warning`); empty if
        the log is missing or unreadable.
    """
    if path is None:
        return ()
    found: dict[str, None] = {}
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                warning = _normalize_warning(line)
                if warning is not None:
                    found.setdefault(warning)
    except OSError:
        return ()
    return tuple(found)


class GetDPError(RuntimeError):
    """Raised when the ``getdp`` subprocess exits with a non-zero return code.

    The message carries the return code, the command line and the last
    lines of the run log. The run directory is kept for diagnosis.

    Attributes:
        returncode: Exit status of the subprocess (``None`` if unknown).
        cmd: The executed command line.
        log_path: The ``getdp.log`` of the failed run, if any.
        run_dir: The run directory of the failed run, if any.
        log_tail: The last lines of the log, as included in the message.
        warnings: The GetDP ``Warning`` lines of the log, de-duplicated (as
            :attr:`SolverResult.warnings`); empty without a log.
    """

    def __init__(
        self,
        returncode: int | None,
        cmd: Sequence[str],
        *,
        log_path: Path | None = None,
        run_dir: Path | None = None,
        reason: str | None = None,
    ) -> None:
        self.returncode = returncode
        self.cmd = list(cmd)
        self.log_path = log_path
        self.run_dir = run_dir
        self.log_tail = _read_log_tail(log_path)
        self.warnings = _read_warnings(log_path)
        headline = reason or f"getdp exited with code {returncode}"
        parts = [headline, f"  cmd: {' '.join(self.cmd)}"]
        if run_dir is not None:
            parts.append(f"  run dir: {run_dir}")
        if log_path is not None:
            parts.append(f"  log: {log_path}")
            tail = self.log_tail or "(empty)"
            parts.append(f"  last {_LOG_TAIL_LINES} log lines:\n{tail}")
        super().__init__("\n".join(parts))


class GetDPTimeoutError(GetDPError):
    """Raised when the ``getdp`` subprocess exceeds the ``run()`` timeout.

    Every process of the run's session (wrapper, ``mpirun``, every rank and
    anything they started) has been terminated (SIGTERM, then SIGKILL after
    a grace period) when this is raised.

    Attributes:
        timeout: The timeout that was exceeded, in seconds.
    """

    def __init__(
        self,
        timeout: float,
        cmd: Sequence[str],
        *,
        returncode: int | None = None,
        log_path: Path | None = None,
        run_dir: Path | None = None,
    ) -> None:
        self.timeout = timeout
        super().__init__(
            returncode,
            cmd,
            log_path=log_path,
            run_dir=run_dir,
            reason=f"getdp timed out after {timeout} s (run processes killed)",
        )


class GetDPCancelledError(GetDPError):
    """Raised when a run is cancelled through the ``cancel`` event of ``run()``.

    If the subprocess had been started, every process of its session has
    been terminated (as for :class:`GetDPTimeoutError`) when this is raised;
    ``log_path`` / ``run_dir`` are then set. A run cancelled before it was
    started has neither (no run directory is created).
    """

    def __init__(
        self,
        cmd: Sequence[str],
        *,
        returncode: int | None = None,
        log_path: Path | None = None,
        run_dir: Path | None = None,
        started: bool = True,
    ) -> None:
        reason = (
            "getdp run cancelled (run processes killed)"
            if started
            else "getdp run cancelled before it was started"
        )
        super().__init__(returncode, cmd, log_path=log_path, run_dir=run_dir, reason=reason)


class GetDPMemoryError(GetDPError):
    """Raised when a run is killed by its hard memory cap (``memory_max``).

    The run's processes together exceeded ``MemoryMax`` of its systemd
    scope: the kernel's cgroup OOM killer fired inside the scope only and
    systemd stopped the whole scope (every MPI rank). Other runs, and the
    rest of the machine, are unaffected. The run directory is kept.

    Attributes:
        memory_max: The cap that was exceeded, in bytes.
        unit: Name of the run's transient systemd scope.
    """

    def __init__(
        self,
        memory_max: int,
        cmd: Sequence[str],
        *,
        unit: str | None = None,
        returncode: int | None = None,
        log_path: Path | None = None,
        run_dir: Path | None = None,
    ) -> None:
        self.memory_max = memory_max
        self.unit = unit
        where = f" in {run_dir}" if run_dir is not None else ""
        super().__init__(
            returncode,
            cmd,
            log_path=log_path,
            run_dir=run_dir,
            reason=(
                f"getdp run{where} killed by its memory cap MemoryMax={memory_max} bytes "
                f"({memory_max / 2**30:.3g} GiB; scope {unit}): raise its memory_per_run / "
                "memory_max, or reduce the problem size"
            ),
        )


def _check_memory_max(memory_max: object) -> int | None:
    """Validate a ``memory_max`` argument (``None`` or a positive int of bytes)."""
    if memory_max is None:
        return None
    if isinstance(memory_max, bool) or not isinstance(memory_max, int):
        raise TypeError(f"memory_max must be an int (bytes) or None, got {memory_max!r}")
    if memory_max <= 0:
        raise ValueError(f"memory_max must be > 0 bytes, got {memory_max}")
    return memory_max


def _check_timeout(timeout: object) -> float | None:
    """Validate a ``timeout`` argument (``None`` or a number of seconds >= 0)."""
    if timeout is None:
        return None
    if isinstance(timeout, bool):
        raise ValueError(f"timeout must be a number of seconds or None, got {timeout!r}")
    if not isinstance(timeout, (int, float)):
        raise TypeError(f"timeout must be a number of seconds or None, got {timeout!r}")
    value = float(timeout)
    if math.isnan(value) or value < 0:
        raise ValueError(f"timeout must be >= 0 seconds (not NaN), got {timeout!r}")
    return value


def _check_solution_file(read_solution: Path | str | None) -> Path | None:
    """The absolute path of the solution file a run reads, or ``None``."""
    if read_solution is None:
        return None
    path = Path(read_solution).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Solution file to read not found: {path}")
    return path


class _CancelRequestedError(Exception):
    """Internal: the ``cancel`` event was set while waiting for the subprocess."""


def _check_include_path(path: Path, what: str) -> None:
    text = str(path)
    if any(c in text for c in _FORBIDDEN_INCLUDE_CHARS):
        raise ValueError(
            f"{what} path {text!r} contains a character that cannot be written in a "
            'GetDP Include string (", \\, newline).'
        )


def _format_petsc_value(key: str, value: PetscOptionValue) -> str | None:
    """Validate one PETSc option value; return its text, ``""`` for a flag, ``None`` to omit."""
    if value is None or value is True:
        return ""
    if value is False:
        return None
    if isinstance(value, int):
        return str(int(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"PETSc option {key!r}: non-finite value {value!r}")
        return repr(float(value))  # shortest round-trip form, e.g. 1e-08
    if not isinstance(value, str):
        raise TypeError(
            f"PETSc option {key!r}: value must be str, int, float, bool or None, "
            f"got {type(value).__name__}"
        )
    if not value or "#" in value or any(c.isspace() for c in value):
        raise ValueError(
            f"PETSc option {key!r}: invalid value {value!r} (must be non-empty, without "
            "whitespace nor '#'; use None for a flag)"
        )
    return value


def _validate_petsc_options(
    options: Mapping[str, PetscOptionValue] | None,
) -> dict[str, PetscOptionValue]:
    """Validated copy of a ``petsc_options`` mapping (``{}`` for ``None``).

    Raises:
        TypeError: on a non-``str`` key or a value of an unsupported type.
        ValueError: on an invalid key or value.
    """
    if options is None:
        return {}
    checked: dict[str, PetscOptionValue] = {}
    for key, value in options.items():
        if not isinstance(key, str):
            raise TypeError(f"PETSc option names must be str, got {type(key).__name__}")
        if key.startswith("-"):
            raise ValueError(f"PETSc option {key!r}: give the name without its leading '-'")
        if not _PETSC_KEY_RE.fullmatch(key):
            raise ValueError(
                f"Invalid PETSc option name {key!r}: must match ^{_PETSC_KEY_RE.pattern}$"
            )
        _format_petsc_value(key, value)
        checked[key] = value
    return checked


def _petsc_option_lines(options: Mapping[str, PetscOptionValue]) -> list[str]:
    """``-key [value]`` lines of validated ``options``, sorted by key (``False`` omitted)."""
    lines = []
    for key in sorted(options):
        text = _format_petsc_value(key, options[key])
        if text is not None:
            lines.append(f"-{key} {text}" if text else f"-{key}")
    return lines


def _normalize_postops(postop: str | Sequence[str] | None) -> list[str]:
    if postop is None:
        return []
    names = [postop] if isinstance(postop, str) else list(postop)
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"PostOperation names must be str, got {type(name).__name__}")
        if not name or name.startswith("-") or any(c.isspace() for c in name):
            raise ValueError(f"Invalid PostOperation name {name!r}")
    return names


class SolverLite:
    """Generic getdp subprocess wrapper for the mpxlite platform.

    Concrete physics solver subclasses pass their resolution_id and a
    list of bundled .pro resource files (typically resolved via
    :mod:`importlib.resources`).

    The instance is a reusable, thread-safe *template*: :meth:`run` never
    mutates it. Each call creates a fresh run directory
    ``runs_dir / run_id``, writes ``workflow.pro`` (Function block of
    scalars) and the top-level manifest ``mpxlite.pro`` into it, and
    invokes the native ``getdp`` binary once with the stage flags
    (``-pre``, ``-cal`` and optionally ``-pos``) chained in a single CLI
    invocation, with the run directory as working directory. The manifest
    ``Include``-s the case files and resources by absolute path, so the
    project directory is only read: it may be shared by concurrent runs
    and may be read-only when ``runs_dir`` points elsewhere.

    Path semantics inside the ``.pro`` files (verified with GetDP 3.5):

    * a relative ``Include "x.pro";`` is resolved against the directory of
      the file that contains it, so case files may keep including their
      siblings relatively;
    * relative output files (``File "x.dat"`` in a ``PostOperation``) and
      the ``.pre`` / ``.res`` files are written next to the main ``.pro``
      file, i.e. into the run directory. A case that writes to an absolute
      path or to ``../...`` escapes the run isolation; don't.

    Args:
        project_dir: Directory holding the case files and (optionally) the mesh.
        project_name: Base name of the case (e.g. ``"demo"``); the
            files ``<project_name>.geo.pro``, ``.physprop.pro`` and
            ``.cond.pro`` must exist in ``project_dir``.
        mesh: Path to the Gmsh ``.msh`` mesh (format ≤ 2.2). Either absolute
            or resolved relative to ``project_dir``.
        resolution_id: Name of the GetDP ``Resolution`` to invoke at the
            preprocess stage.
        resource_files: Ordered list of additional .pro files to ``Include``
            in the manifest, after ``<project>.geo.pro`` and before
            ``<project>.physprop.pro``. Typically the FE machinery and the
            solver formulation.
        workflow: The template :class:`WorkflowData`; a fresh empty one is
            created if not supplied. Per-run overrides go through
            ``run(workflow=...)``.
        verbosity: GetDP verbosity level passed via ``-v`` (0 silent .. 5).
        runs_dir: Parent directory of the run directories. Defaults to
            ``project_dir / "runs"``; created lazily on the first run. A
            relative path is resolved against the current directory.
        petsc_options: Template PETSc options of every run, e.g.
            ``{"pc_factor_mat_solver_type": "mumps"}``. Keys are option
            names without the leading ``-`` (``^[A-Za-z][A-Za-z0-9_]*$``);
            a value ``None`` or ``True`` gives a flag without value,
            ``False`` omits the option, numbers are written with
            ``str`` / ``repr``, strings must be non-empty and contain no
            whitespace nor ``#``. Per-run options of ``run(petsc_options=...)``
            are merged over them. Stored as the read-only
            :attr:`petsc_options`.

    Raises:
        FileNotFoundError: on a missing project directory, case file, mesh
            or resource file.
        ValueError: on an unsupported mesh, a path that cannot be written in
            an ``Include`` string or an invalid PETSc option.
        TypeError: on a PETSc option of an unsupported type.
    """

    def __init__(
        self,
        project_dir: Path | str,
        project_name: str,
        mesh: Path | str,
        resolution_id: str,
        resource_files: Sequence[Path | str] | None = None,
        workflow: WorkflowData | None = None,
        verbosity: int = 3,
        runs_dir: Path | str | None = None,
        petsc_options: Mapping[str, PetscOptionValue] | None = None,
    ) -> None:
        self._petsc_options = MappingProxyType(_validate_petsc_options(petsc_options))
        self.project_dir = Path(project_dir).expanduser().resolve()
        if not self.project_dir.is_dir():
            raise FileNotFoundError(f"Project directory does not exist: {self.project_dir}")
        self.project_name = project_name
        self.resolution_id = resolution_id
        self.resource_files: list[Path] = [
            Path(p).expanduser().resolve() for p in (resource_files or [])
        ]
        self.workflow = workflow if workflow is not None else WorkflowData()
        self.verbosity = int(verbosity)
        self.runs_dir = (
            Path(runs_dir).expanduser().resolve()
            if runs_dir is not None
            else self.project_dir / _RUNS_DIRNAME
        )

        self.mesh = self._resolve_mesh(mesh)
        self._verify_case_files()
        self._verify_resource_files()
        _check_include_path(self.runs_dir, "runs_dir")

    @property
    def petsc_options(self) -> Mapping[str, PetscOptionValue]:
        """Template PETSc options (read-only view; see the class docstring)."""
        return self._petsc_options

    # -- Validation helpers --------------------------------------------

    def _resolve_mesh(self, mesh: Path | str) -> Path:
        mesh_path = Path(mesh).expanduser()
        if not mesh_path.is_absolute():
            mesh_path = self.project_dir / mesh_path
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        self._validate_msh_version(mesh_path)
        return mesh_path.resolve()

    @staticmethod
    def _validate_msh_version(path: Path) -> None:
        """Confirm the mesh is in Gmsh MSH ≤ 2.2 ASCII format (GetDP requirement)."""
        with path.open(encoding="utf-8", errors="replace") as f:
            first = f.readline()
            if first.startswith("$NOD"):
                # Legacy Gmsh format 1.0 — accepted by getdp.
                return
            if not first.startswith("$MeshFormat"):
                raise ValueError(f"Unrecognized mesh format in {path}: first line {first!r}")
            second = f.readline().strip()
        try:
            version = float(second.split()[0])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Cannot parse MSH version from {path}: {second!r}") from exc
        if version > 2.2:
            raise ValueError(
                f"Mesh format version {version} unsupported by GetDP (must be ≤ 2.2): {path}"
            )

    def case_file(self, ext: str) -> Path:
        """Path of a case file of the project.

        Args:
            ext: One of :data:`CASE_EXTENSIONS` (``".geo.pro"``,
                ``".physprop.pro"``, ``".cond.pro"``).

        Returns:
            ``project_dir / (project_name + ext)``.

        Raises:
            ValueError: ``ext`` is not one of :data:`CASE_EXTENSIONS`.
        """
        if ext not in CASE_EXTENSIONS:
            raise ValueError(f"case file extension {ext!r} is not one of {CASE_EXTENSIONS}")
        return self.project_dir / f"{self.project_name}{ext}"

    def _verify_case_files(self) -> None:
        for ext in CASE_EXTENSIONS:
            target = self.case_file(ext)
            if not target.is_file():
                raise FileNotFoundError(f"Required case file missing: {target}")
            _check_include_path(target, "Case file")

    def _verify_resource_files(self) -> None:
        for path in self.resource_files:
            if not path.is_file():
                raise FileNotFoundError(f"Resource .pro file missing: {path}")
            _check_include_path(path, "Resource")

    # -- Per-run preparation -------------------------------------------

    def effective_workflow(self, overrides: Mapping[str, Scalar] | None = None) -> WorkflowData:
        """The scalars a run with ``workflow=overrides`` writes and uses.

        A copy of the template :attr:`workflow` updated with the validated
        ``overrides``. :meth:`run` and :meth:`preprocess` call it once per
        run; a subclass may override it, for instance to derive scalars from
        others or to refuse an inconsistent set, and what it returns is
        what the run uses.

        Args:
            overrides: Per-run scalars, as the ``workflow`` argument of
                :meth:`run`.

        Returns:
            A new :class:`WorkflowData`; the template is not modified.

        Raises:
            ValueError: on an invalid scalar name or value.
            TypeError: on a scalar of an unsupported type.
        """
        scalars = WorkflowData(dict(self.workflow))
        if overrides:
            for key, value in overrides.items():
                scalars[key] = value
        return scalars

    @staticmethod
    def _new_run_id(run_id: str | None) -> str:
        if run_id is None:
            return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
            raise ValueError(f"Invalid run_id {run_id!r}: must match ^[A-Za-z0-9][A-Za-z0-9._-]*$")
        return run_id

    def effective_petsc_options(
        self, overrides: Mapping[str, PetscOptionValue] | None = None
    ) -> dict[str, PetscOptionValue]:
        """The PETSc options a run with ``petsc_options=overrides`` applies.

        The template :attr:`petsc_options` updated with the validated
        ``overrides`` (a value ``False`` is kept: it removes the option from
        the file written). :meth:`run` and :meth:`preprocess` call it once
        per run; a subclass may override it, for instance to add options
        computed from the case, and what it returns is what the run writes.

        Args:
            overrides: Per-run options, as the ``petsc_options`` argument of
                :meth:`run`.

        Returns:
            A new mapping; the template is not modified.

        Raises:
            ValueError: on an invalid option name or value.
            TypeError: on an option of an unsupported type.
        """
        return {**self._petsc_options, **_validate_petsc_options(overrides)}

    @staticmethod
    def _write_petsc_opts(run_dir: Path, options: Mapping[str, PetscOptionValue]) -> Path | None:
        """Write ``petsc.opts`` into ``run_dir``; ``None`` (no file) if no option is set."""
        lines = _petsc_option_lines(options)
        if not lines:
            return None
        path = run_dir / PETSC_OPTIONS_NAME
        header = "# petsc.opts — auto-generated by mpxlite (PETSc options file). Do not edit."
        path.write_text("\n".join([header, *lines]) + "\n", encoding="utf-8")
        return path

    @staticmethod
    def _write_workflow_pro(run_dir: Path, scalars: WorkflowData) -> Path:
        path = run_dir / WORKFLOW_NAME
        scalars.write_pro(path)
        return path

    def _write_manifest(self, run_dir: Path) -> Path:
        """Write the top-level ``mpxlite.pro`` ``Include`` chain into ``run_dir``.

        Every ``Include`` uses an absolute path. Order:
            1. ``workflow.pro``                (Function: scalars, in ``run_dir``)
            2. ``<project>.geo.pro``           (Group: regions)
            3. ``resource_files[*]``           (FE machinery and formulation)
            4. ``<project>.physprop.pro``      (Function: ε, μ, σ, sources)
            5. ``<project>.cond.pro``          (Constraint)

        Args:
            run_dir: The run directory (absolute).

        Returns:
            The absolute path of the manifest file just written.
        """
        includes = [
            run_dir / WORKFLOW_NAME,
            self.case_file(".geo.pro"),
            *self.resource_files,
            self.case_file(".physprop.pro"),
            self.case_file(".cond.pro"),
        ]
        lines = ["// mpxlite.pro — auto-generated. Do not edit by hand."]
        lines += [f'Include "{path}";' for path in includes]
        manifest = run_dir / MANIFEST_NAME
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return manifest

    # -- Public stages -------------------------------------------------

    def run(
        self,
        postop: str | Sequence[str] | None = None,
        *,
        workflow: Mapping[str, Scalar] | None = None,
        run_id: str | None = None,
        launch: LaunchSpec | None = None,
        timeout: float | None = None,
        cancel: threading.Event | None = None,
        petsc_options: Mapping[str, PetscOptionValue] | None = None,
        memory_max: int | None = None,
        read_solution: Path | str | None = None,
    ) -> SolverResult:
        """End-to-end getdp invocation: pre-process, solve, optional post-ops.

        Creates the run directory ``runs_dir / run_id`` (never reused),
        writes ``workflow.pro``, ``mpxlite.pro`` and, when PETSc options are
        set, ``petsc.opts`` into it, then runs::

            <driver> <run_dir>/mpxlite.pro -msh <mesh> -v N
                [-options_file <run_dir>/petsc.opts] [-res <file>]
                -pre <res> -cal [-pos <po> ...]

        as a single subprocess with the run directory as working directory,
        its merged stdout/stderr streamed to ``<run_dir>/getdp.log``.
        Pre-process, solve and post-ops share the in-memory PETSc state, so
        output integrals on internal Fourier surfaces are evaluated
        correctly. The instance is not modified: concurrent calls from
        several threads are safe as long as their ``run_id`` differ.

        Args:
            postop: Name of the ``PostOperation`` to evaluate after the
                solve, or a sequence of names (all evaluated, in order, by
                one ``-pos a b c``). ``None`` or an empty sequence runs
                ``-pre`` + ``-cal`` only.
            workflow: Per-run scalar overrides, merged over a copy of
                :attr:`workflow` and validated by :class:`WorkflowData`.
            run_id: Name of the run directory. Must match
                ``^[A-Za-z0-9][A-Za-z0-9._-]*$``. Default: a timestamp plus
                a random suffix.
            launch: Parallel layout of this run, passed to the subprocess
                through its environment. Default: ``LaunchSpec()``, i.e.
                the process-wide ``MPXLITE_GETDP_*`` defaults.
            timeout: Wall-clock limit in seconds (a number >= 0). On expiry
                every process of the run's session is terminated and
                :class:`GetDPTimeoutError` is raised. ``None`` waits forever.
            cancel: Optional cancellation flag, polled every 0.2 s while the
                subprocess runs (and checked once before anything is
                created). When it is set, the run's processes are
                terminated like on a timeout and :class:`GetDPCancelledError`
                is raised. Executors use it to stop in-flight runs on
                ``Ctrl-C``; any thread may set it.
            petsc_options: Per-run PETSc options, merged over
                :attr:`petsc_options` (per-run values win; ``False`` removes
                a template option). Same rules as the constructor argument.
                When the effective set is not empty it is written to
                ``<run_dir>/petsc.opts`` and passed with ``-options_file``.
            memory_max: Hard memory cap of the run, in bytes, or ``None``
                (default: no cap). The run executes in its own transient
                systemd scope with ``MemoryMax=memory_max`` and no swap
                (see :mod:`mpxlite.memory`); if its processes together
                exceed the cap, they are all killed and
                :class:`GetDPMemoryError` is raised. The cap is mandatory
                once asked for: :exc:`~mpxlite.memory.HardCapUnavailableError`
                is raised (before anything is created) if it cannot be
                enforced on this machine.
            read_solution: A solution file GetDP reads before the stages
                (``-res <file>``), for a resolution that starts from, or
                post-processes, the solution of another run, typically the
                :attr:`SolverResult.solution_path` of that run. A relative
                path is resolved against the current directory. ``None``:
                nothing is read.

        Returns:
            :class:`SolverResult` describing the run directory and log. The
            GetDP warnings of the log are in its ``warnings`` and are each
            logged once on the ``"mpxlite"`` logger.

        Raises:
            ValueError: on an invalid ``run_id``, ``postop``, scalar, PETSc
                option or ``timeout`` (negative, NaN or a bool).
            TypeError: on a PETSc option or a ``timeout`` of an unsupported
                type.
            FileExistsError: if ``runs_dir / run_id`` already exists.
            FileNotFoundError: if there is no engine or no driver, or if
                ``read_solution`` is not a file.
            GetDPError: if getdp exits with a non-zero code.
            GetDPTimeoutError: if ``timeout`` is exceeded.
            GetDPCancelledError: if ``cancel`` is set.
            GetDPMemoryError: if the run exceeds ``memory_max``.
            HardCapUnavailableError: if ``memory_max`` is set but cannot be
                enforced.
        """
        postops = _normalize_postops(postop)
        stages = ["-pre", self.resolution_id, "-cal", *(["-pos", *postops] if postops else [])]
        return self._execute(
            stages,
            workflow=workflow,
            run_id=run_id,
            launch=launch,
            timeout=timeout,
            cancel=cancel,
            petsc_options=petsc_options,
            memory_max=memory_max,
            read_solution=read_solution,
        )

    def preprocess(
        self,
        *,
        workflow: Mapping[str, Scalar] | None = None,
        run_id: str | None = None,
        launch: LaunchSpec | None = None,
        timeout: float | None = None,
        cancel: threading.Event | None = None,
        petsc_options: Mapping[str, PetscOptionValue] | None = None,
        memory_max: int | None = None,
        read_solution: Path | str | None = None,
    ) -> SolverResult:
        """Pre-process only: parse the case and run GetDP ``-pre`` (no solve).

        Same run directory isolation, inputs and error handling as
        :meth:`run`, but the command stops after the pre-processing stage::

            <driver> <run_dir>/mpxlite.pro -msh <mesh> -v N
                [-options_file <run_dir>/petsc.opts] -pre <res>

        GetDP then parses the whole manifest (every ``Include``-d case and
        resource file), reads the mesh and builds the degree-of-freedom
        data of the resolution (constraints, function spaces, links,
        global quantities) into ``<run_dir>/mpxlite.pre``, without
        assembling nor solving anything. It is the cheap way to check a
        case: what GetDP rejects while parsing and pre-processing makes it
        fail (:exc:`GetDPError`, with the log tail), what it only reports
        shows up in ``SolverResult.warnings``. Errors that only occur when
        evaluating expressions during assembly are not detected.

        Args:
            workflow: Per-run scalar overrides, as for :meth:`run`.
            run_id: Name of the run directory, as for :meth:`run`.
            launch: Parallel layout, as for :meth:`run`. The default
                ``LaunchSpec()`` follows ``MPXLITE_GETDP_NP`` (one process
                when it is unset), which is what a check usually wants;
                pass ``LaunchSpec(np=1)`` to force a single process.
            timeout: Wall-clock limit in seconds, as for :meth:`run`.
            cancel: Optional cancellation flag, as for :meth:`run`.
            petsc_options: Per-run PETSc options, as for :meth:`run`
                (written to ``petsc.opts``; pre-processing reads only the
                options it uses, e.g. none of the solver ones).
            memory_max: Hard memory cap in bytes, as for :meth:`run`.
            read_solution: A solution file to read, as for :meth:`run`.

        Returns:
            :class:`SolverResult` of the run directory (``mpxlite.pre``,
            ``getdp.log``, ``workflow.pro``, ``mpxlite.pro``); the GetDP
            warnings are in its ``warnings`` and are each logged once on the
            ``"mpxlite"`` logger.

        Raises:
            Same exceptions as :meth:`run` (except those about ``postop``).
        """
        return self._execute(
            ["-pre", self.resolution_id],
            workflow=workflow,
            run_id=run_id,
            launch=launch,
            timeout=timeout,
            cancel=cancel,
            petsc_options=petsc_options,
            memory_max=memory_max,
            read_solution=read_solution,
        )

    def _execute(
        self,
        stages: Sequence[str],
        *,
        workflow: Mapping[str, Scalar] | None,
        run_id: str | None,
        launch: LaunchSpec | None,
        timeout: float | None,
        cancel: threading.Event | None,
        petsc_options: Mapping[str, PetscOptionValue] | None,
        memory_max: int | None,
        read_solution: Path | str | None,
    ) -> SolverResult:
        """Common body of :meth:`run` and :meth:`preprocess`.

        Validates the per-run arguments, creates the run directory, writes
        its inputs, runs ``<driver> <manifest> -msh <mesh> -v N
        [-options_file ...] [-res <file>] <stages...>`` and collects the
        warnings.
        """
        solution = _check_solution_file(read_solution)
        cap = _check_memory_max(memory_max)
        timeout = _check_timeout(timeout)
        scalars = self.effective_workflow(workflow)
        petsc = self.effective_petsc_options(petsc_options)
        rid = self._new_run_id(run_id)
        spec = launch if launch is not None else LaunchSpec()
        env = spec.to_env(os.environ)
        # One engine per run, decided here and handed to the driver: the
        # driver never looks for a getdp by itself.
        env[ENGINE_ENV] = str(find_engine(env))
        driver = self._resolve_getdp(env)
        if cancel is not None and cancel.is_set():
            raise GetDPCancelledError([driver], started=False)
        if cap is not None:
            require_hard_cap("run")
            _memory.reap_stale_scopes_once()

        run_dir = self.runs_dir / rid
        run_dir.mkdir(parents=True, exist_ok=False)
        self._write_workflow_pro(run_dir, scalars)
        manifest = self._write_manifest(run_dir)
        opts_file = self._write_petsc_opts(run_dir, petsc)

        # -options_file goes before the stage flags: GetDP ignores the PETSc
        # options it does not know, and the -pos name list stays the tail.
        cmd = [
            driver,
            str(manifest),
            "-msh",
            str(self.mesh),
            "-v",
            str(self.verbosity),
            *(["-options_file", str(opts_file)] if opts_file is not None else []),
            *(["-res", str(solution)] if solution is not None else []),
            *stages,
        ]
        scope = MemoryScope.for_run(rid, cap) if cap is not None else None
        returncode, elapsed = self._run_getdp(cmd, run_dir, env, timeout, cancel, scope)
        warnings = _read_warnings(run_dir / LOG_NAME)
        for warning in warnings:
            logger.warning("getdp run %s: %s", rid, warning)
        return SolverResult(
            work_dir=run_dir,
            run_id=rid,
            log_path=run_dir / LOG_NAME,
            elapsed_s=elapsed,
            returncode=returncode,
            cmd=tuple(cmd),
            warnings=warnings,
            memory_max=cap,
        )

    # -- subprocess plumbing -------------------------------------------

    @staticmethod
    def _run_getdp(
        cmd: Sequence[str],
        run_dir: Path,
        env: Mapping[str, str],
        timeout: float | None,
        cancel: threading.Event | None = None,
        scope: MemoryScope | None = None,
    ) -> tuple[int, float]:
        """Run ``cmd`` in its own session, output streamed to the run log.

        The subprocess gets ``/dev/null`` as stdin (``mpirun`` would
        otherwise forward and consume the parent's stdin). Whatever ends the
        wait abnormally (timeout, cancellation, ``KeyboardInterrupt``...),
        every process of the session is terminated before returning.

        With a ``scope``, ``cmd`` is started through ``systemd-run --scope``
        (which exec-s it: same pid, same session) and the scope's result
        tells whether the cgroup OOM killer ended the run.

        When the main process exits normally, whatever it left behind in
        its session (or in its scope) is terminated with a warning. On
        Linux the subprocess also gets ``PR_SET_PDEATHSIG=SIGKILL`` (it is
        killed if the calling thread dies), and while it runs in the main
        thread ``SIGTERM`` / ``SIGHUP`` clean up like ``Ctrl-C`` (see
        :func:`_termination_signals`).

        Returns:
            ``(returncode, elapsed_seconds)`` of a successful run.

        Raises:
            GetDPError: non-zero exit.
            GetDPTimeoutError: ``timeout`` exceeded.
            GetDPCancelledError: ``cancel`` set.
            GetDPMemoryError: the run was killed by the memory cap of ``scope``.
        """
        log_path = run_dir / LOG_NAME
        argv = [*(scope.command_prefix() if scope is not None else []), *cmd]
        logger.info("running %s (cwd=%s)", " ".join(argv), run_dir)
        with _termination_signals(), log_path.open("wb") as log:
            start = time.perf_counter()
            proc: subprocess.Popen[bytes] | None = None
            try:
                # Inside the try: a signal right after the start still terminates the run.
                proc = subprocess.Popen(
                    argv,
                    cwd=str(run_dir),
                    env=dict(env),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    preexec_fn=_parent_death_preexec(),
                )
                _register_live_run(proc, scope)
                returncode = _wait(proc, timeout, cancel)
                elapsed = time.perf_counter() - start
                _cleanup_after_exit(proc, run_dir)
                oom = scope is not None and scope.oom_killed(settle=returncode != 0)
            except subprocess.TimeoutExpired:  # from _wait: proc is set
                _terminate_run(proc, scope)  # type: ignore[arg-type]
                raise GetDPTimeoutError(
                    timeout if timeout is not None else 0.0,
                    cmd,
                    returncode=proc.returncode,  # type: ignore[union-attr]
                    log_path=log_path,
                    run_dir=run_dir,
                ) from None
            except _CancelRequestedError:  # from _wait: proc is set
                _terminate_run(proc, scope)  # type: ignore[arg-type]
                raise GetDPCancelledError(
                    cmd,
                    returncode=proc.returncode,  # type: ignore[union-attr]
                    log_path=log_path,
                    run_dir=run_dir,
                ) from None
            except BaseException:
                # KeyboardInterrupt, SIGTERM / SIGHUP & co.: never leave orphan ranks behind.
                if proc is not None:
                    _terminate_run(proc, scope)
                raise
            finally:
                if proc is not None:
                    _unregister_live_run(proc)
        if oom and scope is not None:
            raise GetDPMemoryError(
                scope.memory_max,
                cmd,
                unit=scope.unit,
                returncode=returncode,
                log_path=log_path,
                run_dir=run_dir,
            )
        if returncode != 0:
            raise GetDPError(returncode, cmd, log_path=log_path, run_dir=run_dir)
        return returncode, elapsed

    @staticmethod
    def _resolve_getdp(env: Mapping[str, str] | None = None) -> str:
        """Locate the driver, the executable that launches the engine.

        1. ``MPXLITE_GETDP``, when it is set: a custom launcher. It must name
           an executable file; a relative path is resolved against the
           current directory, not the run directory.
        2. Otherwise the ``tools/getdp_runner.sh`` wrapper of this package,
           which starts the MPI launcher (mpirun / srun / mpiexec) the
           surrounding environment calls for, pins the CPUs and sets the
           threads.

        Either one executes the engine named by ``MPXLITE_GETDP_BINARY``,
        which :meth:`_execute` sets in the environment of the run to the
        answer of :func:`mpxlite.find_engine`. There is no other driver: a
        ``getdp`` on ``PATH`` is never used.

        Args:
            env: The run's effective environment, read for ``MPXLITE_GETDP``
                (default: :data:`os.environ`).

        Returns:
            The absolute path of the driver.

        Raises:
            FileNotFoundError: ``MPXLITE_GETDP`` does not name an executable
                file, or the wrapper of the package is missing or not
                executable.
        """
        environ = os.environ if env is None else env
        override = environ.get("MPXLITE_GETDP")
        if override:
            path = Path(override).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                return str(path.resolve())  # the run is exec-ed with cwd = its run dir
            raise FileNotFoundError(f"MPXLITE_GETDP={override!r} is not an executable file.")
        from mpxlite import WRAPPER_PATH

        if WRAPPER_PATH.is_file() and os.access(WRAPPER_PATH, os.X_OK):
            return str(WRAPPER_PATH)
        raise FileNotFoundError(
            f"The getdp_runner.sh wrapper of mpxlite ({WRAPPER_PATH}) is missing or not "
            "executable: reinstall mpxlite, or set MPXLITE_GETDP to a custom launcher."
        )


def _wait(
    proc: subprocess.Popen[bytes], timeout: float | None, cancel: threading.Event | None
) -> int:
    """Wait for ``proc``, polling ``cancel`` every :data:`_POLL_S` seconds.

    Raises:
        subprocess.TimeoutExpired: ``timeout`` exceeded.
        _CancelRequestedError: ``cancel`` set.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        if cancel is not None and cancel.is_set():
            raise _CancelRequestedError
        step = _POLL_S
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(proc.args, timeout or 0.0)
            step = min(step, remaining)
        with contextlib.suppress(subprocess.TimeoutExpired):
            return proc.wait(timeout=step)


_PROC = "/proc"
_SESSION_GONE = "_mpxlite_session_gone"  # attribute set on a Popen whose session is over


def _session_pids(sid: int) -> list[int] | None:
    """Live (non-zombie) processes whose session id is ``sid``, from ``/proc``.

    Only meaningful for a session known to be ours: see :func:`_session_members`
    for the pid-reuse rules once its leader has been reaped.

    Returns:
        The pids, or ``None`` when ``/proc`` is unavailable (non-Linux).
    """
    try:
        entries = os.listdir(_PROC)
    except OSError:
        return None
    pids = []
    for name in entries:
        if not name.isdigit():
            continue
        try:
            with open(f"{_PROC}/{name}/stat", "rb") as f:
                stat = f.read()
        except OSError:
            continue  # exited meanwhile
        # "pid (comm) state ppid pgrp session ...": comm may hold spaces or ')'.
        fields = stat[stat.rfind(b")") + 2 :].split()
        if len(fields) > 3 and fields[0] not in (b"Z", b"X") and int(fields[3]) == sid:
            pids.append(int(name))
    return pids


def _pid_in_use(pid: int) -> bool:
    """Whether ``/proc/<pid>`` exists (a process or thread holds the number ``pid``)."""
    return os.path.exists(f"{_PROC}/{pid}")


def _session_members(proc: subprocess.Popen[bytes]) -> list[int] | None:
    """Live processes of the session led by ``proc`` (``start_new_session``).

    While the leader is alive, or reaped but other members remain, Linux
    keeps the number ``proc.pid`` allocated (a session id holds it), so every
    process whose session id is ``proc.pid`` is one the run started (directly
    or not) -- including MPI ranks that ``mpirun`` moved to their own
    process group. Once the session has emptied after the leader was reaped,
    the number is free: a new, unrelated process may get it and ``setsid()``.
    Hence, after the reap, the session is over -- and never scanned again --
    as soon as one scan finds it empty or ``/proc/<proc.pid>`` exists again
    (the leader cannot be both reaped and present: the pid was recycled).

    Returns:
        The pids, or ``None`` when ``/proc`` is unavailable (non-Linux).
    """
    reaped = proc.returncode is not None
    if reaped and getattr(proc, _SESSION_GONE, False):
        return []
    pids = _session_pids(proc.pid)
    if pids is None:
        return None
    if reaped and (not pids or _pid_in_use(proc.pid)):
        setattr(proc, _SESSION_GONE, True)
        return []
    return pids


def _signal_session(proc: subprocess.Popen[bytes], sig: int) -> bool:
    """Send ``sig`` to every live process of the session led by ``proc``.

    Falls back to the process group of ``proc`` when ``/proc`` is unavailable.

    Returns:
        Whether some process of the session may still be alive.
    """
    proc.poll()  # reap the leader if it already exited
    pids = _session_members(proc)
    if pids is None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, sig)
        return proc.poll() is None
    for pid in pids:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, sig)
    return bool(pids)


def _session_alive(proc: subprocess.Popen[bytes]) -> bool:
    """Whether the leader or any other process of its session is still alive."""
    if proc.poll() is None:
        return True
    return bool(_session_members(proc))


def _wait_session(proc: subprocess.Popen[bytes], seconds: float) -> bool:
    """Poll until the session is empty or ``seconds`` elapsed; return whether it emptied."""
    deadline = time.monotonic() + seconds
    while _session_alive(proc):
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(_POLL_S / 2, max(0.0, deadline - time.monotonic())))
    return True


def _terminate_run(proc: subprocess.Popen[bytes], scope: MemoryScope | None) -> None:
    """Terminate the run's session, then whatever is left in its scope (if capped)."""
    with _termination_signals_deferred():
        try:
            _terminate_session(proc)
        finally:
            if scope is not None:
                scope.kill()


def _terminate_session(proc: subprocess.Popen[bytes]) -> None:
    """Terminate every process of the session led by ``proc`` (``start_new_session``).

    SIGTERM to all of them, wait up to :data:`_TERM_GRACE_S` for the session
    to empty, then SIGKILL the survivors and re-scan until none is left (at
    most :data:`_KILL_WAIT_S`). Every wait is bounded. A ``KeyboardInterrupt``
    during the grace period (second ``Ctrl-C``) skips straight to SIGKILL;
    it is re-raised once the cleanup is done. A ``SIGTERM`` / ``SIGHUP``
    arriving meanwhile is deferred until the cleanup is done (see
    :func:`_termination_signals_deferred`).
    """
    with _termination_signals_deferred():
        _terminate_session_now(proc)


def _terminate_session_now(proc: subprocess.Popen[bytes]) -> None:
    """Body of :func:`_terminate_session` (termination signals already deferred)."""
    interrupt: KeyboardInterrupt | None = None
    try:
        if _signal_session(proc, signal.SIGTERM):
            _wait_session(proc, _TERM_GRACE_S)
    except KeyboardInterrupt as exc:
        interrupt = exc
    deadline = time.monotonic() + _KILL_WAIT_S
    while True:
        try:
            if not _signal_session(proc, signal.SIGKILL) or _wait_session(proc, _POLL_S):
                break
        except KeyboardInterrupt as exc:  # keep killing: the phase is bounded
            interrupt = interrupt or exc
        if time.monotonic() >= deadline:
            logger.warning(
                "processes of session %d survived SIGKILL: %s", proc.pid, _session_members(proc)
            )
            break
    try:
        proc.wait(timeout=_KILL_WAIT_S)
    except subprocess.TimeoutExpired:
        logger.warning("getdp process %d could not be reaped", proc.pid)
    if interrupt is not None:
        raise interrupt


# -- process-lifetime safety nets ---------------------------------------------

_PR_SET_PDEATHSIG = 1


@functools.lru_cache(maxsize=1)
def _libc_prctl() -> Callable[..., int] | None:
    """``prctl(2)`` of the C library (Linux only), resolved once; ``None`` elsewhere."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        import ctypes

        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return None
    # int prctl(int option, unsigned long arg2, ..., unsigned long arg5): declared,
    # so that the variadic arguments are passed as unsigned long, not as C int.
    prctl.argtypes = [ctypes.c_int, *[ctypes.c_ulong] * 4]
    prctl.restype = ctypes.c_int
    return prctl  # type: ignore[no-any-return]


def _parent_death_preexec() -> Callable[[], None] | None:
    """A ``preexec_fn`` setting ``PR_SET_PDEATHSIG=SIGKILL`` in the child (Linux only).

    The kernel then SIGKILLs the child when the thread that started it
    dies, including when the whole Python process is killed (``SIGKILL``,
    a crash). The flag survives ``execve`` -- through ``systemd-run
    --scope``, the wrapper and ``taskset`` (verified) -- but applies to
    the direct child only: the wrapper gives MPI ranks their own (see
    ``getdp_runner.sh``). If the parent died before the flag was set, the
    child kills itself at once.

    Returns:
        The function, or ``None`` where ``prctl`` is unavailable.
    """
    prctl = _libc_prctl()
    if prctl is None:
        return None
    parent = os.getpid()

    def preexec() -> None:  # runs in the child between fork and exec: keep it minimal
        prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
        if os.getppid() != parent:
            os.kill(os.getpid(), signal.SIGKILL)

    return preexec


def _cleanup_after_exit(proc: subprocess.Popen[bytes], run_dir: Path) -> None:
    """After a normal exit of ``proc``: terminate whatever it left in its session."""
    if _session_alive(proc):
        logger.warning(
            "getdp run in %s exited (code %s) but left processes in its session: %s; "
            "terminating them",
            run_dir,
            proc.returncode,
            _session_members(proc),
        )
        _terminate_session(proc)


_live_lock = threading.Lock()
_live_runs: dict[int, tuple[subprocess.Popen[bytes], MemoryScope | None]] = {}


def _register_live_run(proc: subprocess.Popen[bytes], scope: MemoryScope | None) -> None:
    with _live_lock:
        _live_runs[proc.pid] = (proc, scope)


def _unregister_live_run(proc: subprocess.Popen[bytes]) -> None:
    with _live_lock:
        _live_runs.pop(proc.pid, None)


@atexit.register
def _terminate_live_runs() -> None:
    """At interpreter exit, terminate the runs still alive (e.g. run in daemon threads).

    SIGTERM is sent to every live run first, so the grace periods overlap;
    each run is then terminated as on a timeout (bounded).
    """
    with _live_lock:
        runs = list(_live_runs.values())
    if not runs:
        return
    logger.warning("interpreter exiting with %d live getdp run(s): terminating them", len(runs))
    for proc, _ in runs:
        with contextlib.suppress(Exception):
            _signal_session(proc, signal.SIGTERM)
    for proc, scope in runs:
        with contextlib.suppress(Exception):
            _terminate_run(proc, scope)


class _TerminationSignalError(BaseException):
    """Internal: ``SIGTERM`` / ``SIGHUP`` received while a run was in progress."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"received signal {signum}")
        self.signum = signum


_TERMINATION_SIGNALS = tuple(
    getattr(signal, name) for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name)
)
_guard_depth = 0  # nesting of _termination_signals() in the main thread
_terminating = False  # a termination signal is being handled
_cleanup_depth = 0  # nesting of _termination_signals_deferred() in the main thread
_deferred_signal: tuple[Any, int] | None = None  # (previous handler, signum) held back


def _on_termination_signal(previous: Any, signum: int, frame: FrameType | None) -> None:
    """Handler installed by :func:`_termination_signals`."""
    global _terminating, _deferred_signal
    if _cleanup_depth > 0:
        # A session cleanup is in progress in the main thread: raising here would
        # abort it before the SIGKILL escalation. Handle the signal once it is done.
        if _deferred_signal is None and not _terminating:
            _deferred_signal = (previous, signum)
            logger.warning("signal %d received during a run cleanup: deferred until done", signum)
        else:
            logger.warning("signal %d ignored: the runs are being terminated", signum)
        return
    if callable(previous):
        previous(signum, frame)  # chained: the user's handler decides (it may raise)
        return
    if _terminating:
        logger.warning("signal %d ignored: the runs are being terminated", signum)
        return
    _terminating = True
    raise _TerminationSignalError(signum)


@contextlib.contextmanager
def _termination_signals_deferred() -> Iterator[None]:
    """Hold back ``SIGTERM`` / ``SIGHUP`` while a run cleanup is in progress.

    A termination signal caught by :func:`_termination_signals` during the
    block (for instance during the grace period of a cleanup started by a
    timeout, a cancellation or ``Ctrl-C``) is recorded instead of raising;
    when the outermost block ends, it is handled as if it arrived then
    (chained handler, or the internal exception that unwinds the runs and
    re-delivers it). The cleanup thus always reaches its SIGKILL escalation.
    Effective in the main thread only (where Python delivers signals).
    """
    global _cleanup_depth, _deferred_signal
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    _cleanup_depth += 1
    try:
        yield
    finally:
        _cleanup_depth -= 1
        if _cleanup_depth == 0 and _deferred_signal is not None:
            previous, signum = _deferred_signal
            _deferred_signal = None
            _on_termination_signal(previous, signum, None)


@contextlib.contextmanager
def _termination_signals() -> Iterator[None]:
    """Make ``SIGTERM`` / ``SIGHUP`` clean up the runs like ``Ctrl-C`` does.

    Active only in the main thread (where Python delivers signals) and
    only for the outermost use. For each signal whose current handler is
    the default one, a handler is installed that raises an internal
    :class:`BaseException` in the main thread: it unwinds through the
    usual cleanup (run sessions and scopes terminated, executors cancel
    their in-flight runs); the original handler is then restored and the
    signal re-delivered, so the process still dies of it. A user handler
    (a Python callable) is chained: it is called and decides (if it
    raises, the cleanup happens; if it returns, the runs go on). An
    ignored signal stays ignored. Further termination signals during the
    cleanup are ignored, and one arriving during a cleanup started
    otherwise (timeout, cancellation, ``Ctrl-C``) is deferred until that
    cleanup is done (:func:`_termination_signals_deferred`), so no cleanup
    can be cut short.
    """
    global _guard_depth, _terminating
    if threading.current_thread() is not threading.main_thread():
        yield  # signals are delivered to the main thread only
        return
    if _guard_depth > 0:  # nested: the outermost guard handles the signals
        _guard_depth += 1
        try:
            yield
        finally:
            _guard_depth -= 1
        return
    previous: dict[int, Any] = {}
    for sig in _TERMINATION_SIGNALS:
        handler = signal.getsignal(sig)
        if handler is None or handler == signal.SIG_IGN:
            continue  # set from C (cannot be restored) or deliberately ignored
        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, functools.partial(_on_termination_signal, handler))
            previous[sig] = handler
    _guard_depth += 1
    received: int | None = None
    try:
        yield
    except _TerminationSignalError as exc:
        received = exc.signum
    finally:
        _guard_depth -= 1
        _terminating = False
        for sig, handler in previous.items():
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sig, handler)
    if received is not None:
        os.kill(os.getpid(), received)  # default action restored: the process dies
        raise SystemExit(128 + received)  # pragma: no cover - only if the signal is blocked
