# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""Gmsh jobs: programs that run Gmsh on files, started as separate processes.

A plug-in never imports Gmsh. What it needs Gmsh to do (make the parts of a
geometry conform, measure them, mesh them) is done by a *job*: a program of
the mpxlite installation that uses the Gmsh API, reads input files and a
parameter document, and writes output files and a report document. Jobs live
beside the engine, outside this package (they are GPL-2.0-or-later, like
Gmsh); this module only runs them, all in the same way
(:func:`run_gmsh_job`):

* every run has its own folder ``runs_dir / run_id``, never reused;
* the request is written there as ``request.json``; the job writes its output
  files there and its report as ``report.json``; its standard output and
  error go to ``job.log``;
* the job runs in its own process session with ``<home>/venv/bin/python``,
  with one OpenMP thread and one Gmsh thread (reproducible meshes), under an
  optional wall-clock limit and an optional hard memory cap (the machinery of
  :meth:`mpxlite.SolverLite.run`);
* a failure raises :class:`GmshJobError`, whose ``kind`` and ``code`` tell a
  plug-in what went wrong so that it can write its own message;
* a job is serial pre-processing, run once by the driving script before the
  parallel run: the runner refuses to run in a process that is one of several
  MPI ranks (:data:`MPI_SIZE_VARIABLES`).

Where the jobs are (:func:`find_gmsh_jobs`): the directory named by
``MPXLITE_GMSH_JOBS`` when it is set (development), otherwise
``<home>/engine/gmsh/jobs`` of the installation
(:func:`mpxlite.installation_home`). A job ``name`` is the program
``<jobs>/<name>.py``.

Request (``request.json``)::

    {"format": "mpxlite-gmsh-job-request", "version": 1, "job": "<name>",
     "inputs": {"<input name>": "<absolute path>", ...},
     "parameters": {...}}

Report (``report.json``), written by the job::

    {"format": "mpxlite-gmsh-job-report", "version": 1, "job": "<name>",
     "status": "ok" | "failed",
     "outputs": {"<output name>": "<path relative to the run folder>", ...},
     "report": {...},          # the result of the job (status "ok")
     "failure": {"kind": ..., "code": ..., "message": ..., "data": {...}},
     "messages": [{"level": "info" | "warning" | "error", "text": ...}, ...],
     "gmsh": "<Gmsh version>", "elapsed_s": <seconds>}
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mpxlite import base as _base
from mpxlite import memory as _memory
from mpxlite.engine import installation_home
from mpxlite.memory import MemoryScope, require_hard_cap

__all__ = [
    "FORMAT_VERSION",
    "GMSH_JOBS_ENV",
    "JOB_FAILURE_KINDS",
    "JOB_LOG_NAME",
    "JOB_REPORT_NAME",
    "JOB_REQUEST_NAME",
    "MPI_SIZE_VARIABLES",
    "REPORT_FORMAT",
    "REQUEST_FORMAT",
    "GmshJobError",
    "GmshJobResult",
    "find_gmsh_jobs",
    "run_gmsh_job",
]

logger = logging.getLogger("mpxlite")

GMSH_JOBS_ENV = "MPXLITE_GMSH_JOBS"
"""Environment variable naming the directory of the Gmsh jobs (development)."""

REQUEST_FORMAT = "mpxlite-gmsh-job-request"
"""``format`` field of a request document."""

REPORT_FORMAT = "mpxlite-gmsh-job-report"
"""``format`` field of a report document."""

FORMAT_VERSION = 1
"""Version of the request and report formats (``version`` field)."""

JOB_REQUEST_NAME = "request.json"
"""Name, in a run folder, of the request document."""

JOB_REPORT_NAME = "report.json"
"""Name, in a run folder, of the report document."""

JOB_LOG_NAME = "job.log"
"""Name, in a run folder, of the log of the job (its standard output and error)."""

JOB_FAILURE_KINDS = ("input", "rule", "gmsh", "timeout", "memory", "crash")
"""Kinds of :class:`GmshJobError`.

* ``input``: an input file or a parameter is missing, unreadable or invalid;
* ``rule``: the input is readable but breaks a rule of the job (for instance
  a geometry that is not a periodic cell);
* ``gmsh``: Gmsh reported an error, or a warning that means a wrong result,
  or a limit of the job (element count) was exceeded;
* ``timeout``: the wall-clock limit of the run was exceeded;
* ``memory``: the run was killed by its memory cap;
* ``crash``: the job ended without a valid report (a defect).
"""

MPI_SIZE_VARIABLES = (
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "SLURM_STEP_NUM_TASKS",
)
"""Environment variables by which an MPI launcher tells a process the number of ranks.

``OMPI_COMM_WORLD_SIZE``: Open MPI's ``mpirun`` / ``mpiexec`` (the MPI of the
engine, 4.x and 5.x); ``PMI_SIZE``: the PMI launchers (MPICH's Hydra, Intel
MPI, ``srun --mpi=pmi2``); ``MV2_COMM_WORLD_SIZE``: MVAPICH2;
``SLURM_STEP_NUM_TASKS``: the tasks of a Slurm job step (``srun``), which a
batch script itself is not. :func:`run_gmsh_job` refuses to run when one of
them is an integer greater than 1 in the environment of the calling process
(code ``job.mpi_ranks``); one rank, or no launcher, passes.
"""

_JOBS_IN_HOME = Path("engine") / "gmsh" / "jobs"
_PYTHON_IN_HOME = Path("venv") / "bin" / "python"
_JOB_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")
_INPUT_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


class GmshJobError(RuntimeError):
    """A Gmsh job failed.

    Raised by :func:`run_gmsh_job`, and by a job itself (its toolkit turns
    it into the ``failure`` of the report).

    Attributes:
        kind: One of :data:`JOB_FAILURE_KINDS`.
        code: Short machine-readable identifier chosen by the job, dotted
            (for instance ``"cell.not_periodic"``); the runner's own codes
            start with ``"job."``.
        message: Human-readable description, in the job's terms.
        data: Machine-readable details (JSON values), for the plug-in's
            message.
        job: Name of the job (``None`` when raised inside the job).
        run_dir: The run folder (kept for diagnosis), or ``None``.
        log_path: The log of the job, or ``None``.
    """

    def __init__(
        self,
        kind: str,
        code: str,
        message: str,
        data: Mapping[str, Any] | None = None,
        *,
        job: str | None = None,
        run_dir: Path | None = None,
        log_path: Path | None = None,
    ) -> None:
        if kind not in JOB_FAILURE_KINDS:
            raise ValueError(f"kind must be one of {JOB_FAILURE_KINDS}, got {kind!r}")
        self.kind = kind
        self.code = str(code)
        self.message = str(message)
        self.data: dict[str, Any] = dict(data or {})
        self.job = job
        self.run_dir = run_dir
        self.log_path = log_path
        where = f" (job {job}, run folder {run_dir})" if job is not None else ""
        super().__init__(f"[{kind}: {self.code}] {self.message}{where}")

    def to_json(self) -> dict[str, Any]:
        """The ``failure`` object of a report."""
        return {"kind": self.kind, "code": self.code, "message": self.message, "data": self.data}


@dataclass(frozen=True)
class GmshJobResult:
    """Outcome of a successful job run.

    Attributes:
        job: Name of the job.
        run_id: Identifier of the run (the name of ``run_dir``).
        run_dir: The run folder: ``request.json``, ``report.json``,
            ``job.log`` and the output files.
        report: The result of the job (the ``report`` object of
            ``report.json``); its content is defined by the job.
        outputs: Output name -> absolute path of the file, in ``run_dir``.
        log_path: ``run_dir / "job.log"``.
        elapsed_s: Wall-clock time of the job process, in seconds.
        messages: The messages of Gmsh the job collected, as
            ``(level, text)`` pairs (warnings and errors that did not fail
            the job, information).
        gmsh_version: Version of the Gmsh the job ran.
    """

    job: str
    run_id: str
    run_dir: Path
    report: Mapping[str, Any]
    outputs: Mapping[str, Path]
    log_path: Path
    elapsed_s: float
    messages: tuple[tuple[str, str], ...] = field(default=())
    gmsh_version: str = ""

    @property
    def warnings(self) -> tuple[str, ...]:
        """The texts of the messages of level ``"warning"``."""
        return tuple(text for level, text in self.messages if level == "warning")


def find_gmsh_jobs(env: Mapping[str, str] | None = None) -> Path:
    """The directory of the Gmsh jobs (see the module documentation).

    Args:
        env: The environment read (default: :data:`os.environ`), for
            ``MPXLITE_GMSH_JOBS`` and ``MPXLITE_HOME``.

    Returns:
        The absolute path of the directory.

    Raises:
        FileNotFoundError: ``MPXLITE_GMSH_JOBS`` does not name a directory,
            or it is unset and the installation has no jobs.
    """
    environ = os.environ if env is None else env
    named = environ.get(GMSH_JOBS_ENV, "")
    if named:
        path = Path(named).expanduser()
        if path.is_dir():
            return path.resolve()
        raise FileNotFoundError(f"{GMSH_JOBS_ENV}={named!r} is not a directory.")
    home = installation_home(environ)
    path = home / _JOBS_IN_HOME
    if path.is_dir():
        return path.resolve()
    raise FileNotFoundError(
        f"No Gmsh jobs at {path}: mpxlite is not installed at {home}. Install mpxlite from "
        f"its bundle (install.sh), or set MPXLITE_HOME to its installation, or "
        f"{GMSH_JOBS_ENV} to a directory of jobs."
    )


def _job_python(env: Mapping[str, str]) -> Path:
    """``<home>/venv/bin/python`` of the installation."""
    home = installation_home(env)
    path = home / _PYTHON_IN_HOME
    if path.is_file() and os.access(path, os.X_OK):
        return path
    raise FileNotFoundError(
        f"No Python at {path}: mpxlite is not installed at {home}. Install mpxlite from "
        "its bundle (install.sh), or set MPXLITE_HOME to its installation."
    )


def _check_job_name(job: object) -> str:
    if not isinstance(job, str) or not job or not set(job) <= _JOB_NAME_CHARS or job[0] == "_":
        raise ValueError(f"a job name is made of a-z, 0-9 and '_' (not first); got {job!r}")
    return job


def _check_inputs(job: str, inputs: Mapping[str, Path | str] | None) -> dict[str, str]:
    """Input name -> absolute path of an existing file."""
    out: dict[str, str] = {}
    for name, value in (inputs or {}).items():
        if not isinstance(name, str) or not name or not set(name) <= _INPUT_NAME_CHARS:
            raise ValueError(f"an input name is made of letters, digits and '_'; got {name!r}")
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise GmshJobError(
                "input",
                "job.input_missing",
                f"the input {name!r} of the job {job} is not a file: {path}",
                {"input": name, "path": str(path)},
                job=job,
            )
        out[name] = str(path)
    return out


def _check_single_process(job: str) -> None:
    """Refuses a calling process that is one of several MPI ranks."""
    for variable in MPI_SIZE_VARIABLES:
        value = os.environ.get(variable, "").strip()
        if value.isdecimal() and int(value) > 1:
            raise GmshJobError(
                "input",
                "job.mpi_ranks",
                f"the Gmsh job {job} was called by one of {int(value)} MPI ranks "
                f"({variable}={value}): a Gmsh job is serial pre-processing, to be run "
                "once, before the parallel run, by the driving script (started without "
                "an MPI launcher, or with one rank); the parallel run is the engine's",
                {"variable": variable, "size": int(value)},
                job=job,
            )


def _check_parameters(parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    """A JSON round trip of the parameters (finite numbers only)."""
    if parameters is None:
        return {}
    if not isinstance(parameters, Mapping):
        raise TypeError(f"parameters must be a mapping, got {type(parameters).__name__}")
    try:
        text = json.dumps(dict(parameters), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"the parameters are not a JSON document: {error}") from None
    return json.loads(text)


def _failure_from_report(job: str, run_dir: Path, log_path: Path, doc: Any) -> GmshJobError:
    failure = doc.get("failure") if isinstance(doc, dict) else None
    if (
        isinstance(failure, dict)
        and failure.get("kind") in JOB_FAILURE_KINDS
        and isinstance(failure.get("code"), str)
        and isinstance(failure.get("message"), str)
        and isinstance(failure.get("data", {}), dict)
    ):
        return GmshJobError(
            failure["kind"],
            failure["code"],
            failure["message"],
            failure.get("data", {}),
            job=job,
            run_dir=run_dir,
            log_path=log_path,
        )
    return GmshJobError(
        "crash",
        "job.report_invalid",
        f"the job {job} reported a failure without a valid failure object: {failure!r}",
        job=job,
        run_dir=run_dir,
        log_path=log_path,
    )


def _crash(job: str, run_dir: Path, log_path: Path, code: str, what: str) -> GmshJobError:
    tail = _base._read_log_tail(log_path)
    return GmshJobError(
        "crash",
        code,
        f"the job {job} {what}; last lines of its log:\n{tail or '(empty)'}",
        {"log_tail": tail},
        job=job,
        run_dir=run_dir,
        log_path=log_path,
    )


def _read_report(job: str, run_dir: Path, log_path: Path, returncode: int) -> dict[str, Any]:
    """The report document; raises the failure it reports, or a crash."""
    path = run_dir / JOB_REPORT_NAME
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise _crash(
            job, run_dir, log_path, "job.no_report", f"exited with code {returncode} and no report"
        ) from None
    except (OSError, ValueError) as error:
        raise _crash(
            job, run_dir, log_path, "job.report_invalid", f"wrote no JSON report: {error}"
        ) from None
    if (
        not isinstance(doc, dict)
        or doc.get("format") != REPORT_FORMAT
        or doc.get("version") != FORMAT_VERSION
    ):
        found = (doc.get("format"), doc.get("version")) if isinstance(doc, dict) else doc
        raise _crash(
            job,
            run_dir,
            log_path,
            "job.report_format",
            f"wrote a report of another format {found!r} (expected {REPORT_FORMAT!r} "
            f"version {FORMAT_VERSION})",
        )
    if doc.get("status") == "failed":
        raise _failure_from_report(job, run_dir, log_path, doc)
    if doc.get("status") != "ok" or returncode != 0:
        raise _crash(
            job,
            run_dir,
            log_path,
            "job.exit",
            f"exited with code {returncode} and status {doc.get('status')!r}",
        )
    return doc


def _outputs(job: str, run_dir: Path, log_path: Path, doc: Mapping[str, Any]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    declared = doc.get("outputs", {})
    if not isinstance(declared, dict):
        raise _crash(job, run_dir, log_path, "job.report_invalid", "reported no outputs object")
    root = run_dir.resolve()
    for name, rel in declared.items():
        path = (root / str(rel)).resolve()
        if root not in path.parents or not path.is_file():
            raise _crash(
                job,
                run_dir,
                log_path,
                "job.output_missing",
                f"reported the output {name!r} = {rel!r}, which is not a file of its run folder",
            )
        out[str(name)] = path
    return out


def _messages(doc: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    found = doc.get("messages", [])
    if not isinstance(found, list):
        return ()
    return tuple(
        (str(m.get("level", "")), str(m.get("text", ""))) for m in found if isinstance(m, dict)
    )


def _run_process(
    job: str,
    argv: list[str],
    run_dir: Path,
    env: Mapping[str, str],
    timeout: float | None,
    scope: MemoryScope | None,
) -> tuple[int, float]:
    """Run the job process in its own session; ``(returncode, elapsed)``.

    Raises:
        GmshJobError: of kind ``timeout`` or ``memory``.
    """
    log_path = run_dir / JOB_LOG_NAME
    full = [*(scope.command_prefix() if scope is not None else []), *argv]
    logger.info("running Gmsh job %s (%s)", job, " ".join(full))
    with _base._termination_signals(), log_path.open("wb") as log:
        start = time.perf_counter()
        proc: subprocess.Popen[bytes] | None = None
        try:
            proc = subprocess.Popen(
                full,
                cwd=str(run_dir),
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=_base._parent_death_preexec(),
            )
            _base._register_live_run(proc, scope)
            returncode = _base._wait(proc, timeout, None)
            elapsed = time.perf_counter() - start
            _base._cleanup_after_exit(proc, run_dir)
            oom = scope is not None and scope.oom_killed(settle=returncode != 0)
        except subprocess.TimeoutExpired:
            _base._terminate_run(proc, scope)  # type: ignore[arg-type]
            raise GmshJobError(
                "timeout",
                "job.timeout",
                f"the job {job} exceeded its time limit of {timeout} s (its processes killed)",
                {"timeout_s": timeout},
                job=job,
                run_dir=run_dir,
                log_path=log_path,
            ) from None
        except BaseException:
            if proc is not None:
                _base._terminate_run(proc, scope)
            raise
        finally:
            if proc is not None:
                _base._unregister_live_run(proc)
    if oom and scope is not None:
        raise GmshJobError(
            "memory",
            "job.memory",
            f"the job {job} was killed by its memory cap of {scope.memory_max} bytes "
            f"({scope.memory_max / 2**30:.3g} GiB)",
            {"memory_max": scope.memory_max},
            job=job,
            run_dir=run_dir,
            log_path=log_path,
        )
    return returncode, elapsed


def run_gmsh_job(
    job: str,
    *,
    inputs: Mapping[str, Path | str] | None = None,
    parameters: Mapping[str, Any] | None = None,
    runs_dir: Path | str,
    run_id: str | None = None,
    timeout: float | None = None,
    memory_max: int | None = None,
    env: Mapping[str, str] | None = None,
) -> GmshJobResult:
    """Run a Gmsh job in a separate process (see the module documentation).

    Args:
        job: Name of the job (``periodic_cell_geometry``, ...): the program
            ``<jobs>/<job>.py`` (:func:`find_gmsh_jobs`).
        inputs: Input name -> input file. The files are read in place (by
            absolute path), never written.
        parameters: The parameters of the job, a JSON document (mapping of
            strings, finite numbers, booleans, ``None``, lists and mappings).
        runs_dir: Directory of the run folders (created if needed).
        run_id: Name of the run folder; it must match
            ``^[A-Za-z0-9][A-Za-z0-9._-]*$`` and not exist yet. Default: a
            timestamp plus a random suffix.
        timeout: Wall-clock limit in seconds (``None``: none). On expiry
            every process of the job is killed.
        memory_max: Hard memory cap of the job process, in bytes (``None``:
            none), as :meth:`mpxlite.SolverLite.run` applies it.
        env: Variables added to the environment of the job process (over
            :data:`os.environ`). ``OMP_NUM_THREADS`` is always ``1``.

    Returns:
        The :class:`GmshJobResult`.

    Raises:
        ValueError: ``job``, an input name or ``run_id`` is malformed, the
            parameters are not a JSON document, ``timeout`` is negative.
        TypeError: an argument has an unsupported type.
        FileNotFoundError: there are no jobs, no such job, or no Python of
            the installation.
        FileExistsError: the run folder exists already.
        HardCapUnavailableError: ``memory_max`` is set but cannot be
            enforced on this machine.
        GmshJobError: the job failed (``kind`` says how), or the calling
            process is one of several MPI ranks (kind ``input``, code
            ``job.mpi_ranks``, data ``variable`` and ``size``; raised before
            anything is created, see :data:`MPI_SIZE_VARIABLES`).
    """
    name = _check_job_name(job)
    timeout = _base._check_timeout(timeout)
    cap = _base._check_memory_max(memory_max)
    params = _check_parameters(parameters)
    _check_single_process(name)
    environ = {**os.environ, **dict(env or {})}
    environ["OMP_NUM_THREADS"] = "1"
    program = find_gmsh_jobs(environ) / f"{name}.py"
    if not program.is_file():
        raise FileNotFoundError(f"No Gmsh job {name!r}: {program} is not a file.")
    python = _job_python(environ)
    files = _check_inputs(name, inputs)
    rid = _base.SolverLite._new_run_id(run_id)
    if cap is not None:
        require_hard_cap("run")
        _memory.reap_stale_scopes_once()
    run_dir = Path(runs_dir).expanduser().resolve() / rid
    run_dir.mkdir(parents=True, exist_ok=False)
    request = {
        "format": REQUEST_FORMAT,
        "version": FORMAT_VERSION,
        "job": name,
        "inputs": files,
        "parameters": params,
    }
    (run_dir / JOB_REQUEST_NAME).write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    argv = [str(python), "-I", str(program), str(run_dir)]
    scope = MemoryScope.for_run(f"gmsh-{rid}", cap) if cap is not None else None
    returncode, elapsed = _run_process(name, argv, run_dir, environ, timeout, scope)
    log_path = run_dir / JOB_LOG_NAME
    doc = _read_report(name, run_dir, log_path, returncode)
    report = doc.get("report", {})
    if not isinstance(report, dict):
        raise _crash(name, run_dir, log_path, "job.report_invalid", "reported no report object")
    result = GmshJobResult(
        job=name,
        run_id=rid,
        run_dir=run_dir,
        report=report,
        outputs=_outputs(name, run_dir, log_path, doc),
        log_path=log_path,
        elapsed_s=elapsed if math.isfinite(elapsed) else 0.0,
        messages=_messages(doc),
        gmsh_version=str(doc.get("gmsh", "")),
    )
    for text in result.warnings:
        logger.warning("Gmsh job %s (%s): %s", name, rid, text)
    return result
