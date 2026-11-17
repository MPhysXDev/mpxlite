# Copyright (c) 2025 MPhysX
# SPDX-License-Identifier: MIT

"""Result of a :meth:`SolverLite.run` invocation.

Minimal record of the artefacts produced by a single end-to-end getdp call:
the run directory, the log file the subprocess output was streamed to, the
wall-clock time, the return code, the executed command and the GetDP
warnings found in the log. Output-parsing
helpers live in :mod:`mpxlite.parsers`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MANIFEST_NAME = "mpxlite.pro"
"""Name, in a run directory, of the main ``.pro`` file written by mpxlite."""

WORKFLOW_NAME = "workflow.pro"
"""Name, in a run directory, of the file of the scalars of the run."""

PETSC_OPTIONS_NAME = "petsc.opts"
"""Name, in a run directory, of the PETSc options file (written when options are set)."""

LOG_NAME = "getdp.log"
"""Name, in a run directory, of the log of the engine."""

PRE_NAME = "mpxlite.pre"
"""Name, in a run directory, of the pre-processing file GetDP writes."""

SOLUTION_NAME = "mpxlite.res"
"""Name, in a run directory, of the solution file GetDP writes."""


@dataclass(frozen=True)
class SolverResult:
    """Artefacts of a single end-to-end getdp invocation.

    Attributes:
        work_dir: The RUN directory. It holds the generated ``workflow.pro``
            and ``mpxlite.pro`` (and ``petsc.opts`` when PETSc options were
            set), the ``getdp.log`` and every getdp-side
            output (``.pre``, ``.res``, ``.dat``, ``.pos``, …). The shared
            case files and mesh are NOT in it (they are referenced by
            absolute path).
        run_id: Identifier of the run (the basename of ``work_dir``).
        log_path: ``work_dir / "getdp.log"``: merged stdout + stderr of
            the getdp subprocess.
        elapsed_s: Wall-clock duration of the subprocess, in seconds.
        returncode: Exit status of the subprocess (0 on success).
        cmd: The executed command line.
        warnings: The GetDP ``Warning`` lines of the log, in order of first
            appearance, de-duplicated; ANSI colours and the ``[rank N]`` tag
            of MPI runs are removed, so a warning printed by every rank
            appears once. GetDP prints warnings only at verbosity >= 2.
        memory_max: The hard memory cap the run executed under, in bytes
            (``None``: not capped). ``cmd`` does not include the
            ``systemd-run`` prefix that enforced it.
    """

    work_dir: Path
    run_id: str
    log_path: Path
    elapsed_s: float
    returncode: int
    cmd: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    memory_max: int | None = None

    @property
    def manifest_path(self) -> Path:
        """``work_dir / "mpxlite.pro"``: the main ``.pro`` file of the run."""
        return self.work_dir / MANIFEST_NAME

    @property
    def workflow_path(self) -> Path:
        """``work_dir / "workflow.pro"``: the scalars of the run.

        :meth:`mpxlite.WorkflowData.read_pro` reads them back.
        """
        return self.work_dir / WORKFLOW_NAME

    @property
    def petsc_options_path(self) -> Path:
        """``work_dir / "petsc.opts"``; it exists only when the run had PETSc options."""
        return self.work_dir / PETSC_OPTIONS_NAME

    @property
    def pre_path(self) -> Path:
        """``work_dir / "mpxlite.pre"``: the pre-processing file written by GetDP."""
        return self.work_dir / PRE_NAME

    @property
    def solution_path(self) -> Path:
        """``work_dir / "mpxlite.res"``: the solution file, when the resolution saves one."""
        return self.work_dir / SOLUTION_NAME

    def path(self, name: str) -> Path:
        """Path of an output file inside the run directory.

        Args:
            name: File name (or relative path) as written by the ``.pro``
                ``PostOperation``, e.g. ``"eff.dat"``.

        Returns:
            ``work_dir / name``. The file is not required to exist.

        Raises:
            ValueError: if ``name`` is absolute or escapes the run directory
                (``..`` component).
        """
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Output name must stay inside the run directory: {name!r}")
        return self.work_dir / rel
