# Copyright (c) 2025 Thierry Valet
# SPDX-License-Identifier: MIT

"""SolverLite: subprocess wrapper around the native getdp binary.

A :class:`SolverLite` instance binds a GetDP project (a triplet of
``<project>.geo.pro`` / ``.physprop.pro`` / ``.cond.pro``) to a mesh and
a named GetDP Resolution, exposes a Python :class:`WorkflowData` for
runtime scalars, and runs the end-to-end pre-process / solve / optional
post-op pipeline through the :meth:`run` method as a single ``getdp``
invocation (one ``mpirun`` launch when the wrapper script is in MPI
mode).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from mpxlite.result import SolverResult
from mpxlite.workflow import WorkflowData

logger = logging.getLogger("mpxlite")

_CASE_EXTENSIONS = (".geo.pro", ".physprop.pro", ".cond.pro")
_MANIFEST_NAME = "mpxlite.pro"
_WORKFLOW_NAME = "workflow.pro"


class GetDPError(RuntimeError):
    """Raised when the ``getdp`` subprocess exits with a non-zero return code."""

    def __init__(
        self,
        returncode: int,
        stdout: str,
        stderr: str,
        cmd: Sequence[str],
    ) -> None:
        snippet = (stderr or stdout or "")[-500:].strip()
        super().__init__(
            f"getdp exited with code {returncode}\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr (last 500 chars): {snippet}"
        )
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.cmd = list(cmd)


class SolverLite:
    """Generic getdp subprocess wrapper for the mpxlite platform.

    Concrete physics solver subclasses pass their resolution_id and a
    list of bundled .pro resource files (typically resolved via
    :mod:`importlib.resources`).

    The instance writes a top-level manifest ``mpxlite.pro`` and a
    ``workflow.pro`` (Function block of scalars) into ``project_dir`` at
    each :meth:`run` call, then invokes the native ``getdp`` binary
    once with the appropriate stage flags (``-pre``, ``-cal`` and
    optionally ``-pos``) chained in a single CLI invocation.

    Args:
        project_dir: Directory holding the case files and (optionally) the mesh.
        project_name: Base name of the case (e.g. ``"chequerboard"``); the
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
        workflow: An initial :class:`WorkflowData`; a fresh empty one is
            created if not supplied.
        verbosity: GetDP verbosity level passed via ``-v`` (0 silent .. 5).
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
    ) -> None:
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

        self.mesh = self._resolve_mesh(mesh)
        self._verify_case_files()
        self._verify_resource_files()

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

    def _verify_case_files(self) -> None:
        for ext in _CASE_EXTENSIONS:
            target = self.project_dir / f"{self.project_name}{ext}"
            if not target.is_file():
                raise FileNotFoundError(f"Required case file missing: {target}")

    def _verify_resource_files(self) -> None:
        for path in self.resource_files:
            if not path.is_file():
                raise FileNotFoundError(f"Resource .pro file missing: {path}")

    # -- Manifest writing ----------------------------------------------

    def _write_workflow_pro(self) -> Path:
        path = self.project_dir / _WORKFLOW_NAME
        self.workflow.write_pro(path)
        return path

    def _write_manifest(self) -> Path:
        """Write the top-level ``mpxlite.pro`` ``Include`` chain.

        Order:
            1. ``workflow.pro``                (Function: scalars)
            2. ``<project>.geo.pro``           (Group: regions)
            3. ``resource_files[*]``           (FE machinery and formulation)
            4. ``<project>.physprop.pro``      (Function: ε, μ, σ, sources)
            5. ``<project>.cond.pro``          (Constraint)

        Returns:
            The absolute path of the manifest file just written.
        """
        manifest = self.project_dir / _MANIFEST_NAME
        lines: list[str] = [
            "// mpxlite.pro — auto-generated. Do not edit by hand.",
            f'Include "{_WORKFLOW_NAME}";',
            f'Include "{self.project_name}.geo.pro";',
        ]
        for resource in self.resource_files:
            lines.append(f'Include "{resource}";')
        lines.append(f'Include "{self.project_name}.physprop.pro";')
        lines.append(f'Include "{self.project_name}.cond.pro";')
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return manifest

    # -- Public stages -------------------------------------------------

    def run(
        self,
        postop: str | None = None,
    ) -> SolverResult:
        """End-to-end getdp invocation: pre-process, solve, optional post-op.

        Writes ``workflow.pro`` and ``mpxlite.pro`` then runs::

            getdp mpxlite.pro -msh ... -pre <resolution_id> -cal [-pos <postop>]

        in a single subprocess call. Compatible with mpirun-wrapped getdp
        binaries: pre-process, solve and post-op share the in-memory
        PETSc state, so output integrals on internal Fourier surfaces are
        evaluated correctly.

        GetDP's CLI accepts only ONE ``-pos`` argument per invocation
        (multiple flags overwrite each other silently); for workflows
        that need multiple outputs in a single solve, define a composite
        ``PostOperation`` in the ``.pro`` that bundles the desired
        ``Print[]`` calls.

        Args:
            postop: name of the ``PostOperation`` to evaluate after the
                solve. ``None`` (default) runs ``-pre`` + ``-cal`` only,
                useful for benchmarking the pre + solve pipeline.

        Returns:
            :class:`SolverResult` carrying the project directory and the
            :class:`subprocess.CompletedProcess`.
        """
        self._write_workflow_pro()
        manifest = self._write_manifest()

        args: list[str] = ["-pre", self.resolution_id, "-cal"]
        if postop is not None:
            args += ["-pos", postop]

        completed = self._run_getdp(manifest, args)
        return SolverResult(work_dir=self.project_dir, completed=completed)

    # -- subprocess plumbing -------------------------------------------

    def _run_getdp(
        self,
        manifest: Path,
        extra_args: Sequence[str],
    ) -> subprocess.CompletedProcess[str]:
        getdp = self._resolve_getdp()
        cmd = [
            getdp,
            str(manifest),
            "-msh",
            str(self.mesh),
            "-v",
            str(self.verbosity),
            *extra_args,
        ]
        logger.info("running %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            cwd=str(self.project_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise GetDPError(result.returncode, result.stdout, result.stderr, cmd)
        return result

    @staticmethod
    def _resolve_getdp() -> str:
        """Locate the ``getdp`` binary.

        Search order:
            1. ``MPXLITE_GETDP`` environment variable (must point to an
               executable file). Typically the ``tools/getdp_runner.sh``
               wrapper that handles ``mpirun`` invocation and
               ``LD_LIBRARY_PATH`` augmentation.
            2. ``getdp`` on ``PATH`` (via :func:`shutil.which`).

        Returns:
            The absolute path of a callable ``getdp`` binary.

        Raises:
            FileNotFoundError: if no usable ``getdp`` binary is found.
        """
        env = os.environ.get("MPXLITE_GETDP")
        if env:
            path = Path(env).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
            raise FileNotFoundError(f"MPXLITE_GETDP={env!r} is not an executable file.")
        which = shutil.which("getdp")
        if which:
            return which
        raise FileNotFoundError(
            "getdp binary not found. Install getdp and put it on PATH, or "
            "set MPXLITE_GETDP to its absolute path."
        )
