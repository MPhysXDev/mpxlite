# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""Smoke test of an mpxlite installation (``mpxlite-smoke``).

1. Engine identity: ``getdp --info`` reports GetDP 3.5.0 with the mpx3 patch,
   complex PETSc 3.26.0 and exactly the expected build options.
2. The ``rod`` Poisson problem of mpxlite's integration tests
   (``-u'' = S`` on the unit segment, ``u(0) = 0``, ``u(1) = V``, P1 elements,
   nodal values exact) in serial, against the exact solution.
3. The same on 2 MPI ranks.
4. MUMPS, asked for the METIS ordering (``ICNTL(7) = 5``), reports that it used
   it (``INFOG(7) = 5``), in serial and on 2 ranks.
5. petsc4py, the Python bindings of the PETSc of the engine: complex scalars,
   the libpetsc of the installation, and a sparse complex system factorised in
   this process by MUMPS with the METIS ordering, against its exact solution.
6. Gmsh for the user's scripts: ``import gmsh`` in the Python of the
   installation imports the module of the engine (``engine/gmsh/gmsh.py``),
   which loads the library of the engine (``engine/lib/libgmsh.so.4.15``),
   reports the expected version, builds a unit cube with OpenCASCADE
   (volume 1) and meshes it.
7. The ``gmsh`` command: the ``gmsh`` on ``PATH`` is the program of the engine
   (``engine/bin/gmsh``); ``gmsh --version`` prints the expected version and
   ``gmsh -info`` lists the graphical interface (``Fltk``), both without a
   display.
8. The Gmsh jobs: :func:`mpxlite.run_gmsh_job` runs the job
   ``periodic_cell_geometry`` of the installation on a two-layer box written
   by this process (an XAO file with named parts) and reports its lattice,
   its parts and the classification of its faces; then the job
   ``layered_section_mesh`` on a band of two columns with a free layer.

Prints one line per check and PASS or FAIL; exit status 0 on PASS. Takes a few
seconds. Needs the environment of ``activate.sh`` (``mpxlite-smoke`` sources it).
Usage: ``mpxlite-smoke [--data DIR] [--keep]``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from pathlib import Path

EXPECTED_OPTIONS = "64Bit Blas[petsc] Kernel Lapack[petsc] MPI NoFortran PETSc"
#: Version of the Gmsh of the engine, and its library in ``engine/lib``.
EXPECTED_GMSH = "4.15.2"
GMSH_LIBRARY = "libgmsh.so.4.15"
V, S = 2.0, 8.0
#: ``-ksp_view`` line of MUMPS when the METIS ordering was used.
METIS_USED = "INFOG(7) (ordering option effectively used after analysis): 5"
MUMPS_METIS = {"pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_7": 5, "ksp_view": None}


def check_engine() -> None:
    """The configured getdp is the mpx3 build on complex PETSc 3.26.0."""
    import mpxlite

    binary = mpxlite.find_engine()
    home = mpxlite.installation_home()
    if not (home / "engine" / "bin" / "getdp").samefile(binary):
        raise RuntimeError(f"{binary} is not the engine of the installation {home}")
    lines = mpxlite.engine_info()
    version = lines.get("Version", "")
    petsc = lines.get("PETSc version", "")
    options = lines.get("Build options", "")
    print(f"  getdp   {binary}")
    print(f"  version {version}")
    print(f"  PETSc   {petsc}")
    print(f"  options {options}")
    if not version.startswith("3.5.0-mpx3"):
        raise RuntimeError(f"not GetDP 3.5.0 with the mpx3 patch: {version!r}")
    if petsc != "3.26.0 (complex arithmetic)":
        raise RuntimeError(f"not complex PETSc 3.26.0: {petsc!r}")
    if options.split() != EXPECTED_OPTIONS.split():
        raise RuntimeError(f"build options {options!r}, expected {EXPECTED_OPTIONS!r}")


class Rod:
    """The rod case on a read-only data directory; every run in its own run directory."""

    def __init__(self, data: Path, runs: Path) -> None:
        from mpxlite import SolverLite, WorkflowData

        self.solver = SolverLite(
            data,
            "rod",
            "rod.msh",
            "Static",
            resource_files=[data / "rod.formulation.pro"],
            workflow=WorkflowData({"V": V, "S": S}),
            verbosity=3,
            runs_dir=runs,
        )

    def run(self, run_id: str, np_ranks: int = 1, metis: bool = False) -> tuple[float, float, str]:
        """Solve; returns (u(0.5), integral of u, log text)."""
        from mpxlite import LaunchSpec, parse_real_table

        t0 = time.perf_counter()
        res = self.solver.run(
            ["mid", "integral"],
            run_id=run_id,
            launch=LaunchSpec(np=np_ranks, threads=1) if np_ranks > 1 else None,
            petsc_options=MUMPS_METIS if metis else None,
            timeout=120,
        )
        mid = float(parse_real_table(res.path("u_mid.dat"))[-1, -1])
        integral = float(parse_real_table(res.path("u_int.dat"))[-1, -1])
        log = res.log_path.read_text(encoding="utf-8", errors="replace")
        print(f"  {run_id}: {np_ranks} rank(s), {time.perf_counter() - t0:.2f} s")
        return mid, integral, log


def check_exact(rod: Rod, run_id: str, np_ranks: int) -> None:
    """u(0.5) and the integral of u match the exact P1 values."""
    mid, integral, log = rod.run(run_id, np_ranks)
    exact_mid = V / 2 + S / 8
    exact_int = V / 2 + S * (1 - 0.1**2) / 12
    err = max(abs(mid / exact_mid - 1), abs(integral / exact_int - 1))
    print(f"  u(0.5) = {mid:.12g} (exact {exact_mid:g}), integral = {integral:.12g}")
    print(f"  max relative error = {err:.1e} (tol 1e-9)")
    if err > 1e-9:
        raise RuntimeError("the solution differs from the exact one")
    if np_ranks > 1 and f"[rank {np_ranks - 1:3d}]" not in log:
        raise RuntimeError(f"getdp did not report rank {np_ranks - 1} (no MPI run?)")


def check_metis(rod: Rod) -> None:
    """MUMPS reports the METIS ordering it was asked for, in serial and on 2 ranks."""
    for run_id, np_ranks in (("metis-serial", 1), ("metis-np2", 2)):
        mid, _, log = rod.run(run_id, np_ranks, metis=True)
        if "package used to perform factorization: mumps" not in log:
            raise RuntimeError(f"{run_id}: MUMPS was not the factorization package")
        if np_ranks > 1 and "[rank   1]" not in log:
            raise RuntimeError(f"{run_id}: getdp did not report rank 1 (no MPI run?)")
        if METIS_USED not in log:
            raise RuntimeError(f"{run_id}: MUMPS did not report the METIS ordering")
        if abs(mid / (V / 2 + S / 8) - 1) > 1e-9:
            raise RuntimeError(f"{run_id}: wrong solution")
        print(f"  {run_id}: MUMPS, {METIS_USED}")


def check_petsc4py() -> None:
    """petsc4py is bound to the PETSc of the engine and factorises with MUMPS and METIS."""
    import numpy as np

    import mpxlite

    try:
        import petsc4py
    except ImportError as exc:
        raise RuntimeError(f"petsc4py is not installed in this environment: {exc}") from exc
    if "petsc4py.PETSc" not in sys.modules:
        petsc4py.init(["mpxlite-smoke", "-skip_petscrc"])
    from petsc4py import PETSc

    print(f"  petsc4py {petsc4py.__version__}, PETSc {'.'.join(map(str, PETSc.Sys.getVersion()))}")
    if not issubclass(PETSc.ScalarType, complex):
        raise RuntimeError(f"petsc4py scalars are {PETSc.ScalarType.__name__}, not complex")
    if not PETSc.Sys.hasExternalPackage("mumps"):
        raise RuntimeError("the PETSc of petsc4py has no MUMPS")
    engine_lib = (mpxlite.installation_home() / "engine" / "lib").resolve()
    with open("/proc/self/maps", encoding="utf-8") as maps:
        loaded = {Path(line.split()[-1]).resolve() for line in maps if "libpetsc" in line}
    print(f"  libpetsc {', '.join(str(p) for p in sorted(loaded))}")
    if not loaded or any(p.parent != engine_lib for p in loaded):
        raise RuntimeError(f"petsc4py did not load the libpetsc of {engine_lib}")

    # A complex tridiagonal system with a known solution.
    n = 200
    exact = np.exp(1j * np.linspace(0.0, 3.0, n))
    mat = PETSc.Mat().createAIJ((n, n), nnz=3, comm=PETSc.COMM_SELF)
    for i in range(n):
        mat.setValue(i, i, 4.0 + 1.0j)
        if i > 0:
            mat.setValue(i, i - 1, -1.0 + 0.5j)
        if i < n - 1:
            mat.setValue(i, i + 1, -1.0 - 0.25j)
    mat.assemble()
    rhs, sol = mat.createVecs()
    x = mat.createVecRight()
    x.setArray(exact)
    mat.mult(x, rhs)
    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    ksp.setOperators(mat)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()
    factor = pc.getFactorMatrix()
    factor.setMumpsIcntl(7, 5)  # METIS
    ksp.solve(rhs, sol)
    ordering = factor.getMumpsInfog(7)
    err = float(np.abs(sol.getArray() - exact).max())
    print(f"  MUMPS in this process, INFOG(7) = {ordering}, max error = {err:.1e} (tol 1e-12)")
    for obj in (ksp, mat, rhs, sol, x):
        obj.destroy()
    if ordering != 5:
        raise RuntimeError(f"MUMPS did not use the METIS ordering (INFOG(7) = {ordering})")
    if err > 1e-12:
        raise RuntimeError("the solution differs from the exact one")


def check_gmsh_module() -> None:
    """``import gmsh`` imports the Gmsh of the engine, which meshes a unit cube."""
    import mpxlite

    engine = mpxlite.installation_home() / "engine"
    import gmsh

    module = Path(gmsh.__file__)
    library = Path(gmsh.libpath)
    print(f"  module  {module}")
    print(f"  library {library}")
    if not module.samefile(engine / "gmsh" / "gmsh.py"):
        raise RuntimeError(f"import gmsh imported {module}, not the module of {engine}")
    if not library.samefile(engine / "lib" / GMSH_LIBRARY):
        raise RuntimeError(f"the Gmsh module loaded {library}, not the library of {engine}/lib")
    if gmsh.__version__ != EXPECTED_GMSH:
        raise RuntimeError(f"Gmsh module {gmsh.__version__}, expected {EXPECTED_GMSH}")
    loaded = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/libgmsh" in line
    }
    if not loaded or any(not Path(p).samefile(engine / "lib" / GMSH_LIBRARY) for p in loaded):
        raise RuntimeError(f"Gmsh libraries loaded by this process: {sorted(loaded)}")
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        built = gmsh.option.getString("General.Version")
        gmsh.model.add("cube")
        cube = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        volume = gmsh.model.occ.getMass(3, cube)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.25)
        gmsh.model.mesh.generate(3)
        nodes = gmsh.model.mesh.getNodes()[0].size
    finally:
        gmsh.finalize()
    print(f"  Gmsh {built}: unit cube (OpenCASCADE), volume {volume:.12g}, {nodes} mesh nodes")
    if built != EXPECTED_GMSH:
        raise RuntimeError(f"Gmsh library {built}, expected {EXPECTED_GMSH}")
    if abs(volume - 1.0) > 1e-9:
        raise RuntimeError(f"the volume of the unit cube is {volume}")
    if nodes == 0:
        raise RuntimeError("the mesh of the unit cube has no node")


def check_gmsh_command() -> None:
    """The ``gmsh`` on ``PATH`` is the program of the engine, which starts without a display."""
    import mpxlite

    engine = mpxlite.installation_home() / "engine"
    found = shutil.which("gmsh")
    print(f"  gmsh    {found}")
    if found is None or not Path(found).samefile(engine / "bin" / "gmsh"):
        raise RuntimeError(f"the gmsh on PATH is {found}, not {engine / 'bin' / 'gmsh'}")
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}

    def gmsh(*args: str) -> str:
        done = subprocess.run(
            [found, *args], env=env, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, timeout=60, check=False,
        )  # fmt: skip
        if done.returncode != 0:
            raise RuntimeError(f"gmsh {' '.join(args)}: exit {done.returncode}: {done.stderr}")
        return done.stdout

    version = gmsh("--version").strip()
    if version != EXPECTED_GMSH:
        raise RuntimeError(f"gmsh --version printed {version!r}, expected {EXPECTED_GMSH}")
    info = dict(
        (key.strip(), value.strip())
        for key, _, value in (line.partition(":") for line in gmsh("-info").splitlines())
    )
    if "Fltk" not in info.get("Build options", "").split():
        raise RuntimeError(f"gmsh -info: no graphical interface (Fltk): {info}")
    print(f"  gmsh --version {version}; gmsh -info: FLTK {info.get('FLTK version', '?')}")


def check_gmsh_job(work: Path) -> None:
    """The geometry job of the installation accepts a two-layer box."""
    import gmsh

    import mpxlite

    source = work / "box.xao"
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box")
        lower = gmsh.model.occ.addBox(0, 0, 0, 2, 1, 0.5)
        upper = gmsh.model.occ.addBox(0, 0, 0.5, 2, 1, 0.5)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [lower], name="lower")
        gmsh.model.addPhysicalGroup(3, [upper], name="upper")
        gmsh.write(str(source))
    finally:
        gmsh.finalize()
    result = mpxlite.run_gmsh_job(
        "periodic_cell_geometry",
        inputs={"geometry": source},
        parameters={"scale": 1.0},
        runs_dir=work / "jobs",
        timeout=300,
    )
    cell = result.report["cell"]
    print(f"  job     {result.run_dir} ({result.elapsed_s:.2f} s)")
    print(f"  lattice {cell['lattice']}, height {cell['height']}")
    if abs(cell["height"] - 1.0) > 1e-9 or abs(cell["sides"][0] - 2.0) > 1e-9:
        raise RuntimeError(f"wrong prism: {cell}")
    names = sorted(part["name"] for part in result.report["parts"])
    if names != ["lower", "upper"]:
        raise RuntimeError(f"parts {names}, expected lower and upper")
    if not result.outputs["geometry"].is_file():
        raise RuntimeError("the job wrote no conformal geometry")


def check_section_job(work: Path) -> None:
    """The section job of the installation meshes a structured band with a free layer."""
    import mpxlite

    result = mpxlite.run_gmsh_job(
        "layered_section_mesh",
        parameters={
            "columns": {"x": [0.0, 1.0, 2.0], "cells": [{"name": "a", "cells": 4},
                                                       {"name": "b", "cells": 4}]},
            "core": {"y": [0.0, 0.25], "rows": 1},
            "above": {"name": "layer", "height": 1.0, "mesh": "free"},
            "sizes": {"floor": [{"size": 0.25}], "base": [{"size": 0.25}], "rate": 0.5,
                      "min": 0.25, "max": 0.5},
        },
        runs_dir=work / "jobs",
        timeout=300,
    )  # fmt: skip
    mesh = result.report["mesh"]
    print(f"  job     {result.run_dir} ({result.elapsed_s:.2f} s)")
    print(f"  {mesh['nodes']} nodes, {mesh['surface_elements']} surface elements")
    if sorted(result.report["physical"]) != ["a", "b", "layer"]:
        raise RuntimeError(f"groups {sorted(result.report['physical'])}, expected a, b, layer")
    if mesh["surface_elements"] <= 16 or not result.outputs["mesh"].is_file():
        raise RuntimeError("the job wrote no mesh of the section")


def main(argv: list[str] | None = None) -> int:
    """Run the checks; returns the exit status."""
    home = os.environ.get("MPXLITE_HOME", "")
    ap = argparse.ArgumentParser(description="mpxlite installation smoke test")
    ap.add_argument(
        "--data",
        type=Path,
        default=Path(home, "tests", "integration", "data") if home else None,
        help="the rod case directory (default: $MPXLITE_HOME/tests/integration/data)",
    )
    ap.add_argument("--keep", action="store_true", help="keep the run directories")
    args = ap.parse_args(argv)

    import mpxlite

    print(f"mpxlite {mpxlite.__version__}, python {sys.version.split()[0]}, home {home or '-'}")
    if args.data is None or not (args.data / "rod.msh").is_file():
        print(f"FAIL: no rod case in {args.data} (set MPXLITE_HOME or --data)")
        return 1
    work = Path(tempfile.mkdtemp(prefix="mpxlite-smoke-"))
    failed: list[str] = []
    t_all = time.perf_counter()
    try:
        rod = Rod(args.data, work / "runs")
        checks: list[tuple[str, Callable[[], None]]] = [
            ("engine identity (getdp --info)", check_engine),
            ("rod Poisson problem, serial, vs exact", lambda: check_exact(rod, "serial", 1)),
            ("rod Poisson problem, 2 MPI ranks, vs exact", lambda: check_exact(rod, "np2", 2)),
            ("MUMPS uses the METIS ordering", lambda: check_metis(rod)),
            ("petsc4py on the PETSc of the engine", check_petsc4py),
            ("import gmsh: the Gmsh of the engine meshes a unit cube", check_gmsh_module),
            ("the gmsh command of the engine (gmsh --version, gmsh -info)", check_gmsh_command),
            ("Gmsh job: periodic_cell_geometry on a box", lambda: check_gmsh_job(work)),
            ("Gmsh job: layered_section_mesh on a band", lambda: check_section_job(work)),
        ]
        for name, fn in checks:
            print(f"[check] {name}", flush=True)
            try:
                fn()
                print("  ok")
            except Exception:
                traceback.print_exc(file=sys.stdout)
                failed.append(name)
                print("  FAILED")
    finally:
        if args.keep:
            print(f"run directories kept in {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
    print(f"smoke test: {time.perf_counter() - t_all:.1f} s")
    if failed:
        print("FAIL: " + ", ".join(failed))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
