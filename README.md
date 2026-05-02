# mpxlite

> **mpx**: multi-physics; **lite**: lightweight.

`mpxlite` is a small Python 3 platform that drives the
[GetDP](http://getdp.info) finite-element solver through a clean,
reproducible workflow. It is **MPI-aware**: the canonical
`solver.run()` entry point is a single subprocess invocation, which
maps naturally to a single `mpirun -np N getdp ...` launch when the
underlying GetDP build is compiled against PETSc + MUMPS + MPI.

`mpxlite` is **physics-agnostic**: it provides only the generic Python
machinery — manifest assembly, scalar container, result wrapper,
output parsers. Concrete physics modules (e.g. `metalab.Meta3D` for
3D periodic metasurfaces) plug in by subclassing `SolverLite` and
bundling their own `.pro` resources.

## Status

Early development. API in flux until v0.2.

## Quick start

Install in editable mode for development:

```bash
git clone <this-repo> mpxlite
cd mpxlite
pip install -e .[test]
nox -s tests       # 56 tests, ~0.5 s
```

For real getdp invocations, point `MPXLITE_GETDP` at the wrapper
script in `tools/`:

```bash
export MPXLITE_GETDP=/path/to/mpxlite/tools/getdp_runner.sh
export MPXLITE_GETDP_BINARY=/opt/getdp/bin/getdp
export MPXLITE_PETSC_LIB_DIR=/opt/petsc/lib   # if needed
export MPXLITE_GETDP_NP=8                      # 8-rank MPI launch
```

After this, every `solver.run(postop=...)` triggers a single

```text
mpirun -np 8 getdp mpxlite.pro -msh ... -pre <res> -cal -pos <op>
```

that distributes the assembled system across ranks via PETSc + MUMPS.

## Environment variables

| Variable                 | Purpose                                                                                          |
|--------------------------|---------------------------------------------------------------------------------------------------|
| `MPXLITE_GETDP`          | Absolute path to the executable invoked by `SolverLite`. May be the `getdp` binary itself, or `tools/getdp_runner.sh` (recommended for MPI / OpenMP workflows). |
| `MPXLITE_GETDP_BINARY`   | (Read by `getdp_runner.sh`.) Absolute path to the actual `getdp` executable. Falls back to `command -v getdp` if unset.                                          |
| `MPXLITE_PETSC_LIB_DIR`  | (Read by `getdp_runner.sh`.) Optional. Prepended to `LD_LIBRARY_PATH`; needed when `getdp` is dynamically linked against a locally-built PETSc outside the system loader path. |
| `MPXLITE_GETDP_NP`       | (Read by `getdp_runner.sh`.) Number of MPI ranks per `solver.run()` (default `1`, no `mpirun`). When `> 1`, the wrapper invokes `mpirun -np $MPXLITE_GETDP_NP …` and requires `mpirun` on `PATH`. |
| `OMP_NUM_THREADS`        | Honoured by OpenBLAS / OpenMP-enabled MUMPS if the underlying `getdp` build links against them. Set to `1` when `MPXLITE_GETDP_NP > 1` to avoid oversubscription unless deliberately running a hybrid MPI + OpenMP layout. |

## Quick links

* `tools/getdp_runner.sh` — MPI / `LD_LIBRARY_PATH` wrapper.
* `src/mpxlite/` — generic Python encapsulation.
* `tests/` — unit tests with `subprocess.run` mocked via
  `pytest-mock`; no real `getdp` required.

## License

MIT — see `LICENSE`.
