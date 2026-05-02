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

## Quick links

* `tools/getdp_runner.sh` — wrapper script that handles
  `LD_LIBRARY_PATH` augmentation and optional `mpirun` invocation
  via `MPXLITE_GETDP_*` and `OMP_NUM_THREADS` environment variables.
* `src/mpxlite/` — generic Python encapsulation.
* `tests/` — unit tests with subprocess mocking (no real `getdp`
  required).

## License

MIT — see `LICENSE`.
