<!-- Copyright (c) 2025-2026 MPhysX -->
<!-- SPDX-License-Identifier: MIT -->

# Changelog

All notable changes to `mpxlite`. Versions follow [semantic versioning](https://semver.org/).
mpxlite serves the plug-in packages of MPhysX, which are updated with it. Release 1.0.0 broke
plug-ins written for 0.3.3, hence the new major version; 1.1.0 only adds. Before 0.3.3, a minor
bump (0.x.0) could break the API.

## 1.1.0 — 2026-11-17

Gmsh in the engine, the Gmsh jobs, and the `gmsh` command. A plug-in contains no GPL code and
does not import the `gmsh` Python module: what it needs Gmsh for is done by the Gmsh jobs of the
installation, separate programs defined by a kind of geometry (files and a parameter document
in, files and a report out). The users' scripts import Gmsh themselves, in the Python of the
installation, and open geometries and meshes in the graphical interface of Gmsh with the `gmsh`
command. Additions only: what plug-ins use of 1.0 is unchanged. Upgrading replaces the engine
(identifier `getdp-3.5.0-mpx3+petsc-3.26.0+gmsh-4.15.2+occt-7.8.1`) and installs the Gmsh jobs
beside its Gmsh.

### Added

- **Gmsh jobs**: `mpxlite.run_gmsh_job(job, inputs=..., parameters=..., runs_dir=..., run_id=None,
  timeout=None, memory_max=None, env=None)` runs a Gmsh job of the installation
  (`engine/gmsh/jobs/<job>.py`, or the directory named by `MPXLITE_GMSH_JOBS`) as a separate
  program in its own run folder (`request.json`, `report.json`, `job.log`), with one thread, an
  optional time limit and memory cap; it returns a `GmshJobResult` (report, outputs by name, log,
  run folder, elapsed time, Gmsh messages) or raises `GmshJobError` (`kind` input / rule / gmsh /
  timeout / memory / crash, `code`, `message`, `data`); `find_gmsh_jobs`, `GMSH_JOBS_ENV`. A job
  is run once by the driving script: `run_gmsh_job` refuses to run in a process that is one of
  several MPI ranks (`OMPI_COMM_WORLD_SIZE`, `PMI_SIZE`, `MV2_COMM_WORLD_SIZE` or
  `SLURM_STEP_NUM_TASKS` greater than 1; code `job.mpi_ranks`). The jobs
  `periodic_cell_geometry` (a named periodic prism cell made conformal, checked, scaled and
  measured), `periodic_cell_mesh` (layers, sizes from named controls, symmetry detection,
  asymmetric unit, symmetric periodic MSH 2.2 mesh) and `layered_section_mesh` (the structured
  and graded MSH 2.2 mesh of a layered two-dimensional section) and their toolkit `mpxlite_gmsh`
  are GPL-2.0-or-later programs in `tools/gmsh/`, outside the Python package; the bundle ships
  them in `gmsh/` and `source/tools/gmsh/`, and the installer puts them in `engine/gmsh/` beside
  the Gmsh module of the engine.
- **`mpxlite.cells`**: the pure-NumPy tools of periodic cells (lattices and plane / space groups,
  symmetry detection on a tagged mesh, asymmetric units, replication, tilings, periodic and
  topology checks, MSH 2.2 writing with `$Periodic`, JSON forms of the symmetry objects).
- **Gmsh 4.15.2 in the engine, with its graphical interface.** `tools/engine/build_engine.sh`
  builds, from sha256-pinned source archives (with a copy at
  <https://github.com/MPhysXDev/mpxlite-engine-sources>), OpenCASCADE 7.8.1 (modules
  FoundationClasses, ModelingData, ModelingAlgorithms and DataExchange only), FLTK 1.3.11 (static,
  with OpenGL and Xft, on the system's libjpeg, libpng and zlib) and Gmsh 4.15.2: the shared
  library with FLTK linked in, its Python module and the `gmsh` executable, with the contributed
  code it carries, without Untangle (its HLBFGS part is free for non-commercial use only), the MPEG
  encoder, PETSc, MPI, MED, CGNS, GMP or Mmg. The meshes are those of the `gmsh` package of PyPI.
  The bundle ships `engine/bin/gmsh`, the library in `engine/lib` with the OpenCASCADE libraries it
  loads, and the module in `engine/gmsh/`; `THIRD_PARTY/` and `VERSIONS.tsv` record Gmsh (a GPL
  binary, with each piece of contributed code compiled in and its licence), FLTK (LGPL-2.0 with the
  FLTK exceptions) and OpenCASCADE (LGPL-2.1 with the Open CASCADE exception) with the URL, tag,
  commit and sha256 of their source archives. Nothing of Gmsh is downloaded by the installer.
- **`import gmsh`** in the Python of an installation imports the Gmsh of the engine (a `.pth`
  file of the venv names `engine/gmsh`), and the **`gmsh` command** of the engine is on `PATH`
  after `source activate.sh`. The library loads the X11, OpenGL and image libraries of the
  system but opens no display unless the graphical interface is started: without a display,
  `import gmsh`, batch `gmsh` runs, the jobs and getdp work. The installer's run-time packages
  gain `libx11-6 libxext6 libxft2 libxrender1 libfontconfig1 libgl1 libglu1-mesa libjpeg-turbo8
  libpng16-16t64 zlib1g` and `libgomp1`; the build prerequisites gain `libfreetype-dev
  libx11-dev libxext-dev libxft-dev libxrender-dev libfontconfig-dev libgl-dev libegl-dev
  libglu1-mesa-dev libjpeg-dev libpng-dev zlib1g-dev`. `mpxlite-smoke` checks that `import gmsh`
  imports the Gmsh of the engine, which meshes a unit cube, that the `gmsh` on `PATH` is the
  engine's (`gmsh --version`, `gmsh -info` with `Fltk`, without a display), and runs the jobs
  `periodic_cell_geometry` and `layered_section_mesh`.
- **Incremental engine build.** The engine is built in six parts (petsc; getdp and petsc4py on
  petsc; occt; fltk; gmsh on occt and fltk); each records what it was built from (pinned
  sources, options, a recipe number, the patch checksum for getdp, the Python version for
  petsc4py, the keys of the parts it depends on) and is skipped when that matches, rebuilt with
  its dependents otherwise; `--force` rebuilds everything; `--print-key PART`,
  `--print-inputs PART` and `--print-dirs PART` print the key, the inputs and the directories of
  a part. The CI caches the engine part by part under these keys
  (`.github/actions/engine-cache-restore`, `engine-cache-save`).
- Top-level names of the launch variables: `mpxlite.GETDP_NP_ENV`, `GETDP_THREADS_ENV`,
  `GETDP_CPUS_ENV`, `GETDP_LAUNCHER_ENV` (`mpxlite.launch.ENV_*` are unchanged).
- `activate.sh` puts every `plugins/<name>/bin` of the installation on `PATH`, after mpxlite's own
  commands. A plug-in writes nothing outside `plugins/<name>/` (its packages in `venv/` apart);
  its commands go in `plugins/<name>/bin/`.
- Tests: `tests/unit/test_gmshjob.py` (the runner with stand-in jobs, no Gmsh),
  `tests/integration/test_gmsh_job_geometry.py`, `test_gmsh_job_mesh.py`,
  `test_gmsh_job_section.py` and `test_gmsh_job_runner.py` (the jobs on real Gmsh, the tests as
  user scripts with `import gmsh`), `tools/gmsh/tests/` (the toolkit).

### Removed

- The Sphinx manual (`docs/`), its build (`nox -s docs`, the extra `docs`, the CI job) and the
  pointers to it. The repository keeps a short `README.md`, `LICENSE`, `COPYRIGHT.md` and the
  licence and notice files.

### Fixed

- `install.sh` printed "Authorization required, but no authorization protocol specified" several
  times (hwloc probing the display when the petsc4py check started Open MPI): every program the
  installer starts now runs with `HWLOC_COMPONENTS=-gl`.
- An installation copied or moved to another directory (`cp -a ~/mpxlite ...`) could not be
  upgraded, its venv naming the old directory: `install.sh` now refuses it with a message saying
  to install afresh into the new location.
- A bundle run over an installation of the same version kept an mpxlite package of another build
  of that version (pip does not reinstall a same-numbered package, and the check compared the
  version only): `install.sh` now checks the package file for file against the wheel of the
  bundle, and a repair replaces a package that differs.
- `README.md` and `README-INSTALL.md` omitted petsc4py (BSD-2-Clause), which the bundle ships,
  from their licence tables.

## 1.0.1 — 2026-11-15

A corrective release of the engine. No change of the Python interface. The engine patch is
now `mpx3`: `getdp --version` prints `3.5.0-mpx3-git-864875cd`, and the engine identifier of
an installation is `getdp-3.5.0-mpx3+petsc-3.26.0`. Upgrading replaces the engine.

### Fixed (engine)

- **Wrong Gauss rules on quadrangles** (a defect of GetDP 3.5.0). The tabulated 3-point and
  7-point rules held wrong ordinates (±0.840896 for ±1/√2, and ±√(3/5) for ±1/√3 in the last
  four points): both integrated exactly only polynomials of degree 1 in the second variable,
  where they must reach degree 2 and 5, without any message. Results computed with
  `NumberOfPoints 3` or `7` on `Quadrangle` with an earlier engine are to be recomputed.
- **Integral post-quantities accept `Type GaussLegendre`** (a limitation of GetDP 3.5.0), as
  the assembly does.

### Added

- `tests/integration/test_engine_quadrature.py`: one engine run integrates a full polynomial
  of the degree of each tabulated rule of lines, triangles, quadrangles, tetrahedra and
  hexahedra, and of three Gauss-Legendre rules in an integral post-quantity, against the exact
  integrals. On the engine of 1.0.0 it fails on the two quadrangle rules. Prisms and pyramids
  are not covered.
- Manual: the degree of exactness of every tabulated rule (the 16-point rule of the
  tetrahedron is exact to degree 4 only); the second-order nodal space on quadrangles needs
  `BF_Node_2F`; a plug-in records the version string of the engine and never tests it.

## 1.0.0 — 2026-11-15

mpxlite gets one interface for plug-ins and one rule for its engine, and ships petsc4py bound to
the PETSc of the engine. This release breaks plug-ins written for 0.3.3: what they took from
mpxlite's internals has a public name now and the internal one is gone, and a `getdp` on `PATH`
is never used. It is therefore a new major version, as the compatibility promise of 0.3.3
requires; from 1.0.0 on that promise covers the interface for plug-ins stated in the manual.

### Changed (breaking)

- **One engine, named in one place.** A run executes the program named by
  `MPXLITE_GETDP_BINARY` or, when it is unset, `engine/bin/getdp` of the installation
  (`MPXLITE_HOME`, else the installation the package is in, else `~/mpxlite`). `PATH` is never
  searched for a `getdp`: the fallback of the Python package (`find_getdp`), of `SolverLite`
  (a bare `getdp` when the wrapper was unusable) and of `getdp_runner.sh` (`command -v getdp`)
  are removed. Each run decides its engine before its directory is created and passes it to
  the driver in `MPXLITE_GETDP_BINARY`; with no engine it raises `FileNotFoundError`.
- `mpxlite.find_getdp()` is replaced by `mpxlite.find_engine(env=None)`, which applies that
  rule to the environment of a run.
- **One interface for plug-ins**, stated in the manual (chapter "The interface for plug-ins"):
  what is listed there is what a plug-in may use, and nothing else is. The private names
  plug-ins had come to use are replaced by public ones and no longer exist:
  `SolverLite._case_file` → `SolverLite.case_file` (it now refuses an extension that is not a
  case file), `SolverLite._effective_workflow` → `SolverLite.effective_workflow`,
  `SolverLite._effective_petsc_options` → `SolverLite.effective_petsc_options`,
  `mpxlite.base._CASE_EXTENSIONS` → `mpxlite.CASE_EXTENSIONS`. The two `effective_` methods are
  the ones a subclass may override: a run writes and uses what they return.
- `SolverLite.run` and `SolverLite.preprocess` take `read_solution`, and `RunRequest` has the
  same field, passed by the executors to every `run`: an override of `run` in a plug-in must
  accept it.
- The package requires Python 3.12 or newer (it was 3.11): the Python of Ubuntu 24.04 (3.12)
  and of 26.04 (3.14) are the ones mpxlite is installed and tested on, and the petsc4py shipped
  with the engine is built for them.
- A plug-in declares a minimum mpxlite version and no upper bound (`mpxlite>=X.Y.Z`); the manual
  section for plug-in installers said `<X+1`. One installation serves every plug-in of the
  machine, and an upper bound in one of them would block the upgrade another one needs.

### Added

- `mpxlite.engine_info(env=None)`: the build information of the engine, the `Key : value` lines
  of `getdp --info` as a mapping (asked once per program file).
- `mpxlite.installation_home(env=None)`: the installation whose engine is used.
- `read_solution` of `SolverLite.run` / `preprocess`: a solution file GetDP reads before the
  stages (`-res`), for a resolution that starts from the solution of another run. There was no
  public way to do this.
- The files of a run directory have public names and paths: the constants `LOG_NAME`,
  `WORKFLOW_NAME`, `MANIFEST_NAME`, `PETSC_OPTIONS_NAME`, `PRE_NAME`, `SOLUTION_NAME`, and the
  properties `SolverResult.workflow_path`, `manifest_path`, `petsc_options_path`, `pre_path`,
  `solution_path` (beside the existing `log_path`).
- `WorkflowData.read_pro(path)` and `WorkflowData.from_pro_string(text)`: the scalars of a file
  written by `write_pro`, for instance the `workflow.pro` of a run.
- Exported from the top-level package: `PetscOptionValue`, `Unbounded`, `resolve_hard_cap`.
- **petsc4py with the engine.** The engine build (`tools/engine/build_engine.sh`) also builds
  petsc4py 3.26.0, the Python bindings in the PETSc source tree, against the PETSc it has just
  built, for the Python of the Ubuntu release. The bundle ships that wheel, its extension module
  relocated to the `engine/lib` of the installation; `install.sh` installs it into `venv/`
  (never a petsc4py from an index, which would be bound to another PETSc) and verifies that it
  loads the `libpetsc` of the installation, on complex scalars, with MUMPS. `mpxlite-smoke`
  factorises a complex system in the Python process with MUMPS and the METIS ordering. Plug-ins
  that need PETSc in their own process use it; they never build one.

### Changed

- `install.sh`, on an upgrade, replaces the engine unless its files are those of the bundle, one
  for one (it compared the engine identifier only): the installation always holds the engine its
  petsc4py was compiled against.
- The engine build needs `python3-dev` and `python3-venv` besides its former packages.

### Other

- Engine build script: when the upstream GetDP repository cannot be reached, GetDP is cloned
  from <https://github.com/MPhysXDev/fem-mpx>, a copy of the GetDP history (with the `mpx3`
  patch as one commit on its branch `mpx3`); when the MUMPS site or the address of the
  OpenCASCADE, FLTK or Gmsh archive cannot be reached, the archive is taken from
  <https://github.com/MPhysXDev/mpxlite-engine-sources>, a copy of the source archives of the
  components used unmodified at the pinned revisions. The pinned commit and checksum are checked
  whatever the origin.
- CI: the nightly and release builds reuse the cached engine and rebuild it only when the build
  script or the GetDP patch changes.

## 0.3.3 — 2026-11-14

mpxlite becomes self-standing and self-installing: it owns its engine (the patched GetDP program
and PETSc with MUMPS, ScaLAPACK and METIS) and installs as a whole, once per machine, from one
release bundle per Ubuntu release. The engine is reduced and licence-clean. No change of the
Python API or behaviour.

### Added

- **Release bundles** `mpxlite-<version>-ubuntu<release>-x86_64.tar.gz` (+ `.sha256`) for
  Ubuntu 24.04 and 26.04 (x86-64 with AVX2, native or WSL2): the engine (relocatable), the
  mpxlite wheel, `install.sh`, the smoke test, mpxlite's tests, `THIRD_PARTY/` and `source/`.
- `install.sh [--prefix DIR] [--no-apt] [--no-smoke]`: installs into `$MPXLITE_HOME` or
  `~/mpxlite` (engine, Python environment with mpxlite and pytest, `activate.sh`, `VERSION`,
  `bin/mpxlite-smoke`, `plugins/`, tests, licences); one apt step for runtime packages only;
  upgrades an older installation in place (engine replaced only when it changed, mpxlite
  upgraded inside the existing environment, plug-ins and `plugins/` kept), verifies the same
  version, never downgrades; works as a normal user and without a terminal.
- `mpxlite-smoke`: engine identity, the rod Poisson problem in serial and on 2 MPI ranks
  against the exact solution, MUMPS with the METIS ordering; a few seconds.
- `tools/engine/build_engine.sh`, the engine build script (pinned PETSc commit and GetDP tag,
  MUMPS tarball checked against a pinned sha256, build-machine information neutralised,
  version and licence records), and `tools/bundle/make_bundle.sh`, the bundle generator (the
  engine collected from what getdp loads, RUNPATH relocated, removed components and machine
  paths refused, `THIRD_PARTY/`, `BUILDINFO`, `SHA256SUMS`).
- Documentation for plug-in authors: how a plug-in installer finds mpxlite (`MPXLITE_HOME`,
  `~/mpxlite`, the `VERSION` file), requires a minimum version, installs or upgrades it from a
  release bundle, installs itself into its environment and records itself under `plugins/`.
  mpxlite keeps backward compatibility for plug-ins within a major line.

### Changed

- The one supported engine is GetDP 3.5.0 + the mpx2 patch, configured with only the
  components mpxlite uses (kernel, MPI, PETSc: no Python interface, no Fortran contributions,
  no other optional component; the binary does not link `libpython`), on complex PETSc 3.26.0
  with **MUMPS as the only direct solver**, ScaLAPACK (needed by MUMPS) and **METIS as the only
  ordering library**, plus the distribution's Open MPI and OpenBLAS. PETSc is configured with
  exactly `--download-mumps --download-scalapack --download-metis`.
- Removed from the engine: **ParMETIS** — its licence (University of Minnesota) forbids
  redistribution and non-research use, which is the reason for this change — and PT-Scotch /
  Scotch, STRUMPACK and SuperLU_DIST, which no supported feature needs. `flex` / `bison` (needed
  by PT-Scotch only) are no longer build prerequisites, nor is cmake >= 3.21 (STRUMPACK).
- The sdist no longer contains `tools/` nor `.github/`: the engine tooling and the GPL patch it
  needs are in the repository and in every bundle's `source/`.

### Licences

- A bundle ships: the getdp binary, a modified version of GetDP 3.5.0 (upstream + the mpx2
  patch, shipped in `source/`), distributed under GPL-3.0-or-later (GetDP is GPL-2.0-or-later;
  linked with the Apache-2.0 METIS); PETSc (BSD-2-Clause), MUMPS (CeCILL-C; PORD public domain,
  AMD- and LAPACK-derived routines BSD-3-Clause), ScaLAPACK (BSD-3-Clause), METIS (Apache-2.0).
  No source of unmodified components; `THIRD_PARTY/` has the upstream revisions, a dated
  modification statement, every licence and notice file and the GPL-3.0 text. What the
  installer downloads (Open MPI, OpenBLAS, GCC runtimes, Python, NumPy, pytest) is listed as not
  distributed. Same statements in the README, `COPYRIGHT.md`, the installation, copyright and
  notice pages and the bundle's `README-INSTALL.md`.

### Tests, docs, build

- Installation guide rewritten: the bundle install first, then building the engine from source
  (the script is the reference; the recipe says what it does), then plug-in installers.
- CI: `ci.yml` (lint, unit tests on Python 3.12 and 3.14, docs with warnings as errors, and a
  bundle job: engine from the cache or built, bundle, install into a clean prefix, smoke test
  and integration tests of the installed copy), `nightly.yml` (engine from the cache, rebuilt only
  when its build script or the GetDP patch changes, and bundle for 24.04 and 26.04, install as a non-root user in fresh containers, full suites, upgrade
  paths) and `release.yml` (tag = version, the nightly workflow, then the GitHub release with
  both bundles, their checksums, the wheel and the sdist). Every job is a script of `tools/ci/`,
  also runnable in Docker (`tools/ci/in_docker.sh`); no secrets, read-only permissions except
  the release job; the test-reporter step is gone.
- `petsc_options` stays a generic pass-through to PETSc; the examples (README, concepts) use what
  the engine has (MUMPS ICNTL settings, PETSc's built-in serial LU, `ksp_view`), and the docs say
  that other PETSc packages can be selected only if the user's own PETSc build has them.
- Integration tests: the factorisation-package selection through `petsc_options` is tested with
  MUMPS and PETSc's built-in LU in serial, MUMPS on 2 ranks; new tests check that a MUMPS ICNTL
  reaches MUMPS (`mat_mumps_icntl_7 5`, the METIS ordering, reported as used by `-ksp_view`;
  serial and 2 ranks) and that a non-default package is
  honoured on 2 ranks (PETSc's built-in LU, which has no parallel factorisation, is refused
  instead of being replaced by MUMPS). The SuperLU_DIST / STRUMPACK cases are gone.
- A unit test of the session cleanup waits until its SIGTERM-ignoring child is set up (it
  could fail on a loaded machine).
- Prepared for the public repository: generic host-package wording (no named plug-in package
  in code, tests or docs; neutral names in the case-definition examples), `contact@mphysx.com`
  as the only contact address (authors and maintainers: MPhysX), clone instructions and project
  URLs pointing at <https://github.com/MPhysXDev/mpxlite>, an MIT header on every docs page and
  test package, a licence section in the README, and the licence notices now say that the
  patch and the GPL text beside it (`tools/getdp/patches/`) are the non-MIT content and live in
  the source repository only.

## 0.3.2 — 2026-11-07

### Fixed

- A `SIGTERM` / `SIGHUP` arriving while a run cleanup started by a timeout, a cancellation,
  `Ctrl-C` or post-exit leftovers is in progress no longer cuts it short: the signal is deferred
  until the SIGKILL escalation and the scope kill are done, then handled as usual (chained user
  handler, or unwind and re-delivery so that the process dies of it).
- Session cleanup can no longer signal an unrelated process that got the leader's pid after the
  session emptied: once the leader is reaped, the session is over (and never re-scanned) as soon
  as one scan finds it empty or `/proc/<leader pid>` exists again. Leftovers that still hold the
  session are terminated as before.
- A signal or `KeyboardInterrupt` between the subprocess start and its registration now
  terminates the run like any other interruption.
- Wrapper: a whitespace-only `MPXLITE_GETDP_LAUNCHER` exits `2` instead of silently running one
  process whatever `NP`; the launcher is no longer glob-expanded against the run directory
  (`-x OMP_*`, `h[0-9]` reach it verbatim).
- A relative `MPXLITE_GETDP` is resolved against the current directory (it was exec-ed relative to
  the run directory and failed after the run directory was created).
- `OMP_NUM_THREADS` given as an OpenMP comma list (`"4,2"`) counts its first entry instead of
  making every `LocalExecutor` request fail with `ValueError`.
- Only `Warning : ...` log lines are collected as GetDP warnings (no more `Warnings: ...`,
  `WarningCount ...`, `Warning,` banners).
- A failed capped run polls `systemctl` every 250 ms (was 50 ms) during its bounded 5 s settle:
  ~20 forks instead of ~100.
- `prctl` (parent-death signal) is called with its declared C signature.

### Removed

- `mpxlite.workflow.SerializableScalar` (alias of `Scalar` kept since 0.1): use `Scalar`.

### Tests, docs, build

- `nox -s tests` is hermetic at collection too: the integration modules' skip conditions
  (`hard_cap_available()`, `getdp --info`) are evaluated lazily, at test setup; a unit test
  collects the whole suite with process creation forbidden.
- Docs: the `MPXLITE_GETDP_THREADS=1` default also applies to CPU-pinned single-rank runs (every
  `LocalExecutor` run); `MPXLITE_GETDP_CPUS` is applied under SLURM/PBS when `NP == 1`;
  `LocalExecutor` is Linux-only (`taskset`); `SolverResult` lists `warnings` and `memory_max`.
- `ruff` pinned (`ruff==0.15.12`) in the lint and format sessions; `.gitignore` names the
  generated `mpxlite.pro` manifest.

## 0.3.1 — 2026-11-01

- Tests are hermetic against the developer's launch settings: an autouse fixture clears
  `MPXLITE_GETDP_NP`, `MPXLITE_GETDP_THREADS`, `MPXLITE_GETDP_CPUS`, `MPXLITE_GETDP_LAUNCHER` and
  pins `OMP_NUM_THREADS=1` for every test (a sourced dev `env.sh` no longer changes results).
- `tools/getdp/patches/COPYING`: full GNU GPL v2 text shipped with the GPL-2.0-or-later GetDP patch (GPL v2 §1).
- Python 3.10 dropped: `requires-python >= 3.11`; CI and nox test 3.11 and 3.12.

## 0.3.0 — 2026-11-01

Memory safety (**API change**), a public pre-processing entry point, and the supported GetDP
build.

### Changed (breaking)

- `LocalExecutor` now **requires** `memory_per_run`, a peak-memory estimate of one run in bytes
  (`mpxlite.UNBOUNDED` explicitly packs by CPU only). Its default `memory_bytes` budget is 80 %
  of `MemAvailable` at construction. Migration: add `memory_per_run=` to every
  `LocalExecutor(...)`.

### Added

- Hard per-run memory caps: where systemd (user manager) and cgroup v2 allow it, each run executes
  in its own transient `systemd-run --user --scope` with `MemoryMax`, `MemorySwapMax=0` and
  `OOMPolicy=stop`. A run that exceeds its cap fails alone with the new `GetDPMemoryError` (a
  `GetDPError`) instead of waking the global OOM killer; the other runs go on.
  `RunRequest(memory_max=...)` overrides the estimate per run, `SolverLite.run(memory_max=...)`
  caps a single run, `hard_cap=None|True|False` (executors) selects auto / required / off;
  `SerialExecutor` accepts `memory_per_run` / `hard_cap` too. `mpxlite.hard_cap_available()`
  probes the machine, `HardCapUnavailableError` is raised when a required cap cannot be enforced.
  `SolverResult.memory_max` and `RunOutcome.memory_max` record the cap a run was started with
  (`None`: uncapped). Inside a SLURM / PBS job (`SLURM_JOB_ID` / `PBS_JOBID`), or when the
  process's cgroup is not under `user@<uid>.service`, the automatic mode does not cap (a user
  scope would escape the job's memory limit); an explicit cap is honoured with a warning. The
  default budget is also bounded by the enclosing cgroup's `memory.max - memory.current`.
- Process lifetime: `SIGTERM` / `SIGHUP` received while runs are in progress in the main thread
  clean up like `Ctrl-C`, then the signal is re-delivered (user handlers are chained, ignored
  signals stay ignored); an `atexit` hook terminates runs still alive at interpreter exit; on
  Linux every run gets `PR_SET_PDEATHSIG=SIGKILL` and the wrapper starts local `mpirun` ranks
  through `setpriv --pdeathsig KILL`, so a SIGKILLed Python process no longer leaves runs
  behind; run scopes record their owner and `mpxlite.memory.reap_stale_scopes()` (run once per
  process before the first capped run) kills the scopes of dead owners only.
- `SolverLite.preprocess(*, workflow=None, run_id=None, launch=None, timeout=None, cancel=None,
  petsc_options=None, memory_max=None) -> SolverResult`: runs the resolution with `getdp -pre`
  only (no assembly, no solve) to check a case, with the same run-directory isolation, inputs,
  launch, cleanup, memory cap, warning collection and exceptions as `run()`.
- `tools/getdp/patches/getdp-3.5.0-mpx2.patch`: patch set of GetDP 3.5.0 that (1) preallocates
  the non-local (dense) rows of distributed matrices exactly, as GetDP does in serial (MPI
  assembly of a DtN case: 80–155 s → < 1 s), and (2) removes the fixed limit of 200 groups of
  entities per element (`NBR_MAX_GROUPS_IN_ELEMENT`, "Reached limit number of groups of
  entities"), which capped a Floquet DtN at ~99 orders (now validated up to 625 orders); outputs
  byte-identical below the old limit. The patched binary reports version `3.5.0-mpx2`. This
  patched build is now the only supported GetDP. It supersedes the
  interim `getdp-3.5.0-mpx1-mpi-nonlocal-prealloc.patch` (part (1) alone), which is removed.

### Fixed

- The MPI integration tests default to the binary under test (`find_getdp()`) instead of
  `/usr/bin/getdp` when `MPXLITE_TEST_MPI_GETDP` is unset.
- Processes a run leaves behind after a normal exit (in its session or its scope) are terminated
  with a warning; a failed scope is reset even when the wait is interrupted; an OOM kill is read
  from the scope's `memory.events`, so a scope that is already clean is not waited for.
- `LocalExecutor(cpus=...)` and `LaunchSpec(np=, threads=, cpus=)` accept numpy integers
  (bools are rejected); an invalid pinned spec fails its request alone instead of the batch.
  `LocalExecutor` falls back to `os.cpu_count()` where `os.sched_getaffinity` is missing.
- `timeout` is validated (negative, NaN or bool → `ValueError`).
- `HardCapUnavailableError` gives advice that fits the caller (`run(memory_max=...)` vs
  `hard_cap=True`); the probe names a cgroup v1 / hybrid hierarchy.
- Wrapper: `MPXLITE_PETSC_LIB_DIR` no longer adds an empty (= current directory) entry to an
  unset `LD_LIBRARY_PATH`.
- `nox -s tests` is hermetic (`-m "not integration"`); the integration suite has its own
  `nox -s integration` session.

### Licensing

- Copyright holder of `mpxlite` is MPhysX (MIT, unchanged licence); headers, `LICENSE`,
  `COPYRIGHT.md`, docs and package metadata made consistent (`License-Expression: MIT`).
- The GetDP patch `tools/getdp/patches/getdp-3.5.0-mpx2.patch` is GPL-2.0-or-later (derived from
  GetDP), the only non-MIT file; it carries its own notice (`tools/getdp/patches/README.md`) and
  is excluded from the Python distributions (wheel and sdist).

### Documentation

- Memory model (budget, hard caps), systemd / cgroup v2 requirements.
- Installation: the only supported stack is GetDP 3.5.0 + the mpx2 patch set on complex
  PETSc 3.26 (MUMPS, STRUMPACK, SuperLU_DIST, Open MPI), with the verified recipe (incl.
  `-DCMAKE_INSTALL_RPATH`). Any other `getdp` (distribution packages, other versions, unpatched
  or real-PETSc builds) is unsupported; `MPXLITE_GETDP_BINARY` can still point at one, outside
  the supported scope. Known GetDP 3.5 pitfalls (`-cpu` under `mpirun` deadlocks, serial `-petsc_prealloc`
  for high-order elements and for many DtN orders) and known limitations of the patch. The
  former real-PETSc recipe is removed.
- Process lifetime guarantees (what happens to runs when Python is interrupted, terminated or
  killed); memory caps inside scheduler jobs.

## 0.2.1 — 2026-10-31

### Added

- PETSc options: `SolverLite(petsc_options=...)` is a read-only template, `run(petsc_options=...)`
  merges over it (per-run wins, `False` removes an option). Keys are given without the leading
  dash; values are validated. A non-empty effective set is written to `<run_dir>/petsc.opts` and
  passed as `-options_file` before `-pre`. `RunRequest` gains `petsc_options`.
- GetDP warnings: the `Warning` lines of `getdp.log` (ANSI colours and `[rank N]` tags stripped,
  de-duplicated in order) are returned in `SolverResult.warnings` and logged once each on the
  `mpxlite` logger; failed runs expose them as `GetDPError.warnings`.

### Tests and documentation

- Integration tests use a configurable MPI binary (`MPXLITE_TEST_MPI_GETDP`) and check
  `petsc_options` (MUMPS / SuperLU_DIST / STRUMPACK, serial and MPI) and warnings.
- Linear solver options, GetDP warnings.

## 0.2.0 — 2026-10-31

Run isolation, parallel layout and executors (**API change** from 0.1).

### Changed (breaking)

- Every `run()` writes only into its own run directory `runs_dir/<run_id>/` (default
  `project_dir/runs`, never reused): the case directory is only read, so concurrent runs can
  share it. The manifest `Include`s inputs by absolute path.
- `run()` never mutates the solver: keyword-only per-run `workflow=` overrides are merged over a
  copy of the template; `postop` may be a list (`-pos a b c`, one solve).
- `SolverResult` records the run directory, `run_id`, log path, elapsed time, return code and
  command (`CompletedProcess` dropped).
- `LaunchSpec.env` may not set the launch variables; `LaunchSpec(launcher=...)` requires `np`;
  NP / THREADS values follow the wrapper's rule (`^[1-9][0-9]{0,8}$`, empty = unset).

### Added

- `LaunchSpec(np, threads, cpus, launcher, env)`: per-run parallel layout passed to the wrapper
  through the environment; one OpenMP thread per rank by default for multi-rank or pinned runs.
- Executors: `RunRequest` / `RunOutcome`, `SerialExecutor` and a FIFO CPU-block
  `LocalExecutor` (each run pinned to its own block of CPUs).
- `run(timeout=..., cancel=threading.Event)`; `GetDPTimeoutError`, `GetDPCancelledError`. On a
  timeout, a cancellation or `Ctrl-C` (also inside `LocalExecutor.submit`) every process of the
  run's session is killed, MPI ranks included (session-wide `/proc` scan, bounded
  SIGTERM → SIGKILL). The subprocess gets `/dev/null` as stdin. `GetDPError` carries the log
  tail, `log_path` and `run_dir`.
- Wrapper `getdp_runner.sh`: NP / THREADS validation, CPU pinning with `taskset -c`
  (`MPXLITE_GETDP_CPUS`, `--bind-to none` for local `mpirun`), explicit launcher used whatever NP.
- `WorkflowData`: numpy scalar coercion, finite / string validation, complex scalars serialised
  as `N[] = Complex[re, im];` (reference them as `N[]`); `Scalar` alias.
- Parsers: `parse_complex_vector_table`, uniform empty-file errors, layout guards (column
  layouts verified with GetDP 3.5.0 and 3.2.0).
- Real-GetDP integration suite (`integration` marker, committed 1-D rod case).

### Fixed

- No bare-`getdp` fallback for multi-rank, pinned or thread-controlled runs (they would silently
  lose MPI launching / pinning / thread setup).

## 0.1.0

Initial release of the host package: `SolverLite`, `WorkflowData`, `SolverResult`, output
parsers and the bundled `getdp_runner.sh` wrapper with cluster launcher auto-detection.
