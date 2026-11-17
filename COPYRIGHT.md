<!-- Copyright (c) 2025-2026 MPhysX -->
<!-- SPDX-License-Identifier: MIT -->

# Copyright

`mpxlite` is © 2025-2026 MPhysX.

Everything in this repository is distributed under the MIT License — see
[`LICENSE`](LICENSE) — **except the content of `tools/getdp/patches/` and of `tools/gmsh/`**:

* [`tools/getdp/patches/getdp-3.5.0-mpx3.patch`](tools/getdp/patches/getdp-3.5.0-mpx3.patch)
  is a patch of GetDP, derived from GetDP 3.5.0 (Copyright (C) 1997-2022 P. Dular and
  C. Geuzaine, University of Liège), with modifications Copyright (c) 2026 MPhysX. It is
  licensed under the GNU General Public License, version 2 or (at your option) any later
  version (SPDX `GPL-2.0-or-later`), like GetDP itself, and is **not** covered by the MIT
  License of `mpxlite`;
* `tools/getdp/patches/COPYING` is the text of the GNU GPL version 2 that accompanies the
  patch (© Free Software Foundation, distributed under its own verbatim-copying terms);
* the Gmsh programs of `mpxlite` in `tools/gmsh/`, Copyright (c) 2026 MPhysX, are licensed
  under the GNU General Public License, version 2 or (at your option) any later version (SPDX
  `GPL-2.0-or-later`), like Gmsh, whose Python module they import; they are **not** covered by
  the MIT License of `mpxlite`: the toolkit of the
  Gmsh jobs [`tools/gmsh/mpxlite_gmsh/`](tools/gmsh/mpxlite_gmsh), the Gmsh jobs
  [`tools/gmsh/jobs/`](tools/gmsh/jobs) and the tests of the toolkit `tools/gmsh/tests/`.
  `tools/gmsh/COPYING` is the GNU GPL version 2 text that accompanies them.

The patch, the Gmsh programs and the GPL texts beside them are the only non-MIT content of the
repository; their notices are `tools/getdp/patches/README.md` and `tools/gmsh/README.md`. The
Python distributions (wheel, sdist) of `mpxlite` contain the MIT package only (with its MIT
modules `mpxlite.gmshjob`, which starts the Gmsh jobs as separate programs, and
`mpxlite.cells`, pure NumPy): `tools/` (the engine build script, the bundle tools, the CI
scripts, the patch and the Gmsh programs) is not part of them. The patch, the Gmsh programs
and the build script are shipped in the source repository
(<https://github.com/MPhysXDev/mpxlite>) and in every release bundle (directory `source/`; the
Gmsh programs also in `gmsh/`, installed in `engine/gmsh/`).

## Gmsh

`mpxlite` runs [Gmsh](https://gmsh.info) (© 1997-2025 C. Geuzaine and J.-F. Remacle,
GPL-2.0-or-later with the exceptions stated in its `LICENSE.txt`) as a separate program:
`mpxlite.run_gmsh_job` (MIT) starts a Gmsh job (files and a JSON request in, files and a JSON
report out); the jobs import the `gmsh` Python module. No
module of the `mpxlite` package imports Gmsh. The Python environment of an installation
imports the Gmsh module of the engine (`import gmsh`) for the scripts of its users. Gmsh
is not in the repository; it is part of the engine, and each release bundle distributes a
**GPL binary** of it (below). Its authors ask that work using Gmsh cite: C. Geuzaine and J.-F. Remacle, "Gmsh: a three-dimensional finite element mesh generator with built-in pre- and post-processing facilities", International Journal for Numerical Methods in Engineering 79(11), 1309-1331, 2009.

## The engine and the release bundles

`mpxlite` runs its engine — GetDP 3.5.0 with the mpx3 patch, on complex PETSc 3.26.0 with
MUMPS, ScaLAPACK and METIS, and Gmsh 4.15.2 with its graphical interface (FLTK 1.3.11) on
OpenCASCADE 7.8.1 — as separate programs started
through subprocesses; it does not link to it. The repository contains the script that builds the engine
(`tools/engine/build_engine.sh`, MIT) but no engine code besides the patch. Each release
bundle (`mpxlite-<version>-ubuntu<release>-x86_64.tar.gz`) distributes, besides `mpxlite`:

| Component | Licence |
|---|---|
| `engine/bin/getdp`: GetDP 3.5.0 ([getdp.info](https://getdp.info), © 1997-2026 P. Dular and C. Geuzaine), modified by the mpx3 patch | distributed under **GPL-3.0-or-later** (GetDP is GPL-2.0-or-later; the binary is linked with the Apache-2.0 METIS) |
| the mpx3 patch (`source/tools/getdp/patches/`) | GPL-2.0-or-later (derived from GetDP) |
| Gmsh 4.15.2, unmodified (© 1997-2025 C. Geuzaine and J.-F. Remacle): the library `engine/lib/libgmsh.so.4.15`, the program `engine/bin/gmsh` and the Python module `engine/gmsh/gmsh.py`, built from the source release with its graphical interface | a **GPL binary**: GPL-2.0-or-later with the exception of its `LICENSE.txt`; the contributed code compiled in keeps its own licences (`THIRD_PARTY/gmsh/`) |
| [FLTK](https://www.fltk.org) 1.3.11, unmodified (© Bill Spitzak and others), linked statically into the Gmsh library: the graphical interface of Gmsh is based in part on the work of the FLTK project | LGPL-2.0 with the FLTK exceptions (static linking permitted; `THIRD_PARTY/fltk/COPYING`) |
| [OpenCASCADE Technology](https://dev.opencascade.org) 7.8.1, unmodified (© Open CASCADE SAS; `engine/lib/libTK*`) | LGPL-2.1 with the Open CASCADE exception |
| [PETSc](https://petsc.org/) 3.26.0 (© Argonne National Laboratory and others) | BSD-2-Clause |
| petsc4py 3.26.0, the Python bindings of PETSc (from the PETSc sources; © Lisandro Dalcin), shipped as a wheel | BSD-2-Clause |
| the Gmsh programs of mpxlite: the toolkit `mpxlite_gmsh/`, the jobs and the toolkit's tests (`gmsh/`, installed in `engine/gmsh/`; `source/tools/gmsh/`; © 2026 MPhysX) | GPL-2.0-or-later |
| [MUMPS](https://mumps-solver.org) 5.9.1 (© CERFACS, IRIT-ENSEEIHT, INPT(ENSEEIHT)-IRIT), linked into libpetsc | CeCILL-C (PORD: public domain; AMD- and LAPACK-derived routines: BSD-3-Clause) |
| [ScaLAPACK](https://www.netlib.org/scalapack/) 2.2.3 (© the University of Tennessee and others) | BSD-3-Clause |
| [METIS](https://github.com/KarypisLab/METIS) 5.1.0 (© the Regents of the University of Minnesota) | Apache-2.0 |

The getdp binary is a modified version of GetDP: upstream GetDP 3.5.0 plus the mpx3 patch,
shipped in the bundle's `source/` with the build script. No other third-party code is modified,
and no source code of unmodified components is included: the bundle's `THIRD_PARTY/README.md`
gives each component's upstream address and exact commit or source-archive checksum (for Gmsh,
FLTK and OpenCASCADE: the release archive, its sha256, and the tag and commit whose tree it is;
a copy of the source archives of the components used unmodified is kept at
<https://github.com/MPhysXDev/mpxlite-engine-sources>, and GetDP with its history and the mpx3
patch at <https://github.com/MPhysXDev/fem-mpx>, branch `mpx3`), a dated statement of the
modification and what is compiled into the engine (Gmsh with each piece of contributed code and
its licence; its Untangle code, free for non-commercial use only, is not compiled in);
`THIRD_PARTY/` holds every licence and notice file, the GPL-3.0 text and `mpxlite`'s own
licence. Because METIS is under the Apache License 2.0 (compatible with the GPL version 3, not
with version 2), the distributed GetDP binary is under the GNU GPL version 3 or later; MUMPS
stays under CeCILL-C, PETSc and ScaLAPACK under their BSD licences. ParMETIS, PT-Scotch /
Scotch, STRUMPACK and SuperLU_DIST are not part of the engine.

Not distributed by the bundles: what the installer installs on the user's machine, each under
its own licence — [Open MPI](https://www.open-mpi.org) and
[OpenBLAS](https://www.openblas.net) (BSD-3-Clause), the GCC runtime libraries
(GPL-3.0-or-later with the GCC Runtime Library Exception; the OpenMP runtime of the Gmsh
library among them), the X11 client libraries, fontconfig, FreeType, OpenGL, GLU,
libjpeg-turbo, libpng and zlib that the Gmsh library loads for its graphical interface
(permissive licences) and Python (PSF-2.0) from Ubuntu; NumPy (BSD-3-Clause) and pytest with
its plug-ins (MIT) from PyPI.
