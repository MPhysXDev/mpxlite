<!-- Copyright (c) 2026 MPhysX -->
<!-- SPDX-License-Identifier: MIT -->

# Installing mpxlite (Ubuntu 24.04 / 26.04, also under WSL2)

This bundle installs mpxlite as a whole, once per machine: the **engine** (the
GetDP 3.5.0 program with the mpx3 patch, on complex PETSc 3.26.0 with MUMPS,
ScaLAPACK and METIS, and Gmsh 4.15.2 with its graphical interface on
OpenCASCADE 7.8.1), the mpxlite Python package, and the Python environment they
run in. Plug-in packages are later installed into that same environment
by their own installers.

There is one bundle per Ubuntu release, for example
`mpxlite-1.1.0-ubuntu24.04-x86_64.tar.gz` for Ubuntu 24.04 and
`mpxlite-1.1.0-ubuntu26.04-x86_64.tar.gz` for 26.04. Use the one of your
release (`grep VERSION_ID /etc/os-release` prints it). The commands below use
the 24.04 name; for 26.04, replace `24.04` by `26.04`.

You need an x86-64 CPU with AVX2 (Intel since 2013, AMD since 2015), about
250 MB of disk, and Internet access for apt and PyPI.

## 1. Windows only: Ubuntu under WSL2

In PowerShell, as administrator, then restart if asked and open "Ubuntu" from
the Start menu (it asks for a Linux user name and password):

    wsl --install -d Ubuntu-24.04

Work in the Linux home directory (`~`), not under `/mnt/c`: the Windows drives
are much slower from WSL. A file in your Windows Downloads folder is at
`/mnt/c/Users/<Windows user>/Downloads/`.

## 2. Install

Download the bundle and its checksum from
<https://github.com/MPhysXDev/mpxlite/releases> (or with `curl`), check, unpack
and install:

    cd ~
    curl -LO https://github.com/MPhysXDev/mpxlite/releases/download/v1.1.0/mpxlite-1.1.0-ubuntu24.04-x86_64.tar.gz
    curl -LO https://github.com/MPhysXDev/mpxlite/releases/download/v1.1.0/mpxlite-1.1.0-ubuntu24.04-x86_64.tar.gz.sha256
    sha256sum -c mpxlite-1.1.0-ubuntu24.04-x86_64.tar.gz.sha256
    tar xzf mpxlite-1.1.0-ubuntu24.04-x86_64.tar.gz
    cd mpxlite-1.1.0-ubuntu24.04-x86_64
    ./install.sh

`install.sh` checks the Ubuntu release, the CPU and AVX2, then installs the
missing runtime packages with one `sudo apt-get install` (it prints the command
first and asks for your password; Open MPI, OpenBLAS, the GCC Fortran and
OpenMP runtimes, the X11, OpenGL and image libraries of the graphical interface
of Gmsh, Python and its venv module). Everything else goes into
`~/mpxlite`: the engine (with Gmsh) and the Gmsh programs of mpxlite (the
toolkit and the jobs, in `engine/gmsh/`), a Python
environment with mpxlite, petsc4py (the Python bindings of the PETSc of the
engine, from the bundle) and pytest, in which `import gmsh` imports the Gmsh of
the engine (for your own scripts), `activate.sh` and the tests. It ends with the smoke test and prints `PASS`. It takes about a minute.

| Option | Effect |
|---|---|
| `--prefix DIR` | install into `DIR` instead of `~/mpxlite` (the environment variable `MPXLITE_HOME` does the same) |
| `--no-apt` | skip the apt step (the runtime packages are already installed, or installed by an administrator) |
| `--no-smoke` | skip the smoke test |

## 3. Use it

Each new terminal needs one command first:

    source ~/mpxlite/activate.sh

It activates the Python environment, points mpxlite to the engine and puts
the `getdp` and `gmsh` commands of the engine on `PATH`: on a desktop (or with
an X server under WSL2), `gmsh cell.xao` opens a geometry or a mesh in the
graphical interface of Gmsh, and a script can call `gmsh.fltk.run()`. Without a
display everything else works the same (batch `gmsh -3 model.geo`, `import
gmsh`, the Gmsh jobs). Check the installation at any time with:

    mpxlite-smoke

and run mpxlite's integration tests against it (about 30 seconds):

    cd ~/mpxlite
    python -m pytest tests/integration

Tests that need a systemd user session (memory caps) are skipped where there is
none, e.g. in a container.

## 4. Upgrade, plug-ins, uninstall

To upgrade, run `./install.sh` of a newer bundle: it upgrades `~/mpxlite` in
place and keeps everything else installed in its environment, including
plug-in packages. Running a bundle of the installed version verifies the
installation (the mpxlite package file for file against the bundle) and repairs
what differs; an older bundle never downgrades it. A copy of `~/mpxlite` made
elsewhere (`cp -a`, `mv`) cannot be upgraded: install afresh into the new
location instead.

Plug-in packages install themselves into `~/mpxlite/venv` and record
themselves under `~/mpxlite/plugins/`; their commands, in
`~/mpxlite/plugins/<name>/bin`, are on `PATH` after
`source ~/mpxlite/activate.sh`. See their own instructions.

To uninstall everything (mpxlite, the engine and every plug-in):

    rm -rf ~/mpxlite

The apt packages are standard Ubuntu runtime libraries; you can leave them.

## Licences

mpxlite is Copyright (c) 2025-2026 MPhysX, under the MIT licence
(`THIRD_PARTY/mpxlite/LICENSE`). It runs the engine as a separate program and
does not link to it.

What this bundle distributes, besides mpxlite:

| Component | Licence |
|---|---|
| `engine/bin/getdp`: GetDP 3.5.0, modified by the mpx3 patch | distributed under GPL-3.0-or-later (GetDP is GPL-2.0-or-later; the binary is linked with the Apache-2.0 METIS) |
| the mpx3 patch (`source/tools/getdp/patches/`) | GPL-2.0-or-later (derived from GetDP) |
| Gmsh 4.15.2, unmodified (© C. Geuzaine and J.-F. Remacle): the library `engine/lib/libgmsh.so.4.15`, the program `engine/bin/gmsh` and the Python module `engine/gmsh/gmsh.py` | a GPL binary: GPL-2.0-or-later with the exception of its `LICENSE.txt`; the contributed code compiled in keeps its own licences (`THIRD_PARTY/gmsh/`) |
| FLTK 1.3.11, unmodified, linked statically into the Gmsh library (its graphical interface is based in part on the work of the FLTK project, https://www.fltk.org) | LGPL-2.0 with the FLTK exceptions, static linking permitted (`THIRD_PARTY/fltk/`) |
| OpenCASCADE Technology 7.8.1, unmodified (`engine/lib/libTK*`) | LGPL-2.1 with the Open CASCADE exception |
| PETSc 3.26.0 (`engine/lib/libpetsc`) | BSD-2-Clause |
| petsc4py 3.26.0, the Python bindings of PETSc (`wheels/`) | BSD-2-Clause |
| the Gmsh programs of mpxlite: the toolkit `mpxlite_gmsh/`, the jobs `jobs/` and the toolkit's tests (`gmsh/`, installed in `engine/gmsh/`) | GPL-2.0-or-later (© 2026 MPhysX; `gmsh/README.md`, `gmsh/COPYING`) |
| MUMPS 5.9.1 (linked into libpetsc) | CeCILL-C (PORD ordering: public domain; AMD- and LAPACK-derived routines: BSD-3-Clause) |
| ScaLAPACK 2.2.3 (`engine/lib/libscalapack`) | BSD-3-Clause |
| METIS 5.1.0 (`engine/lib/libmetis`) | Apache-2.0 |

The getdp binary is a modified version of GetDP: upstream GetDP 3.5.0 plus the
mpx3 patch, which is shipped in `source/` with the script that built the
engine. No other third-party code is modified, and no source code of
unmodified components is included: `THIRD_PARTY/README.md` gives, for each
component, its upstream address and exact commit or source-archive checksum
(a copy of the source archives of the components used unmodified is kept at
<https://github.com/MPhysXDev/mpxlite-engine-sources>, and GetDP with its history and the
mpx3 patch at <https://github.com/MPhysXDev/fem-mpx>, branch `mpx3`), and `THIRD_PARTY/` holds every
licence and notice file.

Not distributed by this bundle (installed on your machine by `install.sh`,
each under its own licence): Open MPI and OpenBLAS (BSD-3-Clause), the GCC
runtime libraries, OpenMP included (GPL-3.0-or-later with the GCC Runtime
Library Exception), the X11 client libraries, fontconfig, FreeType, OpenGL, GLU,
libjpeg-turbo, libpng and zlib (permissive licences, `THIRD_PARTY/README.md`),
and Python (PSF-2.0) from Ubuntu; NumPy (BSD-3-Clause) and
pytest with its plug-ins (MIT) from PyPI. mpxlite runs
[Gmsh](https://gmsh.info) as a separate program, through the Gmsh jobs. Its authors ask that work using Gmsh cite: C. Geuzaine and J.-F. Remacle, "Gmsh: a three-dimensional finite element mesh generator with built-in pre- and post-processing facilities", International Journal for Numerical Methods in Engineering 79(11), 1309-1331, 2009.
