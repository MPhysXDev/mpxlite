<!-- Copyright (c) 2026 MPhysX -->
<!-- SPDX-License-Identifier: MIT -->

# Engine source of this bundle

This directory, with the upstream sources it names, is the corresponding source
of the engine in `engine/` (GetDP 3.5.0 with the mpx3 patch, on complex PETSc
3.26.0 with MUMPS, ScaLAPACK and METIS, and Gmsh 4.15.2 with its graphical
interface on FLTK 1.3.11: library, Python module and program, on OpenCASCADE
7.8.1):

- `tools/engine/build_engine.sh` (MIT): the build script that produced `engine/`.
  It downloads the upstream sources at the exact revisions recorded in
  `THIRD_PARTY/README.md` (GetDP tag `getdp_3_5_0`, PETSc commit of `v3.26.0`;
  PETSc then fetches MUMPS, whose tarball checksum the script checks, ScaLAPACK
  and METIS; the OpenCASCADE 7.8.1, FLTK 1.3.11 and Gmsh 4.15.2 source
  archives, checked by sha256), applies the patch and builds everything in user space. Its header
  lists the Ubuntu packages it needs. Run again, it rebuilds only the parts
  whose inputs changed.
- `tools/getdp/patches/getdp-3.5.0-mpx3.patch` (GPL-2.0-or-later, derived from
  GetDP; modifications Copyright (c) 2026 MPhysX): the only modification of
  third-party code. Its notice is `tools/getdp/patches/README.md`, the GNU GPL
  version 2 text `tools/getdp/patches/COPYING`.

- `tools/gmsh/` (GPL-2.0-or-later, Copyright (c) 2026 MPhysX): the programs
  through which mpxlite runs Gmsh, the toolkit of the Gmsh jobs
  `mpxlite_gmsh/`, the jobs `jobs/` and the tests of the toolkit `tests/` (the
  copy the installer
  uses is `gmsh/` of the bundle). Their notice is `tools/gmsh/README.md`, the
  GNU GPL version 2 text `tools/gmsh/COPYING`. Gmsh
  itself is part of the engine (`engine/lib/libgmsh.so.4.15` with FLTK linked
  in, `engine/bin/gmsh`, `engine/gmsh/gmsh.py`), built by the script above from
  its unmodified source, as FLTK is from its own.

To rebuild the engine (about half an hour on 4 cores; the build is bound to its prefix):

    ./tools/engine/build_engine.sh --prefix ~/opt

The same files are in the mpxlite repository, <https://github.com/MPhysXDev/mpxlite>.

A copy of the upstream sources is kept in two repositories.
<https://github.com/MPhysXDev/fem-mpx> holds GetDP only: its history up to the
pinned release, with the patched source as one commit on its branch `mpx3`.
<https://github.com/MPhysXDev/mpxlite-engine-sources>
holds the source archives of the components used unmodified, at the revisions
above, the MUMPS, OpenCASCADE, FLTK and Gmsh archives among them
(`MUMPS_5.9.1.tar.gz`, `OCCT-7_8_1.tar.gz`, `fltk-1.3.11-source.tar.gz`,
`gmsh-4.15.2-source.tgz`), with their checksums. The build script clones the
first when the upstream GetDP repository cannot be reached, and takes an archive
from the second when the MUMPS site or the address of the archive cannot be
reached.
