<!-- Copyright (c) 2026 MPhysX -->
<!-- SPDX-License-Identifier: MIT -->

# The Gmsh programs of mpxlite — licence notice

The programs in this directory run [Gmsh](https://gmsh.info) for `mpxlite`; they are **not**
part of `mpxlite`'s MIT-licensed code:

* `mpxlite_gmsh/` — the toolkit of the Gmsh jobs (a named geometry, the rules of a periodic
  prism cell, bridging strips, layers, mesh sizes, the symmetric periodic mesh, layered
  two-dimensional sections);
* `jobs/` — the Gmsh jobs (`periodic_cell_geometry.py`, `periodic_cell_mesh.py`,
  `layered_section_mesh.py`), short programs assembled from the toolkit;
* `tests/` — the tests of the toolkit (`python -m pytest tests`).

All of them are:

* Copyright (c) 2026 MPhysX;
* SPDX-License-Identifier: `GPL-2.0-or-later` — licensed under the GNU General Public License,
  version 2 or (at your option) any later version, like Gmsh, whose Python module they import.
  They are **not** covered by the MIT License of `mpxlite`.

A full copy of the GNU GPL v2 accompanies the programs in `COPYING` (this directory); the text
is also at <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.

Gmsh is Copyright (C) 1997-2025 C. Geuzaine and J.-F. Remacle, licensed under the GNU General
Public License, version 2 or (at your option) any later version (`GPL-2.0-or-later`), with the
exceptions stated in its `LICENSE.txt` (<https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/LICENSE.txt>).
No Gmsh code is in this directory. Gmsh 4.15.2 is part of the mpxlite engine: built from its
unmodified source release by `tools/engine/build_engine.sh`, its library
(`engine/lib/libgmsh.so.4.15`), its program (`engine/bin/gmsh`) and Python module
(`engine/gmsh/gmsh.py`, beside these programs in an installation) are distributed in the release bundles as a GPL binary, under the licence
of Gmsh. Its authors ask that work using Gmsh cite:

> C. Geuzaine and J.-F. Remacle, "Gmsh: a three-dimensional finite element mesh generator with
> built-in pre- and post-processing facilities", International Journal for Numerical Methods in
> Engineering 79(11), 1309-1331, 2009.

`mpxlite` (MIT) never imports Gmsh nor these programs: `mpxlite.run_gmsh_job` starts a job as a
separate program (files and a JSON request in, files and a JSON report out). The jobs import the toolkit, `gmsh`, NumPy and the MIT package
`mpxlite` (its pure-NumPy module `mpxlite.cells`).

These programs are shipped in the source repository and in every mpxlite release bundle (in
`gmsh/` and in `source/tools/gmsh/`); the installer puts them in `engine/gmsh/` of the
installation. They are excluded from the Python distributions (wheel and sdist) of `mpxlite`.
