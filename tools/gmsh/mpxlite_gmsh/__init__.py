# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the GNU General Public
# License (the file COPYING of the directory above) for more details.

"""The Gmsh toolkit of the jobs of mpxlite.

The jobs (``../jobs/``) are short programs assembled from this package; they
are started by :func:`mpxlite.run_gmsh_job` (MIT, in the Python package of
mpxlite), which never imports Gmsh. This package imports ``gmsh`` (the module
of the engine), NumPy and :mod:`mpxlite` (its pure-NumPy periodic-cell tools,
:mod:`mpxlite.cells`).

Modules:

* :mod:`mpxlite_gmsh.session` -- Gmsh initialised for a job (one thread,
  messages collected, errors and fatal warnings turned into failures);
* :mod:`mpxlite_gmsh.job` -- the job side of the request/report protocol;
* :mod:`mpxlite_gmsh.named` -- :class:`~mpxlite_gmsh.named.NamedGeometry`, a
  model whose parts are known by name: files, conforming fragment, scaling,
  fragment with new volumes, questions;
* :mod:`mpxlite_gmsh.prism` -- the rules and facts of a periodic prism cell;
* :mod:`mpxlite_gmsh.strips` -- bridging strips between two surface sets;
* :mod:`mpxlite_gmsh.layers` -- layers added above and below a cell;
* :mod:`mpxlite_gmsh.sizing` -- mesh sizes of a cell from named controls;
* :mod:`mpxlite_gmsh.symmetric` -- the symmetric periodic mesh of a tagged
  cell (detection, asymmetric unit, paired faces, meshing, replication);
* :mod:`mpxlite_gmsh.section` -- layered two-dimensional sections: a
  structured core band cut into columns, layers above and below, the size
  law of the free layers, named sides and lines.
"""
