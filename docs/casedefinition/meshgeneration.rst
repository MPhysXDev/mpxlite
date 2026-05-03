.. _mpxlite_meshgeneration:

Mesh generation
===============

This page covers the practical step of turning a tagged geometry into
a ``.msh`` file consumable by GetDP, with an emphasis on the format
and topological constraints that ``mpxlite`` enforces.

Generating the mesh from a ``.geo`` script
------------------------------------------

The canonical command is::

    gmsh -3 -format msh22 my_geometry.geo

* ``-3`` --- generate a 3D mesh. Replace with ``-2`` for a surface
  problem.
* ``-format msh22`` --- write in the legacy MSH 2.2 ASCII format.
  This is the format GetDP supports natively; newer MSH 4.x formats
  are *not* supported and will be rejected by :class:`SolverLite` at
  construction time.

The output file is named after the input by default (``my_geometry.msh``)
unless overridden by ``-o my.msh``.

Format validation
-----------------

When a :class:`SolverLite` is constructed, the mesh path is opened
and the first two lines are inspected:

* If the first line starts with ``$NOD``, the file is treated as a
  legacy Gmsh 1.0 mesh and accepted (GetDP also supports this
  format).
* Otherwise the first line must start with ``$MeshFormat``, and the
  version number on the second line must be ``≤ 2.2``. Anything
  else raises :exc:`ValueError` immediately.

This eager check is intentional: the most common GetDP failure mode
in practice is a mesh saved in MSH 4.x with an opaque error, and
catching it at the Python boundary saves a lot of debugging.

Mesh density
------------

GetDP, like every FE solver, demands enough resolution per
wavelength (or per skin depth, or per gradient length, depending on
the physics) to produce reliable results. Plug-ins document their
own rules of thumb (for ``metalab.Meta3D``, ten edges per wavelength
in the densest dielectric is a reasonable starting point); on the
``mpxlite`` side, the mesh is taken as-is.

Two practical levers for controlling density inside Gmsh:

* ``Mesh.CharacteristicLengthMin`` and
  ``Mesh.CharacteristicLengthMax`` --- global floor / ceiling of the
  characteristic length.
* Per-point characteristic lengths via the third argument of
  ``Point(N) = {x, y, z, lc};``.

Periodic meshes
---------------

When the case includes periodicity (for example a 2D-periodic
metasurface, where opposite lateral faces of the unit cell must be
linked by a Bloch boundary condition), the **node patterns** of the
two paired faces must match exactly. Gmsh handles this through the
``Periodic`` directive in the ``.geo`` script:

.. code-block:: c

    Periodic Surface{102} = {101} Translate{Lx, 0, 0};
    Periodic Surface{104} = {103} Translate{0, Ly, 0};

After this, every node on surface ``102`` has a one-to-one
counterpart on ``101`` translated by ``(Lx, 0, 0)``, and likewise
for ``104`` / ``103``.

The plug-in's resource files then reference the node-to-node mapping
via GetDP ``LinkCplx`` constraints; without the ``Periodic``
directive, the constraint would silently fail and the periodicity
would not be enforced numerically.

Inspection
----------

Two routine sanity checks before the first solve:

* **Visual inspection**: ``gmsh my.msh`` opens the GUI; toggle each
  physical group via *Tools → Visibility* and confirm the expected
  classes are populated and non-empty.
* **Tag check** (CLI)::

      gmsh my.msh -part 1 -info 2>&1 | grep "Number of"

  reports the number of nodes, elements and physical groups; cross-
  check against the ``Group {...}`` block of ``<project>.geo.pro``.

Reproducibility
---------------

For contractual deliverables and CI, treat the ``.geo`` script (or
the parametric Python script) as the source of truth and the
``.msh`` file as a derived artefact. Store both under git --- the
``.msh`` to make the test fixtures self-contained, the ``.geo`` to
allow re-derivation if Gmsh's algorithms drift between releases.
