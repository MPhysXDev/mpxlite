.. _mpxlite_geometry:

Geometry
========

`mpxlite` does not impose a geometric modeller: any tool that produces
a Gmsh ``.msh`` file with **physical-tag annotations** is acceptable.
In practice, the path of least resistance is to write a Gmsh
``.geo`` script and run Gmsh in batch mode on it.

This page describes the platform-side conventions; physics plug-ins
(for example ``metalab.Meta3D``) define on top of them their own list
of topology classes and the way mesh tags are expected to map to
those classes.

Why Gmsh
--------

* GetDP is part of the same ecosystem as Gmsh and natively reads its
  ``.msh`` format.
* Gmsh's ``.geo`` scripting language is small, deterministic and
  easily checked into a git repository.
* The Gmsh Python API offers an alternative for parametric mesh
  generation; see :ref:`mpxlite_meshgeneration`.

Required output format
----------------------

The mesh **must** be in ASCII Gmsh MSH format, version 1.0 or 2.2.
Newer formats (4.0 and above) are not supported by GetDP. The
constructor of :class:`SolverLite` validates this eagerly: a mesh in
MSH 4.x raises :exc:`ValueError` *before* any ``getdp`` call.

To enforce the right format from the command line::

    gmsh -3 -format msh22 my_geometry.geo

For binary scripts (Python), pass ``"Mesh.MshFileVersion"`` = ``2.2``
to the Gmsh option API.

Physical tags --- the user's job
--------------------------------

A bare mesh is not enough: GetDP needs to know which volumes,
surfaces, edges and points belong to which physical region. This is
done by attaching **physical tags** (``Physical Volume``,
``Physical Surface``, ...) to the relevant elementary entities.

Plug-ins document the list of platform topology classes they expect
to see populated (for ``metalab.Meta3D``, the ``Meta*`` family ---
``MetaVol``, ``MetaPMLBot``, ``MetaPMLTop``, ``MetaSurfTop``, ...);
the user's job is to:

1. Build the geometry with whatever modelling tool is convenient.
2. Tag every relevant elementary entity with a numeric **physical
   tag** in the ``.geo`` script.
3. In ``<project>.geo.pro``, map each tag to a platform topology
   class via the ``Group {...}`` block.

Step 3 is the one piece of glue that lives in the case files; it is
deliberately decoupled from the geometric modelling so that the same
mesh can be re-used by several solvers, each interpreting tags
through its own ``.geo.pro``.

Minimal example --- a single tagged volume
------------------------------------------

A trivial unit-cell box, in Gmsh ``.geo`` syntax:

.. code-block:: c

    SetFactory("OpenCASCADE");

    L = 1.0;
    Box(1) = {-L/2, -L/2, -L/2, L, L, L};

    Physical Volume("vol", 1)         = {1};
    Physical Surface("top",   101)    = {Surface{1}[1]};
    Physical Surface("bot",   102)    = {Surface{1}[0]};
    Physical Surface("perio", 103)    = {2, 3, 4, 5};

The numeric tags (``1``, ``101``, ``102``, ``103``) are the
identifiers that the case-file ``Group {...}`` block then references
to populate the platform classes. On the GetDP side, the case file
might read::

    Group {
        Vol         = Region[1];
        SurfTop     = Region[101];
        SurfBot     = Region[102];
        SurfPerio   = Region[103];
    }

Best practices
--------------

* **Make the script parametric.** Express dimensions through Gmsh
  variables (``L = ...;``); this lets the same ``.geo`` drive a mesh
  family with no source duplication.
* **Number tags by class, not by entity.** Reserve 1-99 for volumes,
  100-199 for boundary surfaces, 200-299 for periodic faces, etc.
  This makes the ``.geo.pro`` ``Group`` block readable at a glance.
* **Check the mesh visually once.** ``gmsh my_geometry.msh`` opens
  the mesh in the GUI; the *Tools → Visibility* dialog lets you
  toggle physical groups and confirm that every expected tag is
  populated.
* **Keep the mesh under version control with the script.** A
  reproducible case is one where ``gmsh -3 -format msh22 my.geo``
  produces a known ``my.msh`` byte-for-byte.

Python alternative
------------------

For meshes whose topology depends on continuous parameters (lattice
geometry, parameter sweeps), the Gmsh Python API is more flexible
than ``.geo`` scripts:

.. code-block:: python

    import gmsh
    gmsh.initialize()
    gmsh.model.add("demo")
    gmsh.model.occ.addBox(-0.5, -0.5, -0.5, 1.0, 1.0, 1.0, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [1], tag=1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.mesh.generate(3)
    gmsh.write("demo.msh")
    gmsh.finalize()

The same physical-tag conventions apply.
