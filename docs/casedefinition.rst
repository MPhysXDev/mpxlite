.. _mpxlite_casedefinition:

Case definition
===============

A simulation case is the user-facing input of `mpxlite`: a small set
of plain-text files (and one mesh) that, together, specify what
problem ``getdp`` should solve. This chapter documents the layout,
syntax and conventions of those files at the *platform* level ---
without committing to any particular physics. Plug-ins (for example
``metalab.Meta3D``) build on top of this and add their own topology
and material vocabularies, but the structure described here is
shared by every `mpxlite` solver.

Files in a case
---------------

Each case is described by a triplet of GetDP files plus a mesh, all
sharing a common base name ``<project>``:

============================  =================================================================
File                          Purpose
============================  =================================================================
``<project>.geo.pro``         Maps mesh tags to topology classes
                              (``Group {...}``). Defines the case's topology.
``<project>.physprop.pro``    Per-region material properties and source terms
                              (``Function {...}``).
``<project>.cond.pro``        Case-specific extra constraints (``Constraint {...}``).
                              Often empty: standard constraints (Bloch, Dirichlet, ...) live
                              in the plug-in resources.
``<project>.msh``             The Gmsh mesh, ASCII MSH format ≤ 2.2.
                              May live anywhere; its path is passed explicitly to the solver.
============================  =================================================================

The four files together fully describe a case: change any of them
without touching the others and the next ``solver.run()`` reflects
the new state.

The platform manifest
---------------------

At runtime, :class:`SolverLite` writes a top-level manifest
``mpxlite.pro`` in the project directory. Its job is to ``Include``
the user files and the plug-in resources in the right order; see
:ref:`mpxlite_concepts` for the exact ordering. The user never edits
``mpxlite.pro``: it is auto-generated and overwritten on every
``run()``.

Workflow scalars
----------------

A second auto-generated file, ``workflow.pro``, holds the Python-side
scalars contributed by :class:`WorkflowData`. It is a simple
``Function {...}`` block of name → value pairs and is the first
file ``Include``\ d by the manifest, so every subsequent file (the
case files and the plug-in resources) can reference these scalars by
name as if they were ordinary GetDP constants.

Directory layout in practice
----------------------------

A typical case directory after construction looks like this::

    case/
    ├── demo.geo.pro          (user-edited)
    ├── demo.physprop.pro     (user-edited)
    ├── demo.cond.pro         (user-edited)
    └── demo.msh              (user-generated, see meshgeneration.rst)

After the first ``solver.run()`` ::

    case/
    ├── demo.geo.pro
    ├── demo.physprop.pro
    ├── demo.cond.pro
    ├── demo.msh
    ├── workflow.pro          (auto-generated, scalars)
    ├── mpxlite.pro           (auto-generated, manifest)
    ├── mpxlite.pre           (GetDP pre-processing output)
    ├── M.res                 (GetDP system / solution)
    └── ...                   (any .dat / .pos files written by post-ops)


Reference
---------

The remainder of this chapter walks through the three pieces the user
is expected to author by hand: the geometry script that produces the
mesh, the meshing pass that turns it into a ``.msh`` file, and the
GetDP ``.pro`` mini-language that expresses the rest.

.. toctree::
   :maxdepth: 1

   casedefinition/geometry
   casedefinition/meshgeneration
   casedefinition/pro_syntax
