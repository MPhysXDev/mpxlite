.. _mpxlite_index:

Welcome to mpxlite
==================

`mpxlite` is a lightweight Python encapsulation of the
`GetDP <http://getdp.info>`_ finite-element solver. It exposes the
native ``getdp`` binary through a small, opinionated API designed to
be the **host platform** of higher-level physics plug-ins (for example
`metalab <../metalab>`_ for periodic 3D metasurfaces).

The package is intentionally minimal:

* :class:`mpxlite.SolverLite` --- a subprocess wrapper around ``getdp``
  that validates the case-file triplet and the Gmsh mesh, writes a
  manifest, and runs the end-to-end pre-process / solve / optional
  post-op pipeline as a *single* CLI invocation.
* :class:`mpxlite.WorkflowData` --- a typed dict-like container of
  GetDP scalars, serialized at runtime as a ``Function {...}`` block.
* :class:`mpxlite.SolverResult` --- a frozen dataclass bundling the
  work directory and the :class:`subprocess.CompletedProcess` returned
  by ``getdp``.
* a handful of NumPy-aware parsers for the standard ``.dat`` output
  formats.

A bundled wrapper script, ``tools/getdp_runner.sh``, auto-detects the
local MPI launcher (``mpirun``, ``srun`` or ``mpiexec``) so that the
same Python code runs unchanged on a laptop, on a single-node
SLURM/PBS allocation, and on a multi-node cluster job.

`mpxlite` itself contains no physics: concrete solvers plug in by
subclassing :class:`SolverLite` and bundling their own ``.pro``
resources via the ``resource_files`` attribute.

.. toctree::
    :maxdepth: 1

    notice
    installation
    concepts
    casedefinition
    api
    copyright

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
