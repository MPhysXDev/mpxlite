.. _mpxlite_concepts:

Concepts
========

This chapter is the *page-cœur* of `mpxlite`: it describes the
encapsulation model that the package implements around the GetDP
solver, the lifecycle of a single solver run, and the responsibilities
shared between the host platform and a downstream physics plug-in.
Read it once and the rest of the documentation should fall into
place.

`mpxlite` is **deliberately minimal**: a subprocess wrapper, a typed
scalar container, a one-shot ``run()`` method, a result wrapper and a
handful of output parsers. Anything physics-specific (FE topology,
function spaces, formulations, post-processing) lives in a
*resource bundle* contributed by the plug-in. This separation is what
makes ``mpxlite`` reusable for any GetDP-driven solver.


Architecture in two layers
--------------------------

A real-world `mpxlite` deployment has three actors: the platform,
a plug-in, and a user case::

                 ┌────────────────────────────────────────┐
                 │  user case files + mesh                │
                 │  (chequerboard.geo.pro, .msh, run.py)  │
                 └─────────────────┬──────────────────────┘
                                   │
                                   ▼
       ┌─────────────────────────────────────────────────────┐
       │  Plug-in layer (e.g. metalab.Meta3D)                │
       │   - subclasses SolverLite                           │
       │   - bundles its own .pro resource files             │
       │   - exposes typed setters that write into           │
       │     self.workflow                                   │
       └─────────────────┬───────────────────────────────────┘
                         │
                         ▼
       ┌─────────────────────────────────────────────────────┐
       │  Host layer (mpxlite, this package)                 │
       │   - SolverLite, WorkflowData, SolverResult          │
       │   - manifest auto-generation                        │
       │   - subprocess + MPI launcher abstraction           │
       │   - .dat parsers                                    │
       └─────────────────┬───────────────────────────────────┘
                         │
                         ▼
                 ┌──────────────────────────────────┐
                 │  GetDP CLI + bundled .pro chain  │
                 └──────────────────────────────────┘

The host layer is **physics-agnostic**: it knows how to drive
``getdp``, but does not know what equations are being solved. The
plug-in layer knows the physics but does not know how ``getdp`` is
invoked. The user case is plain text and a mesh; it does not know
about Python at all.


SolverLite --- the platform-level wrapper
-----------------------------------------

:class:`mpxlite.SolverLite` is the central class. An instance binds:

* a project directory, holding the case files and (optionally) the
  mesh;
* a project name, used as the base name of the case-file triplet
  ``<project>.geo.pro`` / ``.physprop.pro`` / ``.cond.pro``;
* a mesh path, validated for Gmsh MSH ≤ 2.2 ASCII format (the only
  format supported by GetDP);
* a GetDP ``Resolution`` name (``resolution_id``);
* a list of bundled ``.pro`` resource files contributed by the plug-in;
* a :class:`WorkflowData` instance (created empty if not supplied).

Construction is **eagerly validating**: the project directory must
exist, every case file must be present, every resource file must be
on disk, and the mesh must be in a GetDP-compatible MSH version.
Errors surface as :exc:`FileNotFoundError` or :exc:`ValueError` from
the constructor --- not minutes later from a cryptic GetDP message.

A plug-in subclass typically looks like this:

.. code-block:: python

    from mpxlite import SolverLite, WorkflowData
    from importlib.resources import files

    class MyPlugin(SolverLite):
        RESOLUTION_ID = "my_resolution"

        def __init__(self, project_dir, project_name, mesh):
            super().__init__(
                project_dir=project_dir,
                project_name=project_name,
                mesh=mesh,
                resolution_id=self.RESOLUTION_ID,
                resource_files=[
                    Path(str(files("myplugin._resources")
                              .joinpath("functionspaces.pro"))),
                    Path(str(files("myplugin._resources")
                              .joinpath("formulation.pro"))),
                ],
                workflow=WorkflowData(),
            )
            self._set_defaults()

        def set_frequency(self, f_hz):
            self.workflow["MY_FREQ"] = float(f_hz)

The plug-in author writes typed setters that mutate ``self.workflow``;
nothing else is needed for a fully working physics solver.


WorkflowData --- typed dict of GetDP scalars
--------------------------------------------

:class:`mpxlite.WorkflowData` is the Python-side container of GetDP
scalars. It implements the :class:`collections.abc.MutableMapping`
protocol, so it behaves like a regular ``dict``:

.. code-block:: python

    >>> wf = WorkflowData()
    >>> wf['MY_FREQ'] = 1.2e10
    >>> wf['LABEL'] = "demo"
    >>> wf['ENABLED'] = True

with two strict validation rules:

* **Names** must match the GetDP identifier syntax
  ``[A-Za-z_][A-Za-z_0-9]*``.
* **Values** must be ``bool``, ``int``, ``float`` or ``str``.

Booleans are coerced to ``1`` / ``0``; strings are quoted and escaped.
Any other type raises :exc:`TypeError` at assignment, ruling out a
whole class of GetDP errors at the Python boundary.

Serialization to a GetDP ``Function {...}`` block is handled by
:meth:`WorkflowData.to_pro_string` and :meth:`WorkflowData.write_pro`.
Keys are emitted in sorted order, so the output is deterministic and
diff-friendly. The serialized file is what the manifest then
``Include``\ s into the GetDP run.

Why a typed container rather than a free-form ``dict``? Because the
serialization step needs to know how to format each value (e.g. a
boolean should become ``1``, not ``True``). Centralizing the
formatting logic in one place removes a long category of subtle
``getdp`` parse errors that only surface at solve time.


The manifest is auto-generated
------------------------------

Every :meth:`SolverLite.run` call writes two files in the project
directory before invoking ``getdp``:

1. ``workflow.pro`` --- the serialization of the current
   :class:`WorkflowData` as a ``Function {...}`` block.
2. ``mpxlite.pro`` --- a top-level manifest that ``Include``\ s, in
   exactly this order:

   1. ``workflow.pro``                  (Function: scalars)
   2. ``<project>.geo.pro``             (Group: regions)
   3. each entry of ``resource_files``  (FE machinery, formulation,
      post-processing)
   4. ``<project>.physprop.pro``        (Function: ε, μ, σ, sources)
   5. ``<project>.cond.pro``            (Constraint)

The order matters: scalars first, then the topology classes that the
case file populates, then the resource files that consume those
classes (function spaces, formulation), then the per-region material
properties that close the ``Function`` blocks declared by the
resources, and finally the case-specific extra constraints.

Both files are auto-generated and overwritten on every ``run()``.
A line at the top of the manifest reminds the reader::

    // mpxlite.pro -- auto-generated. Do not edit by hand.

Hand-editing them is therefore pointless: they are rewritten the next
time the solver is invoked. The right way to influence them is to
edit ``WorkflowData`` (for the scalars) or to add another resource
file in the plug-in (for everything else).


run() --- one-shot end-to-end invocation
----------------------------------------

The single public entry point for executing a solve is
:meth:`SolverLite.run`. Its signature is intentionally narrow:

.. code-block:: python

    result = solver.run(postop="my_postop")        # or postop=None

The method writes ``workflow.pro`` and ``mpxlite.pro``, then runs::

    getdp mpxlite.pro -msh <mesh> -v <verbosity> \
        -pre <resolution_id> -cal [-pos <postop>]

in **one** ``subprocess.run`` call. This is the design that makes
distributed execution natural: pre-process, solve and post-op share
the same in-memory PETSc state, so output integrals (Fourier
projections on internal surfaces, energy norms, etc.) are evaluated
correctly under MPI. There is no separate ``preprocess()`` /
``process()`` /  ``postprocess()`` choreography to keep in sync.

GetDP's CLI accepts only **one** ``-pos`` flag per invocation
(multiple flags overwrite each other silently). For workflows that
need multiple outputs from a single solve, define a composite
``PostOperation`` in the resource ``.pro`` that bundles every desired
``Print[]`` call and pass its name as ``postop``. Calling ``run()``
twice with two different ``postop`` values would re-factorise the
linear system from scratch the second time --- a no-go for any
non-trivial case.

If ``getdp`` exits with a non-zero return code, ``run()`` raises
:exc:`mpxlite.GetDPError` carrying the return code, the captured
stdout and stderr, and the exact command line that was executed. The
exception message displays the last 500 characters of stderr, which
is usually enough to locate the offending ``.pro`` line.


SolverResult --- minimal artefact wrapper
-----------------------------------------

A successful ``run()`` returns a :class:`mpxlite.SolverResult`, a
frozen dataclass with two fields:

* ``work_dir`` --- the project directory, where the manifest, the
  ``workflow.pro``, the mesh, and every GetDP-side output file
  (``.pre``, ``.res``, ``.dat``, ``.pos``, ...) live;
* ``completed`` --- the :class:`subprocess.CompletedProcess` returned
  by ``getdp``, carrying return code, captured stdout, captured
  stderr, and the executed command line.

The class is **deliberately empty** of physics-specific accessors:
``mpxlite`` does not know what files the plug-in's post-processing
wrote. Plug-ins are expected to add their own getters in their
``Solver`` subclass (e.g. ``solver.read_efficiencies()``) that read
the relevant ``.dat`` files via the parsers documented next.


Parsers for GetDP textual outputs
---------------------------------

Three small NumPy-aware helpers cover the majority of textual
``.dat`` formats produced by GetDP's ``Print[]`` operation:

* :func:`mpxlite.parse_complex_table` --- parse a table whose last
  two columns are ``(Re, Im)``. Returns a 1-D :class:`numpy.ndarray`
  of complex values, one per data row. Suitable for
  ``Format FrequencyTable`` outputs (``<freq> <Re> <Im>``) and for
  indexed ``Format Table`` outputs that share the same
  last-two-columns convention.
* :func:`mpxlite.parse_complex_scalar` --- convenience wrapper that
  returns the last data row of a complex table as a Python
  ``complex``. Useful for one-shot frequencies (e.g. an admittance
  evaluated at a single frequency).
* :func:`mpxlite.parse_real_table` --- parse a generic real table
  into a 2-D :class:`numpy.ndarray` of shape ``(n_rows, n_cols)``.
  Used for any format that does not fit the complex convention.

All three raise :exc:`FileNotFoundError` if the path does not exist
and :exc:`ValueError` for empty or malformed files. They ignore
``#``-prefixed comment lines.

For non-trivial output layouts (multi-block files, custom column
ordering), plug-ins are expected to add their own parsers; the three
functions above are the lowest common denominator.


Lifecycle of a single run
-------------------------

Putting everything together, a typical end-to-end call goes through
the following steps:

1. **Construction**:

   .. code-block:: python

      solver = MyPlugin("./case", "demo", "demo.msh")

   The constructor validates the project directory, the case-file
   triplet, the mesh format and the resource files. Defaults are
   pushed into ``self.workflow`` by the plug-in's ``__init__``.

2. **Configuration**:

   .. code-block:: python

      solver.set_frequency(12e9)
      solver.workflow["EXTRA_FLAG"] = True

   Typed setters (or direct ``self.workflow`` writes) populate the
   scalars to be exposed to the ``.pro`` chain.

3. **Execution**:

   .. code-block:: python

      result = solver.run(postop="po_demo")

   The host layer writes ``workflow.pro`` + ``mpxlite.pro`` and
   invokes ``getdp`` once. Whatever stdout and stderr the binary
   produces is captured and bundled in ``result.completed``.

4. **Post-processing**:

   .. code-block:: python

      from mpxlite import parse_complex_table
      data = parse_complex_table(result.work_dir / "demo.dat")

   Output ``.dat`` files are read with the parsers (or with custom
   getters added by the plug-in).

There is intentionally no caching, no implicit state mutation across
calls, no hidden working directory: the project directory is the
single source of truth for both inputs and outputs of a run.


Where to look in the code
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Role
   * - ``src/mpxlite/__init__.py``
     - Top-level exports and the bundled-wrapper path discovery
       (:func:`find_getdp`).
   * - ``src/mpxlite/base.py``
     - :class:`SolverLite`, :exc:`GetDPError` and the manifest
       auto-generation logic.
   * - ``src/mpxlite/workflow.py``
     - :class:`WorkflowData`: identifier validation, value-type
       validation and the ``Function {...}`` serializer.
   * - ``src/mpxlite/result.py``
     - :class:`SolverResult` dataclass.
   * - ``src/mpxlite/parsers.py``
     - :func:`parse_complex_table`, :func:`parse_complex_scalar`
       and :func:`parse_real_table`.
   * - ``src/mpxlite/tools/getdp_runner.sh``
     - Bash wrapper around ``getdp`` with cluster-aware MPI
       launching.
