# Copyright (c) 2025 Thierry Valet
# SPDX-License-Identifier: MIT

"""mpxlite — lightweight Python encapsulation of the GetDP FE solver.

Generic, solver-agnostic platform that drives the native ``getdp``
binary through a clean Python API: a typed scalar container
(:class:`WorkflowData`), a single end-to-end invocation entry point
(:meth:`SolverLite.run`), a result wrapper (:class:`SolverResult`),
and parsers for the standard ``.dat`` output formats.

Concrete physics solvers (e.g. ``metalab.Meta3D`` for 3D periodic
metasurfaces) plug in by subclassing :class:`SolverLite` and bundling
their own ``.pro`` resources via the ``resource_files`` attribute.
"""

__version__ = "0.1.0"

__all__: list[str] = []
