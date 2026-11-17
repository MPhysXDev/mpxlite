# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the GNU General Public
# License (the file COPYING of the directory above) for more details.

"""Gmsh in a job: one thread, its messages collected, its errors failures.

:func:`started` initialises Gmsh for a job: no configuration file, no
terminal output, one thread for everything (``General.NumThreads``,
``Mesh.MaxNumThreads1D/2D/3D``: the same input gives the same mesh), and its
logger on. :class:`Messages` collects what Gmsh logs; :meth:`Messages.check`
turns an error, or a warning that means a wrong result
(:data:`FATAL_WARNINGS`), into a :class:`~mpxlite.GmshJobError` of kind
``gmsh``.
"""

from __future__ import annotations

import contextlib
import re
import sys
from collections.abc import Iterator

import gmsh

from mpxlite.gmshjob import GmshJobError

__all__ = ["FATAL_WARNINGS", "Messages", "started"]

FATAL_WARNINGS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"boolean .*fail",
        r"could not (recover|mesh|create|generate)",
        r"unable to recover",
        r"self[- ]intersect",
        r"invalid (mesh|boundary mesh|surface mesh)",
        r"no (tetrahedra|elements) in volume",
        r"mesh generation error",
        r"\bfailed to\b",
    )
)
"""Gmsh warnings that mean a wrong result: a job fails on them (kind ``gmsh``)."""

_LEVELS = {"Info": "info", "Warning": "warning", "Error": "error"}


class Messages:
    """The messages Gmsh logged, collected from its logger.

    Attributes:
        collected: ``(level, text)`` of every warning and error so far, in
            order (information and progress lines go to the log only).
    """

    def __init__(self) -> None:
        self.collected: list[tuple[str, str]] = []

    def drain(self) -> list[tuple[str, str]]:
        """Read the new messages of the Gmsh logger; print them to the log.

        Returns:
            The new warnings and errors, as ``(level, text)``.
        """
        lines = gmsh.logger.get()
        gmsh.logger.stop()  # the buffer of the logger goes with it
        gmsh.logger.start()
        new: list[tuple[str, str]] = []
        for line in lines:
            head, _, text = line.partition(":")
            level = _LEVELS.get(head.strip())
            if level is None:  # progress lines
                continue
            print(f"gmsh {level}: {text.strip()}", file=sys.stdout)
            if level != "info":
                new.append((level, text.strip()))
        self.collected += new
        return new

    def check(self, step: str) -> None:
        """Drain the logger; fail on an error or a fatal warning of ``step``.

        Args:
            step: What the job was doing, for the failure message.

        Raises:
            GmshJobError: of kind ``gmsh``, code ``gmsh.error`` or
                ``gmsh.warning``, data ``{"step", "message"}``.
        """
        for level, text in self.drain():
            if level == "error":
                raise GmshJobError(
                    "gmsh",
                    "gmsh.error",
                    f"Gmsh reported an error during {step}: {text}",
                    {"step": step, "message": text},
                )
            if any(p.search(text) for p in FATAL_WARNINGS):
                raise GmshJobError(
                    "gmsh",
                    "gmsh.warning",
                    f"Gmsh reported a failure during {step}: {text}",
                    {"step": step, "message": text},
                )

    def as_json(self) -> list[dict[str, str]]:
        """The collected messages as ``{"level", "text"}`` objects."""
        return [{"level": level, "text": text} for level, text in self.collected]


@contextlib.contextmanager
def started() -> Iterator[Messages]:
    """Gmsh initialised for a job (one thread, logger on); finalised at the end.

    Yields:
        The :class:`Messages` of the session.
    """
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        for dim in (1, 2, 3):
            gmsh.option.setNumber(f"Mesh.MaxNumThreads{dim}D", 1)
        gmsh.logger.start()
        messages = Messages()
        try:
            yield messages
        finally:
            messages.drain()
            gmsh.logger.stop()
    finally:
        gmsh.finalize()
