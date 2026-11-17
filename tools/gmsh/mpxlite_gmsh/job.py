# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: GPL-2.0-or-later
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; see the GNU General Public
# License (the file COPYING of the directory above) for more details.

"""The side of a job of the protocol of :mod:`mpxlite.gmshjob`.

A job program is::

    from mpxlite_gmsh.job import Job, run

    def main(job: Job) -> None:
        source = job.input("geometry")
        size = job.number("size", minimum=0.0, strict=True)
        ...
        job.output("mesh", "cell.msh")
        job.report["nodes"] = n

    if __name__ == "__main__":
        run(main)

:func:`run` reads ``request.json`` of the run folder (the argument of the
program), starts Gmsh (:func:`mpxlite_gmsh.session.started`), calls ``main``
and writes ``report.json``: status ``ok`` with the outputs and the report, or
status ``failed`` with the failure (a :class:`~mpxlite.GmshJobError` raised by
the job; an exception of the Gmsh API, kind ``gmsh``; any other exception,
kind ``crash``, its traceback in the log). The exit status is 0 on success,
3 on a reported failure.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import gmsh

from mpxlite.gmshjob import (
    FORMAT_VERSION,
    JOB_REPORT_NAME,
    JOB_REQUEST_NAME,
    REPORT_FORMAT,
    REQUEST_FORMAT,
    GmshJobError,
)
from mpxlite_gmsh.session import Messages, started

__all__ = ["FAILED_EXIT", "Job", "check_number", "jsonable", "run"]

FAILED_EXIT = 3
"""Exit status of a job that reports a failure."""

_MISSING = object()


class Job:
    """One run of a job: its request, its outputs and its report.

    Attributes:
        name: Name of the job.
        run_dir: The run folder.
        inputs: Input name -> path.
        parameters: The parameters (a JSON document).
        report: The result of the job, filled by ``main``.
        outputs: Output name -> path relative to the run folder.
        messages: The messages of Gmsh (:class:`~mpxlite_gmsh.session.Messages`).
    """

    def __init__(self, run_dir: Path, request: Mapping[str, Any], messages: Messages) -> None:
        self.name = str(request.get("job", ""))
        self.run_dir = run_dir
        self.inputs = {str(k): Path(v) for k, v in dict(request.get("inputs", {})).items()}
        self.parameters: dict[str, Any] = dict(request.get("parameters", {}))
        self.report: dict[str, Any] = {}
        self.outputs: dict[str, str] = {}
        self.messages = messages

    # -- inputs and parameters --------------------------------------------

    def input(self, name: str, suffixes: Sequence[str] = ()) -> Path:
        """The path of a required input file.

        Args:
            name: Input name.
            suffixes: Accepted file suffixes (lower case, with the dot);
                empty: any.

        Raises:
            GmshJobError: kind ``input``, code ``input.missing`` or
                ``input.format``.
        """
        if name not in self.inputs:
            raise GmshJobError(
                "input",
                "input.missing",
                f"the job {self.name} needs the input {name!r}",
                {"input": name},
            )
        path = self.inputs[name]
        if suffixes and path.suffix.lower() not in suffixes:
            raise GmshJobError(
                "input",
                "input.format",
                f"the input {name!r} must be a {' or '.join(suffixes)} file; found {path.name}",
                {"input": name, "path": str(path), "suffixes": list(suffixes)},
            )
        return path

    def parameter(self, path: str, default: Any = _MISSING) -> Any:
        """A parameter by its dotted path (``"symmetry.mode"``).

        Args:
            path: Dotted path in the parameter document.
            default: Value when it is absent (default: required).

        Raises:
            GmshJobError: kind ``input``, code ``parameter.missing``.
        """
        node: Any = self.parameters
        for key in path.split("."):
            if not isinstance(node, Mapping) or key not in node or node[key] is None:
                if default is _MISSING:
                    raise GmshJobError(
                        "input",
                        "parameter.missing",
                        f"the job {self.name} needs the parameter {path!r}",
                        {"parameter": path},
                    )
                return default
            node = node[key]
        return node

    def number(
        self,
        path: str,
        default: Any = _MISSING,
        *,
        minimum: float | None = None,
        strict: bool = False,
    ) -> float:
        """A finite number parameter, optionally bounded below.

        Raises:
            GmshJobError: kind ``input``, code ``parameter.missing`` or
                ``parameter.invalid``.
        """
        value = self.parameter(path, default)
        return check_number(path, value, minimum=minimum, strict=strict)

    def fail_input(self, path: str, message: str, **data: Any) -> GmshJobError:
        """The ``parameter.invalid`` failure of a parameter (to raise)."""
        return GmshJobError("input", "parameter.invalid", message, {"parameter": path, **data})

    # -- outputs --------------------------------------------------------

    def path(self, name: str) -> Path:
        """``run_dir / name``: where the job writes an output file."""
        return self.run_dir / name

    def output(self, name: str, file_name: str) -> Path:
        """Declare the output ``name``, the file ``file_name`` of the run folder."""
        self.outputs[name] = file_name
        return self.run_dir / file_name


def check_number(
    path: str, value: Any, *, minimum: float | None = None, strict: bool = False
) -> float:
    """``value`` as a finite float, at least (or above) ``minimum``.

    Raises:
        GmshJobError: kind ``input``, code ``parameter.invalid``.
    """
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise GmshJobError(
            "input",
            "parameter.invalid",
            f"the parameter {path!r} must be a finite number; found {value!r}",
            {"parameter": path, "value": repr(value)},
        )
    if minimum is not None and (value <= minimum if strict else value < minimum):
        relation = "above" if strict else "at least"
        raise GmshJobError(
            "input",
            "parameter.invalid",
            f"the parameter {path!r} must be {relation} {minimum:g}; found {value!r}",
            {"parameter": path, "value": value, "minimum": minimum},
        )
    return float(value)


def jsonable(value: Any) -> Any:
    """``value`` as JSON values: NumPy scalars and arrays, tuples and paths converted.

    A non-finite float becomes ``None``; mapping keys become strings.
    """
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        items = sorted(value) if isinstance(value, set | frozenset) else value
        return [jsonable(v) for v in items]
    if hasattr(value, "tolist"):  # NumPy array or scalar
        return jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_report(run_dir: Path, doc: Mapping[str, Any]) -> None:
    tmp = run_dir / (JOB_REPORT_NAME + ".tmp")
    text = json.dumps(jsonable(doc), indent=2, allow_nan=False)
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, run_dir / JOB_REPORT_NAME)


def _read_request(run_dir: Path) -> dict[str, Any]:
    try:
        request = json.loads((run_dir / JOB_REQUEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise GmshJobError(
            "input", "request.invalid", f"no readable request in {run_dir}: {error}"
        ) from None
    if (
        not isinstance(request, dict)
        or request.get("format") != REQUEST_FORMAT
        or request.get("version") != FORMAT_VERSION
    ):
        raise GmshJobError(
            "input",
            "request.format",
            f"the request is not of the format {REQUEST_FORMAT!r} version {FORMAT_VERSION}",
        )
    return request


def run(main: Callable[[Job], None]) -> None:
    """Run a job program (see the module documentation); exits the process."""
    run_dir = Path(sys.argv[1]).resolve()
    start = time.perf_counter()
    doc: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": FORMAT_VERSION,
        "job": Path(sys.argv[0]).stem,
        "gmsh": "",
    }
    status = 0
    job: Job | None = None
    with started() as messages:
        doc["gmsh"] = gmsh.option.getString("General.Version")
        try:
            request = _read_request(run_dir)
            job = Job(run_dir, request, messages)
            main(job)
            messages.check("the end of the job")
            doc.update(status="ok", outputs=job.outputs, report=job.report)
        except GmshJobError as error:
            doc.update(status="failed", failure=error.to_json())
            status = FAILED_EXIT
        except Exception as error:  # the Gmsh API raises Exception itself
            traceback.print_exc(file=sys.stdout)
            messages.drain()
            if type(error) is Exception:
                failure = GmshJobError(
                    "gmsh",
                    "gmsh.exception",
                    f"Gmsh failed: {error}",
                    {"message": str(error)},
                )
            else:
                failure = GmshJobError(
                    "crash",
                    "job.exception",
                    f"the job raised {type(error).__name__}: {error}",
                    {"exception": type(error).__name__, "message": str(error)},
                )
            doc.update(status="failed", failure=failure.to_json())
            status = FAILED_EXIT
        doc["messages"] = messages.as_json()
    doc["elapsed_s"] = time.perf_counter() - start
    sys.stdout.flush()
    _write_report(run_dir, doc)
    sys.exit(status)
