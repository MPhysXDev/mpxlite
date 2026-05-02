# Copyright (c) 2025 Thierry Valet
# SPDX-License-Identifier: MIT

"""Result of a :meth:`SolverLite.run` invocation.

Minimal wrapper around the artefacts produced by a single end-to-end getdp
call: the project directory and the :class:`subprocess.CompletedProcess`.
Output-parsing helpers (e.g. ``parse_eff_table``) are intentionally kept
out of this module for now and added incrementally as concrete call sites
ask for them.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SolverResult:
    """Artefacts of a single end-to-end getdp invocation.

    The ``work_dir`` attribute is the directory where the case files
    (``.geo.pro``, ``.physprop.pro``, ``.cond.pro``), the manifest
    (``mpxlite.pro``), the workflow (``workflow.pro``), the mesh, and
    every getdp-side output (``.pre``, ``.res``, ``.dat``, ``.pos``, …)
    live. The ``completed`` attribute is the
    :class:`subprocess.CompletedProcess` returned by running getdp,
    carrying return code, captured stdout, stderr, and the executed
    command.
    """

    work_dir: Path
    completed: subprocess.CompletedProcess[str]
