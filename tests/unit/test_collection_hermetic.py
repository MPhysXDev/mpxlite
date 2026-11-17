# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""Collecting the whole suite (integration modules included) starts no process.

``nox -s tests`` deselects the integration tests with ``-m "not integration"``
but pytest still imports their modules: their skip conditions must therefore
be lazy (string-form ``skipif``), or collection would run ``systemd-run`` and
``getdp --info``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_PLUGIN = """
import os
import subprocess

LOG = os.environ["MPXLITE_TEST_EXEC_LOG"]


def _record(what):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(repr(what) + "\\n")


def _popen_init(self, args, *a, **k):
    _record(args)
    raise OSError("process creation forbidden at collection")


def _system(cmd):
    _record(cmd)
    raise OSError("process creation forbidden at collection")


subprocess.Popen.__init__ = _popen_init
os.system = _system
"""


def test_collection_runs_no_subprocess(tmp_path: Path) -> None:
    (tmp_path / "forbid_exec.py").write_text(_PLUGIN, encoding="utf-8")
    log = tmp_path / "exec.log"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(REPO / "src"), str(REPO), env.get("PYTHONPATH", "")]
    )
    env["MPXLITE_TEST_EXEC_LOG"] = str(log)
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "forbid_exec",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "-m",
            "not integration",
            "tests",
        ],
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
        check=False,
    )
    assert out.returncode == 0, out.stdout
    assert "error" not in out.stdout.lower().splitlines()[-1], out.stdout
    assert not log.exists(), f"processes started at collection:\n{log.read_text()}"
