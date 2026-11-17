# Copyright (c) 2025 MPhysX
# SPDX-License-Identifier: MIT

"""Nox session definitions for mpxlite development workflows."""

from __future__ import annotations

import nox

nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["lint", "tests"]

PYTHON_VERSIONS = ["3.12", "3.14"]
# Pinned: a ruff release adding rules to the selected families must not break the lint gate.
RUFF = "ruff==0.15.12"


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the hermetic test suite (no getdp, no systemd) with coverage and JUnit XML.

    The ``integration`` tests are deselected: they run the real ``getdp``
    (and systemd, ``mpirun``) and have their own session.
    """
    session.install("-e", ".[test]")
    session.run(
        "pytest",
        "-m",
        "not integration",
        "--cov=mpxlite",
        "--cov-report=xml",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--junitxml=junit.xml",
        *session.posargs,
    )


@nox.session
def integration(session: nox.Session) -> None:
    """Run the integration suite on the engine (MPXLITE_GETDP_BINARY, else the installation).

    MPI tests use ``MPXLITE_TEST_MPI_GETDP``; tests whose tool is missing are skipped.
    """
    session.install("-e", ".[test]")
    session.run("pytest", "-m", "integration", "tests/integration", *session.posargs)


@nox.session
def lint(session: nox.Session) -> None:
    """Run ruff lint and verify formatting."""
    session.install(RUFF)
    targets = ["src", "tests", "tools", "noxfile.py"]
    session.run("ruff", "check", *targets)
    session.run("ruff", "format", "--check", *targets)


@nox.session(name="format")
def format_(session: nox.Session) -> None:
    """Apply ruff formatter and lint --fix."""
    session.install(RUFF)
    targets = ["src", "tests", "tools", "noxfile.py"]
    session.run("ruff", "check", "--fix", *targets)
    session.run("ruff", "format", *targets)


@nox.session
def build(session: nox.Session) -> None:
    """Build sdist and wheel distributions."""
    session.install("build>=1.0")
    session.run("python", "-m", "build")
