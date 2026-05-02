# Copyright (c) 2025 Thierry Valet
# SPDX-License-Identifier: MIT

"""Nox session definitions for mpxlite development workflows."""

from __future__ import annotations

import nox

nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["lint", "tests"]

PYTHON_VERSIONS = ["3.10", "3.11", "3.12"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite with coverage and JUnit XML reports."""
    session.install("-e", ".[test]")
    session.run(
        "pytest",
        "--cov=mpxlite",
        "--cov-report=xml",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--junitxml=junit.xml",
        *session.posargs,
    )


@nox.session
def lint(session: nox.Session) -> None:
    """Run ruff lint and verify formatting."""
    session.install("ruff>=0.4")
    targets = ["src", "tests", "noxfile.py"]
    session.run("ruff", "check", *targets)
    session.run("ruff", "format", "--check", *targets)


@nox.session(name="format")
def format_(session: nox.Session) -> None:
    """Apply ruff formatter and lint --fix."""
    session.install("ruff>=0.4")
    targets = ["src", "tests", "noxfile.py"]
    session.run("ruff", "check", "--fix", *targets)
    session.run("ruff", "format", *targets)


@nox.session
def build(session: nox.Session) -> None:
    """Build sdist and wheel distributions."""
    session.install("build>=1.0")
    session.run("python", "-m", "build")
