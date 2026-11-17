# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""Shield every test from the developer's launch settings.

A shell that sourced a dev environment (e.g. ``MPXLITE_GETDP_NP=4``) must not
change what a test runs: each test starts with the launch variables cleared
and one OpenMP thread. Tests that exercise these variables set them
explicitly with ``monkeypatch``.
"""

from __future__ import annotations

import pytest

from mpxlite import launch

LAUNCH_ENV_VARS = (launch.ENV_NP, launch.ENV_THREADS, launch.ENV_CPUS, launch.ENV_LAUNCHER)


@pytest.fixture(autouse=True)
def _hermetic_launch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in LAUNCH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
