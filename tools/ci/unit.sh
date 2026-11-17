#!/usr/bin/env bash
# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT
#
# Unit tests (everything but the `integration` marker; no engine needed) in a
# fresh venv of $PYTHON with mpxlite[test] installed from this checkout.
#
# Usage: tools/ci/unit.sh [pytest args...]
# Writes $CI_WORK/junit-unit-<python>.xml.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PYVER="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
V="$CI_WORK/venv-unit-$PYVER"
say "venv (Python $PYVER)"
fresh_venv "$V"
"$V/bin/python" -m pip install -q "${REPO}[test]"
say "unit tests"
cd "$REPO"
"$V/bin/python" -m pytest -m "not integration" -p no:cacheprovider \
    --junitxml="$CI_WORK/junit-unit-$PYVER.xml" "$@"
