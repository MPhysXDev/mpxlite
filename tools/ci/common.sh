# shellcheck shell=bash disable=SC2034
# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT
#
# Shared settings of the tools/ci scripts (sourced, not run).
#
#   REPO      this mpxlite checkout (from the script location)
#   CI_WORK   scratch directory for venvs, reports and builds (default:
#             $RUNNER_TEMP/mpxlite-ci, else /tmp/mpxlite-ci); the scripts
#             write nothing into the checkout
#   PYTHON    interpreter the venvs are made from (default: python3)
#   ENGINE_PREFIX_DEFAULT  where CI builds the engine: a neutral path outside
#             any home directory, since the PETSc build records its prefix

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CI_WORK="${CI_WORK:-${RUNNER_TEMP:-/tmp}/mpxlite-ci}"
mkdir -p "$CI_WORK"
CI_WORK="$(cd "$CI_WORK" && pwd)"
PYTHON="${PYTHON:-python3}"
ENGINE_PREFIX_DEFAULT=/opt/mpxlite-engine
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1

say() { printf '\n=== %s\n' "$*"; }
die() { echo "error: $*" >&2; exit 1; }

# fresh_venv DIR: a new venv from $PYTHON with an up-to-date pip.
fresh_venv() {
    rm -rf "$1"
    "$PYTHON" -m venv "$1"
    "$1/bin/python" -m pip install -q --upgrade pip
}

# The version of this checkout (src/mpxlite/__init__.py).
mpxlite_version() { sed -n 's/^__version__ = "\(.*\)"/\1/p' "$REPO/src/mpxlite/__init__.py"; }

# ruff release of the lint gate (RUFF in noxfile.py).
ruff_pin() { sed -n 's/^RUFF = "\(ruff==[^"]*\)"/\1/p' "$REPO/noxfile.py"; }

# Run as root, or through sudo.
as_root() { if [ "$(id -u)" = 0 ]; then "$@"; else sudo "$@"; fi; }

# ensure_user NAME: a non-root account NAME with passwordless sudo (for the
# installer's apt step), created when missing.
ensure_user() {
    if ! id "$1" >/dev/null 2>&1; then
        as_root useradd --create-home --shell /bin/bash "$1"
        echo "$1 ALL=(ALL) NOPASSWD:ALL" | as_root tee "/etc/sudoers.d/90-$1" >/dev/null
        as_root chmod 440 "/etc/sudoers.d/90-$1"
    fi
}
