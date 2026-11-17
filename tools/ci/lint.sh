#!/usr/bin/env bash
# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT
#
# Lint gate: ruff check and ruff format --check with the pinned ruff release
# of noxfile.py, over the targets of `nox -s lint`; bash -n on every shell
# script.
#
# Usage: tools/ci/lint.sh
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

V="$CI_WORK/venv-lint"
fresh_venv "$V"
"$V/bin/python" -m pip install -q "$(ruff_pin)"
cd "$REPO"
targets=(src tests tools noxfile.py)
"$V/bin/ruff" --version
"$V/bin/ruff" check "${targets[@]}"
"$V/bin/ruff" format --check "${targets[@]}"
git ls-files '*.sh' | while read -r f; do bash -n "$f" || die "bash -n $f"; done
echo "bash -n: $(git ls-files '*.sh' | wc -l) scripts"
