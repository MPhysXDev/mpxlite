#!/usr/bin/env bash
# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT
#
# Build the Python distributions (wheel and sdist) of this checkout into
# OUT_DIR and check what they contain: no file of tools/ or .github/ (the GPL
# patch and the GPL Gmsh programs -- toolkit, jobs -- in
# particular), the MIT licence files and the MIT modules mpxlite.gmshjob and
# mpxlite.cells present.
#
# Usage: tools/ci/dist.sh OUT_DIR
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ $# -eq 1 ] || die "usage: $0 OUT_DIR"
OUT="$1"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
V="$CI_WORK/venv-dist"
fresh_venv "$V"
"$V/bin/python" -m pip install -q build
VERSION="$(mpxlite_version)"
rm -f "$OUT/mpxlite-$VERSION"*
"$V/bin/python" -m build --outdir "$OUT" "$REPO"
WHL="$OUT/mpxlite-$VERSION-py3-none-any.whl"
SDIST="$OUT/mpxlite-$VERSION.tar.gz"
[ -f "$WHL" ] && [ -f "$SDIST" ] || die "missing $WHL or $SDIST"
S="$(tar -tzf "$SDIST")"
W="$("$V/bin/python" -m zipfile -l "$WHL")"
! grep -E "^mpxlite-$VERSION/(tools|\.github)/" <<< "$S" || die "the sdist contains tools/ or .github/"
! grep -iE 'patch|COPYING|mpxlite_gmsh/|/jobs/' <<< "$W$S" || die "a GPL file is in the Python distributions"
grep -q "mpxlite/gmshjob.py" <<< "$W" && grep -q "mpxlite/cells/detect.py" <<< "$W" \
    || die "mpxlite.gmshjob or mpxlite.cells is missing from the wheel"
grep -q "^mpxlite-$VERSION/LICENSE$" <<< "$S" || die "no LICENSE in the sdist"
grep -q "licenses/LICENSE" <<< "$W" || die "no LICENSE in the wheel"
ls -la "$OUT"
