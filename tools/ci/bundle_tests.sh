#!/usr/bin/env bash
# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT
#
# Tests of an installed mpxlite, in a clean environment (env -i) that only
# sources PREFIX/activate.sh, as the owner of PREFIX, on the tests installed
# in PREFIX/tests:
#   quick  smoke test (mpxlite-smoke), the tests of the Gmsh toolkit and the
#          integration suite
#   full   smoke test, the unit suite, the tests of the Gmsh toolkit and the
#          integration suite
#
# Usage: tools/ci/bundle_tests.sh quick|full PREFIX REPORT_DIR
# Writes REPORT_DIR/junit-bundle-*.xml (also when a suite fails).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ $# -eq 3 ] || die "usage: $0 quick|full PREFIX REPORT_DIR"
MODE="$1"; PREFIX="$2"; REPORTS="$3"
case "$MODE" in quick|full) ;; *) die "mode must be quick or full" ;; esac
mkdir -p "$REPORTS"
REPORTS="$(cd "$REPORTS" && pwd)"
OWNER="$(stat -c %U "$PREFIX")"
RUN=()
[ "$OWNER" = "$(id -un)" ] || RUN=(sudo -u "$OWNER" -H)
HOME_DIR="$(getent passwd "$OWNER" | cut -d: -f6)"
# The reports are written by the owner of PREFIX in its home, then copied.
UREP="$HOME_DIR/ci-reports"
"${RUN[@]}" rm -rf "$UREP"
"${RUN[@]}" mkdir -p "$UREP"
collect() { as_root cp "$UREP"/*.xml "$REPORTS"/ 2>/dev/null || true; }
trap collect EXIT

in_env() {
    "${RUN[@]}" env -i HOME="$HOME_DIR" USER="$OWNER" LOGNAME="$OWNER" LANG=C.UTF-8 \
        PATH=/usr/sbin:/usr/bin:/sbin:/bin \
        bash -c '. "$1/activate.sh" && cd "$1" && shift && "$@"' _ "$PREFIX" "$@"
}

say "VERSION"
cat "$PREFIX/VERSION"
say "smoke test"
in_env mpxlite-smoke
rc=0
if [ "$MODE" = full ]; then
    say "unit suite (installed copy)"
    in_env python -m pytest -p no:cacheprovider -m "not integration" tests/unit \
        --junitxml="$UREP/junit-bundle-unit.xml" || rc=1
fi
say "tests of the Gmsh toolkit (installed copy)"
in_env python -m pytest -p no:cacheprovider engine/gmsh/tests \
    --junitxml="$UREP/junit-bundle-gmsh-toolkit.xml" || rc=1
say "integration suite (installed copy)"
in_env python -m pytest -p no:cacheprovider -m integration tests/integration \
    --junitxml="$UREP/junit-bundle-integration.xml" || rc=1
exit "$rc"
