#!/usr/bin/env bash
# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT
#
# Make sure the engine is built in ENGINE_PREFIX (default /opt/mpxlite-engine):
# run tools/engine/build_engine.sh there, which skips the parts that are up to
# date (restored from the cache) and builds the others, then regenerates
# engine-info/. The prefix is created (with sudo when needed) and owned by the
# current user. Needs tools/ci/apt.sh engine.
# The prefix must be the same path on every run: the PETSc build is bound to
# it (actions/cache restores it at the same path).
#
# Usage: tools/ci/engine.sh [ENGINE_PREFIX] [--jobs N]
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PREFIX="$ENGINE_PREFIX_DEFAULT"
if [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; then PREFIX="$1"; shift; fi
if [ ! -d "$PREFIX" ]; then
    mkdir -p "$PREFIX" 2>/dev/null || as_root mkdir -p "$PREFIX"
fi
[ -w "$PREFIX" ] || as_root chown "$(id -u):$(id -g)" "$PREFIX"
say "engine in $PREFIX: parts restored or built"
for part in petsc getdp petsc4py occt fltk gmsh; do
    echo "    $part: $("$REPO/tools/engine/build_engine.sh" --print-key "$part")"
done
T0=$(date +%s)
"$REPO/tools/engine/build_engine.sh" --prefix "$PREFIX" "$@"
echo "engine step: $(( $(date +%s) - T0 )) s"
HWLOC_COMPONENTS=-gl "$PREFIX/getdp-3.5.0-mpx3/bin/getdp" --info
