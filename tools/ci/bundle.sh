#!/usr/bin/env bash
# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT
#
# Build the bundle from the engine in ENGINE_PREFIX with
# tools/bundle/make_bundle.sh, into OUT_DIR, and check that the expected
# mpxlite-<version>-ubuntu<release>-x86_64.tar.gz with a valid .sha256 and the
# files of the bundle contract came out. The checkout must be a git repository
# (the tests shipped are the files git tracks).
#
# Usage: tools/ci/bundle.sh ENGINE_PREFIX OUT_DIR
# Prints the tarball path on the last line.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ $# -eq 2 ] || die "usage: $0 ENGINE_PREFIX OUT_DIR"
ENGINE="$1"; OUT="$2"
VERSION="$(mpxlite_version)"
REL="$(awk '$1 == "os_version" {print $2}' "$ENGINE/engine-info/BUILD")"
NAME="mpxlite-$VERSION-ubuntu$REL-x86_64"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
rm -f "$OUT/$NAME.tar.gz" "$OUT/$NAME.tar.gz.sha256"
say "make_bundle.sh -> $OUT/$NAME.tar.gz"
"$REPO/tools/bundle/make_bundle.sh" --engine "$ENGINE" --out "$OUT"
[ -f "$OUT/$NAME.tar.gz" ] || die "missing $OUT/$NAME.tar.gz"
(cd "$OUT" && sha256sum -c "$NAME.tar.gz.sha256")
# (listing saved first: grep -q on a pipe would SIGPIPE tar under pipefail)
LISTING="$(tar -tzf "$OUT/$NAME.tar.gz")"
for f in install.sh README-INSTALL.md mpxlite_smoke.py BUILDINFO TARGET SHA256SUMS \
         engine/bin/getdp engine/bin/gmsh engine/lib/libgmsh.so.4.15 engine/gmsh/gmsh.py \
         "wheels/mpxlite-$VERSION-py3-none-any.whl" THIRD_PARTY/README.md \
         THIRD_PARTY/gmsh/LICENSE.txt THIRD_PARTY/gmsh/CREDITS.txt \
         THIRD_PARTY/opencascade/LICENSE_LGPL_21.txt THIRD_PARTY/opencascade/OCCT_LGPL_EXCEPTION.txt \
         THIRD_PARTY/fltk/COPYING \
         THIRD_PARTY/GPL-3.0.txt THIRD_PARTY/mpxlite/LICENSE source/tools/engine/build_engine.sh \
         source/tools/getdp/patches/getdp-3.5.0-mpx3.patch source/tools/getdp/patches/COPYING \
         gmsh/README.md gmsh/COPYING gmsh/jobs/periodic_cell_geometry.py \
         gmsh/jobs/periodic_cell_mesh.py gmsh/jobs/layered_section_mesh.py \
         source/tools/gmsh/README.md source/tools/gmsh/COPYING \
         tests/pytest.ini tests/integration/data/rod.msh; do
    grep -qx "$NAME/$f" <<< "$LISTING" || die "no $f in the bundle"
done
[ "$(grep -c "^$NAME/engine/lib/." <<< "$LISTING")" -ge 3 ] || die "engine/lib incomplete"
grep -q "^$NAME/engine/lib/libTKernel\.so\." <<< "$LISTING" || die "no OpenCASCADE in engine/lib"
# The component table of THIRD_PARTY/README.md and BUILDINFO: every component
# with its version, upstream address and a full commit or tarball checksum.
TP="$(tar -xzOf "$OUT/$NAME.tar.gz" "$NAME/THIRD_PARTY/README.md")"
BI="$(tar -xzOf "$OUT/$NAME.tar.gz" "$NAME/BUILDINFO")"
for c in getdp petsc mumps scalapack metis opencascade fltk gmsh; do
    row="$(grep -E "^\| $c( |\|)" <<< "$TP")" || die "THIRD_PARTY/README.md: no row for $c"
    IFS='|' read -r _ _ ver lic up rev files _ <<< "$row"
    for f in "$ver" "$lic" "$up" "$rev" "$files"; do
        [ -n "${f// /}" ] || die "THIRD_PARTY/README.md: empty field for $c: $row"
    done
    grep -qE '^ *https://[^ ]+ *$' <<< "$up" || die "THIRD_PARTY/README.md: upstream of $c: '$up'"
    grep -qE '(commit [0-9a-f]{40}([; ]|$|  *$)|tarball sha256 [0-9a-f]{64})' <<< "$rev" \
        || die "THIRD_PARTY/README.md: revision of $c: '$rev'"
    grep -qE "^component   $c [^ (]" <<< "$BI" || die "BUILDINFO: no version for $c"
    grep -E "^component   $c " <<< "$BI" | grep -qE '(commit [0-9a-f]{40}|tarball sha256 [0-9a-f]{64})' \
        || die "BUILDINFO: no full revision for $c"
done
echo "records: THIRD_PARTY/README.md and BUILDINFO complete for getdp, petsc, mumps, scalapack, metis, opencascade, fltk, gmsh"
# Gmsh is the one of the engine: no gmsh wheel, no other copy of the module or library.
[ "$(grep -cE "/(gmsh\.py|libgmsh\.so[.0-9]*)$" <<< "$LISTING")" = 2 ] || die "Gmsh is in the bundle elsewhere than engine/"
! grep -iE "^$NAME/wheels/gmsh-" <<< "$LISTING" || die "a gmsh wheel is in the bundle"
[ "$(grep -cv "^$NAME/" <<< "$LISTING")" = 0 ] || die "the tarball has entries outside $NAME/"
echo "$OUT/$NAME.tar.gz"
