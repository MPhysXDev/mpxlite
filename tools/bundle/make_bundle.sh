#!/usr/bin/env bash
# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT
#
# Build the mpxlite release bundle mpxlite-<version>-ubuntu<release>-x86_64.tar.gz
# (and its .sha256): the engine (GetDP 3.5.0-mpx3, the Gmsh 4.15.2 library with
# its graphical interface (FLTK 1.3.11 linked in), its Python module and the
# gmsh executable, and the shared libraries they load from the engine build:
# PETSc, ScaLAPACK, METIS, OpenCASCADE; RUNPATH rewritten to $ORIGIN), the
# mpxlite wheel, the petsc4py wheel of the engine build (its extension module
# relocated to the engine/lib of the installation), the Gmsh programs of
# mpxlite (gmsh/: the toolkit mpxlite_gmsh/, the jobs and the toolkit's tests,
# from tools/gmsh/ of this checkout, GPL-2.0-or-later),
# the installer, the smoke test, mpxlite's tests, THIRD_PARTY/ (licences,
# notices, exact upstream revisions) and source/ (the engine build script, the
# mpx3 patch and tools/gmsh/).
#
# Usage:
#   tools/bundle/make_bundle.sh [--engine DIR] [--wheel FILE] [--out DIR]
#
#   --engine DIR   a tools/engine/build_engine.sh prefix (default $HOME/opt), or an
#                  exported copy of one with the same layout:
#                    getdp-3.5.0-mpx3/bin/getdp
#                    petsc-3.26.0-complex/arch-complex-opt/lib/
#                    petsc4py-3.26.0/petsc4py-3.26.0-cp3XX-*.whl
#                    occt-7.8.1/lib/
#                    gmsh-4.15.2/lib/libgmsh.so.4.15, gmsh-4.15.2/lib/gmsh.py,
#                    gmsh-4.15.2/bin/gmsh
#                    engine-info/     (written by build_engine.sh)
#   --wheel FILE   the mpxlite wheel to ship (default: built from this checkout);
#                  its version must be the version of this checkout
#   --out DIR      output directory (default: dist/ of this checkout)
#
# The bundle targets the Ubuntu release the engine was built on (recorded in
# engine-info/BUILD), not the release of the machine running this script. The
# libraries bundled are found by following the DT_NEEDED entries of getdp and
# of the Gmsh library through the non-system RUNPATH directories of the
# engine, recursively; what is not found there comes from the Ubuntu release
# (listed in BUILDINFO and THIRD_PARTY/README.md). The script fails if getdp or
# the Gmsh library loads a library of ParMETIS, (PT-)Scotch, STRUMPACK,
# SuperLU(_DIST) or Python, if the Gmsh library loads a library of the system
# other than the C/C++ and OpenMP runtimes and those of its graphical interface
# (X11, Xext, Xft, Xrender, fontconfig, OpenGL, GLU, libjpeg, libpng, zlib), if
# the gmsh executable loads anything but the Gmsh library and the C/C++
# runtimes, if the engine exports
# symbols of the removed components, if the petsc4py module loads anything but
# libpetsc of the engine and system libraries, if a bundled library is not
# covered by engine-info/VERSIONS.tsv, if getdp or Gmsh is not the expected
# build, or if anything in the bundle names a home directory or the build
# machine. On a machine running the target release it also checks the
# relocated binaries, libraries and modules with ldd and by loading them (no
# display needed).
#
# Needs: python3 with venv (build and patchelf go into a throw-away venv),
# binutils (readelf, nm, strip), git (the tests shipped are the files git tracks).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$REPO/tools/bundle"
ENGINE="$HOME/opt"
WHEEL=""
OUT="$REPO/dist"
while [ $# -gt 0 ]; do
    case "$1" in
        --engine) ENGINE="$2"; shift 2 ;;
        --wheel) WHEEL="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        -h|--help) sed -n '5,48p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
done
die() { echo "error: $*" >&2; exit 1; }

ENGINE_ID="getdp-3.5.0-mpx3+petsc-3.26.0+gmsh-4.15.2+occt-7.8.1"
EXPECTED_OPTIONS="64Bit Blas[petsc] Kernel Lapack[petsc] MPI NoFortran PETSc"
GMSH_VERSION="4.15.2"
OCCT_VERSION="7.8.1"
GMSH_SONAME="libgmsh.so.4.15"
FLTK_VERSION="1.3.11"
EXPECTED_GMSH_OPTIONS="64Bit ALGLIB[contrib] ANN[contrib] Bamg Blossom DIntegration Dlopen DomHex Eigen[contrib] Fltk Gmm[contrib] Hxt Jpeg Kbipack LinuxJoystick MathEx[contrib] Mesh Metis[contrib] Netgen Nii2mesh ONELAB ONELABMetamodel OpenCASCADE OpenCASCADE-CAF OpenGL OpenMP OptHom Parser Plugins Png Post QuadMeshingTools QuadTri Solver TetGen/BR TinyXML2[contrib] Voro++[contrib] WinslowUntangler Zlib tinyobjloader"
# The libraries of the system the Gmsh library may load (DT_NEEDED): the C/C++
# and OpenMP runtimes, and those of its graphical interface (FLTK is linked
# into it); the gmsh executable loads the library and the C/C++ runtimes.
GMSH_SYSTEM_LIBS='lib(c|m|dl|rt|pthread|stdc\+\+|gcc_s|gomp)\.so\.[0-9]+|ld-linux-x86-64\.so\.2'
GMSH_GUI_LIBS='lib(X11|Xext|Xft|Xrender|fontconfig|GL|GLU|jpeg|png16|z)\.so\.[0-9]+'

ENGINE="$(cd "$ENGINE" && pwd)"
INFO="$ENGINE/engine-info"
GETDP_BIN="$ENGINE/getdp-3.5.0-mpx3/bin/getdp"
GMSH_LIB="$ENGINE/gmsh-$GMSH_VERSION/lib/$GMSH_SONAME"
GMSH_PY="$ENGINE/gmsh-$GMSH_VERSION/lib/gmsh.py"
GMSH_EXE="$ENGINE/gmsh-$GMSH_VERSION/bin/gmsh"
PATCH="$REPO/tools/getdp/patches/getdp-3.5.0-mpx3.patch"
for f in "$GETDP_BIN" "$GMSH_LIB" "$GMSH_PY" "$GMSH_EXE" "$INFO/VERSIONS.tsv" "$INFO/BUILD" "$INFO/getdp-info.txt" \
         "$INFO/petscconf.h" "$INFO/gmsh-build-info.txt" "$INFO/gmsh-contrib.txt" "$INFO/gmsh-info.txt" \
         "$INFO/patch/getdp-3.5.0-mpx3.patch" "$INFO/licenses/getdp/GPL-3.0.txt"; do
    [ -e "$f" ] || die "missing: $f (engine not built by the current tools/engine/build_engine.sh? rerun it)"
done
# The records point to the corresponding source (no unmodified source is
# shipped): every field must be there and well formed (the same check as in
# tools/engine/build_engine.sh), and the pinned revisions must be the ones built.
check_versions() {
    awk -F'\t' -v dir="$(dirname "$1")" '
        function bad(m) { printf "%s: %s: %s\n", FILENAME, $1, m > "/dev/stderr"; err = 1 }
        NR == 1 { if ($0 != "component\tversion\tlicence\tupstream\trevision\tlicence_files") bad("bad header"); next }
        {
            n++; seen[$1] = 1
            if (NF != 6) bad("expected 6 fields, got " NF)
            for (i = 1; i <= 6; i++) if ($i ~ /^[[:space:]]*$/ || $i ~ /^ \(/) bad("field " i " empty")
            if ($2 !~ /^v?[0-9]+\.[0-9]+(\.[0-9]+)?([-.][0-9A-Za-z.]+)?( .*)?$/) bad("version \"" $2 "\"")
            if ($4 !~ /^https:\/\/[^ ]+$/) bad("upstream \"" $4 "\"")
            if ($1 == "getdp") re = "^commit [0-9a-f]{40}; mpx3 patch sha256 [0-9a-f]{64}$"
            else if ($1 == "petsc" || $1 == "petsc4py") re = "^commit [0-9a-f]{40} \\(tag v[0-9.]+\\)$"
            else if ($1 == "mumps") re = "^tarball sha256 [0-9a-f]{64}$"
            else if ($1 == "scalapack" || $1 == "metis") re = "^commit [0-9a-f]{40}$"
            else if ($1 == "opencascade" || $1 == "fltk" || $1 == "gmsh") re = "^tarball sha256 [0-9a-f]{64}; tag [A-Za-z0-9_.-]+ commit [0-9a-f]{40} \\(https://[^ ]+\\)$"
            else { bad("unknown component"); re = "." }
            if ($5 !~ re) bad("revision \"" $5 "\"")
            m = split($6, f, " ")
            for (i = 1; i <= m; i++) if (system("test -f \"" dir "/" f[i] "\"") != 0) bad("missing " f[i])
        }
        END {
            if (n != 9 || !seen["getdp"] || !seen["petsc"] || !seen["petsc4py"] || !seen["mumps"] || !seen["scalapack"] || !seen["metis"] || !seen["opencascade"] || !seen["fltk"] || !seen["gmsh"])
                bad("expected the rows getdp, petsc, petsc4py, mumps, scalapack, metis, opencascade, fltk, gmsh")
            exit err
        }' "$1"
}
check_versions "$INFO/VERSIONS.tsv" || die "invalid records in $INFO/VERSIONS.tsv: rerun tools/engine/build_engine.sh on this prefix"
for pin in "getdp	commit 864875cdfc652254f777311d58f510b38e508b56;" \
           "petsc	commit e6a25c9dbe75ac9d50c03f4462d8e39f0a0ce7ad " \
           "petsc4py	commit e6a25c9dbe75ac9d50c03f4462d8e39f0a0ce7ad " \
           "mumps	tarball sha256 659c9b57646b5a003ac618baa1faf9dd2044e46c732b3daaccbc7158003e1b46" \
           "opencascade	tarball sha256 7321af48c34dc253bf8aae3f0430e8cb10976961d534d8509e72516978aa82f5; tag V7_8_1 commit bd2a789f15235755ce4d1a3b07379a2e062fdc2e " \
           "fltk	tarball sha256 92805abc84505e3e7e27aec775ab0754ecb4182fe2d8ff2a9d568ccdcb2811ac; tag release-1.3.11 commit 702172a951a2bb4f25afe31008ef3fcbb5bfb92f " \
           "gmsh	tarball sha256 be3f66f225d27ba9fa014f07e83169285da8a051b0e8ab7103d88066b39bdd3e; tag gmsh_4_15_2 commit 657c8e915f60405e6cad0c8ec7faf812bfff1a60 "; do
    awk -F'\t' -v c="${pin%%	*}" -v r="${pin#*	}" '$1 == c && index($5, r) == 1 {ok = 1} END {exit !ok}' \
        "$INFO/VERSIONS.tsv" || die "VERSIONS.tsv: ${pin%%	*} is not the pinned revision (${pin#*	})"
done
cmp -s "$PATCH" "$INFO/patch/getdp-3.5.0-mpx3.patch" \
    || die "the engine was built with another mpx3 patch than tools/getdp/patches/ of this checkout"
field() { awk -v k="$1" '$1 == k {sub(/^[^ ]+ +/, ""); print; exit}' "$INFO/BUILD"; }
info() { awk -F': ' -v k="$1" '{sub(/ +$/, "", $1)} $1 == k {print $2; exit}' "$INFO/getdp-info.txt"; }
OS_ID="$(field os_id)"
OS_VER="$(field os_version)"
BUILD_PREFIX="$(field prefix)"
ENGINE_DATE="$(field built)"
PY_TAG="$(field python | cut -d' ' -f1)"
case "$PY_TAG" in cp3[0-9]*) ;; *) die "no Python tag in $INFO/BUILD: rerun tools/engine/build_engine.sh on this prefix" ;; esac
P4PY_WHEEL="$(ls "$ENGINE"/petsc4py-3.26.0/petsc4py-3.26.0-"$PY_TAG"-"$PY_TAG"-*.whl 2>/dev/null | head -n 1)"
[ -n "$P4PY_WHEEL" ] || die "no petsc4py wheel for $PY_TAG in $ENGINE/petsc4py-3.26.0: rerun tools/engine/build_engine.sh"
[ "$OS_ID" = ubuntu ] || die "the engine was built on '$OS_ID', not Ubuntu"
case "$OS_VER" in 24.04|26.04) ;; *) die "the engine was built on Ubuntu $OS_VER (supported: 24.04, 26.04)" ;; esac
[ -n "$BUILD_PREFIX" ] || die "no build prefix in $INFO/BUILD"
HOST_VER="$(. /etc/os-release && { [ "$ID" = ubuntu ] && echo "$VERSION_ID" || true; })"

# The engine must be exactly the supported build.
case "$(info Version)" in 3.5.0-mpx3*) ;; *) die "getdp version '$(info Version)' is not 3.5.0-mpx3" ;; esac
[ "$(info 'Build options')" = "$EXPECTED_OPTIONS" ] \
    || die "getdp build options '$(info 'Build options')', expected '$EXPECTED_OPTIONS'"
[ "$(info 'PETSc version')" = "3.26.0 (complex arithmetic)" ] || die "getdp PETSc: '$(info 'PETSc version')'"
[ "$(info 'Packaged by')" = mpxlite ] || die "getdp 'Packaged by: $(info 'Packaged by')' (expected mpxlite)"
[ "$(info 'Build host')" = unspecified ] || die "getdp 'Build host: $(info 'Build host')' (expected unspecified)"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "=== wheel"
python3 -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install -q build patchelf wheel
PATCHELF="$WORK/venv/bin/patchelf"
mkdir -p "$WORK/wheels"
SRC_VERSION="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$REPO/src/mpxlite/__init__.py")"
REV="$(git -C "$REPO" rev-parse HEAD)"
if [ -z "$(git -C "$REPO" status --porcelain)" ]; then
    DIRTY=""
else
    DIRTY=" (with uncommitted changes)"
    echo "warning: $REPO has uncommitted changes (they are bundled as they are)" >&2
fi
if [ -n "$WHEEL" ]; then
    cp "$WHEEL" "$WORK/wheels/"
else
    "$WORK/venv/bin/python" -m build -q --wheel --outdir "$WORK/wheels" "$REPO" >/dev/null
fi
VERSION="$(basename "$WORK"/wheels/mpxlite-*.whl | cut -d- -f2)"
[ "$VERSION" = "$SRC_VERSION" ] || die "wheel version $VERSION, but this checkout is $SRC_VERSION"
WHL="mpxlite-$VERSION-py3-none-any.whl"
[ -f "$WORK/wheels/$WHL" ] || die "expected a pure-Python wheel $WHL"
NAME="mpxlite-$VERSION-ubuntu$OS_VER-x86_64"
B="$WORK/$NAME"
mkdir -p "$B/engine/bin" "$B/engine/lib" "$B/THIRD_PARTY"
mv "$WORK/wheels" "$B/wheels"
echo "    $WHL"

# Gmsh: the build reports the expected version, OpenCASCADE and options.
ginfo() { awk -F': ' -v k="$1" '$1 == k {print $2; exit}' "$INFO/gmsh-build-info.txt"; }
[ "$(ginfo Version)" = "$GMSH_VERSION" ] || die "Gmsh version '$(ginfo Version)', expected $GMSH_VERSION"
[ "$(ginfo 'OCC version')" = "$OCCT_VERSION" ] || die "Gmsh on OpenCASCADE '$(ginfo 'OCC version')', expected $OCCT_VERSION"
[ "$(ginfo 'Build options')" = "$EXPECTED_GMSH_OPTIONS" ] \
    || die "Gmsh build options '$(ginfo 'Build options')', expected '$EXPECTED_GMSH_OPTIONS'"
[ "$(ginfo 'Packaged by')" = mpxlite ] || die "Gmsh 'Packaged by: $(ginfo 'Packaged by')' (expected mpxlite)"
[ "$(ginfo 'Build host')" = unspecified ] || die "Gmsh 'Build host: $(ginfo 'Build host')' (expected unspecified)"
[ "$(ginfo 'FLTK version')" = "$FLTK_VERSION" ] || die "Gmsh on FLTK '$(ginfo 'FLTK version')', expected $FLTK_VERSION"
# The executable is the same build (gmsh -info, recorded by build_engine.sh).
[ "$(awk -F' *: ' '$1 == "Version" {print $2}' "$INFO/gmsh-info.txt")" = "$GMSH_VERSION" ] \
    && [ "$(awk -F' *: ' '$1 == "Build options" {print $2}' "$INFO/gmsh-info.txt")" = "$EXPECTED_GMSH_OPTIONS" ] \
    || die "gmsh -info of the engine is not the expected build ($INFO/gmsh-info.txt)"

echo "=== engine (closure of the DT_NEEDED of getdp and of the Gmsh library through the engine RUNPATHs)"
needed() { readelf -d "$1" | sed -n 's/.*(NEEDED).*\[\(.*\)\]$/\1/p'; }
# Non-system RUNPATH/RPATH directories of ELF file $1, mapped from the build
# prefix to $ENGINE (exported engines).
enginedirs() {
    local d o
    o="$(dirname "$1")"
    readelf -d "$1" | sed -n 's/.*(R\(UN\)\{0,1\}PATH).*\[\(.*\)\]$/\2/p' | tr ':' '\n' | while read -r d; do
        d="${d//\$ORIGIN/$o}"
        case "$d" in ""|/usr/*|/lib|/lib/*|/lib64*) continue ;; esac
        [ "${d#"$BUILD_PREFIX"/}" != "$d" ] && d="$ENGINE/${d#"$BUILD_PREFIX"/}"
        if [ -d "$d" ]; then echo "$d"; fi
    done
}
# closure ROOT ARRAY: the libraries ELF file ROOT loads, recursively; those
# found in the engine go into BUNDLED (name -> file), the others into SYSTEM
# and into the array named ARRAY.
declare -A BUNDLED=() SYSTEM=()
closure() {
    local -n sys_of="$2"
    local queue=("$1") elf so d found
    local -A seen=()
    while [ "${#queue[@]}" -gt 0 ]; do
        elf="${queue[0]}"; queue=("${queue[@]:1}")
        mapfile -t dirs < <(enginedirs "$elf")
        for so in $(needed "$elf"); do
            if [ -n "${seen[$so]:-}" ]; then continue; fi
            seen[$so]=1
            found=""
            for d in "${dirs[@]}"; do
                if [ -e "$d/$so" ]; then found="$(readlink -f "$d/$so")"; break; fi
            done
            if [ -n "$found" ]; then BUNDLED[$so]="$found"; queue+=("$found"); else SYSTEM[$so]=1; sys_of+=("$so"); fi
        done
    done
}
GETDP_SYS=()
GMSH_SYS=()
closure "$GETDP_BIN" GETDP_SYS
closure "$(readlink -f "$GMSH_LIB")" GMSH_SYS
bad="$(printf '%s\n' "${!BUNDLED[@]}" "${!SYSTEM[@]}" | grep -iE 'parmetis|scotch|esmumps|strumpack|superlu|python' || true)"
[ -z "$bad" ] || die "the engine loads libraries of a removed component or of Python: $bad"
# Of the system, Gmsh loads the runtimes and the libraries of its graphical
# interface only (FLTK is linked into it).
bad="$(printf '%s\n' "${GMSH_SYS[@]}" | grep -v -x -E "$GMSH_SYSTEM_LIBS|$GMSH_GUI_LIBS" || true)"
[ -z "$bad" ] || die "the Gmsh library loads libraries it must not load: $bad"
bad="$(needed "$GMSH_EXE" | grep -v -x -E "$GMSH_SONAME|$GMSH_SYSTEM_LIBS" || true)"
[ -z "$bad" ] || die "the gmsh executable loads libraries it must not load: $bad"
install -m 755 "$GETDP_BIN" "$B/engine/bin/getdp"
install -m 755 "$GMSH_EXE" "$B/engine/bin/gmsh"
for so in "${!BUNDLED[@]}"; do install -m 644 "${BUNDLED[$so]}" "$B/engine/lib/$so"; done
install -m 644 "$(readlink -f "$GMSH_LIB")" "$B/engine/lib/$GMSH_SONAME"
# The Gmsh Python module (unmodified): in engine/gmsh/ of the installation,
# beside the Gmsh programs of mpxlite, it loads ../lib/libgmsh.so.4.15.
mkdir -p "$B/engine/gmsh"
install -m 644 "$GMSH_PY" "$B/engine/gmsh/gmsh.py"
grep -q "^GMSH_API_VERSION = \"$GMSH_VERSION\"$" "$B/engine/gmsh/gmsh.py" || die "gmsh.py is not the module of Gmsh $GMSH_VERSION"
grep -q "libname = \"$GMSH_SONAME\"" "$B/engine/gmsh/gmsh.py" || die "gmsh.py does not load $GMSH_SONAME"
# PETSc always compiles its interface functions (MatSTRUMPACKSet*, ...) and the
# DMPlex partitioner stubs; the symbols below exist only when the package
# itself is built in (PETSc implementation, MUMPS ordering wrappers).
bad="$(for f in "$B"/engine/bin/getdp "$B"/engine/lib/*; do nm -D --defined-only "$f" 2>/dev/null || true; done \
       | awk '{print $NF}' \
       | grep -E '^(MatPartitioningCreate_(ParMETIS|PTScotch)|MatMeshToCellGraph_ParMETIS|MatSuperluDist|Petsc_Superlu_dist|MatGetFactor_.*_(superlu_dist|strumpack)|mumps_(parmetis|scotch)|__mumps_ana_ord_wrappers_MOD_mumps_scotch|(SCOTCH|METIS_V3|ParMETIS)_|Py_Initialize)' \
       | sort -u | head -20 || true)"
[ -z "$bad" ] || die "the engine contains symbols of a removed component:"$'\n'"$bad"
bad="$(grep -E '^#define PETSC_HAVE_(PARMETIS|PTSCOTCH|STRUMPACK|SUPERLU|SUPERLU_DIST|SCOTCH) ' "$INFO/petscconf.h" || true)"
[ -z "$bad" ] || die "PETSc was configured with a removed component: $bad"
grep -q '^#define PETSC_HAVE_MUMPS 1' "$INFO/petscconf.h" || die "PETSc was configured without MUMPS"
grep -q '^#define PETSC_HAVE_METIS 1' "$INFO/petscconf.h" || die "PETSc was configured without METIS"

# Each bundled library must belong to a component recorded in VERSIONS.tsv.
declare -A COMP=([getdp]=1 [gmsh]=1 [fltk]=1)
for so in "${!BUNDLED[@]}"; do
    case "$so" in
        libpetsc.so*) COMP[petsc]=1 ;;
        libscalapack.so*) COMP[scalapack]=1 ;;
        libmetis.so*) COMP[metis]=1 ;;
        lib[sdcz]mumps*|libmumps_common*|libpord*) COMP[mumps]=1 ;;
        libTK*.so.*) COMP[opencascade]=1 ;;
        *) die "bundled library $so is not a known engine component (add it here and to VERSIONS.tsv in build_engine.sh)" ;;
    esac
done
# MUMPS is linked statically into libpetsc.
if [ "$(nm -D --defined-only "$B"/engine/lib/libpetsc.so* | grep -c ' zmumps_c$')" -gt 0 ]; then COMP[mumps]=1; fi
# (FLTK is linked statically into the Gmsh library.)
for c in getdp petsc mumps scalapack metis opencascade fltk gmsh; do
    [ -n "${COMP[$c]:-}" ] || die "component $c is not in the engine"
    grep -q "^$c"$'\t' "$INFO/VERSIONS.tsv" || die "component $c is not recorded in $INFO/VERSIONS.tsv"
    [ -d "$INFO/licenses/$c" ] || die "no licence files for $c in $INFO/licenses"
done

strip --strip-unneeded "$B/engine/bin/getdp" "$B/engine/bin/gmsh" "$B"/engine/lib/*
"$PATCHELF" --set-rpath '$ORIGIN/../lib' "$B/engine/bin/getdp"
"$PATCHELF" --set-rpath '$ORIGIN/../lib' "$B/engine/bin/gmsh"
for f in "$B"/engine/lib/*; do "$PATCHELF" --set-rpath '$ORIGIN' "$f"; done
# PETSc records the build machine in libpetsc; build_engine.sh neutralises it.
compiled_on="$(grep -a -o 'Libraries compiled on [0-9-]* [0-9:]* on [^[:space:]]*' "$B"/engine/lib/libpetsc.so* | head -1 || true)"
case "$compiled_on" in
    *" on unspecified") ;;
    *) die "libpetsc names its build machine ('$compiled_on'): rerun tools/engine/build_engine.sh on this prefix" ;;
esac
if [ "$HOST_VER" = "$OS_VER" ]; then
    for f in "$B/engine/bin/getdp" "$B/engine/bin/gmsh" "$B/engine/lib/$GMSH_SONAME"; do
        if env -i ldd "$f" | grep -E "not found|$BUILD_PREFIX|$ENGINE"; then
            die "relocation check failed: $f"
        fi
    done
    # The module, unmodified, finds the library of engine/lib, which loads
    # OpenCASCADE from engine/lib: what an installation does.
    # (-B: no compiled cache written into the bundle.)
    env -i python3 -I -B - "$B/engine" "$GMSH_VERSION" <<'PYEOF' || die "the Gmsh module does not load the library of engine/lib"
import os, sys
engine = os.path.realpath(sys.argv[1])
sys.path.insert(0, os.path.join(engine, "gmsh"))
import gmsh
assert os.path.realpath(gmsh.__file__) == os.path.join(engine, "gmsh", "gmsh.py"), gmsh.__file__
gmsh.initialize(readConfigFiles=False)
gmsh.option.setNumber("General.Terminal", 0)
assert gmsh.option.getString("General.Version") == sys.argv[2]
gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
assert gmsh.model.mesh.getNodes()[0].size > 0
gmsh.finalize()
loaded = sorted({line.split()[-1] for line in open("/proc/self/maps") if "/libgmsh" in line or "/libTK" in line})
lib = os.path.join(engine, "lib")
assert any("/libgmsh" in p for p in loaded), loaded
bad = [p for p in loaded if os.path.dirname(os.path.realpath(p)) != lib]
assert not bad, bad
print(f"    Gmsh {sys.argv[2]}: the module loads {len(loaded)} libraries, all from engine/lib")
PYEOF
    # The relocated executable runs without a display, in batch.
    info="$(env -i "$B/engine/bin/gmsh" -info 2>&1)" || die "engine/bin/gmsh -info failed: $info"
    grep -q "^Version *: $GMSH_VERSION$" <<< "$info" && grep -q "^FLTK version *: $FLTK_VERSION$" <<< "$info" \
        || die "engine/bin/gmsh -info: $info"
    echo "    engine/bin/gmsh -info: Gmsh $GMSH_VERSION, FLTK $FLTK_VERSION (no display)"
    echo "    ldd check on this Ubuntu $HOST_VER: everything resolves in engine/lib or the system"
else
    echo "    (this machine is not Ubuntu $OS_VER: ldd relocation check skipped)"
fi
SYS_LIBS="$(printf '%s\n' "${!SYSTEM[@]}" | sort | tr '\n' ' ' | sed 's/ $//')"
GMSH_SYS_LIBS="$(printf '%s\n' "${GMSH_SYS[@]}" | sort | tr '\n' ' ' | sed 's/ $//')"
ENGINE_LIBS="$(ls "$B/engine/lib" | tr '\n' ' ' | sed 's/ $//')"
echo "    getdp, gmsh + $(ls "$B/engine/lib" | wc -l) libraries: $ENGINE_LIBS"
echo "    from Ubuntu $OS_VER: $SYS_LIBS"
echo "    (of which the Gmsh library loads: $GMSH_SYS_LIBS)"

echo "=== petsc4py (relocated to the engine/lib of the installation)"
# In an installation the module is at
#   <home>/venv/lib/python3.X/site-packages/petsc4py/lib/<arch>/PETSc.*.so
# and libpetsc at <home>/engine/lib: seven directories up, then engine/lib.
P4PY_RUNPATH='$ORIGIN/../../../../../../../engine/lib'
P4PY_NAME="$(basename "$P4PY_WHEEL")"
"$WORK/venv/bin/python" -m wheel unpack -d "$WORK/p4py" "$P4PY_WHEEL" >/dev/null
P4PY_TREE="$(ls -d "$WORK"/p4py/petsc4py-*)"
mapfile -t P4PY_MODULES < <(find "$P4PY_TREE" -name '*.so')
[ "${#P4PY_MODULES[@]}" = 1 ] || die "expected one extension module in $P4PY_NAME, found ${#P4PY_MODULES[@]}"
P4PY_SO="${P4PY_MODULES[0]}"
case "${P4PY_SO#"$P4PY_TREE"/}" in
    petsc4py/lib/*/PETSc.*.so) ;;
    *) die "unexpected location of the petsc4py module: ${P4PY_SO#"$P4PY_TREE"/}" ;;
esac
bad="$(needed "$P4PY_SO" | grep -v -x -E 'libpetsc\.so\.[0-9.]+|libmpi\.so\.[0-9]+|libc\.so\.6|libm\.so\.6|libgcc_s\.so\.1|libpthread\.so\.0' || true)"
[ -z "$bad" ] || die "the petsc4py module loads unexpected libraries: $bad"
P4PY_PETSC="$(needed "$P4PY_SO" | grep -E '^libpetsc\.so')"
[ -e "$B/engine/lib/$P4PY_PETSC" ] || die "the petsc4py module needs $P4PY_PETSC, which is not in engine/lib"
strip --strip-unneeded "$P4PY_SO"
"$PATCHELF" --set-rpath "$P4PY_RUNPATH" "$P4PY_SO"
# petsc.cfg names the PETSc build directory: only the architecture is used
# to find the module; the directory is replaced by a neutral value.
P4PY_CFG="$P4PY_TREE/petsc4py/lib/petsc.cfg"
[ -f "$P4PY_CFG" ] || die "no petsc4py/lib/petsc.cfg in $P4PY_NAME"
sed -i -E 's|^(PETSC_DIR[[:space:]]*=).*|\1 engine|' "$P4PY_CFG"
grep -q -E '^PETSC_ARCH[[:space:]]*=[[:space:]]*[^[:space:]]+' "$P4PY_CFG" || die "no PETSC_ARCH in petsc4py/lib/petsc.cfg"
bad="$(grep -r -l -a -E '/home/|/root/' "$P4PY_TREE" || true)"
[ -z "$bad" ] || die "a home directory path is in the petsc4py wheel: $bad"
"$WORK/venv/bin/python" -m wheel pack -d "$B/wheels" "$P4PY_TREE" >/dev/null
[ -f "$B/wheels/$P4PY_NAME" ] || die "repacking did not give $P4PY_NAME"
if [ "$HOST_VER" = "$OS_VER" ]; then
    # The layout of an installation, to check what the module resolves to.
    STAGE="$WORK/stage"
    SITE="$STAGE/venv/lib/python3.${PY_TAG#cp3}/site-packages"
    mkdir -p "$SITE" "$STAGE/engine"
    cp -a "$B/engine/lib" "$STAGE/engine/lib"
    (cd "$SITE" && "$WORK/venv/bin/python" -m zipfile -e "$B/wheels/$P4PY_NAME" .)
    STAGED_SO="$(find "$SITE" -name 'PETSc.*.so')"
    resolved="$(env -i ldd "$STAGED_SO")"
    if grep -E "not found|$BUILD_PREFIX|$ENGINE" <<< "$resolved"; then die "petsc4py relocation check failed"; fi
    found="$(awk -v so="$P4PY_PETSC" '$1 == so && $2 == "=>" {print $3}' <<< "$resolved")"
    [ -n "$found" ] && [ "$(readlink -f "$found")" = "$(readlink -f "$STAGE/engine/lib/$P4PY_PETSC")" ] \
        || die "the petsc4py module does not resolve $P4PY_PETSC to engine/lib: $resolved"
    echo "    ldd check: $P4PY_PETSC resolves to engine/lib of the installation"
else
    echo "    (this machine is not Ubuntu $OS_VER: ldd relocation check skipped)"
fi
echo "    $P4PY_NAME"

echo "=== installer, smoke test, Gmsh programs, tests, source/"
install -m 755 "$HERE/install.sh" "$B/install.sh"
install -m 644 "$HERE/README-INSTALL.md" "$B/README-INSTALL.md"
install -m 644 "$HERE/mpxlite_smoke.py" "$B/mpxlite_smoke.py"
echo "ubuntu $OS_VER" > "$B/TARGET"
# The Gmsh programs of mpxlite with their notice and the GPL text, installed
# into engine/gmsh/ of the installation: the toolkit mpxlite_gmsh/, the jobs
# and the tests of the toolkit (the files git tracks).
GMSH_FILES=(README.md COPYING)
GMSH_DIRS=(mpxlite_gmsh jobs tests)
mkdir -p "$B/gmsh"
for f in "${GMSH_FILES[@]}"; do install -m 644 "$REPO/tools/gmsh/$f" "$B/gmsh/$f"; done
for d in "${GMSH_DIRS[@]}"; do
    [ -n "$(cd "$REPO" && git ls-files "tools/gmsh/$d")" ] || die "tools/gmsh/$d has no tracked file"
done
(cd "$REPO/tools/gmsh" && git ls-files -z "${GMSH_DIRS[@]}" | tar --null -T - -cf -) | tar -C "$B/gmsh" -xf -
(cd "$REPO" && git ls-files -z tests | tar --null -T - -cf -) | tar -C "$B" -xf -
# The pytest configuration of pyproject.toml, as tests/pytest.ini.
python3 - "$REPO/pyproject.toml" > "$B/tests/pytest.ini" <<'EOF'
import sys, tomllib
opts = tomllib.load(open(sys.argv[1], "rb"))["tool"]["pytest"]["ini_options"]
print("# Copyright (c) 2026 MPhysX\n# SPDX-License-Identifier: MIT\n")
print("# pytest settings of mpxlite (from its pyproject.toml), for the tests of a bundle.")
print("[pytest]")
for key, val in opts.items():
    if key == "testpaths":
        continue
    if isinstance(val, list):
        print(f"{key} =\n" + "\n".join(f"    {v}" for v in val))
    else:
        print(f"{key} = {str(val).lower() if isinstance(val, bool) else val}")
EOF
# The engine build script and the mpx3 patch at the paths the script expects;
# tools/gmsh/ (the Gmsh programs) at its path in the repository.
mkdir -p "$B/source/tools/engine" "$B/source/tools/getdp/patches" "$B/source/tools/gmsh"
install -m 755 "$REPO/tools/engine/build_engine.sh" "$B/source/tools/engine/build_engine.sh"
install -m 644 "$REPO"/tools/getdp/patches/* "$B/source/tools/getdp/patches/"
(cd "$REPO/tools/gmsh" && git ls-files -z . | tar --null -T - -cf -) | tar -C "$B/source/tools/gmsh" -xf -
install -m 644 "$HERE/SOURCE-README.md" "$B/source/README.md"

echo "=== THIRD_PARTY"
T="$B/THIRD_PARTY"
PATCH_SUM="$(sha256sum "$PATCH" | cut -d' ' -f1)"
grep -q "mpx3 patch sha256 $PATCH_SUM" "$INFO/VERSIONS.tsv" \
    || die "engine-info/VERSIONS.tsv does not record the mpx3 patch of this checkout"
for c in getdp petsc petsc4py mumps scalapack metis opencascade fltk gmsh; do
    mkdir -p "$T/$c"
    cp -r "$INFO/licenses/$c"/. "$T/$c/"
done
mv "$T/getdp/GPL-3.0.txt" "$T/GPL-3.0.txt"
mkdir -p "$T/mpxlite"
cp "$REPO/LICENSE" "$REPO/COPYRIGHT.md" "$T/mpxlite/"
GETDP_COMMIT="$(awk -F'\t' '$1 == "getdp" {print $5}' "$INFO/VERSIONS.tsv" | sed 's/^commit \([0-9a-f]*\).*/\1/')"
row() { awk -F'\t' -v c="$1" -v k="$2" '$1 == c {print $k}' "$INFO/VERSIONS.tsv"; }
GMSH_SRC_URL="$(row gmsh 4)"
OCCT_SRC_URL="$(row opencascade 4)"
GMSH_SRC_SUM="$(row gmsh 5 | sed 's/^tarball sha256 \([0-9a-f]*\);.*/\1/')"
OCCT_SRC_SUM="$(row opencascade 5 | sed 's/^tarball sha256 \([0-9a-f]*\);.*/\1/')"
FLTK_SRC_URL="$(row fltk 4)"
FLTK_SRC_SUM="$(row fltk 5 | sed 's/^tarball sha256 \([0-9a-f]*\);.*/\1/')"
ENGINE_SOURCES_REPO="$(sed -n 's/^ENGINE_SOURCES_REPO="\(.*\)"$/\1/p' "$REPO/tools/engine/build_engine.sh")"
[ -n "$ENGINE_SOURCES_REPO" ] || die "no ENGINE_SOURCES_REPO= in tools/engine/build_engine.sh"
grep -qxF "ENGINE_SOURCES_FILES=\"\$ENGINE_SOURCES_REPO/raw/main\"" "$REPO/tools/engine/build_engine.sh" \
    || die "tools/engine/build_engine.sh does not take the source archives from the branch main of ENGINE_SOURCES_REPO"
GETDP_COPY_GIT="$(sed -n 's/^GETDP_COPY_GIT="\(.*\)"$/\1/p' "$REPO/tools/engine/build_engine.sh")"
[ -n "$GETDP_COPY_GIT" ] || die "no GETDP_COPY_GIT= in tools/engine/build_engine.sh"
# The contributed code of Gmsh compiled into the library (engine-info/gmsh-contrib.txt)
# and its licence, as its licence files and notices state it.
declare -A CONTRIB=(
    [ALGLIB]="ALGLIB (Sergey Bochkanov)|GPL-2.0-or-later"
    [ANN]="ANN (University of Maryland, S. Arya, D. Mount)|LGPL-2.1"
    [bamg]="Bamg (from FreeFem++, F. Hecht)|LGPL-2.1-or-later"
    [blossom]="Blossom IV and Concorde (W. Cook et al.)|permission of the authors for use within Gmsh, academic and non-academic"
    [DiscreteIntegration]="DiscreteIntegration (UCL)|MIT-style permissive notice"
    [domhex]="domhex (Gmsh)|the Gmsh licence"
    [eigen]="Eigen (headers)|MPL-2.0"
    [gmm]="Gmm++ (Y. Renard; headers)|LGPL-3.0-or-later with the GCC Runtime Library Exception 3.1"
    [HighOrderMeshOptimizer]="HighOrderMeshOptimizer (UCLouvain-ULiege)|MIT-style permissive notice"
    [hxt]="HXT (Universite catholique de Louvain)|GPL-2.0-or-later with the exception of its LICENSE.txt"
    [kbipack]="KBIPack (S. Suuriniemi)|GPL-2.0"
    [MathEx]="MathEx (SSCILIB, S. Massago)|LGPL-2.1-or-later with static linking exceptions"
    [MeshOptimizer]="MeshOptimizer (UCLouvain-ULiege)|MIT-style permissive notice"
    [MeshQualityOptimizer]="MeshQualityOptimizer (UCLouvain-ULiege)|MIT-style permissive notice"
    [metis]="METIS (Regents of the University of Minnesota)|Apache-2.0"
    [Netgen]="Netgen (J. Schoeberl)|LGPL-2.1"
    [nii2mesh]="nii2mesh (C. Rorden)|BSD-2-Clause"
    [onelab]="ONELAB (Universite de Liege, Universite catholique de Louvain)|MIT-style permissive notice"
    [QuadMeshingTools]="QuadMeshingTools (Gmsh)|the Gmsh licence"
    [QuadTri]="QuadTri (T. Strickler, copyright transferred to C. Geuzaine)|the Gmsh licence"
    [tinyobjloader]="tinyobjloader (S. Fujita)|MIT"
    [tinyxml2]="TinyXML-2 (L. Thomason)|Zlib"
    [voro++]="Voro++ (Lawrence Berkeley National Laboratory)|BSD-3-Clause-LBNL"
    [WinslowUntangler]="WinslowUntangler (Gmsh)|the Gmsh licence"
)
CONTRIB_ROWS=""
while read -r c; do
    [ -n "${CONTRIB[$c]:-}" ] || die "Gmsh contrib/$c is compiled in but has no licence entry here: read its licence and add it to CONTRIB"
    files="$(cd "$T/gmsh" && find "contrib/$c" -type f | LC_ALL=C sort | sed 's|^|`gmsh/|; s|$|`|' | paste -sd, - | sed 's/,/, /g')"
    [ -n "$files" ] || die "no licence or notice file for Gmsh contrib/$c"
    CONTRIB_ROWS+="| ${CONTRIB[$c]%%|*} (in Gmsh, contrib/$c) | ${CONTRIB[$c]#*|} | $files |"$'\n'
done < "$INFO/gmsh-contrib.txt"
TODAY="$(date -u +%Y-%m-%d)"
{
    cat <<EOF
<!-- Copyright (c) 2026 MPhysX -->
<!-- SPDX-License-Identifier: MIT -->

# Third-party components of $NAME

mpxlite $VERSION is under the MIT licence (\`mpxlite/LICENSE\`, \`mpxlite/COPYRIGHT.md\`).
It runs the engine below as separate programs and does not link to it.

## The engine in engine/

\`engine/bin/getdp\`, \`engine/bin/gmsh\`, the shared libraries in \`engine/lib\`
(among them the Gmsh library \`libgmsh.so.4.15\`, which contains FLTK, and the
OpenCASCADE libraries \`libTK*\`) and the Gmsh Python module
\`engine/gmsh/gmsh.py\` were built from the upstream sources listed below with
\`source/tools/engine/build_engine.sh\`, on $(field os), on $ENGINE_DATE. Apart
from stripped symbols and a rewritten library search path (RUNPATH), the
programs and libraries are not modified; \`gmsh.py\` is the file of the Gmsh
source, unmodified.

**Modified program.** \`engine/bin/getdp\` is a modified version of GetDP 3.5.0:
upstream GetDP (tag getdp_3_5_0, commit $GETDP_COMMIT) with the mpx3 patch
\`source/tools/getdp/patches/getdp-3.5.0-mpx3.patch\` applied (sha256
$PATCH_SUM; modifications Copyright (c) 2026 MPhysX, GPL-2.0-or-later, see
\`source/tools/getdp/patches/README.md\`). This modified version was built on
$ENGINE_DATE. The patch is the only modification of third-party code in this
bundle. (Statement dated $TODAY.)

**Licence of the getdp binary.** GetDP and the mpx3 patch are GPL-2.0-or-later
(\`getdp/LICENSE.txt\`). The getdp binary is linked with METIS, which is under the
Apache License 2.0; Apache-2.0 is compatible with version 3 of the GNU GPL, not
with version 2. The getdp binary distributed here is therefore under the **GNU
General Public License version 3 or later (GPL-3.0-or-later)**; the text is in
\`GPL-3.0.txt\`. The libraries keep their own licences (table below).

**Gmsh, a GPL binary.** \`engine/lib/libgmsh.so.4.15\` is a binary of Gmsh
$GMSH_VERSION, unmodified, built from the Gmsh source release (archive
\`$(basename "$GMSH_SRC_URL")\`, sha256 $GMSH_SRC_SUM, the tree of the
$(row gmsh 5 | sed 's/^[^;]*; //')) with its graphical interface,
\`engine/bin/gmsh\` the Gmsh program of the same build (linked to that
library), and \`engine/gmsh/gmsh.py\` its Python module. Gmsh is Copyright (C) 1997-2025 C.
Geuzaine and J.-F. Remacle, under the **GNU General Public License version 2
or later**, with the exception stated in its \`LICENSE.txt\` (\`gmsh/LICENSE.txt\`,
which permits its combination with Netgen, METIS and OpenCASCADE under their
own licences). This bundle distributes that binary under those terms. The
library contains the contributed code listed below (\`gmsh/CREDITS.txt\`,
\`gmsh/contrib/\`), under its own licences. Its authors ask that work using Gmsh
cite: C. Geuzaine and J.-F. Remacle, "Gmsh: a three-dimensional finite element
mesh generator with built-in pre- and post-processing facilities",
International Journal for Numerical Methods in Engineering 79(11), 1309-1331,
2009.

**FLTK.** The graphical interface of Gmsh is based in part on the work of the
FLTK project (https://www.fltk.org). FLTK $FLTK_VERSION, unmodified, built from
its source release (archive \`$(basename "$FLTK_SRC_URL")\`, sha256
$FLTK_SRC_SUM, the tree of the $(row fltk 5 | sed 's/^[^;]*; //')) as static
libraries, is linked into \`engine/lib/libgmsh.so.4.15\`. FLTK is under the GNU
Library General Public License version 2 with the FLTK exceptions
(\`fltk/COPYING\`), whose exception 3 states that static linking to FLTK does not
constitute a derivative work and does not require the source of the program
linked with it; the notices of its source files are in
\`fltk/NOTICES-from-sources.txt\`, its credits in \`fltk/CREDITS\`.

**OpenCASCADE.** The \`libTK*\` libraries are OpenCASCADE Technology
$OCCT_VERSION, unmodified, built from its source release (archive
\`$(basename "$OCCT_SRC_URL")\`, sha256 $OCCT_SRC_SUM, the tree of the
$(row opencascade 5 | sed 's/^[^;]*; //')) with the modules Gmsh uses, under
the GNU Lesser General Public License version 2.1 with the Open CASCADE
exception (\`opencascade/LICENSE_LGPL_21.txt\`,
\`opencascade/OCCT_LGPL_EXCEPTION.txt\`). They are shared libraries, loaded by the
Gmsh library; they can be replaced by other builds of the same version.

**petsc4py.** \`wheels/$P4PY_NAME\` holds the Python bindings of PETSc, from the
PETSc source tree at the commit below, compiled against the libpetsc of
\`engine/lib\` for the Python of Ubuntu $OS_VER ($PY_TAG). Its extension module is
stripped and its library search path points to the \`engine/lib\` of the
installation; \`petsc4py/lib/petsc.cfg\` names no build directory. It is under
the BSD-2-Clause licence of PETSc (\`petsc4py/LICENSE.rst\`).

**Corresponding source.** The corresponding source of the getdp binary, of the
Gmsh library and module and of the bundled libraries is the upstream source at
the exact revisions below (git commit, or sha256 of the source archive), the
mpx3 patch and the build script (\`source/\`), which reproduces the build (see
\`source/README.md\`). No source code of unmodified components is included in
this bundle. A copy of the source archives of the components used unmodified,
the Gmsh, FLTK and OpenCASCADE archives among them, under the same file names
and checksums, is kept at <$ENGINE_SOURCES_REPO>; GetDP, with its history and
the mpx3 patch (branch mpx3), at <${GETDP_COPY_GIT%.git}>.

| Component | Version | Licence | Upstream | Commit / checksum | Licence files (in THIRD_PARTY/) |
|---|---|---|---|---|---|
EOF
    for c in getdp petsc petsc4py mumps scalapack metis opencascade fltk gmsh; do
        awk -F'\t' -v c="$c" '$1 == c {
            files = ""; n = split($6, f, " ")
            for (i = 1; i <= n; i++) {
                sub(/^licenses\//, "", f[i])
                if (f[i] == "getdp/GPL-3.0.txt") f[i] = "GPL-3.0.txt"
                if (f[i] ~ /^gmsh\/contrib\//) continue
                files = files (files != "" ? ", " : "") "`" f[i] "`"
            }
            if (c == "gmsh") files = files ", `gmsh/contrib/` (contributed code, table below)"
            note = (c == "mumps") ? " (linked statically into libpetsc)" : ""
            if (c == "petsc4py") note = " (Python bindings of PETSc, in wheels/)"
            if (c == "gmsh") note = " (library, program and Python module)"
            if (c == "fltk") note = " (static, inside the Gmsh library)"
            printf "| %s%s | %s | %s | %s | %s | %s |\n", c, note, $2, $3, $4, $5, files }' "$INFO/VERSIONS.tsv"
    done
    cat <<EOF

About the licence files: \`getdp/CREDITS.txt\` carries the notices of the avl,
KissFFT and picojson code compiled into GetDP, and
\`getdp/NOTICE-onelab-gmshsocket.txt\` those of its ONELAB and Gmsh socket
headers. \`mumps/LICENSE\` says which parts of MUMPS are not CeCILL-C (full
text: \`mumps/CeCILL-C_V1-en.txt\`, \`mumps/CeCILL-C_V1-fr.txt\`): the PORD
ordering (public domain, \`mumps/PORD-README\`) and routines derived from AMD
and LAPACK (BSD-3-Clause, \`mumps/NOTICE-BSD-3-Clause.txt\`).
\`metis/LICENSE.txt\` is the METIS notice, \`metis/Apache-2.0.txt\` the licence
text (the METIS tree has no NOTICE file). \`gmsh/CREDITS.txt\` carries the
notices of the code of other authors in the Gmsh sources (AVL trees, picojson,
nanoflann, libOL, robin_hood hashing, TetGen/BR boundary recovery, relicensed
for Gmsh), \`gmsh/contrib/<name>/\` the licence and notice files of each
contributed code compiled in, with the leading comments of its source files
that state a copyright or licence (\`NOTICES-from-sources.txt\`).

## The Gmsh programs of mpxlite in gmsh/

\`gmsh/\` holds the programs through which mpxlite runs Gmsh, Copyright (c) 2026
MPhysX, GPL-2.0-or-later (notice \`gmsh/README.md\`, GNU GPL version 2 text
\`gmsh/COPYING\`; also in \`source/tools/gmsh/\`): the toolkit of the Gmsh jobs
\`mpxlite_gmsh/\`, the jobs \`jobs/\` and the tests of the toolkit \`tests/\`. The
installer puts them in \`engine/gmsh/\` of the installation, beside the Gmsh
Python module of the engine; mpxlite (MIT) never imports them: it starts a job
as a separate process, which imports the Gmsh module. The Python environment of the
installation imports that module too (\`import gmsh\`, for the user's scripts).

## What is compiled into the engine

GetDP is configured with \`-DDEFAULT=0\` and only its kernel, MPI and PETSc;
OpenCASCADE with its modules FoundationClasses, ModelingData,
ModelingAlgorithms and DataExchange and no optional third-party product; FLTK
as static libraries with OpenGL and Xft, on the image libraries of the system;
Gmsh with its graphical interface on that FLTK, OpenCASCADE and the contributed
code it carries (\`source/tools/engine/build_engine.sh\`). What that compiles in, and
where its licence or notice is:

| Code | Licence | Notice |
|---|---|---|
| GetDP kernel | GPL-2.0-or-later | \`getdp/LICENSE.txt\` |
| mpx3 patch | GPL-2.0-or-later | \`source/tools/getdp/patches/README.md\` |
| avl (AVL trees, in GetDP) | permissive notice of the Regents of the University of California | \`getdp/CREDITS.txt\` |
| KissFFT (in GetDP) | BSD-3-Clause | \`getdp/CREDITS.txt\` |
| picojson (in GetDP) | BSD-2-Clause | \`getdp/CREDITS.txt\` |
| ONELAB and Gmsh socket headers (in GetDP) | MIT-style notice | \`getdp/NOTICE-onelab-gmshsocket.txt\` |
| GetDP parser generated by Bison | GPL-3.0-or-later with the Bison exception | the parser source (upstream) |
| PETSc | BSD-2-Clause | \`petsc/LICENSE\` |
| MUMPS (static, inside libpetsc) | CeCILL-C | \`mumps/LICENSE\`, \`mumps/CeCILL-C_V1-en.txt\`, \`mumps/CeCILL-C_V1-fr.txt\` |
| PORD (in MUMPS) | public domain | \`mumps/PORD-README\` |
| AMD- and LAPACK-derived routines (in MUMPS) | BSD-3-Clause | \`mumps/NOTICE-BSD-3-Clause.txt\` |
| ScaLAPACK | BSD-3-Clause | \`scalapack/LICENSE\` |
| METIS | Apache-2.0 | \`metis/LICENSE.txt\`, \`metis/Apache-2.0.txt\` |
| OpenCASCADE Technology | LGPL-2.1 with the Open CASCADE exception | \`opencascade/LICENSE_LGPL_21.txt\`, \`opencascade/OCCT_LGPL_EXCEPTION.txt\` |
| FLTK (static, inside the Gmsh library) | LGPL-2.0 with the FLTK exceptions | \`fltk/COPYING\`, \`fltk/NOTICES-from-sources.txt\` |
| Gmsh (library, program and Python module) | GPL-2.0-or-later with the exception of its \`LICENSE.txt\` | \`gmsh/LICENSE.txt\` |
| code of other authors in the Gmsh sources (AVL trees, picojson, nanoflann, libOL, robin_hood, TetGen/BR) | permissive notices; TetGen/BR relicensed under the Gmsh licence | \`gmsh/CREDITS.txt\` |
${CONTRIB_ROWS}
Not compiled in (switched off): the contributed Arpack, PeWe and Bessel
Fortran routines and Sparskit of GetDP, its Python interface and every other
optional component; ParMETIS, PT-Scotch / Scotch, STRUMPACK and SuperLU_DIST
in PETSc; the Visualization, ApplicationFramework and Draw modules of
OpenCASCADE (apart from the toolkits that DataExchange requires, built without
X11, OpenGL and FreeType) and its optional third-party products; the copies of
libjpeg, libpng and zlib that FLTK carries, and its Xinerama, Xfixes, Xcursor
and Cairo support; the Untangle code of Gmsh (HLBFGS, free for non-commercial
use only), its MPEG encoder, and its interfaces to PETSc, MPI, MED, CGNS, GMP,
Mmg, Cairo and every other library of the system.

## Not distributed by this bundle

\`install.sh\` installs the following on the user's machine; they are not part
of this bundle and come under their own licences:

| Component | Licence | Installed from |
|---|---|---|
| $(field mpi) | BSD-3-Clause | Ubuntu $OS_VER (apt) |
| OpenBLAS | BSD-3-Clause | Ubuntu $OS_VER (apt) |
| GCC runtime libraries (libgfortran, libquadmath, libstdc++, libgcc, libgomp) | GPL-3.0-or-later with the GCC Runtime Library Exception | Ubuntu $OS_VER (apt) |
| GNU C library | LGPL-2.1-or-later | Ubuntu $OS_VER (base system) |
| X11 client libraries (libX11, libXext, libXft, libXrender) | MIT/X11 | Ubuntu $OS_VER (apt) |
| fontconfig, FreeType | fontconfig: MIT-style; FreeType: FreeType License (or GPL-2.0) | Ubuntu $OS_VER (apt) |
| OpenGL (libGL of GLVND and Mesa), GLU | MIT-style; GLU: SGI Free Software License B 2.0 | Ubuntu $OS_VER (apt) |
| libjpeg-turbo, libpng, zlib | IJG and BSD-3-Clause; libpng licence; Zlib | Ubuntu $OS_VER (apt) |
| Python 3 and its venv module | PSF-2.0 | Ubuntu $OS_VER (apt) |
| NumPy | BSD-3-Clause | PyPI (pip) |
| pytest, pytest-cov, pytest-mock | MIT | PyPI (pip) |
| their dependencies (pluggy, iniconfig, packaging, coverage, pygments) | MIT, Apache-2.0, BSD-2-Clause | PyPI (pip) |

Libraries the engine loads from the system: $SYS_LIBS (the Gmsh library: $GMSH_SYS_LIBS).
EOF
} > "$T/README.md"

{
    echo "bundle      $NAME"
    echo "built       $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(. /etc/os-release && echo "$PRETTY_NAME") $(uname -m)"
    echo "mpxlite     $VERSION commit $REV$DIRTY"
    echo "engine      $ENGINE_ID, built $ENGINE_DATE on $(field os)"
    grep -E '^(Version|Build options|PETSc version|Packaged by)' "$INFO/getdp-info.txt" | sed 's/^/getdp       /'
    grep -E '^(Version|OCC version|Build options|Packaged by):' "$INFO/gmsh-build-info.txt" | sed 's/^/gmsh        /'
    echo "cflags      $(field cflags)"
    echo "compilers   $(field gcc); $(field gfortran); $(field cmake); $(field mpi)"
    awk -F'\t' 'NR > 1 {printf "component   %s %s (%s)\n", $1, $2, $5}' "$INFO/VERSIONS.tsv"
    echo "petsc4py    $P4PY_NAME ($(field petsc4py | sed 's/^[^;]*; //'))"
    echo "engine/lib  $ENGINE_LIBS"
    echo "system libs $SYS_LIBS"
} > "$B/BUILDINFO"

# Nothing shipped may name a home directory, the build prefix or the build machine.
bad="$(grep -r -l -a -E '/home/|/root/' "$B" || true)"
[ -z "$bad" ] || die "a home directory path is in: $bad"
case "$BUILD_PREFIX" in
    /opt/*|/usr/local/*|/srv/*|/tmp/*) ;;
    *) die "build prefix $BUILD_PREFIX: build the engine under /opt (or /usr/local, /srv) for a release bundle" ;;
esac

# No compiled Python cache is shipped: it records the time a file was copied,
# so two bundles of the same sources would differ (and an upgrade would
# replace an unchanged engine).
cached="$(find "$B" \( -name __pycache__ -o -name '*.pyc' \) -print)"
[ -z "$cached" ] || die "compiled Python caches in the bundle: $cached"

(cd "$B" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)

echo "=== archive"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
tar -C "$WORK" --owner=0 --group=0 --numeric-owner --sort=name -czf "$OUT/$NAME.tar.gz" "$NAME"
(cd "$OUT" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
ls -la "$OUT/$NAME.tar.gz"
cat "$OUT/$NAME.tar.gz.sha256"
