#!/usr/bin/env bash
# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT
#
# Install mpxlite from this bundle, as a whole and once per machine: the engine
# (GetDP 3.5.0-mpx3 on complex PETSc 3.26.0 with MUMPS, ScaLAPACK and METIS, and
# Gmsh 4.15.2 with its graphical interface (FLTK 1.3.11): library, Python module
# and the gmsh command, on OpenCASCADE 7.8.1), the Gmsh programs of mpxlite (the
# toolkit and the jobs), a Python virtual environment with mpxlite, petsc4py
# (bound to the PETSc of the engine) and pytest, in which the Gmsh module of the
# engine is importable (import gmsh), activate.sh (getdp and gmsh on PATH), the
# tests, then a smoke test. For the Ubuntu release in the file TARGET (24.04 or
# 26.04), x86-64 with AVX2, native or under WSL2.
#
# Usage: ./install.sh [--prefix DIR] [--no-apt] [--no-smoke]
#
#   --prefix DIR   install location (default: $MPXLITE_HOME, else ~/mpxlite)
#   --no-apt       skip the apt step (the runtime packages must be installed)
#   --no-smoke     skip the smoke test
#
# With an mpxlite installation already at the location: an older version is
# upgraded in place (the engine is replaced only if it changed; mpxlite is
# upgraded inside the existing venv/, everything else installed there and
# plugins/ are kept); the same version is verified (the mpxlite package file
# for file against the wheel of the bundle) and repaired if it differs; a newer one is left alone
# (no downgrade). An installation copied or moved to the location is refused:
# install afresh there. The exit status is non-zero only on failure.
set -euo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${MPXLITE_HOME:-$HOME/mpxlite}"
DO_APT=1
SMOKE=1
while [ $# -gt 0 ]; do
    case "$1" in
        --prefix) [ $# -ge 2 ] || { echo "--prefix needs a directory" >&2; exit 2; }; PREFIX="$2"; shift 2 ;;
        --prefix=*) PREFIX="${1#*=}"; shift ;;
        --no-apt) DO_APT=0; shift ;;
        --no-smoke) SMOKE=0; shift ;;
        -h|--help) sed -n '5,27p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
done
T0=$(date +%s)
# Every program the installer starts (getdp, the Python of the venv with
# petsc4py, hence Open MPI) keeps hwloc off X11 / OpenGL, as activate.sh does:
# its GL probe prints "Authorization required, but no authorization protocol
# specified" on machines with a display server.
export HWLOC_COMPONENTS=-gl
say() { printf '\n=== %s\n' "$*"; }
die() { echo "error: $*" >&2; exit 1; }

ENGINE_ID="getdp-3.5.0-mpx3+petsc-3.26.0+gmsh-4.15.2+occt-7.8.1"
# Gmsh of the engine: its Python module in engine/gmsh/, its library in
# engine/lib. Beside the module, the Gmsh programs of mpxlite (GPL-2.0-or-later):
# the toolkit package mpxlite_gmsh/, the jobs in jobs/ and the tests of the
# toolkit in tests/, with their notice and licence text; engine/gmsh holds
# nothing else (GMSH_KEEP: the module of the engine and its compiled cache).
# The venv imports the module through the file GMSH_PTH of its site-packages,
# which names engine/gmsh.
GMSH_VERSION=4.15.2
GMSH_FILES=(README.md COPYING)
GMSH_DIRS=(mpxlite_gmsh jobs tests)
GMSH_KEEP=(gmsh.py __pycache__)
GMSH_PTH=mpxlite_engine_gmsh.pth
for f in "${GMSH_FILES[@]}"; do
    [ -f "$BUNDLE/gmsh/$f" ] || die "missing $BUNDLE/gmsh/$f"
done
for d in "${GMSH_DIRS[@]}"; do
    [ -d "$BUNDLE/gmsh/$d" ] || die "missing $BUNDLE/gmsh/$d/"
done
for f in engine/gmsh/gmsh.py engine/lib/libgmsh.so.4.15 engine/bin/gmsh; do
    [ -f "$BUNDLE/$f" ] || die "missing $BUNDLE/$f"
done
TARGET_VER="$(awk '{print $2}' "$BUNDLE/TARGET")"
WHEEL=("$BUNDLE"/wheels/mpxlite-*-py3-none-any.whl)
[ -f "${WHEEL[0]}" ] && [ "${#WHEEL[@]}" = 1 ] || die "expected one mpxlite wheel in $BUNDLE/wheels"
VERSION="$(basename "${WHEEL[0]}" | cut -d- -f2)"
# petsc4py of the engine build: the Python bindings of the PETSc in engine/lib.
P4PY=("$BUNDLE"/wheels/petsc4py-*.whl)
[ -f "${P4PY[0]}" ] && [ "${#P4PY[@]}" = 1 ] || die "expected one petsc4py wheel in $BUNDLE/wheels"
P4PY_TAG="$(basename "${P4PY[0]}" | cut -d- -f3)"

# PEP 440 comparison of two mpxlite versions: prints -1, 0 or 1.
vercmp() {
    python3 - "$1" "$2" <<'EOF'
import re, sys
def key(v):
    m = re.fullmatch(r"(\d+(?:\.\d+)*)(?:(a|b|rc)(\d+))?(?:\.post(\d+))?(?:\.dev(\d+))?", v)
    if not m:
        sys.exit(f"cannot compare version {v!r}")
    rel = [int(x) for x in m[1].split(".")]
    while rel and rel[-1] == 0:
        rel.pop()
    pre = ({"a": 0, "b": 1, "rc": 2}[m[2]], int(m[3])) if m[2] else (3, 0)
    if m[5] is not None and not m[2] and m[4] is None:
        pre = (-1, 0)  # X.devN sorts before X.aN
    post = int(m[4]) if m[4] is not None else -1
    dev = int(m[5]) if m[5] is not None else float("inf")
    return (rel, pre, post, dev)
a, b = key(sys.argv[1]), key(sys.argv[2])
print((a > b) - (a < b))
EOF
}

# Runtime packages of the engine (Open MPI, OpenBLAS, the GCC Fortran runtime,
# the OpenMP runtime of the Gmsh library, the X11, OpenGL and image libraries
# the Gmsh library loads for its graphical interface; they are loaded on a
# machine without display too, and open no display unless the interface is
# started) and of the Python environment.
APT_PACKAGES=(openmpi-bin libopenblas0-pthread libgfortran5 libgomp1
              libx11-6 libxext6 libxft2 libxrender1 libfontconfig1 libgl1 libglu1-mesa
              libjpeg-turbo8 libpng16-16t64 zlib1g python3 python3-venv)
case "$TARGET_VER" in
    24.04) APT_PACKAGES+=(libopenmpi3t64) ;;
    26.04) APT_PACKAGES+=(libopenmpi40) ;;
esac

say "checks"
# (in a subshell: /etc/os-release sets VERSION, among others)
OS_ID="$(. /etc/os-release && echo "${ID:-}")"
OS_VER="$(. /etc/os-release && echo "${VERSION_ID:-}")"
PRETTY_NAME="$(. /etc/os-release && echo "${PRETTY_NAME:-an unknown system}")"
if [ "$OS_ID" != ubuntu ] || [ "$OS_VER" != "$TARGET_VER" ]; then
    die "this bundle is for Ubuntu $TARGET_VER (found $PRETTY_NAME); use the bundle of your release"
fi
[ "$(uname -m)" = x86_64 ] || die "x86-64 required (found $(uname -m))"
for flag in avx2 fma bmi2; do
    grep -qw "$flag" /proc/cpuinfo \
        || die "the engine needs an x86-64-v3 CPU (AVX2, FMA, BMI2); '$flag' is missing. Build it from source (see source/README.md)"
done
mkdir -p "$PREFIX" || die "cannot create $PREFIX"
PREFIX="$(cd "$PREFIX" && pwd)"
echo "    $PRETTY_NAME, $(uname -m), mpxlite $VERSION, prefix $PREFIX"

# What is at the prefix: nothing, a failed fresh install, or an mpxlite installation.
MODE=fresh
if [ -f "$PREFIX/VERSION" ]; then
    OLD_VERSION="$(sed -n 's/^mpxlite=//p' "$PREFIX/VERSION")"
    OLD_ENGINE="$(sed -n 's/^engine=//p' "$PREFIX/VERSION")"
    OLD_TARGET="$(sed -n 's/^target=//p' "$PREFIX/VERSION")"
    [ -n "$OLD_VERSION" ] || die "$PREFIX/VERSION has no mpxlite= line"
    case "$(vercmp "$OLD_VERSION" "$VERSION")" in
        -1) MODE=upgrade ;;
        0) MODE=same ;;
        1)
            echo "    mpxlite $OLD_VERSION, newer than this bundle ($VERSION), is installed in $PREFIX:"
            echo "    nothing to do (no downgrade)."
            exit 0
            ;;
    esac
elif [ -e "$PREFIX/.mpxlite-installing" ]; then
    echo "    an unfinished installation is in $PREFIX: installing again"
elif [ -n "$(ls -A "$PREFIX")" ]; then
    die "$PREFIX is not empty and is not an mpxlite installation (no VERSION file); choose another --prefix"
fi

# A Python environment is bound to the directory it was created in (its
# scripts and activate name it): an installation copied or moved elsewhere
# cannot be upgraded or repaired, only installed afresh.
if [ -f "$PREFIX/venv/pyvenv.cfg" ]; then
    RECORDED="$(sed -n 's/^command = .* -m venv //p' "$PREFIX/venv/pyvenv.cfg" | sed -E 's/^(--?[a-z-]+ +)*//')"
    if [ -z "$RECORDED" ] && [ -f "$PREFIX/venv/bin/pip" ]; then
        RECORDED="$(head -n 1 "$PREFIX/venv/bin/pip" | sed -n 's|^#!\(.*\)/bin/python[0-9.]*$|\1|p')"
    fi
    if [ -n "$RECORDED" ] && [ "$(readlink -f -- "$RECORDED")" != "$(readlink -f -- "$PREFIX/venv")" ]; then
        die "$PREFIX/venv was created as $RECORDED: this installation was copied or moved to $PREFIX, and a Python environment cannot be moved. Install mpxlite afresh into $PREFIX (move the copy aside, or remove it, and run install.sh again), then install the plug-ins again."
    fi
fi

# The engine files at the prefix are those of this bundle.
engine_is_this_bundle() {
    [ -d "$PREFIX/engine" ] || return 1
    (cd "$BUNDLE" && grep ' \./engine/' SHA256SUMS | sed 's| \./engine/| ./|' \
        | (cd "$PREFIX/engine" && sha256sum --quiet -c - >/dev/null 2>&1))
}
# petsc4py of the venv is on complex scalars with MUMPS and has loaded the
# libpetsc of the engine at the prefix.
petsc4py_ok() {
    "$PREFIX/venv/bin/python" - "$PREFIX/engine/lib" <<'EOF'
import os, sys
import petsc4py
petsc4py.init([])
from petsc4py import PETSc
assert issubclass(PETSc.ScalarType, complex), PETSc.ScalarType
assert PETSc.Sys.hasExternalPackage("mumps"), "PETSc without MUMPS"
loaded = [line.split()[-1] for line in open("/proc/self/maps") if "libpetsc" in line]
engine_lib = os.path.realpath(sys.argv[1])
assert loaded and all(os.path.realpath(os.path.dirname(p)) == engine_lib for p in loaded), loaded
EOF
}
# The site-packages directory of the venv.
venv_site() {
    "$PREFIX/venv/bin/python" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'
}
# The Python of the venv, as a user's script runs it (no path of its own: the
# file GMSH_PTH of its site-packages names engine/gmsh), imports the Gmsh
# module of the engine, which loads the Gmsh and OpenCASCADE libraries of
# engine/lib, and Gmsh reports the version of the bundle; the venv has no gmsh
# package of its own.
gmsh_ok() {
    [ "$(cat "$(venv_site)/$GMSH_PTH" 2>/dev/null)" = "$PREFIX/engine/gmsh" ] || return 1
    "$PREFIX/venv/bin/python" -I - "$PREFIX/engine" "$GMSH_VERSION" <<'EOF' || return 1
import os, sys
engine = os.path.realpath(sys.argv[1])
import gmsh
assert os.path.realpath(gmsh.__file__) == os.path.join(engine, "gmsh", "gmsh.py"), gmsh.__file__
gmsh.initialize(readConfigFiles=False)
gmsh.option.setNumber("General.Terminal", 0)
version = gmsh.option.getString("General.Version")
gmsh.finalize()
assert version == sys.argv[2], version
loaded = {line.split()[-1] for line in open("/proc/self/maps") if "/libgmsh" in line or "/libTK" in line}
lib = os.path.join(engine, "lib")
assert any("/libgmsh" in p for p in loaded), loaded
assert all(os.path.dirname(os.path.realpath(p)) == lib for p in loaded), loaded
EOF
    ! venv_has_gmsh
}
# The gmsh command of the engine starts (in batch: no display needed) and is
# the Gmsh of the bundle.
gmsh_command_ok() {
    env -u DISPLAY -u WAYLAND_DISPLAY "$PREFIX/engine/bin/gmsh" -info 2>&1 \
        | grep -q "^Version *: $GMSH_VERSION$"
}
# A gmsh distribution is installed in the venv (a package of PyPI installed
# there by a plug-in or by hand, which would shadow the Gmsh of the engine).
venv_has_gmsh() {
    "$PREFIX/venv/bin/python" -m pip show -q gmsh >/dev/null 2>&1
}
# The mpxlite package of the venv is the one of this bundle, file for file:
# the files of the wheel (compiled caches apart), byte for byte. A package of
# another build with the same version number differs here, not in its version.
mpxlite_is_this_bundle() {
    "$PREFIX/venv/bin/python" - "${WHEEL[0]}" <<'EOF' 2>/dev/null
import importlib.util, pathlib, sys, zipfile
package = pathlib.Path(importlib.util.find_spec("mpxlite").origin).parent
with zipfile.ZipFile(sys.argv[1]) as wheel:
    shipped = {n: wheel.read(n) for n in wheel.namelist() if n.startswith("mpxlite/") and not n.endswith("/")}
installed = {
    p.relative_to(package.parent).as_posix(): p
    for p in package.rglob("*")
    if p.is_file() and "__pycache__" not in p.parts
}
assert set(installed) == set(shipped), sorted(set(installed) ^ set(shipped))
assert all(installed[n].read_bytes() == data for n, data in shipped.items())
EOF
}
# The Gmsh programs of the installation (toolkit, jobs) are those of this
# bundle (compiled caches apart), and engine/gmsh holds nothing else.
gmsh_programs_are_this_bundle() {
    local f d
    for f in "${GMSH_FILES[@]}"; do
        cmp -s "$BUNDLE/gmsh/$f" "$PREFIX/engine/gmsh/$f" || return 1
    done
    for d in "${GMSH_DIRS[@]}"; do
        diff -r -q -x __pycache__ "$BUNDLE/gmsh/$d" "$PREFIX/engine/gmsh/$d" >/dev/null 2>&1 || return 1
    done
    [ -z "$(gmsh_extra_entries)" ]
}
# The entries of engine/gmsh that are neither the Gmsh programs of this bundle
# nor the module of the engine.
gmsh_extra_entries() {
    local e name known
    for e in "$PREFIX/engine/gmsh"/* "$PREFIX/engine/gmsh"/.[!.]*; do
        [ -e "$e" ] || continue
        name="$(basename "$e")"
        for known in "${GMSH_FILES[@]}" "${GMSH_DIRS[@]}" "${GMSH_KEEP[@]}"; do
            [ "$name" = "$known" ] && continue 2
        done
        printf '%s\n' "$e"
    done
}
# Engine and venv health of an existing installation (same version).
verify() {
    local ok=1
    engine_is_this_bundle || ok=0
    gmsh_programs_are_this_bundle || ok=0
    petsc4py_ok 2>/dev/null || ok=0
    gmsh_ok 2>/dev/null || ok=0
    gmsh_command_ok || ok=0
    "$PREFIX/venv/bin/python" -c "import mpxlite, sys; sys.exit(mpxlite.__version__ != '$VERSION')" \
        2>/dev/null || ok=0
    mpxlite_is_this_bundle || ok=0
    [ "$(cat "$PREFIX/VERSION" | sed -n 's/^engine=//p')" = "$ENGINE_ID" ] || ok=0
    "$PREFIX/engine/bin/getdp" --version 2>&1 | grep -q 'mpx3' || ok=0
    [ "$ok" = 1 ]
}
if [ "$MODE" = same ]; then
    say "mpxlite $VERSION is already installed in $PREFIX: verifying"
    if verify; then
        echo "    engine files, getdp, Gmsh (importable, and the gmsh command), the Gmsh programs, petsc4py and the mpxlite package are as in this bundle: nothing to do."
        exit 0
    fi
    echo "    the installation differs from this bundle: repairing it in place"
    MODE=repair
fi

missing=()
for p in "${APT_PACKAGES[@]}"; do
    dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q 'install ok installed' || missing+=("$p")
done
if [ "${#missing[@]}" = 0 ]; then
    say "runtime packages: all installed (${APT_PACKAGES[*]})"
elif [ "$DO_APT" = 1 ]; then
    SUDO=()
    if [ "$(id -u)" != 0 ]; then
        command -v sudo >/dev/null || die "sudo is missing; as root run: apt-get install -y ${missing[*]} (then rerun with --no-apt)"
        SUDO=(sudo)
    fi
    say "runtime packages (apt): ${missing[*]}"
    echo "    ${SUDO[*]}${SUDO[*]:+ }apt-get update && ${SUDO[*]}${SUDO[*]:+ }apt-get install -y --no-install-recommends ${APT_PACKAGES[*]}"
    "${SUDO[@]}" apt-get update -qq
    "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive \
        apt-get install -y -qq --no-install-recommends "${APT_PACKAGES[@]}" >/dev/null
else
    say "apt skipped (--no-apt); missing runtime packages: ${missing[*]}"
fi

# Replace directory $PREFIX/$1 by a copy of $BUNDLE/$1 (or of $2), atomically enough
# that an interrupted run leaves either the old or the new copy.
replace_dir() {
    local name="$1" src="${2:-$BUNDLE/$1}"
    rm -rf "${PREFIX:?}/.new-$name" "${PREFIX:?}/.old-$name"
    cp -a "$src" "$PREFIX/.new-$name"
    if [ -e "$PREFIX/$name" ]; then mv "$PREFIX/$name" "$PREFIX/.old-$name"; fi
    mv "$PREFIX/.new-$name" "$PREFIX/$name"
    rm -rf "${PREFIX:?}/.old-$name"
}

case "$MODE" in
    fresh)
        touch "$PREFIX/.mpxlite-installing"
        say "engine -> $PREFIX/engine"
        replace_dir engine
        ;;
    upgrade|repair)
        [ "$MODE" = upgrade ] && say "upgrading mpxlite $OLD_VERSION -> $VERSION in $PREFIX"
        # The engine of the installation is the engine of the bundle, file
        # for file: petsc4py below is compiled against this very libpetsc.
        if engine_is_this_bundle; then
            say "engine unchanged ($ENGINE_ID, ubuntu-$TARGET_VER, same files)"
        else
            say "engine -> $PREFIX/engine (replaced: ${OLD_ENGINE:-?} ${OLD_TARGET:-?} -> $ENGINE_ID ubuntu-$TARGET_VER)"
            replace_dir engine
        fi
        ;;
esac
GETDP="$PREFIX/engine/bin/getdp"
"$GETDP" --version >/dev/null 2>&1 || {
    "$GETDP" --version || true
    ldd "$GETDP" | grep "not found" || true
    die "getdp does not start (missing runtime packages? rerun without --no-apt)"
}

PY="$PREFIX/venv/bin/python"
if [ -x "$PY" ] && "$PY" -c 'import sys' 2>/dev/null; then
    say "mpxlite $VERSION -> existing venv $PREFIX/venv (other packages kept)"
else
    [ "$MODE" = fresh ] \
        || die "$PREFIX/venv is missing or broken (Ubuntu release upgrade?): move $PREFIX aside and install again"
    say "Python venv -> $PREFIX/venv"
    rm -rf "${PREFIX:?}/venv"
    python3 -m venv "$PREFIX/venv"
fi
# The bundled wheel by path (never a same-named package from an index); its
# dependencies come from PyPI. --upgrade replaces mpxlite only: other packages
# are upgraded only if mpxlite's requirements need it.
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
"$PY" -m pip install -q --upgrade "${WHEEL[0]}[test]"
# pip keeps an installed mpxlite of the same version: in a repair, a package
# of another build of that version is replaced by the one of this bundle.
if ! mpxlite_is_this_bundle; then
    echo "    the mpxlite package in $PREFIX/venv is not the one of this bundle: replacing it"
    "$PY" -m pip install -q --no-deps --force-reinstall "${WHEEL[0]}"
fi
"$PY" -c "import mpxlite, sys; sys.exit(mpxlite.__version__ != '$VERSION')" \
    || die "the venv does not import mpxlite $VERSION"
mpxlite_is_this_bundle || die "the mpxlite package in $PREFIX/venv is not the one of this bundle"

say "petsc4py $(basename "${P4PY[0]}" | cut -d- -f2) (Python bindings of the PETSc of the engine) -> $PREFIX/venv"
VENV_TAG="$("$PY" -c 'import sys; print("cp%d%d" % sys.version_info[:2])')"
[ "$VENV_TAG" = "$P4PY_TAG" ] \
    || die "the Python of $PREFIX/venv is $VENV_TAG, petsc4py of this bundle is for $P4PY_TAG (the Python of Ubuntu $TARGET_VER)"
# By path and without dependencies (NumPy, its only one, came with mpxlite):
# never a petsc4py from an index, which would be bound to another PETSc.
"$PY" -m pip install -q --no-deps --force-reinstall "${P4PY[0]}"
petsc4py_ok || die "petsc4py does not load on the engine of $PREFIX"

say "Gmsh programs (toolkit, jobs) -> $PREFIX/engine/gmsh (beside the Gmsh module of the engine)"
gmsh_extra_entries | while IFS= read -r e; do
    echo "    removing $e: not a Gmsh program of this bundle"
    rm -rf "$e"
done
for f in "${GMSH_FILES[@]}"; do install -m 644 "$BUNDLE/gmsh/$f" "$PREFIX/engine/gmsh/$f"; done
for d in "${GMSH_DIRS[@]}"; do
    rm -rf "${PREFIX:?}/engine/gmsh/$d"
    cp -r "$BUNDLE/gmsh/$d" "$PREFIX/engine/gmsh/$d"
    find "$PREFIX/engine/gmsh/$d" -type d -exec chmod 755 {} + -o -type f -exec chmod 644 {} +
done
# The Gmsh module of the engine, importable from the venv (import gmsh).
printf '%s\n' "$PREFIX/engine/gmsh" > "$(venv_site)/$GMSH_PTH"
# Gmsh is the one of the engine only: a gmsh package in the venv (installed
# there by a plug-in or by hand) is removed.
if venv_has_gmsh; then
    echo "    removing the gmsh package $("$PY" -m pip show gmsh 2>/dev/null | sed -n 's/^Version: //p') of $PREFIX/venv: Gmsh is part of the engine"
    "$PY" -m pip uninstall -q -y gmsh
fi
gmsh_ok || die "Gmsh $GMSH_VERSION of the engine does not load from $PREFIX/engine (missing runtime packages? rerun without --no-apt)"
gmsh_command_ok || die "the gmsh command of the engine ($PREFIX/engine/bin/gmsh -info) does not start (missing runtime packages? rerun without --no-apt)"
echo "    Gmsh $GMSH_VERSION: engine/gmsh/gmsh.py (import gmsh in the venv, $GMSH_PTH), engine/lib/libgmsh.so.4.15 and OpenCASCADE from engine/lib, the gmsh command engine/bin/gmsh"

say "activate.sh, smoke test, tests, licences"
mkdir -p "$PREFIX/bin" "$PREFIX/plugins"
install -m 644 "$BUNDLE/mpxlite_smoke.py" "$PREFIX/bin/mpxlite_smoke.py"
cat > "$PREFIX/bin/mpxlite-smoke" <<EOF
#!/usr/bin/env bash
# mpxlite installation smoke test (written by install.sh)
. "$PREFIX/activate.sh"
exec python "$PREFIX/bin/mpxlite_smoke.py" "\$@"
EOF
chmod 755 "$PREFIX/bin/mpxlite-smoke"
cat > "$PREFIX/activate.sh" <<EOF
# mpxlite environment (written by install.sh); use it with: source $PREFIX/activate.sh
. "$PREFIX/venv/bin/activate"
export MPXLITE_HOME="$PREFIX"
export MPXLITE_GETDP_BINARY="$GETDP"
# mpxlite's commands, then those of the plug-ins (every plugins/<name>/bin, in
# name order, as found when this file is sourced), then the inherited PATH.
_mpxlite_path="$PREFIX/bin:$PREFIX/engine/bin"
_mpxlite_bins="\$(find "$PREFIX/plugins" -mindepth 2 -maxdepth 2 -name bin -type d 2>/dev/null | LC_ALL=C sort)"
while IFS= read -r _mpxlite_bin; do
    if [ -n "\$_mpxlite_bin" ]; then _mpxlite_path="\$_mpxlite_path:\$_mpxlite_bin"; fi
done <<_MPXLITE_BINS
\$_mpxlite_bins
_MPXLITE_BINS
export PATH="\$_mpxlite_path:\$PATH"
unset _mpxlite_path _mpxlite_bins _mpxlite_bin
export HWLOC_COMPONENTS=-gl              # keep hwloc (Open MPI) off X11/OpenGL
export OMP_NUM_THREADS=1
# Open MPI on a single machine without ssh or a hostfile (WSL2, containers):
export OMPI_MCA_plm_rsh_agent=           # no ssh lookup; local launches only
export OMPI_MCA_rmaps_base_oversubscribe=1                   # allow np > cores (Open MPI 4)
export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe  # same, Open MPI 5 (PRRTE)
export OMPI_MCA_btl_vader_single_copy_mechanism=none         # no CMA/ptrace (Open MPI 4)
export OMPI_MCA_btl_sm_single_copy_mechanism=none            # same, Open MPI 5
if [ "\$(id -u)" = 0 ]; then                                 # root (containers): allow mpirun
    export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
fi
EOF
for d in THIRD_PARTY source tests; do replace_dir "$d"; done
install -m 644 "$BUNDLE/README-INSTALL.md" "$PREFIX/README-INSTALL.md"
# VERSION last: it marks a complete installation.
cat > "$PREFIX/.VERSION.new" <<EOF
mpxlite=$VERSION
engine=$ENGINE_ID
target=ubuntu-$TARGET_VER
installed=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
mv "$PREFIX/.VERSION.new" "$PREFIX/VERSION"
rm -f "$PREFIX/.mpxlite-installing"

echo
echo "mpxlite $VERSION installed in $PREFIX ($MODE, $(( $(date +%s) - T0 )) s)."
if [ "$SMOKE" = 1 ]; then
    say "smoke test"
    if "$PREFIX/bin/mpxlite-smoke"; then
        echo "smoke test: PASS"
    else
        echo "smoke test: FAIL (output above); the installation is in $PREFIX"
        exit 1
    fi
fi
echo
echo "Start with:"
echo "    source $PREFIX/activate.sh"
