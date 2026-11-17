#!/usr/bin/env bash
# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT
#
# Build the mpxlite engine from source, in user space, in six parts:
#
#   petsc     PETSc 3.26.0, complex scalars, with the MUMPS direct solver,
#             ScaLAPACK and the METIS ordering
#   getdp     GetDP 3.5.0 with the mpx3 patch, on that PETSc
#   petsc4py  the Python bindings of that PETSc (from the PETSc source tree),
#             for the python3 of this machine
#   occt      OpenCASCADE 7.8.1 (the geometry kernel of Gmsh), with only the
#             modules Gmsh needs
#   fltk      FLTK 1.3.11 (the toolkit of the graphical interface of Gmsh), as
#             static libraries with OpenGL, on X11
#   gmsh      Gmsh 4.15.2: the shared library with its Python module and the
#             gmsh executable (linked to that library), with its graphical
#             interface on the FLTK above and the OpenCASCADE kernel above
#
# This script is the reference of the source build.
#
# PETSc downloads exactly three packages: MUMPS, ScaLAPACK and METIS. GetDP is
# configured with only what mpxlite uses (kernel, MPI, PETSc): no Python, no
# Fortran (hence no contributed Arpack, PeWe or Bessel routines), no BLAS/LAPACK
# of its own, no Gmsh, GSL, SLEPc, Sparskit nor any other optional component.
# OpenCASCADE is built from its source release with the modules
# FoundationClasses, ModelingData, ModelingAlgorithms and DataExchange (and the
# toolkits these require), without Draw, Tcl/Tk, FreeType, X11, OpenGL, TBB or
# any other optional third-party product. FLTK is built from its source release
# as static libraries (position-independent, linked into the Gmsh library) with
# OpenGL, Xft and the image libraries of the system (libjpeg, libpng, zlib; its
# bundled copies are not built), without Xinerama, Xfixes, Xcursor, Cairo nor
# Wayland (FLTK 1.3 is X11 only: the official Gmsh builds for Linux use FLTK
# 1.3, whose OpenGL rendering is faster than that of FLTK 1.4). Gmsh is built
# from its source release with the code it carries (contrib/) and its graphical
# interface (FLTK, OpenGL, GLU, and the system libjpeg, libpng and zlib for
# the image formats), without PETSc, MPI, MED, CGNS, GMP, Cairo, the MPEG
# encoder nor any other library of the system. The library loads the X11 and
# OpenGL libraries but opens no display: only the graphical interface does
# (the gmsh command run without a batch option, gmsh.fltk in Python).
#
# Supported on Ubuntu 24.04 and 26.04, x86-64 (native, WSL2, container, CI
# runner). System prerequisites (one sudo step):
#
#   sudo apt-get install -y --no-install-recommends build-essential gfortran \
#       cmake git python3 python3-dev python3-venv ca-certificates openmpi-bin \
#       libopenmpi-dev libopenblas-dev libfreetype-dev libx11-dev libxext-dev \
#       libxft-dev libxrender-dev libfontconfig-dev libgl-dev libegl-dev \
#       libglu1-mesa-dev libjpeg-dev libpng-dev zlib1g-dev
#
# (libfreetype-dev: the headers the Gmsh configuration requires to enable the
# reading of STEP/IGES names and colours, and the FreeType of Xft for FLTK;
# neither OpenCASCADE nor Gmsh calls FreeType itself. The X11, OpenGL and
# image packages: for FLTK and the graphical interface of Gmsh.)
# The PETSc configure step, the GetDP clone, the OpenCASCADE, FLTK and Gmsh
# source archives and the build tools of petsc4py (Cython, NumPy, setuptools,
# from PyPI) need Internet access.
#
# Usage:
#   tools/engine/build_engine.sh [--prefix DIR] [--jobs N] [--getdp-git URL_OR_DIR] [--force]
#   tools/engine/build_engine.sh --print-key PART | --print-inputs PART | --print-dirs PART
#
#   --prefix DIR   build prefix (default: $HOME/opt)
#   --jobs N       parallel compile jobs (default: min(nproc, 8))
#   --getdp-git X  where to clone GetDP from (default: the upstream repository;
#                  a local clone or mirror works, the tag below is checked out)
#   --force        rebuild every part
#   --print-key PART     print the key of PART (petsc, getdp, petsc4py, occt,
#                  fltk, gmsh): <part>-<20 hex digits of the sha256 of its inputs>,
#                  e.g. for a CI cache key; nothing is built
#   --print-inputs PART  print the inputs the key of PART is made from
#   --print-dirs PART    print the directories of PART, relative to the
#                  prefix (what a CI cache of the part holds)
#
# Incremental build. Each part records, in <part directory>/mpxlite-part/inputs,
# what it was built from: the release of the system, the pinned version and
# commit or archive checksum, the configuration options, a recipe number (raised
# here when the way the part is built changes otherwise), the Python version for
# petsc4py, the checksum of the mpx3 patch for GetDP, and the key of each part
# it depends on (getdp and petsc4py on petsc; gmsh on occt and fltk). A part whose
# record matches is skipped; any other is rebuilt from a clean tree, and the
# parts that depend on it are rebuilt with it (their inputs contain its key).
# A change of the mpx3 patch rebuilds GetDP only. engine-info/ is regenerated
# on every run.
#
# When the upstream GetDP repository cannot be reached, GetDP is cloned from
# GETDP_COPY_GIT below (the GetDP history up to the pinned commit, with the
# mpx3 patch on top as one commit, branch mpx3; it carries no upstream tag, so
# GETDP_TAG is set on the pinned commit in the clone). When the MUMPS site
# cannot be reached, or the address of the OpenCASCADE, FLTK or Gmsh source
# archive cannot be reached or does not give the pinned file, the archive is
# taken, under the file names below, from ENGINE_SOURCES_REPO below (copies of
# the source archives of the components used unmodified). The pinned commit
# and checksum are checked whatever the origin.
#
# Result, under the prefix (the PETSc build is bound to this path):
#   petsc-3.26.0-complex/arch-complex-opt/lib   PETSc (+ MUMPS, ScaLAPACK, METIS)
#   getdp-3.5.0-mpx3/bin/getdp                  the GetDP executable
#   petsc4py-3.26.0/petsc4py-3.26.0-cp3XX-*.whl the petsc4py wheel, linked to
#                                               the libpetsc above
#   occt-7.8.1/lib/libTK*.so.7.8                the OpenCASCADE libraries
#   fltk-1.3.11/lib/libfltk*.a                  the FLTK static libraries (and
#                                               its headers and fltk-config)
#   gmsh-4.15.2/lib/libgmsh.so.4.15             the Gmsh library, linked to the
#   gmsh-4.15.2/lib/gmsh.py                     OpenCASCADE above and containing
#   gmsh-4.15.2/bin/gmsh                        FLTK, its Python module (ctypes)
#                                               and the gmsh executable
#   engine-info/   what tools/bundle/make_bundle.sh needs besides the binaries:
#       VERSIONS.tsv (component, version, licence, upstream, exact revision or
#       archive sha256, licence files), licenses/<component>/ (licence and
#       notice files of every component compiled into the engine), patch/ (the
#       mpx3 patch and its notices), the GetDP configuration (CMake cache
#       options, getdp --info), the PETSc configure options and petscconf.h,
#       the Gmsh build information and the list of its contributed code
#       compiled in, the OpenCASCADE toolkits, the FLTK configuration, and
#       BUILD (release, compilers,
#       inputs and durations of the parts).
# The patch is read from tools/getdp/patches/ of the checkout (or of the
# source/ directory of a bundle) that contains this script.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="$(cd "$HERE/../getdp/patches" 2>/dev/null && pwd || true)"
PREFIX="$HOME/opt"
GETDP_GIT="https://gitlab.onelab.info/getdp/getdp.git"
GETDP_UPSTREAM="https://gitlab.onelab.info/getdp/getdp.git"
# The copy of the GetDP history, cloned when GETDP_GIT cannot be reached.
GETDP_COPY_GIT="https://github.com/MPhysXDev/fem-mpx.git"
# The copies of the source archives of the components used unmodified (the
# MUMPS, OpenCASCADE, FLTK and Gmsh archives among them), and the address of
# their files.
ENGINE_SOURCES_REPO="https://github.com/MPhysXDev/mpxlite-engine-sources"
ENGINE_SOURCES_FILES="$ENGINE_SOURCES_REPO/raw/main"
PETSC_GIT="https://gitlab.com/petsc/petsc.git"
JOBS="$(nproc)"
[ "$JOBS" -gt 8 ] && JOBS=8
FORCE=0
PRINT=""
PRINT_PART=""
PARTS=(petsc getdp petsc4py occt fltk gmsh)

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix) PREFIX="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --getdp-git) GETDP_GIT="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --print-key|--print-inputs|--print-dirs)
            [ $# -ge 2 ] || { echo "$1 needs a part (${PARTS[*]})" >&2; exit 2; }
            PRINT="${1#--print-}"; PRINT_PART="$2"; shift 2 ;;
        -h|--help) sed -n '5,/^set -euo pipefail$/p' "$0" | sed '$d'; exit 0 ;;
        *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
done

# --- pinned sources ----------------------------------------------------------
PETSC_COMMIT="e6a25c9dbe75ac9d50c03f4462d8e39f0a0ce7ad"   # PETSc 3.26.0 (tag v3.26.0)
GETDP_TAG="getdp_3_5_0"                                  # 864875cd
GETDP_COMMIT="864875cdfc652254f777311d58f510b38e508b56"
# MUMPS is the only package PETSc fetches as a tarball (METIS and ScaLAPACK are
# git clones at commits pinned by PETSc). Its checksum is pinned here and
# checked in the petsc part.
MUMPS_SHA256="659c9b57646b5a003ac618baa1faf9dd2044e46c732b3daaccbc7158003e1b46"  # MUMPS_5.9.1.tar.gz
PATCH="$PATCH_DIR/getdp-3.5.0-mpx3.patch"
PETSC_ARCH="arch-complex-opt"
# OpenCASCADE, FLTK and Gmsh: the upstream source archive of the release (sha256),
# which is the tree of the tag (commit) in the upstream repository; the file
# name is also the name of its copy in ENGINE_SOURCES_REPO.
OCCT_VERSION="7.8.1"
OCCT_TAG="V7_8_1"
OCCT_COMMIT="bd2a789f15235755ce4d1a3b07379a2e062fdc2e"
OCCT_GIT="https://github.com/Open-Cascade-SAS/OCCT"
OCCT_ARCHIVE="OCCT-7_8_1.tar.gz"
OCCT_URL="https://github.com/Open-Cascade-SAS/OCCT/archive/refs/tags/V7_8_1.tar.gz"
OCCT_SHA256="7321af48c34dc253bf8aae3f0430e8cb10976961d534d8509e72516978aa82f5"
FLTK_VERSION="1.3.11"
FLTK_TAG="release-1.3.11"
FLTK_COMMIT="702172a951a2bb4f25afe31008ef3fcbb5bfb92f"
FLTK_GIT="https://github.com/fltk/fltk"
FLTK_ARCHIVE="fltk-1.3.11-source.tar.gz"
FLTK_URL="https://github.com/fltk/fltk/releases/download/release-1.3.11/fltk-1.3.11-source.tar.gz"
FLTK_SHA256="92805abc84505e3e7e27aec775ab0754ecb4182fe2d8ff2a9d568ccdcb2811ac"
GMSH_VERSION="4.15.2"
GMSH_TAG="gmsh_4_15_2"
GMSH_COMMIT="657c8e915f60405e6cad0c8ec7faf812bfff1a60"
GMSH_GIT="https://gitlab.onelab.info/gmsh/gmsh"
GMSH_ARCHIVE="gmsh-4.15.2-source.tgz"
GMSH_URL="https://gmsh.info/src/gmsh-4.15.2-source.tgz"
GMSH_SHA256="be3f66f225d27ba9fa014f07e83169285da8a051b0e8ab7103d88066b39bdd3e"
GMSH_SONAME="libgmsh.so.4.15"

# --- recipes: what each part is built with (all of it goes into its key) -----
# Raise a RECIPE number when the way a part is built changes in a way that the
# options below do not show.
PETSC_RECIPE=1
GETDP_RECIPE=1
P4PY_RECIPE=1
OCCT_RECIPE=1
FLTK_RECIPE=1
GMSH_RECIPE=1
OPT='-O3 -march=x86-64-v3 -mtune=generic'
PETSC_CONFIGURE=(
    PETSC_ARCH="$PETSC_ARCH"
    --with-scalar-type=complex
    --with-mpi-dir=/usr
    --with-debugging=0
    --with-shared-libraries=1
    --with-fortran-bindings=0
    --with-x=0
    COPTFLAGS="$OPT" CXXOPTFLAGS="$OPT" FOPTFLAGS="$OPT"
    --with-blaslapack-lib=/usr/lib/x86_64-linux-gnu/libopenblas.so
    --download-mumps --download-scalapack --download-metis
)
# DEFAULT=0 turns every optional component off; the ENABLE_*=0 below say it
# explicitly. CMAKE_POLICY_VERSION_MINIMUM is needed by CMake 4 (Ubuntu 26.04)
# and ignored by CMake 3 (24.04). GETDP_HOST / GETDP_PACKAGER replace the build
# host and user names that getdp --info would print.
GETDP_CMAKE=(
    -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    -DDEFAULT=0
    -DENABLE_KERNEL=1 -DENABLE_MPI=1 -DENABLE_PETSC=1
    -DENABLE_SLEPC=0 -DENABLE_PYTHON=0 -DENABLE_FORTRAN=0
    -DENABLE_ARPACK=0 -DENABLE_PEWE=0 -DENABLE_SPARSKIT=0 -DENABLE_ZITSOL=0
    -DENABLE_BLAS_LAPACK=0 -DENABLE_GMSH=0 -DENABLE_GSL=0 -DENABLE_NR=0
    -DENABLE_MMA=0 -DENABLE_OCTAVE=0 -DENABLE_OPENMP=0 -DENABLE_NX=0
    -DENABLE_MULTIHARMONIC=0 -DENABLE_SMALLFEM=0 -DENABLE_PRIVATE_API=0
    -DENABLE_SYSTEM_CONTRIB=0 -DENABLE_WRAP_PYTHON=0
    -DGETDP_HOST=unspecified -DGETDP_PACKAGER=mpxlite
)
P4PY_TOOLS=("cython>=3" numpy setuptools wheel)
# OpenCASCADE: the four modules Gmsh uses; the toolkits they require from other
# modules (TKService, TKV3d, TKCAF, ... for DataExchange) are added by its
# CMake; every optional third-party product off. Default compiler flags.
OCCT_CMAKE=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    -DBUILD_LIBRARY_TYPE=Shared -DINSTALL_DIR_LAYOUT=Unix
    -DBUILD_MODULE_FoundationClasses=ON -DBUILD_MODULE_ModelingData=ON
    -DBUILD_MODULE_ModelingAlgorithms=ON -DBUILD_MODULE_DataExchange=ON
    -DBUILD_MODULE_Visualization=OFF -DBUILD_MODULE_ApplicationFramework=OFF
    -DBUILD_MODULE_Draw=OFF -DBUILD_MODULE_DETools=OFF
    -DBUILD_DOC_Overview=OFF -DBUILD_SAMPLES_QT=OFF
    -DUSE_TK=OFF -DUSE_FREETYPE=OFF -DUSE_XLIB=OFF -DUSE_OPENGL=OFF -DUSE_GLES2=OFF
    -DUSE_FREEIMAGE=OFF -DUSE_FFMPEG=OFF -DUSE_OPENVR=OFF -DUSE_RAPIDJSON=OFF
    -DUSE_DRACO=OFF -DUSE_TBB=OFF -DUSE_VTK=OFF -DUSE_EIGEN=OFF
)
# FLTK: static libraries, position-independent (they are linked into the Gmsh
# library), with OpenGL and threads (gmsh.fltk.lock), Xft fonts, the image
# libraries of the system (the bundled jpeg, png and zlib are not built), and
# none of the optional X11 extensions Gmsh does without (Xinerama, Xfixes,
# Xcursor), no Cairo; no documentation, no test programs. Default compiler
# flags.
FLTK_CMAKE=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    -DOPTION_BUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DOPTION_USE_GL=ON -DOPTION_USE_THREADS=ON
    -DOPTION_USE_SYSTEM_ZLIB=ON -DOPTION_USE_SYSTEM_LIBJPEG=ON -DOPTION_USE_SYSTEM_LIBPNG=ON
    -DOPTION_USE_XFT=ON -DOPTION_USE_XRENDER=ON -DOPTION_USE_XDBE=ON
    -DOPTION_USE_XINERAMA=OFF -DOPTION_USE_XFIXES=OFF -DOPTION_USE_XCURSOR=OFF
    -DOPTION_CAIRO=OFF -DOPTION_CAIROEXT=OFF -DOPTION_CREATE_LINKS=OFF
    -DOPTION_BUILD_HTML_DOCUMENTATION=OFF -DOPTION_BUILD_PDF_DOCUMENTATION=OFF
    -DFLTK_BUILD_TEST=OFF
)
# Gmsh: the shared library with its Python module, and the gmsh executable
# linked to it (ENABLE_BUILD_DYNAMIC); the graphical interface on the FLTK
# above (fltk-config of that part), with OpenGL and the system libjpeg, libpng
# and zlib; the code of contrib/ as Gmsh carries it (ENABLE_SYSTEM_CONTRIB=0);
# every other option that would use a library of the system off, explicitly;
# no MPEG encoder; OpenCASCADE with its CAF module (names and colours of
# STEP/IGES entities). Untangle off: its HLBFGS code is "freely available for
# non-commercial purposes" only (the WinslowUntangler does that work). Default
# compiler flags (no -march: the floating-point operations are those of the
# upstream builds).
GMSH_CMAKE=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    -DGMSH_RELEASE=1 -DGMSH_HOST=unspecified -DGMSH_PACKAGER=mpxlite
    -DENABLE_BUILD_SHARED=1 -DENABLE_BUILD_DYNAMIC=1 -DENABLE_BUILD_LIB=0
    -DENABLE_FLTK=1 -DENABLE_GRAPHICS=0 -DENABLE_OSMESA=0 -DENABLE_CAIRO=0
    -DENABLE_MPEG_ENCODE=0 -DENABLE_TOUCHBAR=0
    -DENABLE_OCC=1 -DENABLE_OCC_CAF=1 -DENABLE_OCC_STATIC=0 -DENABLE_OCC_TBB=0
    -DENABLE_SYSTEM_CONTRIB=0 -DENABLE_EIGEN=1 -DENABLE_BLAS_LAPACK=0 -DENABLE_OPENMP=1
    -DENABLE_PETSC=0 -DENABLE_SLEPC=0 -DENABLE_MUMPS=0 -DENABLE_MPI=0 -DENABLE_PETSC4PY=0
    -DENABLE_MED=0 -DENABLE_CGNS=0 -DENABLE_GMP=0 -DENABLE_MMG=0 -DENABLE_MESQUITE=0
    -DENABLE_GEOMETRYCENTRAL=0 -DENABLE_POPPLER=0 -DENABLE_P4EST=0 -DENABLE_TCMALLOC=0
    -DENABLE_GETDP=0 -DENABLE_TESTS=0 -DENABLE_ZIPPER=0 -DENABLE_REVOROPT=0
    -DENABLE_OPENACC=0 -DENABLE_3M=0 -DENABLE_PRIVATE_API=0 -DENABLE_NUMPY=0
    -DENABLE_WRAP_PYTHON=0 -DENABLE_WRAP_JAVA=0 -DENABLE_UNTANGLE=0
)

# --- keys --------------------------------------------------------------------
OS_RELEASE="$(. /etc/os-release && echo "${ID:-unknown}-${VERSION_ID:-unknown}")"
sha() { sha256sum | cut -c1-20; }
inputs() {
    case "$1" in
        petsc)
            printf '%s\n' "part=petsc" "os=$OS_RELEASE" "recipe=$PETSC_RECIPE" \
                "commit=$PETSC_COMMIT" "mumps_sha256=$MUMPS_SHA256" \
                "configure=${PETSC_CONFIGURE[*]}" ;;
        getdp)
            [ -f "$PATCH" ] || { echo "mpx3 patch not found next to this script (../getdp/patches/)" >&2; exit 1; }
            printf '%s\n' "part=getdp" "os=$OS_RELEASE" "recipe=$GETDP_RECIPE" \
                "commit=$GETDP_COMMIT" "patch_sha256=$(sha256sum < "$PATCH" | cut -d' ' -f1)" \
                "cmake=${GETDP_CMAKE[*]}" "requires=$(key petsc)" ;;
        petsc4py)
            printf '%s\n' "part=petsc4py" "os=$OS_RELEASE" "recipe=$P4PY_RECIPE" \
                "python=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')" \
                "tools=${P4PY_TOOLS[*]}" "requires=$(key petsc)" ;;
        occt)
            printf '%s\n' "part=occt" "os=$OS_RELEASE" "recipe=$OCCT_RECIPE" \
                "version=$OCCT_VERSION" "archive_sha256=$OCCT_SHA256" "cmake=${OCCT_CMAKE[*]}" ;;
        fltk)
            printf '%s\n' "part=fltk" "os=$OS_RELEASE" "recipe=$FLTK_RECIPE" \
                "version=$FLTK_VERSION" "archive_sha256=$FLTK_SHA256" "cmake=${FLTK_CMAKE[*]}" ;;
        gmsh)
            printf '%s\n' "part=gmsh" "os=$OS_RELEASE" "recipe=$GMSH_RECIPE" \
                "version=$GMSH_VERSION" "archive_sha256=$GMSH_SHA256" "cmake=${GMSH_CMAKE[*]}" \
                "requires=$(key occt)" "requires=$(key fltk)" ;;
        *) echo "unknown part '$1' (parts: ${PARTS[*]})" >&2; exit 2 ;;
    esac
}
key() { local text; text="$(inputs "$1")"; printf '%s-%s\n' "$1" "$(printf '%s\n' "$text" | sha)"; }
# The directories of each part, relative to the prefix (all it writes, its
# record included).
dirs() {
    case "$1" in
        petsc) echo "petsc-3.26.0-complex" ;;
        getdp) printf '%s\n' "getdp-3.5.0-mpx3" "getdp-src-mpx3" ;;
        petsc4py) echo "petsc4py-3.26.0" ;;
        occt) echo "occt-$OCCT_VERSION" ;;
        fltk) echo "fltk-$FLTK_VERSION" ;;
        gmsh) echo "gmsh-$GMSH_VERSION" ;;
        *) echo "unknown part '$1' (parts: ${PARTS[*]})" >&2; exit 2 ;;
    esac
}
if [ -n "$PRINT" ]; then
    case "$PRINT" in
        key) key "$PRINT_PART" ;;
        inputs) inputs "$PRINT_PART" ;;
        dirs) dirs "$PRINT_PART" ;;
    esac
    exit 0
fi

[ -f "$PATCH" ] || { echo "mpx3 patch not found next to this script (../getdp/patches/)" >&2; exit 1; }

# Clean build environment: nothing inherited from other toolchains.
unset LD_LIBRARY_PATH CPATH LIBRARY_PATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH \
      PETSC_DIR SLEPC_DIR PYTHONPATH SPACK_ROOT SPACK_PYTHON CASROOT CMAKE_PREFIX_PATH || true
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export OMP_NUM_THREADS=1
export HWLOC_COMPONENTS=-gl
# 'make check' runs MPI programs: allow root (containers, CI), more ranks
# than cores, and shared memory without CMA (unprivileged containers).
if [ "$(id -u)" = 0 ]; then
    export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
fi
export OMPI_MCA_btl_vader_single_copy_mechanism=none OMPI_MCA_btl_sm_single_copy_mechanism=none
export OMPI_MCA_rmaps_base_oversubscribe=1 PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe

mkdir -p "$PREFIX/downloads"
PREFIX="$(cd "$PREFIX" && pwd)"
DL="$PREFIX/downloads"
PETSC_DIR="$PREFIX/$(dirs petsc)"
GETDP_PREFIX="$PREFIX/$(dirs getdp | sed -n 1p)"
GETDP_SRC="$PREFIX/$(dirs getdp | sed -n 2p)"
P4PY_SRC="$PETSC_DIR/src/binding/petsc4py"
P4PY_DIR="$PREFIX/$(dirs petsc4py)"
OCCT_PREFIX="$PREFIX/$(dirs occt)"
FLTK_PREFIX="$PREFIX/$(dirs fltk)"
GMSH_PREFIX="$PREFIX/$(dirs gmsh)"
BUILD_TREES="$PREFIX/build"
PY_TAG="$(python3 -c 'import sys; print("cp%d%d" % sys.version_info[:2])')"
INFO="$PREFIX/engine-info"
MACHINEINFO="$PETSC_DIR/$PETSC_ARCH/include/petscmachineinfo.h"
now() { date +%s; }
# git of the build trees, whoever owns them: a tree cloned by another user
# (container, CI cache, sudo) is otherwise refused as "dubious ownership".
g() { git -c safe.directory='*' -C "$@"; }

# The record of a part: <part directory>/mpxlite-part/ (inputs, seconds, ...).
part_dir() {
    case "$1" in
        petsc) echo "$PETSC_DIR/$PETSC_ARCH/mpxlite-part" ;;
        getdp) echo "$GETDP_PREFIX/mpxlite-part" ;;
        petsc4py) echo "$P4PY_DIR/mpxlite-part" ;;
        occt) echo "$OCCT_PREFIX/mpxlite-part" ;;
        fltk) echo "$FLTK_PREFIX/mpxlite-part" ;;
        gmsh) echo "$GMSH_PREFIX/mpxlite-part" ;;
    esac
}
# The products of a part are all there.
products() {
    case "$1" in
        petsc) [ -f "$PETSC_DIR/$PETSC_ARCH/lib/libpetsc.so" ] && [ -f "$(part_dir petsc)/mumps-verified" ] \
                   && machineinfo_clean ;;
        getdp) [ -x "$GETDP_PREFIX/bin/getdp" ] && [ -f "$GETDP_SRC/build/CMakeCache.txt" ] \
                   && [ "$(g "$GETDP_SRC" rev-parse HEAD 2>/dev/null)" = "$GETDP_COMMIT" ] ;;
        petsc4py) [ -n "$(p4py_wheel)" ] && [ -f "$P4PY_DIR/build-tools.txt" ] ;;
        occt) [ -f "$OCCT_PREFIX/lib/libTKernel.so" ] && [ -d "$(part_dir occt)/licenses" ] ;;
        fltk) [ -f "$FLTK_PREFIX/lib/libfltk.a" ] && [ -f "$FLTK_PREFIX/lib/libfltk_gl.a" ] \
                  && [ -f "$FLTK_PREFIX/lib/libfltk_images.a" ] && [ -x "$FLTK_PREFIX/bin/fltk-config" ] \
                  && [ -f "$(part_dir fltk)/licenses/COPYING" ] ;;
        gmsh) [ -f "$GMSH_PREFIX/lib/$GMSH_SONAME" ] && [ -f "$GMSH_PREFIX/lib/gmsh.py" ] && [ -x "$GMSH_PREFIX/bin/gmsh" ] \
                  && [ -f "$(part_dir gmsh)/build-info.txt" ] && [ -f "$(part_dir gmsh)/contrib-files.txt" ] ;;
    esac
}
# A part is current when its record has its inputs of today and its products are there.
current() {
    [ "$FORCE" = 0 ] && [ -f "$(part_dir "$1")/inputs" ] \
        && [ "$(cat "$(part_dir "$1")/inputs")" = "$(inputs "$1")" ] && products "$1"
}
# Write the record of part $1 built in $2 seconds.
record() {
    local d
    d="$(part_dir "$1")"
    mkdir -p "$d"
    echo "$2" > "$d/seconds"
    key "$1" > "$d/key"
    inputs "$1" > "$d/inputs"
    echo "    $1 built in $2 s (key $(cat "$d/key"))"
}
skip() { echo "    up to date ($(cat "$(part_dir "$1")/key")), skipping"; }

# PETSc compiles the build machine's host name and kernel release into
# libpetsc (petscmachineinfo.h, printed by -log_view): replace them with
# neutral values, so that the libraries carry no trace of the build machine.
sanitize_machineinfo() {
    sed -i -E \
        -e 's/^("Libraries compiled on [0-9-]+ [0-9:]+ on )[^\\"]*(\\n")/\1unspecified\2/' \
        -e 's/^("Machine characteristics: )[^\\"]*(\\n")/\1Linux-x86_64\2/' "$MACHINEINFO"
}
machineinfo_clean() {
    [ -f "$MACHINEINFO" ] \
        && grep -q '^"Libraries compiled on .* on unspecified\\n"' "$MACHINEINFO" \
        && grep -q '^"Machine characteristics: Linux-x86_64\\n"' "$MACHINEINFO"
}
fetch() { python3 -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$1" "$2"; }
# fetch_pinned NAME URL SHA256: $DL/NAME, the file of that checksum, from URL,
# or else from ENGINE_SOURCES_REPO.
fetch_pinned() {
    local name="$1" url="$2" sum="$3" f="$DL/$1"
    if [ -f "$f" ] && echo "$sum  $f" | sha256sum -c --status; then return 0; fi
    rm -f "$f" "$f.part"
    if fetch "$url" "$f.part" 2>/dev/null && echo "$sum  $f.part" | sha256sum -c --status; then
        mv "$f.part" "$f"
    else
        echo "    $url unreachable or not the pinned file: taking $name from $ENGINE_SOURCES_FILES"
        rm -f "$f.part"
        fetch "$ENGINE_SOURCES_FILES/$name" "$f.part" || { echo "cannot download $name" >&2; exit 1; }
        echo "$sum  $f.part" | sha256sum -c - || { rm -f "$f.part"; echo "$name: checksum mismatch" >&2; exit 1; }
        mv "$f.part" "$f"
    fi
    echo "    $name: sha256 $sum"
}
p4py_wheel() { ls "$P4PY_DIR"/petsc4py-*-"$PY_TAG"-"$PY_TAG"-*.whl 2>/dev/null | head -n 1; }

echo "=== [1/7] PETSc 3.26.0 complex (MUMPS, ScaLAPACK, METIS) -> $PETSC_DIR"
if current petsc; then
    skip petsc
else
    t0=$(now)
    if [ -d "$PETSC_DIR/.git" ]; then
        # A clean tree at the pinned commit: the previous build is removed.
        g "$PETSC_DIR" checkout -q -f "$PETSC_COMMIT" 2>/dev/null \
            || { g "$PETSC_DIR" fetch -q origin; g "$PETSC_DIR" checkout -q -f "$PETSC_COMMIT"; }
        g "$PETSC_DIR" clean -q -f -d -x
    else
        rm -rf "$PETSC_DIR"
        git clone -b release "$PETSC_GIT" "$PETSC_DIR"
        g "$PETSC_DIR" checkout -q "$PETSC_COMMIT"
    fi
    cd "$PETSC_DIR"
    python3 ./configure "${PETSC_CONFIGURE[@]}" --with-make-np="$JOBS"
    # PETSc deletes the MUMPS tarball after extracting it: fetch the same
    # URL once more, check it against the pinned checksum, and check that
    # every file in it is identical in the tree PETSc built from.
    MUMPS_SRC="$(ls -d "$PETSC_DIR/$PETSC_ARCH"/externalpackages/MUMPS_*)"
    MUMPS_VER="${MUMPS_SRC##*/MUMPS_}"
    MUMPS_URL="https://mumps-solver.org/MUMPS_$MUMPS_VER.tar.gz"
    grep -q "https://mumps-solver.org/MUMPS_'+self.version+'.tar.gz" \
        config/BuildSystem/config/packages/MUMPS.py \
        || { echo "PETSc no longer downloads MUMPS from $MUMPS_URL: update this script" >&2; exit 1; }
    TARBALL="$DL/MUMPS_$MUMPS_VER.tar.gz"
    if ! fetch "$MUMPS_URL" "$TARBALL"; then
        echo "    $MUMPS_URL unreachable: taking the same archive from $ENGINE_SOURCES_FILES"
        fetch "$ENGINE_SOURCES_FILES/MUMPS_$MUMPS_VER.tar.gz" "$TARBALL"
    fi
    echo "$MUMPS_SHA256  $TARBALL" | sha256sum -c -
    CHK="$(mktemp -d)"
    tar -C "$CHK" -xzf "$TARBALL"
    (cd "$CHK/MUMPS_$MUMPS_VER" && find . -type f -print0 | while IFS= read -r -d '' f; do
        cmp -s "$f" "$MUMPS_SRC/$f" || { echo "MUMPS source differs from the tarball: $f" >&2; exit 1; }
    done)
    rm -rf "$CHK"
    sanitize_machineinfo
    make PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" all
    make PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" check
    machineinfo_clean || { echo "could not neutralise $MACHINEINFO" >&2; exit 1; }
    mkdir -p "$(part_dir petsc)"
    echo "$MUMPS_URL $MUMPS_SHA256" > "$(part_dir petsc)/mumps-verified"
    record petsc $(( $(now) - t0 ))
fi

echo "=== [2/7] GetDP 3.5.0 + mpx3 -> $GETDP_PREFIX"
if current getdp; then
    skip getdp
else
    t0=$(now)
    rm -rf "$GETDP_PREFIX"
    if [ ! -d "$GETDP_SRC/.git" ]; then
        rm -rf "$GETDP_SRC"
        if ! git clone "$GETDP_GIT" "$GETDP_SRC"; then
            echo "    $GETDP_GIT unreachable: cloning the copy $GETDP_COPY_GIT"
            rm -rf "$GETDP_SRC"
            git clone "$GETDP_COPY_GIT" "$GETDP_SRC"
            g "$GETDP_SRC" tag "$GETDP_TAG" "$GETDP_COMMIT"
        fi
    fi
    # The tag, clean (no earlier patch, no earlier build), then the patch.
    g "$GETDP_SRC" checkout -q -f "$GETDP_TAG"
    g "$GETDP_SRC" clean -q -f -d -x
    [ "$(g "$GETDP_SRC" rev-parse HEAD)" = "$GETDP_COMMIT" ] || { echo "tag $GETDP_TAG is not $GETDP_COMMIT" >&2; exit 1; }
    g "$GETDP_SRC" apply "$PATCH"
    mkdir "$GETDP_SRC/build" && cd "$GETDP_SRC/build"
    PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" cmake .. "${GETDP_CMAKE[@]}" \
        -DCMAKE_INSTALL_PREFIX="$GETDP_PREFIX" \
        -DCMAKE_INSTALL_RPATH="$PETSC_DIR/$PETSC_ARCH/lib"
    make -j "$JOBS"
    make install
    record getdp $(( $(now) - t0 ))
fi

echo "=== [3/7] petsc4py 3.26.0 for $PY_TAG -> $P4PY_DIR"
# The bindings in the PETSc tree, at the PETSc commit, compiled against the
# PETSc just built. The wheel is for the python3 of this machine: the Python
# of the Ubuntu release the bundle targets.
if current petsc4py; then
    skip petsc4py
else
    t0=$(now)
    rm -rf "$P4PY_DIR"
    mkdir -p "$P4PY_DIR"
    P4PY_ENV="$P4PY_DIR/build-env"
    python3 -m venv "$P4PY_ENV"
    "$P4PY_ENV/bin/python" -m pip install -q --disable-pip-version-check --upgrade pip
    "$P4PY_ENV/bin/python" -m pip install -q --disable-pip-version-check "${P4PY_TOOLS[@]}"
    # Built from a copy: the PETSc tree stays as checked out.
    cp -a "$P4PY_SRC" "$P4PY_DIR/src"
    PETSC_DIR="$PETSC_DIR" PETSC_ARCH="$PETSC_ARCH" "$P4PY_ENV/bin/python" -m pip wheel -q \
        --disable-pip-version-check --no-deps --no-build-isolation \
        --wheel-dir "$P4PY_DIR" "$P4PY_DIR/src"
    [ -n "$(p4py_wheel)" ] || { echo "no petsc4py wheel for $PY_TAG in $P4PY_DIR" >&2; exit 1; }
    # The wheel must load, on complex scalars, with MUMPS.
    "$P4PY_ENV/bin/python" -m pip install -q --disable-pip-version-check --no-deps "$(p4py_wheel)"
    "$P4PY_ENV/bin/python" -c '
import petsc4py
petsc4py.init([])
from petsc4py import PETSc
assert issubclass(PETSc.ScalarType, complex), PETSc.ScalarType
assert PETSc.Sys.hasExternalPackage("mumps"), "petsc4py: PETSc without MUMPS"
print("    petsc4py", petsc4py.__version__, "PETSc", PETSc.Sys.getVersion(), PETSc.ScalarType.__name__)'
    "$P4PY_ENV/bin/python" -m pip list --disable-pip-version-check --format=freeze 2>/dev/null \
        | grep -i -E '^(cython|numpy|setuptools|wheel|pip)=' | tr '\n' ' ' | sed 's/ $//' \
        > "$P4PY_DIR/build-tools.txt"
    rm -rf "$P4PY_ENV" "$P4PY_DIR/src"
    record petsc4py $(( $(now) - t0 ))
fi
P4PY_WHEEL="$(p4py_wheel)"

echo "=== [4/7] OpenCASCADE $OCCT_VERSION -> $OCCT_PREFIX"
if current occt; then
    skip occt
else
    t0=$(now)
    rm -rf "$OCCT_PREFIX" "$BUILD_TREES/occt"
    fetch_pinned "$OCCT_ARCHIVE" "$OCCT_URL" "$OCCT_SHA256"
    mkdir -p "$BUILD_TREES/occt/build"
    tar -C "$BUILD_TREES/occt" -xzf "$DL/$OCCT_ARCHIVE"
    OCCT_SRC="$BUILD_TREES/occt/OCCT-${OCCT_TAG#V}"
    grep -q -E "^#define OCC_VERSION_COMPLETE +\"$OCCT_VERSION\"" "$OCCT_SRC/src/Standard/Standard_Version.hxx" \
        || { echo "$OCCT_ARCHIVE is not OpenCASCADE $OCCT_VERSION" >&2; exit 1; }
    cd "$BUILD_TREES/occt/build"
    cmake "$OCCT_SRC" "${OCCT_CMAKE[@]}" -DCMAKE_INSTALL_PREFIX="$OCCT_PREFIX" \
        -DCMAKE_INSTALL_RPATH="$OCCT_PREFIX/lib"
    make -j "$JOBS"
    make install
    # Only the libraries and the headers are kept (no environment scripts,
    # resources nor CMake files).
    rm -rf "$OCCT_PREFIX/bin" "$OCCT_PREFIX/share" "$OCCT_PREFIX/lib/cmake"
    mkdir -p "$(part_dir occt)/licenses"
    cp "$OCCT_SRC/LICENSE_LGPL_21.txt" "$OCCT_SRC/OCCT_LGPL_EXCEPTION.txt" "$(part_dir occt)/licenses/"
    (cd "$OCCT_PREFIX/lib" && ls libTK*.so | sed 's/\.so$//' | sort) > "$(part_dir occt)/toolkits.txt"
    # No library of the system beyond the C/C++ runtime.
    for so in "$OCCT_PREFIX"/lib/libTK*.so; do
        readelf -d "$so" | sed -n 's/.*(NEEDED).*\[\(.*\)\]$/\1/p'
    done | sort -u | grep -v -x -E 'libTK[A-Za-z0-9]+\.so\.[0-9.]+|lib(c|m|dl|rt|pthread|stdc\+\+|gcc_s)\.so\.[0-9]+|ld-linux-x86-64\.so\.2' \
        > "$BUILD_TREES/occt/unexpected" || true
    [ ! -s "$BUILD_TREES/occt/unexpected" ] \
        || { echo "OpenCASCADE links libraries of the system: $(tr '\n' ' ' < "$BUILD_TREES/occt/unexpected")" >&2; exit 1; }
    rm -rf "$BUILD_TREES/occt"
    record occt $(( $(now) - t0 ))
fi

echo "=== [5/7] FLTK $FLTK_VERSION (static libraries, OpenGL, X11) -> $FLTK_PREFIX"
if current fltk; then
    skip fltk
else
    t0=$(now)
    rm -rf "$FLTK_PREFIX" "$BUILD_TREES/fltk"
    fetch_pinned "$FLTK_ARCHIVE" "$FLTK_URL" "$FLTK_SHA256"
    mkdir -p "$BUILD_TREES/fltk/build"
    tar -C "$BUILD_TREES/fltk" -xzf "$DL/$FLTK_ARCHIVE"
    FLTK_SRC="$BUILD_TREES/fltk/fltk-$FLTK_VERSION"
    [ "$(tr -d '[:space:]' < "$FLTK_SRC/fltk_version.dat")" = "$FLTK_VERSION" ] \
        || { echo "$FLTK_ARCHIVE is not FLTK $FLTK_VERSION" >&2; exit 1; }
    cd "$BUILD_TREES/fltk/build"
    cmake "$FLTK_SRC" "${FLTK_CMAKE[@]}" -DCMAKE_INSTALL_PREFIX="$FLTK_PREFIX" | tee cmake.out
    make -j "$JOBS" fltk fltk_gl fltk_images
    # Only the libraries Gmsh links, their headers and fltk-config (by which
    # Gmsh finds them): no fluid, no forms library, no documentation.
    mkdir -p "$FLTK_PREFIX/lib" "$FLTK_PREFIX/bin" "$FLTK_PREFIX/include" "$(part_dir fltk)/licenses"
    for l in fltk fltk_gl fltk_images; do install -m 644 "lib/lib$l.a" "$FLTK_PREFIX/lib/"; done
    cp -r "$FLTK_SRC/FL" "$FLTK_PREFIX/include/FL"
    install -m 644 FL/abi-version.h "$FLTK_PREFIX/include/FL/abi-version.h"
    install -m 755 bin/fltk-config "$FLTK_PREFIX/bin/fltk-config"
    # The image libraries are those of the system: the copies FLTK carries
    # (jpeg, png, zlib) are not built.
    if ls lib/libfltk_jpeg.a lib/libfltk_png.a lib/libfltk_z.a 2>/dev/null | grep -q .; then
        echo "FLTK built its own jpeg, png or zlib (libjpeg-dev, libpng-dev, zlib1g-dev missing?)" >&2; exit 1
    fi
    for opt in HAVE_GL HAVE_GL_GLU_H USE_XFT HAVE_XDBE HAVE_LIBJPEG HAVE_LIBPNG HAVE_LIBZ HAVE_PTHREAD; do
        grep -q "^#define $opt 1" config.h || { echo "FLTK configured without $opt" >&2; exit 1; }
    done
    for opt in HAVE_XINERAMA HAVE_XFIXES HAVE_XCURSOR FLTK_HAVE_CAIRO; do
        if grep -q "^#define $opt 1" config.h; then echo "FLTK configured with $opt" >&2; exit 1; fi
    done
    grep -E '^#define (HAVE_|USE_)' config.h > "$(part_dir fltk)/config.txt"
    "$FLTK_PREFIX/bin/fltk-config" --use-gl --use-images --ldflags > "$(part_dir fltk)/ldflags.txt"
    # Licence (the LGPL with the FLTK exceptions); the leading comment of each
    # source file compiled into the three libraries that carries a permission
    # notice of its own (once per distinct text).
    cp "$FLTK_SRC/COPYING" "$FLTK_SRC/CREDITS" "$(part_dir fltk)/licenses/"
    find src/CMakeFiles/fltk.dir src/CMakeFiles/fltk_gl.dir src/CMakeFiles/fltk_images.dir -name '*.o.d' \
        -exec cat {} + | tr ' ' '\n' | sed 's/:$//' | grep -E '\.(c|cxx|h|H)$' | xargs -r realpath -m \
        | sed -n "s|^$FLTK_SRC/||p" | grep -v '^build/' | LC_ALL=C sort -u > "$(part_dir fltk)/source-files.txt"
    [ -s "$(part_dir fltk)/source-files.txt" ] || { echo "no source file found in the FLTK build" >&2; exit 1; }
    while read -r f; do
        awk 'NR > 80 {exit}
             /^[[:space:]]*(\/\/|\/\*|\*|$)/ || inblock {
                 print; if ($0 ~ /\/\*/) inblock = 1; if ($0 ~ /\*\//) inblock = 0; next }
             {exit}' "$FLTK_SRC/$f" > "$BUILD_TREES/fltk/notice"
        if grep -q -i 'permission' "$BUILD_TREES/fltk/notice"; then
            n="$(sha256sum < "$BUILD_TREES/fltk/notice" | cut -c1-12)"
            if [ ! -f "$BUILD_TREES/fltk/seen-$n" ]; then
                touch "$BUILD_TREES/fltk/seen-$n"
                echo "== $f"
                cat "$BUILD_TREES/fltk/notice"
                echo
            fi
        fi
    done < "$(part_dir fltk)/source-files.txt" > "$(part_dir fltk)/licenses/NOTICES-from-sources.txt"
    [ -s "$(part_dir fltk)/licenses/NOTICES-from-sources.txt" ] || rm "$(part_dir fltk)/licenses/NOTICES-from-sources.txt"
    rm -rf "$BUILD_TREES/fltk"
    rmdir "$BUILD_TREES" 2>/dev/null || true
    record fltk $(( $(now) - t0 ))
fi

echo "=== [6/7] Gmsh $GMSH_VERSION (library, Python module, gmsh executable, graphical interface) -> $GMSH_PREFIX"
if current gmsh; then
    skip gmsh
else
    t0=$(now)
    rm -rf "$GMSH_PREFIX" "$BUILD_TREES/gmsh"
    fetch_pinned "$GMSH_ARCHIVE" "$GMSH_URL" "$GMSH_SHA256"
    mkdir -p "$BUILD_TREES/gmsh/build"
    tar -C "$BUILD_TREES/gmsh" -xzf "$DL/$GMSH_ARCHIVE"
    GMSH_SRC="$BUILD_TREES/gmsh/gmsh-$GMSH_VERSION-source"
    cd "$BUILD_TREES/gmsh/build"
    # The executable finds the library in $GMSH_PREFIX/lib (the library itself
    # finds OpenCASCADE through its build RUNPATH).
    CASROOT="$OCCT_PREFIX" cmake "$GMSH_SRC" "${GMSH_CMAKE[@]}" \
        -DFLTK_CONFIG_SCRIPT="$FLTK_PREFIX/bin/fltk-config" \
        -DCMAKE_BUILD_RPATH="$OCCT_PREFIX/lib" \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,$GMSH_PREFIX/lib" | tee cmake.out
    grep -q "Found OpenCASCADE version $OCCT_VERSION in $OCCT_PREFIX/include/opencascade" cmake.out \
        || { echo "Gmsh is not configured with the OpenCASCADE of $OCCT_PREFIX" >&2; exit 1; }
    grep -q "Using fltk-config script for Fltk ${FLTK_VERSION%.*}$" cmake.out \
        && grep -q "^FLTK_CONFIG_SCRIPT:FILEPATH=$FLTK_PREFIX/bin/fltk-config$" CMakeCache.txt \
        && [ "$("$FLTK_PREFIX/bin/fltk-config" --version)" = "$FLTK_VERSION" ] \
        || { echo "Gmsh is not configured with the FLTK of $FLTK_PREFIX" >&2; exit 1; }
    make -j "$JOBS" shared gmsh
    mkdir -p "$GMSH_PREFIX/lib" "$GMSH_PREFIX/bin" "$(part_dir gmsh)/licenses"
    LIBFILE="$(readlink -f "$GMSH_SONAME")"
    install -m 755 "$LIBFILE" "$GMSH_PREFIX/lib/"
    ln -s "$(basename "$LIBFILE")" "$GMSH_PREFIX/lib/$GMSH_SONAME"
    install -m 644 "$GMSH_SRC/api/gmsh.py" "$GMSH_PREFIX/lib/gmsh.py"
    install -m 755 gmsh "$GMSH_PREFIX/bin/gmsh"
    # The library loads OpenCASCADE from $OCCT_PREFIX and, of the system, the
    # C/C++ and OpenMP runtimes, X11 with Xext, Xft and Xrender, fontconfig,
    # OpenGL and GLU, libjpeg, libpng and zlib (FLTK is inside it); the
    # executable loads the library and the C/C++ runtimes.
    GMSH_SYSTEM_LIBS='lib(c|m|dl|rt|pthread|stdc\+\+|gcc_s|gomp)\.so\.[0-9]+|ld-linux-x86-64\.so\.2'
    GMSH_GUI_LIBS='lib(X11|Xext|Xft|Xrender|fontconfig|freetype|GL|GLU|jpeg|png16|z)\.so\.[0-9]+'
    bad="$(readelf -d "$LIBFILE" | sed -n 's/.*(NEEDED).*\[\(.*\)\]$/\1/p' \
        | grep -v -x -E "libTK[A-Za-z0-9]+\\.so\\.[0-9.]+|$GMSH_SYSTEM_LIBS|$GMSH_GUI_LIBS" || true)"
    [ -z "$bad" ] || { echo "the Gmsh library loads unexpected libraries: $bad" >&2; exit 1; }
    bad="$(readelf -d "$GMSH_PREFIX/bin/gmsh" | sed -n 's/.*(NEEDED).*\[\(.*\)\]$/\1/p' \
        | grep -v -x -E "$GMSH_SONAME|$GMSH_SYSTEM_LIBS" || true)"
    [ -z "$bad" ] || { echo "the gmsh executable loads unexpected libraries: $bad" >&2; exit 1; }
    for f in "$LIBFILE" "$GMSH_PREFIX/bin/gmsh"; do
        if env -i ldd "$f" | grep -E 'not found'; then echo "$f does not resolve" >&2; exit 1; fi
    done
    # FLTK is linked statically: no symbol of it is exported.
    if nm -D --defined-only "$LIBFILE" | awk '{print $NF}' | grep -q -E '^_ZN?[0-9]*Fl_'; then
        echo "the Gmsh library exports FLTK symbols" >&2; exit 1
    fi
    # Licence and notices; the contributed code compiled in (each contrib/<name>
    # with a source or header in the library) with its licence and notice files.
    cp "$GMSH_SRC/LICENSE.txt" "$GMSH_SRC/CREDITS.txt" "$(part_dir gmsh)/licenses/"
    # (from the dependency files of the compiler: the sources and headers
    # compiled into the library, recorded in contrib-files.txt)
    find CMakeFiles/shared.dir -name '*.o.d' -exec cat {} + | tr ' ' '\n' | grep -F "/contrib/" \
        | sed 's/:$//' | xargs -r realpath -m | sed -n "s|^$GMSH_SRC/\(contrib/[^/]*/.*\)|\1|p" \
        | LC_ALL=C sort -u > "$(part_dir gmsh)/contrib-files.txt"
    cut -d/ -f2 "$(part_dir gmsh)/contrib-files.txt" | LC_ALL=C sort -u > "$(part_dir gmsh)/contrib.txt"
    [ -s "$(part_dir gmsh)/contrib.txt" ] || { echo "no contributed code found in the Gmsh build" >&2; exit 1; }
    while read -r c; do
        mkdir -p "$(part_dir gmsh)/licenses/contrib/$c"
        find "$GMSH_SRC/contrib/$c" -maxdepth 1 -type f \
            \( -iname 'licen[cs]e*' -o -iname 'copying*' -o -iname 'readme*' \) \
            -exec cp {} "$(part_dir gmsh)/licenses/contrib/$c/" \;
    done < "$(part_dir gmsh)/contrib.txt"
    # The leading comment of each of those files that states a copyright or a
    # licence (once per distinct text): the notices of the code that has no
    # licence file of its own.
    while read -r c; do
        grep "^contrib/$c/" "$(part_dir gmsh)/contrib-files.txt" | while read -r f; do
            awk 'NR > 80 {exit}
                 /^[[:space:]]*(\/\/|\/\*|\*|$)/ || inblock {
                     print; if ($0 ~ /\/\*/) inblock = 1; if ($0 ~ /\*\//) inblock = 0; next }
                 {exit}' "$GMSH_SRC/$f" > "$BUILD_TREES/gmsh/notice"
            if grep -q -i -E 'copyright|licen[cs]e|permission' "$BUILD_TREES/gmsh/notice"; then
                n="$(sha256sum < "$BUILD_TREES/gmsh/notice" | cut -c1-12)"
                if [ ! -f "$BUILD_TREES/gmsh/seen-$n" ]; then
                    touch "$BUILD_TREES/gmsh/seen-$n"
                    echo "== $f"
                    cat "$BUILD_TREES/gmsh/notice"
                    echo
                fi
            fi
        done > "$(part_dir gmsh)/licenses/contrib/$c/NOTICES-from-sources.txt"
        [ -s "$(part_dir gmsh)/licenses/contrib/$c/NOTICES-from-sources.txt" ] \
            || rm "$(part_dir gmsh)/licenses/contrib/$c/NOTICES-from-sources.txt"
    done < "$(part_dir gmsh)/contrib.txt"
    # Build information of the library, as Gmsh reports it.
    python3 - "$GMSH_PREFIX/lib" > "$(part_dir gmsh)/build-info.txt" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
import gmsh
gmsh.initialize(readConfigFiles=False)
for item in gmsh.option.getString("General.BuildInfo").split("; "):
    key, _, value = item.partition(": ")
    print(f"{key}: {value}")
gmsh.finalize()
EOF
    info_of() { awk -F': ' -v k="$1" '$1 == k {print $2}' "$(part_dir gmsh)/build-info.txt"; }
    [ "$(info_of Version)" = "$GMSH_VERSION" ] || { echo "Gmsh version $(info_of Version)" >&2; exit 1; }
    [ "$(info_of 'OCC version')" = "$OCCT_VERSION" ] || { echo "Gmsh on OCC $(info_of 'OCC version')" >&2; exit 1; }
    [ "$(info_of 'Packaged by')" = mpxlite ] && [ "$(info_of 'Build host')" = unspecified ] \
        || { echo "Gmsh names its build machine or user" >&2; exit 1; }
    for opt in OpenCASCADE OpenCASCADE-CAF Fltk OpenGL Jpeg Png Zlib; do
        grep -qw -- "$opt" <<< " $(info_of 'Build options') " \
            || { echo "Gmsh built without $opt (a build prerequisite missing?)" >&2; exit 1; }
    done
    if grep -qiE '(^| )(PETSc|MPI|Med|Cgns|Gmp|Untangle|Mpeg|Cairo|OSMesa)( |$)' <<< "$(info_of 'Build options')"; then
        echo "Gmsh built with an option that must be off: $(info_of 'Build options')" >&2; exit 1
    fi
    # The executable: the same build, and it runs without a display (batch).
    env -u DISPLAY -u WAYLAND_DISPLAY "$GMSH_PREFIX/bin/gmsh" -info > "$(part_dir gmsh)/gmsh-info.txt" 2>&1
    grep -q "^Version *: $GMSH_VERSION$" "$(part_dir gmsh)/gmsh-info.txt" \
        && grep -q "^Build options *: .*Fltk" "$(part_dir gmsh)/gmsh-info.txt" \
        || { echo "gmsh -info: $(cat "$(part_dir gmsh)/gmsh-info.txt")" >&2; exit 1; }
    rm -rf "$BUILD_TREES/gmsh"
    rmdir "$BUILD_TREES" 2>/dev/null || true
    record gmsh $(( $(now) - t0 ))
fi

echo "=== [7/7] version records and licence files -> $INFO"
EXT="$PETSC_DIR/$PETSC_ARCH/externalpackages"
CONF="$PETSC_DIR/$PETSC_ARCH/lib/petsc/conf"
rm -rf "$INFO"
mkdir -p "$INFO"/licenses/{getdp,petsc,petsc4py,mumps,scalapack,metis,opencascade,fltk,gmsh} "$INFO/patch"
L="$INFO/licenses"
# Lines of file $3 from the first match of regex $1 to the next match of
# regex $2, without the Fortran comment column.
excerpt() { awk -v a="$1" -v b="$2" '$0 ~ a {p=1} p {sub(/^[Cc!] ?/, ""); print} p && $0 ~ b {exit}' "$3"; }
# Full licence texts referenced by short notices (Ubuntu base-files).
common() { [ -f "/usr/share/common-licenses/$1" ] || { echo "/usr/share/common-licenses/$1 not found" >&2; exit 1; }
           cp "/usr/share/common-licenses/$1" "$2"; }

# GetDP: licence, credits (notices of the avl, KissFFT and picojson code
# compiled in) and the notices of the ONELAB and Gmsh socket headers.
cp "$GETDP_SRC/LICENSE.txt" "$GETDP_SRC/CREDITS.txt" "$L/getdp/"
for h in src/common/onelab.h src/common/GmshSocket.h; do
    echo "== $h"
    awk 'NR == 1 || /^\/\// {print; next} {exit}' "$GETDP_SRC/$h"
    echo
done > "$L/getdp/NOTICE-onelab-gmshsocket.txt"
common GPL-3 "$L/getdp/GPL-3.0.txt"
cp "$PATCH" "$PATCH_DIR/COPYING" "$PATCH_DIR/README.md" "$INFO/patch/"
# PETSc, and its Python bindings
cp "$PETSC_DIR/LICENSE" "$L/petsc/"
cp "$P4PY_SRC/LICENSE.rst" "$L/petsc4py/"
# MUMPS (+ PORD; AMD- and LAPACK-derived routines under BSD-3-Clause)
M="$(ls -d "$EXT"/MUMPS_*)"
MUMPS_VER="${M##*/MUMPS_}"
cp "$M/LICENSE" "$M/doc/CeCILL-C_V1-en.txt" "$M/doc/CeCILL-C_V1-fr.txt" "$L/mumps/"
cp "$M/PORD/README" "$L/mumps/PORD-README"
{
    echo "== src/ana_orderings.F"
    excerpt 'AMD, Copyright' 'EVEN IF ADVISED' "$M/src/ana_orderings.F"
    echo
    echo "== src/[sdcz]lr_core.F"
    excerpt 'Copyright \(c\) 1992-' 'SUCH DAMAGE' "$M/src/zlr_core.F"
} > "$L/mumps/NOTICE-BSD-3-Clause.txt"
# ScaLAPACK; METIS (licence notice + full Apache-2.0 text, NOTICE if any)
cp "$EXT/git.scalapack/LICENSE" "$L/scalapack/"
cp "$EXT/git.metis/LICENSE.txt" "$L/metis/"
common Apache-2.0 "$L/metis/Apache-2.0.txt"
for n in NOTICE NOTICE.txt GKlib/NOTICE GKlib/LICENSE.txt; do
    if [ -f "$EXT/git.metis/$n" ]; then cp "$EXT/git.metis/$n" "$L/metis/${n//\//-}"; fi
done
# OpenCASCADE (LGPL-2.1 and its exception); FLTK (the LGPL with the FLTK
# exceptions, credits, the permission notices of its sources); Gmsh (licence
# with its exception, credits, the licence and notice files of the contributed
# code compiled in)
cp "$(part_dir occt)"/licenses/* "$L/opencascade/"
cp "$(part_dir fltk)"/licenses/* "$L/fltk/"
cp "$(part_dir fltk)/config.txt" "$INFO/fltk-config.txt"
cp "$(part_dir fltk)/ldflags.txt" "$INFO/fltk-ldflags.txt"
cp "$(part_dir gmsh)/gmsh-info.txt" "$INFO/gmsh-info.txt"
cp -r "$(part_dir gmsh)"/licenses/. "$L/gmsh/"
cp "$(part_dir gmsh)/build-info.txt" "$INFO/gmsh-build-info.txt"
cp "$(part_dir gmsh)/contrib.txt" "$INFO/gmsh-contrib.txt"
cp "$(part_dir gmsh)/contrib-files.txt" "$INFO/gmsh-contrib-files.txt"
cp "$(part_dir occt)/toolkits.txt" "$INFO/occt-toolkits.txt"

# Every field of the records is determined here, before anything is written;
# a failing query stops the script (set -e), and check_versions below rejects
# an empty or malformed field.
PETSC_VER="$(awk '/define PETSC_VERSION_(MAJOR|MINOR|SUBMINOR) /{v = v (v ? "." : "") $3} END {print v}' \
    "$PETSC_DIR/include/petscversion.h")"
read -r MUMPS_URL MUMPS_SUM < "$(part_dir petsc)/mumps-verified"
GETDP_REV="$(g "$GETDP_SRC" rev-parse "$GETDP_TAG^{commit}")"
PATCH_SUM="$(sha256sum "$PATCH" | cut -d' ' -f1)"
PETSC_REV="$(g "$PETSC_DIR" rev-parse HEAD)"
P4PY_VER="$(basename "$P4PY_WHEEL" | cut -d- -f2)"
[ "$P4PY_VER" = "$PETSC_VER" ] || { echo "petsc4py $P4PY_VER is not the version of PETSc, $PETSC_VER" >&2; exit 1; }
SCALAPACK_VER="$(g "$EXT/git.scalapack" describe --tags --always)"
SCALAPACK_URL="$(g "$EXT/git.scalapack" remote get-url origin)"
SCALAPACK_REV="$(g "$EXT/git.scalapack" rev-parse HEAD)"
METIS_VER="$(g "$EXT/git.metis" describe --tags --always)"
METIS_URL="$(g "$EXT/git.metis" remote get-url origin)"
METIS_REV="$(g "$EXT/git.metis" rev-parse HEAD)"
[ "$GETDP_REV" = "$GETDP_COMMIT" ] || { echo "GetDP tree at $GETDP_REV, not $GETDP_COMMIT" >&2; exit 1; }
[ "$PETSC_REV" = "$PETSC_COMMIT" ] || { echo "PETSc tree at $PETSC_REV, not $PETSC_COMMIT" >&2; exit 1; }
[ "$MUMPS_SUM" = "$MUMPS_SHA256" ] || { echo "MUMPS record $MUMPS_SUM, not $MUMPS_SHA256" >&2; exit 1; }
lic() { (cd "$INFO" && find licenses/"$1" -type f | LC_ALL=C sort | tr '\n' ' ' | sed 's/ $//'); }
{
    printf 'component\tversion\tlicence\tupstream\trevision\tlicence_files\n'
    printf 'getdp\t3.5.0 (tag %s) + mpx3 patch\tGPL-2.0-or-later (binary linked with METIS: GPL-3.0-or-later)\t%s\tcommit %s; mpx3 patch sha256 %s\t%s\n' \
        "$GETDP_TAG" "$GETDP_UPSTREAM" "$GETDP_REV" "$PATCH_SUM" "$(lic getdp)"
    printf 'petsc\t%s\tBSD-2-Clause\t%s\tcommit %s (tag v%s)\t%s\n' \
        "$PETSC_VER" "$PETSC_GIT" "$PETSC_REV" "$PETSC_VER" "$(lic petsc)"
    printf 'petsc4py\t%s\tBSD-2-Clause\t%s\tcommit %s (tag v%s)\t%s\n' \
        "$P4PY_VER" "$PETSC_GIT" "$PETSC_REV" "$PETSC_VER" "$(lic petsc4py)"
    printf 'mumps\t%s\tCeCILL-C (PORD: public domain; AMD- and LAPACK-derived routines: BSD-3-Clause)\t%s\ttarball sha256 %s\t%s\n' \
        "$MUMPS_VER" "$MUMPS_URL" "$MUMPS_SUM" "$(lic mumps)"
    printf 'scalapack\t%s\tBSD-3-Clause\t%s\tcommit %s\t%s\n' \
        "$SCALAPACK_VER" "$SCALAPACK_URL" "$SCALAPACK_REV" "$(lic scalapack)"
    printf 'metis\t%s (PETSc-maintained METIS 5.1.0)\tApache-2.0\t%s\tcommit %s\t%s\n' \
        "$METIS_VER" "$METIS_URL" "$METIS_REV" "$(lic metis)"
    printf 'opencascade\t%s\tLGPL-2.1-only with the Open CASCADE exception\t%s\ttarball sha256 %s; tag %s commit %s (%s)\t%s\n' \
        "$OCCT_VERSION" "$OCCT_URL" "$OCCT_SHA256" "$OCCT_TAG" "$OCCT_COMMIT" "$OCCT_GIT" "$(lic opencascade)"
    printf 'fltk\t%s\tLGPL-2.0 with the FLTK exceptions (static linking permitted)\t%s\ttarball sha256 %s; tag %s commit %s (%s)\t%s\n' \
        "$FLTK_VERSION" "$FLTK_URL" "$FLTK_SHA256" "$FLTK_TAG" "$FLTK_COMMIT" "$FLTK_GIT" "$(lic fltk)"
    printf 'gmsh\t%s\tGPL-2.0-or-later with the exception of its LICENSE.txt (contributed code compiled in: own licences)\t%s\ttarball sha256 %s; tag %s commit %s (%s)\t%s\n' \
        "$GMSH_VERSION" "$GMSH_URL" "$GMSH_SHA256" "$GMSH_TAG" "$GMSH_COMMIT" "$GMSH_GIT" "$(lic gmsh)"
} > "$INFO/VERSIONS.tsv"
# The same check as in tools/bundle/make_bundle.sh: exactly the nine
# components, no empty field, commits of 40 and checksums of 64 hex digits,
# https upstream addresses, licence files that exist.
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
check_versions "$INFO/VERSIONS.tsv" || { echo "invalid records in $INFO/VERSIONS.tsv" >&2; exit 1; }
grep -h '^Configure Options' "$CONF/configure.log" > "$INFO/petsc-configure-options.txt"
cp "$PETSC_DIR/$PETSC_ARCH/include/petscconf.h" "$INFO/petscconf.h"
grep -E '^ENABLE_|^PYTHON|^LAPACK|^BLAS|^GSL|^CMAKE_(C|CXX)_FLAGS_RELEASE:' "$GETDP_SRC/build/CMakeCache.txt" \
    > "$INFO/getdp-cmake-cache.txt"
"$GETDP_PREFIX/bin/getdp" --info > "$INFO/getdp-info.txt" 2>&1 || true
(
    . /etc/os-release
    echo "os          $PRETTY_NAME $(uname -m)"
    echo "os_id       $ID"
    echo "os_version  $VERSION_ID"
    echo "built       $(date -u -r "$GETDP_PREFIX/bin/getdp" +%Y-%m-%dT%H:%M:%SZ)"
    echo "prefix      $PREFIX"
    echo "gcc         $(gcc --version | head -1)"
    echo "gfortran    $(gfortran --version | head -1)"
    echo "cmake       $(cmake --version | head -1)"
    echo "mpi         Open MPI $(ompi_info --parsable 2>/dev/null | sed -n 's/^ompi:version:full://p' | head -1)"
    echo "python      $PY_TAG ($(python3 --version 2>&1))"
    echo "petsc4py    $(basename "$P4PY_WHEEL"); built with $(cat "$P4PY_DIR/build-tools.txt")"
    echo "cflags      -O3 -march=x86-64-v3 -mtune=generic (PETSc, MUMPS, ScaLAPACK, METIS; needs AVX2/FMA/BMI2); OpenCASCADE, FLTK, Gmsh: CMake Release defaults"
    echo "fltk        $FLTK_VERSION (static, in the Gmsh library); $(sed -n 's/^#define \(USE_XFT\|HAVE_GL\|HAVE_XDBE\) 1$/\1/p' "$INFO/fltk-config.txt" | tr '\n' ' ' | sed 's/ $//')"
    echo "gmsh        $GMSH_VERSION, OCC $OCCT_VERSION, FLTK $FLTK_VERSION; $(awk -F': ' '$1 == "Build options" {print $2}' "$INFO/gmsh-build-info.txt")"
    echo "script      tools/engine/build_engine.sh sha256 $(sha256sum "$HERE/build_engine.sh" | cut -d' ' -f1)"
    for p in "${PARTS[@]}"; do
        echo "part        $p $(cat "$(part_dir "$p")/key") built in $(cat "$(part_dir "$p")/seconds") s"
    done
) > "$INFO/BUILD"
cut -f1,2,5 "$INFO/VERSIONS.tsv" | sed 's/\t/ | /g'

echo "=== verify"
grep -E 'Version|Build options|PETSc version|Packaged by|Build host' "$INFO/getdp-info.txt"
grep -E '^(Version|OCC version|Build options|Packaged by|Build host):' "$INFO/gmsh-build-info.txt" | sed 's/^/Gmsh /'
grep '^part ' "$INFO/BUILD"
echo
echo "export MPXLITE_GETDP_BINARY=$GETDP_PREFIX/bin/getdp"
