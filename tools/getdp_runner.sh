#!/usr/bin/env bash
# tools/getdp_runner.sh
# Copyright (c) 2025 Thierry Valet
# SPDX-License-Identifier: MIT
#
# Generic wrapper around the native getdp binary. Provides:
#   - optional LD_LIBRARY_PATH augmentation (for a getdp linked against a
#     locally-built PETSc whose .so files are not on the system loader path);
#   - optional MPI launch (mpirun -np N) for distributed solves with PETSc
#     + MUMPS;
#   - a single, well-documented binding point for the mpxlite SolverLite
#     and any plugin built on top of it
#     (MPXLITE_GETDP=/path/to/mpxlite/tools/getdp_runner.sh).
#
# Required environment:
#   MPXLITE_GETDP_BINARY     absolute path to the actual getdp executable.
#                            If unset, falls back to `command -v getdp`.
#
# Optional environment:
#   MPXLITE_PETSC_LIB_DIR    absolute path to the PETSc shared-library dir.
#                            Prepended to LD_LIBRARY_PATH if set.
#   MPXLITE_GETDP_NP         MPI rank count (default 1, no mpirun).
#                            When > 1, the wrapper invokes:
#                              mpirun -np "$MPXLITE_GETDP_NP" "$GETDP" "$@"
#                            and requires `mpirun` on PATH.
#   OMP_NUM_THREADS          honoured by OpenBLAS / MUMPS-OpenMP if the
#                            underlying getdp build supports them. Not
#                            modified by the wrapper.
#
# Typical bashrc / project env block::
#
#   export MPXLITE_GETDP=/path/to/mpxlite/tools/getdp_runner.sh
#   export MPXLITE_GETDP_BINARY=/opt/getdp/bin/getdp
#   export MPXLITE_PETSC_LIB_DIR=/opt/petsc/lib
#   export MPXLITE_GETDP_NP=8        # 8-rank MPI launch per solver.run()
#   export OMP_NUM_THREADS=1         # avoid oversubscription if NP*OMP > nproc

set -e

GETDP_BIN=${MPXLITE_GETDP_BINARY:-$(command -v getdp || true)}
if [ -z "${GETDP_BIN}" ] || [ ! -x "${GETDP_BIN}" ]; then
    cat >&2 <<EOF
[getdp_runner] cannot locate getdp executable.
Set MPXLITE_GETDP_BINARY=/absolute/path/to/getdp, or make sure getdp is on PATH.
EOF
    exit 127
fi

if [ -n "${MPXLITE_PETSC_LIB_DIR:-}" ]; then
    export LD_LIBRARY_PATH="${MPXLITE_PETSC_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

NP=${MPXLITE_GETDP_NP:-1}
if [ "${NP}" -gt 1 ]; then
    if ! command -v mpirun >/dev/null 2>&1; then
        cat >&2 <<EOF
[getdp_runner] MPXLITE_GETDP_NP=${NP} > 1 but mpirun is not on PATH.
Either install OpenMPI / MPICH, or unset MPXLITE_GETDP_NP for a single-process run.
EOF
        exit 127
    fi
    exec mpirun -np "${NP}" "${GETDP_BIN}" "$@"
else
    exec "${GETDP_BIN}" "$@"
fi
