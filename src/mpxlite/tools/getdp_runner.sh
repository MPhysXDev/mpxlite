#!/usr/bin/env bash
# tools/getdp_runner.sh
# Copyright (c) 2025 Thierry Valet
# SPDX-License-Identifier: MIT
#
# Generic launcher around the native getdp binary. Auto-detects the MPI
# launcher (mpirun / srun / mpiexec.hydra) from the surrounding scheduler
# environment so the same Python script runs unchanged on a laptop, on a
# single-node SLURM/PBS allocation, and on a multi-node allocation.
#
# Required environment:
#   MPXLITE_GETDP_BINARY     absolute path to the actual getdp executable.
#                            If unset, falls back to `command -v getdp`.
#
# Optional environment:
#   MPXLITE_PETSC_LIB_DIR    absolute path to a PETSc shared-library dir.
#                            Prepended to LD_LIBRARY_PATH if set.
#   MPXLITE_GETDP_NP         MPI rank count (default 1, no launcher).
#   MPXLITE_GETDP_LAUNCHER   explicit launcher prefix to use as-is, e.g.
#                            "mpirun --hostfile myhosts -np 16" or
#                            "srun -n 32 --cpu-bind=verbose". When set,
#                            short-circuits the auto-detection below.
#   OMP_NUM_THREADS          honoured by OpenBLAS / MUMPS-OpenMP if the
#                            underlying getdp build supports them.
#
# Auto-detection order (when MPXLITE_GETDP_LAUNCHER is not set and NP > 1):
#   1. SLURM_JOB_ID present + srun on PATH        -> srun -n NP
#   2. PBS_NODEFILE present + mpirun on PATH      -> mpirun --hostfile $PBS_NODEFILE -np NP
#   3. mpirun on PATH (generic local MPI)         -> mpirun -np NP
#
# When NP == 1 (or unset), getdp is exec-ed directly without any launcher,
# regardless of the surrounding scheduler -- a sequential warmup or
# diagnostic call inside an interactive SLURM session does not get wrapped
# in srun, which would be surprising.

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

# Sequential -- no launcher, no scheduler, no surprise.
if [ "${NP}" -le 1 ]; then
    exec "${GETDP_BIN}" "$@"
fi

# Explicit user override -- splits on whitespace, used as-is.
if [ -n "${MPXLITE_GETDP_LAUNCHER:-}" ]; then
    # shellcheck disable=SC2086
    exec ${MPXLITE_GETDP_LAUNCHER} "${GETDP_BIN}" "$@"
fi

# Auto-detect from scheduler environment.
if [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null 2>&1; then
    exec srun -n "${NP}" "${GETDP_BIN}" "$@"
fi

if [ -n "${PBS_NODEFILE:-}" ] && command -v mpirun >/dev/null 2>&1; then
    exec mpirun --hostfile "${PBS_NODEFILE}" -np "${NP}" "${GETDP_BIN}" "$@"
fi

# Generic local MPI.
if ! command -v mpirun >/dev/null 2>&1; then
    cat >&2 <<EOF
[getdp_runner] MPXLITE_GETDP_NP=${NP} > 1 but no MPI launcher detected.
Either install OpenMPI / MPICH (mpirun on PATH), submit inside a SLURM
or PBS allocation, or set MPXLITE_GETDP_LAUNCHER explicitly.
EOF
    exit 127
fi
exec mpirun -np "${NP}" "${GETDP_BIN}" "$@"
