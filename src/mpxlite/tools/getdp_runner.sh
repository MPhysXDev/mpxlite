#!/usr/bin/env bash
# tools/getdp_runner.sh
# Copyright (c) 2025 MPhysX
# SPDX-License-Identifier: MIT
#
# Generic launcher around the native getdp binary. Auto-detects the MPI
# launcher (srun / mpirun) from the surrounding scheduler
# environment so the same Python script runs unchanged on a laptop, on a
# single-node SLURM/PBS allocation, and on a multi-node allocation.
#
# Required environment:
#   MPXLITE_GETDP_BINARY     absolute path of the engine, the getdp
#                            executable to run. mpxlite sets it for each
#                            run; PATH is never searched.
#
# Optional environment:
#   MPXLITE_PETSC_LIB_DIR    absolute path to a PETSc shared-library dir.
#                            Prepended to LD_LIBRARY_PATH if set (no empty
#                            entry is added when LD_LIBRARY_PATH is unset).
#   MPXLITE_GETDP_NP         MPI rank count, integer >= 1 (default 1, no
#                            launcher). Anything else -> exit 2.
#   MPXLITE_GETDP_THREADS    OpenMP threads per rank, integer >= 1. Exported
#                            as OMP_NUM_THREADS (overrides any inherited
#                            value). Anything else -> exit 2.
#   MPXLITE_GETDP_CPUS       logical CPU list for `taskset -c`, e.g. "0-3" or
#                            "4,5,6,7". Confines the whole run (launcher and
#                            every rank) to those CPUs so concurrent runs do
#                            not pile up on the same cores:
#                              NP == 1    taskset -c CPUS getdp ...
#                                         (also under SLURM/PBS)
#                              local MPI  taskset -c CPUS mpirun --bind-to none
#                                           -np NP getdp ...
#                                         (ranks inherit the mask; --bind-to
#                                         none stops Open MPI from re-binding
#                                         every job onto cores 0..NP-1)
#                              LAUNCHER   taskset -c CPUS $LAUNCHER getdp ...
#                                         (one warning on stderr if LAUNCHER
#                                         has no --bind-to: Open MPI would
#                                         re-bind the ranks out of CPUS; the
#                                         launcher is never rewritten)
#                              SLURM/PBS  NP > 1: ignored with one warning on
#                                         stderr (the scheduler owns placement).
#                            taskset missing while needed -> exit 127.
#   MPXLITE_GETDP_LAUNCHER   explicit launcher prefix to use as-is, e.g.
#                            "mpirun --bind-to none -np 16" or
#                            "srun -n 32 --cpu-bind=verbose". When set, it is
#                            used whatever NP (even NP == 1) and
#                            short-circuits the auto-detection below. It must
#                            start the rank count itself (no -np appended).
#                            Split on whitespace, never glob-expanded (no
#                            quoting either); blank (whitespace only) -> exit 2.
#   OMP_NUM_THREADS          honoured by OpenBLAS / MUMPS-OpenMP if the
#                            underlying getdp build supports them. Inherited
#                            unless MPXLITE_GETDP_THREADS is set.
#
# Auto-detection order (when MPXLITE_GETDP_LAUNCHER is not set and NP > 1):
#   1. SLURM_JOB_ID present + srun on PATH        -> srun -n NP
#   2. PBS_NODEFILE present + mpirun on PATH      -> mpirun --hostfile $PBS_NODEFILE -np NP
#   3. mpirun on PATH (generic local MPI)         -> mpirun -np NP
#                                                    (with CPUS: see above)
#
# Parent-death signal of local MPI ranks: with a local mpirun (auto-detected,
# or an explicit launcher whose first word is mpirun / mpiexec / orterun),
# every rank is started through `setpriv --pdeathsig KILL --` (util-linux
# >= 2.33, probed once), so that the ranks die with mpirun if mpirun itself is
# SIGKILLed (e.g. because the Python process died: mpxlite gives mpirun a
# parent-death signal). Open MPI does not kill orphaned ranks by itself.
# Without a usable setpriv the ranks are started as before.
#
# When NP == 1 (or unset) and no MPXLITE_GETDP_LAUNCHER is set, getdp is
# exec-ed directly without any launcher,
# regardless of the surrounding scheduler -- a sequential warmup or
# diagnostic call inside an interactive SLURM session does not get wrapped
# in srun, which would be surprising.

set -e

# The engine is named by the caller: mpxlite sets MPXLITE_GETDP_BINARY in the
# environment of every run (mpxlite.find_engine). PATH is never searched.
GETDP_BIN=${MPXLITE_GETDP_BINARY:-}
if [ -z "${GETDP_BIN}" ] || [ ! -x "${GETDP_BIN}" ]; then
    cat >&2 <<EOF
[getdp_runner] MPXLITE_GETDP_BINARY='${GETDP_BIN}' does not name an executable.
It names the engine (the GetDP program) to run; mpxlite sets it for each run.
EOF
    exit 127
fi

if [ -n "${MPXLITE_PETSC_LIB_DIR:-}" ]; then
    export LD_LIBRARY_PATH="${MPXLITE_PETSC_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Positive decimal integer check (no sign, no leading zeros, bounded length).
is_pos_int() {
    [[ "$1" =~ ^[1-9][0-9]{0,8}$ ]]
}

NP=${MPXLITE_GETDP_NP:-1}
if ! is_pos_int "${NP}"; then
    echo "[getdp_runner] MPXLITE_GETDP_NP='${NP}' is not an integer >= 1." >&2
    exit 2
fi

if [ -n "${MPXLITE_GETDP_THREADS:-}" ]; then
    if ! is_pos_int "${MPXLITE_GETDP_THREADS}"; then
        echo "[getdp_runner] MPXLITE_GETDP_THREADS='${MPXLITE_GETDP_THREADS}'" \
            "is not an integer >= 1." >&2
        exit 2
    fi
    export OMP_NUM_THREADS="${MPXLITE_GETDP_THREADS}"
fi

# CPU pinning prefix: empty array when MPXLITE_GETDP_CPUS is unset or empty.
CPUS=${MPXLITE_GETDP_CPUS:-}
PIN=()
if [ -n "${CPUS}" ]; then
    PIN=(taskset -c "${CPUS}")
fi

require_taskset() {
    if [ -n "${CPUS}" ] && ! command -v taskset >/dev/null 2>&1; then
        cat >&2 <<EOF
[getdp_runner] MPXLITE_GETDP_CPUS=${CPUS} is set but taskset is not on PATH.
Install util-linux (taskset) or unset MPXLITE_GETDP_CPUS.
EOF
        exit 127
    fi
}

warn_cpus_ignored() {
    if [ -n "${CPUS}" ]; then
        echo "[getdp_runner] warning: MPXLITE_GETDP_CPUS=${CPUS} ignored under $1;" \
            "the scheduler owns CPU placement." >&2
    fi
}

# Rank prefix of local mpirun launches (see "Parent-death signal" above).
RANK=()
rank_pdeathsig() {
    if command -v setpriv >/dev/null 2>&1 \
        && setpriv --pdeathsig KILL true >/dev/null 2>&1; then
        RANK=(setpriv --pdeathsig KILL --)
    fi
}

# Explicit user override -- whatever NP; split on whitespace (no pathname
# expansion: the cwd is the run directory), used as-is.
if [ -n "${MPXLITE_GETDP_LAUNCHER:-}" ]; then
    read -r -a LAUNCHER_WORDS <<< "${MPXLITE_GETDP_LAUNCHER}"
    if [ "${#LAUNCHER_WORDS[@]}" -eq 0 ]; then
        echo "[getdp_runner] MPXLITE_GETDP_LAUNCHER='${MPXLITE_GETDP_LAUNCHER}'" \
            "is blank; unset it or give a launcher command." >&2
        exit 2
    fi
    require_taskset
    if [ -n "${CPUS}" ] && [[ "${MPXLITE_GETDP_LAUNCHER}" != *--bind-to* ]]; then
        echo "[getdp_runner] warning: MPXLITE_GETDP_CPUS=${CPUS} with a launcher" \
            "without --bind-to; Open MPI may re-bind the ranks outside CPUS:" \
            "add '--bind-to none' to MPXLITE_GETDP_LAUNCHER." >&2
    fi
    case "${LAUNCHER_WORDS[0]##*/}" in
        mpirun|mpiexec|orterun) rank_pdeathsig ;;
    esac
    exec "${PIN[@]}" "${LAUNCHER_WORDS[@]}" "${RANK[@]}" "${GETDP_BIN}" "$@"
fi

# Sequential -- no launcher, no scheduler, no surprise.
if [ "${NP}" -le 1 ]; then
    require_taskset
    exec "${PIN[@]}" "${GETDP_BIN}" "$@"
fi

# Auto-detect from scheduler environment.
if [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null 2>&1; then
    warn_cpus_ignored SLURM
    exec srun -n "${NP}" "${GETDP_BIN}" "$@"
fi

if [ -n "${PBS_NODEFILE:-}" ] && command -v mpirun >/dev/null 2>&1; then
    warn_cpus_ignored PBS
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
rank_pdeathsig
if [ -n "${CPUS}" ]; then
    require_taskset
    exec "${PIN[@]}" mpirun --bind-to none -np "${NP}" "${RANK[@]}" "${GETDP_BIN}" "$@"
fi
exec mpirun -np "${NP}" "${RANK[@]}" "${GETDP_BIN}" "$@"
