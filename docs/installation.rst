.. _mpxlite_installation:

Installation
============

`mpxlite` is a small pure-Python package, but it drives an external
finite-element solver (``getdp``) which itself links against PETSc,
MUMPS and an MPI implementation. This page covers everything from a
laptop install to a multi-node cluster build.

System requirements
-------------------

==============  =================  ================================
Tool            Minimum version    Used by
==============  =================  ================================
Python          3.10               `mpxlite` runtime
``getdp``       3.5                :class:`SolverLite` and any
                                   subclass; required at runtime
``gmsh``        4.12               only for users who generate their
                                   own meshes (optional, see below)
==============  =================  ================================

Operating system: any Linux distribution with a recent glibc.
macOS works for sequential runs but is not part of the contractual
test matrix. Windows is not supported.

Installing the Python package
-----------------------------

The package follows the modern PEP 621 layout with the ``hatchling``
build backend. Editable install for development::

    git clone <repository-url>
    cd mpxlite
    pip install -e .

For a non-editable install of a tagged release, the same command
without ``-e`` works.

Optional dependency groups (extras)
+++++++++++++++++++++++++++++++++++

Each extra is opt-in and installable via ``pip install -e .[<extra>]``.

==========  ==================================================  =========================================
Extra       Adds                                                When you need it
==========  ==================================================  =========================================
``test``    ``pytest``, ``pytest-cov``, ``pytest-mock``         running ``pytest`` or ``nox -s tests``
``docs``    ``sphinx``, ``furo``, ``sphinx-autoapi``,           building this documentation
            ``myst-parser``
``dev``     ``ruff``, ``nox``, ``build`` and the above          full developer environment
==========  ==================================================  =========================================

Common patterns::

    pip install -e .             # core only
    pip install -e .[test]       # core + pytest
    pip install -e .[dev]        # everything

Installing GetDP
----------------

`mpxlite` invokes a system-wide ``getdp`` binary; it does not download
or build it. Two paths are supported.

Distribution package (sequential only)
++++++++++++++++++++++++++++++++++++++

On Debian / Ubuntu::

    sudo apt install getdp

On macOS (Homebrew)::

    brew install getdp

This is sufficient for laptop use and for the unit-test suite. The
shipped binary is **not** built against PETSc / MUMPS / MPI, so it
cannot run distributed cases. It is also typically a couple of minor
versions behind upstream.

Source build (recommended for non-trivial cases)
++++++++++++++++++++++++++++++++++++++++++++++++

For any non-trivial linear system or for distributed solves, build
``getdp`` from source against a PETSc compiled with MPI and MUMPS.

1. **Build PETSc with MPI + MUMPS**::

      git clone -b release https://gitlab.com/petsc/petsc.git $HOME/petsc
      cd $HOME/petsc
      ./configure --with-mpi --download-mumps --download-scalapack \
                  --download-metis --download-parmetis --with-openmp
      make

   Note the printed ``PETSC_DIR`` and ``PETSC_ARCH``; the libraries
   live under ``$PETSC_DIR/$PETSC_ARCH/lib``.

2. **Build GetDP against that PETSc**::

      git clone https://gitlab.onelab.info/getdp/getdp.git
      cd getdp && mkdir build && cd build
      cmake .. -DENABLE_PETSC=ON \
               -DPETSC_DIR=$HOME/petsc \
               -DPETSC_ARCH=arch-linux-c-debug
      make -j

   The resulting binary is ``build/getdp``.

3. **Make it discoverable**::

      export MPXLITE_GETDP_BINARY=$HOME/getdp/build/getdp
      export MPXLITE_PETSC_LIB_DIR=$HOME/petsc/arch-linux-c-debug/lib

Environment variables
---------------------

`mpxlite` is configured through a small set of ``MPXLITE_*`` variables.
None is mandatory if a working ``getdp`` is on ``PATH`` and runs
sequentially.

==========================  ============================================================================================================================
Variable                    Purpose
==========================  ============================================================================================================================
``MPXLITE_GETDP``           Absolute path to the executable that drives ``getdp``. May be the real binary, a custom launcher, or the bundled
                            ``getdp_runner.sh`` wrapper. When unset, the bundled wrapper is used.
``MPXLITE_GETDP_BINARY``    (Read by ``getdp_runner.sh``.) Absolute path to the actual ``getdp`` executable that the wrapper should ``exec``. Falls back
                            to ``command -v getdp`` if unset.
``MPXLITE_PETSC_LIB_DIR``   (Read by ``getdp_runner.sh``.) Optional. Prepended to ``LD_LIBRARY_PATH``. Useful when ``getdp`` is dynamically linked
                            against a PETSc installed outside the system loader path.
``MPXLITE_GETDP_NP``        (Read by ``getdp_runner.sh``.) Number of MPI ranks per :meth:`SolverLite.run` call (default ``1``, no MPI launcher). When
                            set to ``> 1``, the wrapper switches to MPI mode (see below).
``MPXLITE_GETDP_LAUNCHER``  (Read by ``getdp_runner.sh``.) Optional explicit launcher prefix used as-is, e.g. ``"srun -n 32 --cpu-bind=verbose"``. Short
                            -circuits the auto-detection.
``OMP_NUM_THREADS``         Honoured by OpenBLAS / OpenMP-MUMPS if the underlying ``getdp`` build supports them. Set to ``1`` when
                            ``MPXLITE_GETDP_NP > 1`` to avoid oversubscription unless you deliberately run a hybrid MPI+OpenMP layout.
==========================  ============================================================================================================================

MPI mode and cluster auto-detection
-----------------------------------

The bundled wrapper script ``tools/getdp_runner.sh`` is the single
binding point between `mpxlite` and the surrounding scheduler. When
``MPXLITE_GETDP_NP > 1`` it auto-selects the MPI launcher in this
order:

1. **SLURM** --- if ``SLURM_JOB_ID`` is set and ``srun`` is on
   ``PATH``::

      srun -n $MPXLITE_GETDP_NP getdp ...

2. **PBS / Torque** --- if ``PBS_NODEFILE`` is set and ``mpirun`` is
   available::

      mpirun --hostfile $PBS_NODEFILE -np $MPXLITE_GETDP_NP getdp ...

3. **Generic local MPI** (``mpirun`` on ``PATH``)::

      mpirun -np $MPXLITE_GETDP_NP getdp ...

When ``MPXLITE_GETDP_NP`` is unset or equal to ``1``, ``getdp`` is
``exec``\ ed directly without any launcher --- a sequential warmup or
diagnostic call inside an interactive SLURM session does *not* get
wrapped in ``srun``, which would be surprising.

The user can always force a specific launcher via the
``MPXLITE_GETDP_LAUNCHER`` override; its value is split on whitespace
and used as-is.

Typical project env block::

    export MPXLITE_GETDP=/path/to/mpxlite/tools/getdp_runner.sh
    export MPXLITE_GETDP_BINARY=$HOME/getdp/build/getdp
    export MPXLITE_PETSC_LIB_DIR=$HOME/petsc/arch-linux-c-debug/lib
    export MPXLITE_GETDP_NP=8
    export OMP_NUM_THREADS=1

After this, every :meth:`SolverLite.run` call triggers a single::

    mpirun -np 8 getdp <manifest>.pro -msh ... -pre <res> -cal -pos <op>

that distributes the assembled system across ranks via PETSc + MUMPS,
with the post-operation sharing the same in-memory state as the
solve.

Single-process fallback
+++++++++++++++++++++++

Setting ``MPXLITE_GETDP_NP=1`` (or unsetting it) makes the wrapper
invoke ``getdp`` directly without ``mpirun``. This is the default and
what the unit-test suite uses.

Verifying the installation
--------------------------

Sanity-check the Python install::

    python -c "import mpxlite; print(mpxlite.__version__)"
    python -c "from mpxlite import find_getdp; print(find_getdp())"

The second command resolves the actual ``getdp`` binary; it raises
:exc:`FileNotFoundError` with an actionable message if no usable
binary can be located.

Run the test suite::

    nox -s lint        # ruff
    nox -s tests       # pytest with coverage
    nox -s docs        # build this documentation

The unit tests are hermetic (no real ``getdp`` invocation). Tests that
require a working ``getdp`` are auto-skipped when the binary is
missing.

Troubleshooting
---------------

``mpxlite`` raises :exc:`FileNotFoundError`
+++++++++++++++++++++++++++++++++++++++++++

*Message:* ``getdp binary not found ...`` or ``MPXLITE_GETDP_BINARY=...
is not an executable file``.

The wrapper could not locate a usable ``getdp``. Either install
``getdp`` and put it on ``PATH``, or set ``MPXLITE_GETDP_BINARY`` to
the absolute path of the binary.

``mpxlite`` raises :exc:`mpxlite.GetDPError`
++++++++++++++++++++++++++++++++++++++++++++

The ``getdp`` subprocess exited with a non-zero return code. The
exception message includes the last 500 characters of stderr, the
full command line, and the return code. The full stdout / stderr are
available on the exception object as ``exc.stdout`` / ``exc.stderr``.

Common causes:

* a malformed ``.pro`` file --- check the GetDP error before the
  Python traceback;
* a mesh saved in MSH 4.x (use ``gmsh -3 -format msh22``);
* a PETSc shared library not on ``LD_LIBRARY_PATH`` --- set
  ``MPXLITE_PETSC_LIB_DIR``.

``MPI`` errors at startup
+++++++++++++++++++++++++

*Message:* ``MPXLITE_GETDP_NP=N > 1 but no MPI launcher detected``.

The wrapper found neither ``SLURM_JOB_ID`` nor ``PBS_NODEFILE`` and
no ``mpirun`` on ``PATH``. Either install Open MPI / MPICH, run inside
a SLURM / PBS allocation, or set ``MPXLITE_GETDP_LAUNCHER`` explicitly.
