# Copyright (c) 2025 MPhysX
# SPDX-License-Identifier: MIT

"""Parsers for GetDP textual output files.

Column layouts of complex-valued ``Print[...]`` output, verified empirically with
the supported build (GetDP 3.5.0-mpx3, complex PETSc 3.26). ``t`` is the current
time (``SetTime``), ``f`` the current frequency (``0`` when none is set), ``etype``
the post-element type (``15`` = point), ``id`` its index, ``s`` the curvilinear
abscissa along an ``OnLine`` path (``0`` for ``OnPoint``), ``step`` the time-step
index::

    Print target / Format                leading columns       scalar   vector (3 comp.)
    -----------------------------------  --------------------  -------  -----------------
    q[R], OnGlobal|OnRegion R  Table     t                     Re Im    Rx Ry Rz Ix Iy Iz
    q[R], OnGlobal  FrequencyTable       f                     Re Im    Rx Ry Rz Ix Iy Iz
    q[R], OnGlobal  TimeTable            t                     Re Im    Rx Ry Rz Ix Iy Iz
    q, OnPoint|OnLine  Table             etype id x y z s 0 0  Re Im    Rx Ry Rz Ix Iy Iz
    q, OnPoint|OnLine  TimeTable         step t x y z          Re Im    Rx Ry Rz Ix Iy Iz

In every case the value block is the *last* ``2 * ncomp`` columns, ordered as
all real parts first, then all imaginary parts (``Re_1 .. Re_n Im_1 .. Im_n``,
not interleaved). Vectors always carry three components, even for 2-D problems
(the ``z`` component is then ``0``). Scalars therefore occupy the last two
columns ``(Re, Im)`` — the convention :func:`parse_complex_table` relies on —
and vectors are read with :func:`parse_complex_vector_table`.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

# GetDP post-element type written in column 0 of OnPoint / OnLine ``Format Table`` rows.
_GETDP_POINT_ELEMENT_TYPE = 15
# Number of leading metadata columns in OnPoint / OnLine ``Format Table`` rows
# (etype, id, x, y, z, s, 0, 0).
_GETDP_POINT_TABLE_HEADER = 8
# Complex component counts that are not scalars: vector (3) and tensor (9).
_GETDP_NONSCALAR_NCOMP = (3, 9)


def _load_table(path: Path | str) -> np.ndarray:
    """Load a whitespace-separated numeric table as a non-empty 2-D float array.

    Args:
        path: Path to the output file.

    Returns:
        A 2-D array (``n_rows × n_cols``) with at least one row.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file has no data rows.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Output file not found: {path}")
    # numpy.loadtxt emits a UserWarning when the file has no data rows;
    # we surface that as a ValueError instead.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="loadtxt: input contained no data", category=UserWarning
        )
        data = np.loadtxt(path, ndmin=2, comments="#")
    if data.size == 0:
        raise ValueError(f"No data rows found in {path}")
    return data


def _is_getdp_nonscalar_point_table(data: np.ndarray) -> bool:
    """Return True if ``data`` matches GetDP's OnPoint/OnLine vector/tensor ``Table`` layout.

    The signature is specific enough to be detected reliably: every row starts with
    the point post-element type ``15``, an integer index, and columns 6–7 are zero,
    and the column count is ``8 + 2 * ncomp`` with ``ncomp`` in ``{3, 9}``.

    Args:
        data: Table loaded by :func:`_load_table`.

    Returns:
        Whether the table is a non-scalar OnPoint / OnLine ``Format Table`` dump.
    """
    n_cols = data.shape[1]
    if n_cols not in {_GETDP_POINT_TABLE_HEADER + 2 * n for n in _GETDP_NONSCALAR_NCOMP}:
        return False
    return bool(
        np.all(data[:, 0] == _GETDP_POINT_ELEMENT_TYPE)
        and np.all(data[:, 1] == np.round(data[:, 1]))
        and np.all(data[:, 6:8] == 0.0)
    )


def parse_complex_table(path: Path | str, *, expected_columns: int | None = None) -> np.ndarray:
    """Parse a GetDP table of a complex *scalar*, where the last two columns are (Re, Im).

    Suitable for ``Format FrequencyTable`` outputs (one row per frequency
    point, columns ``<freq> <Re> <Im>``) and for ``Format Table`` / ``TimeTable``
    outputs of a complex scalar on any Print target (see the module docstring
    for the verified layouts).

    Vector tables: a vector row ends with ``Rx Ry Rz Ix Iy Iz``, so reading its
    last two columns would silently return ``Iy + 1j*Iz``. The OnPoint / OnLine
    ``Format Table`` vector layout is detected and rejected. The ``OnGlobal`` /
    ``OnRegion`` and ``TimeTable`` vector layouts *cannot* be detected reliably:
    an ``OnGlobal`` vector row (``t Rx Ry Rz Ix Iy Iz``, 7 columns) is
    indistinguishable from an ``OnPoint`` ``TimeTable`` scalar row
    (``step t x y z Re Im``, also 7 columns), and the file carries no header.
    Callers that know their Print target should pin the layout with
    ``expected_columns`` (3 for a global scalar, 10 for an OnPoint/OnLine
    ``Table`` scalar, 7 for an OnPoint/OnLine ``TimeTable`` scalar) and use
    :func:`parse_complex_vector_table` for vector quantities.

    Args:
        path: Path to the ``.dat`` (or ``.txt``) output file.
        expected_columns: If given, the exact number of columns every row must
            have; a mismatch raises ``ValueError``.

    Returns:
        A 1-D numpy array of complex values, one per data row.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file is empty, has fewer than two columns, does not
            have ``expected_columns`` columns, or is a detectable GetDP vector /
            tensor table.
    """
    data = _load_table(path)
    n_cols = data.shape[1]
    if expected_columns is not None and n_cols != expected_columns:
        raise ValueError(f"Expected {expected_columns} columns in {path}, got {n_cols}")
    if n_cols < 2:
        raise ValueError(f"Expected at least 2 columns in {path}, got {n_cols}")
    if _is_getdp_nonscalar_point_table(data):
        ncomp = (n_cols - _GETDP_POINT_TABLE_HEADER) // 2
        raise ValueError(
            f"{path} looks like a GetDP OnPoint/OnLine table of a {ncomp}-component "
            "quantity; use parse_complex_vector_table(path, ncomp=...) instead"
        )
    return data[:, -2] + 1j * data[:, -1]


def parse_complex_scalar(path: Path | str, *, expected_columns: int | None = None) -> complex:
    """Parse a single complex value from a GetDP output (last data row).

    Convenient for FrequencyTable files representing one-shot results
    (e.g. an admittance evaluated at a single frequency). Same layout rules
    and vector-table caveats as :func:`parse_complex_table`.

    Args:
        path: Path to the ``.dat`` output file.
        expected_columns: If given, the exact number of columns every row must
            have; a mismatch raises ``ValueError``.

    Returns:
        The complex value of the last data row (as a Python ``complex``).

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file is empty, has fewer than two columns, does not
            have ``expected_columns`` columns, or is a detectable GetDP vector /
            tensor table.
    """
    return complex(parse_complex_table(path, expected_columns=expected_columns)[-1])


def parse_complex_vector_table(path: Path | str, ncomp: int = 3) -> np.ndarray:
    """Parse a GetDP table of a complex vector (or tensor) quantity.

    Reads the last ``2 * ncomp`` columns of every row as
    ``Re_1 .. Re_ncomp Im_1 .. Im_ncomp`` (real block, then imaginary block).
    This holds for every Print target and format verified (``OnGlobal``,
    ``OnRegion``, ``OnPoint``, ``OnLine``; ``Table``, ``FrequencyTable``,
    ``TimeTable``), so no target parameter is needed. GetDP always writes three
    components for a vector (``ncomp=3``, even in 2-D) and nine for a tensor
    (``ncomp=9``).

    Args:
        path: Path to the output file.
        ncomp: Number of complex components per row (``>= 1``).

    Returns:
        A complex array of shape ``(n_rows, ncomp)``.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if ``ncomp < 1``, the file is empty, or rows have fewer than
            ``2 * ncomp`` columns.
    """
    if isinstance(ncomp, bool) or not isinstance(ncomp, (int, np.integer)) or ncomp < 1:
        raise ValueError(f"ncomp must be an integer >= 1, got {ncomp!r}")
    ncomp = int(ncomp)
    data = _load_table(path)
    n_cols = data.shape[1]
    if n_cols < 2 * ncomp:
        raise ValueError(
            f"Expected at least {2 * ncomp} columns in {path} for ncomp={ncomp}, got {n_cols}"
        )
    values = data[:, n_cols - 2 * ncomp :]
    return values[:, :ncomp] + 1j * values[:, ncomp:]


def parse_real_table(path: Path | str) -> np.ndarray:
    """Parse a GetDP table of real values into a 2-D numpy array.

    Args:
        path: Path to the output file.

    Returns:
        A 2-D array (``n_rows × n_cols``) of float values.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file has no data rows.
    """
    return _load_table(path)
