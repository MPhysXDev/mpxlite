# Copyright (c) 2025 Thierry Valet
# SPDX-License-Identifier: MIT

"""Parsers for GetDP textual output files.

GetDP can write ``Print[…]`` statements with several formats; the most
useful for typical solver output are ``FrequencyTable`` (one row per
frequency, columns ``<freq> <Re> <Im>``) and ``Table`` (generic tabular
output). Both expose the complex value as the *last two columns* of every
data row — a convention that the helpers in this module rely on.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np


def parse_complex_table(path: Path | str) -> np.ndarray:
    """Parse a GetDP table where the last two columns are (Re, Im).

    Suitable for ``Format FrequencyTable`` outputs (one row per frequency
    point, columns ``<freq> <Re> <Im>``) and for indexed ``Format Table``
    outputs (e.g. ``rs.txt``, ``eff_r1.txt``) that share the same
    last-two-columns convention.

    Args:
        path: Path to the ``.dat`` (or ``.txt``) output file.

    Returns:
        A 1-D numpy array of complex values, one per data row.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file is empty or has fewer than two columns.
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
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}, got {data.shape[1]}")
    return data[:, -2] + 1j * data[:, -1]


def parse_complex_scalar(path: Path | str) -> complex:
    """Parse a single complex value from a GetDP output (last data row).

    Convenient for FrequencyTable files representing one-shot results
    (e.g. an admittance evaluated at a single frequency).

    Args:
        path: Path to the ``.dat`` output file.

    Returns:
        The complex value of the last data row (as a Python ``complex``).

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file is empty or has fewer than two columns.
    """
    return complex(parse_complex_table(path)[-1])


def parse_real_table(path: Path | str) -> np.ndarray:
    """Parse a GetDP table of real values into a 2-D numpy array.

    Args:
        path: Path to the output file.

    Returns:
        A 2-D array (``n_rows × n_cols``) of float values.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Output file not found: {path}")
    return np.loadtxt(path, ndmin=2, comments="#")
