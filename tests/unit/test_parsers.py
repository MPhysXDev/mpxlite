# Copyright (c) 2025 Thierry Valet
# SPDX-License-Identifier: MIT

"""Unit tests for mpxlite.parsers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mpxlite.parsers import (
    parse_complex_scalar,
    parse_complex_table,
    parse_real_table,
)


def test_parse_complex_table_freq_format(tmp_path: Path) -> None:
    p = tmp_path / "yin.dat"
    p.write_text("5.0e8 1.0 2.0\n1.5e9 3.0 -4.0\n", encoding="utf-8")
    arr = parse_complex_table(p)
    assert arr.shape == (2,)
    assert arr[0] == complex(1.0, 2.0)
    assert arr[1] == complex(3.0, -4.0)


def test_parse_complex_table_two_columns(tmp_path: Path) -> None:
    p = tmp_path / "two.dat"
    p.write_text("1.0 2.0\n3.0 -4.0\n", encoding="utf-8")
    arr = parse_complex_table(p)
    np.testing.assert_array_equal(arr, np.array([1 + 2j, 3 - 4j]))


def test_parse_complex_table_too_few_columns(tmp_path: Path) -> None:
    p = tmp_path / "single.dat"
    p.write_text("1.0\n2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least 2 columns"):
        parse_complex_table(p)


def test_parse_complex_table_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_complex_table(tmp_path / "absent.dat")


def test_parse_complex_table_empty_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.dat"
    p.write_text("# only a comment\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No data rows"):
        parse_complex_table(p)


def test_parse_complex_table_strips_comments(tmp_path: Path) -> None:
    p = tmp_path / "with_comments.dat"
    p.write_text("# header\n1.0 2.0\n# tail\n3.0 -4.0\n", encoding="utf-8")
    arr = parse_complex_table(p)
    np.testing.assert_array_equal(arr, np.array([1 + 2j, 3 - 4j]))


def test_parse_complex_table_real_dipole_data(tmp_path: Path) -> None:
    """Reproduce the format observed in Harmony's Yin.dat."""
    p = tmp_path / "yin.dat"
    p.write_text(
        "99999999.99999999  3.220845560910668e-05 0.001359067813552882\n"
        "150000000  0.000132881190463536 0.00239083803968991\n",
        encoding="utf-8",
    )
    arr = parse_complex_table(p)
    assert arr.shape == (2,)
    np.testing.assert_allclose(arr[0].real, 3.220845560910668e-05)
    np.testing.assert_allclose(arr[0].imag, 0.001359067813552882)


def test_parse_complex_scalar(tmp_path: Path) -> None:
    p = tmp_path / "scalar.dat"
    p.write_text("5.0e8 1.0 2.0\n1.5e9 3.0 -4.0\n", encoding="utf-8")
    val = parse_complex_scalar(p)
    assert val == complex(3.0, -4.0)


def test_parse_complex_scalar_single_row(tmp_path: Path) -> None:
    p = tmp_path / "single.dat"
    p.write_text("5.0e8 1.0 2.0\n", encoding="utf-8")
    val = parse_complex_scalar(p)
    assert val == complex(1.0, 2.0)


def test_parse_real_table(tmp_path: Path) -> None:
    p = tmp_path / "real.dat"
    p.write_text("1 2.0 3.0\n4 5.0 6.0\n", encoding="utf-8")
    arr = parse_real_table(p)
    assert arr.shape == (2, 3)
    np.testing.assert_array_equal(arr, np.array([[1, 2, 3], [4, 5, 6]]))


def test_parse_real_table_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_real_table(tmp_path / "absent.dat")
