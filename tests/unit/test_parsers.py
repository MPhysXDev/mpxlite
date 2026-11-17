# Copyright (c) 2025 MPhysX
# SPDX-License-Identifier: MIT

"""Unit tests for mpxlite.parsers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mpxlite.parsers import (
    parse_complex_scalar,
    parse_complex_table,
    parse_complex_vector_table,
    parse_real_table,
)

# Verbatim GetDP 3.5.0 ``Format`` outputs of a complex 2-D problem (unit square,
# SetTime[5]): scalar ``u`` and vector ``g = Complex[1,2] * {d u}``, plus their
# integrals over the domain. GetDP 3.2.0 (real PETSc) writes the same layouts.
S_GLOBAL = " 5 0.4521328316621545 -0.06442585175399579\n"
V_GLOBAL = (
    " 5 -0.9999999999999599 2.584762904423768e-05 0 -1.999999999999919 -4.565854463098867e-05 0\n"
)
S_POINT = "15 13  0.3 0.4 0  0 0 0   0.6333028683988252   -0.09283590091929259  \n"
V_POINT = (
    "15 13  0.3 0.4 0  0 0 0   -0.8918410780342 -0.01679234379180805 0   "
    "-2.281159020266281 0.03267605485189939 0  \n"
)
S_LINE = (
    "15 26  0.1 0.5 0  0 0 0   0.8718206130059689   -0.04042858258605912  \n"
    "15 11  0.5 0.5 0  0.4 0 0   0.4275248646201856   -0.09661887316836287  \n"
    "15 29  0.9 0.5 0  0.8 0 0   0.08038374623232551   -0.02494860380585468  \n"
)
V_LINE = (
    "15 26  0.1 0.5 0  0 0 0   -0.4732222182188954 0.01127087806024119 0   "
    "-2.967873565741581 -0.01794694156614823 0  \n"
    "15 11  0.5 0.5 0  0.4 0 0   -1.154184762610607 -0.0145277351470704 0   "
    "-1.737165742018113 0.03518975019338771 0  \n"
    "15 29  0.9 0.5 0  0.8 0 0   -1.30280953844035 -4.990423679740452e-05 0   "
    "-1.358188886587962 0.0001221779649163146 0  \n"
)
S_LINE_TIMETABLE = (
    "0 5  0.1 0.5 0  0.8718206130059689   -0.04042858258605912  \n"
    "0 5  0.5 0.5 0  0.4275248646201856   -0.09661887316836287  \n"
)
V_POINT_TIMETABLE = (
    "0 5  0.3 0.4 0  -0.8918410780342 -0.01679234379180805 0   "
    "-2.281159020266281 0.03267605485189939 0  \n"
)


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


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


def test_parse_real_table_empty_file_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, "empty.dat", "# only a comment\n")
    with pytest.raises(ValueError, match="No data rows"):
        parse_real_table(p)


def test_parse_complex_scalar_empty_file_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, "empty.dat", "")
    with pytest.raises(ValueError, match="No data rows"):
        parse_complex_scalar(p)


def test_parse_complex_vector_table_empty_file_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, "empty.dat", "\n# nothing\n")
    with pytest.raises(ValueError, match="No data rows"):
        parse_complex_vector_table(p)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (S_GLOBAL, [0.4521328316621545 - 0.06442585175399579j]),
        (S_POINT, [0.6333028683988252 - 0.09283590091929259j]),
        (
            S_LINE,
            [
                0.8718206130059689 - 0.04042858258605912j,
                0.4275248646201856 - 0.09661887316836287j,
                0.08038374623232551 - 0.02494860380585468j,
            ],
        ),
        (
            S_LINE_TIMETABLE,
            [0.8718206130059689 - 0.04042858258605912j, 0.4275248646201856 - 0.09661887316836287j],
        ),
    ],
    ids=["global", "point", "line", "line-timetable"],
)
def test_parse_complex_table_getdp_scalar_layouts(
    tmp_path: Path, text: str, expected: list[complex]
) -> None:
    arr = parse_complex_table(_write(tmp_path, "s.txt", text))
    np.testing.assert_array_equal(arr, np.array(expected))


@pytest.mark.parametrize("text", [V_POINT, V_LINE], ids=["point", "line"])
def test_parse_complex_table_rejects_getdp_point_vector_table(tmp_path: Path, text: str) -> None:
    p = _write(tmp_path, "v.txt", text)
    with pytest.raises(ValueError, match="parse_complex_vector_table"):
        parse_complex_table(p)
    with pytest.raises(ValueError, match="3-component"):
        parse_complex_scalar(p)


def test_parse_complex_table_rejects_getdp_point_tensor_table(tmp_path: Path) -> None:
    row = "15 1  0 0 0  0 0 0   " + " ".join(["1"] * 18) + "\n"
    with pytest.raises(ValueError, match="9-component"):
        parse_complex_table(_write(tmp_path, "t.txt", row))


def test_parse_complex_table_global_vector_needs_expected_columns(tmp_path: Path) -> None:
    """A 7-column global vector is not detectable; ``expected_columns`` guards it."""
    p = _write(tmp_path, "v_glob.txt", V_GLOBAL)
    # Undetectable: silently reads (Im_y, Im_z), hence the guard below.
    assert parse_complex_table(p)[0] == complex(-4.565854463098867e-05, 0.0)
    with pytest.raises(ValueError, match="Expected 3 columns"):
        parse_complex_table(p, expected_columns=3)
    with pytest.raises(ValueError, match="Expected 3 columns"):
        parse_complex_scalar(p, expected_columns=3)


def test_parse_complex_table_expected_columns_match(tmp_path: Path) -> None:
    p = _write(tmp_path, "s_pt.txt", S_POINT)
    assert parse_complex_scalar(p, expected_columns=10) == complex(
        0.6333028683988252, -0.09283590091929259
    )


def test_parse_complex_table_point_like_rows_with_other_width_are_scalar(tmp_path: Path) -> None:
    """Element-type-15 rows are only rejected for the 8 + 2*{3,9} widths."""
    p = _write(tmp_path, "x.txt", "15 1  0 0 0  0 0 0   1 2 3 4\n")
    assert parse_complex_scalar(p) == complex(3.0, 4.0)


def test_parse_complex_table_generic_14_columns_not_rejected(tmp_path: Path) -> None:
    """A 14-column table whose first column is not the point type stays scalar."""
    row = " ".join(str(float(i)) for i in range(14)) + "\n"
    assert parse_complex_scalar(_write(tmp_path, "g.txt", row)) == complex(12.0, 13.0)


V_GLOBAL_EXPECTED = np.array(
    [[-0.9999999999999599 - 1.999999999999919j, 2.584762904423768e-05 - 4.565854463098867e-05j, 0]]
)
V_POINT_EXPECTED = np.array(
    [[-0.8918410780342 - 2.281159020266281j, -0.01679234379180805 + 0.03267605485189939j, 0]]
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (V_GLOBAL, V_GLOBAL_EXPECTED),
        (V_POINT, V_POINT_EXPECTED),
        (V_POINT_TIMETABLE, V_POINT_EXPECTED),
        (
            V_LINE,
            np.array(
                [
                    [
                        -0.4732222182188954 - 2.967873565741581j,
                        0.01127087806024119 - 0.01794694156614823j,
                        0,
                    ],
                    [
                        -1.154184762610607 - 1.737165742018113j,
                        -0.0145277351470704 + 0.03518975019338771j,
                        0,
                    ],
                    [
                        -1.30280953844035 - 1.358188886587962j,
                        -4.990423679740452e-05 + 0.0001221779649163146j,
                        0,
                    ],
                ]
            ),
        ),
    ],
    ids=["global", "point", "point-timetable", "line"],
)
def test_parse_complex_vector_table_getdp_layouts(
    tmp_path: Path, text: str, expected: np.ndarray
) -> None:
    arr = parse_complex_vector_table(_write(tmp_path, "v.txt", text))
    assert arr.shape == expected.shape
    assert arr.dtype == np.complex128
    np.testing.assert_array_equal(arr, expected)


def test_parse_complex_vector_table_ncomp_one_matches_scalar(tmp_path: Path) -> None:
    p = _write(tmp_path, "s.txt", S_LINE)
    arr = parse_complex_vector_table(p, ncomp=1)
    assert arr.shape == (3, 1)
    np.testing.assert_array_equal(arr[:, 0], parse_complex_table(p))


def test_parse_complex_vector_table_ncomp_two(tmp_path: Path) -> None:
    p = _write(tmp_path, "v2.txt", "7 1 2 3 4\n8 5 6 7 8\n")
    arr = parse_complex_vector_table(p, ncomp=2)
    np.testing.assert_array_equal(arr, np.array([[1 + 3j, 2 + 4j], [5 + 7j, 6 + 8j]]))


def test_parse_complex_vector_table_too_few_columns(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 6 columns"):
        parse_complex_vector_table(_write(tmp_path, "s.txt", S_GLOBAL))


@pytest.mark.parametrize("ncomp", [0, -1, 1.5, True])
def test_parse_complex_vector_table_bad_ncomp(tmp_path: Path, ncomp: object) -> None:
    p = _write(tmp_path, "v.txt", V_GLOBAL)
    with pytest.raises(ValueError, match="ncomp"):
        parse_complex_vector_table(p, ncomp=ncomp)  # type: ignore[arg-type]


def test_parse_complex_vector_table_accepts_numpy_int(tmp_path: Path) -> None:
    arr = parse_complex_vector_table(_write(tmp_path, "v.txt", V_GLOBAL), ncomp=np.int64(3))
    np.testing.assert_array_equal(arr, V_GLOBAL_EXPECTED)


def test_parse_complex_vector_table_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_complex_vector_table(tmp_path / "absent.dat")
