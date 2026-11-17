# Copyright (c) 2025 MPhysX
# SPDX-License-Identifier: MIT

"""Unit tests for mpxlite.workflow.WorkflowData."""

from __future__ import annotations

import enum
import math
from pathlib import Path

import numpy as np
import pytest

from mpxlite.workflow import Scalar, WorkflowData


def test_empty_serializes_to_empty_function_block() -> None:
    wf = WorkflowData()
    assert wf.to_pro_string() == "Function {\n}\n"


def test_set_and_get_scalar() -> None:
    wf = WorkflowData()
    wf["FREQ"] = 12.0e9
    assert wf["FREQ"] == 12.0e9


def test_initial_mapping_seeds_data() -> None:
    wf = WorkflowData({"A": 1, "B": 2.0, "C": "hello"})
    assert wf["A"] == 1
    assert wf["B"] == 2.0
    assert wf["C"] == "hello"


def test_invalid_name_starting_with_digit_raises() -> None:
    wf = WorkflowData()
    with pytest.raises(ValueError, match="Invalid GetDP scalar name"):
        wf["1invalid"] = 1


def test_invalid_name_with_dash_raises() -> None:
    wf = WorkflowData()
    with pytest.raises(ValueError, match="Invalid GetDP scalar name"):
        wf["has-dash"] = 1


def test_invalid_name_empty_raises() -> None:
    wf = WorkflowData()
    with pytest.raises(ValueError, match="Invalid GetDP scalar name"):
        wf[""] = 1


def test_non_string_name_raises() -> None:
    wf = WorkflowData()
    with pytest.raises(TypeError, match="must be str"):
        wf[42] = 1  # type: ignore[index]


def test_invalid_value_type_raises() -> None:
    wf = WorkflowData()
    with pytest.raises(TypeError, match="must be bool/int/float/complex/str"):
        wf["X"] = [1, 2, 3]  # type: ignore[assignment]


def test_serialization_keys_sorted() -> None:
    wf = WorkflowData()
    wf["B"] = 2.0
    wf["A"] = 1.0
    text = wf.to_pro_string()
    assert text.index("A =") < text.index("B =")


def test_serialization_int() -> None:
    wf = WorkflowData()
    wf["I"] = 5
    assert "I = 5;" in wf.to_pro_string()


def test_serialization_float() -> None:
    wf = WorkflowData()
    wf["F"] = 1.5e-3
    assert "F = 0.0015;" in wf.to_pro_string()


def test_serialization_string_quoted() -> None:
    wf = WorkflowData()
    wf["S"] = "hello"
    assert 'S = "hello";' in wf.to_pro_string()


def test_serialization_bool_to_int() -> None:
    wf = WorkflowData()
    wf["T"] = True
    wf["F"] = False
    text = wf.to_pro_string()
    assert "T = 1;" in text
    assert "F = 0;" in text


@pytest.mark.parametrize("bad", ['a"b', "a\\b", "a\nb", "a\rb"])
def test_string_with_forbidden_character_raises(bad: str) -> None:
    wf = WorkflowData()
    with pytest.raises(ValueError, match=r"'LABEL'.*forbidden"):
        wf["LABEL"] = bad
    assert "LABEL" not in wf


def test_string_forbidden_check_applies_to_initial_mapping() -> None:
    with pytest.raises(ValueError, match="'S'"):
        WorkflowData({"S": 'x"y'})


def test_write_pro_round_trip(tmp_path: Path) -> None:
    wf = WorkflowData({"FREQ": 12.0e9})
    target = tmp_path / "workflow.pro"
    wf.write_pro(target)
    text = target.read_text(encoding="utf-8")
    assert text.startswith("Function {")
    assert "FREQ = 12000000000.0;" in text
    assert text.rstrip().endswith("}")


def test_mutablemapping_interface() -> None:
    wf = WorkflowData()
    wf["A"] = 1
    wf["B"] = 2
    assert len(wf) == 2
    assert set(wf.keys()) == {"A", "B"}
    del wf["A"]
    assert "A" not in wf
    assert len(wf) == 1


def test_update_method() -> None:
    wf = WorkflowData()
    wf.update({"A": 1, "B": 2})
    assert wf["A"] == 1
    assert wf["B"] == 2


def test_repr_round_trip() -> None:
    wf = WorkflowData({"A": 1})
    assert "WorkflowData" in repr(wf)
    assert "'A': 1" in repr(wf)


# -- numpy coercion -------------------------------------------------------


def test_numpy_float64_serializes_as_plain_literal() -> None:
    wf = WorkflowData()
    wf["F"] = np.float64(1.5)
    assert type(wf["F"]) is float
    text = wf.to_pro_string()
    assert "F = 1.5;" in text
    assert "np." not in text


def test_numpy_float32_coerced_to_float() -> None:
    wf = WorkflowData({"F": np.float32(0.25)})
    assert type(wf["F"]) is float
    assert "F = 0.25;" in wf.to_pro_string()


@pytest.mark.parametrize("value", [np.int8(7), np.int32(7), np.int64(7), np.uint16(7)])
def test_numpy_integer_coerced_to_int(value: np.integer) -> None:
    wf = WorkflowData({"I": value})
    assert type(wf["I"]) is int
    assert "I = 7;" in wf.to_pro_string()


def test_numpy_bool_coerced_to_bool() -> None:
    wf = WorkflowData({"T": np.bool_(True), "F": np.bool_(False)})
    assert type(wf["T"]) is bool
    assert type(wf["F"]) is bool
    text = wf.to_pro_string()
    assert "T = 1;" in text
    assert "F = 0;" in text


def test_bool_stays_bool_not_int() -> None:
    wf = WorkflowData({"T": True})
    assert type(wf["T"]) is bool
    text = wf.to_pro_string()
    assert "T = 1;" in text
    assert "True" not in text


@pytest.mark.parametrize("value", [np.complex64(1 + 2j), np.complex128(1 + 2j)])
def test_numpy_complex_coerced_to_complex(value: np.complexfloating) -> None:
    wf = WorkflowData({"Z": value})
    assert type(wf["Z"]) is complex
    assert "Z[] = Complex[1.0, 2.0];" in wf.to_pro_string()


def test_int_subclass_coerced_to_plain_int() -> None:
    class Mode(enum.IntEnum):
        TE = 3

    wf = WorkflowData({"M": Mode.TE})
    assert type(wf["M"]) is int
    assert "M = 3;" in wf.to_pro_string()


def test_numpy_array_rejected() -> None:
    wf = WorkflowData()
    with pytest.raises(TypeError, match="must be bool/int/float/complex/str"):
        wf["A"] = np.array([1.0, 2.0])  # type: ignore[assignment]


# -- non-finite rejection ---------------------------------------------------


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, np.float64("nan")])
def test_non_finite_float_raises(value: float) -> None:
    wf = WorkflowData()
    with pytest.raises(ValueError, match="'X' must be finite"):
        wf["X"] = value
    assert "X" not in wf


@pytest.mark.parametrize(
    "value",
    [
        complex(math.nan, 0.0),
        complex(0.0, math.inf),
        complex(-math.inf, 1.0),
        np.complex128(np.nan),
    ],
)
def test_non_finite_complex_raises(value: complex) -> None:
    wf = WorkflowData()
    with pytest.raises(ValueError, match="'Z' must have finite real and imaginary parts"):
        wf["Z"] = value


# -- complex serialization ------------------------------------------------


def test_serialization_complex() -> None:
    wf = WorkflowData({"EPS": complex(4.0, -0.1)})
    assert "  EPS[] = Complex[4.0, -0.1];" in wf.to_pro_string()


def test_serialization_complex_zero_imag_stays_complex() -> None:
    wf = WorkflowData({"Z": 3 + 0j})
    assert "Z[] = Complex[3.0, 0.0];" in wf.to_pro_string()


def test_mixed_scalars_single_function_block_sorted() -> None:
    wf = WorkflowData({"Z": 1 - 2j, "A": 1.5, "B": 2, "C": True, "D": "txt"})
    assert wf.to_pro_string() == (
        'Function {\n  A = 1.5;\n  B = 2;\n  C = 1;\n  D = "txt";\n  Z[] = Complex[1.0, -2.0];\n}\n'
    )


def test_real_values_unchanged_format() -> None:
    wf = WorkflowData({"F": 1e-300, "G": -2.5, "I": -4})
    text = wf.to_pro_string()
    assert "F = 1e-300;" in text
    assert "G = -2.5;" in text
    assert "I = -4;" in text


def test_write_pro_round_trip_all_types(tmp_path: Path) -> None:
    wf = WorkflowData(
        {"R": np.float64(0.5), "I": np.int64(3), "B": np.bool_(False), "S": "ok", "Z": 2j}
    )
    target = tmp_path / "workflow.pro"
    wf.write_pro(target)
    assert target.read_text(encoding="utf-8") == wf.to_pro_string()
    text = target.read_text(encoding="utf-8")
    assert "R = 0.5;" in text
    assert "I = 3;" in text
    assert "B = 0;" in text
    assert 'S = "ok";' in text
    assert "Z[] = Complex[0.0, 2.0];" in text


def test_scalar_type_hint_usable() -> None:
    values: dict[str, Scalar] = {"A": 1.0, "Z": 1j}
    assert len(WorkflowData(values)) == 2


# ----- reading a scalars file back ----------------------------------------


def test_read_pro_is_the_inverse_of_write_pro(tmp_path: Path) -> None:
    written = WorkflowData(
        {
            "N": 3,
            "NEG": -7,
            "F": 12.0e9,
            "SMALL": 1.5e-300,
            "Z": complex(4.0, -0.1),
            "ZR": complex(2.0, 0.0),
            "NAME": "a b.c",
            "EMPTY": "",
        }
    )
    path = tmp_path / "workflow.pro"
    written.write_pro(path)
    read = WorkflowData.read_pro(path)
    assert dict(read) == dict(written)
    assert [type(read[k]) for k in ("N", "F", "Z", "ZR", "NAME")] == [
        int,
        float,
        complex,
        complex,
        str,
    ]
    assert read.to_pro_string() == written.to_pro_string()


def test_a_bool_is_read_back_as_the_integer_written() -> None:
    read = WorkflowData.from_pro_string(WorkflowData({"ON": True, "OFF": False}).to_pro_string())
    assert dict(read) == {"ON": 1, "OFF": 0}
    assert type(read["ON"]) is int


def test_from_pro_string_skips_comments_and_blank_lines() -> None:
    text = "// written by hand\nFunction {\n\n  A = 1;\n  // note\n  B[] = Complex[1.0, 2.0];\n}\n"
    assert dict(WorkflowData.from_pro_string(text)) == {"A": 1, "B": complex(1.0, 2.0)}


@pytest.mark.parametrize(
    "line", ["A = ;", "A = one;", "A[] = Complex[1.0];", "Group { }", "A = 1", "1A = 2;"]
)
def test_from_pro_string_refuses_what_it_does_not_write(line: str) -> None:
    with pytest.raises(ValueError, match="Unrecognised line of a scalars file"):
        WorkflowData.from_pro_string(f"Function {{\n  {line}\n}}\n")
