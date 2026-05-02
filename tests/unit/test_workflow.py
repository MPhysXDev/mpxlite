# Copyright (c) 2025 Thierry Valet
# SPDX-License-Identifier: MIT

"""Unit tests for mpxlite.workflow.WorkflowData."""

from __future__ import annotations

from pathlib import Path

import pytest

from mpxlite.workflow import WorkflowData


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
    with pytest.raises(TypeError, match="must be bool/int/float/str"):
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


def test_string_value_escapes_quotes_and_backslashes() -> None:
    wf = WorkflowData()
    wf["S"] = 'a"b\\c'
    assert 'S = "a\\"b\\\\c";' in wf.to_pro_string()


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
