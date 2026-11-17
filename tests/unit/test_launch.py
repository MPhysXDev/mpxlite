# Copyright (c) 2025 MPhysX
# SPDX-License-Identifier: MIT

"""Unit tests for mpxlite.launch.LaunchSpec."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from mpxlite.launch import LaunchSpec, parse_count

# ----- validation ---------------------------------------------------------


def test_defaults() -> None:
    spec = LaunchSpec()
    assert spec.np is None
    assert spec.threads is None
    assert spec.cpus is None
    assert spec.launcher is None
    assert dict(spec.env) == {}


@pytest.mark.parametrize("field", ["np", "threads"])
@pytest.mark.parametrize("bad", [0, -1])
def test_np_threads_must_be_positive(field: str, bad: int) -> None:
    with pytest.raises(ValueError, match=f"{field} must be >= 1"):
        LaunchSpec(**{field: bad})


@pytest.mark.parametrize("field", ["np", "threads"])
@pytest.mark.parametrize("bad", [True, 2.0, "2"])
def test_np_threads_must_be_int(field: str, bad: Any) -> None:
    with pytest.raises(TypeError, match=f"{field} must be an int"):
        LaunchSpec(**{field: bad})


def test_cpus_list_coerced_to_tuple() -> None:
    spec = LaunchSpec(cpus=[3, 1, 2])  # type: ignore[arg-type]
    assert spec.cpus == (3, 1, 2)


@pytest.mark.parametrize(
    ("bad", "exc", "match"),
    [
        ((), ValueError, "must not be empty"),
        ((0, -1), ValueError, ">= 0"),
        ((1, 1), ValueError, "unique"),
        ((1, 2.0), TypeError, "must be an int"),
        ((True,), TypeError, "must be an int"),
        ("0,1", TypeError, "not a string"),
    ],
)
def test_cpus_validation(bad: Any, exc: type[Exception], match: str) -> None:
    with pytest.raises(exc, match=match):
        LaunchSpec(cpus=bad)


def test_numpy_integers_coerced_to_int() -> None:
    """Regression (review S4): numpy CPU ids / counts are ints, stored as plain int."""
    spec = LaunchSpec(np=np.int64(2), threads=np.int32(1), cpus=np.arange(4, 6))
    assert spec.cpus == (4, 5) and spec.np == 2 and spec.threads == 1
    assert all(type(c) is int for c in spec.cpus)
    assert type(spec.np) is int and type(spec.threads) is int
    assert spec.to_env({})["MPXLITE_GETDP_CPUS"] == "4,5"


@pytest.mark.parametrize("bad", [np.True_, np.bool_(False), True])
def test_bools_rejected_everywhere(bad: Any) -> None:
    with pytest.raises(TypeError, match="must be an int"):
        LaunchSpec(cpus=(0, bad))
    with pytest.raises(TypeError, match="must be an int"):
        LaunchSpec(np=bad)


def test_launcher_validation() -> None:
    with pytest.raises(ValueError, match="blank"):
        LaunchSpec(np=2, launcher="  ")
    with pytest.raises(TypeError, match="launcher must be a str"):
        LaunchSpec(np=2, launcher=["mpirun"])  # type: ignore[arg-type]


def test_launcher_requires_explicit_np() -> None:
    with pytest.raises(ValueError, match="launcher requires np"):
        LaunchSpec(launcher="mpirun --bind-to none -np 2")
    assert LaunchSpec(np=2, launcher="mpirun -np 2").launcher == "mpirun -np 2"


@pytest.mark.parametrize(
    ("key", "field"),
    [
        ("MPXLITE_GETDP_NP", "np"),
        ("MPXLITE_GETDP_THREADS", "threads"),
        ("MPXLITE_GETDP_CPUS", "cpus"),
        ("MPXLITE_GETDP_LAUNCHER", "launcher"),
    ],
)
def test_env_rejects_keys_owned_by_fields(key: str, field: str) -> None:
    with pytest.raises(ValueError, match=f"must not set {key}: use LaunchSpec.{field}"):
        LaunchSpec(env={key: "2"})


def test_env_other_mpxlite_keys_allowed() -> None:
    spec = LaunchSpec(env={"MPXLITE_GETDP_BINARY": "/usr/bin/getdp", "MPXLITE_GETDP": "/x"})
    assert spec.to_env({})["MPXLITE_GETDP_BINARY"] == "/usr/bin/getdp"


@pytest.mark.parametrize(
    ("bad", "exc"),
    [
        ({"": "x"}, ValueError),
        ({"A=B": "x"}, ValueError),
        ({"A": "x\0y"}, ValueError),
        ({"A": 1}, TypeError),
        ({1: "x"}, TypeError),
        (["A"], TypeError),
    ],
)
def test_env_validation(bad: Any, exc: type[Exception]) -> None:
    with pytest.raises(exc):
        LaunchSpec(env=bad)


def test_env_is_a_frozen_copy() -> None:
    src = {"A": "1"}
    spec = LaunchSpec(env=src)
    src["A"] = "2"
    assert spec.env["A"] == "1"
    with pytest.raises(TypeError):
        spec.env["B"] = "3"  # type: ignore[index]


def test_frozen_hashable_and_replaceable() -> None:
    spec = LaunchSpec(np=2, cpus=(0, 1), env={"A": "1"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.np = 3  # type: ignore[misc]
    hash(spec)
    other = dataclasses.replace(spec, cpus=(4, 5))
    assert other.cpus == (4, 5)
    assert other.np == 2
    assert dict(other.env) == {"A": "1"}
    assert spec == LaunchSpec(np=2, cpus=(0, 1), env={"A": "1"})


# ----- effective_np / width -----------------------------------------------


def test_effective_np_explicit_wins() -> None:
    assert LaunchSpec(np=4).effective_np({"MPXLITE_GETDP_NP": "8"}) == 4


def test_effective_np_from_env_then_default() -> None:
    assert LaunchSpec().effective_np({"MPXLITE_GETDP_NP": "8"}) == 8
    assert LaunchSpec().effective_np({}) == 1
    assert LaunchSpec().effective_np({"MPXLITE_GETDP_NP": ""}) == 1  # empty == unset


def test_effective_np_explicit_ignores_bad_env() -> None:
    assert LaunchSpec(np=2).effective_np({"MPXLITE_GETDP_NP": "junk"}) == 2


# Exactly the values getdp_runner.sh rejects (see test_runner_script.py).
_BAD_COUNTS = ["0", "-2", "two", "1.5", " 2", "2 ", " ", "02", "+2", "2_0", "1e3", "1234567890"]


@pytest.mark.parametrize("bad", _BAD_COUNTS)
def test_effective_np_bad_env(bad: str) -> None:
    with pytest.raises(ValueError, match="MPXLITE_GETDP_NP"):
        LaunchSpec().effective_np({"MPXLITE_GETDP_NP": bad})


@pytest.mark.parametrize("bad", _BAD_COUNTS)
def test_effective_threads_bad_mpxlite_env(bad: str) -> None:
    with pytest.raises(ValueError, match="MPXLITE_GETDP_THREADS"):
        LaunchSpec().effective_threads({"MPXLITE_GETDP_THREADS": bad})


@pytest.mark.parametrize(
    ("raw", "expected"), [(None, None), ("", None), ("1", 1), ("16", 16), ("999999999", 999999999)]
)
def test_parse_count_valid(raw: str | None, expected: int | None) -> None:
    assert parse_count("X", raw) == expected


def test_width() -> None:
    assert LaunchSpec().width({}) == 1
    assert LaunchSpec(np=4).width({}) == 4
    assert LaunchSpec(np=4, threads=2).width({}) == 8
    assert LaunchSpec(threads=3).width({"MPXLITE_GETDP_NP": "2"}) == 6


# ----- to_env -------------------------------------------------------------


def test_to_env_empty_spec_copies_base() -> None:
    base = {"PATH": "/bin", "MPXLITE_GETDP_NP": "3", "OMP_NUM_THREADS": "2"}
    env = LaunchSpec().to_env(base)
    assert env == base
    assert env is not base


def test_to_env_sets_only_non_none_fields() -> None:
    base = {"MPXLITE_GETDP_THREADS": "9", "KEEP": "1"}
    env = LaunchSpec(np=2, cpus=(0, 1, 2, 3)).to_env(base)
    assert env["MPXLITE_GETDP_NP"] == "2"
    assert env["MPXLITE_GETDP_CPUS"] == "0,1,2,3"
    assert env["MPXLITE_GETDP_THREADS"] == "9"  # inherited
    assert env["KEEP"] == "1"
    assert "MPXLITE_GETDP_LAUNCHER" not in env
    assert base == {"MPXLITE_GETDP_THREADS": "9", "KEEP": "1"}  # not modified


def test_to_env_all_fields_and_env_applied_last() -> None:
    spec = LaunchSpec(
        np=2,
        threads=4,
        cpus=(8, 9),
        launcher="mpirun -np 2",
        env={"OMP_PROC_BIND": "close", "OMP_NUM_THREADS": "7"},
    )
    env = spec.to_env({"OMP_PROC_BIND": "spread"})
    assert env == {
        "MPXLITE_GETDP_NP": "2",
        "MPXLITE_GETDP_THREADS": "4",
        "MPXLITE_GETDP_CPUS": "8,9",
        "MPXLITE_GETDP_LAUNCHER": "mpirun -np 2",
        "OMP_PROC_BIND": "close",
        "OMP_NUM_THREADS": "7",
    }


@pytest.mark.parametrize(
    ("spec", "base", "expected"),
    [
        (LaunchSpec(np=4), {}, "1"),  # multi-rank, threads set nowhere -> 1
        (LaunchSpec(), {"MPXLITE_GETDP_NP": "4"}, "1"),  # np from base env
        (LaunchSpec(np=4, threads=3), {}, "3"),  # explicit threads kept
        (LaunchSpec(np=4), {"MPXLITE_GETDP_THREADS": "2"}, "2"),  # inherited
        (LaunchSpec(np=1), {}, None),  # sequential: untouched
        (LaunchSpec(), {}, None),
    ],
)
def test_to_env_threads_default_for_multi_rank(
    spec: LaunchSpec, base: dict[str, str], expected: str | None
) -> None:
    assert spec.to_env(base).get("MPXLITE_GETDP_THREADS") == expected


def test_to_env_threads_default_respects_omp_num_threads() -> None:
    env = LaunchSpec(np=4).to_env({"OMP_NUM_THREADS": "2"})
    assert "MPXLITE_GETDP_THREADS" not in env
    assert env["OMP_NUM_THREADS"] == "2"


def test_to_env_threads_default_for_pinned_single_rank() -> None:
    assert LaunchSpec(cpus=(3,)).to_env({})["MPXLITE_GETDP_THREADS"] == "1"
    assert "MPXLITE_GETDP_THREADS" not in LaunchSpec(cpus=(3,)).to_env({"OMP_NUM_THREADS": "4"})


@pytest.mark.parametrize(
    ("spec", "base", "expected"),
    [
        (LaunchSpec(), {}, 1),
        (LaunchSpec(threads=3), {"OMP_NUM_THREADS": "8"}, 3),
        (LaunchSpec(), {"MPXLITE_GETDP_THREADS": "2"}, 2),
        (LaunchSpec(), {"OMP_NUM_THREADS": "4"}, 4),
        (LaunchSpec(), {"OMP_NUM_THREADS": "4,2"}, 4),  # OpenMP nested list (F-9)
        (LaunchSpec(), {"OMP_NUM_THREADS": " 6 , 3 , 1 "}, 6),
        (LaunchSpec(np=2), {}, 1),
    ],
)
def test_effective_threads(spec: LaunchSpec, base: dict[str, str], expected: int) -> None:
    assert spec.effective_threads(base) == expected


@pytest.mark.parametrize("bad", ["zero", "0", ",2", "0,4", "x,1"])
def test_effective_threads_bad_env(bad: str) -> None:
    with pytest.raises(ValueError, match="OMP_NUM_THREADS"):
        LaunchSpec().effective_threads({"OMP_NUM_THREADS": bad})


def test_width_uses_inherited_threads() -> None:
    assert LaunchSpec(np=2).width({"MPXLITE_GETDP_THREADS": "3"}) == 6
