# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""JSON forms of the symmetry objects of a periodic cell, by tag *name*.

A Gmsh job reports what it detected and used (:mod:`mpxlite.gmshjob`) as
JSON; tags, which a plug-in knows by name, are written by name. These
functions write and read back the objects of :mod:`mpxlite.cells`:

* :func:`lattice_op_to_json` / :func:`lattice_op_from_json`:
  :class:`~mpxlite.cells.spacegroup.LatticeOp` as
  ``{"name", "matrix", "shift"}`` (lattice coordinates);
* :func:`symmetry_to_json` / :func:`symmetry_from_json`:
  :class:`~mpxlite.cells.detect.DetectedSymmetry`, its tag permutations,
  direction signs, window tags and classes by name;
* :func:`unit_to_json` / :func:`unit_from_json`:
  :class:`~mpxlite.cells.asymmetric.AsymmetricUnit`.

Floats are written as Python ``repr`` gives them (exact round trip); a class
label is a JSON value (a tuple label is written as a list and read back as a
tuple).
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np

from mpxlite.cells.asymmetric import AsymmetricUnit, UnitEdge
from mpxlite.cells.detect import DetectedSymmetry, NearSymmetry
from mpxlite.cells.spacegroup import LatticeOp

__all__ = [
    "lattice_op_from_json",
    "lattice_op_to_json",
    "symmetry_from_json",
    "symmetry_to_json",
    "unit_from_json",
    "unit_to_json",
]


def _label(value: Hashable) -> Any:
    if isinstance(value, tuple):
        return [_label(v) for v in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return repr(value)


def _hashable(value: Any) -> Hashable:
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    return value


def lattice_op_to_json(op: LatticeOp) -> dict[str, Any]:
    """``{"name", "matrix", "shift"}`` of an operation (lattice coordinates)."""
    return {"name": op.name, "matrix": op.matrix.tolist(), "shift": op.shift.tolist()}


def lattice_op_from_json(doc: Mapping[str, Any]) -> LatticeOp:
    """The operation of :func:`lattice_op_to_json`."""
    return LatticeOp(str(doc["name"]), np.asarray(doc["matrix"]), np.asarray(doc["shift"]))


def _names(tags: Mapping[int, int] | Mapping[int, Any], names: Mapping[int, str]) -> dict:
    return {names[t]: v for t, v in tags.items()}


def symmetry_to_json(sym: DetectedSymmetry, names: Mapping[int, str]) -> dict[str, Any]:
    """The JSON form of a detected group, its tags written by name.

    Args:
        sym: The group.
        names: Tag -> name of every tag the group mentions.

    Returns:
        ``{"name", "order", "point_group", "lattice", "lattice_type",
        "holohedry", "centre", "tol", "near_tol", "operations": [{"name",
        "matrix", "shift", "deviation", "permutation", "direction_signs"}],
        "near", "window", "classes", "warnings", "primitive", "orthogonal"}``.
    """
    ops = []
    for op, perm, signs, dev in zip(
        sym.ops, sym.permutations, sym.direction_signs, sym.deviations, strict=True
    ):
        ops.append(
            {
                **lattice_op_to_json(op),
                "deviation": float(dev),
                "permutation": {names[a]: names[b] for a, b in perm.items()},
                "direction_signs": _names(signs, names),
            }
        )
    near = [
        {
            "operation": None if n.op is None else lattice_op_to_json(n.op),
            "matrix": np.asarray(n.matrix).tolist(),
            "deviation": float(n.deviation),
            "tag": None if n.tag is None else names[n.tag],
            "reason": n.reason,
        }
        for n in sym.near
    ]
    return {
        "name": sym.name,
        "order": sym.order,
        "point_group": sym.point_group,
        "lattice": np.asarray(sym.lattice).tolist(),
        "lattice_type": sym.lattice_type,
        "holohedry": [np.asarray(m).tolist() for m in sym.holohedry],
        "centre": np.asarray(sym.centre).tolist(),
        "tol": float(sym.tol),
        "near_tol": float(sym.near_tol),
        "operations": ops,
        "near": near,
        "window": [names[t] for t in sym.window_tags],
        "classes": {names[t]: _label(c) for t, c in sym.classes.items()},
        "warnings": list(sym.warnings),
        "primitive": sym.primitive,
        "orthogonal": sym.orthogonal,
    }


def symmetry_from_json(doc: Mapping[str, Any], tags: Mapping[str, int]) -> DetectedSymmetry:
    """The group of :func:`symmetry_to_json`, its tags given by name.

    Args:
        doc: The JSON form.
        tags: Name -> tag (for instance the physical tags of the mesh).

    Returns:
        The :class:`~mpxlite.cells.detect.DetectedSymmetry`.
    """
    ops = tuple(lattice_op_from_json(o) for o in doc["operations"])
    near = tuple(
        NearSymmetry(
            None if n["operation"] is None else lattice_op_from_json(n["operation"]),
            np.asarray(n["matrix"]),
            float(n["deviation"]),
            None if n["tag"] is None else tags[n["tag"]],
            str(n["reason"]),
        )
        for n in doc["near"]
    )
    return DetectedSymmetry(
        lattice=np.asarray(doc["lattice"], dtype=float),
        lattice_type=str(doc["lattice_type"]),
        holohedry=tuple(np.asarray(m) for m in doc["holohedry"]),
        ops=ops,
        permutations=tuple(
            {tags[a]: tags[b] for a, b in o["permutation"].items()} for o in doc["operations"]
        ),
        direction_signs=tuple(
            {tags[a]: int(s) for a, s in o["direction_signs"].items()} for o in doc["operations"]
        ),
        deviations=tuple(float(o["deviation"]) for o in doc["operations"]),
        name=str(doc["name"]),
        point_group=str(doc["point_group"]),
        centre=np.asarray(doc["centre"], dtype=float),
        tol=float(doc["tol"]),
        near_tol=float(doc["near_tol"]),
        near=near,
        window_tags=tuple(tags[n] for n in doc["window"]),
        warnings=tuple(doc["warnings"]),
        classes={tags[n]: _hashable(c) for n, c in doc["classes"].items()},
    )


def _array(value: np.ndarray | None) -> Any:
    return None if value is None else np.asarray(value).tolist()


def unit_to_json(unit: AsymmetricUnit) -> dict[str, Any]:
    """The JSON form of an asymmetric unit (Cartesian lengths, lattice coordinates)."""
    return {
        "name": unit.name,
        "lattice": np.asarray(unit.lattice).tolist(),
        "operations": [lattice_op_to_json(op) for op in unit.ops],
        "frame": np.asarray(unit.frame).tolist(),
        "origin": np.asarray(unit.origin).tolist(),
        "vertices": np.asarray(unit.vertices).tolist(),
        "edges": [
            {
                "index": e.index,
                "a": _array(e.a),
                "b": _array(e.b),
                "kind": e.kind,
                "partner": e.partner,
                "g": _array(e.g),
                "t": _array(e.t),
                "operation": e.op,
            }
            for e in unit.edges
        ],
        "cuts": [np.asarray(c).tolist() for c in unit.cuts],
        "translations": [list(r) for r in unit.translations],
        "window_lo": np.asarray(unit.window_lo).tolist(),
        "area": unit.area,
    }


def unit_from_json(doc: Mapping[str, Any]) -> AsymmetricUnit:
    """The unit of :func:`unit_to_json`."""

    def opt(value: Any) -> np.ndarray | None:
        return None if value is None else np.asarray(value, dtype=float)

    edges = tuple(
        UnitEdge(
            index=int(e["index"]),
            a=np.asarray(e["a"], dtype=float),
            b=np.asarray(e["b"], dtype=float),
            kind=str(e["kind"]),
            partner=None if e["partner"] is None else int(e["partner"]),
            g=opt(e["g"]),
            t=opt(e["t"]),
            op=str(e["operation"]),
        )
        for e in doc["edges"]
    )
    return AsymmetricUnit(
        name=str(doc["name"]),
        lattice=np.asarray(doc["lattice"], dtype=float),
        ops=tuple(lattice_op_from_json(o) for o in doc["operations"]),
        frame=np.asarray(doc["frame"]),
        origin=np.asarray(doc["origin"], dtype=float),
        vertices=np.asarray(doc["vertices"], dtype=float),
        edges=edges,
        cuts=tuple(np.asarray(c, dtype=float) for c in doc["cuts"]),
        translations=tuple((int(r[0]), int(r[1])) for r in doc["translations"]),
        window_lo=np.asarray(doc["window_lo"], dtype=float),
    )
