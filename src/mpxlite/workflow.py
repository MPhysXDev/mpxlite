# Copyright (c) 2025 MPhysX
# SPDX-License-Identifier: MIT

"""WorkflowData: Python-side container of GetDP scalars.

A :class:`WorkflowData` instance holds a dict of named scalars that gets
serialized as a GetDP ``Function {…}`` block at runtime, providing the
parameters that the .pro files reference at the GetDP level.

Supported value types are ``bool``, ``int``, ``float``, ``complex`` and
``str``; numpy scalars (``np.bool_``, ``np.integer``, ``np.floating``,
``np.complexfloating``) are coerced to the matching builtin when set.

Real values are emitted as GetDP constants (``N = 1.5;``). GetDP constants
must be real, so complex values are emitted as zero-argument functions
(``N[] = Complex[1.0, 2.0];``) and **must be referenced as ``N[]``** in the
``.pro`` files (a bare ``N`` is a GetDP error).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping, MutableMapping
from pathlib import Path
from typing import Any

import numpy as np

# A GetDP scalar: a number (bool/int/float/complex) or a string literal.
# Booleans become 1/0; complex values become ``Name[] = Complex[re, im];``.
Scalar = bool | int | float | complex | str

_VALID_NAME = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")

# Characters GetDP string literals cannot carry safely (no reliable escaping).
_FORBIDDEN_STR_CHARS = ('"', "\\", "\n", "\r")

# The lines to_pro_string() writes, read back by from_pro_string().
_REAL_LINE = re.compile(r"^([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.+?)\s*;$")
_COMPLEX_LINE = re.compile(
    r"^([A-Za-z_][A-Za-z_0-9]*)\[\]\s*=\s*Complex\[\s*([^,\]]+?)\s*,\s*([^\]]+?)\s*\]\s*;$"
)
_INT_VALUE = re.compile(r"[+-]?\d+")


class WorkflowData(MutableMapping[str, Scalar]):
    r"""Typed dict-like container of GetDP scalars.

    Scalars are name → value pairs serialized to a GetDP ``Function {…}``
    block that the manifest .pro file ``Include``s before any other source.
    Subsequent ``.pro`` blocks reference real scalars by name as if they
    were ordinary GetDP constants, and complex scalars as ``Name[]``.

    Names must match the GetDP identifier syntax ``[A-Za-z_][A-Za-z_0-9]*``.
    Values are validated and normalized when set:

    * ``bool`` / ``np.bool_`` → ``bool``, serialized as ``1`` / ``0``;
    * ``int`` / ``np.integer`` → ``int``;
    * ``float`` / ``np.floating`` → ``float`` (must be finite);
    * ``complex`` / ``np.complexfloating`` → ``complex`` (both parts finite),
      serialized as ``Name[] = Complex[re, im];``;
    * ``str`` → quoted verbatim; it must not contain ``"``, ``\``, newline
      or carriage return, which GetDP string literals cannot carry.

    Examples:
        >>> wf = WorkflowData()
        >>> wf['EM3D_FREQ'] = 12.0e9
        >>> wf['EPS_R'] = complex(4.0, -0.1)
        >>> text = wf.to_pro_string()
        >>> 'EM3D_FREQ = 12000000000.0;' in text
        True
        >>> 'EPS_R[] = Complex[4.0, -0.1];' in text
        True
    """

    def __init__(self, initial: Mapping[str, Scalar] | None = None) -> None:
        """Initialize, optionally from an existing mapping.

        Args:
            initial: Optional mapping of (name, value) pairs to seed the data.
        """
        self._data: dict[str, Scalar] = {}
        if initial is not None:
            for key, value in initial.items():
                self[key] = value

    # -- MutableMapping interface --------------------------------------

    def __getitem__(self, key: str) -> Scalar:
        return self._data[key]

    def __setitem__(self, key: str, value: Scalar) -> None:
        self._validate_name(key)
        self._data[key] = self._coerce_value(key, value)

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"WorkflowData({self._data!r})"

    # -- Validation ----------------------------------------------------

    @staticmethod
    def _validate_name(name: Any) -> None:
        if not isinstance(name, str):
            raise TypeError(f"GetDP scalar name must be str, got {type(name).__name__}")
        if not _VALID_NAME.match(name):
            raise ValueError(
                f"Invalid GetDP scalar name {name!r}: must match [A-Za-z_][A-Za-z_0-9]*"
            )

    @staticmethod
    def _coerce_value(name: str, value: Any) -> Scalar:
        """Validate ``value`` and normalize it to an exact builtin type.

        Subclasses (numpy scalars, ``IntEnum``, …) are converted to the exact
        builtin so that serialization via ``repr`` yields a plain literal.

        Args:
            name: Scalar name, used in error messages.
            value: Candidate value.

        Returns:
            The value as an exact ``bool``, ``int``, ``float``, ``complex``
            or ``str``.

        Raises:
            TypeError: If the type is not supported.
            ValueError: If a number is non-finite or a string contains a
                forbidden character.
        """
        # Order matters: bool before int (bool subclasses int), and numpy
        # floating/complex are tested via their abstract numpy bases.
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            real = float(value)
            if not math.isfinite(real):
                raise ValueError(f"GetDP scalar {name!r} must be finite, got {real!r}")
            return real
        if isinstance(value, (complex, np.complexfloating)):
            cplx = complex(value)
            if not (math.isfinite(cplx.real) and math.isfinite(cplx.imag)):
                raise ValueError(
                    f"GetDP scalar {name!r} must have finite real and imaginary parts, got {cplx!r}"
                )
            return cplx
        if isinstance(value, str):
            text = str(value)
            bad = [c for c in _FORBIDDEN_STR_CHARS if c in text]
            if bad:
                raise ValueError(
                    f"GetDP string scalar {name!r} contains forbidden character(s) "
                    f"{', '.join(repr(c) for c in bad)} (GetDP string literals cannot "
                    f'carry ", \\, newline or carriage return)'
                )
            return text
        raise TypeError(
            f"GetDP scalar value must be bool/int/float/complex/str, got {type(value).__name__}"
        )

    # -- Serialization -------------------------------------------------

    @staticmethod
    def _format_line(key: str, value: Scalar) -> str:
        if isinstance(value, bool):
            return f"  {key} = {'1' if value else '0'};"
        if isinstance(value, (int, float)):
            return f"  {key} = {value!r};"
        if isinstance(value, complex):
            return f"  {key}[] = Complex[{value.real!r}, {value.imag!r}];"
        if isinstance(value, str):
            return f'  {key} = "{value}";'
        raise TypeError(  # pragma: no cover  (unreachable: blocked by _coerce_value)
            f"Unsupported value type: {type(value).__name__}"
        )

    def to_pro_string(self) -> str:
        """Serialize the data as a GetDP ``Function {…}`` block string.

        Real scalars become ``  N = <value>;``; complex scalars become
        ``  N[] = Complex[<re>, <im>];`` and must be referenced as ``N[]``.

        Returns:
            A string ready to be written to a ``.pro`` file. Keys are emitted
            in sorted order to ensure reproducible output.
        """
        lines = ["Function {"]
        lines.extend(self._format_line(key, self._data[key]) for key in sorted(self._data))
        lines.append("}")
        return "\n".join(lines) + "\n"

    def write_pro(self, path: Path | str) -> None:
        """Serialize the data to a ``.pro`` file at ``path``.

        Args:
            path: Destination file path. Parent directories must already exist.
        """
        Path(path).write_text(self.to_pro_string(), encoding="utf-8")

    @classmethod
    def from_pro_string(cls, text: str) -> WorkflowData:
        """The scalars of a text written by :meth:`to_pro_string`.

        Each ``Name = value;`` line gives an ``int`` (digits only), a
        ``float`` (any other number) or a ``str`` (quoted); each
        ``Name[] = Complex[re, im];`` line a ``complex``. A ``bool`` is
        written as ``1`` / ``0`` and so comes back as an ``int``. The
        ``Function {`` and ``}`` lines, blank lines and ``//`` comments are
        skipped.

        Args:
            text: The content of the file.

        Returns:
            A new :class:`WorkflowData`.

        Raises:
            ValueError: on a line that :meth:`to_pro_string` does not write.
        """
        data = cls()
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("//") or line in ("Function {", "}"):
                continue
            complex_match = _COMPLEX_LINE.match(line)
            real_match = None if complex_match else _REAL_LINE.match(line)
            try:
                if complex_match is not None:
                    name, re_part, im_part = complex_match.groups()
                    data[name] = complex(float(re_part), float(im_part))
                elif real_match is not None:
                    name, value = real_match.groups()
                    if len(value) >= 2 and value[0] == value[-1] == '"':
                        data[name] = value[1:-1]
                    elif _INT_VALUE.fullmatch(value):
                        data[name] = int(value)
                    else:
                        data[name] = float(value)
                else:
                    raise ValueError("not a scalar line")
            except ValueError as exc:
                raise ValueError(f"Unrecognised line of a scalars file: {raw!r} ({exc})") from exc
        return data

    @classmethod
    def read_pro(cls, path: Path | str) -> WorkflowData:
        """The scalars of a file written by :meth:`write_pro`.

        For instance the ``workflow.pro`` of a run,
        :attr:`mpxlite.SolverResult.workflow_path`.

        Args:
            path: The file.

        Returns:
            A new :class:`WorkflowData` (see :meth:`from_pro_string`).

        Raises:
            ValueError: on a line that :meth:`write_pro` does not write.
        """
        return cls.from_pro_string(Path(path).read_text(encoding="utf-8"))
