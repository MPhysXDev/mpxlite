# Copyright (c) 2025 Thierry Valet
# SPDX-License-Identifier: MIT

"""WorkflowData: Python-side container of GetDP scalars.

A :class:`WorkflowData` instance holds a dict of named scalars that gets
serialized as a GetDP ``Function {…}`` block at runtime, providing the
parameters that the .pro files reference at the GetDP level.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, MutableMapping
from pathlib import Path
from typing import Any

# A GetDP scalar can be a number (real/integer) or a string literal.
# Booleans are coerced to integers (1/0) at serialization time.
SerializableScalar = float | int | str | bool

_VALID_NAME = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")


class WorkflowData(MutableMapping[str, SerializableScalar]):
    """Typed dict-like container of GetDP scalars.

    Scalars are name → value pairs serialized to a GetDP ``Function {…}``
    block that the manifest .pro file ``Include``s before any other source.
    Subsequent ``.pro`` blocks reference these scalars by name as if they
    were ordinary GetDP constants.

    Names must match the GetDP identifier syntax ``[A-Za-z_][A-Za-z_0-9]*``.
    Values must be ``bool``, ``int``, ``float``, or ``str``. Strings are
    quoted and escaped. Booleans become ``1`` / ``0``.

    Examples:
        >>> wf = WorkflowData()
        >>> wf['EM3D_FREQ'] = 12.0e9
        >>> wf['LABEL'] = "demo"
        >>> 'EM3D_FREQ = 12000000000.0' in wf.to_pro_string()
        True
    """

    def __init__(self, initial: Mapping[str, SerializableScalar] | None = None) -> None:
        """Initialize, optionally from an existing mapping.

        Args:
            initial: Optional mapping of (name, value) pairs to seed the data.
        """
        self._data: dict[str, SerializableScalar] = {}
        if initial is not None:
            for key, value in initial.items():
                self[key] = value

    # -- MutableMapping interface --------------------------------------

    def __getitem__(self, key: str) -> SerializableScalar:
        return self._data[key]

    def __setitem__(self, key: str, value: SerializableScalar) -> None:
        self._validate_name(key)
        self._validate_value(value)
        self._data[key] = value

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
    def _validate_value(value: Any) -> None:
        if not isinstance(value, (bool, int, float, str)):
            raise TypeError(
                f"GetDP scalar value must be bool/int/float/str, got {type(value).__name__}"
            )

    # -- Serialization -------------------------------------------------

    @staticmethod
    def _format_value(value: SerializableScalar) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, (int, float)):
            return repr(value)
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        raise TypeError(  # pragma: no cover  (unreachable: blocked by _validate_value)
            f"Unsupported value type: {type(value).__name__}"
        )

    def to_pro_string(self) -> str:
        """Serialize the data as a GetDP ``Function {…}`` block string.

        Returns:
            A string ready to be written to a ``.pro`` file. Keys are emitted
            in sorted order to ensure reproducible output.
        """
        lines = ["Function {"]
        for key in sorted(self._data):
            value = self._data[key]
            lines.append(f"  {key} = {self._format_value(value)};")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def write_pro(self, path: Path | str) -> None:
        """Serialize the data to a ``.pro`` file at ``path``.

        Args:
            path: Destination file path. Parent directories must already exist.
        """
        Path(path).write_text(self.to_pro_string(), encoding="utf-8")
