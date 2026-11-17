# Copyright (c) 2025 MPhysX
# SPDX-License-Identifier: MIT

"""LaunchSpec: per-run description of how ``getdp`` is launched.

A :class:`LaunchSpec` carries the parallel layout of ONE run (MPI rank
count, OpenMP threads per rank, CPU affinity, explicit launcher prefix and
extra environment variables). It is translated into the ``MPXLITE_GETDP_*``
environment contract of the bundled ``tools/getdp_runner.sh`` wrapper by
:meth:`LaunchSpec.to_env`, and handed to the ``getdp`` subprocess through its
environment only: the parent process environment is never modified, so
concurrent runs with different layouts do not interfere.

Process-wide ``MPXLITE_*`` environment variables remain the defaults for
every field left to ``None``.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

ENV_NP = "MPXLITE_GETDP_NP"
ENV_THREADS = "MPXLITE_GETDP_THREADS"
ENV_CPUS = "MPXLITE_GETDP_CPUS"
ENV_LAUNCHER = "MPXLITE_GETDP_LAUNCHER"

# Launch variables owned by a dedicated LaunchSpec field: ``env`` may not set them.
_FIELD_OF_ENV = {ENV_NP: "np", ENV_THREADS: "threads", ENV_CPUS: "cpus", ENV_LAUNCHER: "launcher"}

# Same rule as ``is_pos_int`` in getdp_runner.sh: a positive decimal integer,
# no sign, no leading zero, no blank, at most 9 digits.
_POS_INT_RE = re.compile(r"[1-9][0-9]{0,8}")


def parse_count(name: str, raw: str | None) -> int | None:
    """Parse an ``MPXLITE_GETDP_NP`` / ``_THREADS`` value exactly like the wrapper.

    Args:
        name: Variable name, for the error message.
        raw: The raw value; ``None`` or ``""`` means unset.

    Returns:
        The integer value, or ``None`` when unset.

    Raises:
        ValueError: if ``raw`` is set but does not match ``^[1-9][0-9]{0,8}$``
            (e.g. ``"0"``, ``"02"``, ``"+2"``, ``" 2"``, ``"2_0"``, ``" "``).
    """
    if raw is None or raw == "":
        return None
    if not _POS_INT_RE.fullmatch(raw):
        raise ValueError(f"{name}={raw!r} must be an integer >= 1 (digits only, no leading zero)")
    return int(raw)


def as_int(value: object, what: str) -> int:
    """``value`` as a plain ``int``: ints and numpy integers, never a bool.

    Args:
        value: The value to convert (anything implementing ``__index__``).
        what: Name of the value, for the error message.

    Returns:
        ``operator.index(value)``.

    Raises:
        TypeError: on a bool (Python or numpy) or a non-integer value.
    """
    kind = type(value)
    is_numpy_bool = kind.__module__ == "numpy" and kind.__name__ in ("bool", "bool_")
    if isinstance(value, bool) or is_numpy_bool:
        raise TypeError(f"{what} must be an int, got {value!r}")
    try:
        return operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(f"{what} must be an int, got {type(value).__name__}") from None


def _check_positive_int(name: str, value: object) -> int:
    number = as_int(value, f"LaunchSpec.{name}")
    if number < 1:
        raise ValueError(f"LaunchSpec.{name} must be >= 1, got {number}")
    return number


@dataclass(frozen=True)
class LaunchSpec:
    """Parallel layout and environment of a single ``getdp`` run.

    Every field defaults to ``None`` (or an empty mapping), meaning "inherit
    the process-wide default": the corresponding ``MPXLITE_GETDP_*`` variable
    of the base environment, if any, is then left untouched.

    Attributes:
        np: Number of MPI ranks (an int; numpy integers are accepted and
            stored as ``int``, bools are rejected). ``None`` falls back to ``MPXLITE_GETDP_NP``
            from the base environment, then to 1.
        threads: OpenMP threads per rank (``MPXLITE_GETDP_THREADS``, which
            the wrapper exports as ``OMP_NUM_THREADS``). ``None`` inherits.
        cpus: Logical CPU ids this run may use (``MPXLITE_GETDP_CPUS``,
            applied by the wrapper through ``taskset -c``). Any sequence of
            ints (numpy integers included) is accepted and stored as a
            tuple of ``int``. ``None`` means no pinning.
        launcher: Explicit launcher prefix (``MPXLITE_GETDP_LAUNCHER``),
            e.g. ``"mpirun --bind-to none -np 16"``, used by the wrapper as-is
            whatever the rank count. It must start exactly ``np`` ranks
            (``np`` is what executors reserve CPUs for), so ``np`` must be
            set explicitly with it. With ``cpus``, an Open MPI launcher
            should include ``--bind-to none``. ``None`` inherits.
        env: Extra environment variables, applied last. Stored as a
            read-only copy. It must not set ``MPXLITE_GETDP_NP``,
            ``_THREADS``, ``_CPUS`` nor ``_LAUNCHER``: use the dedicated
            fields (other variables, e.g. ``MPXLITE_GETDP_BINARY``, are fine).

    Examples:
        >>> spec = LaunchSpec(np=2, threads=1, cpus=(4, 5))
        >>> env = spec.to_env({})
        >>> env["MPXLITE_GETDP_NP"], env["MPXLITE_GETDP_CPUS"]
        ('2', '4,5')
        >>> spec.width({})
        2
    """

    np: int | None = None
    threads: int | None = None
    cpus: tuple[int, ...] | None = None
    launcher: str | None = None
    env: Mapping[str, str] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        """Validate the fields and freeze the mutable ones.

        Raises:
            TypeError: if a field has the wrong type.
            ValueError: if a field has an invalid value (``np`` or
                ``threads`` < 1, empty / negative / duplicated ``cpus``,
                blank ``launcher`` or ``launcher`` without ``np``, malformed
                ``env`` key or an ``env`` key owned by a dedicated field).
        """
        if self.np is not None:
            object.__setattr__(self, "np", _check_positive_int("np", self.np))
        if self.threads is not None:
            object.__setattr__(self, "threads", _check_positive_int("threads", self.threads))
        if self.cpus is not None:
            self._validate_cpus()
        if self.launcher is not None:
            if not isinstance(self.launcher, str):
                raise TypeError(
                    f"LaunchSpec.launcher must be a str, got {type(self.launcher).__name__}"
                )
            if not self.launcher.strip():
                raise ValueError("LaunchSpec.launcher must not be blank")
            if self.np is None:
                raise ValueError(
                    "LaunchSpec.launcher requires np to be set explicitly: the launcher "
                    "string must start exactly np ranks (what executors reserve CPUs for)"
                )
        self._validate_env()

    def _validate_cpus(self) -> None:
        if isinstance(self.cpus, (str, bytes)):
            raise TypeError("LaunchSpec.cpus must be a sequence of ints, not a string")
        cpus = tuple(as_int(c, "LaunchSpec.cpus entries") for c in self.cpus)  # type: ignore[union-attr]
        if not cpus:
            raise ValueError("LaunchSpec.cpus must not be empty (use None for no pinning)")
        for cpu in cpus:
            if cpu < 0:
                raise ValueError(f"LaunchSpec.cpus entries must be >= 0, got {cpu}")
        if len(set(cpus)) != len(cpus):
            raise ValueError(f"LaunchSpec.cpus entries must be unique, got {cpus}")
        object.__setattr__(self, "cpus", cpus)

    def _validate_env(self) -> None:
        if not isinstance(self.env, Mapping):
            raise TypeError(f"LaunchSpec.env must be a mapping, got {type(self.env).__name__}")
        frozen: dict[str, str] = {}
        for key, value in self.env.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError(f"LaunchSpec.env entries must be str -> str, got {key!r}")
            if not key or "=" in key or "\0" in key or "\0" in value:
                raise ValueError(f"Invalid environment variable name or value: {key!r}")
            if key in _FIELD_OF_ENV:
                raise ValueError(
                    f"LaunchSpec.env must not set {key}: use LaunchSpec.{_FIELD_OF_ENV[key]}"
                )
            frozen[key] = value
        object.__setattr__(self, "env", MappingProxyType(frozen))

    def effective_np(self, base_env: Mapping[str, str]) -> int:
        """Number of MPI ranks this run will use.

        Args:
            base_env: Environment providing the defaults (usually
                :data:`os.environ`).

        Returns:
            ``np`` if set, else the ``MPXLITE_GETDP_NP`` of ``base_env``
            (parsed by :func:`parse_count`, the wrapper's rule), else 1.

        Raises:
            ValueError: if ``np`` is unset and the ``base_env`` value is not
                a valid integer >= 1.
        """
        if self.np is not None:
            return self.np
        value = parse_count(ENV_NP, base_env.get(ENV_NP))
        return 1 if value is None else value

    def effective_threads(self, base_env: Mapping[str, str]) -> int:
        """OpenMP threads per rank this run will use.

        Args:
            base_env: Environment providing the defaults.

        Returns:
            ``MPXLITE_GETDP_THREADS`` as handed to the subprocess by
            :meth:`to_env` (including its default of 1), else an inherited
            ``OMP_NUM_THREADS`` (its first, outermost-level entry when it is
            an OpenMP comma list such as ``"4,2"``), else 1.

        Raises:
            ValueError: if ``MPXLITE_GETDP_THREADS`` is set but invalid
                (:func:`parse_count`), or ``OMP_NUM_THREADS`` is set but its
                first entry is not an integer >= 1.
        """
        env = self.to_env(base_env)
        threads = parse_count(ENV_THREADS, env.get(ENV_THREADS))
        if threads is not None:
            return threads
        raw = env.get("OMP_NUM_THREADS", "").strip()
        if not raw:
            return 1
        try:
            value = int(raw.split(",", 1)[0])
        except ValueError:
            value = 0
        if value < 1:
            raise ValueError(f"OMP_NUM_THREADS={raw!r} must be an integer >= 1")
        return value

    def width(self, base_env: Mapping[str, str]) -> int:
        """Number of logical CPUs this run occupies.

        Args:
            base_env: Environment providing the defaults.

        Returns:
            ``effective_np(base_env) * effective_threads(base_env)``.
        """
        return self.effective_np(base_env) * self.effective_threads(base_env)

    def to_env(self, base_env: Mapping[str, str]) -> dict[str, str]:
        """Build the subprocess environment for this run.

        Args:
            base_env: Environment to start from (usually :data:`os.environ`);
                it is copied, never modified.

        Returns:
            A new dict: ``base_env`` plus ``MPXLITE_GETDP_NP``,
            ``MPXLITE_GETDP_THREADS``, ``MPXLITE_GETDP_CPUS`` (comma-joined)
            and ``MPXLITE_GETDP_LAUNCHER`` for every non-``None`` field, then
            ``env`` applied last. A variable counts as set when non-empty
            (the wrapper's rule). For a multi-rank or CPU-pinned run whose
            thread count is set nowhere (no ``threads``, no
            ``MPXLITE_GETDP_THREADS``, no ``OMP_NUM_THREADS``),
            ``MPXLITE_GETDP_THREADS=1`` is added: every rank would otherwise
            start a full OpenMP pool and oversubscribe the machine or its CPU
            block (measured ~15x slowdown at 4 ranks on 16 CPUs).
        """
        out = dict(base_env)
        if self.np is not None:
            out[ENV_NP] = str(self.np)
        if self.threads is not None:
            out[ENV_THREADS] = str(self.threads)
        if self.cpus is not None:
            out[ENV_CPUS] = ",".join(str(c) for c in self.cpus)
        if self.launcher is not None:
            out[ENV_LAUNCHER] = self.launcher
        out.update(self.env)
        pinned = bool(out.get(ENV_CPUS))
        if (pinned or _is_multi_rank(out.get(ENV_NP))) and not (
            out.get(ENV_THREADS) or out.get("OMP_NUM_THREADS")
        ):
            out[ENV_THREADS] = "1"
        return out


def _is_multi_rank(raw_np: str | None) -> bool:
    """Whether an ``MPXLITE_GETDP_NP`` value requests more than one rank.

    An invalid value counts as single-rank here; it is reported by
    :meth:`LaunchSpec.effective_np` and by the wrapper.
    """
    try:
        value = parse_count(ENV_NP, raw_np)
    except ValueError:
        return False
    return value is not None and value > 1
