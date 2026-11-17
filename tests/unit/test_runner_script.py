# Copyright (c) 2025 MPhysX
# SPDX-License-Identifier: MIT

"""Unit tests for the bundled ``tools/getdp_runner.sh`` wrapper.

The wrapper is executed for real with ``bash``, but every tool it may exec
(``getdp``, ``mpirun``, ``srun``, ``taskset``) is a stub living in a private
``bin`` directory that is the ONLY entry on ``PATH``. Each stub records its
argv (NUL-separated) and ``OMP_NUM_THREADS`` and exits 0, so the tests assert
the exact command line the wrapper exec-ed. Only ``cat`` (used by the
wrapper's heredoc error messages) is symlinked from the real system.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

import mpxlite

SCRIPT = Path(mpxlite.__file__).resolve().parent / "tools" / "getdp_runner.sh"
BASH = shutil.which("bash") or "/bin/bash"
ARGS = ("case.pro", "-msh", "case.msh", "-solve", "R")

_STUB = """#!{bash}
printf '%s\\0' "{name}" "$@" > "$STUB_LOG"
printf '%s' "${{OMP_NUM_THREADS-<unset>}}" > "$STUB_LOG.omp"
printf '%s' "${{LD_LIBRARY_PATH-<unset>}}" > "$STUB_LOG.ld"
exit 0
"""


@dataclass
class Sandbox:
    bin_dir: Path
    log: Path

    @property
    def getdp(self) -> str:
        return str(self.bin_dir / "getdp")

    def remove(self, tool: str) -> None:
        (self.bin_dir / tool).unlink()


@dataclass
class Outcome:
    returncode: int
    argv: list[str] | None
    omp: str | None
    stderr: str


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("getdp", "mpirun", "srun", "taskset"):
        stub = bin_dir / name
        stub.write_text(_STUB.format(bash=BASH, name=name), encoding="utf-8")
        stub.chmod(0o755)
    cat = shutil.which("cat")
    assert cat is not None
    (bin_dir / "cat").symlink_to(cat)
    return Sandbox(bin_dir=bin_dir, log=tmp_path / "stub.log")


def _run(
    sb: Sandbox, env: dict[str, str] | None = None, binary: bool = True, cwd: Path | None = None
) -> Outcome:
    full_env = {"PATH": str(sb.bin_dir), "STUB_LOG": str(sb.log)}
    if binary:
        full_env["MPXLITE_GETDP_BINARY"] = sb.getdp
    full_env.update(env or {})
    proc = subprocess.run(
        [BASH, str(SCRIPT), *ARGS],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        cwd=cwd,
    )
    argv = sb.log.read_text(encoding="utf-8").split("\0")[:-1] if sb.log.exists() else None
    omp_file = Path(f"{sb.log}.omp")
    omp = omp_file.read_text(encoding="utf-8") if omp_file.exists() else None
    return Outcome(proc.returncode, argv, omp, proc.stderr)


# ----- script sanity -------------------------------------------------------


def test_script_is_executable_and_parses() -> None:
    assert os.access(SCRIPT, os.X_OK)
    subprocess.run([BASH, "-n", str(SCRIPT)], check=True)


# ----- sequential ----------------------------------------------------------


@pytest.mark.parametrize("np_env", [None, "1"])
def test_np_unset_or_one_execs_getdp_directly(sandbox: Sandbox, np_env: str | None) -> None:
    env = {} if np_env is None else {"MPXLITE_GETDP_NP": np_env}
    out = _run(sandbox, env)
    assert out.returncode == 0, out.stderr
    assert out.argv == ["getdp", *ARGS]
    assert out.omp == "<unset>"
    assert out.stderr == ""


def test_getdp_on_path_is_not_used(sandbox: Sandbox) -> None:
    """The engine is named by MPXLITE_GETDP_BINARY; the getdp on PATH is not run."""
    out = _run(sandbox, binary=False)
    assert out.returncode == 127
    assert out.argv is None
    assert "MPXLITE_GETDP_BINARY=''" in out.stderr


def test_np_one_with_cpus_uses_taskset(sandbox: Sandbox) -> None:
    out = _run(sandbox, {"MPXLITE_GETDP_CPUS": "4,5,6,7"})
    assert out.returncode == 0, out.stderr
    assert out.argv == ["taskset", "-c", "4,5,6,7", sandbox.getdp, *ARGS]


def test_np_one_ignores_scheduler(sandbox: Sandbox) -> None:
    out = _run(sandbox, {"SLURM_JOB_ID": "42", "PBS_NODEFILE": "/x"})
    assert out.returncode == 0, out.stderr
    assert out.argv == ["getdp", *ARGS]


@pytest.mark.parametrize("np_env", [None, "1"])
def test_np_one_uses_explicit_launcher(sandbox: Sandbox, np_env: str | None) -> None:
    env = {"MPXLITE_GETDP_LAUNCHER": "srun -n 1", "SLURM_JOB_ID": "42"}
    if np_env is not None:
        env["MPXLITE_GETDP_NP"] = np_env
    out = _run(sandbox, env)
    assert out.returncode == 0, out.stderr
    assert out.argv == ["srun", "-n", "1", sandbox.getdp, *ARGS]


# ----- local MPI -----------------------------------------------------------


def test_np_gt1_local_mpirun(sandbox: Sandbox) -> None:
    out = _run(sandbox, {"MPXLITE_GETDP_NP": "4"})
    assert out.returncode == 0, out.stderr
    assert out.argv == ["mpirun", "-np", "4", sandbox.getdp, *ARGS]


def test_np_gt1_local_mpirun_with_cpus(sandbox: Sandbox) -> None:
    out = _run(sandbox, {"MPXLITE_GETDP_NP": "4", "MPXLITE_GETDP_CPUS": "0-3"})
    assert out.returncode == 0, out.stderr
    assert out.argv == [
        "taskset",
        "-c",
        "0-3",
        "mpirun",
        "--bind-to",
        "none",
        "-np",
        "4",
        sandbox.getdp,
        *ARGS,
    ]
    assert out.stderr == ""


def test_np_gt1_without_mpirun_exits_127(sandbox: Sandbox) -> None:
    sandbox.remove("mpirun")
    out = _run(sandbox, {"MPXLITE_GETDP_NP": "2"})
    assert out.returncode == 127
    assert out.argv is None
    assert "no MPI launcher detected" in out.stderr


def _add_setpriv(sb: Sandbox, probe_ok: bool = True) -> None:
    """A ``setpriv`` stub: the wrapper's probe (``... true``) succeeds or fails."""
    stub = sb.bin_dir / "setpriv"
    stub.write_text(
        f'#!{BASH}\n[ "${{@: -1}}" = true ] && exit {0 if probe_ok else 1}\nexit 0\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)


_RANK = ["setpriv", "--pdeathsig", "KILL", "--"]


@pytest.mark.parametrize("cpus", [None, "0-3"])
def test_local_mpirun_ranks_get_parent_death_signal(sandbox: Sandbox, cpus: str | None) -> None:
    """Ranks die with mpirun (Open MPI leaves them running when mpirun is SIGKILLed)."""
    _add_setpriv(sandbox)
    env = {"MPXLITE_GETDP_NP": "4", **({"MPXLITE_GETDP_CPUS": cpus} if cpus else {})}
    out = _run(sandbox, env)
    assert out.returncode == 0, out.stderr
    assert out.argv is not None
    i = out.argv.index("mpirun")
    tail = ["-np", "4", *_RANK, sandbox.getdp, *ARGS]
    assert out.argv[i + 1 :] == (["--bind-to", "none", *tail] if cpus else tail)


def test_local_mpirun_without_usable_setpriv(sandbox: Sandbox) -> None:
    _add_setpriv(sandbox, probe_ok=False)  # e.g. util-linux < 2.33: no --pdeathsig
    out = _run(sandbox, {"MPXLITE_GETDP_NP": "2"})
    assert out.returncode == 0, out.stderr
    assert out.argv == ["mpirun", "-np", "2", sandbox.getdp, *ARGS]


@pytest.mark.parametrize(
    ("launcher", "ranked"),
    [
        ("mpirun --bind-to none -np 2", True),
        ("/opt/x/bin/mpiexec -n 2", True),
        ("srun -n 2", False),
    ],
)
def test_explicit_launcher_parent_death_signal(
    sandbox: Sandbox, launcher: str, ranked: bool
) -> None:
    _add_setpriv(sandbox)
    if "/opt/x/bin/" in launcher:
        launcher = launcher.replace("/opt/x/bin/", f"{sandbox.bin_dir}/")
        (sandbox.bin_dir / "mpiexec").symlink_to(sandbox.bin_dir / "mpirun")
    out = _run(sandbox, {"MPXLITE_GETDP_NP": "2", "MPXLITE_GETDP_LAUNCHER": launcher})
    assert out.returncode == 0, out.stderr
    assert out.argv is not None
    assert out.argv[-len(ARGS) - 1 - 4 * ranked : -len(ARGS) - 1] == (_RANK if ranked else [])


@pytest.mark.parametrize("inherited", [None, "", "/a:/b"])
def test_petsc_lib_dir_prepended_without_empty_entry(
    sandbox: Sandbox, inherited: str | None
) -> None:
    """An empty LD_LIBRARY_PATH entry would mean the current directory (review nit)."""
    env = {"MPXLITE_PETSC_LIB_DIR": "/opt/petsc/lib"}
    if inherited is not None:
        env["LD_LIBRARY_PATH"] = inherited
    out = _run(sandbox, env)
    assert out.returncode == 0, out.stderr
    ld = Path(f"{sandbox.log}.ld").read_text(encoding="utf-8")
    assert ld == ("/opt/petsc/lib:/a:/b" if inherited else "/opt/petsc/lib")


# ----- explicit launcher ---------------------------------------------------


def test_explicit_launcher(sandbox: Sandbox) -> None:
    out = _run(
        sandbox,
        {"MPXLITE_GETDP_NP": "2", "MPXLITE_GETDP_LAUNCHER": "srun -n 2 --cpu-bind=verbose"},
    )
    assert out.returncode == 0, out.stderr
    assert out.argv == ["srun", "-n", "2", "--cpu-bind=verbose", sandbox.getdp, *ARGS]


def test_explicit_launcher_with_cpus(sandbox: Sandbox) -> None:
    out = _run(
        sandbox,
        {
            "MPXLITE_GETDP_NP": "2",
            "MPXLITE_GETDP_LAUNCHER": "mpirun --bind-to none -np 2",
            "MPXLITE_GETDP_CPUS": "8,9",
        },
    )
    assert out.returncode == 0, out.stderr
    assert out.argv == [
        "taskset",
        "-c",
        "8,9",
        "mpirun",
        "--bind-to",
        "none",
        "-np",
        "2",
        sandbox.getdp,
        *ARGS,
    ]
    assert out.stderr == ""


@pytest.mark.parametrize("np_env", ["1", "2"])
def test_explicit_launcher_with_cpus_without_bind_to_warns(sandbox: Sandbox, np_env: str) -> None:
    out = _run(
        sandbox,
        {
            "MPXLITE_GETDP_NP": np_env,
            "MPXLITE_GETDP_LAUNCHER": "mpirun -np 2",
            "MPXLITE_GETDP_CPUS": "8,9",
        },
    )
    assert out.returncode == 0, out.stderr
    # The launcher is not rewritten.
    assert out.argv == ["taskset", "-c", "8,9", "mpirun", "-np", "2", sandbox.getdp, *ARGS]
    lines = out.stderr.strip().splitlines()
    assert len(lines) == 1
    assert "[getdp_runner] warning" in lines[0]
    assert "--bind-to none" in lines[0]


@pytest.mark.parametrize("blank", [" ", "   ", "\t \t"])
@pytest.mark.parametrize("np_env", ["1", "4"])
def test_blank_explicit_launcher_exits_2(sandbox: Sandbox, blank: str, np_env: str) -> None:
    """A whitespace-only launcher is an error, not a silent single-process run (F-5)."""
    out = _run(sandbox, {"MPXLITE_GETDP_NP": np_env, "MPXLITE_GETDP_LAUNCHER": blank})
    assert out.returncode == 2
    assert out.argv is None
    assert "[getdp_runner] MPXLITE_GETDP_LAUNCHER=" in out.stderr
    assert "blank" in out.stderr


def test_explicit_launcher_not_glob_expanded(sandbox: Sandbox, tmp_path: Path) -> None:
    """Glob characters of the launcher reach it verbatim, whatever the run dir holds (F-6)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in ("x1", "OMP_A", "h1"):
        (run_dir / name).write_text("", encoding="utf-8")
    launcher = "mpirun -x x? -x OMP_* --hostfile h[0-9] -np 2"
    out = _run(sandbox, {"MPXLITE_GETDP_NP": "2", "MPXLITE_GETDP_LAUNCHER": launcher}, cwd=run_dir)
    assert out.returncode == 0, out.stderr
    assert out.argv == [*launcher.split(), sandbox.getdp, *ARGS]


def test_explicit_launcher_without_cpus_does_not_warn(sandbox: Sandbox) -> None:
    out = _run(sandbox, {"MPXLITE_GETDP_NP": "2", "MPXLITE_GETDP_LAUNCHER": "mpirun -np 2"})
    assert out.returncode == 0, out.stderr
    assert out.stderr == ""


# ----- schedulers ----------------------------------------------------------


def test_slurm_uses_srun(sandbox: Sandbox) -> None:
    out = _run(sandbox, {"MPXLITE_GETDP_NP": "8", "SLURM_JOB_ID": "123"})
    assert out.returncode == 0, out.stderr
    assert out.argv == ["srun", "-n", "8", sandbox.getdp, *ARGS]
    assert out.stderr == ""


def test_slurm_ignores_cpus_with_one_warning(sandbox: Sandbox) -> None:
    out = _run(
        sandbox, {"MPXLITE_GETDP_NP": "8", "SLURM_JOB_ID": "123", "MPXLITE_GETDP_CPUS": "0-7"}
    )
    assert out.returncode == 0, out.stderr
    assert out.argv == ["srun", "-n", "8", sandbox.getdp, *ARGS]
    lines = out.stderr.strip().splitlines()
    assert len(lines) == 1
    assert "[getdp_runner] warning" in lines[0]
    assert "MPXLITE_GETDP_CPUS=0-7 ignored under SLURM" in lines[0]


def test_slurm_np_one_still_pins_cpus(sandbox: Sandbox) -> None:
    """Documented: under SLURM/PBS a one-rank run is still pinned to MPXLITE_GETDP_CPUS."""
    out = _run(sandbox, {"SLURM_JOB_ID": "123", "PBS_NODEFILE": "/x", "MPXLITE_GETDP_CPUS": "2,3"})
    assert out.returncode == 0, out.stderr
    assert out.argv == ["taskset", "-c", "2,3", sandbox.getdp, *ARGS]
    assert out.stderr == ""


def test_slurm_without_srun_falls_back_to_mpirun(sandbox: Sandbox) -> None:
    sandbox.remove("srun")
    out = _run(sandbox, {"MPXLITE_GETDP_NP": "2", "SLURM_JOB_ID": "123"})
    assert out.returncode == 0, out.stderr
    assert out.argv == ["mpirun", "-np", "2", sandbox.getdp, *ARGS]


def test_pbs_uses_hostfile(sandbox: Sandbox) -> None:
    out = _run(sandbox, {"MPXLITE_GETDP_NP": "16", "PBS_NODEFILE": "/var/pbs/nodes"})
    assert out.returncode == 0, out.stderr
    assert out.argv == [
        "mpirun",
        "--hostfile",
        "/var/pbs/nodes",
        "-np",
        "16",
        sandbox.getdp,
        *ARGS,
    ]
    assert out.stderr == ""


def test_pbs_ignores_cpus_with_one_warning(sandbox: Sandbox) -> None:
    sandbox.remove("taskset")  # not needed on the scheduler path
    out = _run(
        sandbox,
        {"MPXLITE_GETDP_NP": "4", "PBS_NODEFILE": "/var/pbs/nodes", "MPXLITE_GETDP_CPUS": "0-3"},
    )
    assert out.returncode == 0, out.stderr
    assert out.argv == ["mpirun", "--hostfile", "/var/pbs/nodes", "-np", "4", sandbox.getdp, *ARGS]
    lines = out.stderr.strip().splitlines()
    assert len(lines) == 1
    assert "ignored under PBS" in lines[0]


# ----- validation ----------------------------------------------------------


def test_empty_np_means_default(sandbox: Sandbox) -> None:
    # Empty == unset -> default 1 (shell ${VAR:-1} semantics).
    out = _run(sandbox, {"MPXLITE_GETDP_NP": ""})
    assert out.returncode == 0, out.stderr
    assert out.argv == ["getdp", *ARGS]


# Keep in sync with _BAD_COUNTS in test_launch.py (Python applies the same rule).
_BAD_COUNTS = ["0", "-1", "two", "2.5", " 2", "2 ", " ", "01", "+2", "2_0", "1e3", "1234567890"]


@pytest.mark.parametrize("bad", [*_BAD_COUNTS, "4; rm -rf /"])
def test_bad_np_exits_2(sandbox: Sandbox, bad: str) -> None:
    out = _run(sandbox, {"MPXLITE_GETDP_NP": bad})
    assert out.returncode == 2
    assert out.argv is None
    assert "[getdp_runner] MPXLITE_GETDP_NP=" in out.stderr
    assert "integer >= 1" in out.stderr


@pytest.mark.parametrize("bad", _BAD_COUNTS)
def test_bad_threads_exits_2(sandbox: Sandbox, bad: str) -> None:
    out = _run(sandbox, {"MPXLITE_GETDP_THREADS": bad})
    assert out.returncode == 2
    assert out.argv is None
    assert "[getdp_runner] MPXLITE_GETDP_THREADS=" in out.stderr


@pytest.mark.parametrize("np_env", ["1", "4"])
def test_cpus_without_taskset_exits_127(sandbox: Sandbox, np_env: str) -> None:
    sandbox.remove("taskset")
    out = _run(sandbox, {"MPXLITE_GETDP_NP": np_env, "MPXLITE_GETDP_CPUS": "0-3"})
    assert out.returncode == 127
    assert out.argv is None
    assert "taskset is not on PATH" in out.stderr


def test_missing_getdp_exits_127(sandbox: Sandbox) -> None:
    sandbox.remove("getdp")
    out = _run(sandbox)
    assert out.returncode == 127
    assert "does not name an executable" in out.stderr


# ----- threads -------------------------------------------------------------


@pytest.mark.parametrize("np_env", ["1", "4"])
def test_threads_exported_as_omp_num_threads(sandbox: Sandbox, np_env: str) -> None:
    out = _run(
        sandbox,
        {"MPXLITE_GETDP_NP": np_env, "MPXLITE_GETDP_THREADS": "3", "OMP_NUM_THREADS": "16"},
    )
    assert out.returncode == 0, out.stderr
    assert out.omp == "3"


def test_inherited_omp_num_threads_kept_without_threads(sandbox: Sandbox) -> None:
    out = _run(sandbox, {"OMP_NUM_THREADS": "5"})
    assert out.returncode == 0, out.stderr
    assert out.omp == "5"
