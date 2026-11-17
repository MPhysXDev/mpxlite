# Copyright (c) 2025 MPhysX
# SPDX-License-Identifier: MIT

"""End-to-end tests with the real getdp binary (``integration`` marker).

All runs share the committed, read-only ``data/`` case directory and write
only under ``tmp_path / "runs"`` (checked by the ``make_solver`` fixture).
"""

from __future__ import annotations

import logging
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from mpxlite import (
    WRAPPER_PATH,
    GetDPError,
    GetDPTimeoutError,
    LaunchSpec,
    LocalExecutor,
    RunRequest,
    SolverLite,
    SolverResult,
    WorkflowData,
    find_engine,
    parse_real_table,
)
from mpxlite.base import PetscOptionValue, _session_pids
from tests.integration.rod import (
    DATA_DIR,
    MPI_GETDP_ENV,
    build_options,
    expected_integral,
    expected_mid,
    mpi_getdp,
    mpi_getdp_available,  # noqa: F401 (string-form skipif condition)
)

pytestmark = pytest.mark.integration

SolverFactory = Callable[..., SolverLite]

# String-form conditions: evaluated at test setup, so collecting (or deselecting)
# this module never runs getdp.
_requires_mpi = pytest.mark.skipif(
    "not mpi_getdp_available()",
    reason=f"the MPI getdp is missing or built without MPI (set {MPI_GETDP_ENV})",
)


def _serial_getdp_has_petsc() -> bool:
    try:
        return "PETSc" in build_options(find_engine())
    except FileNotFoundError:
        return False


_requires_petsc = pytest.mark.skipif(
    "not _serial_getdp_has_petsc()", reason="the serial getdp is not built with PETSc"
)
# Memory estimate (= hard cap where available) of one rod run, serial / 2-rank MPI.
_RUN_MEMORY = 256 * 1024**2
_MPI_RUN_MEMORY = 512 * 1024**2
# PETSc's message when the requested factorization package is not in the build.
_NO_SOLVER_PACKAGE = "for possible LU and Cholesky solvers"


def _mid(result: SolverResult) -> float:
    return float(parse_real_table(result.path("u_mid.dat"))[-1, -1])


def _integral(result: SolverResult) -> float:
    return float(parse_real_table(result.path("u_int.dat"))[-1, -1])


def _cpu_list(text: str) -> set[int]:
    """Parse a kernel CPU list such as ``"0-2,5"``."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


# ----- (a) single run ------------------------------------------------------


def test_single_run_outputs_only_in_run_dir(make_solver: SolverFactory, tmp_path: Path) -> None:
    solver = make_solver()
    result = solver.run("mid", run_id="single")
    runs = tmp_path / "runs"
    assert result.work_dir == runs / "single"
    assert [p.name for p in runs.iterdir()] == ["single"]
    names = {p.name for p in result.work_dir.iterdir()}
    assert {"getdp.log", "mpxlite.pro", "workflow.pro", "u_mid.dat"} <= names
    assert "u_int.dat" not in names
    assert _mid(result) == pytest.approx(expected_mid(2.0, 8.0), rel=1e-9)
    assert not (DATA_DIR / "runs").exists()


# ----- (b) several post-operations ----------------------------------------


def test_multiple_postops_one_run(make_solver: SolverFactory) -> None:
    result = make_solver(v=1.0, s=-4.0).run(["mid", "integral"])
    assert result.cmd[-3:] == ("-pos", "mid", "integral")
    assert _mid(result) == pytest.approx(expected_mid(1.0, -4.0), rel=1e-9)
    assert _integral(result) == pytest.approx(expected_integral(1.0, -4.0), rel=1e-9)


# ----- (c) workflow overrides ---------------------------------------------


def test_workflow_overrides_distinct_runs(make_solver: SolverFactory) -> None:
    solver = make_solver(v=2.0, s=8.0)
    base = solver.run("mid", run_id="base")
    over = solver.run("mid", run_id="over", workflow={"V": 10.0})
    assert base.work_dir != over.work_dir
    assert _mid(base) == pytest.approx(expected_mid(2.0, 8.0), rel=1e-9)
    assert _mid(over) == pytest.approx(expected_mid(10.0, 8.0), rel=1e-9)
    assert dict(solver.workflow) == {"V": 2.0, "S": 8.0}


# ----- (d) error path -------------------------------------------------------


def test_bad_postop_raises_with_log_tail(make_solver: SolverFactory, tmp_path: Path) -> None:
    with pytest.raises(GetDPError) as info:
        make_solver().run("no_such_postop", run_id="bad")
    err = info.value
    assert err.returncode == 1
    assert err.run_dir == tmp_path / "runs" / "bad"
    assert err.log_path is not None and err.log_path.is_file()
    assert "Unknown PostOperation" in err.log_tail
    assert "Unknown PostOperation" in str(err)


# ----- (e) concurrent runs ---------------------------------------------------


def test_local_executor_concurrent_overrides(make_solver: SolverFactory) -> None:
    solver = make_solver(verbosity=1)
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    reqs = [
        RunRequest(solver, ["mid", "integral"], workflow={"V": v}, run_id=f"v{i}")
        for i, v in enumerate(values)
    ]
    cpus = sorted(os.sched_getaffinity(0))[:4]
    outcomes = LocalExecutor(cpus=cpus, memory_per_run=_RUN_MEMORY).submit(reqs)
    assert all(o.ok for o in outcomes), [str(o.error) for o in outcomes]
    for outcome, v in zip(outcomes, values, strict=True):
        assert outcome.result is not None
        assert outcome.result.run_id == outcome.request.run_id
        assert _mid(outcome.result) == pytest.approx(expected_mid(v, 8.0), rel=1e-9)
        assert _integral(outcome.result) == pytest.approx(expected_integral(v, 8.0), rel=1e-9)
    dirs = {o.result.work_dir for o in outcomes if o.result}
    assert len(dirs) == len(values)


# ----- (f) MPI ---------------------------------------------------------------


@_requires_mpi
def test_two_rank_mpi_run(make_solver: SolverFactory) -> None:
    launch = LaunchSpec(np=2, env={"MPXLITE_GETDP_BINARY": str(mpi_getdp())})
    result = make_solver(v=3.0, s=2.0).run(["mid", "integral"], launch=launch)
    log = result.log_path.read_text(encoding="utf-8", errors="replace")
    assert "[rank   1]" in log
    assert _mid(result) == pytest.approx(expected_mid(3.0, 2.0), rel=1e-9)
    assert _integral(result) == pytest.approx(expected_integral(3.0, 2.0), rel=1e-9)


# ----- (g) CPU pinning -------------------------------------------------------


_wrapper_pins = "MPXLITE_GETDP_CPUS" in WRAPPER_PATH.read_text(encoding="utf-8")


@_requires_mpi
@pytest.mark.skipif(not _wrapper_pins, reason="getdp_runner.sh lacks MPXLITE_GETDP_CPUS")
@pytest.mark.skipif(len(os.sched_getaffinity(0)) < 4, reason="needs >= 4 CPUs")
def test_concurrent_mpi_runs_get_disjoint_masks(make_solver: SolverFactory, tmp_path: Path) -> None:
    """Every rank records its own affinity mask before exec-ing getdp.

    ``MPXLITE_GETDP_BINARY`` points to a shim that ``mpirun`` starts once per
    rank: it writes ``Cpus_allowed_list`` of /proc/self/status into the run
    directory (its cwd), sleeps a little so both runs overlap, then exec-s
    the MPI getdp (exec keeps the mask). No sampling, no race.
    """
    shim = tmp_path / "getdp_shim.sh"
    shim.write_text(
        "#!/bin/sh\n"
        'grep Cpus_allowed_list /proc/self/status > "affinity.$$.txt"\n'
        "sleep 0.5\n"
        f'exec {mpi_getdp()} "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    launch = LaunchSpec(np=2, env={"MPXLITE_GETDP_BINARY": str(shim)})
    solver = make_solver(verbosity=1)
    reqs = [
        RunRequest(solver, "mid", workflow={"V": v}, run_id=f"pin{i}", launch=launch)
        for i, v in enumerate((1.0, 2.0))
    ]
    cpus = sorted(os.sched_getaffinity(0))[:4]
    outcomes = LocalExecutor(cpus=cpus, memory_per_run=_MPI_RUN_MEMORY).submit(reqs)
    assert all(o.ok for o in outcomes), [str(o.error) for o in outcomes]

    masks: list[set[int]] = []
    for outcome, v in zip(outcomes, (1.0, 2.0), strict=True):
        assert outcome.result is not None and outcome.launch is not None
        block = set(outcome.launch.cpus or ())
        assert len(block) == 2
        files = sorted(outcome.result.work_dir.glob("affinity.*.txt"))
        assert len(files) == 2, files  # one per rank
        for f in files:
            assert _cpu_list(f.read_text().split(":", 1)[1]) == block
        assert _mid(outcome.result) == pytest.approx(expected_mid(v, 8.0), rel=1e-9)
        masks.append(block)
    assert masks[0].isdisjoint(masks[1])


# ----- (h) cleanup of every process of a run ---------------------------------

# Per-process shim: records its session id (field 6 of /proc/self/stat) in the
# run directory (its cwd), then sleeps instead of running getdp.
_SID_LINE = 'read -r -a st < /proc/self/stat; echo "${st[5]}" > "sid.$$"\n'


def _write_shim(path: Path, body: str) -> Path:
    path.write_text("#!/bin/bash\n" + _SID_LINE + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _sids(run_dir: Path) -> set[int]:
    return {int(f.read_text()) for f in run_dir.glob("sid.*")}


@pytest.fixture
def reap_sessions() -> Iterator[list[int]]:
    """Session ids to SIGKILL at teardown, should a test leave survivors."""
    sids: list[int] = []
    yield sids
    for sid in sids:
        for pid in _session_pids(sid) or []:
            os.kill(pid, signal.SIGKILL)


def _wait_for_files(directory: Path, pattern: str, count: int, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while len(list(directory.glob(pattern))) < count and time.monotonic() < deadline:
        time.sleep(0.05)


@_requires_mpi
def test_mpi_timeout_kills_whole_session(
    make_solver: SolverFactory, tmp_path: Path, reap_sessions: list[int]
) -> None:
    """Open MPI puts each rank in its own process group; none may survive a timeout.

    Each rank also starts a background job in yet another process group
    (``set -m``) that neither ``mpirun`` nor a process-group kill reaches:
    only the session-wide cleanup does.
    """
    shim = _write_shim(
        tmp_path / "sleep_shim.sh",
        "set -m\n(exec sleep 300) &\nexec sleep 300\n",
    )
    launch = LaunchSpec(np=2, env={"MPXLITE_GETDP_BINARY": str(shim)})
    solver = make_solver(verbosity=0)
    run_dir = solver.runs_dir / "hang"
    with pytest.raises(GetDPTimeoutError) as info:
        solver.run("mid", run_id="hang", launch=launch, timeout=3.0)
    assert info.value.run_dir == run_dir
    assert len(list(run_dir.glob("sid.*"))) == 2, info.value.log_tail  # one per rank
    sids = _sids(run_dir)
    reap_sessions.extend(sids)
    assert len(sids) == 1
    assert _session_pids(sids.pop()) == []


def test_ctrl_c_under_local_executor_stops_runs(
    make_solver: SolverFactory, tmp_path: Path, reap_sessions: list[int]
) -> None:
    """SIGINT during submit(): in-flight runs are killed, queued runs never start."""
    shim = _write_shim(tmp_path / "sleep_shim.sh", "exec sleep 300\n")
    launch = LaunchSpec(env={"MPXLITE_GETDP_BINARY": str(shim)})
    solver = make_solver(verbosity=0)
    reqs = [RunRequest(solver, "mid", run_id=f"c{i}", launch=launch) for i in range(4)]
    cpus = sorted(os.sched_getaffinity(0))[:2]

    def interrupt_when_started() -> None:
        _wait_for_files(solver.runs_dir, "c*/sid.*", 2)
        os.kill(os.getpid(), signal.SIGINT)

    timer = threading.Thread(target=interrupt_when_started, daemon=True)
    timer.start()
    t0 = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            LocalExecutor(cpus=cpus, memory_per_run=_RUN_MEMORY).submit(reqs)
    finally:
        timer.join(timeout=30)
    elapsed = time.monotonic() - t0
    started = sorted(p.parent.name for p in solver.runs_dir.glob("c*/sid.*"))
    assert started == ["c0", "c1"]
    assert not (solver.runs_dir / "c3").exists()
    sids = set().union(*(_sids(solver.runs_dir / r) for r in started))
    reap_sessions.extend(sids)
    assert len(sids) == 2
    assert all(_session_pids(sid) == [] for sid in sids)
    assert elapsed < 20.0  # not the 300 s of the runs


@_requires_mpi
def test_mpi_run_leaves_parent_stdin_alone(tmp_path: Path) -> None:
    """``mpirun`` forwards the parent's stdin to rank 0; the run must not consume it."""
    child = f"""
import sys
from pathlib import Path
from mpxlite import LaunchSpec, SolverLite, WorkflowData
D = Path({str(DATA_DIR)!r})
s = SolverLite(D, "rod", "rod.msh", "Static", resource_files=[D / "rod.formulation.pro"],
               workflow=WorkflowData({{"V": 1.0, "S": 1.0}}), verbosity=0,
               runs_dir={str(tmp_path / "runs")!r})
s.run("mid", launch=LaunchSpec(np=2, env={{"MPXLITE_GETDP_BINARY": {str(mpi_getdp())!r}}}))
print("STDIN=" + repr(sys.stdin.read()))
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    out = subprocess.run(
        [sys.executable, "-c", child],
        input="payload-for-the-parent\n",
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "STDIN='payload-for-the-parent\\n'" in out.stdout


# ----- (i) PETSc options ------------------------------------------------------


def _factorization_package(log_path: Path) -> str:
    """The ``package used to perform factorization`` that ``-ksp_view`` reported."""
    log = log_path.read_text(encoding="utf-8", errors="replace")
    for line in log.splitlines():
        if "package used to perform factorization:" in line:
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"no -ksp_view factorization report in {log_path}")


def _run_with_package(
    solver: SolverLite,
    package: str,
    run_id: str,
    launch: LaunchSpec | None = None,
    extra: dict[str, PetscOptionValue] | None = None,
) -> SolverResult:
    """Run with ``package`` as LU solver (plus ``extra`` options); skip when the build lacks it."""
    opts: dict[str, PetscOptionValue] = {
        "pc_factor_mat_solver_type": package,
        "ksp_view": None,
        **(extra or {}),
    }
    try:
        return solver.run(["mid", "integral"], run_id=run_id, launch=launch, petsc_options=opts)
    except GetDPError as exc:
        if _NO_SOLVER_PACKAGE in exc.log_tail:
            pytest.skip(f"getdp's PETSc has no {package}")
        raise


@_requires_petsc
def test_petsc_options_select_solver(make_solver: SolverFactory) -> None:
    """The template options reach PETSc; a per-run option overrides them."""
    solver = make_solver(
        v=3.0, s=2.0, petsc_options={"pc_factor_mat_solver_type": "petsc", "ksp_view": None}
    )
    result = solver.run(["mid", "integral"], run_id="builtin")
    opts = result.path("petsc.opts")
    assert opts.read_text(encoding="utf-8").splitlines()[1:] == [
        "-ksp_view",
        "-pc_factor_mat_solver_type petsc",
    ]
    assert result.cmd[result.cmd.index("-options_file") + 1] == str(opts)
    assert result.cmd[-3:] == ("-pos", "mid", "integral")
    assert _factorization_package(result.log_path) == "petsc"
    assert _mid(result) == pytest.approx(expected_mid(3.0, 2.0), rel=1e-9)
    assert _integral(result) == pytest.approx(expected_integral(3.0, 2.0), rel=1e-9)

    plain = solver.run("mid", run_id="plain", petsc_options={"ksp_view": False})
    assert "-ksp_view" not in plain.path("petsc.opts").read_text(encoding="utf-8")
    assert "package used to perform factorization" not in plain.log_path.read_text(
        encoding="utf-8", errors="replace"
    )


@_requires_petsc
def test_petsc_option_unknown_solver_fails(make_solver: SolverFactory) -> None:
    """A factorization package PETSc does not know proves the options are applied."""
    with pytest.raises(GetDPError) as info:
        make_solver().run(
            "mid", run_id="nosuch", petsc_options={"pc_factor_mat_solver_type": "no_such_pkg"}
        )
    assert _NO_SOLVER_PACKAGE in info.value.log_tail


# The factorization packages of the engine: MUMPS (the only external direct
# solver) and PETSc's built-in LU (sequential matrices only).
_SERIAL_PACKAGES = ["mumps", "petsc"]
# A MUMPS control parameter and the ``-ksp_view`` line that reports its effect: ICNTL(7) = 5
# requests the METIS ordering (the ordering library of the engine), INFOG(7) reports
# the ordering MUMPS effectively used (automatic choice on the rod system otherwise: AMF, 2).
_MUMPS_METIS = {"mat_mumps_icntl_7": 5}
_MUMPS_METIS_VIEW = "INFOG(7) (ordering option effectively used after analysis): 5"


@_requires_petsc
@pytest.mark.parametrize("package", _SERIAL_PACKAGES)
def test_petsc_options_serial_packages(make_solver: SolverFactory, package: str) -> None:
    result = _run_with_package(make_solver(v=1.0, s=4.0), package, f"serial-{package}")
    assert _factorization_package(result.log_path) == package
    assert _mid(result) == pytest.approx(expected_mid(1.0, 4.0), rel=1e-9)


@_requires_petsc
def test_petsc_options_serial_mumps_control(make_solver: SolverFactory) -> None:
    """A MUMPS ICNTL set through ``petsc_options`` reaches MUMPS (serial, METIS ordering)."""
    solver = make_solver(v=1.0, s=4.0)
    result = _run_with_package(solver, "mumps", "serial-icntl", extra=_MUMPS_METIS)
    assert _factorization_package(result.log_path) == "mumps"
    assert _MUMPS_METIS_VIEW in result.log_path.read_text(encoding="utf-8", errors="replace")
    assert _mid(result) == pytest.approx(expected_mid(1.0, 4.0), rel=1e-9)


@_requires_mpi
def test_petsc_options_mpi_mumps(make_solver: SolverFactory) -> None:
    """MUMPS selected and tuned (METIS ordering) through ``petsc_options`` on 2 ranks."""
    launch = LaunchSpec(np=2, env={"MPXLITE_GETDP_BINARY": str(mpi_getdp())})
    solver = make_solver(v=1.0, s=4.0)
    result = _run_with_package(solver, "mumps", "mpi-mumps", launch, extra=_MUMPS_METIS)
    log = result.log_path.read_text(encoding="utf-8", errors="replace")
    assert "[rank   1]" in log
    assert "2 MPI processes" in log
    assert _factorization_package(result.log_path) == "mumps"
    assert _MUMPS_METIS_VIEW in log
    assert _mid(result) == pytest.approx(expected_mid(1.0, 4.0), rel=1e-9)
    assert _integral(result) == pytest.approx(expected_integral(1.0, 4.0), rel=1e-9)


@_requires_mpi
def test_petsc_options_mpi_package_honoured(make_solver: SolverFactory) -> None:
    """On 2 ranks a non-default package is honoured, not replaced by MUMPS.

    PETSc's built-in LU has no parallel (MPIAIJ) factorization, so selecting it on 2 ranks
    must fail with PETSc's missing-package message.
    """
    launch = LaunchSpec(np=2, env={"MPXLITE_GETDP_BINARY": str(mpi_getdp())})
    with pytest.raises(GetDPError) as info:
        make_solver().run(
            "mid",
            run_id="mpi-builtin",
            launch=launch,
            petsc_options={"pc_factor_mat_solver_type": "petsc"},
        )
    assert _NO_SOLVER_PACKAGE in info.value.log_tail


# ----- (j) GetDP warnings -----------------------------------------------------

_MISSING_CONSTRAINT = "Constraint 'NoSuch' is not provided"


@pytest.fixture
def warning_solver(make_solver: SolverFactory, tmp_path: Path) -> Callable[..., SolverLite]:
    """Rod solvers whose FunctionSpace references an undefined Constraint (a GetDP warning)."""
    text = (DATA_DIR / "rod.formulation.pro").read_text(encoding="utf-8")
    good = "NameOfConstraint Dirichlet; } }"
    assert good in text
    bad = text.replace(
        good,
        "NameOfConstraint Dirichlet; }\n"
        "                 { NameOfCoef un; EntityType NodesOf; NameOfConstraint NoSuch; } }",
    )
    formulation = tmp_path / "warn.formulation.pro"
    formulation.write_text(bad, encoding="utf-8")

    def factory(verbosity: int = 3) -> SolverLite:
        return make_solver(verbosity=verbosity, resource_files=[formulation])

    return factory


def _assert_one_missing_constraint(warnings: tuple[str, ...]) -> None:
    assert len(warnings) == 1, warnings
    (warning,) = warnings
    assert warning.startswith("Warning : '") and "warn.formulation.pro" in warning
    assert warning.endswith(_MISSING_CONSTRAINT)
    assert "\x1b" not in warning and "[rank" not in warning


def test_warnings_captured(
    warning_solver: Callable[..., SolverLite], caplog: pytest.LogCaptureFixture
) -> None:
    solver = warning_solver()
    with caplog.at_level(logging.WARNING, logger="mpxlite"):
        result = solver.run("mid", run_id="warn")
    _assert_one_missing_constraint(result.warnings)
    assert _mid(result) == pytest.approx(expected_mid(2.0, 8.0), rel=1e-9)
    logged = [r.getMessage() for r in caplog.records if r.name == "mpxlite"]
    assert logged == [f"getdp run warn: {result.warnings[0]}"]


def test_no_warnings_on_clean_case(make_solver: SolverFactory) -> None:
    result = make_solver().run("mid")
    assert result.warnings == ()


def test_warnings_need_verbosity_2(warning_solver: Callable[..., SolverLite]) -> None:
    """GetDP prints warnings only at -v >= 2."""
    assert warning_solver(verbosity=1).run("mid").warnings == ()


def test_warnings_on_getdp_error(warning_solver: Callable[..., SolverLite]) -> None:
    with pytest.raises(GetDPError) as info:
        warning_solver().run("no_such_postop", run_id="bad")
    _assert_one_missing_constraint(info.value.warnings)


@_requires_mpi
def test_mpi_warnings_collapse_ranks(warning_solver: Callable[..., SolverLite]) -> None:
    """Under mpirun every rank prints the warning (coloured, rank-tagged): one is kept."""
    launch = LaunchSpec(np=2, env={"MPXLITE_GETDP_BINARY": str(mpi_getdp())})
    result = warning_solver().run("mid", launch=launch)
    log = result.log_path.read_text(encoding="utf-8", errors="replace")
    assert log.count(_MISSING_CONSTRAINT) >= 1
    _assert_one_missing_constraint(result.warnings)


# ----- (k) preprocess(): getdp -pre only ---------------------------------------


def test_preprocess_pre_only(make_solver: SolverFactory, tmp_path: Path) -> None:
    solver = make_solver()
    result = solver.preprocess(run_id="pre")
    assert result.work_dir == tmp_path / "runs" / "pre"
    assert result.cmd[-2:] == ("-pre", "Static")
    assert result.warnings == ()
    names = {p.name for p in result.work_dir.iterdir()}
    assert {"getdp.log", "mpxlite.pro", "workflow.pro", "mpxlite.pre"} <= names
    assert not any(n.endswith((".res", ".dat")) for n in names), names  # nothing solved
    log = result.log_path.read_text(encoding="utf-8", errors="replace")
    assert "Pre-Processing" in log.replace(" ", "")  # GetDP spaces its stage banners out
    assert "Solve" not in log
    # the full run of the same template still works afterwards
    assert _mid(solver.run("mid")) == pytest.approx(expected_mid(2.0, 8.0), rel=1e-9)


def test_preprocess_collects_warnings(
    warning_solver: Callable[..., SolverLite], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="mpxlite"):
        result = warning_solver().preprocess(run_id="prewarn")
    _assert_one_missing_constraint(result.warnings)
    logged = [r.getMessage() for r in caplog.records if r.name == "mpxlite"]
    assert logged == [f"getdp run prewarn: {result.warnings[0]}"]


def test_preprocess_parse_error_raises(make_solver: SolverFactory, tmp_path: Path) -> None:
    broken = tmp_path / "broken.formulation.pro"
    broken.write_text(
        (DATA_DIR / "rod.formulation.pro").read_text(encoding="utf-8")
        + "\nFunction { undefined_thing = ; }\n",
        encoding="utf-8",
    )
    with pytest.raises(GetDPError) as info:
        make_solver(resource_files=[broken]).preprocess(run_id="broken")
    err = info.value
    assert err.returncode != 0
    assert err.run_dir == tmp_path / "runs" / "broken"
    assert "broken.formulation.pro" in err.log_tail


# ----- (l) read_solution: a run that starts from the solution of another ------


def test_run_reads_the_solution_of_another_run(make_solver: SolverFactory, tmp_path: Path) -> None:
    """``read_solution`` hands the file to GetDP: the resolution reads it and solves nothing."""
    first = make_solver(v=2.0, s=8.0).run("mid", run_id="first")
    assert first.solution_path.is_file()
    assert _mid(first) == pytest.approx(expected_mid(2.0, 8.0), rel=1e-9)
    # Other scalars on purpose: solved anew, u(0.5) would be expected_mid(5, 1).
    reread = SolverLite(
        DATA_DIR,
        "rod",
        "rod.msh",
        "Reread",
        resource_files=[DATA_DIR / "rod.formulation.pro"],
        workflow=WorkflowData({"V": 5.0, "S": 1.0}),
        runs_dir=tmp_path / "runs",
    )
    second = reread.run("mid", run_id="second", read_solution=first.solution_path)
    assert second.cmd[-7:] == (
        "-res",
        str(first.solution_path),
        "-pre",
        "Reread",
        "-cal",
        "-pos",
        "mid",
    )
    assert _mid(second) == pytest.approx(_mid(first), rel=1e-12)
    assert dict(WorkflowData.read_pro(second.workflow_path)) == {"V": 5.0, "S": 1.0}
    assert second.pre_path.is_file() and second.manifest_path.is_file()


@_requires_mpi
def test_preprocess_mpi(warning_solver: Callable[..., SolverLite]) -> None:
    launch = LaunchSpec(np=2, env={"MPXLITE_GETDP_BINARY": str(mpi_getdp())})
    result = warning_solver().preprocess(launch=launch)
    assert (result.work_dir / "mpxlite.pre").is_file()
    _assert_one_missing_constraint(result.warnings)
