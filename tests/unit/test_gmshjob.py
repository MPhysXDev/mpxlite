# Copyright (c) 2026 MPhysX
# SPDX-License-Identifier: MIT

"""The runner of Gmsh jobs (:mod:`mpxlite.gmshjob`), with fake jobs (no Gmsh).

A fake installation holds ``venv/bin/python`` (a script that executes the
Python of these tests) and ``engine/gmsh/jobs/<job>.py``: small programs that
read the request, write outputs and a report, fail, crash or sleep.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

import mpxlite
from mpxlite import gmshjob, memory


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake installation, named by MPXLITE_HOME."""
    root = tmp_path / "home"
    python = root / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    python.chmod(0o755)
    (root / "engine" / "gmsh" / "jobs").mkdir(parents=True)
    monkeypatch.setenv("MPXLITE_HOME", str(root))
    for variable in gmshjob.MPI_SIZE_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    return root


_PROLOGUE = """\
import json, os, sys, time
from pathlib import Path
run = Path(sys.argv[1])
request = json.loads((run / "request.json").read_text())

def report(status, **fields):
    doc = {"format": "mpxlite-gmsh-job-report", "version": 1, "job": request["job"],
           "status": status, "gmsh": "0.0.0", **fields}
    (run / "report.json").write_text(json.dumps(doc))
"""


def _job(home: Path, name: str, body: str) -> Path:
    path = home / "engine" / "gmsh" / "jobs" / f"{name}.py"
    path.write_text(_PROLOGUE + textwrap.dedent(body), encoding="utf-8")
    return path


def _echo(home: Path) -> None:
    _job(
        home,
        "echo",
        """
        text = Path(request["inputs"]["source"]).read_text()
        (run / "copy.txt").write_text(text.upper())
        print("echo ran")
        report(
            "ok",
            outputs={"copy": "copy.txt"},
            report={"parameters": request["parameters"], "omp": os.environ.get("OMP_NUM_THREADS"),
                    "extra": os.environ.get("MPXLITE_TEST_EXTRA"), "cwd": os.getcwd()},
            messages=[{"level": "warning", "text": "a warning"},
                      {"level": "info", "text": "some info"}],
        )
        """,
    )


def test_find_gmsh_jobs(tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert mpxlite.find_gmsh_jobs() == (home / "engine" / "gmsh" / "jobs").resolve()
    other = tmp_path / "dev_jobs"
    other.mkdir()
    assert mpxlite.find_gmsh_jobs({gmshjob.GMSH_JOBS_ENV: str(other)}) == other.resolve()
    with pytest.raises(FileNotFoundError, match="is not a directory"):
        mpxlite.find_gmsh_jobs({gmshjob.GMSH_JOBS_ENV: str(tmp_path / "none")})
    monkeypatch.setenv("MPXLITE_HOME", str(tmp_path / "nothing"))
    with pytest.raises(FileNotFoundError, match="No Gmsh jobs"):
        mpxlite.find_gmsh_jobs()


def test_successful_run(tmp_path: Path, home: Path) -> None:
    _echo(home)
    source = tmp_path / "in.txt"
    source.write_text("abc", encoding="utf-8")
    result = mpxlite.run_gmsh_job(
        "echo",
        inputs={"source": source},
        parameters={"size": 0.5, "names": ["a", "b"], "nested": {"k": None}},
        runs_dir=tmp_path / "runs",
        run_id="r1",
        env={"MPXLITE_TEST_EXTRA": "x", "OMP_NUM_THREADS": "8"},
    )
    assert result.job == "echo" and result.run_id == "r1"
    assert result.run_dir == (tmp_path / "runs" / "r1").resolve()
    assert result.outputs == {"copy": result.run_dir / "copy.txt"}
    assert result.outputs["copy"].read_text() == "ABC"
    assert result.report["parameters"] == {"size": 0.5, "names": ["a", "b"], "nested": {"k": None}}
    assert result.report["omp"] == "1"  # always one thread
    assert result.report["extra"] == "x"
    assert Path(result.report["cwd"]).resolve() == result.run_dir
    assert result.messages == (("warning", "a warning"), ("info", "some info"))
    assert result.warnings == ("a warning",)
    assert result.gmsh_version == "0.0.0"
    assert result.log_path == result.run_dir / "job.log"
    assert "echo ran" in result.log_path.read_text()
    assert result.elapsed_s > 0
    request = json.loads((result.run_dir / "request.json").read_text())
    assert request == {
        "format": "mpxlite-gmsh-job-request",
        "version": 1,
        "job": "echo",
        "inputs": {"source": str(source.resolve())},
        "parameters": {"size": 0.5, "names": ["a", "b"], "nested": {"k": None}},
    }


def test_jobs_directory_of_the_environment(tmp_path: Path, home: Path) -> None:
    """MPXLITE_GMSH_JOBS (here given through ``env``) names the jobs, not the installation."""
    dev = tmp_path / "dev"
    dev.mkdir()
    (dev / "only_here.py").write_text(
        _PROLOGUE + 'report("ok", report={"where": "dev"})\n', encoding="utf-8"
    )
    result = mpxlite.run_gmsh_job(
        "only_here", runs_dir=tmp_path / "runs", env={gmshjob.GMSH_JOBS_ENV: str(dev)}
    )
    assert result.report == {"where": "dev"} and result.outputs == {}
    with pytest.raises(FileNotFoundError, match="No Gmsh job 'only_here'"):
        mpxlite.run_gmsh_job("only_here", runs_dir=tmp_path / "runs")


def test_reported_failure(tmp_path: Path, home: Path) -> None:
    _job(
        home,
        "refuse",
        """
        report("failed", failure={"kind": "rule", "code": "cell.not_periodic",
                                  "message": "not periodic", "data": {"face": "x = 0"}})
        sys.exit(3)
        """,
    )
    with pytest.raises(mpxlite.GmshJobError) as caught:
        mpxlite.run_gmsh_job("refuse", runs_dir=tmp_path / "runs", run_id="f")
    err = caught.value
    assert (err.kind, err.code, err.message, err.data) == (
        "rule",
        "cell.not_periodic",
        "not periodic",
        {"face": "x = 0"},
    )
    assert err.job == "refuse" and err.run_dir == (tmp_path / "runs" / "f").resolve()
    assert err.log_path == err.run_dir / "job.log"
    assert "cell.not_periodic" in str(err) and "not periodic" in str(err)


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ("raise RuntimeError('boom')\n", "job.no_report"),
        ("(run / 'report.json').write_text('{not json')\n", "job.report_invalid"),
        ("report('ok')\ndoc = json.loads((run / 'report.json').read_text())\n"
         "doc['version'] = 99\n(run / 'report.json').write_text(json.dumps(doc))\n",
         "job.report_format"),
        ("report('ok', outputs={'mesh': 'missing.msh'})\n", "job.output_missing"),
        ("report('ok', outputs={'up': '../request.json'})\n", "job.output_missing"),
        ("report('ok')\nsys.exit(2)\n", "job.exit"),
        ("report('failed', failure={'kind': 'nonsense', 'code': 'x', 'message': 'y'})\n",
         "job.report_invalid"),
        ("report('ok', report=[1, 2])\n", "job.report_invalid"),
    ],
)  # fmt: skip
def test_crashes(tmp_path: Path, home: Path, body: str, code: str) -> None:
    _job(home, "broken", body)
    with pytest.raises(mpxlite.GmshJobError) as caught:
        mpxlite.run_gmsh_job("broken", runs_dir=tmp_path / "runs")
    assert caught.value.kind == "crash"
    assert caught.value.code == code


def test_crash_message_has_the_log_tail(tmp_path: Path, home: Path) -> None:
    _job(home, "broken", "print('last words', flush=True)\nraise RuntimeError('boom')\n")
    with pytest.raises(mpxlite.GmshJobError) as caught:
        mpxlite.run_gmsh_job("broken", runs_dir=tmp_path / "runs")
    assert "last words" in caught.value.message
    assert "RuntimeError: boom" in caught.value.data["log_tail"]


def test_timeout_kills_the_job(tmp_path: Path, home: Path) -> None:
    _job(home, "sleeper", "time.sleep(60)\n")
    with pytest.raises(mpxlite.GmshJobError) as caught:
        mpxlite.run_gmsh_job("sleeper", runs_dir=tmp_path / "runs", timeout=0.5)
    assert caught.value.kind == "timeout" and caught.value.code == "job.timeout"
    assert caught.value.data == {"timeout_s": 0.5}


def test_missing_input_fails_before_the_run(tmp_path: Path, home: Path) -> None:
    _echo(home)
    with pytest.raises(mpxlite.GmshJobError) as caught:
        mpxlite.run_gmsh_job(
            "echo", inputs={"source": tmp_path / "none.txt"}, runs_dir=tmp_path / "runs"
        )
    assert caught.value.kind == "input" and caught.value.code == "job.input_missing"
    assert caught.value.data["input"] == "source"
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("variable", gmshjob.MPI_SIZE_VARIABLES)
def test_refused_in_one_of_several_mpi_ranks(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    """A process started by an MPI launcher with several ranks does not run a job."""
    _job(home, "plain", "report('ok')\n")
    monkeypatch.setenv(variable, "4")
    with pytest.raises(mpxlite.GmshJobError) as caught:
        mpxlite.run_gmsh_job("plain", runs_dir=tmp_path / "runs")
    assert caught.value.kind == "input" and caught.value.code == "job.mpi_ranks"
    assert caught.value.data == {"variable": variable, "size": 4}
    assert caught.value.job == "plain"
    assert "once, before the parallel run, by the driving script" in caught.value.message
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("value", ["1", "", " ", "0", "two", "-3", "4.0"])
def test_one_rank_or_no_launcher_runs(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """One rank, no launcher or a value that is not a rank count: the job runs."""
    _job(home, "plain", "report('ok')\n")
    for variable in gmshjob.MPI_SIZE_VARIABLES:
        monkeypatch.setenv(variable, value)
    result = mpxlite.run_gmsh_job("plain", runs_dir=tmp_path / "runs")
    assert result.run_dir.is_dir()


def test_mpi_ranks_of_the_job_environment_do_not_count(tmp_path: Path, home: Path) -> None:
    """Only the calling process is checked, not the variables given to the job."""
    _job(home, "plain", "report('ok')\n")
    mpxlite.run_gmsh_job("plain", runs_dir=tmp_path / "runs", env={"OMPI_COMM_WORLD_SIZE": "4"})


def test_mpi_size_variables() -> None:
    assert gmshjob.MPI_SIZE_VARIABLES == (
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "MV2_COMM_WORLD_SIZE",
        "SLURM_STEP_NUM_TASKS",
    )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"job": "Bad-Name"}, ValueError),
        ({"job": "_private"}, ValueError),
        ({"parameters": {"x": float("nan")}}, ValueError),
        ({"parameters": {"x": object()}}, ValueError),
        ({"parameters": [1, 2]}, TypeError),
        ({"inputs": {"bad name": "x"}}, ValueError),
        ({"timeout": -1.0}, ValueError),
        ({"memory_max": 0}, ValueError),
        ({"run_id": "../escape"}, ValueError),
    ],
)
def test_invalid_arguments(tmp_path: Path, home: Path, kwargs: dict, error: type) -> None:
    _echo(home)
    call = {"job": "echo", "runs_dir": tmp_path / "runs", **kwargs}
    with pytest.raises(error):
        mpxlite.run_gmsh_job(**call)


def test_run_folder_is_never_reused(tmp_path: Path, home: Path) -> None:
    _job(home, "plain", "report('ok')\n")
    mpxlite.run_gmsh_job("plain", runs_dir=tmp_path / "runs", run_id="same")
    with pytest.raises(FileExistsError):
        mpxlite.run_gmsh_job("plain", runs_dir=tmp_path / "runs", run_id="same")


def test_no_python_in_the_installation(tmp_path: Path, home: Path) -> None:
    _job(home, "plain", "report('ok')\n")
    (home / "venv" / "bin" / "python").unlink()
    with pytest.raises(FileNotFoundError, match="No Python"):
        mpxlite.run_gmsh_job("plain", runs_dir=tmp_path / "runs")


def test_memory_cap_unavailable(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _job(home, "plain", "report('ok')\n")
    monkeypatch.setattr(memory, "hard_cap_available", lambda refresh=False: False)
    with pytest.raises(mpxlite.HardCapUnavailableError):
        mpxlite.run_gmsh_job("plain", runs_dir=tmp_path / "runs", memory_max=2**30)
    assert not (tmp_path / "runs").exists()


def test_error_kinds() -> None:
    err = mpxlite.GmshJobError("gmsh", "mesh.too_many_elements", "too many", {"limit": 10})
    assert err.to_json() == {
        "kind": "gmsh",
        "code": "mesh.too_many_elements",
        "message": "too many",
        "data": {"limit": 10},
    }
    assert err.job is None and err.run_dir is None
    with pytest.raises(ValueError, match="kind"):
        mpxlite.GmshJobError("other", "x", "y")
    assert gmshjob.JOB_FAILURE_KINDS == ("input", "rule", "gmsh", "timeout", "memory", "crash")
