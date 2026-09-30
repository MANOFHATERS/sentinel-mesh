"""Live runs: re-train the fast models, or re-run the evaluation, from the dashboard.

The Models and Evaluation pages show saved results by default: instant, and the same
numbers every time. This module is the optional "run it now" behind them, so a reader can
watch the numbers being produced instead of taking them on trust.

Three kinds of run, one at a time (they are CPU-heavy, and a second click while one is
running is refused rather than queued):

``retrain``
    Trains the autoencoder, the GNN and the response policy again under a chosen seed,
    in-process, and reports them in the same shape as the Models page. It **never
    replaces the models serving the live incident flow** — a re-train is a second
    opinion (a different seed giving similar numbers is the point), not a hot swap.
``eval_quick``
    ``scripts/evaluate.py`` on a smaller corpus, a minute or so.
``eval_full``
    The complete evaluation, every gate, several minutes.

Both evaluations run ``scripts/evaluate.py`` as a **subprocess** writing to their own
file: PRD §9.3 says one pipeline produces every number, so the dashboard runs that
pipeline rather than a second copy of it, and a crash or a slow gate cannot take the
server down. The saved report the Evaluation page opens with is never overwritten.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from sentinel.core.errors import SentinelError

__all__ = ["Busy", "Job", "RunKind", "RunManager", "RunsUnavailable", "default_script"]


class Busy(SentinelError):
    """A run is already in progress."""


class RunsUnavailable(SentinelError):
    """The evaluation script is not on disk (an installed package, not a checkout)."""


class RunKind:
    RETRAIN: Final = "retrain"
    QUICK: Final = "eval_quick"
    FULL: Final = "eval_full"
    #: Real public data (sentinel.real): the detector on real flows, and a real repository.
    REAL_NETWORK: Final = "real_network"
    REAL_SCAN: Final = "real_scan"
    #: Kinds that run in-process through a handler and cannot be cancelled.
    HANDLED: Final = (REAL_NETWORK, REAL_SCAN)
    ALL: Final = (RETRAIN, QUICK, FULL, REAL_NETWORK, REAL_SCAN)


#: Fixed argument lists: the only user-supplied value that reaches a command line is the
#: seed, and it is validated as an integer, so there is nothing to inject into.
_QUICK_ARGS: Final[tuple[str, ...]] = ("--n", "5000", "--graph", "--kb", "--policy")
_FULL_ARGS: Final[tuple[str, ...]] = (
    "--n",
    "20000",
    "--cross-dataset",
    "--graph",
    "--kb",
    "--policy",
    "--augment",
    "--agents",
    "--codescan",
    "--supplychain",
    "--connectors",
    "--dashboard",
)
_TIMEOUT: Final[dict[str, float]] = {RunKind.QUICK: 900.0, RunKind.FULL: 3600.0}
_MAX_SEED: Final[int] = 2**31 - 1


def default_script() -> Path:
    return Path(__file__).resolve().parents[3] / "scripts" / "evaluate.py"


@dataclass(slots=True)
class Job:
    id: str
    kind: str
    seed: int
    started_by: str
    started: float
    state: str = "running"  # running | done | failed | cancelled
    finished: float | None = None
    log: deque[str] = field(default_factory=lambda: deque(maxlen=12))
    result: dict[str, Any] | None = None
    error: str | None = None
    exit_code: int | None = None
    proc: subprocess.Popen[str] | None = None
    cancel_requested: bool = False
    params: dict[str, Any] = field(default_factory=dict)

    def view(self, now: float) -> dict[str, Any]:
        end = self.finished if self.finished is not None else now
        return {
            "id": self.id,
            "kind": self.kind,
            "seed": self.seed,
            "started_by": self.started_by,
            "state": self.state,
            "elapsed_s": round(end - self.started, 1),
            "log": list(self.log),
            "error": self.error,
            "exit_code": self.exit_code,
            "params": self.params,
            "result": self.result,
        }


class RunManager:
    def __init__(
        self,
        *,
        workdir: Path,
        retrain: Callable[[int], dict[str, Any]],
        read_evaluation: Callable[[Path], dict[str, Any]],
        script: Path | None = None,
        python: str = sys.executable,
        now: Callable[[], float] = time.time,
        handlers: Mapping[str, Callable[[int, dict[str, Any]], dict[str, Any]]] | None = None,
        availability: Callable[[], dict[str, bool]] | None = None,
    ) -> None:
        self._handlers = dict(handlers or {})
        self._availability = availability
        self._workdir = workdir
        self._retrain = retrain
        self._read_evaluation = read_evaluation
        self._script = script if script is not None else default_script()
        self._python = python
        self._now = now
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._latest: dict[str, str] = {}
        self._running: str | None = None

    @property
    def evaluation_available(self) -> bool:
        return self._script.is_file()

    # -- control ---------------------------------------------------------------- #

    def start(self, kind: str, seed: int, *, by: str, params: dict[str, Any] | None = None) -> Job:
        if kind not in RunKind.ALL:
            raise ValueError(f"unknown run kind {kind!r}")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _MAX_SEED:
            raise ValueError(f"seed must be an integer between 0 and {_MAX_SEED}")
        if kind in RunKind.HANDLED and kind not in self._handlers:
            raise RunsUnavailable(f"{kind} is not enabled on this server")
        if kind in (RunKind.QUICK, RunKind.FULL) and not self.evaluation_available:
            raise RunsUnavailable("scripts/evaluate.py is not present in this installation")
        with self._lock:
            if self._running is not None:
                raise Busy("another run is in progress; wait for it or cancel it")
            job = Job(secrets.token_hex(6), kind, seed, by, self._now(), params=dict(params or {}))
            self._jobs[job.id] = job
            self._latest[kind] = job.id
            self._running = job.id
        if kind == RunKind.RETRAIN:
            target = self._run_retrain
        elif kind in RunKind.HANDLED:
            target = self._run_handler
        else:
            target = self._run_evaluation
        threading.Thread(target=target, args=(job,), name=f"run-{job.id}", daemon=True).start()
        return job

    def cancel(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        with self._lock:
            # Flag first, then stop the process if it exists yet: a cancel that lands
            # before the worker has spawned it is honoured the moment it does.
            if job.state == "running" and job.kind in (RunKind.QUICK, RunKind.FULL):
                job.cancel_requested = True
                if job.proc is not None:
                    job.proc.terminate()
        return job

    def describe(self, job: Job) -> dict[str, Any]:
        return job.view(self._now())

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def latest(self) -> dict[str, Any]:
        now = self._now()
        jobs = {kind: self._jobs[jid].view(now) for kind, jid in self._latest.items()}
        return {
            "available": {
                RunKind.RETRAIN: True,
                RunKind.QUICK: self.evaluation_available,
                RunKind.FULL: self.evaluation_available,
                **{kind: kind in self._handlers for kind in RunKind.HANDLED},
                **(self._availability() if self._availability else {}),
            },
            "busy": self._running,
            "jobs": jobs,
        }

    # -- workers ---------------------------------------------------------------- #

    def _finish(self, job: Job, state: str, *, error: str | None = None) -> None:
        job.state = state
        job.error = error
        job.finished = self._now()
        with self._lock:
            if self._running == job.id:
                self._running = None

    def _run_retrain(self, job: Job) -> None:
        try:
            job.log.append(f"training the autoencoder, GNN and policy under seed {job.seed}")
            job.result = self._retrain(job.seed)
        except Exception as exc:  # a failed re-train is reported, never fatal to the server
            self._finish(job, "failed", error=f"{type(exc).__name__}: {exc}")
        else:
            self._finish(job, "done")

    def _run_handler(self, job: Job) -> None:
        try:
            job.log.append("working on real public data")
            job.result = self._handlers[job.kind](job.seed, job.params)
        except Exception as exc:  # reported on the page; never fatal to the server
            self._finish(job, "failed", error=f"{type(exc).__name__}: {exc}"[:400])
        else:
            self._finish(job, "done")

    def _run_evaluation(self, job: Job) -> None:
        out_dir = self._workdir / f"run-{job.id}"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / "evaluation.json"
            extra = _QUICK_ARGS if job.kind == RunKind.QUICK else _FULL_ARGS
            command = [
                self._python,
                str(self._script),
                *extra,
                "--seed",
                str(job.seed),
                "--out",
                str(out),
            ]
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
            with self._lock:
                job.proc = subprocess.Popen(  # fixed argv, integer seed
                    command,
                    cwd=self._script.parent.parent,
                    env=env,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                if job.cancel_requested:
                    job.proc.terminate()
            watchdog = threading.Timer(_TIMEOUT[job.kind], job.proc.kill)
            watchdog.daemon = True
            watchdog.start()
            try:
                assert job.proc.stdout is not None
                for line in job.proc.stdout:
                    line = line.strip()
                    if line:
                        job.log.append(line[:200])
                job.exit_code = job.proc.wait()
            finally:
                watchdog.cancel()
            if job.cancel_requested:
                self._finish(job, "cancelled")
            elif out.is_file():
                # The evaluation exits non-zero when a gate fails but still writes its
                # report; a failed gate is a result to show, not an error to hide.
                job.result = self._read_evaluation(out)
                self._finish(job, "done")
            else:
                self._finish(
                    job,
                    "failed",
                    error=f"evaluation exited {job.exit_code} without writing a report",
                )
        except Exception as exc:
            self._finish(job, "failed", error=f"{type(exc).__name__}: {exc}")
