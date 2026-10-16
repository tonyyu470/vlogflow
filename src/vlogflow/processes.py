"""Cancellation and resource limits for processes owned by one job."""

from __future__ import annotations

import os
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import psutil

GIB = 1024**3


class Cancelled(RuntimeError):
    pass


class ResourceLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class Budget:
    min_start_gib: float = 0.4
    reserve_gib: float = 0.3
    max_rss_gib: float = 3.5
    max_swap_growth_mib: float = 128
    timeout: float = 720


MLX_BUDGET = Budget(4.1, 1.6, 3.5, 128, 720)
ASR_BUDGET = Budget(1.3, 0.5, 1.5, 128, 300)


def terminate_owned(child: subprocess.Popen):
    if child.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(child.pid, signal.SIGTERM)
    else:
        for p in psutil.Process(child.pid).children(recursive=True):
            p.terminate()
        child.terminate()
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(child.pid, signal.SIGKILL)
        else:
            for p in psutil.Process(child.pid).children(recursive=True):
                p.kill()
            child.kill()
        child.wait()


class Runner:
    def __init__(self, logs: Path, cancel=lambda: False, phase=lambda message: None):
        self.logs = logs
        logs.mkdir(parents=True, exist_ok=True)
        self.cancel = cancel
        self.phase = phase
        self.stats = []

    def check(self):
        if self.cancel():
            raise Cancelled("Task cancelled; completed intermediates are retained")

    def preflight(self, budget: Budget):
        self.check()
        available = psutil.virtual_memory().available / GIB
        if available < budget.min_start_gib:
            raise ResourceLimit(
                f"Available memory {available:.1f} GiB; this stage needs {budget.min_start_gib:.1f} GiB. Close unused applications and retry."
            )
        if psutil.disk_usage(self.logs).free < 512 * 1024**2:
            raise ResourceLimit("Less than 512 MiB free disk space. Free space and retry.")

    def run(self, argv: list[str], *, budget: Budget | None = None, env: dict | None = None) -> str:
        budget = budget or Budget()
        self.preflight(budget)
        log = self.logs / f"{uuid.uuid4().hex}.log"
        started = time.monotonic()
        swap0 = psutil.swap_memory().used
        stats = dict(peak_rss=0, min_available=psutil.virtual_memory().available, swap_growth=0)
        with log.open("wb") as handle:
            child = subprocess.Popen(
                [str(v) for v in argv],
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=os.name == "posix",
                env=env,
            )
            try:
                process = psutil.Process(child.pid)
                while child.poll() is None:
                    self.check()
                    available = psutil.virtual_memory().available
                    growth = max(0, psutil.swap_memory().used - swap0)
                    try:
                        rss = process.memory_info().rss + sum(
                            p.memory_info().rss for p in process.children(recursive=True)
                        )
                    except psutil.NoSuchProcess:
                        rss = 0
                    stats.update(
                        peak_rss=max(stats["peak_rss"], rss),
                        min_available=min(stats["min_available"], available),
                        swap_growth=max(stats["swap_growth"], growth),
                    )
                    if (
                        available < budget.reserve_gib * GIB
                        or rss > budget.max_rss_gib * GIB
                        or growth > budget.max_swap_growth_mib * 1024**2
                    ):
                        raise ResourceLimit(
                            "Memory protection stopped this stage. Close unused applications and retry; completed files are retained."
                        )
                    if time.monotonic() - started > budget.timeout:
                        raise RuntimeError("Stage exceeded its time limit. Shorten the input and retry.")
                    time.sleep(0.1)
                self.check()
            finally:
                terminate_owned(child)
                stats.update(seconds=round(time.monotonic() - started, 2), returncode=child.returncode)
                self.stats.append(stats)
        output = log.read_text(errors="replace")
        if child.returncode:
            raise RuntimeError(f"{Path(argv[0]).name} failed (exit {child.returncode}): {output[-2500:]}")
        return output
