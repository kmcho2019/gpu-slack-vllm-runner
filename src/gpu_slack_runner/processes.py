"""Process tree utilities for managed filler jobs."""

from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass

import psutil

from gpu_slack_runner.state import ManagedJob


@dataclass(frozen=True, slots=True)
class TerminationResult:
    """Result from attempting to stop a managed job."""

    pid: int
    terminated: bool
    killed: bool
    message: str


def is_pid_alive(pid: int) -> bool:
    """Return whether a process ID exists and is not a zombie."""

    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def process_tree_pids(pid: int) -> set[int]:
    """Return PID plus recursive child PIDs, ignoring inaccessible processes."""

    try:
        proc = psutil.Process(pid)
        children = proc.children(recursive=True)
        return {pid, *(child.pid for child in children)}
    except psutil.Error:
        return {pid}


def managed_pid_set(jobs: list[ManagedJob]) -> set[int]:
    """Return root and child PIDs for all live managed jobs."""

    pids: set[int] = set()
    for job in jobs:
        if is_pid_alive(job.pid):
            pids.update(process_tree_pids(job.pid))
    return pids


def terminate_process_tree(pid: int, timeout_seconds: int) -> TerminationResult:
    """Gracefully terminate a process tree, then kill remaining processes."""

    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return TerminationResult(pid=pid, terminated=True, killed=False, message="process already exited")
    except psutil.Error as exc:
        return TerminationResult(pid=pid, terminated=False, killed=False, message=str(exc))

    procs = [root, *root.children(recursive=True)]
    for proc in reversed(procs):
        try:
            proc.send_signal(signal.SIGTERM)
        except psutil.NoSuchProcess:
            continue
        except psutil.Error:
            continue

    gone, alive = psutil.wait_procs(procs, timeout=timeout_seconds)
    killed = False
    if alive:
        killed = True
        for proc in reversed(alive):
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                continue
            except psutil.Error:
                continue
        psutil.wait_procs(alive, timeout=10)

    # Best effort: if the root is still around but no longer killable due to a race, report accurately.
    still_alive = is_pid_alive(pid)
    if not still_alive:
        return TerminationResult(
            pid=pid,
            terminated=True,
            killed=killed,
            message="terminated" if not killed else "terminated after SIGKILL",
        )
    return TerminationResult(pid=pid, terminated=False, killed=killed, message="process still alive")


def pid_owner(pid: int) -> str:
    """Return username for a PID when accessible."""

    try:
        return psutil.Process(pid).username()
    except psutil.Error:
        return "unknown"


def current_user() -> str:
    """Return current effective user name."""

    try:
        return psutil.Process(os.getpid()).username()
    except psutil.Error:
        return "unknown"
