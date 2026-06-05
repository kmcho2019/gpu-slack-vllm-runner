"""State files for managed filler jobs."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ManagedJob:
    """Metadata for one managed filler job."""

    job_id: str
    pid: int
    gpus: list[int]
    command: list[str]
    started_at: float
    stdout_log: str
    stderr_log: str
    config_path: str | None = None

    @property
    def age_seconds(self) -> float:
        """Return elapsed wall-clock seconds since the job was launched."""

        return time.time() - self.started_at


def ensure_runtime_dirs(*paths: Path) -> None:
    """Create runtime directories if they do not exist."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def job_state_path(state_dir: Path, job_id: str) -> Path:
    """Return the path for a managed job state file."""

    return state_dir / f"job_{job_id}.json"


def write_job(state_dir: Path, job: ManagedJob) -> None:
    """Persist managed job metadata atomically enough for local scheduler use."""

    ensure_runtime_dirs(state_dir)
    path = job_state_path(state_dir, job.job_id)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(asdict(job), f, indent=2, sort_keys=True)
    tmp.replace(path)


def read_jobs(state_dir: Path) -> list[ManagedJob]:
    """Read all valid managed job records from the state directory."""

    if not state_dir.exists():
        return []
    jobs: list[ManagedJob] = []
    for path in sorted(state_dir.glob("job_*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                raw: dict[str, Any] = json.load(f)
            jobs.append(
                ManagedJob(
                    job_id=str(raw["job_id"]),
                    pid=int(raw["pid"]),
                    gpus=[int(x) for x in raw.get("gpus", [])],
                    command=[str(x) for x in raw.get("command", [])],
                    started_at=float(raw["started_at"]),
                    stdout_log=str(raw.get("stdout_log", "")),
                    stderr_log=str(raw.get("stderr_log", "")),
                    config_path=None if raw.get("config_path") is None else str(raw.get("config_path")),
                )
            )
        except Exception:
            # Bad state files should not break the scheduler; leave them for inspection.
            continue
    return jobs


def remove_job(state_dir: Path, job_id: str) -> None:
    """Remove one managed job record if it exists."""

    path = job_state_path(state_dir, job_id)
    if path.exists():
        path.unlink()


def cooldown_path(state_dir: Path, gpu_index: int) -> Path:
    """Return cooldown marker path for one GPU."""

    return state_dir / f"cooldown_gpu_{gpu_index}.json"


def set_gpu_cooldown(state_dir: Path, gpu_index: int, seconds: int) -> None:
    """Set a temporary cooldown marker for one GPU."""

    ensure_runtime_dirs(state_dir)
    until = time.time() + max(0, seconds)
    path = cooldown_path(state_dir, gpu_index)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"gpu": gpu_index, "until": until}, f)


def active_gpu_cooldowns(state_dir: Path) -> dict[int, float]:
    """Return active GPU cooldowns and remove expired markers."""

    if not state_dir.exists():
        return {}
    now = time.time()
    active: dict[int, float] = {}
    for path in state_dir.glob("cooldown_gpu_*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            gpu = int(raw["gpu"])
            until = float(raw["until"])
            if until > now:
                active[gpu] = until
            else:
                path.unlink(missing_ok=True)
        except Exception:
            continue
    return active
