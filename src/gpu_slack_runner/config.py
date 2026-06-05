"""Configuration loading for gpu-slack-vllm-runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class IdlePolicy:
    """Rules used to decide whether a GPU is safe to use as slack capacity."""

    gpu_utilization_below_pct: int = 10
    min_free_memory_mib: int = 8_000
    require_no_foreign_compute_process: bool = True
    allow_low_utilization_with_foreign_process: bool = False
    exclude_gpus: list[int] = field(default_factory=list)


@dataclass(slots=True)
class JobConfig:
    """Command and resource shape for the filler workload."""

    name: str = "vllm-synthetic-generation"
    command: list[str] = field(default_factory=list)
    gpus_per_job: int = 1
    max_jobs: int = 8
    cooldown_seconds_after_stop: int = 120
    stop_timeout_seconds: int = 45
    environment: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ArchiveConfig:
    """Age-based archive policy for generated data and logs."""

    enabled: bool = True
    min_age_hours: int = 24
    interval_seconds: int = 86_400
    archive_dir: Path = Path("data/archive")


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime paths and loop cadence."""

    poll_interval_seconds: int = 300
    state_dir: Path = Path("state")
    log_dir: Path = Path("logs")
    output_dir: Path = Path("data/output/generations")
    repo_root: Path = Path(".")


@dataclass(slots=True)
class AppConfig:
    """Top-level application configuration."""

    idle_policy: IdlePolicy = field(default_factory=IdlePolicy)
    job: JobConfig = field(default_factory=JobConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    config_path: Path | None = None


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping in config, got {type(value).__name__}")
    return value


def _path(value: Any, default: str | Path, base: Path) -> Path:
    raw = Path(str(value if value is not None else default)).expanduser()
    if raw.is_absolute():
        return raw
    return (base / raw).resolve()


def load_config(path: str | Path) -> AppConfig:
    """Load YAML config and resolve relative paths against the config directory."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = _as_dict(yaml.safe_load(f))

    base = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent

    idle_raw = _as_dict(raw.get("idle_policy"))
    job_raw = _as_dict(raw.get("job"))
    runtime_raw = _as_dict(raw.get("runtime"))
    archive_raw = _as_dict(raw.get("archive"))

    idle = IdlePolicy(
        gpu_utilization_below_pct=int(idle_raw.get("gpu_utilization_below_pct", 10)),
        min_free_memory_mib=int(idle_raw.get("min_free_memory_mib", 8_000)),
        require_no_foreign_compute_process=bool(
            idle_raw.get("require_no_foreign_compute_process", True)
        ),
        allow_low_utilization_with_foreign_process=bool(
            idle_raw.get("allow_low_utilization_with_foreign_process", False)
        ),
        exclude_gpus=[int(x) for x in idle_raw.get("exclude_gpus", [])],
    )

    command = job_raw.get("command") or []
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise TypeError("job.command must be a list of strings")

    env = job_raw.get("environment") or {}
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise TypeError("job.environment must be a mapping of string keys to string values")

    job = JobConfig(
        name=str(job_raw.get("name", "vllm-synthetic-generation")),
        command=command,
        gpus_per_job=int(job_raw.get("gpus_per_job", 1)),
        max_jobs=int(job_raw.get("max_jobs", 8)),
        cooldown_seconds_after_stop=int(job_raw.get("cooldown_seconds_after_stop", 120)),
        stop_timeout_seconds=int(job_raw.get("stop_timeout_seconds", 45)),
        environment=env,
    )

    runtime = RuntimeConfig(
        poll_interval_seconds=int(runtime_raw.get("poll_interval_seconds", 300)),
        state_dir=_path(runtime_raw.get("state_dir"), "state", base),
        log_dir=_path(runtime_raw.get("log_dir"), "logs", base),
        output_dir=_path(runtime_raw.get("output_dir"), "data/output/generations", base),
        repo_root=_path(runtime_raw.get("repo_root"), ".", base),
    )
    archive = ArchiveConfig(
        enabled=bool(archive_raw.get("enabled", True)),
        min_age_hours=int(archive_raw.get("min_age_hours", 24)),
        interval_seconds=int(archive_raw.get("interval_seconds", 86_400)),
        archive_dir=_path(archive_raw.get("archive_dir"), "data/archive", base),
    )

    if job.gpus_per_job < 1:
        raise ValueError("job.gpus_per_job must be >= 1")
    if job.max_jobs < 1:
        raise ValueError("job.max_jobs must be >= 1")
    if idle.gpu_utilization_below_pct < 0 or idle.gpu_utilization_below_pct > 100:
        raise ValueError("idle_policy.gpu_utilization_below_pct must be between 0 and 100")
    if archive.min_age_hours < 1:
        raise ValueError("archive.min_age_hours must be >= 1")
    if archive.interval_seconds < 3600:
        raise ValueError("archive.interval_seconds must be >= 3600")

    return AppConfig(idle_policy=idle, job=job, runtime=runtime, archive=archive, config_path=config_path)
