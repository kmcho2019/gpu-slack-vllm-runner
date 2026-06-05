"""Scheduler logic for launching and stopping GPU slack jobs."""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field

from gpu_slack_runner.config import AppConfig
from gpu_slack_runner.gpu import GpuStatus, collect_gpu_status
from gpu_slack_runner.processes import (
    is_pid_alive,
    managed_pid_set,
    pid_owner,
    terminate_process_tree,
)
from gpu_slack_runner.state import (
    ManagedJob,
    active_gpu_cooldowns,
    ensure_runtime_dirs,
    read_jobs,
    remove_job,
    set_gpu_cooldown,
    write_job,
)


@dataclass(frozen=True, slots=True)
class GpuDecision:
    """Decision metadata for one GPU."""

    index: int
    idle: bool
    reason: str
    utilization_gpu_pct: int
    memory_free_mib: int
    foreign_pids: list[int] = field(default_factory=list)
    managed_pids: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Result of one scheduler check."""

    decisions: list[GpuDecision]
    active_jobs_before: list[ManagedJob]
    active_jobs_after: list[ManagedJob]
    started_jobs: list[ManagedJob]
    stopped_jobs: list[str]
    cleaned_jobs: list[str]
    dry_run: bool = False


def _format_command(command: list[str], variables: dict[str, str]) -> list[str]:
    return [part.format(**variables) for part in command]


def _job_id() -> str:
    return f"{int(time.time())}-{uuid.uuid4().hex[:8]}"


def _distributed_port(gpus: list[int]) -> str:
    return str(52_000 + gpus[0] * 100)


def _job_has_foreign_process(job: ManagedJob, statuses: list[GpuStatus], managed_pids: set[int]) -> bool:
    gpu_status = {s.index: s for s in statuses}
    for gpu_idx in job.gpus:
        status = gpu_status.get(gpu_idx)
        if status is None:
            continue
        if any(pid not in managed_pids for pid in status.compute_pids):
            return True
    return False


def _decide_idle_gpus(
    config: AppConfig,
    statuses: list[GpuStatus],
    jobs: list[ManagedJob],
    known_managed_pids: set[int],
) -> list[GpuDecision]:
    policy = config.idle_policy
    managed_gpus = {gpu for job in jobs if is_pid_alive(job.pid) for gpu in job.gpus}
    cooldowns = active_gpu_cooldowns(config.runtime.state_dir)
    decisions: list[GpuDecision] = []

    for status in statuses:
        all_pids = status.compute_pids
        managed_here = sorted(pid for pid in all_pids if pid in known_managed_pids)
        foreign_here = sorted(pid for pid in all_pids if pid not in known_managed_pids)

        if status.index in cooldowns:
            remaining = int(max(0, cooldowns[status.index] - time.time()))
            decisions.append(
                GpuDecision(
                    index=status.index,
                    idle=False,
                    reason=f"cooldown active for {remaining}s",
                    utilization_gpu_pct=status.utilization_gpu_pct,
                    memory_free_mib=status.memory_free_mib,
                    foreign_pids=foreign_here,
                    managed_pids=managed_here,
                )
            )
            continue

        if status.index in policy.exclude_gpus:
            decisions.append(
                GpuDecision(
                    index=status.index,
                    idle=False,
                    reason="excluded by config",
                    utilization_gpu_pct=status.utilization_gpu_pct,
                    memory_free_mib=status.memory_free_mib,
                    foreign_pids=foreign_here,
                    managed_pids=managed_here,
                )
            )
            continue

        if status.index in managed_gpus or managed_here:
            decisions.append(
                GpuDecision(
                    index=status.index,
                    idle=False,
                    reason="managed filler job already running",
                    utilization_gpu_pct=status.utilization_gpu_pct,
                    memory_free_mib=status.memory_free_mib,
                    foreign_pids=foreign_here,
                    managed_pids=managed_here,
                )
            )
            continue

        if status.memory_free_mib < policy.min_free_memory_mib:
            decisions.append(
                GpuDecision(
                    index=status.index,
                    idle=False,
                    reason=f"free memory {status.memory_free_mib} MiB < {policy.min_free_memory_mib} MiB",
                    utilization_gpu_pct=status.utilization_gpu_pct,
                    memory_free_mib=status.memory_free_mib,
                    foreign_pids=foreign_here,
                    managed_pids=managed_here,
                )
            )
            continue

        no_foreign_process = len(foreign_here) == 0
        low_utilization = status.utilization_gpu_pct < policy.gpu_utilization_below_pct

        if no_foreign_process and low_utilization:
            decisions.append(
                GpuDecision(
                    index=status.index,
                    idle=True,
                    reason="no foreign compute process and low utilization",
                    utilization_gpu_pct=status.utilization_gpu_pct,
                    memory_free_mib=status.memory_free_mib,
                    foreign_pids=foreign_here,
                    managed_pids=managed_here,
                )
            )
            continue

        if no_foreign_process:
            decisions.append(
                GpuDecision(
                    index=status.index,
                    idle=False,
                    reason=f"utilization {status.utilization_gpu_pct}% >= {policy.gpu_utilization_below_pct}%",
                    utilization_gpu_pct=status.utilization_gpu_pct,
                    memory_free_mib=status.memory_free_mib,
                    foreign_pids=foreign_here,
                    managed_pids=managed_here,
                )
            )
            continue

        if (
            low_utilization
            and not policy.require_no_foreign_compute_process
            and policy.allow_low_utilization_with_foreign_process
        ):
            decisions.append(
                GpuDecision(
                    index=status.index,
                    idle=True,
                    reason="low utilization despite existing foreign process; aggressive mode enabled",
                    utilization_gpu_pct=status.utilization_gpu_pct,
                    memory_free_mib=status.memory_free_mib,
                    foreign_pids=foreign_here,
                    managed_pids=managed_here,
                )
            )
            continue

        owners = ", ".join(f"{pid}:{pid_owner(pid)}" for pid in foreign_here) or "none"
        decisions.append(
            GpuDecision(
                index=status.index,
                idle=False,
                reason=f"foreign compute process present ({owners})",
                utilization_gpu_pct=status.utilization_gpu_pct,
                memory_free_mib=status.memory_free_mib,
                foreign_pids=foreign_here,
                managed_pids=managed_here,
            )
        )

    return decisions


def _make_gpu_groups(idle_gpu_indices: list[int], gpus_per_job: int) -> list[list[int]]:
    return [
        idle_gpu_indices[i : i + gpus_per_job]
        for i in range(0, len(idle_gpu_indices), gpus_per_job)
        if len(idle_gpu_indices[i : i + gpus_per_job]) == gpus_per_job
    ]


def _start_job(config: AppConfig, gpus: list[int], dry_run: bool = False) -> ManagedJob:
    runtime = config.runtime
    job_cfg = config.job
    ensure_runtime_dirs(runtime.state_dir, runtime.log_dir, runtime.output_dir)

    job_id = _job_id()
    gpu_csv = ",".join(str(gpu) for gpu in gpus)
    now = time.strftime("%Y%m%d-%H%M%S")
    stdout_log = runtime.log_dir / f"{job_cfg.name}-{job_id}-stdout.log"
    stderr_log = runtime.log_dir / f"{job_cfg.name}-{job_id}-stderr.log"

    variables = {
        "job_id": job_id,
        "gpu": str(gpus[0]),
        "gpu0": str(gpus[0]),
        "gpus": gpu_csv,
        "num_gpus": str(len(gpus)),
        "distributed_port": _distributed_port(gpus),
        "timestamp": now,
        "repo_root": str(runtime.repo_root),
        "state_dir": str(runtime.state_dir),
        "log_dir": str(runtime.log_dir),
        "output_dir": str(runtime.output_dir),
        "config_path": str(config.config_path or ""),
    }
    command = _format_command(job_cfg.command, variables)
    if not command:
        raise ValueError("job.command is empty; nothing to launch")

    fake_pid = -1
    managed = ManagedJob(
        job_id=job_id,
        pid=fake_pid,
        gpus=gpus,
        command=command,
        started_at=time.time(),
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        config_path=str(config.config_path) if config.config_path else None,
    )
    if dry_run:
        return managed

    env = os.environ.copy()
    env.update({key: value.format(**variables) for key, value in job_cfg.environment.items()})
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu_csv,
            "GPU_SLACK_RUNNER_MANAGED": "1",
            "GPU_SLACK_RUNNER_JOB_ID": job_id,
            "GPU_SLACK_RUNNER_VISIBLE_GPUS": gpu_csv,
        }
    )

    with stdout_log.open("ab") as out, stderr_log.open("ab") as err:
        proc = subprocess.Popen(
            command,
            cwd=runtime.repo_root,
            env=env,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )

    managed.pid = int(proc.pid)
    write_job(runtime.state_dir, managed)
    return managed


def check_once(config: AppConfig, dry_run: bool = False) -> CheckResult:
    """Run one scheduling pass: cleanup, stop conflicting jobs, then launch on idle GPUs."""

    ensure_runtime_dirs(config.runtime.state_dir, config.runtime.log_dir, config.runtime.output_dir)
    statuses = collect_gpu_status()
    jobs_before = read_jobs(config.runtime.state_dir)
    cleaned: list[str] = []
    stopped: list[str] = []

    # Remove stale state records for jobs whose root process has exited.
    live_jobs: list[ManagedJob] = []
    for job in jobs_before:
        if is_pid_alive(job.pid):
            live_jobs.append(job)
        else:
            cleaned.append(job.job_id)
            if not dry_run:
                remove_job(config.runtime.state_dir, job.job_id)

    managed_pids = managed_pid_set(live_jobs)

    # Polite preemption: if a real user process appears on a GPU occupied by filler, stop filler.
    still_live_jobs: list[ManagedJob] = []
    for job in live_jobs:
        if _job_has_foreign_process(job, statuses, managed_pids):
            stopped.append(job.job_id)
            if not dry_run:
                terminate_process_tree(job.pid, config.job.stop_timeout_seconds)
                for gpu_idx in job.gpus:
                    set_gpu_cooldown(config.runtime.state_dir, gpu_idx, config.job.cooldown_seconds_after_stop)
                remove_job(config.runtime.state_dir, job.job_id)
        else:
            still_live_jobs.append(job)

    jobs_after_stop = read_jobs(config.runtime.state_dir) if not dry_run else still_live_jobs
    managed_pids = managed_pid_set(jobs_after_stop)
    decisions = _decide_idle_gpus(config, statuses, jobs_after_stop, managed_pids)
    idle_gpu_indices = [d.index for d in decisions if d.idle]
    gpu_groups = _make_gpu_groups(idle_gpu_indices, config.job.gpus_per_job)

    active_count = len([job for job in jobs_after_stop if is_pid_alive(job.pid)])
    start_budget = max(0, config.job.max_jobs - active_count)
    started: list[ManagedJob] = []

    for group in gpu_groups[:start_budget]:
        started.append(_start_job(config, group, dry_run=dry_run))

    active_after = read_jobs(config.runtime.state_dir) if not dry_run else [*jobs_after_stop, *started]
    return CheckResult(
        decisions=decisions,
        active_jobs_before=jobs_before,
        active_jobs_after=active_after,
        started_jobs=started,
        stopped_jobs=stopped,
        cleaned_jobs=cleaned,
        dry_run=dry_run,
    )


def stop_all(config: AppConfig, dry_run: bool = False) -> list[str]:
    """Stop all managed jobs tracked by this runner."""

    stopped: list[str] = []
    for job in read_jobs(config.runtime.state_dir):
        stopped.append(job.job_id)
        if not dry_run:
            terminate_process_tree(job.pid, config.job.stop_timeout_seconds)
            for gpu_idx in job.gpus:
                set_gpu_cooldown(config.runtime.state_dir, gpu_idx, config.job.cooldown_seconds_after_stop)
            remove_job(config.runtime.state_dir, job.job_id)
    return stopped


def status_json(config: AppConfig) -> str:
    """Return machine-readable scheduler and GPU status."""

    statuses = collect_gpu_status()
    jobs = read_jobs(config.runtime.state_dir)
    managed_pids = managed_pid_set(jobs)
    decisions = _decide_idle_gpus(config, statuses, jobs, managed_pids)
    payload = {
        "gpus": [
            {
                "index": s.index,
                "uuid": s.uuid,
                "name": s.name,
                "utilization_gpu_pct": s.utilization_gpu_pct,
                "utilization_memory_pct": s.utilization_memory_pct,
                "memory_used_mib": s.memory_used_mib,
                "memory_total_mib": s.memory_total_mib,
                "memory_free_mib": s.memory_free_mib,
                "compute_pids": sorted(s.compute_pids),
            }
            for s in statuses
        ],
        "decisions": [asdict(decision) for decision in decisions],
        "managed_jobs": [
            {
                "job_id": job.job_id,
                "pid": job.pid,
                "alive": is_pid_alive(job.pid),
                "gpus": job.gpus,
                "age_seconds": round(job.age_seconds, 3),
                "stdout_log": job.stdout_log,
                "stderr_log": job.stderr_log,
                "command": job.command,
            }
            for job in jobs
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)
