from gpu_slack_runner.config import AppConfig
from gpu_slack_runner.gpu import GpuStatus
from gpu_slack_runner.scheduler import _decide_idle_gpus, _start_job


def test_busy_gpu_without_process_is_not_idle(tmp_path) -> None:
    config = AppConfig()
    config.runtime.state_dir = tmp_path
    status = GpuStatus(
        index=0,
        uuid="GPU-0",
        name="test",
        utilization_gpu_pct=50,
        utilization_memory_pct=0,
        memory_used_mib=1024,
        memory_total_mib=80_000,
        compute_processes=[],
    )

    decision = _decide_idle_gpus(config, [status], [], set())[0]

    assert not decision.idle
    assert decision.reason == "utilization 50% >= 10%"


def test_start_job_expands_gpu_specific_distributed_port(tmp_path) -> None:
    config = AppConfig()
    config.runtime.repo_root = tmp_path
    config.runtime.state_dir = tmp_path / "state"
    config.runtime.log_dir = tmp_path / "logs"
    config.runtime.output_dir = tmp_path / "out"
    config.job.command = ["run", "--master-port", "{distributed_port}"]

    job6 = _start_job(config, [6], dry_run=True)
    job7 = _start_job(config, [7], dry_run=True)

    assert job6.command == ["run", "--master-port", "52600"]
    assert job7.command == ["run", "--master-port", "52700"]
