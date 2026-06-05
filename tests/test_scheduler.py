from gpu_slack_runner.config import AppConfig
from gpu_slack_runner.gpu import GpuStatus
from gpu_slack_runner.scheduler import _decide_idle_gpus


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
