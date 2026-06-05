from pathlib import Path

from gpu_slack_runner.config import load_config


def test_load_default_config() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    assert config.idle_policy.gpu_utilization_below_pct == 10
    assert config.job.gpus_per_job == 1
    assert config.runtime.poll_interval_seconds == 1800
    assert "gpu-slack-vllm-generate" in config.job.command
    assert "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4" in config.job.command
    assert "131072" in config.job.command
    assert "1.0" in config.job.command
    assert "262144" in config.job.command
    assert "fp8" in config.job.command
    assert "flashinfer_trtllm" in config.job.command
    assert "--batch-size" in config.job.command
    assert "1" in config.job.command
    assert config.job.environment["VLLM_PORT"] == "{distributed_port}"
    assert "VLLM_USE_FLASHINFER_MOE_FP4" not in config.job.environment
    assert "VLLM_FLASHINFER_MOE_BACKEND" not in config.job.environment
