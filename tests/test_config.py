from pathlib import Path

from gpu_slack_runner.config import load_config


def test_load_default_config() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    assert config.idle_policy.gpu_utilization_below_pct == 10
    assert config.job.gpus_per_job == 1
    assert "gpu-slack-vllm-generate" in config.job.command
