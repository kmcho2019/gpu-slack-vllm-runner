import gzip
import json
import os
import time

from gpu_slack_runner.archive import archive_once
from gpu_slack_runner.config import AppConfig, ArchiveConfig
from gpu_slack_runner.state import ManagedJob, write_job


def test_archive_once_compresses_old_outputs(tmp_path) -> None:
    config = AppConfig()
    config.runtime.state_dir = tmp_path / "state"
    config.runtime.log_dir = tmp_path / "logs"
    config.runtime.output_dir = tmp_path / "out"
    config.archive = ArchiveConfig(min_age_hours=1, interval_seconds=3600, archive_dir=tmp_path / "archive")
    config.runtime.log_dir.mkdir()
    config.runtime.output_dir.mkdir()

    generation = config.runtime.output_dir / "old.jsonl"
    generation.write_text(json.dumps({"type": "job_start"}) + "\n" + json.dumps({"type": "generation"}) + "\n")
    log = config.runtime.log_dir / "old.log"
    log.write_text("first\nsecond\n")
    for path in [generation, log]:
        old = time.time() - 7200
        os.utime(path, (old, old))

    result = archive_once(config)

    assert len(result.archived) == 2
    assert not generation.exists()
    assert not log.exists()
    archived_generation = config.archive.archive_dir / "generations" / time.strftime("%Y-%m-%d") / "old.jsonl.gz"
    archived_log = config.archive.archive_dir / "logs" / time.strftime("%Y-%m-%d") / "old.log.gz"
    assert archived_generation.exists()
    assert archived_log.exists()
    with gzip.open(archived_generation, "rt", encoding="utf-8") as f:
        assert json.loads(f.readline())["type"] == "job_start"
    manifest = (config.archive.archive_dir / "manifest.jsonl").read_text().splitlines()
    assert len(manifest) == 2


def test_archive_once_skips_active_files(tmp_path) -> None:
    config = AppConfig()
    config.runtime.state_dir = tmp_path / "state"
    config.runtime.log_dir = tmp_path / "logs"
    config.runtime.output_dir = tmp_path / "out"
    config.archive = ArchiveConfig(min_age_hours=1, interval_seconds=3600, archive_dir=tmp_path / "archive")
    config.runtime.log_dir.mkdir()
    config.runtime.output_dir.mkdir()

    output = config.runtime.output_dir / "20260101-active-job.jsonl"
    stdout = config.runtime.log_dir / "active-stdout.log"
    stderr = config.runtime.log_dir / "active-stderr.log"
    for path in [output, stdout, stderr]:
        path.write_text(json.dumps({"type": "job_start"}) + "\n" if path.suffix == ".jsonl" else "log\n")
        old = time.time() - 7200
        os.utime(path, (old, old))

    write_job(
        config.runtime.state_dir,
        ManagedJob(
            job_id="active-job",
            pid=123,
            gpus=[0],
            command=["run"],
            started_at=time.time(),
            stdout_log=str(stdout),
            stderr_log=str(stderr),
        ),
    )

    result = archive_once(config)

    assert not result.archived
    assert len(result.skipped_active) == 3
    assert output.exists()
    assert stdout.exists()
    assert stderr.exists()
