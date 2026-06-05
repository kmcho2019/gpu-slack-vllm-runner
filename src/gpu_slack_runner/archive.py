"""Archive old logs and generation JSONL files."""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_slack_runner.config import AppConfig
from gpu_slack_runner.state import read_jobs

JSONL_TYPES = {"job_start", "generation", "job_end", "job_error"}


@dataclass(frozen=True, slots=True)
class ArchiveResult:
    """Summary of one archive pass."""

    archived: list[str]
    skipped_active: list[str]
    skipped_young: int
    dry_run: bool


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonl_counts(path: Path) -> dict[str, int]:
    counts = {name: 0 for name in sorted(JSONL_TYPES)}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record["type"]
            assert kind in JSONL_TYPES, f"Unknown JSONL record type: {kind}"
            counts[kind] += 1
    return {key: value for key, value in counts.items() if value}


def _line_count(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _archive_file(config: AppConfig, kind: str, path: Path, dry_run: bool) -> str:
    """Compress one old file and append a manifest entry."""

    assert kind in {"generation", "log"}
    date = time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime))
    target_dir = config.archive.archive_dir / f"{kind}s" / date
    target = target_dir / f"{path.name}.gz"
    assert not target.exists(), f"Archive already exists: {target}"

    metadata: dict[str, Any] = {
        "type": "archive_entry",
        "kind": kind,
        "source": str(path),
        "archive": str(target),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "created_at": time.time(),
    }
    if kind == "generation":
        metadata["record_counts"] = _jsonl_counts(path)
    elif kind == "log":
        metadata["line_count"] = _line_count(path)
    else:
        raise AssertionError(f"Unknown archive kind: {kind}")

    if dry_run:
        return str(target)

    target_dir.mkdir(parents=True, exist_ok=True)
    with path.open("rb") as src, gzip.open(target, "wb") as dst:
        dst.writelines(src)
    with (config.archive.archive_dir / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(metadata, sort_keys=True) + "\n")
    path.unlink()
    return str(target)


def archive_once(config: AppConfig, dry_run: bool = False) -> ArchiveResult:
    """Compress old output files and leave active job files alone."""

    if not config.archive.enabled:
        return ArchiveResult([], [], 0, dry_run)

    now = time.time()
    min_age = config.archive.min_age_hours * 3600.0
    jobs = read_jobs(config.runtime.state_dir)
    active_logs = {Path(job.stdout_log) for job in jobs} | {Path(job.stderr_log) for job in jobs}
    active_job_ids = {job.job_id for job in jobs}
    archived: list[str] = []
    skipped_active: list[str] = []
    skipped_young = 0

    # Only archive the two append-heavy file kinds this project owns.
    files = [("generation", path) for path in sorted(config.runtime.output_dir.glob("*.jsonl"))]
    files += [("log", path) for path in sorted(config.runtime.log_dir.glob("*.log"))]

    for kind, path in files:
        if path in active_logs or any(job_id in path.name for job_id in active_job_ids):
            skipped_active.append(str(path))
            continue
        if now - path.stat().st_mtime < min_age:
            skipped_young += 1
            continue
        archived.append(_archive_file(config, kind, path, dry_run))

    return ArchiveResult(archived, skipped_active, skipped_young, dry_run)
