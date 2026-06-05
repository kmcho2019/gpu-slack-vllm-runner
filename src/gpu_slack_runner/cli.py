"""Command line interface for gpu-slack-vllm-runner."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from gpu_slack_runner.config import load_config
from gpu_slack_runner.scheduler import check_once, status_json, stop_all


def _default_config() -> str:
    return "configs/default.yaml"


def _print_check_summary(result: object) -> None:
    # Kept deliberately simple so the CLI has no rich/textual dependency.
    started = getattr(result, "started_jobs")
    stopped = getattr(result, "stopped_jobs")
    cleaned = getattr(result, "cleaned_jobs")
    decisions = getattr(result, "decisions")
    dry = getattr(result, "dry_run")

    print(f"dry_run={dry}")
    print(f"started={len(started)} stopped={len(stopped)} cleaned={len(cleaned)}")
    for decision in decisions:
        state = "IDLE" if decision.idle else "BUSY"
        print(
            f"gpu={decision.index:<2} {state:<4} util={decision.utilization_gpu_pct:>3}% "
            f"free_mem={decision.memory_free_mib:>7} MiB reason={decision.reason}"
        )
    for job in started:
        print(f"START job_id={job.job_id} pid={job.pid} gpus={job.gpus} cmd={' '.join(job.command)}")
    for job_id in stopped:
        print(f"STOP job_id={job_id}")
    for job_id in cleaned:
        print(f"CLEAN job_id={job_id}")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""

    parser = argparse.ArgumentParser(
        prog="gpu-slack",
        description="Launch polite filler jobs on idle NVIDIA GPUs.",
    )
    parser.add_argument(
        "--config",
        default=_default_config(),
        help="Path to YAML config. Defaults to configs/default.yaml.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Show GPU and managed-job status as JSON.")
    status.add_argument("--json", action="store_true", help="Accepted for readability; output is JSON.")

    check = sub.add_parser("check", help="Run one scheduling pass.")
    check.add_argument("--dry-run", action="store_true", help="Do not start or stop any process.")

    daemon = sub.add_parser("daemon", help="Run repeated scheduling checks.")
    daemon.add_argument("--dry-run", action="store_true", help="Do not start or stop any process.")
    daemon.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=None,
        help="Override runtime.poll_interval_seconds from config.",
    )

    stop = sub.add_parser("stop", help="Stop all managed filler jobs.")
    stop.add_argument("--dry-run", action="store_true", help="Only print which jobs would be stopped.")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(Path(args.config))

    if args.command == "status":
        print(status_json(config))
        return 0

    if args.command == "check":
        result = check_once(config, dry_run=args.dry_run)
        _print_check_summary(result)
        return 0

    if args.command == "daemon":
        interval = args.poll_interval_seconds or config.runtime.poll_interval_seconds
        if interval < 60:
            print("Refusing poll interval < 60 seconds; use >=60 to avoid scheduler noise.", file=sys.stderr)
            return 2
        while True:
            try:
                result = check_once(config, dry_run=args.dry_run)
                _print_check_summary(result)
            except Exception as exc:  # pragma: no cover - daemon resilience
                print(f"ERROR: {exc}", file=sys.stderr)
            sys.stdout.flush()
            sys.stderr.flush()
            time.sleep(interval)

    if args.command == "stop":
        stopped = stop_all(config, dry_run=args.dry_run)
        print(f"stopped={len(stopped)} dry_run={args.dry_run}")
        for job_id in stopped:
            print(f"STOP job_id={job_id}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
