#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed or not on PATH. Install uv first, then rerun this script." >&2
  exit 1
fi

# Scheduler-only environment. Fast and enough for status/check without vLLM.
uv sync

echo "Scheduler environment ready."
echo "For real vLLM filler jobs, run: uv sync --extra vllm"
