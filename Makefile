.PHONY: sync sync-vllm status dry-run check stop lint test

sync:
	uv sync

sync-vllm:
	uv sync --extra vllm

status:
	uv run --no-sync gpu-slack --config configs/default.yaml status

dry-run:
	uv run --no-sync gpu-slack --config configs/default.yaml check --dry-run

check:
	uv run --no-sync gpu-slack --config configs/default.yaml check

stop:
	uv run --no-sync gpu-slack --config configs/default.yaml stop

lint:
	uv run ruff check src tests

test:
	uv run pytest
