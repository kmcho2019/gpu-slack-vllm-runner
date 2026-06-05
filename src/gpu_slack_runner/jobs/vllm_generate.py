"""Bounded vLLM offline generation filler job.

This module is intentionally lazy-imported so the scheduler can run without vLLM installed.
Install the optional extra on GPU hosts with:

    uv sync --extra vllm
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_STOP_REQUESTED = False


def _handle_stop(signum: int, frame: object) -> None:  # noqa: ARG001
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)


def _read_prompts(path: Path) -> list[str]:
    """Read all prompts from a JSONL file with a required prompt field."""

    assert path.exists(), f"Missing prompt file: {path}"
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj["prompt"]
            assert isinstance(prompt, str)
            assert prompt
            prompts.append(prompt)
    assert prompts, f"No prompts found in {path}"
    return prompts


def _cycle_batches(prompts: list[str], batch_size: int) -> Iterator[list[str]]:
    """Yield fixed-size prompt batches indefinitely."""

    idx = 0
    n = len(prompts)
    while True:
        batch = [prompts[(idx + offset) % n] for offset in range(batch_size)]
        idx = (idx + batch_size) % n
        yield batch


def _import_vllm() -> tuple[Any, Any]:
    try:
        from vllm import LLM, SamplingParams  # type: ignore[import-not-found, unused-ignore]
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is not installed. Install with `uv sync --extra vllm` or `uv pip install vllm`."
        ) from exc
    return LLM, SamplingParams


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the vLLM generation job."""

    parser = argparse.ArgumentParser(description="Run bounded vLLM synthetic generation as GPU filler.")
    parser.add_argument("--model", required=True, help="HF model name or local model path.")
    parser.add_argument("--prompts", default="data/input/prompts.jsonl", help="Prompt JSONL path.")
    parser.add_argument("--output-dir", default="data/output/generations", help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=16, help="Prompts per vLLM batch.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max generated tokens per prompt.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling top-p.")
    parser.add_argument("--time-budget-min", type=float, default=50.0, help="Exit after this many minutes.")
    parser.add_argument("--max-batches", type=int, default=0, help="Optional hard cap; 0 means no cap.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument("--max-model-len", type=int, default=0, help="vLLM max model length; 0 uses model default.")
    parser.add_argument("--max-num-seqs", type=int, default=0, help="vLLM max concurrent sequences; 0 uses vLLM default.")
    parser.add_argument("--kv-cache-dtype", default="auto", help="vLLM KV cache dtype, e.g. auto or fp8.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.82,
        help="vLLM GPU memory utilization fraction. Keep below 1.0 for politeness.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to vLLM. Only use with trusted models.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="vLLM dtype, e.g. auto, half, bfloat16, float16.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the bounded vLLM generation workload."""

    _install_signal_handlers()
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.time_budget_min <= 0:
        raise ValueError("--time-budget-min must be > 0")

    LLM, SamplingParams = _import_vllm()
    prompts = _read_prompts(Path(args.prompts))

    job_id = os.environ.get("GPU_SLACK_RUNNER_JOB_ID", f"manual-{int(time.time())}")
    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    output_path = Path(args.output_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}-{job_id}.jsonl"

    llm_kwargs: dict[str, Any] = {}
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.max_num_seqs:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.kv_cache_dtype != "auto":
        llm_kwargs["kv_cache_dtype"] = args.kv_cache_dtype

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        **llm_kwargs,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    start = time.time()
    deadline = start + args.time_budget_min * 60.0
    batch_iter = _cycle_batches(prompts, args.batch_size)
    batch_idx = 0
    total_outputs = 0

    metadata = {
        "type": "job_start",
        "job_id": job_id,
        "model": args.model,
        "visible_gpus": visible_gpus,
        "tensor_parallel_size": args.tensor_parallel_size,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "kv_cache_dtype": args.kv_cache_dtype,
        "time": time.time(),
    }
    _write_jsonl(output_path, [metadata])

    while not _STOP_REQUESTED and time.time() < deadline:
        if args.max_batches and batch_idx >= args.max_batches:
            break
        batch = next(batch_iter)
        outputs = llm.generate(batch, sampling_params)
        records: list[dict[str, Any]] = []
        now = time.time()
        for prompt, output in zip(batch, outputs, strict=False):
            text = output.outputs[0].text if output.outputs else ""
            records.append(
                {
                    "type": "generation",
                    "job_id": job_id,
                    "batch_idx": batch_idx,
                    "model": args.model,
                    "visible_gpus": visible_gpus,
                    "prompt": prompt,
                    "text": text,
                    "finish_reason": output.outputs[0].finish_reason if output.outputs else None,
                    "created_at": now,
                }
            )
        _write_jsonl(output_path, records)
        batch_idx += 1
        total_outputs += len(records)

    _write_jsonl(
        output_path,
        [
            {
                "type": "job_end",
                "job_id": job_id,
                "reason": "signal" if _STOP_REQUESTED else "budget_or_limit",
                "batches": batch_idx,
                "total_outputs": total_outputs,
                "elapsed_seconds": round(time.time() - start, 3),
                "time": time.time(),
            }
        ],
    )
    print(f"Wrote {total_outputs} generations to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
