# gpu-slack-vllm-runner

A polite GPU slack-filling runner for an 8-GPU research server.

The runner periodically checks NVIDIA GPU state with NVML. If a GPU is idle, it launches a bounded vLLM offline generation job on that GPU. If a non-managed compute process appears on a GPU that is currently occupied by a managed filler job, the runner terminates the filler job so normal research workloads take priority.

## Intended use

Use this for opportunistic synthetic token generation during GPU slack time.

Default behavior is conservative:

- A GPU is eligible only when it has no foreign compute process.
- GPU utilization must be below `idle_policy.gpu_utilization_below_pct`, default 10%.
- The GPU must have at least `idle_policy.min_free_memory_mib` free, default 8000 MiB.
- One bounded vLLM job is launched per idle GPU by default.
- The default model is `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`.
- Generated JSONL is stored under `data/output/generations/`.
- The systemd timer runs one scheduling pass every 30 minutes.

## Repository layout

```text
.
├── configs/
│   ├── default.yaml                 # conservative 1-GPU-per-job config
│   ├── aggressive-low-util.yaml      # optional co-scheduling mode
│   └── tensor-parallel-2gpu.yaml     # example for 2-GPU vLLM jobs
├── data/
│   ├── input/prompts.jsonl           # synthetic generation prompts
│   └── output/generations/           # JSONL generation outputs
├── logs/                             # stdout/stderr logs per filler job
├── scripts/
│   ├── bootstrap.sh                  # uv environment setup
│   ├── install_systemd_user.sh        # install 30-minute user timer
│   └── uninstall_systemd_user.sh
├── src/gpu_slack_runner/
│   ├── cli.py                        # gpu-slack CLI
│   ├── config.py                     # YAML config loader
│   ├── gpu.py                        # NVML GPU monitoring
│   ├── scheduler.py                  # scheduling / preemption logic
│   ├── processes.py                  # process-tree handling
│   ├── state.py                      # managed job state files
│   └── jobs/vllm_generate.py          # bounded vLLM generation job
├── state/                            # managed job PID/state files
├── systemd/                          # user systemd templates
└── pyproject.toml                     # uv project definition
```

## Install

### 1. Create the scheduler environment

```bash
git clone <your-repo-url> gpu-slack-vllm-runner
cd gpu-slack-vllm-runner
./scripts/bootstrap.sh
```

This installs only the scheduler dependencies.

### 2. Install vLLM for real generation jobs

```bash
uv sync --extra vllm
```

This can be heavy because vLLM brings its CUDA/PyTorch stack. Use a fresh environment on the GPU server.

## Python Development

Use the same `uv` environment for local code checks. The Python code toolchain is intentionally small:

```bash
uv sync
uv run ruff check src tests
uv run ty check
uv run pytest
```

The Makefile wraps those commands:

```bash
make lint
make type
make test
make quality
```

`ruff` handles linting and import formatting checks. `ty` is the project type checker. `pytest` runs the unit tests.

### 3. Smoke-test GPU detection

```bash
uv run --no-sync gpu-slack --config configs/default.yaml status
```

### 4. Dry-run a scheduling pass

```bash
uv run --no-sync gpu-slack --config configs/default.yaml check --dry-run
```

### 5. Run persistently in tmux

```bash
tmux new -s gpu-slack
uv run --no-sync gpu-slack --config configs/default.yaml daemon --poll-interval-seconds 1800
```

Detach with `Ctrl-b d`. Reattach with `tmux attach -t gpu-slack`.

### 6. Run one real scheduling pass

```bash
uv run --no-sync gpu-slack --config configs/default.yaml check
```

### 7. Stop all managed filler jobs

```bash
uv run --no-sync gpu-slack --config configs/default.yaml stop
```

## Install 30-Minute Systemd Timer

The default timer runs one check every 30 minutes.

```bash
./scripts/install_systemd_user.sh configs/default.yaml
```

Check status:

```bash
systemctl --user status gpu-slack-runner.timer
journalctl --user -u gpu-slack-runner.service -f
```

Uninstall:

```bash
./scripts/uninstall_systemd_user.sh
```

For server reboots, user timers may require lingering:

```bash
loginctl enable-linger "$USER"
```

Ask your administrator before enabling this on shared systems.

## Optional Daemon Mode

The timer is usually enough for opportunistic token generation. For a visible persistent process during bring-up, run daemon mode in tmux with the same 30-minute cadence:

```bash
uv run --no-sync gpu-slack --config configs/default.yaml daemon --poll-interval-seconds 1800
```

Use `tmux attach -t gpu-slack` to inspect it later, or install the user systemd timer once the dry-run output looks right.

## Config Guide

Important fields in `configs/default.yaml`:

The default Nemotron workload follows NVIDIA's offline-compatible recommendations: FlashInfer FP4 MoE environment flags, `trust_remote_code`, `max_model_len=262144`, `max_num_seqs=8`, `kv_cache_dtype=fp8`, `temperature=1.0`, and `top_p=1.0`. Server-only flags such as port, served model name, and tool-call parser are not used by this offline JSONL generator.


```yaml
idle_policy:
  gpu_utilization_below_pct: 10
  min_free_memory_mib: 8000
  require_no_foreign_compute_process: true
  allow_low_utilization_with_foreign_process: false

job:
  gpus_per_job: 1
  max_jobs: 8
  command:
    - uv
    - run
    - --no-sync
    - gpu-slack-vllm-generate
    - --model
    - nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
    - --time-budget-min
    - "50"
    - --max-tokens
    - "131072"
    - --temperature
    - "1.0"
    - --top-p
    - "1.0"
    - --tensor-parallel-size
    - "{num_gpus}"
    - --max-model-len
    - "262144"
    - --max-num-seqs
    - "8"
    - --kv-cache-dtype
    - fp8
    - --trust-remote-code
```

Placeholders available in `job.command` and `job.environment`:

- `{gpu}` / `{gpu0}`: first GPU index in the assigned group
- `{gpus}`: comma-separated GPU list
- `{num_gpus}`: number of GPUs assigned to the job
- `{job_id}`: unique managed job ID
- `{repo_root}`: repo root path
- `{output_dir}`: generation output directory
- `{log_dir}`: log directory
- `{state_dir}`: state directory
- `{config_path}`: config path

## Prompt Inputs

`data/input/prompts.jsonl` is the default prompt source. Each line is a JSON object with a required `prompt` string. Extra fields such as `source`, `id`, and `categories` are allowed and ignored by the generator.

The checked-in prompt file contains CUDA, Triton, FlashAttention, BLAS, and EDA GPU-kernel prompts, plus converted CVDP SystemVerilog RTL code-generation tasks.

## Output Format

Each vLLM filler job writes a JSONL file under `data/output/generations/`.

Example records:

```jsonl
{"type":"job_start","job_id":"...","model":"...","visible_gpus":"0","batch_size":16}
{"type":"generation","job_id":"...","batch_idx":0,"prompt":"...","text":"..."}
{"type":"job_end","job_id":"...","reason":"budget_or_limit","total_outputs":512}
```

## Operational Recommendations

1. Start with `check --dry-run` for several hours before allowing real jobs.
2. Keep `require_no_foreign_compute_process: true` unless the server users explicitly agree to co-scheduling.
3. Set `--gpu-memory-utilization` below 0.9 so vLLM does not reserve all memory.
4. Use bounded jobs, for example 50 minutes, so repeated checks never accumulate stale jobs.
5. For shared systems, document that these are low-priority filler jobs and can be killed at any time.
6. Use daemon mode in tmux during bring-up; use the systemd timer once the dry-run output is boring.

## Common Commands

```bash
# Show status
uv run --no-sync gpu-slack --config configs/default.yaml status

# Dry-run scheduler
uv run --no-sync gpu-slack --config configs/default.yaml check --dry-run

# Start filler jobs on currently idle GPUs
uv run --no-sync gpu-slack --config configs/default.yaml check

# Stop all managed filler jobs
uv run --no-sync gpu-slack --config configs/default.yaml stop

# Run more frequent checks
uv run --no-sync gpu-slack --config configs/default.yaml daemon --poll-interval-seconds 1800
```

## Notes

- This runner manages only jobs it started and recorded under `state/`.
- It does not kill other users' jobs.
- It does not require root if NVML process visibility is available to the user.
- If your system uses Slurm, Kubernetes, or another cluster scheduler, integrate this as a low-priority/preemptible queue job instead of bypassing the scheduler.
