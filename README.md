# gpu-slack-vllm-runner

A polite GPU slack-filling runner for an 8-GPU research server.

The runner periodically checks NVIDIA GPU state with NVML. If a GPU is idle, it launches a bounded vLLM offline generation job on that GPU. If a non-managed compute process appears on a GPU that is currently occupied by a managed filler job, the runner terminates the filler job so normal research workloads take priority.

## Intended use

Use this for opportunistic synthetic token generation during GPU slack time.

Default behavior is conservative:

- A GPU is eligible only when it has no foreign compute process and enough free memory.
- One filler job is launched per idle GPU.
- Each filler job runs for a bounded time budget, default 50 minutes.
- The hourly systemd timer runs one scheduling pass every hour.
- A daemon mode is also provided for more frequent preemption checks.

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
│   ├── install_systemd_user.sh        # install hourly user timer
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

### 3. Smoke-test GPU detection

```bash
uv run --no-sync gpu-slack --config configs/default.yaml status
```

### 4. Dry-run a scheduling pass

```bash
uv run --no-sync gpu-slack --config configs/default.yaml check --dry-run
```

### 5. Run one real scheduling pass

```bash
uv run --no-sync gpu-slack --config configs/default.yaml check
```

### 6. Stop all managed filler jobs

```bash
uv run --no-sync gpu-slack --config configs/default.yaml stop
```

## Install hourly systemd timer

The default timer runs one check every hour.

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

## Optional daemon mode

The hourly timer matches the original requirement, but it only observes conflicts once per hour. For more polite preemption, run daemon mode with a 5-minute interval:

```bash
uv run --no-sync gpu-slack --config configs/default.yaml daemon --poll-interval-seconds 300
```

You can create a separate long-running systemd service for daemon mode if your lab prefers faster preemption.

## Config guide

Important fields in `configs/default.yaml`:

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
    - Qwen/Qwen2.5-1.5B-Instruct
    - --time-budget-min
    - "50"
    - --tensor-parallel-size
    - "{num_gpus}"
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

## Output format

Each vLLM filler job writes a JSONL file under `data/output/generations/`.

Example records:

```jsonl
{"type":"job_start","job_id":"...","model":"...","visible_gpus":"0","batch_size":16}
{"type":"generation","job_id":"...","batch_idx":0,"prompt":"...","text":"..."}
{"type":"job_end","job_id":"...","reason":"budget_or_limit","total_outputs":512}
```

## Operational recommendations

1. Start with `check --dry-run` for several hours before allowing real jobs.
2. Keep `require_no_foreign_compute_process: true` unless the server users explicitly agree to co-scheduling.
3. Set `--gpu-memory-utilization` below 0.9 so vLLM does not reserve all memory.
4. Use bounded jobs, for example 50 minutes, so the hourly timer never accumulates stale jobs.
5. For shared systems, document that these are low-priority filler jobs and can be killed at any time.
6. Consider running the daemon mode every 5 minutes if normal user jobs must preempt filler jobs quickly.

## Common commands

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
uv run --no-sync gpu-slack --config configs/default.yaml daemon --poll-interval-seconds 300
```

## Notes

- This runner manages only jobs it started and recorded under `state/`.
- It does not kill other users' jobs.
- It does not require root if NVML process visibility is available to the user.
- If your system uses Slurm, Kubernetes, or another cluster scheduler, integrate this as a low-priority/preemptible queue job instead of bypassing the scheduler.
