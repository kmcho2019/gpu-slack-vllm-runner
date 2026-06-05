"""NVML-backed GPU discovery and monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class GpuProcess:
    """One compute process reported by NVML."""

    pid: int
    used_memory_mib: int | None = None


@dataclass(frozen=True, slots=True)
class GpuStatus:
    """Current state of one NVIDIA GPU."""

    index: int
    uuid: str
    name: str
    utilization_gpu_pct: int
    utilization_memory_pct: int
    memory_used_mib: int
    memory_total_mib: int
    compute_processes: list[GpuProcess] = field(default_factory=list)

    @property
    def memory_free_mib(self) -> int:
        """Return free framebuffer memory in MiB."""

        return self.memory_total_mib - self.memory_used_mib

    @property
    def compute_pids(self) -> set[int]:
        """Return compute process IDs for this GPU."""

        return {p.pid for p in self.compute_processes}


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _get_compute_processes(pynvml: Any, handle: Any) -> list[GpuProcess]:
    """Collect compute processes with compatibility across NVML bindings."""

    process_fns = [
        "nvmlDeviceGetComputeRunningProcesses_v3",
        "nvmlDeviceGetComputeRunningProcesses_v2",
        "nvmlDeviceGetComputeRunningProcesses",
    ]
    last_error: Exception | None = None
    for fn_name in process_fns:
        fn = getattr(pynvml, fn_name, None)
        if fn is None:
            continue
        try:
            raw_processes = fn(handle)
            processes: list[GpuProcess] = []
            for proc in raw_processes:
                used = getattr(proc, "usedGpuMemory", None)
                used_mib = None if used is None or used < 0 else int(used // (1024 * 1024))
                processes.append(GpuProcess(pid=int(proc.pid), used_memory_mib=used_mib))
            return processes
        except Exception as exc:  # pragma: no cover - depends on installed driver/NVML version
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return []


def collect_gpu_status() -> list[GpuStatus]:
    """Return status for all NVIDIA GPUs visible to NVML."""

    try:
        import pynvml  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - import check is environment dependent
        raise RuntimeError(
            "pynvml is unavailable. Install the project with `uv sync` or install nvidia-ml-py."
        ) from exc

    try:
        pynvml.nvmlInit()
    except Exception as exc:  # pragma: no cover - depends on host GPU driver
        raise RuntimeError("NVML initialization failed. Is the NVIDIA driver installed?") from exc

    statuses: list[GpuStatus] = []
    try:
        count = int(pynvml.nvmlDeviceGetCount())
        for idx in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                uuid = _decode(pynvml.nvmlDeviceGetUUID(handle))
            except Exception:  # pragma: no cover - depends on NVML version
                uuid = f"GPU-{idx}"
            try:
                name = _decode(pynvml.nvmlDeviceGetName(handle))
            except Exception:  # pragma: no cover - depends on NVML version
                name = "unknown"

            statuses.append(
                GpuStatus(
                    index=idx,
                    uuid=uuid,
                    name=name,
                    utilization_gpu_pct=int(util.gpu),
                    utilization_memory_pct=int(util.memory),
                    memory_used_mib=int(mem.used // (1024 * 1024)),
                    memory_total_mib=int(mem.total // (1024 * 1024)),
                    compute_processes=_get_compute_processes(pynvml, handle),
                )
            )
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # pragma: no cover - best effort cleanup
            pass
    return statuses
