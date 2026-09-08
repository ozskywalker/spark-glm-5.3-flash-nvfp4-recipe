"""Bound a vLLM worker's device memory, and name whatever crosses the bound.

Adapted near-verbatim from mmastrac/glm-5.3-flash-4x-gx10's
`image/patches/spark_mem_trace.py` (MIT), found while investigating this
project's own long-unsolved "silent kill" crash signature (a worker dies
with zero CUDA traceback and zero dmesg entry -- see recipes/VALIDATION.md's
crash entries and the `memfree_memavailable_gap` memory note). Their own
rationale, unchanged here because it applies identically on our hardware:

vLLM never calls set_per_process_memory_fraction, so nothing bounds a
worker. gpu_memory_utilization only sizes the KV pool; every allocation
outside it -- the sparse indexer scores chunk x context/kpool, and grows
with the session -- comes from host memory, because on GB10 there is no
separate VRAM to exhaust first. The node stops forking and the watchdog
resets it, and nothing in any log says which allocation did it.

A fraction turns that into torch.OutOfMemoryError, so one request fails and
the node keeps serving. Arming the trace as well turns the same moment into
a stack naming the call site, since the observer runs in the allocating
thread at the point of failure.

To find a culprit rather than survive it, set the fraction BELOW the level
that hurts: the first allocation past it is reported and the node stays up.

Not adapted: their file targets a 4-node NVFP4/Marlin topology; this one is
a straight copy for our 2-node EXL3/TP2 setup, since the mechanism itself
(a per-process memory-fraction cap plus a CUDA-allocator OOM observer) is
topology- and quant-backend-agnostic -- it hooks torch's own allocator, not
anything EXL3- or NVFP4-specific.

Env, all empty by default so an unset deployment behaves like upstream:
  TORCH_MEM_FRACTION       ceiling as a fraction of device memory, e.g. 0.90
  TORCH_MEM_TRACE_DIR      directory for the snapshot; also arms the recorder
  TORCH_MEM_TRACE_ENTRIES  allocation records kept (default 100000)
  TORCH_MEM_STATS_DIR      directory for the per-worker counter file
  TORCH_MEM_STATS_INTERVAL seconds between samples (default 15)
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback

import torch

from vllm.logger import init_logger

logger = init_logger(f"vllm.{__name__}")

_GiB = 1 << 30
_reported = threading.Event()


def _observer(trace_dir: str):
    def on_oom(*args) -> None:
        # (device, alloc, device_allocated, device_free) upstream, taken
        # positionally so a signature change degrades to a bare report.
        size = args[1] if len(args) > 1 else -1
        free = args[3] if len(args) > 3 else -1
        logger.error(
            "glm53-oom-observer: allocation of %.2f GiB crossed the cap (%.2f GiB free)",
            size / _GiB,
            free / _GiB,
        )
        logger.error(
            "glm53-oom-observer: allocating stack:\n%s",
            "".join(traceback.format_stack()[:-1]),
        )
        if _reported.is_set():
            return
        _reported.set()
        path = os.path.join(trace_dir, f"mem-snapshot-pid{os.getpid()}.pickle")
        try:
            torch.cuda.memory._dump_snapshot(path)
            logger.error("glm53-oom-observer: wrote %s (open at pytorch.org/memory_viz)", path)
        except Exception as exc:
            logger.error("glm53-oom-observer: snapshot failed: %s", exc)

    return on_oom


def arm(device) -> None:
    """Apply the cap and trace this worker asks for. Safe to call once per worker."""
    frac = os.environ.get("TORCH_MEM_FRACTION", "").strip()
    if frac:
        torch.cuda.set_per_process_memory_fraction(float(frac), device)
        logger.info("glm53-oom-observer: worker capped at %s of device memory", frac)

    _start_reporter()

    trace_dir = os.environ.get("TORCH_MEM_TRACE_DIR", "").strip()
    if not trace_dir:
        return
    os.makedirs(trace_dir, exist_ok=True)
    entries = int(os.environ.get("TORCH_MEM_TRACE_ENTRIES", "").strip() or "100000")
    # Recording keeps a stack per allocation, so it costs both time and memory;
    # entries bounds the second.
    torch.cuda.memory._record_memory_history(max_entries=entries)
    torch._C._cuda_attach_out_of_memory_observer(_observer(trace_dir))
    logger.info(
        "glm53-oom-observer: memory trace armed, %d entries, snapshots to %s",
        entries,
        trace_dir,
    )


def _report(path: str, interval: float) -> None:
    while True:
        try:
            free, total = torch.cuda.mem_get_info()
            tmp = f"{path}.tmp"
            with open(tmp, "w") as fh:
                json.dump(
                    {
                        "pid": os.getpid(),
                        "ts": time.time(),
                        "allocated": torch.cuda.memory_allocated(),
                        "reserved": torch.cuda.memory_reserved(),
                        "max_reserved": torch.cuda.max_memory_reserved(),
                        "device_free": free,
                        "device_total": total,
                    },
                    fh,
                )
            os.replace(tmp, path)
        except Exception:
            pass
        time.sleep(interval)


def _start_reporter() -> None:
    """Publish this worker's allocator counters for another process to read.

    The counters are per process and every reader is another one, so a
    worker that does not write them cannot be asked. What the pair of
    numbers answers is which side is growing: reserved is what torch holds,
    and the host's MemAvailable is what is left, so the gap between them is
    everything else -- NCCL, cuBLAS, the JIT caches. This is the same
    MemAvailable this project's own memfree_memavailable_gap investigation
    reads from /proc/meminfo; this reporter is the per-process half of that
    picture.
    """
    stats_dir = os.environ.get("TORCH_MEM_STATS_DIR", "").strip()
    if not stats_dir:
        return
    os.makedirs(stats_dir, exist_ok=True)
    interval = float(os.environ.get("TORCH_MEM_STATS_INTERVAL", "").strip() or "15")
    path = os.path.join(stats_dir, f"mem-stats-pid{os.getpid()}.json")
    threading.Thread(target=_report, args=(path, interval), daemon=True).start()
    logger.info(
        "glm53-oom-observer: publishing allocator counters to %s every %.0fs",
        path,
        interval,
    )
