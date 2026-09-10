#!/usr/bin/env python3
"""Install the OOM-observer/memory-cap runtime and wire it into worker init.

See overlay/glm53_oom_observer_runtime.py for the full mechanism and
rationale (adapted from mmastrac/glm-5.3-flash-4x-gx10's
`spark_mem_trace.py`/`worker_memory_cap.py`, MIT). Two effects, both opt-in
via env var and both no-ops when unset -- an unpatched-behaving deployment
that happens to carry this file is byte-for-byte upstream at runtime:

  TORCH_MEM_FRACTION   caps this worker at a fraction of device memory via
                       torch.cuda.set_per_process_memory_fraction, so an
                       over-budget allocation raises torch.OutOfMemoryError
                       (one request fails, the node stays up) instead of
                       silently exhausting host memory until the node stops
                       forking and the watchdog resets it with no log trail.
  TORCH_MEM_TRACE_DIR  arms torch's allocator OOM observer + memory-history
                       recorder, so the moment either the above cap or CUDA's
                       own allocator raises, a full allocating-thread stack
                       trace and a memory_viz-loadable snapshot are written,
                       instead of the process dying with zero CUDA traceback
                       and zero dmesg entry -- this project's long-standing
                       "silent kill" crash signature (see recipes/
                       VALIDATION.md's crash entries).

Applied by copying the runtime module to dist-packages root (so a bare
`import glm53_oom_observer_runtime` resolves from any worker process,
matching how overlay/exl3.py and every other top-level overlay module in
this build are already reached) and inserting one import + one call right
after gpu_worker.py picks its CUDA device -- the earliest point a worker's
device is known, and well before the memory profile / KV-cache sizing pass
that gpu_memory_utilization governs, so a fraction set here does not fight
that pass, it bounds what's left outside it.

Conventions match overlay/patch_indexer_workspace.py: pinned ANCHOR, MARK
sentinel, verified_state(), idempotent prepare(), atomic replace, pyc clear,
fail-closed on drift, --preflight CLI mode.

Usage::

    python3 patch_oom_observer.py              # apply
    python3 patch_oom_observer.py --preflight  # validate anchors only
"""
from __future__ import annotations

import os
import shutil
import stat
import sys
from pathlib import Path


RUNTIME_SRC = Path(
    os.environ.get("GLM53_OOM_OBSERVER_RUNTIME_SRC", "/opt/glm53/glm53_oom_observer_runtime.py")
)
RUNTIME_DST = Path(
    os.environ.get(
        "GLM53_OOM_OBSERVER_RUNTIME_DST",
        "/usr/local/lib/python3.12/dist-packages/glm53_oom_observer_runtime.py",
    )
)
WORKER = Path(
    os.environ.get(
        "GLM53_GPU_WORKER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py",
    )
)

MARK = "            # [glm53-oom-observer] Arm before the memory profile /\n"

ANCHOR = (
    '            self.device = torch.device(f"cuda:{visible_device_index}")\n'
    "            torch.accelerator.set_device_index(self.device)\n"
)

PATCHED = (
    '            self.device = torch.device(f"cuda:{visible_device_index}")\n'
    "            torch.accelerator.set_device_index(self.device)\n"
    "            # [glm53-oom-observer] Arm before the memory profile /\n"
    "            # KV-cache sizing pass so TORCH_MEM_FRACTION bounds what's\n"
    "            # left OUTSIDE gpu_memory_utilization's own accounting,\n"
    "            # rather than fighting it. No-op unless TORCH_MEM_FRACTION\n"
    "            # or TORCH_MEM_TRACE_DIR is set -- see overlay/glm53_oom_\n"
    "            # observer_runtime.py.\n"
    "            import glm53_oom_observer_runtime\n"
    "\n"
    "            glm53_oom_observer_runtime.arm(self.device)\n"
)


def verified_state(text: str) -> bool:
    return text.count(MARK) == 1 and text.count(PATCHED) == 1


def prepare(source: str) -> tuple[str, str]:
    """Idempotent, fail-closed. Returns ``(text, action)``."""
    marks = source.count(MARK)
    if marks:
        if marks != 1 or not verified_state(source):
            raise ValueError(
                "partial/inconsistent oom-observer patch "
                f"(marks={marks}) -- refusing to touch a half-patched file"
            )
        return source, "already present"

    n_anchor = source.count(ANCHOR)
    if n_anchor != 1:
        raise ValueError(
            f"pinned gpu_worker device-init anchor drifted (found {n_anchor}, "
            "expected 1) -- re-derive the patch after a base-image bump"
        )
    out = source.replace(ANCHOR, PATCHED, 1)
    if not verified_state(out):
        raise ValueError("oom-observer post-patch verification failed")
    return out, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-oom-observer.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if not cache.is_dir():
        return
    for pyc in cache.glob(f"{target.stem}*.pyc"):
        pyc.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    preflight_only = "--preflight" in argv[1:]

    if not WORKER.is_file():
        raise SystemExit(f"missing {WORKER}")
    source = WORKER.read_text()
    try:
        patched, action = prepare(source)
    except ValueError as exc:
        raise SystemExit(f"oom-observer preflight failed: {exc}") from exc
    compile(patched, str(WORKER), "exec")

    if preflight_only:
        print(f"{WORKER.name}: oom-observer preflight OK ({action})")
        if not RUNTIME_DST.is_file() and not RUNTIME_SRC.is_file():
            print(
                f"  note: neither {RUNTIME_DST} nor {RUNTIME_SRC} exist yet "
                "(expected pre-build; the COPY step installs the runtime "
                "module before this patch runs)"
            )
        return 0

    if not RUNTIME_SRC.is_file():
        raise SystemExit(
            f"missing {RUNTIME_SRC} -- add a Dockerfile COPY step for "
            "overlay/glm53_oom_observer_runtime.py before running this patch"
        )
    RUNTIME_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(RUNTIME_SRC, RUNTIME_DST)

    if patched != source:
        replace_file(WORKER, patched)
        clear_pyc(WORKER)
    print(f"{WORKER.name}: oom-observer {action}; runtime installed at {RUNTIME_DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
