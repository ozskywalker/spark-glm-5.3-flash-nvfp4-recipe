#!/usr/bin/env python3
"""Turn a vLLM torch-profiler Chrome trace into sub-forward-pass attribution.

Answers the three questions VALIDATION.md's "Request pipeline timing" section
left open, none of which /metrics can reach:

  1. Inside a forward pass, which kernel families actually consume the time?
     (MLA attention vs sparse indexer vs mamba vs EXL3 MoE vs norms vs comms)
  2. Inside an engine step, how much is GPU work vs CPU-side orchestration?
     Reported as GPU-busy vs wall span over the profiled window; a low busy
     fraction at batch=1 is the signature of launch-bound decode.
  3. Which individual kernels are worth a closer look, with their input shapes
     and (when --with-flops was on) FLOP estimates.

Kernel classification is name-pattern based and deliberately ordered
specific-before-generic; anything unmatched lands in "other" and is listed
explicitly rather than silently folded in, so a misclassification is visible
rather than hidden.

Usage:
  analyze_trace.py TRACE[.json|.json.gz] [--top 25] [--json OUT.json]
"""

import argparse
import collections
import gzip
import json
import os
import sys

# Ordered: first match wins. Patterns are lowercase substrings.
KERNEL_CATEGORIES = [
    ("comms",      ["nccl", "all_reduce", "allreduce", "all_gather", "reduce_scatter"]),
    ("moe_exl3",   ["exl3", "trellis", "mcg", "fat_gemm", "fatgemm", "moe", "expert"]),
    ("attention",  ["flash", "fmha", "mla", "attn", "attention", "paged"]),
    ("indexer",    ["indexer", "sparse_idx", "topk_idx", "select_idx"]),
    ("mamba_ssm",  ["mamba", "causal_conv1d", "conv1d", "selective_scan", "chunk_scan",
                    "chunk_state", "state_passing", "bmm_chunk", "ssm", "chunk_cumsum"]),
    ("norm",       ["rms_norm", "rmsnorm", "layernorm", "layer_norm", "fused_add_rms"]),
    ("quant",      ["quant", "dequant", "scaled_mm", "fp8", "nvfp4", "scale"]),
    ("sampling",   ["sample", "argmax", "softmax", "top_p", "top_k", "topp", "topk",
                    "multinomial", "logits", "penalt"]),
    ("gemm",       ["gemm", "cutlass", "nvjet", "cublas", "sgemm", "hgemm", "dot",
                    "matmul", "mm_", "addmm", "linear"]),
    ("embedding",  ["embedding", "index_select", "gather"]),
    ("memory",     ["memcpy", "memset", "copy", "cat_", "contiguous", "clone",
                    "pad", "slice", "narrow", "transpose", "permute", "reshape"]),
    ("elementwise", ["elementwise", "vectorized", "unrolled", "add", "mul", "silu",
                     "gelu", "swiglu", "activation", "sigmoid", "where", "masked"]),
]

# vLLM's own record_function scopes (VLLM_CUSTOM_SCOPES_FOR_PROFILING=1).
SCOPE_PREFIXES = ("gpu_model_runner:", "llm_engine step:", "schedule:",
                  "ngram_proposer_gpu:")


def load_trace(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def classify(name):
    low = name.lower()
    for cat, pats in KERNEL_CATEGORIES:
        if any(p in low for p in pats):
            return cat
    return "other"


def busy_span(intervals):
    """Union length of [start, end) intervals — GPU busy time without
    double-counting kernels overlapping across streams."""
    if not intervals:
        return 0.0, 0.0, 0.0
    intervals.sort()
    merged_total = 0.0
    cur_s, cur_e = intervals[0]
    lo, hi = cur_s, cur_e
    for s, e in intervals[1:]:
        if s > cur_e:
            merged_total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
        hi = max(hi, e)
    merged_total += cur_e - cur_s
    return merged_total, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    data = load_trace(args.trace)
    events = data.get("traceEvents", data if isinstance(data, list) else [])

    kernels = []            # (name, dur_us, ts, args)
    scopes = collections.defaultdict(lambda: {"count": 0, "dur_us": 0.0})
    steps = []
    cat_counts = collections.Counter()

    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        dur = float(ev.get("dur", 0) or 0)
        ts = float(ev.get("ts", 0) or 0)
        cat_counts[cat] += 1

        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append((name, dur, ts, ev.get("args", {})))
        elif cat in ("user_annotation", "cpu_op", "cuda_profiler_range"):
            if name.startswith(SCOPE_PREFIXES):
                s = scopes[name]
                s["count"] += 1
                s["dur_us"] += dur
            elif name.startswith("ProfilerStep"):
                steps.append((name, dur, ts))

    if not kernels:
        print("No GPU kernel events found. Event categories present:",
              dict(cat_counts.most_common(12)), file=sys.stderr)

    by_cat = collections.defaultdict(lambda: {"dur_us": 0.0, "count": 0})
    by_kernel = collections.defaultdict(lambda: {"dur_us": 0.0, "count": 0,
                                                 "cat": "", "shapes": None, "flops": 0.0})
    intervals = []
    for name, dur, ts, kargs in kernels:
        c = classify(name)
        by_cat[c]["dur_us"] += dur
        by_cat[c]["count"] += 1
        k = by_kernel[name]
        k["dur_us"] += dur
        k["count"] += 1
        k["cat"] = c
        if k["shapes"] is None:
            k["shapes"] = kargs.get("Input Dims") or kargs.get("input_dims")
        try:
            k["flops"] += float(kargs.get("flops", 0) or 0)
        except (TypeError, ValueError):
            pass
        intervals.append((ts, ts + dur))

    gpu_busy_us, t_lo, t_hi = busy_span(intervals)
    wall_us = t_hi - t_lo
    total_kernel_us = sum(v["dur_us"] for v in by_cat.values())

    out = {
        "trace": os.path.basename(args.trace),
        "n_kernel_events": len(kernels),
        "n_profiler_steps": len(steps),
        "wall_span_ms": round(wall_us / 1000, 3),
        "gpu_busy_ms": round(gpu_busy_us / 1000, 3),
        "gpu_busy_pct": round(100 * gpu_busy_us / wall_us, 2) if wall_us else None,
        "gpu_idle_ms": round((wall_us - gpu_busy_us) / 1000, 3),
        "sum_kernel_ms": round(total_kernel_us / 1000, 3),
        "categories": {},
        "scopes": {},
        "top_kernels": [],
    }

    print(f"\n=== {os.path.basename(args.trace)} ===")
    print(f"kernel events: {len(kernels)}   profiler steps: {len(steps)}")
    print(f"wall span: {wall_us/1000:.1f} ms   GPU busy: {gpu_busy_us/1000:.1f} ms "
          f"({100*gpu_busy_us/wall_us:.1f}%)   GPU idle: {(wall_us-gpu_busy_us)/1000:.1f} ms"
          if wall_us else "no span")

    print("\n--- GPU time by kernel family ---")
    print(f"{'category':<14}{'ms':>10}{'% kernel':>10}{'count':>9}{'us/call':>10}")
    for cat, v in sorted(by_cat.items(), key=lambda kv: -kv[1]["dur_us"]):
        pct = 100 * v["dur_us"] / total_kernel_us if total_kernel_us else 0
        out["categories"][cat] = {"ms": round(v["dur_us"] / 1000, 3),
                                  "pct_of_kernel_time": round(pct, 2),
                                  "count": v["count"],
                                  "us_per_call": round(v["dur_us"] / v["count"], 2)}
        print(f"{cat:<14}{v['dur_us']/1000:>10.2f}{pct:>10.1f}{v['count']:>9}"
              f"{v['dur_us']/v['count']:>10.1f}")

    if scopes:
        print("\n--- engine scopes (CPU wall, VLLM_CUSTOM_SCOPES_FOR_PROFILING) ---")
        print(f"{'scope':<46}{'ms':>10}{'count':>8}{'us/call':>10}")
        for name, v in sorted(scopes.items(), key=lambda kv: -kv[1]["dur_us"]):
            out["scopes"][name] = {"ms": round(v["dur_us"] / 1000, 3),
                                   "count": v["count"],
                                   "us_per_call": round(v["dur_us"] / v["count"], 2)}
            print(f"{name:<46}{v['dur_us']/1000:>10.2f}{v['count']:>8}"
                  f"{v['dur_us']/v['count']:>10.1f}")

    print(f"\n--- top {args.top} kernels ---")
    print(f"{'ms':>9}{'%':>7}{'cnt':>7}{'us/call':>9}  {'cat':<12} name")
    ranked = sorted(by_kernel.items(), key=lambda kv: -kv[1]["dur_us"])[:args.top]
    for name, v in ranked:
        pct = 100 * v["dur_us"] / total_kernel_us if total_kernel_us else 0
        entry = {"name": name, "cat": v["cat"], "ms": round(v["dur_us"] / 1000, 3),
                 "pct": round(pct, 2), "count": v["count"],
                 "us_per_call": round(v["dur_us"] / v["count"], 2)}
        if v["flops"]:
            entry["gflops_total"] = round(v["flops"] / 1e9, 2)
        if v["shapes"]:
            entry["shapes"] = str(v["shapes"])[:200]
        out["top_kernels"].append(entry)
        short = name if len(name) <= 78 else name[:75] + "..."
        print(f"{v['dur_us']/1000:>9.2f}{pct:>7.1f}{v['count']:>7}"
              f"{v['dur_us']/v['count']:>9.1f}  {v['cat']:<12} {short}")

    unmatched = [ (n, v) for n, v in by_kernel.items() if v["cat"] == "other" ]
    if unmatched:
        unmatched.sort(key=lambda kv: -kv[1]["dur_us"])
        print(f"\n--- unclassified kernels (top 10 of {len(unmatched)}) ---")
        for n, v in unmatched[:10]:
            print(f"{v['dur_us']/1000:>9.2f} ms  {n[:90]}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
