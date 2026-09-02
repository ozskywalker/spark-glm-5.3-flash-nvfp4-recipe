#!/usr/bin/env python3
"""Aggregate a py-spy speedscope profile into "what is this process doing".

The question this answers is narrower than a flamegraph: at batch=1, is the
serving process burning CPU on real orchestration work, or is it parked in a
wait (GPU sync, IPC, spin-wait)? Those look identical in a %CPU number and
call for opposite fixes, so every frame gets bucketed into wait-vs-work and
the split is reported up front.

Self-time is attributed to the leaf frame of each sample; the bucket for a
sample is decided by scanning its whole stack for the most specific match, so
a cuda synchronize called from inside the sampler is counted as a GPU wait
rather than as sampler work.

Usage: analyze_pyspy.py PROFILE.speedscope.json [--top 25] [--json OUT.json]
"""

import argparse
import collections
import json


# Ordered specific -> generic; first match against any frame in the stack wins.
BUCKETS = [
    # Deliberately narrow: a bare "wait" or "synchronize" substring pulls in the
    # shm_broadcast spin loop and mislabels an IPC spin as a GPU stall, which
    # inverts the conclusion. Match only genuinely CUDA-specific frames here.
    ("gpu_wait",    ["cuda.synchronize", "cuda_synchronize", "event.synchronize",
                     "stream.synchronize", "cudastreamsynchronize",
                     "torch.cuda.synchronize", "cuda/streams.py"]),
    ("ipc_spinwait", ["shm_broadcast", "acquire_read", "acquire_write",
                      "recv_multipart", "socket.recv", "dequeue", "zmq"]),
    ("sleep_idle",  ["time.sleep", "sched_yield", "select.select", "epoll",
                     "selectors.py", "queue.py", "threading.py"]),
    ("nccl_comm",   ["nccl", "all_reduce", "allreduce", "distributed"]),
    ("spec_decode", ["spec_decode", "propose", "drafter", "eagle", "mtp", "rejection_sample"]),
    ("sampler",     ["sampler", "sample_", "logits", "topk", "top_p", "penalt"]),
    ("model_fwd",   ["model_runner", "execute_model", "forward", "graph.replay",
                     "cudagraph", "_call_impl", "module."]),
    ("scheduler",   ["scheduler", "schedule", "kv_cache_manager", "block_pool",
                     "allocate_slots", "coordinator"]),
    ("detokenize",  ["detokenize", "tokenizer", "decode_", "incremental"]),
    ("serialize",   ["json", "serial", "encode", "pickle", "msgpack"]),
    ("output_proc", ["output_processor", "process_outputs", "request_output",
                     "stream", "queue.put"]),
]


def load(path):
    with open(path) as f:
        return json.load(f)


def bucket_for(stack_names):
    joined = [n.lower() for n in stack_names]
    for bucket, pats in BUCKETS:
        for n in joined:
            if any(p in n for p in pats):
                return bucket
    return "other_python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    data = load(args.profile)
    frames = data["shared"]["frames"]

    def fname(i):
        f = frames[i]
        file = (f.get("file") or "").split("/")[-1]
        return f"{f.get('name','?')} ({file}:{f.get('line','?')})"

    def raw_name(i):
        f = frames[i]
        return f"{f.get('name','')} {f.get('file','')}"

    self_time = collections.Counter()
    bucket_time = collections.Counter()
    total = 0.0

    for prof in data.get("profiles", []):
        samples = prof.get("samples", [])
        weights = prof.get("weights", [1.0] * len(samples))
        for stack, w in zip(samples, weights):
            if not stack:
                continue
            total += w
            self_time[fname(stack[-1])] += w
            bucket_time[bucket_for([raw_name(i) for i in stack])] += w

    unit = data.get("profiles", [{}])[0].get("unit", "none")
    out = {"profile": args.profile, "unit": unit, "total_weight": round(total, 3),
           "buckets": {}, "top_self": []}

    print(f"\n=== {args.profile} ===")
    print(f"total sampled weight: {total:.2f} ({unit})\n")
    print("--- where the process actually is (stack-classified) ---")
    print(f"{'bucket':<16}{'weight':>12}{'%':>9}")
    waits = 0.0
    for b, w in bucket_time.most_common():
        pct = 100 * w / total if total else 0
        out["buckets"][b] = {"weight": round(w, 3), "pct": round(pct, 2)}
        if b in ("gpu_wait", "ipc_spinwait", "sleep_idle", "nccl_comm"):
            waits += w
        print(f"{b:<16}{w:>12.2f}{pct:>9.1f}")
    out["wait_pct"] = round(100 * waits / total, 2) if total else None
    out["work_pct"] = round(100 * (total - waits) / total, 2) if total else None
    print(f"\nwaiting (gpu/ipc-spin/idle/nccl): {100*waits/total:.1f}%   "
          f"doing python work: {100*(total-waits)/total:.1f}%")

    print(f"\n--- top {args.top} frames by self time ---")
    print(f"{'weight':>10}{'%':>8}  frame")
    for name, w in self_time.most_common(args.top):
        pct = 100 * w / total if total else 0
        out["top_self"].append({"frame": name, "weight": round(w, 3), "pct": round(pct, 2)})
        print(f"{w:>10.2f}{pct:>8.1f}  {name[:110]}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
