#!/usr/bin/env python3
"""Drive vLLM's /start_profile ... workload ... /stop_profile cycle.

Requires the server to have been launched with a profiler config (see
recipes/glm-5.3-flash-exl3-v2-profiling.yaml); without it those endpoints
don't exist and this exits with a clear message rather than a 404 traceback.

The workload matters more than it looks: a profiled window that contains only
decode steps and a profiled window that contains the prefill answer different
questions, so --workload selects the shape and --prefill-only stops generation
after a single token to isolate prefill.

Usage:
  probe_profile_run.py --workload prose|structured|medium|longctx
                       [--prefill-only] [--max-tokens N] [--base-url URL]
"""

import argparse
import json
import time
import urllib.error
import urllib.request

WORKLOADS = {
    "prose": "Write a short paragraph about the history of computing.",
    "structured": "Count from 1 to 100, one number per line, digits only, no commentary.",
    "medium": ("Summarize the following in two sentences: " +
               ("The DGX Spark is a compact AI supercomputer built around the GB10 Grace "
                "Blackwell Superchip, pairing a 20-core Arm CPU with a Blackwell-generation "
                "GPU and unified memory architecture. It targets local development and "
                "fine-tuning of large models without requiring cloud GPU access, and two "
                "units can be linked over ConnectX networking to run larger models via "
                "tensor parallelism across both boards. ") * 3),
}


def post(base, path, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else b"{}"
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
        try:
            return json.loads(body) if body.strip() else {}
        except json.JSONDecodeError:
            return {"raw": body[:400]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://10.7.0.87:8000")
    ap.add_argument("--model", default="glm-5.3-flash-exl3-v2")
    ap.add_argument("--workload", default="prose")
    ap.add_argument("--prefill-only", action="store_true",
                    help="cap generation at 1 token so the window is prefill-dominated")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--longctx-tokens", type=int, default=100000,
                    help="approximate prompt size for --workload longctx")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    if args.workload == "longctx":
        filler = ("The quick brown fox jumps over the lazy dog. Pack my box with five "
                  "dozen liquor jugs. How vexingly quick daft zebras jump. ")
        # ~1.35 tokens/word for this filler; overshoot then let the server truncate nothing.
        reps = max(1, args.longctx_tokens // 30)
        prompt = ("Read this text and then answer.\n" + filler * reps +
                  "\nQuestion: what animal is mentioned most often?")
    else:
        prompt = WORKLOADS[args.workload]

    max_tokens = 1 if args.prefill_only else (args.max_tokens or
                                              (250 if args.workload == "structured" else 128))

    try:
        print("start_profile:", post(base, "/start_profile", timeout=120))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise SystemExit("No /start_profile endpoint — server was not launched with "
                             "a --profiler-config. Use recipes/glm-5.3-flash-exl3-v2-profiling.yaml.")
        raise

    t0 = time.perf_counter()
    for i in range(args.repeat):
        body = post(base, "/v1/chat/completions", {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        u = body.get("usage", {})
        print(f"  run {i+1}: prompt_tokens={u.get('prompt_tokens')} "
              f"completion_tokens={u.get('completion_tokens')}")
    workload_s = time.perf_counter() - t0

    print("stop_profile:", post(base, "/stop_profile", timeout=600))
    print(f"workload wall: {workload_s:.2f}s "
          f"(workload={args.workload}, prefill_only={args.prefill_only}, max_tokens={max_tokens})")
    print("Traces are written on each TP rank's node under the configured "
          "torch_profiler_dir; collect them with collect_traces.sh")


if __name__ == "__main__":
    main()
