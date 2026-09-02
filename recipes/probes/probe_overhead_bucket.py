#!/usr/bin/env python3
"""Decompose the residual "overhead" in a request's end-to-end time.

vLLM's own histograms account for queue + prefill + decode. Subtracting those
from end-to-end leaves a consistent 1-2% residual that no metric explains —
it is some mix of tokenization, detokenization, response serialization and
HTTP transfer. This probe separates those by varying one dimension at a time:
sweep output length at a fixed prompt (isolates per-output-token cost), then
sweep prompt length at a fixed output (isolates per-prompt-token cost).

Server-side `e2e` is used as the baseline rather than client wall time so that
network RTT to the probe host does not contaminate the per-token slopes; the
client-vs-server gap is reported separately as the transport share.

Usage: probe_overhead_bucket.py [--base-url URL] [--model NAME] [--reps 4]
"""

import argparse
import json
import os
import statistics
import time
import urllib.request

COUNTERS = [
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_decode_time_seconds_sum",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
]

FILLER = ("The DGX Spark is a compact AI supercomputer built around the GB10 Grace "
          "Blackwell Superchip with unified memory. ")


def snapshot(base):
    with urllib.request.urlopen(base + "/metrics", timeout=30) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name = line.partition("{")[0]
        if name in COUNTERS:
            try:
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                pass
    return out


def run_case(base, model, prompt, max_tokens, reps):
    before = snapshot(base)
    walls = []
    for _ in range(reps):
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "temperature": 0,
                   "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(base + "/v1/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as r:
            json.load(r)
        walls.append(time.perf_counter() - t0)
    time.sleep(0.3)
    after = snapshot(base)
    d = {k: after.get(k, 0) - before.get(k, 0) for k in COUNTERS}
    n = reps
    e2e = d["vllm:e2e_request_latency_seconds_sum"] / n
    acct = (d["vllm:request_queue_time_seconds_sum"] +
            d["vllm:request_prefill_time_seconds_sum"] +
            d["vllm:request_decode_time_seconds_sum"]) / n
    return {
        "prompt_tokens": round(d["vllm:prompt_tokens_total"] / n, 1),
        "gen_tokens": round(d["vllm:generation_tokens_total"] / n, 1),
        "srv_e2e_s": round(e2e, 4),
        "accounted_s": round(acct, 4),
        "overhead_s": round(e2e - acct, 5),
        "overhead_pct": round(100 * (e2e - acct) / e2e, 2) if e2e else None,
        "client_wall_s": round(statistics.fmean(walls), 4),
        "transport_s": round(statistics.fmean(walls) - e2e, 5),
    }


def slope(xs, ys):
    if len(xs) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den) if den else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://10.7.0.87:8000")
    ap.add_argument("--model", default="glm-5.3-flash-exl3-v2")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    results = {"output_sweep": [], "prompt_sweep": []}
    short = "Write about computing."

    print("=== output-length sweep (prompt fixed) ===")
    for mt in (16, 64, 160, 320):
        r = run_case(base, args.model, short, mt, args.reps)
        r["case"] = f"out={mt}"
        results["output_sweep"].append(r)
        print(json.dumps(r), flush=True)

    print("\n=== prompt-length sweep (output fixed at 32) ===")
    for reps_filler in (1, 40, 160, 400):
        prompt = "Summarize: " + FILLER * reps_filler
        r = run_case(base, args.model, prompt, 32, args.reps)
        r["case"] = f"filler={reps_filler}"
        results["prompt_sweep"].append(r)
        print(json.dumps(r), flush=True)

    o = results["output_sweep"]
    p = results["prompt_sweep"]
    s_out = slope([x["gen_tokens"] for x in o], [x["overhead_s"] for x in o])
    s_pro = slope([x["prompt_tokens"] for x in p], [x["overhead_s"] for x in p])
    summary = {
        "us_per_output_token": round(s_out * 1e6, 2) if s_out is not None else None,
        "us_per_prompt_token": round(s_pro * 1e6, 2) if s_pro is not None else None,
        "mean_transport_s": round(statistics.fmean(
            [x["transport_s"] for x in o + p]), 5),
    }
    results["summary"] = summary
    print("\n=== slopes ===")
    print(json.dumps(summary, indent=2))

    out = args.out or os.path.join(os.path.dirname(__file__), "overhead_bucket.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
