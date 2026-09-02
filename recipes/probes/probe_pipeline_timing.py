#!/usr/bin/env python3
"""Collects real per-stage pipeline metrics (queue/prefill/decode/TTFT, spec
decode acceptance, FLOPs/bytes) from the live vLLM EngineCore metrics
exposition, across a few representative workloads, via before/after deltas.

Usage: probe_pipeline_timing.py [--base-url http://10.7.0.87:8000] [--model glm-5.3-flash-exl3-v2]

Used to build the "Anatomy of a GLM-5.3-Flash Request" artifact (2026-09-02) —
see VALIDATION.md's "Request pipeline timing" section for the results.
"""
import argparse
import json
import os
import time
import urllib.request

_ap = argparse.ArgumentParser()
_ap.add_argument("--base-url", default="http://10.7.0.87:8000")
_ap.add_argument("--model", default="glm-5.3-flash-exl3-v2")
_args, _ = _ap.parse_known_args()

BASE = _args.base_url.rstrip("/")
MODEL = _args.model

METRIC_NAMES = [
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_decode_time_seconds_sum",
    "vllm:request_inference_time_seconds_sum",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:inter_token_latency_seconds_sum",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:estimated_flops_per_gpu_total",
    "vllm:estimated_read_bytes_per_gpu_total",
    "vllm:estimated_write_bytes_per_gpu_total",
]


def snapshot():
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
        text = r.read().decode()
    out = {}
    per_pos = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name, _, rest = line.partition("{")
        if name in METRIC_NAMES:
            val = float(line.rsplit(" ", 1)[1])
            out[name] = out.get(name, 0.0) + val
        if name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            pos = rest.split('position="')[1].split('"')[0]
            val = float(line.rsplit(" ", 1)[1])
            per_pos[pos] = val
    return out, per_pos


def chat(messages, max_tokens, temperature=0):
    payload = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
               "temperature": temperature,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                  data=json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        json.load(r)
    return time.perf_counter() - t0


def run_workload(name, messages, max_tokens, n=6):
    before, before_pos = snapshot()
    wall = []
    for _ in range(n):
        wall.append(chat(messages, max_tokens))
    time.sleep(0.3)
    after, after_pos = snapshot()
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in METRIC_NAMES}
    pos_delta = {k: after_pos.get(k, 0) - before_pos.get(k, 0) for k in after_pos}
    count = delta["vllm:e2e_request_latency_seconds_sum"] and n
    result = {
        "workload": name, "n": n, "wall_s_per_req": round(sum(wall) / len(wall), 3),
        "avg_e2e_s": round(delta["vllm:e2e_request_latency_seconds_sum"] / n, 4),
        "avg_queue_s": round(delta["vllm:request_queue_time_seconds_sum"] / n, 5),
        "avg_prefill_s": round(delta["vllm:request_prefill_time_seconds_sum"] / n, 4),
        "avg_decode_s": round(delta["vllm:request_decode_time_seconds_sum"] / n, 4),
        "avg_inference_s": round(delta["vllm:request_inference_time_seconds_sum"] / n, 4),
        "avg_ttft_s": round(delta["vllm:time_to_first_token_seconds_sum"] / n, 4),
        "avg_prompt_tokens": round(delta["vllm:prompt_tokens_total"] / n, 1),
        "avg_gen_tokens": round(delta["vllm:generation_tokens_total"] / n, 1),
        "draft_tokens": delta["vllm:spec_decode_num_draft_tokens_total"],
        "accepted_tokens": delta["vllm:spec_decode_num_accepted_tokens_total"],
        "accept_rate": round(delta["vllm:spec_decode_num_accepted_tokens_total"] /
                              delta["vllm:spec_decode_num_draft_tokens_total"], 4)
                       if delta["vllm:spec_decode_num_draft_tokens_total"] else None,
        "accept_per_pos": pos_delta,
        "flops_total": delta["vllm:estimated_flops_per_gpu_total"],
        "read_bytes_total": delta["vllm:estimated_read_bytes_per_gpu_total"],
        "write_bytes_total": delta["vllm:estimated_write_bytes_per_gpu_total"],
    }
    decode_tok_s = (delta["vllm:generation_tokens_total"] - n) / delta["vllm:request_decode_time_seconds_sum"] \
        if delta["vllm:request_decode_time_seconds_sum"] else None
    result["decode_tok_s"] = round(decode_tok_s, 2) if decode_tok_s else None
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    results = []
    results.append(run_workload(
        "short_prose_cold",
        [{"role": "user", "content": "Write a short paragraph about the history of computing."}],
        128, n=6))
    results.append(run_workload(
        "structured_counting",
        [{"role": "user", "content": "Count from 1 to 100, one number per line, digits only, no commentary."}],
        250, n=6))
    results.append(run_workload(
        "medium_prompt_summarize",
        [{"role": "user", "content": (
            "Summarize the following in two sentences: " +
            "The DGX Spark is a compact AI supercomputer built around the GB10 Grace Blackwell "
            "Superchip, pairing a 20-core Arm CPU with a Blackwell-generation GPU and unified "
            "memory architecture. It targets local development and fine-tuning of large models "
            "without requiring cloud GPU access, and two units can be linked over ConnectX "
            "networking to run larger models via tensor parallelism across both boards. "
        ) * 3}],
        150, n=6))
    out_path = os.path.join(os.path.dirname(__file__), "pipeline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("DONE")
