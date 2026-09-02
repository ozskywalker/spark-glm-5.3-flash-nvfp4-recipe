#!/usr/bin/env python3
"""Concurrency sweep with per-stage server-side attribution.

Extends probe_pipeline_timing.py from single-stream to a concurrency sweep,
answering the question that probe left open: queue time is free at
concurrency=1, but where does it stop being free, and does aggregate
throughput keep scaling once it does?

For each concurrency level it reports, per request: client-side e2e, and
server-side queue / prefill / decode splits from Prometheus counter deltas.
It also samples the instantaneous scheduler gauges (num_requests_running,
num_requests_waiting_by_reason, kv_cache_usage_perc) throughout the run,
since those are gauges and vanish once the run ends.

Usage:
  probe_concurrency_pipeline.py [--base-url URL] [--model NAME]
                                [--levels 1,2,4,8] [--waves 3]
                                [--workload prose|structured] [--out FILE]
"""

import argparse
import json
import os
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

COUNTERS = [
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_decode_time_seconds_sum",
    "vllm:request_inference_time_seconds_sum",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:num_preemptions_total",
]
GAUGES = [
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
]

WORKLOADS = {
    "prose": ("Write a short paragraph about the history of computing.", 128),
    "structured": ("Count from 1 to 100, one number per line, digits only, no commentary.", 250),
}


def fetch_metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=30) as r:
        return r.read().decode()


def parse(text):
    counters, gauges, waiting_by_reason = {}, {}, {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name, _, rest = line.partition("{")
        try:
            val = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        if name in COUNTERS:
            counters[name] = counters.get(name, 0.0) + val
        elif name in GAUGES:
            gauges[name] = val
        elif name == "vllm:num_requests_waiting_by_reason":
            reason = rest.split('reason="')[1].split('"')[0]
            waiting_by_reason[reason] = val
    return counters, gauges, waiting_by_reason


class GaugeSampler(threading.Thread):
    """Gauges are instantaneous; sample them while the load is actually on."""

    def __init__(self, base, interval=0.25):
        super().__init__(daemon=True)
        self.base, self.interval = base, interval
        self.samples, self._stop_evt = [], threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                _, g, wbr = parse(fetch_metrics(self.base))
                self.samples.append({**g, **{f"waiting_{k}": v for k, v in wbr.items()}})
            except Exception:
                pass
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()

    def summary(self):
        if not self.samples:
            return {}
        keys = set().union(*(s.keys() for s in self.samples))
        out = {}
        for k in keys:
            vals = [s.get(k, 0.0) for s in self.samples]
            out[k + "_max"] = round(max(vals), 4)
            out[k + "_mean"] = round(statistics.fmean(vals), 4)
        out["n_samples"] = len(self.samples)
        return out


def one_request(base, model, prompt, max_tokens):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.load(r)
    return time.perf_counter() - t0, body["usage"]["completion_tokens"]


def run_level(base, model, prompt, max_tokens, concurrency, waves):
    before, _, _ = parse(fetch_metrics(base))
    sampler = GaugeSampler(base)
    sampler.start()
    lat, toks = [], 0
    t_start = time.perf_counter()
    for _ in range(waves):
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(one_request, base, model, prompt, max_tokens)
                       for _ in range(concurrency)]
            for f in futures:
                dt, n = f.result()
                lat.append(dt)
                toks += n
    wall = time.perf_counter() - t_start
    sampler.stop()
    sampler.join(timeout=2)
    time.sleep(0.4)
    after, _, _ = parse(fetch_metrics(base))
    d = {k: after.get(k, 0) - before.get(k, 0) for k in COUNTERS}
    n = concurrency * waves

    return {
        "concurrency": concurrency,
        "waves": waves,
        "n_requests": n,
        "wall_s": round(wall, 3),
        "aggregate_tok_s": round(toks / wall, 2),
        "gen_tokens": toks,
        "client_e2e_p50": round(statistics.median(lat), 3),
        "client_e2e_max": round(max(lat), 3),
        "srv_e2e_avg": round(d["vllm:e2e_request_latency_seconds_sum"] / n, 4),
        "srv_queue_avg": round(d["vllm:request_queue_time_seconds_sum"] / n, 5),
        "srv_prefill_avg": round(d["vllm:request_prefill_time_seconds_sum"] / n, 4),
        "srv_decode_avg": round(d["vllm:request_decode_time_seconds_sum"] / n, 4),
        "srv_ttft_avg": round(d["vllm:time_to_first_token_seconds_sum"] / n, 4),
        "queue_pct_of_e2e": round(100 * d["vllm:request_queue_time_seconds_sum"] /
                                  d["vllm:e2e_request_latency_seconds_sum"], 2)
        if d["vllm:e2e_request_latency_seconds_sum"] else 0.0,
        "preemptions": d["vllm:num_preemptions_total"],
        "accept_rate": round(d["vllm:spec_decode_num_accepted_tokens_total"] /
                             d["vllm:spec_decode_num_draft_tokens_total"], 4)
        if d["vllm:spec_decode_num_draft_tokens_total"] else None,
        "gauges": sampler.summary(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://10.7.0.87:8000")
    ap.add_argument("--model", default="glm-5.3-flash-exl3-v2")
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--waves", type=int, default=3)
    ap.add_argument("--workload", default="prose", choices=sorted(WORKLOADS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    prompt, max_tokens = WORKLOADS[args.workload]
    results = []
    for level in [int(x) for x in args.levels.split(",")]:
        r = run_level(base, args.model, prompt, max_tokens, level, args.waves)
        r["workload"] = args.workload
        results.append(r)
        print(json.dumps(r), flush=True)

    out = args.out or os.path.join(os.path.dirname(__file__),
                                   f"concurrency_{args.workload}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
