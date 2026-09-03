#!/usr/bin/env python3
"""Long-context concurrency sweep for GLM-5.3-Flash (sparkrun recipe validation).

The shipped max_num_seqs=4 -> 16 win (+171.6%, see VALIDATION.md) was measured
entirely on SHORT prompts, with KV only 41% used at c=16 -- it was never
validated under real long-context load, and this project's own boot logs show
a real reason to be cautious: this hybrid mamba/MLA architecture pads the
attention block size up to match the mamba state page size ("Setting
attention block size to N tokens to ensure attention page size is >= mamba
page size"), and every KV-cache group (MLA attention, each mamba/GDN state
component, the sparse indexer) pins at least one such block per running
request REGARDLESS of how short that request's actual content is (see
vllm-project/vllm#54458 for the general mechanism on other hybrid-model
deployments). This probe measures the real effect directly instead of
inferring it from log lines: build documents at a couple of realistic
long-context sizes, fire them concurrently at increasing concurrency, and
watch KV-cache-usage%, queueing, and whether anything fails or degrades badly.

Usage:
  probe_longctx_concurrency.py [--base-url URL] [--model NAME]
                                [--tokens 20000,60000] [--levels 2,4,8]
                                [--max-tokens 60]
Prints one JSON line per (doc_size, concurrency) cell, then a summary table.
"""

import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

GAUGES = ["vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc"]

SUBJECTS = ["the migration patterns of arctic terns", "the invention of the printing press",
            "deep-sea hydrothermal vents", "the history of the Suez Canal",
            "how bilingualism affects cognitive aging", "the geology of the Deccan Traps",
            "the domestication of horses", "ant colony optimization algorithms",
            "the restoration of wetlands in Florida", "medieval guild regulations"]
ASPECTS = ["economic consequences", "technical challenges", "political context",
           "environmental impact", "key historical figures", "measurement methods"]


def make_sentence(rng):
    s, a, n = rng.choice(SUBJECTS), rng.choice(ASPECTS), rng.randint(1732, 1989)
    return (f"Archival note {n}: the committee reviewed {s} with particular attention to "
            f"{a}, cross-referencing field measurements against the registry copies.")


def build_document(target_tokens, base_url, model, seed):
    rng = random.Random(seed)
    target_chars = int(target_tokens * 5.7)
    paras, chars = [], 0
    while chars < target_chars:
        para = " ".join(make_sentence(rng) for _ in range(rng.randint(4, 8)))
        paras.append(para)
        chars += len(para) + 1
    text = "\n".join(paras)

    def tok_count(t):
        payload = json.dumps({"model": model, "prompt": t, "add_special_tokens": False}).encode()
        req = urllib.request.Request(base_url + "/tokenize", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return len(json.load(r)["tokens"])

    lo, hi = 0, len(paras)
    while lo < hi:
        mid = (lo + hi) // 2
        if tok_count("\n".join(paras[:mid + 1])) < target_tokens - 200:
            lo = mid + 1
        else:
            hi = mid
    text = "\n".join(paras[:lo + 1])
    n = tok_count(text)
    while n > target_tokens - 100:
        text = text[:int(len(text) * (target_tokens - 150) / n)]
        n = tok_count(text)
    return text + "\n\nSummarize the above in one sentence.", n


def fetch_metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=30) as r:
        return r.read().decode()


def parse_gauges(text):
    g = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name, _, _ = line.partition("{")
        if name in GAUGES:
            try:
                g[name] = float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                pass
    return g


class GaugeSampler(threading.Thread):
    def __init__(self, base, interval=1.0):
        super().__init__(daemon=True)
        self.base, self.interval = base, interval
        self.samples, self._stop_evt = [], threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                self.samples.append(parse_gauges(fetch_metrics(self.base)))
            except Exception:
                pass
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()

    def summary(self):
        if not self.samples:
            return {}
        out = {}
        for k in GAUGES:
            vals = [s.get(k, 0.0) for s in self.samples]
            out[k.split(":")[1] + "_max"] = round(max(vals), 2)
        return out


def one_request(base, model, prompt, max_tokens, timeout):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.load(r)
        return {"ok": True, "dt": time.perf_counter() - t0,
                "finish_reason": body["choices"][0]["finish_reason"]}
    except Exception as e:
        return {"ok": False, "dt": time.perf_counter() - t0, "error": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://10.7.0.87:8000")
    ap.add_argument("--model", default="glm-5.3-flash-exl3-v2")
    ap.add_argument("--tokens", default="20000,60000")
    ap.add_argument("--levels", default="2,4,8")
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    doc_sizes = [int(x) for x in args.tokens.split(",")]
    levels = [int(x) for x in args.levels.split(",")]

    docs = {}
    for size in doc_sizes:
        print(f"building {size}-token document...", file=sys.stderr, flush=True)
        text, actual = build_document(size, args.base_url, args.model, seed=size)
        docs[size] = text
        print(f"  built: {actual} actual tokens", file=sys.stderr, flush=True)

    results = []
    for size in doc_sizes:
        for conc in levels:
            sampler = GaugeSampler(args.base_url)
            sampler.start()
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=conc) as ex:
                futures = [ex.submit(one_request, args.base_url, args.model, docs[size],
                                     args.max_tokens, args.timeout) for _ in range(conc)]
                outcomes = [f.result() for f in futures]
            wall = time.perf_counter() - t0
            sampler.stop()
            sampler.join(timeout=2)

            n_ok = sum(1 for o in outcomes if o["ok"])
            n_fail = conc - n_ok
            lat_ok = [o["dt"] for o in outcomes if o["ok"]]
            row = {
                "doc_tokens": size,
                "concurrency": conc,
                "n_ok": n_ok,
                "n_fail": n_fail,
                "wall_s": round(wall, 1),
                "lat_p50": round(statistics.median(lat_ok), 1) if lat_ok else None,
                "lat_max": round(max(lat_ok), 1) if lat_ok else None,
                "errors": [o["error"] for o in outcomes if not o["ok"]][:3],
                **sampler.summary(),
            }
            print(json.dumps(row), flush=True)
            results.append(row)

    print("\n=== SUMMARY ===", flush=True)
    print(f"{'doc_tokens':>10} {'conc':>5} {'ok':>4} {'fail':>5} {'wall_s':>7} "
          f"{'p50':>6} {'max':>7} {'kv_max%':>8} {'waiting_max':>12}", flush=True)
    for r in results:
        print(f"{r['doc_tokens']:>10} {r['concurrency']:>5} {r['n_ok']:>4} {r['n_fail']:>5} "
              f"{r['wall_s']:>7} {r.get('lat_p50', '-'):>6} {r.get('lat_max', '-'):>7} "
              f"{r.get('kv_cache_usage_perc_max', '-'):>8} {r.get('num_requests_waiting_max', '-'):>12}",
              flush=True)


if __name__ == "__main__":
    main()
