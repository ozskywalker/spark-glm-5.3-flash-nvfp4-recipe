#!/usr/bin/env python3
"""Prompt-length bisection for the DFlash2@7168 silent-kill crash.

VALIDATION.md "DFlash2 k=7 re-validated at MNBT=7168": a prompt-length sweep
up to ~8-11K tokens killed the worker with this project's documented silent
signature -- EngineDeadError, no CUDA OOM, no assertion, no worker traceback.
Only 14.6 GiB KV is available with the drafter resident.

Critical detail from fleet_watchdog.sh's own comments: /v1/models returns 200
even with a dead engine core. Only /health reliably returns 503 on
EngineDeadError. This script checks /health after every single request, not
/v1/models, and stops at the first sign of trouble rather than continuing to
probe a dead engine.

Usage:
  probe_dflash2_bisect.py [--base-url URL] [--model NAME]
                          [--lengths 4000,6000,8000,9000,10000,11000,12000]
Sequential (not concurrent) -- one request per length, in increasing order,
health-checked after each. Stops immediately on the first failure.
"""

import argparse
import json
import random
import sys
import time
import urllib.request

SUBJECTS = ["the migration patterns of arctic terns", "the invention of the printing press",
            "deep-sea hydrothermal vents", "the history of the Suez Canal",
            "how bilingualism affects cognitive aging", "the geology of the Deccan Traps"]
ASPECTS = ["economic consequences", "technical challenges", "political context",
           "environmental impact", "key historical figures", "measurement methods"]


def make_sentence(rng):
    s, a, n = rng.choice(SUBJECTS), rng.choice(ASPECTS), rng.randint(1732, 1989)
    return (f"Archival note {n}: the committee reviewed {s} with particular attention to "
            f"{a}, cross-referencing field measurements against the registry copies.")


def build_prompt(target_tokens, base_url, model, seed):
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
        with urllib.request.urlopen(req, timeout=60) as r:
            return len(json.load(r)["tokens"])

    lo, hi = 0, len(paras)
    while lo < hi:
        mid = (lo + hi) // 2
        if tok_count("\n".join(paras[:mid + 1])) < target_tokens - 100:
            lo = mid + 1
        else:
            hi = mid
    text = "\n".join(paras[:lo + 1])
    return text + "\n\nSummarize the above in one sentence."


def check_health(base_url, timeout=10):
    try:
        req = urllib.request.Request(base_url + "/health")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception as e:
        return False


def try_length(base_url, model, target_tokens, timeout=180):
    prompt = build_prompt(target_tokens, base_url, model, seed=target_tokens)
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 20, "temperature": 0}
    req = urllib.request.Request(base_url + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.load(r)
        dt = time.perf_counter() - t0
        return {"ok": True, "dt": round(dt, 1), "finish_reason": body["choices"][0]["finish_reason"]}
    except Exception as e:
        return {"ok": False, "dt": round(time.perf_counter() - t0, 1), "error": str(e)[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://10.7.0.87:8000")
    ap.add_argument("--model", default="glm-5.3-flash-exl3-v2")
    ap.add_argument("--lengths", default="4000,6000,8000,9000,10000,11000,12000")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]

    for length in lengths:
        print(f"--- trying {length} tokens ---", file=sys.stderr, flush=True)
        result = try_length(args.base_url, args.model, length)
        result["target_tokens"] = length
        print(json.dumps(result), flush=True)

        if not result["ok"]:
            print(f"FAILED at {length} tokens: {result.get('error')}", file=sys.stderr, flush=True)
            print("BISECTION STOPPED -- request itself failed", flush=True)
            sys.exit(1)

        healthy = check_health(args.base_url)
        print(json.dumps({"target_tokens": length, "post_request_health": healthy}), flush=True)
        if not healthy:
            print(f"ENGINE UNHEALTHY after {length} tokens (request succeeded but /health failed)",
                  file=sys.stderr, flush=True)
            print("BISECTION STOPPED -- engine died after a successful-looking request", flush=True)
            sys.exit(1)

        print(f"OK at {length} tokens (dt={result['dt']}s, finish_reason={result['finish_reason']})",
              file=sys.stderr, flush=True)

    print("ALL LENGTHS SURVIVED -- no crash found in this range", flush=True)


if __name__ == "__main__":
    main()
