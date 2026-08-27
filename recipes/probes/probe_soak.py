#!/usr/bin/env python3
"""Stability soak probe for GLM-5.3-Flash (sparkrun recipe validation).

Mixed sequential + concurrent traffic to shake out the GB10 unified-memory
edge (the failure mode that killed 32G KV configs: concurrent 20K prefills —
docs/SM121-CRASH-FORENSICS-2026-08-27.md). Every response must be HTTP 200
with non-empty content and a healthy finish_reason (stop, or length when the
reply hits the max_tokens cap); the endpoint must still answer after the soak.

Usage: probe_soak.py [--base-url http://10.7.0.87:8000] [--rounds 3] [--conc 3]
Exit 0 = soak passed.
"""

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

FAILURES = []
PROMPTS = [
    "What is 17 * 23? Answer with just the number.",
    "Name three primary colors, comma separated.",
    "Write a haiku about GPUs.",
    "Translate 'good morning' into French, German, and Spanish.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "What year did the Berlin Wall fall?",
    "Give two synonyms for 'fast'.",
    "Explain what a KV cache is in one sentence.",
    "List the first five prime numbers.",
    "Who wrote 'One Hundred Years of Solitude'?",
]


def one_request(base, model, prompt, max_tokens=120, timeout=600):
    t0 = time.perf_counter()
    payload = {"model": model, "max_tokens": max_tokens,
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.load(r)
        choice = (out.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        return {"ok": True, "latency": time.perf_counter() - t0,
                "finish": choice.get("finish_reason"),
                "content_len": len(content.strip()),
                "completion_tokens": (out.get("usage") or {}).get("completion_tokens"),
                "prompt": prompt[:30]}
    except Exception as e:  # noqa: BLE001 - probe must report any failure mode
        return {"ok": False, "latency": time.perf_counter() - t0,
                "error": repr(e)[:200], "prompt": prompt[:30]}


def report(tag, results):
    bad = [r for r in results if not (r["ok"] and r["content_len"] > 0 and r["finish"] in ("stop", "length"))]
    lats = sorted(r["latency"] for r in results)
    med = lats[len(lats) // 2] if lats else float("nan")
    print(f"[{'PASS' if not bad else 'FAIL'}] {tag}: {len(results) - len(bad)}/{len(results)} ok, "
          f"median {med:.2f}s, max {max(lats):.2f}s", flush=True)
    for r in bad:
        print(f"       BAD: {r}", flush=True)
        FAILURES.append(f"{tag}:{r['prompt']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://10.7.0.87:8000")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--rounds", type=int, default=3, help="sequential rounds over the prompt set")
    ap.add_argument("--conc", type=int, default=3, help="concurrent requests per wave")
    ap.add_argument("--waves", type=int, default=3, help="concurrent waves")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    # sequential rounds
    for rnd in range(1, args.rounds + 1):
        results = [one_request(base, args.model, p) for p in PROMPTS]
        report(f"sequential-round-{rnd}", results)

    # concurrent waves (different prompt slices per wave)
    for wave in range(1, args.waves + 1):
        prompts = [PROMPTS[(wave * args.conc + i) % len(PROMPTS)] for i in range(args.conc)]
        with ThreadPoolExecutor(max_workers=args.conc) as pool:
            results = list(pool.map(lambda p: one_request(base, args.model, p), prompts))
        report(f"concurrent-wave-{wave} (x{args.conc})", results)

    # endpoint still alive?
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
            alive = r.status == 200
    except Exception as e:  # noqa: BLE001
        alive = False
        print(f"       models endpoint error: {e!r}", flush=True)
    print(f"[{'PASS' if alive else 'FAIL'}] endpoint-alive-after-soak")
    if not alive:
        FAILURES.append("endpoint-alive-after-soak")

    print(f"\n{'SOAK PASSED' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
