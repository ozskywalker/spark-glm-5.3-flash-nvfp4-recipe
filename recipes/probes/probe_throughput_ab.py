#!/usr/bin/env python3
"""Larger-n decode-throughput benchmark for A/B comparisons.

probe_sanity.py's bench section (n=3, 128 max_tokens) is a regression
smoke test, not built to detect a small, real effect against this
cluster's own run-to-run noise (observed swings of ~15% between individual
runs at baseline). This script trades speed for statistical usefulness:
one fixed prompt, temperature=0, a longer forced completion (more decode
steps per run averages out per-request scheduling/startup jitter), and a
much larger n -- enough to compare two configs' medians against their own
spread, not just eyeball two overlapping ranges.

Usage: probe_throughput_ab.py [--base-url URL] [--model NAME] [--runs 20]
                               [--max-tokens 400] [--label baseline]
Prints one JSON line per run, then a summary line with mean/median/stdev.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request

PROMPT = (
    "Write a detailed, technical explanation of how TCP congestion control "
    "works, covering slow start, congestion avoidance, and fast retransmit. "
    "Aim for around 400 words."
)


def run_once(base_url, model, max_tokens, timeout=120):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    start = time.perf_counter()
    first_token = None
    completion_tokens = None
    finish_reason = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if first_token is None and (delta.get("content") or delta.get("reasoning_content")):
                    first_token = time.perf_counter()
                finish_reason = choices[0].get("finish_reason") or finish_reason
            usage = event.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens")
    end = time.perf_counter()
    ttft = (first_token - start) if first_token is not None else float("nan")
    decode_seconds = max(end - (first_token or start), 1e-9)
    decode_tokens = max((completion_tokens or 0) - 1, 0)
    return {
        "ttft_s": round(ttft, 4),
        "total_s": round(end - start, 4),
        "completion_tokens": completion_tokens,
        "decode_tok_s": round(decode_tokens / decode_seconds, 3),
        "finish_reason": finish_reason,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://10.7.0.87:8000")
    ap.add_argument("--model", default="glm-5.3-flash-exl3-v2")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    results = []
    for i in range(1, args.runs + 1):
        try:
            r = run_once(args.base_url, args.model, args.max_tokens)
        except Exception as e:
            print(json.dumps({"run": i, "label": args.label, "error": str(e)}), flush=True)
            continue
        r["run"] = i
        r["label"] = args.label
        print(json.dumps(r), flush=True)
        results.append(r)
        time.sleep(0.3)

    if not results:
        print("NO SUCCESSFUL RUNS", file=sys.stderr)
        sys.exit(1)

    tok_s = [r["decode_tok_s"] for r in results]
    ttft = [r["ttft_s"] for r in results]
    summary = {
        "label": args.label,
        "n": len(results),
        "decode_tok_s_mean": round(statistics.mean(tok_s), 3),
        "decode_tok_s_median": round(statistics.median(tok_s), 3),
        "decode_tok_s_stdev": round(statistics.stdev(tok_s), 3) if len(tok_s) > 1 else 0.0,
        "decode_tok_s_min": round(min(tok_s), 3),
        "decode_tok_s_max": round(max(tok_s), 3),
        "ttft_s_mean": round(statistics.mean(ttft), 3),
        "ttft_s_median": round(statistics.median(ttft), 3),
    }
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
