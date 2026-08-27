#!/usr/bin/env python3
"""Short-context sanity probe for GLM-5.3-Flash (sparkrun recipe validation).

Checks, against a running endpoint:
  1. /v1/models lists the served model id.
  2. Non-streaming chat: coherent reply, finish_reason=stop, no <think> leakage
     into content (reasoning must arrive via reasoning_content, thinking-off).
  3. Streaming chat: first token arrives, usage counts present.
  4. TTFT/decode measurement over 3 runs (port of ../probes/bench_glm53.py).

Usage: probe_sanity.py [--base-url http://10.7.0.87:8000] [--model glm-5.3-flash]
Exit 0 = all checks passed; nonzero = failure (prints FAIL lines).
"""

import argparse
import json
import sys
import time
import urllib.request

FAILURES = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok:
        FAILURES.append(name)


def post(base, path, payload, timeout=600):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def chat(base, model, messages, max_tokens=200, stream=False, timeout=600):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "chat_template_kwargs": {"enable_thinking": False}}
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return post(base, "/v1/chat/completions", payload, timeout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://10.7.0.87:8000")
    ap.add_argument("--model", default="glm-5.3-flash")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    # 1. models endpoint
    with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
        models = json.load(r)
    ids = [m["id"] for m in models.get("data", [])]
    check("models-lists-served-id", args.model in ids, f"ids={ids}")

    # 1b. Prometheus metrics exposition enabled (no --disable-log-stats)
    with urllib.request.urlopen(base + "/metrics", timeout=30) as r:
        metrics_text = r.read().decode()
    vllm_metrics = [ln.split("{")[0] for ln in metrics_text.splitlines()
                    if ln.startswith("vllm:")]
    check("metrics-enabled", bool(vllm_metrics),
          f"{len(vllm_metrics)} vllm:* series; e.g. {vllm_metrics[:2]}")

    # 2. non-streaming chat, thinking off
    t0 = time.perf_counter()
    with chat(base, args.model, [{"role": "user", "content":
              "Say hello and name yourself. Then state the capital of France in one word."}],
              max_tokens=200) as r:
        out = json.load(r)
    dt = time.perf_counter() - t0
    choice = (out.get("choices") or [{}])[0]
    content = choice.get("message", {}).get("content") or ""
    reasoning = choice.get("message", {}).get("reasoning_content") or ""
    check("chat-nonempty", len(content.strip()) > 0, f"{dt:.1f}s, {len(content)} chars")
    check("chat-finish-stop", choice.get("finish_reason") == "stop",
          f"finish_reason={choice.get('finish_reason')}")
    check("chat-no-think-leak", "<think>" not in content and "</think>" not in content,
          "reasoning must not leak into content")
    check("chat-coherent", ("paris" in content.lower()) or ("france" in content.lower()),
          content.strip()[:120].replace("\n", " "))
    print(f"       reply: {content.strip()[:160]!r} (reasoning_content {len(reasoning)} chars)", flush=True)

    # 3. streaming chat + usage
    first = None
    pieces = []
    usage = None
    t0 = time.perf_counter()
    with chat(base, args.model, [{"role": "user", "content":
              "Count from one to five, digits only."}], max_tokens=50, stream=True) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            chs = ev.get("choices") or []
            if chs:
                delta = chs[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece and first is None:
                    first = time.perf_counter() - t0
                pieces.append(piece)
    text = "".join(pieces)
    check("stream-first-token", first is not None and first < 60, f"ttft={first:.2f}s" if first else "no tokens")
    check("stream-usage", bool(usage and usage.get("prompt_tokens")), f"usage={usage}")
    check("stream-counts", all(d in text for d in "12345"), text.strip()[:80])
    print(f"       ttft={first:.2f}s prompt_tokens={usage.get('prompt_tokens') if usage else '?'}", flush=True)

    # 4. TTFT/decode bench, 3 runs
    for run in range(1, 4):
        t0 = time.perf_counter()
        first = None
        completion_tokens = None
        with chat(base, args.model, [{"role": "user", "content":
                  "Write a short paragraph about the history of computing."}],
                  max_tokens=128, stream=True) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                if ev.get("usage"):
                    completion_tokens = ev["usage"].get("completion_tokens")
                chs = ev.get("choices") or []
                if chs and chs[0].get("delta", {}).get("content") and first is None:
                    first = time.perf_counter() - t0
        total = time.perf_counter() - t0
        decode = max(total - (first or 0), 1e-9)
        toks = max((completion_tokens or 1) - 1, 0)
        print(json.dumps({"bench_run": run, "ttft_s": round(first or -1, 3),
                          "total_s": round(total, 3),
                          "decode_tok_s": round(toks / decode, 2)}), flush=True)
        check(f"bench-run-{run}", first is not None and completion_tokens)

    print(f"\n{'ALL SHORT-CONTEXT CHECKS PASSED' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
