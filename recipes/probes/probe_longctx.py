#!/usr/bin/env python3
"""Long-context probe for GLM-5.3-Flash (sparkrun recipe validation).

Builds a synthetic document sized to a target prompt-token count (measured
EXACTLY via the server's /v1/tokenize endpoint), plants secret codes at
~25% / ~50% / ~75% depth plus one near the end, then asks the model to
retrieve them. Verifies:

  - the request is accepted at near-max depth (no OOM / rejection),
  - usage.prompt_tokens lands within tolerance of the target,
  - the model actually attends across the context (codes retrieved),
  - streaming stays alive for the whole multi-minute prefill.

Default target: 250,000 prompt tokens (max_model_len is 262,144 on the TP2
recipe; the shipped 3 GiB KV pool is ~370K tokens so a single request fits
with room to spare).
A full-depth prefill takes many minutes — run with a patient timeout.

Usage:
  probe_longctx.py [--base-url http://10.7.0.87:8000] [--tokens 250000]
Exit 0 = all checks passed.
"""

import argparse
import json
import random
import sys
import time
import urllib.request

FAILURES = []

SUBJECTS = ["the migration patterns of arctic terns", "the invention of the printing press",
            "deep-sea hydrothermal vents", "the history of the Suez Canal",
            "how bilingualism affects cognitive aging", "the geology of the Deccan Traps",
            "the domestication of horses", "ant colony optimization algorithms",
            "the restoration of wetlands in Florida", "medieval guild regulations",
            "the discovery of penicillin", "monsoon dynamics in South Asia",
            "the construction of Gothic cathedrals", "fermentation in food preservation",
            "the evolution of the metric system", "coral bleaching events",
            "the Silk Road trade in lapis lazuli", "how sonar was developed",
            "the physiology of hibernation", "the design of Roman aqueducts"]
ASPECTS = ["economic consequences", "technical challenges", "political context",
           "environmental impact", "key historical figures", "measurement methods",
           "common misconceptions", "notable failures", "modern legacy", "early records"]


def make_sentence(rng):
    s = rng.choice(SUBJECTS)
    a = rng.choice(ASPECTS)
    n = rng.randint(1732, 1989)
    return (f"Archival note {n}: the committee reviewed {s} with particular attention to "
            f"{a}, cross-referencing field measurements against the registry copies and "
            f"recording dissenting opinions in the annex for later ratification.")


def build_document(target_tokens, base_url, model, seed=1234):
    """Generate filler text, then trim/pad to the exact token target via /tokenize."""
    rng = random.Random(seed)
    # this archival style runs ~5.6 chars/token under BPE (repetitive phrasing);
    # overshoot then trim by binary search
    target_chars = int(target_tokens * 5.7)
    paras = []
    chars = 0
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

    # safety loop: append more paragraphs if still short of target
    while tok_count(text) < target_tokens - 200:
        for _ in range(len(paras)):
            para = " ".join(make_sentence(rng) for _ in range(rng.randint(4, 8)))
            paras.append(para)
        text = "\n".join(paras)

    # trim by paragraphs until under target (fast convergence, ~1-2 tokenize calls)
    lo, hi = 0, len(paras)
    while lo < hi:
        mid = (lo + hi) // 2
        if tok_count("\n".join(paras[:mid + 1])) < target_tokens - 200:
            lo = mid + 1
        else:
            hi = mid
    text = "\n".join(paras[:lo + 1])
    n = tok_count(text)
    # fine trim by words if still over
    while n > target_tokens - 100:
        text = text[:int(len(text) * (target_tokens - 150) / n)]
        n = tok_count(text)
    return text, n


def plant_codes(text):
    """Insert codes at 25/50/75% depth and near the end; return (text, codes)."""
    codes = [f"CODE-{i}-{''.join(random.Random(42 + i).choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(4))}"
             for i in range(1, 5)]
    lines = text.split("\n")
    marks = [int(len(lines) * f) for f in (0.25, 0.50, 0.75, 0.97)]
    for code, mark in zip(codes, marks):
        lines[mark] = (lines[mark] + f" CLASSIFIED VERIFICATION MARKER: {code}.")
    return "\n".join(lines), codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://10.7.0.87:8000")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--tokens", type=int, default=250000, help="target prompt tokens")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=7200, help="request timeout seconds")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    print(f"building ~{args.tokens}-token document (tokenized via /v1/tokenize)...", flush=True)
    t0 = time.perf_counter()
    text, n_tok = build_document(args.tokens, base, args.model)
    text, codes = plant_codes(text)
    print(f"document ready: ~{n_tok} tokens before markers, {len(text)} chars, "
          f"{time.perf_counter() - t0:.0f}s build time", flush=True)
    print(f"planted codes: {codes}", flush=True)

    question = ("The document above contains four CLASSIFIED VERIFICATION MARKERS, "
                "each with a code of the form CODE-<digit>-<4 letters>. "
                "List all four codes in the order they appear, nothing else.")
    payload = {"model": args.model, "max_tokens": args.max_tokens, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user", "content": text + "\n\n" + question}]}

    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    print(f"streaming long-context request (timeout {args.timeout}s)...", flush=True)
    t0 = time.perf_counter()
    first = None
    pieces = []
    usage = None
    finish = None
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                if ev.get("usage"):
                    usage = ev["usage"]
                chs = ev.get("choices") or []
                if chs:
                    finish = chs[0].get("finish_reason") or finish
                    piece = (chs[0].get("delta") or {}).get("content") or ""
                    if piece and first is None:
                        first = time.perf_counter() - t0
                        print(f"first token after {first:.1f}s", flush=True)
                    pieces.append(piece)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] request-failed — {e!r}", flush=True)
        sys.exit(1)
    total = time.perf_counter() - t0
    answer = "".join(pieces)

    def check(name, ok, detail=""):
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""), flush=True)
        if not ok:
            FAILURES.append(name)

    prompt_tokens = (usage or {}).get("prompt_tokens") or 0
    check("prompt-tokens-near-target",
          abs(prompt_tokens - args.tokens) <= max(500, args.tokens * 0.02),
          f"prompt_tokens={prompt_tokens}, target={args.tokens}")
    check("ttft-reasonable", first is not None and first < 1800,
          f"ttft={first:.1f}s" if first else "no tokens")
    check("finish-stop", finish == "stop", f"finish_reason={finish}")
    found = [c for c in codes if c in answer]
    check("codes-retrieved", len(found) == len(codes),
          f"{len(found)}/{len(codes)} found; answer={answer.strip()[:200]!r}")
    print(f"       total={total:.1f}s  prefill+decode; "
          f"completion_tokens={(usage or {}).get('completion_tokens')}", flush=True)

    print(f"\n{'LONG-CONTEXT CHECKS PASSED' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
