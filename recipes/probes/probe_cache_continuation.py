#!/usr/bin/env python3
"""Cache-continuation probe for GLM-5.3-Flash (sparkrun recipe validation).

Exercises vLLM's prefix cache with a shared long prefix + divergent suffixes —
the request shape that exposed the sglang hybrid-arch corruption on the Qwen
lane (shared prefix + divergent suffix -> degenerate output). For GLM/vLLM we
verify the same shape stays coherent:

  A. long shared document (~6K tokens) ending with a planted code,
  B. request 1: prefix + "what is the code?"  (cold prefill, caches prefix)
  C. request 2: SAME prefix + different question (partial cache hit)
  D. request 3: repeat of request 1 (full cache hit) — must agree with B
  E. divergent-suffix sanity: outputs are non-degenerate (no "!!!!!" runs,
     no empty content, finish=stop).

Usage: probe_cache_continuation.py [--base-url http://10.7.0.87:8000]
Exit 0 = all checks passed.
"""

import argparse
import json
import random
import sys
import time
import urllib.request

FAILURES = []


def make_doc(paras=48, seed=7):
    rng = random.Random(seed)
    subjects = ["tidal energy pilot projects", "orchard pollination contracts",
                "the archive of merchant ship logs", "alpine railway signalling",
                "urban heat island mitigation", "the provenance of a bronze mirror"]
    aspects = ["budget overruns", "seasonal staffing", "instrument calibration",
               "community consultations", "weather delays", "insurance claims"]
    out = []
    for i in range(paras):
        s = rng.choice(subjects)
        a = rng.choice(aspects)
        out.append(f"Minute {100 + i}: the subcommittee discussed {s}, focusing on {a}, "
                   f"and resolved to revisit the estimates at the next quarterly review "
                   f"pending confirmation from the field offices.")
    code = "CODE-9-" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(4))
    out.append(f"Final minute: the session closed with the CLASSIFIED VERIFICATION MARKER: {code}.")
    return "\n".join(out), code


def chat(base, model, content, max_tokens=60, timeout=900):
    payload = {"model": model, "max_tokens": max_tokens, "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    dt = time.perf_counter() - t0
    choice = (out.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    return {"latency": dt, "content": (msg.get("content") or "").strip(),
            "finish": choice.get("finish_reason"),
            "prompt_tokens": (out.get("usage") or {}).get("prompt_tokens"),
            "cached_tokens": ((out.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")}


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok:
        FAILURES.append(name)


def degenerate(text):
    """Detect degenerate repetition like the '!!!!!' corruption signature."""
    if not text:
        return True
    if len(set(text)) <= 3 and len(text) > 12:
        return True
    for width in (1, 2, 3):
        chunk = text[:width]
        if chunk and chunk * 12 in text:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://10.7.0.87:8000")
    ap.add_argument("--model", default="glm-5.3-flash")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    doc, code = make_doc()
    print(f"shared document: {len(doc)} chars, planted code {code}", flush=True)

    r1 = chat(base, args.model, doc + "\n\nWhat was the CLASSIFIED VERIFICATION MARKER code? Reply with the code only.")
    check("cache-cold-code-found", code in r1["content"], f"{r1['content'][:80]!r} in {r1['latency']:.1f}s")
    check("cache-cold-finish-stop", r1["finish"] == "stop", f"finish={r1['finish']}")
    check("cache-cold-non-degenerate", not degenerate(r1["content"]), repr(r1["content"][:80]))

    r2 = chat(base, args.model, doc + "\n\nHow many minutes are recorded in the document? Reply with just a number.")
    check("cache-partial-coherent", len(r2["content"]) > 0 and r2["finish"] == "stop",
          f"{r2['content'][:80]!r} in {r2['latency']:.1f}s")
    check("cache-partial-non-degenerate", not degenerate(r2["content"]), repr(r2["content"][:80]))

    r3 = chat(base, args.model, doc + "\n\nWhat was the CLASSIFIED VERIFICATION MARKER code? Reply with the code only.")
    check("cache-warm-code-found", code in r3["content"], f"{r3['content'][:80]!r} in {r3['latency']:.1f}s")
    # temperature=0 makes warm/cold agree exactly; tolerate whitespace diffs only
    check("cache-warm-consistent", r3["content"].split() == r1["content"].split(),
          f"warm={r3['content'][:40]!r} cold={r1['content'][:40]!r}")
    # prefix cache must actually engage on the repeat (vLLM reports cached_tokens
    # in prompt_tokens_details; if the field is absent entirely, say so instead
    # of silently passing)
    if r3["cached_tokens"] is not None:
        check("cache-prefix-hit", r3["cached_tokens"] > 0,
              f"cached_tokens={r3['cached_tokens']} of prompt_tokens={r3['prompt_tokens']}")
    else:
        print(f"[INFO] cache-prefix-hit — server does not report cached_tokens; cannot assert", flush=True)
    print(f"       cold={r1['latency']:.2f}s partial={r2['latency']:.2f}s warm={r3['latency']:.2f}s "
          f"(cached_tokens warm={r3['cached_tokens']})", flush=True)

    print(f"\n{'CACHE-CONTINUATION CHECKS PASSED' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
