#!/usr/bin/env python3
"""Mixed-prefill scheduling A/B probe for GLM-5.3-Flash (sparkrun recipe validation).

Measures the real cost/benefit of `GLM53_MIXED_PREFILL_CHUNK` values, read
straight from the scheduler logic shipped in the image
(`_glm53_mixed_prefill_policy` in the patched vllm/v1/core/sched/scheduler.py):
"skip" (cap=0) holds ANY request with remaining prompt tokens completely off
the schedule for as long as any peer is decoding; a numeric value N instead
lets it advance up to N tokens of prefill per step.

Run this identical script against two boots (GLM53_MIXED_PREFILL_CHUNK=skip
vs =512) with --label to tag which, then diff the two summary blocks.

  A. WARM FOLLOW-UP -- reproduces the PR80 "gate v2" test that found `skip`
     stalling near-fully-cached repeats (VALIDATION.md: follow-up TTFT
     scaled 76-89% with the concurrent generation's own duration). Prime a
     long prefix, start a decode on it, then fire an identical-prompt
     max_tokens=1 request mid-decode. A solo (uncontended) control isolates
     the pure cost of having a decoding peer at all.

  B. COLD INTAKE -- the general "fairer scheduling" claim gate v2 did NOT
     target: a brand-new, never-before-seen prompt arrives while an
     unrelated generation is decoding. `skip` should fully stall it
     regardless of chunk size, since it isn't cached at all; a numeric cap
     should let it make steady per-step progress instead. Also reports the
     victim's own decode duration with vs without the intruder, since a
     nonzero cap spends some of the victim's step budget on the intruder.

Usage:
  probe_mixed_prefill_ab.py [--base-url URL] [--model NAME] [--label skip]
                             [--reps 3] [--victim-tokens 300]
Prints one JSON line per trial, then a summary block per scenario.
"""

import argparse
import json
import random
import statistics
import sys
import threading
import time
import urllib.request

SUBJECTS = ["tidal energy pilot projects", "orchard pollination contracts",
            "the archive of merchant ship logs", "alpine railway signalling",
            "urban heat island mitigation", "the provenance of a bronze mirror",
            "the migration patterns of arctic terns", "deep-sea hydrothermal vents"]
ASPECTS = ["budget overruns", "seasonal staffing", "instrument calibration",
           "community consultations", "weather delays", "insurance claims",
           "economic consequences", "technical challenges"]


def make_doc(target_tokens, seed):
    rng = random.Random(seed)
    target_chars = int(target_tokens * 5.7)
    paras, chars = [], 0
    while chars < target_chars:
        s, a, n = rng.choice(SUBJECTS), rng.choice(ASPECTS), rng.randint(100, 9999)
        para = (f"Minute {n}: the subcommittee discussed {s}, focusing on {a}, "
                f"and resolved to revisit the estimates at the next quarterly review.")
        paras.append(para)
        chars += len(para) + 1
    return "\n".join(paras)


def chat(base, model, content, max_tokens, timeout=600, temperature=0):
    """Streaming request; returns TTFT and total latency."""
    payload = json.dumps({
        "model": model, "max_tokens": max_tokens, "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False}, "stream": True,
        "messages": [{"role": "user", "content": content}],
    }).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    first_token, finish_reason = None, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw_line in r:
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
    end = time.perf_counter()
    return {"ttft_s": round((first_token or end) - t0, 3),
            "total_s": round(end - t0, 3), "finish": finish_reason}


def bg_chat(base, model, content, max_tokens, timeout):
    """Fire a chat in a background thread; returns (thread, result-dict)."""
    result = {}
    def target():
        result["value"] = chat(base, model, content, max_tokens, timeout)
    t = threading.Thread(target=target)
    t.start()
    return t, result


def scenario_warm_followup(base, model, rep, victim_tokens, timeout):
    prompt = make_doc(16000, seed=1000 + rep) + "\n\nSummarize the above in exactly one sentence."

    # 1. Prime the prefix cache (cold, sequential).
    prime = chat(base, model, prompt, max_tokens=20, timeout=timeout)

    # 2. Solo control: identical warm-cache hit, no concurrent decode.
    solo = chat(base, model, prompt, max_tokens=1, timeout=timeout)

    # 3. Start the victim decode on the identical (now-warm) prompt.
    vt, vres = bg_chat(base, model, prompt, victim_tokens, timeout)
    time.sleep(1.0)

    # 4. Fire the follow-up mid-decode.
    followup = chat(base, model, prompt, max_tokens=1, timeout=timeout)
    vt.join(timeout=timeout)
    victim = vres.get("value", {"total_s": None, "finish": "timeout"})

    return {
        "scenario": "warm_followup", "rep": rep,
        "prime_total_s": prime["total_s"],
        "solo_followup_ttft_s": solo["ttft_s"],
        "contended_followup_ttft_s": followup["ttft_s"],
        "contended_followup_finish": followup["finish"],
        "victim_total_s": victim["total_s"],
        "victim_finish": victim["finish"],
    }


def scenario_cold_intake(base, model, rep, victim_tokens, timeout):
    victim_prompt = make_doc(8000, seed=2000 + rep) + \
        "\n\nSummarize the above in one sentence, then add related commentary."
    intruder_prompt = make_doc(4000, seed=3000 + rep) + \
        "\n\nWhat is the main theme? Answer in one sentence."

    # Solo control: the intruder prompt alone, no contention.
    solo = chat(base, model, intruder_prompt, max_tokens=40, timeout=timeout)

    # Victim decode running, then a brand-new cold intruder request arrives.
    vt, vres = bg_chat(base, model, victim_prompt, victim_tokens, timeout)
    time.sleep(1.0)
    intruder = chat(base, model, intruder_prompt, max_tokens=40, timeout=timeout)
    vt.join(timeout=timeout)
    victim = vres.get("value", {"total_s": None, "finish": "timeout"})

    return {
        "scenario": "cold_intake", "rep": rep,
        "solo_intruder_ttft_s": solo["ttft_s"],
        "contended_intruder_ttft_s": intruder["ttft_s"],
        "contended_intruder_finish": intruder["finish"],
        "victim_total_s": victim["total_s"],
        "victim_finish": victim["finish"],
    }


def summarize(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return {}
    out = {f"{key}_mean": round(statistics.mean(vals), 3), f"{key}_median": round(statistics.median(vals), 3)}
    if len(vals) > 1:
        out[f"{key}_stdev"] = round(statistics.stdev(vals), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://10.7.0.87:8000")
    ap.add_argument("--model", default="glm-5.3-flash-exl3-v2")
    ap.add_argument("--label", default="run")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--victim-tokens", type=int, default=300)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--scenario", choices=["warm", "cold", "both"], default="both")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    warm_rows, cold_rows = [], []
    for rep in range(1, args.reps + 1):
        if args.scenario in ("warm", "both"):
            row = scenario_warm_followup(base, args.model, rep, args.victim_tokens, args.timeout)
            row["label"] = args.label
            print(json.dumps(row), flush=True)
            warm_rows.append(row)
        if args.scenario in ("cold", "both"):
            row = scenario_cold_intake(base, args.model, rep, args.victim_tokens, args.timeout)
            row["label"] = args.label
            print(json.dumps(row), flush=True)
            cold_rows.append(row)

    print(f"\n=== SUMMARY (label={args.label}, reps={args.reps}) ===", flush=True)
    if warm_rows:
        summary = {"scenario": "warm_followup", "label": args.label, "n": len(warm_rows)}
        summary.update(summarize(warm_rows, "solo_followup_ttft_s"))
        summary.update(summarize(warm_rows, "contended_followup_ttft_s"))
        summary.update(summarize(warm_rows, "victim_total_s"))
        print("SUMMARY " + json.dumps(summary), flush=True)
    if cold_rows:
        summary = {"scenario": "cold_intake", "label": args.label, "n": len(cold_rows)}
        summary.update(summarize(cold_rows, "solo_intruder_ttft_s"))
        summary.update(summarize(cold_rows, "contended_intruder_ttft_s"))
        summary.update(summarize(cold_rows, "victim_total_s"))
        print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
