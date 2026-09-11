# Performance history

All numbers from `probes/probe_sanity.py` (decode), `probes/probe_longctx.py`
/ `probe_longctx_concurrency.py` (prefill/TTFT), or `probe_throughput_ab.py`
(rigorous A/B, noted where used) on this project's actual 2x DGX Spark GB10
TP=2 cluster. Decode ranges are the raw min-max across bench runs in a
session, not a single cherry-picked number. Full investigation detail for
any row: `recipes/VALIDATION.md` (current version) or
`recipes/validation-archive/<version>.md` (prior versions).

**Latest first.**

## Current production: v20-upstreamsync, two recipes sharing one image (2026-09-11)

Same `glm53-exl3-v20-upstreamsync:local` image, toggled by one env var
(`GLM53_DENSE_FP8`). **Default recipe ships it on** — this fleet's prefill
throughput has grown enough since v1 (~860-900 → ~1,500-1,700+ tok/s, see
the version history below) that trading some of it for a real decode win
is the right call now, given the ~85/15 heavy/short traffic split and
growing agentic use. A sibling recipe keeps it off for workloads that are
consistently prefill-dominated.

| Metric | **Default** (`...-vllm.yaml`, dense-FP8 **on**) | **maxprefill sibling** (dense-FP8 off) |
|---|---|---|
| Decode | 31.54-36.76 tok/s (final-recipe confirmation) / 32.59-34.15 tok/s (A/B) | 27.96-29.99 tok/s (n=9, 3 boots) |
| Prefill @16K | ~1,437 tok/s (TTFT 11.1s) | ~1,375 tok/s (TTFT 11.6s) |
| Prefill @64K | ~1,512 tok/s (TTFT 42.3s) | ~1,791 tok/s (TTFT 35.7s) |

Decode gain **+14-17%**, prefill cost @64K **-15.6%** — essentially
unchanged from the original v16-densefp8 A/B (+12% / -14.7%@64K), measured
9 days later against v15-combined; kernel work since (E3, GB10
router-GEMM, FlashKDA, MoE-gate-dedup) didn't shift the tradeoff in either
direction. Coherence/accuracy check (6 prompts incl. a CJK translation,
temp=0, on vs. off) came back clean — no garbling, no wrong answers, only
ordinary paraphrase-level drift. Still PROVISIONAL per upstream's own
commit message (no full KLD panel run). Full record: `VALIDATION.md`,
"Dense-FP8 promoted to default, maxprefill sibling added".

The `v19-indexercompat` baseline numbers below (routine-upstream-sync
fixes only, no throughput change expected) still apply as the pre-existing
platform both v20 recipes build on:

**Decode kernel composition re-measured 2026-09-11 (batch=1, CUDA graphs
on)**: gemm 53.0%/52.6%, moe_exl3 34.1%/32.8%, comms 5.0%/6.6%, attention
0.5%/0.5%, mamba_ssm 0.3%/0.3% (rank0/rank1) — essentially unchanged from
the 2026-09-02 trace (gemm 52.5%/49.8%, moe 32.8%/31.3%). The single
largest kernel is still the undersized-tile Ampere WMMA GEMM
(`cutlass_80_wmma...16x16`, 36.2-36.3% of GPU time) — the custom-kernel
opportunity that trace identified is confirmed still live. Full detail,
plus the still-missing piece for any future revisit (a fresh real-traffic
workload sample — the one this project has is from 2026-09-06): `VALIDATION.md`.

## EXL3 version history

| Version | Decode tok/s | Prefill / TTFT | Notes |
|---|---|---|---|
| v19-indexercompat | 27.4-29.85 (2 runs) | TTFT@16K 9.7-10.7s | #54048 (GB10 router-GEMM GEMM tier) + #51718 forward-compat hardening |
| v18-gb10gemm | 27.95-29.59 | TTFT@16K 10.7s | cuBLAS out_dtype router GEMM fix (vLLM #54048), at/above prior baseline |
| v16-densefp8 (**not promoted**) | 29.8-33.0 (+12%) | @16K 1,412 tok/s / TTFT 11.3s (**-12.4%**); @64K 1,495 tok/s / TTFT 42.8s (**-14.7%**) | FP8 dense projections: real decode win, real prefill loss — wrong tradeoff for this prefill-dominated fleet, kept in tree unshipped |
| v15-combined | 26.9-28.5 | @16K 1,612 tok/s / TTFT 9.9s; @64K 1,752 tok/s / TTFT 36.5s | FlashKDA (+2-6%) + MoE gate dedup; baseline the v16-densefp8 A/B was measured against |
| v12-combined | 25.9-27.8 | @16K ~1,557 (TTFT 10.2-10.3s); @64K ~1,623 (39.4s); @128K ~1,607 (79.6s); @240K ~1,575 (152.3s) | MiaAI-Lab E3 grouped-MoE kernel; flat prefill curve through the full 262K context ceiling, no cliff |
| v9-fatfork | 26.45-28.58 (3-run); 25.667 (n=10 A/B) | @16.3K ~1,510 tok/s (**+24.0%** vs. v8); @33.7K ~1,524 tok/s | sparkglm grouped-prefill fat-expert kernel; real production window: avg TTFT 14.04s, effective prefill ~1,275 tok/s at ~101.5K avg prompt tokens |
| v8 (gmu 0.84) | 28.47-28.81 | — | 0.86 confirmed unsafe post-reboot (kernel 1032 regression); 0.84 shipped |
| v8 (race fix) | 25.5-28.5 | TTFT 70.4s (large context) | vLLM #50729 mamba state-copy race backport |
| v7 | 26.04-28.19 | TTFT 17.9s / 33.3s at two context sizes | K-pool/indexer prefill-triggered crash fix |
| v6 | 23.93-28.33 | — | max_num_seqs 4→16, validated safe under long-context concurrent load |
| v5 | 24.17-26.74 | — | K-pool tail cache crash-on-long-output fix |
| v4 | 25.8-28.28 (median 27.25) | TTFT@100K 82.9-85.5s | FP8 LM head, rigorous n=20 A/B: **+8.2% decode** (23.658→25.599 tok/s, pooled-SE t≈9.95) |
| v2 | 23.55-25.09 (median 24.12) | @100K 933.3 tok/s / TTFT 107.1s (**+14.3%** vs. v1); @250K 1,048.4 tok/s / TTFT 238.4s (**+13.9%**) | 4 upstream PR validation round; promoted to default |
| v1 EXL3 (DFlash2 k=7) | 26.49-28.29 (median) | — | first EXL3 config to beat the NVFP4 baseline outright (21.56 tok/s median) |
| v1 EXL3 (MTP-only) | 19.40-21.56 | — | behind NVFP4 baseline at this point — not yet the "bigger swing" |

## Pre-EXL3: NVFP4 line (superseded, kept for history)

| Version | Decode tok/s | Prefill / TTFT | Notes |
|---|---|---|---|
| v1 NVFP4 (CUDA graphs, TP2) | 25.1-26.3 (+15-20% vs. `--enforce-eager`) | 249,951 tokens: TTFT 195.3s, 4/4 codes | Last validated NVFP4 config before the EXL3 switch |
| bf16 TP2 (v7, root README numbering — distinct from the EXL3 v7 above) | 14.3 | — | Original day-0 deployment |
| fp8+MTP-4 TP2 (v8, root README numbering) | 25-26 | — | — |
| fp8+MTP-4 TP4 (v8, flagship, root README numbering) | 35.7 (peak 36.8) | TTFT 0.204s | 4-node, 1M-token native context |

Root-README NVFP4 numbering (v7/v8 above) is a **separate, older versioning
track** from the EXL3 line's own v7/v8 — see `README.md` for that
deployment's full history, kept independent since it predates and is
unrelated to the EXL3 quantization switch.
