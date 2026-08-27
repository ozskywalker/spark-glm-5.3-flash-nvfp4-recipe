# GLM-5.3-Flash-NVFP4 sparkrun recipes — validation record (2026-08-27)

Cluster: sparkrun `default` — head `10.7.0.87` (spark-2dd4, rank 0) + worker
`127.0.0.1` (spark-276f, rank 1). Recipe: `glm-5.3-flash-nvfp4-vllm.yaml`
(TP=2, image `radixark/vllm-glm53-flash:sm121-v9`, weights pre-placed under
`/models` on both nodes, chat template at `/models/glm53-flash-chat_template_mm.jinja`).

## Result: ALL CHECKS PASSED

| Check | Result |
|---|---|
| Model load (InstantTensor, 91.23 GiB/rank) | ~80 s weights + 40 s warmup to serving |
| `/v1/models` serves `glm-5.3-flash` | PASS |
| Prometheus metrics (`/metrics`, 417 `vllm:*` series) | PASS |
| Short-context chat (coherent, thinking-off, no `<think>` leak) | PASS |
| Streaming TTFT / decode | 0.29-0.31 s / 23.7-27.4 tok/s (MTP-4) |
| Stability soak (2x10 sequential + 2x3 concurrent) | PASS, endpoint alive after |
| Cache continuation (shared ~6K prefix, cold/partial/warm) | PASS, no degenerate output |
| Long context **249,951 tokens** (95% of 262,144 max) | PASS — TTFT 180.6 s, 4/4 planted codes retrieved at 25/50/75/97% depth |

## Key finding: KV pool ceiling on this cluster

The upstream repo's TP2 configs (`4445787956` = 4.14 GiB, and the 672K-token
`5905580032` = 5.5 GiB record) **do not survive on spark-2dd4/spark-276f**:
three boots died in the silent NVRM first-touch kill (worker `exit code: None`,
no Python traceback, no dmesg/journal trace) during warmup or seconds after
startup. A boot without the pin showed vLLM's profiler computing
**"Available KV cache memory: -0.02 GiB"** at gmu 0.85 — these nodes have
~4-6 GiB less effective unified-memory headroom than the upstream record nodes.

The shipped recipe therefore pins `--kv-cache-memory 3221225472` (3 GiB/rank
~= 370K fp8 KV tokens = 1.4x max_model_len), which boots clean and passed the
full suite including the 250K-token prefill. Raise only with live memory
watching + the flush ritual (`scripts/prelaunch_flush.sh`).

## Operational notes

- Pre-launch ritual is mandatory: `scripts/prelaunch_flush.sh <hosts> --during-load`
  (drop_caches on all ranks + cache-flusher loop during load).
- Boot to ready: ~4.5 min with warm page cache (image pre-distributed).
- The TP4 recipe (`glm-5.3-flash-nvfp4-vllm-tp4.yaml`) mirrors the upstream
  tp4 script but is NOT validated here (cluster has 2 nodes).
- Probes: `probes/probe_sanity.py`, `probe_soak.py`,
  `probe_cache_continuation.py`, `probe_longctx.py --tokens 250000`.
  All dependency-free (urllib); sized via the server's `/tokenize` endpoint.

## NVFP4 KV cache lane: not servable on sm121-v9 (confirmed 2026-08-27)

`glm-5.3-flash-nvfp4-kvnvfp4-vllm.yaml` (`--kv-cache-dtype nvfp4`) fails fast
at VllmConfig validation, before weight load:

```
nvfp4 KV cache is not supported with MLA (Multi-head Latent Attention)
backends. Please use a different --kv-cache-dtype (e.g., 'fp8' or 'auto')
for MLA models such as DeepSeek.
```

No MLA-sparse backend in the image supports nvfp4 (SM90: auto/bf16/fp8/fp8_e4m3;
SM120: + fp8_ds_mla; TRTLLM-SM10: fp16/bf16/fp8; plain FlashInfer nvfp4 requires
the SM100 non-MLA trtllm-gen kernel). The custom `nvfp4_ds_mla` kernel work
required is specced in docs/NVFP4-KV-BUILD-SPEC.md but was never built. The
recipe is kept as the target lane for that build.

## Reproduce

```bash
./scripts/prelaunch_flush.sh 10.7.0.87,127.0.0.1 --during-load
sparkrun run glm-5.3-flash-nvfp4-vllm.yaml          # uses default cluster
python3 probes/probe_sanity.py --base-url http://10.7.0.87:8000
python3 probes/probe_soak.py --base-url http://10.7.0.87:8000 --rounds 2 --waves 2
python3 probes/probe_cache_continuation.py --base-url http://10.7.0.87:8000
python3 probes/probe_longctx.py --base-url http://10.7.0.87:8000 --tokens 250000
```
