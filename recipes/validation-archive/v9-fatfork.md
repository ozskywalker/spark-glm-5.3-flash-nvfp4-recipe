# VALIDATION.md archive: v9-fatfork

_Archived from `recipes/VALIDATION.md` lines 3714-3887 (git-blame-verified,_
_no content altered). See `recipes/VALIDATION.md` for the current index and_
_where to look for common problems._

---


## TileLang JIT cache persistence: a real gap found by reading a sibling project, verified end-to-end (2026-09-05)

At the user's request, reviewed `Enntity/sparkglm` (a different open project also running GLM-5.3-Flash on exactly two DGX Spark GB10s, sharing our checkpoint, `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`, and much of our `GLM53_*` patch lineage -- their `tests/` names line up almost 1:1 with our own K-pool/indexer/spinwait/mixed-prefill patches) for anything applicable to this recipe. Their `start.sh` carries this comment verbatim: *"Overlay FS ~/.triton and ~/.tilelang die on container recreate (TP=2 JIT stall -> 600s NCCL watchdog). Persist next to the vLLM cache."* -- and they bind-mount a host-persisted `TILELANG_CACHE_DIR` alongside `TRITON_CACHE_DIR`.

**Checked whether this applies to us, directly, before touching anything.** Our production image already persists a whole family of JIT/compile caches under sparkrun's per-recipe host mount at `/cache/runtime` -- `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `FLASHINFER_CACHE_DIR`, `CUTE_DSL_CACHE_DIR`, and `FLASH_ATTENTION_CUTE_DSL_CACHE_DIR` are all baked into the image pointing there. `TILELANG_CACHE_DIR` was conspicuously absent from that list. TileLang's own `env.py` defaults it to `~/.tilelang/cache`; inspected the live container's `HOME` and found it set to `/tmp` (not `/root`, despite the image running plenty of other tools' caches under `/root/.cache/...`-style paths) -- so TileLang's actual default resolved to `/tmp/.tilelang/cache`, entirely inside the ephemeral container filesystem. Confirmed directly on the (unmodified) v8@0.84 production boot: that ephemeral path held 3.6 MiB of compiled kernels, **including the `mhc_pre_big_fuse_with_norm_tilelang` family** -- the exact kernel whose first-ever JIT compile preceded the fourth production crash by 20 seconds (see "A FOURTH production crash" above, which concluded "no crash-specific fix found anywhere for this kernel"). Every container recreate was silently discarding this cache and forcing a fresh from-scratch compile of every previously-seen TileLang kernel shape the next time serving traffic happened to need it -- live, mid-request, on a TP2 collective.

**Fix**: added `TILELANG_CACHE_DIR: /cache/runtime/tilelang` to the recipe's `env:` block -- the identical pattern the image's own baked-in vars already use, landing under the same sparkrun-persisted mount with no new volume declaration needed. Did not set `TILELANG_TMP_DIR`; TileLang derives it from `TILELANG_CACHE_DIR` by default (`os.path.join(Environment.TILELANG_CACHE_DIR, "tmp")`).

**Verified end-to-end, not just "the env var is set":**

1. Live in the new container: `env | grep -i tilelang` -> `TILELANG_CACHE_DIR=/cache/runtime/tilelang`, and `docker inspect` confirmed `/cache/runtime` is the same per-recipe host-persisted mount (`~/.cache/sparkrun/runtime-cache/vllm/models-926ad449`) already used by Triton et al.
2. After the first boot on the fix: 8 kernel directories materialized under the *host* path (not the container-ephemeral one), and `/tmp/.tilelang` inside the container was empty (0 files) -- TileLang is writing to the persisted location, full stop.
3. **The actual test**: stopped the container, ran the mandatory `prelaunch_flush.sh`, and relaunched clean (a genuine recreate -- new container ID, new PIDs). After the second boot reached `Application startup complete`: the set of kernel directories was byte-for-byte identical to the pre-recreate listing (`diff` clean), and a spot-checked `executable.so`'s mtime and md5sum were unchanged across the recreate. TileLang reused the cached kernel; it did not recompile it.
4. `/health` returned 200 after both boots; a real chat-completion request against the fixed boot returned a correct, clean response.

This doesn't prove the crash-4 kernel shape specifically will now warm silently instead of stalling mid-serving (that kernel was hit by real traffic ~18 minutes into serving, not at boot, so reproducing the exact trigger isn't practical to force) -- but the general mechanism this fix targets (any TileLang kernel shape now compiles once per image lifetime instead of once per container recreate) is now verified working, and it directly closes the persistence gap the sibling project's comment identified. Low risk: it only redirects an existing, already-consumed env var to a path the image already trusts for identical caches; a bad interaction would show up as a boot failure, and both test boots served cleanly.

**Not yet done**: sparkglm also ships `boot-shape-warmup.sh`, which deliberately burns known JIT shapes right after `/health` so any first-compile stall happens before real traffic depends on the collective, rather than mid-request. That's a complementary, separate change (needs enumerating which TileLang/Triton shapes are actually reachable in production, e.g. the DFlash2 BLOCK-size ladder they describe does not apply to us since we run MTP-2, not DFlash2) -- not attempted here.

## Adopting sparkglm's grouped-prefill fat-expert kernel + a tinyGLM fixture: build and dispatch-correctness phase (2026-09-06)

At the user's request: pursue both sparkglm's GPU-resident grouped-prefill kernel (the fork discussed in "The Fat-Expert Fork" artifact) and a tinyGLM-equivalent fixture, following the plan at `.claude/plans/dapper-moseying-clock.md`. Scope, confirmed with the user first: grouped-prefill only (`EXL3_GROUPED_PREFILL_K4`); sparkglm's separate decode-side cooperative kernel (`EXL3_DECODE_COOP_K4`) is out of scope -- their own docs flag it as weaker evidence, and it's cleanly separable in both the CUDA source and the Python dispatch layer.

### Phase A: the graft, and why it's lower-risk than it sounds

Found a local clone of the actual upstream source, `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks @ c190db1`, at `/home/luser/build-exl3-v2` -- the real source our production image is built from, not just "an equivalent recipe." Confirmed it and sparkglm build from the **identical base image digest** and **identical exllamav3 commit** (`c5d9c657966ffeeaa9353f0cc899f18629da4a13`), so this is a graft onto a matching toolchain.

New build tree at `recipes/build/glm53-exl3-fatfork/`, seeded from that known-working source: `overlay/exl3_fat_gemm.cu`/`.cuh` replaced with sparkglm's versions (confirmed scoped entirely to fat-expert/grouped-prefill, zero decode-coop references by direct grep), `overlay/patch_exl3_fat_kernel.py` extended with the ~15-line pybind11-binding diff for the new grouped-prefill symbols only. `overlay/exl3.py` was taken **wholesale** from sparkglm rather than hand-merged -- checked first that every name our own `tests/test_exl3_overlay.py` imports from it (17 names) exists exactly once in their file, and that every decode-coop reference is defensively guarded (`decode_coop_enabled()` gates all of it; the one C++ call site additionally checks `hasattr(exllamav3_ext, "exl3_decode_moe_k4")` before calling), so excluding `exl3_decode_moe.cu/.cuh` from the compile leaves that code path inert, not broken.

Built with `docker build -t glm53-exl3-fatfork:local .`. First attempt failed at the project's own build-time self-test suite: `test_exl3_overlay.py` asserted `EXL3_FAT_DIAG_SCHEMA == 1`, sparkglm's file ships schema `2`. Diffed the two schemas directly: purely additive (`sym_fat_gemm_pair`, `sym_fat_swiglu_had`, `fat_pair_enabled`, `fat_fused_activation_enabled` added to `EXL3_FAT_DIAG_KEYS`), not a structural break -- updated the hardcoded `1` to `2` with a comment explaining why. Second build succeeded end-to-end, including the full existing patch self-test suite (suppress-stops, scheduler-decode-floor, hybrid-prefix-hit, xgrammar, kpool-tail, spinwait, indexer-workspace, ablit -- all still pass against the grafted files) and a build-time assertion (added) that the 8 new grouped-prefill pybind symbols exist AND `exl3_decode_moe_k4` does not.

Verified directly in the built image (`docker run --rm --entrypoint python3 ... -c "import torch; import exllamav3_ext as e; ..."`, importing torch first -- a bare `import exllamav3_ext` fails with `ImportError: libc10.so`, the same failure mode observed earlier this session): all 6 new grouped-prefill symbols present, `exl3_decode_moe_k4` absent. Image: `sha256:1dced847312c8b683aca7d9e464f8a6320052c94d076dd44d1556246d9797576`.

### Phase B: tinyGLM, and three real bugs it caught before a full checkpoint boot

`recipes/scripts/make_tinyglm.py`, adapted from sparkglm's script (Apache-2.0, credited) but built from *our* real checkpoint's `config.json` read directly rather than copied from their published example -- their example uses a flatter schema (`architectures: ["Glm5NextForCausalLM"]`, no explicit `linear_attn_config.kda_layers`/`full_attn_layers` index lists) than what our installed vLLM actually parses for this checkpoint. 4 layers (keeping the real 3:1 KDA:full-attention repeating pattern), 16 experts, 256 vocab, dummy weights, `--load-format dummy`.

This is exactly the class of problem tinyGLM exists to catch cheaply, and it earned its keep immediately -- three real, unrelated bugs surfaced across three ~10-second boot attempts, none of which would have been fun to debug at the end of an 8-minute real checkpoint load:

1. **`FileNotFoundError: processor_config.json`.** Omitting `vision_config` entirely (matching sparkglm's own text-only fixture design) doesn't produce a text-only code path for this checkpoint's actual architecture string, `Glm5NextForConditionalGeneration` -- that class unconditionally wires in vLLM's multimodal processor regardless of `vision_config`'s presence. Fix: kept the real `vision_config` verbatim (cheap under `--load-format dummy`) and copied `processor_config.json` byte-for-byte from the real checkpoint; added 6 multimodal special-token ids sized to fit the reduced vocab (the real checkpoint's ids, 154830+, don't fit a 256-token vocab).
2. **`RuntimeError: EXL3 mcg marker is not the MCG int32 ...; packed ABI mismatch`.** vLLM's stock `--load-format dummy` leaves integer tensors uninitialized, but EXL3's loader validates a specific packed marker value as an ABI check -- a generic dummy loader can't satisfy a quantization-specific integrity check. sparkglm already solved this: `_initialize_tiny_dummy_exl3()` in their `overlay/exl3.py`, gated by `SPARKGLM_TINY_DUMMY=1`, fills a deterministic per-expert-distinct valid trellis pattern. Wasn't obvious from the diagnostics alone -- found by grepping their file for the exact error text and finding the existing, already-wired bypass.
3. **Stale schema assertion** (Phase A, above) -- caught by the *build-time* self-test, before this fixture was even reachable.

**Verified, not just "it booted":**
- `Application startup complete`, `/health` returns 200.
- `exl3 e2 diag schema=2 configured_tier=kernel effective_tier=kernel tier_reason=kernel_ok ... fat_pair_enabled=1 fat_fused_activation_enabled=1 ... cap_ok=1` -- the grouped-prefill-capable tier is configured *and* effective, not silently falling back.
- `exl3 grouped prefill scratch: rows=16384 bytes=301994244` logged during CUDA graph capture -- the grouped-prefill C++ path actually executed (scratch allocation happens inside the dispatch call), under TP2, under graph capture, without crashing.
- A real chat-completion request round-tripped cleanly (well-formed response, correct `finish_reason`, no NaN/error -- content is the expected meaningless dummy-weight garbage).
- All of this project's own existing runtime patches (mamba race-condition fix, K-pool tail fix, K-pool/indexer fix, mixed-prefill env fix) printed their normal confirmation lines in every process (entrypoint + both `__mp_main__` children), unaffected by the graft.
- **Determinism**: stopped, fully recreated the container (new container ID, new PIDs), reissued an identical request -- byte-identical output both times.

Practical note for future use: this checkpoint's tiny weight footprint (~1.6 GiB) means a tinyGLM boot still requests its configured *fraction* of total device memory for KV sizing (e.g. 0.84 -> ~96 GiB requested), so it cannot run concurrently with the real production boot on the same GB10 pair -- confirmed by watching a first attempt sit in "pending" against live v8 traffic, and choosing (with the user's confirmation, given live traffic was active) to stop v8 briefly rather than risk today's already-thin memory margin by trying to co-locate them. v8 was back up and serving within one full reboot cycle afterward.

### Phase D: real-checkpoint validation -- a real, reproducible prefill win, a real KV-capacity cost

`recipes/glm-5.3-flash-exl3-v9-fatfork-vllm.yaml` (candidate, carries v8's full patch stack plus `container: glm53-exl3-fatfork:local` and `EXL3_GROUPED_PREFILL_K4: "1"`) booted against the real 164 GiB checkpoint. `prelaunch_flush.sh` run before every boot in this phase, as always.

**Memory margin**: head-node mem-avail settled at ~6.6 GiB / 5.3% and held flat there, comfortably clear of earlyoom's 2% trigger -- not the knife-edge this project found and fixed for 0.86 earlier this session. `Application startup complete` reached normally, `/health` returned 200.

**KV cache capacity -- a real cost, but the exact size is noisier than a clean before/after**: consumed memory (weights + non-torch + the grouped-prefill scratch allocation) came in at 89.65-89.75 GiB across v9-fatfork's two boots, leaving 6.91-8.87 GiB for KV cache. v8's own boots the same session ranged 83.03-89.02 GiB consumed / 6.91-15.23... no -- corrected after re-checking: **v8's own KV-cache-in-use varied 7.38-15 GiB across this session's several boots**, driven by host free-memory state at profiling time (the same startup-time "Free memory on device" readout this file has documented varying by several GiB between otherwise-identical boots all session). One fresh v8 restore immediately after this Phase D's v9-fatfork tests measured only 7.38 GiB -- inside v9-fatfork's own 6.91-8.87 GiB range, not clearly above it. **Conclusion: there is very likely a real, directional KV-capacity cost from the grouped-prefill scratch allocation (sparkglm's own docs quantify it at ~1.2 GiB/rank), but this session's data cannot cleanly separate that fixed cost from this fleet's already-documented boot-to-boot free-memory variance** -- an honest gap, not a claimed number. A same-boot-cycle, back-to-back A/B (stop/flush/relaunch each arm immediately before measuring, no other boots in between) would be needed to isolate it precisely; not done here given time already spent.

**Correctness**: `probe_sanity.py --model glm-5.3-flash-exl3-v2` -- ALL SHORT-CONTEXT CHECKS PASSED (chat, streaming, coherence, no reasoning-leak, 3-run bench at 26.45-28.58 tok/s).

**Decode, confirmed unaffected** (expected -- grouped-prefill only touches the prefill-time MoE dispatch path): `probe_throughput_ab.py` n=10, mean **25.667 tok/s** (stdev 0.705) -- statistically indistinguishable from this same session's v8@0.84 baselines (25.609 and 25.908 tok/s, both n=10). `probe_pipeline_timing.py`'s two short-prompt shapes (23 and 32 prompt tokens) were likewise unchanged from the v8 baseline measured earlier the same session.

**Prefill at small scale, a real signal on repetitive content**: `probe_pipeline_timing.py`'s `medium_prompt_summarize` shape (275 prompt tokens, the same paragraph repeated 3x) dropped from v8's 1.409s prefill / 1.497s TTFT to **0.515s prefill / 0.596s TTFT** -- a ~63% reduction at a scale far below the 128-row fat-expert threshold on average routing (275 tokens x 8-of-288 routing implies ~7.6 rows/expert on average). The repeated-paragraph structure is the likely explanation: repetition concentrates routing onto fewer experts than the average would suggest, pushing at least one past the fat threshold even at this small scale. Treated as a lead, not the headline number, given n=6 at one shape.

**Prefill at scale -- the real test, and the answer to this session's open question**: built a small dedicated probe (`probe_prefill_at_scale.py`, using `probe_pipeline_timing.py`'s server-side `vllm:request_prefill_time_seconds_sum`/`vllm:prompt_tokens_total` metric deltas, not client TTFT) with a fresh cache-unique prompt per run (a random nonce prefix defeats `--enable-prefix-caching`, which a first careless version of this test did not do and silently deflated its own numbers on repeat runs -- caught and fixed before trusting the result, same discipline as the `GLM53_MIXED_PREFILL_CHUNK` env-propagation check earlier this session). At ~16,315 actual tokens:

| Config | n | mean prefill tok/s | range |
|---|---:|---:|---:|
| v8@0.84 (baseline) | 4 | **1,217.9** | 1,200.5-1,229.3 |
| v9-fatfork | 4 (of 5; one early-boot outlier at 4,579 tok/s excluded as a one-off, not reproduced across 4 subsequent runs) | **~1,510** | 1,499.4-1,514.7 |

**+24.0% effective prefill at ~16.3K tokens, under this fleet's actual MTP-2 config.** This is larger, proportionally, than sparkglm's own claimed +9.6% at ~33K under DFlash2 k=7 -- directly answering this session's open question (whether their DFlash2-measured win transfers to MTP-2, given this project's own prior finding that MTP-vs-DFlash2 bookkeeping differences can change a scheduler-adjacent result's outcome): it does, and by more, not less, on our config. Not yet tested at sparkglm's own ~33K scale specifically, given time already spent; the 16.3K result is tight (four runs within a 15 tok/s band) and not treated as needing further confirmation at this scale.

**Concurrency**: `probe_concurrency_pipeline.py --levels 1,4,8 --waves 2 --workload prose` -- clean at every level, zero preemptions, aggregate tok/s scaled as expected (27.37 -> 51.63 -> 94.45), acceptance rates in the normal 0.67-0.76 range for this workload.

**Stability soak**: `probe_soak.py --rounds 4 --conc 3 --waves 3` -- SOAK PASSED, all sequential rounds and concurrent waves 100% ok, endpoint alive after soak.

**One honestly-reported, unattributed signal**: `dmesg` on the head node shows NVRM `Out of memory [NV_ERR_NO_MEMORY]` kernel-level log lines at several points across this session's testing (18:16, 21:20, 21:30-21:31, 22:26-22:27, 22:38 local). These correlate with clusters of rapid stop/restart cycles (tinyGLM iteration, the TileLang-fix boots earlier the same session) rather than steady-state operation, and **the first occurrence (18:16) predates any of today's fat-expert-kernel work** -- this project's TileLang-cache testing earlier the same session already did several rapid recreates before sparkglm's kernel was ever touched. Every boot in this session, on both v8 and v9-fatfork, reached `Application startup complete` and passed its correctness checks regardless. Not attributing this to the grouped-prefill kernel specifically -- flagging it as a background signal worth watching if it recurs, consistent with how this file has always treated correlational-but-unconfirmed evidence (see the TileLang/#54317 crash entries above).

## Follow-ups: isolating the KV-capacity cost, and whether `gpu_memory_utilization` can go above 0.84 (2026-09-06)

Two follow-ups from the promotion decision above, chased before actually promoting: pin down the KV-capacity tradeoff precisely (the first write-up's number was contaminated by this fleet's own known boot-to-boot free-memory variance), and see whether v9-fatfork's larger memory margin at 0.84 (vs v8's razor-thin post-reboot 0.86) means it can safely run at a higher `gpu_memory_utilization` to claw back some of that KV capacity.

### KV-capacity cost, isolated with a true back-to-back A/B

Stopped v8, flushed, and booted v9-fatfork *immediately* after reading v8's own boot-time memory-profiler line, minimizing the time gap between the two measurements (rather than comparing boots from different points in the session, which is what produced the earlier, noisier "40-45% smaller" estimate). Confirmed the two boots actually shared comparable host state before trusting the comparison: free memory at startup was 106.18 GiB (v8) vs 106.08 GiB (v9-fatfork) -- 0.1 GiB apart.

| | consumed (weights+non-torch) | peak activation | CUDA graph | KV cache in use |
|---|---:|---:|---:|---:|
| v8@0.84 | 89.02 GiB | 5.81 GiB | 0.66 GiB | 7.38 GiB |
| v9-fatfork@0.84 | 89.66 GiB | 5.81 GiB | 0.68 GiB | 6.75 GiB |

**+0.64 GiB consumed per rank, -0.63 GiB KV cache per rank -- an ~8.5% KV-capacity reduction**, not the ~40-45% the noisier same-session-but-different-boot comparison suggested. Consistent with, and close to half of, sparkglm's own documented ~1.2 GiB/rank scratch figure (their number may reflect a different scratch-sizing path or measurement point). This is the number to trust going forward; the earlier estimate is superseded.

### `gpu_memory_utilization`: tested 0.85 and 0.86, both real cause for concern -- 0.84 stands, fleet-wide

Tested whether v9-fatfork's healthier idle margin at 0.84 (~5.3-5.6%, vs v8's post-reboot 0.86 knife-edge of ~1.3-1.6%) meant headroom existed to raise it, using `sparkrun run ... -o gpu_memory_utilization=X` (no recipe edit) to iterate quickly.

**0.86**: booted clean, KV cache rose to 9.25 GiB. Idle margin held in a *stable, oscillating* 3.4-4.3% band (the background cache-flusher's periodic reclaims visibly punctuating a slow decline) -- categorically healthier than v8's fixed ~1.3-1.6% knife-edge at the same nominal value. But under the actual **~33K-token prefill stress test** (the same test used to validate prefill speed, which also exercises real KV allocation), margin dropped to **2.44-2.84%** and stayed there -- close enough to the 2% earlyoom trigger, under a realistic serving condition (a long-context request with its KV blocks retained by prefix caching), that this was not called safe.

**0.85**: booted clean, KV cache **12.52 GiB** (nearly matching v8's own upper range -- would have closed almost the entire capacity gap). Idle margin measured **worse** than 0.86's (~2.76-3.01%, vs 0.86's 3.4-4.3%) -- itself a sign of how much this fleet's free-memory readings vary between successive boots, not a clean monotonic function of the GMU value. Running the identical 33K-token stress test **crashed it**: `EngineDeadError`, "AsyncLLM output_handler failed", no CUDA traceback, no explicit OOM message from vLLM itself. Checked `dmesg` directly for the exact crash window (03:40:23 UTC) -- the nearest NVRM kernel-level entry was 3.5 minutes earlier, not at the crash itself. **This is the same silent-kill signature this file has already documented and never fully root-caused** (the TileLang/#54317-pattern crashes, Crash B, Crash 4, above) -- no forensic trace at all, on either the application or kernel side.

**Confirmed the floor holds**: re-ran the identical 33K-token stress test against v9-fatfork@0.84 immediately after the 0.85 crash. Clean, 1,522.6-1,526.8 tok/s (matching the earlier ~1,510-1,524 tok/s range -- the prefill win is stable across 16K-34K and independent of `gpu_memory_utilization`, as expected). Margin under load: 2.86-2.99%, stable, comfortably above 2%.

**Notable, and worth tracking going forward**: 0.84's *own* margin has visibly eroded over the course of this single session -- ~5.3-5.6% idle, no-load, earlier in the day; ~2.86-2.99% under load, several hours and many container recreates later. The user's own hypothesis, raised live during this testing: this may be accumulating host-level memory pressure/fragmentation across a long uptime and many rapid reboots, not something specific to any one recipe -- consistent with v8 and v9-fatfork showing nearly identical GPU-side accounting (consumed memory within 0.64 GiB of each other) while host margin varies far more than that between boots. Periodic host reboots, or more aggressive host-memory hygiene beyond the existing `prelaunch_flush.sh` ritual, are worth investigating separately; not attempted here. **Conclusion: `gpu_memory_utilization` stays at 0.84, fleet-wide, for both v8 and v9-fatfork. This is a shared host-memory-margin ceiling, not a v9-fatfork-specific limitation** -- nothing here counts against promoting v9-fatfork itself.

## Verdict: promote

A real, reproducible, larger-than-sparkglm's-own +24.0% effective prefill win at ~16.3K-34K tokens, on this fleet's actual MTP-2 config, with zero measured decode regression, clean correctness/concurrency/soak results, a precisely-isolated ~8.5% KV-capacity cost (small, not the ~40-45% first estimated), and a memory margin that holds at the same `gpu_memory_utilization=0.84` this fleet already runs. **v9-fatfork promoted to production**, replacing v8. `gpu_memory_utilization` stays at 0.84 -- raising it is a fleet-wide question for a future session, not something this promotion should wait on or attempt.

## Upstream check: MiaAI-Lab's own MNBT/max_num_seqs guidance changed -- re-tested, staying at 7168 (2026-09-06)

Routine upstream check (MiaAI-Lab main, `c190db1` -> `3021f24`; Enntity/sparkglm; the specific vLLM issues this project tracks) turned up nothing code-level new from either recipe repo since our pins -- sparkglm has zero new commits, MiaAI-Lab's only changes are a TP4 lane (not applicable, this cluster is 2-node/TP2), GitHub issue-template meta, and one doc update worth checking: their `.env.example` comment for `MAX_NUM_BATCHED_TOKENS` now reads *"Current maintainer default at MAX_NUM_SEQS=4: 7168. On an independent MAX_NUM_SEQS=16 geometry, 2048 gave the best measured throughput/KV balance. Re-run 2048 vs 7168 before changing."*

This lands squarely on a gap in our own validation history: the original MNBT 2048->7168 test (+28.4%/+19.5% prefill, "max_num_batched_tokens 2048 -> 7168" above) was run **before** `max_num_seqs` was promoted 4->16 the next day -- the two changes were never tested together until now, and upstream is independently reporting the opposite conclusion at exactly this project's current concurrency setting.

**Re-tested directly** (`-o max_num_batched_tokens=2048` override on the shipped v9-fatfork recipe, `max_num_seqs=16` and the grouped-prefill kernel both held constant, same cache-unique-prompt methodology as the grouped-prefill A/B above):

| Metric | MNBT 7168 (shipped) | MNBT 2048 | Change |
|---|---:|---:|---:|
| Prefill @16.3K | ~1,510 tok/s | 1,295-1,298 tok/s (steady-state, n=3 of 4) | **-14%** |
| Prefill @33.7K | ~1,524 tok/s | 1,290.6-1,293.8 tok/s (n=3) | **-15%** |
| Consumed memory (weights+non-torch) | 89.66-89.75 GiB | 83.84 GiB | -5.8 GiB |
| Peak activation | 5.7-5.81 GiB | 2.49 GiB | -3.2 GiB |
| KV cache in use | 6.75-9.25 GiB | **15.89 GiB** | +1.7-2.4x |

**The tradeoff is real in both directions, not a free lunch.** Smaller MNBT genuinely shrinks the activation buffer (scales with the batching window, as expected) and appears to also shrink the grouped-prefill kernel's own scratch allocation (the 5.8 GiB drop in "consumed" memory is too large to be activation alone) -- together roughly doubling usable KV capacity. But raw prefill throughput drops ~14-15% at both tested scales, consistently -- this is upstream's own "independent geometry" finding, reproduced here, not contradicted.

**Verdict: keep 7168.** This project's own concurrency validation (`max_num_seqs` 4->16 section above) already showed KV was not the binding constraint at this fleet's tested load -- 41% peak utilization at 16 concurrent requests, zero preemptions. Trading away 14-15% of the prefill win this session was specifically built to get, in exchange for KV headroom this fleet isn't currently using, is the wrong trade *for this deployment* -- though evidently the right one for whatever workload produced upstream's "independent geometry" number. Re-check this if a future workload actually starts pressuring KV capacity (many concurrent long-context sessions, a `max_num_seqs` increase past 16, or a `gpu_memory_utilization` reduction) -- the 2048 config is now measured and ready to reconsider, not rejected outright.

## First real-world production window on v9-fatfork: 657 requests, zero failures, and a live re-run of the exact crash-4 trigger (2026-09-06)

Everything up to this point validated v9-fatfork with synthetic probes -- correctness suites, controlled A/Bs, concurrency sweeps, a soak. This is the first entry built from **actual user-driven production traffic**: ~15 hours of continuous uptime on the shipped config (`gpu_memory_utilization=0.84`, `max_num_batched_tokens=7168`), no restarts, no synthetic load generators.

**Aggregate metrics, read directly from `/metrics`:**

| Metric | Value |
|---|---:|
| Requests completed | 657 |
| `finished_reason=stop` | 657 (100%) |
| `finished_reason` = length / abort / error / repetition | 0 / 0 / 0 / 0 |
| Preemptions | 0 |
| Total prompt tokens | 66,704,984 |
| Prefix-cache hit rate | **93.4%** (62,354,432 / 66,745,593 queries) |
| New (cold) prompt tokens actually computed | ~4,354,136 |
| Total generation tokens | 118,853 |
| Avg TTFT | 14.04 s |
| Avg E2E latency | 20.62 s |
| Avg prefill time / avg decode time per request | 5.20 s / 6.58 s |
| Effective new-token prefill rate (cold tokens / summed prefill time) | ~1,275 tok/s |
| MTP-2 draft acceptance | 76.4% (71,711 / 93,842) |

**Zero failures of any kind.** No aborts, no errors, no preemptions, no truncated (`length`) completions -- every one of 657 requests ran to a natural stop. `dmesg` and the head node's `earlyoom` log show nothing across the entire window -- no NVRM entries, no OOM kills. Head-node memory margin held throughout (currently 3.12% available, never approached the 2% earlyoom trigger), though swap-free crept down slightly over the session (79.9% -> 77.9%) -- the same slow erosion pattern already flagged in the `gpu_memory_utilization` follow-ups above, now observed under real rather than synthetic load, still nowhere near concerning.

**The workload shape**: average ~101,500 prompt tokens per request against only ~181 output tokens, with a 93.4% cache-hit rate -- this is sustained, incrementally-growing-context usage (a long session or agentic loop keeping a large working context resident across many turns), not a stream of independent one-off prompts. The effective new-token prefill rate (~1,275 tok/s) sits a bit below the isolated solo-benchmark number (~1,510-1,524 tok/s) as expected -- real traffic carries scheduling contention and mixed request shapes an isolated A/B doesn't.

**The finding that actually matters**: at 12:28 PM EDT, `mhc_pre_big_fuse_with_norm_tilelang` -- the exact TileLang kernel whose first-ever JIT compile preceded the fourth production crash (see "A FOURTH production crash" above) by 20 seconds -- compiled live, under real traffic, for the first time on this boot:

```
16:28:46 UTC  WARNING jit_monitor: TileLang JIT compilation during inference:
              mhc_pre_big_fuse_with_norm_tilelang. This causes a latency spike...
16:28:46 UTC  TileLang begins to compile kernel `mhc_pre_big_fuse_with_norm_tilelang`
16:28:51 UTC  TileLang completes to compile kernel `mhc_pre_big_fuse_with_norm_tilelang`
```

**The server did not crash.** It was still healthy, serving cleanly, 3+ hours later at the time of this check. This is the first real-world test of v9's two relevant fixes acting together against the actual trigger condition that killed a prior boot -- the backported Mamba state-copy race fix (`#50729`, present since v8) and the TileLang JIT-cache persistence fix (this session, Sep 5). Worth being precise about what this does and doesn't show: v8 already had the race fix alone and still crashed 18 minutes after this exact kernel shape's live compile, so the race fix by itself was evidently insufficient for us; this is the first time the *combination* has actually been exercised against the trigger, and it held. Not proof of causation -- this project's own standard for that bar, per every other crash entry above -- but a real, specific, positive data point, not a vague "it's been fine so far."

Confirmed the newly-compiled kernel is now sitting in the host-persisted TileLang cache (kernel directory count: 8 -> 14 since the last check), so a future container recreate will load it from cache rather than risk a live recompile again. Two other novel shapes also JIT-compiled live without incident in the same window -- `BuildPrefillChunkMetadataKernel.kernel` (~11:17 AM) and `_kpool_softmax_rotate_write_cache_kernel`/`_kpool_tail_seed_kernel` (~12:52 PM) -- neither preceded by any prior crash signature, but now also cached for next time regardless.

**Verdict: the longest and first real-traffic validation window for v9-fatfork is clean.** No promotion decision changes here -- v9-fatfork was already production -- but this is the first evidence point from real usage rather than synthetic testing, and it happens to include a direct, favorable re-run of the one open question (does this project's crash-4 fix combination actually hold) that synthetic testing couldn't have manufactured on its own.
