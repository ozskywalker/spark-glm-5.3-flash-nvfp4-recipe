# VALIDATION.md archive: v12-combined

_Archived from `recipes/VALIDATION.md` lines 3888-3961 (git-blame-verified,_
_no content altered). See `recipes/VALIDATION.md` for the current index and_
_where to look for common problems._

---


## Backporting vLLM upstream PRs, MiaAI-Lab's E3 grouped-MoE kernel, and a real memory-pressure finding at long context: v12-combined promoted

An investigate-only upstream sweep (production traffic running, no changes made) surfaced five workstreams: five vLLM PRs to evaluate for backport, MiaAI-Lab's own "E3" grouped-MoE kernel as an alternative to the sparkglm-derived kernel already in production, a pilot of `mmastrac/mentat` as a lower-overhead Ray replacement, root-causing this project's long-unsolved "silent kill" crash signature, and small corroborating fixes from a sibling 4x-GB10 project. Full plan at `/home/luser/.claude/plans/dapper-moseying-clock.md`. User gave an explicit go-ahead to schedule downtime and execute once the investigation-only report was reviewed; production was stopped (`sparkrun stop`) to free the GPUs for testing.

### Track 1: vLLM PR backports -- smaller than planned, for real reasons

Five PRs were investigated (`#55736`, `#55738`, `#55222`, `#55563`, `#55737`), all unmerged in vLLM mainline. Checked each against this image's actual installed vLLM source, not just the PR diffs -- found real version drift, not just risk-aversion reasons to defer:

- **Applied**: `#55736` commit 2/3 only ("Write the absorbed MQA query token-major and skip the empty RoPE concat") -- verified byte-for-byte against this image's installed `mla_attention.py`/`flashinfer_mla_sparse.py`, unconditional (correctness-neutral perf fix, no env flag). New overlay patch: `patch_nope_mqa_fix.py`.
- **Deferred, real structural mismatches found**:
  - `#55736` commit 1 (KDA Triton stride cleanup): target module (`ops/third_party/kda/`) doesn't exist in this build -- our `kda.py` imports from a differently-organized `vllm.third_party.flash_linear_attention.ops.kda` instead. Real rewrite, not a cherry-pick.
  - `#55736` commit 3 (MoE router-GEMM dedup): our installed `model.py` (~line 266) explicitly computes `router_logits, _ = self.gate(hidden_states)` before calling `self.experts(...)`, with an existing comment "pre-computed router_logits, so compute them here unconditionally" -- reads as an intentional design choice this PR would override. Confirmed our EXL3 `Exl3Config.apply()` takes precomputed `topk_ids`/`topk_weights`, not `router_logits` directly -- the actual risk is whether `MoERunner`'s internal logits computation (which this commit relies on) produces identical results to the current explicit call. A silently-wrong routing bug is a correctness risk, not just perf -- not worth taking without a much closer trace than this session had time for.
  - `#55738` (dense/masked-MHA NoPE prefill): one target file (`flash_attn.py`) fails to import in this exact image (`ImportError: cannot import name 'compile_flash_attn_varlen_func_from_specs'`, real version skew), and `_is_masked_mha_available` in `sparse_mla_attention.py` is structurally different (hard-gated on `is_device_capability_family(100)`, not the tuple-list check the PR's diff assumes).
  - `#55737` FlashKDA: the compiled `vllm._flashkda_C.abi3.so` extension already exists in this image (confirmed via `importlib.util.find_spec` inside the container), but zero Python integration is wired into `kda.py` -- a real feature-addition-sized task, not a patch. Deferred to its own future pass.
- **`#55222`**: fp8-plan-dtype part is SM90-only, doesn't apply to SM121/GB10. The indexer-workspace fix is a naive `get_max_prefill_buffer_size(vllm_config) // self.index_kpool` -- our own `overlay/patch_indexer_workspace.py` (`GLM53_INDEXER_WORKSPACE=rightsize`) is a strict superset: it computes the legal per-step maximum (`min(max_num_seqs, max_num_batched_tokens) * cdiv(max_model_len + num_speculative_tokens, compress_ratio)`, clamped to never exceed stock) plus a fail-closed builder-init cross-check between `hf_text_config.index_kpool` and the runtime's `kv_cache_spec.compress_ratio`. Nothing to adopt.
- **`#55563`** (closed, self-withdrawn by its author, not superseded): fixes the same 2048-wide SM120 top-k buffer overflow via a different strategy (shrink `index_topk` 2048->2045) than what we already carry. Checked our own build: it already has its own independent fix for this exact bug (a Dockerfile-level "drop the lowest-ranked pool instead of the tail" strategy: 511 pools + 3 tail = 2047 of 2048 candidates, keeping `select_k=512`'s fused fast path) -- reasoned independently of `#55563`. Nothing to adopt.

Track 5's `busy_loop_s` tuning item (from the 4x-gx10 sibling repo) turned out to be the same knob as our existing `GLM53_SPINWAIT_MS`, already tested and rejected at the equivalent low value (16ms wins over 2ms on this fleet). The PDL-off-SM12x item was already present in our Dockerfile, independently arrived at for the same "races KDA state kernels" reason the 4x-gx10 repo found. `thinking_budget_guard` (vllm#50473) doesn't apply -- this project doesn't use `thinking_token_budget` anywhere.

### Track 2: MiaAI-Lab's E3 grouped-MoE kernel -- grafted, tier-reconciled, and it works

MiaAI-Lab's own independently-developed grouped-MoE kernel (`EXL3_FAT_GROUPED`, commits `1a0feb0`/`2c0ebe5`) is architecturally distinct from sparkglm's kernel already in production -- not stackable as-is, both solve the same fat-expert-prefill problem with different designs (sparkglm's piggybacks inside the existing "kernel" tier; E3 adds a whole new top-level tier above it). Grafted into a new build (`recipes/build/glm53-exl3-v11-e3/`, seeded from the current production build tree, NOT from E3's own config defaults -- `MAX_MODEL_LEN`/`GPU_MEM_UTIL` changes from `2c0ebe5`/`f4ef41e` were deliberately not pulled, since `GPU_MEM_UTIL=0.85` is a value this fleet has already proven crashes under load).

**Tier-ladder reconciliation** (`overlay/exl3.py`): final `_FAT_TIERS = ("grouped", "kernel", "batched", "sorted", "legacy")` -- E3 sits above sparkglm's unchanged kernel-tier dispatch. E3's eligibility gate runs first, once per layer at weight-load time; if `EXL3_FAT_GROUPED=1` but a layer is ineligible, it deliberately downgrades to `"kernel"` so sparkglm's own shape gate becomes the real fallback rather than racing it. **Caught a real bug during the graft**: E3's eligibility check originally ran before `layer._exl3_inners` was populated, which would have silently disabled E3 for every layer (`no_inners` on every check) -- fixed by moving the call to run after `_exl3_inners` exists, matching what upstream's own diff required. Scratch caches kept deliberately separate (`_FAT_GROUPED_CACHE` vs. sparkglm's existing `_GROUPED_PREFILL_SCRATCH_CACHE`). Diagnostic schema conflict resolved: our fork's schema was already at 2 for sparkglm's key set; upstream's E3 patch also wanted schema 2 for a different key set -- renumbered E3's bump to schema 3 (union of both key sets) rather than let "schema 2" silently mean two different things.

Build succeeded after fixing two real bugs (a missing Dockerfile `COPY` for the new CUDA source files; a diagnostic-schema test assertion). Symbol check confirmed both sparkglm's `exl3_grouped_prefill_k4` and all 5 new E3 symbols present, `exl3_decode_moe_k4` correctly absent.

### Combined into v12-combined; tinyGLM gates passed for all three builds

Combined Track 1's NoPE MQA fix and Track 4's OOM-observer diagnostic (below) onto Track 2's E3 graft in one final pre-downtime candidate, `recipes/build/glm53-exl3-v12-combined/`, rather than validating four separate real-checkpoint boots. Each of v10 (Track 1 alone), v11-e3 (Track 2 alone), and v12-combined passed the same tinyGLM dispatch-correctness gate: clean TP2 boot, correct diagnostic tier (`effective_tier=kernel` for v10, `effective_tier=grouped grouped_eligible=1` for v11-e3 and v12-combined), a real chat completion round-trips, and two identical requests produce byte-identical output (determinism confirmed) in all three cases.

### Real-checkpoint validation: E3 helps at scale, and a genuine safety finding along the way

Real-checkpoint sanity probe passed cleanly on v12-combined (decode 25.9-27.8 tok/s, matching prior baselines). Prefill measured via `probe_longctx.py` at increasing context sizes, with the memory margin checked between every run (idle earlier in this session already looked deceptively fine right up until real load exposed risk -- same discipline as every prior GMU investigation in this file).

**Found and fixed a real methodology bug in `probe_longctx.py` first**: `build_document()` used a hardcoded `seed=1234`, so repeat runs at the same `--tokens` value produced byte-identical prompts -- a warm prefix-cache hit on the second/third run, not a real speedup (TTFT dropped from 11.7s to 3.3s for an *identical* 16K-token prompt across three back-to-back runs before this was caught). Added a `--seed` argument (default: a fresh random seed each run) so cold-prefill timing is trustworthy without an explicit opt-in to reuse content.

With that fixed, cache-unique cold-prefill numbers on v12-combined (E3 active, `EXL3_FAT_GROUPED=1`):

| Context (actual prompt tokens) | TTFT | Effective prefill |
|---|---|---|
| ~15,960 (n=3, consistent) | 10.2-10.3s | ~1,557 tok/s |
| ~63,970 | 39.4s | ~1,623 tok/s |
| ~127,910 | 79.6s | ~1,607 tok/s |
| ~239,960 | 152.3s | ~1,575 tok/s |

No cliff, no degradation up through our real `max_model_len` ceiling (262,144) -- holds steady across the whole range at ~1,560-1,620 tok/s, comparable to or modestly better than v9-fatfork's own already-validated ~1,510-1,524 tok/s at the 16K scale (sparkglm kernel alone). This is real, new evidence: MiaAI-Lab's own E3 benchmarks stop at 100K and explicitly flag 256K/300K as "unmeasured" -- this is the first measurement of E3's behavior at our actual production context range, on our actual workload/quantization, and it holds up.

**Then, at the 240K run, real NVRM out-of-memory events appeared on both nodes** (`NVRM: nvCheckOkFailedNoLog: Check failed: Out of memory [NV_ERR_NO_MEMORY] ... _memdescAllocInternal`), and node 2's margin dropped to 2.52% mem-available with swap-free already at 79.85% -- uncomfortably close to earlyoom's simultaneous 2%/80% trigger condition. **The container did not crash** -- the request completed correctly (all 4 planted codes retrieved) and the server kept serving. This is the exact crash signature this project has never been able to attribute (see every "silent kill" entry in this file, and tonyd2wild's issue #19 investigated earlier this session) -- caught live, for the first time, with a concrete trigger (a ~240K-token single-request prefill).

**Isolated whether E3 caused it**: rebuilt a control config (`glm-5.3-flash-exl3-v12-control-vllm.yaml`, identical image, `EXL3_FAT_GROUPED=0` -- sparkglm's kernel only, matching production's exact kernel choice) and re-ran the identical 240K test. **The same NVRM out-of-memory events reproduced on both nodes**, at the equivalent point in the test, with a comparable margin drop (node 2: 5.44% -> 3.27% mem-avail, vs. 4.75% -> 2.52% with E3 on -- similar magnitude). Interestingly, the control's cold prefill was *slower* at this scale (169.2s TTFT / ~1,418 tok/s vs. E3's 152.3s / ~1,575 tok/s), consistent with E3 giving a real ~11% win at this exact context size in this single comparison.

**Conclusion: this is a pre-existing, kernel-independent memory-pressure risk at ~240K-token single-request prefill on this fleet at `gpu_memory_utilization=0.84`, not something E3 introduces.** It already exists in current production (which shares the same sparkglm kernel, same GMU, same general memory budget) -- it just hadn't been tested at this exact scale before (this project's own controlled long-context validations have gone up to ~33-34K "actual tokens"; real production traffic averages ~101.5K with occasional excursions, per the "First real-world production window" entry above, but a clean 240K single-request case hadn't been manufactured and watched this closely before). All memory recovered fully (90-93% available) within seconds of stopping both test boots -- this is boot-footprint pressure, not a permanent host-level leak.

### Track 3: Mentat -- closed, wrong mechanism

See the dedicated memory note `mentat_track_closed` -- this deployment uses vLLM's `mp` executor backend, not `ray`, so there is no Ray control-plane overhead for `mmastrac/mentat` (a Ray-executor-specific shim) to replace. Not pursued further; a future "switch to `ray`+Mentat" investigation would need its own justification independent of Mentat's overhead claim.

### Track 4: OOM-observer diagnostic -- built, verified, not yet armed

Ported `mmastrac/glm-5.3-flash-4x-gx10`'s `spark_mem_trace.py`/`worker_memory_cap.py` mechanism (`TORCH_MEM_FRACTION` per-worker memory cap + a CUDA-allocator OOM-observer that dumps a stack trace instead of a silent death + memory-history recorder + periodic stats reporter) into `overlay/patch_oom_observer.py` + `overlay/glm53_oom_observer_runtime.py`, now baked into `v12-combined`. Preflight-verified twice (once by the implementing agent, once independently by the coordinator) against the real installed `gpu_worker.py` -- anchor text matched byte-for-byte. **Inert by default** (`TORCH_MEM_FRACTION`/`TORCH_MEM_TRACE_DIR`/`TORCH_MEM_STATS_DIR` all unset in the promoted recipe) -- this session's real-checkpoint testing found the actual NVRM OOM trigger (see above) before this diagnostic was ever armed, so it hasn't been exercised against a real event yet. Worth arming on a future long-context stress pass specifically targeting the 240K-token risk found above.

### Track 5: closed, all four items resolved

See the Track 1 section above (PDL-off-SM12x already present, `busy_loop_s` resolved as a duplicate of `GLM53_SPINWAIT_MS`, `thinking_budget_guard` not applicable, SM121 NoPE/MLA + K-pool/indexer items were corroboration only, folded into Track 1's own findings).

### Verdict: promote v12-combined

Clean win, no new risk: E3 holds up at real scale (comparable-to-better prefill throughput from 16K through 240K, no correctness issues, tinyGLM and real-checkpoint gates both pass), Track 1's one safe backport is verified byte-for-byte compatible, and Track 4's diagnostic rides along for free. The 240K memory-pressure finding is real and worth its own follow-up, but it is **not a reason to withhold promotion** -- it affects the current production kernel equally, isn't new, and doesn't get worse under v12-combined (E3 was, if anything, faster and no less safe at that exact scale in this comparison). Promoted 2026-09-08: `recipes/glm-5.3-flash-exl3-v12-combined-vllm.yaml` is now the shipped default, `EXL3_FAT_GROUPED=1` and `EXL3_GROUPED_PREFILL_K4=1` both set (E3 primary, sparkglm as the real fallback chain), `gpu_memory_utilization` unchanged at 0.84. `v9-fatfork` retained as the last known-good rollback target.

### New track identified, needs human review before further autonomous work

**Investigate the ~240K-token single-request prefill memory-pressure risk properly**, per this session's own instruction to document new tracks for human review rather than chase them autonomously. Candidate next steps, none yet started: arm Track 4's OOM-observer (`TORCH_MEM_FRACTION`/`TORCH_MEM_TRACE_DIR`) on a future boot and deliberately reproduce this exact 240K scenario to get a real stack trace instead of just a margin/dmesg correlation; determine whether this is specific to single-request prefill (vs. the same total tokens spread across concurrent smaller requests); consider whether `gpu_memory_utilization` needs a context-length-aware ceiling rather than one fixed value fleet-wide; cross-reference against tonyd2wild's issue #19 (`MemFree` vs. `MemAvailable`, `watermark_boost_factor`) findings from earlier this session -- see the `memfree_memavailable_gap` memory note.
