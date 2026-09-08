# v12-combined: Track 1 + Track 2 + Track 4, combined pre-downtime candidate

Seeded from `recipes/build/glm53-exl3-v11-e3` (Track 2's E3 kernel graft),
plus:

- **Track 1** (`recipes/build/glm53-exl3-v10-vllmpatch/NOTES.md`): vLLM PR
  #55736 commit 2/3, the NoPE MQA write-path fix. Unconditional.
- **Track 4** (this session, `mmastrac/glm-5.3-flash-4x-gx10`-derived):
  `TORCH_MEM_FRACTION` + CUDA-allocator OOM-observer + memory-history
  recorder + per-worker stats reporter. Inert unless `TORCH_MEM_FRACTION`/
  `TORCH_MEM_TRACE_DIR`/`TORCH_MEM_STATS_DIR` are set at runtime -- zero
  behavior change by default.

Carries Track 2's full E3 graft unchanged: `_FAT_TIERS = ("grouped",
"kernel", "batched", "sorted", "legacy")`, sparkglm's existing
`exl3_grouped_prefill_k4` kernel-tier dispatch untouched, E3's own
`exl3_fat_moe_*` symbols and `grouped_fat_eligibility` gate. Both are
selectable independently at runtime: `EXL3_GROUPED_PREFILL_K4=1` (sparkglm,
already shipped) and `EXL3_FAT_GROUPED=1` (E3, new, default off) --
E3 wins when both are set and the layer is eligible; falls through to
sparkglm's kernel tier otherwise.

See `/home/luser/.claude/plans/dapper-moseying-clock.md` for the full plan
this build closes out, and `recipes/build/glm53-exl3-v10-vllmpatch/NOTES.md`
for what vLLM-PR items were deferred and why (real version drift found in
this image's installed vLLM, not risk-aversion).

## Validation status (update as testing proceeds)

- [x] Symbol check (both sparkglm's and E3's pybind11 symbols present)
- [ ] tinyGLM dispatch-correctness gate
- [ ] Real-checkpoint boot + memory-margin check
- [ ] Real-checkpoint A/B vs. current production (v9-fatfork)
