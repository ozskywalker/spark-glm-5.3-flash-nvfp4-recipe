#!/usr/bin/env bash
# GB10 pre-launch ritual for GLM-5.3-Flash (see docs/KV-HUNT-672K-TP2-RECORD.md):
#   1. drop_caches on every rank IMMEDIATELY before launch (hardening rule —
#      the one production crash happened on a boot that skipped this).
#   2. with --during-load, start the background cache-flusher loop on each node
#      for the duration of model load (keeps page cache small so NVRM can carve
#      the KV slab; NVIDIA KB 5776 remedy).
#   3. check host memory fragmentation (nr_free_pages_blocks, zone Normal)
#      after the flush and attempt a compaction pass if it looks low —
#      raw MemAvailable% is NOT a reliable signal for the NVRM
#      `_memdescAllocInternal` NV_ERR_NO_MEMORY failure class (found
#      2026-09-08): a node can show healthy MemAvailable% while having zero
#      contiguous higher-order free blocks, which is what that allocation
#      actually needs. This is advisory (WARN, not a hard fail) — the
#      threshold below is based on two data points, not a validated
#      statistical floor. See recipes/VALIDATION.md and the
#      host_fragmentation_xid31_reboot_risk memory note.
#   4. check host MemAvailable against what gpu_memory_utilization is about
#      to request, BEFORE the image pull/weight load/sync -- ported from
#      MiaAI-Lab PR #39 (2026-09-15 upstream-check round). On GB10 the GPU
#      shares the unified memory pool with the host; a host process holding
#      memory (not a container -- distinct from any container-level check)
#      leaves less than gpu_memory_utilization x MemTotal free, and the
#      worker then dies MID-BRING-UP with a `ValueError: Free memory on
#      device ... less than desired` deep in a hard-to-read worker log,
#      after already paying for the image pull and weight load. This is
#      FATAL by default (override via GLM53_PREFLIGHT_SKIP_MEMORY_CHECK=1)
#      -- unlike the fragmentation check above, this failure mode is a
#      guaranteed, deterministic hard-stop, not a probabilistic risk, so
#      failing fast before the expensive part of boot is strictly better
#      than failing loud mid-bring-up. Deliberately reads MemAvailable, not
#      MemFree, for the same reason the fragmentation check does -- a large
#      weight rsync fills reclaimable page cache and would false-fail on
#      MemFree.
#
# Usage:
#   prelaunch_flush.sh <host1,host2,...> [--during-load]
set -euo pipefail

# Below this many contiguous free blocks in zone Normal, warn and attempt a
# compaction pass. Chosen well above the observed-crash value (0, fresh boot
# immediately before a 240K prefill failure) and well below the two
# observed-healthy values (49,152 on live idle production; 147,456 on a
# settled boot's own compile event) — deliberately conservative so it only
# fires on a genuinely depleted node, not routine variance.
FRAG_WARN_THRESHOLD="${FRAG_WARN_THRESHOLD:-2000}"

# This fork's shipped gpu_memory_utilization (v20-upstreamsync, both dense-FP8
# and maxprefill sibling recipes) -- override if launching a recipe with a
# different value. Headroom matches MiaAI-Lab's own default; both are
# overridable independently.
GLM53_PREFLIGHT_GMU="${GLM53_PREFLIGHT_GMU:-0.84}"
GLM53_PREFLIGHT_MEMORY_HEADROOM_KIB="${GLM53_PREFLIGHT_MEMORY_HEADROOM_KIB:-2097152}"
GLM53_PREFLIGHT_SKIP_MEMORY_CHECK="${GLM53_PREFLIGHT_SKIP_MEMORY_CHECK:-0}"

HOSTS="${1:?usage: prelaunch_flush.sh <host1,host2,...> [--during-load]}"
MODE="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

check_fragmentation() {
  local host="$1"
  local blocks blocks_after
  scp -q -o BatchMode=yes "$SCRIPT_DIR/frag_check_remote.sh" "$host":/tmp/glm53_frag_check.sh
  blocks="$(ssh -o BatchMode=yes "$host" 'bash /tmp/glm53_frag_check.sh' 2>/dev/null || echo "")"
  if [ -z "$blocks" ]; then
    echo "  fragmentation check: could not read nr_free_pages_blocks (unexpected /proc/zoneinfo format) — skipping"
    return
  fi
  if [ "$blocks" -ge "$FRAG_WARN_THRESHOLD" ]; then
    echo "  fragmentation check: $blocks contiguous free blocks in zone Normal (>= $FRAG_WARN_THRESHOLD, OK)"
    return
  fi
  echo "  WARNING: only $blocks contiguous free blocks in zone Normal (threshold $FRAG_WARN_THRESHOLD)."
  echo "           This node may be fragmented enough to risk an NVRM NV_ERR_NO_MEMORY failure under"
  echo "           large-allocation pressure (long-context prefill soon after this boot, in particular)."
  echo "           Attempting a synchronous compaction pass..."
  ssh -o BatchMode=yes "$host" 'echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null 2>&1' || true
  blocks_after="$(ssh -o BatchMode=yes "$host" 'bash /tmp/glm53_frag_check.sh' 2>/dev/null || echo "$blocks")"
  if [ "$blocks_after" -ge "$FRAG_WARN_THRESHOLD" ]; then
    echo "  post-compaction: $blocks_after contiguous free blocks — above threshold, proceeding is more comfortable."
  else
    echo "  post-compaction: $blocks_after contiguous free blocks — still below threshold."
    echo "  WARNING: a boot right now carries elevated risk of the NVRM/Xid-31 failure class documented"
    echo "           2026-09-08 (see recipes/VALIDATION.md). This is advisory, not a hard block — consider"
    echo "           letting the node sit idle longer before a long-context boot, or proceed with awareness."
  fi
}

check_memory_available() {
  local host="$1"
  local meminfo mem_total_kib mem_avail_kib threshold_kib
  meminfo="$(ssh -o BatchMode=yes "$host" 'grep -E "^MemTotal:|^MemAvailable:" /proc/meminfo' 2>/dev/null || echo "")"
  mem_total_kib="$(echo "$meminfo" | awk '/^MemTotal:/ {print $2}')"
  mem_avail_kib="$(echo "$meminfo" | awk '/^MemAvailable:/ {print $2}')"
  if [ -z "$mem_total_kib" ] || [ -z "$mem_avail_kib" ]; then
    echo "  memory preflight: could not read /proc/meminfo — skipping"
    return 0
  fi
  # Integer KiB arithmetic throughout; GMU is e.g. 0.84 so scale by 100 first.
  threshold_kib=$(( mem_total_kib * $(printf '%.0f' "$(echo "$GLM53_PREFLIGHT_GMU * 100" | bc)") / 100 + GLM53_PREFLIGHT_MEMORY_HEADROOM_KIB ))
  if [ "$mem_avail_kib" -ge "$threshold_kib" ]; then
    echo "  memory preflight: ${mem_avail_kib} KiB available >= ${threshold_kib} KiB needed (gmu=${GLM53_PREFLIGHT_GMU} + $((GLM53_PREFLIGHT_MEMORY_HEADROOM_KIB / 1048576)) GiB headroom), OK"
    return 0
  fi
  echo "  MEMORY PREFLIGHT FAILED on $host: only ${mem_avail_kib} KiB (MemAvailable) free," >&2
  echo "    but gpu_memory_utilization=${GLM53_PREFLIGHT_GMU} on a ${mem_total_kib} KiB host needs" >&2
  echo "    ${threshold_kib} KiB free (incl. $((GLM53_PREFLIGHT_MEMORY_HEADROOM_KIB / 1048576)) GiB headroom)." >&2
  echo "    A host process is holding memory a container-level check won't see. Booting now would" >&2
  echo "    pay for the image pull + weight load, then die mid-bring-up with a hard-to-read" >&2
  echo "    'Free memory on device ... less than desired' worker error instead. Free memory on" >&2
  echo "    $host first, or set GLM53_PREFLIGHT_SKIP_MEMORY_CHECK=1 to proceed anyway." >&2
  return 1
}

IFS=',' read -ra HOST_LIST <<< "$HOSTS"
for host in "${HOST_LIST[@]}"; do
  echo "== $host =="
  # fail loudly if the flush could not run (sudo -n refused, etc.) — a node
  # that silently skipped drop_caches is how the production crash happened
  ssh -o BatchMode=yes "$host" \
    'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || { echo "FLUSH FAILED" >&2; exit 1; }; echo flushed'
  check_fragmentation "$host"
  if [ "$GLM53_PREFLIGHT_SKIP_MEMORY_CHECK" != "1" ]; then
    check_memory_available "$host"
  fi
  if [ "$MODE" = "--during-load" ]; then
    scp -q -o BatchMode=yes "$SCRIPT_DIR/cache_flusher_remote.sh" "$host":/tmp/glm53_cache_flusher.sh
    # stop any stale flusher via pidfile (pkill -f would match this very
    # ssh command line — it contains the script path), then start fresh.
    # The flusher self-terminates after 25 min regardless.
    ssh -o BatchMode=yes "$host" \
      'cat /tmp/glm53_cache_flusher.pid 2>/dev/null | xargs -r kill 2>/dev/null; nohup bash /tmp/glm53_cache_flusher.sh >/tmp/glm53_cache_flusher.log 2>&1 & echo $! > /tmp/glm53_cache_flusher.pid; echo "flusher started pid $(cat /tmp/glm53_cache_flusher.pid)"'
  fi
done
echo "pre-launch flush complete on: $HOSTS"
