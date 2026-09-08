#!/usr/bin/env bash
# Reports nr_free_pages_blocks for zone Normal (contiguous higher-order free
# blocks) from /proc/zoneinfo. Run remotely by prelaunch_flush.sh.
#
# Why this and not MemAvailable%: the 2026-09-08 host/driver-level trace of
# the 240K NVRM `_memdescAllocInternal` NV_ERR_NO_MEMORY failure found that
# raw MemAvailable% looked similar (~2-4.5%) in both a successful and a
# failed boot -- the field that actually discriminated was contiguous
# free-block count (147,456 on the successful run vs. 0 on the failed one).
# A node can show acceptable raw free memory while having zero usable
# contiguous blocks for the allocation NVRM needs. See recipes/VALIDATION.md
# and the host_fragmentation_xid31_reboot_risk memory note.
set -euo pipefail

awk '
  /^Node/ && /zone/ { in_normal = ($NF == "Normal") }
  in_normal && /nr_free_pages_blocks/ { print $2; exit }
' /proc/zoneinfo
