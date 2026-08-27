#!/usr/bin/env bash
# GB10 pre-launch ritual for GLM-5.3-Flash (see docs/KV-HUNT-672K-TP2-RECORD.md):
#   1. drop_caches on every rank IMMEDIATELY before launch (hardening rule —
#      the one production crash happened on a boot that skipped this).
#   2. with --during-load, start the background cache-flusher loop on each node
#      for the duration of model load (keeps page cache small so NVRM can carve
#      the KV slab; NVIDIA KB 5776 remedy).
#
# Usage:
#   prelaunch_flush.sh <host1,host2,...> [--during-load]
set -euo pipefail

HOSTS="${1:?usage: prelaunch_flush.sh <host1,host2,...> [--during-load]}"
MODE="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IFS=',' read -ra HOST_LIST <<< "$HOSTS"
for host in "${HOST_LIST[@]}"; do
  echo "== $host =="
  # fail loudly if the flush could not run (sudo -n refused, etc.) — a node
  # that silently skipped drop_caches is how the production crash happened
  ssh -o BatchMode=yes "$host" \
    'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || { echo "FLUSH FAILED" >&2; exit 1; }; echo flushed'
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
