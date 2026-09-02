#!/usr/bin/env bash
# Pull torch-profiler traces off both TP ranks after a /stop_profile.
#
# Traces are written inside each node's container (torch_profiler_dir), one
# set per rank, so they have to be copied out of the container and then off
# the node. Rank 0 lives on the head node; rank 1 on the worker. Both are
# collected because a TP-imbalance question ("is one rank waiting on the
# other?") can only be answered by comparing them.
#
# Usage: collect_traces.sh [OUTDIR] [PROFILE_DIR_IN_CONTAINER]

set -uo pipefail

OUTDIR="${1:-./traces}"
PROFDIR="${2:-/profiles}"
NODES="${NODES:-10.7.0.87 10.7.0.142}"

mkdir -p "$OUTDIR"

for NODE in $NODES; do
  CONTAINER=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$NODE" \
    "docker ps --format '{{.Names}}' | grep sparkrun | head -1" 2>/dev/null)
  if [[ -z "$CONTAINER" ]]; then
    echo "[$NODE] no sparkrun container, skipping"
    continue
  fi
  FILES=$(ssh -o BatchMode=yes "$NODE" \
    "docker exec $CONTAINER bash -lc 'ls -1 $PROFDIR/ 2>/dev/null'" 2>/dev/null)
  if [[ -z "$FILES" ]]; then
    echo "[$NODE/$CONTAINER] $PROFDIR is empty"
    continue
  fi
  echo "[$NODE/$CONTAINER] found:"
  echo "$FILES" | sed 's/^/    /'
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    ssh -o BatchMode=yes "$NODE" "docker cp '$CONTAINER:$PROFDIR/$f' '/tmp/$f'" >/dev/null 2>&1 \
      && scp -q -o BatchMode=yes "$NODE:/tmp/$f" "$OUTDIR/${NODE//./_}__$f" \
      && ssh -o BatchMode=yes "$NODE" "rm -f '/tmp/$f'" >/dev/null 2>&1
  done <<< "$FILES"
done

echo
echo "collected into $OUTDIR:"
ls -laS "$OUTDIR" 2>/dev/null | head -20
