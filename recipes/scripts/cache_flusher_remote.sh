#!/usr/bin/env bash
# Cache-flusher loop for GB10 during GLM-5.3-Flash model load.
# Keep GB10 page cache small so NVRM can allocate the KV slab.
# Runs for 25 min max, flushes whenever Cached > 40 GiB. NVIDIA KB 5776 remedy.
# (Port of ../cache_flusher.sh; sudo -n variant for sparkrun-provisioned sudoers.)
end=$((SECONDS+1500))
while [ $SECONDS -lt $end ]; do
  c=$(awk '/^Cached:/{print int($2/1048576)}' /proc/meminfo)
  if [ "${c:-0}" -gt 40 ]; then
    sync
    echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
  fi
  sleep 5
done
