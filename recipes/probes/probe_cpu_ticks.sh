#!/usr/bin/env bash
# Real CPU-seconds (not wall-time-in-frame) for EngineCore + Worker_TP0
# during a fixed decode workload, via /proc/<pid>/stat utime+stime deltas.
#
# py-spy's %-of-samples-in-a-frame cannot tell "genuinely spinning and
# burning CPU" apart from "blocked in a syscall, using ~0 CPU" — both show
# as "the process is in this frame" to a wall-clock sampler. This is the
# metric that actually answers PR96's claim (-85.3% EngineCore CPU).
#
# Usage: probe_cpu_ticks.sh [NODE] [CONTAINER] [LABEL]

set -euo pipefail
NODE="${1:-10.7.0.87}"
CONTAINER="${2:-}"
LABEL="${3:-run}"
BASE_URL="${BASE_URL:-http://10.7.0.87:8000}"
MODEL="${MODEL:-glm-5.3-flash-exl3-v2}"

if [[ -z "$CONTAINER" ]]; then
  CONTAINER=$(ssh -o BatchMode=yes "$NODE" "docker ps --format '{{.Names}}' | grep sparkrun | head -1")
fi

read_ticks() {
  # sum of utime+stime (field 14+15) in clock ticks, for both PIDs
  ssh -o BatchMode=yes "$NODE" "docker exec $CONTAINER bash -lc '
    for P in EngineCore Worker_TP0; do
      PID=\$(pgrep -f \"VLLM::\$P\" | head -1)
      [ -z \"\$PID\" ] && { echo \"\$P 0\"; continue; }
      read -r _ _ _ _ _ _ _ _ _ _ _ _ _ UT ST _ < /proc/\$PID/stat
      echo \"\$P \$((UT+ST))\"
    done
  '"
}

HZ=$(ssh -o BatchMode=yes "$NODE" "docker exec $CONTAINER getconf CLK_TCK" 2>/dev/null || echo 100)

echo "[$LABEL] before:"
BEFORE=$(read_ticks)
echo "$BEFORE"

T0=$(date +%s.%N)
python3 - "$BASE_URL" "$MODEL" <<'PYEOF'
import json, sys, urllib.request
base, model = sys.argv[1], sys.argv[2]
payload = {"model": model,
           "messages": [{"role": "user", "content":
                         "Write a detailed 900-word essay about the history of computing, "
                         "covering mechanical calculators, vacuum tubes, transistors, "
                         "integrated circuits, and modern accelerators."}],
           "max_tokens": 900, "temperature": 0,
           "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(base + "/v1/chat/completions",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=600) as r:
    body = json.load(r)
print("completion_tokens:", body["usage"]["completion_tokens"])
PYEOF
T1=$(date +%s.%N)
WALL=$(python3 -c "print(f'{$T1-$T0:.3f}')")

echo "[$LABEL] after:"
AFTER=$(read_ticks)
echo "$AFTER"

echo "[$LABEL] wall=${WALL}s hz=$HZ"
paste <(echo "$BEFORE") <(echo "$AFTER") | while read -r P B _ A; do
  DT=$(python3 -c "print(f'{($A-$B)/$HZ:.3f}')")
  PCT=$(python3 -c "print(f'{100*($A-$B)/$HZ/$WALL:.1f}')")
  echo "[$LABEL] $P: cpu_s=$DT  (${PCT}% of wall)"
done
