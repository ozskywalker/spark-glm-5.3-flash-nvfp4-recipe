#!/usr/bin/env bash
# Process-level CPU profile of the vLLM serving processes under a live load.
#
# Why this exists: the Prometheus metrics say decode is 94-96% of a short
# request's end-to-end time, but they cannot say whether that decode time is
# GPU work or CPU-side orchestration. At batch=1 on a 20-core GB10 those are
# very different problems with very different fixes. py-spy sampling the
# EngineCore and Worker processes while a generation is actually in flight
# separates them.
#
# py-spy needs CAP_SYS_PTRACE and the container runs as an unprivileged user
# under ptrace_scope=1, so every exec here is -u root --privileged. Without both
# py-spy returns a bare 'Permission Denied'.
#
# py-spy is installed into the container at run time (it is not in the image,
# and the install does not survive a container restart, so this re-installs
# every time rather than assuming).
#
# Usage:
#   probe_cpu_profile.sh [NODE] [CONTAINER] [DURATION_S] [OUTDIR]
# Defaults target the head node / rank 0.

set -euo pipefail

NODE="${1:-10.7.0.87}"
CONTAINER="${2:-}"
DURATION="${3:-25}"
OUTDIR="${4:-./cpu_profiles}"
BASE_URL="${BASE_URL:-http://10.7.0.87:8000}"
MODEL="${MODEL:-glm-5.3-flash-exl3-v2}"

if [[ -z "$CONTAINER" ]]; then
  CONTAINER=$(ssh -o BatchMode=yes "$NODE" "docker ps --format '{{.Names}}' | grep sparkrun | head -1")
fi
echo "node=$NODE container=$CONTAINER duration=${DURATION}s"

mkdir -p "$OUTDIR"

# pip may install into a user prefix (HOME is /tmp in this image), so resolve
# the binary by search rather than trusting PATH.
ssh -o BatchMode=yes "$NODE" "docker exec -u root --privileged $CONTAINER bash -lc '
  PYSPY=\$(command -v py-spy 2>/dev/null || find /tmp/.local/bin /root/.local/bin /usr/local/bin -name py-spy -type f 2>/dev/null | head -1)
  if [ -z \"\$PYSPY\" ]; then
    pip install -q py-spy >/dev/null 2>&1
    PYSPY=\$(find /tmp/.local/bin /root/.local/bin /usr/local/bin -name py-spy -type f 2>/dev/null | head -1)
  fi
  [ -n \"\$PYSPY\" ] || { echo NO_PYSPY; exit 1; }
  echo \"py-spy ready at \$PYSPY\"
'"

# Drive a long, steady single-stream generation so the sampled window is
# dominated by the decode loop rather than by prefill or by an idle server.
python3 - "$BASE_URL" "$MODEL" <<'PYEOF' &
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
print("load-generator completion_tokens:", body["usage"]["completion_tokens"])
PYEOF
LOADPID=$!

sleep 3   # let prefill finish so the sample window is decode-steady-state

for PROC in EngineCore Worker_TP0; do
  echo "--- sampling $PROC for ${DURATION}s ---"
  ssh -o BatchMode=yes "$NODE" "docker exec -u root --privileged $CONTAINER bash -lc '
    PYSPY=\$(command -v py-spy 2>/dev/null || find /tmp/.local/bin /root/.local/bin /usr/local/bin -name py-spy -type f 2>/dev/null | head -1)
    PID=\$(pgrep -f \"VLLM::$PROC\" | head -1)
    echo \"$PROC pid=\$PID\"
    \$PYSPY record --pid \$PID --duration $DURATION --rate 200 --nonblocking \
      --format speedscope -o /tmp/pyspy_$PROC.speedscope.json 2>&1 | tail -2
    \$PYSPY record --pid \$PID --duration 8 --rate 200 --nonblocking \
      --format flamegraph -o /tmp/pyspy_$PROC.svg >/dev/null 2>&1 || true
    ls -la /tmp/pyspy_$PROC.* 2>/dev/null
  '" &
done
wait

wait $LOADPID 2>/dev/null || true

for PROC in EngineCore Worker_TP0; do
  for EXT in speedscope.json svg; do
    ssh -o BatchMode=yes "$NODE" "docker cp $CONTAINER:/tmp/pyspy_$PROC.$EXT /tmp/pyspy_$PROC.$EXT" 2>/dev/null || continue
    scp -q -o BatchMode=yes "$NODE:/tmp/pyspy_$PROC.$EXT" "$OUTDIR/" 2>/dev/null || true
  done
done

echo "collected:"
ls -la "$OUTDIR"
