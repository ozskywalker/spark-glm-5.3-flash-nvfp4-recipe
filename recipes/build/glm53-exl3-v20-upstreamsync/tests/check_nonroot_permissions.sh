#!/usr/bin/env bash
# Post-build, pre-deploy check: verify every file touched by an overlay/patch_*.py
# is actually readable by the container's real runtime user, not just by root.
#
# Why this exists (2026-09-16): patch_apc_no_store.py's atomic_write() wrote its
# replacement file via tempfile.mkstemp() (mode 0600) + os.replace() (a rename,
# which installs the temp file's mode verbatim, not a copy that would keep the
# target's own mode). The Docker build applies every overlay/patch_*.py as root,
# so the resulting 0600-root-owned file was invisible there -- every self-check
# in the Dockerfile's own RUN step runs as root too. The container actually
# serves as a non-root user (uid:gid 1000:1000, matching the host `luser`
# account, passed via `docker run --user` at launch time -- NOT baked into the
# image, confirmed via `docker inspect <image> --format '{{.Config.User}}'`
# returning empty). That non-root process lost read access to a core vLLM file
# and the whole engine crashed with a bare PermissionError at first launch.
# This script catches that class of regression before it ever reaches a real
# boot. See recipes/VALIDATION.md, "Backport deployment incident, 2026-09-16".
#
# Usage: ./check_nonroot_permissions.sh [image_tag]
#   (default image_tag: glm53-exl3-v20-upstreamsync:local)
#
# Run this after every docker build that touches any overlay/patch_*.py,
# before deploying. Exits non-zero and prints exactly which file(s) a
# non-root runtime user can't read.
set -euo pipefail

IMAGE="${1:-glm53-exl3-v20-upstreamsync:local}"
RUNTIME_UID="${GLM53_RUNTIME_UID:-1000}"
RUNTIME_GID="${GLM53_RUNTIME_GID:-1000}"

echo "checking non-root (uid:gid ${RUNTIME_UID}:${RUNTIME_GID}) read access on ${IMAGE} ..."

docker run --rm --user "${RUNTIME_UID}:${RUNTIME_GID}" --entrypoint python3 "$IMAGE" -c "
import os, sys

# Every file any overlay/patch_*.py in this build tree is known to write to,
# as of 2026-09-16. Add a new file here whenever a new patch writes one.
files = [
    '/usr/local/lib/python3.12/dist-packages/vllm/v1/request.py',
    '/usr/local/lib/python3.12/dist-packages/vllm/sampling_params.py',
    '/usr/local/lib/python3.12/dist-packages/vllm/v1/core/block_pool.py',
    '/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/utils/api_utils.py',
    '/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/completion/serving.py',
    '/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/completion/protocol.py',
    '/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py',
    '/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py',
    '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/router/gate_linear.py',
    '/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py',
    '/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py',
    '/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py',
]
ok = True
for f in files:
    if not os.path.exists(f):
        print(f'SKIP (not present in this image)   {f}')
        continue
    readable = os.access(f, os.R_OK)
    mode = oct(os.stat(f).st_mode & 0o777)
    status = 'OK' if readable else 'FAIL -- NOT READABLE'
    if not readable:
        ok = False
    print(f'{status:22s} mode={mode}  {f}')
print()
print('whoami: uid=%d gid=%d' % (os.getuid(), os.getgid()))
if not ok:
    print()
    print('FAIL: one or more files are not readable by the runtime user.')
    print('Likely cause: a patch wrote the file via a temp-file+os.replace()')
    print('pattern without preserving the original mode (see atomic_write() in')
    print('overlay/patch_apc_no_store.py for the fixed reference implementation).')
    sys.exit(1)
print()
print('PASS: all known patched files are readable by the non-root runtime user.')
"
