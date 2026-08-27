#!/bin/sh
for i in {1..9}; do docker build -f "Dockerfile.glm53-sm121-v$i" -t "radixark/vllm-glm53-flash:glm53-sm121-v$i" . || break; done