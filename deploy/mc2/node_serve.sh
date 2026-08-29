#!/usr/bin/env bash
# Runs ON each node via: srun --ntasks-per-node=1 --gpus-per-node=8 node_serve.sh
# (canonical SLURM GPU binding -> each task gets its node's 8 GPUs at full speed.)
# Config comes from the exported env (srun --export=ALL).
#
# IMPORTANT: do NOT `exec` podman — run it in the background and `wait`, with a
# trap that force-removes the container on TERM/INT/EXIT. Otherwise SLURM's kill
# can't reap the container -> "Kill task failed" -> the node gets DRAINED.
set -uo pipefail
H=$(hostname -s)
# Each node records its OWN routable IPv4 (robust vs flaky head-side DNS/IPv6).
mkdir -p "$OUT" 2>/dev/null || true
hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^(10|172|192)\.' | head -1 > "$OUT/ip.$H" || true
FLAGS=(--rm --replace --network host --ipc host --device=/dev/kfd --device=/dev/dri
       --group-add keep-groups --security-opt seccomp=unconfined --cap-add SYS_PTRACE)

cleanup() { podman rm -f "srv-$H" >/dev/null 2>&1 || true; kill "${TELE:-0}" 2>/dev/null || true; }
trap 'cleanup; exit 0' TERM INT
trap cleanup EXIT

bash "$SHARED/deploy/gpu_telemetry.sh" "$OUT/telemetry.$H.csv" 2 & TELE=$!

if [[ "$BACKEND" == "sglang" ]]; then
  podman run "${FLAGS[@]}" --name "srv-$H" \
    -v "$MODEL_PATH:$MODEL_PATH:ro" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    "$SGLANG_IMAGE" python3 -m sglang.launch_server \
      --model-path "$MODEL_PATH" --served-model-name "$SERVED" \
      --host 0.0.0.0 --port "$PORT" --tp "$TP" --context-length "$MAX_MODEL_LEN" \
      --dtype bfloat16 --enable-metrics --trust-remote-code &
else
  podman run "${FLAGS[@]}" --name "srv-$H" \
    -v "$MODEL_PATH:$MODEL_PATH:ro" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    "$VLLM_IMAGE" python3 -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" --served-model-name "$SERVED" \
      --host 0.0.0.0 --port "$PORT" --dtype bfloat16 \
      --tensor-parallel-size "$TP" --gpu-memory-utilization 0.90 \
      --max-model-len "$MAX_MODEL_LEN" --trust-remote-code --enforce-eager --no-enable-log-requests &
fi
SVPID=$!
wait "$SVPID"
