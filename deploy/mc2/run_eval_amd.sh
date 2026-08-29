#!/usr/bin/env bash
#SBATCH --job-name=odc-eval-amd
#SBATCH --partition=mi355x
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err
#
# ============================================================================
#  OpenDC-Infer-LC — ONE-FILE end-to-end eval on AMD MI355X.
#  Backend-aware: BACKEND = vllm | sglang  (trtllm = CUDA/B200 only, blocked here).
#
#  Two steps, one file:
#    STEP 1 — build the per-tokenizer dataset (idempotent; skipped if present)
#    STEP 2 — run the whole eval: serve -> health gate -> benchmark -> results
#
#  USAGE (one command):
#    sbatch deploy/mc2/run_eval_amd.sh                         # vLLM (default)
#    sbatch --export=ALL,BACKEND=sglang deploy/mc2/run_eval_amd.sh
# ============================================================================
set -euo pipefail
set -x

# ---- configuration (override via sbatch --export=ALL,VAR=...) --------------
SHARED="${SHARED:-/scratch/mc2/skypilot/shared/yue/opendc-infer-lc}"
MODEL_PATH="${MODEL_PATH:-/scratch/mc2/skypilot/shared/model/Qwen3-235B-A22B-Instruct-2507}"
BACKEND="${BACKEND:-vllm}"                     # vllm | sglang | tgi
SERVED="${SERVED:-odc-qwen3-235b}"
PORT="${PORT:-8300}"
TP="${TP:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-135168}"
DATASET="${DATASET:-prompts_qwen3_235b}"
DATASET_CONFIG="${DATASET_CONFIG:-$SHARED/configs/datasets.yaml}"
WORKLOADS="${WORKLOADS:-lc_8k lc_32k lc_128k lc_cache}"
WARMUP_S="${WARMUP_S:-15}"; MEASURE_S="${MEASURE_S:-60}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"; ACCELERATORS="${ACCELERATORS:-8}"

# ---- study toggles ---------------------------------------------------------
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"   # 1=eager (conservative); 0=enable graphs (faster)
QUANT="${QUANT:-}"                    # ""=BF16 | fp8 | fp4   (separate leaderboard category)
CACHE_MODE="${CACHE_MODE:-warmed}"    # warmed (default) | cold (prefix cache OFF) | prefilled
LOAD_MODE="${LOAD_MODE:-closed}"      # closed (concurrency ladder) | open (Poisson rps sweep)
OPEN_MAX_RPS="${OPEN_MAX_RPS:-64}"
# build a result tag from the toggles (explicit ifs: safe under set -e)
if [[ -z "${TAG:-}" ]]; then
  TAG="${QUANT:-bf16}_${CACHE_MODE}"
  if [[ "$ENFORCE_EAGER" == 0 ]]; then TAG="${TAG}_graph"; fi
  if [[ "$LOAD_MODE" == open ]]; then TAG="${TAG}_open"; fi
fi

# Container images (ROCm / gfx950). Harness runs in the proven vLLM image.
VLLM_IMAGE="${VLLM_IMAGE:-docker.io/rocm/vllm:rocm7.13.0_gfx950-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1}"
SGLANG_IMAGE="${SGLANG_IMAGE:-docker.io/lmsysorg/sglang-rocm:v0.5.12.post1-rocm700-mi35x-20260526}"
TGI_IMAGE="${TGI_IMAGE:-ghcr.io/huggingface/text-generation-inference:latest-rocm}"  # NB: confirm a valid ROCm tag; gfx950 support unverified
BENCH_IMAGE="${BENCH_IMAGE:-$VLLM_IMAGE}"      # the harness needs only transformers+stdlib

OUT="${OUT:-$SHARED/results/${BACKEND}_${TAG}_$(basename "$MODEL_PATH")_${SLURM_JOB_ID:-local}}"
mkdir -p "$OUT"

HEALTH_PATH="/v1/models"
case "$BACKEND" in
  vllm)   SERVE_IMAGE="$VLLM_IMAGE" ;;
  sglang) SERVE_IMAGE="$SGLANG_IMAGE" ;;
  tgi)    SERVE_IMAGE="$TGI_IMAGE"; HEALTH_PATH="/health" ;;
  trtllm|trt) echo "FATAL: TensorRT-LLM is CUDA-only (no ROCm). Run it on the B200 cluster."; exit 2 ;;
  *) echo "FATAL: unknown BACKEND=$BACKEND"; exit 2 ;;
esac

[[ -f "${MODEL_PATH}/config.json" ]] || { echo "FATAL: model not visible on $(hostname)"; exit 1; }
podman system migrate 2>/dev/null || true
for img in "$SERVE_IMAGE" "$BENCH_IMAGE"; do podman image exists "$img" || podman pull "$img"; done

# device/runtime flags shared by all ROCm containers
ROCM_FLAGS=(--network host --ipc host --device=/dev/kfd --device=/dev/dri
            --group-add keep-groups --security-opt seccomp=unconfined --cap-add SYS_PTRACE)

# per-backend extra serve flags derived from the study toggles
VLLM_EXTRA=(); SGLANG_EXTRA=(); TGI_EXTRA=()
if [[ "$ENFORCE_EAGER" == 1 ]]; then VLLM_EXTRA+=(--enforce-eager); SGLANG_EXTRA+=(--disable-cuda-graph); fi
if [[ -n "$QUANT" ]]; then VLLM_EXTRA+=(--quantization "$QUANT"); SGLANG_EXTRA+=(--quantization "$QUANT"); TGI_EXTRA+=(--quantize "$QUANT"); fi
if [[ "$CACHE_MODE" == cold ]]; then VLLM_EXTRA+=(--no-enable-prefix-caching); SGLANG_EXTRA+=(--disable-radix-cache); fi

# ############################################################################
# # STEP 1 — BUILD DATASET  (idempotent: skipped if already built)
# ############################################################################
if [[ -f "$SHARED/data/$DATASET/MANIFEST.json" ]]; then
  echo "[step 1] dataset $DATASET present -> skip"
else
  echo "[step 1] building dataset $DATASET ..."
  podman run --rm "${ROCM_FLAGS[@]}" \
    -v "$SHARED:$SHARED" -v "${MODEL_PATH}:${MODEL_PATH}:ro" \
    -e PYTHONPATH="$SHARED/src" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    "$BENCH_IMAGE" \
    python3 -m opendc_data.build --config "$DATASET_CONFIG" \
      --tokenizer "hf:$MODEL_PATH" --out "$SHARED/data/$DATASET"
fi

# ############################################################################
# # STEP 2 — RUN EVAL  (serve -> health gate -> benchmark -> results)
# ############################################################################
# ---- 2a. serve (backend-specific, background) ------------------------------
case "$BACKEND" in
  vllm)
    podman run --rm --name "srv-${SLURM_JOB_ID:-local}" "${ROCM_FLAGS[@]}" \
      -v "${MODEL_PATH}:${MODEL_PATH}:ro" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      "$SERVE_IMAGE" \
      python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" --served-model-name "$SERVED" \
        --host 0.0.0.0 --port "$PORT" --dtype bfloat16 \
        --tensor-parallel-size "$TP" --gpu-memory-utilization 0.90 \
        --max-model-len "$MAX_MODEL_LEN" --trust-remote-code \
        --no-enable-log-requests "${VLLM_EXTRA[@]}" > "$OUT/server.log" 2>&1 &
    ;;
  sglang)
    podman run --rm --name "srv-${SLURM_JOB_ID:-local}" "${ROCM_FLAGS[@]}" \
      -v "${MODEL_PATH}:${MODEL_PATH}:ro" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      "$SERVE_IMAGE" \
      python3 -m sglang.launch_server \
        --model-path "$MODEL_PATH" --served-model-name "$SERVED" \
        --host 0.0.0.0 --port "$PORT" --tp "$TP" \
        --context-length "$MAX_MODEL_LEN" --dtype bfloat16 \
        --enable-metrics --trust-remote-code "${SGLANG_EXTRA[@]}" > "$OUT/server.log" 2>&1 &
    ;;
  tgi)
    # TGI is officially tested on MI210/250/300; gfx950 (MI355X) support is unverified.
    podman run --rm --name "srv-${SLURM_JOB_ID:-local}" "${ROCM_FLAGS[@]}" \
      -v "${MODEL_PATH}:${MODEL_PATH}:ro" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      "$SERVE_IMAGE" \
      text-generation-launcher \
        --model-id "$MODEL_PATH" --hostname 0.0.0.0 --port "$PORT" \
        --num-shard "$TP" --dtype bfloat16 --trust-remote-code \
        --max-input-tokens "$(( MAX_MODEL_LEN - 2048 ))" \
        --max-total-tokens "$MAX_MODEL_LEN" \
        --max-batch-prefill-tokens "$MAX_MODEL_LEN" "${TGI_EXTRA[@]}" > "$OUT/server.log" 2>&1 &
    ;;
esac
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null||true; podman rm -f srv-${SLURM_JOB_ID:-local} 2>/dev/null||true; [[ -n "${TELE:-}" ]]&&kill $TELE 2>/dev/null||true' EXIT

# ---- 2b. telemetry + health gate -------------------------------------------
bash "$SHARED/deploy/gpu_telemetry.sh" "$OUT/telemetry.csv" 2 & TELE=$!
echo "[step 2] waiting for $BACKEND on :$PORT ..."
for i in $(seq 1 480); do
  curl -fsS "http://127.0.0.1:$PORT$HEALTH_PATH" >/dev/null 2>&1 && { echo "healthy after ${i}0s"; break; }
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "FATAL: server died"; tail -60 "$OUT/server.log"; exit 1; }
  sleep 10
done

# prime the server: the first inference can trigger kernel/graph compilation
# (esp. SGLang) that would otherwise consume the first workload's timed window.
echo "[warmup] priming first-request compile ..."
for _ in 1 2; do
  curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"warm up\"}],\"max_tokens\":8}" \
    >/dev/null 2>&1 || true
done

DSHASH=$(python3 -c "import json;print(json.load(open('$SHARED/data/$DATASET/MANIFEST.json'))['dataset_version_hash'])" 2>/dev/null||echo "")
cat > "$OUT/run_meta.json" <<JSON
{"model":"$SERVED","model_path":"$MODEL_PATH","backend":"$BACKEND","tp":$TP,
 "quant":"${QUANT:-bf16}","cache_mode":"$CACHE_MODE","enforce_eager":$ENFORCE_EAGER,
 "load_mode":"$LOAD_MODE","tag":"$TAG","max_model_len":$MAX_MODEL_LEN,
 "hostname":"$(hostname)","slurm_job_id":"${SLURM_JOB_ID:-local}",
 "dataset":"$DATASET","dataset_version_hash":"$DSHASH","warmup_s":$WARMUP_S,"measure_s":$MEASURE_S}
JSON

# ---- 2c. benchmark (harness in BENCH_IMAGE; localhost via --network host) ---
podman run --rm "${ROCM_FLAGS[@]}" \
  -v "$SHARED:$SHARED" -v "${MODEL_PATH}:${MODEL_PATH}:ro" \
  -e PYTHONPATH="$SHARED/src" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e OPENDC_NO_THINK="${OPENDC_NO_THINK:-}" \
  "$BENCH_IMAGE" \
  bash -lc "
    for w in $WORKLOADS; do
      echo \"=== capacity: \$w ===\"
      python3 -m opendc_bench.cli capacity \
        --base-url http://127.0.0.1:$PORT --model $SERVED \
        --tokenizer hf:$MODEL_PATH --data $SHARED/data/$DATASET \
        --workload \$w --warmup $WARMUP_S --measure $MEASURE_S \
        --mode $LOAD_MODE --max-concurrency $MAX_CONCURRENCY --open-max-rps $OPEN_MAX_RPS \
        --accelerators $ACCELERATORS \
        --server-metrics-url http://127.0.0.1:$PORT/metrics \
        --timeout 600 --out $OUT/\${w}.json || echo \"WARN: \$w failed\"
    done
  "

echo "[done] results -> $OUT"; ls -la "$OUT"
