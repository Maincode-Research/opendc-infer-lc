#!/usr/bin/env bash
#SBATCH --job-name=odc-eval-b200
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err
# EDIT: add your cluster's --partition / --account.
#
# ============================================================================
#  OpenDC-Infer-LC — ONE-FILE end-to-end eval on NVIDIA B200 (CUDA).
#  Same harness + datasets as the AMD runner; only the serve layer differs.
#  Backend-aware: BACKEND = vllm | sglang | trtllm.
#
#    STEP 1 — build the per-tokenizer dataset (idempotent; skipped if present)
#    STEP 2 — serve -> health gate -> benchmark -> results
#
#  USAGE:
#    sbatch deploy/b200/run_eval_b200.sh                          # vLLM
#    sbatch --export=ALL,BACKEND=trtllm deploy/b200/run_eval_b200.sh
#    sbatch --export=ALL,BACKEND=sglang deploy/b200/run_eval_b200.sh
#
#  Container runtime: defaults to `docker run --gpus all`. If your cluster uses
#  enroot/pyxis or podman, change RUNTIME / the GPU flag near the top.
# ============================================================================
set -euo pipefail
set -x

# ---- configuration (override via sbatch --export=ALL,VAR=...) --------------
REPO="${REPO:-$HOME/opendc-infer-lc}"          # repo on a mount visible to compute nodes
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-235B-A22B-Instruct-2507}"
BACKEND="${BACKEND:-vllm}"                     # vllm | sglang | trtllm
SERVED="${SERVED:-odc-qwen3-235b}"
PORT="${PORT:-8300}"
TP="${TP:-8}"                                  # one B200 node = 8 GPUs
MAX_MODEL_LEN="${MAX_MODEL_LEN:-135168}"       # 132K. For 256K-1M see the DCA note in README.
DATASET="${DATASET:-prompts_qwen3_235b}"
DATASET_CONFIG="${DATASET_CONFIG:-$REPO/configs/datasets.yaml}"
WORKLOADS="${WORKLOADS:-lc_8k lc_32k lc_128k lc_cache}"
WARMUP_S="${WARMUP_S:-15}"; MEASURE_S="${MEASURE_S:-60}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"; ACCELERATORS="${ACCELERATORS:-8}"
LOAD_MODE="${LOAD_MODE:-closed}"; OPEN_MAX_RPS="${OPEN_MAX_RPS:-64}"
QUANT="${QUANT:-}"                             # ""=BF16 | fp8 | nvfp4 (Blackwell FP4)

# CUDA container images (pin versions for reproducible submissions).
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
SGLANG_IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:latest}"
TRTLLM_IMAGE="${TRTLLM_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.2.0}"
BENCH_IMAGE="${BENCH_IMAGE:-$VLLM_IMAGE}"      # harness needs only transformers+stdlib

# Container runtime + GPU flag (adjust for your site).
RUN="${RUN:-docker run --rm --gpus all --ipc=host --network host --shm-size=32g}"

TAG="${TAG:-${QUANT:-bf16}_${LOAD_MODE}}"
OUT="${OUT:-$REPO/results/${BACKEND}_${TAG}_$(basename "$MODEL_PATH")_${SLURM_JOB_ID:-local}}"
mkdir -p "$OUT"
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "FATAL: model not visible on $(hostname)"; exit 1; }

HEALTH_PATH="/v1/models"
case "$BACKEND" in
  vllm)   SERVE_IMAGE="$VLLM_IMAGE" ;;
  sglang) SERVE_IMAGE="$SGLANG_IMAGE" ;;
  trtllm) SERVE_IMAGE="$TRTLLM_IMAGE"; HEALTH_PATH="/health" ;;
  *) echo "FATAL: unknown BACKEND=$BACKEND"; exit 2 ;;
esac

QUANT_VLLM=(); QUANT_SGLANG=()
[[ -n "$QUANT" ]] && QUANT_VLLM=(--quantization "$QUANT") && QUANT_SGLANG=(--quantization "$QUANT")

# ############################################################################
# # STEP 1 — BUILD DATASET  (idempotent)
# ############################################################################
if [[ -f "$REPO/data/$DATASET/MANIFEST.json" ]]; then
  echo "[step 1] dataset $DATASET present -> skip"
else
  echo "[step 1] building dataset $DATASET ..."
  $RUN --entrypoint bash \
    -v "$REPO:$REPO" -v "${MODEL_PATH}:${MODEL_PATH}:ro" \
    -e PYTHONPATH="$REPO/src" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    "$BENCH_IMAGE" -lc \
    "python3 -m opendc_data.build --config '$DATASET_CONFIG' --tokenizer 'hf:$MODEL_PATH' --out '$REPO/data/$DATASET'"
fi

# ############################################################################
# # STEP 2 — RUN EVAL
# ############################################################################
# ---- 2a. serve (background) ------------------------------------------------
case "$BACKEND" in
  vllm)
    $RUN --name "srv-${SLURM_JOB_ID:-local}" \
      -v "${MODEL_PATH}:${MODEL_PATH}:ro" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      --entrypoint bash "$SERVE_IMAGE" -lc \
      "python3 -m vllm.entrypoints.openai.api_server --model '$MODEL_PATH' \
         --served-model-name '$SERVED' --host 0.0.0.0 --port $PORT --dtype bfloat16 \
         --tensor-parallel-size $TP --gpu-memory-utilization 0.90 \
         --max-model-len $MAX_MODEL_LEN --trust-remote-code ${QUANT_VLLM[*]:-}" \
      > "$OUT/server.log" 2>&1 &
    ;;
  sglang)
    $RUN --name "srv-${SLURM_JOB_ID:-local}" \
      -v "${MODEL_PATH}:${MODEL_PATH}:ro" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      --entrypoint bash "$SERVE_IMAGE" -lc \
      "python3 -m sglang.launch_server --model-path '$MODEL_PATH' \
         --served-model-name '$SERVED' --host 0.0.0.0 --port $PORT --tp $TP \
         --context-length $MAX_MODEL_LEN --dtype bfloat16 --enable-metrics --trust-remote-code ${QUANT_SGLANG[*]:-}" \
      > "$OUT/server.log" 2>&1 &
    ;;
  trtllm)
    # trtllm-serve builds an engine from the HF checkpoint on first launch.
    $RUN --name "srv-${SLURM_JOB_ID:-local}" \
      -v "${MODEL_PATH}:${MODEL_PATH}:ro" \
      --entrypoint bash "$SERVE_IMAGE" -lc \
      "trtllm-serve '$MODEL_PATH' --host 0.0.0.0 --port $PORT \
         --tp_size $TP --max_seq_len $MAX_MODEL_LEN \
         --kv_cache_free_gpu_memory_fraction 0.9" \
      > "$OUT/server.log" 2>&1 &
    ;;
esac
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null||true; docker rm -f srv-${SLURM_JOB_ID:-local} 2>/dev/null||true; [[ -n "${TELE:-}" ]]&&kill $TELE 2>/dev/null||true' EXIT

# ---- 2b. telemetry (nvidia-smi auto-detected) + health gate ----------------
bash "$REPO/deploy/gpu_telemetry.sh" "$OUT/telemetry.csv" 2 & TELE=$!
echo "[step 2] waiting for $BACKEND on :$PORT ..."
for i in $(seq 1 480); do
  curl -fsS "http://127.0.0.1:$PORT$HEALTH_PATH" >/dev/null 2>&1 && { echo "healthy after ${i}0s"; break; }
  kill -0 "$SRV_PID" 2>/dev/null || { echo "FATAL: server died"; tail -60 "$OUT/server.log"; exit 1; }
  sleep 10
done

DSHASH=$(python3 -c "import json;print(json.load(open('$REPO/data/$DATASET/MANIFEST.json'))['dataset_version_hash'])" 2>/dev/null||echo "")
cat > "$OUT/run_meta.json" <<JSON
{"model":"$SERVED","backend":"$BACKEND","hardware":"B200","tp":$TP,"quant":"${QUANT:-bf16}",
 "load_mode":"$LOAD_MODE","max_model_len":$MAX_MODEL_LEN,"hostname":"$(hostname)",
 "slurm_job_id":"${SLURM_JOB_ID:-local}","dataset":"$DATASET","dataset_version_hash":"$DSHASH"}
JSON

# ---- 2c. benchmark (harness in BENCH_IMAGE) --------------------------------
$RUN --entrypoint bash \
  -v "$REPO:$REPO" -v "${MODEL_PATH}:${MODEL_PATH}:ro" \
  -e PYTHONPATH="$REPO/src" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  "$BENCH_IMAGE" -lc "
    for w in $WORKLOADS; do
      echo \"=== capacity: \$w ===\"
      python3 -m opendc_bench.cli capacity \
        --base-url http://127.0.0.1:$PORT --model $SERVED \
        --tokenizer hf:$MODEL_PATH --data $REPO/data/$DATASET \
        --workload \$w --warmup $WARMUP_S --measure $MEASURE_S \
        --mode $LOAD_MODE --max-concurrency $MAX_CONCURRENCY --open-max-rps $OPEN_MAX_RPS \
        --accelerators $ACCELERATORS \
        --server-metrics-url http://127.0.0.1:$PORT/metrics \
        --timeout 600 --out $OUT/\${w}.json || echo \"WARN: \$w failed\"
    done
  "

echo "[done] results -> $OUT"; ls -la "$OUT"
