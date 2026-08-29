#!/usr/bin/env bash
#SBATCH --job-name=odc-eval-mn
#SBATCH --partition=mi355x
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err
#
# ============================================================================
#  OpenDC-Infer-LC — MULTI-NODE scaling (Efficiency(N), spec 2.7) on MI355X.
#  Data-parallel: one full-node vLLM replica per node, all behind a single
#  round-robin router -> the load generator sees ONE endpoint.
#
#  Submit with the node count, e.g.:   sbatch --nodes=4 deploy/mc2/run_eval_amd_multinode.sh
#  Compute Efficiency(N) afterwards:
#    python -m opendc_bench.analyze scaling results/mn_*_N1_* results/mn_*_N2_* ... --workload lc_32k
# ============================================================================
set -uo pipefail
set -x

SHARED="${SHARED:-/scratch/mc2/skypilot/shared/yue/opendc-infer-lc}"
MODEL_PATH="${MODEL_PATH:-/scratch/mc2/skypilot/shared/model/Qwen3-235B-A22B-Instruct-2507}"
SERVED="${SERVED:-odc-qwen3-235b}"
DATASET="${DATASET:-prompts_qwen3_235b}"
TP="${TP:-8}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-135168}"
PORT="${PORT:-8300}"; ROUTER_PORT="${ROUTER_PORT:-8400}"
WORKLOADS="${WORKLOADS:-lc_32k lc_cache}"
WARMUP_S="${WARMUP_S:-15}"; MEASURE_S="${MEASURE_S:-60}"; MAX_CONCURRENCY="${MAX_CONCURRENCY:-256}"
IMAGE="${IMAGE:-docker.io/rocm/vllm:rocm7.13.0_gfx950-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1}"

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
N=${#NODES[@]}
ACCEL=$(( N * 8 ))
OUT="$SHARED/results/mn_$(basename "$MODEL_PATH")_N${N}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
echo "== multi-node N=$N nodes=[${NODES[*]}] -> $OUT =="

ROCM_FLAGS=(--network host --ipc host --device=/dev/kfd --device=/dev/dri \
            --group-add keep-groups --security-opt seccomp=unconfined --cap-add SYS_PTRACE)

# --- one full-node vLLM replica + telemetry per node ------------------------
for node in "${NODES[@]}"; do
  srun --nodes=1 --ntasks=1 --nodelist="$node" --exclusive bash -lc "
    bash $SHARED/deploy/gpu_telemetry.sh '$OUT/telemetry.$node.csv' 2 &
    exec podman run --rm --name srv-${SLURM_JOB_ID}-$node ${ROCM_FLAGS[*]} \
      -v '$MODEL_PATH:$MODEL_PATH:ro' -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      '$IMAGE' python3 -m vllm.entrypoints.openai.api_server \
        --model '$MODEL_PATH' --served-model-name '$SERVED' --host 0.0.0.0 --port $PORT \
        --dtype bfloat16 --tensor-parallel-size $TP --gpu-memory-utilization 0.90 \
        --max-model-len $MAX_MODEL_LEN --trust-remote-code --enforce-eager --no-enable-log-requests" \
    > "$OUT/server.$node.log" 2>&1 &
done

# --- wait for every replica to be healthy (use FQDN; cluster-resolvable) ----
DOMAIN=$(hostname -d 2>/dev/null)
declare -a BACKENDS
for node in "${NODES[@]}"; do
  host="$node${DOMAIN:+.$DOMAIN}"
  BACKENDS+=("$host:$PORT")
  echo "waiting for replica on $host:$PORT ..."
  deadline=$(( $(date +%s) + 2400 ))
  until curl -fsS "http://$host:$PORT/v1/models" >/dev/null 2>&1; do
    [[ $(date +%s) -gt $deadline ]] && { echo "FATAL: $host health timeout"; scancel "$SLURM_JOB_ID"; exit 1; }
    sleep 10
  done
  echo "  $host healthy."
done
echo "all $N replicas healthy."
BLIST=$(IFS=,; echo "${BACKENDS[*]}")

# --- router on head node (single endpoint) ----------------------------------
podman run --rm --name router-${SLURM_JOB_ID} --network host \
  -v "$SHARED:$SHARED" -e PYTHONPATH="$SHARED/src" "$IMAGE" \
  python3 -m opendc_bench.router --host 0.0.0.0 --port "$ROUTER_PORT" --backends "$BLIST" \
  > "$OUT/router.log" 2>&1 &
sleep 4

cat > "$OUT/run_meta.json" <<JSON
{"run_id":"mn_N${N}_${SLURM_JOB_ID}","backend":"vllm","model":"$SERVED","nodes":$N,
 "accelerators":$ACCEL,"tp":$TP,"replicas":$N,"router_backends":"$BLIST",
 "max_model_len":$MAX_MODEL_LEN,"slurm_nodelist":"$SLURM_JOB_NODELIST"}
JSON

# --- benchmark against the router (harness in container) --------------------
podman run --rm --network host -v "$SHARED:$SHARED" -v "$MODEL_PATH:$MODEL_PATH:ro" \
  -e PYTHONPATH="$SHARED/src" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 "$IMAGE" \
  bash -lc "
    for w in $WORKLOADS; do
      echo \"=== N=$N capacity: \$w ===\"
      python3 -m opendc_bench.cli capacity \
        --base-url http://127.0.0.1:$ROUTER_PORT --model $SERVED \
        --tokenizer hf:$MODEL_PATH --data $SHARED/data/$DATASET \
        --workload \$w --warmup $WARMUP_S --measure $MEASURE_S \
        --max-concurrency $MAX_CONCURRENCY --accelerators $ACCEL \
        --timeout 600 --out $OUT/\${w}.json || echo \"WARN: \$w failed\"
    done
  "

podman rm -f router-${SLURM_JOB_ID} 2>/dev/null || true
for node in "${NODES[@]}"; do podman rm -f srv-${SLURM_JOB_ID}-$node 2>/dev/null || true; done
echo "== done N=$N -> $OUT =="
