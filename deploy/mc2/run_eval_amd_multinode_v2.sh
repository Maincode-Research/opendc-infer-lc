#!/usr/bin/env bash
#SBATCH --job-name=odc-mn2
#SBATCH --partition=mi355x
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err
#
# ============================================================================
#  Multi-node scaling v2 (Efficiency(N), spec 2.7) — PRODUCTION routing.
#   * per-node replica launched via canonical `srun --ntasks-per-node=1
#     --gpus-per-node=8` (fixes the GPU/RCCL crippling of the nested-srun v1);
#   * single endpoint via sgl-router (cache-aware prefix routing), which fronts
#     BOTH vLLM and SGLang OpenAI workers.
#  Submit: sbatch --nodes=N deploy/mc2/run_eval_amd_multinode_v2.sh
# ============================================================================
set -uo pipefail
set -x

export SHARED="${SHARED:-/scratch/mc2/skypilot/shared/yue/opendc-infer-lc}"
export MODEL_PATH="${MODEL_PATH:-/scratch/mc2/skypilot/shared/model/Qwen3-235B-A22B-Instruct-2507}"
export BACKEND="${BACKEND:-vllm}"            # vllm | sglang  (router is sgl-router either way)
export SERVED="${SERVED:-odc-qwen3-235b}"
export DATASET="${DATASET:-prompts_qwen3_235b}"
export TP="${TP:-8}"; export MAX_MODEL_LEN="${MAX_MODEL_LEN:-135168}"
export PORT="${PORT:-8300}"; ROUTER_PORT="${ROUTER_PORT:-8400}"
export VLLM_IMAGE="${VLLM_IMAGE:-docker.io/rocm/vllm:rocm7.13.0_gfx950-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1}"
export SGLANG_IMAGE="${SGLANG_IMAGE:-docker.io/lmsysorg/sglang-rocm:v0.5.12.post1-rocm700-mi35x-20260526}"
ROUTER_IMAGE="${ROUTER_IMAGE:-$SGLANG_IMAGE}"   # sgl-router ships in the SGLang image
ROUTER_POLICY="${ROUTER_POLICY:-cache_aware}"   # cache_aware | round_robin | power_of_two | random
WORKLOADS="${WORKLOADS:-lc_32k lc_cache}"
WARMUP_S="${WARMUP_S:-15}"; MEASURE_S="${MEASURE_S:-60}"; MAX_CONCURRENCY="${MAX_CONCURRENCY:-256}"

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
N=${#NODES[@]}; ACCEL=$(( N * 8 )); DOMAIN=$(hostname -d 2>/dev/null)
export OUT="$SHARED/results/mn2_${BACKEND}_${ROUTER_POLICY}_$(basename "$MODEL_PATH")_N${N}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
echo "== v2 N=$N backend=$BACKEND router=$ROUTER_POLICY -> $OUT =="

# --- launch one replica per node (canonical GPU binding; backgrounded) ------
srun --ntasks-per-node=1 --gpus-per-node=8 --export=ALL \
  bash "$SHARED/deploy/mc2/node_serve.sh" > "$OUT/serve.srun.log" 2>&1 &
SRUN_PID=$!
# On exit: stop the router, then SIGTERM the srun so each node_serve.sh trap
# force-removes its container (prevents "Kill task failed" -> node DRAIN).
trap 'podman rm -f router-${SLURM_JOB_ID} 2>/dev/null||true; kill -TERM $SRUN_PID 2>/dev/null||true; sleep 5' EXIT

# --- wait for all replicas healthy (FQDN) -----------------------------------
declare -a WURLS
for node in "${NODES[@]}"; do
  # Each node wrote its own IPv4 to $OUT/ip.<short> (robust; avoids head DNS/IPv6).
  ipf="$OUT/ip.$node"; t=$(( $(date +%s) + 400 ))
  while [[ ! -s "$ipf" ]]; do [[ $(date +%s) -gt $t ]] && break; sleep 3; done
  ip=$(cat "$ipf" 2>/dev/null); ip=${ip:-$node}
  WURLS+=("http://$ip:$PORT")
  echo "waiting for replica $node ($ip):$PORT ..."
  deadline=$(( $(date +%s) + 2400 ))
  until curl -4 -fsS "http://$ip:$PORT/v1/models" >/dev/null 2>&1; do
    [[ $(date +%s) -gt $deadline ]] && { echo "FATAL: $node ($ip) health timeout"; exit 1; }
    sleep 10
  done
  echo "  $ip healthy."
done
echo "all $N replicas healthy."

# --- production router: sgl-router (cache-aware) -----------------------------
# cache_aware needs a tokenizer to compute prefixes -> give it the model path.
ROUTER_EXTRA=(); [[ "$ROUTER_POLICY" == cache_aware ]] && ROUTER_EXTRA=(--model-path "$MODEL_PATH")
podman image exists "$ROUTER_IMAGE" || podman pull "$ROUTER_IMAGE"   # pre-pull so launch is fast
podman run --rm --replace --name "router-${SLURM_JOB_ID}" --network host \
  -v "$MODEL_PATH:$MODEL_PATH:ro" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 "$ROUTER_IMAGE" \
  python3 -m sglang_router.launch_router \
    --host 0.0.0.0 --port "$ROUTER_PORT" --policy "$ROUTER_POLICY" \
    --worker-urls "${WURLS[@]}" "${ROUTER_EXTRA[@]}" > "$OUT/router.log" 2>&1 &
echo "waiting for router on :$ROUTER_PORT ..."
rdl=$(( $(date +%s) + 600 ))
until curl -fsS "http://127.0.0.1:$ROUTER_PORT/v1/models" >/dev/null 2>&1; do
  [[ $(date +%s) -gt $rdl ]] && { echo "FATAL: router not responding"; tail -20 "$OUT/router.log"; exit 1; }
  sleep 10
done
echo "router up"

cat > "$OUT/run_meta.json" <<JSON
{"run_id":"mn2_N${N}_${SLURM_JOB_ID}","backend":"$BACKEND","router":"sgl-router/$ROUTER_POLICY",
 "model":"$SERVED","nodes":$N,"accelerators":$ACCEL,"tp":$TP,"replicas":$N,
 "worker_urls":"${WURLS[*]}","max_model_len":$MAX_MODEL_LEN,"slurm_nodelist":"$SLURM_JOB_NODELIST"}
JSON

# --- benchmark against the router -------------------------------------------
podman run --rm --network host -v "$SHARED:$SHARED" -v "$MODEL_PATH:$MODEL_PATH:ro" \
  -e PYTHONPATH="$SHARED/src" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 "$VLLM_IMAGE" \
  bash -lc "
    for w in $WORKLOADS; do
      echo \"=== N=$N capacity: \$w ===\"
      python3 -m opendc_bench.cli capacity \
        --base-url http://127.0.0.1:$ROUTER_PORT --model $SERVED \
        --tokenizer hf:$MODEL_PATH --data $SHARED/data/$DATASET \
        --workload \$w --warmup $WARMUP_S --measure $MEASURE_S \
        --max-concurrency $MAX_CONCURRENCY --accelerators $ACCEL \
        --server-metrics-url http://127.0.0.1:$ROUTER_PORT/metrics \
        --timeout 600 --out $OUT/\${w}.json || echo \"WARN: \$w failed\"
    done
  "
echo "== v2 done N=$N -> $OUT =="
