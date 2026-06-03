#!/usr/bin/env bash
# ============================================================================
#  OpenDC-Infer-LC — submit the WHOLE AMD matrix with one command.
#
#  Sweeps backends (and optionally models) by sbatch-ing run_eval_amd.sh once
#  per cell. Each cell is its own exclusive 1-node MI355X job, so they queue
#  and run in parallel across idle nodes.
#
#  USAGE:
#    bash deploy/mc2/run_all.sh                 # default matrix below
#    BACKENDS="vllm sglang" bash deploy/mc2/run_all.sh
#
#  NOTE: TensorRT-LLM is CUDA-only -> run it on the B200 cluster, not here.
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$HERE/run_eval_amd.sh"

# ---- the matrix (override via env) -----------------------------------------
BACKENDS="${BACKENDS:-vllm sglang tgi}"
WORKLOADS="${WORKLOADS:-lc_8k lc_32k lc_128k lc_cache}"
WARMUP_S="${WARMUP_S:-15}"; MEASURE_S="${MEASURE_S:-60}"; MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"

# (model, served-name, dataset, tp) triples — add rows to grow the sweep.
# Default: the 235B headline model. Uncomment the 32B medium tier to add it.
MODELS=(
  "/scratch/mc2/skypilot/shared/model/Qwen3-235B-A22B-Instruct-2507|odc-qwen3-235b|prompts_qwen3_235b|8"
  # "/scratch/mc2/skypilot/shared/model/Qwen2.5-32B-Instruct|odc-qwen25-32b|prompts_qwen32b|8"
)

echo "Submitting matrix: backends=[$BACKENDS] x ${#MODELS[@]} model(s), workloads=[$WORKLOADS]"
port=8300
for row in "${MODELS[@]}"; do
  IFS='|' read -r mpath served dataset tp <<<"$row"
  for be in $BACKENDS; do
    port=$((port+1))                      # unique port per cell (avoids clashes if co-scheduled)
    jid=$(sbatch --parsable \
      --export=ALL,BACKEND="$be",MODEL_PATH="$mpath",SERVED="${served}-${be}",DATASET="$dataset",\
TP="$tp",PORT="$port",WORKLOADS="$WORKLOADS",WARMUP_S="$WARMUP_S",MEASURE_S="$MEASURE_S",MAX_CONCURRENCY="$MAX_CONCURRENCY" \
      "$RUNNER")
    echo "  submitted job $jid : backend=$be model=$(basename "$mpath") port=$port"
  done
done
echo "Done. Watch with: squeue -u \$USER ; results land in results/<backend>_<model>_<jobid>/"
echo "Aggregate later:  python -m opendc_bench.analyze leaderboard results --workload lc_32k"
