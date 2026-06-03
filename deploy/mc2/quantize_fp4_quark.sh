#!/usr/bin/env bash
#SBATCH --job-name=odc-quark-fp4
#SBATCH --partition=mi355x
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err
#
# ============================================================================
#  FP4 prep — quantize a BF16 checkpoint to MXFP4 with AMD Quark, for the FP4
#  (CDNA4) efficiency study. Produces a vLLM-loadable MXFP4 checkpoint dir.
#
#  After this finishes, run the benchmark by pointing the runner at the output:
#    sbatch --export=ALL,MODEL_PATH=<OUT_DIR>,SERVED=odc-qwen3-235b-fp4,TAG=fp4 \
#           deploy/mc2/run_eval_amd.sh
#  (vLLM auto-detects the Quark MXFP4 quantization_config; no --quantization needed.)
#
#  CAVEATS (this is a template — verify on first run):
#   * MXFP4 of a 235B MoE is heavy; weight-only (mxfp4) avoids activation
#     calibration. Activation quant (w_mxfp4_a_mxfp4) needs a calibration set.
#   * Quark + (any) calibration data may need network; HF_HUB_OFFLINE is left
#     UNSET here so Quark can fetch its default calib dataset if required.
#   * The exclude_layers recipe below is the documented MoE recipe; tune for
#     Qwen3-235B-A22B if a layer name differs.
# ============================================================================
set -euo pipefail
set -x

SHARED="${SHARED:-/scratch/mc2/skypilot/shared/yue/opendc-infer-lc}"
MODEL_PATH="${MODEL_PATH:-/scratch/mc2/skypilot/shared/model/Qwen3-235B-A22B-Instruct-2507}"
OUT_DIR="${OUT_DIR:-/scratch/mc2/skypilot/shared/model/Qwen3-235B-A22B-MXFP4}"
QUANT_SCHEME="${QUANT_SCHEME:-mxfp4}"          # mxfp4 (weight-only) | w_mxfp4_a_mxfp4
IMAGE="${IMAGE:-docker.io/rocm/vllm:rocm7.13.0_gfx950-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1}"
# AMD Quark example script (clone once on a networked host: git clone --depth 1
# https://github.com/amd/Quark). The `quark` package itself is in the image.
QUARK_REPO="${QUARK_REPO:-/scratch/mc2/skypilot/shared/yue/Quark}"
LLM_PTQ="$QUARK_REPO/examples/torch/language_modeling/llm_ptq"

[[ -f "${MODEL_PATH}/config.json" ]] || { echo "FATAL: model not visible on $(hostname)"; exit 1; }
[[ -f "$LLM_PTQ/quantize_quark.py" ]] || { echo "FATAL: Quark examples missing at $QUARK_REPO (git clone https://github.com/amd/Quark)"; exit 1; }
mkdir -p "$(dirname "$OUT_DIR")"
podman system migrate 2>/dev/null || true
podman image exists "$IMAGE" || podman pull "$IMAGE"

ROCM_FLAGS=(--network host --ipc host --device=/dev/kfd --device=/dev/dri
            --group-add keep-groups --security-opt seccomp=unconfined --cap-add SYS_PTRACE)

# File-to-file MXFP4: bypasses model loading + calibration, quantizes safetensors
# shard-by-shard (offline-safe, low memory). No calib dataset / network needed.
podman run --rm "${ROCM_FLAGS[@]}" \
  -v "$(dirname "$MODEL_PATH"):$(dirname "$MODEL_PATH")" \
  -v "$QUARK_REPO:$QUARK_REPO" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  "$IMAGE" \
  bash -lc "cd '$LLM_PTQ' && python3 quantize_quark.py \
      --model_dir '$MODEL_PATH' --output_dir '$OUT_DIR' \
      --quant_scheme '$QUANT_SCHEME' \
      --exclude_layers '*self_attn*' '*mlp.gate' '*lm_head' \
      --file2file_quantization --skip_evaluation"

echo "[done] MXFP4 checkpoint -> $OUT_DIR"
ls -la "$OUT_DIR" | head
echo "Next: sbatch --export=ALL,MODEL_PATH=$OUT_DIR,SERVED=odc-qwen3-235b-fp4,TAG=fp4 $SHARED/deploy/mc2/run_eval_amd.sh"
