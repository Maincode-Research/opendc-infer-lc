# OpenDC-Infer-LC on NVIDIA B200 (CUDA)

This is the B200/CUDA leg of OpenDC-Infer-LC. **The benchmark harness and
datasets are identical to the AMD side** — only the serve/launch layer differs.
You run the same metric definitions and prompt sets, so results are directly
comparable to the MI355X numbers.

## Can I use the AMD runner directly?

**No** — `deploy/mc2/run_eval_amd.sh` is ROCm-specific (podman, `/dev/kfd`,
`rocm/*` images, `mi355x` partition). Use **`deploy/b200/run_eval_b200.sh`**
instead. Everything under `src/` and `configs/` is shared and unchanged.

What's different on B200 vs MI355X:

| | MI355X (AMD) | B200 (NVIDIA) |
|---|---|---|
| Runtime / GPU flag | `podman … --device=/dev/kfd` | `docker run --gpus all` |
| vLLM image | `rocm/vllm:…gfx950…` | `vllm/vllm-openai:latest` |
| SGLang image | `lmsysorg/sglang:…-mi35x` | `lmsysorg/sglang:latest` |
| **TensorRT-LLM** | ✗ (no ROCm) | ✓ `nvcr.io/nvidia/tensorrt-llm/release:1.2.0` |
| **>256K context (DCA / 1M)** | ✗ (unsupported on ROCm) | ✓ supported |

## Setup (once)

1. Put the repo on a path visible to your compute nodes (set `REPO`).
2. Put the model there too (set `MODEL_PATH`), e.g.
   `Qwen/Qwen3-235B-A22B-Instruct-2507`.
3. Make sure compute nodes can run the container runtime with GPUs. The script
   defaults to `docker run --gpus all`; if your cluster uses **enroot/pyxis** or
   **podman**, edit the `RUN=` line at the top of the runner.

## Run the whole eval (one command)

```bash
sbatch deploy/b200/run_eval_b200.sh                          # vLLM, BF16
sbatch --export=ALL,BACKEND=trtllm deploy/b200/run_eval_b200.sh
sbatch --export=ALL,BACKEND=sglang deploy/b200/run_eval_b200.sh
```

It does **Step 1** (build dataset if missing) then **Step 2** (serve → health
gate → benchmark all workloads → results). Output lands in
`results/<backend>_<tag>_<model>_<jobid>/`.

Override anything via `--export`, e.g.:
```bash
sbatch --export=ALL,BACKEND=vllm,QUANT=fp8,MEASURE_S=120 deploy/b200/run_eval_b200.sh
sbatch --export=ALL,BACKEND=vllm,LOAD_MODE=open,WORKLOADS=lc_32k deploy/b200/run_eval_b200.sh
```

Key knobs: `BACKEND` (vllm|sglang|trtllm), `MODEL_PATH`, `TP` (=8 for one node),
`MAX_MODEL_LEN`, `WORKLOADS`, `WARMUP_S`/`MEASURE_S`, `MAX_CONCURRENCY`,
`QUANT` (fp8 / nvfp4), `LOAD_MODE` (closed|open).

## The B200-only experiments worth running

These are blocked on AMD/ROCm and are the cross-vendor story:

1. **TensorRT-LLM** backend — the third backend we can't run on MI355X.
2. **Long context 256K / 512K / ~1M** via dual-chunk attention (DCA), which
   *is* supported on the CUDA path. To do this:
   - build the 1M dataset: it's already defined in `configs/datasets_long.yaml`
     (workloads `lc_256k`, `lc_512k`, `lc_960k`);
   - serve with the 1M config: make a symlink-farm model dir whose `config.json`
     is the model's `config_1m.json` (so DCA is enabled), point `MODEL_PATH` at
     it, and set `MAX_MODEL_LEN=1000000`:
     ```bash
     FARM=$HOME/models/qwen3_235b_1m
     mkdir -p "$FARM"; for f in "$MODEL_PATH"/*; do ln -sf "$f" "$FARM/$(basename "$f")"; done
     cp "$MODEL_PATH/config_1m.json" "$FARM/config.json"
     sbatch --export=ALL,MODEL_PATH="$FARM",MAX_MODEL_LEN=1000000,\
DATASET=prompts_qwen3_235b_long,DATASET_CONFIG=$REPO/configs/datasets_long.yaml,\
WORKLOADS="lc_256k lc_512k lc_960k",MEASURE_S=240,WARMUP_S=20 \
       deploy/b200/run_eval_b200.sh
     ```
3. **FP4 (nvfp4)** — Blackwell-native FP4 efficiency (`QUANT=nvfp4`).

## Post-processing (same as AMD)

```bash
python -m opendc_bench.analyze leaderboard results --workload lc_32k
python -m opendc_bench.analyze energy   results/<run>     # tokens/joule (nvidia-smi power)
python -m opendc_bench.analyze speedup  results/<run> --cold lc_32k --warm lc_cache
```

Telemetry auto-detects `nvidia-smi`, so tokens/joule works out of the box. The
top-level `README.md` and `paper/metrics.tex` describe the full metric set.
