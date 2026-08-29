# Handoff: OpenDC-Infer-LC — B200 leg

Hi — this is the long-context / RAG **inference-serving benchmark** for the EMNLP
paper. I've built and run the **AMD MI355X** side; the **NVIDIA B200** side is
yours. Everything is reusable — same harness + datasets, only the launch layer
differs.

- **Code / harness / runners:** https://github.com/MaincodeHQ/opendc-infer-lc
- **Frozen datasets (private):** https://huggingface.co/datasets/Maincode/opendc-infer-lc-data

---

## 1. What I did (AMD MI355X, Qwen3-235B-A22B, vLLM/ROCm)

- Built the **dataset builder** (RULER-style needle-in-haystack + LC-Cache,
  exact token lengths, answer keys) and a **backend-neutral harness** that
  reports SLO-qualified goodput, latency tails (p50/95/99), reliability,
  prefix-cache hit rate, tokens/joule, and a quality guardrail.
- Workloads: **LC-8K / LC-32K / LC-128K / LC-Cache** (32K shared prefix).
- Ran on mc2 (8× MI355X, TP8, BF16, 132K context). Headline findings:
  - **SGLang > vLLM**, and **SGLang clears the LC-128K SLO** where vLLM (eager) doesn't.
  - **Prefix caching ≈ 15×** on LC-Cache (warmed vs cold).
  - **Dynamic FP8 *lowered* goodput** (quality intact) — pre-quantized MXFP4 is the better FP4 path.
  - **128K prefill (~20 s TTFT) is the wall**; not an eager-mode artifact.
- **Blocked on ROCm → your job on B200:**
  - **TensorRT-LLM** (CUDA-only).
  - **>256K context (512K/1M)** via dual-chunk attention (DCA) — unsupported by
    vLLM's ROCm backend, **works on CUDA**.
  - Cross-vendor (B200 vs MI355X) comparison — the novelty axis.

## 2. B200 cluster — basic instructions

```bash
git clone https://github.com/MaincodeHQ/opendc-infer-lc.git
cd opendc-infer-lc
# put the model on a path your compute nodes can see; set REPO + MODEL_PATH
```

One command does build → serve → benchmark → results:

```bash
sbatch deploy/b200/run_eval_b200.sh                          # vLLM, BF16
sbatch --export=ALL,BACKEND=trtllm deploy/b200/run_eval_b200.sh   # TensorRT-LLM
sbatch --export=ALL,BACKEND=sglang deploy/b200/run_eval_b200.sh
```

- Edit the top of `deploy/b200/run_eval_b200.sh`: `REPO`, `MODEL_PATH`, your
  SLURM `--partition`, and the container runtime (defaults to
  `docker run --gpus all`; change `RUN=` if you use enroot/pyxis or podman).
- Results land in `results/<backend>_<tag>_<model>_<jobid>/`.
- **The B200-only runs to prioritize** (see `deploy/b200/README.md` §"B200-only
  experiments"): TensorRT-LLM backend, and **256K/512K/1M** long context via the
  1M (DCA) config — there's a copy-paste recipe in that README.
- Full details + cross-vendor differences: **`deploy/b200/README.md`**.

## 3. Using the data

Two equivalent options — the runner handles either:

- **(a) Let it build (default).** `run_eval_b200.sh` **Step 1** auto-builds the
  per-tokenizer dataset if missing (deterministic; ~few min, CPU). Nothing to do.
- **(b) Download the frozen set from HF** (private — request access / use a token):
  ```bash
  pip install huggingface_hub
  huggingface-cli download Maincode/opendc-infer-lc-data --repo-type dataset \
      --local-dir data    # gives data/prompts_qwen3_235b/ etc.
  ```

Notes:
- Datasets are **per-tokenizer** — use the folder matching the model you serve
  (`prompts_qwen3_235b/` for Qwen3-235B). Lengths are in *that tokenizer's* tokens.
- Each folder has a `MANIFEST.json` with a `dataset_version_hash` — **pin it in
  any result/submission** so runs are provably on identical data.
- `prompts_qwen3_235b_long/` (256K/512K/960K) is for the B200 long-context runs.

Ping me with questions. — yue
