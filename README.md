# OpenDC-Infer-LC

An open, hardware- and backend-neutral benchmark for **long-context and RAG
inference serving** in data-center environments. Instead of peak tokens/sec, it
measures whether a serving system can sustain long-context workloads under
realistic SLOs, and reports **SLO-qualified goodput** plus a broad set of
secondary metrics (latency tails, capacity, reliability, cache behavior,
hardware efficiency, scaling, and a quality guardrail).

This repo contains the **dataset builder**, the **reference harness**, and the
**cluster run scripts**. It is currently exercised on AMD MI355X (vLLM/ROCm); it
is backend- and vendor-neutral (vLLM / SGLang / TGI / TensorRT-LLM, AMD or
NVIDIA) — any OpenAI-compatible streaming endpoint works.

---

## Repository layout

```
opendc-infer-lc/
├── README.md                 ← this file
├── requirements.txt          ← deps for the dataset build (transformers/tokenizers/pyyaml)
├── configs/
│   ├── datasets.yaml         ← standard workloads (8K/32K/128K/cache)
│   ├── datasets_long.yaml    ← extreme-length probe (256K/512K/960K)
│   └── datasets.smoke.yaml   ← tiny offline smoke config
├── src/
│   ├── opendc_data/          ← dataset builder (RULER-style NIAH + LC-Cache)
│   │   ├── build.py  tokenizer.py  haystack.py  tasks.py
│   └── opendc_bench/         ← reference harness (pure stdlib, no torch needed)
│       ├── client.py         ← async streaming client (TTFT/TPOT/E2E contract)
│       ├── metrics.py        ← SLO profiles + goodput + percentiles + sub-rates
│       ├── runner.py         ← closed/open-loop load gen + capacity search
│       ├── quality.py        ← quality guardrail (needle recall)
│       ├── servermetrics.py  ← scrape vLLM /metrics (prefix-cache hit, KV)
│       ├── router.py         ← round-robin proxy for multi-node (one endpoint)
│       ├── analyze.py        ← leaderboard / scaling / energy / speedup / drift
│       ├── mockserver.py     ← offline mock endpoint for testing
│       └── cli.py            ← `point` and `capacity` commands
├── deploy/
│   ├── mc2/                  ← AMD MI355X (ROCm)
│   │   ├── run_eval_amd.sh   ← ★ ONE-FILE end-to-end eval (build → serve → bench)
│   │   └── run_all.sh        ← submit the whole AMD matrix (one job per backend)
│   ├── b200/                 ← NVIDIA B200 (CUDA)
│   │   ├── run_eval_b200.sh  ← ★ ONE-FILE eval (vllm/sglang/trtllm; DCA/1M)
│   │   └── README.md         ← B200 quickstart + cross-vendor notes
│   ├── gpu_telemetry.py/.sh  ← AMD/NVIDIA power/util/temp sampler
│   └── README.md             ← deployment guide
├── paper/metrics.tex         ← metrics section + table for the paper
└── data/                     ← frozen per-tokenizer prompt sets (generated)
```

---

## Quickstart — run the whole eval on AMD (one command)

`deploy/mc2/run_eval_amd.sh` is a single SLURM job that does everything:

- **Step 1 — build dataset** (idempotent; skipped if already present)
- **Step 2 — run eval**: serve the model (vLLM/ROCm in podman) → health gate →
  benchmark all workloads (full metric set) → write results → tear down

```bash
sbatch deploy/mc2/run_eval_amd.sh
```

Override anything via `--export`:

```bash
sbatch --export=ALL,WORKLOADS="lc_8k lc_32k",MEASURE_S=120,MAX_CONCURRENCY=128 \
       deploy/mc2/run_eval_amd.sh
```

Results land in `results/<backend>_<tag>_<model>_<jobid>/` as one JSON per
workload (`lc_8k.json`, …) plus `run_meta.json`, `server.log`, `telemetry.csv`.

**Sweep the whole matrix** (one job per backend cell):
```bash
bash deploy/mc2/run_all.sh                       # vllm + sglang + tgi × all workloads
```

**Study toggles** (via `--export`):
```bash
sbatch --export=ALL,BACKEND=sglang        deploy/mc2/run_eval_amd.sh   # backend
sbatch --export=ALL,ENFORCE_EAGER=0       deploy/mc2/run_eval_amd.sh   # HIP graphs
sbatch --export=ALL,QUANT=fp8             deploy/mc2/run_eval_amd.sh   # FP8 (separate category)
sbatch --export=ALL,CACHE_MODE=cold,WORKLOADS=lc_cache deploy/mc2/run_eval_amd.sh
sbatch --export=ALL,LOAD_MODE=open,WORKLOADS=lc_32k    deploy/mc2/run_eval_amd.sh
```

**FP4 (MXFP4) study** — quantize once with AMD Quark, then benchmark the output:
```bash
sbatch deploy/mc2/quantize_fp4_quark.sh          # -> <model>-MXFP4/ (needs Quark examples cloned)
sbatch --export=ALL,MODEL_PATH=<…>-MXFP4,SERVED=odc-qwen3-235b-fp4,TAG=fp4 deploy/mc2/run_eval_amd.sh
```

> **mc2 gotcha (important):** `/home` is **node-local** on the compute nodes,
> so the repo + datasets must live on a **shared** mount
> (`/scratch/mc2/skypilot/shared/...`), and the harness runs **inside the
> serving container** (which already has transformers/torch). The script
> defaults to those shared paths.

### Defaults (edit at the top of the script or via `--export`)

| Var | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `…/model/Qwen3-235B-A22B-Instruct-2507` | local model dir (shared) |
| `MAX_MODEL_LEN` | `135168` | 132K — fits 128K input + output |
| `WORKLOADS` | `lc_8k lc_32k lc_128k lc_cache` | which workloads to run |
| `TP` | `8` | tensor-parallel (one MI355X node) |
| `WARMUP_S` / `MEASURE_S` | `15` / `60` | warmup + timed window per point |
| `MAX_CONCURRENCY` | `64` | closed-loop ladder ceiling |
| `BACKEND` | `vllm` | `vllm` \| `sglang` \| `tgi` |
| `ENFORCE_EAGER` | `1` | `0` enables HIP graphs |
| `QUANT` | _(none)_ | `fp8` (separate category) |
| `CACHE_MODE` | `warmed` | `cold` (prefix cache off) \| `prefilled` |
| `LOAD_MODE` | `closed` | `open` (Poisson rps sweep) |

---

## Workloads (`configs/datasets.yaml`)

| Split | Input tokens | Max output | Scenario |
|---|---|---|---|
| `lc_8k` | 8,192 | 512 | standard RAG |
| `lc_32k` | 32,768 | 512 | enterprise doc QA (main leaderboard) |
| `lc_128k` | 131,072 | 1024 | long-doc / codebase QA |
| `lc_cache` | 32K shared prefix + multi-turn | 256/turn | agentic / repeated-prefix RAG |

Each prompt is RULER-style needle-in-a-haystack with an exact token length and
an answer key (for the quality guardrail). Datasets are **per-tokenizer** and
frozen with a `dataset_version_hash` in `MANIFEST.json` — pin it in submissions.

---

## Metrics

Primary: **SLO-qualified goodput** = `Σ vᵢyᵢ / T` over requests meeting the SLO
profile, counting only successful **and** quality-passing requests.

Secondary (all emitted by the harness): TTFT/TPOT/E2E p50/p95/p99; output &
total (input+output) tok/s; requests/s; tokens/s per accelerator; max
SLO-qualified concurrency + saturation boundary; success rate and
timeout/OOM/server/client/conn sub-rates; prefix-cache hit rate + KV usage
(from the backend's `/metrics`); cache-reuse speedup; tokens/joule (from power
telemetry); run-to-run variance and p99 drift; multi-node `Efficiency(N)`; and
the quality guardrail (`QualityRatio ≥ 0.98` vs a BF16 reference).

See `paper/metrics.tex` (Table with an availability column) for the full list
and definitions.

---

## Post-processing

```bash
python -m opendc_bench.analyze leaderboard results --workload lc_32k
python -m opendc_bench.analyze scaling results/*_N1_* results/*_N2_* --workload lc_32k
python -m opendc_bench.analyze energy   results/<run>          # tokens/joule
python -m opendc_bench.analyze speedup  results/<run> --cold lc_32k --warm lc_cache
python -m opendc_bench.analyze variance results/run_a results/run_b --workload lc_32k
python -m opendc_bench.analyze drift    results/<run>/raw.jsonl    # p99 tail drift
```

---

## Offline development (no cluster, no GPU)

The harness is dependency-free stdlib and ships a mock endpoint:

```bash
python -m opendc_data.build --config configs/datasets.smoke.yaml --tokenizer char --out data/_smoke
python -m opendc_bench.mockserver --port 8000 &        # fake streaming server
PYTHONPATH=src python -m opendc_bench.cli capacity \
    --base-url http://127.0.0.1:8000 --model test --tokenizer char \
    --data data/_smoke --workload lc_8k --warmup 1 --measure 3 --max-concurrency 8
```

For real datasets you need `pip install -r requirements.txt` and the model
tokenizer (HF id or local path; pass as `--tokenizer hf:<path-or-id>`).

---

## Backend & precision support observed on AMD MI355X (Qwen3-235B-A22B)

| | Status on MI355X (gfx950) |
|---|---|
| vLLM (ROCm) | ✓ works |
| SGLang (ROCm) | ✓ works; clears the LC-128K SLO where eager vLLM does not |
| TGI (ROCm) | ✗ officially MI210/250/300 only — gfx950 unverified/unsupported |
| TensorRT-LLM | ✗ CUDA-only → run on the B200 leg |
| BF16 | ✓ reference |
| FP8 (dynamic `--quantization fp8`) | ✓ runs, quality intact, but **lowered** goodput on this stack |
| FP4 (MXFP4 via AMD Quark) | ◐ offline file-to-file quantization (`quantize_fp4_quark.sh`) |
| Context > 256K (DCA / 1M) | ✗ DCA unsupported by vLLM's ROCm attention backend → B200 only |

## Notes / known constraints

- The model's **native context is 262144 (256K)**; LC-128K fits at the default
  132K served length. >256K needs DCA (CUDA path, see `deploy/b200/`).
- Server startup (large-model load) happens **before** the timed window.
- Single runs near an SLO boundary vary run-to-run; use the stability/variance
  analyzer (`opendc_bench.analyze variance`) across repeats before concluding.

See `deploy/README.md` for the AMD runner/sweep and `deploy/b200/README.md` for
the NVIDIA B200 leg.
