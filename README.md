# OpenDC-Infer-LC

An open, hardware- and backend-neutral benchmark for **long-context and RAG
inference serving** in data-center environments. Instead of peak tokens/sec, it
measures whether a serving system can sustain long-context workloads under
realistic SLOs, and reports **SLO-qualified goodput** plus a broad set of
secondary metrics (latency tails, capacity, reliability, cache behavior,
hardware efficiency, scaling, and a quality guardrail).

The harness drives **any OpenAI-compatible streaming endpoint** — vLLM, SGLang,
TGI, TensorRT-LLM, or a hosted API, on AMD or NVIDIA. It never launches your
server, so nothing about your stack has to match ours.

---

## Test your system in three commands

```bash
git clone https://github.com/MaincodeHQ/opendc-infer-lc && cd opendc-infer-lc
pip install ".[hf]"          # harness + real tokenizers
                             # (once published: pip install "opendc-infer-lc[hf]")

# 0. no server needed — verify the install end to end (mock endpoint, ~30s)
opendc-bench selftest

# 1. build the prompt set for YOUR model's tokenizer (once; deterministic)
opendc-data build --config datasets.yaml \
    --tokenizer hf:meta-llama/Llama-3.1-8B-Instruct --out data/prompts_llama31

# 2. point it at your already-running endpoint
opendc-bench suite --base-url http://127.0.0.1:8000 --data data/prompts_llama31 \
    --accelerators 8 --label my-stack
```

That's it. `suite` waits for the endpoint to come up, discovers the served model
name from `/v1/models` and the tokenizer from the dataset manifest, runs a
capacity search for every workload in the dataset, scrapes `/metrics` if the
backend exposes it, and writes a comparable result directory:

```
results/my-stack_<timestamp>/
├── run_meta.json     endpoint, hardware, dataset_version_hash  ← pin this
├── lc_8k.json …      full capacity ladder per workload
├── summary.csv       machine-readable
└── summary.md        the table below
```

```
| workload | goodput tok/s | C* | sat | ttft p95 | tpot p95 | e2e p95 | succ  | quality |
|----------|---------------|----|-----|----------|----------|---------|-------|---------|
| lc_8k    | 7592.0        | 32 | 64  | 0.611    | 0.0212   | 8.03    | 1.000 | 1.000   |
```

**Useful flags:** `--workloads lc_32k lc_128k` (subset) · `--mode open
--open-max-rps 32` (Poisson arrivals instead of a concurrency ladder) ·
`--max-concurrency 256` · `--measure 300` · `--backend vllm --quant fp8`
(recorded in `run_meta.json`) · `--server-metrics-url ''` (disable scraping).

### Container (no Python setup)

```bash
docker build -t opendc-infer-lc .
docker run --rm --network host -v "$PWD/data:/data" -v "$PWD/results:/results" \
    opendc-infer-lc suite --base-url http://127.0.0.1:8000 \
    --data /data/prompts_llama31 --out /results/my-stack
```

The image is CPU-only (~370 MB, or ~180 MB without the `[hf]` extra) and serves nothing — run it next to your server,
or on another host with `--base-url http://<server>:8000`.

### Editable install (development)

```bash
pip install -e ".[hf]"
opendc-bench selftest
```

### Prompt sets

Datasets are **per-tokenizer** (lengths are exact token counts for *your*
tokenizer) and frozen with a `dataset_version_hash` in `MANIFEST.json`. Two
options, both fine — pin the hash either way:

- **Build them** (above). Deterministic: same config + same tokenizer → the
  same bytes on any machine, with builder ≥ 0.2.0.
- **Download** pre-built sets for the tokenizers we publish, from the
  companion Hugging Face dataset (see `data/README.md`).

The real-text workloads (`datasets_rag.yaml`, `datasets_realqa.yaml`,
`datasets_1m.yaml`) pack genuine documents from LongBench; fetch those sources
once with:

```bash
opendc-data fetch-corpus --out data/_public_src     # stdlib only, no extra deps
```

It pulls LongBench's `data.zip` and writes the six subsets in the exact form the
builder expects — including the prompt-template split and 50-document limit the
published 1M sets were built from, so the corpora come out record-for-record
identical to ours (`--limit 0` keeps all 200 whole-document records instead).

### Reporting a result

A submission is the result directory as-is. It already carries everything a
reader needs to trust and reproduce the number: `dataset_version_hash`, the
served model, harness version, load mode, warmup/measure windows, accelerator
count, and the full ladder — not just the winning point. Runs near an SLO
boundary vary; repeat and use `analyze variance` before claiming a win.

---

## Repository layout

```
opendc-infer-lc/
├── pyproject.toml            ← installable package (`opendc-bench`, `opendc-data`)
├── Dockerfile                ← CPU-only harness image
├── configs/
│   ├── datasets.yaml         ← standard workloads (8K/32K/128K/cache)
│   ├── datasets_long.yaml    ← extreme-length probe (256K/512K/960K)
│   ├── datasets_rag.yaml     ← industry RAG shapes (real documents)
│   ├── datasets_realqa.yaml  ← public real multi-doc QA (LongBench)
│   ├── datasets_1m.yaml      ← 1M-token applications (repo code, multi-doc)
│   └── datasets.smoke.yaml   ← tiny offline smoke config
├── src/
│   ├── opendc_data/          ← dataset builder (RULER-style NIAH + LC-Cache)
│   │   ├── build.py  tokenizer.py  haystack.py  tasks.py  fetch.py  cli.py
│   └── opendc_bench/         ← reference harness (pure stdlib, no torch needed)
│       ├── suite.py          ← ★ one-command runner for any endpoint
│       ├── client.py         ← async streaming client (TTFT/TPOT/E2E contract)
│       ├── metrics.py        ← SLO profiles + goodput + percentiles + sub-rates
│       ├── runner.py         ← closed/open-loop load gen + capacity search
│       ├── quality.py        ← quality guardrail (needle recall)
│       ├── servermetrics.py  ← scrape vLLM /metrics (prefix-cache hit, KV)
│       ├── router.py         ← round-robin proxy for multi-node (one endpoint)
│       ├── analyze.py        ← leaderboard / scaling / energy / speedup / drift
│       ├── mockserver.py     ← offline mock endpoint for testing
│       └── cli.py            ← `suite`, `point`, `capacity`, `selftest`
├── deploy/                   ← OUR cluster recipes (optional; see below)
│   ├── mc2/                  ← AMD MI355X (ROCm, SLURM + podman)
│   ├── b200/                 ← NVIDIA B200 (CUDA)
│   └── gpu_telemetry.py/.sh  ← AMD/NVIDIA power/util/temp sampler
├── paper/metrics.tex         ← metrics section + table for the paper
└── data/                     ← frozen per-tokenizer prompt sets (generated)
```

---

## Reference deployment: serve + benchmark in one SLURM job (our cluster)

Everything below is **optional** — it is how *we* run the benchmark on the mc2
AMD cluster, kept as a worked example. You do not need SLURM, podman or ROCm to
use the benchmark; `opendc-bench suite` is the supported entry point.

### One command on AMD MI355X

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

All post-processing is `python -m opendc_bench.analyze <cmd>` (or
`opendc-analyze <cmd>` after install):

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

The harness is dependency-free stdlib and ships a mock endpoint, so the whole
pipeline — build → serve → measure → score → report — runs on a laptop:

```bash
opendc-bench selftest              # does all of the below in a temp dir, asserts PASS
```

Or drive the pieces by hand:

```bash
opendc-data build --config datasets.smoke.yaml --tokenizer char --out data/_smoke
python -m opendc_bench.mockserver --port 8000 &        # fake streaming server
opendc-bench suite --base-url http://127.0.0.1:8000 --data data/_smoke \
    --tokenizer char --warmup 1 --measure 3 --max-concurrency 8
```

`selftest` is also the CI check — if it passes on your machine, the harness,
metrics, quality guardrail and reporting all work; only the endpoint is left.

For real datasets you need the `[hf]` extra and the model tokenizer (HF id or
local path; pass as `--tokenizer hf:<path-or-id>`).

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
