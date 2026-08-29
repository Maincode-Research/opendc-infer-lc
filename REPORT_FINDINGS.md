# OpenDC-Infer-LC — Findings Report

**Author:** Yue · **Platform:** mc2 (AMD MI355X / gfx950, CDNA4) + 1× NVIDIA B200
**Models:** Qwen3-235B-A22B-Instruct-2507 (235B total / 22B active) · Qwen3.5-35B-A3B (35B / 3B active)
**Backends:** vLLM, SGLang (both vendors) · TensorRT-LLM (B200) · TGI (attempted, AMD)
**Workloads:** LC-8K, LC-32K, LC-128K, LC-Cache (32K shared prefix) · SLO-qualified goodput is the primary metric.

---

## TL;DR

We benchmarked long-context / RAG inference serving across two models, three backends, two
vendors, four workloads, multiple precisions, and multi-node scaling. Seven findings below.
**Two of them are non-obvious and I want to flag specifically:** (a) the small 35B model serves
the cached-RAG workload fast but **answers it wrong 60–75% of the time** — it fails the quality
guardrail where the 235B passes 100%; and (b) **HIP graphs (non-eager) *regress* performance on
our MI355X** — eager mode is actually our best config, which matters for how we report the
AMD-vs-NVIDIA comparison.

---

## What we ran

- Reference harness: closed-loop concurrency sweep (C = 1, 2, 4, 8, …) per workload until an SLO
  profile is violated; reports SLO-qualified goodput, latency percentiles, cache hit rate,
  tokens/joule, and a needle-recall quality guardrail (QualityRatio ≥ 0.98).
- Single-node = one TP=8 replica filling a node. Multi-node = N data-parallel TP=8 replicas
  behind one production router (sgl-router), sweeping N ∈ {1, 2, 4, 8} (up to 64 GPUs).

---

## Findings

### (1) The 35B's ~45× goodput is **concurrency**, not a faster request
At LC-8K the 35B hits ~1451 tok/s (SGLang) vs ~31 for the 235B. But at **concurrency 1** the
per-stream rates are within ~2× (30.7 vs 14.7 tok/s, vLLM). The gap is **SLO-qualified
concurrency**: the 22B-active 235B breaks the 120 ms TPOT bound by C=4 (capping at C\*=2), while
the 3B-active 35B holds TPOT ≤ 67 ms all the way to C=64.
**Reason:** MoE sparsity (3B active) doesn't make one request much faster — it makes the *batched
decode* far cheaper, so under a tail-latency SLO you fit a ~32× larger batch per node.

### (2) Backend dominates; only SGLang clears 128K
SGLang beats vLLM on every workload of both models and is the **only** config that satisfies the
heavy SLO at LC-128K on AMD (235B 0.9 tok/s; 35B 170 tok/s); vLLM fails 128K on both.
**Reason:** it's prefill scheduling, not hardware — vLLM's 128K TTFT (~20 s) blows the 10 s bound;
SGLang's chunked prefill brings it to ~7 s.

### (3) Dynamic FP8 hurts; pre-quantized MXFP4 is the FP4 lever
Dynamic FP8 *lowers* goodput on both models (235B: 16.8 vs 22.2 at LC-8K) with the worst
tokens/joule — on-the-fly dequant isn't amortized. **MXFP4** (offline AMD Quark) on the 235B
*matches* BF16 goodput at ~9% lower power → best tokens/joule among vLLM precisions.
**Reason:** block-FP4 from a pre-quantized checkpoint moves the cost offline; dynamic FP8 pays it
every step. (35B MXFP4 is blocked — vLLM's fused-MoE MXFP4 loader can't read the Quark
qwen3_5_moe checkpoint; reported as a coverage gap.)

### (4) Prefix-cache value scales with how prefill-bound the model is
Cold vs warmed prefix cache collapses goodput ~15× on the 235B (≈47 → ≈3) but only ~1.1× on the
35B. **Reason:** the 3B-active 35B re-prefills 32K cheaply (caching saves little); the 22B-active
235B is prefill-bound and benefits enormously. Prefix-cache infra pays off most for large,
prefill-bound models — the data-center RAG regime.

### (5) Energy efficiency tracks goodput
Mean power varies little across configs (340–534 W), so tokens/joule is goodput-driven: SGLang
leads on both models; MXFP4 is the most efficient vLLM precision on the 235B. Systems wins and
sustainability wins are the same wins.

### (6) Multi-node scaling on a single shared prefix is **routing-bound**
DP scaling on LC-Cache (one shared prefix), raw SLO-qualified throughput, baseline = 68.3 tok/s:

| N (GPUs) | cache-aware | Eff | round-robin | Eff |
|---|---|---|---|---|
| 1 (8)  | 68.3  | 1.00 | 68.3  | 1.00 |
| 2 (16) | —     | —    | 136.5 | **1.00** |
| 4 (32) | 68.3  | **0.25** | 204.8 | 0.75 |
| 8 (64) | 68.3  | **0.13** | —     | — |

**Reason:** the workload has *one* byte-identical prefix. **Cache-aware** routes by prefix → pins
ALL traffic to one replica → throughput is flat at the single-replica rate → Eff = 1/N (extra GPUs
buy nothing). **Round-robin** spreads load → near-linear (2× at N=2, 3× at N=4) but loses prefix
reuse (each replica re-prefills). This is the cache-locality vs. load-balance tradeoff; a
production RAG router should do both (shard distinct prefixes, replicate hot ones).

### (7) Cross-vendor: B200 leads short/medium + 128K; MI355X wins cached-RAG (235B)

| Workload | Best MI355X | Best B200 | Winner |
|---|---|---|---|
| LC-8K | 31.3 (SGLang) | 38.3 (SGLang) | B200 ~1.2× |
| LC-32K | 6.5 (SGLang) | 9.1 (SGLang) | B200 ~1.4× |
| LC-128K | 0.9; vLLM **fails** | ~1.2 (all 3 backends qualify) | **B200** |
| LC-Cache | **99.6 (SGLang)** | 64 (vLLM/SGLang); 129.9 (TRT-LLM) | **MI355X** on shared backends; only CUDA-only TRT-LLM beats it |

**Reasons:** B200's Blackwell prefill FLOPS pull 128K TTFT under the 10 s bound (real silicon
advantage), and the CUDA stack (incl. TensorRT-LLM, which has no AMD equivalent here) is more
mature. **MI355X wins LC-Cache** because it has **more HBM (288 GB vs ~192 GB/GPU)** → larger
resident KV cache → more shared-prefix reuse, exactly what LC-Cache rewards.

---

## Two things I want the team to know specifically

### A. The 35B fails the LC-Cache quality guardrail (speed ≠ correctness)
The 35B is SLO-qualified on *latency* for LC-Cache but its **needle-recall quality is 0.2–0.6**
(the 235B is 1.0) — uniformly, on single-node AND multi-node, independent of concurrency. The
3B-active model serves the 32K-shared-prefix queries fast but answers the embedded needle wrong
most of the time.
- **Why this was easy to miss:** the headline `goodput` metric *silently* already filters out
  quality-failing tokens (goodput = valid-only output ÷ time). The single-node and cross-node
  goodput numbers matched, so the story looked consistent until I split goodput into raw ×
  quality per load point — that's when the ~0.3 quality jumped out.
- **How we report it now:** 35B LC-Cache goodput is marked as valid-only (‡) in the main table;
  the scaling table uses *raw* throughput (with the quality failure flagged separately) so the
  routing/scaling effect isn't masked by the quality factor. The gap itself is a result: a
  concrete speed-vs-quality cost of the small MoE.

### B. Non-eager (HIP graphs) is WORSE on MI355X — eager is our best config
Enabling HIP/CUDA graphs (jobs 551, 563) **lowered** throughput (235B LC-8K 22.2 → 15.9, −28%;
35B LC-8K 952 → 586, −38%) and the long-context ladders didn't even complete — C=1 already missed
the SLO, so the sweep stopped at the first rung.
- **Why:** graph capture *succeeds* (~13 s, ~10 GiB), but graphs only optimize the *decode*
  kernel-launch overhead — and our long-context workloads are *prefill*-bound, so there's no
  upside. The downside is (i) fixed-shape padding waste, and (ii) an immature gfx950 (CDNA4) /
  ROCm 7 / vLLM 0.19.1 replay path that adds synchronization rather than removing launch overhead.
  Note the ~10 GiB graph pool is **not** the problem here: MI355X has 288 GB HBM/GPU (more than
  B200's ~192 GB), so with the 235B at ~59 GB/GPU there's still ~200 GB for KV cache — 10 GiB is
  negligible. Net: slower, despite ample memory headroom.
- **Consequence for the paper:** the AMD eager numbers are NOT a handicap — they're AMD's best.
  So the B200-vs-MI355X comparison is fair (both at their best config), and B200's lead is genuine
  rather than an eager-vs-graph artifact. (On a mature CUDA stack graphs usually help; gfx950 just
  isn't there yet — same root cause as TGI not serving gfx950 at all.)

---

## Practical recommendations
1. **Serve long-context with SGLang** on MI355X (only backend that clears 128K; best goodput/energy).
2. **For RAG with a hot shared prefix, don't use pure cache-aware routing for scale-out** — it
   pins to one replica. Use hybrid (replicate hot prefixes) or round-robin if you can afford to
   lose cache reuse.
3. **Use MXFP4 (offline), not dynamic FP8**, when you want FP4 efficiency on the 235B.
4. **Don't enable HIP graphs on gfx950** with the current stack — run eager.
5. **Validate quality, not just latency** — a small MoE can be fast and wrong (the 35B LC-Cache case).

## Open items / caveats
- 35B MXFP4 blocked (vLLM fused-MoE loader); 235B 1M-context (dual-chunk attention) unsupported on
  ROCm; TGI doesn't serve gfx950 — all reported as explicit coverage gaps.
- B200 runs were the **235B** (filename said 35B — confirmed mislabeled). No genuine 35B-on-B200
  run yet; if one exists it should show ~950–1450 tok/s at LC-8K.
- Worth confirming the B200 graph mode for a fully rigorous cross-vendor statement, though since
  AMD-eager ≥ AMD-graph, B200's lead holds regardless.
