# NOTE — Public real-text RAG workloads (LongBench) for OpenDC-Infer-LC

**What / why:** our RAG workloads were all synthetic (seeded filler + "magic number"
needles). Great for control (no contamination, exact 32K length, no cache-rate
inflation) but a reviewer will ask whether the findings hold on *real* text. So we
added two **public** real-text RAG workloads for external validity, built from
**LongBench** (the standard public long-context suite, `zai-org/LongBench` on HF).

## The two workloads

| Workload | Source (LongBench task) | Shape | Grading |
|---|---|---|---|
| `rag_realqa` (**RAG-RealQA**) | HotpotQA | real Wikipedia multi-hop multi-doc QA, short answer | token-F1 |
| `rag_sciqa` (**RAG-SciQA**) | QASPER | scientific-paper (NLP) QA, incl. abstractive/unanswerable | token-F1 |

Both are short-answer (`max_output_tokens=64`), single-node MI355X, BF16, 32K.

## The "real-life" construction approach (the key idea)

We do **not** just truncate a LongBench example to 32K (middle-truncation can drop the
gold span). Instead, per request:

1. Take one LongBench example — its real multi-document context is the **answerable
   core** — and place it **first**.
2. **Pack with more real passages** sampled (seeded) from *other* LongBench documents
   in the same corpus, appended after the core, until we overshoot 32K.
3. `fit_to_length` truncates to **exactly 32,768 tokens** from the end — so truncation
   only trims appended *distractors*, never the gold context.
4. Append the **question after truncation**, so it's always present.

Net: a genuine long-context, real-text, multi-document retrieval prompt (one
answerable document among many real distractors) — the real-text analogue of our
synthetic RAG-TopK. Graded by LongBench's own token-F1 (≥0.5 pass), which the harness
already implemented (`quality.py`, `answer_type="f1"`). 200 instances per workload
(= LongBench test-split size), built per served-model tokenizer, length verified at
32,767–32,768 tokens.

## How to build / run

- **Source data** (committed, pinned): `data/_public_src/{hotpotqa,qasper}.jsonl`
  (extracted from LongBench `data.zip`).
- **Builder**: `build_realqa()` in `src/opendc_data/tasks.py`; dispatch + LongBench
  loader (`_load_longbench`) in `src/opendc_data/build.py`. Source-agnostic — any
  LongBench QA jsonl works via a split's `source_path`.
- **Config**: splits `rag_realqa` / `rag_sciqa` in `configs/datasets_rag.yaml` (and a
  standalone `configs/datasets_realqa.yaml`). SLO = interactive (`metrics.py`).
- **Build** (needs the model tokenizer; do it in the vLLM container):
  `python3 -m opendc_data.build --config configs/datasets_realqa.yaml --tokenizer hf:<MODEL> --out <tmp>`
  then copy `rag_realqa*.jsonl` / `rag_sciqa*.jsonl` into `data/prompts_<model>/`.
  Already built for both 235B and 35B.
- **Run**: `sbatch --export=ALL,BACKEND=<vllm|sglang>,MODEL_PATH=...,SERVED=...,DATASET=...,WORKLOADS="rag_realqa rag_sciqa" deploy/mc2/run_eval_amd.sh`

## ⚠️ Gotcha that WILL bite you on the 35B

`Qwen3.5-35B-A3B` is a **reasoning model** (`<think>` in its chat template); the 235B
(Instruct-2507) is not. The harness sends `/v1/chat/completions`, so on a 64-token
short-answer budget the 35B spends the **entire budget thinking** and never emits the
answer → **F1 = exactly 0.00** (raw throughput looks fine, success=1.0). Don't trust a
35B F1 of 0.

**Fix:** run the 35B with `OPENDC_NO_THINK=1` — the client injects
`chat_template_kwargs={"enable_thinking": false}` (wired in `client.py` +
`run_eval_amd.sh`). With thinking off, 35B HotpotQA F1 went 0.00 → 0.73–0.75.
(Caveat: the synthetic RAG-TopK/Report 35B runs used default thinking mode, so their
35B quality is partly thinking-truncated — rerun with `OPENDC_NO_THINK=1` if you want
cross-corpus 35B numbers to be consistent.)

## Headline results (single-node MI355X, BF16, F1)

| Model·Backend | RAG-RealQA (HotpotQA) | RAG-SciQA (QASPER) |
|---|---|---|
| 235B vLLM   | 1.2@C1, F1 0.58 | 1.2@C1, F1 0.25 |
| 235B SGLang | F1 0.57 (failed interactive TTFT @C1) | 1.2@C1, F1 0.26 |
| 35B vLLM (no-think)   | 3.7@C1, F1 0.73 | 1.7@C1, F1 0.16 |
| 35B SGLang (no-think) | 5.2@C1, F1 **0.75** | 6.9@C2, F1 **0.28** |

**Two takeaways:**
1. **Synthetic overstates quality** — 235B is F1 1.0 on synthetic RAG-TopK but 0.58
   (Wikipedia) / 0.25 (scientific) on real QA. Measure quality on real text.
2. **Tier ordering is workload-dependent** — on real multi-hop QA the *smaller* 35B
   (0.73–0.75) beats the 235B (0.58), inverting the synthetic gap. The "35B
   fast-but-wrong" story is specific to adversarial same-format needle retrieval, not
   real RAG. Scientific QA is hard for both (0.16–0.28). Throughput story unchanged
   (35B more batchable; prefill-bound 235B caps at C*=1).

## Paper artifacts
`paper/results_realqa.tex` (`tab:realqa`), external-validity paragraph in
`paper/results_analysis.tex`, workload defs in `paper/methodology.tex` +
`paper/rag_appendix.tex` (`subsec:rag_details`), LongBench cite in `paper/rag_refs.bib`.
