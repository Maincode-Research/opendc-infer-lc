# Deployment

One-file, end-to-end runners — each does: **Step 1** build dataset (idempotent)
→ **Step 2** serve → health gate → benchmark all workloads → results.

```
deploy/
├── mc2/
│   ├── run_eval_amd.sh   ← AMD MI355X (ROCm, podman). BACKEND = vllm | sglang | tgi
│   └── run_all.sh        ← submit the whole AMD matrix (one job per backend cell)
├── b200/
│   ├── run_eval_b200.sh  ← NVIDIA B200 (CUDA, docker). BACKEND = vllm | sglang | trtllm
│   └── README.md         ← B200 quickstart + cross-vendor notes
└── gpu_telemetry.py/.sh  ← AMD/NVIDIA power/util/temp sampler (auto-detected)
```

## AMD (MI355X)
```bash
sbatch deploy/mc2/run_eval_amd.sh                          # vLLM, BF16, all workloads
sbatch --export=ALL,BACKEND=sglang deploy/mc2/run_eval_amd.sh
bash   deploy/mc2/run_all.sh                                # vllm + sglang + tgi sweep
```
Study toggles (via `--export`): `ENFORCE_EAGER=0` (graphs), `QUANT=fp8`,
`CACHE_MODE=cold`, `LOAD_MODE=open`.

> mc2 note: `/home` is node-local, so the repo + datasets live on the shared
> mount (`/scratch/mc2/skypilot/shared/yue/opendc-infer-lc`) and the harness
> runs inside the serving container. The runner defaults to those paths.

## NVIDIA (B200)
See [`b200/README.md`](b200/README.md). Same harness/datasets; CUDA serve layer;
adds TensorRT-LLM and >256K (DCA/1M) context.

## Results
```bash
python -m opendc_bench.analyze leaderboard results --workload lc_32k
python -m opendc_bench.analyze energy   results/<run>          # tokens/joule
python -m opendc_bench.analyze speedup  results/<run> --cold lc_32k --warm lc_cache
python -m opendc_bench.analyze scaling  results/*_N1_* results/*_N2_* --workload lc_32k
```
