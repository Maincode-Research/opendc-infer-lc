"""Benchmark CLI.

Examples:
  # single closed-loop point
  python -m opendc_bench.cli point --base-url http://127.0.0.1:8000 \
      --model test --tokenizer char --data data/_smoke --workload lc_8k \
      --mode closed --level 8 --warmup 2 --measure 5

  # capacity search (concurrency ladder until SLO break)
  python -m opendc_bench.cli capacity --base-url http://node01:8000 \
      --model Qwen/Qwen2.5-32B-Instruct --tokenizer hf:Qwen/Qwen2.5-32B-Instruct \
      --data data/prompts_qwen32b --workload lc_32k \
      --warmup 30 --measure 300 --out results/qwen32b_lc32k.json
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys

from opendc_data.tokenizer import build_tokenizer

from .client import StreamClient
from .dataset import load_split, split_path
from .runner import capacity_search, run_point


def _common(ap):
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True, help="'char' or 'hf:<id>'")
    ap.add_argument("--data", required=True, help="dataset dir (per-tokenizer)")
    ap.add_argument("--workload", required=True,
                    help="lc_8k|lc_32k|lc_128k|lc_cache|lc_256k|lc_512k|lc_960k|lc_1m")
    ap.add_argument("--warmup", type=float, default=30.0)
    ap.add_argument("--measure", type=float, default=300.0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--accelerators", type=int, default=None,
                    help="GPU count, for tokens/s-per-accelerator")
    ap.add_argument("--server-metrics-url", default=None,
                    help="e.g. http://127.0.0.1:8300/metrics to scrape cache/KV stats")
    ap.add_argument("--out", default=None)


def main():
    ap = argparse.ArgumentParser(prog="opendc_bench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("point", help="single load point")
    _common(p)
    p.add_argument("--mode", choices=["closed", "open"], default="closed")
    p.add_argument("--level", type=float, required=True,
                   help="concurrency (closed) or rps (open)")
    p.add_argument("--dump-raw", default=None,
                   help="write per-request JSONL (for p99-drift / stability analysis)")

    c = sub.add_parser("capacity", help="closed-loop concurrency ladder (or open-loop rps sweep)")
    _common(c)
    c.add_argument("--max-concurrency", type=int, default=1024)
    c.add_argument("--mode", choices=["closed", "open"], default="closed",
                   help="closed = concurrency ladder; open = Poisson rps sweep")
    c.add_argument("--open-max-rps", type=float, default=64.0)

    args = ap.parse_args()

    tok = build_tokenizer(args.tokenizer)
    client = StreamClient(args.base_url, args.model, tok, request_timeout=args.timeout)
    records, gold = load_split(split_path(args.data, args.workload))

    if args.cmd == "point":
        m, raw = asyncio.run(run_point(
            client, records, gold, args.workload,
            mode=args.mode, level=args.level,
            warmup_s=args.warmup, measure_s=args.measure,
            n_accelerators=args.accelerators))
        out = dataclasses.asdict(m)
        if args.dump_raw:
            with open(args.dump_raw, "w") as f:
                for r in raw:
                    if r.is_warmup:
                        continue
                    f.write(json.dumps({
                        "id": r.id, "start_time": r.start_time,
                        "completion_time": r.completion_time,
                        "ttft": r.ttft, "tpot": r.tpot, "e2e": r.e2e,
                        "n_input_tokens": r.n_input_tokens,
                        "n_output_tokens": r.n_output_tokens,
                        "success": r.success, "quality_ok": r.quality_ok,
                        "error": r.error}) + "\n")
            print(f"wrote raw -> {args.dump_raw}", file=sys.stderr)
    else:
        def on_point(lvl_, m_):
            unit = "rps" if args.mode == "open" else "C"
            print(f"  {unit}={lvl_:<6} goodput={m_.goodput_tokens_per_s:8.1f} tok/s  "
                  f"ttft_p95={m_.ttft['p95']}  tpot_p95={m_.tpot['p95']}  "
                  f"succ={m_.success_rate:.3f}  q={m_.quality_score:.2f}  "
                  f"qualified={m_.slo_qualified}", file=sys.stderr)
        if args.mode == "open":
            from .runner import rate_search
            out = asyncio.run(rate_search(
                client, records, gold, args.workload,
                warmup_s=args.warmup, measure_s=args.measure,
                max_rps=args.open_max_rps,
                n_accelerators=args.accelerators, on_point=on_point))
        else:
            out = asyncio.run(capacity_search(
                client, records, gold, args.workload,
                warmup_s=args.warmup, measure_s=args.measure,
                max_concurrency=args.max_concurrency,
                n_accelerators=args.accelerators, on_point=on_point))

    # optional: scrape server-side cache/KV metrics once at the end
    if args.server_metrics_url:
        from .servermetrics import scrape_server_metrics
        out["server_metrics"] = scrape_server_metrics(args.server_metrics_url)

    text = json.dumps(out, indent=2, default=str)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
