"""Benchmark CLI.

Examples:
  # everything at once against any OpenAI-compatible endpoint (recommended)
  opendc-bench suite --base-url http://127.0.0.1:8000 --data data/prompts_qwen3_235b

  # no server, no GPU: end-to-end self test on a built-in mock endpoint
  opendc-bench selftest

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


def _selftest(args) -> int:
    """Build a tiny char-tokenizer dataset, serve a mock endpoint in-process,
    and run the full suite against it. Exercises builder -> harness -> metrics
    -> quality -> summary with no GPU, no network and no model."""
    import shutil
    import tempfile
    import threading
    import types

    from . import mockserver
    from opendc_data import build as data_build

    work = tempfile.mkdtemp(prefix="opendc-selftest-")
    data_dir = os.path.join(work, "data")
    cfg = _bundled_config("datasets.smoke.yaml")
    print(f"[selftest] config={cfg}\n[selftest] workdir={work}")

    argv = sys.argv
    try:
        sys.argv = ["opendc-data", "--config", cfg, "--tokenizer", "char", "--out", data_dir]
        data_build.main()
    finally:
        sys.argv = argv

    srv = mockserver.serve(args.port, ttft=0.01, tpot=0.001, model="opendc-mock")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        from .suite import run_suite
        rc = run_suite(types.SimpleNamespace(
            base_url=f"http://127.0.0.1:{args.port}", data=data_dir, model=None,
            tokenizer="char", workloads=None, out=os.path.join(work, "results"),
            label="selftest", warmup=1.0, measure=3.0, timeout=30.0, mode="closed",
            max_concurrency=4, open_max_rps=4.0, accelerators=None,
            server_metrics_url="", wait_healthy=30.0, backend="mock",
            quant=None, notes="opendc-bench selftest"))
    finally:
        srv.shutdown()

    if args.keep:
        print(f"[selftest] kept {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    print("[selftest] PASS" if rc == 0 else "[selftest] FAIL")
    return rc


def _bundled_config(name: str) -> str:
    """Resolve a shipped workload config whether running from a git clone or
    from an installed wheel."""
    from opendc_data.build import resolve_config
    return resolve_config(name)


def main():
    ap = argparse.ArgumentParser(prog="opendc-bench")
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

    s_ = sub.add_parser("suite", help="run every workload against one endpoint (portable entry point)")
    s_.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. http://host:8000")
    s_.add_argument("--data", required=True, help="dataset dir built by `opendc-data build`")
    s_.add_argument("--model", default=None, help="served model name (default: discovered from /v1/models)")
    s_.add_argument("--tokenizer", default=None, help="'char' or 'hf:<id>' (default: from the dataset MANIFEST)")
    s_.add_argument("--workloads", default=None, help="space/comma separated (default: every split in --data)")
    s_.add_argument("--out", default=None, help="result dir (default: results/<label>_<timestamp>)")
    s_.add_argument("--label", default=None, help="short run name, used in the result dir + summary")
    s_.add_argument("--warmup", type=float, default=15.0)
    s_.add_argument("--measure", type=float, default=60.0)
    s_.add_argument("--timeout", type=float, default=600.0)
    s_.add_argument("--mode", choices=["closed", "open"], default="closed")
    s_.add_argument("--max-concurrency", type=int, default=64)
    s_.add_argument("--open-max-rps", type=float, default=64.0)
    s_.add_argument("--accelerators", type=int, default=None, help="GPU count, for tokens/s per accelerator")
    s_.add_argument("--server-metrics-url", default="auto",
                    help="'auto' (=<base-url>/metrics), an explicit URL, or '' to disable")
    s_.add_argument("--wait-healthy", type=float, default=120.0,
                    help="seconds to wait for /v1/models before starting (0 = do not wait)")
    s_.add_argument("--backend", default=None, help="free-text label recorded in run_meta.json (vllm|sglang|...)")
    s_.add_argument("--quant", default=None, help="free-text label recorded in run_meta.json (bf16|fp8|...)")
    s_.add_argument("--notes", default=None, help="free-text note recorded in run_meta.json")

    t_ = sub.add_parser("selftest", help="offline end-to-end check: mock server + smoke dataset, no GPU")
    t_.add_argument("--port", type=int, default=8765)
    t_.add_argument("--keep", action="store_true", help="keep the temp dataset/results dir")

    args = ap.parse_args()

    if args.cmd == "suite":
        from .suite import run_suite
        sys.exit(run_suite(args))
    if args.cmd == "selftest":
        sys.exit(_selftest(args))

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
