"""One-command suite runner: point the harness at *any* OpenAI-compatible
streaming endpoint and get a complete, comparable result directory.

This is the portable entry point. It knows nothing about SLURM, podman, ROCm or
CUDA — bring your own server (vLLM / SGLang / TGI / TensorRT-LLM / a hosted API)
and it will wait for it to become healthy, discover what it can, run every
workload's capacity search, and write the submission artifacts:

    <out>/run_meta.json     environment + dataset version hash (pin this)
    <out>/<workload>.json   full capacity search per workload
    <out>/summary.md        human-readable table
    <out>/summary.csv       machine-readable table

Anything not supplied on the command line is inferred: the served model name
from /v1/models, the tokenizer from the dataset MANIFEST, and the workload list
from the split files present in the dataset directory.
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import platform
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from opendc_data.tokenizer import build_tokenizer

from . import __version__
from .client import StreamClient
from .dataset import load_split, split_path
from .runner import capacity_search, rate_search

# Canonical ordering for reports; unknown splits are appended alphabetically.
WORKLOAD_ORDER = [
    "lc_8k", "lc_32k", "lc_128k", "lc_cache",
    "lc_256k", "lc_512k", "lc_960k", "lc_1m",
    "rag_topk", "rag_report", "rag_realqa", "rag_sciqa",
]


def _get_json(url: str, timeout: float = 10.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "ignore"))
    except Exception:
        return None


def wait_healthy(base_url: str, seconds: float, quiet: bool = False) -> Optional[dict]:
    """Poll <base>/v1/models until it answers. Returns the parsed response."""
    deadline = time.monotonic() + seconds
    url = base_url.rstrip("/") + "/v1/models"
    attempt = 0
    while True:
        js = _get_json(url, timeout=5.0)
        if js is not None:
            return js
        if time.monotonic() >= deadline:
            return None
        attempt += 1
        if not quiet and attempt % 6 == 1:
            left = int(deadline - time.monotonic())
            print(f"[wait] {url} not ready yet ({left}s left)", file=sys.stderr)
        time.sleep(5.0)


def discover_model(models_json: Optional[dict]) -> Optional[str]:
    try:
        return models_json["data"][0]["id"]
    except Exception:
        return None


def discover_tokenizer(data_dir: str) -> Optional[str]:
    """Read the tokenizer the dataset was built with from its MANIFEST."""
    man = os.path.join(data_dir, "MANIFEST.json")
    if not os.path.exists(man):
        return None
    try:
        with open(man, encoding="utf-8") as f:
            name = json.load(f).get("tokenizer")
    except Exception:
        return None
    if not name:
        return None
    return "char" if name == "char-approx" else f"hf:{name}"


def discover_workloads(data_dir: str) -> List[str]:
    names = set()
    for p in glob.glob(os.path.join(data_dir, "*.jsonl")):
        base = os.path.basename(p)
        if base.endswith(".warmup.jsonl"):
            continue
        names.add(base[: -len(".jsonl")])
    ordered = [w for w in WORKLOAD_ORDER if w in names]
    return ordered + sorted(names - set(ordered))


def dataset_hash(data_dir: str) -> Optional[str]:
    man = os.path.join(data_dir, "MANIFEST.json")
    if not os.path.exists(man):
        return None
    try:
        with open(man, encoding="utf-8") as f:
            return json.load(f).get("dataset_version_hash")
    except Exception:
        return None


def _summary_rows(results: Dict[str, dict], mode: str) -> List[dict]:
    rows = []
    for w, res in results.items():
        if not res:
            rows.append({"workload": w, "status": "FAILED"})
            continue
        pts = res.get("points", [])
        best = None
        for p in pts:
            if p.get("slo_qualified"):
                best = p
        row = {
            "workload": w,
            "status": "ok" if best else "no_qualified_point",
            "goodput_tok_s": res.get("highest_qualified_goodput"),
            "capacity": (res.get("highest_qualified_rps") if mode == "open"
                         else res.get("highest_qualified_concurrency")),
            "saturation": (res.get("saturation_rps") if mode == "open"
                           else res.get("saturation_concurrency")),
            "tokens_per_accel": res.get("highest_qualified_tokens_per_accel"),
        }
        if best:
            row["ttft_p95"] = (best.get("ttft") or {}).get("p95")
            row["tpot_p95"] = (best.get("tpot") or {}).get("p95")
            row["e2e_p95"] = (best.get("e2e") or {}).get("p95")
            row["success_rate"] = best.get("success_rate")
            row["quality_score"] = best.get("quality_score")
        rows.append(row)
    return rows


_SUMMARY_COLS = ["workload", "status", "goodput_tok_s", "capacity", "saturation",
                 "tokens_per_accel", "ttft_p95", "tpot_p95", "e2e_p95",
                 "success_rate", "quality_score"]


def write_summary(out_dir: str, rows: List[dict], meta: dict, mode: str) -> str:
    cap_label = "rps*" if mode == "open" else "C*"
    with open(os.path.join(out_dir, "summary.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(_SUMMARY_COLS) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(c) is None else str(r.get(c))
                             for c in _SUMMARY_COLS) + "\n")

    header = ["workload", "goodput tok/s", cap_label, "sat", "ttft p95",
              "tpot p95", "e2e p95", "succ", "quality"]

    def fmt(r):
        def n(v, d=2):
            return "-" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
        return [r["workload"], n(r.get("goodput_tok_s"), 1), n(r.get("capacity")),
                n(r.get("saturation")), n(r.get("ttft_p95"), 3), n(r.get("tpot_p95"), 4),
                n(r.get("e2e_p95"), 2), n(r.get("success_rate"), 3),
                n(r.get("quality_score"), 3)]

    body = [fmt(r) for r in rows]
    widths = [max(len(h), *(len(b[i]) for b in body)) if body else len(h)
              for i, h in enumerate(header)]
    lines = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |",
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines += ["| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(b)) + " |"
              for b in body]

    md = [f"# OpenDC-Infer-LC run — {meta.get('label') or os.path.basename(out_dir)}", "",
          f"- endpoint: `{meta['base_url']}`",
          f"- model: `{meta['model']}`",
          f"- dataset: `{meta['data_dir']}` (`dataset_version_hash` = `{meta.get('dataset_version_hash')}`)",
          f"- tokenizer: `{meta['tokenizer']}`",
          f"- load mode: `{mode}`  warmup {meta['warmup_s']}s / measure {meta['measure_s']}s per point",
          f"- accelerators: {meta.get('accelerators')}",
          f"- harness: opendc-infer-lc {meta['harness_version']}", "",
          "Primary metric is **SLO-qualified goodput**: output tokens/s counting only",
          "requests that met the workload's SLO profile *and* passed the quality guardrail.",
          ""] + lines + [""]
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(lines))
    return path


def run_suite(args) -> int:
    data_dir = args.data
    if not os.path.isdir(data_dir):
        print(f"FATAL: --data {data_dir!r} is not a directory. Build one with:\n"
              f"  opendc-data build --config datasets.yaml --tokenizer hf:<model> --out {data_dir}",
              file=sys.stderr)
        return 2

    models_json = None
    if args.wait_healthy > 0:
        models_json = wait_healthy(args.base_url, args.wait_healthy)
        if models_json is None:
            print(f"FATAL: {args.base_url} never became healthy within "
                  f"{args.wait_healthy}s (GET /v1/models)", file=sys.stderr)
            return 3

    model = args.model or discover_model(models_json) or discover_model(
        _get_json(args.base_url.rstrip('/') + "/v1/models"))
    if not model:
        print("FATAL: could not discover the served model name; pass --model",
              file=sys.stderr)
        return 2

    tokenizer = args.tokenizer or discover_tokenizer(data_dir)
    if not tokenizer:
        print("FATAL: no MANIFEST.json in --data and no --tokenizer given",
              file=sys.stderr)
        return 2

    workloads = ([w for w in args.workloads.replace(",", " ").split() if w]
                 if args.workloads else discover_workloads(data_dir))
    if not workloads:
        print(f"FATAL: no *.jsonl splits found in {data_dir}", file=sys.stderr)
        return 2

    smetrics = args.server_metrics_url
    if smetrics == "auto":
        smetrics = args.base_url.rstrip("/") + "/metrics"

    out_dir = args.out or os.path.join(
        "results", f"{args.label or 'run'}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "label": args.label,
        "base_url": args.base_url,
        "model": model,
        "backend": args.backend,
        "quant": args.quant,
        "tokenizer": tokenizer,
        "data_dir": os.path.abspath(data_dir),
        "dataset_version_hash": dataset_hash(data_dir),
        "workloads": workloads,
        "load_mode": args.mode,
        "warmup_s": args.warmup,
        "measure_s": args.measure,
        "max_concurrency": args.max_concurrency,
        "open_max_rps": args.open_max_rps,
        "accelerators": args.accelerators,
        "harness_version": __version__,
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": args.notes,
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[suite] endpoint={args.base_url} model={model} tokenizer={tokenizer}")
    print(f"[suite] data={data_dir} hash={meta['dataset_version_hash']}")
    print(f"[suite] workloads={' '.join(workloads)} -> {out_dir}\n")

    tok = build_tokenizer(tokenizer)
    client = StreamClient(args.base_url, model, tok, request_timeout=args.timeout)

    results: Dict[str, dict] = {}
    for w in workloads:
        path = split_path(data_dir, w)
        if not os.path.exists(path):
            print(f"[skip] {w}: {path} missing", file=sys.stderr)
            results[w] = {}
            continue
        print(f"=== {w} ===")

        def on_point(level, m, _w=w):
            unit = "rps" if args.mode == "open" else "C"
            print(f"  {unit}={level:<6} goodput={m.goodput_tokens_per_s:8.1f} tok/s  "
                  f"ttft_p95={m.ttft['p95']}  tpot_p95={m.tpot['p95']}  "
                  f"succ={m.success_rate:.3f}  q={m.quality_score:.2f}  "
                  f"qualified={m.slo_qualified}", file=sys.stderr)

        try:
            records, gold = load_split(path)
            if args.mode == "open":
                res = asyncio.run(rate_search(
                    client, records, gold, w,
                    warmup_s=args.warmup, measure_s=args.measure,
                    max_rps=args.open_max_rps,
                    n_accelerators=args.accelerators, on_point=on_point))
            else:
                res = asyncio.run(capacity_search(
                    client, records, gold, w,
                    warmup_s=args.warmup, measure_s=args.measure,
                    max_concurrency=args.max_concurrency,
                    n_accelerators=args.accelerators, on_point=on_point))
        except KeyboardInterrupt:
            print("[suite] interrupted", file=sys.stderr)
            break
        except Exception as e:  # one bad workload must not lose the whole run
            print(f"WARN: workload {w} failed: {type(e).__name__}: {e}", file=sys.stderr)
            results[w] = {}
            continue

        if smetrics:
            from .servermetrics import scrape_server_metrics
            res["server_metrics"] = scrape_server_metrics(smetrics)
        results[w] = res
        with open(os.path.join(out_dir, f"{w}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=str)

    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    rows = _summary_rows(results, args.mode)
    print()
    write_summary(out_dir, rows, meta, args.mode)
    print(f"\n[suite] results -> {out_dir}")
    return 0 if any(r["status"] != "FAILED" for r in rows) else 1
