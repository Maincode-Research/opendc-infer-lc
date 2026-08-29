"""Build CLI: read a dataset config, emit frozen JSONL splits + MANIFEST.json.

The MANIFEST carries a SHA-256 over every emitted artifact (the dataset
*version hash*) plus per-split achieved-length statistics. Pin that hash in
every benchmark submission so runs are provably on identical data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from typing import Dict, List

import yaml

from . import __version__
import re

from .tasks import (
    build_realqa,
    build_cache_record,
    build_niah_multikey,
    build_niah_single,
    build_rag_report,
    build_rag_topk,
    build_summarization,
    build_repocode,
)
from .tokenizer import build_tokenizer


# ---------------------------------------------------------------------------
# Path portability: configs must work from a git clone, an installed wheel, or
# a container, on someone else's machine. Nothing here may assume our layout.
# ---------------------------------------------------------------------------

def resolve_config(spec: str) -> str:
    """Accept a path, or the bare name of a shipped config ('datasets.yaml')."""
    if os.path.exists(spec):
        return spec
    name = spec if spec.endswith((".yaml", ".yml")) else spec + ".yaml"
    tried = [spec]
    for cand in (os.path.join("configs", name), name):
        tried.append(cand)
        if os.path.exists(cand):
            return cand
    try:  # shipped inside the wheel as opendc_data.configs
        from importlib import resources
        p = resources.files("opendc_data.configs").joinpath(name)
        if p.is_file():
            return str(p)
        tried.append(str(p))
    except Exception:
        pass
    raise SystemExit(f"config not found: {spec!r} (tried: {', '.join(tried)})")


def resolve_source(spec: str, config_path: str) -> str:
    """Resolve a corpus path from a config. Supports `$VARS`, `~`, colon-
    separated lists, `py:<module>` (the install dir of an importable package),
    and paths relative to $OPENDC_DATA_ROOT, the cwd, or the config's repo."""
    roots = [os.environ.get("OPENDC_DATA_ROOT"), os.getcwd(),
             os.path.dirname(os.path.abspath(config_path)),
             os.path.dirname(os.path.dirname(os.path.abspath(config_path)))]
    out = []
    for part in str(spec).split(":"):
        if not part:
            continue
        part = os.path.expanduser(os.path.expandvars(part))
        if part.startswith("py+"):  # py+transformers -> installed package dir
            mod = part[3:]
            try:
                import importlib
                out.append(os.path.dirname(importlib.import_module(mod).__file__))
                continue
            except Exception as e:
                raise SystemExit(f"cannot locate package {mod!r} for a repocode "
                                 f"corpus ({type(e).__name__}); pip install it or "
                                 f"point code_roots at a directory")
        if os.path.isabs(part) or os.path.exists(part):
            out.append(part)
            continue
        for r in roots:
            if r and os.path.exists(os.path.join(r, part)):
                out.append(os.path.join(r, part))
                break
        else:
            raise SystemExit(
                f"corpus path not found: {part!r}\n  looked in: "
                + ", ".join(r for r in roots if r)
                + "\n  set OPENDC_DATA_ROOT=<dir containing it>, or use an absolute path")
    return ":".join(out)


def _derive_seed(base: int, *parts) -> int:
    """Stable per-sample seed. Parts may be ints or strings; everything goes
    through SHA-256, never Python's built-in hash() — that is salted per
    process (PYTHONHASHSEED), which would make builds non-reproducible across
    machines and break the dataset_version_hash contract."""
    h = hashlib.sha256(("|".join(str(p) for p in (base, *parts))).encode()).digest()
    return int.from_bytes(h[:8], "big")


def _write_jsonl(path: str, records: List[Dict]) -> str:
    """Write records, return sha256 of the file bytes."""
    h = hashlib.sha256()
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            line = json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n"
            f.write(line)
            h.update(line.encode("utf-8"))
    return h.hexdigest()


def _length_stats(records: List[Dict], field: str) -> Dict:
    vals = [r[field] for r in records if field in r]
    if not vals:
        return {}
    return {
        "n": len(vals), "min": min(vals), "max": max(vals),
        "mean": round(statistics.mean(vals), 1),
        "stdev": round(statistics.pstdev(vals), 1) if len(vals) > 1 else 0.0,
    }


_LONGBENCH_CACHE: Dict[str, tuple] = {}


def _load_longbench(path: str) -> tuple:
    """Load a LongBench QA jsonl once; return (records, passage_pool). The pool
    is the set of individual real passages (split on the 'Passage N:' headers)
    used as real distractors when packing to the target length."""
    if path not in _LONGBENCH_CACHE:
        records = [json.loads(l) for l in open(path, encoding="utf-8")]
        pool: List[str] = []
        for r in records:
            for p in re.split(r"\nPassage \d+:\n", r["context"]):
                p = p.strip()
                if len(p) > 200:
                    pool.append(p)
        _LONGBENCH_CACHE[path] = (records, pool)
    return _LONGBENCH_CACHE[path]



_DOCS_CACHE: Dict[str, tuple] = {}


def _load_docs(path: str) -> tuple:
    """Load LongBench-style jsonl(s) for whole-DOCUMENT tasks (summarization).
    `path` may be colon-separated: gold records come from the FIRST file, while
    the distractor pool is the union of all of them. A single 50-doc corpus
    (~2.9M chars) cannot supply 983K tokens of DISTINCT filler, and repeating it
    would make a degenerate haystack — so pooling real corpora is required at 1M.
    Pool entries are whole documents, not passages."""
    if path not in _DOCS_CACHE:
        paths = path.split(":")
        records = [json.loads(l) for l in open(paths[0], encoding="utf-8")]
        pool = []
        for pth in paths:
            for l in open(pth, encoding="utf-8"):
                ctx = json.loads(l).get("context", "")
                if len(ctx) > 500:
                    pool.append(ctx)
        _DOCS_CACHE[path] = (records, pool)
    return _DOCS_CACHE[path]


_CODE_CACHE: Dict[str, tuple] = {}


def _load_code_corpus(roots: str, max_file_chars: int = 120_000) -> tuple:
    """Collect real .py files from one or more roots (colon-separated) and find
    symbols defined EXACTLY ONCE corpus-wide, so 'which file defines X' has a
    single unambiguous answer."""
    if roots not in _CODE_CACHE:
        files, sym_paths = [], {}
        for root in roots.split(":"):
            rp = os.path.abspath(root)
            for dirpath, dirnames, filenames in os.walk(rp):
                dirnames[:] = [d for d in dirnames
                               if d not in {"__pycache__", ".git", "node_modules", "test", "tests"}]
                for fn in filenames:
                    if not fn.endswith(".py"):
                        continue
                    full = os.path.join(dirpath, fn)
                    try:
                        text = open(full, encoding="utf-8", errors="replace").read()
                    except OSError:
                        continue
                    if not (500 < len(text) <= max_file_chars):
                        continue
                    rel = os.path.relpath(full, os.path.dirname(rp))
                    files.append({"path": rel, "text": text})
                    for m in re.finditer(r"^\s*(?:def|class)\s+([A-Za-z_]\w*)", text, re.M):
                        sym_paths.setdefault(m.group(1), set()).add(rel)
        candidates = [
            {"symbol": sym, "path": next(iter(paths))}
            for sym, paths in sorted(sym_paths.items())
            if len(paths) == 1 and len(sym) >= 6 and not sym.startswith("_")
        ]
        _CODE_CACHE[roots] = (files, candidates)
    return _CODE_CACHE[roots]


def _build_niah_split(tok, name, spec, base_seed, depths, distractors,
                      count_override=None, config_path=""):
    n = count_override if count_override is not None else spec["n_samples"]
    target = spec["target_tokens"]
    max_out = spec["max_output_tokens"]
    tasks = spec["tasks"]
    records: List[Dict] = []
    idx = 0
    for task_name, frac in tasks.items():
        k = round(n * frac)
        for j in range(k):
            depth = depths[j % len(depths)]
            seed = _derive_seed(base_seed, name, task_name, j)
            if task_name == "niah_single":
                rec = build_niah_single(tok, idx, seed, target, max_out, depth, name)
            elif task_name == "niah_multikey":
                rec = build_niah_multikey(tok, idx, seed, target, max_out, depth, name, distractors)
            elif task_name == "rag_topk":
                rec = build_rag_topk(tok, idx, seed, target, max_out, depth, name,
                                     spec.get("k_passages", 32),
                                     spec.get("distractors", distractors))
            elif task_name == "rag_report":
                rec = build_rag_report(tok, idx, seed, target, max_out, name,
                                       spec.get("n_facts", 5))
            elif task_name == "realqa":
                recs, pool = _load_longbench(resolve_source(spec["source_path"], config_path))
                rec = build_realqa(tok, idx, seed, target, max_out, name, recs, pool)
            elif task_name == "summarization":
                recs, pool = _load_docs(resolve_source(spec["source_path"], config_path))
                rec = build_summarization(tok, idx, seed, target, max_out, name,
                                          recs, pool, spec.get("source", "longbench"))
            elif task_name == "repocode":
                cfiles, cands = _load_code_corpus(resolve_source(spec["code_roots"], config_path))
                if not cands:
                    raise ValueError(f"no uniquely-defined symbols in {spec['code_roots']!r}")
                rec = build_repocode(tok, idx, seed, target, max_out, name, cfiles, cands)
            else:
                raise ValueError(f"Unknown task {task_name!r} in split {name!r}")
            records.append(rec)
            idx += 1
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Build OpenDC-Infer-LC prompt sets.")
    ap.add_argument("--config", required=True,
                    help="path to a workload YAML, or the bare name of a shipped "
                         "one (e.g. datasets.yaml)")
    ap.add_argument("--tokenizer", required=True, help="'char' or 'hf:<model_id>'")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    args.config = resolve_config(args.config)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tok = build_tokenizer(args.tokenizer)
    smoke = getattr(tok, "smoke", False)
    os.makedirs(args.out, exist_ok=True)

    base_seed = cfg.get("seed", 0)
    depths = cfg.get("depths", [0.0, 0.5, 1.0])
    distractors = cfg.get("multikey_distractors", 3)
    warmup_n = cfg.get("warmup", {}).get("per_split", 0)

    manifest = {
        "builder_version": __version__,
        "tokenizer": tok.name,
        "smoke": smoke,
        "base_seed": base_seed,
        "config": cfg,
        "splits": {},
        "files": {},
    }

    for name, spec in cfg["splits"].items():
        if name == "lc_cache":
            records = []
            for p in range(spec.get("n_prefixes", 1)):
                seed = _derive_seed(base_seed, 0xCACE, p)
                records.append(build_cache_record(
                    tok, p, seed, spec["prefix_tokens"], spec["keys_in_prefix"],
                    spec["turns_per_prefix"], spec["max_output_tokens"],
                ))
            path = os.path.join(args.out, f"{name}.jsonl")
            sha = _write_jsonl(path, records)
            manifest["files"][f"{name}.jsonl"] = sha
            manifest["splits"][name] = {
                "n_prefixes": len(records),
                "turns_total": sum(len(r["turns"]) for r in records),
                "prefix_length": _length_stats(records, "prefix_tokens"),
                "prefix_sha256": [r["prefix_sha256"] for r in records],
            }
            continue

        records = _build_niah_split(tok, name, spec, base_seed, depths, distractors,
                                    config_path=args.config)
        path = os.path.join(args.out, f"{name}.jsonl")
        sha = _write_jsonl(path, records)
        manifest["files"][f"{name}.jsonl"] = sha
        manifest["splits"][name] = {
            "n_samples": len(records),
            "achieved_tokens": _length_stats(records, "achieved_tokens"),
            "target_tokens": spec["target_tokens"],
            "tasks": spec["tasks"],
        }

        # Held-out warmup set (distinct seeds).
        if warmup_n:
            wrecords = _build_niah_split(
                tok, name, spec, base_seed + 999_983,
                depths, distractors, count_override=warmup_n,
                config_path=args.config,
            )
            wpath = os.path.join(args.out, f"{name}.warmup.jsonl")
            wsha = _write_jsonl(wpath, wrecords)
            manifest["files"][f"{name}.warmup.jsonl"] = wsha

    # Dataset version hash = sha256 over the sorted per-file hashes.
    digest = hashlib.sha256()
    for fname in sorted(manifest["files"]):
        digest.update(fname.encode())
        digest.update(manifest["files"][fname].encode())
    manifest["dataset_version_hash"] = digest.hexdigest()

    with open(os.path.join(args.out, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Built dataset -> {args.out}")
    print(f"  tokenizer: {tok.name}  smoke={smoke}")
    print(f"  dataset_version_hash: {manifest['dataset_version_hash']}")
    for name, s in manifest["splits"].items():
        if "achieved_tokens" in s and s["achieved_tokens"]:
            a = s["achieved_tokens"]
            print(f"  {name}: n={s['n_samples']} "
                  f"len[min/mean/max]={a['min']}/{a['mean']}/{a['max']} "
                  f"(target {s['target_tokens']})")
        else:
            print(f"  {name}: {s}")


if __name__ == "__main__":
    main()
