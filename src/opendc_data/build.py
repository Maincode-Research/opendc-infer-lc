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
from .tasks import (
    build_cache_record,
    build_niah_multikey,
    build_niah_single,
)
from .tokenizer import build_tokenizer


def _derive_seed(base: int, *parts: int) -> int:
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


def _build_niah_split(tok, name, spec, base_seed, depths, distractors, count_override=None):
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
            seed = _derive_seed(base_seed, hash(name) & 0xFFFF, hash(task_name) & 0xFFFF, j)
            if task_name == "niah_single":
                rec = build_niah_single(tok, idx, seed, target, max_out, depth, name)
            elif task_name == "niah_multikey":
                rec = build_niah_multikey(tok, idx, seed, target, max_out, depth, name, distractors)
            else:
                raise ValueError(f"Unknown task {task_name!r} in split {name!r}")
            records.append(rec)
            idx += 1
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Build OpenDC-Infer-LC prompt sets.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True, help="'char' or 'hf:<model_id>'")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

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

        records = _build_niah_split(tok, name, spec, base_seed, depths, distractors)
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
