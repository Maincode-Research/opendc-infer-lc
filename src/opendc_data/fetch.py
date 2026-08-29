"""Fetch the public source corpora the real-text workloads build from.

The RAG / RealQA / 1M splits pack genuine documents rather than synthetic
filler, sourced from LongBench v1. This downloads them into the layout the
configs expect (`data/_public_src/<name>.jsonl`, one raw record per line) so a
new user can reproduce every workload, not just the synthetic ones.

    opendc-data fetch-corpus --out data/_public_src

Stdlib only — no `datasets` dependency. LongBench ships its splits inside a
single `data.zip` (its loader script no longer works with datasets>=4), so we
pull that archive and extract the subsets we need.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile

# LongBench v1 subsets referenced by configs/datasets_{rag,realqa,1m}.yaml.
LONGBENCH_SUBSETS = ["hotpotqa", "qasper", "2wikimqa", "gov_report", "qmsum", "multi_news"]

# The QA subsets are used raw. The three whole-document subsets feed
# `build_summarization`, which reads `input` as the query — and LongBench ships
# gov_report/multi_news with an EMPTY `input`, because their instruction lives
# in LongBench's own prompt template. We therefore apply the template split
# once here (header onto `context`, instruction into `input`), reproducing the
# corpora the published 1M prompt sets were built from. Without this the
# summarization prompts would carry an empty query.
_SUMMARIZATION_PREP = {
    "gov_report": ("Report:\n", "Now, write a one-page summary of the report.", False),
    "qmsum": ("Transcript:\n", "Query: ", True),
    "multi_news": ("News:\n", "Now, write a one-page summary of all the news.", False),
}
# Records kept per whole-document subset in the published sets.
DEFAULT_SUMMARIZATION_LIMIT = 50
# THUDM/LongBench redirects here after the org rename.
LONGBENCH_URL = "https://huggingface.co/datasets/zai-org/LongBench/resolve/main/data.zip"
EXPECTED_FIELDS = {"input", "context", "answers"}


def _download(url: str, dest: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "opendc-infer-lc"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r[get ] {done >> 20}/{total >> 20} MiB", end="", flush=True)
    print()


def _prepare(name: str, records: list, limit: int = 0) -> list:
    """Apply the LongBench prompt-template split to a whole-document subset."""
    prep = _SUMMARIZATION_PREP.get(name)
    if not prep:
        return records  # QA subsets are used raw and in full
    header, instruction, from_input = prep
    out = []
    for i, r in enumerate(records[:limit] if limit else records):
        # trailing whitespace stripped: matches the published corpora exactly
        context = header + r["context"].rstrip()
        out.append({
            "input": (instruction + r["input"]) if from_input else instruction,
            "context": context,
            "answers": r["answers"],
            "length": len(context),
            "dataset": r.get("dataset", name),
            "language": r.get("language", "en"),
            "all_classes": r.get("all_classes"),
            "_id": f"{name}-{i:04d}",
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download the public corpora used by the real-text workloads.")
    ap.add_argument("--out", default="data/_public_src", help="output directory")
    ap.add_argument("--names", nargs="*", default=LONGBENCH_SUBSETS,
                    help=f"LongBench subsets (default: {' '.join(LONGBENCH_SUBSETS)})")
    ap.add_argument("--url", default=LONGBENCH_URL, help="LongBench data.zip URL")
    ap.add_argument("--archive", default=None,
                    help="use an already-downloaded data.zip instead of fetching")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    ap.add_argument("--limit", type=int, default=DEFAULT_SUMMARIZATION_LIMIT,
                    help="records kept per whole-document subset (gov_report/qmsum/"
                         f"multi_news); {DEFAULT_SUMMARIZATION_LIMIT} reproduces the "
                         "published 1M sets. 0 = keep all. QA subsets are never trimmed.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    wanted = [n for n in args.names
              if args.force or not os.path.exists(os.path.join(args.out, f"{n}.jsonl"))]
    for n in args.names:
        if n not in wanted:
            print(f"[skip] {os.path.join(args.out, n + '.jsonl')} exists")
    if not wanted:
        return

    tmp = None
    try:
        archive = args.archive
        if not archive:
            tmp = tempfile.mkdtemp(prefix="opendc-corpus-")
            archive = os.path.join(tmp, "data.zip")
            print(f"[get ] {args.url}")
            _download(args.url, archive)

        with zipfile.ZipFile(archive) as z:
            members = {os.path.basename(m): m for m in z.namelist()
                       if m.endswith(".jsonl")}
            missing = [n for n in wanted if f"{n}.jsonl" not in members]
            if missing:
                raise SystemExit(
                    f"subset(s) not in the archive: {', '.join(missing)}\n"
                    f"  available: {', '.join(sorted(k[:-6] for k in members))}")
            for n in wanted:
                dest = os.path.join(args.out, f"{n}.jsonl")
                with z.open(members[f"{n}.jsonl"]) as src:
                    records = [json.loads(l) for l in src.read().decode("utf-8").splitlines() if l.strip()]
                got = set(records[0])
                if not EXPECTED_FIELDS <= got:
                    raise SystemExit(
                        f"{n}: unexpected schema {sorted(got)}; the builder needs "
                        f"{sorted(EXPECTED_FIELDS)}")
                prepared = _prepare(n, records, args.limit)
                with open(dest, "w", encoding="utf-8") as f:
                    for r in prepared:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                note = " (template applied)" if n in _SUMMARIZATION_PREP else ""
                print(f"[ok  ] {dest}  n={len(prepared)}{note}")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    root = os.path.abspath(os.path.dirname(args.out) or ".")
    print(f"\nDone. Build a real-text dataset with:\n"
          f"  OPENDC_DATA_ROOT={root} \\\n"
          f"  opendc-data build --config datasets_rag.yaml "
          f"--tokenizer hf:<model> --out data/prompts_rag")


if __name__ == "__main__":
    main()
