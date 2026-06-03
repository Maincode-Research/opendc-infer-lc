"""Load frozen prompt sets into flat request records the harness can fire.

Non-cache splits map 1:1 to requests. LC-Cache expands each prefix's turns into
requests whose prompt is `prefix + turn.query` (byte-identical prefix across
turns so the server's prefix cache can engage).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple


def load_split(path: str) -> Tuple[List[dict], Dict[str, dict]]:
    """Return (requests, gold_by_id).

    requests carry: id, workload, prompt, max_output_tokens.
    gold_by_id maps id -> {answers, answer_type}.
    """
    requests: List[dict] = []
    gold: Dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["workload"] == "lc_cache":
                prefix = rec["prefix"]
                for turn in rec["turns"]:
                    rid = f"{rec['id']}/turn_{turn['turn']:04d}"
                    requests.append({
                        "id": rid,
                        "workload": "lc_cache",
                        "prompt": prefix + turn["query"],
                        "max_output_tokens": turn["max_output_tokens"],
                    })
                    gold[rid] = {"answers": turn["answers"],
                                 "answer_type": turn["answer_type"]}
            else:
                requests.append({
                    "id": rec["id"],
                    "workload": rec["workload"],
                    "prompt": rec["prompt"],
                    "max_output_tokens": rec["max_output_tokens"],
                })
                gold[rec["id"]] = {"answers": rec["answers"],
                                   "answer_type": rec["answer_type"]}
    return requests, gold


def split_path(data_dir: str, workload: str, warmup: bool = False) -> str:
    suffix = ".warmup.jsonl" if warmup else ".jsonl"
    return os.path.join(data_dir, f"{workload}{suffix}")
