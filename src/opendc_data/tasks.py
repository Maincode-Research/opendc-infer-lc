"""RULER-style task builders.

Each builder returns a record dict ready for JSONL serialization. Needles use a
distinctive "magic number" pattern so the answer is an unambiguous exact-match
target for the quality guardrail (spec 2.11).
"""
from __future__ import annotations

import random
from typing import Dict, List

from .haystack import make_filler
from .tokenizer import Tokenizer, fit_to_length

# Distinctive keys so needles are findable but not guessable from context.
_KEYWORDS = [
    "azimuth", "basalt", "cinnabar", "dovetail", "ember", "fathom", "gimbal",
    "halyard", "isobar", "jetty", "kelvin", "lattice", "marl", "nadir",
    "obelisk", "pylon", "quartz", "ravine", "sextant", "tundra",
]


def _value(rng: random.Random) -> str:
    return str(rng.randint(1_000_000, 9_999_999))


def _needle(key: str, value: str) -> str:
    return f" One of the special magic numbers for {key} is: {value}. "


_HEADER = (
    "You are given a long document. Somewhere in it are one or more sentences "
    "stating a special magic number for a keyword. Read the document and answer "
    "the question at the end.\n\nDocument:\n"
)


def _footer(key: str) -> str:
    return (
        f"\n\nQuestion: What is the special magic number for {key}? "
        "Answer with the number only."
    )


def build_niah_single(
    tok: Tokenizer, idx: int, seed: int, target_tokens: int,
    max_output_tokens: int, depth: float, workload: str,
) -> Dict:
    rng = random.Random(seed)
    key = rng.choice(_KEYWORDS)
    value = _value(rng)
    filler = make_filler(seed ^ 0xA5A5, approx_chars=target_tokens * 5)
    prompt, achieved = fit_to_length(
        tok, _HEADER, filler, _footer(key),
        needles=[_needle(key, value)], needle_depths=[depth],
        target_tokens=target_tokens,
    )
    return {
        "id": f"{workload}/niah_single/{idx:06d}",
        "workload": workload, "task": "niah_single",
        "tokenizer": tok.name,
        "target_tokens": target_tokens, "achieved_tokens": achieved,
        "depth": depth,
        "prompt": prompt, "max_output_tokens": max_output_tokens,
        "answers": [value], "answer_type": "exact_match",
        "needles": [{"key": key, "value": value, "depth": depth}],
        "seed": seed,
    }


def build_niah_multikey(
    tok: Tokenizer, idx: int, seed: int, target_tokens: int,
    max_output_tokens: int, depth: float, workload: str, distractors: int,
) -> Dict:
    """One queried needle plus `distractors` decoy needles (different keys) at
    spread-out depths. Tests retrieval under competing similar patterns."""
    rng = random.Random(seed)
    n = distractors + 1
    keys = rng.sample(_KEYWORDS, n)
    values = [_value(rng) for _ in range(n)]
    needles = [_needle(k, v) for k, v in zip(keys, values)]
    # Queried needle sits at the requested depth; distractors spread evenly.
    depths = [depth]
    for j in range(distractors):
        depths.append(round((j + 1) / (distractors + 1), 4))
    prompt, achieved = fit_to_length(
        tok, _HEADER, make_filler(seed ^ 0x5A5A, approx_chars=target_tokens * 5),
        _footer(keys[0]), needles=needles, needle_depths=depths,
        target_tokens=target_tokens,
    )
    return {
        "id": f"{workload}/niah_multikey/{idx:06d}",
        "workload": workload, "task": "niah_multikey",
        "tokenizer": tok.name,
        "target_tokens": target_tokens, "achieved_tokens": achieved,
        "depth": depth,
        "prompt": prompt, "max_output_tokens": max_output_tokens,
        "answers": [values[0]], "answer_type": "exact_match",
        "needles": [
            {"key": k, "value": v, "depth": d, "queried": (i == 0)}
            for i, (k, v, d) in enumerate(zip(keys, values, depths))
        ],
        "seed": seed,
    }


# --- LC-Cache -------------------------------------------------------------

_CACHE_HEADER = (
    "You are analyzing a fixed reference document across multiple questions. "
    "The document below stays the same for every question.\n\nReference document:\n"
)


def build_cache_prefix(
    tok: Tokenizer, prefix_idx: int, seed: int, prefix_tokens: int,
    keys_in_prefix: int,
) -> tuple[str, int, List[Dict]]:
    """Build one byte-identical shared prefix containing `keys_in_prefix`
    embedded magic numbers. Returns (prefix_text, achieved_tokens, facts)."""
    rng = random.Random(seed)
    keys = rng.sample(_KEYWORDS, min(keys_in_prefix, len(_KEYWORDS)))
    values = [_value(rng) for _ in keys]
    needles = [_needle(k, v) for k, v in zip(keys, values)]
    depths = [round((i + 1) / (len(keys) + 1), 4) for i in range(len(keys))]
    # No footer in the prefix — queries are appended per turn by the harness.
    prefix, achieved = fit_to_length(
        tok, _CACHE_HEADER, make_filler(seed ^ 0xC0FFEE, approx_chars=prefix_tokens * 5),
        footer="", needles=needles, needle_depths=depths,
        target_tokens=prefix_tokens,
    )
    facts = [{"key": k, "value": v} for k, v in zip(keys, values)]
    return prefix, achieved, facts


def build_cache_record(
    tok: Tokenizer, prefix_idx: int, seed: int, prefix_tokens: int,
    keys_in_prefix: int, turns_per_prefix: int, max_output_tokens: int,
) -> Dict:
    import hashlib

    prefix, achieved, facts = build_cache_prefix(
        tok, prefix_idx, seed, prefix_tokens, keys_in_prefix
    )
    rng = random.Random(seed ^ 0x7777)
    turns = []
    for t in range(turns_per_prefix):
        fact = facts[rng.randrange(len(facts))]
        turns.append({
            "turn": t,
            "query": (
                f"\n\nQuestion: What is the special magic number for "
                f"{fact['key']}? Answer with the number only."
            ),
            "answers": [fact["value"]],
            "answer_type": "exact_match",
            "max_output_tokens": max_output_tokens,
        })
    return {
        "id": f"lc_cache/prefix_{prefix_idx:04d}",
        "workload": "lc_cache",
        "prefix": prefix,
        "prefix_tokens": achieved,
        "prefix_sha256": hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
        "tokenizer": tok.name,
        "turns": turns,
        "seed": seed,
    }
