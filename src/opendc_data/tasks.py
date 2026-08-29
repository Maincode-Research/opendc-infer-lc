"""RULER-style task builders.

Each builder returns a record dict ready for JSONL serialization. Needles use a
distinctive "magic number" pattern so the answer is an unambiguous exact-match
target for the quality guardrail (spec 2.11).
"""
from __future__ import annotations

import random
import re
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


# --- Industry RAG shapes --------------------------------------------------
# Two workloads that model how enterprise RAG actually arrives, rather than a
# single contiguous haystack: RAG-TopK (k retrieved passages from different
# documents, exactly one carrying the answer) and RAG-Report (multi-document
# context with a long structured output that must reproduce every required fact).


def _passages_filler(seed: int, k_passages: int, approx_chars: int) -> str:
    """Filler assembled as `k_passages` independently-seeded passages with
    document headers, so the context looks like a top-k retrieval set rather
    than one contiguous document. The caller's length control still truncates
    to the exact token budget."""
    per = max(64, approx_chars // max(1, k_passages))
    parts: List[str] = []
    for i in range(k_passages):
        parts.append(f"\n[Document {i + 1}]\n")
        parts.append(make_filler(seed ^ (0x9E3779B1 + i * 0x101), approx_chars=per))
    return "".join(parts)


_RAG_HEADER = (
    "You are given several retrieved passages. Exactly one of them states a "
    "special magic number for a keyword; the others are unrelated distractors. "
    "Read the passages and answer the question at the end.\n\nRetrieved passages:\n"
)


def build_rag_topk(
    tok: Tokenizer, idx: int, seed: int, target_tokens: int,
    max_output_tokens: int, depth: float, workload: str,
    k_passages: int, distractors: int,
) -> Dict:
    """Top-k retrieval: one queried needle plus `distractors` decoy needles
    (different keys), spread across `k_passages` retrieved chunks. Distinct
    retrieved content per query -> fragmented prefill and low cross-query cache
    reuse, unlike the single-haystack workloads."""
    rng = random.Random(seed)
    n = distractors + 1
    keys = rng.sample(_KEYWORDS, n)
    values = [_value(rng) for _ in range(n)]
    needles = [_needle(k, v) for k, v in zip(keys, values)]
    depths = [depth]
    for j in range(distractors):
        depths.append(round((j + 1) / (distractors + 1), 4))
    filler = _passages_filler(seed ^ 0x70504B, k_passages, approx_chars=target_tokens * 5)
    prompt, achieved = fit_to_length(
        tok, _RAG_HEADER, filler, _footer(keys[0]),
        needles=needles, needle_depths=depths, target_tokens=target_tokens,
    )
    return {
        "id": f"{workload}/rag_topk/{idx:06d}",
        "workload": workload, "task": "rag_topk",
        "tokenizer": tok.name,
        "target_tokens": target_tokens, "achieved_tokens": achieved,
        "depth": depth, "n_passages": k_passages,
        "prompt": prompt, "max_output_tokens": max_output_tokens,
        "answers": [values[0]], "answer_type": "exact_match",
        "needles": [
            {"key": k, "value": v, "depth": d, "queried": (i == 0)}
            for i, (k, v, d) in enumerate(zip(keys, values, depths))
        ],
        "seed": seed,
    }


_REPORT_HEADER = (
    "You are given a long multi-document context. Several special magic numbers "
    "are stated in it, each tied to a keyword. Read the documents, then write a "
    "report.\n\nDocuments:\n"
)


_REALQA_HEADER = (
    "You are given several passages retrieved from Wikipedia. Read them and "
    "answer the question at the end as concisely as possible (a few words), "
    "using only information stated in the passages.\n\nRetrieved passages:\n"
)


def _realqa_footer(question: str) -> str:
    return f"\n\nQuestion: {question}\nAnswer:"


def build_realqa(
    tok: Tokenizer, idx: int, seed: int, target_tokens: int,
    max_output_tokens: int, workload: str,
    records: List[Dict], pool: List[str],
) -> Dict:
    """Real public multi-document QA (external validity). Each request uses one
    LongBench HotpotQA example (real Wikipedia passages + a real multi-hop
    question + gold short answer) as the answerable core, then packs additional
    *real* passages sampled from the corpus until the prompt reaches the target
    length. The gold context is placed first so the exact-length truncation only
    trims appended distractors, and the question is appended after truncation so
    it is always present. Graded by token-F1 (answer_type='f1'), the standard
    multi-doc-QA metric. This is the real-text analogue of the synthetic
    RAG-TopK shape."""
    rng = random.Random(seed)
    base = records[idx % len(records)]
    question = " ".join(base["input"].split())
    answers = [a.strip() for a in base["answers"] if a and a.strip()]
    base_ctx = base["context"]
    # Append real distractor passages (from other documents) until we comfortably
    # overshoot the budget; fit_to_length then truncates to exact target tokens.
    overshoot_chars = target_tokens * 6
    cur = len(base_ctx)
    # Sample the WHOLE pool in random order. A fixed cap (was 600) silently
    # under-fills at long targets: fit_to_length then repeats the filler to
    # reach the budget, so a 983K-token prompt would be the same ~165K tokens
    # of distractors cycled ~6x -- a degenerate haystack, not a real one.
    picks = rng.sample(range(len(pool)), len(pool))
    extra: List[str] = []
    used = 0
    for pi in picks:
        if cur >= overshoot_chars:
            break
        extra.append(f"\n\nPassage:\n{pool[pi]}")
        cur += len(pool[pi])
        used += 1
    filler = base_ctx + "".join(extra)
    # True when the corpus could not supply enough distinct text and
    # fit_to_length had to cycle the filler; callers should report it.
    filler_repeated = cur < overshoot_chars
    prompt, achieved = fit_to_length(
        tok, _REALQA_HEADER, filler, _realqa_footer(question),
        needles=[], needle_depths=[], target_tokens=target_tokens,
    )
    return {
        "id": f"{workload}/realqa/{idx:06d}",
        "workload": workload, "task": "realqa",
        "tokenizer": tok.name,
        "target_tokens": target_tokens, "achieved_tokens": achieved,
        "source": "longbench/hotpotqa", "source_id": base.get("_id"),
        "distractor_passages": used, "filler_repeated": filler_repeated,
        "prompt": prompt, "max_output_tokens": max_output_tokens,
        "answers": answers, "answer_type": "f1",
        "question": question,
        "seed": seed,
    }


def _report_footer(keys: List[str]) -> str:
    listed = ", ".join(keys)
    return (
        "\n\nTask: Write a short report that states the special magic number for "
        f"each of these keywords: {listed}. Put each on its own line as "
        "'<keyword>: <number>'."
    )


def build_rag_report(
    tok: Tokenizer, idx: int, seed: int, target_tokens: int,
    max_output_tokens: int, workload: str, n_facts: int,
) -> Dict:
    """Report generation: a multi-document context with `n_facts` required facts
    that the long output must all reproduce. Exercises the decode-bound regime
    and long-generation tail latency the short-output workloads never reach.
    Quality is `all_exact`: every required magic number must appear in the
    output (digit-boundary match)."""
    rng = random.Random(seed)
    n = max(1, n_facts)
    keys = rng.sample(_KEYWORDS, n)
    values = [_value(rng) for _ in range(n)]
    needles = [_needle(k, v) for k, v in zip(keys, values)]
    depths = [round((i + 1) / (n + 1), 4) for i in range(n)]
    filler = _passages_filler(seed ^ 0x12E901, n, approx_chars=target_tokens * 5)
    prompt, achieved = fit_to_length(
        tok, _REPORT_HEADER, filler, _report_footer(keys),
        needles=needles, needle_depths=depths, target_tokens=target_tokens,
    )
    return {
        "id": f"{workload}/rag_report/{idx:06d}",
        "workload": workload, "task": "rag_report",
        "tokenizer": tok.name,
        "target_tokens": target_tokens, "achieved_tokens": achieved,
        "n_facts": n,
        "prompt": prompt, "max_output_tokens": max_output_tokens,
        "answers": values, "answer_type": "all_exact",
        "needles": [{"key": k, "value": v} for k, v in zip(keys, values)],
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# E1 — realistic million-token application workloads
#
# All three follow the proven build_realqa construction: the answerable core is
# placed FIRST, real distractor material is packed after it, fit_to_length
# truncates from the end (so truncation only ever trims distractors), and the
# question is appended AFTER truncation so it is always present.
# ---------------------------------------------------------------------------

_SUMM_HEADER = (
    "You are given a large collection of real documents. Exactly one of them is "
    "the TARGET DOCUMENT, which appears first, immediately after this line. The "
    "remaining documents are unrelated and provided only as context. Read the "
    "target document and answer the instruction at the end about THAT document "
    "only.\n\nTARGET DOCUMENT:\n"
)


def _summ_footer(query: str) -> str:
    return (
        "\n\nInstruction (about the TARGET DOCUMENT at the very beginning of "
        f"this input): {query}\nAnswer:"
    )


def build_summarization(
    tok: Tokenizer, idx: int, seed: int, target_tokens: int,
    max_output_tokens: int, workload: str,
    records: List[Dict], pool: List[str], source: str = "longbench",
) -> Dict:
    """Million-token *retrieve-then-summarize*: one real gold document (LongBench
    gov_report / qmsum / multi_news) placed first as the target, packed out to
    the target length with other real documents as distractors, graded by ROUGE-L
    against that document's reference summary.

    Note on framing: summarizing the ENTIRE haystack has no single gold
    reference and is therefore not ROUGE-gradable, so the task is deliberately
    'summarize the target document embedded in a large corpus' — the realistic
    long-context shape (one relevant document among many) and the direct
    analogue of the rag_realqa construction."""
    rng = random.Random(seed)
    base = records[idx % len(records)]
    query = " ".join(base["input"].split())
    answers = [a for a in base["answers"] if a and a.strip()]
    core = base["context"]

    overshoot_chars = target_tokens * 6
    cur = len(core)
    picks = rng.sample(range(len(pool)), len(pool))
    extra: List[str] = []
    used = 0
    for pi in picks:
        if cur >= overshoot_chars:
            break
        extra.append(f"\n\nUNRELATED DOCUMENT:\n{pool[pi]}")
        cur += len(pool[pi])
        used += 1
    filler = core + "".join(extra)
    prompt, achieved = fit_to_length(
        tok, _SUMM_HEADER, filler, _summ_footer(query),
        needles=[], needle_depths=[], target_tokens=target_tokens,
    )
    return {
        "id": f"{workload}/summarization/{idx:06d}",
        "workload": workload, "task": "summarization",
        "tokenizer": tok.name,
        "target_tokens": target_tokens, "achieved_tokens": achieved,
        "source": source, "source_id": base.get("_id"),
        "distractor_docs": used, "filler_repeated": cur < overshoot_chars,
        "prompt": prompt, "max_output_tokens": max_output_tokens,
        "answers": answers, "answer_type": "rouge",
        "question": query,
        "seed": seed,
    }


_CODE_HEADER = (
    "You are given the source code of a software repository. Each file begins "
    "with a line of the form '# FILE: <path>'. Read the code and answer the "
    "question at the end. Answer with the file path only.\n\nREPOSITORY:\n"
)


def _code_footer(symbol: str) -> str:
    return (
        f"\n\nQuestion: Exactly one file in this repository defines the Python "
        f"function or class named `{symbol}`. Give the value of its '# FILE:' "
        f"path, and nothing else.\nAnswer:"
    )


def build_repocode(
    tok: Tokenizer, idx: int, seed: int, target_tokens: int,
    max_output_tokens: int, workload: str,
    files: List[Dict], candidates: List[Dict],
) -> Dict:
    """Repository-scale code analysis. Packs real source files to the target
    length and asks which file defines a symbol that occurs EXACTLY ONCE across
    the whole packed corpus, so the answer is unambiguous and exact-match
    gradeable. This is genuine cross-file retrieval over a repository-sized
    context rather than a synthetic needle."""
    rng = random.Random(seed)
    cand = candidates[idx % len(candidates)]
    symbol, gold_path = cand["symbol"], cand["path"]

    by_path = {f["path"]: f["text"] for f in files}
    core = f"# FILE: {gold_path}\n{by_path[gold_path]}"

    overshoot_chars = target_tokens * 6
    cur = len(core)
    others = [f for f in files if f["path"] != gold_path]
    picks = rng.sample(range(len(others)), len(others))
    extra: List[str] = []
    used = 0
    for pi in picks:
        f = others[pi]
        # never admit a distractor that also defines the symbol
        if re.search(rf"^\s*(def|class)\s+{re.escape(symbol)}\b", f["text"], re.M):
            continue
        if cur >= overshoot_chars:
            break
        extra.append(f"\n\n# FILE: {f['path']}\n{f['text']}")
        cur += len(f["text"])
        used += 1
    filler = core + "".join(extra)
    prompt, achieved = fit_to_length(
        tok, _CODE_HEADER, filler, _code_footer(symbol),
        needles=[], needle_depths=[], target_tokens=target_tokens,
    )
    return {
        "id": f"{workload}/repocode/{idx:06d}",
        "workload": workload, "task": "repocode",
        "tokenizer": tok.name,
        "target_tokens": target_tokens, "achieved_tokens": achieved,
        "source": "local-repo", "symbol": symbol,
        "distractor_files": used, "filler_repeated": cur < overshoot_chars,
        "prompt": prompt, "max_output_tokens": max_output_tokens,
        "answers": [gold_path], "answer_type": "path_match",
        "question": f"which file defines {symbol}",
        "seed": seed,
    }
