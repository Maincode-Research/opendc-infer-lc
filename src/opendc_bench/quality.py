"""Quality guardrail scoring (spec 2.11).

For our RULER-style NIAH answers the score is exact-match recall: did the model
emit the magic-number answer string? For LC-Cache we additionally check
cross-turn consistency (the same prefix fact answered identically across turns).

QualityRatio = score_submission / score_reference  (caller computes the ratio
against a BF16 reference run; here we just produce per-request pass/fail and a
run-level score in [0,1]).
"""
from __future__ import annotations

import re
from typing import Dict, List

from .metrics import RequestResult


def score_request(answers: List[str], answer_type: str, output_text: str) -> bool:
    text = output_text.strip()
    if answer_type == "exact_match":
        # accept if any gold answer appears as a token-ish substring
        return any(re.search(rf"(?<!\d){re.escape(a)}(?!\d)", text) for a in answers)
    if answer_type == "recall":
        return all(a.lower() in text.lower() for a in answers)
    if answer_type == "f1":
        # lightweight token-F1 >= 0.5 threshold
        gold = set(re.findall(r"\w+", " ".join(answers).lower()))
        pred = set(re.findall(r"\w+", text.lower()))
        if not gold:
            return True
        inter = len(gold & pred)
        if inter == 0:
            return False
        p, r = inter / max(len(pred), 1), inter / len(gold)
        return (2 * p * r / (p + r)) >= 0.5
    raise ValueError(f"unknown answer_type {answer_type!r}")


def apply_quality(results: List[RequestResult], gold_by_id: Dict[str, dict]) -> float:
    """Annotate each result's quality_ok in place; return run-level score
    (fraction of successful requests that pass). gold_by_id maps request id ->
    {answers, answer_type}."""
    n, ok = 0, 0
    for r in results:
        g = gold_by_id.get(r.id)
        if g is None:
            continue
        if not r.success:
            r.quality_ok = False
            continue
        passed = score_request(g["answers"], g["answer_type"], r.output_text)
        r.quality_ok = passed
        n += 1
        ok += int(passed)
    return (ok / n) if n else 0.0
