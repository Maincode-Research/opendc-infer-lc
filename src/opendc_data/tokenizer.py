"""Tokenizer wrapper with exact token-length control.

Two backends:
  * HFTokenizer   -- real path, wraps transformers.AutoTokenizer.
  * CharTokenizer -- offline smoke path, 1 token == 1 unicode codepoint.

Both expose encode(text) -> list[int] and decode(ids) -> str so the length
control in `fit_to_length` is backend-agnostic.
"""
from __future__ import annotations

from typing import List, Protocol


class Tokenizer(Protocol):
    name: str

    def encode(self, text: str) -> List[int]: ...
    def decode(self, ids: List[int]) -> str: ...


class CharTokenizer:
    """Codepoint tokenizer. For offline smoke tests only — lengths are in
    characters, not real subword tokens."""

    def __init__(self) -> None:
        self.name = "char-approx"
        self.smoke = True

    def encode(self, text: str) -> List[int]:
        return [ord(c) for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(chr(i) for i in ids)


class HFTokenizer:
    """Wraps a Hugging Face AutoTokenizer. Encodes without special tokens so
    that concatenation in token space is well defined."""

    def __init__(self, model_id: str) -> None:
        from transformers import AutoTokenizer  # imported lazily

        self.name = model_id
        self.smoke = False
        self._tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)


def build_tokenizer(spec: str) -> Tokenizer:
    """spec is 'char' or 'hf:<model_id>'."""
    if spec == "char":
        return CharTokenizer()
    if spec.startswith("hf:"):
        return HFTokenizer(spec[len("hf:"):])
    raise ValueError(f"Unknown tokenizer spec {spec!r}; use 'char' or 'hf:<id>'")


def fit_to_length(
    tok: Tokenizer,
    header: str,
    filler: str,
    footer: str,
    needles: List[str],
    needle_depths: List[float],
    target_tokens: int,
) -> tuple[str, int]:
    """Assemble `header + <filler with needles inserted at depths> + footer`
    so the whole thing tokenizes to exactly `target_tokens` ids in this
    tokenizer's id space.

    Needles are inserted into the filler token stream at the given relative
    depths. The filler region is truncated/padded so the grand total hits the
    target exactly. Returns (prompt_text, achieved_tokens) where
    achieved_tokens re-tokenizes the *decoded* text (the value the serving
    harness will actually see).
    """
    if len(needles) != len(needle_depths):
        raise ValueError("needles and needle_depths must align")

    h_ids = tok.encode(header)
    f_ids = tok.encode(footer)
    needle_ids = [tok.encode(n) for n in needles]
    needle_budget = sum(len(n) for n in needle_ids)

    filler_budget = target_tokens - len(h_ids) - len(f_ids) - needle_budget
    if filler_budget < 0:
        raise ValueError(
            f"target_tokens={target_tokens} too small for header+footer+needles "
            f"({len(h_ids) + len(f_ids) + needle_budget})"
        )

    # Build a filler token stream of exactly `filler_budget` ids, repeating the
    # supplied filler text as needed (the builder passes enough varied text).
    base = tok.encode(filler)
    if not base:
        base = tok.encode("padding ")
    filler_ids: List[int] = []
    while len(filler_ids) < filler_budget:
        filler_ids.extend(base)
    filler_ids = filler_ids[:filler_budget]

    # Insert needles at their depths. Sort by depth so earlier insertions do
    # not shift later target indices.
    order = sorted(range(len(needle_ids)), key=lambda i: needle_depths[i])
    # Work on a copy; compute insertion indices against the *original* length
    # then apply with running offset.
    insertions = []
    for i in order:
        idx = int(round(needle_depths[i] * len(filler_ids)))
        idx = max(0, min(idx, len(filler_ids)))
        insertions.append((idx, needle_ids[i]))
    offset = 0
    body = list(filler_ids)
    for idx, nids in insertions:
        at = idx + offset
        body[at:at] = nids
        offset += len(nids)

    final_ids = h_ids + body + f_ids
    # Exact in id space by construction; guard against off-by-one from rounding.
    final_ids = final_ids[:target_tokens]

    text = tok.decode(final_ids)
    achieved = len(tok.encode(text))
    return text, achieved
