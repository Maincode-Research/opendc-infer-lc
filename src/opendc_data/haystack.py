"""Seeded synthetic filler ("haystack") generation.

We generate varied, natural-looking distractor sentences from templates with a
seeded RNG rather than repeating a fixed passage. Variety matters: trivially
repetitive text is both easy to compress and inflates prefix-cache hit rates,
which would distort the very cache metrics the benchmark measures. The content
is license-clean (fully synthetic) and fully reproducible from the seed.
"""
from __future__ import annotations

import random
from typing import List

_ADJ = [
    "quiet", "distant", "ancient", "restless", "hollow", "amber", "northern",
    "weathered", "silent", "narrow", "golden", "frozen", "scattered", "vivid",
    "modest", "tangled", "luminous", "rugged", "shallow", "fleeting",
]
_NOUN = [
    "harbor", "ledger", "orchard", "signal", "meadow", "engine", "archive",
    "current", "lantern", "bridge", "compass", "thicket", "furnace", "garden",
    "terminal", "estuary", "corridor", "beacon", "quarry", "pasture",
]
_VERB = [
    "drifted past", "obscured", "outlasted", "circled", "mirrored", "unsettled",
    "anchored", "echoed through", "narrowed toward", "scattered across",
    "lingered near", "overtook", "bordered", "framed", "trailed behind",
]
_CONN = [
    "Meanwhile,", "In the report,", "By most accounts,", "As noted earlier,",
    "Across the district,", "For the record,", "On the whole,",
    "According to the survey,", "In passing,", "Over several seasons,",
]


def _sentence(rng: random.Random) -> str:
    return (
        f"{rng.choice(_CONN)} the {rng.choice(_ADJ)} {rng.choice(_NOUN)} "
        f"{rng.choice(_VERB)} the {rng.choice(_ADJ)} {rng.choice(_NOUN)}."
    )


def make_filler(seed: int, approx_chars: int) -> str:
    """Return seeded filler text of at least `approx_chars` characters.

    The caller (length control) re-tokenizes and truncates to the exact token
    budget, so we just need to overshoot a little.
    """
    rng = random.Random(seed)
    out: List[str] = []
    n = 0
    # ~110 chars/sentence; overshoot by 20% to be safe before token truncation.
    target = int(approx_chars * 1.2) + 256
    while n < target:
        s = _sentence(rng)
        out.append(s)
        n += len(s) + 1
    return " ".join(out)
