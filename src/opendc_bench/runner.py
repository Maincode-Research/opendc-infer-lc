"""Load generation (spec 2.8) and capacity search (spec 2.6).

Two modes:
  * closed-loop concurrency: C workers issue back-to-back requests; used to
    build capacity curves;
  * open-loop Poisson: requests arrive at a fixed offered rate independent of
    completions; better reflects production traffic.

Each run = warmup phase + timed measurement window (spec 2.8). Warmup requests
(those that *started* before the measurement window opens) are tagged and
excluded from metrics.
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

from .client import StreamClient
from .metrics import RequestResult, RunMetrics, WORKLOAD_PROFILE, aggregate
from .quality import apply_quality


async def _closed_loop(
    client: StreamClient, records: List[dict], concurrency: int,
    warmup_s: float, measure_s: float,
) -> Tuple[List[RequestResult], float]:
    t0 = time.monotonic()
    warmup_end = t0 + warmup_s
    deadline = warmup_end + measure_s
    results: List[RequestResult] = []
    counter = {"i": 0}

    def next_record() -> dict:
        rec = records[counter["i"] % len(records)]
        counter["i"] += 1
        return rec

    async def worker():
        while time.monotonic() < deadline:
            rec = next_record()
            res = await client.complete(rec)
            res.is_warmup = res.start_time < warmup_end
            results.append(res)

    await asyncio.gather(*[worker() for _ in range(concurrency)])
    return results, measure_s


async def _open_loop(
    client: StreamClient, records: List[dict], rate_rps: float,
    warmup_s: float, measure_s: float, seed: int = 0,
    max_inflight: int = 4096,
) -> Tuple[List[RequestResult], float]:
    t0 = time.monotonic()
    warmup_end = t0 + warmup_s
    deadline = warmup_end + measure_s
    rng = random.Random(seed)
    results: List[RequestResult] = []
    sem = asyncio.Semaphore(max_inflight)
    tasks: List[asyncio.Task] = []
    idx = 0

    async def fire(rec: dict):
        async with sem:
            res = await client.complete(rec)
        res.is_warmup = res.start_time < warmup_end
        results.append(res)

    # Schedule Poisson arrivals in real time until the deadline.
    next_arrival = t0
    while next_arrival < deadline:
        now = time.monotonic()
        if next_arrival > now:
            await asyncio.sleep(next_arrival - now)
        rec = records[idx % len(records)]
        idx += 1
        tasks.append(asyncio.create_task(fire(rec)))
        # exponential inter-arrival
        next_arrival += rng.expovariate(rate_rps)

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    return results, measure_s


def finalize(
    results: List[RequestResult], gold: Dict[str, dict], workload: str,
    window_s: float, require_quality: bool = True,
    n_accelerators: Optional[int] = None,
) -> RunMetrics:
    apply_quality(results, gold)
    return aggregate(results, workload, window_s,
                     profile=WORKLOAD_PROFILE.get(workload),
                     require_quality=require_quality,
                     n_accelerators=n_accelerators)


async def run_point(
    client: StreamClient, records: List[dict], gold: Dict[str, dict],
    workload: str, *, mode: str, level: float,
    warmup_s: float, measure_s: float, seed: int = 0,
    n_accelerators: Optional[int] = None,
) -> Tuple[RunMetrics, List[RequestResult]]:
    """Run a single load point. `level` is concurrency (closed) or rps (open).
    Returns (metrics, raw per-request results)."""
    if mode == "closed":
        results, T = await _closed_loop(client, records, int(level), warmup_s, measure_s)
    elif mode == "open":
        results, T = await _open_loop(client, records, level, warmup_s, measure_s, seed)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return finalize(results, gold, workload, T, n_accelerators=n_accelerators), results


def metrics_to_dict(level, m: RunMetrics, label: str = "concurrency") -> Dict:
    """Full per-point metric record (the complete secondary-metric set).
    `label` is 'concurrency' (closed-loop) or 'offered_rps' (open-loop)."""
    return {
        label: level,
        "slo_qualified": m.slo_qualified,
        "slo_breakdown": m.slo_breakdown,
        # primary + throughput
        "goodput_tokens_per_s": round(m.goodput_tokens_per_s, 3),
        "raw_output_tokens_per_s": round(m.raw_tokens_per_s, 3),
        "total_tokens_per_s": round(m.total_tokens_per_s, 3),
        "requests_per_s": round(m.requests_per_s, 4),
        "tokens_per_accelerator": (round(m.tokens_per_accelerator, 3)
                                   if m.tokens_per_accelerator is not None else None),
        # latency p50/p95/p99
        "ttft": m.ttft, "tpot": m.tpot, "e2e": m.e2e,
        # reliability
        "success_rate": round(m.success_rate, 4),
        "timeout_rate": round(m.timeout_rate, 4),
        "oom_rate": round(m.oom_rate, 4),
        "server_error_rate": round(m.server_error_rate, 4),
        "client_error_rate": round(m.client_error_rate, 4),
        "conn_error_rate": round(m.conn_error_rate, 4),
        # quality
        "quality_score": round(m.quality_score, 4),
        "quality_eligible": m.quality_eligible,
        "n_total": m.n_total,
    }


async def capacity_search(
    client: StreamClient, records: List[dict], gold: Dict[str, dict],
    workload: str, *, warmup_s: float, measure_s: float,
    ladder: Optional[List[int]] = None, max_concurrency: int = 1024,
    n_accelerators: Optional[int] = None,
    on_point: Optional[Callable[[int, RunMetrics], None]] = None,
) -> Dict:
    """Closed-loop concurrency ladder until the first SLO violation (spec 2.6).
    Reports the highest SLO-qualified point and the first violating point
    (saturation boundary)."""
    if ladder is None:
        ladder = []
        c = 1
        while c <= max_concurrency:
            ladder.append(c)
            c *= 2

    points = []
    highest_qualified = None
    saturation = None
    for c in ladder:
        m, _ = await run_point(client, records, gold, workload,
                               mode="closed", level=c,
                               warmup_s=warmup_s, measure_s=measure_s,
                               n_accelerators=n_accelerators)
        points.append((c, m))
        if on_point:
            on_point(c, m)
        if m.slo_qualified:
            highest_qualified = (c, m)
        else:
            saturation = (c, m)
            break

    return {
        "workload": workload,
        "ladder": ladder,
        "points": [metrics_to_dict(c, m) for c, m in points],
        "highest_qualified_concurrency": highest_qualified[0] if highest_qualified else None,
        "highest_qualified_goodput": (
            round(highest_qualified[1].goodput_tokens_per_s, 3) if highest_qualified else None),
        "highest_qualified_tokens_per_accel": (
            round(highest_qualified[1].tokens_per_accelerator, 3)
            if highest_qualified and highest_qualified[1].tokens_per_accelerator is not None
            else None),
        "saturation_concurrency": saturation[0] if saturation else None,
    }


async def rate_search(
    client: StreamClient, records: List[dict], gold: Dict[str, dict],
    workload: str, *, warmup_s: float, measure_s: float,
    ladder: Optional[List[float]] = None, max_rps: float = 64.0,
    n_accelerators: Optional[int] = None, seed: int = 0,
    on_point: Optional[Callable[[float, RunMetrics], None]] = None,
) -> Dict:
    """Open-loop offered-rate sweep (Poisson arrivals): raise rps until the SLO
    profile is violated. Reports highest SLO-qualified offered rate + the first
    non-qualified rate (saturation), per spec 2.8."""
    if ladder is None:
        ladder = []
        r = 1.0
        while r <= max_rps:
            ladder.append(r)
            r *= 2

    points = []
    highest_qualified = None
    saturation = None
    for rps in ladder:
        m, _ = await run_point(client, records, gold, workload,
                               mode="open", level=rps,
                               warmup_s=warmup_s, measure_s=measure_s,
                               seed=seed, n_accelerators=n_accelerators)
        points.append((rps, m))
        if on_point:
            on_point(rps, m)
        if m.slo_qualified:
            highest_qualified = (rps, m)
        else:
            saturation = (rps, m)
            break

    return {
        "workload": workload,
        "mode": "open",
        "ladder": ladder,
        "points": [metrics_to_dict(r, m, label="offered_rps") for r, m in points],
        "highest_qualified_rps": highest_qualified[0] if highest_qualified else None,
        "highest_qualified_goodput": (
            round(highest_qualified[1].goodput_tokens_per_s, 3) if highest_qualified else None),
        "saturation_rps": saturation[0] if saturation else None,
    }
