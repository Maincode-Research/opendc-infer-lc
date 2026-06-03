"""Scrape a vLLM (or compatible) Prometheus /metrics endpoint for cache/KV
behavior (spec 2.7 cache metrics). Best-effort: returns {} if unreachable.

vLLM exposes counters like:
  vllm:gpu_prefix_cache_queries_total, vllm:gpu_prefix_cache_hits_total,
  vllm:gpu_cache_usage_perc, vllm:num_requests_running/waiting, ...
We derive prefix-cache hit rate from the hit/query counters when present, and
pass through a few gauges. Dependency-free (urllib).
"""
from __future__ import annotations

import re
import urllib.request
from typing import Dict, Optional


def _parse_prometheus(text: str) -> Dict[str, float]:
    """metric_name{labels} value  ->  {metric_name: summed value}."""
    out: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-\d.eE+]+)", line)
        if not m:
            continue
        name, _, val = m.groups()
        try:
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            pass
    return out


def scrape_server_metrics(url: str, timeout: float = 5.0) -> Dict[str, Optional[float]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            text = r.read().decode("utf-8", "ignore")
    except Exception as e:
        return {"available": False, "error": f"{type(e).__name__}"}

    p = _parse_prometheus(text)

    def find(*needles, exclude=()):
        """First metric whose name contains all needles (and none of exclude)."""
        for name, val in p.items():
            nl = name.lower()
            if all(n in nl for n in needles) and not any(x in nl for x in exclude):
                return name, val
        return None, None

    # Counter/gauge names vary across backends & versions; match by substring.
    # vLLM: vllm:gpu_prefix_cache_{hits,queries}_total (+ a hit_rate gauge);
    # SGLang: sglang:cache_hit_rate gauge, sglang:token_usage.
    _, hits = find("prefix_cache", "hit", exclude=("rate",))
    _, queries = find("prefix_cache", "quer")
    hit_rate = None
    if hits is not None and queries:
        hit_rate = hits / queries
    if hit_rate is None:  # direct hit-rate gauge (vLLM or SGLang)
        _, hit_rate = find("cache", "hit", "rate")
    _, kv = find("kv_cache", "usage")
    if kv is None:
        _, kv = find("gpu_cache", "usage")
    if kv is None:
        _, kv = find("token_usage")          # SGLang

    return {
        "available": True,
        "prefix_cache_hit_rate": hit_rate,
        "prefix_cache_hits_total": hits,
        "prefix_cache_queries_total": queries,
        "kv_cache_usage_perc": kv,
        "num_requests_running": find("num_requests_running")[1],
        "num_requests_waiting": find("num_requests_waiting")[1],
    }
