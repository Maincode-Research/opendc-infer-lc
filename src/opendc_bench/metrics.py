"""SLO profiles and metric aggregation (spec 2.5, 2.6, 2.7).

Pure functions over a list of RequestResult so they are unit-testable without a
server.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RequestResult:
    """Per-request measurement (spec 2.9 contract)."""
    id: str
    workload: str
    start_time: float            # monotonic, set just before send
    ttft: Optional[float]        # seconds to first non-empty token
    tpot: Optional[float]        # seconds/token after the first
    e2e: Optional[float]         # seconds to clean stream close
    completion_time: float       # monotonic at stream close (or failure)
    n_output_tokens: int         # re-tokenized output length (y_i)
    n_input_tokens: int = 0      # re-tokenized prompt length
    output_text: str = ""
    success: bool = False        # not timed out / cancelled / malformed
    quality_ok: Optional[bool] = None  # filled by the quality guardrail
    error: Optional[str] = None  # category: timeout|oom|server_error|client_error|conn_error
    is_warmup: bool = False


@dataclass(frozen=True)
class SLOProfile:
    name: str
    ttft_p95_max: float          # seconds
    tpot_p95_max: float          # seconds/token
    e2e_p95_max: float           # seconds
    success_rate_min: float      # fraction


# Table 2.
INTERACTIVE = SLOProfile("interactive_rag", 3.0, 0.120, 60.0, 0.99)
HEAVY = SLOProfile("heavy_long_context", 10.0, 0.200, 180.0, 0.98)

# Workload -> profile (spec 2.5).
WORKLOAD_PROFILE: Dict[str, SLOProfile] = {
    "lc_8k": INTERACTIVE,
    "lc_32k": INTERACTIVE,
    "lc_cache": INTERACTIVE,   # 32K configured context -> interactive
    "lc_128k": HEAVY,
    # extreme long-context probes use the heavy profile
    "lc_256k": HEAVY,
    "lc_512k": HEAVY,
    "lc_960k": HEAVY,
    "lc_1m": HEAVY,
}


def percentile(values: List[float], q: float) -> Optional[float]:
    """Nearest-rank percentile; q in [0,100]. None if no values."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    # nearest-rank: rank = ceil(q/100 * n), 1-indexed
    import math
    rank = max(1, math.ceil(q / 100.0 * len(xs)))
    return xs[min(rank, len(xs)) - 1]


@dataclass
class RunMetrics:
    workload: str
    profile: str
    window_seconds: float
    n_total: int
    n_success: int
    success_rate: float
    ttft: Dict[str, Optional[float]] = field(default_factory=dict)
    tpot: Dict[str, Optional[float]] = field(default_factory=dict)
    e2e: Dict[str, Optional[float]] = field(default_factory=dict)
    output_tokens_total: int = 0
    input_tokens_total: int = 0
    goodput_tokens_per_s: float = 0.0
    raw_tokens_per_s: float = 0.0          # successful output tokens / T
    total_tokens_per_s: float = 0.0        # (input+output) of successful / T
    requests_per_s: float = 0.0
    tokens_per_accelerator: Optional[float] = None  # goodput / n_accelerators
    slo_qualified: bool = False
    slo_breakdown: Dict[str, bool] = field(default_factory=dict)
    quality_eligible: int = 0
    quality_score: float = 0.0             # fraction of successful that passed quality
    # reliability sub-rates (fraction of all measured requests)
    timeout_rate: float = 0.0
    oom_rate: float = 0.0
    server_error_rate: float = 0.0
    client_error_rate: float = 0.0
    conn_error_rate: float = 0.0


def _pcts(values: List[Optional[float]]) -> Dict[str, Optional[float]]:
    vs = [v for v in values if v is not None]
    return {
        "p50": percentile(vs, 50),
        "p95": percentile(vs, 95),
        "p99": percentile(vs, 99),
    }


def aggregate(
    results: List[RequestResult],
    workload: str,
    window_seconds: float,
    profile: Optional[SLOProfile] = None,
    require_quality: bool = True,
    n_accelerators: Optional[int] = None,
) -> RunMetrics:
    """Aggregate measurement-window results into RunMetrics + SLO qualification.

    A request contributes to goodput only if it is successful AND (when
    `require_quality`) passed the quality guardrail (v_i in spec 2.6).
    Warmup results must already be filtered out by the caller.
    """
    profile = profile or WORKLOAD_PROFILE.get(workload, HEAVY)
    measured = [r for r in results if not r.is_warmup]
    n_total = len(measured)
    succ = [r for r in measured if r.success]
    n_success = len(succ)
    success_rate = (n_success / n_total) if n_total else 0.0

    ttft = _pcts([r.ttft for r in succ])
    tpot = _pcts([r.tpot for r in succ])
    e2e = _pcts([r.e2e for r in succ])

    # v_i: successful and (if required) quality-eligible.
    def valid(r: RequestResult) -> bool:
        if not r.success:
            return False
        if require_quality and r.quality_ok is False:
            return False
        return True

    valid_reqs = [r for r in succ if valid(r)]
    goodput_tokens = sum(r.n_output_tokens for r in valid_reqs)
    raw_tokens = sum(r.n_output_tokens for r in succ)
    total_tokens = sum(r.n_output_tokens + r.n_input_tokens for r in succ)
    input_tokens = sum(r.n_input_tokens for r in succ)

    # quality score: fraction of successful requests that passed the guardrail.
    n_quality_passed = sum(1 for r in succ if r.quality_ok)
    quality_score = (n_quality_passed / n_success) if n_success else 0.0

    # reliability sub-rates over all measured requests, by error category.
    def rate(cat: str) -> float:
        if not n_total:
            return 0.0
        return sum(1 for r in measured if r.error == cat) / n_total

    T = window_seconds if window_seconds > 0 else 1.0
    m = RunMetrics(
        workload=workload, profile=profile.name, window_seconds=window_seconds,
        n_total=n_total, n_success=n_success, success_rate=success_rate,
        ttft=ttft, tpot=tpot, e2e=e2e,
        output_tokens_total=raw_tokens, input_tokens_total=input_tokens,
        goodput_tokens_per_s=goodput_tokens / T,
        raw_tokens_per_s=raw_tokens / T,
        total_tokens_per_s=total_tokens / T,
        requests_per_s=n_total / T,
        tokens_per_accelerator=((goodput_tokens / T) / n_accelerators
                                if n_accelerators else None),
        quality_eligible=len(valid_reqs),
        quality_score=quality_score,
        timeout_rate=rate("timeout"), oom_rate=rate("oom"),
        server_error_rate=rate("server_error"),
        client_error_rate=rate("client_error"),
        conn_error_rate=rate("conn_error"),
    )

    # SLO qualification (spec 2.6): all p95 thresholds + success rate.
    checks = {
        "ttft_p95": (ttft["p95"] is not None and ttft["p95"] < profile.ttft_p95_max),
        "tpot_p95": (tpot["p95"] is not None and tpot["p95"] < profile.tpot_p95_max),
        "e2e_p95": (e2e["p95"] is not None and e2e["p95"] < profile.e2e_p95_max),
        "success_rate": success_rate > profile.success_rate_min,
    }
    m.slo_breakdown = checks
    m.slo_qualified = all(checks.values())
    return m
