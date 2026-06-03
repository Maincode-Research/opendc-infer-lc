"""Post-process run directories into leaderboards + scaling efficiency.

Reads the per-workload capacity JSON files written by `cli capacity`
(`highest_qualified_goodput`, `highest_qualified_concurrency`, etc.) and the
`run_meta.json` written by run_benchmark.sh.

Commands:
  leaderboard <parent_dir> [--workload lc_32k]
      Rank every run under <parent_dir> by SLO-qualified goodput.
  scaling <run_dir> <run_dir> ... --workload lc_32k
      Compute Efficiency(N) = goodput(N) / (N * goodput(1)) (spec 2.7).
  energy <run_dir>
      Best-effort tokens/joule from telemetry CSV power columns.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Optional


def _load_meta(run_dir: str) -> dict:
    p = os.path.join(run_dir, "run_meta.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def _node_count(run_dir: str, meta: dict) -> int:
    m = re.search(r"_N(\d+)_", os.path.basename(run_dir.rstrip("/")))
    if m:
        return int(m.group(1))
    nl = meta.get("slurm_nodelist", "")
    tele = glob.glob(os.path.join(run_dir, "telemetry.*.csv"))
    return max(len(tele), 1)


def _workload_goodput(run_dir: str, workload: str) -> Optional[float]:
    p = os.path.join(run_dir, f"{workload}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get("highest_qualified_goodput")


def cmd_leaderboard(args):
    runs = [d for d in glob.glob(os.path.join(args.parent, "*")) if os.path.isdir(d)]
    rows = []
    for d in runs:
        meta = _load_meta(d)
        g = _workload_goodput(d, args.workload)
        if g is None:
            continue
        cap = json.load(open(os.path.join(d, f"{args.workload}.json")))
        rows.append((g, os.path.basename(d), meta.get("backend", "?"),
                     meta.get("model", "?"), meta.get("quant", "?"),
                     cap.get("highest_qualified_concurrency")))
    rows.sort(reverse=True)
    print(f"\n== Leaderboard: {args.workload} (SLO-qualified goodput) ==")
    print(f"{'goodput tok/s':>14}  {'C*':>5}  {'backend':<8} {'quant':<6} run")
    for g, name, backend, model, quant, c in rows:
        print(f"{g:>14.1f}  {str(c):>5}  {backend:<8} {quant:<6} {name}")


def cmd_scaling(args):
    pts = []
    for d in args.run_dirs:
        meta = _load_meta(d)
        n = _node_count(d, meta)
        g = _workload_goodput(d, args.workload)
        if g is not None:
            pts.append((n, g, os.path.basename(d)))
    if not pts:
        print("no usable runs"); return
    pts.sort()
    n1, g1, _ = pts[0]
    print(f"\n== Scaling efficiency: {args.workload} (baseline N={n1}) ==")
    print(f"{'N':>3}  {'goodput tok/s':>14}  {'Efficiency(N)':>13}")
    for n, g, name in pts:
        eff = g / (n / n1 * g1) if g1 else float("nan")
        print(f"{n:>3}  {g:>14.1f}  {eff:>13.3f}   {name}")


def _power_series(csv_path):
    """Return [(ts, total_power_W)] summing all GPUs per timestamp.
    Expects header ts,gpu,util_pct,mem_used_mb,power_w,temp_c."""
    by_ts = {}
    try:
        with open(csv_path) as f:
            header = f.readline().strip().split(",")
            try:
                tcol = header.index("ts"); pcol = header.index("power_w")
            except ValueError:
                return []
            for line in f:
                parts = line.strip().split(",")
                try:
                    ts = float(parts[tcol]); pw = float(parts[pcol])
                except (ValueError, IndexError):
                    continue
                by_ts[ts] = by_ts.get(ts, 0.0) + pw
    except OSError:
        return []
    return sorted(by_ts.items())


def _avg_power_and_energy(csv_path):
    """(avg_total_power_W, energy_J) via trapezoidal integration over ts."""
    s = _power_series(csv_path)
    if len(s) < 2:
        return (s[0][1] if s else 0.0), 0.0
    energy = 0.0
    for (t0, p0), (t1, p1) in zip(s, s[1:]):
        energy += 0.5 * (p0 + p1) * (t1 - t0)
    span = s[-1][0] - s[0][0]
    avg = energy / span if span > 0 else 0.0
    return avg, energy


def cmd_energy(args):
    csvs = [os.path.join(args.run_dir, "telemetry.csv")]
    csvs = [c for c in csvs if os.path.exists(c)] or glob.glob(
        os.path.join(args.run_dir, "telemetry.*.csv"))
    if not csvs:
        print("no telemetry CSV found"); return
    avg_w = energy = 0.0
    for c in csvs:
        a, e = _avg_power_and_energy(c)
        avg_w += a; energy += e
    print(f"\n== Energy: {os.path.basename(args.run_dir)} ==")
    print(f"avg total power: {avg_w:.0f} W  ({len(csvs)} telemetry file(s)), "
          f"energy over window: {energy/1000:.1f} kJ")
    if avg_w <= 0:
        print("(no power samples -> tokens/joule unavailable)"); return
    for p in sorted(glob.glob(os.path.join(args.run_dir, "lc_*.json"))):
        try:
            cap = json.load(open(p))
        except Exception:
            continue
        g = cap.get("highest_qualified_goodput")
        if g:
            print(f"  {os.path.basename(p)[:-5]:<10} tokens/joule = "
                  f"{g/avg_w:.4f}  (goodput {g} tok/s / {avg_w:.0f} W)")


def cmd_speedup(args):
    """Cache-reuse speedup = TTFT(cold workload) / TTFT(warm workload) at C=1."""
    def ttft_c1(workload, pct="p50"):
        d = json.load(open(os.path.join(args.run_dir, f"{workload}.json")))
        for pt in d["points"]:
            if pt["concurrency"] == 1:
                t = pt.get("ttft", {})
                return t.get(pct) if isinstance(t, dict) else None
        return None
    cold = ttft_c1(args.cold); warm = ttft_c1(args.warm)
    print(f"\n== Cache-reuse speedup ({args.cold} vs {args.warm}) ==")
    print(f"TTFT p50 @C=1: cold={cold}  warm={warm}")
    if cold and warm:
        print(f"speedup = {cold/warm:.2f}x")


def cmd_variance(args):
    """Run-to-run variance of peak SLO-qualified goodput across run dirs."""
    import statistics
    vals = []
    for d in args.run_dirs:
        g = _workload_goodput(d, args.workload)
        if g is not None:
            vals.append(g)
    print(f"\n== Run-to-run variance: {args.workload} (n={len(vals)}) ==")
    if len(vals) < 2:
        print("need >=2 runs"); return
    mean = statistics.mean(vals); sd = statistics.pstdev(vals)
    print(f"goodput: mean={mean:.2f}  stdev={sd:.2f}  cv={sd/mean*100:.1f}%  values={vals}")


def cmd_drift(args):
    """p99 (or p95) tail-latency drift across the measurement window, from a
    per-request raw dump (point --dump-raw)."""
    recs = [json.loads(l) for l in open(args.raw)]
    recs = [r for r in recs if r.get("e2e") is not None]
    if not recs:
        print("no completed requests in raw dump"); return
    recs.sort(key=lambda r: r["completion_time"])
    t0, t1 = recs[0]["completion_time"], recs[-1]["completion_time"]
    nb = args.buckets
    span = (t1 - t0) or 1.0
    buckets = [[] for _ in range(nb)]
    for r in recs:
        b = min(nb - 1, int((r["completion_time"] - t0) / span * nb))
        buckets[b].append(r["e2e"])
    import math
    def pctl(xs, q):
        xs = sorted(xs)
        if not xs:
            return None
        return xs[min(len(xs), max(1, math.ceil(q/100*len(xs)))) - 1]
    print(f"\n== Tail-latency drift (E2E p{args.pct}, {nb} buckets) ==")
    series = []
    for i, b in enumerate(buckets):
        v = pctl(b, args.pct)
        series.append(v)
        print(f"  bucket {i}: n={len(b):<4} p{args.pct}={v}")
    nn = [v for v in series if v is not None]
    if len(nn) >= 2:
        print(f"drift (last-first) = {nn[-1]-nn[0]:+.3f}s  (max-min {max(nn)-min(nn):.3f}s)")


def main():
    ap = argparse.ArgumentParser(prog="opendc_bench.analyze")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lb = sub.add_parser("leaderboard"); lb.add_argument("parent")
    lb.add_argument("--workload", default="lc_32k"); lb.set_defaults(fn=cmd_leaderboard)
    sc = sub.add_parser("scaling"); sc.add_argument("run_dirs", nargs="+")
    sc.add_argument("--workload", default="lc_32k"); sc.set_defaults(fn=cmd_scaling)
    en = sub.add_parser("energy"); en.add_argument("run_dir"); en.set_defaults(fn=cmd_energy)
    sp = sub.add_parser("speedup"); sp.add_argument("run_dir")
    sp.add_argument("--cold", default="lc_32k"); sp.add_argument("--warm", default="lc_cache")
    sp.set_defaults(fn=cmd_speedup)
    va = sub.add_parser("variance"); va.add_argument("run_dirs", nargs="+")
    va.add_argument("--workload", default="lc_32k"); va.set_defaults(fn=cmd_variance)
    dr = sub.add_parser("drift"); dr.add_argument("raw")
    dr.add_argument("--buckets", type=int, default=6); dr.add_argument("--pct", type=int, default=99)
    dr.set_defaults(fn=cmd_drift)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
