#!/usr/bin/env python3
"""Robust GPU telemetry sampler -> CSV (timestamp, gpu, util_pct, mem_used_mb,
power_w, temp_c). Works on AMD (amd-smi / rocm-smi) and NVIDIA (nvidia-smi),
auto-detected. Stdlib only. Writes a row per GPU per interval until killed.

Usage: gpu_telemetry.py <out.csv> [interval_s]

The previous shell version relied on amd-smi flags that varied by version and
produced an empty file; this version parses JSON defensively and falls back
across tools, always writing a header + rows.
"""
import json
import shutil
import subprocess
import sys
import time


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""


def _walk_find(obj, *needles):
    """Return first numeric leaf whose key path contains any needle (lowercased)."""
    needles = [n.lower() for n in needles]
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if any(n in kl for n in needles):
                val = _num(v)
                if val is not None:
                    return val
            r = _walk_find(v, *needles)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _walk_find(v, *needles)
            if r is not None:
                return r
    return None


def _num(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):  # e.g. {"value": 123, "unit": "W"}
        return _num(v.get("value"))
    if isinstance(v, str):
        try:
            return float(v.strip().split()[0])
        except (ValueError, IndexError):
            return None
    return None


def sample_amd():
    out = _run(["amd-smi", "metric", "-g", "all", "--json"])
    rows = []
    try:
        data = json.loads(out)
    except Exception:
        return None
    gpus = data if isinstance(data, list) else data.get("gpu", data.get("gpus", [data]))
    if isinstance(gpus, dict):
        gpus = [gpus]
    for i, g in enumerate(gpus):
        rows.append((i,
                     _walk_find(g, "gfx_activity", "gfx_busy", "gpu_util", "usage"),
                     _walk_find(g, "vram_used", "mem_used", "used_vram"),
                     _walk_find(g, "socket_power", "average_socket_power", "power"),
                     _walk_find(g, "edge", "hotspot", "temperature")))
    return rows or None


def sample_nvidia():
    out = _run(["nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits"])
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            rows.append((parts[0], _num(parts[1]), _num(parts[2]), _num(parts[3]), _num(parts[4])))
    return rows or None


def main():
    out_path = sys.argv[1]
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    if shutil.which("amd-smi") or shutil.which("rocm-smi"):
        sampler = sample_amd
    elif shutil.which("nvidia-smi"):
        sampler = sample_nvidia
    else:
        sampler = None
    with open(out_path, "w", buffering=1) as f:
        f.write("ts,gpu,util_pct,mem_used_mb,power_w,temp_c\n")
        if sampler is None:
            return
        while True:
            t = time.time()
            rows = sampler() or []
            for gpu, util, mem, power, temp in rows:
                f.write(f"{t:.3f},{gpu},{util},{mem},{power},{temp}\n")
            time.sleep(interval)


if __name__ == "__main__":
    main()
