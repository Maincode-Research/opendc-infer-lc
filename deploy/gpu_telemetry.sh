#!/usr/bin/env bash
# Sample GPU util/mem/power/temp to CSV until killed (AMD or NVIDIA).
# Usage: gpu_telemetry.sh <out.csv> [interval_s]
# Delegates to the robust stdlib Python sampler (the earlier amd-smi flag-based
# version produced empty files on some amd-smi versions).
set -uo pipefail
OUT="${1:?out csv}"
INT="${2:-2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if command -v python3 >/dev/null 2>&1; then
  exec python3 "$HERE/gpu_telemetry.py" "$OUT" "$INT"
fi
# --- fallback: legacy tool-specific paths (only if no python3) -------------

vendor="${GPU_VENDOR:-auto}"
if [[ "$vendor" == "auto" ]]; then
  if command -v amd-smi >/dev/null 2>&1 || command -v rocm-smi >/dev/null 2>&1; then
    vendor=amd
  elif command -v nvidia-smi >/dev/null 2>&1; then
    vendor=nvidia
  fi
fi

echo "telemetry vendor=$vendor -> $OUT" >&2

if [[ "$vendor" == "nvidia" ]]; then
  exec nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv,nounits -l "$INT" > "$OUT"
elif [[ "$vendor" == "amd" ]]; then
  if command -v amd-smi >/dev/null 2>&1; then
    # amd-smi monitor: per-GPU usage, mem, power, temp.
    exec amd-smi monitor --power --temperature --gpu-util --mem-util --csv \
      --interval "$INT" --file "$OUT"
  else
    # rocm-smi fallback loop.
    echo "timestamp,gpu,gpu_use_pct,mem_use_pct,power_w,temp_c" > "$OUT"
    while true; do
      ts="$(date -Is)"
      rocm-smi --showuse --showmemuse --showpower --showtemp --csv 2>/dev/null \
        | tail -n +2 \
        | awk -F',' -v ts="$ts" '{print ts","$0}' >> "$OUT"
      sleep "$INT"
    done
  fi
else
  echo "no GPU telemetry tool found; writing empty $OUT" >&2
  : > "$OUT"
fi
